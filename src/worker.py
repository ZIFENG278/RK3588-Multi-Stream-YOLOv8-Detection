from threading import Thread
from queue import Queue, Empty, Full
import time
from src.config import Config
from src.rknn_executor import RKNN_model_container
import cv2
import numpy as np

from src.yolov8 import YOLOv8Tool
from src.yolo26 import YOLO26Tool
from src.visualization import Visualizer


def build_detector_tool(config: Config):
    model_type = getattr(config, 'model_type', 'yolov8').lower()
    if model_type == 'yolo26':
        return YOLO26Tool(config)
    return YOLOv8Tool(config)


class FrameTask:
    """Task for NPU inference."""

    def __init__(self, stream_id: int, frame: np.ndarray,
                 orig_shape: tuple, processed: np.ndarray, video_fps: float = 0.0):
        self.stream_id = stream_id
        self.frame = frame
        self.orig_shape = orig_shape
        self.processed = processed
        self.video_fps = video_fps
        # Timing (ms)
        self.decode_time = 0.0
        self.preprocess_time = 0.0
        self.infer_time = 0.0
        self.postprocess_time = 0.0
        self.draw_time = 0.0


class PostProcessWorker(Thread):
    """Post-process + draw — all in one thread so main loop only does display."""

    def __init__(self, input_queue: Queue, output_queue: Queue, stop_event, config: Config):
        super().__init__(daemon=True)
        self.stop_event = stop_event
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.config = config
        self.detector_tool = build_detector_tool(self.config)
        self.visualizer = Visualizer(config)

    def run(self):
        while not self.stop_event.is_set():
            try:
                task = self.input_queue.get(timeout=0.05)
            except Empty:
                continue

            t1 = time.perf_counter()
            detections = self.detector_tool.postprocess(
                task.processed, [task.orig_shape]
            )[0]
            task.postprocess_time = (time.perf_counter() - t1) * 1000

            # Draw in postprocess thread — main loop only needs to display
            t_draw = time.perf_counter()
            task.frame = self.visualizer.draw_detections(task.frame, detections)
            task.draw_time = (time.perf_counter() - t_draw) * 1000

            try:
                self.output_queue.put(task, timeout=1.0)
            except:
                pass


class NPUWorker(Thread):
    """NPU inference worker."""

    def __init__(self, worker_id: int, model: RKNN_model_container, input_queue: Queue,
                 output_queue: Queue, stop_event):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.model = model
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            try:
                task = self.input_queue.get(timeout=0.05)
            except Empty:
                continue

            t0 = time.perf_counter()
            task.processed = self.model.run(task.processed)
            # print(len(task.processed), [p.shape for p in task.processed])
            task.infer_time = (time.perf_counter() - t0) * 1000

            try:
                self.output_queue.put(task, timeout=1.0)
            except:
                pass  # output queue full, skip




class DecodeWorker(Thread):
    """Video/camera decode worker."""

    # Limits for high-resolution cameras
    MAX_WIDTH = 1920
    MAX_HEIGHT = 1080
    MAX_FPS = 30

    def __init__(self, stream_id: int, video_path, output_task: Queue,
                 input_size: int, stop_event, config: Config):
        super().__init__(daemon=True)
        self.stream_id = stream_id
        self.video_path = video_path
        self.output_task = output_task
        self.input_size = input_size
        self.config = config
        self.stop_event = stop_event
        self.detector_tool = build_detector_tool(self.config)
        self.cap = None
        self.source_label = (
            f"camera index {video_path}"
            if isinstance(video_path, int)
            else str(video_path)
        )

    def _try_set_camera_resolution(self, cap, width, height, fps):
        """Try to set camera resolution. Return True if successful."""
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        if actual_w != width or actual_h != height:
            print(f"Stream {self.stream_id}: Camera resolution {actual_w}x{actual_h} "
                  f"(requested {width}x{height}), fps={actual_fps}")
            return False
        return True

    def _open_camera_v4l2(self, reopen=False):
        """Open USB camera using GStreamer v4l2src with resolution limit."""
        action = "Reopened" if reopen else "Opened"

        # Try H264 pipeline first (for cameras that output H264 stream at 1280x720@30fps)
        h264_pipeline = (
            f"v4l2src device=/dev/video{self.video_path} ! "
            f"video/x-h264,width=1280,height=720,framerate=30/1 ! "
            f"h264parse ! mppvideodec ! videoconvert ! appsink sync=False"
        )

        cap = cv2.VideoCapture(h264_pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Stream {self.stream_id}: {action} camera {self.video_path} "
                  f"via v4l2src (H264) at {actual_w}x{actual_h}@{actual_fps}fps")
            return cap
        cap.release()

        # Try MJPEG pipeline (for cameras that output JPEG)
        mjpeg_pipeline = (
            f"v4l2src device=/dev/video{self.video_path} ! "
            f"image/jpeg,width=1280,height=720,framerate=30/1 ! "
            f"jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink sync=False"
        )

        cap = cv2.VideoCapture(mjpeg_pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Stream {self.stream_id}: {action} camera {self.video_path} "
                  f"via v4l2src (MJPEG) at {actual_w}x{actual_h}@{actual_fps}fps")
            return cap
        cap.release()

        # Fallback to raw video pipeline
        raw_pipeline = (
            f"v4l2src device=/dev/video{self.video_path} ! "
            f"video/x-raw,width={self.MAX_WIDTH},height={self.MAX_HEIGHT},framerate={self.MAX_FPS} ! "
            f"videoconvert ! video/x-raw,format=BGR ! appsink sync=False"
        )

        cap = cv2.VideoCapture(raw_pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Stream {self.stream_id}: {action} camera {self.video_path} "
                  f"via v4l2src (raw) at {actual_w}x{actual_h}@{actual_fps}fps")
            return cap

        cap.release()
        return None

    def _open_camera_cv2(self, reopen=False):
        """Open USB camera using OpenCV directly with resolution limit."""
        action = "Reopened" if reopen else "Opened"

        # OpenCV cannot decode H264, so for H264 we need GStreamer with software decoder
        # Try GStreamer with avdec_h264 (software decode) or jpegdec
        gst_pipeline = (
            f"v4l2src device=/dev/video{self.video_path} ! "
            f"video/x-h264,width=1280,height=720,framerate=30/1 ! "
            f"h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! appsink sync=False"
        )

        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Stream {self.stream_id}: {action} camera {self.video_path} "
                  f"via GStreamer (H264 soft) at {actual_w}x{actual_h}@{actual_fps}fps")
            return cap
        cap.release()

        # Try MJPEG via GStreamer with jpegdec
        mjpeg_pipeline = (
            f"v4l2src device=/dev/video{self.video_path} ! "
            f"image/jpeg,width=1280,height=720,framerate=30/1 ! "
            f"jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink sync=False"
        )

        cap = cv2.VideoCapture(mjpeg_pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Stream {self.stream_id}: {action} camera {self.video_path} "
                  f"via GStreamer (MJPEG) at {actual_w}x{actual_h}@{actual_fps}fps")
            return cap
        cap.release()

        # Last resort: OpenCV direct with YUYV raw format
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Stream {self.stream_id}: Cannot open camera index {self.video_path}")
            return None

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"Stream {self.stream_id}: {action} camera index {self.video_path} "
              f"at {actual_w}x{actual_h}@{actual_fps}fps via OpenCV")
        return cap

    def _open_video(self, reopen: bool = False):
        """Open camera or video source with GStreamer or fallback to OpenCV."""
        if isinstance(self.video_path, int):
            # Try GStreamer v4l2src first for USB camera
            if self.config.use_vpu:
                cap = self._open_camera_v4l2(reopen)
                if cap is not None:
                    return cap

            # Fallback to OpenCV
            return self._open_camera_cv2(reopen)

        # Video file: use GStreamer pipelines
        if self.config.use_vpu:
            sync_str = "True" if self.config.sync else "False"
            pipelines = [
                (
                    f"filesrc location={self.video_path} ! "
                    f"qtdemux ! h264parse ! mppvideodec ! videoconvert ! "
                    f"video/x-raw,format=BGR ! appsink sync={sync_str}",
                    "mppvideodec (hardware)"
                ),
                # (
                #     f"filesrc location={self.video_path} ! "
                #     f"decodebin ! videoconvert ! "
                #     f"video/x-raw,format=BGR ! appsink sync=False",
                #     "decodebin (auto-detect)"
                # ),
            ]

            action = "Reopened" if reopen else "Opened"
            for pipeline, label in pipelines:
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    print(f"Stream {self.stream_id}: {action} with {label}")
                    return cap
                cap.release()

        # Fallback to OpenCV
        cap = cv2.VideoCapture(self.video_path)
        if cap.isOpened():
            action = "Reopened" if reopen else "Opened"
            print(f"Stream {self.stream_id}: {action} with OpenCV (software)")
            return cap

        print(f"Error: Cannot open video {self.video_path}")
        return None

    def run(self):
        self.cap = self._open_video()
        if self.cap is None or not self.cap.isOpened():
            print(f"Stream {self.stream_id}: {self.source_label} init failed, signaling stop")
            self.stop_event.set()
            return

        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        reconnect_delay = 0.5  # seconds

        while not self.stop_event.is_set():
            t0 = time.perf_counter()
            ret, frame = self.cap.read()

            if not ret:
                print(f"Stream {self.stream_id}: {self.source_label} disconnected, "
                      f"reconnecting in {reconnect_delay}s...")
                self.cap.release()

                # Wait before reconnecting
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 10.0)  # backoff, max 10s

                self.cap = self._open_video(reopen=True)
                if self.cap is None:
                    # Failed to reconnect, keep trying
                    print(f"Stream {self.stream_id}: Reconnect failed, retrying...")
                    continue

                # Reset delay on successful reopen
                reconnect_delay = 0.5
                video_fps = self.cap.get(cv2.CAP_PROP_FPS)
                frame_interval = 1.0 / video_fps if video_fps > 0 else 0

                # Try reading first frame
                ret, frame = self.cap.read()
                if not ret:
                    print(f"Stream {self.stream_id}: Reconnect read failed, retrying...")
                    continue

            decode_time = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            processed = self.detector_tool.preprocess(frame)
            processed = np.expand_dims(processed, axis=0)
            preprocess_time = (time.perf_counter() - t1) * 1000
            orig_shape = (frame.shape[0], frame.shape[1])

            task = FrameTask(
                stream_id=self.stream_id,
                frame=frame,
                orig_shape=orig_shape,
                processed=processed,
                video_fps=video_fps
            )
            task.decode_time = decode_time
            task.preprocess_time = preprocess_time

            try:
                self.output_task.put(task, timeout=0.5)
            except Full:
                print("Warning: Task queue full, skipping frame")

        if self.cap:
            self.cap.release()
        print(f"Stream {self.stream_id}: Worker stopped")
