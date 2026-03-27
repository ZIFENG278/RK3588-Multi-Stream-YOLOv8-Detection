from threading import Thread
from queue import Queue, Empty, Full
import time
from src.config import Config
from src.rknn_executor import RKNN_model_container
import cv2
import numpy as np

from src.yolov8 import YOLOv8Tool
from src.visualization import Visualizer


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
        self.yolov8_tool = YOLOv8Tool(self.config)
        self.visualizer = Visualizer(config)

    def run(self):
        while not self.stop_event.is_set():
            try:
                task = self.input_queue.get(timeout=0.05)
            except Empty:
                continue

            t1 = time.perf_counter()
            detections = self.yolov8_tool.postprocess(
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
    """Video decode worker."""

    def __init__(self, stream_id: int, video_path: str, output_task: Queue,
                 input_size: int, stop_event, config: Config):
        super().__init__(daemon=True)
        self.stream_id = stream_id
        self.video_path = video_path
        self.output_task = output_task
        self.input_size = input_size
        self.config = config
        self.stop_event = stop_event
        self.yolov8_tool = YOLOv8Tool(self.config)
        self.cap = None

    def _open_video(self, reopen: bool = False):
        """Open video with GStreamer or fallback to OpenCV."""
        action = "Reopened" if reopen else "Opened"
        
        if self.config.use_vpu:
            pipelines = [
                (
                    f"filesrc location={self.video_path} ! "
                    f"qtdemux ! h264parse ! mppvideodec ! videoconvert ! "
                    f"video/x-raw,format=BGR ! appsink sync=False",
                    "mppvideodec (hardware)"
                ),
                (
                    f"filesrc location={self.video_path} ! "
                    f"decodebin ! videoconvert ! "
                    f"video/x-raw,format=BGR ! appsink sync=False",
                    "decodebin (auto-detect)"
                ),
            ]
            
            for pipeline, label in pipelines:
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    print(f"Stream {self.stream_id}: {action} with {label}")
                    return cap
                cap.release()
        
        # Fallback to OpenCV
        cap = cv2.VideoCapture(self.video_path)
        if cap.isOpened():
            print(f"Stream {self.stream_id}: {action} with OpenCV (software)")
            return cap
        
        print(f"Error: Cannot open video {self.video_path}")
        return None

    def run(self):
        self.cap = self._open_video()
        if self.cap is None or not self.cap.isOpened():
            return

        video_fps = self.cap.get(cv2.CAP_PROP_FPS)

        while not self.stop_event.is_set():
            t0 = time.perf_counter()
            ret, frame = self.cap.read()
            if not ret:
                # Video ended, reopen for looping
                print(f"Stream {self.stream_id}: Video ended, restarting...")
                self.cap.release()
                self.cap = self._open_video(reopen=True)
                if self.cap is None:
                    break
                
                # Try reading first frame of new cycle
                ret, frame = self.cap.read()
                if not ret:
                    print(f"Stream {self.stream_id}: Failed to read frame after reopening")
                    break
            
            decode_time = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            processed = self.yolov8_tool.preprocess(frame)
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
                pass  # queue full, skip frame

        if self.cap:
            self.cap.release()
