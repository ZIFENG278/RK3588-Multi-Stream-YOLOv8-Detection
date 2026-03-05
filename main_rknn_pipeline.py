#!/usr/bin/env python3
"""
Multi-stream YOLOv8 detection on RK3588 NPU - Pipeline Architecture.

Architecture:
- Video decode threads: decode multiple videos in parallel
- Preprocess: letterbox + BGR2RGB
- NPU inference pool: 3 cores continuously processing
- Result queue: collect results, sort by frame_id
- Display: show frames in order

Usage:
    python main_rknn_pipeline.py [--num-streams N] [--num-cores C]
"""

import argparse
import os
import sys
import time
from pathlib import Path
from threading import Thread, Event
from queue import Queue, Empty
from typing import List, Optional, Dict
import numpy as np
import cv2

from src_rknn.config import Config
from src_rknn.rknn_inference import YOLOv8RKNN
from src.visualization import Visualizer, GridDisplay

# RK3588 has 3 NPU cores
MAX_NPU_CORES = 3


class FrameTask:
    """Task for NPU inference."""

    def __init__(self, stream_id: int, frame: np.ndarray,
                 orig_shape: tuple, processed: np.ndarray, video_fps: float = 0.0):
        self.stream_id = stream_id
        self.frame = frame
        self.orig_shape = orig_shape
        self.processed = processed
        self.video_fps = video_fps  # 原始视频帧率
        self.detections = None
        # Timing (ms)
        self.decode_time = 0.0
        self.preprocess_time = 0.0
        self.infer_time = 0.0
        self.postprocess_time = 0.0


class DecodeWorker(Thread):
    """Video decode worker - continuously decode frames."""

    def __init__(self, stream_id: int, video_path: str, task_queue: Queue,
                 input_size: int, stop_event):
        super().__init__(daemon=True)
        self.stream_id = stream_id
        self.video_path = video_path
        self.task_queue = task_queue
        self.input_size = input_size
        self.stop_event = stop_event

        self.cap = None

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Letterbox preprocess."""
        h, w = frame.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

        return cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    def run(self):
        """Main decode loop."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"Error: Cannot open video {self.video_path}")
            return

        # Get video FPS (fixed value from video file)
        video_fps = self.cap.get(cv2.CAP_PROP_FPS)

        while not self.stop_event.is_set():
            t0 = time.perf_counter()
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    break
            decode_time = (time.perf_counter() - t0) * 1000

            # Preprocess
            t1 = time.perf_counter()
            processed = self._preprocess(frame)
            preprocess_time = (time.perf_counter() - t1) * 1000
            orig_shape = (frame.shape[0], frame.shape[1])

            # Create task and put to queue
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
                self.task_queue.put(task, timeout=0.1)
            except:
                pass  # Queue full, skip this frame

        if self.cap:
            self.cap.release()


class NPUWorker(Thread):
    """NPU inference worker - continuously process tasks from queue."""

    def __init__(self, worker_id: int, model: YOLOv8RKNN, input_queue: Queue,
                 output_queue: Queue, stop_event):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.model = model
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event

    def run(self):
        """Main inference loop."""
        while not self.stop_event.is_set():
            try:
                task = self.input_queue.get(timeout=0.01)
            except Empty:
                continue

            # Run inference (only NPU, no preprocess)
            t0 = time.perf_counter()
            outputs = self.model.infer(task.processed, core_id=0)
            task.infer_time = (time.perf_counter() - t0) * 1000

            # Postprocess
            t1 = time.perf_counter()
            detections = self.model.postprocess(outputs, [task.orig_shape])[0]
            task.postprocess_time = (time.perf_counter() - t1) * 1000

            task.detections = detections

            # Put to output queue
            try:
                self.output_queue.put(task, timeout=0.1)
            except:
                pass


class PipelineDetector:
    """Pipeline-based multi-stream detector."""

    def __init__(self, config: Config, video_paths: List[str], num_cores: int = 3):
        self.config = config
        self.video_paths = video_paths
        self.num_streams = len(video_paths)
        self.num_cores = min(num_cores, MAX_NPU_CORES)

        print(f"Initializing pipeline: {self.num_streams} streams on {self.num_cores} cores...")

        # Load models (one per NPU core)
        self.models: List[YOLOv8RKNN] = []
        for core_id in range(self.num_cores):
            print(f"  Loading model for core {core_id}...")
            model = YOLOv8RKNN(
                model_path=config.model_path,
                conf_threshold=config.conf_threshold,
                iou_threshold=config.iou_threshold,
                input_size=config.input_size,
                num_cores=1,
                core_id=core_id
            )
            self.models.append(model)

        # Queues
        self.task_queue = Queue(maxsize=100)  # Tasks to NPU
        self.result_queue = Queue(maxsize=100)  # Results from NPU

        # Stop event
        self.stop_event = Event()
        self.stop_event.clear()

        # Workers
        self.decode_workers: List[DecodeWorker] = []
        self.npu_workers: List[NPUWorker] = []

        # Visualization (only if display enabled)
        self.grid_display = GridDisplay(config, self.num_streams) if config.display_results else None

        self.frame_count = 0

    def start(self):
        """Start all workers."""
        # Start NPU workers
        for i in range(self.num_cores):
            worker = NPUWorker(
                worker_id=i,
                model=self.models[i],
                input_queue=self.task_queue,
                output_queue=self.result_queue,
                stop_event=self.stop_event
            )
            worker.start()
            self.npu_workers.append(worker)

        # Start decode workers
        for i, path in enumerate(self.video_paths):
            worker = DecodeWorker(
                stream_id=i,
                video_path=path,
                task_queue=self.task_queue,
                input_size=self.config.input_size,
                stop_event=self.stop_event
            )
            worker.start()
            self.decode_workers.append(worker)

        print(f"Started {len(self.decode_workers)} decode workers and {len(self.npu_workers)} NPU workers")

    def stop(self):
        """Stop all workers."""
        self.stop_event.set()

        for worker in self.decode_workers:
            worker.join(timeout=1.0)
        for worker in self.npu_workers:
            worker.join(timeout=1.0)

        for model in self.models:
            model.release()

        if self.grid_display is not None:
            self.grid_display.destroy()

    def run(self, max_frames: Optional[int] = None):
        """Main run loop - collect results and display."""
        self.start()

        # Visualizer for each stream
        visualizers = [Visualizer(self.config) for _ in range(self.num_streams)]

        # Latest frames for display
        latest_frames: Dict[int, np.ndarray] = {}  # stream_id -> (frame, detections)
        video_fps = [0.0] * self.num_streams  # 原始视频帧率 (固定)
        process_fps = [0.0] * self.num_streams  # 推理帧率 (实时计算)
        frame_times = [time.time()] * self.num_streams

        # Timing stats
        stats = {
            'decode': [],
            'preprocess': [],
            'inference': [],
            'postprocess': [],
            'draw': [],
        }

        try:
            while True:
                # Collect results from NPU
                try:
                    task = self.result_queue.get(timeout=0.01)

                    # Draw detections (统计耗时)
                    t_draw = time.perf_counter()
                    visualizer = visualizers[task.stream_id]
                    frame_with_det = visualizer.draw_detections(task.frame, task.detections)
                    draw_time = (time.perf_counter() - t_draw) * 1000

                    # Store timing
                    stats['decode'].append(task.decode_time)
                    stats['preprocess'].append(task.preprocess_time)
                    stats['inference'].append(task.infer_time)
                    stats['postprocess'].append(task.postprocess_time)
                    stats['draw'].append(draw_time)

                    # Store for display (already drawn)
                    latest_frames[task.stream_id] = frame_with_det

                    # Store video FPS (fixed from video file)
                    # if video_fps[task.stream_id] == 0.0:
                    video_fps[task.stream_id] = task.video_fps

                    # Update process FPS (real-time)
                    now = time.time()
                    dt = now - frame_times[task.stream_id]
                    if dt > 0:
                        process_fps[task.stream_id] = 1.0 / dt
                    frame_times[task.stream_id] = now

                    self.frame_count += 1

                except Empty:
                    pass

                # Check if we have frames from all streams
                if len(latest_frames) >= self.num_streams:
                    # Display if enabled
                    if self.grid_display is not None:
                        # Collect frames in stream order (already drawn)
                        display_frames = []
                        for i in range(self.num_streams):
                            if i in latest_frames:
                                display_frames.append(latest_frames[i])
                            else:
                                display_frames.append(np.zeros((240, 426, 3), dtype=np.uint8))

                        # Display: video_fps (fixed) and process_fps (realtime)
                        grid = self.grid_display.create_grid(
                            display_frames,
                            # [[] for _ in range(self.num_streams)],
                            video_fps,
                            process_fps
                        )
                        if not self.grid_display.show(grid, 1):
                            break

                    latest_frames.clear()

                # Check frame limit
                if max_frames and self.frame_count >= max_frames:
                    break

                time.sleep(0.001)

        finally:
            # Print stats
            print(f"\n=== Pipeline Statistics ===")
            print(f"Total frames: {self.frame_count}")
            print(f"NPU Cores: {self.num_cores}")
            if stats['inference']:
                print(f"\nTiming (avg ms):")
                print(f"  Decode:      {np.mean(stats['decode']):.2f} ms")
                print(f"  Preprocess:  {np.mean(stats['preprocess']):.2f} ms")
                print(f"  Inference:   {np.mean(stats['inference']):.2f} ms")
                print(f"  Postprocess: {np.mean(stats['postprocess']):.2f} ms")
                print(f"  Draw:        {np.mean(stats['draw']):.2f} ms")
                total = (np.mean(stats['decode']) + np.mean(stats['preprocess']) +
                         np.mean(stats['inference']) + np.mean(stats['postprocess']) +
                         np.mean(stats['draw']))
                print(f"  Total:       {total:.2f} ms")
            self.stop()


def get_video_files(video_dir: str, num_streams: int) -> list:
    """Get video files from directory."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}

    video_files = []
    for f in sorted(os.listdir(video_dir)):
        if Path(f).suffix.lower() in video_extensions:
            video_files.append(os.path.join(video_dir, f))

    if len(video_files) < num_streams:
        while len(video_files) < num_streams:
            video_files.extend(video_files[:num_streams - len(video_files)])

    return video_files[:num_streams]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Multi-stream YOLOv8 Pipeline on RK3588 NPU'
    )
    parser.add_argument('--model', type=str, default='yolov8n.rknn')
    parser.add_argument('--video-dir', type=str, default='video')
    parser.add_argument('--num-streams', type=int, default=9)
    parser.add_argument('--num-cores', type=int, default=3)
    parser.add_argument('--conf-threshold', type=float, default=0.4)
    parser.add_argument('--iou-threshold', type=float, default=0.45)
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--max-frames', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.video_dir):
        print(f"Error: Video directory not found: {args.video_dir}")
        sys.exit(1)

    if not os.path.isfile(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    video_files = get_video_files(args.video_dir, args.num_streams)

    config = Config(
        model_path=args.model,
        video_dir=args.video_dir,
        output_dir='output',
        num_streams=args.num_streams,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        save_video=False,
        display_results=not args.no_display
    )

    print(f"\nStarting RKNN pipeline detection...")
    print(f"Streams: {args.num_streams}, Cores: {args.num_cores}\n")

    detector = PipelineDetector(config, video_files, args.num_cores)
    detector.run(max_frames=args.max_frames)

    print("\nDetection complete!")


if __name__ == '__main__':
    main()
