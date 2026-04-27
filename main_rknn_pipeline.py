#!/usr/bin/env python3
"""
Multi-stream YOLOv8 detection on Rockchip NPU - Pipeline Architecture.

Architecture:
- Decode workers: decode + preprocess in parallel
- NPU pool: 3 cores continuously processing inference
- PostProcess worker: 1 dedicated thread for postprocess + draw
- Main loop: collect results and display only

Usage:
    python main_rknn_pipeline.py [--num-streams N] [--num-cores C]
"""

import argparse
import os
import sys
import time
from pathlib import Path
from threading import Event
from queue import Queue, Empty
from typing import List, Optional, Dict
import cv2
import numpy as np

from src.config import Config
from src.worker import NPUWorker, DecodeWorker, PostProcessWorker
from src.visualization import GridDisplay, VideoWriter

try:
    from rknnlite.api import RKNNLite as RKNN
    from src.rknn_executor import RKNN_model_container
except ImportError:
    print("ERROR: rknnlite package not found. Install with: pip install rknnlite2")
    sys.exit(1)

SUPPORTED_SOCS = {'rk3588', 'rk3576'}


def get_soc_npu_config(soc: str):
    SOC = soc.lower()

    if SOC == 'rk3588':
        MAX_NPU_CORES = 6
        core_masks = [RKNN.NPU_CORE_0, RKNN.NPU_CORE_1, RKNN.NPU_CORE_2]
    elif SOC == 'rk3576':
        MAX_NPU_CORES = 4
        core_masks = [RKNN.NPU_CORE_0, RKNN.NPU_CORE_1]
    else:
        raise ValueError(f"Unsupported SoC '{soc}'. Supported: {sorted(SUPPORTED_SOCS)}")

    return SOC, MAX_NPU_CORES, core_masks


class PipelineDetector:
    def __init__(self, config: Config, video_paths: List):
        self.config = config
        self.video_paths = video_paths
        self.num_streams = self.config.num_streams
        self.soc, self.max_npu_cores, self.core_masks = get_soc_npu_config(self.config.soc)
        self.num_cores = min(self.config.num_cores, self.max_npu_cores)

        print(f"Initializing pipeline ({self.soc}): {self.num_streams} streams on {self.num_cores} cores...")

        # Load models (one per NPU core)
        self.models: List[RKNN_model_container] = []
        for core_id in range(self.num_cores):
            core_mask = self.core_masks[core_id % len(self.core_masks)]
            print(f"  Model {core_id} on NPU core mask {core_mask}...")
            self.models.append(RKNN_model_container(
                model_path=config.model_path,
                target=self.soc,
                core_mask=core_mask
            ))

        # Queues
        self.npu_input_queue = Queue(maxsize=20)
        self.npu_output_queue = Queue(maxsize=20)
        self.postprocess_queue = Queue(maxsize=20)

        self.stop_event = Event()
        self.stop_event.clear()

        self.video_decode_workers: List[DecodeWorker] = []
        self.npu_workers: List[NPUWorker] = []
        self.postprocess_workers: List[PostProcessWorker] = []

        self.grid_display = GridDisplay(config, self.num_streams) if config.display_results else None
        self.frame_count = 0

            # Video writer — independent of display, so --no-display still saves video
        self.video_writer = None
        if config.save_video:
            os.makedirs(config.output_dir, exist_ok=True)
            out_path = os.path.join(config.output_dir, 'detection_grid.mp4')
            # Use first video's FPS as output FPS
            video_fps = round(cv2.VideoCapture(self.video_paths[0]).get(cv2.CAP_PROP_FPS))
            # GridDisplay may be None when --no-display, use GridDisplay temporarily for grid size
            _gd = GridDisplay(config, self.num_streams)
            grid_w = _gd.grid_cols * _gd.cell_size[0]
            grid_h = _gd.grid_rows * _gd.cell_size[1]
            self.video_writer = VideoWriter(
                out_path, fps=video_fps, frame_size=(grid_w, grid_h),
                use_vpu=config.use_vpu
            )
            print(f"  Saving video to: {out_path} ({grid_w}x{grid_h}) @ {video_fps:.0f}fps")

    def start(self):
        # NPU workers first (they need models loaded)
        for i in range(self.num_cores):
            w = NPUWorker(
                worker_id=i,
                model=self.models[i],
                input_queue=self.npu_input_queue,
                output_queue=self.npu_output_queue,
                stop_event=self.stop_event
            )
            w.start()
            self.npu_workers.append(w)

        # Decode workers
        for i, path in enumerate(self.video_paths):
            w = DecodeWorker(
                stream_id=i,
                video_path=path,
                output_task=self.npu_input_queue,
                input_size=self.config.input_size,
                stop_event=self.stop_event,
                config=self.config
            )
            w.start()
            self.video_decode_workers.append(w)

        # PostProcess workers
        for i in range(self.config.num_postprocess):
            w = PostProcessWorker(
                input_queue=self.npu_output_queue,
                output_queue=self.postprocess_queue,
                stop_event=self.stop_event,
                config=self.config
            )
            w.start()
            self.postprocess_workers.append(w)

        print(f"Started: {len(self.video_decode_workers)} decode, "
              f"{len(self.npu_workers)} NPU, {len(self.postprocess_workers)} postprocess")

    def stop(self):
        self.stop_event.set()

        for worker in self.video_decode_workers:
            worker.join(timeout=1.0)
        for worker in self.npu_workers:
            worker.join(timeout=1.0)
        for worker in self.postprocess_workers:
            worker.join(timeout=1.0)

        for model in self.models:
            model.release()

        if self.grid_display is not None:
            self.grid_display.destroy()

        if self.video_writer is not None:
            self.video_writer.release()
            print(f"  Video saved.")

    def run(self):
        self.start()

        latest_frames: Dict[int, np.ndarray] = {}
        video_fps = [0.0] * self.num_streams
        process_fps = [0.0] * self.num_streams
        frame_times = [time.time()] * self.num_streams

        stats = {'decode': [], 'preprocess': [], 'inference': [], 'postprocess': [], 'draw': []}

        # FPS tracking per stream
        inf_count = [0] * self.num_streams

        try:
            while True:
                try:
                    task = self.postprocess_queue.get(timeout=0.05)
                except Empty:
                    if self.stop_event.is_set():
                        break
                    time.sleep(0.001)
                    continue

                # task.frame is already drawn by PostProcessWorker
                stats['decode'].append(task.decode_time)
                stats['preprocess'].append(task.preprocess_time)
                stats['inference'].append(task.infer_time)
                stats['postprocess'].append(task.postprocess_time)
                stats['draw'].append(task.draw_time)

                latest_frames[task.stream_id] = task.frame
                video_fps[task.stream_id] = task.video_fps

                # Per-stream FPS
                inf_count[task.stream_id] += 1
                now = time.time()
                dt = now - frame_times[task.stream_id]
                if dt >= 1.0:
                    process_fps[task.stream_id] = inf_count[task.stream_id] / dt
                    inf_count[task.stream_id] = 0
                    frame_times[task.stream_id] = now

                self.frame_count += 1

                # Display when all streams have at least one frame
                if len(latest_frames) >= self.num_streams:
                    display_frames = [
                        latest_frames.get(i, np.zeros((240, 426, 3), dtype=np.uint8))
                        for i in range(self.num_streams)
                    ]
                    # grid_creator is grid_display if available, otherwise a temporary one
                    grid_creator = self.grid_display if self.grid_display is not None else GridDisplay(self.config, self.num_streams)
                    grid = grid_creator.create_grid(display_frames, video_fps, process_fps)

                    if self.video_writer is not None:
                        self.video_writer.write(grid)

                    if self.grid_display is not None:
                        if not self.grid_display.show(grid, 1):
                            break

                    latest_frames.clear()

                if self.config.max_frames and self.frame_count >= self.config.max_frames:
                    break

        finally:
            self._print_stats(stats, process_fps)
            self.stop()

    def _print_stats(self, stats, process_fps):
        print(f"\n{'='*55}")
        print(f"  Total frames:      {self.frame_count}")
        print(f"  NPU cores:         {self.num_cores}")
        if stats['inference']:
            def avg(lst): return np.mean(lst) if lst else 0
            print(f"\n  Timing (avg ms):")
            print(f"    Decode:        {avg(stats['decode']):8.2f} ms")
            print(f"    Preprocess:    {avg(stats['preprocess']):8.2f} ms")
            print(f"    Inference:     {avg(stats['inference']):8.2f} ms")
            print(f"    Postprocess:   {avg(stats['postprocess']):8.2f} ms")
            print(f"    Draw:          {avg(stats['draw']):8.2f} ms")
            total = (avg(stats['decode']) + avg(stats['preprocess']) +
                     avg(stats['inference']) + avg(stats['postprocess']) +
                     avg(stats['draw']))
            print(f"    Total E2E:     {total:8.2f} ms")
            print(f"\n  Throughput:")
            for i, fps in enumerate(process_fps):
                if fps > 0:
                    print(f"    Stream {i}: {fps:.1f} fps")
        print(f"{'='*55}")


def get_video_files(video_dir: str, num_streams: int) -> list:
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    video_files = []
    for f in sorted(os.listdir(video_dir)):
        if Path(f).suffix.lower() in video_extensions:
            video_files.append(os.path.join(video_dir, f))

    if len(video_files) < num_streams:
        while len(video_files) < num_streams:
            video_files.extend(video_files[:num_streams - len(video_files)])

    return video_files[:num_streams]


def parse_camera_indexes(raw_indexes: Optional[str]) -> List[int]:
    if not raw_indexes:
        return []

    camera_indexes = []
    for token in raw_indexes.split(','):
        token = token.strip()
        if not token:
            continue
        try:
            camera_indexes.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid camera index '{token}'. Use comma-separated integers.") from exc
    return camera_indexes


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-stream YOLOv8 on Rockchip NPU')
    parser.add_argument('--model', type=str, default='yolov8n-i8-3588.rknn')
    parser.add_argument('--model-type', type=str, default='yolov8', choices=['yolov8', 'yolo26'])
    parser.add_argument('--soc', type=str, default='rk3588', choices=sorted(SUPPORTED_SOCS))
    parser.add_argument('--video-dir', type=str, default='video')
    parser.add_argument('--img-src', type=str, default=None,
                        help='Path to an image or directory of images for detection')
    parser.add_argument('--img-output-dir', type=str, default='output',
                        help='Directory to save detection results when using --img-src')
    parser.add_argument('--camera-indexes', type=str, default='')
    parser.add_argument('--num-streams', type=int, default=6)
    parser.add_argument('--num-cores', type=int, default=3)
    parser.add_argument('--num-postprocess', type=int, default=3)
    parser.add_argument('--conf-threshold', type=float, default=0.25)
    parser.add_argument('--iou-threshold', type=float, default=0.7)
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--save-video', action='store_true')
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--label-file', type=str, default=None,
                        help='Path to label txt file (one class name per line)')
    parser.add_argument(
    '--use-vpu',
    action=argparse.BooleanOptionalAction,
    default=True,
    help='Use VPU for inference'
)
    parser.add_argument(
    '--sync',
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Sync video playback to original fps'
)
    return parser.parse_args()


def run_images(config: Config, img_src: str, output_dir: str):
    """Run detection on images (single image or directory) and save results."""
    from src.rknn_executor import RKNN_model_container
    from src.worker import build_detector_tool
    from src.visualization import Visualizer

    # Determine if img_src is a file or directory
    if os.path.isfile(img_src):
        img_paths = [img_src]
    elif os.path.isdir(img_src):
        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        img_paths = [
            os.path.join(img_src, f)
            for f in sorted(os.listdir(img_src))
            if Path(f).suffix.lower() in img_extensions
        ]
        if not img_paths:
            print(f"Error: No images found in directory: {img_src}")
            return
    else:
        print(f"Error: Invalid path: {img_src}")
        return

    print(f"Loading model: {config.model_path}")
    model = RKNN_model_container(
        model_path=config.model_path,
        target=config.soc,
        core_mask=RKNN.NPU_CORE_0
    )

    detector = build_detector_tool(config)
    visualizer = Visualizer(config)

    os.makedirs(output_dir, exist_ok=True)

    total_images = len(img_paths)
    total_preprocess = 0
    total_infer = 0
    total_postprocess = 0
    total_detections = 0

    for i, img_path in enumerate(img_paths, 1):
        print(f"\n[{i}/{total_images}] Processing: {img_path}")
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  Warning: Cannot read image, skipping")
            continue

        orig_h, orig_w = frame.shape[:2]

        # Preprocess
        t0 = time.perf_counter()
        processed = detector.preprocess(frame)
        processed = np.expand_dims(processed, axis=0)
        preprocess_time = (time.perf_counter() - t0) * 1000

        # Inference
        t1 = time.perf_counter()
        outputs = model.run(processed)
        infer_time = (time.perf_counter() - t1) * 1000

        # Postprocess
        t2 = time.perf_counter()
        detections = detector.postprocess(outputs, [(orig_h, orig_w)])[0]
        postprocess_time = (time.perf_counter() - t2) * 1000

        # Draw
        result_frame = visualizer.draw_detections(frame, detections)

        # Save result
        img_name = Path(img_path).name
        output_path = os.path.join(output_dir, f"result_{img_name}")
        cv2.imwrite(output_path, result_frame)

        print(f"  Size: {orig_w}x{orig_h}, Detections: {len(detections)}, "
              f"Preprocess: {preprocess_time:.2f}ms, Infer: {infer_time:.2f}ms, "
              f"Postprocess: {postprocess_time:.2f}ms")

        total_preprocess += preprocess_time
        total_infer += infer_time
        total_postprocess += postprocess_time
        total_detections += len(detections)

    print(f"\n{'='*55}")
    print(f"  Total images:     {total_images}")
    print(f"  Total detections: {total_detections}")
    print(f"  Avg timing (ms):")
    print(f"    Preprocess:    {total_preprocess / total_images:.2f} ms")
    print(f"    Inference:     {total_infer / total_images:.2f} ms")
    print(f"    Postprocess:  {total_postprocess / total_images:.2f} ms")
    print(f"  Results saved to: {output_dir}")
    print(f"{'='*55}")

    model.release()


def main():
    args = parse_args()

    try:
        camera_indexes = parse_camera_indexes(args.camera_indexes)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not os.path.isfile(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    # Image mode (single image or directory)
    if args.img_src:
        config = Config(
            soc=args.soc,
            model_type=args.model_type,
            model_path=args.model,
            input_size=640,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            display_results=False,
            label_file=args.label_file,
            use_vpu=args.use_vpu
        )
        run_images(config, args.img_src, args.img_output_dir)
        return

    if camera_indexes:
        input_sources = camera_indexes
        args.num_streams = len(camera_indexes)
    else:
        if not os.path.isdir(args.video_dir):
            print(f"Error: Video directory not found: {args.video_dir}")
            sys.exit(1)
        input_sources = get_video_files(args.video_dir, args.num_streams)

    try:
        config = Config(
            soc=args.soc,
            model_type=args.model_type,
            model_path=args.model,
            video_dir=args.video_dir,
            camera_indexes=camera_indexes or None,

            num_streams=args.num_streams,
            num_postprocess=args.num_postprocess,
            num_cores=args.num_cores,

            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,

            display_results=not args.no_display,
            max_frames=args.max_frames,

            save_video=args.save_video,
            output_dir=args.output_dir,
            label_file=args.label_file,

            use_vpu=args.use_vpu,
            sync=args.sync

        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    source_type = "camera" if camera_indexes else "video"
    print(
        f"\nRKNN pipeline: {config.num_streams} {source_type} streams, "
        f"model={config.model_type}, "
        f"{config.num_cores} NPU cores, {config.num_postprocess} postprocess workers\n"
    )
    detector = PipelineDetector(config, input_sources)
    detector.run()
    print("Done.")


if __name__ == '__main__':
    main()
