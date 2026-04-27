"""Visualization module for drawing detections on frames."""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from src.config import Config


class Visualizer:
    """
    Visualizer for drawing detection results on frames.
    """
    
    # Color palette for different classes
    COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
        (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
        (0, 255, 128), (128, 0, 255), (0, 128, 255), (255, 255, 128), (255, 128, 255),
    ]
    
    def __init__(self, config: Config):
        """
        Initialize visualizer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.thickness = 2
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Tuple[int, int, int, int, float, int]]
    ) -> np.ndarray:
        """
        Draw detections on frame.
        
        Args:
            frame: Input frame (BGR)
            detections: List of (x1, y1, x2, y2, conf, class)
            
        Returns:
            Frame with detections drawn
        """
        result = frame.copy()
        
        for x1, y1, x2, y2, conf, cls in detections:
            # Get color for class
            color = self.COLORS[cls % len(self.COLORS)]
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, self.thickness)
            
            # Draw label
            label = f"{self.config.class_names[cls]} {conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, self.font, self.font_scale, self.thickness
            )
            
            # Label background
            cv2.rectangle(
                result,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w, y1),
                color,
                -1
            )
            
            # Label text
            cv2.putText(
                result,
                label,
                (x1, y1 - baseline - 2),
                self.font,
                self.font_scale,
                (255, 255, 255),
                self.thickness
            )
        
        return result
    
    def draw_stream_label(
        self,
        frame: np.ndarray,
        stream_id: int,
        video_fps: Optional[float] = None,
        process_fps: Optional[float] = None
    ) -> np.ndarray:
        """
        Draw stream ID and FPS on frame.

        Args:
            frame: Input frame
            stream_id: Stream identifier
            video_fps: Original video fps
            process_fps: Processing fps

        Returns:
            Frame with label
        """
        result = frame.copy()

        # Build label text
        label = f"Stream {stream_id}"
        if video_fps is not None:
            label += f" | Vid:{video_fps:.0f}"
        if process_fps is not None:
            label += f" Inf:{process_fps:.0f}"

        # Draw semi-transparent background (only one copy needed)
        cv2.rectangle(result, (0, 0), (200, 30), (0, 0, 0), -1)
        cv2.addWeighted(result, 0.6, frame, 0.4, 0, result)
        
        # Draw text
        cv2.putText(
            result,
            label,
            (10, 22),
            self.font,
            0.6,
            (255, 255, 255),
            2
        )
        
        return result


class GridDisplay:
    """
    Display multiple video streams in a grid layout.
    """

    # RK3588 VPU encoding: width must be 16-aligned, height must be 2-aligned
    ALIGN_W = 16
    ALIGN_H = 2

    def __init__(self, config: Config, num_streams: int = 9):
        """
        Initialize grid display.

        Args:
            config: Configuration object
            num_streams: Number of streams to display
        """
        self.config = config
        self.num_streams = num_streams
        self.visualizer = Visualizer(config)

        # Calculate grid dimensions
        self.grid_rows = int(np.ceil(np.sqrt(num_streams)))
        self.grid_cols = int(np.ceil(num_streams / self.grid_rows))

        # Display settings
        self.window_name = "Multi-Stream Detection"

        # Get screen size and calculate max grid size
        screen_w, screen_h = self._get_screen_size()
        screen_w = int(screen_w * 0.98)
        screen_h = int(screen_h * 0.98)

        # Calculate max cell size that fits screen (for reference)
        max_cell_w = screen_w // self.grid_cols
        max_cell_h = screen_h // self.grid_rows

        # Use max possible cell size while keeping 16:9 aspect ratio
        aspect_ratio = 16 / 9
        if max_cell_w / max_cell_h > aspect_ratio:
            cell_h = max_cell_h
            cell_w = int(cell_h * aspect_ratio)
        else:
            cell_w = max_cell_w
            cell_h = int(cell_w / aspect_ratio)

        # Align to VPU requirements
        cell_w = (cell_w // self.ALIGN_W) * self.ALIGN_W
        cell_h = (cell_h // self.ALIGN_H) * self.ALIGN_H

        self.cell_size = (cell_w, cell_h)

    def _get_screen_size(self):
        """Get screen resolution using xrandr."""
        try:
            import subprocess
            output = subprocess.check_output(['xrandr'], encoding='utf-8')
            for line in output.split('\n'):
                if '*' in line:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if '*' in p and i > 0:
                            size = parts[i-1].split('x')
                            if len(size) == 2:
                                return int(size[0]), int(size[1])
        except:
            pass
        return 1920, 1080  # Default fallback
    
    def create_grid(
        self,
        frames: List[np.ndarray],
        # all_detections: List[List[Tuple[int, int, int, int, float, int]]],
        video_fps_values: Optional[List[float]] = None,
        process_fps_values: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Create a grid display from multiple frames.

        Args:
            frames: List of frames
            all_detections: List of detections per frame
            video_fps_values: Original video fps per stream
            process_fps_values: Processing fps per stream

        Returns:
            Combined grid image
        """
        # Create empty grid
        grid_h = self.grid_rows * self.cell_size[1]
        grid_w = self.grid_cols * self.cell_size[0]
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        for idx in range(self.num_streams):
            row = idx // self.grid_cols
            col = idx % self.grid_cols
            
            if idx < len(frames) and frames[idx] is not None:
                # Frame already has detections drawn in main loop, just add label
                frame = frames[idx]
                orig_h, orig_w = frame.shape[:2]

                # Draw stream label with both video fps and process fps
                video_fps = video_fps_values[idx] if video_fps_values else None
                proc_fps = process_fps_values[idx] if process_fps_values else None
                frame = self.visualizer.draw_stream_label(frame, idx, video_fps, proc_fps)

                # Resize maintaining aspect ratio, no upscaling
                cell_w, cell_h = self.cell_size
                frame_aspect = orig_w / orig_h
                cell_aspect = cell_w / cell_h

                if frame_aspect > cell_aspect:
                    # Frame is wider - fit to width
                    resize_w = cell_w
                    resize_h = int(cell_w / frame_aspect)
                else:
                    # Frame is taller - fit to height
                    resize_h = cell_h
                    resize_w = int(cell_h * frame_aspect)

                # Don't upscale
                if resize_w > orig_w or resize_h > orig_h:
                    resize_w = orig_w
                    resize_h = orig_h

                # Resize and center in cell
                resized = cv2.resize(frame, (resize_w, resize_h))
                cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                y_offset = (cell_h - resize_h) // 2
                x_offset = (cell_w - resize_w) // 2
                cell[y_offset:y_offset+resize_h, x_offset:x_offset+resize_w] = resized
                frame = cell
            else:
                # Empty cell with stream number
                frame = np.zeros((self.cell_size[1], self.cell_size[0], 3), dtype=np.uint8)
                cv2.putText(
                    frame,
                    f"Stream {idx}",
                    (self.cell_size[0] // 2 - 50, self.cell_size[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )
            
            # Place in grid
            y_start = row * self.cell_size[1]
            y_end = y_start + self.cell_size[1]
            x_start = col * self.cell_size[0]
            x_end = x_start + self.cell_size[0]
            grid[y_start:y_end, x_start:x_end] = frame
        
        return grid
    
    def show(self, grid: np.ndarray, delay: int = 1) -> bool:
        """
        Display grid in window.

        Args:
            grid: Grid image to display
            delay: Wait time in ms

        Returns:
            False if 'q' was pressed, True otherwise
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(self.window_name, grid)
        key = cv2.waitKey(delay) & 0xFF
        return key != ord('q')

    def destroy(self):
        """Close display window."""
        cv2.destroyAllWindows()


class VideoWriter:
    def __init__(
        self,
        output_path: str,
        fps: float,
        frame_size: Tuple[int, int],
        codec: str = "mp4v",
        use_vpu: bool = False
    ):
        self.output_path = output_path
        
        self.fps = fps
        print(self.fps)
        
        self.codec = codec
        self.use_vpu = use_vpu

        # RK3588 硬规则：宽度必须 16 对齐，高度 2 对齐
        w, h = frame_size
        self.w = (w + 15) // 16 * 16
        self.h = (h + 15) // 16 * 16
        self.frame_size = (self.w, self.h)

        if use_vpu:
            if codec.lower() in ["mp4v", "avc1", "h264"]:
                enc = "mpph264enc"
                parser = "h264parse"
            elif codec.lower() in ["hev1", "hvc1", "h265"]:
                enc = "mpph265enc"
                parser = "h265parse"
            else:
                use_vpu = False

            if use_vpu:
                pipeline = (
                    f"appsrc ! "
                    f"video/x-raw,format=BGR,width={self.w},height={self.h},framerate=30/1 ! "
                    f"videoconvert ! video/x-raw,format=I420 ! "
                    f"mpph264enc ! h264parse ! mp4mux ! "
                    f"filesink location={self.output_path} sync=false"
                )
                self.writer = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, self.fps, self.frame_size)

                if self.writer.isOpened():
                    print(f"Use RK3588 VPU Decode: {enc}")
                else:
                    print("Can not open VPU encoder, falling back to software encoding")
                    use_vpu = False

        if not use_vpu:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)
            print("Use software encoding")

    def write(self, frame: np.ndarray):
        frame = cv2.resize(frame, (self.w, self.h))
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        self.writer.write(frame)

    def release(self):
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
