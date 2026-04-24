"""Configuration for RKNN multi-stream video detection."""

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class Config:
    """Configuration class for RKNN multi-stream detection."""

    # Model settings
    soc: str = "rk3588"
    model_type: str = "yolov8"
    model_path: str = "yolov8n.rknn"
    input_size: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.7

    # Video settings
    video_dir: str = "video"
    output_dir: str = "output"
    num_streams: int = 6
    camera_indexes: Optional[List[int]] = None

    # RKNN settings
    num_cores: int = 3  # Use 3 NPU cores on RK3588
    batch_size: int = 1

    # Output settings
    save_video: bool = False
    display_results: bool = True
    label_file: Optional[str] = None

    # COCO class names (80 classes)
    class_names: List[str] = None

    max_frames: int = None  # For testing, limit number of frames to process

    num_postprocess: int = 3  # Number of postprocess threads

    use_vpu: bool = True  # Whether to use VPU for inference

    def __post_init__(self):
        if self.label_file:
            label_path = Path(self.label_file)
            if not label_path.is_file():
                raise FileNotFoundError(f"Label file not found: {self.label_file}")
            with label_path.open('r', encoding='utf-8') as f:
                labels = [line.strip() for line in f if line.strip()]
            if not labels:
                raise ValueError(f"Label file is empty: {self.label_file}")
            self.class_names = labels
        elif self.class_names is None:
            self.class_names = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]
