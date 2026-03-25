"""Configuration for RKNN multi-stream video detection."""

from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    """Configuration class for RKNN multi-stream detection."""

    # Model settings
    model_path: str = "yolov8n.rknn"
    input_size: int = 640
    conf_threshold: float = 0.4
    iou_threshold: float = 0.45

    # Video settings
    video_dir: str = "video"
    output_dir: str = "output"
    num_streams: int = 6

    # RKNN settings
    num_cores: int = 3  # Use 3 NPU cores on RK3588
    batch_size: int = 1

    # Output settings
    save_video: bool = False
    display_results: bool = True

    # COCO class names (80 classes)
    class_names: List[str] = None

    max_frames: int = None  # For testing, limit number of frames to process

    num_postprocess: int = 3  # Number of postprocess threads

    def __post_init__(self):
        if self.class_names is None:
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
