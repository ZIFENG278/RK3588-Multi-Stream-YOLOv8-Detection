"""Configuration for multi-stream video detection."""

from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    """Configuration class for multi-stream detection."""
    
    # Model settings
    model_path: str = "yolov8n-640.onnx"
    input_size: int = 640
    conf_threshold: float = 0.4
    iou_threshold: float = 0.45
    
    # Video settings
    video_dir: str = "video"
    output_dir: str = "output"
    num_streams: int = 9
    
    # Processing settings
    batch_size: int = 4  # Process multiple frames in batch
    num_decoder_threads: int = 9  # One per stream
    display_fps: bool = True
    
    # Output settings
    save_video: bool = False
    display_results: bool = True
    
    # COCO class names (80 classes)
    class_names: List[str] = None
    
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


# Default configuration instance
default_config = Config()
