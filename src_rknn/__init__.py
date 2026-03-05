"""RKNN multi-stream detection package."""

from src_rknn.config import Config
from src_rknn.rknn_inference import YOLOv8RKNN, post_process
# from src_rknn.processor_rknn import MultiStreamDetector, StreamProcessor
# from src_rknn.processor_rknn_multicore import MultiCoreDetector
# from src_rknn.processor_rknn_batch4 import BatchRKNNDetector
# from src_rknn.processor_rknn_async import AsyncDetector

__all__ = ['Config', 'YOLOv8RKNN', 'post_process', 'MultiStreamDetector', 'StreamProcessor', 'MultiCoreDetector', 'BatchRKNNDetector', 'AsyncDetector']
