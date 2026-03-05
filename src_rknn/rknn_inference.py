"""RKNN inference engine for YOLOv8 on RK3588."""

import numpy as np
import cv2
from typing import List, Tuple, Optional
import time

# Try to import RKNN, fail gracefully if not available
try:
    from rknn.api import RKNN
    from py_utils.rknn_executor import RKNN_model_container
    RKNN_AVAILABLE = True
except ImportError:
    RKNN_AVAILABLE = False


# YOLOv8 COCO classes
COCO_CLASSES = [
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


def dfl(position: np.ndarray) -> np.ndarray:
    """Distribution Focal Loss - converts distribution to box coordinates.

    Pure numpy implementation (faster than torch).

    Args:
        position: Input tensor [N, 4*16, H, W] (for YOLOv8)

    Returns:
        Processed tensor
    """
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num

    # Reshape: [N, 4, 16, H, W]
    x = position.reshape(n, p_num, mc, h, w)

    # Softmax on axis 2 (the 16 dimension)
    x_max = x.max(axis=2, keepdims=True)
    exp_x = np.exp(x - x_max)
    y = exp_x / exp_x.sum(axis=2, keepdims=True)

    # Weighted sum: [1, 1, 16, 1, 1]
    acc_matrix = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    y = (y * acc_matrix).sum(axis=2)

    return y


def box_process(position: np.ndarray, img_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """Process box predictions from YOLOv8 output.

    Args:
        position: Box predictions [N, 4, H, W]
        img_size: Target image size (width, height)

    Returns:
        Processed boxes in xyxy format [N, 4, H, W]
    """
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([img_size[1] // grid_h, img_size[0] // grid_w]).reshape(1, 2, 1, 1)

    # Apply DFL
    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    return xyxy


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.45) -> np.ndarray:
    """Non-maximum suppression using OpenCV (much faster).

    Args:
        boxes: Bounding boxes [N, 4] in xyxy format
        scores: Confidence scores [N]
        threshold: IoU threshold

    Returns:
        Indices of boxes to keep
    """
    # Use OpenCV NMS
    # Note: OpenCV NMSBoxes requires scores to be 1D
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        scores.tolist(),
        score_threshold=0.0,  # Already filtered
        nms_threshold=threshold
    )
    if len(indices) == 0:
        return np.array([], dtype=np.int32)
    return indices.flatten() if indices.ndim > 1 else indices


def post_process(
    outputs: List[np.ndarray],
    obj_thresh: float = 0.25,
    nms_thresh: float = 0.45,
    img_size: Tuple[int, int] = (640, 640)
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Post-process YOLOv8 RKNN model outputs.

    Args:
        outputs: List of model outputs (9 outputs for YOLOv8)
        obj_thresh: Object confidence threshold
        nms_thresh: NMS IoU threshold
        img_size: Input image size (width, height)

    Returns:
        boxes: [N, 4] in xyxy format
        classes: [N] class indices
        scores: [N] confidence scores
    """
    boxes_list, scores_list, classes_conf_list = [], [], []
    default_branch = 3
    pair_per_branch = len(outputs) // default_branch

    # Process each detection scale
    for i in range(default_branch):
        boxes_list.append(box_process(outputs[pair_per_branch * i], img_size))
        classes_conf_list.append(outputs[pair_per_branch * i + 1])
        scores_list.append(np.ones_like(
            outputs[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32
        ))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    # Flatten and concatenate
    boxes = np.concatenate([sp_flatten(_v) for _v in boxes_list])
    classes_conf = np.concatenate([sp_flatten(_v) for _v in classes_conf_list])
    scores = np.concatenate([sp_flatten(_v) for _v in scores_list])

    # Filter by confidence threshold
    box_confidences = scores.reshape(-1)
    class_max_score = np.max(classes_conf, axis=-1)
    classes = np.argmax(classes_conf, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= obj_thresh)
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    scores = (class_max_score * box_confidences)[_class_pos]

    if len(scores) == 0:
        return None, None, None

    # NMS per class
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c_arr = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s, nms_thresh)
        if len(keep) > 0:
            nboxes.append(b[keep])
            nclasses.append(c_arr[keep])
            nscores.append(s[keep])

    if not nclasses:
        return None, None, None

    return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)


class YOLOv8RKNN:
    """YOLOv8 RKNN inference engine for RK3588.

    Supports multi-core NPU inference with batch processing.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        num_cores: int = 3,
        core_id: int = 0
    ):
        """Initialize RKNN inference engine.

        Args:
            model_path: Path to .rknn model file
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            input_size: Input image size
            num_cores: Number of NPU cores to use (1-3)
            core_id: Core ID for single-core mode (0, 1, or 2)
        """
        if not RKNN_AVAILABLE:
            raise RuntimeError("RKNN not available. Install rknn-toolkit2.")

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.num_cores = num_cores
        self.core_id = core_id

        # Load models on each NPU core
        self.models: List[RKNN_model_container] = []
        self._load_models()

        # Timing stats
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0
        self.detect_count = 0

    def _load_models(self):
        """Load RKNN models on each NPU core."""
        # NPU core constants
        core_masks = [RKNN.NPU_CORE_0, RKNN.NPU_CORE_1, RKNN.NPU_CORE_2]

        if self.num_cores == 1:
            # Single core mode: load on specific core_id
            core_mask = core_masks[self.core_id]
            print(f"Loading RKNN model on core {self.core_id} (mask={core_mask})...")
            model = RKNN_model_container(
                self.model_path,
                target='rk3588',
                core_mask=core_mask
            )
            self.models.append(model)
            print(f"  Core {self.core_id}: Model loaded")
        # else:
        #     # Multi-core mode: load on all cores
        #     print(f"Loading RKNN model on {self.num_cores} NPU cores...")
        #     for i in range(self.num_cores):
        #         core_mask = core_masks[i]
        #         print(f"  Loading on core {i} (mask={core_mask})...")
        #         model = RKNN_model_container(
        #             self.model_path,
        #             target='rk3588',
        #             core_mask=core_mask
        #         )
        #         self.models.append(model)
        #         print(f"  Core {i}: Model loaded")

        print(f"Successfully loaded {len(self.models)} models")

    # def preprocess(self, images: List[np.ndarray]) -> np.ndarray:
    #     """Preprocess images for inference.

    #     Args:
    #         images: List of BGR images

    #     Returns:
    #         Preprocessed batch [N, H, W, 3] in RGB format
    #     """
    #     import cv2
    #     start = time.time()

    #     batch = []
    #     for img in images:
    #         # Letterbox resize
    #         h, w = img.shape[:2]
    #         scale = min(self.input_size / h, self.input_size / w)
    #         new_h, new_w = int(h * scale), int(w * scale)

    #         resized = cv2.resize(img, (new_w, new_h))

    #         # Create padded image
    #         padded = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
    #         pad_h = (self.input_size - new_h) // 2
    #         pad_w = (self.input_size - new_w) // 2
    #         padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized

    #         # BGR to RGB
    #         rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    #         batch.append(rgb)

    #     result = np.stack(batch, axis=0)
    #     self.preprocess_time += time.time() - start
    #     return result

    def infer(self, batch: np.ndarray, core_id: int = 0) -> List[np.ndarray]:
        """Run inference on a specific NPU core.

        Args:
            batch: Preprocessed batch [N, H, W, 3]
            core_id: NPU core ID (0, 1, or 2)

        Returns:
            List of model outputs
        """
        start = time.time()
        outputs = self.models[core_id].run([batch])
        self.inference_time += time.time() - start
        return outputs

    def postprocess(
        self,
        outputs: List[np.ndarray],
        original_shapes: List[Tuple[int, int]]
    ) -> List[List[Tuple[int, int, int, int, float, int]]]:
        """Post-process model outputs.

        Args:
            outputs: Model outputs
            original_shapes: Original image shapes

        Returns:
            List of detections per image: [(x1, y1, x2, y2, conf, class), ...]
        """
        start = time.time()
        batch_size = len(original_shapes)
        results = []

        for i in range(batch_size):
            # Extract single image output
            single_output = [out[i:i+1] for out in outputs]
            boxes, classes, scores = post_process(
                single_output,
                self.conf_threshold,
                self.iou_threshold,
                (self.input_size, self.input_size)
            )

            if boxes is None:
                results.append([])
                continue

            # Scale boxes back to original size
            orig_h, orig_w = original_shapes[i]
            scale = min(self.input_size / orig_h, self.input_size / orig_w)
            pad_x = (self.input_size - orig_w * scale) / 2
            pad_y = (self.input_size - orig_h * scale) / 2

            detections = []
            for j in range(len(boxes)):
                x1, y1, x2, y2 = boxes[j]
                # Remove padding and scale
                x1 = int((x1 - pad_x) / scale)
                y1 = int((y1 - pad_y) / scale)
                x2 = int((x2 - pad_x) / scale)
                y2 = int((y2 - pad_y) / scale)

                # Clip to image bounds
                x1 = max(0, min(x1, orig_w))
                y1 = max(0, min(y1, orig_h))
                x2 = max(0, min(x2, orig_w))
                y2 = max(0, min(y2, orig_h))

                detections.append((x1, y1, x2, y2, float(scores[j]), int(classes[j])))

            results.append(detections)

        self.postprocess_time += time.time() - start
        self.detect_count += batch_size
        return results

    # def detect(
    #     self,
    #     images: List[np.ndarray],
    #     core_id: int = 0,
    #     original_shapes: Optional[List[Tuple[int, int]]] = None
    # ) -> List[List[Tuple[int, int, int, int, float, int]]]:
    #     """Full detection pipeline.

    #     Args:
    #         images: List of BGR images
    #         core_id: NPU core to use
    #         original_shapes: Original image shapes (h, w). If None, inferred from images.

    #     Returns:
    #         List of detections per image
    #     """
    #     if len(images) == 0:
    #         return []

    #     if original_shapes is None:
    #         original_shapes = [(img.shape[0], img.shape[1]) for img in images]

    #     batch = self.preprocess(images)
    #     outputs = self.infer(batch, core_id)
    #     return self.postprocess(outputs, original_shapes)

    # def detect_batch(
    #     self,
    #     images_dict: dict
    # ) -> dict:
    #     """Batch detection across multiple NPU cores.

    #     Args:
    #         images_dict: Dict mapping core_id to list of images

    #     Returns:
    #         Dict mapping core_id to detection results
    #     """
    #     results = {}
    #     for core_id, images in images_dict.items():
    #         if len(images) > 0:
    #             results[core_id] = self.detect(images, core_id)
    #     return results

    # def get_timing_stats(self) -> dict:
    #     """Get timing statistics."""
    #     if self.detect_count == 0:
    #         return {}
    #     return {
    #         'avg_preprocess': (self.preprocess_time / self.detect_count) * 1000,
    #         'avg_inference': (self.inference_time / self.detect_count) * 1000,
    #         'avg_postprocess': (self.postprocess_time / self.detect_count) * 1000,
    #     }

    def release(self):
        """Release all models."""
        for model in self.models:
            model.release()
        print("All RKNN models released")
