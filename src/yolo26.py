import numpy as np
import cv2
from typing import List, Tuple


class YOLO26Tool:
    def __init__(self, config):
        self.config = config
        self.strides = [8, 16, 32]

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = min(self.config.input_size / h, self.config.input_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        padded = np.full((self.config.input_size, self.config.input_size, 3), 114, dtype=np.uint8)
        pad_x = (self.config.input_size - new_w) // 2
        pad_y = (self.config.input_size - new_h) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

        return padded

    def postprocess(
        self,
        outputs: List[np.ndarray],
        original_shapes: List[Tuple[int, int]]
    ) -> List[List[Tuple[int, int, int, int, float, int]]]:
        if len(outputs) < 6:
            return [[] for _ in original_shapes]

        results = []
        for i, (orig_h, orig_w) in enumerate(original_shapes):
            single_output = [out[i:i+1] for out in outputs]
            results.append(self._decode_single(single_output, (orig_h, orig_w)))
        return results

    def _decode_single(
        self,
        outputs: List[np.ndarray],
        original_shape: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int, float, int]]:
        h0, w0 = original_shape
        scale = min(self.config.input_size / h0, self.config.input_size / w0)
        pad_x = (self.config.input_size - w0 * scale) / 2
        pad_y = (self.config.input_size - h0 * scale) / 2

        all_boxes, all_scores, all_classes = [], [], []

        for i in range(3):
            feat = outputs[i][0]       # (4, h, w)
            cls = outputs[i + 3][0]   # (80, h, w)
            _, h, w = feat.shape
            stride = self.strides[i]

            # Vectorized: compute all coordinates at once
            yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            cx = (xx + 0.5) * stride
            cy = (yy + 0.5) * stride

            # Box coordinates: (l, t, r, b) -> (x1, y1, x2, y2)
            l, t, r, b = feat[0].ravel(), feat[1].ravel(), feat[2].ravel(), feat[3].ravel()
            cx_flat, cy_flat = cx.ravel(), cy.ravel()

            x1 = cx_flat - l * stride
            y1 = cy_flat - t * stride
            x2 = cx_flat + r * stride
            y2 = cy_flat + b * stride

            # Scale to original image
            x1 = np.clip((x1 - pad_x) / scale, 0, w0)
            y1 = np.clip((y1 - pad_y) / scale, 0, h0)
            x2 = np.clip((x2 - pad_x) / scale, 0, w0)
            y2 = np.clip((y2 - pad_y) / scale, 0, h0)

            boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

            # Vectorized sigmoid + class selection
            cls_reshaped = cls.reshape(-1, cls.shape[0]).T  # (80, h*w)
            cls_softmax = 1 / (1 + np.exp(-cls_reshaped))   # sigmoid
            scores_all = cls_softmax.max(axis=0)              # (h*w,)
            classes_all = cls_softmax.argmax(axis=0).astype(np.int32)  # (h*w,)

            # Filter by confidence
            mask = scores_all >= self.config.conf_threshold
            if mask.any():
                all_boxes.append(boxes[mask])
                all_scores.append(scores_all[mask])
                all_classes.append(classes_all[mask])

        if not all_boxes:
            return []

        boxes = np.concatenate(all_boxes)
        scores = np.concatenate(all_scores)
        classes = np.concatenate(all_classes)

        # Use cv2.dnn.NMSBoxes for efficient NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            score_threshold=self.config.conf_threshold,
            nms_threshold=self.config.iou_threshold
        )

        if len(indices) == 0:
            return []

        indices = indices.flatten() if indices.ndim > 1 else indices

        detections = []
        for idx in indices:
            x1, y1, x2, y2 = boxes[idx]
            detections.append((int(x1), int(y1), int(x2), int(y2), float(scores[idx]), int(classes[idx])))
        return detections
