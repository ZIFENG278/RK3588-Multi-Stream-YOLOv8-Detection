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

        # reference preprocess: float32 + [0,1] + NCHW
        return np.transpose(padded.astype(np.float32) / 255.0, (2, 0, 1))

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

        raw = []
        for i in range(3):
            feat = outputs[i][0]
            cls = outputs[i + 3][0]
            _, h, w = feat.shape
            stride = self.strides[i]

            for yy in range(h):
                for xx in range(w):
                    cls_vec = self._sigmoid(cls[:, yy, xx])
                    best_cls = int(np.argmax(cls_vec))
                    best_conf = float(np.max(cls_vec))
                    if best_conf < self.config.conf_threshold:
                        continue

                    l, t, r, b = feat[0, yy, xx], feat[1, yy, xx], feat[2, yy, xx], feat[3, yy, xx]
                    cx = (xx + 0.5) * stride
                    cy = (yy + 0.5) * stride

                    x1 = cx - l * stride
                    y1 = cy - t * stride
                    x2 = cx + r * stride
                    y2 = cy + b * stride

                    x1 = np.clip((x1 - pad_x) / scale, 0, w0)
                    y1 = np.clip((y1 - pad_y) / scale, 0, h0)
                    x2 = np.clip((x2 - pad_x) / scale, 0, w0)
                    y2 = np.clip((y2 - pad_y) / scale, 0, h0)

                    raw.append([x1, y1, x2, y2, best_conf, best_cls])

        if not raw:
            return []

        raw = np.array(raw, dtype=np.float32)
        keep = self._nms_indices(raw[:, :4], raw[:, 4], self.config.iou_threshold)

        detections = []
        for idx in keep:
            x1, y1, x2, y2, score, cls_id = raw[idx]
            detections.append((int(x1), int(y1), int(x2), int(y2), float(score), int(cls_id)))
        return detections

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return inter / (area1 + area2 - inter + 1e-6)

    def _nms_indices(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        order = scores.argsort()[::-1]
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            ious = self._iou(boxes[i], boxes[order[1:]])
            order = order[1:][ious < iou_threshold]
        return np.array(keep, dtype=np.int32)
