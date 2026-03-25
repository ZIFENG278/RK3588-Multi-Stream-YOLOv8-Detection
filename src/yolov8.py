import numpy as np
import cv2
from typing import List, Tuple, Optional


class YOLOv8Tool:
    def __init__(self, config):
        self.config = config

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Letterbox preprocess."""
        h, w = frame.shape[:2]
        scale = min(self.config.input_size / h, self.config.input_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        padded = np.full((self.config.input_size, self.config.input_size, 3), 114, dtype=np.uint8)
        pad_x = (self.config.input_size - new_w) // 2
        pad_y = (self.config.input_size - new_h) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        return cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    def postprocess(
        self,
        outputs: List[np.ndarray],
        original_shapes: List[Tuple[int, int]]
    ) -> List[List[Tuple[int, int, int, int, float, int]]]:
        batch_size = len(original_shapes)
        results = []

        for i in range(batch_size):
            single_output = [out[i:i+1] for out in outputs]
            # for output in single_output:
            #     print(output.shape)
            # print("----")
            boxes, classes, scores = self.post_process(
                single_output,
                self.config.conf_threshold,
                self.config.iou_threshold,
                (self.config.input_size, self.config.input_size)
            )


            if boxes is None:
                results.append([])
                continue

            orig_h, orig_w = original_shapes[i]
            scale = min(self.config.input_size / orig_h, self.config.input_size / orig_w)
            pad_x = (self.config.input_size - orig_w * scale) / 2
            pad_y = (self.config.input_size - orig_h * scale) / 2

            detections = []
            for j in range(len(boxes)):
                x1, y1, x2, y2 = boxes[j]
                x1 = int((x1 - pad_x) / scale)
                y1 = int((y1 - pad_y) / scale)
                x2 = int((x2 - pad_x) / scale)
                y2 = int((y2 - pad_y) / scale)

                x1 = max(0, min(x1, orig_w))
                y1 = max(0, min(y1, orig_h))
                x2 = max(0, min(x2, orig_w))
                y2 = max(0, min(y2, orig_h))

                detections.append((x1, y1, x2, y2, float(scores[j]), int(classes[j])))
            results.append(detections)

        return results

    def dfl(self, position: np.ndarray) -> np.ndarray:
        """Distribution Focal Loss."""
        n, c, h, w = position.shape
        p_num = 4
        mc = c // p_num
        x = position.reshape(n, p_num, mc, h, w)
        x_max = x.max(axis=2, keepdims=True)
        exp_x = np.exp(x - x_max)
        y = exp_x / exp_x.sum(axis=2, keepdims=True)
        acc_matrix = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
        return (y * acc_matrix).sum(axis=2)

    def box_process(self, position: np.ndarray, img_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([img_size[1] // grid_h, img_size[0] // grid_w]).reshape(1, 2, 1, 1)
        position = self.dfl(position)
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        return np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    def nms_boxes(self, boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.45) -> np.ndarray:
        """NMS using OpenCV — boxes/scores stay as numpy arrays."""
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            score_threshold=0.0,
            nms_threshold=threshold
        )
        if len(indices) == 0:
            return np.array([], dtype=np.int32)
        return indices.flatten() if indices.ndim > 1 else indices

    def post_process(
        self,
        outputs: List[np.ndarray],
        obj_thresh: float = 0.25,
        nms_thresh: float = 0.45,
        img_size: Tuple[int, int] = (640, 640)
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Post-process YOLOv8 RKNN model outputs."""
        boxes_list, scores_list, classes_conf_list = [], [], []
        default_branch = 3
        pair_per_branch = len(outputs) // default_branch

        for i in range(default_branch):
            boxes_list.append(self.box_process(outputs[pair_per_branch * i], img_size))
            classes_conf_list.append(outputs[pair_per_branch * i + 1])
            scores_list.append(np.ones_like(
                outputs[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32
            ))

        def sp_flatten(_in):
            ch = _in.shape[1]
            return _in.transpose(0, 2, 3, 1).reshape(-1, ch)

        boxes = np.concatenate([sp_flatten(_v) for _v in boxes_list])
        classes_conf = np.concatenate([sp_flatten(_v) for _v in classes_conf_list])
        scores = np.concatenate([sp_flatten(_v) for _v in scores_list])

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
            b, c_arr, s = boxes[inds], classes[inds], scores[inds]
            keep = self.nms_boxes(b, s, nms_thresh)
            if len(keep) > 0:
                nboxes.append(b[keep])
                nclasses.append(c_arr[keep])
                nscores.append(s[keep])

        if not nclasses:
            return None, None, None
        return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)
