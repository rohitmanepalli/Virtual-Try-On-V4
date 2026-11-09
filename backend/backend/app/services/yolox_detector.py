"""
Person detector: YOLOX-S (Apache 2.0, OpenCV model zoo) run by OpenCV's DNN module, so no torch is needed.

Exposes the slice of the Ultralytics API the YOLO service uses: `.task`, `.to(device)` and `.predict(...)` returning
results with `.boxes.xyxy/.conf/.cls` as numpy arrays. Class ids follow 80-class COCO (person == 0).
Logic follows the OpenCV zoo reference (models/object_detection_yolox/yolox.py).
"""
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

INPUT_SIZE = 640
STRIDES = (8, 16, 32)
PAD_VALUE = 114  # YOLOX letterbox grey


def _anchors():
    grids, strides = [], []
    for s in STRIDES:
        n = INPUT_SIZE // s
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((n * n, 1), s))
    return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)


class YoloxDetector:
    task = "detect"

    def __init__(self, model_path: str):
        self.net = cv2.dnn.readNet(model_path)
        self.grids, self.strides = _anchors()
        self.device = "cpu"

    def to(self, device: str):
        return self  # OpenCV DNN runs on the CPU here

    @staticmethod
    def _letterbox(bgr: np.ndarray):
        h, w = bgr.shape[:2]
        ratio = min(INPUT_SIZE / h, INPUT_SIZE / w)
        resized = cv2.resize(bgr, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), PAD_VALUE, dtype=np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        return canvas, ratio

    def predict(self, source, conf=0.25, iou=0.45, classes=None, device=None, verbose=False, **_):
        if isinstance(source, Image.Image):
            bgr = cv2.cvtColor(np.asarray(source.convert("RGB")), cv2.COLOR_RGB2BGR)
        else:
            # Ultralytics treats ndarray input as OpenCV BGR; keep that contract
            bgr = source if source.ndim == 3 else cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        img, ratio = self._letterbox(bgr)
        self.net.setInput(np.transpose(img.astype(np.float32), (2, 0, 1))[np.newaxis])
        dets = self.net.forward(self.net.getUnconnectedOutLayersNames())[0][0]

        centers = (dets[:, :2] + self.grids) * self.strides
        sizes = np.exp(dets[:, 2:4]) * self.strides
        scores = dets[:, 4:5] * dets[:, 5:]
        cls = scores.argmax(1)
        best = scores.max(1)

        wanted = best >= conf
        if classes is not None:
            wanted &= np.isin(cls, classes)
        xywh = np.concatenate([centers - sizes / 2, sizes], 1)[wanted]
        best, cls = best[wanted], cls[wanted]
        keep = np.asarray(cv2.dnn.NMSBoxes(xywh.tolist(), best.tolist(), conf, iou), dtype=int).reshape(-1)

        xyxy = np.concatenate([xywh[keep, :2], xywh[keep, :2] + xywh[keep, 2:]], 1) / ratio  # back to source pixels
        return [SimpleNamespace(boxes=_Boxes(xyxy, best[keep], cls[keep].astype(np.float32)), masks=None)]


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = xyxy, conf, cls

    def __len__(self):
        return len(self.conf)
