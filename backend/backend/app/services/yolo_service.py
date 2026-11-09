import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from app.services.yolox_detector import YoloxDetector as YOLO

from app.config import (
    YOLO_CONFIDENCE,
    YOLO_IOU,
    YOLO_MODEL_PATH,
)

logger = logging.getLogger(__name__)

# COCO dataset class index for 'person'
PERSON_CLASS_ID = 0
PERSON_CLASS_NAME = "person"


@dataclass
class PersonDetection:
    """Structured representation of a single detected person."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int = PERSON_CLASS_ID
    class_name: str = PERSON_CLASS_NAME
    area: int = 0

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": round(float(self.confidence), 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "area": self.area,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class YOLODetectionResult:
    """Structured result containing all person detections and primary person metadata."""

    detections: List[PersonDetection] = field(default_factory=list)
    primary_person: Optional[PersonDetection] = None
    count: int = 0
    image_width: int = 0
    image_height: int = 0
    device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "primary_person": self.primary_person.to_dict() if self.primary_person else None,
            "count": self.count,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "device": self.device,
        }


class YOLOService:
    """
    Isolated, reusable person-detection service (YOLOX-S on OpenCV DNN).
    
    Loads the YOLO model ONCE and reuses the loaded weights across requests.
    Runs on the CPU through OpenCV, so no torch is needed.
    Never alters user images or draws graphics.
    """

    _model_cache: Dict[str, YOLO] = {}
    _cache_lock = threading.Lock()

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: Optional[float] = None,
        iou: Optional[float] = None,
    ):
        raw_path = model_path or YOLO_MODEL_PATH or "assets/yolox_s.onnx"
        self.model_path = self._resolve_model_path(raw_path)
        self.confidence = float(confidence if confidence is not None else YOLO_CONFIDENCE)
        self.iou = float(iou if iou is not None else YOLO_IOU)
        self.device = "cpu"
        self.model = self._get_or_load_model(self.model_path, self.device)

    @classmethod
    def _resolve_model_path(cls, path_str: str) -> str:
        """Resolve model path against common locations (backend root, assets, or CWD)."""
        if os.path.isabs(path_str) and os.path.exists(path_str):
            return path_str

        # Check relative to backend directory
        backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        candidate_in_backend = os.path.join(backend_dir, path_str)
        if os.path.exists(candidate_in_backend):
            return os.path.abspath(candidate_in_backend)

        # Check relative to current working directory
        if os.path.exists(path_str):
            return os.path.abspath(path_str)

        # Not found: returned as given so the load error names the missing file
        return path_str

    @classmethod
    def _get_or_load_model(cls, model_path: str, device: str) -> YOLO:
        """Load the YOLO model ONCE and cache it for all subsequent service instances."""
        cache_key = f"{model_path}:{device}"
        with cls._cache_lock:
            if cache_key in cls._model_cache:
                logger.debug(f"Reusing cached YOLO model for {cache_key}")
                return cls._model_cache[cache_key]

            logger.info(f"Loading YOLO model from '{model_path}' on device '{device}'...")
            model = YOLO(model_path)
            # Pre-transfer model weights to target device if supported
            try:
                model.to(device)
            except Exception as e:
                logger.warning(f"Could not explicitly move YOLO model to {device}: {e}")

            cls._model_cache[cache_key] = model
            logger.info(f"YOLO model loaded successfully from '{model_path}' on device '{device}'")
            return model

    @staticmethod
    def clamp_bbox(
        x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int
    ) -> Tuple[int, int, int, int]:
        """Clamp bounding box coordinates strictly within image boundaries [0, W] and [0, H]."""
        cx1 = max(0, min(int(round(x1)), img_w))
        cy1 = max(0, min(int(round(y1)), img_h))
        cx2 = max(0, min(int(round(x2)), img_w))
        cy2 = max(0, min(int(round(y2)), img_h))

        # Enforce canonical ordering
        if cx1 > cx2:
            cx1, cx2 = cx2, cx1
        if cy1 > cy2:
            cy1, cy2 = cy2, cy1

        return cx1, cy1, cx2, cy2

    @staticmethod
    def is_valid_bbox(x1: int, y1: int, x2: int, y2: int, min_area: int = 1) -> bool:
        """Validate that a bounding box has positive dimensions and minimum area."""
        w = x2 - x1
        h = y2 - y1
        return w > 0 and h > 0 and (w * h) >= min_area

    @classmethod
    def select_primary_person(
        cls, detections: List[PersonDetection]
    ) -> Optional[PersonDetection]:
        """
        Deterministically select the primary person from multiple detections.
        
        Selection Policy:
        - Validates all bounding boxes.
        - Ranks candidates by prominence score: score = area * (confidence ** 2).
          This prioritizes foreground human subjects occupying significant frame area
          while weighting detection certainty.
        - Resolves ties using confidence followed by bounding box area.
        - Safely returns None if detections list is empty.
        """
        valid_dets = [d for d in detections if cls.is_valid_bbox(d.x1, d.y1, d.x2, d.y2)]
        if not valid_dets:
            return None

        # Sort deterministically descending
        sorted_dets = sorted(
            valid_dets,
            key=lambda d: (d.area * (d.confidence ** 2), d.confidence, d.area),
            reverse=True,
        )
        return sorted_dets[0]

    def detect_person(
        self,
        image: Union[Image.Image, np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> YOLODetectionResult:
        """
        Run YOLO inference to detect people in the provided image.

        Args:
            image: PIL.Image.Image or numpy.ndarray (RGB / BGR OpenCV format).
            conf: Optional confidence threshold override.
            iou: Optional IoU threshold override.

        Returns:
            YOLODetectionResult containing all detected people, primary person, and metadata.
            Never modifies the input image.
        """
        if image is None:
            raise ValueError("Input image cannot be None")

        # Extract image dimensions safely without altering the input
        if isinstance(image, Image.Image):
            img_w, img_h = image.size
        elif isinstance(image, np.ndarray):
            if image.ndim < 2:
                raise ValueError("NumPy image must have at least 2 dimensions (H, W)")
            img_h, img_w = image.shape[:2]
        else:
            raise TypeError(f"Unsupported image type: {type(image)}. Expected PIL.Image.Image or np.ndarray.")

        if img_w <= 0 or img_h <= 0:
            return YOLODetectionResult(
                detections=[],
                primary_person=None,
                count=0,
                image_width=img_w,
                image_height=img_h,
                device=self.device,
            )

        eff_conf = float(conf if conf is not None else self.confidence)
        eff_iou = float(iou if iou is not None else self.iou)

        # Run inference filtering for person class (COCO class 0)
        # verbose=False ensures clean server logs without spamming per-token progress
        results = self.model.predict(
            source=image,
            conf=eff_conf,
            iou=eff_iou,
            classes=[PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
        )

        detections: List[PersonDetection] = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy_tensor = np.asarray(boxes.xyxy)
                conf_tensor = np.asarray(boxes.conf)
                cls_tensor = np.asarray(boxes.cls)

                for i in range(len(boxes)):
                    c_id = int(cls_tensor[i])
                    # Verify class is person
                    if c_id != PERSON_CLASS_ID:
                        continue

                    raw_x1, raw_y1, raw_x2, raw_y2 = xyxy_tensor[i]
                    c_conf = float(conf_tensor[i])

                    # Clamp strictly to image boundaries
                    cx1, cy1, cx2, cy2 = self.clamp_bbox(raw_x1, raw_y1, raw_x2, raw_y2, img_w, img_h)

                    # Validate box
                    if not self.is_valid_bbox(cx1, cy1, cx2, cy2):
                        continue

                    box_area = (cx2 - cx1) * (cy2 - cy1)

                    det = PersonDetection(
                        x1=cx1,
                        y1=cy1,
                        x2=cx2,
                        y2=cy2,
                        confidence=c_conf,
                        class_id=PERSON_CLASS_ID,
                        class_name=PERSON_CLASS_NAME,
                        area=box_area,
                    )
                    detections.append(det)

        # Deterministically select primary person
        primary = self.select_primary_person(detections)

        # Structured logging without exposing user image data
        logger.info(f"number of person detections: {len(detections)}")
        if primary is not None:
            logger.info(
                f"primary person confidence: {primary.confidence:.4f} "
                f"(bbox: [{primary.x1}, {primary.y1}, {primary.x2}, {primary.y2}], area: {primary.area})"
            )
        else:
            logger.info("primary person confidence: N/A (0 person detections)")

        return YOLODetectionResult(
            detections=detections,
            primary_person=primary,
            count=len(detections),
            image_width=img_w,
            image_height=img_h,
            device=self.device,
        )

    def get_primary_person(
        self,
        image: Union[Image.Image, np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> Optional[PersonDetection]:
        """Convenience helper to detect and return ONLY the primary person, or None."""
        result = self.detect_person(image, conf=conf, iou=iou)
        return result.primary_person

    def get_all_people(
        self,
        image: Union[Image.Image, np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> List[PersonDetection]:
        """Convenience helper to return all detected people."""
        result = self.detect_person(image, conf=conf, iou=iou)
        return result.detections


# Global thread-safe singleton instance
_GLOBAL_YOLO_SERVICE: Optional[YOLOService] = None
_GLOBAL_LOCK = threading.Lock()


def get_yolo_service() -> YOLOService:
    """Access or lazily initialize the singleton YOLOService instance."""
    global _GLOBAL_YOLO_SERVICE
    if _GLOBAL_YOLO_SERVICE is None:
        with _GLOBAL_LOCK:
            if _GLOBAL_YOLO_SERVICE is None:
                _GLOBAL_YOLO_SERVICE = YOLOService()
    return _GLOBAL_YOLO_SERVICE
