"""
src/detection/detector.py
--------------------------
YOLOv8 vehicle detector with drone / aerial top-view optimization.

Features:
  - COCO → AIC21 class mapping:
      COCO 2 (car)        → Car
      COCO 5 (bus)        → Bus (or counted as Car for AIC21 rules)
      COCO 7 (truck)      → Truck
      COCO 3 (motorcycle) → Motorcycle
      COCO 1 (bicycle)    → Bicycle
  - Drone Aerial Optimization:
      * High-resolution inference (640, 960, 1280) for small aerial vehicles.
      * Drone-tuned confidence & IoU thresholds for roof views and dense queues.
      * Sliced / Tiled Overlapping Inference (SAHI-style) with NMS merging.
      * Letterboxing and aspect ratio preservation.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# COCO ID → AIC21 class mapping
# ---------------------------------------------------------------------------
COCO_TO_AIC21: Dict[int, str] = {
    1: "Bicycle",       # bicycle
    2: "Car",           # car
    3: "Motorcycle",    # motorcycle
    5: "Bus",           # bus
    7: "Truck",         # truck
}

AIC21_CLASS_ID: Dict[str, int] = {
    "Car": 1,
    "Truck": 2,
    "Bus": 3,
    "Motorcycle": 4,
    "Bicycle": 5,
}

TARGET_COCO_IDS = set(COCO_TO_AIC21.keys())  # {1, 2, 3, 5, 7}


@dataclass
class Detection:
    """Single vehicle detection on one frame."""
    frame_id: int
    timestamp_sec: float
    coco_class_id: int
    aic21_class_name: str       # "Car", "Truck", "Bus", "Motorcycle", "Bicycle"
    aic21_class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def bottom_center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, float(self.y2))

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def is_countable(self) -> bool:
        return self.aic21_class_name in ("Car", "Bus", "Truck", "Motorcycle", "Bicycle")

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "aic21_class_name": self.aic21_class_name,
            "aic21_class_id": self.aic21_class_id,
            "confidence": round(self.confidence, 4),
            "x1": round(self.x1, 1),
            "y1": round(self.y1, 1),
            "x2": round(self.x2, 1),
            "y2": round(self.y2, 1),
        }


class VehicleDetector:
    """
    YOLOv8-based vehicle detector with support for drone aerial perspectives.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = 0.25,
        iou_threshold: float = 0.50,
        input_size: int = 960,
        device: Optional[str] = None,
        drone_mode: bool = False,
    ):
        """
        Args:
            model_path:    Path to .pt weights.
            confidence:    Detection confidence threshold (default 0.25 for aerial roofs).
            iou_threshold: NMS IoU threshold (0.50 for dense traffic).
            input_size:    Inference image size (640, 960, 1280).
            device:        "cuda:0", "cpu", or None (auto).
            drone_mode:    If True, optimizes parameters for top-angle drone footage.
        """
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.drone_mode = drone_mode
        self.confidence = 0.22 if drone_mode else confidence
        self.iou_threshold = 0.55 if drone_mode else iou_threshold
        self.input_size = 960 if drone_mode and input_size < 960 else input_size

        resolved_path = str(model_path) if (model_path and Path(model_path).exists()) else "yolov8n.pt"
        log.info(f"Loading YOLOv8 detector from: {resolved_path} on {device}")
        self.model = YOLO(resolved_path)
        self.model.to(self.device)

        log.info(
            f"VehicleDetector ready | device={device} | conf={self.confidence} | "
            f"iou={self.iou_threshold} | imgsz={self.input_size} | drone_mode={drone_mode}"
        )

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp_sec: float = 0.0,
    ) -> List[Detection]:
        """Run detection on a single BGR frame."""
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            verbose=False,
            classes=list(TARGET_COCO_IDS),
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                coco_id = int(box.cls.item())
                if coco_id not in TARGET_COCO_IDS:
                    continue
                aic21_name = COCO_TO_AIC21[coco_id]
                xyxy = box.xyxy[0].tolist()
                detections.append(Detection(
                    frame_id=frame_id,
                    timestamp_sec=timestamp_sec,
                    coco_class_id=coco_id,
                    aic21_class_name=aic21_name,
                    aic21_class_id=AIC21_CLASS_ID[aic21_name],
                    confidence=float(box.conf.item()),
                    x1=xyxy[0],
                    y1=xyxy[1],
                    x2=xyxy[2],
                    y2=xyxy[3],
                ))

        return detections


class SlicedDroneDetector:
    """
    Overlapping tiled (sliced) inference for high-resolution aerial drone footage.
    Slices the image into overlapping sub-regions, runs batched inference,
    and merges boundary-overlapping detections using Non-Maximum Suppression (NMS).
    """

    def __init__(
        self,
        base_detector: VehicleDetector,
        tile_size: Tuple[int, int] = (640, 640),
        overlap_ratio: float = 0.20,
        nms_threshold: float = 0.45,
    ):
        self.detector = base_detector
        self.tile_w, self.tile_h = tile_size
        self.overlap_ratio = overlap_ratio
        self.nms_threshold = nms_threshold

    def _generate_slices(self, img_w: int, img_h: int) -> List[Tuple[int, int, int, int]]:
        """Generate overlapping tile coordinates: (x1, y1, x2, y2)."""
        stride_x = int(self.tile_w * (1.0 - self.overlap_ratio))
        stride_y = int(self.tile_h * (1.0 - self.overlap_ratio))

        x_starts = list(range(0, max(1, img_w - self.tile_w + 1), stride_x))
        if not x_starts or x_starts[-1] + self.tile_w < img_w:
            x_starts.append(max(0, img_w - self.tile_w))

        y_starts = list(range(0, max(1, img_h - self.tile_h + 1), stride_y))
        if not y_starts or y_starts[-1] + self.tile_h < img_h:
            y_starts.append(max(0, img_h - self.tile_h))

        slices = []
        for ys in y_starts:
            for xs in x_starts:
                slices.append((xs, ys, min(img_w, xs + self.tile_w), min(img_h, ys + self.tile_h)))
        return slices

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp_sec: float = 0.0,
    ) -> List[Detection]:
        """Run sliced overlapping detection and merge detections with NMS."""
        h, w = frame.shape[:2]
        slices = self._generate_slices(w, h)

        all_boxes = []
        all_confs = []
        all_classes = []

        # 1. Full frame low-resolution pass (catches large vehicles)
        full_dets = self.detector.detect(frame, frame_id, timestamp_sec)
        for d in full_dets:
            all_boxes.append([d.x1, d.y1, d.x2, d.y2])
            all_confs.append(d.confidence)
            all_classes.append((d.coco_class_id, d.aic21_class_name, d.aic21_class_id))

        # 2. Tiled high-resolution passes (catches tiny top-view vehicles)
        tile_crops = []
        tile_offsets = []
        for (x1, y1, x2, y2) in slices:
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                tile_crops.append(crop)
                tile_offsets.append((x1, y1))

        if tile_crops:
            # Batch infer on tiles
            results = self.detector.model(
                tile_crops,
                conf=self.detector.confidence,
                iou=self.detector.iou_threshold,
                imgsz=self.tile_w,
                verbose=False,
                classes=list(TARGET_COCO_IDS),
            )
            for i, res in enumerate(results):
                ox, oy = tile_offsets[i]
                if res.boxes is None:
                    continue
                for box in res.boxes:
                    cid = int(box.cls.item())
                    if cid not in TARGET_COCO_IDS:
                        continue
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf.item())
                    all_boxes.append([xyxy[0] + ox, xyxy[1] + oy, xyxy[2] + ox, xyxy[3] + oy])
                    all_confs.append(conf)
                    aic_name = COCO_TO_AIC21[cid]
                    all_classes.append((cid, aic_name, AIC21_CLASS_ID[aic_name]))

        if not all_boxes:
            return []

        # 3. Non-Maximum Suppression across merged tiles
        boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
        confs_tensor = torch.tensor(all_confs, dtype=torch.float32)

        keep_indices = torchvision.ops.nms(boxes_tensor, confs_tensor, self.nms_threshold)
        keep = keep_indices.cpu().numpy()

        merged_detections = []
        for idx in keep:
            b = all_boxes[idx]
            cid, aic_name, aic_id = all_classes[idx]
            merged_detections.append(Detection(
                frame_id=frame_id,
                timestamp_sec=timestamp_sec,
                coco_class_id=cid,
                aic21_class_name=aic_name,
                aic21_class_id=aic_id,
                confidence=all_confs[idx],
                x1=b[0],
                y1=b[1],
                x2=b[2],
                y2=b[3],
            ))

        return merged_detections
