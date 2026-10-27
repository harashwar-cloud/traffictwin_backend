"""
src/data/annotation_loader.py
------------------------------
Loads AIC21 Track 1 annotation files:
  - ROI polygons (ROIs/*.txt)
  - Movement descriptions (movement_description/*.txt)
  - Ground-truth counting CSV (counting_gt_sample/*.csv)

Returns strongly-typed structures. No fabrication.
"""

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import Point, Polygon

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ROIPolygon:
    camera_id: str
    vertices: List[Tuple[int, int]]     # [(x, y), ...]
    polygon: Polygon = field(init=False)

    def __post_init__(self):
        if len(self.vertices) < 3:
            raise ValueError(f"ROI for {self.camera_id} has fewer than 3 vertices")
        self.polygon = Polygon(self.vertices)

    def contains_point(self, x: float, y: float, margin: float = 0.0) -> bool:
        import cv2
        dist = cv2.pointPolygonTest(self.numpy_vertices(), (float(x), float(y)), True)
        return dist >= margin

    def contains_bottom_center(self, x1: float, y1: float, x2: float, y2: float, margin: float = 0.0) -> bool:
        """
        Check if bottom-center of bounding box (contact point on road) is inside ROI polygon.
        """
        import cv2
        cx = (x1 + x2) / 2.0
        by = float(y2)
        dist = cv2.pointPolygonTest(self.numpy_vertices(), (float(cx), by), True)
        return dist >= margin

    def numpy_vertices(self) -> np.ndarray:
        """Return vertices as np.array of shape (N, 2) — for cv2.polylines and cv2.pointPolygonTest."""
        return np.array(self.vertices, dtype=np.int32)


@dataclass
class Movement:
    movement_id: int
    description: str          # e.g. "right-turning movement"
    movement_type: str        # "right", "left", "through", "on_ramp", "off_ramp", "unknown"


@dataclass
class CameraAnnotation:
    camera_id: str
    intersection_type: str
    camera_type: str          # "normal camera", "fisheye camera", "long range PTZ camera"
    movements: Dict[int, Movement]
    notes: List[str]


@dataclass
class CountingEvent:
    """Single ground-truth counting event from the GT CSV."""
    video_clip_id: int
    frame_id: int
    movement_id: int
    vehicle_class_id: int     # 1=Car, 2=Truck
    vehicle_class_name: str   # "Car" or "Truck"


# ---------------------------------------------------------------------------
# Movement type inference
# ---------------------------------------------------------------------------

def _infer_movement_type(description: str) -> str:
    desc = description.lower()
    if "right" in desc:
        return "right"
    if "left" in desc:
        return "left"
    if "through" in desc:
        return "through"
    if "on ramp" in desc or "on_ramp" in desc:
        return "on_ramp"
    if "off ramp" in desc or "off_ramp" in desc:
        return "off_ramp"
    return "unknown"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_roi(roi_path: Path) -> Optional[ROIPolygon]:
    """
    Parse a ROI text file (one 'x,y' coordinate per line).
    Returns ROIPolygon or None on error.
    """
    camera_id = roi_path.stem
    vertices = []
    try:
        with open(roi_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) == 2:
                    vertices.append((int(parts[0]), int(parts[1])))
        if len(vertices) < 3:
            log.warning(f"ROI {camera_id}: only {len(vertices)} vertices — skipping")
            return None
        return ROIPolygon(camera_id=camera_id, vertices=vertices)
    except Exception as e:
        log.error(f"Failed to load ROI {roi_path}: {e}")
        return None


def load_all_rois(roi_dir: Path) -> Dict[str, ROIPolygon]:
    """Load all ROI files in directory. Returns {camera_id: ROIPolygon}."""
    rois = {}
    if not roi_dir.exists() or not roi_dir.is_dir():
        log.error(f"ROI directory not found or invalid: {roi_dir}")
        return rois

    for f in sorted(roi_dir.glob("*.txt")):
        roi = load_roi(f)
        if roi:
            rois[roi.camera_id] = roi
            
    if len(rois) == 0:
        log.warning(f"No ROIs found in {roi_dir}")
    else:
        log.info(f"Loaded ROIs for {len(rois)} cameras: {list(rois.keys())}")
    return rois


def load_movement_description(mov_path: Path) -> Optional[CameraAnnotation]:
    """Parse a movement description text file."""
    camera_id = mov_path.stem
    try:
        with open(mov_path, "r") as f:
            lines = [line.strip() for line in f.readlines()]

        non_empty = [l for l in lines if l]
        intersection_type = non_empty[0] if len(non_empty) > 0 else "unknown"
        camera_type = non_empty[1] if len(non_empty) > 1 else "unknown"

        movements = {}
        notes = []
        in_note = False
        for line in non_empty[2:]:
            if line.lower().startswith("movement "):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    try:
                        mov_id = int(parts[0].strip().lower().replace("movement", "").strip())
                        mov_desc = parts[1].strip()
                        movements[mov_id] = Movement(
                            movement_id=mov_id,
                            description=mov_desc,
                            movement_type=_infer_movement_type(mov_desc)
                        )
                    except ValueError:
                        pass
            elif line.lower().startswith("note"):
                in_note = True
                notes.append(line)
            elif in_note:
                notes.append(line)

        return CameraAnnotation(
            camera_id=camera_id,
            intersection_type=intersection_type,
            camera_type=camera_type,
            movements=movements,
            notes=notes
        )
    except Exception as e:
        log.error(f"Failed to load movement description {mov_path}: {e}")
        return None


def load_all_movements(mov_dir: Path) -> Dict[str, CameraAnnotation]:
    """Load all movement description files. Returns {camera_id: CameraAnnotation}."""
    annotations = {}
    if not mov_dir.exists() or not mov_dir.is_dir():
        log.error(f"Movement directory not found or invalid: {mov_dir}")
        return annotations

    for f in sorted(mov_dir.glob("*.txt")):
        ann = load_movement_description(f)
        if ann:
            annotations[ann.camera_id] = ann
            
    if len(annotations) == 0:
        log.warning(f"No movement descriptions found in {mov_dir}")
    else:
        log.info(f"Loaded movement descriptions for {len(annotations)} cameras: {list(annotations.keys())}")
    return annotations


def load_counting_gt(csv_path: Path) -> List[CountingEvent]:
    """
    Parse counting GT CSV.
    Format: video_clip_id, frame_id, movement_id, vehicle_class_id
    Returns list of CountingEvent sorted by frame_id.
    """
    CLASS_NAMES = {1: "Car", 2: "Truck"}
    events = []
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_id = int(row["vehicle_class_id"])
                events.append(CountingEvent(
                    video_clip_id=int(row["video_clip_id"]),
                    frame_id=int(row["frame_id"]),
                    movement_id=int(row["movement_id"]),
                    vehicle_class_id=class_id,
                    vehicle_class_name=CLASS_NAMES.get(class_id, "Unknown")
                ))
        events.sort(key=lambda e: e.frame_id)
        log.info(f"Loaded {len(events)} GT counting events from {csv_path.name}")
    except Exception as e:
        log.error(f"Failed to load GT CSV {csv_path}: {e}")
    return events


def summarize_gt(events: List[CountingEvent]) -> Dict:
    """Summarize GT events into counts per movement and class."""
    from collections import defaultdict
    summary = {
        "total": len(events),
        "by_class": defaultdict(int),
        "by_movement": defaultdict(int),
        "by_movement_and_class": defaultdict(lambda: defaultdict(int)),
        "frame_range": [None, None]
    }
    frames = []
    for e in events:
        summary["by_class"][e.vehicle_class_name] += 1
        summary["by_movement"][e.movement_id] += 1
        summary["by_movement_and_class"][e.movement_id][e.vehicle_class_name] += 1
        frames.append(e.frame_id)
    if frames:
        summary["frame_range"] = [min(frames), max(frames)]
    # Convert defaultdicts for JSON serialization
    summary["by_class"] = dict(summary["by_class"])
    summary["by_movement"] = dict(summary["by_movement"])
    summary["by_movement_and_class"] = {
        k: dict(v) for k, v in summary["by_movement_and_class"].items()
    }
    return summary
