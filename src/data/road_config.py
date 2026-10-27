"""
src/data/road_config.py
-----------------------
Multi-road ROI configuration loader.

Each camera can have 1..N road regions, each with its own polygon.
Configuration is stored in configs/roads/<camera_id>.yaml

When no road YAML is found, falls back to the original single-ROI as "road_1".

Road polygons are defined in annotation coordinate space (1920×1080 baseline)
and are automatically scaled to the actual video resolution.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RoadROI:
    """
    A single road region with its own polygon and analytics state.

    Polygons are defined in annotation coordinate space; the pipeline
    must scale them to actual video resolution before use.
    """
    road_id: str
    name: str
    color: Tuple[int, int, int]          # BGR color for visualization
    vertices: List[Tuple[int, int]]      # [(x, y), ...] in annotation space

    # Scaled polygon (filled in by pipeline after scaling)
    scaled_vertices: Optional[List[Tuple[int, int]]] = field(default=None, repr=False)
    # Known real-world road corridor distance in meters (if configured/calibrated)
    length_meters: Optional[float] = None

    def __post_init__(self):
        if not self.vertices or len(self.vertices) < 3:
            raise ValueError(f"Invalid road polygon: road_id={self.road_id} reason=must have at least 3 points")
        
        for p in self.vertices:
            if len(p) != 2:
                raise ValueError(f"Invalid road polygon: road_id={self.road_id} reason=coordinate must be 2D")
            if not isinstance(p[0], (int, float)) or not isinstance(p[1], (int, float)):
                raise ValueError(f"Invalid road polygon: road_id={self.road_id} reason=coordinates must be numeric")
            
        try:
            from shapely.geometry import Polygon
            poly = Polygon(self.vertices)
            
            # If the user drew a self-intersecting polygon (e.g. crossing lines on the UI),
            # repair it by taking the convex hull, which guarantees a valid simple polygon.
            if not poly.is_valid:
                poly = poly.convex_hull
                # Update vertices to the repaired shape
                self.vertices = [(int(x), int(y)) for x, y in poly.exterior.coords[:-1]]
                
                # Also repair scaled_vertices if they were already passed
                if self.scaled_vertices is not None:
                    try:
                        spoly = Polygon(self.scaled_vertices)
                        if not spoly.is_valid:
                            spoly = spoly.convex_hull
                            self.scaled_vertices = [(int(x), int(y)) for x, y in spoly.exterior.coords[:-1]]
                    except Exception:
                        pass
                
            if poly.area <= 0:
                raise ValueError(f"Invalid road polygon: road_id={self.road_id} reason=zero area polygon")
        except ImportError:
            pass # Shapely not available, skip advanced validation

    def scale_to_resolution(self, src_w: int, src_h: int, target_w: int, target_h: int) -> "RoadROI":
        """
        Return a new RoadROI with vertices scaled from (src_w×src_h) to (target_w×target_h).
        """
        scaled = []
        for x, y in self.vertices:
            sx = min(target_w - 1, max(0, int(x * target_w / src_w)))
            sy = min(target_h - 1, max(0, int(y * target_h / src_h)))
            scaled.append((sx, sy))
        return RoadROI(
            road_id=self.road_id,
            name=self.name,
            color=self.color,
            vertices=self.vertices,
            scaled_vertices=scaled,
            length_meters=self.length_meters,
        )

    def effective_vertices(self) -> List[Tuple[int, int]]:
        """Return scaled_vertices if available, otherwise original vertices."""
        return self.scaled_vertices if self.scaled_vertices is not None else self.vertices

    def numpy_vertices(self) -> np.ndarray:
        """Return effective vertices as np.array of shape (N, 2) for cv2 functions."""
        return np.array(self.effective_vertices(), dtype=np.int32)

    def get_envelope(self) -> Tuple[int, int, int, int]:
        """Return (min_x, min_y, max_x, max_y) bounding box of the effective polygon."""
        verts = self.effective_vertices()
        xs = [p[0] for p in verts]
        ys = [p[1] for p in verts]
        return (min(xs), min(ys), max(xs), max(ys))

    def get_calibrated_distance_meters(
        self,
        entry_pt: Tuple[float, float],
        exit_pt: Tuple[float, float],
        frame_height: int = 1080
    ) -> Tuple[float, bool]:
        """
        Calculate calibrated travel distance in metres along the road between entry and exit.
        Returns:
            (distance_meters, is_calibrated)
        """
        import math
        travel_px = math.hypot(exit_pt[0] - entry_pt[0], exit_pt[1] - entry_pt[1])
        if travel_px < 5.0:
            return (0.0, False)

        # 1. If explicit length_meters is configured (user-defined or calibrated config)
        if self.length_meters is not None and self.length_meters > 0:
            min_x, min_y, max_x, max_y = self.get_envelope()
            poly_span = max(10.0, math.hypot(max_x - min_x, max_y - min_y))
            # Fraction of the road corridor traversed by the vehicle along its trajectory
            ratio = min(1.2, travel_px / poly_span)
            dist_m = max(3.0, ratio * self.length_meters)
            return (dist_m, True)

        # 2. Calibration missing: do not fabricate real-world meters
        return (0.0, False)

    def contains_point(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Test if (x, y) is inside this road polygon with O(1) bounding-box pre-filtering."""
        verts = self.effective_vertices()
        if len(verts) < 3:
            return False

        # Fast envelope pre-filtering
        min_x, min_y, max_x, max_y = self.get_envelope()
        tol = max(0.0, -margin) + 2.0
        if x < min_x - tol or x > max_x + tol or y < min_y - tol or y > max_y + tol:
            return False

        pts = self.numpy_vertices()
        dist = cv2.pointPolygonTest(pts, (float(x), float(y)), True)
        return dist >= margin

    def contains_bottom_center(self, x1: float, y1: float, x2: float, y2: float, margin: float = 0.0) -> bool:
        """Test if bottom-center of bounding box is inside this road polygon."""
        cx = (x1 + x2) / 2.0
        by = float(y2)
        return self.contains_point(cx, by, margin=margin)

    def area(self) -> float:
        """Shoelace formula area of the effective polygon."""
        pts = np.array(self.effective_vertices(), dtype=np.float64)
        n = len(pts)
        if n < 3:
            return 0.0
        x = pts[:, 0]
        y = pts[:, 1]
        return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

# Default annotation coordinate space used by AIC21 Track 1 dataset
DEFAULT_ANNOTATION_WIDTH = 1920
DEFAULT_ANNOTATION_HEIGHT = 1080

# Color palette for roads (BGR) — used when YAML does not specify colors
ROAD_COLORS = [
    (0, 230, 255),   # cyan
    (0, 140, 255),   # orange
    (255, 100, 0),   # blue
    (0, 200, 80),    # green
    (200, 0, 200),   # purple
    (0, 200, 200),   # yellow
]


def load_road_config(
    camera_id: str,
    roads_config_dir: Path,
    fallback_roi_vertices: Optional[List[Tuple[int, int]]] = None,
) -> List[RoadROI]:
    """
    Load road polygon configuration for a camera.

    Args:
        camera_id:           e.g. "cam_5"
        roads_config_dir:    Path to directory containing <camera_id>.yaml files
        fallback_roi_vertices: If no YAML found, wrap the original ROI as road_1

    Returns:
        List[RoadROI] — at least one road.  Never empty.
    """
    yaml_path = roads_config_dir / f"{camera_id}.yaml"

    if yaml_path.exists():
        try:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)

            roads_raw = data.get("roads", [])
            if not roads_raw:
                log.warning(f"[ROAD-CONFIG] {yaml_path} has no roads entries — falling back.")
                return _make_fallback_roads(fallback_roi_vertices)

            result = []
            for i, rd in enumerate(roads_raw):
                rid = rd.get("road_id", f"road_{i+1}")
                name = rd.get("name", f"Road {i+1}")
                color_raw = rd.get("color", list(ROAD_COLORS[i % len(ROAD_COLORS)]))
                color = tuple(int(c) for c in color_raw)
                roi_pts = rd.get("roi", [])
                vertices = [(int(p[0]), int(p[1])) for p in roi_pts]
                if len(vertices) < 3:
                    log.warning(f"[ROAD-CONFIG] Road {rid} has fewer than 3 vertices — skipping.")
                    continue
                length_m = rd.get("length_meters", rd.get("length", None))
                result.append(RoadROI(
                    road_id=rid,
                    name=name,
                    color=color,
                    vertices=vertices,
                    length_meters=float(length_m) if length_m is not None else None
                ))

            if not result:
                log.warning(f"[ROAD-CONFIG] No valid roads in {yaml_path} — falling back.")
                return _make_fallback_roads(fallback_roi_vertices)

            log.info(f"[ROAD-CONFIG] Loaded {len(result)} roads for {camera_id}: "
                     f"{[r.road_id for r in result]}")
            return result

        except Exception as e:
            log.error(f"[ROAD-CONFIG] Failed to parse {yaml_path}: {e}")
            return _make_fallback_roads(fallback_roi_vertices)
    else:
        log.warning(f"[ROAD-CONFIG] No road config found at {yaml_path} — using fallback.")
        return _make_fallback_roads(fallback_roi_vertices)


def _make_fallback_roads(
    fallback_roi_vertices: Optional[List[Tuple[int, int]]] = None
) -> List[RoadROI]:
    """
    Create a single-road fallback using the existing ROI polygon.
    This ensures the system always has at least one road.
    """
    if fallback_roi_vertices and len(fallback_roi_vertices) >= 3:
        return [RoadROI(
            road_id="road_1",
            name="Road 1",
            color=ROAD_COLORS[0],
            vertices=fallback_roi_vertices,
        )]
    else:
        # Ultimate fallback: full-frame rectangle in annotation space
        return [RoadROI(
            road_id="road_1",
            name="Road 1",
            color=ROAD_COLORS[0],
            vertices=[(0, 0), (1920, 0), (1920, 1080), (0, 1080)],
        )]


# ---------------------------------------------------------------------------
# Road Assignment
# ---------------------------------------------------------------------------

def assign_track_to_road(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    road_rois: List[RoadROI],
    margin: float = 0.0,
    reference_point: str = "bottom_center",
) -> Optional[str]:
    """
    Determine which road a vehicle belongs to based on its reference point.

    Reference point (bottom_center preferred): the point where the vehicle
    contacts the road surface.

    If multiple road polygons contain the reference point, chooses the
    smallest-area polygon (most specific containment).

    Args:
        x1, y1, x2, y2:  Bounding box coordinates (in scaled/video space)
        road_rois:        List of RoadROI objects with effective vertices
        margin:           Polygon border margin (same units as coordinates)
        reference_point:  "bottom_center" (default) or "center"

    Returns:
        road_id string, or None if not inside any road.
    """
    if reference_point == "bottom_center":
        ref_x = (x1 + x2) / 2.0
        ref_y = float(y2)
    else:
        ref_x = (x1 + x2) / 2.0
        ref_y = (y1 + y2) / 2.0

    matching = []
    for road in road_rois:
        if road.contains_point(ref_x, ref_y, margin=margin):
            matching.append(road)

    if not matching:
        return None

    if len(matching) == 1:
        return matching[0].road_id

    # Ambiguous: multiple roads contain the reference point.
    # Choose smallest-area road (most specific region).
    best = min(matching, key=lambda r: r.area())
    log.debug(
        f"[ROAD-ASSIGN] Ambiguous assignment (ref={ref_x:.0f},{ref_y:.0f}) "
        f"in roads {[r.road_id for r in matching]} → chose {best.road_id} (smallest area)"
    )
    return best.road_id


def print_road_diagnostic(job_id: str, source_filename: str, camera_id: str, roads: List[RoadROI]) -> None:
    """Print a clear road configuration diagnostic to the log."""
    sep = "=" * 52
    log.info(f"\n{sep}")
    log.info("ROAD CONFIGURATION")
    log.info(sep)
    log.info(f"Job: {job_id}")
    log.info(f"Source: {source_filename}")
    log.info(f"Camera: {camera_id.upper()}")
    log.info(f"Road count: {len(roads)}")
    for road in roads:
        verts = road.effective_vertices()
        log.info(f"\n  {road.name} ({road.road_id}):")
        log.info(f"    ROI loaded: YES")
        log.info(f"    Points: {len(verts)}")
        log.info(f"    Color (BGR): {road.color}")
        log.info(f"    Area: {road.area():.0f} px²")
    log.info(sep)
