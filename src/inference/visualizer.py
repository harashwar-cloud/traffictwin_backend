"""
src/inference/visualizer.py
-----------------------------
Annotates video frames with:
  - Vehicle bounding boxes (color-coded by class)
  - Track IDs + vehicle class + confidence + road assignment label
  - Single or multi-road ROI polygon overlays
  - Traffic state display
  - Real-time counts panel

All drawing uses OpenCV — no external dependencies.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.data.annotation_loader import ROIPolygon
from src.data.road_config import RoadROI
from src.tracking.tracker import Track
from src.analytics.traffic_state import IntersectionTrafficState, TrafficState

log = logging.getLogger(__name__)

# Color palette (BGR format)
COLORS = {
    "Car":        (0, 220, 0),       # green
    "Bus":        (255, 180, 0),     # cyan-amber
    "Truck":      (0, 165, 255),     # orange
    "Motorcycle": (200, 200, 0),     # yellow-cyan
    "Bicycle":    (220, 100, 220),   # purple/magenta
    "Unknown":    (180, 180, 180),
    "roi":        (0, 255, 255),     # yellow
    TrafficState.LOW:      (0, 200, 0),
    TrafficState.MEDIUM:   (0, 165, 255),
    TrafficState.HIGH:     (0, 0, 255),
    TrafficState.CRITICAL: (128, 0, 255),
    TrafficState.UNKNOWN:  (150, 150, 150),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# ROI Drawing — single (legacy) and multi-road
# ---------------------------------------------------------------------------

# Pre-allocated scratch buffer for zero-allocation SIMD blending
_OVERLAY_BUFFER: Optional[np.ndarray] = None
_CACHED_ROAD_KEY = None
_CACHED_ROAD_ANN: List[Dict[str, Any]] = []


def draw_roi(
    frame: np.ndarray,
    roi: ROIPolygon,
    alpha: float = 0.12,
) -> np.ndarray:
    """Draw semi-transparent single ROI polygon on frame (legacy compatibility)."""
    global _OVERLAY_BUFFER
    h, w = frame.shape[:2]
    if _OVERLAY_BUFFER is None or _OVERLAY_BUFFER.shape != frame.shape:
        _OVERLAY_BUFFER = np.empty_like(frame)

    np.copyto(_OVERLAY_BUFFER, frame)
    pts = roi.numpy_vertices().reshape((-1, 1, 2))
    cv2.fillPoly(_OVERLAY_BUFFER, [pts], COLORS["roi"])
    cv2.addWeighted(_OVERLAY_BUFFER, alpha, frame, 1.0 - alpha, 0, dst=frame)
    cv2.polylines(frame, [pts], isClosed=True, color=COLORS["roi"], thickness=2)
    return frame


def draw_road_rois(
    frame: np.ndarray,
    road_rois: List[RoadROI],
    alpha: float = 0.12,
) -> np.ndarray:
    """
    Draw all road ROI polygons on frame using SIMD-accelerated pre-allocated buffer.
    Reuses buffer across frames with zero heap re-allocation and precomputed text layout.
    """
    global _OVERLAY_BUFFER, _CACHED_ROAD_KEY, _CACHED_ROAD_ANN
    if not road_rois:
        return frame

    h, w = frame.shape[:2]
    if _OVERLAY_BUFFER is None or _OVERLAY_BUFFER.shape != frame.shape:
        _OVERLAY_BUFFER = np.empty_like(frame)

    # Check if road layout changed; if so, pre-compute pill geometries once
    road_key = (h, w, tuple((r.road_id, tuple(r.effective_vertices())) for r in road_rois))
    if road_key != _CACHED_ROAD_KEY:
        ann_list = []
        for road in road_rois:
            pts = road.numpy_vertices().reshape((-1, 1, 2))
            verts = np.array(road.effective_vertices(), dtype=np.float32)
            cx = int(verts[:, 0].mean())
            cy = int(verts[:, 1].mean())
            short_id = road.road_id.upper().replace("_", " ")
            (tw, th), _ = cv2.getTextSize(short_id, FONT, 0.65, 2)
            lx = max(0, cx - tw // 2)
            ly = max(th + 8, cy)
            ann_list.append({
                "pts": pts,
                "color": road.color,
                "text": short_id,
                "pill": (lx - 4, ly - th - 6, lx + tw + 4, ly + 4),
                "text_pos": (lx, ly - 2),
            })
        _CACHED_ROAD_ANN = ann_list
        _CACHED_ROAD_KEY = road_key

    # Fast in-place SIMD overlay copy + fill + blend (takes ~1.3 ms on 1080p)
    np.copyto(_OVERLAY_BUFFER, frame)
    for ann in _CACHED_ROAD_ANN:
        cv2.fillPoly(_OVERLAY_BUFFER, [ann["pts"]], ann["color"])

    cv2.addWeighted(_OVERLAY_BUFFER, alpha, frame, 1.0 - alpha, 0, dst=frame)

    # Draw pre-computed outlines and centroid pills
    for ann in _CACHED_ROAD_ANN:
        cv2.polylines(frame, [ann["pts"]], isClosed=True, color=ann["color"], thickness=2)
        p = ann["pill"]
        cv2.rectangle(frame, (p[0], p[1]), (p[2], p[3]), (0, 0, 0), -1)
        cv2.rectangle(frame, (p[0], p[1]), (p[2], p[3]), ann["color"], 2)
        cv2.putText(frame, ann["text"], ann["text_pos"], FONT, 0.65, ann["color"], 2)

    return frame



# ---------------------------------------------------------------------------
# Track Drawing — with road assignment labels
# ---------------------------------------------------------------------------

def draw_tracks(
    frame: np.ndarray,
    tracks: List[Track],
    roi: Optional[ROIPolygon] = None,
    road_rois: Optional[List[RoadROI]] = None,
    road_assignment: Optional[Dict[int, str]] = None,
    movement_states: Optional[Dict[int, str]] = None,
    roi_states: Optional[Dict[int, str]] = None,
    show_outside_roi: bool = True,
    margin: float = 0.0,
    reference_point: str = "bottom_center",
    show_track_id: bool = True,
    show_class: bool = True,
    thickness: int = 2,
    font_scale: float = 0.50,
) -> np.ndarray:
    """
    Draw bounding boxes and labels for tracked vehicles with distinct ROI color-coding.

    Color coding:
      - Inside ROI: Bright Vibrant Green (0, 230, 0)
      - Transitioning (Entering): Cyan (255, 200, 0)
      - Transitioning (Exiting): Orange (0, 140, 255)
      - Outside ROI: Muted Slate Gray (120, 120, 120)
    """
    for track in tracks:
        # Determine ROI membership status
        roi_st = roi_states.get(track.track_id) if roi_states else None
        road_label = ""

        if roi_st == "INSIDE":
            color = (0, 230, 0)       # Green
            status_tag = "IN"
        elif roi_st == "ENTERING":
            color = (255, 200, 0)     # Cyan
            status_tag = "ENTER"
        elif roi_st == "EXITING":
            color = (0, 140, 255)     # Orange
            status_tag = "EXIT"
        elif roi_st == "OUTSIDE":
            color = (120, 120, 120)   # Gray
            status_tag = "OUT"
        else:
            # Fallback to road assignment
            assigned_road_id = road_assignment.get(track.track_id) if road_assignment else None
            if assigned_road_id and assigned_road_id != "OUTSIDE":
                color = (0, 230, 0)
                status_tag = "IN"
            else:
                color = (120, 120, 120)
                status_tag = "OUT"

        if road_rois is not None and road_assignment:
            assigned = road_assignment.get(track.track_id)
            if assigned and assigned != "OUTSIDE":
                num = assigned.split("_")[-1] if "_" in assigned else assigned
                road_label = f"R{num}"

        if not show_outside_roi and status_tag == "OUT":
            continue

        x1, y1, x2, y2 = int(track.x1), int(track.y1), int(track.x2), int(track.y2)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Build label: #ID CLS CONF [ROAD] [MOV] [TAG]
        label_parts = []
        if show_track_id:
            label_parts.append(f"#{track.track_id}")
        if show_class:
            cls_name = track.aic21_class_name
            cls_short = cls_name[:4].upper() if cls_name == "Bicycle" else cls_name[:3].upper()
            label_parts.append(cls_short)

        label_parts.append(f"{track.confidence:.2f}")

        if road_label:
            label_parts.append(road_label)
        label_parts.append(status_tag)

        if movement_states:
            mov_state = movement_states.get(track.track_id, "UNKNOWN")
            st_short = "MOV" if mov_state == "MOVING" else ("STOP" if mov_state in ("STATIONARY", "REST") else "...")
            label_parts.append(st_short)

        label = " | ".join(label_parts)

        # Background for label — above bounding box
        (tw, th), _ = cv2.getTextSize(label, FONT, font_scale, 1)
        label_y = max(y1 - 4, th + 6)

        cv2.rectangle(frame, (x1, label_y - th - 5), (x1 + tw + 6, label_y + 2), (0, 0, 0), -1)
        cv2.rectangle(frame, (x1, label_y - th - 5), (x1 + tw + 6, label_y + 2), color, 1)
        cv2.putText(frame, label, (x1 + 3, label_y - 2), FONT, font_scale, color, 1)

        # Draw representative anchor point (bottom-center for road contact)
        ref_x = int((x1 + x2) / 2)
        ref_y = int(y2 if reference_point == "bottom_center" else (y1 + y2) / 2)
        dot_color = (0, 255, 0) if status_tag == "IN" else ((0, 200, 255) if status_tag in ("ENTER", "EXIT") else (160, 160, 160))
        cv2.circle(frame, (ref_x, ref_y), 3, dot_color, -1)

    return frame


# ---------------------------------------------------------------------------
# Count Panel
# ---------------------------------------------------------------------------

def draw_count_panel(
    frame: np.ndarray,
    total: int,
    car_count: int,
    truck_count: int,
    frame_id: int,
    timestamp_sec: float,
    traffic_state: Optional[str] = None,
    road_summaries: Optional[List[Dict]] = None,
    current_global: int = 0,
    flow_rate: float = 0.0,
    avg_speed: float = 0.0,
    queue_length: int = 0,
    bus_count: int = 0,
    motorcycle_count: int = 0,
    initial_occupancy: int = 0,
    cumulative_entries: int = 0,
    cumulative_exits: int = 0,
    invariant_verified: bool = True,
) -> np.ndarray:
    """
    Draw a count panel in the top-right corner.
    Clearly separates Current ROI Occupancy, Cumulative Entries/Exits, and Invariant Verification.
    """
    h, w = frame.shape[:2]

    # Determine panel height based on roads
    num_roads = len(road_summaries) if road_summaries else 0
    base_lines = 10
    road_lines = num_roads * 3
    panel_h = 24 + (base_lines + road_lines) * 18 + 10
    panel_w = 340
    x_offset = w - panel_w - 10
    y_offset = 10

    # Fast localized in-place darkening (zero memory allocations)
    sub = frame[y_offset : y_offset + panel_h, x_offset : x_offset + panel_w]
    if sub.size > 0:
        sub[:] = (sub.astype(np.uint16) * 46 >> 8).astype(np.uint8)
        cv2.rectangle(frame, (x_offset, y_offset), (x_offset + panel_w, y_offset + panel_h), (60, 60, 60), 1)

    lines = [
        (f"Frame: {frame_id} | T: {timestamp_sec:.1f}s", (200, 200, 200)),
        (f"CURRENT ROI OCCUPANCY: {current_global} veh", (0, 255, 0)),
        (f"ENTRIES: {cumulative_entries} | EXITS: {cumulative_exits}", (0, 220, 255)),
        (f"INVARIANT: {'[PASS] ' + str(current_global) + ' = ' + str(initial_occupancy) + ' + ' + str(cumulative_entries) + ' - ' + str(cumulative_exits) if invariant_verified else '[FAIL] MISMATCH'}", (0, 255, 128) if invariant_verified else (0, 0, 255)),
        (f"CUMULATIVE TOTAL: {total} (Car:{car_count} Trk:{truck_count})", (220, 220, 220)),
        (f"FLOW (30s): {flow_rate:.1f} veh/min (Rolling)", (255, 220, 100)),
        (f"QUEUE: {queue_length} veh | SPEED: {avg_speed:.1f} km/h", (100, 220, 255)),
    ]
    if traffic_state:
        lines.append((f"STATE: {traffic_state}", (255, 255, 255)))

    if road_summaries:
        lines.append(("─" * 26, (120, 120, 120)))
        for rs in road_summaries:
            rname = rs.get("name", rs.get("road_id", "?"))
            rveh = rs.get("vehicles", 0)
            rst = rs.get("traffic_state", "LOW")
            cact = rs.get("current_active", 0)
            cmov = rs.get("current_moving", 0)
            crest = rs.get("current_rest", rs.get("current_stationary", 0))
            lines.append((f"{rname[:14]}: Vis {cact} [{rst[:3]}]", (240, 240, 240)))
            lines.append((f"  Mov: {cmov}  |  Stop: {crest}", (180, 180, 180)))
            lines.append((f"  Cumul: {rveh}", (160, 160, 160)))

    for i, (line_text, color) in enumerate(lines):
        if line_text.startswith("STATE:"):
            state_val = line_text.split(":")[-1].strip()
            ts = TrafficState(state_val) if state_val in [s.value for s in TrafficState] else TrafficState.UNKNOWN
            color = COLORS.get(ts, (255, 255, 255))
        cv2.putText(
            frame, line_text,
            (x_offset + 8, y_offset + 18 + i * 18),
            FONT, 0.44, color, 1, cv2.LINE_AA
        )

    return frame


# ---------------------------------------------------------------------------
# Traffic State Banner
# ---------------------------------------------------------------------------

def draw_traffic_state_banner(
    frame: np.ndarray,
    state: IntersectionTrafficState,
) -> np.ndarray:
    """Draw a traffic state banner at the bottom of the frame with localized ROI blend."""
    h, w = frame.shape[:2]
    banner_h = 32
    y_start = h - banner_h

    ts = state.overall_state
    color = COLORS.get(ts, (100, 100, 100))

    # Fast localized in-place alpha blend
    sub = frame[y_start : h, 0 : w]
    if sub.size > 0:
        c_arr = np.array(color, dtype=np.uint16)
        sub[:] = ((sub.astype(np.uint16) * 64 + c_arr * 192) >> 8).astype(np.uint8)

    text = (
        f"TRAFFIC: {ts.value}  |  "
        f"Flow (30s): {state.total_flow_rate_per_min:.1f} veh/min  |  "
        f"Window: {state.total_vehicles_in_window} vehicles"
    )
    if state.direction_imbalance.detected:
        text += f"  |  IMBALANCE: mov {state.direction_imbalance.dominant_movement}"

    cv2.putText(frame, text, (10, y_start + 22),
                FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Diagnostic Road Mapping Image
# ---------------------------------------------------------------------------

def save_road_diagnostic_image(
    frame: np.ndarray,
    road_rois: List[RoadROI],
    output_path: str,
    job_id: str = "",
) -> bool:
    """
    Save a diagnostic image showing all road polygons on the first frame.
    This lets operators verify road polygon placement before trusting vehicle counts.

    Args:
        frame:       First frame from the source video (original, unmodified)
        road_rois:   Scaled road ROI list
        output_path: Path to save the diagnostic JPEG
        job_id:      For labeling

    Returns:
        True if saved successfully.
    """
    try:
        diag = frame.copy()
        diag = draw_road_rois(diag, road_rois, alpha=0.20)

        h, w = diag.shape[:2]
        title = f"ROAD MAPPING DIAGNOSTIC  |  Job: {job_id}  |  Roads: {len(road_rois)}"
        cv2.putText(diag, title, (10, 30), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(diag, title, (10, 30), FONT, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

        for i, road in enumerate(road_rois):
            info = f"  {road.name} | {len(road.effective_vertices())} pts | area={road.area():.0f}px²"
            cv2.putText(diag, info, (10, 55 + i * 22), FONT, 0.55, road.color, 2, cv2.LINE_AA)

        import os
        from pathlib import Path
        os.makedirs(str(Path(output_path).parent), exist_ok=True)
        success = cv2.imwrite(output_path, diag, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if success:
            log.info(f"[DIAGNOSTIC] Road mapping image saved: {output_path}")
        else:
            log.warning(f"[DIAGNOSTIC] Failed to save road mapping image: {output_path}")
        return success
    except Exception as e:
        log.error(f"[DIAGNOSTIC] Exception saving road diagnostic: {e}")
        return False


# ---------------------------------------------------------------------------
# Video Writer
# ---------------------------------------------------------------------------

def create_video_writer(
    output_path: str,
    width: int,
    height: int,
    fps: float,
) -> cv2.VideoWriter:
    """Create an OpenCV VideoWriter for MP4 output."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter at: {output_path}")
    return writer
