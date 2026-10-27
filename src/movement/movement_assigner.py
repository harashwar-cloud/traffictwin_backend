"""
src/movement/movement_assigner.py
-----------------------------------
Assigns a movement_id to a vehicle when it exits the ROI.

Strategy (trajectory-based):
  1. Detect when a track's center transitions INSIDE → OUTSIDE the ROI polygon.
  2. Compute approach direction: angle from which vehicle ENTERED the ROI.
  3. Compute exit direction: angle at which vehicle EXITS the ROI.
  4. Map (approach_direction, turn_type) → movement_id.

For 4-leg signalized intersections (12 movements):
  - 4 approach quadrants × 3 turn types (right, through, left) = 12 movements
  - Approach angle determines which leg (N/S/E/W)
  - Exit angle vs approach angle determines turn type

For highway / single-approach cameras:
  - Use exit boundary segment index → movement_id mapping

Limitation:
  - Approximate — official scoring uses human-annotated frames.
  - Movement assignment accuracy improves with longer tracked trajectories.
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from src.data.annotation_loader import CameraAnnotation, ROIPolygon
from src.tracking.tracker import TrackStore

log = logging.getLogger(__name__)


@dataclass
class EntryEvent:
    """Describes a vehicle entering the valid ROI polygon."""
    track_id: int
    frame_id: int
    timestamp_sec: float
    aic21_class_name: str
    aic21_class_id: int
    entry_x: float
    entry_y: float


@dataclass
class ExitEvent:
    """Describes a vehicle movement assignment."""
    track_id: int
    frame_id: int
    timestamp_sec: float
    aic21_class_name: str
    aic21_class_id: int
    exit_x: float
    exit_y: float
    movement_id: Optional[int]
    movement_description: str
    approach_angle_deg: Optional[float] = None
    exit_angle_deg: Optional[float] = None
    turn_type: Optional[str] = None  # "right", "through", "left"


class MovementAssigner:
    """
    Trajectory-based vehicle movement assigner for AIC21 Track 1.

    Uses vehicle trajectory (entry angle + exit angle) to assign movements.
    This handles both intersection cameras and highway cameras.

    For 4-leg intersections (12 movements):
      - Divides the ROI centroid space into 4 approach quadrants
      - Determines turn type from change in angle: right/through/left
      - Maps (quadrant, turn_type) → movement_id

    For highway / single-approach cameras:
      - Uses ROI exit segment index → movement_id
    """

    def __init__(
        self,
        roi: ROIPolygon,
        camera_annotation: CameraAnnotation,
        min_track_length: int = 5,
        exit_margin: float = 20.0,
        border_margin: float = 10.0,
        reference_point: str = "bottom_center",
    ):
        """
        Args:
            roi:                ROIPolygon for this camera.
            camera_annotation:  Parsed movement descriptions.
            min_track_length:   Minimum track frames before assigning movement.
            exit_margin:        Distance (pixels) inside ROI that counts as near exit.
            border_margin:      Safety inset margin inside ROI polygon to exclude border objects.
            reference_point:    "bottom_center" (road contact) or "center".
        """
        self.roi = roi
        self.camera = camera_annotation
        self.min_track_length = min_track_length
        self.exit_margin = exit_margin
        self.border_margin = border_margin
        self.reference_point = reference_point

        # ROI centroid (used as reference for angle computation)
        self.roi_centroid = self._compute_centroid()

        # Build ROI boundary segments (for fallback)
        self.boundary_segments = self._build_boundary_segments()

        # Classify camera type for routing logic
        self.n_movements = len(camera_annotation.movements)
        self.intersection_type = camera_annotation.intersection_type.lower()
        self.is_intersection = any(w in self.intersection_type for w in
                                   ["intersection", "signalized", "allway", "stop"])
        self.is_highway = any(w in self.intersection_type for w in
                              ["highway", "ramp", "segment"])

        # For intersection cameras: build movement lookup
        # Movement IDs are sorted; grouped as blocks of 3 (right/through/left) per leg
        self.movement_ids_sorted = sorted(camera_annotation.movements.keys())

        # Track previous in/out state: {track_id: was_inside}
        self._prev_inside: Dict[int, bool] = {}
        # Track first entry position: {track_id: (cx, cy)}
        self._entry_pos: Dict[int, Tuple[float, float]] = {}
        # Track counted IDs on ROI entry: Set[track_id]
        self._counted_tracks: set = set()
        # Track assigned movement IDs: {track_id: movement_id}
        self._assigned_movements: Dict[int, int] = {}

        log.info(
            f"MovementAssigner: camera={roi.camera_id} | "
            f"margin={self.border_margin}px | ref={self.reference_point} | "
            f"segments={len(self.boundary_segments)} | "
            f"movements={self.n_movements} | "
            f"type={'intersection' if self.is_intersection else 'highway'}"
        )

    def _compute_centroid(self) -> Tuple[float, float]:
        """Compute centroid of ROI polygon."""
        xs = [v[0] for v in self.roi.vertices]
        ys = [v[1] for v in self.roi.vertices]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _build_boundary_segments(self) -> List[LineString]:
        """Build list of LineString segments from ROI polygon boundary."""
        verts = self.roi.vertices
        segments = []
        n = len(verts)
        for i in range(n):
            p1 = verts[i]
            p2 = verts[(i + 1) % n]
            segments.append(LineString([p1, p2]))
        return segments

    def _find_nearest_segment(self, x: float, y: float) -> int:
        """Find the index of the boundary segment nearest to point (x, y)."""
        pt = Point(x, y)
        min_dist = float("inf")
        nearest_idx = 0
        for i, seg in enumerate(self.boundary_segments):
            dist = pt.distance(seg)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        return nearest_idx

    def _angle_from_centroid(self, cx: float, cy: float) -> float:
        """
        Compute angle (degrees, 0=right/East, ccw) from ROI centroid to point.
        Returns angle in [0, 360).
        """
        dx = cx - self.roi_centroid[0]
        dy = self.roi_centroid[1] - cy   # flip Y (image coords: y increases downward)
        angle = math.degrees(math.atan2(dy, dx))
        return angle % 360

    def _angle_to_quadrant(self, angle_deg: float, n_legs: int = 4) -> int:
        """
        Map angle to approach quadrant index [0, n_legs-1].
        For 4-leg intersection: 0=E, 1=N, 2=W, 3=S (quadrant 0-3, 90° each).
        """
        sector = 360.0 / n_legs
        return int((angle_deg + sector / 2) % 360 / sector) % n_legs

    def _compute_turn_type(
        self,
        approach_angle: float,
        exit_angle: float,
    ) -> str:
        """
        Determine turn type from approach direction to exit direction.

        Convention (standard traffic):
          - Right turn: exit is ~90° clockwise from approach direction
          - Through:    exit is ~180° from approach direction (vehicle continues straight)
          - Left turn:  exit is ~90° counter-clockwise from approach direction
          - U-turn:     exit is same as approach direction (excluded from counting)

        Angle diff is measured as (exit - approach) mod 360.
        For image coordinates (y flipped): clockwise = decreasing angle
        """
        diff = (exit_angle - approach_angle) % 360

        # With image Y-axis flipped, clockwise turn appears as angle DECREASE
        # Through movement: vehicle exits roughly opposite to entry → diff ≈ 180°
        # Right turn (image frame): diff ≈ 90° (vehicle turns to its right)
        # Left turn (image frame): diff ≈ 270° (vehicle turns to its left)
        # U-turn: diff ≈ 0° or 360°

        if 45 <= diff < 135:
            return "right"
        elif 135 <= diff < 225:
            return "through"
        elif 225 <= diff < 315:
            return "left"
        else:
            return "u_turn"  # Not counted per dataset rules

    def _assign_movement_intersection(
        self,
        approach_angle: float,
        exit_angle: float,
        n_legs: int = 4,
    ) -> Optional[int]:
        """
        Assign movement_id for an intersection camera.

        Pattern from dataset (4-leg, 12 movements):
          Movements 1-3:  leg 1 (right, through, left)
          Movements 4-6:  leg 2 (right, through, left)
          Movements 7-9:  leg 3 (right, through, left)
          Movements 10-12: leg 4 (right, through, left)

        Turn type ordering per leg: right=0, through=1, left=2
        """
        turn_type = self._compute_turn_type(approach_angle, exit_angle)

        if turn_type == "u_turn":
            return None   # U-turns not counted

        approach_quad = self._angle_to_quadrant(approach_angle, n_legs)

        turn_offset = {"right": 0, "through": 1, "left": 2}.get(turn_type, 1)
        movement_idx = approach_quad * 3 + turn_offset

        if movement_idx < len(self.movement_ids_sorted):
            return self.movement_ids_sorted[movement_idx]
        return self.movement_ids_sorted[-1]  # fallback to last movement

    def _assign_movement_highway(
        self,
        exit_x: float,
        exit_y: float,
    ) -> Optional[int]:
        """
        Assign movement_id for highway/ramp cameras.
        Uses exit boundary segment → movement index (evenly distributed).
        """
        seg_idx = self._find_nearest_segment(exit_x, exit_y)
        n_seg = len(self.boundary_segments)
        n_mov = len(self.movement_ids_sorted)
        if n_mov == 0:
            return None
        mov_idx = min(int(seg_idx * n_mov / n_seg), n_mov - 1)
        return self.movement_ids_sorted[mov_idx]

    def _assign_movement_single_approach(
        self,
        exit_x: float,
        exit_y: float,
        approach_angle: float,
    ) -> Optional[int]:
        """
        Assign movement_id for single-approach intersections (3 movements).
        Uses exit angle relative to approach to determine right/through/left.
        """
        # Single approach: exit is always away from camera
        # Map to right/through/left by exit angle from centroid
        exit_angle = self._angle_from_centroid(exit_x, exit_y)
        turn_type = self._compute_turn_type(approach_angle, exit_angle)
        if turn_type == "u_turn":
            return None
        turn_offsets = {"right": 0, "through": 1, "left": 2}
        turn_offset = turn_offsets.get(turn_type, 1)
        if turn_offset < len(self.movement_ids_sorted):
            return self.movement_ids_sorted[turn_offset]
        return self.movement_ids_sorted[-1]



    def process_frame(
        self,
        frame_id: int,
        timestamp_sec: float,
        track_store: TrackStore,
        active_tracks: list,
    ) -> Tuple[List[EntryEvent], List[ExitEvent]]:
        """
        Process ROI entries and movement exits for all active tracks on this frame.

        Returns:
            Tuple: (entry_events, exit_events)
        """
        entry_events = []
        exit_events = []

        for track in active_tracks:
            tid = track.track_id

            # Skip non-countable (motorcycles)
            if not track.is_countable():
                continue

            cx, cy = (track.bottom_center if self.reference_point == "bottom_center" else (track.center_x, track.center_y))
            currently_inside = track.is_inside_roi(self.roi, margin=self.border_margin, reference_point=self.reference_point)
            was_inside = self._prev_inside.get(tid, None)

            # 1. ROI ENTRY COUNTING: Count vehicle when reference point is observed inside ROI
            if currently_inside and tid not in self._counted_tracks:
                self._counted_tracks.add(tid)
                self._entry_pos[tid] = (cx, cy)
                entry_events.append(EntryEvent(
                    track_id=tid,
                    frame_id=frame_id,
                    timestamp_sec=timestamp_sec,
                    aic21_class_name=track.aic21_class_name,
                    aic21_class_id=track.aic21_class_id,
                    entry_x=cx,
                    entry_y=cy,
                ))
                log.debug(
                    f"ENTRY COUNT: track={tid} class={track.aic21_class_name} "
                    f"frame={frame_id} t={timestamp_sec:.1f}s"
                )

            # Record entry position when track enters ROI from outside
            if not was_inside and currently_inside and tid not in self._entry_pos:
                self._entry_pos[tid] = (cx, cy)

            # 2. MOVEMENT ASSIGNMENT: Evaluate trajectory when vehicle progresses or exits
            if tid not in self._assigned_movements:
                traj_len = track_store.track_length(tid)
                is_exiting = (was_inside and not currently_inside)

                if traj_len >= self.min_track_length and (is_exiting or traj_len >= 8):
                    entry_pos = self._entry_pos.get(tid, (cx, cy))
                    approach_angle = self._angle_from_centroid(entry_pos[0], entry_pos[1])
                    exit_angle = self._angle_from_centroid(cx, cy)

                    movement_id = None
                    turn_type = None

                    if self.is_intersection and approach_angle is not None:
                        n_legs = max(1, self.n_movements // 3)
                        turn_type = self._compute_turn_type(approach_angle, exit_angle)
                        if turn_type != "u_turn":
                            movement_id = self._assign_movement_intersection(
                                approach_angle, exit_angle, n_legs
                            )
                    elif self.is_highway:
                        movement_id = self._assign_movement_highway(cx, cy)
                    elif self.n_movements <= 3 and approach_angle is not None:
                        movement_id = self._assign_movement_single_approach(
                            cx, cy, approach_angle
                        )
                    else:
                        seg_idx = self._find_nearest_segment(cx, cy)
                        n_seg = len(self.boundary_segments)
                        n_mov = len(self.movement_ids_sorted)
                        if n_mov > 0 and n_seg > 0:
                            mov_idx = min(int(seg_idx * n_mov / n_seg), n_mov - 1)
                            movement_id = self.movement_ids_sorted[mov_idx]

                    if movement_id is not None:
                        movement_desc = ""
                        if movement_id in self.camera.movements:
                            movement_desc = self.camera.movements[movement_id].description

                        self._assigned_movements[tid] = movement_id
                        track_store.mark_counted(tid, movement_id)

                        exit_events.append(ExitEvent(
                            track_id=tid,
                            frame_id=frame_id,
                            timestamp_sec=timestamp_sec,
                            aic21_class_name=track.aic21_class_name,
                            aic21_class_id=track.aic21_class_id,
                            exit_x=cx,
                            exit_y=cy,
                            movement_id=movement_id,
                            movement_description=movement_desc,
                            approach_angle_deg=approach_angle,
                            exit_angle_deg=exit_angle,
                            turn_type=turn_type,
                        ))

            self._prev_inside[tid] = currently_inside

        return entry_events, exit_events

    def reset(self) -> None:
        """Reset internal state (call between different videos)."""
        self._prev_inside.clear()
        self._entry_pos.clear()
        self._counted_tracks.clear()
        self._assigned_movements.clear()
