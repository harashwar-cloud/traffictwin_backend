"""
src/counting/roi_counter.py
----------------------------
High-precision polygon-based ROI vehicle counting and tracking for TrafficTwin.

Counting logic:
  1. A vehicle enters the polygon: increase current vehicle count by 1.
  2. A vehicle remains inside the polygon: keep it counted without incrementing again.
  3. A vehicle moves within the polygon: keep it counted.
  4. A vehicle crosses outside the polygon: decrease current vehicle count by 1.
  5. A vehicle moves back inside the polygon: increase current count by 1 again.
  6. A vehicle outside the polygon: do not include in current count.

Mathematical Invariant:
  Current ROI Count = Initial Occupancy + Cumulative Entries - Cumulative Exits

Features:
  - Track-ID based counting (not raw detections per frame).
  - Bottom-center bounding box anchor point (road contact point) or center.
  - Dual-margin boundary hysteresis to suppress jitter at polygon boundaries.
  - Multi-frame confirmation for stable entry and exit transitions.
  - Temporary track loss grace period and deterministic stale track expiration.
  - Re-entry support without permanent counted_ids lock.
  - Fast bounding-box envelope pre-filtering before cv2.pointPolygonTest.
  - Support for single-polygon and multi-road polygon regions.
"""

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from src.movement.movement_assigner import EntryEvent, ExitEvent

log = logging.getLogger(__name__)


class TrackROIState(str, Enum):
    OUTSIDE = "OUTSIDE"
    ENTERING = "ENTERING"
    INSIDE = "INSIDE"
    EXITING = "EXITING"
    LOST = "LOST"
    EXPIRED = "EXPIRED"


@dataclass
class ROIEntryEvent:
    """Event emitted when a vehicle is confirmed to have entered the ROI."""
    track_id: int
    frame_id: int
    timestamp_sec: float
    class_name: str
    class_id: int
    road_id: str
    anchor_x: float
    anchor_y: float
    bbox: Tuple[float, float, float, float]


@dataclass
class ROIExitEvent:
    """Event emitted when a vehicle is confirmed to have exited the ROI."""
    track_id: int
    frame_id: int
    timestamp_sec: float
    class_name: str
    class_id: int
    road_id: str
    anchor_x: float
    anchor_y: float
    bbox: Tuple[float, float, float, float]
    reason: str = "normal"  # "normal" boundary exit or "expired" track timeout


@dataclass
class ROICountSnapshot:
    """Complete snapshot of live and cumulative vehicle counts inside the ROI."""
    # Current Occupancy inside ROI
    current_count: int = 0
    initial_occupancy: int = 0
    cumulative_entries: int = 0
    cumulative_exits: int = 0
    invariant_verified: bool = True

    # Active vehicles currently inside ROI
    active_track_ids: List[int] = field(default_factory=list)
    by_class: Dict[str, int] = field(default_factory=dict)
    by_road: Dict[str, int] = field(default_factory=dict)

    # Cumulative stats
    cumulative_by_class: Dict[str, int] = field(default_factory=dict)
    cumulative_by_road: Dict[str, int] = field(default_factory=dict)

    # Visual states for rendering bounding boxes
    track_states: Dict[int, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "current_count": self.current_count,
            "initial_occupancy": self.initial_occupancy,
            "cumulative_entries": self.cumulative_entries,
            "cumulative_exits": self.cumulative_exits,
            "invariant_verified": self.invariant_verified,
            "active_track_ids": list(self.active_track_ids),
            "by_class": dict(self.by_class),
            "by_road": dict(self.by_road),
            "cumulative_by_class": dict(self.cumulative_by_class),
            "cumulative_by_road": dict(self.cumulative_by_road),
            "track_states": dict(self.track_states),
        }


# ==============================================================================
# ROIVehicleCounter — Primary Polygon-Based Counter
# ==============================================================================

class ROIVehicleCounter:
    """
    Precision polygon-based vehicle counter for user-defined ROIs.
    Maintains exact agreement:
        Current Count = Initial Occupancy + Entries - Exits
    """

    def __init__(
        self,
        road_rois: Optional[List[Any]] = None,
        reference_point: str = "bottom_center",
        boundary_tolerance: float = 8.0,
        entry_confirm_frames: int = 2,
        exit_confirm_frames: int = 3,
        lost_grace_frames: int = 12,
        time_window_seconds: float = 60.0,
        track_initial_occupancy: bool = False,
    ):
        """
        Args:
            road_rois: List of RoadROI or objects with effective_vertices() / numpy_vertices()
            reference_point: "bottom_center" (default) or "center"
            boundary_tolerance: Margin in pixels for hysteresis band
            entry_confirm_frames: Consecutive frames inside to confirm entry
            exit_confirm_frames: Consecutive frames outside to confirm exit
            lost_grace_frames: Frames before a lost track is considered permanently expired
            time_window_seconds: Rolling window for flow rate estimation
        """
        self.reference_point = reference_point
        self.boundary_tolerance = float(boundary_tolerance)
        self.entry_confirm_frames = max(1, int(entry_confirm_frames))
        self.exit_confirm_frames = max(1, int(exit_confirm_frames))
        self.lost_grace_frames = max(1, int(lost_grace_frames))
        self.time_window = float(time_window_seconds)
        self.track_initial_occupancy = track_initial_occupancy

        # Polygon definitions & bounding box pre-filtering envelopes
        self.road_rois: List[Any] = []
        self._road_envelopes: Dict[str, Tuple[float, float, float, float]] = {}
        if road_rois:
            self.set_rois(road_rois)

        # Invariant metrics
        self._initial_occupancy: int = 0
        self._cumulative_entries: int = 0
        self._cumulative_exits: int = 0

        # Current occupants: set of track_ids confirmed inside ROI
        self._current_occupants: Set[int] = set()

        # Cumulative track history (for telemetry and unique class counting)
        self._cumulative_class_counts: Dict[str, int] = defaultdict(int)
        self._cumulative_road_counts: Dict[str, int] = defaultdict(int)

        self._track_states: Dict[int, Dict[str, Any]] = {}

        # Log of entry and exit events
        self._entry_events: List[ROIEntryEvent] = []
        self._exit_events: List[ROIExitEvent] = []

        # Flow rate tracking: deque of (timestamp_sec, track_id)
        self._flow_events: deque = deque()

        # Startup initialization flag
        self._is_startup: bool = True
        self._startup_frame: Optional[int] = None

        log.info(
            f"ROIVehicleCounter initialized | ref={reference_point} | "
            f"tol={boundary_tolerance}px | entry_confirm={self.entry_confirm_frames} | "
            f"exit_confirm={self.exit_confirm_frames} | grace={self.lost_grace_frames} | "
            f"init_occ={track_initial_occupancy}"
        )

    def set_rois(self, road_rois: List[Any]) -> None:
        """Update or register road ROI polygons with bounding-box pre-filtering."""
        self.road_rois = list(road_rois)
        self._road_envelopes.clear()

        for road in self.road_rois:
            rid = getattr(road, "road_id", "roi")
            verts = getattr(road, "effective_vertices", lambda: getattr(road, "vertices", []))()
            if len(verts) >= 3:
                xs = [p[0] for p in verts]
                ys = [p[1] for p in verts]
                self._road_envelopes[rid] = (min(xs), min(ys), max(xs), max(ys))
            else:
                log.warning(f"Road {rid} has fewer than 3 vertices; skipping envelope creation.")

    def _get_anchor_point(self, x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
        """Calculate representative anchor point."""
        if self.reference_point == "bottom_center":
            return ((x1 + x2) / 2.0, float(y2))
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _test_containment(self, px: float, py: float, margin: float = 0.0) -> Tuple[bool, Optional[str], float]:
        """
        Test if (px, py) is inside any active road ROI with bounding box pre-filtering.
        Returns: (is_inside, best_road_id, max_distance)
        """
        if not self.road_rois:
            return False, None, -999.0

        best_road_id = None
        max_dist = -999.0
        is_inside = False

        for road in self.road_rois:
            rid = getattr(road, "road_id", "roi")
            env = self._road_envelopes.get(rid)

            # 1. Bounding box pre-filtering: O(1) early rejection
            if env:
                min_x, min_y, max_x, max_y = env
                tol = max(0.0, -margin) + 2.0
                if px < min_x - tol or px > max_x + tol or py < min_y - tol or py > max_y + tol:
                    continue

            # 2. Precise OpenCV polygon test
            pts = getattr(road, "numpy_vertices", lambda: np.array(road.vertices, dtype=np.int32))()
            dist = cv2.pointPolygonTest(pts, (float(px), float(py)), True)

            if dist > max_dist:
                max_dist = dist
                best_road_id = rid

            if dist >= margin:
                is_inside = True

        return is_inside, best_road_id, max_dist

    def update(
        self,
        frame_id: int,
        timestamp_sec: float,
        tracks: List[Any],
    ) -> ROICountSnapshot:
        """
        Process tracks on the current frame and update ROI occupancy.
        Handles entries, exits, boundary jitter, re-entries, and track timeouts.
        """
        # Determine startup phase (e.g. initial frame)
        if self._is_startup:
            self._startup_frame = frame_id

        active_track_ids_in_frame = set()
        current_frame_tracks: Dict[int, Any] = {}

        for track in tracks:
            # Only count countable vehicles (Cars, Trucks, Buses, Motorcycles, Bicycles)
            is_cnt = getattr(track, "is_countable", lambda: True)()
            if not is_cnt:
                continue

            tid = int(track.track_id)
            active_track_ids_in_frame.add(tid)
            current_frame_tracks[tid] = track

        # --- A. Update Observed Tracks ---
        for tid, track in current_frame_tracks.items():
            x1, y1, x2, y2 = float(track.x1), float(track.y1), float(track.x2), float(track.y2)
            anchor_x, anchor_y = self._get_anchor_point(x1, y1, x2, y2)
            cls_name = getattr(track, "aic21_class_name", "Car")
            cls_id = getattr(track, "aic21_class_id", 1)
            bbox = (x1, y1, x2, y2)

            # Instantaneous polygon test
            # Entry condition: point is strictly inside polygon (dist >= 0)
            # Exit condition: point is outside polygon past the hysteresis margin (dist < -boundary_tolerance)
            is_physically_inside, instant_road_id, dist = self._test_containment(anchor_x, anchor_y, margin=0.0)

            if tid not in self._track_states:
                if self._is_startup and self.track_initial_occupancy and is_physically_inside:
                    # Vehicle already present inside ROI at analysis startup (initial occupancy mode)
                    self._track_states[tid] = {
                        "state": TrackROIState.INSIDE,
                        "road_id": instant_road_id,
                        "candidate_road_id": None,
                        "inside_frames": self.entry_confirm_frames,
                        "outside_frames": 0,
                        "lost_frames": 0,
                        "last_seen_frame": frame_id,
                        "last_seen_ts": timestamp_sec,
                        "class_name": cls_name,
                        "class_id": cls_id,
                        "anchor_x": anchor_x,
                        "anchor_y": anchor_y,
                        "bbox": bbox,
                    }
                    self._initial_occupancy += 1
                    self._current_occupants.add(tid)
                    self._cumulative_class_counts[cls_name] += 1
                    if instant_road_id:
                        self._cumulative_road_counts[instant_road_id] += 1
                    self._flow_events.append((timestamp_sec, tid))
                    log.debug(f"[ROI STARTUP OCCUPANT] track={tid} ({cls_name}) at ({anchor_x:.1f},{anchor_y:.1f})")
                else:
                    initial_state = TrackROIState.ENTERING if is_physically_inside else TrackROIState.OUTSIDE
                    inside_init = 1 if is_physically_inside else 0
                    self._track_states[tid] = {
                        "state": initial_state,
                        "road_id": instant_road_id if is_physically_inside else None,
                        "candidate_road_id": instant_road_id if is_physically_inside else None,
                        "inside_frames": inside_init,
                        "outside_frames": 0 if is_physically_inside else 1,
                        "lost_frames": 0,
                        "last_seen_frame": frame_id,
                        "last_seen_ts": timestamp_sec,
                        "class_name": cls_name,
                        "class_id": cls_id,
                        "anchor_x": anchor_x,
                        "anchor_y": anchor_y,
                        "bbox": bbox,
                    }
                    # If entry confirmation threshold is 1, confirm immediately
                    if is_physically_inside and inside_init >= self.entry_confirm_frames:
                        self._confirm_entry(tid, frame_id, timestamp_sec, cls_name, cls_id, instant_road_id, anchor_x, anchor_y, bbox)
                continue

            # --- EXISTING TRACK UPDATE ---
            tstate = self._track_states[tid]
            tstate["last_seen_frame"] = frame_id
            tstate["last_seen_ts"] = timestamp_sec
            tstate["lost_frames"] = 0
            tstate["anchor_x"] = anchor_x
            tstate["anchor_y"] = anchor_y
            tstate["bbox"] = bbox
            tstate["class_name"] = cls_name
            tstate["class_id"] = cls_id

            current_st = tstate["state"]

            if current_st in (TrackROIState.INSIDE, TrackROIState.EXITING):
                # Currently considered inside ROI (or exiting candidate)
                # Check if it has crossed outside beyond hysteresis threshold
                # Dual threshold: to exit, dist must fall below -self.boundary_tolerance
                if dist < -self.boundary_tolerance:
                    tstate["outside_frames"] += 1
                    tstate["inside_frames"] = 0
                    tstate["state"] = TrackROIState.EXITING

                    if tstate["outside_frames"] >= self.exit_confirm_frames:
                        self._confirm_exit(tid, frame_id, timestamp_sec, cls_name, cls_id, tstate["road_id"], anchor_x, anchor_y, bbox, reason="normal")
                else:
                    # Vehicle remains comfortably inside or within boundary buffer
                    tstate["outside_frames"] = 0
                    tstate["inside_frames"] += 1
                    tstate["state"] = TrackROIState.INSIDE
                    if instant_road_id:
                        tstate["road_id"] = instant_road_id

            elif current_st in (TrackROIState.OUTSIDE, TrackROIState.ENTERING, TrackROIState.EXPIRED):
                # Currently considered outside ROI (or entering candidate)
                if dist >= 0.0:  # Physically inside polygon
                    tstate["inside_frames"] += 1
                    tstate["outside_frames"] = 0
                    tstate["state"] = TrackROIState.ENTERING
                    tstate["candidate_road_id"] = instant_road_id

                    if tstate["inside_frames"] >= self.entry_confirm_frames:
                        self._confirm_entry(tid, frame_id, timestamp_sec, cls_name, cls_id, instant_road_id, anchor_x, anchor_y, bbox)
                else:
                    tstate["inside_frames"] = 0
                    tstate["outside_frames"] += 1
                    tstate["state"] = TrackROIState.OUTSIDE
                    tstate["candidate_road_id"] = None

        # --- B. Handle Missing / Occluded Tracks (Grace Period & Expiration) ---
        for tid, tstate in list(self._track_states.items()):
            if tid not in active_track_ids_in_frame:
                tstate["lost_frames"] += 1

                if tstate["state"] in (TrackROIState.INSIDE, TrackROIState.EXITING, TrackROIState.LOST):
                    if tstate["lost_frames"] <= self.lost_grace_frames:
                        # In grace period: keep inside count stable, do not jitter on single-frame miss
                        tstate["state"] = TrackROIState.LOST
                    else:
                        # Exceeded grace timeout: track expired
                        self._confirm_exit(
                            tid,
                            frame_id,
                            timestamp_sec,
                            tstate["class_name"],
                            tstate["class_id"],
                            tstate["road_id"],
                            tstate["anchor_x"],
                            tstate["anchor_y"],
                            tstate["bbox"],
                            reason="expired",
                        )
                        tstate["state"] = TrackROIState.EXPIRED

                elif tstate["state"] == TrackROIState.ENTERING:
                    # Cancel unconfirmed entry if track vanished
                    if tstate["lost_frames"] > 3:
                        tstate["state"] = TrackROIState.OUTSIDE
                        tstate["inside_frames"] = 0

        # After first frame is processed, startup phase is complete
        if self._is_startup:
            self._is_startup = False

        # Invariant self-check
        self.assert_invariant()

        return self.get_snapshot()

    def _confirm_entry(
        self,
        track_id: int,
        frame_id: int,
        timestamp_sec: float,
        class_name: str,
        class_id: int,
        road_id: Optional[str],
        anchor_x: float,
        anchor_y: float,
        bbox: Tuple[float, float, float, float],
    ) -> None:
        """Confirm a validated entry event into the ROI."""
        tstate = self._track_states[track_id]
        tstate["state"] = TrackROIState.INSIDE
        tstate["road_id"] = road_id
        tstate["inside_frames"] = self.entry_confirm_frames
        tstate["outside_frames"] = 0

        if track_id not in self._current_occupants:
            self._current_occupants.add(track_id)
            self._cumulative_entries += 1
            self._cumulative_class_counts[class_name] += 1
            if road_id:
                self._cumulative_road_counts[road_id] += 1
            self._flow_events.append((timestamp_sec, track_id))

            ev = ROIEntryEvent(
                track_id=track_id,
                frame_id=frame_id,
                timestamp_sec=timestamp_sec,
                class_name=class_name,
                class_id=class_id,
                road_id=road_id or "roi",
                anchor_x=anchor_x,
                anchor_y=anchor_y,
                bbox=bbox,
            )
            self._entry_events.append(ev)
            log.debug(
                f"[ROI ENTRY] track={track_id} ({class_name}) road={road_id} "
                f"frame={frame_id} current_count={len(self._current_occupants)}"
            )

    def _confirm_exit(
        self,
        track_id: int,
        frame_id: int,
        timestamp_sec: float,
        class_name: str,
        class_id: int,
        road_id: Optional[str],
        anchor_x: float,
        anchor_y: float,
        bbox: Tuple[float, float, float, float],
        reason: str = "normal",
    ) -> None:
        """Confirm a validated exit event from the ROI."""
        tstate = self._track_states[track_id]
        tstate["state"] = TrackROIState.OUTSIDE if reason == "normal" else TrackROIState.EXPIRED
        tstate["road_id"] = None
        tstate["outside_frames"] = self.exit_confirm_frames
        tstate["inside_frames"] = 0

        if track_id in self._current_occupants:
            self._current_occupants.remove(track_id)
            self._cumulative_exits += 1

            ev = ROIExitEvent(
                track_id=track_id,
                frame_id=frame_id,
                timestamp_sec=timestamp_sec,
                class_name=class_name,
                class_id=class_id,
                road_id=road_id or "roi",
                anchor_x=anchor_x,
                anchor_y=anchor_y,
                bbox=bbox,
                reason=reason,
            )
            self._exit_events.append(ev)
            log.debug(
                f"[ROI EXIT] track={track_id} ({class_name}) reason={reason} "
                f"frame={frame_id} current_count={len(self._current_occupants)}"
            )

    def assert_invariant(self) -> bool:
        """
        Validate the fundamental counting formula:
        Current count == Initial occupancy + Cumulative entries - Cumulative exits
        """
        expected_current = self._initial_occupancy + self._cumulative_entries - self._cumulative_exits
        actual_current = len(self._current_occupants)
        if actual_current != expected_current:
            log.error(
                f"[ROI INVARIANT VIOLATION] Actual={actual_current} != "
                f"Expected={expected_current} (Initial={self._initial_occupancy} + "
                f"Entries={self._cumulative_entries} - Exits={self._cumulative_exits})"
            )
            return False
        return True

    def get_snapshot(self) -> ROICountSnapshot:
        """Generate structured count snapshot for rendering and API telemetry."""
        # Class breakdown of active occupants inside ROI
        active_classes: Dict[str, int] = defaultdict(int)
        active_roads: Dict[str, int] = defaultdict(int)
        track_states: Dict[int, str] = {}

        for tid in self._current_occupants:
            tinfo = self._track_states.get(tid)
            if tinfo:
                active_classes[tinfo["class_name"]] += 1
                rid = tinfo.get("road_id")
                if rid:
                    active_roads[rid] += 1

        for tid, tinfo in self._track_states.items():
            st = tinfo["state"]
            if st in (TrackROIState.INSIDE, TrackROIState.LOST):
                track_states[tid] = "INSIDE"
            elif st == TrackROIState.ENTERING:
                track_states[tid] = "ENTERING"
            elif st == TrackROIState.EXITING:
                track_states[tid] = "EXITING"
            else:
                track_states[tid] = "OUTSIDE"

        curr_count = len(self._current_occupants)
        inv_ok = (curr_count == self._initial_occupancy + self._cumulative_entries - self._cumulative_exits)

        return ROICountSnapshot(
            current_count=curr_count,
            initial_occupancy=self._initial_occupancy,
            cumulative_entries=self._cumulative_entries,
            cumulative_exits=self._cumulative_exits,
            invariant_verified=inv_ok,
            active_track_ids=list(self._current_occupants),
            by_class=dict(active_classes),
            by_road=dict(active_roads),
            cumulative_by_class=dict(self._cumulative_class_counts),
            cumulative_by_road=dict(self._cumulative_road_counts),
            track_states=track_states,
        )

    def get_flow_rate(self, current_timestamp_sec: float) -> float:
        """Vehicles per minute entering the ROI within rolling time window."""
        cutoff = current_timestamp_sec - self.time_window
        while self._flow_events and self._flow_events[0][0] < cutoff:
            self._flow_events.popleft()

        unique_ids = set(ev[1] for ev in self._flow_events)
        effective_window_sec = max(5.0, min(self.time_window, current_timestamp_sec))
        return round(len(unique_ids) / (effective_window_sec / 60.0), 1)

    def reset(self) -> None:
        """Reset counter state between videos."""
        self._initial_occupancy = 0
        self._cumulative_entries = 0
        self._cumulative_exits = 0
        self._current_occupants.clear()
        self._cumulative_class_counts.clear()
        self._cumulative_road_counts.clear()
        self._track_states.clear()
        self._entry_events.clear()
        self._exit_events.clear()
        self._flow_events.clear()
        self._is_startup = True
        self._startup_frame = None


# ==============================================================================
# Legacy Adapter: VehicleCounter
# ==============================================================================

@dataclass
class CountResult:
    """Snapshot of accumulated vehicle counts (AIC21 compatibility)."""
    total: int = 0
    by_class: Dict[str, int] = field(default_factory=dict)
    by_movement: Dict[int, int] = field(default_factory=dict)
    by_movement_and_class: Dict[int, Dict[str, int]] = field(default_factory=dict)
    by_time_window: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "by_class": dict(self.by_class),
            "by_movement": {str(k): v for k, v in self.by_movement.items()},
            "by_movement_and_class": {
                str(k): dict(v)
                for k, v in self.by_movement_and_class.items()
            },
            "by_time_window": dict(self.by_time_window),
        }


class VehicleCounter:
    """
    Accumulates unique vehicle counts on ROI Entry and assigns movements.
    Backward-compatible adapter that wraps ROI entry/exit recording.
    """

    def __init__(self, time_window_seconds: float = 60.0):
        self.time_window = time_window_seconds

        # Cumulative counters
        self._total: int = 0
        self._counted_track_ids: set = set()
        self._by_class: Dict[str, int] = defaultdict(int)
        self._by_movement: Dict[int, int] = defaultdict(int)
        self._by_movement_and_class: Dict[int, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._by_time_window: Dict[str, int] = defaultdict(int)
        self._entry_events: List[EntryEvent] = []
        self._exit_events: List[ExitEvent] = []

        log.info(f"VehicleCounter ready | time_window={time_window_seconds}s")

    def register_entry(self, event: EntryEvent) -> None:
        """Record one vehicle entry (= one unique count)."""
        if event.track_id in self._counted_track_ids:
            return

        self._counted_track_ids.add(event.track_id)
        self._total += 1
        self._by_class[event.aic21_class_name] += 1

        # Time bucket
        bucket_idx = int(event.timestamp_sec // self.time_window)
        t_start = bucket_idx * self.time_window
        t_end = t_start + self.time_window
        bucket_key = f"{t_start:.0f}-{t_end:.0f}s"
        self._by_time_window[bucket_key] += 1

        self._entry_events.append(event)

    def register_entries(self, events: List[EntryEvent]) -> None:
        for event in events:
            self.register_entry(event)

    def register_exit(self, event: ExitEvent) -> None:
        if event.movement_id is not None:
            self._by_movement[event.movement_id] += 1
            self._by_movement_and_class[event.movement_id][event.aic21_class_name] += 1

        self._exit_events.append(event)

    def register_exits(self, events: List[ExitEvent]) -> None:
        for event in events:
            self.register_exit(event)

    def get_result(self) -> CountResult:
        return CountResult(
            total=self._total,
            by_class=dict(self._by_class),
            by_movement=dict(self._by_movement),
            by_movement_and_class={
                k: dict(v) for k, v in self._by_movement_and_class.items()
            },
            by_time_window=dict(self._by_time_window),
        )

    def get_flow_rate(
        self,
        window_start_sec: float,
        window_end_sec: float,
        movement_id: Optional[int] = None,
    ) -> float:
        window_duration_min = (window_end_sec - window_start_sec) / 60.0
        if window_duration_min <= 0:
            return 0.0

        count = 0
        for event in self._exit_events:
            if window_start_sec <= event.timestamp_sec < window_end_sec:
                if movement_id is None or event.movement_id == movement_id:
                    count += 1

        return count / window_duration_min

    def get_all_exit_events(self) -> List[ExitEvent]:
        return list(self._exit_events)

    def get_counts_at_frame(self, max_frame_id: int) -> CountResult:
        result = CountResult()
        by_class: Dict[str, int] = defaultdict(int)
        by_movement: Dict[int, int] = defaultdict(int)
        by_movement_and_class: Dict[int, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        for event in self._exit_events:
            if event.frame_id <= max_frame_id:
                result.total += 1
                by_class[event.aic21_class_name] += 1
                if event.movement_id is not None:
                    by_movement[event.movement_id] += 1
                    by_movement_and_class[event.movement_id][event.aic21_class_name] += 1

        result.by_class = dict(by_class)
        result.by_movement = dict(by_movement)
        result.by_movement_and_class = {k: dict(v) for k, v in by_movement_and_class.items()}
        return result

    def reset(self) -> None:
        self._total = 0
        self._counted_track_ids.clear()
        self._by_class.clear()
        self._by_movement.clear()
        self._by_movement_and_class.clear()
        self._by_time_window.clear()
        self._exit_events.clear()
        self._entry_events.clear()

    def summary_str(self) -> str:
        lines = [
            f"Total vehicles counted: {self._total}",
            f"  Cars:   {self._by_class.get('Car', 0)}",
            f"  Trucks: {self._by_class.get('Truck', 0)}",
            "  By movement:",
        ]
        for mov_id in sorted(self._by_movement.keys()):
            lines.append(
                f"    Movement {mov_id}: {self._by_movement[mov_id]} "
                f"(Car={self._by_movement_and_class[mov_id].get('Car', 0)}, "
                f"Truck={self._by_movement_and_class[mov_id].get('Truck', 0)})"
            )
        return "\n".join(lines)

