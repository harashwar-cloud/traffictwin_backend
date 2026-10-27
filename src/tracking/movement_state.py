"""
src/tracking/movement_state.py
------------------------------
Robust Moving vs Stationary vehicle classifier with:
  - Bounding-box center jitter smoothing (Exponential Moving Average)
  - Perspective-aware displacement thresholding
  - Multi-frame observation window
  - Hysteresis confirmation to prevent state flickering
  - Three distinct states: "MOVING", "STATIONARY", "UNKNOWN"
"""

import collections
import logging
import math
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


class VehicleMovementDetector:
    """
    Classifies tracked vehicles into:
      - MOVING: displacement exceeds movement threshold over observation window.
      - STATIONARY: movement remains below threshold for the configured duration.
      - UNKNOWN: insufficient observations or unreliable tracking history.

    Accounts for bounding-box jitter via EMA coordinate smoothing
    and camera perspective scaling.
    """

    def __init__(
        self,
        window_seconds: float = 0.9,
        start_moving_threshold_px_per_sec: float = 14.0,
        stop_moving_threshold_px_per_sec: float = 7.0,
        moving_confirm_frames: int = 3,
        rest_confirm_frames: int = 5,
        min_history_seconds: float = 0.4,
        smoothing_alpha: float = 0.65,
        frame_height: float = 1080.0,
    ):
        self.window_seconds = window_seconds
        self.start_moving_threshold = start_moving_threshold_px_per_sec
        self.stop_moving_threshold = stop_moving_threshold_px_per_sec
        self.moving_confirm_frames = moving_confirm_frames
        self.rest_confirm_frames = rest_confirm_frames
        self.min_history_seconds = min_history_seconds
        self.smoothing_alpha = smoothing_alpha
        self.frame_height = frame_height

        # track_id -> state dict
        self.states: Dict[int, dict] = {}

    def update(
        self,
        track_id: int,
        x: float,
        y: float,
        timestamp_sec: float,
    ) -> str:
        """
        Updates position history with coordinate smoothing and returns current state:
        'MOVING', 'STATIONARY', or 'UNKNOWN'
        """
        if track_id not in self.states:
            self.states[track_id] = {
                "state": "UNKNOWN",
                "candidate_state": None,
                "candidate_frames": 0,
                "smoothed_x": float(x),
                "smoothed_y": float(y),
                # deque of (timestamp, smoothed_x, smoothed_y)
                "history": collections.deque(),
                "stationary_since": None,
            }

        state = self.states[track_id]

        # 1. Bounding box coordinate smoothing (reduces detector jitter)
        alpha = self.smoothing_alpha
        sx = alpha * x + (1.0 - alpha) * state["smoothed_x"]
        sy = alpha * y + (1.0 - alpha) * state["smoothed_y"]
        state["smoothed_x"] = sx
        state["smoothed_y"] = sy

        history = state["history"]
        history.append((timestamp_sec, sx, sy))

        # 2. Prune history outside time window
        while len(history) > 1 and (timestamp_sec - history[0][0]) > self.window_seconds:
            history.popleft()

        dt_history = timestamp_sec - history[0][0]

        # 3. If insufficient history, state remains UNKNOWN (or previous locked state)
        if dt_history < self.min_history_seconds or len(history) < 3:
            return state["state"]

        # 4. Calculate displacement over the observation window
        old_t, old_x, old_y = history[0]
        dx = sx - old_x
        dy = sy - old_y
        pixel_distance = math.hypot(dx, dy)

        speed_px_s = pixel_distance / dt_history if dt_history > 0 else 0.0

        # Perspective factor: vehicles further up the frame (smaller y) cover less pixels per meter
        # Normalize so distant vehicles moving slowly in pixels are still correctly identified as moving
        y_norm = max(0.1, min(1.0, sy / max(1.0, self.frame_height)))
        perspective_factor = max(0.45, min(1.6, 0.45 + 1.15 * y_norm))
        normalized_speed = speed_px_s / perspective_factor

        # 5. Raw candidate state based on hysteresis thresholds
        if normalized_speed >= self.start_moving_threshold:
            raw_state = "MOVING"
            state["stationary_since"] = None
        elif normalized_speed <= self.stop_moving_threshold:
            raw_state = "STATIONARY"
            if state["stationary_since"] is None:
                state["stationary_since"] = timestamp_sec
        else:
            # In deadband between stop and start threshold: retain existing state
            raw_state = state["state"] if state["state"] != "UNKNOWN" else "STATIONARY"

        # 6. Hysteresis confirmation (requires N consecutive frames to switch state)
        current = state["state"]
        if current == "UNKNOWN":
            state["state"] = raw_state
            state["candidate_state"] = None
            state["candidate_frames"] = 0
        elif raw_state != current:
            if state["candidate_state"] == raw_state:
                state["candidate_frames"] += 1
            else:
                state["candidate_state"] = raw_state
                state["candidate_frames"] = 1

            required_frames = self.moving_confirm_frames if raw_state == "MOVING" else self.rest_confirm_frames
            if state["candidate_frames"] >= required_frames:
                state["state"] = raw_state
                state["candidate_state"] = None
                state["candidate_frames"] = 0
        else:
            state["candidate_state"] = None
            state["candidate_frames"] = 0

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"[MOVEMENT] ID={track_id} dist={pixel_distance:.1f}px "
                f"speed={speed_px_s:.1f}px/s (norm={normalized_speed:.1f}) state={state['state']}"
            )

        return state["state"]

    def cleanup_lost_tracks(self, active_track_ids: List[int]) -> None:
        """Remove state for tracks that are no longer active to prevent memory leaks."""
        active_set = set(active_track_ids)
        lost_ids = [tid for tid in self.states if tid not in active_set]
        for tid in lost_ids:
            del self.states[tid]

    def get_state(self, track_id: int) -> str:
        """Return the current classified state of track_id ('MOVING', 'STATIONARY', or 'UNKNOWN')."""
        st = self.states.get(track_id)
        return st["state"] if st else "UNKNOWN"

    def is_moving(self, track_id: int) -> bool:
        """Return True if vehicle is confirmed to be moving."""
        return self.get_state(track_id) == "MOVING"

    def is_stationary(self, track_id: int) -> bool:
        """Return True if vehicle is confirmed to be stationary."""
        return self.get_state(track_id) in ("STATIONARY", "REST")
