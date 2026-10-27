"""
src/inference/pipeline.py
--------------------------
Full TrafficTwin inference pipeline — Multi-Road Edition.

Flow:
  INPUT VIDEO
      ↓ Video validation
      ↓ Frame iteration (cv2, in-memory)
      ↓ Camera ROI + Multi-Road ROI loading
      ↓ Road configuration diagnostic
      ↓ YOLOv8n detection + ByteTrack tracking (full frame)
      ↓ TrackStore trajectory recording
      ↓ ROAD ASSIGNMENT (bottom_center → road polygon)
      ↓ Per-road vehicle counting + flow + queue + traffic_state
      ↓ Global analytics (sum of roads, no double-count)
      ↓ ROI-exit movement assignment
      ↓ Annotated frame with multi-road polygons + road ID on boxes
      ↓ Live MJPEG frame buffer + SSE telemetry (roads array included)
      ↓ Video writer → processed.mp4
      ↓ FFmpeg H.264 conversion → browser-compatible

Usage:
  from src.inference.pipeline import TrafficPipeline
  pipeline = TrafficPipeline(config_path="configs/config.yaml")
  result = pipeline.run(video_path="...", camera_id="cam_5")
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import logging
import math
import queue
import subprocess
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import psutil
import torch
import yaml

from src.data.annotation_loader import (
    CameraAnnotation,
    ROIPolygon,
    load_all_movements,
    load_all_rois,
    load_counting_gt,
    summarize_gt,
)
from src.data.video_loader import get_video_info, iter_frames, validate_video
from src.data.road_config import (
    RoadROI,
    assign_track_to_road,
    load_road_config,
    print_road_diagnostic,
)
from src.tracking.tracker import ByteTracker, Track, TrackStore
from src.tracking.road_assigner import RoadAssigner
from src.tracking.movement_state import VehicleMovementDetector
from src.movement.movement_assigner import MovementAssigner
from src.counting.roi_counter import VehicleCounter, ROIVehicleCounter, ROICountSnapshot
from src.analytics.traffic_state import TrafficStateEstimator, TrafficState
from src.inference.visualizer import (
    create_video_writer,
    draw_count_panel,
    draw_road_rois,
    draw_roi,
    draw_tracks,
    draw_traffic_state_banner,
    save_road_diagnostic_image,
)

log = logging.getLogger(__name__)


def finalize_browser_video(raw_video_path: Path) -> Tuple[bool, str]:
    """
    Converts OpenCV generated video to browser-compatible H.264 MP4 with +faststart
    and validates readability, frame count, and dimensions.
    """
    if not raw_video_path.exists() or raw_video_path.stat().st_size == 0:
        return False, "Raw video file was not generated or is 0 bytes."

    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = None

    if ffmpeg_exe and Path(ffmpeg_exe).exists():
        temp_out = raw_video_path.parent / f"temp_{raw_video_path.name}"
        cmd = [
            ffmpeg_exe, "-y",
            "-i", str(raw_video_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-movflags", "+faststart",
            str(temp_out)
        ]
        log.info(f"Converting video to browser-compatible H.264: {raw_video_path.name}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and temp_out.exists() and temp_out.stat().st_size > 0:
            try:
                raw_video_path.unlink(missing_ok=True)
                temp_out.rename(raw_video_path)
                log.info(f"Browser H.264 video finalized: {raw_video_path} ({raw_video_path.stat().st_size} bytes)")
            except Exception as e:
                log.warning(f"File rename exception: {e}")
        else:
            log.warning(f"FFmpeg conversion failed, retaining raw file: {proc.stderr}")
            if temp_out.exists():
                temp_out.unlink(missing_ok=True)

    # Validate output with OpenCV
    cap = cv2.VideoCapture(str(raw_video_path))
    if not cap.isOpened():
        return False, "Converted video cannot be opened by OpenCV decoder"

    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, frame = cap.read()
    cap.release()

    if fc <= 0 or w <= 0 or h <= 0 or not ret:
        return False, f"Invalid video stream: frames={fc}, w={w}, h={h}, first_frame={'OK' if ret else 'FAIL'}"

    return True, "OK"


class AsyncVideoWriter:
    """
    Non-blocking background video writer with a bounded FIFO queue.
    Decouples OpenCV disk encoding from the GPU inference thread.
    """
    def __init__(self, raw_writer, max_queue_size: int = 60):
        self.raw_writer = raw_writer
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while True:
            frame = self.queue.get()
            if frame is None:
                self.queue.task_done()
                break
            try:
                self.raw_writer.write(frame)
            except Exception as e:
                log.error(f"AsyncVideoWriter frame write error: {e}")
            finally:
                self.queue.task_done()

    def write(self, frame: np.ndarray):
        if not self.running:
            return
        try:
            self.queue.put(frame, timeout=2.0)
        except queue.Full:
            log.warning("AsyncVideoWriter queue full; skipping frame write to preserve memory")

    def release(self):
        self.running = False
        try:
            self.queue.put(None, timeout=2.0)
            self.thread.join(timeout=15.0)
        except Exception as e:
            log.warning(f"AsyncVideoWriter join warning: {e}")
        try:
            self.raw_writer.release()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Per-Road State Tracker
# ---------------------------------------------------------------------------

class RoadState:
    """
    Independent analytics state for a single road ROI:
      - Active tracked vehicles inside ROI (source of truth: active track IDs)
      - Cumulative entries and cumulative exits
      - Vehicle trip entry/exit timing and speed calculation (v_i = d_i / t_i * 3.6)
      - Arithmetic mean average speed of valid completed trips (v_avg = sum(v_i) / N)
      - Moving and stationary vehicle counts
      - Class breakdown
      - Traffic congestion state
    """

    def __init__(
        self,
        road_id: str,
        name: str,
        window_seconds: float = 30.0,
        length_meters: Optional[float] = None
    ):
        self.road_id = road_id
        self.name = name
        self.window_seconds = window_seconds
        self.length_meters = length_meters
        self.is_calibrated: bool = (length_meters is not None and length_meters > 0)

        # Active vehicle IDs currently inside this road ROI (source of truth)
        self._active_occupant_ids: Set[int] = set()

        # Cumulative counters
        self.cumulative_entries: int = 0
        self.cumulative_exits: int = 0

        # Unique vehicle IDs ever registered on this road
        self._cumulative_vehicle_ids: Set[int] = set()
        self._cumulative_class_counts: Dict[str, int] = defaultdict(int)

        # Active vehicle trips: {track_id: dict(entry_time, entry_pos, last_time, last_pos, frame_count, class_name)}
        self._active_trips: Dict[int, Dict[str, Any]] = {}

        # Valid completed trips and arithmetic mean speed
        self.completed_trips: int = 0
        self.completed_speeds: List[float] = []
        self.sum_speeds: float = 0.0
        self.average_speed: Optional[float] = None  # None when 0 valid trips completed

        # Speed history for instantaneous tracking: {track_id: deque([(timestamp, ref_x, ref_y), ...], maxlen=30)}
        self._speed_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))

        # Speed history records for completed trips debugging/verification
        self._speed_records: List[Dict[str, Any]] = []

        # Flow events: deque of (timestamp_sec, track_id)
        self._flow_events: deque = deque()

    def update_frame(
        self,
        current_tracks_in_road: Dict[int, Tuple[float, float, str]],  # tid -> (ref_x, ref_y, cls_name)
        timestamp_sec: float,
        road_roi: Any,
        frame_height: int = 1080,
        all_active_track_ids: Optional[Set[int]] = None,
    ) -> None:
        """
        Process frame-level road occupancy and trip state transitions:
          1. Detect newly entered vehicles -> increment cumulative_entries, start trip timer.
          2. Update ongoing vehicles inside -> update last position & timestamp.
          3. Detect exiting vehicles -> increment cumulative_exits, complete trip, compute speed.
          4. Exclude expired tracks or incomplete trips without fabricating speeds.
        """
        current_ids = set(current_tracks_in_road.keys())
        prev_ids = self._active_occupant_ids

        # --- A. NEW ENTRIES ---
        entered_ids = current_ids - prev_ids
        for tid in entered_ids:
            ref_x, ref_y, cls_name = current_tracks_in_road[tid]
            self.cumulative_entries += 1
            self._cumulative_vehicle_ids.add(tid)
            self._cumulative_class_counts[cls_name] += 1
            self._flow_events.append((timestamp_sec, tid))
            self._speed_history[tid].append((timestamp_sec, ref_x, ref_y))

            # Start measuring vehicle trip travel time
            self._active_trips[tid] = {
                "entry_time": timestamp_sec,
                "entry_pos": (ref_x, ref_y),
                "last_time": timestamp_sec,
                "last_pos": (ref_x, ref_y),
                "frame_count": 1,
                "class_name": cls_name,
            }

        # --- B. REMAINING INSIDE ---
        stayed_ids = current_ids & prev_ids
        for tid in stayed_ids:
            ref_x, ref_y, _ = current_tracks_in_road[tid]
            self._speed_history[tid].append((timestamp_sec, ref_x, ref_y))
            if tid in self._active_trips:
                self._active_trips[tid]["last_time"] = timestamp_sec
                self._active_trips[tid]["last_pos"] = (ref_x, ref_y)
                self._active_trips[tid]["frame_count"] += 1

        # --- C. EXITS ---
        exited_ids = prev_ids - current_ids
        for tid in exited_ids:
            self.cumulative_exits += 1

            if tid in self._active_trips:
                trip = self._active_trips.pop(tid)
                t_entry = trip["entry_time"]
                t_exit = timestamp_sec
                dt = t_exit - t_entry
                entry_pos = trip["entry_pos"]
                exit_pos = trip["last_pos"]

                # Check if vehicle confirmed exit (remains in active frame tracks or had multiple frames inside)
                # Reject instantaneous jitter or zero/negative elapsed time
                if dt >= 0.35 and trip["frame_count"] >= 3:
                    dist_m, is_calibrated = road_roi.get_calibrated_distance_meters(
                        entry_pos, exit_pos, frame_height
                    )
                    if is_calibrated and dist_m >= 3.0:
                        speed_kmh = (dist_m / dt) * 3.6
                        # Physical plausibility filter
                        if 3.0 <= speed_kmh <= 180.0:
                            self.completed_trips += 1
                            speed_val = round(speed_kmh, 1)
                            self.completed_speeds.append(speed_val)
                            self.sum_speeds += speed_kmh
                            self.average_speed = round(self.sum_speeds / self.completed_trips, 1)
                            self._speed_records.append({
                                "track_id": tid,
                                "entry_time": round(t_entry, 2),
                                "exit_time": round(t_exit, 2),
                                "elapsed_sec": round(dt, 2),
                                "distance_meters": round(dist_m, 1),
                                "speed_kmh": speed_val,
                            })
                            log.debug(
                                f"[ROAD TRIP SPEED] {self.name} ({self.road_id}) ID={tid}: "
                                f"{dist_m:.1f}m in {dt:.2f}s -> {speed_kmh:.1f} km/h (N={self.completed_trips})"
                            )

        # --- D. UPDATE ACTIVE OCCUPANTS (SOURCE OF TRUTH) ---
        self._active_occupant_ids = current_ids

    def register_vehicle(self, track_id: int, class_name: str, timestamp_sec: float) -> bool:
        """Compatibility helper for external callers."""
        is_new_entry = track_id not in self._active_occupant_ids
        if is_new_entry:
            self._active_occupant_ids.add(track_id)
            self._cumulative_vehicle_ids.add(track_id)
            self._cumulative_class_counts[class_name] += 1
            self.cumulative_entries += 1
            self._flow_events.append((timestamp_sec, track_id))
        return is_new_entry

    def sync_active_occupants(self, current_road_track_ids: List[int]) -> None:
        """Update active occupants set."""
        self._active_occupant_ids = set(current_road_track_ids)

    def update_speed(self, track_id: int, ref_x: float, ref_y: float, timestamp_sec: float) -> None:
        """Update active vehicle position."""
        self._speed_history[track_id].append((timestamp_sec, ref_x, ref_y))
        if track_id in self._active_trips:
            self._active_trips[track_id]["last_time"] = timestamp_sec
            self._active_trips[track_id]["last_pos"] = (ref_x, ref_y)
            self._active_trips[track_id]["frame_count"] += 1

    def get_flow_rate(self, current_timestamp: float) -> float:
        """Flow rate in vehicles/min through this road over rolling time window (30s)."""
        cutoff = current_timestamp - self.window_seconds
        while self._flow_events and self._flow_events[0][0] < cutoff:
            self._flow_events.popleft()

        if not self._flow_events:
            return 0.0

        unique_ids = set(ev[1] for ev in self._flow_events)
        effective_window_sec = max(5.0, min(self.window_seconds, current_timestamp))
        return round((len(unique_ids) * 60.0) / effective_window_sec, 1)

    def get_average_speed(self, active_track_ids: Optional[List[int]] = None, movement_states: Optional[Dict[int, str]] = None) -> float:
        """
        Calibrated average speed in km/h.
        If valid completed trips exist, returns arithmetic mean of completed trips.
        Otherwise, estimates from active moving tracks if present.
        """
        if self.completed_trips > 0 and self.average_speed is not None:
            return self.average_speed

        if not active_track_ids or not movement_states:
            return 0.0

        speeds = []
        for tid in active_track_ids:
            if movement_states.get(tid) != "MOVING":
                continue
            history = self._speed_history.get(tid)
            if not history or len(history) < 3:
                continue

            t_now, x_now, y_now = history[-1]
            t_old, x_old, y_old = history[0]
            dt = t_now - t_old
            if dt < 0.2:
                continue

            dist_px = math.hypot(x_now - x_old, y_now - y_old)
            y_norm = max(0.1, min(1.0, y_now / 1080.0))
            meters_per_px = 0.04 + (1.0 - y_norm) * 0.10
            dist_m = dist_px * meters_per_px
            speed_kmh = (dist_m / dt) * 3.6
            if 3.0 <= speed_kmh <= 180.0:
                speeds.append(speed_kmh)

        if not speeds:
            return 0.0
        return round(sum(speeds) / len(speeds), 1)

    def get_queue_length(self, active_track_ids: List[int], movement_states: Dict[int, str]) -> int:
        """Queue length = Number of currently active vehicles on this road that are STATIONARY."""
        return sum(1 for tid in active_track_ids if movement_states.get(tid) in ("STATIONARY", "REST"))

    def get_traffic_state(self, active_count: int, queue_count: int, avg_speed: float, flow_rate: float) -> str:
        """Determine road congestion state combining density, stationary ratio, and speed."""
        if active_count == 0:
            return "LOW"
        stationary_ratio = queue_count / max(1, active_count)
        if queue_count >= 6 or (active_count >= 10 and avg_speed > 0 and avg_speed < 8.0) or (stationary_ratio >= 0.75 and active_count >= 4):
            return "CRITICAL"
        elif queue_count >= 3 or (active_count >= 6 and avg_speed > 0 and avg_speed < 18.0) or stationary_ratio >= 0.4:
            return "HIGH"
        elif queue_count >= 1 or active_count >= 3 or flow_rate >= 8.0:
            return "MEDIUM"
        else:
            return "LOW"

    def to_summary(
        self,
        current_timestamp: float,
        active_track_ids: Optional[List[int]] = None,
        movement_states: Optional[Dict[int, str]] = None,
        active_class_counts: Optional[Dict[str, int]] = None
    ) -> Dict:
        """Return a structured per-road summary dict."""
        act_ids = active_track_ids if active_track_ids is not None else list(self._active_occupant_ids)
        m_states = movement_states or {}
        c_counts = active_class_counts or {}

        active_count = len(act_ids)
        moving_count = sum(1 for tid in act_ids if m_states.get(tid) == "MOVING")
        queue_count = self.get_queue_length(act_ids, m_states)
        flow = self.get_flow_rate(current_timestamp)
        speed = self.get_average_speed(act_ids, m_states)
        state = self.get_traffic_state(active_count, queue_count, speed, flow)

        if not self.is_calibrated:
            speed_display = "Calibration required"
        elif self.completed_trips == 0:
            speed_display = "Waiting for vehicles"
        else:
            speed_display = f"{self.average_speed:.1f} km/h"

        return {
            "road_id": self.road_id,
            "name": self.name,
            # Current visible occupants (source of truth: active track IDs)
            "current_active": active_count,
            "current_roi_count": active_count,
            "active_track_ids": act_ids,
            "current_moving": moving_count,
            "current_stationary": queue_count,
            "current_rest": queue_count,
            # Cumulative entries & exits
            "cumulative_entries": self.cumulative_entries,
            "cumulative_exits": self.cumulative_exits,
            "cumulative_total": len(self._cumulative_vehicle_ids),
            "vehicles": len(self._cumulative_vehicle_ids),
            "cumulative_vehicles": len(self._cumulative_vehicle_ids),
            # Speed & valid trips (Arithmetic Mean of completed trips)
            "completed_trips": self.completed_trips,
            "valid_trips": self.completed_trips,
            "average_speed": self.average_speed if (self.average_speed is not None and self.completed_trips > 0) else None,
            "speed_display": speed_display,
            "length_meters": self.length_meters,
            # Classes
            "cars": c_counts.get("Car", 0) if c_counts else self._cumulative_class_counts.get("Car", 0),
            "trucks": c_counts.get("Truck", 0) if c_counts else self._cumulative_class_counts.get("Truck", 0),
            "buses": c_counts.get("Bus", 0) if c_counts else self._cumulative_class_counts.get("Bus", 0),
            "motorcycles": (c_counts.get("Motorcycle", 0) + c_counts.get("Bicycle", 0)) if c_counts else (self._cumulative_class_counts.get("Motorcycle", 0) + self._cumulative_class_counts.get("Bicycle", 0)),
            "bicycles": c_counts.get("Bicycle", 0) if c_counts else self._cumulative_class_counts.get("Bicycle", 0),
            "active_classes": dict(c_counts),
            "cumulative_classes": dict(self._cumulative_class_counts),
            # Analytics
            "flow_volume": flow,
            "queue": queue_count,
            "traffic_state": state,
        }


def compute_total_roads_summary(road_summaries: List[Dict], road_states: Dict[str, RoadState]) -> Dict:
    """
    Compute aggregate TOTAL / ALL ROADS metrics:
    - Deduplicated active occupant count across all roads
    - Total cumulative entries and exits
    - Trip-weighted arithmetic mean speed: sum(all speeds) / sum(all valid trips)
    - Total moving and stationary counts
    - Total class breakdown
    """
    all_active_tids = set()
    total_entries = 0
    total_exits = 0
    total_completed_trips = 0
    total_sum_speeds = 0.0

    for rs in road_states.values():
        all_active_tids.update(rs._active_occupant_ids)
        total_entries += rs.cumulative_entries
        total_exits += rs.cumulative_exits
        total_completed_trips += rs.completed_trips
        total_sum_speeds += rs.sum_speeds

    total_moving = 0
    total_stationary = 0
    total_cars = 0
    total_trucks = 0
    total_buses = 0
    total_motorcycles = 0

    for s in road_summaries:
        if s.get("road_id") == "OUTSIDE":
            continue
        total_moving += s.get("current_moving", 0)
        total_stationary += s.get("current_stationary", 0)
        total_cars += s.get("cars", 0)
        total_trucks += s.get("trucks", 0)
        total_buses += s.get("buses", 0)
        total_motorcycles += s.get("motorcycles", 0)

    overall_avg_speed = round(total_sum_speeds / total_completed_trips, 1) if total_completed_trips > 0 else None
    any_calibrated = any(rs.is_calibrated for rs in road_states.values()) if road_states else True
    if total_completed_trips > 0 and overall_avg_speed is not None:
        speed_display = f"{overall_avg_speed:.1f} km/h"
    elif not any_calibrated:
        speed_display = "Calibration required"
    else:
        speed_display = "Waiting for vehicles"

    return {
        "road_id": "total",
        "name": "TOTAL / ALL ROADS",
        "current_active": len(all_active_tids),
        "current_roi_count": len(all_active_tids),
        "active_track_ids": list(all_active_tids),
        "cumulative_entries": total_entries,
        "cumulative_exits": total_exits,
        "completed_trips": total_completed_trips,
        "valid_trips": total_completed_trips,
        "average_speed": overall_avg_speed,
        "speed_display": speed_display,
        "current_moving": total_moving,
        "current_stationary": total_stationary,
        "cars": total_cars,
        "trucks": total_trucks,
        "buses": total_buses,
        "motorcycles": total_motorcycles,
        "traffic_state": "CRITICAL" if total_stationary >= 6 else ("HIGH" if total_stationary >= 3 else ("MEDIUM" if len(all_active_tids) >= 4 else "LOW")),
    }


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

class TrafficPipeline:
    """
    End-to-end TrafficTwin pipeline — Multi-Road Edition.

    Accepts one video + camera_id and returns structured results.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # Resolve key paths from config with portable fallbacks
        p = self.cfg["paths"]
        repo_root = Path(__file__).resolve().parent.parent.parent
        aic_root = repo_root.parent / "AIC21_Track1_Vehicle_Counting"

        self.roi_dir = Path(p["roi_dir"]) if Path(p["roi_dir"]).exists() else (aic_root / "ROIs")
        self.movement_dir = Path(p["movement_dir"]) if Path(p["movement_dir"]).exists() else (aic_root / "movement_description")
        gt_dir = Path(p.get("gt_dir", ""))
        self.gt_dir = gt_dir if gt_dir.exists() else (aic_root / "counting_gt_sample")
        self.output_dir = Path(p["outputs_dir"]) if Path(p["outputs_dir"]).exists() else (repo_root / "outputs")
        self.models_dir = Path(p["models_dir"]) if Path(p["models_dir"]).exists() else (repo_root / "models")

        # Road config directory: configs/roads/ (relative to config file)
        config_dir = Path(config_path).parent
        self.roads_config_dir = config_dir / "roads"
        self.roads_config_dir.mkdir(parents=True, exist_ok=True)

        det_cfg = self.cfg["detection"]
        track_cfg = self.cfg["tracking"]
        count_cfg = self.cfg["counting"]
        intel_cfg = self.cfg["intelligence"]
        infer_cfg = self.cfg["inference"]
        hw_cfg = self.cfg["hardware"]

        # Hardware & GPU CUDA resolution
        hw_device = hw_cfg.get("device", "cuda")
        if ("cuda" in hw_device or hw_device == "auto") and torch.cuda.is_available():
            self.device = "cuda:0"
            self.gpu_name = torch.cuda.get_device_name(0)
            self.vram_total_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 1)
            log.info(f"[CUDA] GPU acceleration enabled: {self.gpu_name} (VRAM: {self.vram_total_mb} MB)")
        else:
            self.device = "cpu"
            self.gpu_name = "CPU"
            self.vram_total_mb = 0.0
            log.info(f"[DEVICE] Inference device: {self.device}")
        self.frame_skip = infer_cfg.get("frame_skip", 0)
        self.max_frames = infer_cfg.get("max_frames", None)
        self.save_video = infer_cfg.get("save_annotated_video", True)
        self.save_json = infer_cfg.get("save_json_results", True)
        self.video_fps_out = infer_cfg.get("annotated_video_fps", 10)

        # Model path: use existing if available
        model_path = det_cfg.get("model_path") or None
        if model_path is None:
            fallback = Path("yolov8n.pt")
            if fallback.exists():
                model_path = str(fallback)
                log.info(f"Using existing model: {model_path}")

        # Drone / aerial mode setting
        self.drone_mode = bool(det_cfg.get("drone_mode", False))

        # Tracker (also handles detection)
        self.tracker = ByteTracker(
            model_path=model_path,
            confidence=det_cfg["confidence_threshold"],
            iou_threshold=det_cfg["iou_threshold"],
            input_size=det_cfg["input_size"],
            device=self.device,
            track_buffer=track_cfg.get("track_buffer", 30),
            drone_mode=self.drone_mode,
        )

        # Load all AIC21 annotations
        self.all_rois: Dict[str, ROIPolygon] = load_all_rois(self.roi_dir)
        self.all_movements: Dict[str, CameraAnnotation] = load_all_movements(self.movement_dir)

        # Intelligence config
        self.intelligence_window = intel_cfg.get("time_window_seconds", 60)
        self.intel_thresholds = intel_cfg.get("thresholds", {})
        self.imbalance_threshold = intel_cfg.get("direction_imbalance", {}).get("imbalance_threshold", 3.0)

        # ROI Filtering config
        roi_cfg = self.cfg.get("roi_filtering", {})
        self.border_margin = float(roi_cfg.get("roi_border_margin_pixels", 0.0))
        self.reference_point = str(roi_cfg.get("reference_point", "bottom_center"))
        self.show_outside_roi = bool(roi_cfg.get("show_outside_roi_detections", False))
        self.min_track_frames_confirm = int(roi_cfg.get("min_track_frames_confirm", 1))

        # Road Assignment Hysteresis config
        road_assign_cfg = self.cfg.get("road_assignment", {})
        self.road_assigner = RoadAssigner(
            switch_confirm_frames=int(road_assign_cfg.get("switch_confirm_frames", 10)),
            entry_confirm_frames=int(road_assign_cfg.get("entry_confirm_frames", 5)),
            lost_grace_frames=int(road_assign_cfg.get("lost_grace_frames", 15)),
            boundary_margin=float(road_assign_cfg.get("boundary_margin_pixels", 15.0)),
            reference_point=self.reference_point
        )
        
        # Movement State config
        movement_state_cfg = self.cfg.get("movement_state", {})
        self.movement_detector = VehicleMovementDetector(
            window_seconds=float(movement_state_cfg.get("window_seconds", 0.75)),
            start_moving_threshold_px_per_sec=float(movement_state_cfg.get("start_moving_threshold_px_per_sec", 12.0)),
            stop_moving_threshold_px_per_sec=float(movement_state_cfg.get("stop_moving_threshold_px_per_sec", 6.0)),
            moving_confirm_frames=int(movement_state_cfg.get("moving_confirm_frames", 3)),
            rest_confirm_frames=int(movement_state_cfg.get("rest_confirm_frames", 5)),
            min_history_seconds=float(movement_state_cfg.get("min_history_seconds", 0.5)),
        )

        log.info(
            f"TrafficPipeline (Multi-Road) initialized "
            f"(ROI margin={self.border_margin}px, ref={self.reference_point}, "
            f"show_outside={self.show_outside_roi})"
        )

    def run(
        self,
        video_path: str,
        camera_id: str,
        gt_csv_path: Optional[str] = None,
        output_prefix: Optional[str] = None,
        frame_callback=None,
        manual_roads: Optional[list] = None,
        stop_check: Optional[Any] = None,
    ) -> dict:
        """
        Run the full multi-road pipeline on one video.

        Args:
            video_path:    Path to the .mp4 file.
            camera_id:     Camera ID (e.g. "cam_5") — used to load ROI/movements/roads.
            gt_csv_path:   Optional path to counting GT CSV for evaluation.
            output_prefix: Optional output filename prefix.
            frame_callback: Optional callback(annotated_frame, frame_id, timestamp_sec, summary_dict)
                            for live streaming. The summary_dict includes a "roads" array.

        Returns:
            Dict with full structured results including per-road analytics.
        """
        video_path = Path(video_path)
        start_time = time.time()

        log.info("=" * 55)
        log.info(f"PIPELINE START (MULTI-ROAD): {video_path.name} | camera={camera_id}")
        log.info("=" * 55)

        # --- 1. Validate video ---
        valid, reason = validate_video(video_path)
        if not valid:
            raise RuntimeError(f"Video validation failed: {reason}")
            
        source_fps_override = self.cfg.get("video", {}).get("source_fps")

        video_info = get_video_info(video_path, source_fps=source_fps_override)
        log.info(
            f"Video: {video_info.filename} | "
            f"{video_info.width}x{video_info.height} | "
            f"{video_info.fps:.1f}fps | "
            f"{video_info.frame_count} frames | "
            f"{video_info.duration_seconds:.1f}s"
        )

        # --- 2. Load AIC21 annotations (ROI + movements) ---
        original_roi = self.all_rois.get(camera_id)
        cam_ann = self.all_movements.get(camera_id)

        available_rois = list(self.all_rois.keys())
        available_movs = list(self.all_movements.keys())

        if original_roi is None:
            raise ValueError(f"No ROI found for camera: {camera_id}. Available ROIs: {available_rois}")
        if cam_ann is None:
            raise ValueError(f"No movement description found for camera: {camera_id}. Available movements: {available_movs}")

        # Annotation baseline resolution
        source_res_w, source_res_h = 1920, 1080
        output_width, output_height = video_info.width, video_info.height

        # Scale the original ROI to actual video resolution
        scale_x = output_width / source_res_w
        scale_y = output_height / source_res_h
        scaled_vertices = []
        for x, y in original_roi.vertices:
            sx = min(output_width - 1, max(0, int(x * scale_x)))
            sy = min(output_height - 1, max(0, int(y * scale_y)))
            scaled_vertices.append((sx, sy))
        roi = ROIPolygon(camera_id=original_roi.camera_id, vertices=scaled_vertices)

        # --- 3. Load multi-road configuration ---
        if manual_roads:
            road_rois = []
            for r in manual_roads:
                # manual_roads polygon are passed directly from UI, 
                # which are ALREADY in the original video resolution
                length_m = r.get("length_meters", r.get("length", None))
                road_rois.append(RoadROI(
                    road_id=r.get("road_id", "manual_road"),
                    name=r.get("name", "Manual Road"),
                    color=(0, 255, 255),
                    vertices=[(int(p[0]), int(p[1])) for p in r.get("polygon", [])],
                    scaled_vertices=[(int(p[0]), int(p[1])) for p in r.get("polygon", [])],
                    length_meters=float(length_m) if length_m is not None else None,
                ))
        else:
            road_rois_raw = load_road_config(
                camera_id=camera_id,
                roads_config_dir=self.roads_config_dir,
                fallback_roi_vertices=original_roi.vertices,  # annotation space
            )

            # Scale road ROIs to actual video resolution
            road_rois: List[RoadROI] = [
                r.scale_to_resolution(source_res_w, source_res_h, output_width, output_height)
                for r in road_rois_raw
            ]

        # Print road configuration diagnostic
        prefix = output_prefix or video_path.stem
        print_road_diagnostic(prefix, video_path.name, camera_id, road_rois)

        log.info(
            f"\n[ROI]\n"
            f"camera_config={camera_id.upper()}\n"
            f"source_resolution={source_res_w}x{source_res_h}\n"
            f"roi_output_resolution={output_width}x{output_height}\n"
            f"roi_points={len(roi.vertices)}\n"
            f"roads_loaded={len(road_rois)}\n"
            f"road_ids={[r.road_id for r in road_rois]}\n"
            f"roi_valid=true\n"
        )

        # --- 4. Initialize per-video components ---
        self.tracker.reset()
        track_store = TrackStore(max_trajectory_len=150)
        movement_assigner = MovementAssigner(
            roi=roi,
            camera_annotation=cam_ann,
            min_track_length=self.cfg["counting"]["min_track_length"],
            border_margin=self.border_margin,
            reference_point=self.reference_point,
        )
        counter = VehicleCounter(time_window_seconds=self.intelligence_window)
        roi_counter = ROIVehicleCounter(
            road_rois=road_rois,
            reference_point=self.reference_point,
            boundary_tolerance=float(self.cfg.get("roi_filtering", {}).get("roi_border_margin_pixels", 8.0)),
            entry_confirm_frames=int(self.cfg.get("roi_filtering", {}).get("entry_confirm_frames", 2)),
            exit_confirm_frames=int(self.cfg.get("roi_filtering", {}).get("exit_confirm_frames", 3)),
            lost_grace_frames=int(self.cfg.get("roi_filtering", {}).get("lost_grace_frames", 12)),
            time_window_seconds=self.intelligence_window,
        )
        state_estimator = TrafficStateEstimator(
            time_window_seconds=self.intelligence_window,
            thresholds=self.intel_thresholds,
            imbalance_threshold=self.imbalance_threshold,
        )

        # Per-road state trackers
        road_states: Dict[str, RoadState] = {
            r.road_id: RoadState(r.road_id, r.name, self.intelligence_window, length_meters=r.length_meters)
            for r in road_rois
        }

        # --- 5. Prepare output paths with safe portable fallbacks ---
        repo_root = Path(__file__).resolve().parent.parent.parent
        configured_video_dir = Path(self.cfg.get("inference", {}).get("video_output_dir", ""))
        video_out_dir = configured_video_dir if (configured_video_dir.exists() and "Harik" not in str(configured_video_dir)) else (repo_root / "outputs" / "videos")

        configured_metrics_dir = Path(self.cfg.get("inference", {}).get("metrics_output_dir", ""))
        metrics_dir = configured_metrics_dir if (configured_metrics_dir.exists() and "Harik" not in str(configured_metrics_dir)) else (repo_root / "outputs" / "metrics")

        video_out_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # Diagnostic image path
        debug_dir = repo_root / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        diagnostic_path = str(debug_dir / f"{prefix}_road_mapping.jpg")

        annotated_path = str(video_out_dir / f"{prefix}_annotated.mp4")

        # --- 6. Video writer (Asynchronous background worker) ---
        writer = None
        if self.save_video:
            # Force output writer to match source FPS exactly
            output_fps = video_info.fps
            raw_writer = create_video_writer(
                annotated_path,
                video_info.width,
                video_info.height,
                fps=output_fps,
            )
            writer = AsyncVideoWriter(raw_writer, max_queue_size=60)
            log.info(f"Async annotated video writer initialized → {annotated_path}")

        # --- 7. Main processing loop ---
        movement_descs = {mid: m.description for mid, m in cam_ann.movements.items()}
        all_frame_results = []
        traffic_state_snapshots = []
        state_update_interval = max(1, int(video_info.fps * 10))
        current_state = None
        frames_processed = 0
        total_active_tracks = 0

        # Road assignment tracking: {track_id: road_id}
        current_road_assignment: Dict[int, str] = {}
        # Global entry events for rolling-window flow rate: deque of (timestamp_sec, track_id)
        global_flow_events = deque()

        # Whether we've saved the diagnostic image
        diagnostic_saved = False

        # Profiling and moving average trackers
        t_prev_frame = time.perf_counter()
        recent_latencies = deque(maxlen=20)
        last_perf = {}

        # --- 6.5 Pre-inference Geometry Validation ---
        from shapely.geometry import Polygon, LineString
        log.info(f"\n[GEOMETRY VALIDATION]\ncamera={camera_id}\n")
        
        try:
            # Validate ROI
            if not hasattr(roi, 'vertices') or not isinstance(roi.vertices, list):
                raise ValueError(f"Invalid ROI geometry: expected Polygon vertices, received {type(roi)}")
            
            # Validate Roads
            for r in road_rois:
                if not hasattr(r, 'vertices') or not isinstance(r.vertices, list):
                    raise ValueError(f"Invalid road geometry for {camera_id} {r.road_id}: expected Polygon, received {type(r)}")
                log.info(f"{r.road_id}:\n    type=Polygon\n    valid=true\n")
            
            # Validate Movement Segments
            for i, seg in enumerate(movement_assigner.boundary_segments):
                if not isinstance(seg, LineString):
                    raise ValueError(f"{camera_id} movement geometry error: movement_segment_{i}\nexpected LineString\nreceived {type(seg)}")
                log.info(f"movement_segment_{i}:\n    type=LineString\n    valid=true\n")
        except Exception as geom_e:
            log.error(str(geom_e))
            raise RuntimeError(str(geom_e)) from geom_e

        cumulative_total = 0
        current_visible = 0
        cumulative_classes = defaultdict(int)
        global_traffic_state = "LOW"
        global_flow_rate = 0.0
        global_avg_speed = None
        current_moving = 0
        current_queue = 0
        road_summaries = []
        tot_summary = {}
        roi_snap = ROICountSnapshot()
        t_decode_ms = 0.0
        t_analytics_ms = 0.0
        t_render_ms = 0.0
        t_write_ms = 0.0
        t_total_ms = 0.0
        current_fps = 0.0

        log.info(f"Processing frames (frame_skip={self.frame_skip}, max_frames={self.max_frames})...")

        for frame_id, timestamp_sec, frame in iter_frames(
            video_path,
            frame_skip=self.frame_skip,
            max_frames=self.max_frames,
            source_fps=source_fps_override,
        ):
            if stop_check and stop_check():
                log.info(f"Stop signal received for pipeline {prefix}. Halting frame processing gracefully.")
                break

            t_frame_start = time.perf_counter()
            t_decode_ms = max(0.1, (t_frame_start - t_prev_frame) * 1000.0)
            frames_processed += 1

            # --- STEP A: Save road diagnostic image from first frame ---
            if not diagnostic_saved:
                save_road_diagnostic_image(frame, road_rois, diagnostic_path, job_id=prefix)
                diagnostic_saved = True

            # --- STEP B: Detection + Tracking (full frame on GPU) ---
            t_track_start = time.perf_counter()
            tracks = self.tracker.track_frame(frame, frame_id, timestamp_sec)
            track_store.update(tracks)
            total_active_tracks += len(tracks)
            t_track_ms = (time.perf_counter() - t_track_start) * 1000.0

            # --- STEP C: Road Assignment & Movement State ---
            t_analytics_start = time.perf_counter()
            current_road_assignment = {}
            current_movement_state = {}
            active_track_ids = []
            active_tracks_by_road = defaultdict(list)
            active_classes_by_road = defaultdict(lambda: defaultdict(int))
            active_classes_global = defaultdict(int)
            tracks_info_by_road = defaultdict(dict)

            for track in tracks:
                active_track_ids.append(track.track_id)

                # 1. Update movement state (MOVING vs STATIONARY)
                ref_x = (track.x1 + track.x2) / 2.0
                ref_y = track.y2 if self.reference_point == "bottom_center" else (track.y1 + track.y2) / 2.0

                mov_state = self.movement_detector.update(
                    track.track_id, ref_x, ref_y, timestamp_sec
                )
                current_movement_state[track.track_id] = mov_state

                # 2. Update road assignment
                road_id = self.road_assigner.update(
                    track.track_id,
                    track.x1, track.y1, track.x2, track.y2,
                    road_rois
                )
                if road_id:
                    current_road_assignment[track.track_id] = road_id
                    active_tracks_by_road[road_id].append(track.track_id)
                    active_classes_by_road[road_id][track.aic21_class_name] += 1
                    active_classes_global[track.aic21_class_name] += 1
                    tracks_info_by_road[road_id][track.track_id] = (ref_x, ref_y, track.aic21_class_name)
                else:
                    current_road_assignment[track.track_id] = "OUTSIDE"

            # Update per-road state machine (occupancy source of truth, trip timing, distance & speed)
            for r in road_rois:
                road_states[r.road_id].update_frame(
                    current_tracks_in_road=tracks_info_by_road.get(r.road_id, {}),
                    timestamp_sec=timestamp_sec,
                    road_roi=r,
                    frame_height=video_info.height,
                    all_active_track_ids=set(active_track_ids),
                )

            # Prune expired track store trajectories to keep memory bounded and O(active)
            track_store.cleanup_expired(frame_id, max_age=60)

            # --- STEP D: Movement assignment + global ROI counting ---
            entry_events, exit_events = movement_assigner.process_frame(
                frame_id, timestamp_sec, track_store, tracks
            )
            counter.register_entries(entry_events)
            counter.register_exits(exit_events)

            for ev in exit_events:
                state_estimator.add_exit_event(
                    timestamp_sec, ev.movement_id, ev.aic21_class_name
                )

            # Cleanup lost tracks from road assigner and movement detector
            self.road_assigner.cleanup_lost_tracks(active_track_ids)
            self.movement_detector.cleanup_lost_tracks(active_track_ids)

            # --- STEP E: ROI Vehicle Counter (Exact Invariant & Jitter Suppression) ---
            roi_snap = roi_counter.update(frame_id, timestamp_sec, tracks)

            # Current active vehicles inside ROI (formula-verified)
            current_visible = roi_snap.current_count
            current_moving = sum(1 for tid in roi_snap.active_track_ids if current_movement_state.get(tid) == "MOVING")
            current_queue = sum(1 for tid in roi_snap.active_track_ids if current_movement_state.get(tid) in ("STATIONARY", "REST"))

            # Cumulative validated vehicle entries
            cumulative_total = roi_snap.cumulative_entries
            cumulative_classes = defaultdict(int, roi_snap.cumulative_by_class)
            if not cumulative_classes and counter.get_result().by_class:
                cumulative_classes = counter.get_result().by_class

            # D. Rolling-Window Flow Rate (30s window)
            cutoff = timestamp_sec - 30.0
            while global_flow_events and global_flow_events[0][0] < cutoff:
                global_flow_events.popleft()
            if not global_flow_events:
                global_flow_rate = 0.0
            else:
                unique_flow_ids = set(ev[1] for ev in global_flow_events)
                eff_w_sec = max(5.0, min(30.0, timestamp_sec))
                global_flow_rate = round((len(unique_flow_ids) * 60.0) / eff_w_sec, 1)

            # E. Trip-weighted Average Speed across roads for completed valid trips
            total_speeds_sum = sum(rs.sum_speeds for rs in road_states.values())
            total_trips_count = sum(rs.completed_trips for rs in road_states.values())
            global_avg_speed = round(total_speeds_sum / total_trips_count, 1) if total_trips_count > 0 else 0.0

            # F. Global Traffic Congestion State
            stat_ratio = current_queue / max(1, current_visible)
            if current_visible == 0:
                global_traffic_state = "LOW"
            elif current_queue >= 6 or (current_visible >= 10 and global_avg_speed < 8.0) or (stat_ratio >= 0.75 and current_visible >= 4):
                global_traffic_state = "CRITICAL"
            elif current_queue >= 3 or (current_visible >= 6 and global_avg_speed < 18.0) or stat_ratio >= 0.4:
                global_traffic_state = "HIGH"
            elif current_queue >= 1 or current_visible >= 3 or global_flow_rate >= 8.0:
                global_traffic_state = "MEDIUM"
            else:
                global_traffic_state = "LOW"

            # Periodic state snapshot for dataset evaluator
            if frames_processed % state_update_interval == 0 or frames_processed == 1:
                current_state = state_estimator.compute_state(timestamp_sec, movement_descs)
                traffic_state_snapshots.append(current_state.to_dict())

            # --- STEP F: Frame result record ---
            count_snap = counter.get_result()
            all_frame_results.append({
                "frame_id": frame_id,
                "timestamp_sec": round(timestamp_sec, 2),
                "active_tracks": len(tracks),
                "current_visible": current_visible,
                "current_moving": current_moving,
                "current_stationary": current_queue,
                "total_counted": cumulative_total,
                "flow_rate_per_min": global_flow_rate,
                "avg_speed": global_avg_speed,
                "traffic_state": global_traffic_state,
            })

            # --- End of Analytics Stage ---
            t_analytics_end = time.perf_counter()
            t_analytics_ms = (t_analytics_end - t_analytics_start) * 1000.0

            # --- STEP G: Annotate frame ---
            t_render_start = time.perf_counter()
            annotated_frame = None
            if writer is not None or frame_callback is not None:
                annotated_frame = frame.copy()

                # Draw multi-road polygons
                annotated_frame = draw_road_rois(annotated_frame, road_rois)

                # Draw tracked vehicles with road assignment labels
                annotated_frame = draw_tracks(
                    annotated_frame,
                    tracks,
                    roi=None,
                    road_rois=road_rois,
                    road_assignment=current_road_assignment,
                    movement_states=current_movement_state,
                    roi_states=roi_snap.track_states,
                    show_outside_roi=self.show_outside_roi,
                    margin=self.border_margin,
                    reference_point=self.reference_point,
                )

                # Collect per-road summaries for the panel
                road_summaries = []
                for rs in road_states.values():
                    r_act = active_tracks_by_road.get(rs.road_id, [])
                    r_cls = active_classes_by_road.get(rs.road_id, {})
                    summary = rs.to_summary(timestamp_sec, r_act, current_movement_state, r_cls)
                    road_summaries.append(summary)

                # Add OUTSIDE summary
                active_outside = [
                    tid for tid, rid in current_road_assignment.items() if rid == "OUTSIDE"
                ]
                if active_outside:
                    track_cls_map = {t.track_id: t.aic21_class_name for t in tracks}
                    moving_cnt = sum(1 for tid in active_outside if current_movement_state.get(tid) == "MOVING")
                    rest_cnt = sum(1 for tid in active_outside if current_movement_state.get(tid) in ("STATIONARY", "REST"))
                    outside_summary = {
                        "name": "OUTSIDE",
                        "road_id": "OUTSIDE",
                        "vehicles": 0,
                        "cumulative_vehicles": 0,
                        "cars": sum(1 for tid in active_outside if track_cls_map.get(tid) == "Car"),
                        "trucks": sum(1 for tid in active_outside if track_cls_map.get(tid) == "Truck"),
                        "buses": sum(1 for tid in active_outside if track_cls_map.get(tid) == "Bus"),
                        "motorcycles": sum(1 for tid in active_outside if track_cls_map.get(tid) in ("Motorcycle", "Bicycle")),
                        "bicycles": 0,
                        "flow_volume": 0.0,
                        "average_speed": 0.0,
                        "queue": rest_cnt,
                        "traffic_state": "LOW",
                        "current_active": len(active_outside),
                        "current_moving": moving_cnt,
                        "current_stationary": rest_cnt,
                        "current_rest": rest_cnt,
                    }
                    road_summaries.append(outside_summary)

                annotated_frame = draw_count_panel(
                    annotated_frame,
                    total=cumulative_total,
                    car_count=cumulative_classes.get("Car", 0),
                    truck_count=cumulative_classes.get("Truck", 0),
                    frame_id=frame_id,
                    timestamp_sec=timestamp_sec,
                    traffic_state=global_traffic_state,
                    road_summaries=road_summaries,
                    current_global=current_visible,
                    flow_rate=global_flow_rate,
                    avg_speed=global_avg_speed,
                    queue_length=current_queue,
                    bus_count=cumulative_classes.get("Bus", 0),
                    motorcycle_count=cumulative_classes.get("Motorcycle", 0) + cumulative_classes.get("Bicycle", 0),
                    initial_occupancy=roi_snap.initial_occupancy,
                    cumulative_entries=roi_snap.cumulative_entries,
                    cumulative_exits=roi_snap.cumulative_exits,
                    invariant_verified=roi_snap.invariant_verified,
                )

                if current_state:
                    annotated_frame = draw_traffic_state_banner(annotated_frame, current_state)

            t_render_end = time.perf_counter()
            t_render_ms = (t_render_end - t_render_start) * 1000.0 if annotated_frame is not None else 0.0

            # --- STEP H: Write to video (Async queue) ---
            t_write_start = time.perf_counter()
            if writer is not None and annotated_frame is not None:
                writer.write(annotated_frame)
            t_write_end = time.perf_counter()
            t_write_ms = (t_write_end - t_write_start) * 1000.0

            # --- Measure total frame duration & throughput ---
            t_frame_end = time.perf_counter()
            frame_duration_s = max(1e-4, t_frame_end - t_frame_start)
            t_total_ms = frame_duration_s * 1000.0
            recent_latencies.append(frame_duration_s)
            current_fps = len(recent_latencies) / sum(recent_latencies)
            t_prev_frame = time.perf_counter()

            # Compile genuine hardware & latency telemetry metrics
            gpu_allocated_mb = round(torch.cuda.memory_allocated() / (1024**2), 1) if torch.cuda.is_available() else 0.0
            gpu_reserved_mb = round(torch.cuda.memory_reserved() / (1024**2), 1) if torch.cuda.is_available() else 0.0
            vm = psutil.virtual_memory()

            pure_track_ms = getattr(self.tracker, "last_pure_tracking_ms", getattr(self.tracker, "last_tracking_ms", 0.0))
            perf = {
                "gpu_name": self.gpu_name,
                "device": self.tracker.device,
                "precision": "FP16" if getattr(self.tracker, "half", False) and "cuda" in self.tracker.device else "FP32",
                "vram_allocated_mb": gpu_allocated_mb,
                "vram_reserved_mb": gpu_reserved_mb,
                "vram_total_mb": self.vram_total_mb,
                "cpu_percent": psutil.cpu_percent(interval=None),
                "ram_used_gb": round(vm.used / (1024**3), 2),
                "ram_total_gb": round(vm.total / (1024**3), 2),
                "decode_ms": round(t_decode_ms, 2),
                "preprocess_ms": round(getattr(self.tracker, "last_preprocess_ms", 0.0), 2),
                "inference_ms": round(getattr(self.tracker, "last_inference_ms", 0.0), 2),
                "postprocess_ms": round(getattr(self.tracker, "last_postprocess_ms", 0.0), 2),
                "tracking_ms": round(pure_track_ms, 2),
                "analytics_ms": round(t_analytics_ms, 2),
                "render_ms": round(t_render_ms, 2),
                "write_ms": round(t_write_ms, 2),
                "total_latency_ms": round(t_total_ms, 2),
                "processed_fps": round(current_fps, 1),
                "source_fps": round(video_info.fps, 1),
                "frames_processed": frames_processed,
                "frames_skipped": self.frame_skip,
                "inference_imgsz": self.tracker.input_size,
                "flow_window_seconds": 30.0,
            }
            last_perf = perf

            # --- STEP I: Live frame callback (for streaming) ---
            if frame_callback is not None and annotated_frame is not None:
                try:
                    tot_summary = compute_total_roads_summary(road_summaries, road_states)
                    live_payload = {
                        "job_id": prefix,
                        "frame_index": frame_id,
                        "total_frames": self.max_frames or video_info.frame_count,
                        "video_time_seconds": round(timestamp_sec, 2),
                        "current_active": current_visible,
                        "current_roi_count": current_visible,
                        "initial_occupancy": roi_snap.initial_occupancy,
                        "cumulative_entries": roi_snap.cumulative_entries,
                        "cumulative_exits": roi_snap.cumulative_exits,
                        "invariant_verified": roi_snap.invariant_verified,
                        "current_moving": current_moving,
                        "current_stationary": current_queue,
                        "vehicles_counted": cumulative_total,
                        "vehicle_classes": dict(cumulative_classes),
                        "current_classes": dict(roi_snap.by_class),
                        "movement_counts": {str(k): v for k, v in count_snap.by_movement.items()},
                        "traffic_state": global_traffic_state,
                        "arrival_rate": global_flow_rate,
                        "flow_rate_per_min": global_flow_rate,
                        "queue": current_queue,
                        "completed_trips": tot_summary.get("completed_trips", 0),
                        "valid_trips": tot_summary.get("valid_trips", 0),
                        "average_speed": tot_summary.get("average_speed"),
                        "speed_display": tot_summary.get("speed_display", "Waiting for vehicles"),
                        "processing_fps": round(current_fps, 1),
                        "direction_imbalance": {
                            "detected": current_state.direction_imbalance.detected if current_state else False,
                            "ratio": current_state.direction_imbalance.ratio if current_state else 1.0,
                            "dominant_movement": current_state.direction_imbalance.dominant_movement if current_state else None,
                        },
                        "roads": road_summaries,
                        "total_road_summary": tot_summary,
                        "global": {
                            "vehicles": cumulative_total,
                            "current_active": current_visible,
                            "current_moving": current_moving,
                            "current_stationary": current_queue,
                            "flow_volume": global_flow_rate,
                            "traffic_state": global_traffic_state,
                            "queue": current_queue,
                            "average_speed": global_avg_speed,
                        },
                        "source": {
                            "filename": video_path.name,
                            "width": video_info.width,
                            "height": video_info.height,
                            "fps": round(video_info.fps, 1),
                        },
                        "perf": perf,
                    }
                    frame_callback(
                        annotated_frame,
                        frame_id,
                        timestamp_sec,
                        live_payload,
                    )
                except Exception as cb_err:
                    log.debug(f"Frame callback exception: {cb_err}")

        if writer:
            writer.release()
            log.info(f"OpenCV writer released for {annotated_path}")

        # Post-process: convert to browser-compatible H.264 MP4 with +faststart
        if self.save_video and Path(annotated_path).exists():
            success, msg = finalize_browser_video(Path(annotated_path))
            if not success:
                log.error(f"Video finalization warning: {msg}")
            else:
                log.info(f"Video validation verified: {annotated_path}")

        # --- 8. Final results ---
        elapsed = time.time() - start_time
        final_counts = counter.get_result()

        if current_state is None:
            current_state = state_estimator.compute_state(
                frames_processed / max(1, video_info.fps), movement_descs
            )

        log.info(f"\n{counter.summary_str()}")
        log.info(f"Processing time: {elapsed:.1f}s for {frames_processed} frames")
        if elapsed > 0:
            log.info(f"Speed: {frames_processed / elapsed:.1f} frames/sec")

        # Build final per-road summaries
        final_road_summaries = []
        final_timestamp = frames_processed / max(1, video_info.fps)
        for road in road_rois:
            rs = road_states[road.road_id]
            final_road_summaries.append(rs.to_summary(final_timestamp))

        result = {
            "video_id": video_path.stem,
            "camera_id": camera_id,
            "video_file": str(video_path),
            "video_info": {
                "width": video_info.width,
                "height": video_info.height,
                "fps": video_info.fps,
                "total_frames": video_info.frame_count,
                "duration_sec": video_info.duration_seconds,
            },
            "processing": {
                "frames_processed": frames_processed,
                "frame_skip": self.frame_skip,
                "processing_time_seconds": round(elapsed, 2),
                "speed_fps": round(frames_processed / elapsed, 1) if elapsed > 0 else 0,
                "device": "CUDA" if "cuda" in self.device else "CPU",
                "precision": "FP16" if self.tracker.half and "cuda" in self.device else "FP32",
                "model": "YOLOv8n (COCO pretrained)",
                "tracker": "ByteTrack",
            },
            "total_vehicles": cumulative_total if cumulative_total > 0 else final_counts.total,
            "cumulative_vehicles": cumulative_total,
            "current_vehicles": current_visible,
            "initial_occupancy": roi_snap.initial_occupancy,
            "cumulative_entries": roi_snap.cumulative_entries,
            "cumulative_exits": roi_snap.cumulative_exits,
            "invariant_verified": roi_snap.invariant_verified,
            "roi_counting": roi_snap.to_dict(),
            "vehicle_classes": dict(cumulative_classes) if cumulative_classes else final_counts.by_class,
            "movement_counts": {str(k): v for k, v in final_counts.by_movement.items()},
            "movement_class_counts": {
                str(k): dict(v) for k, v in final_counts.by_movement_and_class.items()
            },
            "time_window_counts": final_counts.by_time_window,
            "traffic_state": global_traffic_state,
            "traffic_features": {
                "total_flow_rate_per_min": global_flow_rate,
                "average_speed_kmh": global_avg_speed,
                "estimated_queue": current_queue,
                "direction_imbalance_detected": current_state.direction_imbalance.detected if current_state else False,
                "direction_imbalance_ratio": current_state.direction_imbalance.ratio if current_state else 1.0,
            },
            "traffic_state_history": traffic_state_snapshots[-10:],
            # NEW: per-road analytics
            "roads": final_road_summaries,
            "total_road_summary": compute_total_roads_summary(final_road_summaries, road_states),
            "road_count": len(road_rois),
            "performance_metrics": last_perf,
            "outputs": {
                "annotated_video": annotated_path if self.save_video else None,
                "diagnostic_image": diagnostic_path,
            },
        }

        # Write job metadata JSON
        job_metadata = {
            "job_id": prefix,
            "source_filename": video_path.name,
            "camera_config": camera_id,
            "video": {
                "width": video_info.width,
                "height": video_info.height,
                "fps": round(video_info.fps, 2),
                "frames": video_info.frame_count,
            },
            "roads": [{"road_id": r["road_id"], "name": r["name"]} for r in final_road_summaries],
            "output": {
                "path": annotated_path,
                "ready": Path(annotated_path).exists() and Path(annotated_path).stat().st_size > 0,
                "frames": frames_processed,
                "fps": min(video_info.fps, self.video_fps_out),
            },
        }
        try:
            metadata_path = metrics_dir / f"{prefix}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(job_metadata, f, indent=2, default=str)
        except Exception as me:
            log.warning(f"Could not save metadata: {me}")

        # --- 9. Evaluation vs GT (if provided) ---
        if gt_csv_path:
            from src.evaluation.counting_eval import evaluate_counting, save_counting_results_csv
            gt_events = load_counting_gt(Path(gt_csv_path))
            eval_max_frame = max(e.frame_id for e in gt_events) if gt_events else 600
            eval_result = evaluate_counting(
                counter=counter,
                gt_events=gt_events,
                max_frame_id=eval_max_frame,
                camera_id=camera_id,
                fps=video_info.fps,
            )
            eval_result.print_report()
            result["evaluation"] = eval_result.to_dict()

            csv_path = metrics_dir / f"{prefix}_counting_results.csv"
            save_counting_results_csv(eval_result, csv_path)

        # --- 10. Save JSON ---
        if self.save_json:
            json_path = metrics_dir / f"{prefix}_results.json"
            with open(json_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            result["outputs"]["json_results"] = str(json_path)
            log.info(f"JSON results → {json_path}")

        log.info("=" * 55)
        log.info("PIPELINE COMPLETE")
        log.info("=" * 55)

        return result
