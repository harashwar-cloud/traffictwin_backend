"""
src/tracking/tracker.py
------------------------
ByteTrack-based multi-object tracker for AIC21 vehicle tracking.
Uses ultralytics built-in tracker (no additional install required).

Each vehicle gets a persistent track_id that survives brief occlusions.
Trajectory history is maintained in TrackStore.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Track:
    """A single tracked vehicle on a specific frame."""
    track_id: int
    frame_id: int
    timestamp_sec: float
    aic21_class_name: str        # "Car", "Truck", "Motorcycle"
    aic21_class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Vehicle contact point with the road: ((x1 + x2)/2, y2)."""
        return ((self.x1 + self.x2) / 2.0, float(self.y2))

    def is_inside_roi(self, roi, margin: float = 10.0, reference_point: str = "bottom_center") -> bool:
        if roi is None:
            return True
        if reference_point == "bottom_center":
            return roi.contains_bottom_center(self.x1, self.y1, self.x2, self.y2, margin=margin)
        return roi.contains_point(self.center_x, self.center_y, margin=margin)

    def is_countable(self) -> bool:
        return self.aic21_class_name in ("Car", "Bus", "Truck", "Motorcycle", "Bicycle")

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "frame_id": self.frame_id,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "aic21_class_name": self.aic21_class_name,
            "aic21_class_id": self.aic21_class_id,
            "confidence": round(self.confidence, 4),
            "x1": round(self.x1, 1),
            "y1": round(self.y1, 1),
            "x2": round(self.x2, 1),
            "y2": round(self.y2, 1),
            "center_x": round(self.center_x, 1),
            "center_y": round(self.center_y, 1),
        }


# ---------------------------------------------------------------------------
# TrackStore — persistent trajectory memory
# ---------------------------------------------------------------------------

class TrackStore:
    """
    Stores the trajectory (sequence of positions) for every track_id.
    Used by movement assigner and ROI counter to determine exit direction.
    """

    def __init__(self, max_trajectory_len: int = 150):
        """
        Args:
            max_trajectory_len: Maximum number of positions stored per track.
                                Older positions are discarded (FIFO).
        """
        self.max_len = max_trajectory_len
        # {track_id: [(frame_id, cx, cy), ...]}
        self._trajectories: Dict[int, List[Tuple[int, float, float]]] = {}
        # {track_id: class_name}  — most recently seen class
        self._classes: Dict[int, str] = {}
        # {track_id: bool}  — has this track been counted?
        self._counted: Dict[int, bool] = {}
        # {track_id: int}  — assigned movement_id at time of counting
        self._movement: Dict[int, Optional[int]] = {}

    def update(self, tracks: List[Track]) -> None:
        """Register new track observations for this frame."""
        for t in tracks:
            if t.track_id not in self._trajectories:
                self._trajectories[t.track_id] = []
                self._counted[t.track_id] = False
                self._movement[t.track_id] = None
            traj = self._trajectories[t.track_id]
            traj.append((t.frame_id, t.center_x, t.center_y))
            if len(traj) > self.max_len:
                traj.pop(0)
            self._classes[t.track_id] = t.aic21_class_name

    def get_trajectory(self, track_id: int) -> List[Tuple[int, float, float]]:
        """Return [(frame_id, cx, cy), ...] for a track_id."""
        return self._trajectories.get(track_id, [])

    def get_last_positions(self, track_id: int, n: int = 5) -> List[Tuple[float, float]]:
        """Return the last N (cx, cy) positions of a track."""
        traj = self._trajectories.get(track_id, [])
        return [(p[1], p[2]) for p in traj[-n:]]

    def get_class(self, track_id: int) -> str:
        return self._classes.get(track_id, "Unknown")

    def is_counted(self, track_id: int) -> bool:
        return self._counted.get(track_id, False)

    def mark_counted(self, track_id: int, movement_id: Optional[int] = None) -> None:
        self._counted[track_id] = True
        self._movement[track_id] = movement_id

    def get_movement(self, track_id: int) -> Optional[int]:
        return self._movement.get(track_id, None)

    def track_length(self, track_id: int) -> int:
        return len(self._trajectories.get(track_id, []))

    def all_track_ids(self) -> List[int]:
        return list(self._trajectories.keys())

    def active_track_ids(self, current_frame_id: int, max_age: int = 30) -> List[int]:
        """Return track IDs seen within the last max_age frames."""
        active = []
        for tid, traj in self._trajectories.items():
            if traj and (current_frame_id - traj[-1][0]) <= max_age:
                active.append(tid)
        return active

    def cleanup_expired(self, current_frame_id: int, max_age: int = 60) -> int:
        """
        Prune inactive tracks not seen for max_age frames.
        Keeps trajectories, classes, and movement lookups O(active_tracks) and bounded in memory.
        """
        expired = [
            tid for tid, traj in self._trajectories.items()
            if traj and (current_frame_id - traj[-1][0]) > max_age
        ]
        for tid in expired:
            self._trajectories.pop(tid, None)
            self._classes.pop(tid, None)
            self._counted.pop(tid, None)
            self._movement.pop(tid, None)
        return len(expired)



# ---------------------------------------------------------------------------
# ByteTracker — wraps ultralytics YOLO tracker
# ---------------------------------------------------------------------------

# COCO IDs for vehicles (must match detector)
# 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
TARGET_COCO_IDS = {1, 2, 3, 5, 7}
COCO_TO_AIC21 = {1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
AIC21_CLASS_ID = {"Car": 1, "Truck": 2, "Bus": 3, "Motorcycle": 4, "Bicycle": 5}


_YOLO_WEIGHTS_CACHE: Dict[Tuple[str, str], torch.nn.Module] = {}
_TRACKER_LOCK = threading.Lock()


def get_yolo_for_session(model_path: Optional[str], device: str) -> YOLO:
    """
    Returns an independent YOLO instance with its own predictor and ByteTrack state,
    while sharing the underlying PyTorch neural network weights in GPU memory.
    This prevents cross-video tracking collisions while maintaining a minimal VRAM footprint (~25MB total).
    """
    resolved_path = str(model_path) if (model_path and Path(model_path).exists()) else "yolov8n.pt"
    cache_key = (resolved_path, device)
    with _TRACKER_LOCK:
        model_instance = YOLO(resolved_path)
        if cache_key in _YOLO_WEIGHTS_CACHE:
            # Share the loaded PyTorch nn.Module directly (zero extra VRAM allocation)
            model_instance.model = _YOLO_WEIGHTS_CACHE[cache_key]
        else:
            log.info(f"[GPU-CACHE] Loading base YOLO model into weights cache: {resolved_path} on {device}")
            model_instance.to(device)
            _YOLO_WEIGHTS_CACHE[cache_key] = model_instance.model
        return model_instance


class ByteTracker:
    """
    Wraps ultralytics model.track() to provide ByteTrack.
    Returns List[Track] per frame, with persistent track_ids.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = 0.35,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        device: Optional[str] = None,
        track_buffer: int = 30,
        use_half: Optional[bool] = None,
        drone_mode: bool = False,
    ):
        if device is None or device == "cuda":
            if torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"
        self.device = str(device)
        self.drone_mode = drone_mode
        self.confidence = 0.22 if drone_mode else confidence
        self.iou_threshold = 0.55 if drone_mode else iou_threshold
        self.input_size = 960 if drone_mode and input_size < 960 else input_size
        self.track_buffer = 45 if drone_mode else track_buffer
        self.half = use_half if use_half is not None else ("cuda" in self.device)
        self.last_inference_ms = 0.0
        self.last_tracking_ms = 0.0
        self.last_preprocess_ms = 0.0
        self.last_postprocess_ms = 0.0
        self.last_pure_tracking_ms = 0.0
        self.last_total_tracker_ms = 0.0

        # Each tracker gets its own YOLO wrapper so ByteTrack state is isolated
        self.model = get_yolo_for_session(model_path, self.device)
        self._frame_count = 0

        # GPU Warmup (only on first init)
        try:
            dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            kwargs = {"imgsz": self.input_size, "device": self.device, "verbose": False}
            if self.device != "cpu":
                kwargs["half"] = self.half
                
            with torch.inference_mode():
                self.model.predict(dummy, **kwargs)
            
            device_type = "CUDA" if "cuda" in self.device else "CPU"
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
            precision = "FP16" if self.half and device_type == "CUDA" else "FP32"
            
            log.info(
                f"[{device_type}] ByteTracker ready | device={self.device} (precision={precision}) | "
                f"GPU={gpu_name} | imgsz={self.input_size} | drone_mode={drone_mode}"
            )
        except Exception as e:
            log.warning(f"Tracker warmup exception: {e}")

    def reset(self):
        """Reset internal tracker state for this session."""
        self._frame_count = 0
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            self.model.predictor.trackers = []

    def track_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp_sec: float = 0.0,
    ) -> List[Track]:
        """
        Run detection + ByteTrack on a single frame on GPU with inference_mode().

        Args:
            frame:         BGR numpy array.
            frame_id:      Original frame index in the video.
            timestamp_sec: Time offset in seconds.

        Returns:
            List of Track objects with persistent track_ids.
        """
        kwargs = {
            "conf": self.confidence,
            "iou": self.iou_threshold,
            "imgsz": self.input_size,
            "device": self.device,
            "persist": True,
            "verbose": False,
            "classes": list(TARGET_COCO_IDS),
            "tracker": "bytetrack.yaml",
        }
        if self.device != "cpu":
            kwargs["half"] = self.half
            
        t0 = time.perf_counter()
        with _TRACKER_LOCK:
            with torch.inference_mode():
                results = self.model.track(frame, **kwargs)
        total_call_ms = (time.perf_counter() - t0) * 1000.0
        self.last_total_tracker_ms = total_call_ms

        for r in results:
            if hasattr(r, "speed") and r.speed:
                self.last_preprocess_ms = float(r.speed.get("preprocess", 0.0))
                self.last_inference_ms = float(r.speed.get("inference", 0.0))
                self.last_postprocess_ms = float(r.speed.get("postprocess", 0.0))

        # Reconciled timing: ByteTrack tracking association time is exclusive of detection/inference
        det_time = self.last_preprocess_ms + self.last_inference_ms + self.last_postprocess_ms
        self.last_pure_tracking_ms = max(0.05, total_call_ms - det_time)
        # Store pure tracking time to prevent double-counting in dashboard stage sums
        self.last_tracking_ms = self.last_pure_tracking_ms

        tracks = []
        for result in results:
            boxes = result.boxes
            if boxes is None or boxes.id is None or len(boxes) == 0:
                continue

            # Vectorized GPU -> CPU batch transfer
            # (Eliminates dozens of blocking per-scalar cudaStreamSynchronize calls per frame)
            boxes_cpu = boxes.cpu()
            if boxes_cpu.id is None:
                continue

            ids = boxes_cpu.id.numpy().astype(int)
            clss = boxes_cpu.cls.numpy().astype(int)
            confs = boxes_cpu.conf.numpy().astype(float)
            xyxys = boxes_cpu.xyxy.numpy().astype(float)

            for i in range(len(ids)):
                coco_id = int(clss[i])
                if coco_id not in TARGET_COCO_IDS:
                    continue
                aic21_name = COCO_TO_AIC21[coco_id]
                box_xyxy = xyxys[i]
                tracks.append(Track(
                    track_id=int(ids[i]),
                    frame_id=frame_id,
                    timestamp_sec=timestamp_sec,
                    aic21_class_name=aic21_name,
                    aic21_class_id=AIC21_CLASS_ID[aic21_name],
                    confidence=float(confs[i]),
                    x1=float(box_xyxy[0]),
                    y1=float(box_xyxy[1]),
                    x2=float(box_xyxy[2]),
                    y2=float(box_xyxy[3]),
                ))

        self._frame_count += 1
        return tracks

    def reset(self) -> None:
        """Reset tracker state (call between different videos)."""
        with _TRACKER_LOCK:
            self._frame_count = 0
            if hasattr(self.model, "predictor") and self.model.predictor is not None:
                predictor = self.model.predictor
                if hasattr(predictor, "trackers") and predictor.trackers is not None:
                    for t in predictor.trackers:
                        if hasattr(t, "reset"):
                            try:
                                t.reset()
                            except Exception:
                                pass
                if hasattr(predictor, "vid_path"):
                    try:
                        predictor.vid_path = None
                    except Exception:
                        pass
            log.debug("Tracker state reset.")
