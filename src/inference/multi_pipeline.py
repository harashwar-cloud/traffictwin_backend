"""
src/inference/multi_pipeline.py
---------------------------------
Multi-Camera Simultaneous GPU Inference Engine for TrafficTwin.
Processes all 5 camera video feeds concurrently using shared GPU YOLOv8n Tensor Core FP16
while maintaining strictly independent ByteTrack, ROI, Movement, and Metric states per camera.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from src.analytics.traffic_state import TrafficState, TrafficStateEstimator
from src.counting.roi_counter import VehicleCounter
from src.data.annotation_loader import (
    CameraAnnotation,
    ROIPolygon,
    load_all_movements,
    load_all_rois,
)
from src.data.video_loader import get_video_info
from src.inference.visualizer import (
    create_video_writer,
    draw_count_panel,
    draw_roi,
    draw_tracks,
    draw_traffic_state_banner,
)
from src.movement.movement_assigner import MovementAssigner
from src.tracking.tracker import ByteTracker, TrackStore

log = logging.getLogger(__name__)


class MultiCameraPipeline:
    """
    Simultaneous 5-Camera Traffic Perception & Digital Twin Engine.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.half = "cuda" in self.device
        self.input_size = self.cfg.get("inference", {}).get("input_size", 640)
        self.confidence = self.cfg.get("detection", {}).get("confidence_threshold", 0.35)
        self.iou_threshold = self.cfg.get("detection", {}).get("iou_threshold", 0.45)
        self.max_frames = self.cfg.get("inference", {}).get("max_frames", 600)

        # Output paths
        self.output_dir_videos = self.project_root / "outputs" / "videos"
        self.output_dir_metrics = self.project_root / "outputs" / "metrics"
        self.output_dir_videos.mkdir(parents=True, exist_ok=True)
        self.output_dir_metrics.mkdir(parents=True, exist_ok=True)

        # Load Annotations & ROIs
        data_root = self.project_root.parent / "AIC21_Track1_Vehicle_Counting"
        roi_dir = data_root / "ROIs" if (data_root / "ROIs").exists() else data_root / "cam_roi"
        self.all_rois = load_all_rois(roi_dir)
        self.all_movements = load_all_movements(data_root / "movement_description")

        log.info(
            f"[MULTI-CAM] Initialized MultiCameraPipeline | device={self.device} "
            f"(half={self.half}) | ROIs loaded: {len(self.all_rois)}"
        )

    def run_all(
        self,
        camera_jobs: List[Dict[str, str]],  # [{"camera_id": "cam_5", "job_id": "job_123", "video_path": "..."}, ...]
        max_frames: Optional[int] = None,
        frame_callback: Optional[Callable[[str, np.ndarray, int, float, dict], None]] = None,
    ) -> Dict[str, dict]:
        """
        Run simultaneous ML processing for all requested cameras.
        """
        if max_frames is not None:
            self.max_frames = max_frames

        # 1. Initialize per-camera resources
        trackers: Dict[str, ByteTracker] = {}
        track_stores: Dict[str, TrackStore] = {}
        movement_assigners: Dict[str, MovementAssigner] = {}
        counters: Dict[str, VehicleCounter] = {}
        state_estimators: Dict[str, TrafficStateEstimator] = {}
        caps: Dict[str, cv2.VideoCapture] = {}
        writers: Dict[str, cv2.VideoWriter] = {}
        annotated_paths: Dict[str, str] = {}
        results: Dict[str, dict] = {}
        movement_descs_map: Dict[str, dict] = {}

        border_margin = float(self.cfg.get("roi_filtering", {}).get("roi_border_margin_pixels", 0.0))
        ref_point = self.cfg.get("roi_filtering", {}).get("reference_point", "bottom_center")

        for c_info in camera_jobs:
            cid = c_info["camera_id"]
            jid = c_info["job_id"]
            vpath = str(c_info["video_path"])

            # Camera-specific ROI & Movements
            roi = self.all_rois.get(cid)
            camera_ann = self.all_movements.get(cid, CameraAnnotation(
                camera_id=cid, intersection_type="signalized", camera_type="standard", movements={}, notes=[]
            ))
            movement_descs = {
                m_id: m.description for m_id, m in camera_ann.movements.items()
            }
            movement_descs_map[cid] = movement_descs

            # Create independent instances
            trackers[cid] = ByteTracker(
                device=self.device,
                confidence=self.confidence,
                iou_threshold=self.iou_threshold,
                input_size=self.input_size,
                use_half=self.half,
            )
            track_stores[cid] = TrackStore()
            if roi:
                movement_assigners[cid] = MovementAssigner(
                    roi=roi,
                    camera_annotation=camera_ann,
                    min_track_length=int(self.cfg.get("movement", {}).get("min_track_length", 5)),
                    exit_margin=float(self.cfg.get("movement", {}).get("exit_boundary_margin", 20)),
                    border_margin=border_margin,
                    reference_point=ref_point,
                )
            else:
                movement_assigners[cid] = None

            counters[cid] = VehicleCounter(time_window_seconds=60.0)
            state_estimators[cid] = TrafficStateEstimator(
                time_window_seconds=float(self.cfg.get("intelligence", {}).get("time_window_seconds", 60.0)),
                thresholds=self.cfg.get("intelligence", {}).get("thresholds", {}),
                imbalance_threshold=float(self.cfg.get("intelligence", {}).get("direction_imbalance", {}).get("imbalance_threshold", 3.0)),
            )

            # Open Video
            cap = cv2.VideoCapture(vpath)
            if not cap.isOpened():
                log.error(f"[MULTI-CAM] Could not open video for {cid}: {vpath}")
                continue
            caps[cid] = cap

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
            temp_annotated = str(self.output_dir_videos / f"temp_{jid}_annotated.mp4")
            annotated_paths[cid] = temp_annotated
            writers[cid] = create_video_writer(temp_annotated, fps=fps, width=width, height=height)

        log.info(f"[MULTI-CAM] Processing {len(caps)} cameras simultaneously on GPU...")

        frame_idx = 0
        t_global_start = time.perf_counter()

        # 2. Main Multi-Camera Loop
        while True:
            frame_idx += 1
            if self.max_frames and frame_idx > self.max_frames:
                break

            any_frame_read = False
            t_batch_start = time.perf_counter()

            for c_info in camera_jobs:
                cid = c_info["camera_id"]
                jid = c_info["job_id"]

                if cid not in caps:
                    continue

                cap = caps[cid]
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                any_frame_read = True
                timestamp_sec = frame_idx / 10.0

                # Inference on GPU
                tracker = trackers[cid]
                tracks = tracker.track_frame(frame, frame_idx, timestamp_sec)
                track_stores[cid].update(tracks)

                # ROI Entry Counting + Movement Assignment
                m_assigner = movement_assigners.get(cid)
                if m_assigner:
                    entry_events, exit_events = m_assigner.process_frame(
                        frame_idx, timestamp_sec, track_stores[cid], tracks
                    )
                    counters[cid].register_entries(entry_events)
                    counters[cid].register_exits(exit_events)
                    for ev in exit_events:
                        state_estimators[cid].add_exit_event(
                            timestamp_sec, ev.movement_id, ev.aic21_class_name
                        )

                # Periodic Traffic State Calculation
                current_state = None
                if frame_idx % 30 == 0 or frame_idx == 1:
                    current_state = state_estimators[cid].compute_state(
                        timestamp_sec, movement_descs_map.get(cid, {})
                    )

                # Visualization / HUD
                annotated = frame.copy()
                roi = self.all_rois.get(cid)
                if roi:
                    annotated = draw_roi(annotated, roi)
                annotated = draw_tracks(annotated, tracks, roi=roi, show_outside_roi=False)
                snap = counters[cid].get_result()
                annotated = draw_count_panel(
                    annotated,
                    total=snap.total,
                    car_count=snap.by_class.get("Car", 0),
                    truck_count=snap.by_class.get("Truck", 0),
                    frame_id=frame_idx,
                    timestamp_sec=timestamp_sec,
                    traffic_state=current_state.overall_state.value if current_state else "LOW",
                )
                if current_state:
                    annotated = draw_traffic_state_banner(annotated, current_state)

                if cid in writers and writers[cid] is not None:
                    writers[cid].write(annotated)

                # Build real-time live payload
                count_snap = counters[cid].get_result()
                cur_dur_min = max(0.05, timestamp_sec / 60.0)
                cur_flow = round(count_snap.total / cur_dur_min, 1)
                cur_queue = min(85, max(2, int(cur_flow * 0.42)))

                live_payload = {
                    "job_id": jid,
                    "camera_id": cid,
                    "frame_index": frame_idx,
                    "total_frames": self.max_frames,
                    "video_time_seconds": round(timestamp_sec, 2),
                    "vehicles_counted": count_snap.total,
                    "vehicle_classes": {
                        "Car": count_snap.by_class.get("Car", 0),
                        "Truck": count_snap.by_class.get("Truck", 0),
                    },
                    "movement_counts": {str(k): v for k, v in count_snap.by_movement.items()},
                    "traffic_state": current_state.overall_state.value if current_state else "LOW",
                    "arrival_rate": cur_flow,
                    "queue": cur_queue,
                    "processing_fps": round(1.0 / max(1e-4, time.perf_counter() - t_batch_start), 1),
                    "direction_imbalance": {
                        "detected": current_state.direction_imbalance.detected if current_state else False,
                        "ratio": current_state.direction_imbalance.ratio if current_state else 1.0,
                        "dominant_movement": current_state.direction_imbalance.dominant_movement if current_state else None,
                    },
                }

                if frame_callback is not None:
                    try:
                        frame_callback(cid, annotated, frame_idx, timestamp_sec, live_payload)
                    except Exception as cb_err:
                        log.debug(f"Multi-cam callback error: {cb_err}")

            if not any_frame_read:
                break

        # 3. Clean up resources & finalize outputs
        for cid, cap in caps.items():
            cap.release()

        for cid, writer in writers.items():
            writer.release()

        total_elapsed = time.perf_counter() - t_global_start
        log.info(f"[MULTI-CAM] All {len(camera_jobs)} camera processing loops finished in {total_elapsed:.2f}s")

        # Convert each to H.264 MP4 and save JSON
        for c_info in camera_jobs:
            cid = c_info["camera_id"]
            jid = c_info["job_id"]

            temp_path = Path(annotated_paths.get(cid, ""))
            final_h264 = self.output_dir_videos / f"{jid}_annotated.mp4"

            if temp_path.exists():
                try:
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", str(temp_path),
                        "-c:v", "libx264",
                        "-preset", "veryfast",
                        "-crf", "23",
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        str(final_h264)
                    ]
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if temp_path.exists():
                        temp_path.unlink()
                except Exception as fe:
                    log.warning(f"FFmpeg conversion error for {cid}: {fe}")
                    if temp_path.exists():
                        temp_path.rename(final_h264)

            count_res = counters[cid].get_result()
            final_state = state_estimators[cid].compute_state(
                self.max_frames / 10.0, movement_descs_map.get(cid, {})
            )
            flow_rate = round(count_res.total / max(0.05, (self.max_frames / 10.0) / 60.0), 1)

            res_dict = {
                "video": Path(c_info["video_path"]).name,
                "camera_id": cid,
                "job_id": jid,
                "total_vehicles": count_res.total,
                "vehicle_classes": dict(count_res.by_class),
                "movement_counts": {str(k): v for k, v in count_res.by_movement.items()},
                "traffic_state": final_state.overall_state.value,
                "traffic_features": {
                    "total_flow_rate_per_min": flow_rate,
                    "direction_imbalance_detected": final_state.direction_imbalance.detected,
                    "direction_imbalance_ratio": final_state.direction_imbalance.ratio,
                },
                "processing": {
                    "frames_processed": self.max_frames,
                    "processing_time_sec": round(total_elapsed, 2),
                    "fps": round(self.max_frames / max(0.1, total_elapsed), 1),
                },
                "outputs": {
                    "annotated_video": str(final_h264),
                    "json_results": str(self.output_dir_metrics / f"{jid}_results.json")
                }
            }

            with open(self.output_dir_metrics / f"{jid}_results.json", "w") as f:
                json.dump(res_dict, f, indent=2)

            results[cid] = res_dict

        return results
