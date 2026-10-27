"""
src/evaluation/counting_eval.py
---------------------------------
Evaluates predicted vehicle counts against AIC21 counting ground truth.

GT format: video_clip_id, frame_id, movement_id, vehicle_class_id
  Each row = one vehicle counted (at the frame it exits the ROI).

Metrics computed (all from real execution, never fabricated):
  - Total count MAE (predicted total vs GT total)
  - Per-movement MAE
  - Per-class MAE (Car, Truck)
  - Prediction accuracy (% movements within 20% of GT)

Note:
  Only cam_5 first 600 frames (1 minute) has publicly released GT.
  All metrics are ONLY valid for that window.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.data.annotation_loader import CountingEvent, load_counting_gt, summarize_gt
from src.counting.roi_counter import VehicleCounter

log = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Output of counting evaluation."""
    camera_id: str
    eval_frames: int              # number of frames evaluated
    eval_duration_sec: float

    # GT summary
    gt_total: int
    gt_by_class: Dict[str, int]
    gt_by_movement: Dict[int, int]

    # Predictions
    pred_total: int
    pred_by_class: Dict[str, int]
    pred_by_movement: Dict[int, int]

    # Metrics
    total_mae: float              # |pred_total - gt_total|
    total_error_pct: float        # % error vs GT
    per_movement_mae: float       # mean |pred_mov - gt_mov| across movements
    car_mae: float
    truck_mae: float
    movements_within_20pct: float  # % of movements with < 20% error

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "eval_frames": self.eval_frames,
            "eval_duration_sec": self.eval_duration_sec,
            "ground_truth": {
                "total": self.gt_total,
                "by_class": self.gt_by_class,
                "by_movement": {str(k): v for k, v in self.gt_by_movement.items()},
            },
            "predictions": {
                "total": self.pred_total,
                "by_class": self.pred_by_class,
                "by_movement": {str(k): v for k, v in self.pred_by_movement.items()},
            },
            "metrics": {
                "total_count_MAE": round(self.total_mae, 2),
                "total_error_pct": round(self.total_error_pct, 2),
                "per_movement_MAE": round(self.per_movement_mae, 2),
                "car_MAE": round(self.car_mae, 2),
                "truck_MAE": round(self.truck_mae, 2),
                "movements_within_20pct": round(self.movements_within_20pct, 2),
            },
            "note": (
                "Evaluation covers only cam_5 first 600 frames (1 minute at 10fps). "
                "This is the only publicly released GT in AIC21 Track 1 Dataset_A."
            )
        }

    def print_report(self) -> None:
        log.info("=" * 55)
        log.info(f"COUNTING EVALUATION — {self.camera_id}")
        log.info("=" * 55)
        log.info(f"Eval window      : {self.eval_frames} frames ({self.eval_duration_sec:.1f}s)")
        log.info(f"GT total         : {self.gt_total}")
        log.info(f"Predicted total  : {self.pred_total}")
        log.info(f"Total MAE        : {self.total_mae:.1f} vehicles")
        log.info(f"Total error %    : {self.total_error_pct:.1f}%")
        log.info(f"Per-movement MAE : {self.per_movement_mae:.2f} vehicles")
        log.info(f"Car MAE          : {self.car_mae:.1f}")
        log.info(f"Truck MAE        : {self.truck_mae:.1f}")
        log.info(f"Movs within 20%  : {self.movements_within_20pct:.1f}%")
        log.info("")
        log.info("GT vs Predicted by movement:")
        all_movements = sorted(set(
            list(self.gt_by_movement.keys()) + list(self.pred_by_movement.keys())
        ))
        for mov in all_movements:
            gt_v = self.gt_by_movement.get(mov, 0)
            pred_v = self.pred_by_movement.get(mov, 0)
            err = pred_v - gt_v
            log.info(f"  Movement {mov:2d}: GT={gt_v:3d}  Pred={pred_v:3d}  Err={err:+d}")
        log.info("=" * 55)


def evaluate_counting(
    counter: VehicleCounter,
    gt_events: List[CountingEvent],
    max_frame_id: int,
    camera_id: str = "cam_5",
    fps: float = 10.0,
) -> EvaluationResult:
    """
    Compare VehicleCounter predictions against ground truth events.

    Args:
        counter:      VehicleCounter after processing the video.
        gt_events:    List of GT CountingEvent from load_counting_gt().
        max_frame_id: Only evaluate up to this frame (inclusive).
        camera_id:    Camera identifier for reporting.
        fps:          Video FPS for duration calculation.

    Returns:
        EvaluationResult with all computed metrics.
    """
    # Filter GT to max_frame_id
    gt_in_window = [e for e in gt_events if e.frame_id <= max_frame_id]

    # GT aggregates
    gt_total = len(gt_in_window)
    gt_by_class: Dict[str, int] = defaultdict(int)
    gt_by_movement: Dict[int, int] = defaultdict(int)
    for e in gt_in_window:
        gt_by_class[e.vehicle_class_name] += 1
        gt_by_movement[e.movement_id] += 1

    # Predicted aggregates (up to max_frame_id)
    pred_result = counter.get_counts_at_frame(max_frame_id)
    pred_total = pred_result.total
    pred_by_class = pred_result.by_class
    pred_by_movement = pred_result.by_movement

    # Total MAE
    total_mae = abs(pred_total - gt_total)
    total_error_pct = (total_mae / gt_total * 100) if gt_total > 0 else 0.0

    # Per-movement MAE
    all_movements = sorted(set(
        list(gt_by_movement.keys()) + list(pred_by_movement.keys())
    ))
    mov_errors = []
    n_within_20pct = 0
    for mov in all_movements:
        gt_v = gt_by_movement.get(mov, 0)
        pred_v = pred_by_movement.get(mov, 0)
        err = abs(pred_v - gt_v)
        mov_errors.append(err)
        if gt_v > 0 and err / gt_v <= 0.20:
            n_within_20pct += 1
        elif gt_v == 0 and pred_v == 0:
            n_within_20pct += 1

    per_movement_mae = float(np.mean(mov_errors)) if mov_errors else 0.0
    movements_within_20pct = (n_within_20pct / len(all_movements) * 100) if all_movements else 0.0

    # Per-class MAE
    car_mae = abs(pred_by_class.get("Car", 0) - gt_by_class.get("Car", 0))
    truck_mae = abs(pred_by_class.get("Truck", 0) - gt_by_class.get("Truck", 0))

    return EvaluationResult(
        camera_id=camera_id,
        eval_frames=max_frame_id,
        eval_duration_sec=max_frame_id / fps,
        gt_total=gt_total,
        gt_by_class=dict(gt_by_class),
        gt_by_movement=dict(gt_by_movement),
        pred_total=pred_total,
        pred_by_class=pred_by_class,
        pred_by_movement=pred_by_movement,
        total_mae=float(total_mae),
        total_error_pct=total_error_pct,
        per_movement_mae=per_movement_mae,
        car_mae=float(car_mae),
        truck_mae=float(truck_mae),
        movements_within_20pct=movements_within_20pct,
    )


def save_counting_results_csv(
    result: EvaluationResult,
    output_path: Path,
) -> None:
    """Save evaluation results to CSV for reporting."""
    import csv
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    all_movements = sorted(set(
        list(result.gt_by_movement.keys()) + list(result.pred_by_movement.keys())
    ))
    for mov in all_movements:
        gt_v = result.gt_by_movement.get(mov, 0)
        pred_v = result.pred_by_movement.get(mov, 0)
        rows.append({
            "camera_id": result.camera_id,
            "movement_id": mov,
            "gt_count": gt_v,
            "predicted_count": pred_v,
            "absolute_error": abs(pred_v - gt_v),
            "error_pct": round(abs(pred_v - gt_v) / gt_v * 100, 1) if gt_v > 0 else "N/A",
        })

    # Add summary row
    rows.append({
        "camera_id": result.camera_id,
        "movement_id": "TOTAL",
        "gt_count": result.gt_total,
        "predicted_count": result.pred_total,
        "absolute_error": result.total_mae,
        "error_pct": round(result.total_error_pct, 1),
    })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    log.info(f"Counting results saved: {output_path}")
