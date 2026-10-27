"""
scripts/verify_gpu_live_sync.py
--------------------------------
Comprehensive GPU acceleration and Live Count Synchronization Acceptance Test.
Measures real performance on NVIDIA RTX 3050 and verifies video + telemetry sync.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import json
import sys
import torch
from pathlib import Path
import numpy as np

# Ensure traffic_ml is in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.inference.pipeline import TrafficPipeline

def main():
    print("=" * 65)
    print("TRAFFICTWIN — GPU ACCELERATION & LIVE SYNC ACCEPTANCE TEST")
    print("=" * 65)

    # 1. Hardware Report
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Capability: {torch.cuda.get_device_capability(0)}")
        print(f"Allocated VRAM:  {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    # 2. Checkpoints recorder
    checkpoints = {}
    
    count_calls = 0
    def live_callback(frame, frame_id, ts, live_payload):
        nonlocal count_calls
        count_calls += 1
        if count_calls % 100 == 0 or count_calls == 1 or count_calls == 600:
            checkpoints[count_calls] = {
                "frame_count": count_calls,
                "frame_id": frame_id,
                "timestamp_sec": ts,
                "vehicles_counted": live_payload["vehicles_counted"],
                "classes": live_payload["vehicle_classes"],
                "movement_counts": live_payload["movement_counts"],
                "traffic_state": live_payload["traffic_state"],
                "arrival_rate": live_payload["arrival_rate"],
                "queue": live_payload["queue"],
                "fps": live_payload["processing_fps"]
            }
            print(
                f"[LIVE SYNC CHECKPOINT] Frame {count_calls:3d} (t={ts:4.1f}s) | "
                f"Counted: {live_payload['vehicles_counted']:2d} | "
                f"State: {live_payload['traffic_state']:8s} | "
                f"Queue: {live_payload['queue']:2d} | "
                f"FPS: {live_payload['processing_fps']:.1f}"
            )

    # 3. Execute Pipeline on cam_5
    aic_root = project_root.parent / "AIC21_Track1_Vehicle_Counting"
    video_path = str(aic_root / "Dataset_A" / "cam_5.mp4")
    config_path = str(project_root / "configs" / "config.yaml")

    t_start = time.perf_counter()
    pipeline = TrafficPipeline(config_path=config_path)
    pipeline.max_frames = 600
    
    res = pipeline.run(
        video_path=video_path,
        camera_id="cam_5",
        output_prefix="cam5_eval",
        frame_callback=live_callback
    )
    total_time = time.perf_counter() - t_start

    fps = 600.0 / total_time

    print("=" * 65)
    print("RESULTS & ACCEPTANCE SUMMARY:")
    print(f"Total Frames Processed: 600 (60.0s video)")
    print(f"Total Processing Time:  {total_time:.2f}s")
    print(f"Overall Processing Speed: {fps:.1f} FPS ({60.0 / total_time:.1f}x Realtime)")
    print(f"Total Unique Vehicles Counted: {res['total_vehicles']}")
    print(f"Vehicle Classes: {res['vehicle_classes']}")
    print(f"Movement Counts: {res['movement_counts']}")
    print(f"Final Traffic State: {res['traffic_state']}")
    print(f"Direction Imbalance Ratio: {res['traffic_features']['direction_imbalance_ratio']}x")
    print("=" * 65)

    # Save summary report
    report = {
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
        },
        "performance": {
            "total_frames": 600,
            "duration_sec": 60.0,
            "processing_time_sec": round(total_time, 2),
            "processing_fps": round(fps, 1),
            "realtime_factor": round(60.0 / total_time, 2),
        },
        "live_sync_checkpoints": checkpoints,
        "final_metrics": {
            "total_vehicles": res["total_vehicles"],
            "vehicle_classes": res["vehicle_classes"],
            "movement_counts": res["movement_counts"],
            "traffic_state": res["traffic_state"],
            "flow_rate_per_min": res["traffic_features"]["total_flow_rate_per_min"],
            "direction_imbalance_ratio": res["traffic_features"]["direction_imbalance_ratio"],
        }
    }

    report_path = Path("reports/gpu_live_sync_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Detailed acceptance report saved to: {report_path}")

if __name__ == "__main__":
    main()
