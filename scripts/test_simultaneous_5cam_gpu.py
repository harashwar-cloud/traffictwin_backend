import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.inference.multi_pipeline import MultiCameraPipeline

def main():
    print("=" * 65)
    print("TRAFFICTWIN — 5-CAMERA SIMULTANEOUS GPU INFERENCE ACCEPTANCE TEST")
    print("=" * 65)

    pipeline = MultiCameraPipeline(config_path="configs/config.yaml")

    data_dir = project_root.parent / "AIC21_Track1_Vehicle_Counting" / "Dataset_A"
    camera_jobs = [
        {"camera_id": "cam_5", "job_id": "eval_cam_5", "video_path": str(data_dir / "cam_5.mp4")},
        {"camera_id": "cam_1", "job_id": "eval_cam_1", "video_path": str(data_dir / "cam_1.mp4")},
        {"camera_id": "cam_7", "job_id": "eval_cam_7", "video_path": str(data_dir / "cam_7.mp4")},
        {"camera_id": "cam_9", "job_id": "eval_cam_9", "video_path": str(data_dir / "cam_9.mp4")},
        {"camera_id": "cam_10", "job_id": "eval_cam_10", "video_path": str(data_dir / "cam_10.mp4")},
    ]

    per_cam_frames = {}

    def callback(cid, annotated, fid, ts, payload):
        per_cam_frames[cid] = per_cam_frames.get(cid, 0) + 1
        if fid in (10, 30, 50):
            print(
                f"[SIMULTANEOUS LIVE] Camera {cid:6s} | Frame {fid:2d} | "
                f"Count: {payload['vehicles_counted']:2d} | "
                f"State: {payload['traffic_state']:8s} | "
                f"Speed: {payload['processing_fps']:.1f} FPS"
            )

    t0 = time.perf_counter()
    results = pipeline.run_all(camera_jobs, max_frames=50, frame_callback=callback)
    elapsed = time.perf_counter() - t0

    print("=" * 65)
    print(f"5-Camera Simultaneous Run Completed in {elapsed:.2f}s ({50 * 5 / elapsed:.1f} Total Video FPS)")
    for cid, res in results.items():
        print(f"  {cid:6s} -> Vehicles: {res['total_vehicles']:2d} | State: {res['traffic_state']:8s} | Video: {Path(res['outputs']['annotated_video']).name}")
    print("=" * 65)

if __name__ == "__main__":
    main()
