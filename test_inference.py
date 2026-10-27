import cv2
import yaml
from pathlib import Path
from src.inference.pipeline import TrafficPipeline
from src.data.video_loader import get_video_info

print("Initializing Pipeline...")
pipeline = TrafficPipeline(config_path="configs/config.yaml")

print("Validating Annotations for cam_5...")
print("ROI loaded:", "cam_5" in pipeline.all_rois)
print("Movement loaded:", "cam_5" in pipeline.all_movements)

repo_root = Path(__file__).resolve().parent.parent
default_video = repo_root / "AIC21_Track1_Vehicle_Counting" / "Dataset_A" / "cam_5.mp4"
video_path = str(default_video) if default_video.exists() else "C:/Users/haras/Desktop/openhack/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_5.mp4"

if not Path(video_path).exists():
    print(f"Video not found: {video_path}")
else:
    info = get_video_info(Path(video_path))
    print(f"Video Info: {info.width}x{info.height}, {info.fps} FPS")
    
    print("Running Inference (10 frames)...")
    pipeline.max_frames = 10
    pipeline.save_video = False
    
    res = pipeline.run(video_path, camera_id="cam_5")
    
    print("Test passed! Vehicles counted:", res.get("total_vehicles", 0))
    print("Device used:", res.get("processing", {}).get("device"))
    print("Precision:", res.get("processing", {}).get("precision"))
