import requests
import json
import time
import cv2

from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
video_path = str(repo_root / "AIC21_Track1_Vehicle_Counting" / "Dataset_A" / "cam_5.mp4")
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# Polygon with a self-intersection (bowtie)
roads_json = json.dumps([
    {
        "road_id": "road_1",
        "name": "North Road",
        "polygon": [[0, 0], [100, 100], [0, 100], [100, 0]]
    }
])

data = {
    "max_frames": 50,
    "roads_json": roads_json
}
files = {
    "file": open(video_path, "rb")
}
print("Uploading video...")
upload_res = requests.post("http://127.0.0.1:8000/api/video/upload", data={"camera_id": "cam_5"}, files=files).json()
job_id = upload_res["job_id"]
print("Upload returned:", upload_res)

print("Starting inference...")
data["job_id"] = job_id
inf_res = requests.post("http://127.0.0.1:8000/api/inference/start", data=data).json()
print("Inference started:", inf_res)

while True:
    status_res = requests.get(f"http://127.0.0.1:8000/api/inference/{job_id}/status").json()
    print("Status:", status_res["status"])
    if status_res["status"] in ["READY", "ERROR"]:
        print("Final result:", status_res)
        break
    time.sleep(2)
