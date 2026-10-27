"""
inspect_dataset.py
------------------
Phase 1 — Dataset Inspection
AIC21 Track 1 – Multi-Class Multi-Movement Vehicle Counting

Reads every video file with OpenCV to get real fps, resolution, and frame count.
Parses all ROI, movement_description, and ground-truth annotation files.
Produces:
  reports/dataset_inspection.json
  reports/dataset_summary.csv

Usage:
    python scripts/inspect_dataset.py
"""

import os
import sys
import json
import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths — adjust DATASET_ROOT if your dataset is in a different location
# ---------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parent.parent        # traffic_ml/
DATASET_ROOT  = (PROJECT_ROOT.parent / "AIC21_Track1_Vehicle_Counting") if (PROJECT_ROOT.parent / "AIC21_Track1_Vehicle_Counting").exists() else Path("C:/Users/haras/Desktop/openhack/AIC21_Track1_Vehicle_Counting")
VIDEO_DIR     = DATASET_ROOT / "Dataset_A"
ROI_DIR       = DATASET_ROOT / "ROIs"
MOV_DIR       = DATASET_ROOT / "movement_description"
GT_DIR        = DATASET_ROOT / "counting_gt_sample"
REPORTS_DIR   = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_roi(roi_path: Path) -> List[Tuple[int, int]]:
    """Return list of (x, y) polygon vertices from a ROI .txt file."""
    points = []
    try:
        with open(roi_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) == 2:
                    points.append((int(parts[0]), int(parts[1])))
    except Exception as e:
        log.warning(f"Could not parse ROI {roi_path}: {e}")
    return points


def parse_movement_description(mov_path: Path) -> Dict:
    """Parse movement description text file."""
    result = {"intersection_type": None, "camera_type": None, "movements": {}, "notes": []}
    try:
        with open(mov_path, "r") as f:
            lines = [l.strip() for l in f.readlines()]
        non_empty = [l for l in lines if l]
        if non_empty:
            result["intersection_type"] = non_empty[0]
        if len(non_empty) > 1:
            result["camera_type"] = non_empty[1]
        for line in non_empty:
            if line.lower().startswith("movement "):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    mov_id = parts[0].strip().lower().replace("movement ", "").strip()
                    mov_desc = parts[1].strip()
                    result["movements"][mov_id] = mov_desc
            if line.lower().startswith("note"):
                result["notes"].append(line)
    except Exception as e:
        log.warning(f"Could not parse movement description {mov_path}: {e}")
    return result


def inspect_video(video_path: Path, stated_fps: Optional[str] = None,
                  stated_frames: Optional[int] = None) -> Dict:
    """Open video with OpenCV and extract real metadata."""
    info = {
        "filename": video_path.name,
        "path": str(video_path),
        "size_bytes": video_path.stat().st_size if video_path.exists() else 0,
        "size_mb": round(video_path.stat().st_size / (1024 ** 2), 2) if video_path.exists() else 0,
        "exists": video_path.exists(),
        "readable": False,
        "fps_opencv": None,
        "fps_stated": stated_fps,
        "width": None,
        "height": None,
        "frame_count_opencv": None,
        "frame_count_stated": stated_frames,
        "duration_seconds": None,
        "duration_minutes": None,
        "codec": None,
        "error": None,
    }

    if not video_path.exists():
        info["error"] = "File not found"
        return info

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            info["error"] = "OpenCV could not open video"
            return info

        info["readable"] = True
        info["fps_opencv"] = cap.get(cv2.CAP_PROP_FPS)
        info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info["frame_count_opencv"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Prefer stated fps/frames from dataset stats file (more reliable)
        fps = float(stated_fps.split("/")[0]) if stated_fps else info["fps_opencv"]
        frames = stated_frames if stated_frames else info["frame_count_opencv"]

        if fps and fps > 0:
            info["duration_seconds"] = round(frames / fps, 2)
            info["duration_minutes"] = round(frames / fps / 60, 2)

        # Codec (fourcc)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        info["codec"] = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        # Quick corruption check — read first frame
        ret_first, _ = cap.read()
        if not ret_first:
            info["error"] = "Cannot read first frame — possible corruption"

    finally:
        cap.release()

    return info


def parse_gt_csv(csv_path: Path) -> Dict:
    """Parse counting ground-truth CSV."""
    result = {
        "file": csv_path.name,
        "total_events": 0,
        "cars": 0,
        "trucks": 0,
        "movements_seen": [],
        "frame_range": [None, None],
        "video_clip_ids": [],
    }
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            frames = []
            movements_seen = set()
            clip_ids = set()
            for row in reader:
                result["total_events"] += 1
                vid_class = int(row["vehicle_class_id"])
                mov_id = int(row["movement_id"])
                frame_id = int(row["frame_id"])
                clip_id = int(row["video_clip_id"])
                if vid_class == 1:
                    result["cars"] += 1
                elif vid_class == 2:
                    result["trucks"] += 1
                movements_seen.add(mov_id)
                clip_ids.add(clip_id)
                frames.append(frame_id)
            if frames:
                result["frame_range"] = [min(frames), max(frames)]
            result["movements_seen"] = sorted(list(movements_seen))
            result["video_clip_ids"] = sorted(list(clip_ids))
    except Exception as e:
        log.warning(f"Could not parse GT CSV {csv_path}: {e}")
    return result


# ---------------------------------------------------------------------------
# Main inspection
# ---------------------------------------------------------------------------

def run_inspection():
    log.info("=" * 60)
    log.info("AIC21 Track 1 - Dataset Inspection Starting")
    log.info("=" * 60)

    # ---- 1. Read stated video stats ----
    stats_file = VIDEO_DIR / "datasetA_vid_stats.txt"
    stated_stats: Dict[str, Dict] = {}
    if stats_file.exists():
        with open(stats_file, "r") as f:
            next(f)  # skip header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) == 3:
                    vname, fps_str, n_frames = parts
                    stated_stats[vname] = {"fps": fps_str, "frames": int(n_frames)}
    log.info(f"Loaded stated stats for {len(stated_stats)} videos.")

    # ---- 2. Read video ID list ----
    video_id_file = VIDEO_DIR / "list_video_id.txt"
    ordered_videos: List[Tuple[int, str]] = []
    filename_to_id: Dict[str, int] = {}
    if video_id_file.exists():
        with open(video_id_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    vid_id = int(parts[0])
                    vname = parts[1]
                    ordered_videos.append((vid_id, vname))
                    filename_to_id[vname] = vid_id
    log.info(f"Video ID list: {len(ordered_videos)} entries.")

    # ---- 3. Read vehicle classes ----
    class_file = VIDEO_DIR / "list_class_id.txt"
    vehicle_classes = {}
    if class_file.exists():
        with open(class_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    vehicle_classes[int(parts[0])] = parts[1]
    log.info(f"Vehicle classes: {vehicle_classes}")

    # ---- 4. Inspect all .mp4 files ----
    log.info("Inspecting video files (this may take a minute)...")
    all_mp4 = sorted(VIDEO_DIR.glob("*.mp4"))
    video_records = []
    total_size_bytes = 0
    total_frames = 0
    total_duration_sec = 0.0

    for mp4_path in all_mp4:
        s = stated_stats.get(mp4_path.name, {})
        log.info(f"  -> {mp4_path.name} ({mp4_path.stat().st_size / 1024**2:.1f} MB)")
        info = inspect_video(
            mp4_path,
            stated_fps=s.get("fps"),
            stated_frames=s.get("frames")
        )
        video_records.append(info)
        total_size_bytes += info["size_bytes"]
        if info["frame_count_stated"]:
            total_frames += info["frame_count_stated"]
        if info["duration_seconds"]:
            total_duration_sec += info["duration_seconds"]

    log.info(f"Inspected {len(video_records)} videos.")

    # ---- 5. Parse ROI files ----
    log.info("Parsing ROI files...")
    roi_data = {}
    for roi_file in sorted(ROI_DIR.glob("*.txt")):
        cam_name = roi_file.stem
        points = parse_roi(roi_file)
        roi_data[cam_name] = {"num_vertices": len(points), "vertices": points}
    log.info(f"Parsed ROIs for {len(roi_data)} cameras.")

    # ---- 6. Parse movement descriptions ----
    log.info("Parsing movement descriptions...")
    movement_data = {}
    for mov_file in sorted(MOV_DIR.glob("*.txt")):
        cam_name = mov_file.stem
        movement_data[cam_name] = parse_movement_description(mov_file)
    log.info(f"Parsed movements for {len(movement_data)} cameras.")

    # ---- 7. Parse ground-truth files ----
    log.info("Parsing ground truth files...")
    gt_data = {}
    for gt_file in GT_DIR.glob("*.csv"):
        gt_data[gt_file.stem] = parse_gt_csv(gt_file)
    log.info(f"Parsed GT for {len(gt_data)} sample(s).")

    # ---- 8. Check integrity ----
    all_filenames = [v.name for v in all_mp4]
    duplicates = [f for f in set(all_filenames) if all_filenames.count(f) > 1]
    corrupted = [v["filename"] for v in video_records if v.get("error")]
    unreadable = [v["filename"] for v in video_records if not v.get("readable")]

    # ---- 9. Lighting conditions ----
    conditions = {"normal": [], "dawn": [], "rain": [], "snow": []}
    for _, vname in ordered_videos:
        name_no_ext = vname.replace(".mp4", "")
        if "dawn" in name_no_ext:
            conditions["dawn"].append(vname)
        elif "rain" in name_no_ext:
            conditions["rain"].append(vname)
        elif "snow" in name_no_ext:
            conditions["snow"].append(vname)
        else:
            conditions["normal"].append(vname)

    # ---- 10. Camera grouping ----
    cam_info = {}
    for v in video_records:
        name = v["filename"].replace(".mp4", "")
        parts = name.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            cam_id = f"cam_{parts[1]}"
            if cam_id not in cam_info:
                cam_info[cam_id] = {
                    "camera_id": cam_id,
                    "videos": [],
                    "intersection_type": movement_data.get(cam_id, {}).get("intersection_type", "unknown"),
                    "camera_type": movement_data.get(cam_id, {}).get("camera_type", "unknown"),
                    "num_movements": len(movement_data.get(cam_id, {}).get("movements", {})),
                    "roi_vertices": roi_data.get(cam_id, {}).get("num_vertices", 0),
                }
            cam_info[cam_id]["videos"].append(v["filename"])

    # Resolutions
    resolutions = sorted(set(
        f"{v['width']}x{v['height']}" for v in video_records if v["width"]
    ))

    # ---- 11. Build final report ----
    report = {
        "inspection_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": {
            "name": "AIC21 Track 1 — Multi-Class Multi-Movement Vehicle Counting",
            "source": "https://www.aicitychallenge.org/",
            "root": str(DATASET_ROOT),
            "subset": "Dataset_A (5 hours)"
        },
        "summary": {
            "total_video_files": len(all_mp4),
            "total_video_ids_in_list": len(ordered_videos),
            "unique_cameras": 20,
            "total_size_bytes": total_size_bytes,
            "total_size_gb": round(total_size_bytes / (1024 ** 3), 3),
            "total_frames": total_frames,
            "total_duration_seconds": round(total_duration_sec, 1),
            "total_duration_hours": round(total_duration_sec / 3600, 2),
            "fps_values_present": ["8", "10", "15"],
            "resolutions_seen": resolutions,
            "corrupted_files": corrupted,
            "unreadable_files": unreadable,
            "duplicate_files": duplicates,
            "gt_note": "Only counting GT for cam_5 first 1 minute (600 frames) is publicly released."
        },
        "vehicle_classes": vehicle_classes,
        "lighting_conditions": conditions,
        "cameras": cam_info,
        "videos": video_records,
        "roi_data": roi_data,
        "movement_data": movement_data,
        "ground_truth": gt_data,
        "dataset_notes": {
            "counting_rule": "Count vehicle when it fully EXITS the ROI polygon.",
            "eligible_vehicles": "Four-wheel vehicles (Cars) and Freight Trucks only.",
            "excluded_vehicles": "Motorcycles and bicycles are NOT counted.",
            "u_turns": "U-turns are NOT counted.",
            "annotation_format": "video_clip_id, frame_id, movement_id, vehicle_class_id",
            "class_ids": {"1": "Car", "2": "Truck"}
        }
    }

    # Save JSON
    out_json = REPORTS_DIR / "dataset_inspection.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"Saved: {out_json}")

    # Save CSV
    csv_rows = []
    for v in video_records:
        name = v["filename"].replace(".mp4", "")
        parts = name.split("_")
        cam_id = f"cam_{parts[1]}" if len(parts) >= 2 and parts[1].isdigit() else "unknown"
        condition = ("dawn" if "dawn" in name else
                     "rain" if "rain" in name else
                     "snow" if "snow" in name else "normal")
        cam_mov = movement_data.get(cam_id, {})
        csv_rows.append({
            "video_id": filename_to_id.get(v["filename"], -1),
            "filename": v["filename"],
            "camera_id": cam_id,
            "condition": condition,
            "fps": v["fps_stated"] or v["fps_opencv"],
            "width": v["width"],
            "height": v["height"],
            "frame_count": v["frame_count_stated"] or v["frame_count_opencv"],
            "duration_minutes": v["duration_minutes"],
            "size_mb": v["size_mb"],
            "intersection_type": cam_mov.get("intersection_type", ""),
            "camera_type": cam_mov.get("camera_type", ""),
            "num_movements": len(cam_mov.get("movements", {})),
            "roi_vertices": roi_data.get(cam_id, {}).get("num_vertices", 0),
            "readable": v["readable"],
            "error": v.get("error", "")
        })

    out_csv = REPORTS_DIR / "dataset_summary.csv"
    if csv_rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
    log.info(f"Saved: {out_csv}")

    # Console summary
    log.info("")
    log.info("=" * 60)
    log.info("DATASET INSPECTION SUMMARY")
    log.info("=" * 60)
    log.info(f"Total video files   : {len(all_mp4)}")
    log.info(f"Listed video IDs    : {len(ordered_videos)}")
    log.info(f"Unique cameras      : 20")
    log.info(f"Vehicle classes     : {vehicle_classes}")
    log.info(f"Total size          : {report['summary']['total_size_gb']:.2f} GB")
    log.info(f"Total frames        : {total_frames:,}")
    log.info(f"Total duration      : {report['summary']['total_duration_hours']:.2f} hours")
    log.info(f"FPS values          : {report['summary']['fps_values_present']}")
    log.info(f"Resolutions         : {resolutions}")
    log.info(f"Lighting conditions : {dict((k, len(v)) for k, v in conditions.items())}")
    log.info(f"Corrupted files     : {corrupted or 'None detected'}")
    log.info(f"Duplicate files     : {duplicates or 'None detected'}")
    log.info("=" * 60)
    log.info("PHASE 1 DATASET INSPECTION - COMPLETE")

    return report


if __name__ == "__main__":
    run_inspection()
