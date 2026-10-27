# TrafficTwin — Phase Progress Report
# AIC21 Track 1 — Multi-Class Multi-Movement Vehicle Counting

| Phase | Description | Status | Notes |
|-------|-------------|--------|-------|
| PHASE 1 | Environment + Dataset Inspection | ✅ PASS | Hardware: RTX 3050 4GB, 15.68GB RAM. Dataset: 31 videos, 20 cameras, 5.35 GB, 185,446 frames, 4.98 hours |
| PHASE 2 | Annotation + Label Analysis | ⬜ PENDING | |
| PHASE 3 | Preprocessing + Config | ⬜ PENDING | |
| PHASE 4 | Baseline Detector + Tracker | ⬜ PENDING | |
| PHASE 5 | Vehicle Counting | ⬜ PENDING | |
| PHASE 6 | Movement Analysis | ⬜ PENDING | |
| PHASE 7 | Traffic Intelligence Layer | ⬜ PENDING | |
| PHASE 8 | Evaluation | ⬜ PENDING | |
| PHASE 9 | Final Inference Pipeline | ⬜ PENDING | |
| PHASE 10 | API-Ready Outputs | ⬜ PENDING | |
| PHASE 11 | Testing | ⬜ PENDING | |
| PHASE 12 | Documentation | ⬜ PENDING | |

---

## PHASE 1 RESULTS — VERIFIED REAL DATA

### Hardware
| Property | Value |
|----------|-------|
| OS | Windows 10 Build 26200 |
| CPU | Intel Core i7-12th Gen (8P+4E), 12 threads |
| RAM | 15.68 GB total, ~5.4 GB free |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM | 4.0 GB (3.65 GB free) |
| CUDA | 12.1 / Driver 595.71 |
| Python | 3.10.0rc2 |
| PyTorch | 2.5.1+cu121 |
| Ultralytics | 8.4.21 (YOLOv8 + ByteTrack built-in) |

### Dataset
| Property | Value |
|----------|-------|
| Dataset | AIC21 Track 1 Dataset_A |
| Total videos | 31 |
| Unique cameras | 20 |
| Total size | 5.35 GB |
| Total frames | 185,446 |
| Total duration | 4.98 hours |
| FPS values | 8 fps (cam_7), 10 fps (most), 15 fps (cam_4) |
| Resolutions | 1280×720, 1280×960, 1920×1080, 2560×1920 |
| Vehicle classes | 1=Car, 2=Truck (motorbikes excluded from counting) |
| Lighting | 20 normal, 4 dawn, 6 rain, 1 snow |
| Corruption | None detected |
| Duplicates | None detected |

### Annotation Files (Ground Truth)
| Type | Coverage |
|------|---------|
| ROI polygons | All 20 cameras (5–6 vertices each) |
| Movement descriptions | All 20 cameras (3–12 movements per camera) |
| Counting GT | cam_5 only — first 600 frames (1 minute at 10fps): 96 car events, 1 truck event across movements 1-12 |

### Camera Types
| Type | Cameras |
|------|---------|
| highway ramp | cam_1, cam_2, cam_3, cam_7 |
| four-leg signalized intersection (fisheye) | cam_4, cam_5, cam_6 |
| three-leg allway-stop (fisheye) | cam_8 |
| four-leg allway-stop | cam_9 |
| single-approach intersection | cam_10, cam_11, cam_12, cam_13 |
| various intersections | cam_14–cam_20 |

### Dataset Split (Video-level)
| Split | Cameras | Videos |
|-------|---------|--------|
| TRAIN | 14 cameras | 23 videos |
| VAL | 3 cameras | 3 videos |
| TEST | 3 cameras (incl. cam_5 with GT) | 5 videos |

---

## PHASE 1 DECISIONS

### Selected Detector: YOLOv8n
- Fits 4GB VRAM (baseline inference ~300–500ms/frame batch)
- Already has ultralytics 8.4.21 installed
- Existing yolov8n.pt found on disk at `D:\Dinesh project\New folder (2)\yolov8n.pt`
- COCO-pretrained: car(2), bus(5), truck(7) → mapped to AIC21 classes Car/Truck
- Can upgrade to YOLOv8s if performance is insufficient

### Selected Tracker: ByteTrack (ultralytics built-in)
- Built into ultralytics 8.4.21 — zero additional install
- lap (0.5.13) already installed for assignment
- Handles occlusion better than SORT via two-stage matching
- No separate dependency needed

### Selected Intelligence Layer: Traffic-State Estimation
- Uses rolling per-movement flow rates (vehicles/minute per movement)
- Classifies intersection traffic state: LOW / MEDIUM / HIGH / CRITICAL
- Also detects direction imbalance (congestion in one leg vs. another)
- Outputs actionable JSON per time window — usable by web app
- Thresholds derived from actual cam_5 GT data (96 vehicles in 1 minute = ~8 vehicles/min total)

### Evaluation Strategy
- Counting: MAE, RMSE vs. cam_5 GT (600 frames, movements 1–12)
- Detection: qualitative visual inspection (no per-frame bbox GT available)
- Traffic State: output consistency check (deterministic from flow rates)
