"""
scripts/benchmark_bottleneck.py
--------------------------------
Comprehensive diagnostic tool to measure and profile the exact time spent in:
1. Video decoding (cv2.VideoCapture.read)
2. YOLOv8 detection + ByteTrack (GPU inference, CUDA sync)
3. Road ROI & Movement state analytics
4. MOI trajectory assignment & counting
5. Frame annotation & drawing (OpenCV CPU drawing)
6. VideoWriter disk writing
7. Live streaming JPEG compression (cv2.imencode)
8. GPU Memory & System CPU profiling
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import psutil
from pathlib import Path
import numpy as np
import torch
import cv2

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.video_loader import get_video_info
from src.data.annotation_loader import load_all_rois, load_all_movements
from src.data.road_config import load_road_config
from src.tracking.tracker import ByteTracker, TrackStore
from src.tracking.road_assigner import RoadAssigner
from src.tracking.movement_state import VehicleMovementDetector
from src.movement.movement_assigner import MovementAssigner
from src.counting.roi_counter import VehicleCounter
from src.analytics.traffic_state import TrafficStateEstimator
from src.inference.visualizer import (
    create_video_writer,
    draw_count_panel,
    draw_road_rois,
    draw_tracks,
    draw_traffic_state_banner,
)

def benchmark(num_frames: int = 100):
    video_path = project_root.parent / "AIC21_Track1_Vehicle_Counting" / "Dataset_A" / "cam_5.mp4"
    camera_id = "cam_5"

    print("=" * 70)
    print("TRAFFICTWIN — COMPREHENSIVE BOTTLENECK DIAGNOSTIC & BENCHMARK")
    print("=" * 70)
    print(f"PyTorch Version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available:  {cuda_avail}")
    if cuda_avail:
        print(f"GPU Device:      {torch.cuda.get_device_name(0)}")
        print(f"Device Count:    {torch.cuda.device_count()}")
        print(f"VRAM Total:      {torch.cuda.get_device_properties(0).total_memory / (1024**2):.1f} MB")
        print(f"Allocated VRAM:  {torch.cuda.memory_allocated(0) / (1024**2):.1f} MB")
    print(f"CPU Physical:    {psutil.cpu_count(logical=False)} cores | Logical: {psutil.cpu_count(logical=True)} cores")
    print(f"System RAM:      {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"Video File:      {video_path}")
    print(f"Benchmark Frames: {num_frames}")
    print("-" * 70)

    # 1. Video capture init
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Info: {w}x{h} @ {fps} FPS, Total Frames: {total_vid_frames}")

    # 2. Init components
    all_rois = load_all_rois(project_root.parent / "AIC21_Track1_Vehicle_Counting" / "ROIs")
    all_movements = load_all_movements(project_root.parent / "AIC21_Track1_Vehicle_Counting" / "movement_description")
    roi = all_rois[camera_id]
    cam_ann = all_movements[camera_id]
    road_rois = load_road_config(camera_id, project_root / "configs" / "roads", roi.vertices)
    road_rois = [r.scale_to_resolution(1920, 1080, w, h) for r in road_rois]

    tracker = ByteTracker(
        model_path=str(project_root / "yolov8n.pt"),
        confidence=0.35,
        iou_threshold=0.45,
        input_size=640,
        device="cuda" if cuda_avail else "cpu",
        use_half=cuda_avail,
    )
    track_store = TrackStore()
    road_assigner = RoadAssigner(switch_confirm_frames=10, entry_confirm_frames=5, lost_grace_frames=15, boundary_margin=15.0)
    movement_detector = VehicleMovementDetector()
    movement_assigner = MovementAssigner(roi=roi, camera_annotation=cam_ann)
    counter = VehicleCounter(time_window_seconds=60.0)
    state_estimator = TrafficStateEstimator(time_window_seconds=60.0)

    out_video_path = str(project_root / "outputs" / "videos" / "benchmark_test.mp4")
    writer = create_video_writer(out_video_path, w, h, fps)

    # Profiling buckets
    t_decode_list = []
    t_track_list = []
    t_analytics_list = []
    t_annotate_list = []
    t_write_list = []
    t_jpeg_list = []
    t_total_frame_list = []

    print("\nWarming up tracker & GPU...")
    dummy = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(5):
        tracker.track_frame(dummy, 0, 0.0)
    if cuda_avail:
        torch.cuda.synchronize()
    print("Warmup complete. Starting benchmark...\n")

    # Start benchmark
    bench_start = time.perf_counter()
    frame_idx = 0

    while frame_idx < num_frames:
        t_f0 = time.perf_counter()

        # A. Decode
        t_d0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        t_d1 = time.perf_counter()
        t_decode_list.append((t_d1 - t_d0) * 1000.0)

        ts = frame_idx / fps

        # B. YOLO + ByteTrack
        t_t0 = time.perf_counter()
        if cuda_avail:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            tracks = tracker.track_frame(frame, frame_idx, ts)
            end_event.record()
            torch.cuda.synchronize()
            gpu_time_ms = start_event.elapsed_time(end_event)
            t_t1 = time.perf_counter()
            track_time_ms = (t_t1 - t_t0) * 1000.0
        else:
            tracks = tracker.track_frame(frame, frame_idx, ts)
            t_t1 = time.perf_counter()
            track_time_ms = (t_t1 - t_t0) * 1000.0
            gpu_time_ms = 0.0
        t_track_list.append(track_time_ms)
        track_store.update(tracks)

        # C. Road Assignment & Analytics
        t_a0 = time.perf_counter()
        current_road_assignment = {}
        current_movement_state = {}
        active_ids = []
        for tr in tracks:
            active_ids.append(tr.track_id)
            ref_x = (tr.x1 + tr.x2) / 2.0
            ref_y = tr.y2
            mstate = movement_detector.update(tr.track_id, ref_x, ref_y, ts)
            current_movement_state[tr.track_id] = mstate
            rid = road_assigner.update(tr.track_id, tr.x1, tr.y1, tr.x2, tr.y2, road_rois)
            current_road_assignment[tr.track_id] = rid or "OUTSIDE"

        entry_ev, exit_ev = movement_assigner.process_frame(frame_idx, ts, track_store, tracks)
        counter.register_entries(entry_ev)
        counter.register_exits(exit_ev)
        for e in exit_ev:
            state_estimator.add_exit_event(ts, e.movement_id, e.aic21_class_name)
        road_assigner.cleanup_lost_tracks(active_ids)
        movement_detector.cleanup_lost_tracks(active_ids)

        if frame_idx % 30 == 0:
            cstate = state_estimator.compute_state(ts)
        t_a1 = time.perf_counter()
        t_analytics_list.append((t_a1 - t_a0) * 1000.0)

        # D. Frame Annotation
        t_draw0 = time.perf_counter()
        ann = frame.copy()
        ann = draw_road_rois(ann, road_rois)
        ann = draw_tracks(ann, tracks, road_rois=road_rois, road_assignment=current_road_assignment, movement_states=current_movement_state)
        csnap = counter.get_result()
        ann = draw_count_panel(ann, csnap.total, csnap.by_class.get("Car", 0), csnap.by_class.get("Truck", 0), frame_idx, ts)
        t_draw1 = time.perf_counter()
        t_annotate_list.append((t_draw1 - t_draw0) * 1000.0)

        # E. Video Writer Write
        t_w0 = time.perf_counter()
        writer.write(ann)
        t_w1 = time.perf_counter()
        t_write_list.append((t_w1 - t_w0) * 1000.0)

        # F. JPEG Encode (for streaming)
        t_j0 = time.perf_counter()
        _, _ = cv2.imencode(".jpg", ann, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        t_j1 = time.perf_counter()
        t_jpeg_list.append((t_j1 - t_j0) * 1000.0)

        t_f1 = time.perf_counter()
        t_total_frame_list.append((t_f1 - t_f0) * 1000.0)

        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f"Processed {frame_idx}/{num_frames} frames... Current Frame Latency: {(t_f1 - t_f0)*1000.0:.1f}ms (FPS: {1.0/(t_f1 - t_f0):.1f})")

    cap.release()
    writer.release()
    total_time = time.perf_counter() - bench_start
    overall_fps = num_frames / total_time

    # Calculate statistics
    avg_decode = np.mean(t_decode_list)
    avg_track = np.mean(t_track_list)
    avg_analytics = np.mean(t_analytics_list)
    avg_annotate = np.mean(t_annotate_list)
    avg_write = np.mean(t_write_list)
    avg_jpeg = np.mean(t_jpeg_list)
    avg_total = np.mean(t_total_frame_list)

    print("\n" + "=" * 70)
    print("DETAILED BOTTLENECK ANALYSIS RESULTS")
    print("=" * 70)
    print(f"Total Frames Processed:    {num_frames}")
    print(f"Total Wall Time:          {total_time:.2f} s")
    print(f"End-to-End Pipeline FPS:   {overall_fps:.2f} FPS")
    print(f"Average Frame Latency:    {avg_total:.2f} ms")
    print("-" * 70)
    print(f"{'Pipeline Stage':<35} | {'Avg Time (ms)':<14} | {'% of Frame Time':<15}")
    print("-" * 70)
    print(f"{'1. Video Decoding (cv2.read)':<35} | {avg_decode:<14.2f} | {avg_decode/avg_total*100:<15.1f}%")
    print(f"{'2. YOLOv8n + ByteTrack (GPU Track)':<35} | {avg_track:<14.2f} | {avg_track/avg_total*100:<15.1f}%")
    print(f"{'3. Road ROI & State Analytics':<35} | {avg_analytics:<14.2f} | {avg_analytics/avg_total*100:<15.1f}%")
    print(f"{'4. Visual Annotation (OpenCV)':<35} | {avg_annotate:<14.2f} | {avg_annotate/avg_total*100:<15.1f}%")
    print(f"{'5. VideoWriter Write (Disk mp4v)':<35} | {avg_write:<14.2f} | {avg_write/avg_total*100:<15.1f}%")
    print(f"{'6. Live MJPEG JPEG Compression':<35} | {avg_jpeg:<14.2f} | {avg_jpeg/avg_total*100:<15.1f}%")
    print("-" * 70)
    if cuda_avail:
        print(f"Peak VRAM Allocated:       {torch.cuda.max_memory_allocated(0)/(1024**2):.1f} MB")
        print(f"Current VRAM Allocated:    {torch.cuda.memory_allocated(0)/(1024**2):.1f} MB")
        print(f"Reserved VRAM:             {torch.cuda.memory_reserved(0)/(1024**2):.1f} MB")
    print("=" * 70)

if __name__ == "__main__":
    benchmark(num_frames=80)
