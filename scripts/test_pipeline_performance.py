"""
scripts/test_pipeline_performance.py
------------------------------------
Executes the full TrafficPipeline with all optimizations on cam_5.mp4
Measures actual steady-state processing FPS, inference time, VRAM, and vehicle counting.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
from pathlib import Path
import torch
import psutil

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.inference.pipeline import TrafficPipeline

def test_pipeline(num_frames: int = 100):
    print("=" * 70)
    print("TRAFFICTWIN — FULL OPTIMIZED PIPELINE VALIDATION BENCHMARK")
    print("=" * 70)

    video_path = project_root.parent / "AIC21_Track1_Vehicle_Counting" / "Dataset_A" / "cam_5.mp4"
    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        return

    print(f"CUDA Available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name:        {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM:      {torch.cuda.get_device_properties(0).total_memory / (1024**2):.1f} MB")
        print(f"Initial VRAM:    {torch.cuda.memory_allocated() / (1024**2):.1f} MB")

    print(f"Host CPU:        {psutil.cpu_count(logical=False)} physical / {psutil.cpu_count(logical=True)} logical cores")
    print(f"Host RAM:        {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"Test Video:      {video_path.name}")
    print(f"Benchmark:       {num_frames} frames")
    print("-" * 70)

    live_fps_samples = []
    live_latencies = []
    last_perf = {}

    def live_cb(frame, frame_id, ts, payload):
        fps = payload.get("processing_fps", 0.0)
        perf = payload.get("perf", {})
        if fps > 0:
            live_fps_samples.append(fps)
        if perf.get("total_latency_ms"):
            live_latencies.append(perf["total_latency_ms"])
        if frame_id % 25 == 0:
            print(f"[LIVE CB] Frame {frame_id:3d} | FPS: {fps:4.1f} | Latency: {perf.get('total_latency_ms', 0):4.1f}ms | GPU Inf: {perf.get('inference_ms', 0):4.1f}ms | Track: {perf.get('tracking_ms', 0):4.1f}ms | Render: {perf.get('render_ms', 0):4.1f}ms | VRAM: {perf.get('vram_allocated_mb', 0):.1f}MB")

    pipeline = TrafficPipeline(config_path=str(project_root / "configs" / "config.yaml"))
    pipeline.max_frames = num_frames
    pipeline.save_video = True

    t0 = time.perf_counter()
    result = pipeline.run(
        video_path=str(video_path),
        camera_id="cam_5",
        output_prefix="perf_test",
        frame_callback=live_cb,
    )
    total_elapsed = time.perf_counter() - t0

    print("\n" + "=" * 70)
    print("BENCHMARK EXECUTION RESULTS")
    print("=" * 70)
    print(f"Frames Processed:        {result['processing']['frames_processed']}")
    print(f"Total Wall-Clock Time:   {total_elapsed:.2f} s")
    print(f"Overall Effective FPS:   {num_frames / total_elapsed:.2f} FPS")
    if live_fps_samples:
        print(f"Average Pipeline FPS:    {sum(live_fps_samples)/len(live_fps_samples):.2f} FPS")
        print(f"Peak Pipeline FPS:       {max(live_fps_samples):.2f} FPS")
    if live_latencies:
        print(f"Average Frame Latency:   {sum(live_latencies)/len(live_latencies):.2f} ms")
        print(f"Min Frame Latency:       {min(live_latencies):.2f} ms")

    perf = result.get("performance_metrics", {})
    if perf:
        decode_ms = perf.get('decode_ms', 0)
        pre_ms = perf.get('preprocess_ms', 0)
        inf_ms = perf.get('inference_ms', 0)
        post_ms = perf.get('postprocess_ms', 0)
        track_ms = perf.get('tracking_ms', 0)
        ana_ms = perf.get('analytics_ms', 0)
        ren_ms = perf.get('render_ms', 0)
        wrt_ms = perf.get('write_ms', 0)
        tot_ms = perf.get('total_latency_ms', 0)
        stage_sum = decode_ms + pre_ms + inf_ms + post_ms + track_ms + ana_ms + ren_ms + wrt_ms

        print("\n--- Sub-Millisecond Stage Breakdown (Last Steady-State) ---")
        print(f"  Device:                {perf.get('device')} ({perf.get('precision')})")
        print(f"  Inference Resolution: {perf.get('inference_imgsz')}x{perf.get('inference_imgsz')}")
        print(f"  1. Video Decode:       {decode_ms:6.2f} ms")
        print(f"  2. YOLO Preprocess:    {pre_ms:6.2f} ms")
        print(f"  3. YOLOv8n Inference:  {inf_ms:6.2f} ms")
        print(f"  4. NMS Postprocess:    {post_ms:6.2f} ms")
        print(f"  5. ByteTrack (Pure):   {track_ms:6.2f} ms")
        print(f"  6. Traffic Analytics:  {ana_ms:6.2f} ms")
        print(f"  7. Visualizer Render:  {ren_ms:6.2f} ms")
        print(f"  8. Async Video Write:  {wrt_ms:6.2f} ms")
        print(f"  -------------------------------------")
        print(f"  Sum of Stages:         {stage_sum:6.2f} ms")
        print(f"  Total Wall Latency:    {tot_ms:6.2f} ms  (Reconciled diff: {abs(stage_sum - tot_ms):.2f} ms)")
        print(f"  VRAM Allocated:        {perf.get('vram_allocated_mb'):.1f} MB / {perf.get('vram_total_mb'):.1f} MB")
        print(f"  Host CPU:              {perf.get('cpu_percent'):.1f}%")


    print("\n--- Counting & Traffic State Verification ---")
    print(f"  Total Vehicles Counted: {result.get('total_vehicles')}")
    print(f"  Vehicle Breakdown:      {result.get('vehicle_classes')}")
    print(f"  Overall Traffic State:  {result.get('traffic_state')}")
    print(f"  Road Count:             {result.get('road_count')}")
    for rd in result.get("roads", []):
        print(f"    - Road {rd.get('name')}: {rd.get('vehicles')} veh, flow: {rd.get('flow_volume')} v/m, state: {rd.get('traffic_state')}")

    print("\n--- Output Verification ---")
    out_video = result.get("outputs", {}).get("annotated_video")
    if out_video and Path(out_video).exists():
        print(f"  Annotated Video Exists: True ({Path(out_video).stat().st_size} bytes)")
    else:
        print(f"  Annotated Video Exists: False")
    print("=" * 70)

if __name__ == "__main__":
    test_pipeline(num_frames=100)
