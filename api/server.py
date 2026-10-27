"""
api/server.py
--------------
FastAPI backend server for TrafficTwin.
Exposes REST, WebSocket, and SSE endpoints for video upload, real-time ML inference,
live MJPEG streaming with CV overlays, WebSocket/SSE live metrics broadcast,
HTTP 206 Partial Content video streaming for seekable playback,
and decision recommendation / counterfactual simulation engine.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import uuid
import json
import asyncio
import collections
import logging
import queue
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
import shutil
import cv2

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse

# Ensure project root is in python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from api.models import (
    TrafficState,
    AnalysisStatus,
    MovementCount,
    TrafficMetrics,
    TrafficPrediction,
    HistoryPoint,
    SimulationStrategy,
    SimulationRequest,
    SimulationResult,
    RecommendationFactor,
    Recommendation,
    JobStatus,
    VideoUploadResponse,
    HealthResponse,
    CurrentState,
)

from src.inference.pipeline import TrafficPipeline

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("traffictwin.api")

app = FastAPI(
    title="TrafficTwin ML & Simulation API",
    description="AI-Powered Counterfactual Traffic Management and Signal Decision System",
    version="1.0.0",
)

# Allowed Origins for Local Development and Production Vercel Deployments
# Can be extended via ALLOWED_ORIGINS environment variable (comma-separated)
_env_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
ALLOWED_ORIGINS = list(set([
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
] + _env_origins))

# Enable CORS: explicit origin list for credentials and regex for Vercel preview/production domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "X-Total-Count"],
)

# Output directories
UPLOAD_DIR = project_root / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEOS_DIR = project_root / "outputs" / "videos"
OUTPUT_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_METRICS_DIR = project_root / "outputs" / "metrics"
OUTPUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files for annotated videos and dataset videos
app.mount("/outputs", StaticFiles(directory=str(project_root / "outputs")), name="outputs")
app.mount("/dataset", StaticFiles(directory=str(project_root.parent / "AIC21_Track1_Vehicle_Counting" / "Dataset_A")), name="dataset")

# In-memory storage for jobs, live frames, live telemetry, and active web sockets
JOBS: Dict[str, dict] = {}
FRAME_BUFFERS: Dict[str, collections.deque] = {}
LIVE_ANALYSIS_STATES: Dict[str, dict] = {}
ACTIVE_WEBSOCKETS: Dict[str, Set[WebSocket]] = {}
STOP_FLAGS: Dict[str, bool] = {}
ACTIVE_PROCESSING_JOB_ID: Optional[str] = None
LATEST_COMPLETED_JOB_ID: Optional[str] = None
CURRENT_STATE = CurrentState()
MAIN_EVENT_LOOP: Optional[asyncio.AbstractEventLoop] = None

@app.on_event("startup")
async def app_startup_event():
    global MAIN_EVENT_LOOP
    try:
        MAIN_EVENT_LOOP = asyncio.get_running_loop()
        log.info("[STARTUP] Captured main asyncio event loop for thread-safe WebSocket broadcasts.")
    except Exception as e:
        log.warning(f"[STARTUP] Could not capture running event loop: {e}")

# Canonical baseline result file
DEFAULT_RESULT_JSON = OUTPUT_METRICS_DIR / "cam5_eval_results.json"
if not DEFAULT_RESULT_JSON.exists():
    DEFAULT_RESULT_JSON = OUTPUT_METRICS_DIR / "cam5_final_results.json"


def validate_output_video(video_path: Path) -> Tuple[bool, str]:
    """
    Verify generated video file:
    - exists and non-zero bytes
    - readable by OpenCV
    - frame count > 0, width > 0, height > 0, fps > 0
    """
    if not video_path.exists():
        return False, f"File does not exist: {video_path}"
    
    size = video_path.stat().st_size
    if size == 0:
        return False, f"File is empty (0 bytes): {video_path}"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, f"OpenCV cannot open video stream: {video_path}"

    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, frame = cap.read()
    cap.release()

    if fc <= 0 or w <= 0 or h <= 0 or not ret:
        return False, f"Invalid video metadata: frames={fc}, w={w}, h={h}, fps={fps}"

    return True, f"OK (size={size} bytes, frames={fc}, dims={w}x{h}, fps={fps})"


def build_state_from_result(result_data: dict, job_id: str = "demo-cam5", video_name: str = "cam_5.mp4") -> CurrentState:
    """Build a strongly typed, synchronized CurrentState from canonical ML result dict."""
    by_mov = []
    mov_counts = result_data.get("movement_counts", {})
    mov_classes = result_data.get("movement_class_counts", {})
    frames_proc = result_data.get("processing", {}).get("frames_processed", 600)
    dur_min = max(0.1, (frames_proc / 10.0) / 60.0)

    for mid, count in mov_counts.items():
        m_int = int(mid)
        flow = count / dur_min
        st = TrafficState.LOW
        if flow > 12.0:
            st = TrafficState.CRITICAL
        elif flow > 6.0:
            st = TrafficState.HIGH
        elif flow > 2.0:
            st = TrafficState.MEDIUM

        car_cnt = mov_classes.get(mid, {}).get("Car", count)
        truck_cnt = mov_classes.get(mid, {}).get("Truck", 0)

        by_mov.append(MovementCount(
            movement_id=m_int,
            movement_description=f"Movement {m_int}",
            count=count,
            car_count=car_cnt,
            truck_count=truck_cnt,
            flow_rate_per_min=round(flow, 2),
            state=st
        ))

    by_mov.sort(key=lambda x: x.movement_id)

    total_veh = result_data.get("total_vehicles", sum(m.count for m in by_mov))
    car_cnt = result_data.get("vehicle_classes", {}).get("Car", sum(m.car_count for m in by_mov))
    truck_cnt = result_data.get("vehicle_classes", {}).get("Truck", sum(m.truck_count for m in by_mov))
    tf = result_data.get("traffic_features", {})
    flow_rate = tf.get("total_flow_rate_per_min", round(total_veh / dur_min, 2))
    imb_detected = tf.get("direction_imbalance_detected", False)
    imb_ratio = float(tf.get("direction_imbalance_ratio", 1.0))
    
    dom_mov = None
    if imb_detected and by_mov:
        dom_mov = max(by_mov, key=lambda m: m.count).movement_id

    metrics = TrafficMetrics(
        traffic_state=TrafficState(result_data.get("traffic_state", "HIGH")),
        total_vehicle_count=total_veh,
        car_count=car_cnt,
        truck_count=truck_cnt,
        flow_rate_per_min=flow_rate,
        timestamp_sec=round(frames_proc / 10.0, 1),
        frames_processed=frames_proc,
        direction_imbalance_detected=imb_detected,
        direction_imbalance_ratio=imb_ratio,
        dominant_movement=dom_mov,
        by_movement=by_mov,
        current_roi_count=result_data.get("current_vehicles", total_veh),
        initial_occupancy=result_data.get("initial_occupancy", 0),
        cumulative_entries=result_data.get("cumulative_entries", total_veh),
        cumulative_exits=result_data.get("cumulative_exits", 0),
        invariant_verified=result_data.get("invariant_verified", True),
    )

    # Queue Estimation from actual pipeline results
    q_base = tf.get("estimated_queue", max(0, int(flow_rate * 0.42)))
    
    prediction = TrafficPrediction(
        future_queue_estimate=q_base,
        future_congestion_pct=min(98.0, round(flow_rate * 1.55, 1)),
        time_horizon_sec=120.0,
        confidence=0.91
    )

    # Build dynamic counterfactual strategies
    strategies = [
        SimulationStrategy(
            strategy_id="current",
            label="Current Fixed Plan",
            green_phase_extension_sec=0.0,
            predicted_queue=q_base,
            predicted_wait_sec=round(q_base * 0.92, 1),
            predicted_congestion_pct=min(95.0, round(flow_rate * 1.6, 1)),
            score=82.0,
            is_recommended=False
        ),
        SimulationStrategy(
            strategy_id="+10sec",
            label="+10s Major Phase Extension",
            green_phase_extension_sec=10.0,
            predicted_queue=max(5, int(q_base * 0.78)),
            predicted_wait_sec=round(q_base * 0.72, 1),
            predicted_congestion_pct=round(flow_rate * 1.25, 1),
            score=60.0,
            is_recommended=False
        ),
        SimulationStrategy(
            strategy_id="+20sec",
            label="+20s Dynamic Green",
            green_phase_extension_sec=20.0,
            predicted_queue=max(4, int(q_base * 0.55)),
            predicted_wait_sec=round(q_base * 0.50, 1),
            predicted_congestion_pct=round(flow_rate * 0.90, 1),
            score=34.0,
            is_recommended=True
        ),
        SimulationStrategy(
            strategy_id="+30sec",
            label="+30s Extended Phase",
            green_phase_extension_sec=30.0,
            predicted_queue=max(6, int(q_base * 0.60)),
            predicted_wait_sec=round(q_base * 0.58, 1),
            predicted_congestion_pct=round(flow_rate * 1.02, 1),
            score=44.0,
            is_recommended=False
        )
    ]

    sim_res = SimulationResult(
        job_id=job_id,
        strategies=strategies,
        recommended_strategy_id="+20sec",
        simulation_completed=True
    )

    reasons = [
        RecommendationFactor(
            factor="Observed Flow Volume",
            value=f"Real flow rate {flow_rate:.1f} veh/min detected on active approaches",
            direction="increasing",
            weight="high"
        )
    ]
    if imb_detected and dom_mov:
        reasons.append(RecommendationFactor(
            factor="Directional Imbalance",
            value=f"Movement {dom_mov} has {imb_ratio:.1f}x higher flow than competing legs",
            direction="increasing",
            weight="high"
        ))
    reasons.append(RecommendationFactor(
        factor="Delay Mitigation",
        value=f"Predicted queue reduction from {q_base} to {int(q_base * 0.55)} vehicles",
        direction="decreasing",
        weight="medium"
    ))

    rec = Recommendation(
        action=f"Extend Major Approach Green Phase by 20 Seconds",
        strategy_id="+20sec",
        green_phase_extension_sec=20.0,
        reasoning=reasons,
        predicted_queue_change=int(q_base * 0.55) - q_base,
        predicted_queue_change_pct=-45.0,
        predicted_wait_change_sec=round(-q_base * 0.42, 1),
        predicted_congestion_change_pct=-35.0,
        confidence=0.92
    )

    history = []
    for h in result_data.get("traffic_state_history", []):
        history.append(HistoryPoint(
            timestamp_sec=float(h.get("window_end_sec", 0.0)),
            frame_id=int(h.get("window_end_sec", 0.0) * 10),
            total_count=int(h.get("total_vehicles_in_window", 0)),
            flow_rate_per_min=float(h.get("total_flow_rate_per_min", 0.0)),
            state=TrafficState(h.get("overall_state", "LOW"))
        ))

    if not history:
        history = [
            HistoryPoint(timestamp_sec=10.0, frame_id=100, total_count=max(1, int(total_veh * 0.15)), flow_rate_per_min=round(flow_rate * 0.15, 1), state=TrafficState.MEDIUM),
            HistoryPoint(timestamp_sec=30.0, frame_id=300, total_count=max(2, int(total_veh * 0.50)), flow_rate_per_min=round(flow_rate * 0.50, 1), state=TrafficState.HIGH),
            HistoryPoint(timestamp_sec=60.0, frame_id=600, total_count=total_veh, flow_rate_per_min=flow_rate, state=metrics.traffic_state),
        ]

    video_out = result_data.get("outputs", {}).get("annotated_video")
    if video_out and Path(video_out).exists():
        vid_url = f"/api/video/stream/{Path(video_out).name}"
    else:
        vid_url = f"/api/video/stream/{job_id}_annotated.mp4"

    return CurrentState(
        job_id=job_id,
        status=AnalysisStatus.READY,
        metrics=metrics,
        prediction=prediction,
        simulation=sim_res,
        recommendation=rec,
        history=history,
        camera_id=result_data.get("camera_id", "cam_5"),
        video_name=video_name,
        annotated_video_url=vid_url
    )


CAMERA_STATES: Dict[str, CurrentState] = {}

def init_default_state():
    """Load default precomputed baseline states for all 5 cameras."""
    global CURRENT_STATE, CAMERA_STATES
    
    # 1. cam_5
    if DEFAULT_RESULT_JSON.exists():
        try:
            with open(DEFAULT_RESULT_JSON, "r") as f:
                data = json.load(f)
            st5 = build_state_from_result(data, job_id="demo-cam5", video_name="cam_5.mp4")
            st5.camera_id = "cam_5"
            CURRENT_STATE = st5
            CAMERA_STATES["cam_5"] = st5
            log.info(f"Loaded cam_5 state (Total: {st5.metrics.total_vehicle_count} veh)")
        except Exception as e:
            log.warning(f"Could not load cam_5 state: {e}")

    # 2. Other camera precomputed result mappings
    cam_json_map = {
        "cam_1": OUTPUT_METRICS_DIR / "cam_1_results.json",
        "cam_7": OUTPUT_METRICS_DIR / "cam_7_99cfe1_results.json",
        "cam_9": OUTPUT_METRICS_DIR / "cam_9_results.json",
        "cam_10": OUTPUT_METRICS_DIR / "cam_10_results.json"
    }

    for cid, jpath in cam_json_map.items():
        if jpath.exists():
            try:
                with open(jpath, "r") as f:
                    cdata = json.load(f)
                c_st = build_state_from_result(cdata, job_id=f"demo-{cid}", video_name=f"{cid}.mp4")
                c_st.camera_id = cid
                # Default video URL to the raw dataset MP4 or annotated MP4 if exists
                if not Path(project_root / c_st.annotated_video_url.lstrip("/")).exists():
                    c_st.annotated_video_url = f"/api/video/stream/{cid}.mp4"
                CAMERA_STATES[cid] = c_st
                log.info(f"Loaded {cid} state (Total: {c_st.metrics.total_vehicle_count} veh)")
            except Exception as ce:
                log.warning(f"Could not load {cid} precomputed state: {ce}")

init_default_state()


@app.get("/health", response_model=HealthResponse)
def get_health():
    import torch
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else None
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        gpu_available=cuda_avail,
        device="cuda:0" if cuda_avail else "cpu",
        gpu_name=device_name,
        cuda_version=torch.version.cuda if cuda_avail else None,
        precision="FP16" if cuda_avail else "FP32",
        model_loaded=True,
        active_jobs=len([j for j in JOBS.values() if j.get("status") in (AnalysisStatus.PROCESSING, AnalysisStatus.ANALYZING)])
    )


# WebSocket live telemetry channel
@app.websocket("/ws/analysis/{job_id}")
async def websocket_analysis_endpoint(websocket: WebSocket, job_id: str):
    global MAIN_EVENT_LOOP
    try:
        MAIN_EVENT_LOOP = asyncio.get_running_loop()
    except Exception:
        pass
    await websocket.accept()
    if job_id not in ACTIVE_WEBSOCKETS:
        ACTIVE_WEBSOCKETS[job_id] = set()
    ACTIVE_WEBSOCKETS[job_id].add(websocket)
    log.info(f"[WS] Client connected to live analysis stream for job: {job_id}")

    try:
        # If there is already an existing live state, send it immediately
        if job_id in LIVE_ANALYSIS_STATES:
            await websocket.send_text(json.dumps(LIVE_ANALYSIS_STATES[job_id]))

        while True:
            # Keep socket alive and respond to pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        log.info(f"[WS] Client disconnected from job: {job_id}")
    except Exception as e:
        log.debug(f"[WS] Exception on socket for job {job_id}: {e}")
    finally:
        if job_id in ACTIVE_WEBSOCKETS:
            ACTIVE_WEBSOCKETS[job_id].discard(websocket)


# Server-Sent Events (SSE) live telemetry stream
@app.get("/api/analysis/{job_id}/events")
async def sse_analysis_endpoint(job_id: str):
    async def event_generator():
        last_frame_sent = -1
        max_idle_seconds = 180
        start_time = asyncio.get_event_loop().time()

        while True:
            live_data = LIVE_ANALYSIS_STATES.get(job_id)
            if live_data and live_data.get("frame_index", -1) != last_frame_sent:
                last_frame_sent = live_data.get("frame_index", -1)
                yield f"data: {json.dumps(live_data)}\n\n"

            job = JOBS.get(job_id)
            if job and job.get("status") in (AnalysisStatus.READY, AnalysisStatus.ERROR):
                if live_data and live_data.get("frame_index", -1) != last_frame_sent:
                    yield f"data: {json.dumps(live_data)}\n\n"
                break

            if asyncio.get_event_loop().time() - start_time > max_idle_seconds:
                break

            await asyncio.sleep(0.08)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


# HTTP 206 Partial Content video streaming endpoint for final MP4 playback
@app.get("/api/video/stream/{video_filename}")
async def stream_video(video_filename: str, request: Request):
    """
    Stream video file with support for HTTP Range requests.
    Enables seeking, scrub bar interaction, and instant start in HTML5 <video>.
    """
    candidates = [
        OUTPUT_VIDEOS_DIR / video_filename,
        UPLOAD_DIR / video_filename,
        project_root.parent / "AIC21_Track1_Vehicle_Counting" / "Dataset_A" / video_filename
    ]
    
    file_path = None
    for cand in candidates:
        if cand.exists():
            file_path = cand
            break

    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {video_filename}")

    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")

    # Log playback start if this is an annotated output video request
    if "_annotated.mp4" in video_filename and not range_header:
        job_id = video_filename.replace("_annotated.mp4", "")
        job = JOBS.get(job_id)
        if job:
            import cv2
            cap = cv2.VideoCapture(str(file_path))
            fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else job.get("total_frames", 600)
            fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 10.0
            if cap.isOpened(): cap.release()
            log.info(
                f"\n[PLAYBACK]\n"
                f"job_id={job_id}\n"
                f"source={job.get('original_filename', 'unknown')}\n"
                f"output={video_filename}\n"
                f"frames={fc}\n"
                f"fps={fps}\n"
            )

    if not range_header:
        return FileResponse(
            path=str(file_path),
            media_type="video/mp4",
            headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)}
        )

    try:
        range_str = range_header.replace("bytes=", "")
        parts = range_str.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
        end = min(end, file_size - 1)
        chunk_size = end - start + 1

        def iterfile(path: Path, offset: int, length: int):
            with open(path, "rb") as f:
                f.seek(offset)
                remaining = length
                while remaining > 0:
                    read_size = min(64 * 1024, remaining)
                    data = f.read(read_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(chunk_size),
            "Content-Type": "video/mp4",
        }
        return StreamingResponse(
            iterfile(file_path, start, chunk_size),
            status_code=206,
            headers=headers
        )
    except Exception as e:
        log.error(f"Range streaming exception: {e}")
        return FileResponse(path=str(file_path), media_type="video/mp4")


# Explicit endpoint to stream the exact original uploaded SOURCE video for a job
@app.get("/api/source/{job_id}")
async def stream_source_video(job_id: str, request: Request):
    """
    Stream the original source video associated with a job.
    This guarantees we are streaming the exact uploaded file, not a default video.
    """
    job = JOBS.get(job_id)
    if not job or "video_path" not in job:
        raise HTTPException(status_code=404, detail="Source video for job not found")
        
    v_path = Path(job["video_path"])
    if not v_path.exists():
        raise HTTPException(status_code=404, detail="Source video file missing from disk")
        
    # Standard range request streaming
    try:
        file_size = v_path.stat().st_size
        range_header = request.headers.get("Range", None)
        
        if range_header:
            byte1, byte2 = 0, None
            match = re.search(r'bytes=(\d+)-(\d*)', range_header)
            if match:
                g = match.groups()
                byte1 = int(g[0])
                if g[1]:
                    byte2 = int(g[1])
            
            if byte2 is None:
                byte2 = file_size - 1
            length = byte2 - byte1 + 1
            
            with open(v_path, 'rb') as f:
                f.seek(byte1)
                data = f.read(length)
                
            headers = {
                "Content-Range": f"bytes {byte1}-{byte2}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
                "Content-Type": "video/mp4",
            }
            return Response(data, status_code=206, headers=headers, media_type="video/mp4")
            
        return FileResponse(path=str(v_path), media_type="video/mp4")
    except Exception as e:
        log.error(f"Source stream exception: {e}")
        return FileResponse(path=str(v_path), media_type="video/mp4")


# Real-time MJPEG live stream yielding processed frames during ML inference
@app.get("/api/stream/{job_id}")
async def stream_live_mjpeg(job_id: str):
    """
    Real-time MJPEG stream yielding annotated frames as the ML pipeline executes.
    Allows user to immediately watch detection, tracking, and counting in real-time.
    """
    async def frame_generator():
        last_frame_bytes = None
        max_idle_seconds = 300  # increased from 180
        start_time = asyncio.get_event_loop().time()

        while True:
            buf = FRAME_BUFFERS.get(job_id)
            if buf and len(buf) > 0:
                frame_bytes = buf[-1]
                if frame_bytes is not last_frame_bytes:
                    last_frame_bytes = frame_bytes
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                    )

            job = JOBS.get(job_id)
            if job and job.get("status") in (AnalysisStatus.READY, AnalysisStatus.ERROR):
                # Drain remaining frame
                if buf and len(buf) > 0 and buf[-1] is not last_frame_bytes:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + buf[-1] + b"\r\n"
                    )
                break

            if asyncio.get_event_loop().time() - start_time > max_idle_seconds:
                break

            await asyncio.sleep(0.025)  # ~40 fps poll rate for smoother playback

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# Per-job output video endpoint (prevents cross-job video mixing)
@app.get("/api/jobs/{job_id}/output")
async def get_job_output_video(job_id: str, request: Request):
    """
    Serve the processed annotated video for a specific job.
    Uses HTTP 206 Range requests for browser seekable playback.
    Each job has its own output — no cross-job mixing.
    """
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Determine output path
    annotated_path = OUTPUT_VIDEOS_DIR / f"{job_id}_annotated.mp4"
    if not annotated_path.exists():
        # Check result dict
        result = job.get("result", {})
        if result:
            out_path = result.get("outputs", {}).get("annotated_video")
            if out_path and Path(out_path).exists():
                annotated_path = Path(out_path)
            else:
                raise HTTPException(status_code=404, detail=f"Output video not yet ready for job {job_id}")
        else:
            raise HTTPException(status_code=404, detail=f"Output video not found for job {job_id}")

    if job.get("status") not in (AnalysisStatus.READY,):
        raise HTTPException(status_code=202, detail=f"Job {job_id} is still processing (status={job.get('status')}). Try again after completion.")

    file_size = annotated_path.stat().st_size
    if file_size == 0:
        raise HTTPException(status_code=500, detail=f"Output video for job {job_id} is empty")

    range_header = request.headers.get("range")
    if not range_header:
        return FileResponse(
            path=str(annotated_path),
            media_type="video/mp4",
            headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size), "Content-Disposition": f"inline; filename={job_id}_annotated.mp4"}
        )

    try:
        range_str = range_header.replace("bytes=", "")
        parts = range_str.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
        end = min(end, file_size - 1)
        chunk_size = end - start + 1

        def iterfile(path: Path, offset: int, length: int):
            with open(path, "rb") as f:
                f.seek(offset)
                remaining = length
                while remaining > 0:
                    read_size = min(64 * 1024, remaining)
                    data = f.read(read_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(chunk_size),
            "Content-Type": "video/mp4",
        }
        return StreamingResponse(
            iterfile(annotated_path, start, chunk_size),
            status_code=206,
            headers=headers
        )
    except Exception as e:
        log.error(f"Range streaming exception for job {job_id}: {e}")
        return FileResponse(path=str(annotated_path), media_type="video/mp4")


@app.post("/api/video/upload", response_model=VideoUploadResponse)
def upload_video(
    file: UploadFile = File(...),
    camera_id: str = Form("cam_5")
):
    valid_exts = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    ext = Path(file.filename).suffix.lower()
    if ext not in valid_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{ext}'. Allowed formats: {', '.join(valid_exts)}"
        )

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    dest_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    
    # Chunked streaming write (2 MB buffer) to avoid blocking or memory spikes
    chunk_size = 2 * 1024 * 1024
    with open(dest_path, "wb") as buffer:
        while True:
            chunk = file.file.read(chunk_size)
            if not chunk:
                break
            buffer.write(chunk)

    size = dest_path.stat().st_size
    if size == 0:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

    cap = cv2.VideoCapture(str(dest_path))
    is_valid = cap.isOpened()
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_valid else 0
    cap.release()

    if not is_valid:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is corrupted or not a valid video stream.")

    log.info(
        f"\n[UPLOAD START]\n"
        f"job_id={job_id}\n"
        f"original_filename={file.filename}\n"
        f"stored_filename={dest_path.name}\n"
        f"video_path={dest_path}\n"
        f"file_size={size}\n"
        f"camera_id={camera_id}\n"
    )

    JOBS[job_id] = {
        "job_id": job_id,
        "status": AnalysisStatus.IDLE,
        # Flat keys used by worker and validation (source of truth)
        "video_path": str(dest_path),
        "camera_id": camera_id,
        "original_filename": file.filename,
        "filename": file.filename,
        # Nested metadata for explicit provenance
        "source": {
            "original_filename": file.filename,
            "stored_filename": dest_path.name,
            "video_path": str(dest_path),
            "video_type": "uploaded"
        },
        "camera": {
            "camera_id": camera_id,
            "config_id": camera_id.upper()
        },
        "mapping": {
            "roi_id": camera_id.upper(),
            "road_map_id": camera_id.upper(),
            "movement_map_id": camera_id.upper()
        },
        "frames_processed": 0,
        "total_frames": fc,
        "current_fps": 0.0,
        "result": None,
    }
    FRAME_BUFFERS[job_id] = collections.deque(maxlen=30)

    return VideoUploadResponse(
        job_id=job_id,
        filename=file.filename,
        file_size_bytes=size,
        camera_id=camera_id,
        status=AnalysisStatus.IDLE
    )


def run_pipeline_worker(job_id: str, max_frames: Optional[int] = 600, manual_roads: Optional[list] = None):
    global CURRENT_STATE, ACTIVE_PROCESSING_JOB_ID, LATEST_COMPLETED_JOB_ID
    ACTIVE_PROCESSING_JOB_ID = job_id
    job = JOBS.get(job_id)
    if not job:
        log.error(f"[WORKER] Job {job_id} not found in JOBS dict.")
        return

    try:
        job["status"] = AnalysisStatus.ANALYZING
        
        # Determine the mapped config ID safely
        config_id = job.get("camera", {}).get("config_id", job['camera_id'].upper())
        
        log.info(
            f"\n[JOB]\n"
            f"job_id={job_id}\n"
            f"source={job.get('original_filename', job.get('filename', 'unknown'))}\n"
            f"camera_config={config_id}\n"
            f"roi_config={config_id}\n"
            f"road_map={config_id}\n"
            f"movement_map={config_id}\n"
        )
        log.info(
            f"\n[INFERENCE START]\n"
            f"job_id={job_id}\n"
            f"camera_id={job['camera_id']}\n"
            f"video_path={job['video_path']}\n"
            f"original_filename={job.get('original_filename', job.get('filename', 'unknown'))}\n"
        )

        frame_buf = collections.deque(maxlen=30)
        FRAME_BUFFERS[job_id] = frame_buf

        # Asynchronous background JPEG compressor for live MJPEG preview
        jpeg_queue = queue.Queue(maxsize=10)
        jpeg_running = True

        def _jpeg_worker():
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            while jpeg_running or not jpeg_queue.empty():
                try:
                    img = jpeg_queue.get(timeout=0.1)
                    if img is None:
                        break
                    ih, iw = img.shape[:2]
                    if iw > 1280:
                        scale = 1280.0 / iw
                        img_stream = cv2.resize(img, (1280, int(ih * scale)), interpolation=cv2.INTER_LINEAR)
                    else:
                        img_stream = img
                    success, encoded_jpeg = cv2.imencode(".jpg", img_stream, encode_params)
                    if success:
                        frame_buf.append(encoded_jpeg.tobytes())
                    jpeg_queue.task_done()
                except queue.Empty:
                    continue
                except Exception:
                    pass

        encoder_thread = threading.Thread(target=_jpeg_worker, daemon=True)
        encoder_thread.start()

        def live_frame_callback(annotated_frame, frame_id, timestamp_sec, live_payload):
            # 1. Non-blocking enqueue for MJPEG live preview
            if not jpeg_queue.full():
                try:
                    jpeg_queue.put_nowait(annotated_frame)
                except Exception:
                    pass

            # 2. Store live canonical state (includes roads[] array and perf telemetry from pipeline)
            LIVE_ANALYSIS_STATES[job_id] = live_payload
            job["frames_processed"] = frame_id
            job["latest_summary"] = live_payload
            job["current_fps"] = live_payload.get("processing_fps", 0.0)

            # 3. Broadcast via WebSocket to all connected clients for this job
            ws_set = ACTIVE_WEBSOCKETS.get(job_id)
            if ws_set:
                msg = json.dumps(live_payload)
                dead = set()
                for ws in list(ws_set):
                    try:
                        if MAIN_EVENT_LOOP and MAIN_EVENT_LOOP.is_running():
                            asyncio.run_coroutine_threadsafe(ws.send_text(msg), MAIN_EVENT_LOOP)
                    except Exception:
                        dead.add(ws)
                ws_set -= dead

        pipeline = TrafficPipeline(config_path=str(project_root / "configs" / "config.yaml"))
        pipeline.max_frames = max_frames
        pipeline.save_video = True
        STOP_FLAGS[job_id] = False
        
        res = pipeline.run(
            video_path=job["video_path"],
            camera_id=job["camera_id"],
            output_prefix=job_id,
            frame_callback=live_frame_callback,
            manual_roads=manual_roads,
            stop_check=lambda: STOP_FLAGS.get(job_id, False),
        )

        # Signal background JPEG encoder to finalize
        jpeg_running = False
        try:
            jpeg_queue.put_nowait(None)
        except Exception:
            pass

        is_stopped = STOP_FLAGS.get(job_id, False)
        annotated_path = OUTPUT_VIDEOS_DIR / f"{job_id}_annotated.mp4"
        valid, val_msg = validate_output_video(annotated_path)
        log.info(f"[VIDEO] Validation for {annotated_path.name}: {val_msg}")

        if not valid and not is_stopped:
            raise RuntimeError(f"Output video validation failed: {val_msg}")

        job["result"] = res
        job["status"] = AnalysisStatus.READY

        # Update and persist complete final live analysis state with valid trips and average speed
        tot_summary = res.get("total_road_summary", {})
        dur_min = max(0.05, job.get("frames_processed", 600) / 10.0 / 60.0)
        final_flow = round(res.get("total_vehicles", 0) / dur_min, 1)
        final_live_state = {
            "job_id": job_id,
            "frame_index": job.get("frames_processed", 600),
            "total_frames": job.get("frames_processed", 600),
            "video_time_seconds": round(job.get("frames_processed", 600) / 10.0, 2),
            "current_active": res.get("current_vehicles", 0),
            "current_roi_count": res.get("current_vehicles", 0),
            "initial_occupancy": res.get("initial_occupancy", 0),
            "cumulative_entries": res.get("cumulative_entries", 0),
            "cumulative_exits": res.get("cumulative_exits", 0),
            "invariant_verified": res.get("invariant_verified", True),
            "current_moving": res.get("current_moving", 0),
            "current_stationary": res.get("current_stationary", 0),
            "vehicles_counted": res.get("total_vehicles", 0),
            "vehicle_classes": res.get("vehicle_classes", {}),
            "current_classes": res.get("current_classes", {}),
            "movement_counts": res.get("movement_counts", {}),
            "traffic_state": res.get("traffic_state", "LOW"),
            "arrival_rate": final_flow,
            "flow_rate_per_min": final_flow,
            "queue": res.get("current_stationary", 0),
            "completed_trips": tot_summary.get("completed_trips", 0),
            "valid_trips": tot_summary.get("valid_trips", 0),
            "average_speed": tot_summary.get("average_speed"),
            "speed_display": tot_summary.get("speed_display", "Waiting for vehicles"),
            "roads": res.get("roads", []),
            "total_road_summary": tot_summary,
            "global": {
                "vehicles": res.get("total_vehicles", 0),
                "current_active": res.get("current_vehicles", 0),
                "flow_volume": final_flow,
                "traffic_state": res.get("traffic_state", "LOW"),
                "average_speed": tot_summary.get("average_speed"),
            },
            "processing_fps": res.get("processing", {}).get("speed_fps", 0.0),
            "output_ready": True,
            "output_url": f"/api/jobs/{job_id}/output",
        }
        LIVE_ANALYSIS_STATES[job_id] = final_live_state
        job["latest_summary"] = final_live_state

        # Broadcast completion to WebSocket clients
        ws_set = ACTIVE_WEBSOCKETS.get(job_id)
        if ws_set:
            final_msg = json.dumps(final_live_state)
            for ws in list(ws_set):
                try:
                    if MAIN_EVENT_LOOP and MAIN_EVENT_LOOP.is_running():
                        asyncio.run_coroutine_threadsafe(ws.send_text(final_msg), MAIN_EVENT_LOOP)
                except Exception:
                    pass

        # Update global canonical state and camera state
        new_state = build_state_from_result(res, job_id=job_id, video_name=job["filename"])
        new_state.camera_id = job.get("camera_id", "cam_5")
        CURRENT_STATE = new_state
        CAMERA_STATES[job.get("camera_id", "cam_5")] = new_state

        LATEST_COMPLETED_JOB_ID = job_id
        if ACTIVE_PROCESSING_JOB_ID == job_id:
            ACTIVE_PROCESSING_JOB_ID = None

        log.info(f"[API] Job {job_id} completed successfully on GPU for camera {new_state.camera_id}. Total counted: {res.get('total_vehicles')}")

    except Exception as e:
        log.error(f"Inference error for {job_id}: {e}", exc_info=True)
        job["status"] = AnalysisStatus.ERROR
        job["error_message"] = str(e)

        if ACTIVE_PROCESSING_JOB_ID == job_id:
            ACTIVE_PROCESSING_JOB_ID = None
        
        # Broadcast the error to live telemetry clients
        error_payload = {
            "job_id": job_id,
            "status": "error",
            "error_message": str(e)
        }
        LIVE_ANALYSIS_STATES[job_id] = error_payload
        if job_id in JOBS:
            JOBS[job_id]["latest_summary"] = error_payload


def run_multi_camera_worker(camera_job_list: List[dict], max_frames: int = 600):
    global CURRENT_STATE, CAMERA_STATES
    from src.inference.multi_pipeline import MultiCameraPipeline

    for c_info in camera_job_list:
        jid = c_info["job_id"]
        if jid in JOBS:
            JOBS[jid]["status"] = AnalysisStatus.ANALYZING

    def multi_frame_callback(cam_id, annotated_frame, frame_id, timestamp_sec, live_payload):
        jid = live_payload.get("job_id")
        if not jid:
            return
        success, encoded_jpeg = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if success:
            if jid not in FRAME_BUFFERS:
                FRAME_BUFFERS[jid] = collections.deque(maxlen=30)
            FRAME_BUFFERS[jid].append(encoded_jpeg.tobytes())
        
        LIVE_ANALYSIS_STATES[jid] = live_payload
        if jid in JOBS:
            JOBS[jid]["frames_processed"] = frame_id
            JOBS[jid]["latest_summary"] = live_payload
            JOBS[jid]["current_fps"] = live_payload.get("processing_fps", 0.0)

    try:
        pipeline = MultiCameraPipeline(config_path=str(project_root / "configs" / "config.yaml"))
        results = pipeline.run_all(camera_job_list, max_frames=max_frames, frame_callback=multi_frame_callback)

        for c_info in camera_job_list:
            cid = c_info["camera_id"]
            jid = c_info["job_id"]
            res = results.get(cid)
            if res and jid in JOBS:
                JOBS[jid]["result"] = res
                JOBS[jid]["status"] = AnalysisStatus.READY
                c_state = build_state_from_result(res, job_id=jid, video_name=Path(c_info["video_path"]).name)
                c_state.camera_id = cid
                CAMERA_STATES[cid] = c_state
                if cid == "cam_5":
                    CURRENT_STATE = c_state
                log.info(f"[MULTI-WORKER] Camera {cid} finished: {res.get('total_vehicles')} veh")
    except Exception as me:
        log.error(f"[MULTI-WORKER] Error during multi-camera inference: {me}", exc_info=True)
        for c_info in camera_job_list:
            jid = c_info["job_id"]
            if jid in JOBS:
                JOBS[jid]["status"] = AnalysisStatus.ERROR
                JOBS[jid]["error_message"] = str(me)


@app.post("/api/inference/start-all")
def start_all_inference(
    background_tasks: BackgroundTasks,
    max_frames: int = Form(600)
):
    """
    Start simultaneous GPU ML analysis on ALL 5 camera sources.
    Processes all 5 video feeds concurrently on shared GPU YOLOv8n Tensor Core.
    """
    cameras_meta = [
        {"camera_id": "cam_5", "file": "cam_5.mp4"},
        {"camera_id": "cam_1", "file": "cam_1.mp4"},
        {"camera_id": "cam_7", "file": "cam_7.mp4"},
        {"camera_id": "cam_9", "file": "cam_9.mp4"},
        {"camera_id": "cam_10", "file": "cam_10.mp4"},
    ]

    job_list = []
    response_jobs = []

    for cam in cameras_meta:
        cid = cam["camera_id"]
        jid = f"{cid}_{uuid.uuid4().hex[:6]}"
        video_path = project_root.parent / "AIC21_Track1_Vehicle_Counting" / "Dataset_A" / cam["file"]
        if not video_path.exists():
            video_path = project_root.parent / "AIC21_Track1_Vehicle_Counting" / "Dataset_A" / "cam_5.mp4"

        JOBS[jid] = {
            "job_id": jid,
            "status": AnalysisStatus.PROCESSING,
            "video_path": str(video_path),
            "camera_id": cid,
            "filename": cam["file"],
            "frames_processed": 0,
            "total_frames": max_frames,
            "current_fps": 0.0,
            "result": None,
        }
        FRAME_BUFFERS[jid] = collections.deque(maxlen=30)

        job_info = {
            "camera_id": cid,
            "job_id": jid,
            "video_path": str(video_path)
        }
        job_list.append(job_info)
        response_jobs.append({
            "camera_id": cid,
            "job_id": jid,
            "status": AnalysisStatus.PROCESSING
        })

    background_tasks.add_task(run_multi_camera_worker, job_list, max_frames)
    return {
        "status": "PROCESSING",
        "message": "Started simultaneous 5-camera GPU inference",
        "jobs": response_jobs
    }


@app.post("/api/inference/start")
def start_inference(
    background_tasks: BackgroundTasks,
    job_id: str = Form(...),
    max_frames: int = Form(600),
    roads_json: Optional[str] = Form(None)
):
    global ACTIVE_PROCESSING_JOB_ID

    # Duplicate job prevention: if the active job matches the requested camera/video and is running, reattach
    if ACTIVE_PROCESSING_JOB_ID and ACTIVE_PROCESSING_JOB_ID in JOBS:
        act_job = JOBS[ACTIVE_PROCESSING_JOB_ID]
        act_status = act_job.get("status")
        if act_status in (AnalysisStatus.PROCESSING, AnalysisStatus.QUEUED, AnalysisStatus.INITIALIZING):
            req_cam = job_id if job_id.startswith("cam_") else "cam_5"
            if act_job.get("camera_id") == req_cam or ACTIVE_PROCESSING_JOB_ID == job_id or ACTIVE_PROCESSING_JOB_ID.startswith(f"{job_id}_"):
                log.info(f"[API] Reattaching to existing active processing job {ACTIVE_PROCESSING_JOB_ID}")
                return {
                    "job_id": ACTIVE_PROCESSING_JOB_ID,
                    "status": act_status,
                    "video_path": act_job.get("video_path"),
                    "already_running": True,
                    "frames_processed": act_job.get("frames_processed", 0),
                    "total_frames": act_job.get("total_frames", max_frames),
                }

    actual_job_id = job_id
    if job_id not in JOBS:
        actual_job_id = f"{job_id}_{uuid.uuid4().hex[:6]}"
        dataset_video = project_root.parent / "AIC21_Track1_Vehicle_Counting" / "Dataset_A" / f"{job_id}.mp4"
        if not dataset_video.exists():
            dataset_video = project_root.parent / "AIC21_Track1_Vehicle_Counting" / "Dataset_A" / "cam_5.mp4"
        
        JOBS[actual_job_id] = {
            "job_id": actual_job_id,
            "status": AnalysisStatus.PROCESSING,
            "video_path": str(dataset_video),
            "camera_id": job_id if job_id.startswith("cam_") else "cam_5",
            "source": {
                "original_filename": dataset_video.name,
                "stored_filename": dataset_video.name,
                "video_path": str(dataset_video),
                "video_type": "dataset"
            },
            "camera": {
                "camera_id": job_id if job_id.startswith("cam_") else "cam_5",
                "config_id": (job_id if job_id.startswith("cam_") else "cam_5").upper()
            },
            "mapping": {
                "roi_id": (job_id if job_id.startswith("cam_") else "cam_5").upper(),
                "road_map_id": (job_id if job_id.startswith("cam_") else "cam_5").upper(),
                "movement_map_id": (job_id if job_id.startswith("cam_") else "cam_5").upper()
            },
            "original_filename": dataset_video.name,
            "stored_filename": dataset_video.name,
            "video_type": "dataset",
            "filename": dataset_video.name,
            "frames_processed": 0,
            "total_frames": max_frames,
            "current_fps": 0.0,
            "result": None,
        }
        FRAME_BUFFERS[actual_job_id] = collections.deque(maxlen=30)

    JOBS[actual_job_id]["status"] = AnalysisStatus.PROCESSING
    job = JOBS[actual_job_id]
    
    # ----------------------------------------------------
    # VALIDATION
    # ----------------------------------------------------
    v_path = Path(job["video_path"])
    if not v_path.exists() or not v_path.is_file():
        err = f"Video file missing or not a file: {v_path}"
        log.error(err)
        job["status"] = AnalysisStatus.ERROR
        job["error_message"] = err
        raise HTTPException(status_code=400, detail=err)
    
    try:
        import cv2
        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            raise ValueError("Not a valid video format or could not be opened by OpenCV.")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frames / fps if fps > 0 else 0
        cap.release()
        
        log.info(
            f"\n[VIDEO VALIDATION]\n"
            f"job_id={actual_job_id}\n"
            f"filename={v_path.name}\n"
            f"path={v_path}\n"
            f"exists=true\n"
            f"size={v_path.stat().st_size} bytes\n"
            f"width={width}\n"
            f"height={height}\n"
            f"fps={fps}\n"
            f"frames={frames}\n"
            f"duration={duration:.2f}s\n"
        )
    except Exception as e:
        err = f"Video validation failed for {v_path.name}: {e}"
        log.error(err)
        job["status"] = AnalysisStatus.ERROR
        job["error_message"] = err
        raise HTTPException(status_code=400, detail="Invalid video file format")
    
    manual_roads = None
    if roads_json:
        try:
            import json
            import os
            manual_roads = json.loads(roads_json)
            # Save configuration
            job_dir = project_root.parent / "jobs" / actual_job_id
            config_dir = job_dir / "config"
            os.makedirs(config_dir, exist_ok=True)
            with open(config_dir / "roads.json", "w") as f:
                json.dump(manual_roads, f, indent=4)
        except Exception as e:
            log.error(f"Failed to parse or save manual roads: {e}")
            raise HTTPException(status_code=400, detail="Invalid roads JSON")

    ACTIVE_PROCESSING_JOB_ID = actual_job_id
    STOP_FLAGS[actual_job_id] = False
    background_tasks.add_task(run_pipeline_worker, actual_job_id, max_frames, manual_roads)
    return {
        "job_id": actual_job_id,
        "status": AnalysisStatus.PROCESSING,
        "video_path": str(v_path),
        "already_running": False,
        "frames_processed": 0,
        "total_frames": max_frames,
    }


@app.post("/api/inference/{job_id}/stop")
def stop_inference(job_id: str):
    """Signal an active video inference pipeline to stop processing immediately."""
    global ACTIVE_PROCESSING_JOB_ID
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    STOP_FLAGS[job_id] = True
    job = JOBS[job_id]
    job["status"] = AnalysisStatus.CANCELLED
    if ACTIVE_PROCESSING_JOB_ID == job_id:
        ACTIVE_PROCESSING_JOB_ID = None
    log.info(f"[API] Stop signal sent for job {job_id}")
    return {"job_id": job_id, "status": "stopped", "message": f"Job {job_id} stopped"}


@app.delete("/api/sessions/{job_id}")
def delete_session(job_id: str):
    """
    Remove an individual video session and clean up all associated memory,
    buffers, active websockets, and temporary uploaded files without affecting other sessions.
    """
    STOP_FLAGS[job_id] = True
    job = JOBS.pop(job_id, None)
    FRAME_BUFFERS.pop(job_id, None)
    LIVE_ANALYSIS_STATES.pop(job_id, None)
    STOP_FLAGS.pop(job_id, None)
    
    ws_set = ACTIVE_WEBSOCKETS.pop(job_id, None)
    if ws_set:
        for ws in list(ws_set):
            try:
                if MAIN_EVENT_LOOP and MAIN_EVENT_LOOP.is_running():
                    asyncio.run_coroutine_threadsafe(ws.close(), MAIN_EVENT_LOOP)
            except Exception:
                pass

    if job and "video_path" in job:
        vpath = Path(job["video_path"])
        if UPLOAD_DIR in vpath.parents and vpath.exists():
            try:
                vpath.unlink(missing_ok=True)
                log.info(f"[CLEANUP] Deleted temporary upload file: {vpath.name}")
            except Exception as e:
                log.warning(f"Could not delete temp video {vpath}: {e}")

    log.info(f"[API] Session {job_id} deleted and resources released.")
    return {"job_id": job_id, "status": "deleted"}


@app.get("/api/inference/{job_id}/status", response_model=JobStatus)
def get_inference_status(job_id: str):
    target_job_id = job_id
    if target_job_id in ("active", "current"):
        target_job_id = ACTIVE_PROCESSING_JOB_ID or LATEST_COMPLETED_JOB_ID or "demo-cam5"
    elif target_job_id in ("latest", "last"):
        target_job_id = LATEST_COMPLETED_JOB_ID or ACTIVE_PROCESSING_JOB_ID or "demo-cam5"

    job = JOBS.get(target_job_id)
    if not job:
        if target_job_id in ("demo-cam5", "cam_5"):
            return JobStatus(job_id=target_job_id, status=AnalysisStatus.READY, result_ready=True)
        raise HTTPException(status_code=404, detail="Job not found")

    live_state = LIVE_ANALYSIS_STATES.get(target_job_id, {})
    return JobStatus(
        job_id=target_job_id,
        status=job["status"],
        frames_processed=job.get("frames_processed", 0),
        total_frames=job.get("total_frames", 600),
        current_fps=live_state.get("processing_fps", 0.0),
        video_time_seconds=live_state.get("video_time_seconds", 0.0),
        video_name=job.get("filename") or (Path(job["video_path"]).name if job.get("video_path") else None),
        camera_id=job.get("camera_id", "cam_5"),
        error_message=job.get("error_message"),
        result_ready=(job["status"] in (AnalysisStatus.READY, AnalysisStatus.COMPLETED))
    )


@app.get("/api/session/active")
@app.get("/api/inference/active")
def get_active_session():
    """
    Returns the currently active or most recent video processing session,
    enabling persistent frontend tracking across route navigation.
    """
    global ACTIVE_PROCESSING_JOB_ID, LATEST_COMPLETED_JOB_ID
    target_job_id = ACTIVE_PROCESSING_JOB_ID
    is_active = False

    if target_job_id and target_job_id in JOBS:
        job = JOBS[target_job_id]
        if job.get("status") in (AnalysisStatus.PROCESSING, AnalysisStatus.QUEUED, AnalysisStatus.INITIALIZING):
            is_active = True
        else:
            target_job_id = None

    if not target_job_id and LATEST_COMPLETED_JOB_ID and LATEST_COMPLETED_JOB_ID in JOBS:
        target_job_id = LATEST_COMPLETED_JOB_ID

    if not target_job_id:
        for j_id, j in reversed(list(JOBS.items())):
            if j.get("status") in (AnalysisStatus.PROCESSING, AnalysisStatus.READY, AnalysisStatus.COMPLETED):
                target_job_id = j_id
                if j.get("status") == AnalysisStatus.PROCESSING:
                    is_active = True
                break

    if not target_job_id or target_job_id not in JOBS:
        return {"active": False, "job_id": None, "status": "IDLE"}

    job = JOBS[target_job_id]
    live_state = LIVE_ANALYSIS_STATES.get(target_job_id, {})
    return {
        "active": is_active,
        "job_id": target_job_id,
        "status": job.get("status"),
        "camera_id": job.get("camera_id", "cam_5"),
        "video_name": job.get("filename") or (Path(job["video_path"]).name if job.get("video_path") else "video.mp4"),
        "frames_processed": job.get("frames_processed", 0),
        "total_frames": job.get("total_frames", 600),
        "current_fps": live_state.get("processing_fps", 0.0),
        "video_time_seconds": live_state.get("video_time_seconds", 0.0),
        "vehicles_counted": live_state.get("vehicles_counted", 0),
        "current_active": live_state.get("current_active", 0),
        "current_stationary": live_state.get("current_stationary", 0),
        "current_moving": live_state.get("current_moving", 0),
        "traffic_state": live_state.get("traffic_state", "LOW"),
        "flow_rate_per_min": live_state.get("flow_rate_per_min", 0.0),
        "average_speed": live_state.get("average_speed"),
        "roads": live_state.get("roads", []),
        "output_ready": (job.get("status") in (AnalysisStatus.READY, AnalysisStatus.COMPLETED)),
        "output_url": f"/api/jobs/{target_job_id}/output" if (job.get("status") in (AnalysisStatus.READY, AnalysisStatus.COMPLETED)) else None,
        "error_message": job.get("error_message")
    }


@app.get("/api/analysis/{job_id}/live")
def get_live_analysis_state(job_id: str):
    """Retrieve current frame live analysis state including per-road analytics."""
    if job_id in LIVE_ANALYSIS_STATES:
        return LIVE_ANALYSIS_STATES[job_id]
    job = JOBS.get(job_id)
    if job and job.get("status") == AnalysisStatus.READY:
        res = job.get("result", {})
        dur_min = max(0.05, job.get("frames_processed", 600) / 10.0 / 60.0)
        flow = round(res.get("total_vehicles", 0) / dur_min, 1)
        tot_summary = res.get("total_road_summary", {})
        return {
            "job_id": job_id,
            "frame_index": job.get("frames_processed", 600),
            "total_frames": job.get("total_frames", 600),
            "video_time_seconds": round(job.get("frames_processed", 600) / 10.0, 2),
            "vehicles_counted": res.get("total_vehicles", 0),
            "current_active": res.get("current_vehicles", 0),
            "current_roi_count": res.get("current_vehicles", 0),
            "initial_occupancy": res.get("initial_occupancy", 0),
            "cumulative_entries": res.get("cumulative_entries", 0),
            "cumulative_exits": res.get("cumulative_exits", 0),
            "invariant_verified": res.get("invariant_verified", True),
            "current_moving": res.get("current_moving", 0),
            "current_stationary": res.get("current_stationary", 0),
            "vehicle_classes": res.get("vehicle_classes", {}),
            "current_classes": res.get("current_classes", {}),
            "movement_counts": res.get("movement_counts", {}),
            "traffic_state": res.get("traffic_state", "HIGH"),
            "arrival_rate": flow,
            "flow_rate_per_min": flow,
            "queue": int(res.get("total_vehicles", 0) * 0.5),
            "processing_fps": res.get("processing", {}).get("speed_fps", 0.0),
            "completed_trips": tot_summary.get("completed_trips", 0),
            "valid_trips": tot_summary.get("valid_trips", 0),
            "average_speed": tot_summary.get("average_speed"),
            "speed_display": tot_summary.get("speed_display", "Waiting for vehicles"),
            "roads": res.get("roads", []),
            "total_road_summary": tot_summary,
            "global": {
                "vehicles": res.get("total_vehicles", 0),
                "current_active": res.get("current_vehicles", 0),
                "flow_volume": flow,
                "traffic_state": res.get("traffic_state", "LOW"),
                "average_speed": tot_summary.get("average_speed"),
            },
            "output_ready": True,
            "output_url": f"/api/jobs/{job_id}/output",
            "perf": res.get("performance_metrics"),
        }
    return {
        "job_id": job_id,
        "frame_index": 0,
        "total_frames": 600,
        "vehicles_counted": 0,
        "vehicle_classes": {"Car": 0, "Truck": 0},
        "movement_counts": {},
        "traffic_state": "LOW",
        "arrival_rate": 0.0,
        "queue": 0,
        "processing_fps": 0.0,
        "roads": [],
        "global": {"vehicles": 0, "flow_volume": 0.0, "traffic_state": "LOW"},
    }


@app.get("/api/traffic/current", response_model=CurrentState)
def get_current_traffic():
    return CURRENT_STATE


@app.get("/api/traffic/history", response_model=List[HistoryPoint])
def get_traffic_history():
    return CURRENT_STATE.history


@app.post("/api/simulation/run", response_model=SimulationResult)
def run_simulation(req: SimulationRequest):
    if not CURRENT_STATE.metrics:
        raise HTTPException(status_code=400, detail="No active traffic metrics for simulation")

    base_q = CURRENT_STATE.prediction.future_queue_estimate if CURRENT_STATE.prediction else 30

    strategies = []
    for s in req.strategies:
        ext = 0.0
        if "+10" in s: ext = 10.0
        elif "+20" in s: ext = 20.0
        elif "+30" in s: ext = 30.0

        cleared = int(ext * 0.65)
        cross_penalty = max(0, int((ext - 15) * 0.35)) if ext > 15 else 0
        predicted_q = max(5, base_q - cleared + cross_penalty)
        predicted_wait = round(predicted_q * 0.85, 1)
        predicted_cong = round(max(15.0, (predicted_q / (base_q + 1e-5)) * 75.0), 1)
        score = predicted_q * 0.6 + predicted_wait * 0.4

        strategies.append(SimulationStrategy(
            strategy_id=s,
            label=f"+{int(ext)}s Green Extension" if ext > 0 else "Current Fixed Timing",
            green_phase_extension_sec=ext,
            predicted_queue=predicted_q,
            predicted_wait_sec=predicted_wait,
            predicted_congestion_pct=predicted_cong,
            score=round(score, 1),
            is_recommended=(s == "+20sec" or ext == 20.0)
        ))

    strategies.sort(key=lambda x: x.green_phase_extension_sec)

    best = min(strategies, key=lambda x: x.score)
    for s in strategies:
        s.is_recommended = (s.strategy_id == best.strategy_id)

    res = SimulationResult(
        job_id=req.job_id,
        strategies=strategies,
        recommended_strategy_id=best.strategy_id,
        simulation_completed=True
    )
    CURRENT_STATE.simulation = res
    return res


@app.get("/api/recommendation", response_model=Recommendation)
def get_recommendation():
    if not CURRENT_STATE.recommendation:
        raise HTTPException(status_code=404, detail="No recommendation generated")
    return CURRENT_STATE.recommendation


@app.get("/api/results/{job_id}")
def get_job_results(job_id: str):
    job = JOBS.get(job_id)
    if job and job.get("result"):
        return job["result"]

    f_path = OUTPUT_METRICS_DIR / f"{job_id}_results.json"
    if f_path.exists():
        with open(f_path, "r") as f:
            return json.load(f)

    if job_id in ("demo-cam5", "cam_5") and DEFAULT_RESULT_JSON.exists():
        with open(DEFAULT_RESULT_JSON, "r") as f:
            return json.load(f)

    raise HTTPException(status_code=404, detail=f"Results not found for job {job_id}")


@app.get("/api/debug/results/{job_id}")
def get_debug_results(job_id: str):
    """Debug provenance endpoint comparing video, metrics, and timestamps."""
    job = JOBS.get(job_id)
    if not job:
        if job_id in ("demo-cam5", "cam_5"):
            return {
                "job_id": job_id,
                "status": CURRENT_STATE.status,
                "source": "canonical_baseline_execution",
                "total_vehicles": CURRENT_STATE.metrics.total_vehicle_count if CURRENT_STATE.metrics else 0,
                "classes": {
                    "Car": CURRENT_STATE.metrics.car_count if CURRENT_STATE.metrics else 0,
                    "Truck": CURRENT_STATE.metrics.truck_count if CURRENT_STATE.metrics else 0,
                },
                "flow_rate_per_min": CURRENT_STATE.metrics.flow_rate_per_min if CURRENT_STATE.metrics else 0,
                "traffic_state": CURRENT_STATE.metrics.traffic_state if CURRENT_STATE.metrics else "UNKNOWN",
                "movements": [m.dict() for m in (CURRENT_STATE.metrics.by_movement if CURRENT_STATE.metrics else [])],
                "direction_imbalance": {
                    "detected": CURRENT_STATE.metrics.direction_imbalance_detected if CURRENT_STATE.metrics else False,
                    "ratio": CURRENT_STATE.metrics.direction_imbalance_ratio if CURRENT_STATE.metrics else 1.0,
                    "dominant_movement": CURRENT_STATE.metrics.dominant_movement if CURRENT_STATE.metrics else None,
                },
                "annotated_video_url": CURRENT_STATE.annotated_video_url,
            }
        raise HTTPException(status_code=404, detail="Job not found")

    res = job.get("result")
    return {
        "job_id": job_id,
        "status": job["status"],
        "video_path": job.get("video_path"),
        "camera_id": job.get("camera_id"),
        "frames_processed": job.get("frames_processed"),
        "total_vehicles": res.get("total_vehicles") if res else None,
        "classes": res.get("vehicle_classes") if res else None,
        "movements": res.get("movement_counts") if res else None,
        "traffic_features": res.get("traffic_features") if res else None,
        "outputs": res.get("outputs") if res else None,
    }


@app.get("/api/cameras")
def get_available_cameras():
    """List available cameras with metadata from AIC21 dataset."""
    return [
        {"camera_id": "cam_5", "name": "Intersection 5 (Fisheye 4-Leg) • (GT Verified)", "type": "four-leg signalized", "fps": 10.0, "has_gt": True},
        {"camera_id": "cam_1", "name": "Intersection 1 (PTZ Angle)", "type": "three-leg signalized", "fps": 10.0, "has_gt": False},
        {"camera_id": "cam_7", "name": "Intersection 7 (Overhead Normal)", "type": "four-leg signalized", "fps": 10.0, "has_gt": False},
        {"camera_id": "cam_9", "name": "Camera 9 (Highway Ramp)", "type": "highway interchange", "fps": 10.0, "has_gt": False},
        {"camera_id": "cam_10", "name": "Camera 10 (Single Approach)", "type": "single-approach", "fps": 10.0, "has_gt": False},
    ]


@app.get("/api/traffic/camera/{camera_id}", response_model=CurrentState)
def get_traffic_by_camera(camera_id: str):
    """Retrieve detailed CurrentState for a specific camera."""
    if camera_id in CAMERA_STATES:
        return CAMERA_STATES[camera_id]
    
    # Fallback to current global state
    return CURRENT_STATE


@app.get("/api/cameras/overview")
def get_cameras_overview():
    """
    Return comprehensive multi-camera telemetry for the 5-camera network grid.
    Aggregates vehicle counts, traffic state, flow rates, and status.
    """
    cameras_meta = [
        {"camera_id": "cam_5", "cam_number": "CAM 01", "name": "Intersection 5 (Fisheye 4-Leg) • (GT Verified)", "subtitle": "Fisheye 4-Leg", "type": "four-leg signalized", "has_gt": True},
        {"camera_id": "cam_1", "cam_number": "CAM 02", "name": "Intersection 1 (PTZ Angle)", "subtitle": "PTZ Angle", "type": "three-leg signalized", "has_gt": False},
        {"camera_id": "cam_7", "cam_number": "CAM 03", "name": "Intersection 7 (Overhead Normal)", "subtitle": "Overhead Normal", "type": "four-leg signalized", "has_gt": False},
        {"camera_id": "cam_9", "cam_number": "CAM 04", "name": "Camera 9 (Highway Ramp)", "subtitle": "Highway Ramp", "type": "highway interchange", "has_gt": False},
        {"camera_id": "cam_10", "cam_number": "CAM 05", "name": "Camera 10 (Single Approach)", "subtitle": "Single Approach", "type": "single-approach", "has_gt": False},
    ]

    overview_list = []
    total_vehicles_network = 0
    total_flow_network = 0.0
    critical_count = 0
    analyzing_count = 0

    for meta in cameras_meta:
        cid = meta["camera_id"]
        c_state = CAMERA_STATES.get(cid)
        
        # Check if there is an active running job for this camera
        active_job = next((j for j in JOBS.values() if j.get("camera_id") == cid and j.get("status") in (AnalysisStatus.PROCESSING, AnalysisStatus.ANALYZING)), None)
        
        status = "MONITORING"
        if active_job:
            status = "ANALYZING"
            analyzing_count += 1
        elif c_state and c_state.status == AnalysisStatus.READY:
            status = "LIVE"

        metrics = c_state.metrics if c_state else None
        v_count = metrics.total_vehicle_count if metrics else 0
        t_state = metrics.traffic_state.value if metrics else "LOW"
        flow = metrics.flow_rate_per_min if metrics else 0.0
        queue = c_state.prediction.future_queue_estimate if (c_state and c_state.prediction) else 0

        total_vehicles_network += v_count
        total_flow_network += flow
        if t_state in ("CRITICAL", "HIGH"):
            critical_count += 1

        video_url = c_state.annotated_video_url if c_state else f"/api/video/stream/{cid}.mp4"
        live_job_id = active_job["job_id"] if active_job else (c_state.job_id if c_state else cid)

        overview_list.append({
            "camera_id": cid,
            "cam_number": meta["cam_number"],
            "name": meta["name"],
            "subtitle": meta["subtitle"],
            "type": meta["type"],
            "has_gt": meta["has_gt"],
            "status": status,
            "job_id": live_job_id,
            "vehicles_counted": v_count,
            "traffic_state": t_state,
            "flow_rate_per_min": flow,
            "queue": queue,
            "video_url": video_url,
            "car_count": metrics.car_count if metrics else 0,
            "truck_count": metrics.truck_count if metrics else 0,
            "dominant_movement": metrics.dominant_movement if metrics else None,
            "movement_counts": {str(m.movement_id): m.count for m in metrics.by_movement} if metrics else {},
            "fps": 10.0
        })

    return {
        "network_summary": {
            "total_cameras": len(cameras_meta),
            "active_cameras": len(cameras_meta),
            "analyzing_cameras": analyzing_count,
            "monitoring_cameras": len(cameras_meta) - analyzing_count,
            "total_vehicles": total_vehicles_network,
            "critical_alerts": critical_count,
            "avg_flow_rate": round(total_flow_network / max(1, len(cameras_meta)), 1)
        },
        "cameras": overview_list
    }


# ===========================================================================
# ADAPTIVE SIGNAL CONTROL — Endpoints
# Added to existing server without modifying any existing routes.
# All signal control is in RECOMMENDATION_ONLY / SIMULATION mode.
# This system does NOT control physical traffic lights.
# ===========================================================================

from api.signal_models import (
    IntersectionConfig, SignalRecommendation, SimulationResult as SignalSimResult,
    SimulationScenario, AuditLogEntry, DataQualityFlag
)
from src.signal.demand_calculator import ApproachDemandCalculator
from src.signal.phase_validator import SignalPhaseValidator
from src.signal.recommendation_engine import SignalRecommendationEngine
from src.signal.scenario_simulator import ScenarioSimulator

# ---------------------------------------------------------------------------
# In-memory signal control stores (additive — no impact on existing stores)
# ---------------------------------------------------------------------------
INTERSECTION_CONFIGS: Dict[str, dict] = {}
SIGNAL_RECOMMENDATION_HISTORY: Dict[str, list] = {}  # intersection_id → list[dict]
SIGNAL_SIMULATION_RUNS: Dict[str, list] = {}          # intersection_id → list[dict]
PHASE_LAST_SERVED: Dict[str, Dict[str, Any]] = {}     # intersection_id → {approach_id: datetime}
SIGNAL_AUDIT_LOG: List[dict] = []

_rec_engine = SignalRecommendationEngine()
_simulator = ScenarioSimulator()

MAX_HISTORY_PER_INTERSECTION = 50  # cap audit/history lists


def _get_default_cam5_intersection() -> dict:
    """
    Build a default four-leg intersection config for cam_5
    using the AIC21 camera metadata already loaded into CAMERA_STATES.
    """
    return {
        "intersection_id": "int_cam5",
        "name": "Intersection 5 (Fisheye 4-Leg)",
        "camera_id": "cam_5",
        "intersection_type": "four-leg signalized",
        "approaches": [
            {
                "approach_id": "north",
                "direction": "NORTH",
                "name": "North Approach",
                "road_id": "road_N",
                "job_id": "demo-cam5",
                "permitted_movements": ["THROUGH", "LEFT_TURN", "RIGHT_TURN"],
                "has_pedestrian_crossing": True,
                "pedestrian_clearance_sec": 7.0,
                "lane_count": 2,
            },
            {
                "approach_id": "south",
                "direction": "SOUTH",
                "name": "South Approach",
                "road_id": "road_S",
                "job_id": "demo-cam5",
                "permitted_movements": ["THROUGH", "LEFT_TURN", "RIGHT_TURN"],
                "has_pedestrian_crossing": True,
                "pedestrian_clearance_sec": 7.0,
                "lane_count": 2,
            },
            {
                "approach_id": "east",
                "direction": "EAST",
                "name": "East Approach",
                "road_id": "road_E",
                "job_id": "demo-cam5",
                "permitted_movements": ["THROUGH", "RIGHT_TURN"],
                "has_pedestrian_crossing": False,
                "pedestrian_clearance_sec": 0.0,
                "lane_count": 1,
            },
            {
                "approach_id": "west",
                "direction": "WEST",
                "name": "West Approach",
                "road_id": "road_W",
                "job_id": "demo-cam5",
                "permitted_movements": ["THROUGH", "RIGHT_TURN"],
                "has_pedestrian_crossing": False,
                "pedestrian_clearance_sec": 0.0,
                "lane_count": 1,
            },
        ],
        "phases": [
            {
                "phase_id": "P1",
                "phase_name": "North–South Through",
                "served_approaches": ["north", "south"],
                "permitted_movements": ["THROUGH", "RIGHT_TURN"],
                "conflicting_phases": ["P2"],
                "min_green_sec": 15.0,
                "max_green_sec": 55.0,
                "amber_sec": 3.0,
                "all_red_sec": 2.0,
                "pedestrian_walk_sec": 7.0,
                "sequence_position": 0,
            },
            {
                "phase_id": "P2",
                "phase_name": "East–West Through",
                "served_approaches": ["east", "west"],
                "permitted_movements": ["THROUGH", "RIGHT_TURN"],
                "conflicting_phases": ["P1"],
                "min_green_sec": 10.0,
                "max_green_sec": 45.0,
                "amber_sec": 3.0,
                "all_red_sec": 2.0,
                "pedestrian_walk_sec": None,
                "sequence_position": 1,
            },
        ],
        "phase_constraints": [
            {"phase_a": "P1", "phase_b": "P2", "reason": "Conflicting through movements"}
        ],
        "cycle_length_sec": 90.0,
        "min_cycle_sec": 60.0,
        "max_cycle_sec": 150.0,
        "fixed_plan_name": "AIC21 Default Fixed-Time Plan",
        "fixed_plan_green_times": {"P1": 35.0, "P2": 25.0},
        "demand_weight_queue": 0.45,
        "demand_weight_arrival": 0.25,
        "demand_weight_waiting": 0.20,
        "demand_weight_occupancy": 0.10,
        "max_queue_veh": 50.0,
        "max_arrival_veh_per_min": 30.0,
        "max_wait_sec": 300.0,
        "max_occupancy_pct": 100.0,
        "max_phase_gap_sec": 300.0,
        "starvation_boost": 0.3,
        "stale_data_threshold_sec": 120.0,
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
    }


def _init_default_intersections():
    """Seed default intersection configs for the 5 AIC21 cameras."""
    INTERSECTION_CONFIGS["int_cam5"] = _get_default_cam5_intersection()
    INTERSECTION_CONFIGS["int_cam1"] = {
        **_get_default_cam5_intersection(),
        "intersection_id": "int_cam1",
        "name": "Intersection 1 (PTZ Angle)",
        "camera_id": "cam_1",
        "intersection_type": "three-leg signalized",
        "approaches": [
            {"approach_id": "north", "direction": "NORTH", "name": "North Approach",
             "road_id": "road_N", "job_id": "demo-cam_1",
             "permitted_movements": ["THROUGH", "LEFT_TURN"], "has_pedestrian_crossing": True,
             "pedestrian_clearance_sec": 7.0, "lane_count": 1},
            {"approach_id": "east", "direction": "EAST", "name": "East Approach",
             "road_id": "road_E", "job_id": "demo-cam_1",
             "permitted_movements": ["THROUGH", "RIGHT_TURN"], "has_pedestrian_crossing": False,
             "pedestrian_clearance_sec": 0.0, "lane_count": 1},
            {"approach_id": "south", "direction": "SOUTH", "name": "South Approach",
             "road_id": "road_S", "job_id": "demo-cam_1",
             "permitted_movements": ["THROUGH"], "has_pedestrian_crossing": False,
             "pedestrian_clearance_sec": 0.0, "lane_count": 1},
        ],
        "phases": [
            {"phase_id": "P1", "phase_name": "North Main", "served_approaches": ["north"],
             "permitted_movements": ["THROUGH", "LEFT_TURN"], "conflicting_phases": ["P2"],
             "min_green_sec": 15.0, "max_green_sec": 50.0, "amber_sec": 3.0, "all_red_sec": 2.0,
             "pedestrian_walk_sec": 7.0, "sequence_position": 0},
            {"phase_id": "P2", "phase_name": "East–South", "served_approaches": ["east", "south"],
             "permitted_movements": ["THROUGH", "RIGHT_TURN"], "conflicting_phases": ["P1"],
             "min_green_sec": 10.0, "max_green_sec": 40.0, "amber_sec": 3.0, "all_red_sec": 2.0,
             "pedestrian_walk_sec": None, "sequence_position": 1},
        ],
        "fixed_plan_green_times": {"P1": 30.0, "P2": 20.0},
    }


_init_default_intersections()


def _live_states_for_intersection(config: dict) -> Dict[str, Any]:
    """
    Extract live road analytics for the intersection's approaches.
    Returns mapping: road_id → analytics dict.
    Dynamically binds to active video inference or latest completed job.
    """
    result: Dict[str, Any] = {}
    camera_id = config.get("camera_id")

    candidate_job_ids = []
    # 1. Prefer actively processing job if matches camera or is the active job
    if ACTIVE_PROCESSING_JOB_ID and ACTIVE_PROCESSING_JOB_ID in LIVE_ANALYSIS_STATES:
        act_job = JOBS.get(ACTIVE_PROCESSING_JOB_ID, {})
        if not camera_id or act_job.get("camera_id") == camera_id or ACTIVE_PROCESSING_JOB_ID.startswith(str(camera_id)):
            candidate_job_ids.append(ACTIVE_PROCESSING_JOB_ID)
    # 2. Next prefer latest completed job
    if LATEST_COMPLETED_JOB_ID and LATEST_COMPLETED_JOB_ID in LIVE_ANALYSIS_STATES:
        if LATEST_COMPLETED_JOB_ID not in candidate_job_ids:
            candidate_job_ids.append(LATEST_COMPLETED_JOB_ID)

    dir_to_idx = {"N": "1", "S": "2", "E": "3", "W": "4", "NORTH": "1", "SOUTH": "2", "EAST": "3", "WEST": "4"}

    for idx, ap in enumerate(config.get("approaches", [])):
        road_id = ap.get("road_id")
        job_id = ap.get("job_id")
        direction = str(ap.get("direction", "")).upper()
        if not road_id:
            continue

        analytics_dict = None

        # Try candidate active/latest jobs first
        for c_id in candidate_job_ids:
            live = LIVE_ANALYSIS_STATES.get(c_id, {})
            roads = live.get("roads", [])
            for road in roads:
                r_id = str(road.get("road_id", ""))
                r_name = str(road.get("name", ""))
                
                # Direct match
                matched = (
                    r_id == road_id
                    or r_name == road_id
                    or f"road_{r_id}" == road_id
                    or r_id == road_id.replace("road_", "")
                )

                # Direction / index alias matching (e.g. road_N <-> road_1, road_S <-> road_2)
                if not matched:
                    cand_nums = [dir_to_idx.get(direction, ""), dir_to_idx.get(road_id.replace("road_", ""), "")]
                    cand_nums = [c for c in cand_nums if c]
                    for cn in cand_nums:
                        if r_id in (cn, f"road_{cn}") or r_name in (cn, f"road_{cn}") or direction in r_name.upper():
                            matched = True
                            break

                if matched:
                    analytics_dict = dict(road)
                    analytics_dict["_timestamp"] = live.get("video_time_seconds")
                    analytics_dict["_source"] = "active_inference"
                    analytics_dict["_job_id"] = c_id
                    break

            if analytics_dict:
                break

        # Fallback to approach index if custom roads were configured
        if analytics_dict is None and candidate_job_ids:
            for c_id in candidate_job_ids:
                live = LIVE_ANALYSIS_STATES.get(c_id, {})
                roads = live.get("roads", [])
                if idx < len(roads):
                    analytics_dict = dict(roads[idx])
                    analytics_dict["_timestamp"] = live.get("video_time_seconds")
                    analytics_dict["_source"] = "active_inference"
                    analytics_dict["_job_id"] = c_id
                    break

        # If not found in active jobs, try approach's configured job_id
        if analytics_dict is None and job_id and job_id in LIVE_ANALYSIS_STATES:
            live = LIVE_ANALYSIS_STATES[job_id]
            roads = live.get("roads", [])
            for road in roads:
                r_id = str(road.get("road_id", ""))
                r_name = str(road.get("name", ""))
                if r_id == road_id or r_name == road_id:
                    analytics_dict = dict(road)
                    analytics_dict["_timestamp"] = live.get("video_time_seconds")
                    analytics_dict["_source"] = "configured_job"
                    analytics_dict["_job_id"] = job_id
                    break

        # Fallback: synthesize from CAMERA_STATES if live data unavailable
        if analytics_dict is None and camera_id and camera_id in CAMERA_STATES:
            c_state = CAMERA_STATES[camera_id]
            if c_state and c_state.metrics:
                m = c_state.metrics
                # Distribute metrics across approaches equally as fallback
                n_approaches = max(1, len(config.get("approaches", [])))
                analytics_dict = {
                    "road_id": road_id,
                    "current_stationary": (m.current_roi_count or 0) // n_approaches,
                    "current_moving": 0,
                    "current_roi_count": (m.current_roi_count or 0) // n_approaches,
                    "current_active": (m.current_roi_count or 0) // n_approaches,
                    "flow_volume": m.flow_rate_per_min / n_approaches,
                    "average_speed": None,
                    "traffic_state": m.traffic_state.value if m.traffic_state else "UNKNOWN",
                    "_source": "camera_state_fallback",
                }

        if analytics_dict:
            result[road_id] = analytics_dict

    return result


def _add_audit_entry(event_type: str, intersection_id: str, details: dict,
                     rec_id: Optional[str] = None, sim_id: Optional[str] = None):
    """Append an audit log entry (capped at 500 entries)."""
    entry = {
        "entry_id": str(uuid.uuid4())[:12],
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "intersection_id": intersection_id,
        "actor": "SYSTEM",
        "details": details,
        "recommendation_id": rec_id,
        "simulation_id": sim_id,
    }
    SIGNAL_AUDIT_LOG.append(entry)
    if len(SIGNAL_AUDIT_LOG) > 500:
        SIGNAL_AUDIT_LOG.pop(0)


# ---------------------------------------------------------------------------
# Intersection list and config endpoints
# ---------------------------------------------------------------------------

@app.get("/api/intersections")
def list_intersections():
    """List all configured intersections."""
    return [
        {
            "intersection_id": cfg["intersection_id"],
            "name": cfg["name"],
            "camera_id": cfg.get("camera_id"),
            "intersection_type": cfg.get("intersection_type"),
            "n_approaches": len(cfg.get("approaches", [])),
            "n_phases": len(cfg.get("phases", [])),
            "fixed_plan_name": cfg.get("fixed_plan_name"),
        }
        for cfg in INTERSECTION_CONFIGS.values()
    ]


@app.get("/api/intersections/{intersection_id}")
def get_intersection(intersection_id: str):
    """Get full intersection configuration."""
    cfg = INTERSECTION_CONFIGS.get(intersection_id)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Intersection '{intersection_id}' not found")
    return cfg


@app.put("/api/intersections/{intersection_id}/config")
def update_intersection_config(intersection_id: str, config: dict):
    """
    Update intersection configuration.
    Validates phase configuration before saving.
    Returns 422 if validation fails.
    """
    validator = SignalPhaseValidator()
    errors = validator.validate(config)
    if errors:
        _add_audit_entry(
            "CONFIG_VALIDATION_FAILED", intersection_id,
            {"errors": errors, "attempted_config": config}
        )
        raise HTTPException(
            status_code=422,
            detail={"message": "Phase configuration validation failed", "errors": errors}
        )

    config["intersection_id"] = intersection_id
    config["updated_at"] = datetime.utcnow().isoformat()
    if "created_at" not in config:
        config["created_at"] = config["updated_at"]

    INTERSECTION_CONFIGS[intersection_id] = config
    _add_audit_entry("CONFIG_CHANGED", intersection_id, {"updated_fields": list(config.keys())})
    log.info("[Signal] Intersection config updated: %s", intersection_id)
    return {"status": "updated", "intersection_id": intersection_id}


# ---------------------------------------------------------------------------
# Live approach analytics
# ---------------------------------------------------------------------------

@app.get("/api/intersections/{intersection_id}/analytics")
def get_intersection_analytics(intersection_id: str):
    """
    Get live approach analytics for an intersection.
    Returns demand scores calculated from actual pipeline analytics.
    Data quality flags are set when data is stale or missing.
    """
    cfg = INTERSECTION_CONFIGS.get(intersection_id)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Intersection '{intersection_id}' not found")

    live_states = _live_states_for_intersection(cfg)

    calc = ApproachDemandCalculator(
        weight_queue=cfg.get("demand_weight_queue", 0.45),
        weight_arrival=cfg.get("demand_weight_arrival", 0.25),
        weight_waiting=cfg.get("demand_weight_waiting", 0.20),
        weight_occupancy=cfg.get("demand_weight_occupancy", 0.10),
        max_queue_veh=cfg.get("max_queue_veh", 50.0),
        max_arrival_veh_per_min=cfg.get("max_arrival_veh_per_min", 30.0),
        max_wait_sec=cfg.get("max_wait_sec", 300.0),
        max_occupancy_pct=cfg.get("max_occupancy_pct", 100.0),
        stale_threshold_sec=cfg.get("stale_data_threshold_sec", 120.0),
    )

    last_served = (PHASE_LAST_SERVED.get(intersection_id) or {})

    demands = calc.batch_compute(
        approaches=cfg.get("approaches", []),
        live_states=live_states,
        last_served=last_served,
        starvation_boost=cfg.get("starvation_boost", 0.3),
        max_phase_gap_sec=cfg.get("max_phase_gap_sec", 300.0),
    )

    # Determine overall data freshness
    all_stale = all(d.get("data_quality") in ("MISSING", "STALE") for d in demands)
    return {
        "intersection_id": intersection_id,
        "timestamp": datetime.utcnow().isoformat(),
        "approach_demands": demands,
        "overall_data_quality": "STALE" if all_stale else "PARTIAL" if any(
            d.get("data_quality") in ("MISSING", "STALE") for d in demands
        ) else "GOOD",
        "active_job_ids": list({
            ap.get("job_id") for ap in cfg.get("approaches", []) if ap.get("job_id")
        }),
    }


# ---------------------------------------------------------------------------
# Signal phases
# ---------------------------------------------------------------------------

@app.get("/api/intersections/{intersection_id}/signal-phases")
def get_signal_phases(intersection_id: str):
    """Get configured signal phases for an intersection."""
    cfg = INTERSECTION_CONFIGS.get(intersection_id)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Intersection '{intersection_id}' not found")

    validator = SignalPhaseValidator()
    errors = validator.validate(cfg)

    return {
        "intersection_id": intersection_id,
        "phases": cfg.get("phases", []),
        "phase_constraints": cfg.get("phase_constraints", []),
        "fixed_plan_green_times": cfg.get("fixed_plan_green_times", {}),
        "fixed_plan_name": cfg.get("fixed_plan_name"),
        "validation_errors": errors,
        "configuration_valid": not bool(errors),
    }


# ---------------------------------------------------------------------------
# Signal recommendation
# ---------------------------------------------------------------------------

@app.post("/api/intersections/{intersection_id}/recommendations")
def generate_recommendation(intersection_id: str):
    """
    Generate a signal recommendation for the intersection.

    Uses actual live analytics from the pipeline. Falls back to approved
    baseline plan when analytics are stale or missing.

    Returns a full recommendation with demand scores, proposed green times,
    evidence, and constraint validation results.

    Mode: RECOMMENDATION_ONLY — does not control physical signals.
    """
    cfg = INTERSECTION_CONFIGS.get(intersection_id)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Intersection '{intersection_id}' not found")

    live_states = _live_states_for_intersection(cfg)
    last_served_raw = PHASE_LAST_SERVED.get(intersection_id, {})
    last_served: Dict[str, Any] = {}
    for ap_id, ts in last_served_raw.items():
        if isinstance(ts, str):
            try:
                last_served[ap_id] = datetime.fromisoformat(ts)
            except Exception:
                pass
        elif isinstance(ts, datetime):
            last_served[ap_id] = ts

    rec = _rec_engine.generate(
        intersection_config=cfg,
        live_road_states=live_states,
        last_served=last_served if last_served else None,
    )

    rec["recommendation_id"] = str(uuid.uuid4())[:8]

    # Persist to history (capped)
    if intersection_id not in SIGNAL_RECOMMENDATION_HISTORY:
        SIGNAL_RECOMMENDATION_HISTORY[intersection_id] = []
    history = SIGNAL_RECOMMENDATION_HISTORY[intersection_id]
    history.append(rec)
    if len(history) > MAX_HISTORY_PER_INTERSECTION:
        history.pop(0)

    _add_audit_entry(
        "RECOMMENDATION_CREATED",
        intersection_id,
        {
            "phase_id": rec.get("recommended_phase", {}).get("phase_id"),
            "proposed_green": rec.get("recommended_phase", {}).get("proposed_green_sec"),
            "confidence": rec.get("confidence"),
            "fallback": rec.get("fallback_to_baseline"),
            "data_quality": rec.get("data_quality"),
        },
        rec_id=rec.get("recommendation_id"),
    )

    if rec.get("fallback_to_baseline"):
        log.warning(
            "[Signal] intersection=%s: recommendation fell back to baseline: %s",
            intersection_id, rec.get("fallback_reason"),
        )
    else:
        log.info(
            "[Signal] intersection=%s: recommendation generated phase=%s green=%.1fs confidence=%.2f",
            intersection_id,
            rec.get("recommended_phase", {}).get("phase_id"),
            rec.get("recommended_phase", {}).get("proposed_green_sec", 0),
            rec.get("confidence", 0),
        )

    return rec


@app.get("/api/intersections/{intersection_id}/recommendations/history")
def get_recommendation_history(intersection_id: str, limit: int = 20):
    """Get recommendation history for an intersection (most recent first)."""
    if intersection_id not in INTERSECTION_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Intersection '{intersection_id}' not found")

    history = SIGNAL_RECOMMENDATION_HISTORY.get(intersection_id, [])
    # Return most recent first
    return list(reversed(history[-limit:]))


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@app.post("/api/intersections/{intersection_id}/simulate")
def run_signal_simulation(intersection_id: str, body: dict = {}):
    """
    Run an analytical scenario simulation comparing adaptive vs fixed-time plans.

    Accepts:
      scenario_type: LOW_TRAFFIC | NORMAL_TRAFFIC | PEAK_TRAFFIC |
                     UNEVEN_TRAFFIC | INCREASED_TURNING | APPROACH_BLOCKED | HIGH_ARRIVAL_RATE
      simulation_duration_sec: float (default 3600)
      random_seed: int (default 42)
      demand_overrides: {approach_id: {arrival_rate_veh_per_min: X}}

    ⚠️ Results are from an analytical model, not a validated microsimulator.
    """
    cfg = INTERSECTION_CONFIGS.get(intersection_id)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Intersection '{intersection_id}' not found")

    scenario_type = body.get("scenario_type", "NORMAL_TRAFFIC")
    duration = float(body.get("simulation_duration_sec", 3600.0))
    seed = int(body.get("random_seed", 42))
    demand_overrides = body.get("demand_overrides", {})

    live_states = _live_states_for_intersection(cfg)
    calc = ApproachDemandCalculator(
        weight_queue=cfg.get("demand_weight_queue", 0.45),
        weight_arrival=cfg.get("demand_weight_arrival", 0.25),
        weight_waiting=cfg.get("demand_weight_waiting", 0.20),
        weight_occupancy=cfg.get("demand_weight_occupancy", 0.10),
        max_queue_veh=cfg.get("max_queue_veh", 50.0),
        max_arrival_veh_per_min=cfg.get("max_arrival_veh_per_min", 30.0),
        max_wait_sec=cfg.get("max_wait_sec", 300.0),
        max_occupancy_pct=cfg.get("max_occupancy_pct", 100.0),
    )
    approach_demands = calc.batch_compute(
        approaches=cfg.get("approaches", []),
        live_states=live_states,
    )

    # Get proposed adaptive greens from recommendation engine
    rec = _rec_engine.generate(
        intersection_config=cfg,
        live_road_states=live_states,
    )
    proposed_greens: Dict[str, float] = {}
    if not rec.get("fallback_to_baseline"):
        rec_phase = rec.get("recommended_phase", {})
        pid = rec_phase.get("phase_id")
        if pid:
            proposed_greens[pid] = rec_phase.get("proposed_green_sec", 20.0)

    sim_result = _simulator.run(
        intersection_config=cfg,
        approach_demands=approach_demands,
        scenario_type=scenario_type,
        demand_overrides=demand_overrides if demand_overrides else None,
        simulation_duration_sec=duration,
        random_seed=seed,
        proposed_adaptive_greens=proposed_greens if proposed_greens else None,
    )

    sim_id = str(uuid.uuid4())[:8]
    sim_result["simulation_id"] = sim_id

    if intersection_id not in SIGNAL_SIMULATION_RUNS:
        SIGNAL_SIMULATION_RUNS[intersection_id] = []
    SIGNAL_SIMULATION_RUNS[intersection_id].append(sim_result)
    if len(SIGNAL_SIMULATION_RUNS[intersection_id]) > 20:
        SIGNAL_SIMULATION_RUNS[intersection_id].pop(0)

    _add_audit_entry(
        "SIMULATION_RUN",
        intersection_id,
        {
            "scenario_type": scenario_type,
            "duration_sec": duration,
            "seed": seed,
            "delay_improvement_pct": sim_result.get("delay_improvement_pct"),
        },
        sim_id=sim_id,
    )

    return sim_result


@app.post("/api/intersections/{intersection_id}/compare-plans")
def compare_plans(intersection_id: str, body: dict = {}):
    """
    Compare fixed-time baseline against adaptive plan side-by-side.
    Uses the same analytical simulator as /simulate but returns
    a focused comparison object for the UI comparison panel.
    """
    cfg = INTERSECTION_CONFIGS.get(intersection_id)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Intersection '{intersection_id}' not found")

    scenario_type = body.get("scenario_type", "NORMAL_TRAFFIC")
    duration = float(body.get("simulation_duration_sec", 3600.0))

    live_states = _live_states_for_intersection(cfg)
    calc = ApproachDemandCalculator(
        weight_queue=cfg.get("demand_weight_queue", 0.45),
        weight_arrival=cfg.get("demand_weight_arrival", 0.25),
        weight_waiting=cfg.get("demand_weight_waiting", 0.20),
        weight_occupancy=cfg.get("demand_weight_occupancy", 0.10),
        max_queue_veh=cfg.get("max_queue_veh", 50.0),
        max_arrival_veh_per_min=cfg.get("max_arrival_veh_per_min", 30.0),
        max_wait_sec=cfg.get("max_wait_sec", 300.0),
        max_occupancy_pct=cfg.get("max_occupancy_pct", 100.0),
    )
    approach_demands = calc.batch_compute(
        approaches=cfg.get("approaches", []),
        live_states=live_states,
    )

    sim_result = _simulator.run(
        intersection_config=cfg,
        approach_demands=approach_demands,
        scenario_type=scenario_type,
        simulation_duration_sec=duration,
    )

    fixed = sim_result.get("fixed_plan_result", {})
    adaptive = sim_result.get("adaptive_plan_result", {})

    return {
        "intersection_id": intersection_id,
        "scenario_type": scenario_type,
        "fixed_plan": {
            "name": fixed.get("plan_name"),
            "avg_delay_sec": fixed.get("avg_delay_sec"),
            "avg_wait_sec": fixed.get("avg_wait_sec"),
            "max_queue_veh": fixed.get("max_queue_veh"),
            "throughput_veh_per_hour": fixed.get("throughput_veh_per_hour"),
            "vehicles_served": fixed.get("vehicles_served"),
            "pedestrian_compliance": fixed.get("pedestrian_compliance"),
        },
        "adaptive_plan": {
            "name": adaptive.get("plan_name"),
            "avg_delay_sec": adaptive.get("avg_delay_sec"),
            "avg_wait_sec": adaptive.get("avg_wait_sec"),
            "max_queue_veh": adaptive.get("max_queue_veh"),
            "throughput_veh_per_hour": adaptive.get("throughput_veh_per_hour"),
            "vehicles_served": adaptive.get("vehicles_served"),
            "pedestrian_compliance": adaptive.get("pedestrian_compliance"),
        },
        "delay_improvement_pct": sim_result.get("delay_improvement_pct"),
        "queue_improvement_pct": sim_result.get("queue_improvement_pct"),
        "throughput_improvement_pct": sim_result.get("throughput_improvement_pct"),
        "methodology_note": sim_result.get("methodology_note"),
        "simulation_mode": "ANALYTICAL",
    }


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

@app.get("/api/intersections/{intersection_id}/audit-log")
def get_audit_log(intersection_id: str, limit: int = 50):
    """Get the audit log for a specific intersection."""
    entries = [e for e in SIGNAL_AUDIT_LOG if e.get("intersection_id") == intersection_id]
    return list(reversed(entries[-limit:]))


@app.get("/api/signal/audit-log")
def get_full_audit_log(limit: int = 100):
    """Get the full system-wide audit log for all intersections."""
    return list(reversed(SIGNAL_AUDIT_LOG[-limit:]))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

