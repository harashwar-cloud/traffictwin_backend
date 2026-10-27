"""
api/models.py
--------------
Pydantic models for the TrafficTwin FastAPI backend.
These define the contract between ML pipeline and frontend.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel


class TrafficState(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class AnalysisStatus(str, Enum):
    IDLE = "IDLE"
    QUEUED = "QUEUED"
    INITIALIZING = "INITIALIZING"
    UPLOADING = "UPLOADING"
    PROCESSING = "PROCESSING"
    ANALYZING = "ANALYZING"
    SIMULATING = "SIMULATING"
    PAUSED = "PAUSED"
    READY = "READY"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


# ---------------------------------------------------------------------------
# Traffic metrics
# ---------------------------------------------------------------------------

class MovementCount(BaseModel):
    movement_id: int
    movement_description: str
    count: int
    car_count: int
    truck_count: int
    flow_rate_per_min: float
    state: TrafficState


class TrafficMetrics(BaseModel):
    traffic_state: TrafficState
    total_vehicle_count: int
    car_count: int
    truck_count: int
    flow_rate_per_min: float
    timestamp_sec: float
    frames_processed: int
    direction_imbalance_detected: bool
    direction_imbalance_ratio: float
    dominant_movement: Optional[int] = None
    by_movement: List[MovementCount] = []
    current_roi_count: Optional[int] = None
    initial_occupancy: Optional[int] = None
    cumulative_entries: Optional[int] = None
    cumulative_exits: Optional[int] = None
    invariant_verified: Optional[bool] = None


class TrafficPrediction(BaseModel):
    future_queue_estimate: int
    future_congestion_pct: float
    time_horizon_sec: float
    confidence: float


class HistoryPoint(BaseModel):
    timestamp_sec: float
    frame_id: int
    total_count: int
    flow_rate_per_min: float
    state: TrafficState


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class SimulationStrategy(BaseModel):
    strategy_id: str              # "current", "+10sec", "+20sec", "+30sec"
    label: str                    # Human-readable label
    green_phase_extension_sec: float
    predicted_queue: int
    predicted_wait_sec: float
    predicted_congestion_pct: float
    score: float                  # Lower is better
    is_recommended: bool


class SimulationRequest(BaseModel):
    job_id: str
    strategies: List[str] = ["current", "+10sec", "+20sec", "+30sec"]


class SimulationResult(BaseModel):
    job_id: str
    strategies: List[SimulationStrategy]
    recommended_strategy_id: str
    simulation_completed: bool


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

class RecommendationFactor(BaseModel):
    factor: str
    value: str
    direction: str    # "increasing", "decreasing", "stable"
    weight: str       # "high", "medium", "low"


class Recommendation(BaseModel):
    action: str
    strategy_id: str
    green_phase_extension_sec: float
    reasoning: List[RecommendationFactor]
    predicted_queue_change: int
    predicted_queue_change_pct: float
    predicted_wait_change_sec: float
    predicted_congestion_change_pct: float
    confidence: float


# ---------------------------------------------------------------------------
# Job / status
# ---------------------------------------------------------------------------

class JobStatus(BaseModel):
    job_id: str
    status: AnalysisStatus
    progress_pct: Optional[float] = None    # None = indeterminate
    frames_processed: int = 0
    total_frames: int = 0
    current_fps: float = 0.0
    video_time_seconds: Optional[float] = None
    video_name: Optional[str] = None
    camera_id: Optional[str] = None
    error_message: Optional[str] = None
    result_ready: bool = False


class VideoUploadResponse(BaseModel):
    job_id: str
    filename: str
    file_size_bytes: int
    camera_id: str
    status: AnalysisStatus


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    gpu_available: bool
    device: Optional[str] = "cpu"
    gpu_name: Optional[str] = None
    cuda_version: Optional[str] = None
    precision: Optional[str] = "FP32"
    model_loaded: bool
    active_jobs: int


# ---------------------------------------------------------------------------
# Full current state (what the frontend polls)
# ---------------------------------------------------------------------------

class CurrentState(BaseModel):
    job_id: Optional[str] = None
    status: AnalysisStatus = AnalysisStatus.IDLE
    metrics: Optional[TrafficMetrics] = None
    prediction: Optional[TrafficPrediction] = None
    simulation: Optional[SimulationResult] = None
    recommendation: Optional[Recommendation] = None
    history: List[HistoryPoint] = []
    camera_id: Optional[str] = None
    video_name: Optional[str] = None
    annotated_video_url: Optional[str] = None
