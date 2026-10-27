"""
api/signal_models.py
---------------------
Pydantic models for the Adaptive Signal Control module.

These models define the contract between:
  - The signal recommendation engine (backend)
  - The Adaptive Signal Control page (frontend)

Scope: recommendation-only / decision-support.
This system does NOT control physical traffic signals.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MovementDirection(str, Enum):
    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"
    NORTHEAST = "NORTHEAST"
    NORTHWEST = "NORTHWEST"
    SOUTHEAST = "SOUTHEAST"
    SOUTHWEST = "SOUTHWEST"
    UNKNOWN = "UNKNOWN"


class VehicleMovementType(str, Enum):
    THROUGH = "THROUGH"
    LEFT_TURN = "LEFT_TURN"
    RIGHT_TURN = "RIGHT_TURN"
    U_TURN = "U_TURN"


class DataQualityFlag(str, Enum):
    GOOD = "GOOD"          # Fresh, complete data
    STALE = "STALE"        # Data older than configured staleness threshold
    MISSING = "MISSING"    # No data received for this approach
    PARTIAL = "PARTIAL"    # Some metrics missing but enough for estimation


class RecommendationStatus(str, Enum):
    WAITING_FOR_DATA = "WAITING_FOR_DATA"
    READY = "READY"
    PENDING_OFFICER_REVIEW = "PENDING_OFFICER_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    SUPERSEDED = "SUPERSEDED"
    INVALIDATED = "INVALIDATED"
    GENERATED = "GENERATED"
    REVIEWED = "REVIEWED"
    OFFICER_REVIEWED = "OFFICER_REVIEWED"


class SimulationScenarioType(str, Enum):
    LOW_TRAFFIC = "LOW_TRAFFIC"
    NORMAL_TRAFFIC = "NORMAL_TRAFFIC"
    PEAK_TRAFFIC = "PEAK_TRAFFIC"
    UNEVEN_TRAFFIC = "UNEVEN_TRAFFIC"
    INCREASED_TURNING = "INCREASED_TURNING"
    APPROACH_BLOCKED = "APPROACH_BLOCKED"
    HIGH_ARRIVAL_RATE = "HIGH_ARRIVAL_RATE"


# ---------------------------------------------------------------------------
# Approach configuration
# ---------------------------------------------------------------------------

class ApproachConfig(BaseModel):
    """Configuration for a single road approach at an intersection."""
    approach_id: str = Field(..., description="Unique identifier for this approach")
    direction: MovementDirection = Field(..., description="Compass direction of the approach")
    name: str = Field(..., description="Human-readable name, e.g. 'North Approach'")
    road_id: Optional[str] = Field(None, description="Road ROI ID from the existing analytics pipeline")
    job_id: Optional[str] = Field(None, description="Inference job ID providing analytics for this approach")
    permitted_movements: List[VehicleMovementType] = Field(
        default_factory=lambda: [VehicleMovementType.THROUGH],
        description="Permitted vehicle movements on this approach"
    )
    has_pedestrian_crossing: bool = Field(False, description="Whether pedestrians cross on this approach")
    pedestrian_clearance_sec: float = Field(
        7.0,
        description="Required pedestrian clearance interval in seconds (when applicable)"
    )
    lane_count: int = Field(1, ge=1, description="Number of lanes on this approach")
    saturation_flow_veh_per_hour: Optional[float] = Field(
        None,
        description="Calibrated saturation flow rate (vehicles/hour). None = uncalibrated"
    )


# ---------------------------------------------------------------------------
# Signal phase configuration
# ---------------------------------------------------------------------------

class SignalPhase(BaseModel):
    """A single signal phase definition."""
    phase_id: str = Field(..., description="Unique phase identifier, e.g. 'P1'")
    phase_name: str = Field(..., description="Human-readable name, e.g. 'North-South Through'")
    served_approaches: List[str] = Field(
        ...,
        description="Approach IDs that receive green in this phase"
    )
    permitted_movements: List[VehicleMovementType] = Field(
        default_factory=list,
        description="Vehicle movements permitted during this phase"
    )
    conflicting_phases: List[str] = Field(
        default_factory=list,
        description="Phase IDs that must NOT run simultaneously (safety constraint)"
    )
    min_green_sec: float = Field(
        10.0,
        ge=5.0,
        description="Minimum green duration in seconds (pedestrian minimum)"
    )
    max_green_sec: float = Field(
        60.0,
        le=180.0,
        description="Maximum green duration in seconds"
    )
    amber_sec: float = Field(
        3.0,
        ge=3.0,
        description="Amber/yellow clearance interval in seconds"
    )
    all_red_sec: float = Field(
        2.0,
        ge=1.0,
        description="All-red safety clearance interval in seconds"
    )
    pedestrian_walk_sec: Optional[float] = Field(
        None,
        description="Pedestrian WALK interval in seconds (when applicable)"
    )
    sequence_position: int = Field(
        0,
        ge=0,
        description="Position in the approved phase sequence (0-indexed)"
    )


class PhaseConstraint(BaseModel):
    """Constraint linking two phases that must not overlap."""
    phase_a: str
    phase_b: str
    reason: str


# ---------------------------------------------------------------------------
# Intersection configuration
# ---------------------------------------------------------------------------

class IntersectionConfig(BaseModel):
    """Complete configuration for a single intersection."""
    intersection_id: str = Field(..., description="Unique intersection identifier")
    name: str = Field(..., description="Human-readable intersection name")
    camera_id: Optional[str] = Field(None, description="Primary camera ID for this intersection")
    intersection_type: str = Field(
        "four-leg signalized",
        description="Type: four-leg, three-leg, roundabout, etc."
    )

    approaches: List[ApproachConfig] = Field(
        default_factory=list,
        description="Configured road approaches"
    )
    phases: List[SignalPhase] = Field(
        default_factory=list,
        description="Configured signal phases (must be validated before use)"
    )
    phase_constraints: List[PhaseConstraint] = Field(
        default_factory=list,
        description="Explicit conflict constraints between phases"
    )

    # Cycle parameters
    cycle_length_sec: float = Field(
        90.0,
        ge=30.0,
        le=300.0,
        description="Approved cycle length in seconds"
    )
    min_cycle_sec: float = Field(60.0, ge=30.0)
    max_cycle_sec: float = Field(150.0, le=300.0)

    # Baseline fixed-time plan
    fixed_plan_name: Optional[str] = Field(None, description="Name of the existing fixed-time plan")
    fixed_plan_green_times: Dict[str, float] = Field(
        default_factory=dict,
        description="Fixed-time green allocations: {phase_id: green_sec}"
    )

    # Demand model weights (initial experimental parameters)
    demand_weight_queue: float = Field(0.45, ge=0.0, le=1.0)
    demand_weight_arrival: float = Field(0.25, ge=0.0, le=1.0)
    demand_weight_waiting: float = Field(0.20, ge=0.0, le=1.0)
    demand_weight_occupancy: float = Field(0.10, ge=0.0, le=1.0)

    # Normalization thresholds
    max_queue_veh: float = Field(50.0, gt=0.0, description="Queue count mapped to demand score 1.0")
    max_arrival_veh_per_min: float = Field(30.0, gt=0.0, description="Arrival rate mapped to score 1.0")
    max_wait_sec: float = Field(300.0, gt=0.0, description="Waiting time mapped to score 1.0")
    max_occupancy_pct: float = Field(100.0, gt=0.0, description="ROI occupancy % mapped to score 1.0")

    # Starvation prevention
    max_phase_gap_sec: float = Field(
        300.0,
        description="Maximum elapsed time before an approach priority is boosted"
    )
    starvation_boost: float = Field(
        0.3,
        ge=0.0,
        le=0.5,
        description="Demand score boost added when starvation threshold is exceeded"
    )

    # Data quality
    stale_data_threshold_sec: float = Field(
        120.0,
        description="Age of analytics data beyond which it is considered stale"
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("demand_weight_queue", "demand_weight_arrival",
                     "demand_weight_waiting", "demand_weight_occupancy",
                     mode="before")
    @classmethod
    def weights_must_sum_to_one(cls, v: float) -> float:
        # Per-field validation only; sum validated at model level
        if not (0.0 <= v <= 1.0):
            raise ValueError("Weight must be in [0, 1]")
        return v


# ---------------------------------------------------------------------------
# Live approach analytics (snapshot from pipeline)
# ---------------------------------------------------------------------------

class ApproachDemand(BaseModel):
    """Real-time demand snapshot for one approach, derived from pipeline analytics."""
    approach_id: str
    direction: MovementDirection
    name: str
    road_id: Optional[str] = None

    # Raw metrics (from existing analytics pipeline)
    queue_length_veh: Optional[float] = Field(None, description="Current vehicles stationary/queued")
    arrival_rate_veh_per_min: Optional[float] = Field(None, description="Vehicle arrivals per minute")
    avg_wait_sec: Optional[float] = Field(None, description="Estimated average waiting time")
    occupancy_pct: Optional[float] = Field(None, description="ROI occupancy percentage 0-100")
    avg_speed_kmh: Optional[float] = Field(None, description="Average vehicle speed (None if uncalibrated)")
    moving_count: Optional[int] = Field(None, description="Currently moving vehicles in ROI")
    stationary_count: Optional[int] = Field(None, description="Currently stationary vehicles in ROI")
    traffic_state: Optional[str] = Field(None, description="LOW/MEDIUM/HIGH/CRITICAL from pipeline")

    # Normalized demand scores (0.0–1.0)
    score_queue: Optional[float] = None
    score_arrival: Optional[float] = None
    score_waiting: Optional[float] = None
    score_occupancy: Optional[float] = None
    demand_score: Optional[float] = Field(
        None,
        description="Composite demand score D_i = 0.45Q + 0.25A + 0.20W + 0.10O"
    )

    # Data quality
    data_quality: DataQualityFlag = DataQualityFlag.MISSING
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    data_age_sec: Optional[float] = Field(None, description="Age of the source analytics data")
    last_updated: Optional[datetime] = None

    # Starvation tracking
    last_served_at: Optional[datetime] = None
    time_since_served_sec: Optional[float] = None
    starvation_boost_applied: bool = False


# ---------------------------------------------------------------------------
# Signal recommendation
# ---------------------------------------------------------------------------

class PhaseRecommendation(BaseModel):
    """Recommended action for a single phase."""
    phase_id: str
    phase_name: str
    served_approaches: List[str]
    proposed_green_sec: float
    aggregate_demand_score: float
    per_approach_demand: List[ApproachDemand] = []

    # Evidence
    queue_evidence: str = ""
    wait_evidence: str = ""
    reason: str = ""

    # Constraint compliance
    min_green_respected: bool = True
    max_green_respected: bool = True
    conflicts_cleared: bool = True
    pedestrian_clearance_included: bool = True


class SignalRecommendation(BaseModel):
    """Full signal recommendation for one intersection at one point in time."""
    recommendation_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    intersection_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Source metrics provenance
    source_job_ids: List[str] = []
    source_metric_ages_sec: Dict[str, float] = {}

    # Recommended phase
    recommended_phase: PhaseRecommendation

    # Full demand picture for all approaches
    approach_demands: List[ApproachDemand] = []

    # Comparison against fixed-time baseline
    fixed_plan_comparison: Optional[Dict[str, Any]] = None

    # Overall status
    data_quality: DataQualityFlag = DataQualityFlag.GOOD
    status: RecommendationStatus = RecommendationStatus.GENERATED
    mode: str = Field("SIMULATION", description="SIMULATION or RECOMMENDATION_ONLY. Never LIVE_CONTROL.")
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    fallback_to_baseline: bool = Field(
        False,
        description="True when data quality is insufficient for recommendation"
    )
    fallback_reason: Optional[str] = None

    # Audit & Officer Decision Support
    constraint_validation_passed: bool = True
    constraint_violations: List[str] = []
    officer_decision: Optional[str] = Field(None, description="APPROVED or REJECTED")
    officer_id: Optional[str] = Field(None, description="Identifier of the reviewing officer")
    decision_timestamp: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    approved_green_sec: Optional[float] = None
    trigger_source: Optional[str] = Field("MANUAL", description="SCHEDULED_8S or MANUAL")


# ---------------------------------------------------------------------------
# Video Readiness & Approach Mapping Models
# ---------------------------------------------------------------------------

class SessionReadinessItem(BaseModel):
    session_id: str
    camera_id: str
    video_name: str
    status: str
    frames_processed: int = 0
    total_frames: int = 600
    progress_pct: int = 0
    latest_timestamp_sec: Optional[float] = None
    roads_mapped: List[str] = []
    is_valid: bool = True
    validation_note: str = ""


class VideoReadinessReport(BaseModel):
    intersection_id: str
    is_ready: bool
    valid_session_count: int
    required_min: int = 3
    allowed_max: int = 4
    coverage_status: str  # "COMPLETE" | "PARTIAL" | "MISSING"
    uncovered_approaches: List[str] = []
    recommendation_available: bool
    reasons: List[str] = []
    sessions: List[SessionReadinessItem] = []


class SignalStatusResponse(BaseModel):
    intersection_id: str
    current_phase_id: str
    current_phase_name: str
    served_approaches: List[str] = []
    phase_start_time: str
    phase_end_time: Optional[str] = None
    duration_sec: float
    elapsed_sec: float
    remaining_green_sec: Optional[float] = None
    is_timing_available: bool = True
    is_simulated: bool = True
    recommendation_trigger_sec: float = 8.0
    should_trigger_recommendation: bool = False


class ApprovalRequest(BaseModel):
    officer_id: str = "officer_1"
    approved_green_sec: float


class RejectionRequest(BaseModel):
    officer_id: str = "officer_1"
    rejection_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class SimulationScenario(BaseModel):
    """Scenario definition for a signal timing simulation."""
    intersection_id: str
    scenario_type: SimulationScenarioType = SimulationScenarioType.NORMAL_TRAFFIC
    scenario_name: str = "Normal Traffic"
    description: str = ""

    # Demand overrides per approach (None = use live analytics)
    approach_demand_overrides: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Override analytics per approach: {approach_id: {metric: value}}"
    )

    # Plans to compare
    compare_fixed_plan: bool = True
    compare_adaptive_plan: bool = True

    # Run parameters
    simulation_duration_sec: float = Field(
        3600.0,
        ge=60.0,
        description="Simulated duration in seconds"
    )
    random_seed: int = Field(42, description="Random seed for reproducibility")


class ApproachSimResult(BaseModel):
    """Simulation result for one approach."""
    approach_id: str
    direction: str
    avg_delay_sec: float
    avg_wait_sec: float
    max_queue_veh: int
    throughput_veh_per_hour: float
    max_wait_sec: float
    vehicles_served: int
    pedestrian_compliance: Optional[bool] = None


class PlanSimResult(BaseModel):
    """Simulation result for one timing plan (fixed or adaptive)."""
    plan_name: str
    plan_type: str  # "fixed" or "adaptive"
    avg_delay_sec: float
    avg_wait_sec: float
    max_queue_veh: int
    throughput_veh_per_hour: float
    max_wait_sec: float
    vehicles_served: int
    pedestrian_compliance: bool
    per_approach: List[ApproachSimResult] = []
    invalid_recommendations: int = 0
    notes: str = ""


class SimulationResult(BaseModel):
    """Full simulation run result."""
    simulation_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    intersection_id: str
    scenario: SimulationScenario
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    fixed_plan_result: Optional[PlanSimResult] = None
    adaptive_plan_result: Optional[PlanSimResult] = None

    # Improvement metrics (only claimed when both plans are simulated)
    delay_improvement_pct: Optional[float] = Field(
        None,
        description="Positive = adaptive plan is better. None if comparison is not available."
    )
    queue_improvement_pct: Optional[float] = None
    throughput_improvement_pct: Optional[float] = None

    # Methodology note
    methodology_note: str = (
        "Results are from an analytical queuing model (D/D/1), "
        "not a validated microsimulator (SUMO). "
        "Do not interpret as guaranteed real-world improvements."
    )
    simulation_mode: str = "ANALYTICAL"  # "ANALYTICAL" or "SUMO"

    data_quality: DataQualityFlag = DataQualityFlag.GOOD


# ---------------------------------------------------------------------------
# Audit log entry
# ---------------------------------------------------------------------------

class AuditLogEntry(BaseModel):
    """Audit record for recommendation creation, config changes, and simulation runs."""
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: str  # "RECOMMENDATION_CREATED", "CONFIG_CHANGED", "SIMULATION_RUN", etc.
    intersection_id: str
    actor: str = "SYSTEM"  # In future: operator username
    details: Dict[str, Any] = {}
    recommendation_id: Optional[str] = None
    simulation_id: Optional[str] = None
