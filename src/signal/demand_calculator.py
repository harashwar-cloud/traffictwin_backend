"""
src/signal/demand_calculator.py
----------------------------------
Approach-Demand Calculation Module.

Computes a composite demand score D_i for each road approach:

    D_i = w_Q * Q_i + w_A * A_i + w_W * W_i + w_O * O_i

Where:
    Q_i  = normalized queue demand       (0–1)
    A_i  = normalized arrival-rate demand (0–1)
    W_i  = normalized waiting-time demand (0–1)
    O_i  = normalized occupancy demand    (0–1)

Default weights (initial experimental parameters — NOT validated traffic-
engineering constants):
    w_Q = 0.45,  w_A = 0.25,  w_W = 0.20,  w_O = 0.10

These weights are stored in IntersectionConfig and adjustable without code changes.

Requirements satisfied:
  - Missing metrics handled: treated as 0 demand, data quality flagged
  - Stale observations: data quality flagged, confidence reduced
  - NaN/Inf prevented: all division guarded with max(0, min(...))
  - No double-counting: Q and O are related but Q = stationary only, O = all ROI
  - Per-approach independence: each score calculated separately
  - Score is separate from vehicle-detection confidence
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _safe_normalize(value: Optional[float], max_val: float) -> Optional[float]:
    """
    Normalize value to [0, 1] using a configurable maximum.

    Returns None if value is None (missing metric).
    Returns 0.0 if value is <= 0.
    Returns 1.0 if value >= max_val.
    Guards against NaN, Inf, and zero/negative max_val.
    """
    if value is None:
        return None
    if not math.isfinite(value):
        log.warning("Non-finite metric value %s replaced with 0", value)
        return 0.0
    if not math.isfinite(max_val) or max_val <= 0:
        log.error("Invalid normalization threshold %s; returning 0", max_val)
        return 0.0
    normalized = max(0.0, min(1.0, value / max_val))
    return normalized


def _coalesce_zero(normalized: Optional[float]) -> Tuple[float, bool]:
    """
    Return (score, missing_flag).
    When a metric is missing, the score is 0.0 and missing_flag is True.
    0 demand is a safe default — it never inflates the recommendation.
    """
    if normalized is None:
        return 0.0, True
    return normalized, False


# ---------------------------------------------------------------------------
# Main calculator
# ---------------------------------------------------------------------------

class ApproachDemandCalculator:
    """
    Stateless calculator for per-approach demand scores.

    All state resides in the intersection configuration.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        weight_queue: float = 0.45,
        weight_arrival: float = 0.25,
        weight_waiting: float = 0.20,
        weight_occupancy: float = 0.10,
        # Normalization thresholds
        max_queue_veh: float = 50.0,
        max_arrival_veh_per_min: float = 30.0,
        max_wait_sec: float = 300.0,
        max_occupancy_pct: float = 100.0,
        # Staleness
        stale_threshold_sec: float = 120.0,
        **kwargs: Any
    ):
        if config is not None and not isinstance(config, (int, float)):
            cfg = config
            def _get(k: str, default: Any) -> Any:
                if isinstance(cfg, dict):
                    return cfg.get(k, default)
                return getattr(cfg, k, default)
            weight_queue = _get("demand_weight_queue", 0.45)
            weight_arrival = _get("demand_weight_arrival", 0.25)
            weight_waiting = _get("demand_weight_waiting", 0.20)
            weight_occupancy = _get("demand_weight_occupancy", 0.10)
            max_queue_veh = _get("max_queue_veh", 50.0)
            max_arrival_veh_per_min = _get("max_arrival_veh_per_min", 30.0)
            max_wait_sec = _get("max_wait_sec", 300.0)
            max_occupancy_pct = _get("max_occupancy_pct", 100.0)
            stale_threshold_sec = _get("stale_data_threshold_sec", 120.0)
        elif isinstance(config, (int, float)):
            weight_queue = float(config)

        # Validate weights sum approximately to 1.0
        total = weight_queue + weight_arrival + weight_waiting + weight_occupancy
        if not math.isclose(total, 1.0, abs_tol=0.05):
            log.warning(
                "Demand weights sum to %.3f (expected 1.0). "
                "Scores will not be in [0,1]. Consider normalizing weights.",
                total
            )

        self.w_q = weight_queue
        self.w_a = weight_arrival
        self.w_w = weight_waiting
        self.w_o = weight_occupancy

        self.max_queue = max(0.001, max_queue_veh)
        self.max_arrival = max(0.001, max_arrival_veh_per_min)
        self.max_wait = max(0.001, max_wait_sec)
        self.max_occ = max(0.001, max_occupancy_pct)

        self.stale_threshold_sec = stale_threshold_sec

    def calculate_approach_demand(
        self,
        approach: Any,
        road_analytics: Optional[Dict[str, Any]] = None,
        time_since_served_sec: Optional[float] = None,
        data_age_sec: Optional[float] = None
    ) -> Any:
        from api.signal_models import ApproachDemand, MovementDirection
        app_id = getattr(approach, "approach_id", str(approach))
        direction = getattr(approach, "direction", MovementDirection.UNKNOWN)
        name = getattr(approach, "name", app_id)

        if data_age_sec is None and road_analytics and "last_updated" in road_analytics:
            lu = road_analytics["last_updated"]
            if isinstance(lu, str):
                try:
                    lu_dt = datetime.fromisoformat(lu)
                    if lu_dt.tzinfo is None:
                        lu_dt = lu_dt.replace(tzinfo=timezone.utc)
                    data_age_sec = max(0.0, (datetime.now(timezone.utc) - lu_dt).total_seconds())
                except Exception:
                    pass

        last_served = None
        if time_since_served_sec is not None:
            last_served = datetime.now(timezone.utc) - timedelta(seconds=time_since_served_sec)

        res = self.compute(
            approach_id=app_id,
            road_analytics=road_analytics,
            data_age_sec=data_age_sec,
            last_served_at=last_served,
            max_phase_gap_sec=120.0
        )
        res["direction"] = direction
        res["name"] = name
        return ApproachDemand(**res)

    def compute(
        self,
        approach_id: str,
        road_analytics: Optional[Dict[str, Any]],
        data_age_sec: Optional[float] = None,
        last_served_at: Optional[datetime] = None,
        starvation_boost: float = 0.3,
        max_phase_gap_sec: float = 300.0,
    ) -> Dict[str, Any]:
        """
        Compute demand score for one approach.

        Parameters
        ----------
        approach_id : str
            Approach identifier for logging.
        road_analytics : dict | None
            Analytics dict from LIVE_ANALYSIS_STATES[job_id]["roads"][i].
            Expected keys: current_stationary, flow_volume, average_speed,
                           current_roi_count, current_active.
        data_age_sec : float | None
            Age of the analytics data in seconds. Used for staleness detection.
        last_served_at : datetime | None
            When this approach was last given a green phase (for starvation prevention).
        starvation_boost : float
            Extra demand added when starvation threshold exceeded (0–0.5).
        max_phase_gap_sec : float
            Seconds before starvation boost activates.

        Returns
        -------
        dict with keys matching ApproachDemand fields.
        """
        from api.signal_models import DataQualityFlag  # avoid circular at import

        result: Dict[str, Any] = {
            "approach_id": approach_id,
            "queue_length_veh": None,
            "arrival_rate_veh_per_min": None,
            "avg_wait_sec": None,
            "occupancy_pct": None,
            "avg_speed_kmh": None,
            "moving_count": None,
            "stationary_count": None,
            "traffic_state": None,
            "score_queue": None,
            "score_arrival": None,
            "score_waiting": None,
            "score_occupancy": None,
            "demand_score": None,
            "data_quality": DataQualityFlag.MISSING,
            "confidence": 0.0,
            "data_age_sec": data_age_sec,
            "last_updated": None,
            "time_since_served_sec": None,
            "starvation_boost_applied": False,
        }

        # ------------------------------------------------------------------ #
        # 1. Check data availability
        # ------------------------------------------------------------------ #
        if road_analytics is None:
            log.warning("[DemandCalc] approach=%s: road_analytics is None → MISSING", approach_id)
            return result

        # ------------------------------------------------------------------ #
        # 2. Staleness check
        # ------------------------------------------------------------------ #
        stale = False
        if data_age_sec is not None and data_age_sec > self.stale_threshold_sec:
            stale = True
            log.warning(
                "[DemandCalc] approach=%s: analytics age %.1fs > threshold %.1fs → STALE",
                approach_id, data_age_sec, self.stale_threshold_sec,
            )
            result["data_quality"] = DataQualityFlag.STALE

        # ------------------------------------------------------------------ #
        # 3. Extract raw metrics (safe extraction with None fallback)
        # ------------------------------------------------------------------ #
        queue_veh = self._safe_get(road_analytics, "current_stationary")
        if queue_veh is None:
            queue_veh = self._safe_get(road_analytics, "queue")
        arrival = self._safe_get(road_analytics, "flow_volume")
        if arrival is None:
            arrival = self._safe_get(road_analytics, "arrival_rate")
        avg_speed = self._safe_get(road_analytics, "average_speed")  # km/h
        roi_count = self._safe_get(road_analytics, "current_roi_count") or \
                    self._safe_get(road_analytics, "current_active") or \
                    self._safe_get(road_analytics, "roi_count")
        moving = self._safe_get(road_analytics, "current_moving")
        stationary = self._safe_get(road_analytics, "current_stationary")
        traffic_state = road_analytics.get("traffic_state")

        # Estimate or extract occupancy
        occupancy_pct: Optional[float] = self._safe_get(road_analytics, "occupancy")
        if occupancy_pct is None and roi_count is not None and self.max_queue > 0:
            occupancy_pct = min(100.0, (roi_count / self.max_queue) * 100.0)

        # Estimate or extract waiting time
        avg_wait_sec: Optional[float] = self._safe_get(road_analytics, "average_wait") or \
                                        self._safe_get(road_analytics, "avg_wait")
        if avg_wait_sec is None and queue_veh is not None:
            avg_wait_sec = max(0.0, queue_veh * 8.0)

        # ------------------------------------------------------------------ #
        # 4. Store raw metrics
        # ------------------------------------------------------------------ #
        result["queue_length_veh"] = queue_veh
        result["arrival_rate_veh_per_min"] = arrival
        result["avg_wait_sec"] = avg_wait_sec
        result["occupancy_pct"] = occupancy_pct
        result["avg_speed_kmh"] = avg_speed
        result["moving_count"] = int(moving) if moving is not None else None
        result["stationary_count"] = int(stationary) if stationary is not None else None
        result["traffic_state"] = traffic_state
        result["last_updated"] = datetime.now(timezone.utc)

        # ------------------------------------------------------------------ #
        # 5. Normalize metrics
        # ------------------------------------------------------------------ #
        s_q = _safe_normalize(queue_veh, self.max_queue)
        s_a = _safe_normalize(arrival, self.max_arrival)
        s_w = _safe_normalize(avg_wait_sec, self.max_wait)
        s_o = _safe_normalize(occupancy_pct, self.max_occ)

        result["score_queue"] = s_q
        result["score_arrival"] = s_a
        result["score_waiting"] = s_w
        result["score_occupancy"] = s_o

        # ------------------------------------------------------------------ #
        # 6. Composite demand score
        # ------------------------------------------------------------------ #
        q_val, q_miss = _coalesce_zero(s_q)
        a_val, a_miss = _coalesce_zero(s_a)
        w_val, w_miss = _coalesce_zero(s_w)
        o_val, o_miss = _coalesce_zero(s_o)

        demand = (
            self.w_q * q_val +
            self.w_a * a_val +
            self.w_w * w_val +
            self.w_o * o_val
        )

        # Guard against floating-point artefacts
        demand = max(0.0, min(1.0, demand))
        if not math.isfinite(demand):
            log.error("[DemandCalc] approach=%s: demand score is non-finite, resetting to 0", approach_id)
            demand = 0.0

        result["demand_score"] = round(demand, 4)

        # ------------------------------------------------------------------ #
        # 7. Data quality and confidence
        # ------------------------------------------------------------------ #
        missing_count = sum([q_miss, a_miss, w_miss, o_miss])
        if not stale:
            if missing_count == 0:
                result["data_quality"] = DataQualityFlag.GOOD
                result["confidence"] = 1.0
            elif missing_count <= 2:
                result["data_quality"] = DataQualityFlag.PARTIAL
                result["confidence"] = 1.0 - (missing_count * 0.25)
            else:
                result["data_quality"] = DataQualityFlag.PARTIAL
                result["confidence"] = 0.2

        if stale:
            result["confidence"] = max(0.0, result["confidence"] - 0.5)

        # ------------------------------------------------------------------ #
        # 8. Starvation prevention boost
        # ------------------------------------------------------------------ #
        if last_served_at is not None:
            now = datetime.now(timezone.utc)
            if last_served_at.tzinfo is None:
                last_served_at = last_served_at.replace(tzinfo=timezone.utc)
            elapsed = (now - last_served_at).total_seconds()
            result["time_since_served_sec"] = round(elapsed, 1)

            if elapsed > max_phase_gap_sec:
                boost_amount = min(starvation_boost, 1.0 - demand)
                if boost_amount > 0:
                    old_demand = demand
                    demand = min(1.0, demand + boost_amount)
                    result["demand_score"] = round(demand, 4)
                    result["starvation_boost_applied"] = True
                    log.info(
                        "[DemandCalc] approach=%s starvation boost %.3f applied "
                        "(%.3f → %.3f, elapsed=%.0fs)",
                        approach_id, boost_amount, old_demand, demand, elapsed,
                    )

        log.debug(
            "[DemandCalc] approach=%s D=%.3f Q=%.2f A=%.2f W=%.2f O=%.2f quality=%s",
            approach_id,
            result["demand_score"] or 0,
            q_val, a_val, w_val, o_val,
            result["data_quality"],
        )

        return result

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _safe_get(d: Dict[str, Any], key: str) -> Optional[float]:
        """Get a numeric value from a dict, returning None if missing or non-finite."""
        v = d.get(key)
        if v is None:
            return None
        try:
            f = float(v)
            return f if math.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    def batch_compute(
        self,
        approaches: List[Dict[str, Any]],
        live_states: Dict[str, Dict[str, Any]],
        last_served: Optional[Dict[str, datetime]] = None,
        starvation_boost: float = 0.3,
        max_phase_gap_sec: float = 300.0,
        now: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compute demand scores for all configured approaches.

        Parameters
        ----------
        approaches : list[dict]
            Approach configs from IntersectionConfig.approaches.
        live_states : dict
            Mapping of road_id → analytics dict from the pipeline.
        last_served : dict | None
            Mapping of approach_id → datetime when last served.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        results = []
        for ap in approaches:
            ap_id = ap.get("approach_id") or ap.get("approach_id", "unknown")
            road_id = ap.get("road_id")

            # Retrieve analytics for this approach's road ROI
            analytics = None
            data_age_sec = None
            if road_id and road_id in live_states:
                analytics = live_states[road_id]
            elif ap_id and ap_id in live_states:
                analytics = live_states[ap_id]
                # Estimate age from last_updated field if available
                ts = analytics.get("last_updated") or analytics.get("timestamp")
                if ts:
                    try:
                        if isinstance(ts, str):
                            ts = datetime.fromisoformat(ts)
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        data_age_sec = (now - ts).total_seconds()
                    except Exception:
                        data_age_sec = None

            lsa = (last_served or {}).get(ap_id)
            score_dict = self.compute(
                approach_id=ap_id,
                road_analytics=analytics,
                data_age_sec=data_age_sec,
                last_served_at=lsa,
                starvation_boost=starvation_boost,
                max_phase_gap_sec=max_phase_gap_sec,
            )
            # Merge approach direction / name from config
            score_dict["direction"] = ap.get("direction", "UNKNOWN")
            score_dict["name"] = ap.get("name", ap_id)
            score_dict["road_id"] = road_id
            results.append(score_dict)

        return results
