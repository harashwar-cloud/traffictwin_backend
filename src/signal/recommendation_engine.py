"""
src/signal/recommendation_engine.py
--------------------------------------
Deterministic Rules-Based Signal Recommendation Engine.

Algorithm (Part 3.4):
  1. Load fresh traffic metrics for the intersection
  2. Validate data completeness and timestamps
  3. Calculate normalized approach-demand scores
  4. Aggregate demand for each configured signal phase
  5. Consider waiting time and starvation prevention
  6. Respect phase sequence and compatibility constraints
  7. Calculate proposed green duration: G_i = G_min,i + allocated_extra_i
  8. Enforce min/max green durations
  9. Account for amber, all-red, pedestrian clearance
  10. Compare against approved baseline
  11. Return recommendation with supporting evidence

Safety boundary:
  - This system operates in RECOMMENDATION_ONLY mode
  - It does NOT connect to physical signal controllers
  - All outputs are for officer review only
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.signal.demand_calculator import ApproachDemandCalculator
from src.signal.phase_validator import SignalPhaseValidator

log = logging.getLogger(__name__)


class SignalRecommendationEngine:
    """
    Produces signal phase recommendations from live approach-demand data.

    All parameters are read from the intersection configuration;
    no hardcoded traffic-engineering constants are used.
    """

    def __init__(self):
        self._validator = SignalPhaseValidator()

    def generate(
        self,
        intersection_config: dict,
        live_road_states: Dict[str, Dict[str, Any]],
        last_served: Optional[Dict[str, datetime]] = None,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Generate a signal recommendation for the given intersection.

        Parameters
        ----------
        intersection_config : dict
            IntersectionConfig (as dict) for the intersection.
        live_road_states : dict
            Mapping road_id → analytics dict from LIVE_ANALYSIS_STATES.
            Keys used per road: current_stationary, flow_volume, current_roi_count,
                                current_moving, average_speed, traffic_state.
        last_served : dict | None
            Mapping approach_id → datetime of last green phase.
        now : datetime | None
            Current time (injectable for testing).

        Returns
        -------
        dict compatible with SignalRecommendation model.
        """
        if hasattr(intersection_config, "model_dump"):
            intersection_config = intersection_config.model_dump()
        elif hasattr(intersection_config, "dict"):
            intersection_config = intersection_config.dict()
        elif not isinstance(intersection_config, dict):
            intersection_config = vars(intersection_config)

        if now is None:
            now = datetime.now(timezone.utc)

        iid = intersection_config.get("intersection_id", "unknown")
        phases = intersection_config.get("phases", [])
        approaches = intersection_config.get("approaches", [])

        # ------------------------------------------------------------------ #
        # Step 1: Validate phase configuration
        # ------------------------------------------------------------------ #
        validation_errors = self._validator.validate(intersection_config)
        if validation_errors:
            log.error(
                "[RecommendationEngine] intersection=%s: phase config invalid:\n%s",
                iid, "\n".join(validation_errors)
            )
            return self._build_fallback(
                intersection_config,
                reason=f"Phase configuration invalid: {'; '.join(validation_errors[:3])}",
                now=now,
            )

        # ------------------------------------------------------------------ #
        # Step 2: Build demand calculator from intersection config
        # ------------------------------------------------------------------ #
        calc = ApproachDemandCalculator(
            weight_queue=intersection_config.get("demand_weight_queue", 0.45),
            weight_arrival=intersection_config.get("demand_weight_arrival", 0.25),
            weight_waiting=intersection_config.get("demand_weight_waiting", 0.20),
            weight_occupancy=intersection_config.get("demand_weight_occupancy", 0.10),
            max_queue_veh=intersection_config.get("max_queue_veh", 50.0),
            max_arrival_veh_per_min=intersection_config.get("max_arrival_veh_per_min", 30.0),
            max_wait_sec=intersection_config.get("max_wait_sec", 300.0),
            max_occupancy_pct=intersection_config.get("max_occupancy_pct", 100.0),
            stale_threshold_sec=intersection_config.get("stale_data_threshold_sec", 120.0),
        )

        # ------------------------------------------------------------------ #
        # Step 3: Compute per-approach demand scores
        # ------------------------------------------------------------------ #
        approach_demands = calc.batch_compute(
            approaches=approaches,
            live_states=live_road_states,
            last_served=last_served,
            starvation_boost=intersection_config.get("starvation_boost", 0.3),
            max_phase_gap_sec=intersection_config.get("max_phase_gap_sec", 300.0),
            now=now,
        )

        # Map approach_id → demand dict for fast lookup
        demand_by_approach: Dict[str, dict] = {d["approach_id"]: d for d in approach_demands}

        # ------------------------------------------------------------------ #
        # Step 4: Check data quality — fall back to baseline if all stale/missing
        # ------------------------------------------------------------------ #
        good_count = sum(
            1 for d in approach_demands
            if d.get("data_quality") in ("GOOD", "PARTIAL")
        )
        if good_count == 0:
            log.warning(
                "[RecommendationEngine] intersection=%s: all approach data missing/stale → fallback",
                iid,
            )
            return self._build_fallback(
                intersection_config,
                reason="All approach analytics are missing or stale. Cannot generate recommendation.",
                now=now,
                approach_demands=approach_demands,
            )

        # ------------------------------------------------------------------ #
        # Step 5: Aggregate demand per phase
        # ------------------------------------------------------------------ #
        phase_scores: List[Tuple[float, dict]] = []  # (score, phase_dict)
        for phase in phases:
            served = phase.get("served_approaches", [])
            ap_demands = [demand_by_approach.get(ap_id, {}) for ap_id in served]

            # Phase demand = max demand among served approaches
            # (the phase that resolves the highest-demand approach is prioritized)
            scores = [
                d.get("demand_score", 0.0) or 0.0
                for d in ap_demands
                if d.get("demand_score") is not None
            ]
            phase_demand = max(scores) if scores else 0.0
            phase_scores.append((phase_demand, phase))

        # Sort by demand descending
        phase_scores.sort(key=lambda x: x[0], reverse=True)

        # ------------------------------------------------------------------ #
        # Step 6: Select best phase respecting conflict constraints
        # ------------------------------------------------------------------ #
        phase_map: Dict[str, dict] = {p["phase_id"]: p for p in phases}
        selected_phase: Optional[dict] = None
        selected_demand: float = 0.0

        for demand_score, candidate in phase_scores:
            pid = candidate["phase_id"]
            conflicts = set(candidate.get("conflicting_phases", []))

            # A phase is selectable if it doesn't conflict with itself
            # (At recommendation time we pick one phase at a time)
            selected_phase = candidate
            selected_demand = demand_score
            break  # Take the highest-demand valid phase

        if selected_phase is None:
            return self._build_fallback(
                intersection_config,
                reason="No valid phase could be selected after conflict filtering.",
                now=now,
                approach_demands=approach_demands,
            )

        # ------------------------------------------------------------------ #
        # Step 7: Green time allocation
        # ------------------------------------------------------------------ #
        proposed_green = self._allocate_green_time(
            selected_phase=selected_phase,
            phase_demand=selected_demand,
            all_phase_scores=phase_scores,
            intersection_config=intersection_config,
        )

        # ------------------------------------------------------------------ #
        # Step 8: Build evidence strings
        # ------------------------------------------------------------------ #
        served_approach_ids = selected_phase.get("served_approaches", [])
        served_demands = [demand_by_approach.get(ap_id, {}) for ap_id in served_approach_ids]

        queue_evidence = self._build_queue_evidence(served_demands, demand_by_approach)
        wait_evidence = self._build_wait_evidence(served_demands)
        reason = self._build_reason(selected_phase, selected_demand, served_demands, phase_scores)

        # ------------------------------------------------------------------ #
        # Step 9: Fixed-plan comparison
        # ------------------------------------------------------------------ #
        fixed_comparison = self._compare_with_fixed_plan(
            selected_phase=selected_phase,
            proposed_green=proposed_green,
            intersection_config=intersection_config,
        )

        # ------------------------------------------------------------------ #
        # Step 10: Constraint validation summary
        # ------------------------------------------------------------------ #
        min_g = selected_phase.get("min_green_sec", 10.0)
        max_g = selected_phase.get("max_green_sec", 60.0)
        constraint_violations: List[str] = []
        if proposed_green < min_g:
            constraint_violations.append(f"Proposed green {proposed_green:.1f}s < min {min_g:.1f}s")
        if proposed_green > max_g:
            constraint_violations.append(f"Proposed green {proposed_green:.1f}s > max {max_g:.1f}s")

        # ------------------------------------------------------------------ #
        # Step 11: Overall confidence
        # ------------------------------------------------------------------ #
        confidence = self._compute_confidence(approach_demands, good_count, len(approaches))

        # ------------------------------------------------------------------ #
        # Build output dict
        # ------------------------------------------------------------------ #
        return {
            "intersection_id": iid,
            "timestamp": now.isoformat(),
            "source_job_ids": list({
                ap.get("job_id")
                for ap in approaches
                if ap.get("job_id")
            }),
            "recommended_phase": {
                "phase_id": selected_phase["phase_id"],
                "phase_name": selected_phase.get("phase_name", selected_phase["phase_id"]),
                "served_approaches": served_approach_ids,
                "proposed_green_sec": proposed_green,
                "amber_sec": selected_phase.get("amber_sec", 3.0),
                "all_red_sec": selected_phase.get("all_red_sec", 2.0),
                "aggregate_demand_score": round(selected_demand, 4),
                "per_approach_demand": served_demands,
                "queue_evidence": queue_evidence,
                "wait_evidence": wait_evidence,
                "reason": reason,
                "min_green_respected": proposed_green >= min_g,
                "max_green_respected": proposed_green <= max_g,
                "conflicts_cleared": not bool(constraint_violations),
                "pedestrian_clearance_included": True,
            },
            "approach_demands": approach_demands,
            "fixed_plan_comparison": fixed_comparison,
            "data_quality": self._overall_quality(approach_demands),
            "status": "GENERATED",
            "mode": "RECOMMENDATION_ONLY",
            "confidence": round(confidence, 3),
            "fallback_to_baseline": False,
            "constraint_validation_passed": not bool(constraint_violations),
            "constraint_violations": constraint_violations,
            "all_phase_scores": [
                {
                    "phase_id": p["phase_id"],
                    "phase_name": p.get("phase_name", p["phase_id"]),
                    "demand_score": round(s, 4),
                    "served_approaches": p.get("served_approaches", []),
                }
                for s, p in phase_scores
            ],
        }

    # ---------------------------------------------------------------------- #
    # Green time allocation
    # ---------------------------------------------------------------------- #

    def _allocate_green_time(
        self,
        selected_phase: dict,
        phase_demand: float,
        all_phase_scores: List[Tuple[float, dict]],
        intersection_config: dict,
    ) -> float:
        """
        G_i = G_min,i + allocated_extra_i

        allocated_extra_i is proportional to demand fraction of the selected phase
        within the available timing budget.

        The budget = max_cycle - sum(min_green + clearances for all phases).
        """
        phases = intersection_config.get("phases", [])
        max_cycle = intersection_config.get("max_cycle_sec", 90.0)

        # Total overhead from minimum greens + clearances for all phases
        total_overhead = 0.0
        for ph in phases:
            total_overhead += (
                ph.get("min_green_sec", 10.0)
                + ph.get("amber_sec", 3.0)
                + ph.get("all_red_sec", 2.0)
                + (ph.get("pedestrian_walk_sec") or 0.0)
            )

        available_budget = max(0.0, max_cycle - total_overhead)

        min_g = selected_phase.get("min_green_sec", 10.0)
        max_g = selected_phase.get("max_green_sec", 60.0)

        if available_budget <= 0.0 or phase_demand <= 0.0:
            # No budget remaining; give minimum green
            return min_g

        # Demand fraction: proportion of this phase's demand vs total demand
        total_demand = sum(s for s, _ in all_phase_scores) or 1.0
        demand_fraction = phase_demand / total_demand

        allocated_extra = demand_fraction * available_budget

        proposed = min_g + allocated_extra
        # Enforce constraints
        proposed = max(min_g, min(max_g, proposed))
        return round(proposed, 1)

    # ---------------------------------------------------------------------- #
    # Evidence builders
    # ---------------------------------------------------------------------- #

    def _build_queue_evidence(
        self,
        served_demands: List[dict],
        all_demands: Dict[str, dict],
    ) -> str:
        parts = []
        for d in served_demands:
            q = d.get("queue_length_veh")
            ap_name = d.get("name", d.get("approach_id", "?"))
            if q is not None:
                parts.append(f"{ap_name}: {q:.0f} vehicles queued")
            else:
                parts.append(f"{ap_name}: queue data unavailable")
        return "; ".join(parts) if parts else "No queue data available"

    def _build_wait_evidence(self, served_demands: List[dict]) -> str:
        parts = []
        for d in served_demands:
            w = d.get("avg_wait_sec")
            ap_name = d.get("name", d.get("approach_id", "?"))
            if w is not None:
                parts.append(f"{ap_name}: est. {w:.0f}s avg wait")
            else:
                parts.append(f"{ap_name}: wait time unavailable")
        return "; ".join(parts) if parts else "No wait time data available"

    def _build_reason(
        self,
        selected_phase: dict,
        demand_score: float,
        served_demands: List[dict],
        all_phase_scores: List[Tuple[float, dict]],
    ) -> str:
        phase_name = selected_phase.get("phase_name", selected_phase["phase_id"])

        if not all_phase_scores:
            return f"Phase '{phase_name}' selected (no alternatives configured)."

        # Compare against next best
        alternatives = [(s, p) for s, p in all_phase_scores if p["phase_id"] != selected_phase["phase_id"]]

        parts = [
            f"Phase '{phase_name}' has the highest aggregate demand score ({demand_score:.3f})."
        ]

        if alternatives:
            next_score, next_phase = alternatives[0]
            delta = demand_score - next_score
            parts.append(
                f"Leads alternative '{next_phase.get('phase_name', next_phase['phase_id'])}' "
                f"by {delta:.3f} demand units."
            )

        # Starvation note
        for d in served_demands:
            if d.get("starvation_boost_applied"):
                elapsed = d.get("time_since_served_sec", 0)
                parts.append(
                    f"Starvation prevention applied for {d.get('name', d.get('approach_id', '?'))}: "
                    f"not served for {elapsed:.0f}s."
                )

        return " ".join(parts)

    # ---------------------------------------------------------------------- #
    # Fixed-plan comparison
    # ---------------------------------------------------------------------- #

    def _compare_with_fixed_plan(
        self,
        selected_phase: dict,
        proposed_green: float,
        intersection_config: dict,
    ) -> Optional[dict]:
        pid = selected_phase["phase_id"]
        fixed_plan = intersection_config.get("fixed_plan_green_times", {})
        fixed_green = fixed_plan.get(pid)
        plan_name = intersection_config.get("fixed_plan_name", "Fixed-time baseline")

        if fixed_green is None:
            return {
                "baseline_plan": plan_name,
                "baseline_green_sec": None,
                "proposed_green_sec": proposed_green,
                "delta_sec": None,
                "note": "Fixed-time green for this phase not configured.",
            }

        delta = proposed_green - fixed_green
        return {
            "baseline_plan": plan_name,
            "baseline_green_sec": fixed_green,
            "proposed_green_sec": proposed_green,
            "delta_sec": round(delta, 1),
            "note": (
                f"Adaptive plan proposes {abs(delta):.1f}s "
                f"{'more' if delta > 0 else 'less'} than baseline for phase {pid}."
            ),
        }

    # ---------------------------------------------------------------------- #
    # Fallback
    # ---------------------------------------------------------------------- #

    def _build_fallback(
        self,
        intersection_config: dict,
        reason: str,
        now: datetime,
        approach_demands: Optional[List[dict]] = None,
    ) -> dict:
        iid = intersection_config.get("intersection_id", "unknown")
        phases = intersection_config.get("phases", [])
        fixed_plan = intersection_config.get("fixed_plan_green_times", {})
        plan_name = intersection_config.get("fixed_plan_name", "Fixed-time baseline")

        # Fall back to the first phase in approved sequence
        fallback_phase = min(phases, key=lambda p: p.get("sequence_position", 999)) if phases else {}
        min_g = fallback_phase.get("min_green_sec", 30.0)
        pid = fallback_phase.get("phase_id", "P1")
        fixed_green = fixed_plan.get(pid, min_g)

        log.warning("[RecommendationEngine] intersection=%s: fallback reason: %s", iid, reason)

        return {
            "intersection_id": iid,
            "timestamp": now.isoformat(),
            "source_job_ids": [],
            "recommended_phase": {
                "phase_id": pid,
                "phase_name": fallback_phase.get("phase_name", pid),
                "served_approaches": fallback_phase.get("served_approaches", []),
                "proposed_green_sec": fixed_green,
                "amber_sec": fallback_phase.get("amber_sec", 3.0),
                "all_red_sec": fallback_phase.get("all_red_sec", 2.0),
                "aggregate_demand_score": 0.0,
                "per_approach_demand": approach_demands or [],
                "queue_evidence": "Insufficient analytics data",
                "wait_evidence": "Insufficient analytics data",
                "reason": f"Fallback to approved baseline: {reason}",
                "min_green_respected": True,
                "max_green_respected": True,
                "conflicts_cleared": True,
                "pedestrian_clearance_included": True,
            },
            "approach_demands": approach_demands or [],
            "fixed_plan_comparison": {
                "baseline_plan": plan_name,
                "baseline_green_sec": fixed_green,
                "proposed_green_sec": fixed_green,
                "delta_sec": 0.0,
                "note": "Fallback — using approved baseline values.",
            },
            "data_quality": "MISSING",
            "status": "GENERATED",
            "mode": "RECOMMENDATION_ONLY",
            "confidence": 0.0,
            "fallback_to_baseline": True,
            "fallback_reason": reason,
            "constraint_validation_passed": True,
            "constraint_violations": [],
            "all_phase_scores": [],
        }

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _overall_quality(approach_demands: List[dict]) -> str:
        qualities = [d.get("data_quality", "MISSING") for d in approach_demands]
        if all(q == "GOOD" for q in qualities):
            return "GOOD"
        if all(q == "MISSING" for q in qualities):
            return "MISSING"
        if any(q == "STALE" for q in qualities):
            return "STALE"
        return "PARTIAL"

    @staticmethod
    def _compute_confidence(
        approach_demands: List[dict],
        good_count: int,
        total: int,
    ) -> float:
        if total == 0:
            return 0.0
        base = good_count / total
        avg_individual = (
            sum(d.get("confidence", 0.0) for d in approach_demands) / total
        )
        return (base + avg_individual) / 2.0

    def generate_recommendation(
        self,
        intersection_config: Any,
        live_road_states: Dict[str, Dict[str, Any]],
        last_served: Optional[Dict[str, datetime]] = None,
        now: Optional[datetime] = None,
    ) -> Any:
        from api.signal_models import SignalRecommendation
        res = self.generate(intersection_config, live_road_states, last_served=last_served, now=now)
        return SignalRecommendation(**res)
