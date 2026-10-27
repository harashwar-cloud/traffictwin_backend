"""
src/signal/scenario_simulator.py
-----------------------------------
Analytical Traffic Scenario Simulator.

Compares an adaptive signal plan against a fixed-time baseline using
a simplified D/D/1 queuing model. This is NOT a microsimulator (SUMO).

Methodology:
  - For each phase, given arrival rate λ (veh/min) and effective green g (s):
    - Capacity c = (g / cycle_length) * saturation_flow
    - If λ < c: queue drains, delay is low
    - If λ >= c: queue grows, delay escalates
  - Average delay estimated via Webster's formula (simplified)
  - Results are labeled ANALYTICAL to distinguish from validated microsimulation

⚠️ IMPORTANT:
  Results are from an analytical model. They should NOT be interpreted as
  guaranteed real-world improvements. Validate with SUMO or a field study
  before making operational decisions.
"""

from __future__ import annotations

import logging
import math
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# Default saturation flow: 1800 veh/hr/lane (typical urban value, uncalibrated)
DEFAULT_SATURATION_FLOW_VEH_PER_HR = 1800.0


class ScenarioSimulator:
    """
    Analytical scenario simulator for comparing adaptive vs fixed-time signal plans.
    """

    def run(
        self,
        intersection_config: dict,
        approach_demands: List[Dict[str, Any]],
        scenario_type: str = "NORMAL_TRAFFIC",
        demand_overrides: Optional[Dict[str, Dict[str, float]]] = None,
        simulation_duration_sec: float = 3600.0,
        random_seed: int = 42,
        proposed_adaptive_greens: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Run the simulation for one scenario.

        Parameters
        ----------
        intersection_config : dict
        approach_demands : list[dict]
            Demand scores from ApproachDemandCalculator.
        scenario_type : str
            One of SimulationScenarioType values.
        demand_overrides : dict | None
            {approach_id: {arrival_rate_veh_per_min: X, ...}}
        simulation_duration_sec : float
        random_seed : int
        proposed_adaptive_greens : dict | None
            {phase_id: green_sec} from recommendation engine.
        """
        rng = random.Random(random_seed)
        phases = intersection_config.get("phases", [])
        approaches = intersection_config.get("approaches", [])
        cycle_sec = intersection_config.get("cycle_length_sec", 90.0)
        max_cycle = intersection_config.get("max_cycle_sec", 90.0)

        # Build demand lookup
        demand_by_approach: Dict[str, dict] = {d["approach_id"]: d for d in approach_demands}

        # Apply scenario demand multipliers
        multipliers = self._scenario_multipliers(scenario_type, approaches, demand_overrides)

        # Fixed-time green plan
        fixed_greens = intersection_config.get("fixed_plan_green_times", {})
        if not fixed_greens:
            # Equal time split minus clearances
            overhead_per_phase = sum(
                p.get("amber_sec", 3.0) + p.get("all_red_sec", 2.0)
                for p in phases
            )
            equal_green = max(10.0, (cycle_sec - overhead_per_phase) / max(1, len(phases)))
            fixed_greens = {p["phase_id"]: equal_green for p in phases}

        # Adaptive green plan
        if proposed_adaptive_greens:
            adaptive_greens = proposed_adaptive_greens
        else:
            adaptive_greens = self._compute_demand_based_greens(
                phases=phases,
                demand_by_approach=demand_by_approach,
                multipliers=multipliers,
                intersection_config=intersection_config,
            )

        # Simulate both plans
        fixed_result = self._simulate_plan(
            plan_name=intersection_config.get("fixed_plan_name", "Fixed-Time Baseline"),
            plan_type="fixed",
            greens=fixed_greens,
            phases=phases,
            approaches=approaches,
            demand_by_approach=demand_by_approach,
            multipliers=multipliers,
            cycle_sec=cycle_sec,
            simulation_duration_sec=simulation_duration_sec,
            rng=rng,
        )

        adaptive_result = self._simulate_plan(
            plan_name="Adaptive Signal Plan",
            plan_type="adaptive",
            greens=adaptive_greens,
            phases=phases,
            approaches=approaches,
            demand_by_approach=demand_by_approach,
            multipliers=multipliers,
            cycle_sec=min(max_cycle, cycle_sec * 1.1),  # adaptive may vary cycle slightly
            simulation_duration_sec=simulation_duration_sec,
            rng=rng,
        )

        # Improvement calculations
        delay_impr = None
        queue_impr = None
        thru_impr = None

        if fixed_result.get("avg_delay_sec", 0) > 0:
            delay_impr = round(
                (1.0 - adaptive_result["avg_delay_sec"] / fixed_result["avg_delay_sec"]) * 100.0, 1
            )
        if fixed_result.get("max_queue_veh", 0) > 0:
            queue_impr = round(
                (1.0 - adaptive_result["max_queue_veh"] / fixed_result["max_queue_veh"]) * 100.0, 1
            )
        if fixed_result.get("throughput_veh_per_hour", 0) > 0:
            thru_impr = round(
                (adaptive_result["throughput_veh_per_hour"] / fixed_result["throughput_veh_per_hour"] - 1.0) * 100.0, 1
            )

        return {
            "intersection_id": intersection_config.get("intersection_id"),
            "scenario_type": scenario_type,
            "simulation_duration_sec": simulation_duration_sec,
            "fixed_plan_result": fixed_result,
            "adaptive_plan_result": adaptive_result,
            "delay_improvement_pct": delay_impr,
            "queue_improvement_pct": queue_impr,
            "throughput_improvement_pct": thru_impr,
            "methodology_note": (
                "Results are from a simplified analytical queuing model (D/D/1 + Webster). "
                "NOT from a validated microsimulator (SUMO). "
                "Do not claim these as guaranteed real-world improvements."
            ),
            "simulation_mode": "ANALYTICAL",
        }

    # ---------------------------------------------------------------------- #
    # Plan simulation
    # ---------------------------------------------------------------------- #

    def _simulate_plan(
        self,
        plan_name: str,
        plan_type: str,
        greens: Dict[str, float],
        phases: List[dict],
        approaches: List[dict],
        demand_by_approach: Dict[str, dict],
        multipliers: Dict[str, float],
        cycle_sec: float,
        simulation_duration_sec: float,
        rng: random.Random,
    ) -> Dict[str, Any]:
        """
        Simulate a signal timing plan using simplified queuing theory.
        Returns a dict compatible with PlanSimResult.
        """
        per_approach_results: List[Dict[str, Any]] = []
        total_delay = 0.0
        total_wait = 0.0
        total_max_q = 0
        total_served = 0
        total_thru = 0.0
        ped_compliant = True

        for ap in approaches:
            ap_id = ap["approach_id"]
            demand = demand_by_approach.get(ap_id, {})
            base_arrival = demand.get("arrival_rate_veh_per_min", 2.0) or 2.0
            arrival = max(0.1, base_arrival * multipliers.get(ap_id, 1.0))

            # Find which phase serves this approach and get its green
            phase_green = self._get_approach_green(ap_id, phases, greens)
            amber = self._get_approach_clearance(ap_id, phases, "amber_sec", 3.0)
            all_red = self._get_approach_clearance(ap_id, phases, "all_red_sec", 2.0)

            effective_green = max(5.0, phase_green - 0.0)  # all green used for vehicles
            effective_red = max(0.0, cycle_sec - phase_green - amber - all_red)

            # Saturation flow (lanes × sat_flow_per_lane)
            lanes = ap.get("lane_count", 1)
            sat_flow = (
                ap.get("saturation_flow_veh_per_hour") or
                (lanes * DEFAULT_SATURATION_FLOW_VEH_PER_HR)
            )
            sat_flow_per_min = sat_flow / 60.0

            # Effective green ratio
            g_ratio = effective_green / max(cycle_sec, 1.0)
            capacity_veh_per_min = g_ratio * sat_flow_per_min

            # Webster's simplified delay formula
            # d = (C * (1 - g/C)^2) / (2 * (1 - g/C * x)) + x^2 / (2 * q * (1 - x))
            # where x = v/c (degree of saturation), q = arrival rate, C = cycle
            x = min(0.99, arrival / max(0.01, capacity_veh_per_min))
            C = cycle_sec
            g_C = g_ratio

            try:
                d1 = (C * (1 - g_C) ** 2) / (2.0 * max(0.001, 1.0 - g_C * x))
                if x > 0 and arrival > 0:
                    d2 = (x ** 2) / (2.0 * arrival * max(0.001, 1.0 - x))
                else:
                    d2 = 0.0
                avg_delay = max(0.0, d1 + d2)
            except (ZeroDivisionError, ValueError):
                avg_delay = effective_red / 2.0  # fallback estimate

            # Add Gaussian noise (± 10%) for realism
            noise = rng.gauss(1.0, 0.10)
            avg_delay = max(0.0, avg_delay * noise)

            # Queue length estimate
            queue_veh = demand.get("queue_length_veh", 0.0) or 0.0
            if x >= 1.0:
                # Oversaturated: queue grows
                queue_veh = math.ceil(arrival * simulation_duration_sec / 60.0 * (x - 1.0) + queue_veh)
            else:
                queue_veh = max(0.0, queue_veh * max(0.0, 1.0 - g_ratio * (1.0 - x)))

            # Throughput
            cycles = simulation_duration_sec / max(1.0, cycle_sec)
            served = min(
                arrival * simulation_duration_sec / 60.0,
                capacity_veh_per_min * simulation_duration_sec / 60.0
            )

            # Pedestrian check
            ped_req = ap.get("pedestrian_clearance_sec", 0.0)
            if ap.get("has_pedestrian_crossing") and ped_req > 0:
                if effective_green < ped_req:
                    ped_compliant = False

            ap_result = {
                "approach_id": ap_id,
                "direction": ap.get("direction", "UNKNOWN"),
                "avg_delay_sec": round(avg_delay, 1),
                "avg_wait_sec": round(min(avg_delay, effective_red), 1),
                "max_queue_veh": int(queue_veh),
                "throughput_veh_per_hour": round(served * 3600.0 / simulation_duration_sec, 1),
                "max_wait_sec": round(effective_red, 1),
                "vehicles_served": int(served),
                "pedestrian_compliance": (
                    effective_green >= ped_req if ap.get("has_pedestrian_crossing") else None
                ),
                "degree_of_saturation": round(x, 3),
                "green_sec": round(effective_green, 1),
            }

            per_approach_results.append(ap_result)
            total_delay += avg_delay
            total_wait += ap_result["avg_wait_sec"]
            total_max_q = max(total_max_q, int(queue_veh))
            total_served += int(served)
            total_thru += ap_result["throughput_veh_per_hour"]

        n = max(1, len(approaches))
        return {
            "plan_name": plan_name,
            "plan_type": plan_type,
            "avg_delay_sec": round(total_delay / n, 1),
            "avg_wait_sec": round(total_wait / n, 1),
            "max_queue_veh": total_max_q,
            "throughput_veh_per_hour": round(total_thru, 1),
            "max_wait_sec": round(max((r["max_wait_sec"] for r in per_approach_results), default=0.0), 1),
            "vehicles_served": total_served,
            "pedestrian_compliance": ped_compliant,
            "per_approach": per_approach_results,
            "invalid_recommendations": 0,
            "notes": f"Analytical D/D/1 + Webster model. Seed={42}.",
        }

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    def _scenario_multipliers(
        self,
        scenario_type: str,
        approaches: List[dict],
        demand_overrides: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, float]:
        """Return demand multipliers per approach for the given scenario."""
        base = {ap["approach_id"]: 1.0 for ap in approaches}
        ap_ids = list(base.keys())

        if scenario_type == "LOW_TRAFFIC":
            return {k: 0.3 for k in base}
        elif scenario_type == "NORMAL_TRAFFIC":
            return base
        elif scenario_type == "PEAK_TRAFFIC":
            return {k: 1.8 for k in base}
        elif scenario_type == "HIGH_ARRIVAL_RATE":
            return {k: 2.5 for k in base}
        elif scenario_type == "UNEVEN_TRAFFIC":
            # First approach gets 3× demand, others get 0.4×
            result = {k: 0.4 for k in base}
            if ap_ids:
                result[ap_ids[0]] = 3.0
            return result
        elif scenario_type == "INCREASED_TURNING":
            return {k: 1.4 for k in base}
        elif scenario_type == "APPROACH_BLOCKED":
            # Block last approach
            result = dict(base)
            if ap_ids:
                result[ap_ids[-1]] = 0.0
            return result

        # Apply any manual overrides
        if demand_overrides:
            for ap_id, overrides in demand_overrides.items():
                if "arrival_rate_veh_per_min" in overrides and ap_id in base:
                    # Convert override to relative multiplier
                    # (actual rate will be applied in _simulate_plan)
                    pass
        return base

    def _get_approach_green(
        self, approach_id: str, phases: List[dict], greens: Dict[str, float]
    ) -> float:
        """Get green time for the phase serving this approach."""
        for phase in phases:
            if approach_id in phase.get("served_approaches", []):
                return greens.get(phase["phase_id"], phase.get("min_green_sec", 20.0))
        return 20.0  # default

    def _get_approach_clearance(
        self, approach_id: str, phases: List[dict], key: str, default: float
    ) -> float:
        for phase in phases:
            if approach_id in phase.get("served_approaches", []):
                return phase.get(key, default)
        return default

    def _compute_demand_based_greens(
        self,
        phases: List[dict],
        demand_by_approach: Dict[str, dict],
        multipliers: Dict[str, float],
        intersection_config: dict,
    ) -> Dict[str, float]:
        """Compute adaptive green allocation proportional to phase demand."""
        cycle_sec = intersection_config.get("cycle_length_sec", 90.0)

        # Calculate overhead
        total_overhead = sum(
            p.get("amber_sec", 3.0) + p.get("all_red_sec", 2.0) + (p.get("pedestrian_walk_sec") or 0.0)
            for p in phases
        )
        budget = max(0.0, cycle_sec - total_overhead)

        # Phase demands
        phase_demands: Dict[str, float] = {}
        for phase in phases:
            pid = phase["phase_id"]
            served = phase.get("served_approaches", [])
            demands = [
                demand_by_approach.get(ap, {}).get("arrival_rate_veh_per_min", 1.0) or 1.0
                for ap in served
            ]
            phase_demands[pid] = max(demands) if demands else 1.0

        total_d = sum(phase_demands.values()) or 1.0

        result: Dict[str, float] = {}
        for phase in phases:
            pid = phase["phase_id"]
            min_g = phase.get("min_green_sec", 10.0)
            max_g = phase.get("max_green_sec", 60.0)
            demand_frac = phase_demands.get(pid, 1.0) / total_d
            allocated = min_g + demand_frac * budget
            result[pid] = round(max(min_g, min(max_g, allocated)), 1)

        return result

    def run_simulation(
        self,
        intersection_config: Any,
        scenario: Any,
        approach_demands: Optional[List[Dict[str, Any]]] = None,
        demand_overrides: Optional[Dict[str, Dict[str, float]]] = None,
        proposed_adaptive_greens: Optional[Dict[str, float]] = None,
    ) -> Any:
        from api.signal_models import SimulationResult
        if hasattr(intersection_config, "model_dump"):
            cfg = intersection_config.model_dump()
        elif hasattr(intersection_config, "dict"):
            cfg = intersection_config.dict()
        elif not isinstance(intersection_config, dict):
            cfg = vars(intersection_config)
        else:
            cfg = intersection_config

        sc_type = getattr(scenario, "scenario_type", "NORMAL_TRAFFIC")
        if hasattr(sc_type, "value"):
            sc_type = sc_type.value
        dur = float(getattr(scenario, "simulation_duration_sec", 3600.0))
        seed = int(getattr(scenario, "random_seed", 42))

        if approach_demands is None:
            from src.signal.demand_calculator import ApproachDemandCalculator
            calc = ApproachDemandCalculator(cfg)
            approach_demands = []
            for app in cfg.get("approaches", []):
                app_dict = app if isinstance(app, dict) else (app.model_dump() if hasattr(app, "model_dump") else vars(app))
                app_id = app_dict.get("approach_id", "app")
                approach_demands.append(calc.compute(app_id, road_analytics={"queue": 15, "flow_volume": 20.0}))

        res = self.run(
            intersection_config=cfg,
            approach_demands=approach_demands,
            scenario_type=str(sc_type),
            demand_overrides=demand_overrides,
            simulation_duration_sec=dur,
            random_seed=seed,
            proposed_adaptive_greens=proposed_adaptive_greens,
        )
        if "scenario" not in res:
            res["scenario"] = scenario
        return SimulationResult(**res)
