"""
scripts/run_signal_tests.py
---------------------------
Test runner for Adaptive Signal Optimization modules.
"""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime, timezone, timedelta
from api.signal_models import (
    IntersectionConfig, ApproachConfig, SignalPhase,
    SimulationScenario, SimulationScenarioType, MovementDirection, DataQualityFlag
)
from src.signal.demand_calculator import ApproachDemandCalculator
from src.signal.phase_validator import SignalPhaseValidator
from src.signal.recommendation_engine import SignalRecommendationEngine
from src.signal.scenario_simulator import ScenarioSimulator


def run_tests():
    passed = 0
    total = 0

    def assert_true(cond, msg):
        nonlocal passed, total
        total += 1
        if not cond:
            print(f"  [FAIL] {msg}")
            raise AssertionError(msg)
        else:
            print(f"  [PASS] {msg}")
            passed += 1

    print("\n=== 1. TESTING APPROACH DEMAND CALCULATOR ===")
    app_n = ApproachConfig(
        approach_id="app_north",
        direction=MovementDirection.NORTH,
        name="Northbound Approach",
        road_id="road_n",
        job_id="demo-cam_5",
        permitted_movements=["THROUGH", "RIGHT_TURN"],
        has_pedestrian_crossing=True,
        pedestrian_clearance_sec=12.0,
        lane_count=2
    )

    config = IntersectionConfig(
        intersection_id="test_int",
        name="Test Intersection",
        intersection_type="four-leg signalized",
        approaches=[app_n],
        phases=[],
        phase_constraints=[],
        cycle_length_sec=120,
        min_cycle_sec=60,
        max_cycle_sec=180,
        fixed_plan_green_times={"phase_1": 30},
        demand_weight_queue=0.45,
        demand_weight_arrival=0.25,
        demand_weight_waiting=0.20,
        demand_weight_occupancy=0.10,
        max_queue_veh=30,
        max_arrival_veh_per_min=60,
        max_wait_sec=90,
        max_occupancy_pct=100.0,
        stale_data_threshold_sec=120.0
    )

    calc = ApproachDemandCalculator(config)
    road_data = {
        "queue": 15,
        "flow_volume": 30.0,
        "average_wait": 45.0,
        "occupancy": 50.0,
        "current_moving": 10,
        "current_stationary": 15,
        "average_speed": 28.5,
        "traffic_state": "MODERATE",
        "last_updated": datetime.now(timezone.utc).isoformat()
    }

    demand = calc.calculate_approach_demand(app_n, road_data)
    assert_true(demand is not None, "Demand object created")
    assert_true(demand.approach_id == "app_north", "Approach ID matches")
    assert_true(0.45 <= demand.demand_score <= 0.55, f"Demand score expected ~0.50, got {demand.demand_score}")
    assert_true(demand.data_quality == DataQualityFlag.GOOD, "Data quality marked GOOD")

    # Empty data test
    empty_demand = calc.calculate_approach_demand(app_n, None)
    assert_true(empty_demand.demand_score in [0.0, None], "Empty data produces 0.0 or None demand score")
    assert_true(empty_demand.data_quality in [DataQualityFlag.MISSING, DataQualityFlag.STALE], "Empty data quality flagged")

    # Starvation test
    boosted = calc.calculate_approach_demand(app_n, road_data, time_since_served_sec=150.0)
    assert_true(boosted.starvation_boost_applied is True, "Starvation boost applied after 150s")
    assert_true(boosted.demand_score > demand.demand_score, "Boosted score is higher than normal score")

    print("\n=== 2. TESTING SIGNAL PHASE VALIDATOR ===")
    app_s = ApproachConfig(approach_id="app_s", direction=MovementDirection.SOUTH, name="South", permitted_movements=["THROUGH"], has_pedestrian_crossing=True, pedestrian_clearance_sec=12.0, lane_count=2)
    app_e = ApproachConfig(approach_id="app_e", direction=MovementDirection.EAST, name="East", permitted_movements=["THROUGH"], has_pedestrian_crossing=True, pedestrian_clearance_sec=10.0, lane_count=2)
    app_w = ApproachConfig(approach_id="app_w", direction=MovementDirection.WEST, name="West", permitted_movements=["THROUGH"], has_pedestrian_crossing=True, pedestrian_clearance_sec=10.0, lane_count=2)

    p1 = SignalPhase(
        phase_id="p1", phase_name="Phase 1: N-S",
        served_approaches=["app_north", "app_s"],
        permitted_movements=["THROUGH"],
        conflicting_phases=["p2"],
        min_green_sec=15.0, max_green_sec=60.0, amber_sec=4.0, all_red_sec=2.0,
        pedestrian_walk_sec=12.0, sequence_position=1
    )
    p2 = SignalPhase(
        phase_id="p2", phase_name="Phase 2: E-W",
        served_approaches=["app_e", "app_w"],
        permitted_movements=["THROUGH"],
        conflicting_phases=["p1"],
        min_green_sec=15.0, max_green_sec=60.0, amber_sec=4.0, all_red_sec=2.0,
        pedestrian_walk_sec=10.0, sequence_position=2
    )

    four_leg = IntersectionConfig(
        intersection_id="int_4leg",
        name="Four-Leg Intersection",
        intersection_type="four-leg signalized",
        approaches=[app_n, app_s, app_e, app_w],
        phases=[p1, p2],
        phase_constraints=[{"phase_a": "p1", "phase_b": "p2", "reason": "Cross-traffic conflict"}],
        cycle_length_sec=120, min_cycle_sec=60, max_cycle_sec=180,
        fixed_plan_name="Standard Fixed Plan",
        fixed_plan_green_times={"p1": 30.0, "p2": 30.0},
        stale_data_threshold_sec=120.0
    )

    validator = SignalPhaseValidator(four_leg)
    is_valid, errors = validator.validate_configuration()
    if not is_valid:
        print("Validation errors:", errors)
    assert_true(is_valid is True, "Intersection configuration valid")

    # Min green violation test (10s < 15s)
    is_v, errs = validator.validate_proposed_phase(p1, proposed_green_sec=10.0)
    assert_true(is_v is False, "Enforces minimum green bound (<15s rejected)")

    # Max green violation test (75s > 60s)
    is_v, errs = validator.validate_proposed_phase(p1, proposed_green_sec=75.0)
    assert_true(is_v is False, "Enforces maximum green bound (>60s rejected)")

    # Valid green duration (35s)
    is_v, errs = validator.validate_proposed_phase(p1, proposed_green_sec=35.0)
    assert_true(is_v is True, "Valid green time (35s) approved")

    print("\n=== 3. TESTING RECOMMENDATION ENGINE ===")
    rec_engine = SignalRecommendationEngine()
    now_iso = datetime.now(timezone.utc).isoformat()
    approach_data = {
        "app_north": {"queue": 26, "flow_volume": 48.0, "average_wait": 52.0, "occupancy": 82.0, "last_updated": now_iso},
        "app_s": {"queue": 22, "flow_volume": 42.0, "average_wait": 44.0, "occupancy": 75.0, "last_updated": now_iso},
        "app_e": {"queue": 3, "flow_volume": 8.0, "average_wait": 10.0, "occupancy": 15.0, "last_updated": now_iso},
        "app_w": {"queue": 2, "flow_volume": 6.0, "average_wait": 8.0, "occupancy": 12.0, "last_updated": now_iso},
    }

    rec = rec_engine.generate_recommendation(four_leg, approach_data)
    assert_true(rec is not None, "Generated recommendation object")
    assert_true(rec.recommended_phase.phase_id == "p1", "Selected Phase 1 (N-S) under heavy corridor demand")
    assert_true(rec.recommended_phase.proposed_green_sec > 30.0, f"Proposed green extended ({rec.recommended_phase.proposed_green_sec}s > 30s baseline)")
    assert_true(rec.recommended_phase.proposed_green_sec <= 60.0, "Proposed green capped by safety maximum (<=60s)")
    assert_true(rec.constraint_validation_passed is True, "Constraint validation passed")

    print("\n=== 4. TESTING SCENARIO SIMULATOR ===")
    sim = ScenarioSimulator()
    scenario = SimulationScenario(
        intersection_id="int_4leg",
        scenario_type=SimulationScenarioType.PEAK_TRAFFIC,
        simulation_duration_sec=3600,
        random_seed=42
    )
    sim_res = sim.run_simulation(four_leg, scenario)
    assert_true(sim_res is not None, "Simulation completed")
    assert_true(sim_res.delay_improvement_pct is not None and sim_res.delay_improvement_pct > 0, f"Delay improved by {sim_res.delay_improvement_pct:.1f}%")
    assert_true(sim_res.queue_improvement_pct is not None and sim_res.queue_improvement_pct >= 0, f"Queue reduced by {sim_res.queue_improvement_pct:.1f}%")
    assert_true(sim_res.throughput_improvement_pct is not None and sim_res.throughput_improvement_pct >= 0, f"Throughput increased by {sim_res.throughput_improvement_pct:.1f}%")

    print(f"\n==========================================")
    print(f" ALL {passed}/{total} SIGNAL OPTIMIZATION TESTS PASSED!")
    print(f"==========================================\n")


if __name__ == "__main__":
    run_tests()
