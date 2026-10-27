"""
tests/test_signal_recommendation.py
-----------------------------------
Unit tests for SignalRecommendationEngine and ScenarioSimulator using standard unittest.
"""

import unittest
from datetime import datetime, timezone, timedelta
from src.signal.recommendation_engine import SignalRecommendationEngine
from src.signal.scenario_simulator import ScenarioSimulator
from api.signal_models import (
    IntersectionConfig, ApproachConfig, SignalPhase,
    SimulationScenario, SimulationScenarioType, MovementDirection
)


class TestSignalRecommendation(unittest.TestCase):

    def setUp(self):
        self.app_n = ApproachConfig(
            approach_id="app_n", direction=MovementDirection.NORTH, name="North",
            permitted_movements=["THROUGH", "RIGHT_TURN"], has_pedestrian_crossing=True,
            pedestrian_clearance_sec=12.0, lane_count=2
        )
        self.app_s = ApproachConfig(
            approach_id="app_s", direction=MovementDirection.SOUTH, name="South",
            permitted_movements=["THROUGH", "RIGHT_TURN"], has_pedestrian_crossing=True,
            pedestrian_clearance_sec=12.0, lane_count=2
        )
        self.app_e = ApproachConfig(
            approach_id="app_e", direction=MovementDirection.EAST, name="East",
            permitted_movements=["THROUGH", "RIGHT_TURN"], has_pedestrian_crossing=True,
            pedestrian_clearance_sec=10.0, lane_count=2
        )
        self.app_w = ApproachConfig(
            approach_id="app_w", direction=MovementDirection.WEST, name="West",
            permitted_movements=["THROUGH", "RIGHT_TURN"], has_pedestrian_crossing=True,
            pedestrian_clearance_sec=10.0, lane_count=2
        )

        self.p1 = SignalPhase(
            phase_id="p1", phase_name="Phase 1: N-S",
            served_approaches=["app_n", "app_s"],
            permitted_movements=["THROUGH", "RIGHT_TURN"],
            conflicting_phases=["p2"],
            min_green_sec=15.0, max_green_sec=60.0, amber_sec=4.0, all_red_sec=2.0,
            pedestrian_walk_sec=12.0, sequence_position=1
        )
        self.p2 = SignalPhase(
            phase_id="p2", phase_name="Phase 2: E-W",
            served_approaches=["app_e", "app_w"],
            permitted_movements=["THROUGH", "RIGHT_TURN"],
            conflicting_phases=["p1"],
            min_green_sec=15.0, max_green_sec=60.0, amber_sec=4.0, all_red_sec=2.0,
            pedestrian_walk_sec=10.0, sequence_position=2
        )

        self.four_leg = IntersectionConfig(
            intersection_id="int_4leg",
            name="Four-Leg Intersection",
            intersection_type="four-leg signalized",
            approaches=[self.app_n, self.app_s, self.app_e, self.app_w],
            phases=[self.p1, self.p2],
            phase_constraints=[{"phase_a": "p1", "phase_b": "p2", "reason": "Cross-traffic conflict"}],
            cycle_length_sec=120,
            min_cycle_sec=60,
            max_cycle_sec=180,
            fixed_plan_name="Standard Daytime Plan",
            fixed_plan_green_times={"p1": 30.0, "p2": 30.0},
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

    def test_recommendation_engine_heavy_north_south(self):
        engine = SignalRecommendationEngine()
        now_iso = datetime.now(timezone.utc).isoformat()
        
        approach_data = {
            "app_n": {"queue": 25, "flow_volume": 45.0, "average_wait": 50.0, "occupancy": 80.0, "last_updated": now_iso},
            "app_s": {"queue": 20, "flow_volume": 40.0, "average_wait": 40.0, "occupancy": 70.0, "last_updated": now_iso},
            "app_e": {"queue": 3, "flow_volume": 8.0, "average_wait": 10.0, "occupancy": 15.0, "last_updated": now_iso},
            "app_w": {"queue": 2, "flow_volume": 5.0, "average_wait": 8.0, "occupancy": 10.0, "last_updated": now_iso},
        }

        rec = engine.generate_recommendation(self.four_leg, approach_data)
        self.assertIsNotNone(rec)
        self.assertIsNotNone(rec.recommended_phase)
        self.assertEqual(rec.recommended_phase.phase_id, "p1")
        self.assertGreater(rec.recommended_phase.proposed_green_sec, 30.0)
        self.assertLessEqual(rec.recommended_phase.proposed_green_sec, 60.0)
        self.assertTrue(rec.constraint_validation_passed)
        self.assertFalse(rec.fallback_to_baseline)

    def test_recommendation_engine_fallback_on_stale_data(self):
        engine = SignalRecommendationEngine()
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=300)).isoformat()
        
        approach_data = {
            "app_n": {"queue": 25, "flow_volume": 45.0, "last_updated": stale_time},
            "app_s": {"queue": 20, "flow_volume": 40.0, "last_updated": stale_time},
            "app_e": {"queue": 3, "flow_volume": 8.0, "last_updated": stale_time},
            "app_w": {"queue": 2, "flow_volume": 5.0, "last_updated": stale_time},
        }

        rec = engine.generate_recommendation(self.four_leg, approach_data)
        self.assertIsNotNone(rec)
        self.assertTrue(rec.fallback_to_baseline)
        self.assertEqual(rec.recommended_phase.proposed_green_sec, self.four_leg.fixed_plan_green_times["p1"])

    def test_scenario_simulator_peak_traffic(self):
        simulator = ScenarioSimulator()
        scenario = SimulationScenario(
            intersection_id="int_4leg",
            scenario_type=SimulationScenarioType.PEAK_TRAFFIC,
            simulation_duration_sec=3600,
            random_seed=42
        )
        result = simulator.run_simulation(self.four_leg, scenario)
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.fixed_plan_result)
        self.assertIsNotNone(result.adaptive_plan_result)
        self.assertIsNotNone(result.delay_improvement_pct)
        self.assertGreater(result.delay_improvement_pct, 0.0)
        self.assertIsNotNone(result.queue_improvement_pct)
        self.assertGreaterEqual(result.queue_improvement_pct, 0.0)
        self.assertIsNotNone(result.throughput_improvement_pct)
        self.assertGreaterEqual(result.throughput_improvement_pct, 0.0)


if __name__ == "__main__":
    unittest.main()
