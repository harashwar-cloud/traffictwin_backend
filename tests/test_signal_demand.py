"""
tests/test_signal_demand.py
---------------------------
Unit tests for ApproachDemandCalculator using standard unittest.
"""

import unittest
from datetime import datetime, timezone, timedelta
from src.signal.demand_calculator import ApproachDemandCalculator
from api.signal_models import ApproachConfig, IntersectionConfig, DataQualityFlag, MovementDirection


class TestSignalDemand(unittest.TestCase):

    def setUp(self):
        self.sample_approach = ApproachConfig(
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

        self.sample_config = IntersectionConfig(
            intersection_id="test_int",
            name="Test Intersection",
            intersection_type="four-leg signalized",
            approaches=[self.sample_approach],
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

    def test_demand_calculator_basic(self):
        calc = ApproachDemandCalculator(self.sample_config)
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
        demand = calc.calculate_approach_demand(self.sample_approach, road_data)
        self.assertIsNotNone(demand)
        self.assertEqual(demand.approach_id, "app_north")
        self.assertEqual(demand.direction, MovementDirection.NORTH)
        self.assertTrue(0.0 <= demand.demand_score <= 1.0)
        self.assertAlmostEqual(demand.demand_score, 0.50, delta=0.06)
        self.assertEqual(demand.data_quality, DataQualityFlag.GOOD)
        self.assertGreater(demand.confidence, 0.8)

    def test_demand_calculator_empty_data(self):
        calc = ApproachDemandCalculator(self.sample_config)
        demand = calc.calculate_approach_demand(self.sample_approach, None)
        self.assertIsNotNone(demand)
        self.assertIn(demand.data_quality, [DataQualityFlag.MISSING, DataQualityFlag.STALE])
        self.assertIn(demand.demand_score, [0.0, None])

    def test_demand_calculator_starvation_boost(self):
        calc = ApproachDemandCalculator(self.sample_config)
        road_data = {
            "queue": 5,
            "flow_volume": 10.0,
            "average_wait": 20.0,
            "occupancy": 20.0,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        demand = calc.calculate_approach_demand(self.sample_approach, road_data, time_since_served_sec=150.0)
        self.assertTrue(demand.starvation_boost_applied)
        base_demand = calc.calculate_approach_demand(self.sample_approach, road_data, time_since_served_sec=10.0)
        self.assertGreater(demand.demand_score, base_demand.demand_score)

    def test_demand_calculator_stale_data(self):
        calc = ApproachDemandCalculator(self.sample_config)
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=200)).isoformat()
        road_data = {
            "queue": 20,
            "flow_volume": 40.0,
            "average_wait": 60.0,
            "occupancy": 70.0,
            "last_updated": stale_time
        }
        demand = calc.calculate_approach_demand(self.sample_approach, road_data)
        self.assertEqual(demand.data_quality, DataQualityFlag.STALE)
        self.assertLess(demand.confidence, 0.6)


if __name__ == "__main__":
    unittest.main()
