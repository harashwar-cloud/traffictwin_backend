"""
tests/test_signal_phases.py
---------------------------
Unit tests for SignalPhaseValidator using standard unittest.
"""

import unittest
from src.signal.phase_validator import SignalPhaseValidator
from api.signal_models import SignalPhase, IntersectionConfig, ApproachConfig, MovementDirection


class TestSignalPhases(unittest.TestCase):

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

        self.config = IntersectionConfig(
            intersection_id="int_test",
            name="Test 4-Leg Intersection",
            intersection_type="four-leg signalized",
            approaches=[self.app_n, self.app_s, self.app_e, self.app_w],
            phases=[self.p1, self.p2],
            phase_constraints=[{"phase_a": "p1", "phase_b": "p2", "reason": "Cross-traffic conflict"}],
            cycle_length_sec=120,
            min_cycle_sec=60,
            max_cycle_sec=180,
            fixed_plan_green_times={"p1": 35.0, "p2": 25.0}
        )

    def test_phase_validator_valid_config(self):
        validator = SignalPhaseValidator(self.config)
        is_valid, errors = validator.validate_configuration()
        self.assertTrue(is_valid, f"Validation failed with: {errors}")
        self.assertEqual(len(errors), 0)

    def test_phase_validator_enforce_min_max_green(self):
        validator = SignalPhaseValidator(self.config)
        
        # Under minimum green (10s < 15s)
        is_valid_under, errs_under = validator.validate_proposed_phase(self.p1, proposed_green_sec=10.0)
        self.assertFalse(is_valid_under)
        self.assertTrue(any("Minimum green" in e for e in errs_under))
        
        # Over maximum green (75s > 60s)
        is_valid_over, errs_over = validator.validate_proposed_phase(self.p1, proposed_green_sec=75.0)
        self.assertFalse(is_valid_over)
        self.assertTrue(any("Maximum green" in e for e in errs_over))
        
        # Valid green (35s)
        is_valid_ok, errs_ok = validator.validate_proposed_phase(self.p1, proposed_green_sec=35.0)
        self.assertTrue(is_valid_ok)
        self.assertEqual(len(errs_ok), 0)

    def test_phase_validator_pedestrian_clearance(self):
        validator = SignalPhaseValidator(self.config)
        is_valid, errs = validator.validate_proposed_phase(self.p1, proposed_green_sec=10.0)
        self.assertFalse(is_valid)


if __name__ == "__main__":
    unittest.main()
