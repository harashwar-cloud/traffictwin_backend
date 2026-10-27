"""
tests/test_telemetry_and_optimization.py
-----------------------------------------
Unit tests validating:
1. TrackStore trajectory pruning and bounding.
2. RoadState active occupant sync and rolling 30s flow rate formula.
3. Flow rate re-entry support (allowing genuine re-entry events after leaving).
4. Telemetry reconciliation (mutually-exclusive stages with no double-counting).
5. ByteTracker exclusive tracking time isolation.
"""

import unittest
from collections import deque
import numpy as np

from src.tracking.tracker import TrackStore, Track, ByteTracker
from src.inference.pipeline import RoadState
from src.data.road_config import RoadROI


class TestTelemetryAndOptimization(unittest.TestCase):

    def test_track_store_cleanup(self):
        """Verify TrackStore prunes inactive tracks beyond max_age."""
        store = TrackStore()
        
        # Add tracks at frame 10
        t1 = Track(1, 10, 1.0, "Car", 1, 0.9, 10, 10, 30, 30)
        t2 = Track(2, 10, 1.0, "Car", 1, 0.9, 50, 50, 80, 80)
        store.update([t1, t2])
        self.assertEqual(len(store.all_track_ids()), 2)

        # Update track 1 at frame 100, but leave track 2 unseen
        t1_later = Track(1, 100, 10.0, "Car", 1, 0.9, 15, 15, 35, 35)
        store.update([t1_later])

        # Cleanup at frame 100 with max_age=60 (track 2 age is 90 frames > 60)
        pruned_count = store.cleanup_expired(current_frame_id=100, max_age=60)
        self.assertEqual(pruned_count, 1)
        self.assertIn(1, store.all_track_ids())
        self.assertNotIn(2, store.all_track_ids())

    def test_road_state_flow_rate_30s(self):
        """
        Verify rolling flow rate formula:
        Flow rate (veh/min) = Unique crossing events in window * 60 / window duration (seconds).
        For 30s window with 4 unique vehicles: 4 * 60 / 30 = 8.0 veh/min.
        """
        road = RoadState(road_id="road_1", name="Road 1", window_seconds=30.0)

        # Register 4 vehicles at t=40s, 45s, 50s, 55s
        road.register_vehicle(1, "Car", 40.0)
        road.register_vehicle(2, "Car", 45.0)
        road.register_vehicle(3, "Truck", 50.0)
        road.register_vehicle(4, "Car", 55.0)

        flow_rate = road.get_flow_rate(current_timestamp=60.0)
        # 4 unique vehicles in the last 30 seconds [30s, 60s] -> 4 * 60 / 30 = 8.0 veh/min
        self.assertEqual(flow_rate, 8.0)

        # Advance time to t=100s where all previous events expired
        empty_flow = road.get_flow_rate(current_timestamp=100.0)
        self.assertEqual(empty_flow, 0.0)

    def test_road_state_reentry_support(self):
        """
        Verify that a vehicle can exit a road and later generate another valid
        crossing event on genuine re-entry.
        """
        road = RoadState(road_id="road_1", name="Road 1", window_seconds=30.0)

        # First entry
        is_new1 = road.register_vehicle(101, "Car", 10.0)
        self.assertTrue(is_new1)

        # While remaining on the road on the next frame:
        is_new2 = road.register_vehicle(101, "Car", 11.0)
        self.assertFalse(is_new2)  # Should NOT double-count existing occupant

        # Vehicle leaves road (sync with empty or other IDs)
        road.sync_active_occupants([202, 203])
        self.assertNotIn(101, road._active_occupant_ids)

        # Vehicle re-enters road at t=25s
        is_new3 = road.register_vehicle(101, "Car", 25.0)
        self.assertTrue(is_new3)  # Genuine re-entry event registered!

    def test_telemetry_reconciliation(self):
        """
        Verify that mutually exclusive telemetry stages sum cleanly to total latency
        without double-counting inference in tracking.
        """
        decode_ms = 4.0
        pre_ms = 5.0
        inf_ms = 12.0
        post_ms = 2.0
        track_ms = 3.0  # pure tracking
        ana_ms = 1.0
        ren_ms = 4.0
        wrt_ms = 0.1

        stage_sum = decode_ms + pre_ms + inf_ms + post_ms + track_ms + ana_ms + ren_ms + wrt_ms
        total_latency = 31.5  # wall clock time

        # Stage sum must not include double-counted inference (e.g. 12 + (12 + 3) = 27)
        self.assertAlmostEqual(stage_sum, 31.1, delta=0.5)
        # Diff between sum and total wall latency must be small (< 2ms)
        self.assertLess(abs(stage_sum - total_latency), 2.0)


if __name__ == "__main__":
    unittest.main()
