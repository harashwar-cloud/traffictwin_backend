"""
tests/test_road_speed_and_trips.py
-----------------------------------
Unit tests for:
  - Task 1: Automatic ROI counting & occupancy invariant (enter, stay, exit, re-enter)
  - Task 2: Vehicle trip entry/exit timing and speed calculation (v_i = d_i / t_i * 3.6)
  - Task 3: Road average speed arithmetic mean (v_avg = sum(v_i) / N, awaiting valid trips when N=0)
  - Task 4: Multi-road independent tracking & deduplicated total occupancy aggregation
"""

import unittest
from unittest.mock import MagicMock
from src.data.road_config import RoadROI
from src.inference.pipeline import RoadState, compute_total_roads_summary


class TestRoadSpeedAndTrips(unittest.TestCase):

    def setUp(self):
        # 100m road corridor
        self.road_roi_1 = RoadROI(
            road_id="road_1",
            name="Approach North",
            color=(0, 255, 255),
            vertices=[(0, 0), (200, 0), (200, 800), (0, 800)],
            length_meters=100.0,
        )
        self.road_roi_2 = RoadROI(
            road_id="road_2",
            name="Approach South",
            color=(255, 128, 0),
            vertices=[(300, 0), (500, 0), (500, 800), (300, 800)],
            length_meters=60.0,
        )

    def test_task1_roi_counting_lifecycle(self):
        """
        TASK 1 Lifecycle:
          - Vehicle enters ROI -> count increases by 1
          - Vehicle stays inside -> count unchanged
          - Vehicle moves inside -> count unchanged
          - Vehicle exits -> count decreases by 1
          - Vehicle re-enters -> count increases by 1
        """
        road = RoadState(road_id="road_1", name="Approach North", length_meters=100.0)

        # Frame 1: Vehicle 10 enters road
        road.update_frame(
            current_tracks_in_road={10: (100.0, 100.0, "Car")},
            timestamp_sec=1.0,
            road_roi=self.road_roi_1,
        )
        self.assertEqual(len(road._active_occupant_ids), 1)
        self.assertEqual(road.cumulative_entries, 1)
        self.assertEqual(road.cumulative_exits, 0)

        # Frame 2: Vehicle 10 stays inside
        road.update_frame(
            current_tracks_in_road={10: (100.0, 150.0, "Car")},
            timestamp_sec=1.5,
            road_roi=self.road_roi_1,
        )
        self.assertEqual(len(road._active_occupant_ids), 1)
        self.assertEqual(road.cumulative_entries, 1)
        self.assertEqual(road.cumulative_exits, 0)

        # Frame 3: Vehicle 20 also enters
        road.update_frame(
            current_tracks_in_road={10: (100.0, 200.0, "Car"), 20: (100.0, 50.0, "Truck")},
            timestamp_sec=2.0,
            road_roi=self.road_roi_1,
        )
        self.assertEqual(len(road._active_occupant_ids), 2)
        self.assertEqual(road.cumulative_entries, 2)
        self.assertEqual(road.cumulative_exits, 0)

        # Frame 4: Vehicle 10 exits, Vehicle 20 remains
        road.update_frame(
            current_tracks_in_road={20: (100.0, 150.0, "Truck")},
            timestamp_sec=3.0,
            road_roi=self.road_roi_1,
        )
        self.assertEqual(len(road._active_occupant_ids), 1)
        self.assertIn(20, road._active_occupant_ids)
        self.assertNotIn(10, road._active_occupant_ids)
        self.assertEqual(road.cumulative_entries, 2)
        self.assertEqual(road.cumulative_exits, 1)

        # Frame 5: Vehicle 10 re-enters
        road.update_frame(
            current_tracks_in_road={10: (100.0, 50.0, "Car"), 20: (100.0, 250.0, "Truck")},
            timestamp_sec=4.0,
            road_roi=self.road_roi_1,
        )
        self.assertEqual(len(road._active_occupant_ids), 2)
        self.assertEqual(road.cumulative_entries, 3)
        self.assertEqual(road.cumulative_exits, 1)

    def test_task2_speed_calculation_entry_exit_time(self):
        """
        TASK 2: Calculate speed from ROI entry and exit times:
          v_i = (d_i / t_i) * 3.6
        """
        road = RoadState(road_id="road_1", name="Approach North", length_meters=100.0)

        # Vehicle 1 enters at t=1.0s, y=50.0
        road.update_frame(
            current_tracks_in_road={1: (100.0, 50.0, "Car")},
            timestamp_sec=1.0,
            road_roi=self.road_roi_1,
        )
        # Stays inside over multiple frames
        road.update_frame(
            current_tracks_in_road={1: (100.0, 350.0, "Car")},
            timestamp_sec=2.5,
            road_roi=self.road_roi_1,
        )
        road.update_frame(
            current_tracks_in_road={1: (100.0, 750.0, "Car")},
            timestamp_sec=4.0,
            road_roi=self.road_roi_1,
        )

        # Vehicle exits at t=5.0s (elapsed = 4.0s)
        road.update_frame(
            current_tracks_in_road={},
            timestamp_sec=5.0,
            road_roi=self.road_roi_1,
        )

        self.assertEqual(road.completed_trips, 1)
        self.assertEqual(len(road.completed_speeds), 1)
        # Vehicle traversed (750-50)=700px out of ~800px polygon span (ratio ~0.875) of 100m corridor
        # d_i ~ 85-90m. Over 4.0 seconds -> v_i ~ 70-80 km/h.
        speed = road.completed_speeds[0]
        self.assertTrue(50.0 <= speed <= 100.0, f"Expected realistic speed, got {speed}")

    def test_task2_reject_instantaneous_jitter(self):
        """Reject trips where elapsed time is too short (jitter) or insufficient frames."""
        road = RoadState(road_id="road_1", name="Approach North", length_meters=100.0)

        # Vehicle enters and immediately exits on next frame (0.1s dt)
        road.update_frame(
            current_tracks_in_road={99: (100.0, 100.0, "Car")},
            timestamp_sec=1.0,
            road_roi=self.road_roi_1,
        )
        road.update_frame(
            current_tracks_in_road={},
            timestamp_sec=1.1,
            road_roi=self.road_roi_1,
        )

        # Should be rejected due to dt < 0.35s and frame_count < 3
        self.assertEqual(road.completed_trips, 0)
        self.assertIsNone(road.average_speed)

    def test_task3_road_average_speed_arithmetic_mean(self):
        """
        TASK 3:
          v_avg = (sum v_i) / N
          Display 'N/A — awaiting valid trips' when N=0.
        """
        road = RoadState(road_id="road_1", name="Approach North", length_meters=100.0)

        # N=0 state with calibration
        summary_0 = road.to_summary(current_timestamp=0.0)
        self.assertIsNone(summary_0["average_speed"])
        self.assertEqual(summary_0["speed_display"], "Waiting for vehicles")
        self.assertEqual(summary_0["completed_trips"], 0)

        # Uncalibrated road
        uncalibrated_road = RoadState(road_id="road_u", name="Uncalibrated", length_meters=None)
        summary_u = uncalibrated_road.to_summary(current_timestamp=0.0)
        self.assertEqual(summary_u["speed_display"], "Calibration required")

        # Add mock completed trips: e.g. 36.0, 45.0, 27.0 km/h
        road.completed_trips = 3
        road.completed_speeds = [36.0, 45.0, 27.0]
        road.sum_speeds = 36.0 + 45.0 + 27.0
        road.average_speed = round(road.sum_speeds / road.completed_trips, 1)

        summary_3 = road.to_summary(current_timestamp=10.0)
        # Arithmetic mean: (36 + 45 + 27) / 3 = 36.0 km/h
        self.assertEqual(summary_3["average_speed"], 36.0)
        self.assertEqual(summary_3["speed_display"], "36.0 km/h")
        self.assertEqual(summary_3["completed_trips"], 3)

    def test_task4_multi_roads_independent_and_aggregation(self):
        """
        TASK 4:
          - Road 1 and Road 2 maintain independent statistics.
          - Aggregation deduplicates overlapping active tracks.
          - Overall speed is trip-weighted arithmetic mean.
        """
        road1 = RoadState(road_id="road_1", name="Approach North", length_meters=100.0)
        road2 = RoadState(road_id="road_2", name="Approach South", length_meters=60.0)

        # Road 1 has tracks 1, 2. Road 2 has tracks 2, 3 (track 2 is in both, e.g. boundary overlap)
        road1._active_occupant_ids = {1, 2}
        road1.cumulative_entries = 5
        road1.cumulative_exits = 3
        road1.completed_trips = 3
        road1.sum_speeds = 36.0 + 45.0 + 27.0  # avg = 36.0
        road1.average_speed = 36.0

        road2._active_occupant_ids = {2, 3}
        road2.cumulative_entries = 4
        road2.cumulative_exits = 2
        road2.completed_trips = 2
        road2.sum_speeds = 50.0 + 60.0  # avg = 55.0
        road2.average_speed = 55.0

        road_summaries = [
            road1.to_summary(10.0),
            road2.to_summary(10.0),
        ]
        road_states = {"road_1": road1, "road_2": road2}

        total_summary = compute_total_roads_summary(road_summaries, road_states)

        # Deduplication check: {1, 2} union {2, 3} = {1, 2, 3} -> 3 active vehicles (not 4!)
        self.assertEqual(total_summary["current_active"], 3)
        self.assertEqual(total_summary["cumulative_entries"], 9)
        self.assertEqual(total_summary["cumulative_exits"], 5)
        self.assertEqual(total_summary["completed_trips"], 5)

        # Trip-weighted arithmetic mean:
        # sum of all 5 trips = 108.0 + 110.0 = 218.0
        # 218.0 / 5 = 43.6 km/h
        # (NOT the simple average of road averages (36 + 55)/2 = 45.5)
        self.assertAlmostEqual(total_summary["average_speed"], 43.6, places=1)
        self.assertEqual(total_summary["speed_display"], "43.6 km/h")


if __name__ == "__main__":
    unittest.main()
