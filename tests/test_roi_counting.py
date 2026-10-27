"""
tests/test_roi_counting.py
--------------------------
Comprehensive unit tests for:
  - Anchor point calculation (bottom-center for road contact)
  - Polygon containment with O(1) bounding-box pre-filtering
  - Dual-margin boundary hysteresis and jitter suppression
  - Temporary track loss grace period and stale track expiration
  - Invariant validation: Current count = Initial occupancy + Entries - Exits
  - Vehicle re-entry without permanent lock
"""

import unittest
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from src.data.road_config import RoadROI
from src.counting.roi_counter import (
    ROIVehicleCounter,
    TrackROIState,
    ROICountSnapshot,
)


@dataclass
class MockTrack:
    track_id: int
    frame_id: int
    timestamp_sec: float
    aic21_class_name: str
    aic21_class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def is_countable(self) -> bool:
        return self.aic21_class_name in ("Car", "Bus", "Truck", "Motorcycle", "Bicycle")

    @property
    def bottom_center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, float(self.y2))


class TestROICounting(unittest.TestCase):
    def setUp(self):
        # Define a square ROI polygon from (100, 100) to (500, 500)
        self.roi_vertices = [(100, 100), (500, 100), (500, 500), (100, 500)]
        self.road = RoadROI(
            road_id="road_1",
            name="Main Road",
            color=(0, 255, 0),
            vertices=self.roi_vertices,
        )
        self.counter = ROIVehicleCounter(
            road_rois=[self.road],
            reference_point="bottom_center",
            boundary_tolerance=8.0,
            entry_confirm_frames=2,
            exit_confirm_frames=3,
            lost_grace_frames=5,
        )

    def test_01_anchor_point_calculation(self):
        """Verify representative anchor point is bottom-center of bounding box."""
        # Box from x=150..250, y=200..300
        # Bottom center should be x=200, y=300
        cx, cy = self.counter._get_anchor_point(150.0, 200.0, 250.0, 300.0)
        self.assertEqual(cx, 200.0)
        self.assertEqual(cy, 300.0)

    def test_02_containment_and_envelope_prefiltering(self):
        """Points clearly outside bounding box envelope should be rejected in O(1)."""
        # Point inside: (300, 300)
        is_in, rid, dist = self.counter._test_containment(300.0, 300.0, margin=0.0)
        self.assertTrue(is_in)
        self.assertEqual(rid, "road_1")
        self.assertGreater(dist, 0.0)

        # Point far outside: (900, 900)
        is_in, rid, dist = self.counter._test_containment(900.0, 900.0, margin=0.0)
        self.assertFalse(is_in)

    def test_03_prompt_counting_example(self):
        """
        Verify the exact scenario from the prompt table:
          - Three vehicles enter: current = 3
          - One vehicle moves inside: current = 3
          - One vehicle exits: current = 2
          - Two new vehicles enter: current = 4
          - Two vehicles exit: current = 2
        """
        counter = ROIVehicleCounter(
            road_rois=[self.road],
            entry_confirm_frames=1,
            exit_confirm_frames=1,
            boundary_tolerance=5.0,
        )

        # Event 1: Three vehicles enter (IDs 1, 2, 3) inside ROI at frame 10 (t=1.0)
        # BBoxes inside ROI (y2=300 is inside 100..500)
        tracks_ev1 = [
            MockTrack(1, 10, 1.0, "Car", 1, 0.9, 150, 200, 250, 300),
            MockTrack(2, 10, 1.0, "Car", 1, 0.9, 250, 200, 350, 300),
            MockTrack(3, 10, 1.0, "Car", 1, 0.9, 350, 200, 450, 300),
        ]
        snap = counter.update(10, 1.0, tracks_ev1)
        self.assertEqual(snap.current_count, 3)
        self.assertEqual(snap.cumulative_entries, 3)
        self.assertEqual(snap.cumulative_exits, 0)
        self.assertTrue(snap.invariant_verified)

        # Event 2: One vehicle moves inside ROI (no count change)
        tracks_ev2 = [
            MockTrack(1, 11, 1.1, "Car", 1, 0.9, 160, 210, 260, 310),
            MockTrack(2, 11, 1.1, "Car", 1, 0.9, 250, 200, 350, 300),
            MockTrack(3, 11, 1.1, "Car", 1, 0.9, 350, 200, 450, 300),
        ]
        snap = counter.update(11, 1.1, tracks_ev2)
        self.assertEqual(snap.current_count, 3)
        self.assertEqual(snap.cumulative_entries, 3)
        self.assertEqual(snap.cumulative_exits, 0)
        self.assertTrue(snap.invariant_verified)

        # Event 3: One vehicle exits (Vehicle 3 moves outside to y2=600 > 500)
        tracks_ev3 = [
            MockTrack(1, 12, 1.2, "Car", 1, 0.9, 160, 210, 260, 310),
            MockTrack(2, 12, 1.2, "Car", 1, 0.9, 250, 200, 350, 300),
            MockTrack(3, 12, 1.2, "Car", 1, 0.9, 350, 500, 450, 600),  # OUTSIDE
        ]
        snap = counter.update(12, 1.2, tracks_ev3)
        self.assertEqual(snap.current_count, 2)
        self.assertEqual(snap.cumulative_entries, 3)
        self.assertEqual(snap.cumulative_exits, 1)
        self.assertTrue(snap.invariant_verified)

        # Event 4: Two new vehicles enter (IDs 4 and 5)
        tracks_ev4 = [
            MockTrack(1, 13, 1.3, "Car", 1, 0.9, 160, 210, 260, 310),
            MockTrack(2, 13, 1.3, "Car", 1, 0.9, 250, 200, 350, 300),
            MockTrack(4, 13, 1.3, "Truck", 2, 0.9, 200, 200, 300, 300),
            MockTrack(5, 13, 1.3, "Car", 1, 0.9, 300, 200, 400, 300),
        ]
        snap = counter.update(13, 1.3, tracks_ev4)
        self.assertEqual(snap.current_count, 4)
        self.assertEqual(snap.cumulative_entries, 5)
        self.assertEqual(snap.cumulative_exits, 1)
        self.assertTrue(snap.invariant_verified)

        # Event 5: Two vehicles exit (IDs 1 and 2 move outside to y2=700)
        tracks_ev5 = [
            MockTrack(1, 14, 1.4, "Car", 1, 0.9, 160, 600, 260, 700),  # OUTSIDE
            MockTrack(2, 14, 1.4, "Car", 1, 0.9, 250, 600, 350, 700),  # OUTSIDE
            MockTrack(4, 14, 1.4, "Truck", 2, 0.9, 200, 200, 300, 300), # INSIDE
            MockTrack(5, 14, 1.4, "Car", 1, 0.9, 300, 200, 400, 300),   # INSIDE
        ]
        snap = counter.update(14, 1.4, tracks_ev5)
        self.assertEqual(snap.current_count, 2)
        self.assertEqual(snap.cumulative_entries, 5)
        self.assertEqual(snap.cumulative_exits, 3)
        self.assertTrue(snap.invariant_verified)

    def test_04_boundary_hysteresis_and_jitter_suppression(self):
        """
        Oscillating point within boundary tolerance (+/- 3px around border y=500)
        should NOT trigger repeated entry/exit transitions.
        """
        counter = ROIVehicleCounter(
            road_rois=[self.road],
            boundary_tolerance=8.0,
            entry_confirm_frames=2,
            exit_confirm_frames=3,
        )

        # Frame 1: Vehicle 10 enters ROI (y2=450, comfortably inside)
        counter.update(1, 0.1, [MockTrack(10, 1, 0.1, "Car", 1, 0.9, 200, 350, 300, 450)])
        # Frame 2: Confirmed inside
        snap = counter.update(2, 0.2, [MockTrack(10, 2, 0.2, "Car", 1, 0.9, 200, 350, 300, 450)])
        self.assertEqual(snap.current_count, 1)
        self.assertEqual(snap.cumulative_entries, 1)
        self.assertEqual(snap.cumulative_exits, 0)

        # Vehicle moves right to the boundary (y=500) and jitters: y2 = 498, 502, 497, 503, 499, 502
        # None of these drop below y=508 (dist < -8px), so exit is NOT triggered!
        jitter_y = [498, 502, 497, 503, 499, 502, 498, 504]
        for f, y in enumerate(jitter_y, start=3):
            snap = counter.update(f, f * 0.1, [MockTrack(10, f, f * 0.1, "Car", 1, 0.9, 200, y - 50, 300, y)])
            # Count must remain exactly 1, no exits triggered!
            self.assertEqual(snap.current_count, 1, f"Failed at frame {f} with y={y}")
            self.assertEqual(snap.cumulative_entries, 1)
            self.assertEqual(snap.cumulative_exits, 0)
            self.assertTrue(snap.invariant_verified)

    def test_05_reentry_support(self):
        """
        A vehicle that exits the ROI and later re-enters must be counted again,
        satisfying Current = Initial + Entries - Exits.
        """
        counter = ROIVehicleCounter(
            road_rois=[self.road],
            boundary_tolerance=5.0,
            entry_confirm_frames=2,
            exit_confirm_frames=2,
        )

        # 1. Vehicle 20 enters ROI (y=300)
        counter.update(1, 0.1, [MockTrack(20, 1, 0.1, "Car", 1, 0.9, 200, 200, 300, 300)])
        snap = counter.update(2, 0.2, [MockTrack(20, 2, 0.2, "Car", 1, 0.9, 200, 200, 300, 300)])
        self.assertEqual(snap.current_count, 1)
        self.assertEqual(snap.cumulative_entries, 1)
        self.assertEqual(snap.cumulative_exits, 0)

        # 2. Vehicle 20 drives outside ROI (y=650) for 2 frames
        counter.update(3, 0.3, [MockTrack(20, 3, 0.3, "Car", 1, 0.9, 200, 550, 300, 650)])
        snap = counter.update(4, 0.4, [MockTrack(20, 4, 0.4, "Car", 1, 0.9, 200, 550, 300, 650)])
        self.assertEqual(snap.current_count, 0)
        self.assertEqual(snap.cumulative_entries, 1)
        self.assertEqual(snap.cumulative_exits, 1)

        # 3. Vehicle 20 makes a loop and re-enters ROI (y=300) for 2 frames
        counter.update(5, 0.5, [MockTrack(20, 5, 0.5, "Car", 1, 0.9, 200, 200, 300, 300)])
        snap = counter.update(6, 0.6, [MockTrack(20, 6, 0.6, "Car", 1, 0.9, 200, 200, 300, 300)])
        self.assertEqual(snap.current_count, 1)
        self.assertEqual(snap.cumulative_entries, 2)  # Re-entry counted!
        self.assertEqual(snap.cumulative_exits, 1)
        self.assertTrue(snap.invariant_verified)

    def test_06_temporary_loss_grace_period_and_expiration(self):
        """
        Temporary detection loss (e.g. 2 frames) should keep the track inside.
        Exceeding grace period (5 frames) must expire track and preserve invariant.
        """
        counter = ROIVehicleCounter(
            road_rois=[self.road],
            entry_confirm_frames=1,
            exit_confirm_frames=2,
            lost_grace_frames=4,
        )

        # Frame 1: Vehicle 30 enters
        snap = counter.update(1, 0.1, [MockTrack(30, 1, 0.1, "Car", 1, 0.9, 200, 200, 300, 300)])
        self.assertEqual(snap.current_count, 1)

        # Frame 2-3: Detector temporarily drops track (empty list) -> within grace period
        counter.update(2, 0.2, [])
        snap = counter.update(3, 0.3, [])
        self.assertEqual(snap.current_count, 1)  # Still counted in grace period!
        self.assertTrue(snap.invariant_verified)

        # Frame 4: Detector recovers track inside -> grace period resets
        snap = counter.update(4, 0.4, [MockTrack(30, 4, 0.4, "Car", 1, 0.9, 200, 200, 300, 300)])
        self.assertEqual(snap.current_count, 1)

        # Frame 5-9: Track missing for 5 frames (> grace 4) -> expires
        for f in range(5, 10):
            snap = counter.update(f, f * 0.1, [])

        # Track should be expired, count should drop to 0, invariant preserved!
        self.assertEqual(snap.current_count, 0)
        self.assertEqual(snap.cumulative_entries, 1)
        self.assertEqual(snap.cumulative_exits, 1)
        self.assertTrue(snap.invariant_verified)

    def test_07_startup_initial_occupancy(self):
        """Vehicles present at analysis start must be tracked as initial occupancy."""
        counter = ROIVehicleCounter(
            road_rois=[self.road],
            entry_confirm_frames=1,
            exit_confirm_frames=1,
            track_initial_occupancy=True,
        )

        # First frame has 2 vehicles already inside
        initial_tracks = [
            MockTrack(50, 1, 0.1, "Car", 1, 0.9, 200, 200, 300, 300),
            MockTrack(51, 1, 0.1, "Truck", 2, 0.9, 300, 200, 400, 300),
        ]
        snap = counter.update(1, 0.1, initial_tracks)
        self.assertEqual(snap.initial_occupancy, 2)
        self.assertEqual(snap.cumulative_entries, 0)
        self.assertEqual(snap.cumulative_exits, 0)
        self.assertEqual(snap.current_count, 2)
        self.assertTrue(snap.invariant_verified)

        # A new vehicle enters at frame 2
        snap = counter.update(2, 0.2, initial_tracks + [
            MockTrack(52, 2, 0.2, "Car", 1, 0.9, 150, 150, 250, 250)
        ])
        self.assertEqual(snap.initial_occupancy, 2)
        self.assertEqual(snap.cumulative_entries, 1)
        self.assertEqual(snap.current_count, 3)
        self.assertTrue(snap.invariant_verified)


if __name__ == "__main__":
    unittest.main()
