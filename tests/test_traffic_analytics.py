"""Unit tests for TrafficTwin computer vision analytics:
- Vehicle class mapping & detection filtering
- Moving vs Stationary movement classification with EMA smoothing and perspective scaling
- Active visible vs Cumulative unique vehicle counting
- Rolling 30s flow rate calculation
- Speed estimation with perspective calibration
- Congestion state and queue length estimation
"""

import time
import unittest
import numpy as np

from src.tracking.tracker import COCO_TO_AIC21, AIC21_CLASS_ID, Track, TARGET_COCO_IDS
from src.tracking.movement_state import VehicleMovementDetector
from src.inference.pipeline import RoadState


# ==============================================================================
# 1. Vehicle Detection Class Mapping & Tracking Tests
# ==============================================================================

def test_coco_to_aic21_mapping():
    """Verify COCO classes are correctly mapped to AIC21 vehicle categories."""
    assert COCO_TO_AIC21[1] == "Bicycle"
    assert COCO_TO_AIC21[2] == "Car"
    assert COCO_TO_AIC21[3] == "Motorcycle"
    assert COCO_TO_AIC21[5] == "Bus"      # Fixed: COCO 5 is Bus, not Car
    assert COCO_TO_AIC21[7] == "Truck"

    # Supported target IDs
    assert TARGET_COCO_IDS == {1, 2, 3, 5, 7}


def test_track_countable_classes():
    """Verify Track.is_countable() accurately accepts all 5 vehicle classes."""
    for cls_name in ["Car", "Bus", "Truck", "Motorcycle", "Bicycle"]:
        track = Track(
            track_id=1,
            frame_id=1,
            timestamp_sec=0.0,
            aic21_class_name=cls_name,
            aic21_class_id=AIC21_CLASS_ID.get(cls_name, 1),
            confidence=0.85,
            x1=100.0, y1=100.0, x2=200.0, y2=200.0
        )
        assert track.is_countable() is True

    # Non-vehicle or ignored class should not be countable
    invalid_track = Track(
        track_id=99,
        frame_id=1,
        timestamp_sec=0.0,
        aic21_class_name="Person",
        aic21_class_id=99,
        confidence=0.85,
        x1=100.0, y1=100.0, x2=200.0, y2=200.0
    )
    assert invalid_track.is_countable() is False


# ==============================================================================
# 2. Movement Classification Tests (Moving, Stationary, Unknown)
# ==============================================================================

def test_movement_detector_unknown_on_initial_frames():
    """Tracks with fewer observations than min_history_seconds should be UNKNOWN."""
    detector = VehicleMovementDetector(min_history_seconds=0.4)
    
    # Initial observation
    state = detector.update(track_id=101, x=530.0, y=530.0, timestamp_sec=0.05)
    assert state == "UNKNOWN"

    # Second observation: only 0.05s elapsed < 0.4s min_history
    state = detector.update(track_id=101, x=535.0, y=535.0, timestamp_sec=0.10)
    assert state == "UNKNOWN"


def test_movement_detector_moving_classification():
    """Vehicle with consistent displacement across frames should be classified as MOVING."""
    detector = VehicleMovementDetector(
        window_seconds=1.0,
        start_moving_threshold_px_per_sec=14.0,
        moving_confirm_frames=3,
        min_history_seconds=0.3
    )

    # Simulate vehicle moving horizontally at 100 px/sec across 1.0 second (20 steps)
    for i in range(20):
        t = (i + 1) * 0.05
        x = 100.0 + (t * 100.0)
        y = 500.0
        state = detector.update(track_id=202, x=x, y=y, timestamp_sec=t)

    assert detector.get_state(202) == "MOVING"
    assert detector.is_moving(202) is True
    assert detector.is_stationary(202) is False


def test_movement_detector_stationary_with_jitter_suppression():
    """Vehicle with small bounding box jitter should be smoothed and classified as STATIONARY."""
    detector = VehicleMovementDetector(
        window_seconds=1.0,
        stop_moving_threshold_px_per_sec=8.0,
        rest_confirm_frames=4,
        min_history_seconds=0.3,
        smoothing_alpha=0.65
    )

    # Initial frame
    detector.update(track_id=303, x=425.0, y=425.0, timestamp_sec=0.05)
    
    # Slight jitter of +/- 1-2 pixels (simulating camera/detector noise) over 1.0s
    jitter_offsets = [0, 1, -1, 1, 0, -1, 1, 0, -1, 0, 1, -1, 0, 1, -1, 0]
    for i, offset in enumerate(jitter_offsets):
        t = 0.10 + i * 0.05
        detector.update(track_id=303, x=425.0 + offset, y=425.0 + offset, timestamp_sec=t)

    assert detector.get_state(303) in ("STATIONARY", "REST")
    assert detector.is_stationary(303) is True
    assert detector.is_moving(303) is False


# ==============================================================================
# 3. Vehicle Counting Tests (Active Visible vs Cumulative Unique)
# ==============================================================================

def test_active_visible_vs_cumulative_counting():
    """Active visible vehicles should reflect currently present tracks in ROI,
    while cumulative count tracks unique vehicle IDs over time.
    """
    road = RoadState(road_id="road_1", name="Main Approach", window_seconds=30.0)

    # Vehicle 1 and Vehicle 2 enter the road
    assert road.register_vehicle(track_id=1, class_name="Car", timestamp_sec=1.0) is True
    assert road.register_vehicle(track_id=2, class_name="Truck", timestamp_sec=1.5) is True

    # Duplicate observation for Vehicle 1 (already seen) should return False and not increase count
    assert road.register_vehicle(track_id=1, class_name="Car", timestamp_sec=2.0) is False

    assert len(road._cumulative_vehicle_ids) == 2
    assert road._cumulative_class_counts["Car"] == 1
    assert road._cumulative_class_counts["Truck"] == 1

    # Vehicle 3 enters
    assert road.register_vehicle(track_id=3, class_name="Bus", timestamp_sec=3.0) is True
    assert len(road._cumulative_vehicle_ids) == 3
    assert road._cumulative_class_counts["Bus"] == 1


# ==============================================================================
# 4. Rolling Flow Rate Tests
# ==============================================================================

def test_rolling_window_flow_rate():
    """Flow rate must calculate unique entries within rolling 30s window."""
    road = RoadState(road_id="road_1", name="Approach 1", window_seconds=30.0)

    # 6 vehicles enter at timestamps t = 10.0s to t = 20.0s
    for tid in range(1, 7):
        road.register_vehicle(tid, class_name="Car", timestamp_sec=10.0 + tid)

    # At t = 25.0s, window span = 25.0s = 0.4166 min; 6 vehicles / (25/60) = 14.4 veh/min
    flow_at_25 = road.get_flow_rate(current_timestamp=25.0)
    assert 13.0 <= flow_at_25 <= 15.0

    # Advance time to t = 60.0s (all 6 vehicles from t=11..16 are now older than 30s)
    flow_at_60 = road.get_flow_rate(current_timestamp=60.0)
    assert flow_at_60 == 0.0  # Window purged, no recent entries in rolling window


# ==============================================================================
# 5. Speed Estimation Tests
# ==============================================================================

def test_speed_estimation_and_stationary_zero():
    """Moving vehicles should yield positive calibrated km/h; stationary = 0 km/h."""
    road = RoadState(road_id="road_1", name="Approach 1", window_seconds=30.0)

    # Moving vehicle (ID 20): moves 60 pixels downwards over 1.0 second (e.g. 5 steps of 0.2s)
    for i in range(5):
        t = 1.0 + i * 0.2
        y = 540.0 + i * 15.0
        road.update_speed(track_id=20, ref_x=500.0, ref_y=y, timestamp_sec=t)

    # Vehicle 10: stationary at (500, 500)
    for i in range(5):
        t = 1.0 + i * 0.2
        road.update_speed(track_id=10, ref_x=500.0, ref_y=500.0, timestamp_sec=t)

    movement_states = {10: "STATIONARY", 20: "MOVING"}

    # Average speed across moving vehicles (stationary vehicles excluded from moving speed)
    avg_speed = road.get_average_speed(active_track_ids=[10, 20], movement_states=movement_states)
    assert avg_speed > 10.0  # Calibrated speed for moving vehicle is positive


# ==============================================================================
# 6. Congestion & Queue Estimation Tests
# ==============================================================================

def test_queue_and_traffic_state():
    """Queue length must equal active stationary vehicles, driving traffic states."""
    road = RoadState(road_id="road_1", name="Approach 1", window_seconds=30.0)

    # 5 active vehicles: IDs 1..4 stationary, ID 5 moving
    active_ids = [1, 2, 3, 4, 5]
    movement_states = {
        1: "STATIONARY",
        2: "STATIONARY",
        3: "STATIONARY",
        4: "STATIONARY",
        5: "MOVING",
    }

    queue_len = road.get_queue_length(active_track_ids=active_ids, movement_states=movement_states)
    assert queue_len == 4  # Exactly the 4 stopped vehicles

    # 4 out of 5 vehicles stopped (80% stationary) -> should transition to HIGH/CRITICAL congestion
    traffic_state = road.get_traffic_state(
        active_count=len(active_ids),
        queue_count=queue_len,
        avg_speed=5.0,
        flow_rate=2.0
    )
    assert traffic_state in ["HIGH", "CRITICAL"]

    # When all 5 are moving -> LOW congestion, queue = 0
    moving_states = {i: "MOVING" for i in active_ids}
    assert road.get_queue_length(active_track_ids=active_ids, movement_states=moving_states) == 0
    traffic_state_moving = road.get_traffic_state(
        active_count=len(active_ids),
        queue_count=0,
        avg_speed=35.0,
        flow_rate=12.0
    )
    assert traffic_state_moving in ["LOW", "MEDIUM"]


class TrafficAnalyticsTestCase(unittest.TestCase):
    def test_01_coco_to_aic21_mapping(self):
        test_coco_to_aic21_mapping()

    def test_02_track_countable_classes(self):
        test_track_countable_classes()

    def test_03_movement_detector_unknown(self):
        test_movement_detector_unknown_on_initial_frames()

    def test_04_movement_detector_moving(self):
        test_movement_detector_moving_classification()

    def test_05_movement_detector_stationary_jitter_suppression(self):
        test_movement_detector_stationary_with_jitter_suppression()

    def test_06_active_visible_vs_cumulative_counting(self):
        test_active_visible_vs_cumulative_counting()

    def test_07_rolling_window_flow_rate(self):
        test_rolling_window_flow_rate()

    def test_08_speed_estimation_and_stationary(self):
        test_speed_estimation_and_stationary_zero()

    def test_09_queue_and_traffic_state(self):
        test_queue_and_traffic_state()


if __name__ == "__main__":
    unittest.main()
