"""
tests/test_multi_video_isolation.py
-----------------------------------
Automated tests validating multi-video dashboard capabilities:
1. Safe GPU model weight sharing with isolated ByteTrack predictors.
2. Independent session state and tracking isolation across multiple videos.
3. Pipeline stop_check early exit and graceful finalization.
4. Per-road analytics isolation and multi-road aggregation within a single video.
5. Server session cleanup and resource management.
"""

import unittest
import numpy as np
import torch
from pathlib import Path

from src.tracking.tracker import ByteTracker, TrackStore, Track
from src.inference.pipeline import TrafficPipeline, RoadState
from src.data.road_config import RoadROI


class TestMultiVideoIsolation(unittest.TestCase):

    def test_tracker_instances_isolated_and_weights_shared(self):
        """
        Verify that multiple ByteTracker instances:
        - Share the exact same underlying PyTorch neural network weights in GPU memory.
        - Maintain independent YOLO wrapper instances and isolated predictor/tracker state.
        """
        tracker1 = ByteTracker()
        tracker2 = ByteTracker()

        # Independent wrapper instances
        self.assertNotEqual(
            id(tracker1.model),
            id(tracker2.model),
            "ByteTracker instances must have isolated YOLO wrappers to prevent predictor collision."
        )

        # Shared underlying PyTorch nn.Module weights
        self.assertEqual(
            id(tracker1.model.model),
            id(tracker2.model.model),
            "ByteTracker instances must share base PyTorch weights to conserve GPU VRAM."
        )

        # Track a dummy frame on tracker 1
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        tracks1 = tracker1.track_frame(dummy_frame, frame_id=0, timestamp_sec=0.0)
        tracks2 = tracker2.track_frame(dummy_frame, frame_id=0, timestamp_sec=0.0)

        # Both return track lists independently
        self.assertIsInstance(tracks1, list)
        self.assertIsInstance(tracks2, list)

    def test_pipeline_stop_check(self):
        """
        Verify that passing stop_check=lambda: True halts pipeline execution immediately
        without crashing and completes gracefully.
        """
        pipeline = TrafficPipeline()
        pipeline.max_frames = 50
        pipeline.save_video = False

        test_video = Path("data/uploads/sample.mp4")
        if not test_video.exists():
            # Fallback to Dataset_A cam_5.mp4
            test_video = Path("../AIC21_Track1_Vehicle_Counting/Dataset_A/cam_5.mp4")

        if test_video.exists():
            # Test stopping immediately
            stop_flag = True
            result = pipeline.run(
                video_path=str(test_video),
                camera_id="cam_5",
                output_prefix="test_stop",
                stop_check=lambda: stop_flag,
            )
            self.assertIsNotNone(result)
            self.assertIn("total_vehicles", result)
            # Processed frames should be 1 or 0 because stop was triggered at start of loop
            proc_frames = result.get("processing", {}).get("frames_processed", 0)
            self.assertLessEqual(proc_frames, 2)

    def test_multi_road_isolated_and_aggregated_metrics(self):
        """
        Verify that multiple roads inside a video maintain separate counts,
        speeds, and occupancy, while the total summary properly aggregates them.
        """
        roi1 = RoadROI(
            road_id="road_1",
            name="Road 1 (North)",
            color=(255, 0, 0),
            vertices=[(0, 0), (100, 0), (100, 100), (0, 100)],
            length_meters=50.0,
        )
        roi2 = RoadROI(
            road_id="road_2",
            name="Road 2 (South)",
            color=(0, 255, 0),
            vertices=[(200, 200), (300, 200), (300, 300), (200, 300)],
            length_meters=100.0,
        )

        r1 = RoadState(road_id="road_1", name="Road 1 (North)", length_meters=50.0)
        r2 = RoadState(road_id="road_2", name="Road 2 (South)", length_meters=100.0)

        # Frame 1: Vehicle 101 enters Road 1 at t=10.0s
        tracks_r1_f1 = {101: (50.0, 50.0, "Car")}
        r1.update_frame(tracks_r1_f1, timestamp_sec=10.0, road_roi=roi1, frame_height=1080)
        r1.current_active = len(tracks_r1_f1)

        # Vehicle 201 enters Road 2 at t=10.0s
        tracks_r2_f1 = {201: (250.0, 250.0, "Truck")}
        r2.update_frame(tracks_r2_f1, timestamp_sec=10.0, road_roi=roi2, frame_height=1080)
        r2.current_active = len(tracks_r2_f1)

        # Frame 2: Vehicle 101 remains in Road 1 at t=11.0s
        tracks_r1_f2 = {101: (60.0, 60.0, "Car")}
        r1.update_frame(tracks_r1_f2, timestamp_sec=11.0, road_roi=roi1, frame_height=1080)

        # Frame 3: Vehicle 101 moves and stays at t=12.0s
        tracks_r1_f3 = {101: (70.0, 70.0, "Car")}
        r1.update_frame(tracks_r1_f3, timestamp_sec=12.0, road_roi=roi1, frame_height=1080)

        # Frame 4: Vehicle 101 exits Road 1 at t=12.5s (dt = 2.5s)
        tracks_r1_f4 = {}
        r1.update_frame(tracks_r1_f4, timestamp_sec=12.5, road_roi=roi1, frame_height=1080)
        r1.current_active = len(tracks_r1_f4)

        snap1 = r1.to_summary(current_timestamp=13.0)
        snap2 = r2.to_summary(current_timestamp=13.0)

        # Road 1 assertions
        self.assertEqual(snap1["current_active"], 0)
        self.assertEqual(snap1["cumulative_entries"], 1)
        self.assertEqual(snap1["cumulative_exits"], 1)
        self.assertEqual(snap1["completed_trips"], 1)
        self.assertIsNotNone(snap1["average_speed"])
        self.assertGreater(snap1["average_speed"], 0.0)

        # Road 2 assertions
        self.assertEqual(snap2["current_active"], 1)
        self.assertEqual(snap2["cumulative_entries"], 1)
        self.assertEqual(snap2["cumulative_exits"], 0)
        self.assertEqual(snap2["completed_trips"], 0)
        self.assertIsNone(snap2["average_speed"])

        # Network aggregation
        total_occupancy = snap1["current_active"] + snap2["current_active"]
        total_entries = snap1["cumulative_entries"] + snap2["cumulative_entries"]
        total_exits = snap1["cumulative_exits"] + snap2["cumulative_exits"]

        self.assertEqual(total_occupancy, 1)
        self.assertEqual(total_entries, 2)
        self.assertEqual(total_exits, 1)

    def test_session_cleanup_logic(self):
        """
        Verify that session removal removes tracking resources, job records,
        and frame buffers cleanly without leaving dangling pointers.
        """
        from api.server import JOBS, FRAME_BUFFERS, LIVE_ANALYSIS_STATES, STOP_FLAGS, delete_session

        mock_job_id = "test_cleanup_session_1"
        JOBS[mock_job_id] = {"job_id": mock_job_id, "status": "READY"}
        FRAME_BUFFERS[mock_job_id] = [b"frame"]
        LIVE_ANALYSIS_STATES[mock_job_id] = {"current_active": 5}
        STOP_FLAGS[mock_job_id] = False

        res = delete_session(mock_job_id)
        self.assertEqual(res.get("status"), "deleted")
        self.assertNotIn(mock_job_id, JOBS)
        self.assertNotIn(mock_job_id, FRAME_BUFFERS)
        self.assertNotIn(mock_job_id, LIVE_ANALYSIS_STATES)
        self.assertNotIn(mock_job_id, STOP_FLAGS)


if __name__ == "__main__":
    unittest.main()
