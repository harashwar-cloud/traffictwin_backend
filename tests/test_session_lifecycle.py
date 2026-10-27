import unittest
from fastapi.testclient import TestClient
import api.server as server
from api.models import AnalysisStatus


class TestSessionLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(server.app)

    def setUp(self):
        # Reset globals before each test
        server.ACTIVE_PROCESSING_JOB_ID = None
        server.LATEST_COMPLETED_JOB_ID = None

    def test_active_session_idle(self):
        """When no video is running, /api/session/active reports active: False."""
        res = self.client.get("/api/session/active")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertFalse(data["active"])

    def test_active_session_tracking(self):
        """When a video job is active, /api/session/active returns its live metrics and progress."""
        test_job_id = "test_cam5_active_123"
        server.JOBS[test_job_id] = {
            "job_id": test_job_id,
            "status": AnalysisStatus.PROCESSING,
            "camera_id": "cam_5",
            "video_path": "Dataset_A/cam_5.mp4",
            "filename": "cam_5.mp4",
            "frames_processed": 142,
            "total_frames": 600,
        }
        server.LIVE_ANALYSIS_STATES[test_job_id] = {
            "job_id": test_job_id,
            "frame_index": 142,
            "total_frames": 600,
            "video_time_seconds": 14.2,
            "vehicles_counted": 18,
            "current_active": 7,
            "current_stationary": 3,
            "current_moving": 4,
            "traffic_state": "MEDIUM",
            "flow_rate_per_min": 15.5,
            "processing_fps": 28.4,
            "roads": [
                {"road_id": "road_1", "name": "North Approach", "vehicles": 5, "current_stationary": 2, "traffic_state": "MEDIUM"},
                {"road_id": "road_2", "name": "South Approach", "vehicles": 8, "current_stationary": 1, "traffic_state": "LOW"},
            ]
        }
        server.ACTIVE_PROCESSING_JOB_ID = test_job_id

        res = self.client.get("/api/session/active")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["active"])
        self.assertEqual(data["job_id"], test_job_id)
        self.assertEqual(data["frames_processed"], 142)
        self.assertEqual(data["vehicles_counted"], 18)
        self.assertEqual(data["current_active"], 7)
        self.assertAlmostEqual(data["current_fps"], 28.4, places=1)

        # Test inference status alias for "active"
        alias_res = self.client.get("/api/inference/active/status")
        self.assertEqual(alias_res.status_code, 200)
        alias_data = alias_res.json()
        self.assertEqual(alias_data["job_id"], test_job_id)
        self.assertEqual(alias_data["frames_processed"], 142)

    def test_signals_live_demand_binding(self):
        """Signals module dynamically binds to real-time metrics of active video processing job."""
        test_job_id = "test_cam5_signals_456"
        server.JOBS[test_job_id] = {
            "job_id": test_job_id,
            "status": AnalysisStatus.PROCESSING,
            "camera_id": "cam_5",
            "video_path": "cam_5.mp4",
            "filename": "cam_5.mp4",
        }
        server.LIVE_ANALYSIS_STATES[test_job_id] = {
            "job_id": test_job_id,
            "frame_index": 210,
            "video_time_seconds": 21.0,
            "roads": [
                {"road_id": "road_1", "current_stationary": 9, "flow_volume": 22.0, "traffic_state": "HIGH"},
                {"road_id": "road_2", "current_stationary": 4, "flow_volume": 12.0, "traffic_state": "LOW"},
                {"road_id": "road_3", "current_stationary": 1, "flow_volume": 6.0, "traffic_state": "LOW"},
                {"road_id": "road_4", "current_stationary": 11, "flow_volume": 25.0, "traffic_state": "CRITICAL"},
            ]
        }
        server.ACTIVE_PROCESSING_JOB_ID = test_job_id

        config = server.INTERSECTION_CONFIGS["int_cam5"]
        live_states = server._live_states_for_intersection(config)

        # Assert approaches road_N (North Approach -> Road 1) and road_W (West Approach -> Road 4) received active video inference data
        self.assertIn("road_N", live_states)
        self.assertIn("road_W", live_states)
        self.assertEqual(live_states["road_N"]["_source"], "active_inference")
        self.assertEqual(live_states["road_N"]["_job_id"], test_job_id)
        self.assertEqual(live_states["road_N"]["current_stationary"], 9)
        self.assertEqual(live_states["road_W"]["current_stationary"], 11)

    def test_duplicate_job_reattach(self):
        """Starting an already running camera session reattaches without spawning duplicate workers."""
        test_job_id = "cam_5_existing_789"
        server.JOBS[test_job_id] = {
            "job_id": test_job_id,
            "status": AnalysisStatus.PROCESSING,
            "camera_id": "cam_5",
            "video_path": str(server.project_root.parent / "AIC21_Track1_Vehicle_Counting" / "Dataset_A" / "cam_5.mp4"),
            "filename": "cam_5.mp4",
            "frames_processed": 85,
            "total_frames": 600,
        }
        server.ACTIVE_PROCESSING_JOB_ID = test_job_id

        # Attempt to start inference on cam_5 while already running
        res = self.client.post("/api/inference/start", data={"job_id": "cam_5", "max_frames": 600})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("already_running"))
        self.assertEqual(data["job_id"], test_job_id)
        self.assertEqual(data["frames_processed"], 85)

    def test_stop_inference_lifecycle(self):
        """Stopping an active job cancels status and clears ACTIVE_PROCESSING_JOB_ID."""
        test_job_id = "test_cancel_999"
        server.JOBS[test_job_id] = {
            "job_id": test_job_id,
            "status": AnalysisStatus.PROCESSING,
            "camera_id": "cam_5",
        }
        server.ACTIVE_PROCESSING_JOB_ID = test_job_id

        res = self.client.post(f"/api/inference/{test_job_id}/stop")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(server.JOBS[test_job_id]["status"], AnalysisStatus.CANCELLED)
        self.assertIsNone(server.ACTIVE_PROCESSING_JOB_ID)


if __name__ == "__main__":
    unittest.main()
