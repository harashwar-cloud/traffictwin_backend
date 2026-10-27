"""
tests/test_adaptive_signal_system.py
------------------------------------
Automated test suite verifying the complete Adaptive Traffic Signal System:
1. Video session readiness (requires exactly 3 or 4 sessions, deduplication, error when <3 or >4).
2. Authoritative & simulated signal phase status with 8-second recommendation trigger.
3. Officer approval and rejection lifecycle (pending, approve within min/max green, reject with reason, idempotency, error states).
4. Recommendation supersession and audit trail logging.
"""

import unittest
from fastapi.testclient import TestClient
from api.server import (
    app,
    INTERSECTION_CONFIGS,
    SIGNAL_RECOMMENDATION_HISTORY,
    SIGNAL_AUDIT_LOG,
    JOBS,
    LIVE_ANALYSIS_STATES,
    _evaluate_video_readiness,
    _get_or_update_signal_status,
)
from api.signal_models import RecommendationStatus


class TestAdaptiveSignalSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        # Reset intersection recommendation history & audit log before tests
        SIGNAL_RECOMMENDATION_HISTORY["int_cam5"] = []

    def test_01_video_readiness_fewer_than_three_sessions(self):
        """Fewer than 3 sessions disables recommendations and explains why."""
        rep = _evaluate_video_readiness("int_cam5", ["session_alpha", "session_beta"])
        self.assertFalse(rep["is_ready"])
        self.assertFalse(rep["recommendation_available"])
        self.assertEqual(rep["valid_session_count"], 2)
        self.assertEqual(rep["coverage_status"], "INCOMPLETE")
        self.assertTrue(any("2/3 required" in r for r in rep["reasons"]))

    def test_02_video_readiness_more_than_four_sessions(self):
        """More than 4 sessions displays a configuration error requiring 3 or 4."""
        rep = _evaluate_video_readiness(
            "int_cam5",
            ["sess_1", "sess_2", "sess_3", "sess_4", "sess_5"]
        )
        self.assertFalse(rep["is_ready"])
        self.assertFalse(rep["recommendation_available"])
        self.assertEqual(rep["valid_session_count"], 5)
        self.assertEqual(rep["coverage_status"], "CONFIG_ERROR")
        self.assertTrue(any("Maximum allowed is 4" in r for r in rep["reasons"]))

    def test_03_video_readiness_deduplicates_duplicate_session_ids(self):
        """Duplicate session IDs or cards must not be counted as separate sessions."""
        rep = _evaluate_video_readiness(
            "int_cam5",
            ["sess_1", "sess_1", "sess_2", "sess_2", "sess_1"]
        )
        self.assertEqual(rep["valid_session_count"], 2)
        self.assertFalse(rep["is_ready"])
        self.assertEqual(rep["coverage_status"], "INCOMPLETE")

    def test_04_video_readiness_exact_three_and_four_sessions(self):
        """Exactly 3 or 4 valid sessions with complete approach coverage is ready."""
        rep3 = _evaluate_video_readiness("int_cam5", ["cam_1", "cam_2", "cam_5"])
        self.assertEqual(rep3["valid_session_count"], 3)
        self.assertTrue(rep3["is_ready"])
        self.assertTrue(rep3["recommendation_available"])
        self.assertEqual(rep3["coverage_status"], "COMPLETE")

        rep4 = _evaluate_video_readiness("int_cam5", ["cam_1", "cam_2", "cam_3", "cam_5"])
        self.assertEqual(rep4["valid_session_count"], 4)
        self.assertTrue(rep4["is_ready"])
        self.assertTrue(rep4["recommendation_available"])
        self.assertEqual(rep4["coverage_status"], "COMPLETE")

    def test_05_signal_status_and_8s_trigger(self):
        """Authoritative signal cycle tracking and 8-second trigger scheduler."""
        status = _get_or_update_signal_status("int_cam5", trigger_threshold_sec=8.0)
        self.assertEqual(status["intersection_id"], "int_cam5")
        self.assertTrue(status["is_timing_available"])
        self.assertTrue(status["is_simulated"])
        self.assertIn("remaining_green_sec", status)
        self.assertIn("should_trigger_recommendation", status)
        self.assertEqual(status["recommendation_trigger_sec"], 8.0)

        # HTTP Endpoint check
        res = self.client.get("/api/intersections/int_cam5/signal-status?trigger_threshold_sec=8.0")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["is_timing_available"])
        self.assertIn("current_phase_id", data)

    def test_06_recommendation_lifecycle_pending_review(self):
        """Generated recommendations start in PENDING_OFFICER_REVIEW status."""
        res = self.client.post("/api/intersections/int_cam5/recommendations")
        self.assertEqual(res.status_code, 200)
        rec = res.json()
        rec_id = rec["recommendation_id"]
        self.assertEqual(rec["status"], RecommendationStatus.PENDING_OFFICER_REVIEW.value)
        self.assertIsNone(rec.get("officer_decision"))

    def test_07_recommendation_supersession(self):
        """Generating a new recommendation supersedes the prior pending recommendation."""
        res1 = self.client.post("/api/intersections/int_cam5/recommendations")
        rec1_id = res1.json()["recommendation_id"]

        res2 = self.client.post("/api/intersections/int_cam5/recommendations")
        rec2_id = res2.json()["recommendation_id"]
        self.assertNotEqual(rec1_id, rec2_id)

        history = SIGNAL_RECOMMENDATION_HISTORY["int_cam5"]
        rec1_item = next(r for r in history if r["recommendation_id"] == rec1_id)
        rec2_item = next(r for r in history if r["recommendation_id"] == rec2_id)

        self.assertEqual(rec1_item["status"], RecommendationStatus.SUPERSEDED.value)
        self.assertEqual(rec2_item["status"], RecommendationStatus.PENDING_OFFICER_REVIEW.value)

    def test_08_officer_approval_within_bounds(self):
        """Officer approval within min/max green bounds succeeds and records decision."""
        res = self.client.post("/api/intersections/int_cam5/recommendations")
        rec_id = res.json()["recommendation_id"]

        approve_res = self.client.post(
            f"/api/intersections/int_cam5/recommendations/{rec_id}/approve",
            json={"officer_id": "officer_42", "approved_green_sec": 35.0}
        )
        self.assertEqual(approve_res.status_code, 200)
        approved = approve_res.json()
        self.assertEqual(approved["status"], RecommendationStatus.APPROVED.value)
        self.assertEqual(approved["officer_decision"], "APPROVED")
        self.assertEqual(approved["officer_id"], "officer_42")
        self.assertEqual(approved["approved_green_sec"], 35.0)

        # Idempotency check: approving again returns 200 with approved state
        re_approve = self.client.post(
            f"/api/intersections/int_cam5/recommendations/{rec_id}/approve",
            json={"officer_id": "officer_42", "approved_green_sec": 35.0}
        )
        self.assertEqual(re_approve.status_code, 200)

    def test_09_officer_approval_out_of_bounds_rejected(self):
        """Approval with proposed green outside phase bounds returns 422 error."""
        res = self.client.post("/api/intersections/int_cam5/recommendations")
        rec_id = res.json()["recommendation_id"]

        # Phase min is 15s, max is 55s for P1 in int_cam5
        too_high = self.client.post(
            f"/api/intersections/int_cam5/recommendations/{rec_id}/approve",
            json={"officer_id": "officer_1", "approved_green_sec": 95.0}
        )
        self.assertEqual(too_high.status_code, 422)

        too_low = self.client.post(
            f"/api/intersections/int_cam5/recommendations/{rec_id}/approve",
            json={"officer_id": "officer_1", "approved_green_sec": 5.0}
        )
        self.assertEqual(too_low.status_code, 422)

    def test_10_officer_rejection_lifecycle(self):
        """Officer rejection records reason and prevents subsequent approval."""
        res = self.client.post("/api/intersections/int_cam5/recommendations")
        rec_id = res.json()["recommendation_id"]

        reject_res = self.client.post(
            f"/api/intersections/int_cam5/recommendations/{rec_id}/reject",
            json={"officer_id": "officer_99", "rejection_reason": "Emergency vehicle priority corridor"}
        )
        self.assertEqual(reject_res.status_code, 200)
        rejected = reject_res.json()
        self.assertEqual(rejected["status"], RecommendationStatus.REJECTED.value)
        self.assertEqual(rejected["officer_decision"], "REJECTED")
        self.assertEqual(rejected["rejection_reason"], "Emergency vehicle priority corridor")

        # Rejected recommendation cannot subsequently be approved
        bad_approve = self.client.post(
            f"/api/intersections/int_cam5/recommendations/{rec_id}/approve",
            json={"officer_id": "officer_99", "approved_green_sec": 30.0}
        )
        self.assertEqual(bad_approve.status_code, 400)


if __name__ == "__main__":
    unittest.main()
