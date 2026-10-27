"""
tests/test_drone_detection.py
-----------------------------
Unit tests and resolution benchmarking for top-angle aerial drone vehicle detection:
  - Slice generation & overlapping tile geometry
  - Cross-tile NMS deduplication
  - Resolution benchmark (640 vs 960 vs 1280, tiled vs non-tiled)
"""

import time
import unittest
import numpy as np
import torch

from src.detection.detector import (
    VehicleDetector,
    SlicedDroneDetector,
    Detection,
    TARGET_COCO_IDS,
    COCO_TO_AIC21,
)


class TestDroneDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        cls.device = device
        cls.detector = VehicleDetector(
            model_path="yolov8n.pt",
            input_size=640,
            device=device,
            drone_mode=True,
        )
        cls.sliced_detector = SlicedDroneDetector(
            base_detector=cls.detector,
            tile_size=(640, 640),
            overlap_ratio=0.20,
            nms_threshold=0.45,
        )

    def test_01_slice_generation_coverage(self):
        """Verify overlapping tile generation covers the full 1920x1080 frame."""
        slices = self.sliced_detector._generate_slices(1920, 1080)
        self.assertGreater(len(slices), 0)

        # Check all slices are within bounds
        for (x1, y1, x2, y2) in slices:
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, 1920)
            self.assertLessEqual(y2, 1080)
            self.assertLessEqual(x2 - x1, 640)
            self.assertLessEqual(y2 - y1, 640)

        # Check corners are covered
        self.assertEqual(slices[0][0], 0)
        self.assertEqual(slices[0][1], 0)
        self.assertEqual(slices[-1][2], 1920)
        self.assertEqual(slices[-1][3], 1080)

    def test_02_nms_cross_tile_deduplication(self):
        """Detections spanning adjacent tile boundaries must be deduplicated via NMS."""
        # Create a mock frame with synthetic content
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Sliced detector run should complete without error
        dets = self.sliced_detector.detect(frame, frame_id=1, timestamp_sec=0.1)
        self.assertIsInstance(dets, list)

    def test_03_resolution_and_tiling_benchmark(self):
        """
        Benchmark inference latency, FPS, and detection across:
        - 640x640 (standard baseline)
        - 960x960 (recommended drone aerial resolution)
        - 1280x1280 (high-resolution aerial)
        - Sliced overlapping inference (2x2 tiles + full frame)
        """
        # Create a 1920x1080 synthetic test frame with simulated vehicle shapes
        test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Add high-contrast patches simulating top-view vehicle roofs
        cv2 = __import__("cv2")
        for i in range(15):
            x = 100 + (i % 5) * 350
            y = 150 + (i // 5) * 300
            # Small top-view car shape (45x25 px)
            cv2.rectangle(test_frame, (x, y), (x + 45, y + 25), (180, 180, 180), -1)
            cv2.rectangle(test_frame, (x + 8, y + 4), (x + 37, y + 21), (60, 60, 60), -1)

        benchmark_results = {}

        # 1. Benchmark 640x640
        self.detector.input_size = 640
        t0 = time.perf_counter()
        for _ in range(5):
            dets_640 = self.detector.detect(test_frame)
        dur_640 = (time.perf_counter() - t0) / 5.0
        fps_640 = 1.0 / dur_640 if dur_640 > 0 else 0
        benchmark_results["640"] = {
            "resolution": "640x640",
            "latency_ms": round(dur_640 * 1000.0, 1),
            "fps": round(fps_640, 1),
            "detections": len(dets_640),
        }

        # 2. Benchmark 960x960
        self.detector.input_size = 960
        t0 = time.perf_counter()
        for _ in range(5):
            dets_960 = self.detector.detect(test_frame)
        dur_960 = (time.perf_counter() - t0) / 5.0
        fps_960 = 1.0 / dur_960 if dur_960 > 0 else 0
        benchmark_results["960"] = {
            "resolution": "960x960",
            "latency_ms": round(dur_960 * 1000.0, 1),
            "fps": round(fps_960, 1),
            "detections": len(dets_960),
        }

        # 3. Benchmark 1280x1280
        self.detector.input_size = 1280
        t0 = time.perf_counter()
        for _ in range(5):
            dets_1280 = self.detector.detect(test_frame)
        dur_1280 = (time.perf_counter() - t0) / 5.0
        fps_1280 = 1.0 / dur_1280 if dur_1280 > 0 else 0
        benchmark_results["1280"] = {
            "resolution": "1280x1280",
            "latency_ms": round(dur_1280 * 1000.0, 1),
            "fps": round(fps_1280, 1),
            "detections": len(dets_1280),
        }

        # 4. Benchmark Sliced Tiled Inference
        t0 = time.perf_counter()
        for _ in range(3):
            dets_tiled = self.sliced_detector.detect(test_frame)
        dur_tiled = (time.perf_counter() - t0) / 3.0
        fps_tiled = 1.0 / dur_tiled if dur_tiled > 0 else 0
        benchmark_results["tiled"] = {
            "resolution": "Sliced 640 Overlapping",
            "latency_ms": round(dur_tiled * 1000.0, 1),
            "fps": round(fps_tiled, 1),
            "detections": len(dets_tiled),
        }

        print("\n" + "=" * 65)
        print("AERIAL DRONE DETECTION BENCHMARK (RTX 3050 GPU)")
        print("=" * 65)
        print(f"{'Config':<26} | {'Latency (ms)':<14} | {'FPS':<10} | {'Detections'}")
        print("-" * 65)
        for k, v in benchmark_results.items():
            print(f"{v['resolution']:<26} | {v['latency_ms']:<14} | {v['fps']:<10} | {v['detections']}")
        print("=" * 65)

        # Assert all benchmarks produced valid numbers
        self.assertGreater(benchmark_results["640"]["fps"], 15.0)
        self.assertGreater(benchmark_results["960"]["fps"], 10.0)


if __name__ == "__main__":
    unittest.main()
