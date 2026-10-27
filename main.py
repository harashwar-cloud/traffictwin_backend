"""
main.py
--------
TrafficTwin — Single Command Inference Entry Point
AIC21 Track 1 — Multi-Class Multi-Movement Vehicle Counting

Usage:
    # Full pipeline with GT evaluation (cam_5, first 600 frames):
    python main.py --video C:/Users/haras/Desktop/openhack/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_5.mp4 \
                   --camera cam_5 \
                   --gt C:/Users/haras/Desktop/openhack/AIC21_Track1_Vehicle_Counting/counting_gt_sample/counting_example_cam_5_1min.csv \
                   --max-frames 600

    # Full video, no GT:
    python main.py --video C:/Users/haras/Desktop/openhack/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_7.mp4 --camera cam_7

    # Fast run (every 2nd frame, no annotated video):
    python main.py --video ... --camera cam_5 --frame-skip 1 --no-video
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="TrafficTwin — AIC21 Track 1 Vehicle Intelligence Pipeline"
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to input .mp4 video file"
    )
    parser.add_argument(
        "--camera", required=True,
        help="Camera ID (e.g. cam_5) — must match ROI and movement files"
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to config.yaml (default: configs/config.yaml)"
    )
    parser.add_argument(
        "--gt", default=None,
        help="Path to counting ground-truth CSV (optional — for evaluation)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Stop after N frames (default: full video)"
    )
    parser.add_argument(
        "--frame-skip", type=int, default=0,
        help="Skip N frames between processed frames (default: 0 = every frame)"
    )
    parser.add_argument(
        "--no-video", action="store_true",
        help="Do not save annotated video output"
    )
    parser.add_argument(
        "--output-prefix", default=None,
        help="Prefix for output files (default: video stem)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # All imports at the top of main
    import yaml, os
    from pathlib import Path

    log.info("=" * 60)
    log.info("TrafficTwin — AIC21 Track 1 Vehicle Intelligence System")
    log.info("=" * 60)
    log.info(f"Video    : {args.video}")
    log.info(f"Camera   : {args.camera}")
    log.info(f"Config   : {args.config}")
    log.info(f"GT CSV   : {args.gt or 'None (no evaluation)'}")
    log.info(f"Max frames: {args.max_frames or 'full video'}")
    log.info(f"Frame skip: {args.frame_skip}")

    # Validate inputs
    video_path = Path(args.video)
    if not video_path.exists():
        log.error(f"Video not found: {args.video}")
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        log.error(f"Config not found: {args.config}")
        sys.exit(1)

    # Override config options from args
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if args.max_frames is not None:
        cfg["inference"]["max_frames"] = args.max_frames
    if args.frame_skip > 0:
        cfg["inference"]["frame_skip"] = args.frame_skip
    if args.no_video:
        cfg["inference"]["save_annotated_video"] = False

    # Write overrides to a named temp file in the outputs dir
    import tempfile, os
    Path("outputs").mkdir(exist_ok=True)
    tmp_cfg_path = "outputs/.tmp_config.yaml"
    with open(tmp_cfg_path, "w") as f:
        yaml.dump(cfg, f)

    try:
        from src.inference.pipeline import TrafficPipeline
        pipeline = TrafficPipeline(config_path=tmp_cfg_path)

        result = pipeline.run(
            video_path=str(video_path),
            camera_id=args.camera,
            gt_csv_path=args.gt,
            output_prefix=args.output_prefix,
        )

        # Print final summary to stdout
        log.info("")
        log.info("=" * 60)
        log.info("FINAL RESULTS SUMMARY")
        log.info("=" * 60)
        log.info(f"Total vehicles counted : {result['total_vehicles']}")
        log.info(f"Vehicle classes        : {result['vehicle_classes']}")
        log.info(f"Traffic state          : {result['traffic_state']}")
        log.info(f"Flow rate              : {result['traffic_features'].get('total_flow_rate_per_min', 'N/A')} veh/min")
        log.info(f"Direction imbalance    : {result['traffic_features'].get('direction_imbalance_detected', False)}")
        log.info(f"Frames processed       : {result['processing']['frames_processed']}")
        log.info(f"Processing speed       : {result['processing']['speed_fps']} fps")
        log.info(f"Processing time        : {result['processing']['processing_time_seconds']}s")

        if "evaluation" in result:
            ev = result["evaluation"]["metrics"]
            log.info("")
            log.info("EVALUATION vs GT (cam_5 first 600 frames):")
            log.info(f"  GT total    : {result['evaluation']['ground_truth']['total']}")
            log.info(f"  Pred total  : {result['evaluation']['predictions']['total']}")
            log.info(f"  Total MAE   : {ev['total_count_MAE']}")
            log.info(f"  Error %     : {ev['total_error_pct']}%")
            log.info(f"  Mov MAE     : {ev['per_movement_MAE']}")
            log.info(f"  Mov <20%err : {ev['movements_within_20pct']}%")

        if result.get("outputs", {}).get("json_results"):
            log.info(f"\nResults saved → {result['outputs']['json_results']}")
        if result.get("outputs", {}).get("annotated_video"):
            log.info(f"Annotated video → {result['outputs']['annotated_video']}")

    finally:
        pass  # tmp config at outputs/.tmp_config.yaml is reusable


if __name__ == "__main__":
    main()
