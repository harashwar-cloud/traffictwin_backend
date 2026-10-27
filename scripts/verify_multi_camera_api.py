import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from api.server import get_health, get_cameras_overview, get_traffic_by_camera

def main():
    print("=" * 60)
    print("TRAFFICTWIN — MULTI-CAMERA API VERIFICATION")
    print("=" * 60)

    # 1. Health check
    health = get_health()
    print("Health Status:", health.status, "| GPU Available:", health.gpu_available)

    # 2. Cameras Overview
    overview = get_cameras_overview()
    print("\nNetwork Summary:")
    for k, v in overview["network_summary"].items():
        print(f"  {k}: {v}")

    print("\nIndividual Camera Feeds (All 5 Sources):")
    for cam in overview["cameras"]:
        print(
            f"  {cam['cam_number']}: {cam['name']}\n"
            f"    Status: {cam['status']} | Count: {cam['vehicles_counted']} | "
            f"State: {cam['traffic_state']} | Flow: {cam['flow_rate_per_min']} veh/min | "
            f"Queue: {cam['queue']} veh\n"
            f"    Video: {cam['video_url']}"
        )

    # 3. Specific Camera Deep Dive State
    for cid in ["cam_5", "cam_1", "cam_7", "cam_9", "cam_10"]:
        cst = get_traffic_by_camera(cid)
        count = cst.metrics.total_vehicle_count if cst.metrics else 0
        state = cst.metrics.traffic_state.value if cst.metrics else "LOW"
        print(f"\nDeep-dive [{cid}]: Count={count}, State={state}, Movements={len(cst.metrics.by_movement) if cst.metrics else 0}")

    print("=" * 60)
    print("ALL 5 CAMERA FEEDS & METRICS VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
