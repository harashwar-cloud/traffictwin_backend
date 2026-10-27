# TrafficTwin — Metric Definitions & Synchronization Architecture

## 1. Overview
TrafficTwin provides a single, canonical source of truth for all computer vision, tracking, counting, traffic intelligence, and signal simulation telemetry. Every metric produced by the ML inference pipeline is strictly defined, calculated on the backend, and consumed directly by the React web interface without client-side duplication or hardcoded fallbacks.

---

## 2. Core Metric Definitions

### `total_vehicle_count`
- **Definition**: The count of unique tracked vehicles (Cars and Trucks) whose reference point (bottom-center of bounding box) is observed inside the validated Region of Interest (ROI) within the configured safety margin and confirmed across minimum track frames.
- **Unit**: Integer count (`vehicles`).
- **Rule**: Each unique `track_id` is registered at most once (`counted_track_ids.add(track_id)`). Transient border objects or detections outside the ROI are ignored.

### `vehicle_classes` (`car_count`, `truck_count`)
- **Definition**: Breakdown of unique counted vehicles by AIC21 vehicle classification:
  - **Car** (Class 1): Passenger cars, SUVs, vans, pickup trucks, and transit buses.
  - **Truck** (Class 2): Freight trucks, articulated heavy vehicles, and commercial haulers.
  - *Non-counted*: Two-wheelers (Motorcycles / Bicycles) are detected for collision safety but excluded from vehicular counting.
- **Invariant**: `car_count + truck_count == total_vehicle_count`.

### `flow_rate_per_min` (Arrival Rate)
- **Definition**: The rate of vehicle throughput or arrival normalized per minute over the analyzed temporal window.
- **Formula**:
  $$\text{Arrival Rate} = \frac{\text{Vehicles Counted in Window}}{\text{Window Duration (seconds)} / 60}$$
- **Unit**: Vehicles per minute (`veh/min`).

### `traffic_state`
- **Definition**: Intersection-wide congestion classification derived from rolling flow rate thresholds:
  - **LOW**: Flow rate $< 2.0\text{ veh/min}$ per movement.
  - **MEDIUM**: Flow rate $2.0 - 6.0\text{ veh/min}$ per movement.
  - **HIGH**: Flow rate $6.0 - 12.0\text{ veh/min}$ per movement.
  - **CRITICAL**: Flow rate $> 12.0\text{ veh/min}$ per movement.
- **Overall State**: Evaluated as the highest severity among active movements or total aggregate flow.

### `estimated_queue` (`future_queue_estimate`)
- **Definition**: Estimated accumulation of queued vehicles during peak cycle load when arrival flow exceeds nominal clearance discharge capacity.
- **Formula**:
  $$\text{Estimated Queue} = \max\left(0, \text{Round}\left(\text{Arrival Rate} \times \text{Congestion Factor} - \text{Clearance Capacity}\right)\right)$$
- **Unit**: Vehicles (`veh`).

### `movement_counts` (`by_movement`)
- **Definition**: Vehicle counts assigned to specific Movements of Interest (MOIs) based on vehicle trajectory (approach direction + exit angle) across intersection legs.
- **Unit**: Unique vehicles per movement ID.
- **Invariant**: Sum of movement counts + unassigned in-flight tracks equals `total_vehicle_count`.

### `direction_imbalance` / `bottleneck`
- **Definition**: Identifies directional demand asymmetry where the dominant movement flow significantly exceeds competing approaches.
- **Formula**:
  $$\text{Imbalance Ratio} = \frac{\text{Dominant Movement Flow Rate}}{\min(\text{Active Competing Movement Flow Rates})}$$
- **Trigger**: Flagged when $\text{Imbalance Ratio} \ge 3.0\text{x}$.

---

## 3. Data Flow & Single Source of Truth

```plaintext
Surveillance Stream / MP4 Video
             ↓
Frame-by-Frame YOLOv8 Detection
             ↓
Bottom-Center ROI Polygon Filtering (Safety Border Inset)
             ↓
ByteTrack Multi-Object Tracking (Confirmed Track IDs)
             ↓
Unique Vehicle & MOI Counting Engine
             ↓
Traffic State & Imbalance Intelligence
             ↓
┌─────────────────────────────────────────────────────────────┐
│                    Canonical Job Result                     │
│  ├── Live MJPEG Frame Stream (/api/stream/{job_id})         │
│  ├── Final H.264 MP4 Video (/api/video/stream/{filename})   │
│  └── Synchronized JSON API (/api/results/{job_id})          │
└─────────────────────────────────────────────────────────────┘
             ↓
FastAPI Backend (/api/traffic/current & /api/debug/results/{job_id})
             ↓
React Frontend Dashboard (Overview, Live Analysis, Simulator, History)
```
