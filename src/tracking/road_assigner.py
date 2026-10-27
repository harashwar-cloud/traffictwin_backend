import logging
from typing import Dict, List, Optional, Tuple, Any

from src.data.road_config import RoadROI, assign_track_to_road

log = logging.getLogger(__name__)

class RoadAssigner:
    def __init__(
        self,
        switch_confirm_frames: int = 10,
        entry_confirm_frames: int = 5,
        lost_grace_frames: int = 15,
        boundary_margin: float = 15.0,
        reference_point: str = "bottom_center"
    ):
        self.switch_confirm_frames = switch_confirm_frames
        self.entry_confirm_frames = entry_confirm_frames
        self.lost_grace_frames = lost_grace_frames
        self.boundary_margin = boundary_margin
        self.reference_point = reference_point
        
        # track_id -> dict of state
        self.states: Dict[int, Dict[str, Any]] = {}

    def update(
        self,
        track_id: int,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        road_rois: List[RoadROI]
    ) -> Optional[str]:
        # 1. Instantaneous assignment (which road is the point physically in right now?)
        instant_road_id = assign_track_to_road(
            x1, y1, x2, y2, road_rois,
            margin=self.boundary_margin,
            reference_point=self.reference_point
        )

        # Initialize if new track
        if track_id not in self.states:
            self.states[track_id] = {
                "road_id": instant_road_id,
                "candidate_road_id": None,
                "candidate_frames": 0,
                "outside_current_road_frames": 0,
                "lost_frames": 0,
            }
            return instant_road_id
            
        state = self.states[track_id]
        
        # If currently no confirmed road, just adopt the instant one immediately (or after entry confirm)
        if state["road_id"] is None:
            if instant_road_id is not None:
                if state["candidate_road_id"] == instant_road_id:
                    state["candidate_frames"] += 1
                else:
                    state["candidate_road_id"] = instant_road_id
                    state["candidate_frames"] = 1
                    
                if state["candidate_frames"] >= self.entry_confirm_frames:
                    state["road_id"] = instant_road_id
                    state["candidate_road_id"] = None
                    state["candidate_frames"] = 0
            else:
                state["candidate_road_id"] = None
                state["candidate_frames"] = 0
            return state["road_id"]
            
        # We have a current confirmed road.
        # Check if we are still inside it (using margin)
        # We use assign_track_to_road which already applies the margin.
        
        # Determine if we are outside current road
        is_outside_current = True
        
        # Does the instant_road_id match our current confirmed road?
        if instant_road_id == state["road_id"]:
            is_outside_current = False
        else:
            # Maybe the instant road is None or another road, but are we still inside the current road's buffered polygon?
            # Let's explicitly check the current road with the margin.
            current_road = next((r for r in road_rois if r.road_id == state["road_id"]), None)
            if current_road is not None:
                if self.reference_point == "bottom_center":
                    rx, ry = (x1 + x2) / 2.0, float(y2)
                else:
                    rx, ry = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                
                # Check containment explicitly for current road with margin
                if current_road.contains_point(rx, ry, margin=self.boundary_margin):
                    is_outside_current = False
                    # Force instant_road_id to be current if we are within its margin to prevent jitter
                    instant_road_id = state["road_id"]
                    
        # Update outside frames counter
        if is_outside_current:
            state["outside_current_road_frames"] += 1
        else:
            state["outside_current_road_frames"] = 0
            
        # Update lost grace frames (when totally outside any road)
        if instant_road_id is None:
            state["lost_frames"] += 1
            if state["lost_frames"] > self.lost_grace_frames:
                # Completely lost from any road for too long
                state["road_id"] = None
                state["candidate_road_id"] = None
                state["candidate_frames"] = 0
        else:
            state["lost_frames"] = 0
            
        # Update candidate road
        if instant_road_id is not None and instant_road_id != state["road_id"]:
            if state["candidate_road_id"] == instant_road_id:
                state["candidate_frames"] += 1
            else:
                state["candidate_road_id"] = instant_road_id
                state["candidate_frames"] = 1
                
            # Check for transition: must be outside current AND inside candidate for enough frames
            if (state["outside_current_road_frames"] >= self.switch_confirm_frames and 
                state["candidate_frames"] >= self.entry_confirm_frames):
                
                log.debug(f"[ROAD SWITCH] ID={track_id} {state['road_id']} -> {instant_road_id} confirmed=true")
                state["road_id"] = instant_road_id
                state["candidate_road_id"] = None
                state["candidate_frames"] = 0
                state["outside_current_road_frames"] = 0
        else:
            state["candidate_road_id"] = None
            state["candidate_frames"] = 0
            
        return state["road_id"]
        
    def cleanup_lost_tracks(self, active_track_ids: List[int]) -> None:
        """Remove state for tracks that are permanently lost."""
        lost_ids = [tid for tid in self.states if tid not in active_track_ids]
        for tid in lost_ids:
            del self.states[tid]
