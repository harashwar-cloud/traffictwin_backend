"""
src/data/video_loader.py
-------------------------
OpenCV-based frame iterator for AIC21 videos.
Yields (frame_id, timestamp_sec, frame_bgr) tuples.
Handles all FPS values present in the dataset: 8, 10, 15.

No frames are extracted to disk — processed in memory only.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Metadata extracted from a video file."""
    path: str
    filename: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration_seconds: float
    camera_id: str          # e.g. "cam_5"


def get_video_info(video_path: Path, source_fps: Optional[float] = None) -> Optional[VideoInfo]:
    """
    Open video and read metadata without loading frames.
    Returns VideoInfo or None if the file cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error(f"Cannot open video: {video_path}")
        return None
    try:
        if source_fps and source_fps > 0:
            fps = source_fps
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0

        # Infer camera_id from filename: cam_5.mp4 → "cam_5"
        stem = video_path.stem         # e.g. "cam_5_rain"
        parts = stem.split("_")
        # Camera ID is always cam_N (first two parts)
        if len(parts) >= 2 and parts[1].isdigit():
            camera_id = f"cam_{parts[1]}"
        else:
            camera_id = stem

        return VideoInfo(
            path=str(video_path),
            filename=video_path.name,
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
            duration_seconds=round(duration, 2),
            camera_id=camera_id
        )
    finally:
        cap.release()


def iter_frames(
    video_path: Path,
    frame_skip: int = 0,
    max_frames: Optional[int] = None,
    start_frame: int = 0,
    resize_width: Optional[int] = None,
    source_fps: Optional[float] = None,
) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    """
    Iterate over frames of a video.

    Args:
        video_path:   Path to the .mp4 file.
        frame_skip:   Skip N frames between each yielded frame.
                      0 = yield every frame.
                      1 = yield every other frame.
        max_frames:   Stop after yielding this many frames (None = full video).
        start_frame:  Start reading from this frame index (0-based).
        resize_width: If set, resize frame to this width (preserving aspect ratio).

    Yields:
        (frame_id, timestamp_sec, frame_bgr)
        frame_id      — 0-based index in the original video
        timestamp_sec — time in seconds from video start
        frame_bgr     — numpy array (H, W, 3) in BGR format
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error(f"Cannot open video: {video_path}")
        return

    if source_fps and source_fps > 0:
        fps = source_fps
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 10.0  # fallback

    # Seek to start frame if needed
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_id = start_frame
    yielded = 0
    skip_counter = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_id = frame_id
            frame_id += 1

            # Apply frame_skip
            if skip_counter > 0:
                skip_counter -= 1
                continue

            skip_counter = frame_skip  # reset skip counter

            # Resize if requested
            if resize_width is not None and frame.shape[1] != resize_width:
                scale = resize_width / frame.shape[1]
                new_h = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_LINEAR)

            timestamp_sec = current_id / fps

            yield current_id, timestamp_sec, frame

            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                break

    finally:
        cap.release()

    log.debug(f"iter_frames: yielded {yielded} frames from {video_path.name}")


def validate_video(video_path: Path) -> Tuple[bool, str]:
    """
    Quick validation: check that the file opens and the first frame reads.
    Returns (is_valid, reason).
    """
    if not video_path.exists():
        return False, "File not found"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "OpenCV cannot open file"

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return False, "First frame unreadable"

    return True, "OK"
