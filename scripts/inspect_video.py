"""
scripts/inspect_video.py
-------------------------
Diagnostic utility to inspect video files and report:
FILE, SIZE, FORMAT, CODEC, FPS, WIDTH, HEIGHT, FRAME_COUNT, DURATION, READABLE, BROWSER_COMPATIBLE
"""

import sys
import os
import subprocess
from pathlib import Path
import cv2

try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = None


def inspect_video(file_path: str):
    p = Path(file_path)
    print("=" * 60)
    print("VIDEO INSPECTION REPORT")
    print("=" * 60)
    print(f"FILE         : {p.resolve()}")
    if not p.exists():
        print(f"EXISTS       : FALSE (File not found)")
        print(f"READABLE     : FALSE")
        print("=" * 60)
        return False

    size_bytes = p.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    print(f"EXISTS       : TRUE")
    print(f"SIZE         : {size_bytes} bytes ({size_mb:.2f} MB)")
    print(f"FORMAT       : {p.suffix.lower()}")

    cap = cv2.VideoCapture(str(p))
    is_readable = cap.isOpened()
    print(f"READABLE     : {is_readable}")

    if not is_readable:
        print("ERROR        : OpenCV cannot open video file")
        print("=" * 60)
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
    duration = frame_count / fps if fps > 0 else 0.0

    ret, first_frame = cap.read()
    cap.release()

    print(f"FOURCC CODEC : '{fourcc_str}' (int: {fourcc_int})")
    print(f"FPS          : {fps:.2f}")
    print(f"WIDTH        : {width}")
    print(f"HEIGHT       : {height}")
    print(f"FRAME_COUNT  : {frame_count}")
    print(f"DURATION     : {duration:.2f}s")
    print(f"FIRST_FRAME  : {'OK' if ret and first_frame is not None else 'FAILED'}")

    # Inspect with ffmpeg if available
    codec_name = "unknown"
    is_h264 = False
    if FFMPEG_EXE and Path(FFMPEG_EXE).exists():
        try:
            cmd = [FFMPEG_EXE, "-i", str(p)]
            proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
            stderr = proc.stderr
            for line in stderr.splitlines():
                if "Stream #0" in line and "Video:" in line:
                    print(f"FFMPEG STREAM: {line.strip()}")
                    if "h264" in line.lower() or "avc" in line.lower():
                        is_h264 = True
        except Exception as e:
            print(f"FFMPEG ERROR : {e}")

    browser_playable = is_h264 or (fourcc_str.lower() in ("avc1", "h264"))
    print(f"BROWSER PLAYABLE (H.264 MP4): {browser_playable}")
    print("=" * 60)
    return is_readable and frame_count > 0 and size_bytes > 0


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "outputs/videos/cam5_eval_annotated.mp4"
    inspect_video(target)
