"""
pipeline/extract_frames.py
--------------------------
Extracts frames from a video file using ffmpeg.
Outputs numbered .jpg images into the specified directory.
"""

import subprocess
import shutil
from pathlib import Path


def check_ffmpeg():
    """Verify ffmpeg is available on the system."""
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError(
            "ffmpeg not found. Please install it:\n"
            "  Ubuntu/Debian : sudo apt install ffmpeg\n"
            "  macOS         : brew install ffmpeg\n"
            "  Docker        : already included in the provided Dockerfile."
        )


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: float = 2.0,
    max_frames: int = 150,
) -> int:
    """
    Extract frames from a video using ffmpeg.

    Args:
        video_path  : Path to input video file.
        output_dir  : Directory to save extracted frames.
        fps         : Frames per second to extract.
        max_frames  : Maximum number of frames to keep.

    Returns:
        Number of frames extracted.
    """
    check_ffmpeg()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove any existing frames to avoid mixing old/new runs
    for f in output_dir.glob("*.jpg"):
        f.unlink()

    output_pattern = str(output_dir / "frame_%05d.jpg")

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"fps={fps}",           # sample at target fps
        "-q:v", "2",                    # high quality JPEG (1=best, 31=worst)
        "-frames:v", str(max_frames),   # hard cap on frame count
        output_pattern,
        "-y",                           # overwrite without prompting
        "-loglevel", "warning",
    ]

    print(f"  Running ffmpeg @ {fps} fps (max {max_frames} frames)...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed with exit code {result.returncode}.\n"
            f"stderr: {result.stderr}"
        )

    frames = sorted(output_dir.glob("*.jpg"))
    n = len(frames)

    if n == 0:
        return 0

    print(f"  Frames saved to: {output_dir}")
    return n