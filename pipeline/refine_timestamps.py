"""
Stage 2: Timestamp Refinement

Takes rough cut points and refines them to sub-frame precision by extracting
frames at high FPS around each cut point and doing pixel comparisons.
"""

import subprocess
import os
import numpy as np
from PIL import Image


def extract_frames_around_cut(
    video_path: str,
    cut_time: float,
    window: float = 0.25,
    fps: int = 30,
    output_dir: str = "/tmp/refinement_frames",
) -> list[tuple[float, str]]:
    """
    Extract frames at high FPS in a window around a cut point.
    Returns list of (timestamp, frame_path) tuples.
    """
    os.makedirs(output_dir, exist_ok=True)

    start = max(0, cut_time - window)
    duration = window * 2

    # Use a unique prefix to avoid collisions
    prefix = f"cut_{cut_time:.3f}"

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-vf", f"fps={fps}",
        "-frame_pts", "1",
        "-q:v", "2",
        os.path.join(output_dir, f"{prefix}_%04d.jpg"),
    ]
    subprocess.run(cmd, capture_output=True, text=True)

    # Collect extracted frames with their timestamps
    frames = []
    frame_files = sorted([
        f for f in os.listdir(output_dir)
        if f.startswith(prefix) and f.endswith(".jpg")
    ])

    for i, fname in enumerate(frame_files):
        timestamp = start + (i / fps)
        frames.append((timestamp, os.path.join(output_dir, fname)))

    return frames


def compute_frame_difference(frame_a_path: str, frame_b_path: str) -> float:
    """
    Compute normalized pixel difference between two frames.
    Returns a value between 0 (identical) and 1 (completely different).
    """
    try:
        img_a = Image.open(frame_a_path).convert("RGB").resize((160, 90))
        img_b = Image.open(frame_b_path).convert("RGB").resize((160, 90))

        arr_a = np.array(img_a, dtype=np.float32)
        arr_b = np.array(img_b, dtype=np.float32)

        # Mean absolute difference normalized to [0, 1]
        diff = np.mean(np.abs(arr_a - arr_b)) / 255.0
        return float(diff)
    except Exception:
        return 0.0


def refine_cut_point(
    video_path: str,
    rough_cut_time: float,
    window: float = 0.25,
    refinement_fps: int = 30,
) -> float:
    """
    Refine a single cut point to sub-frame precision.
    Finds the frame pair with the largest pixel difference in the window.
    """
    output_dir = f"/tmp/refinement_{rough_cut_time:.3f}"

    frames = extract_frames_around_cut(
        video_path, rough_cut_time,
        window=window, fps=refinement_fps,
        output_dir=output_dir,
    )

    if len(frames) < 2:
        return rough_cut_time

    # Compare consecutive frames
    max_diff = 0.0
    best_cut_time = rough_cut_time

    for i in range(len(frames) - 1):
        ts_a, path_a = frames[i]
        ts_b, path_b = frames[i + 1]

        diff = compute_frame_difference(path_a, path_b)
        if diff > max_diff:
            max_diff = diff
            # The cut happens at the boundary between the two frames
            best_cut_time = (ts_a + ts_b) / 2.0

    # Cleanup
    for _, path in frames:
        try:
            os.remove(path)
        except OSError:
            pass
    try:
        os.rmdir(output_dir)
    except OSError:
        pass

    return round(best_cut_time, 3)


def refine_all_cuts(
    video_path: str,
    shots: list[dict],
    window: float = 0.25,
    refinement_fps: int = 30,
) -> list[dict]:
    """
    Refine all shot boundaries. The first shot always starts at 0.
    """
    print(f"  Refining {len(shots)} shot boundaries (±{window}s at {refinement_fps}fps)...")

    refined_shots = []
    for i, shot in enumerate(shots):
        new_shot = dict(shot)

        # Don't refine the very start (0.0) or very end
        if shot["start_seconds"] > 0.1:
            refined_start = refine_cut_point(
                video_path, shot["start_seconds"],
                window=window, refinement_fps=refinement_fps,
            )
            new_shot["start_seconds"] = refined_start

        refined_shots.append(new_shot)

    # Fix up end times: each shot's end = next shot's start
    for i in range(len(refined_shots) - 1):
        refined_shots[i]["end_seconds"] = refined_shots[i + 1]["start_seconds"]

    # Recalculate durations
    for shot in refined_shots:
        shot["duration"] = round(shot["end_seconds"] - shot["start_seconds"], 3)

    # Filter out any shots that became zero/negative duration after refinement
    refined_shots = [s for s in refined_shots if s["duration"] > 0.02]

    print(f"  Refinement complete: {len(refined_shots)} shots")
    return refined_shots


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.refine_timestamps <video_path>")
        sys.exit(1)

    from pipeline.cut_detection import detect_cuts
    shots = detect_cuts(sys.argv[1])
    refined = refine_all_cuts(sys.argv[1], shots)
    print(json.dumps(refined[:5], indent=2))
