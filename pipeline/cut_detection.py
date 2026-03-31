"""
Stage 1: Cut Detection

Uses PySceneDetect (ContentDetector + AdaptiveDetector) and optionally
ffmpeg's scene filter as a second opinion. Merges results and deduplicates.
"""

import subprocess
import re
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector


def detect_scenes_pyscenedetect(
    video_path: str,
    content_threshold: float = 27.0,
    adaptive_threshold: float = 3.0,
    min_scene_len_frames: int = 5,
) -> list[dict]:
    """
    Run PySceneDetect with both ContentDetector and AdaptiveDetector,
    then merge and deduplicate results.

    Returns list of {start_seconds, end_seconds} for each detected scene.
    """
    video = open_video(video_path)
    fps = video.frame_rate

    # --- Pass 1: ContentDetector (pixel-level frame diffs) ---
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(
            threshold=content_threshold,
            min_scene_len=min_scene_len_frames,
        )
    )
    scene_manager.detect_scenes(video)
    content_scenes = scene_manager.get_scene_list()

    # --- Pass 2: AdaptiveDetector (adaptive thresholding) ---
    video = open_video(video_path)  # re-open
    scene_manager2 = SceneManager()
    scene_manager2.add_detector(
        AdaptiveDetector(
            adaptive_threshold=adaptive_threshold,
            min_scene_len=min_scene_len_frames,
        )
    )
    scene_manager2.detect_scenes(video)
    adaptive_scenes = scene_manager2.get_scene_list()

    # Collect all cut points (start of each scene except the first)
    cut_points = set()
    for scene_list in [content_scenes, adaptive_scenes]:
        for start, end in scene_list:
            cut_points.add(start.get_seconds())

    # Sort and convert to scene list
    cut_points = sorted(cut_points)

    return cut_points, fps


def detect_scenes_ffmpeg(
    video_path: str,
    threshold: float = 0.3,
) -> list[float]:
    """
    Use ffmpeg's scene detection filter as a second opinion.
    Returns list of timestamps (seconds) where scene changes were detected.
    """
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse pts_time from showinfo output
    timestamps = []
    for line in result.stderr.split("\n"):
        match = re.search(r"pts_time:([\d.]+)", line)
        if match:
            timestamps.append(float(match.group(1)))

    return timestamps


def merge_cut_points(
    pyscene_cuts: list[float],
    ffmpeg_cuts: list[float],
    merge_threshold: float = 0.15,
) -> list[float]:
    """
    Merge cut points from both detectors. If two cuts are within
    merge_threshold seconds of each other, keep only the PySceneDetect one
    (generally more accurate).
    """
    all_cuts = set()

    # Add all PySceneDetect cuts first (higher priority)
    for cut in pyscene_cuts:
        all_cuts.add(round(cut, 3))

    # Add ffmpeg cuts only if they're not near an existing cut
    for cut in ffmpeg_cuts:
        is_duplicate = any(abs(cut - existing) < merge_threshold for existing in all_cuts)
        if not is_duplicate:
            all_cuts.add(round(cut, 3))

    return sorted(all_cuts)


def cuts_to_shots(cut_points: list[float], video_duration: float) -> list[dict]:
    """
    Convert a sorted list of cut timestamps into shot ranges.
    Each shot is {start_seconds, end_seconds, duration}.
    """
    shots = []

    # First shot starts at 0
    all_boundaries = [0.0] + cut_points + [video_duration]

    for i in range(len(all_boundaries) - 1):
        start = all_boundaries[i]
        end = all_boundaries[i + 1]
        if end - start < 0.01:  # Skip tiny gaps
            continue
        shots.append({
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "duration": round(end - start, 3),
        })

    return shots


def detect_cuts(
    video_path: str,
    content_threshold: float = 27.0,
    adaptive_threshold: float = 3.0,
    min_scene_len_frames: int = 5,
    use_ffmpeg: bool = True,
    video_duration: float = None,
) -> list[dict]:
    """
    Full cut detection pipeline. Returns list of shot dicts.
    """
    print("  Running PySceneDetect (ContentDetector + AdaptiveDetector)...")
    pyscene_cuts, fps = detect_scenes_pyscenedetect(
        video_path,
        content_threshold=content_threshold,
        adaptive_threshold=adaptive_threshold,
        min_scene_len_frames=min_scene_len_frames,
    )
    print(f"  PySceneDetect found {len(pyscene_cuts)} cut points")

    ffmpeg_cuts = []
    if use_ffmpeg:
        print("  Running ffmpeg scene filter...")
        ffmpeg_cuts = detect_scenes_ffmpeg(video_path)
        print(f"  ffmpeg found {len(ffmpeg_cuts)} cut points")

    # Merge
    merged_cuts = merge_cut_points(pyscene_cuts, ffmpeg_cuts)
    print(f"  Merged: {len(merged_cuts)} unique cut points")

    # Convert to shot list
    if video_duration is None:
        # Get duration from ffprobe
        from pipeline.profiler import get_video_metadata
        video_duration = get_video_metadata(video_path)["duration"]

    shots = cuts_to_shots(merged_cuts, video_duration)
    print(f"  Generated {len(shots)} shots")

    return shots


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.cut_detection <video_path>")
        sys.exit(1)

    shots = detect_cuts(sys.argv[1])
    print(json.dumps(shots[:10], indent=2))
    print(f"... ({len(shots)} total shots)")
