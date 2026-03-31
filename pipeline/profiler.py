"""
Stage 0: Video Profiler

Extracts metadata and characterizes the video BEFORE the main analysis.
This drives adaptive parameter selection (FPS, thresholds, batch sizes).
"""

import subprocess
import json
import os


def get_video_metadata(video_path: str) -> dict:
    """
    Use ffprobe to extract video metadata: duration, fps, resolution, codec, etc.
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    probe = json.loads(result.stdout)

    # Find the video stream
    video_stream = None
    audio_stream = None
    for stream in probe.get("streams", []):
        if stream["codec_type"] == "video" and video_stream is None:
            video_stream = stream
        elif stream["codec_type"] == "audio" and audio_stream is None:
            audio_stream = stream

    if not video_stream:
        raise RuntimeError("No video stream found")

    # Parse FPS (can be fractional like "30000/1001")
    fps_str = video_stream.get("r_frame_rate", "30/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        native_fps = float(num) / float(den)
    else:
        native_fps = float(fps_str)

    duration = float(probe["format"].get("duration", 0))
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))

    return {
        "duration": duration,
        "native_fps": native_fps,
        "width": width,
        "height": height,
        "has_audio": audio_stream is not None,
        "codec": video_stream.get("codec_name", "unknown"),
        "total_frames": int(float(video_stream.get("nb_frames", duration * native_fps))),
        "file_size_mb": os.path.getsize(video_path) / (1024 * 1024),
    }


def estimate_edit_pace(video_path: str, sample_duration: float = 10.0) -> dict:
    """
    Quick scene-change scan on a sample of the video to estimate editing pace.
    Uses ffmpeg's scene detection filter on the first N seconds.
    Returns estimated cuts/second to classify as slow, medium, or fast-paced.
    """
    cmd = [
        "ffmpeg", "-i", video_path,
        "-t", str(sample_duration),
        "-vf", "select='gt(scene,0.3)',showinfo",
        "-vsync", "vfr",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Count scene changes from stderr (showinfo outputs there)
    scene_count = result.stderr.count("pts_time:")

    cuts_per_second = scene_count / sample_duration if sample_duration > 0 else 0

    if cuts_per_second > 2.0:
        pace = "fast"
    elif cuts_per_second > 0.5:
        pace = "medium"
    else:
        pace = "slow"

    return {
        "sample_duration": sample_duration,
        "scene_changes_in_sample": scene_count,
        "cuts_per_second": round(cuts_per_second, 2),
        "pace": pace,
    }


def classify_video(video_path: str, gemini_client=None) -> dict:
    """
    Optional: Use Gemini to classify the video type from a few sampled frames.
    Falls back to heuristics if no client is provided.
    """
    metadata = get_video_metadata(video_path)
    pace_info = estimate_edit_pace(video_path)

    # Heuristic classification based on duration and pace
    duration = metadata["duration"]
    pace = pace_info["pace"]

    if duration < 90 and pace in ("medium", "fast"):
        video_type = "commercial"
    elif duration < 90 and pace == "slow":
        video_type = "cinematic_short"
    elif pace == "fast":
        video_type = "music_video"
    else:
        video_type = "general"

    return {
        "metadata": metadata,
        "pace": pace_info,
        "video_type": video_type,
    }


def get_adaptive_params(profile: dict) -> dict:
    """
    Based on the video profile, return tuned parameters for each pipeline stage.
    """
    pace = profile["pace"]["pace"]
    video_type = profile["video_type"]
    duration = profile["metadata"]["duration"]

    # Cut detection sensitivity
    if pace == "fast":
        # Fast editing: lower threshold to catch rapid cuts
        scene_threshold = 20.0
        adaptive_threshold = 2.5
        min_scene_len_frames = 3
    elif pace == "medium":
        scene_threshold = 27.0
        adaptive_threshold = 3.0
        min_scene_len_frames = 5
    else:
        scene_threshold = 30.0
        adaptive_threshold = 3.5
        min_scene_len_frames = 8

    # Gemini sampling FPS
    if pace == "fast":
        gemini_fps = 8
    elif pace == "medium":
        gemini_fps = 4
    else:
        gemini_fps = 2

    # Batch size for chunked Gemini analysis
    if duration > 180:
        shots_per_batch = 3
    elif duration > 60:
        shots_per_batch = 5
    else:
        shots_per_batch = 8

    # Refinement window (seconds around each cut to do high-FPS comparison)
    refinement_window = 0.25  # ±250ms

    # Gemini model selection
    # 2.5 Pro is better at timestamps, 3 Pro better at descriptions
    gemini_model = "gemini-2.5-pro"

    return {
        "scene_threshold": scene_threshold,
        "adaptive_threshold": adaptive_threshold,
        "min_scene_len_frames": min_scene_len_frames,
        "gemini_fps": gemini_fps,
        "shots_per_batch": shots_per_batch,
        "refinement_window": refinement_window,
        "gemini_model": gemini_model,
        "video_type": video_type,
        "pace": pace,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.profiler <video_path>")
        sys.exit(1)

    path = sys.argv[1]
    profile = classify_video(path)
    params = get_adaptive_params(profile)

    print("=== Video Profile ===")
    print(json.dumps(profile, indent=2))
    print("\n=== Adaptive Parameters ===")
    print(json.dumps(params, indent=2))
