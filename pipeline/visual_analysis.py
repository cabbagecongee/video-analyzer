"""
Stage 4: Chunked Visual Analysis via Gemini

Instead of sending the entire video in one pass, we send individual shots
or small batches to Gemini. This keeps token counts manageable and forces
the model to focus on each segment.

Two modes:
  - Keyframe mode: Extract key frames per shot, send as images
  - Clip mode: Extract short video clips per shot, send as video
"""

import os
import re
import json
import time
import subprocess
import mimetypes
from typing import Union
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


def get_client():
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def extract_keyframes(
    video_path: str,
    shot: dict,
    num_frames: int = 3,
    output_dir: str = "/tmp/keyframes",
) -> list[str]:
    """
    Extract keyframes from a shot: first, middle, and last frame.
    Returns list of image file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    start = shot["start_seconds"]
    end = shot["end_seconds"]
    duration = end - start

    if duration < 0.05:
        timestamps = [start]
    elif num_frames == 1:
        timestamps = [(start + end) / 2]
    elif num_frames == 2:
        timestamps = [start + 0.01, end - 0.01]
    else:
        timestamps = [
            start + 0.01,
            (start + end) / 2,
            max(start + 0.02, end - 0.01),
        ]

    frame_paths = []
    for i, ts in enumerate(timestamps):
        output_path = os.path.join(
            output_dir,
            f"shot_{start:.3f}_frame_{i}.jpg"
        )
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(ts),
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, text=True)
        if os.path.exists(output_path):
            frame_paths.append(output_path)

    return frame_paths


def extract_clip(
    video_path: str,
    shot: dict,
    output_dir: str = "/tmp/clips",
) -> str:
    """
    Extract a short video clip for a single shot.
    """
    os.makedirs(output_dir, exist_ok=True)

    start = shot["start_seconds"]
    duration = shot["duration"]
    output_path = os.path.join(output_dir, f"shot_{start:.3f}.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    return output_path


def build_shot_prompt(shot_index: int, shot: dict, context: str = "") -> str:
    """
    Build the analysis prompt for a single shot.
    """
    return f"""Analyze this shot from a video. This is shot #{shot_index + 1}.
It spans from {format_timestamp(shot['start_seconds'])} to {format_timestamp(shot['end_seconds'])} ({shot['duration']:.2f}s).

{context}

For this shot, provide:
- cut: Type of transition INTO this shot (hard cut, dissolve, match cut, whip pan, fade, J-cut, L-cut, etc.)
- shotType: Camera framing (extreme wide, wide, medium wide, medium, medium close-up, close-up, extreme close-up, insert, etc.)
- angle: Camera angle (eye-level, low angle, high angle, dutch/tilted, bird's eye, worm's eye, over-the-shoulder, POV, etc.)
- lens: Apparent focal length feel (ultra-wide, wide, normal, telephoto, macro)
- focus: What's in focus and depth of field description (shallow DOF with subject sharp and background blurred, deep focus with everything sharp, rack focus, soft focus, etc.)
- movement: Camera movement - be very specific (static/locked off, slow push in, pull back, pan left, pan right, tilt up, tilt down, dolly forward, dolly back, tracking left, tracking right, crane up, crane down, handheld, Steadicam, whip pan, zoom in, zoom out, orbital, etc.). If the camera is moving even slightly, do NOT call it static.
- composition: 3-5 detailed sentences describing: the framing and visual layout, all subjects visible and their positions, props/environment details, lighting quality and direction, color palette/grade, mood/atmosphere. Be specific enough that a filmmaker could recreate this frame.
- graphicsOverlays: Any text on screen (exact wording in quotes), logos, lower thirds, subtitles, motion graphics, VFX, or post-production elements. "None" if nothing.
- musicAndSFX: Describe ONLY background music and sound effects. Do NOT transcribe dialogue or voiceover — that is handled separately. Examples: "Upbeat electronic music", "Dramatic orchestral swell", "Whoosh transition SFX", "Ambient crowd noise". "None" if silence.

IMPORTANT: 
- Pay close attention to camera MOVEMENT. Even subtle pushes, drifts, or zooms should be noted. Only mark as "static" if the camera is truly locked off with zero movement.
- Be precise about what's actually visible on screen, not what you might expect.
- For graphicsOverlays, transcribe ALL visible text exactly as shown.
- For musicAndSFX, do NOT include any spoken words, dialogue, or voiceover. Only music and sound effects.

Output valid JSON:
{{
  "cut": "...",
  "shotType": "...",
  "angle": "...",
  "lens": "...",
  "focus": "...",
  "movement": "...",
  "composition": "...",
  "graphicsOverlays": "...",
  "musicAndSFX": "..."
}}"""


def build_batch_prompt(shots_batch: list[tuple[int, dict]], context: str = "") -> str:
    """
    Build a prompt for analyzing a batch of consecutive shots.
    """
    shot_descriptions = []
    for idx, shot in shots_batch:
        shot_descriptions.append(
            f"Shot #{idx + 1}: {format_timestamp(shot['start_seconds'])} to "
            f"{format_timestamp(shot['end_seconds'])} ({shot['duration']:.2f}s)"
        )

    shots_list = "\n".join(shot_descriptions)

    return f"""Analyze these consecutive shots from a video. 

{shots_list}

{context}

For EACH shot, provide:
- cut: Type of transition INTO this shot (hard cut, dissolve, match cut, whip pan, fade, J-cut, L-cut, etc.)
- shotType: Camera framing (extreme wide, wide, medium wide, medium, medium close-up, close-up, extreme close-up, insert)
- angle: Camera angle (eye-level, low angle, high angle, dutch/tilted, bird's eye, worm's eye, over-the-shoulder, POV)
- lens: Apparent focal length feel (ultra-wide, wide, normal, telephoto, macro)
- focus: What's in focus and depth of field
- movement: Camera movement - be VERY specific. Even subtle pushes, drifts, zooms, or handheld micro-movements should be noted. Only say "static" if truly locked off.
- composition: 3-5 detailed sentences. Describe framing, subjects, props, environment, lighting, color, mood. Be specific enough for recreation.
- graphicsOverlays: Exact text on screen in quotes, logos, graphics. "None" if nothing.
- musicAndSFX: ONLY background music and sound effects. Do NOT transcribe any dialogue or voiceover — that is handled separately. Examples: "Tense synth drone", "Bass drop", "Woosh SFX". "None" if silence.

IMPORTANT:
- Analyze each shot SEPARATELY. Each shot boundary is a distinct edit/cut.
- Camera movement: if there's even slight motion, describe it. "Static" means zero movement.
- Transcribe ALL visible text exactly.
- For musicAndSFX: NO spoken words, dialogue, or voiceover. Music and sound effects ONLY.

Output valid JSON array:
[
  {{
    "shot_index": 1,
    "cut": "...",
    "shotType": "...",
    "angle": "...",
    "lens": "...",
    "focus": "...",
    "movement": "...",
    "composition": "...",
    "graphicsOverlays": "...",
    "musicAndSFX": "..."
  }}
]"""


def analyze_shot_keyframes(
    client,
    frame_paths: list[str],
    shot_index: int,
    shot: dict,
    model: str = "gemini-2.5-pro",
    context: str = "",
) -> dict:
    """
    Analyze a single shot using extracted keyframes (image mode).
    """
    parts = []

    # Images first (Gemini recommends media before text)
    for path in frame_paths:
        with open(path, "rb") as f:
            image_data = f.read()
        parts.append(types.Part(inline_data=types.Blob(
            mime_type="image/jpeg",
            data=image_data,
        )))

    # Then the prompt
    parts.append(types.Part(text=build_shot_prompt(shot_index, shot, context)))

    response = client.models.generate_content(
        model=model,
        contents=types.Content(parts=parts),
        config=types.GenerateContentConfig(
            max_output_tokens=4096,
            temperature=0.2,
        ),
    )

    result = parse_json_response(response.text)
    if isinstance(result, list):
        return result[0] if result else {}
    return result

def analyze_batch_with_video(
    client,
    video_path: str,
    shots_batch: list[tuple[int, dict]],
    model: str = "gemini-2.5-pro",
    fps: int = 4,
    context: str = "",
) -> list[dict]:
    """
    Analyze a batch of consecutive shots by uploading the relevant video segment.
    """
    batch_start = shots_batch[0][1]["start_seconds"]
    batch_end = shots_batch[-1][1]["end_seconds"]
    batch_duration = batch_end - batch_start

    clip_path = f"/tmp/batch_{batch_start:.3f}_{batch_end:.3f}.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(batch_start),
        "-i", video_path,
        "-t", str(batch_duration),
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac",
        clip_path,
    ]
    subprocess.run(cmd, capture_output=True, text=True)

    # Upload clip
    mime_type, _ = mimetypes.guess_type(clip_path)
    if not mime_type:
        mime_type = "video/mp4"

    with open(clip_path, "rb") as f:
        uploaded = client.files.upload(file=f, config={"mime_type": mime_type})

    # Wait for processing
    while True:
        info = client.files.get(name=uploaded.name)
        if info.state.name == "ACTIVE":
            break
        if info.state.name == "FAILED":
            raise Exception(f"Upload failed: {info.error}")
        time.sleep(2)

    # Build request
    parts = [
        types.Part(
            file_data=types.FileData(file_uri=uploaded.uri),
            video_metadata=types.VideoMetadata(fps=fps),
        ),
        types.Part(text=build_batch_prompt(shots_batch, context)),
    ]

    response = client.models.generate_content(
        model=model,
        contents=types.Content(parts=parts),
        config=types.GenerateContentConfig(
            max_output_tokens=16384,
            media_resolution="MEDIA_RESOLUTION_HIGH",
            temperature=0.2,
        ),
    )

    # Cleanup
    try:
        os.remove(clip_path)
    except OSError:
        pass

    result = parse_json_response(response.text)
    if isinstance(result, list):
        return result
    elif isinstance(result, dict) and "shots" in result:
        return result["shots"]
    else:
        return [result]


def analyze_all_shots(
    video_path: str,
    shots: list[dict],
    model: str = "gemini-2.5-pro",
    fps: int = 4,
    batch_size: int = 5,
    use_keyframes_for_short: bool = True,
    short_threshold: float = 1.0,
    context: str = "",
    max_retries: int = 2,
) -> list[dict]:
    """
    Analyze all shots using a mix of keyframe and clip modes.
    Includes retry logic: if a batch fails, retries each shot individually.
    """
    client = get_client()
    results = []

    # Group shots into batches
    batches = []
    current_batch = []
    for i, shot in enumerate(shots):
        current_batch.append((i, shot))
        if len(current_batch) >= batch_size:
            batches.append(current_batch)
            current_batch = []
    if current_batch:
        batches.append(current_batch)

    print(f"  Analyzing {len(shots)} shots in {len(batches)} batches...")

    for batch_idx, batch in enumerate(batches):
        print(f"  Batch {batch_idx + 1}/{len(batches)} "
              f"(shots {batch[0][0] + 1}-{batch[-1][0] + 1})...")

        try:
            if len(batch) == 1 and batch[0][1]["duration"] < short_threshold:
                # Single short shot: use keyframe mode
                idx, shot = batch[0]
                frames = extract_keyframes(video_path, shot)
                analysis = analyze_shot_keyframes(
                    client, frames, idx, shot, model, context
                )
                results.append(analysis)

                for f in frames:
                    try:
                        os.remove(f)
                    except OSError:
                        pass
            else:
                # Batch: use clip mode
                batch_results = analyze_batch_with_video(
                    client, video_path, batch, model, fps, context
                )
                results.extend(batch_results)

        except Exception as e:
            print(f"  WARNING: Batch {batch_idx + 1} failed: {e}")
            print(f"  Retrying each shot individually with keyframes...")

            # RETRY: try each shot individually with keyframes
            for idx, shot in batch:
                retried = False
                for attempt in range(max_retries):
                    try:
                        print(f"    Retrying shot {idx + 1} (attempt {attempt + 1}/{max_retries})...")
                        frames = extract_keyframes(video_path, shot)
                        analysis = analyze_shot_keyframes(
                            client, frames, idx, shot, model, context
                        )
                        results.append(analysis)
                        retried = True

                        for f_path in frames:
                            try:
                                os.remove(f_path)
                            except OSError:
                                pass
                        break  # Success, stop retrying

                    except Exception as retry_e:
                        print(f"    Attempt {attempt + 1} failed: {retry_e}")
                        if attempt < max_retries - 1:
                            time.sleep(3)  # Brief pause before next retry

                if not retried:
                    print(f"    Shot {idx + 1} failed after {max_retries} retries, using placeholder")
                    results.append({
                        "cut": "unknown",
                        "shotType": "unknown",
                        "angle": "unknown",
                        "lens": "unknown",
                        "focus": "unknown",
                        "movement": "unknown",
                        "composition": "Analysis failed after retries",
                        "graphicsOverlays": "None",
                        "musicAndSFX": "None",
                    })

    return results


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS.mmm format."""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:06.3f}"


def parse_json_response(text: str) -> Union[dict, list]:
    """Extract JSON from Gemini response text."""
    # Try to find JSON in code blocks
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    json_str = match.group(1).strip() if match else text.strip()

    # Try parsing as-is
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Try finding array or object boundaries
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = json_str.find(start_char)
        end = json_str.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(json_str[start:end + 1])
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not parse JSON from response: {text[:200]}...")