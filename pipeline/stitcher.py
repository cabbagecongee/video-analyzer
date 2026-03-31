"""
Stage 5 & 6: Audio Alignment + Stitch & Validate

Stage 5: Maps Whisper transcript words to shot boundaries.
Stage 6: Merges everything into final JSON, validates, and formats.
"""

from pipeline.audio import get_words_for_shot
from pipeline.visual_analysis import format_timestamp


def align_audio_to_shots(
    shots: list[dict],
    transcript: dict,
    visual_analyses: list[dict],
) -> list[dict]:
    """
    Combine shot boundaries, visual analysis, and audio transcript
    into the final merged shot data.
    """
    merged = []

    for i, shot in enumerate(shots):
        # Get the visual analysis for this shot
        visual = visual_analyses[i] if i < len(visual_analyses) else {}

        # Get the transcript text for this shot's time window
        dialogue = get_words_for_shot(
            transcript,
            shot["start_seconds"],
            shot["end_seconds"],
        )

        # Build audio description from two clean sources:
        # 1. Whisper provides dialogue/VO (accurate word-level timestamps)
        # 2. Gemini provides music/SFX (musicAndSFX field, no dialogue)
        audio_parts = []

        if dialogue:
            audio_parts.append(f'VO/Dialogue: "{dialogue}"')

        # Get music/SFX from Gemini's dedicated field (no dialogue contamination)
        music_sfx = visual.get("musicAndSFX", "")
        if not music_sfx or music_sfx == "None":
            # Fallback: check old "audio" field but strip anything that looks like dialogue
            music_sfx = visual.get("audio", "")
        if music_sfx and music_sfx not in ("None", "none", ""):
            # Only keep it if it doesn't look like transcribed dialogue
            lower = music_sfx.lower()
            is_likely_dialogue = any(marker in lower for marker in [
                'vo:', 'voiceover:', 'dialogue:', 'narrator:', 'says', 'speaking',
            ])
            if not is_likely_dialogue:
                audio_parts.append(f'Music/SFX: {music_sfx}')

        audio_text = " | ".join(audio_parts) if audio_parts else "None"

        merged_shot = {
            "id": i + 1,
            "startTime": format_timestamp(shot["start_seconds"]),
            "endTime": format_timestamp(shot["end_seconds"]),
            "cut": visual.get("cut", "hard cut"),
            "shotType": visual.get("shotType", "unknown"),
            "angle": visual.get("angle", "unknown"),
            "lens": visual.get("lens", "unknown"),
            "focus": visual.get("focus", "unknown"),
            "movement": visual.get("movement", "unknown"),
            "composition": visual.get("composition", ""),
            "graphicsOverlays": visual.get("graphicsOverlays", "None"),
            "audio": audio_text,
        }

        merged.append(merged_shot)

    return merged


def validate_shots(shots: list[dict], video_duration: float) -> list[str]:
    """
    Validate the final shot list for common issues.
    Returns a list of warning strings.
    """
    warnings = []

    if not shots:
        warnings.append("No shots in output!")
        return warnings

    # Check first shot starts at or near 0
    first_start = parse_timestamp(shots[0]["startTime"])
    if first_start > 0.1:
        warnings.append(f"First shot starts at {shots[0]['startTime']}, expected ~00:00.000")

    # Check last shot ends near video duration
    last_end = parse_timestamp(shots[-1]["endTime"])
    if abs(last_end - video_duration) > 1.0:
        warnings.append(
            f"Last shot ends at {shots[-1]['endTime']} but video is "
            f"{format_timestamp(video_duration)}"
        )

    # Check for gaps and overlaps between consecutive shots
    for i in range(len(shots) - 1):
        current_end = parse_timestamp(shots[i]["endTime"])
        next_start = parse_timestamp(shots[i + 1]["startTime"])

        gap = next_start - current_end
        if gap > 0.1:
            warnings.append(
                f"Gap of {gap:.3f}s between shot {i + 1} and {i + 2} "
                f"({shots[i]['endTime']} -> {shots[i + 1]['startTime']})"
            )
        elif gap < -0.1:
            warnings.append(
                f"Overlap of {-gap:.3f}s between shot {i + 1} and {i + 2}"
            )

    # Check for suspiciously short or long shots
    for i, shot in enumerate(shots):
        start = parse_timestamp(shot["startTime"])
        end = parse_timestamp(shot["endTime"])
        duration = end - start

        if duration < 0.04:
            warnings.append(f"Shot {i + 1} is very short ({duration:.3f}s)")
        if duration > 30:
            warnings.append(
                f"Shot {i + 1} is very long ({duration:.1f}s) - "
                f"may contain missed cuts"
            )

    # Check for missing fields
    required_fields = ["cut", "shotType", "movement", "composition"]
    for i, shot in enumerate(shots):
        for field in required_fields:
            value = shot.get(field, "")
            if not value or value == "unknown":
                warnings.append(f"Shot {i + 1} has missing/unknown {field}")

    return warnings


def fix_shot_continuity(shots: list[dict], video_duration: float) -> list[dict]:
    """
    Fix minor continuity issues:
    - Ensure first shot starts at 00:00.000
    - Ensure last shot ends at video duration
    - Close small gaps between shots
    """
    if not shots:
        return shots

    fixed = [dict(s) for s in shots]  # Deep copy

    # Fix first shot start
    fixed[0]["startTime"] = "00:00.000"

    # Fix last shot end
    fixed[-1]["endTime"] = format_timestamp(video_duration)

    # Close gaps: each shot's end should equal next shot's start
    for i in range(len(fixed) - 1):
        next_start = fixed[i + 1]["startTime"]
        fixed[i]["endTime"] = next_start

    return fixed


def build_final_output(
    shots: list[dict],
    video_duration: float,
    profile: dict = None,
    usage_stats: dict = None,
) -> dict:
    """
    Build the final JSON output compatible with the viewer.
    """
    # Fix continuity
    fixed_shots = fix_shot_continuity(shots, video_duration)

    # Validate
    warnings = validate_shots(fixed_shots, video_duration)
    if warnings:
        print(f"  Validation warnings ({len(warnings)}):")
        for w in warnings:
            print(f"    - {w}")

    output = {"shots": fixed_shots}

    # Add metadata
    if profile:
        output["pipeline_metadata"] = {
            "video_type": profile.get("video_type", "unknown"),
            "pace": profile.get("pace", {}),
            "duration": video_duration,
        }

    if usage_stats:
        output["usage"] = usage_stats

    return output


def parse_timestamp(time_str: str) -> float: 
    """Parse MM:SS.mmm to seconds."""
    import re
    match = re.match(r"(\d+):(\d+)\.(\d+)", time_str)
    if not match:
        return 0.0
    minutes = int(match.group(1))
    seconds = int(match.group(2))
    millis = int(match.group(3))
    return minutes * 60 + seconds + millis / 1000.0