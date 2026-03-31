"""
Rerun Gemini analysis on shots that have thin compositions or unknown field values.

Usage:
    python rerun_weak_shots.py analysis/polymarket/polymarket_analysis.json videos/polymarket.mp4
    python rerun_weak_shots.py analysis/polymarket/polymarket_analysis.json videos/polymarket.mp4 --min-words 15
"""

import io
import os
import sys
import json
import argparse
import re
from contextlib import redirect_stdout

from pipeline.visual_analysis import (
    get_client,
    extract_keyframes,
    analyze_shot_keyframes,
    format_timestamp,
)
from check_quality import analyze_json as quality_analyze, print_report as quality_print_report, get_video_duration


UNKNOWN_VALUES = {"unknown", "n/a", ""}


def parse_timestamp(ts: str) -> float:
    match = re.match(r"(\d+):(\d+)\.(\d+)", str(ts))
    if not match:
        return 0.0
    return int(match.group(1)) * 60 + int(match.group(2)) + int(match.group(3)) / 1000.0


def find_weak_shots(shots: list, min_words: int = 10) -> list[int]:
    """Return 0-based indices of shots with thin compositions or unknown field values."""
    weak = []
    fields_to_check = ["cut", "shotType", "angle", "lens", "focus", "movement"]
    for i, shot in enumerate(shots):
        composition = shot.get("composition", "")
        word_count = len(composition.split())
        has_thin_composition = word_count < min_words

        has_unknown = any(
            str(shot.get(f, "")).lower().strip() in UNKNOWN_VALUES
            for f in fields_to_check
        )

        if has_thin_composition or has_unknown:
            weak.append(i)

    return weak


def shot_to_pipeline_format(shot: dict) -> dict:
    """Convert a final output shot back to pipeline internal format for keyframe extraction."""
    return {
        "start_seconds": parse_timestamp(shot.get("startTime", "00:00.000")),
        "end_seconds": parse_timestamp(shot.get("endTime", "00:00.000")),
        "duration": (
            parse_timestamp(shot.get("endTime", "00:00.000"))
            - parse_timestamp(shot.get("startTime", "00:00.000"))
        ),
    }


def rerun_shots(json_path: str, video_path: str, min_words: int = 10, model: str = "gemini-2.5-pro"):
    with open(json_path) as f:
        data = json.load(f)

    shots = data.get("shots", [])
    weak_indices = find_weak_shots(shots, min_words)

    if not weak_indices:
        print("No weak shots found. Nothing to rerun.")
        return

    print(f"Found {len(weak_indices)} weak shots: {[i + 1 for i in weak_indices]}")
    print(f"Rerunning with model: {model}\n")

    client = get_client()

    for i in weak_indices:
        shot = shots[i]
        print(f"  Shot {i + 1} [{shot.get('startTime')} - {shot.get('endTime')}] "
              f"(composition: {len(shot.get('composition', '').split())} words)...")

        pipeline_shot = shot_to_pipeline_format(shot)
        frames = extract_keyframes(video_path, pipeline_shot)

        if not frames:
            print(f"    WARNING: Could not extract frames for shot {i + 1}, skipping.")
            continue

        try:
            new_analysis = analyze_shot_keyframes(client, frames, i, pipeline_shot, model)

            # Update fields — preserve startTime, endTime, id, audio from original
            for field in ["cut", "shotType", "angle", "lens", "focus", "movement",
                          "composition", "graphicsOverlays"]:
                if field in new_analysis:
                    shots[i][field] = new_analysis[field]

            print(f"    Updated: composition now {len(shots[i].get('composition', '').split())} words")
        except Exception as e:
            print(f"    ERROR on shot {i + 1}: {e}")
        finally:
            for f_path in frames:
                try:
                    os.remove(f_path)
                except OSError:
                    pass

    # Save updated JSON
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved updated analysis to {json_path}")

    # Re-run quality check
    output_dir = os.path.dirname(json_path)
    video_name = os.path.splitext(os.path.basename(json_path))[0].replace("_analysis", "")
    video_duration = get_video_duration(video_path)
    quality = quality_analyze(data, video_duration)
    buf = io.StringIO()
    with redirect_stdout(buf):
        quality_print_report(quality)
    report_text = buf.getvalue()
    print(report_text)
    quality_path = os.path.join(output_dir, video_name + "_quality.txt")
    with open(quality_path, "w") as f:
        f.write(report_text)
    print(f"Saved updated quality report to {quality_path}")


def main():
    parser = argparse.ArgumentParser(description="Rerun Gemini on weak shots")
    parser.add_argument("json_path", help="Path to analysis JSON file")
    parser.add_argument("video_path", help="Path to the original video file")
    parser.add_argument("--min-words", type=int, default=10,
                        help="Minimum composition word count (default: 10)")
    parser.add_argument("--model", default="gemini-2.5-pro",
                        help="Gemini model to use (default: gemini-2.5-pro)")
    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        print(f"Error: JSON file not found: {args.json_path}")
        sys.exit(1)
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    rerun_shots(args.json_path, args.video_path, args.min_words, args.model)


if __name__ == "__main__":
    main()
