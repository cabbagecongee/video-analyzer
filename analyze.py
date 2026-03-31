"""
Shot Analysis Pipeline - Main Entry Point

A multi-stage pipeline for accurate shot-by-shot video analysis.

Usage:
    python analyze.py <video_path> [options]

Options:
    --model MODEL       Gemini model to use (default: gemini-2.5-pro)
    --fps N             Gemini sampling FPS for clip mode (default: adaptive)
    --whisper-model M   Whisper model size: tiny, base, small, medium, large (default: base)
    --batch-size N      Shots per Gemini batch (default: adaptive)
    --skip-refinement   Skip timestamp refinement step (faster but less precise)
    --skip-whisper      Skip Whisper transcription (use Gemini for audio too)
    --output PATH       Output JSON path (default: <video>_analysis.json)
    --verbose           Print detailed progress
"""

import io
import os
import sys
import json
import time
import argparse
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_quality import analyze_json as quality_analyze, print_report as quality_print_report, get_video_duration

from pipeline.profiler import classify_video, get_adaptive_params
from pipeline.cut_detection import detect_cuts
from pipeline.refine_timestamps import refine_all_cuts
from pipeline.audio import extract_and_transcribe
from pipeline.visual_analysis import analyze_all_shots
from pipeline.stitcher import align_audio_to_shots, build_final_output


def run_pipeline(
    video_path: str,
    model: str = None,
    fps: int = None,
    whisper_model: str = "base",
    batch_size: int = None,
    skip_refinement: bool = False,
    skip_whisper: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Run the full analysis pipeline on a video.
    """
    start_time = time.time()

    # =========================================================
    # Stage 0: Profile the video and get adaptive parameters
    # =========================================================
    print("=" * 60)
    print("STAGE 0: Profiling video...")
    print("=" * 60)

    profile = classify_video(video_path)
    params = get_adaptive_params(profile)

    print(f"  Video type: {profile['video_type']}")
    print(f"  Duration: {profile['metadata']['duration']:.1f}s")
    print(f"  Resolution: {profile['metadata']['width']}x{profile['metadata']['height']}")
    print(f"  Native FPS: {profile['metadata']['native_fps']:.1f}")
    print(f"  Edit pace: {profile['pace']['pace']} "
          f"({profile['pace']['cuts_per_second']} cuts/sec in sample)")

    # Override adaptive params with CLI args if provided
    if model:
        params["gemini_model"] = model
    if fps:
        params["gemini_fps"] = fps
    if batch_size:
        params["shots_per_batch"] = batch_size

    print(f"\n  Adaptive parameters:")
    print(f"    Scene threshold: {params['scene_threshold']}")
    print(f"    Gemini FPS: {params['gemini_fps']}")
    print(f"    Batch size: {params['shots_per_batch']}")
    print(f"    Model: {params['gemini_model']}")

    video_duration = profile["metadata"]["duration"]

    # =========================================================
    # Stage 1: Cut Detection
    # =========================================================
    print("\n" + "=" * 60)
    print("STAGE 1: Detecting cuts...")
    print("=" * 60)

    shots = detect_cuts(
        video_path,
        content_threshold=params["scene_threshold"],
        adaptive_threshold=params["adaptive_threshold"],
        min_scene_len_frames=params["min_scene_len_frames"],
        video_duration=video_duration,
    )

    print(f"  Found {len(shots)} shots")

    # =========================================================
    # Stage 2: Timestamp Refinement
    # =========================================================
    if not skip_refinement:
        print("\n" + "=" * 60)
        print("STAGE 2: Refining timestamps...")
        print("=" * 60)

        shots = refine_all_cuts(
            video_path, shots,
            window=params["refinement_window"],
            refinement_fps=30,
        )
    else:
        print("\n[Skipping timestamp refinement]")

    # =========================================================
    # Stage 3: Audio Extraction & Transcription
    # =========================================================
    transcript = {"words": [], "segments": [], "full_text": ""}

    if not skip_whisper and profile["metadata"]["has_audio"]:
        print("\n" + "=" * 60)
        print("STAGE 3: Extracting and transcribing audio...")
        print("=" * 60)

        try:
            # Build a prompt to help Whisper with brand names / domain terms
            # based on the video filename
            whisper_prompt = build_whisper_prompt(video_path)

            transcript = extract_and_transcribe(
                video_path,
                model_size=whisper_model,
                initial_prompt=whisper_prompt,
            )
            print(f"  Transcribed: {transcript['full_text'][:100]}...")
        except Exception as e:
            print(f"  WARNING: Whisper transcription failed: {e}")
            print(f"  Falling back to Gemini for audio analysis")
    else:
        if skip_whisper:
            print("\n[Skipping Whisper transcription]")
        else:
            print("\n[No audio track detected]")

    # =========================================================
    # Stage 4: Visual Analysis (Chunked Gemini)
    # =========================================================
    print("\n" + "=" * 60)
    print("STAGE 4: Analyzing visuals with Gemini...")
    print("=" * 60)

    # Build context string based on video type
    context = build_context_hint(profile)

    visual_analyses = analyze_all_shots(
        video_path,
        shots,
        model=params["gemini_model"],
        fps=params["gemini_fps"],
        batch_size=params["shots_per_batch"],
        context=context,
    )

    print(f"  Analyzed {len(visual_analyses)} shots")

    # =========================================================
    # Stage 5: Audio Alignment
    # =========================================================
    print("\n" + "=" * 60)
    print("STAGE 5: Aligning audio to shots...")
    print("=" * 60)

    merged_shots = align_audio_to_shots(shots, transcript, visual_analyses)
    print(f"  Aligned {len(merged_shots)} shots with audio")

    # =========================================================
    # Stage 6: Stitch & Validate
    # =========================================================
    print("\n" + "=" * 60)
    print("STAGE 6: Stitching and validating...")
    print("=" * 60)

    elapsed = time.time() - start_time
    usage_stats = {
        "pipeline_time_seconds": round(elapsed, 1),
        "whisper_model": whisper_model if not skip_whisper else "skipped",
        "gemini_model": params["gemini_model"],
    }

    output = build_final_output(
        merged_shots,
        video_duration,
        profile=profile,
        usage_stats=usage_stats,
    )

    print(f"\n  Pipeline complete in {elapsed:.1f}s")
    print(f"  Final output: {len(output['shots'])} shots")

    return output


def build_context_hint(profile: dict) -> str:
    """
    Build a context hint for Gemini based on video classification.
    Helps the model focus on what matters for this type of video.
    """
    video_type = profile["video_type"]
    pace = profile["pace"]["pace"]

    hints = []

    if video_type == "commercial":
        hints.append(
            "This is a commercial/advertisement. Pay special attention to: "
            "brand logos, product shots, text overlays, call-to-action graphics, "
            "and voiceover alignment with visuals."
        )
    elif video_type == "music_video":
        hints.append(
            "This is a music video with fast-paced editing. Pay special attention to: "
            "rapid shot changes, performance footage vs. narrative footage, "
            "lighting changes, color grading shifts, and visual effects."
        )
    elif video_type == "cinematic_short":
        hints.append(
            "This is a cinematic piece. Pay special attention to: "
            "camera movements (even subtle ones), depth of field choices, "
            "lighting direction and quality, and precise framing."
        )

    if pace == "fast":
        hints.append(
            "This video has fast-paced editing. Many shots are very short "
            "(under 1 second). Be precise about each individual shot."
        )

    return " ".join(hints)


def build_whisper_prompt(video_path: str) -> str:
    """
    Build an initial_prompt for Whisper based on the video filename.
    This helps Whisper correctly transcribe brand names and domain terms
    that it would otherwise mishear (e.g., "bitcoin" -> "hey coin").
    """
    filename = os.path.basename(video_path).lower()

    # Base prompt with common terms across all videos
    terms = [
        "prediction markets", "cryptocurrency", "Bitcoin", "blockchain",
        "AI", "artificial intelligence",
    ]

    # Add video-specific terms based on filename
    if "polymarket" in filename or "gte" in filename:
        terms.extend([
            "Polymarket", "GTE", "prediction market", "betting",
            "odds", "probability", "outcome", "wager",
        ])
    elif "claude" in filename:
        terms.extend([
            "Claude", "Anthropic", "AI assistant", "language model",
        ])
    elif "micro1" in filename or "micro" in filename:
        terms.extend([
            "Micro1", "hiring", "engineering", "talent", "remote",
        ])
    elif "drake" in filename or "first person" in filename:
        terms.extend([
            "Drake", "J. Cole", "First Person Shooter", "hip-hop",
            "rap", "verse", "bars",
        ])
    elif "riot" in filename or "dor" in filename:
        terms.extend([
            "Riot Games", "Dor Brothers", "League of Legends",
            "gaming", "esports",
        ])

    return ", ".join(terms)


def main():
    parser = argparse.ArgumentParser(
        description="Accurate shot-by-shot video analysis pipeline"
    )
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--model", default=None,
                        help="Gemini model (default: adaptive)")
    parser.add_argument("--fps", type=int, default=None,
                        help="Gemini sampling FPS (default: adaptive)")
    parser.add_argument("--whisper-model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Shots per Gemini batch (default: adaptive)")
    parser.add_argument("--skip-refinement", action="store_true",
                        help="Skip timestamp refinement")
    parser.add_argument("--skip-whisper", action="store_true",
                        help="Skip Whisper transcription")
    parser.add_argument("--output", default=None,
                        help="Output JSON path")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("--rerun-weak", action="store_true",
                        help="After pipeline, re-analyze shots with thin compositions or unknown fields")
    parser.add_argument("--rerun-min-words", type=int, default=10,
                        help="Minimum composition word count for --rerun-weak (default: 10)")

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    # Run pipeline
    result = run_pipeline(
        video_path=args.video_path,
        model=args.model,
        fps=args.fps,
        whisper_model=args.whisper_model,
        batch_size=args.batch_size,
        skip_refinement=args.skip_refinement,
        skip_whisper=args.skip_whisper,
        verbose=args.verbose,
    )

    # Save output
    output_path = args.output
    if not output_path:
        video_name = os.path.splitext(os.path.basename(args.video_path))[0]
        output_dir = os.path.join("analysis", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, video_name + "_analysis.json")
    else:
        output_dir = os.path.dirname(output_path)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved {len(result['shots'])} shots to {output_path}")

    # Run quality check and save report
    video_duration = get_video_duration(args.video_path)
    quality = quality_analyze(result, video_duration)
    buf = io.StringIO()
    with redirect_stdout(buf):
        quality_print_report(quality)
    report_text = buf.getvalue()
    print(report_text)
    quality_path = os.path.join(output_dir, video_name + "_quality.txt")
    with open(quality_path, "w") as f:
        f.write(report_text)
    print(f"Saved quality report to {quality_path}")

    # Optionally re-analyze weak shots
    if args.rerun_weak:
        from rerun_weak_shots import rerun_shots
        print("\n" + "=" * 60)
        print("RERUN: Re-analyzing weak shots...")
        print("=" * 60)
        rerun_shots(output_path, args.video_path, args.rerun_min_words, args.model or "gemini-2.5-pro")


if __name__ == "__main__":
    main()