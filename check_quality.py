"""
Shot Analysis Quality Checker

Aligned to the evaluation criteria from the project README:
  1. Shot boundaries line up with real cuts
  2. Motion/angle/shot type correct
  3. Audio text accurate to what is said

Also checks robustness across different video types and code clarity.

Usage:
    python check_quality.py analysis.json [--video video.mp4]
    python check_quality.py analysis.json --compare baseline.json
"""

import json
import sys
import re
import argparse
from collections import Counter


def parse_timestamp(time_str: str) -> float:
    """Parse MM:SS.mmm to seconds."""
    match = re.match(r"(\d+):(\d+)\.(\d+)", str(time_str))
    if not match:
        return 0.0
    return int(match.group(1)) * 60 + int(match.group(2)) + int(match.group(3)) / 1000.0


def analyze_json(data: dict, video_duration: float = None) -> dict:
    """Run all quality checks on a shot analysis JSON."""
    shots = data.get("shots", [])
    if not shots:
        return {"error": "No shots found"}

    results = {}

    durations = []
    for s in shots:
        start = parse_timestamp(s.get("startTime", "00:00.000"))
        end = parse_timestamp(s.get("endTime", "00:00.000"))
        durations.append(round(end - start, 3))

    # ==================================================================
    # CRITERION 1: "Shot boundaries line up with real cuts"
    # ==================================================================
    results["criterion_1_boundaries"] = {}
    c1 = results["criterion_1_boundaries"]

    c1["total_shots"] = len(shots)
    c1["total_duration_covered"] = round(sum(durations), 3)
    c1["avg_shot_duration"] = round(sum(durations) / len(durations), 3)
    c1["median_shot_duration"] = round(sorted(durations)[len(durations) // 2], 3)
    c1["min_shot_duration"] = min(durations)
    c1["max_shot_duration"] = max(durations)

    # Gaps and overlaps (timestamps should be seamless)
    gaps = []
    overlaps = []
    for i in range(len(shots) - 1):
        current_end = parse_timestamp(shots[i].get("endTime", "00:00.000"))
        next_start = parse_timestamp(shots[i + 1].get("startTime", "00:00.000"))
        diff = round(next_start - current_end, 3)
        if diff > 0.05:
            gaps.append({"between_shots": [i + 1, i + 2], "gap_seconds": diff})
        elif diff < -0.05:
            overlaps.append({"between_shots": [i + 1, i + 2], "overlap_seconds": -diff})

    c1["gaps"] = gaps
    c1["gap_count"] = len(gaps)
    c1["overlaps"] = overlaps
    c1["overlap_count"] = len(overlaps)

    # Coverage
    first_start = parse_timestamp(shots[0].get("startTime", "00:00.000"))
    c1["starts_at_zero"] = first_start < 0.1

    if video_duration:
        last_end = parse_timestamp(shots[-1].get("endTime", "00:00.000"))
        c1["video_duration"] = video_duration
        c1["last_shot_end"] = last_end
        c1["coverage_pct"] = round((last_end / video_duration) * 100, 1)

    # Likely missed cuts (very long shots in a fast-paced video)
    likely_missed_cuts = [
        {"shot": i + 1, "duration": d, "startTime": shots[i]["startTime"]}
        for i, d in enumerate(durations) if d > 15
    ]
    c1["likely_missed_cuts"] = likely_missed_cuts

    # Likely false cuts (impossibly short shots)
    likely_false_cuts = [
        {"shot": i + 1, "duration": d, "startTime": shots[i]["startTime"]}
        for i, d in enumerate(durations) if d < 0.08
    ]
    c1["likely_false_cuts"] = likely_false_cuts

    # ==================================================================
    # CRITERION 2: "Motion/angle/shot type correct"
    # These can't be verified without watching, but we flag things
    # that are likely wrong so you know where to spot-check.
    # ==================================================================
    results["criterion_2_visual_accuracy"] = {}
    c2 = results["criterion_2_visual_accuracy"]

    # --- Movement: flag all "static" shots for manual review ---
    # The README specifically calls out "shots labeled as static even though
    # the camera is clearly pushing in / zooming"
    static_shots = []
    for i, s in enumerate(shots):
        mov = s.get("movement", "").lower()
        if "static" in mov or "locked" in mov:
            static_shots.append({
                "shot": i + 1,
                "startTime": s["startTime"],
                "duration": durations[i],
                "movement": s.get("movement", ""),
                "composition": s.get("composition", "")[:80] + "...",
            })

    c2["static_shots_to_verify"] = static_shots
    c2["static_count"] = len(static_shots)
    c2["static_pct"] = round((len(static_shots) / len(shots)) * 100, 1)

    # Movement variety (if everything is "static" or "pan", something's wrong)
    movement_types = Counter()
    for s in shots:
        m = s.get("movement", "unknown").strip().lower()
        if not m or m in ("unknown", "n/a"):
            movement_types["unknown"] += 1
        elif "static" in m or "locked" in m:
            movement_types["static"] += 1
        elif any(w in m for w in ["push", "dolly in", "move in", "creep in"]):
            movement_types["push in / dolly in"] += 1
        elif any(w in m for w in ["pull", "dolly out", "dolly back", "move out"]):
            movement_types["pull out / dolly back"] += 1
        elif "pan" in m:
            movement_types["pan"] += 1
        elif "tilt" in m:
            movement_types["tilt"] += 1
        elif "track" in m:
            movement_types["tracking"] += 1
        elif "zoom" in m:
            movement_types["zoom"] += 1
        elif "handheld" in m or "hand-held" in m:
            movement_types["handheld"] += 1
        elif "crane" in m:
            movement_types["crane"] += 1
        elif "steadicam" in m or "steady" in m:
            movement_types["steadicam"] += 1
        else:
            movement_types["other: " + m[:30]] += 1

    c2["movement_distribution"] = dict(movement_types.most_common())
    c2["movement_variety"] = len(movement_types)

    # Shot type distribution
    shot_types = Counter(s.get("shotType", "unknown").lower() for s in shots)
    c2["shot_type_distribution"] = dict(shot_types.most_common())

    # Angle distribution
    angles = Counter(s.get("angle", "unknown").lower() for s in shots)
    c2["angle_distribution"] = dict(angles.most_common())

    # Composition detail
    comp_lengths = [len(s.get("composition", "").split()) for s in shots]
    c2["avg_composition_words"] = round(sum(comp_lengths) / len(comp_lengths), 1)

    # Flag thin compositions (evaluators want enough detail to recreate the shot)
    thin_compositions = [
        {
            "shot": i + 1,
            "startTime": shots[i]["startTime"],
            "word_count": comp_lengths[i],
            "composition": shots[i].get("composition", ""),
        }
        for i in range(len(shots)) if comp_lengths[i] < 10
    ]
    c2["thin_compositions"] = thin_compositions

    # Flag repetitive/generic compositions
    comp_texts = [s.get("composition", "").lower() for s in shots]
    repetitive_starts = Counter(c.split(".")[0][:50] for c in comp_texts if c)
    c2["most_repeated_composition_starts"] = dict(
        (k, v) for k, v in repetitive_starts.most_common(3) if v > 2
    )

    # ==================================================================
    # CRITERION 3: "Audio text accurate to what is said"
    # Again can't fully verify, but we surface the audio for quick review.
    # ==================================================================
    results["criterion_3_audio_accuracy"] = {}
    c3 = results["criterion_3_audio_accuracy"]

    # List all audio fields for quick skim review
    audio_entries = []
    for i, s in enumerate(shots):
        audio = s.get("audio", "None")
        if audio and audio not in ("None", "none", ""):
            audio_entries.append({
                "shot": i + 1,
                "startTime": s["startTime"],
                "endTime": s["endTime"],
                "audio": audio,
            })

    c3["audio_entries"] = audio_entries
    c3["shots_with_audio"] = len(audio_entries)
    c3["shots_without_audio"] = len(shots) - len(audio_entries)
    c3["audio_coverage_pct"] = round((len(audio_entries) / len(shots)) * 100, 1)

    # Flag shots where audio seems too long for the shot duration
    # (might indicate VO misalignment — assigned to wrong shot)
    misaligned_audio = []
    for i, s in enumerate(shots):
        audio = s.get("audio", "")
        if not audio or audio in ("None", "none"):
            continue
        word_count = len(audio.split())
        duration = durations[i]
        # Average speaking rate ~2.5 words/sec. Flag if > 4 words/sec.
        if duration > 0 and word_count / duration > 4.0:
            misaligned_audio.append({
                "shot": i + 1,
                "startTime": s["startTime"],
                "duration": duration,
                "word_count": word_count,
                "words_per_sec": round(word_count / duration, 1),
                "audio": audio[:100] + ("..." if len(audio) > 100 else ""),
            })

    c3["possibly_misaligned_audio"] = misaligned_audio

    # Flag consecutive shots with no audio in what might be a VO section
    # (could indicate VO dumped into one shot instead of spread across several)
    audio_dump_suspects = []
    for i, s in enumerate(shots):
        audio = s.get("audio", "")
        if not audio or audio in ("None", "none"):
            continue
        word_count = len(audio.split())
        if word_count > 30:
            audio_dump_suspects.append({
                "shot": i + 1,
                "startTime": s["startTime"],
                "word_count": word_count,
                "audio_preview": audio[:120] + "...",
            })

    c3["possible_audio_dumps"] = audio_dump_suspects

    # ==================================================================
    # FIELD COMPLETENESS (supporting check)
    # ==================================================================
    results["field_completeness"] = {}
    fc = results["field_completeness"]

    expected_fields = ["cut", "shotType", "angle", "lens", "focus", "movement",
                       "composition", "graphicsOverlays", "audio"]
    missing = Counter()
    unknown = Counter()
    empty = Counter()

    for s in shots:
        for field in expected_fields:
            val = s.get(field, "")
            if field not in s:
                missing[field] += 1
            elif not val or val == "":
                empty[field] += 1
            elif str(val).lower() in ("unknown", "n/a"):
                unknown[field] += 1

    fc["missing"] = dict(missing) if missing else "all present"
    fc["unknown"] = dict(unknown) if unknown else "none"
    fc["empty"] = dict(empty) if empty else "none"

    # Graphics overlays
    graphics_filled = sum(1 for s in shots
                          if s.get("graphicsOverlays", "None") not in ("None", "none", "", None))
    fc["shots_with_graphics"] = graphics_filled

    return results


def get_video_duration(video_path: str) -> float:
    """Get video duration via ffprobe."""
    import subprocess
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception:
        return None


def print_report(results: dict):
    """Pretty print the quality report, organized by evaluation criteria."""

    print("\n" + "=" * 70)
    print("  SHOT ANALYSIS QUALITY REPORT")
    print("  (aligned to evaluation criteria)")
    print("=" * 70)

    # ---- CRITERION 1 ----
    c1 = results["criterion_1_boundaries"]
    print(f"\n{'─' * 70}")
    print(f"  CRITERION 1: Shot boundaries line up with real cuts")
    print(f"{'─' * 70}")
    print(f"  Total shots:       {c1['total_shots']}")
    print(f"  Duration covered:  {c1['total_duration_covered']:.1f}s")
    if "video_duration" in c1:
        print(f"  Video duration:    {c1['video_duration']:.1f}s")
        print(f"  Coverage:          {c1['coverage_pct']}%")
    print(f"  Starts at zero:    {'PASS' if c1['starts_at_zero'] else 'FAIL'}")
    print(f"  Gaps:              {c1['gap_count']}  {'PASS' if c1['gap_count'] == 0 else 'WARN'}")
    print(f"  Overlaps:          {c1['overlap_count']}  {'PASS' if c1['overlap_count'] == 0 else 'WARN'}")

    if c1["gaps"]:
        print(f"\n  Gaps (first 5):")
        for g in c1["gaps"][:5]:
            print(f"    {g['gap_seconds']:.3f}s gap between shots {g['between_shots']}")

    if c1["overlaps"]:
        print(f"\n  Overlaps (first 5):")
        for o in c1["overlaps"][:5]:
            print(f"    {o['overlap_seconds']:.3f}s overlap between shots {o['between_shots']}")

    print(f"\n  Shot durations:  avg={c1['avg_shot_duration']:.2f}s  "
          f"median={c1['median_shot_duration']:.2f}s  "
          f"min={c1['min_shot_duration']:.3f}s  max={c1['max_shot_duration']:.2f}s")

    if c1["likely_missed_cuts"]:
        print(f"\n  LIKELY MISSED CUTS (shots >15s — verify these have no internal edits):")
        for s in c1["likely_missed_cuts"]:
            print(f"    Shot {s['shot']} at {s['startTime']} ({s['duration']:.1f}s)")

    if c1["likely_false_cuts"]:
        print(f"\n  LIKELY FALSE CUTS (shots <80ms — probably not real shots):")
        for s in c1["likely_false_cuts"]:
            print(f"    Shot {s['shot']} at {s['startTime']} ({s['duration']*1000:.0f}ms)")

    # ---- CRITERION 2 ----
    c2 = results["criterion_2_visual_accuracy"]
    print(f"\n{'─' * 70}")
    print(f"  CRITERION 2: Motion / angle / shot type correct")
    print(f"{'─' * 70}")

    print(f"\n  Movement distribution:")
    for movement, count in c2["movement_distribution"].items():
        bar = "█" * min(count, 40)
        print(f"    {movement:28s} {count:3d} {bar}")

    print(f"\n  Movement variety: {c2['movement_variety']} distinct types")
    if c2["static_pct"] > 60:
        print(f"  WARNING: {c2['static_pct']}% shots labeled static — likely over-reported")

    print(f"\n  Shot type distribution:")
    for st, count in c2["shot_type_distribution"].items():
        print(f"    {st:28s} {count:3d}")

    print(f"\n  Angle distribution:")
    for a, count in c2["angle_distribution"].items():
        print(f"    {a:28s} {count:3d}")

    print(f"\n  Composition detail: avg {c2['avg_composition_words']} words/shot")
    if c2["thin_compositions"]:
        print(f"  THIN COMPOSITIONS ({len(c2['thin_compositions'])} shots — not enough detail to recreate):")
        for tc in c2["thin_compositions"][:5]:
            print(f"    Shot {tc['shot']} at {tc['startTime']} ({tc['word_count']} words): "
                  f"\"{tc['composition'][:70]}...\"")

    if c2["static_shots_to_verify"]:
        print(f"\n  STATIC SHOTS TO SPOT-CHECK ({c2['static_count']} total — are these really static?):")
        for ss in c2["static_shots_to_verify"][:8]:
            print(f"    Shot {ss['shot']} at {ss['startTime']} ({ss['duration']:.2f}s): {ss['composition']}")

    # ---- CRITERION 3 ----
    c3 = results["criterion_3_audio_accuracy"]
    print(f"\n{'─' * 70}")
    print(f"  CRITERION 3: Audio text accurate to what is said")
    print(f"{'─' * 70}")
    print(f"  Shots with audio:    {c3['shots_with_audio']}/{c3['shots_with_audio'] + c3['shots_without_audio']} "
          f"({c3['audio_coverage_pct']}%)")

    if c3["possibly_misaligned_audio"]:
        print(f"\n  POSSIBLY MISALIGNED AUDIO (too many words for shot duration):")
        for ma in c3["possibly_misaligned_audio"][:5]:
            print(f"    Shot {ma['shot']} at {ma['startTime']} ({ma['duration']:.2f}s, "
                  f"{ma['words_per_sec']} words/sec): \"{ma['audio']}\"")

    if c3["possible_audio_dumps"]:
        print(f"\n  POSSIBLE AUDIO DUMPS (>30 words in one shot — VO lumped together?):")
        for ad in c3["possible_audio_dumps"][:5]:
            print(f"    Shot {ad['shot']} at {ad['startTime']} ({ad['word_count']} words): "
                  f"\"{ad['audio_preview']}\"")

    print(f"\n  ALL AUDIO (skim for accuracy):")
    for ae in c3["audio_entries"]:
        audio_preview = ae["audio"][:90] + ("..." if len(ae["audio"]) > 90 else "")
        print(f"    [{ae['startTime']} - {ae['endTime']}] {audio_preview}")

    # ---- FIELD COMPLETENESS ----
    fc = results["field_completeness"]
    print(f"\n{'─' * 70}")
    print(f"  Field completeness")
    print(f"{'─' * 70}")
    print(f"  Missing fields: {fc['missing']}")
    print(f"  Unknown values: {fc['unknown']}")
    print(f"  Empty values:   {fc['empty']}")
    print(f"  Shots with graphics overlays: {fc['shots_with_graphics']}")
    print()


def compare_analyses(results_a: dict, results_b: dict, label_a: str, label_b: str):
    """Print a side-by-side comparison of two analyses."""
    c1a = results_a["criterion_1_boundaries"]
    c1b = results_b["criterion_1_boundaries"]
    c2a = results_a["criterion_2_visual_accuracy"]
    c2b = results_b["criterion_2_visual_accuracy"]
    c3a = results_a["criterion_3_audio_accuracy"]
    c3b = results_b["criterion_3_audio_accuracy"]

    print(f"\n{'=' * 70}")
    print(f"  COMPARISON: {label_a} vs {label_b}")
    print(f"{'=' * 70}")

    print(f"\n  {'Metric':<40s} {label_a:>12s} {label_b:>12s}")
    print(f"  {'─' * 64}")

    rows = [
        ("Total shots", c1a["total_shots"], c1b["total_shots"]),
        ("Duration covered (s)", c1a["total_duration_covered"], c1b["total_duration_covered"]),
        ("Avg shot duration (s)", c1a["avg_shot_duration"], c1b["avg_shot_duration"]),
        ("Gaps", c1a["gap_count"], c1b["gap_count"]),
        ("Overlaps", c1a["overlap_count"], c1b["overlap_count"]),
        ("Likely missed cuts", len(c1a["likely_missed_cuts"]), len(c1b["likely_missed_cuts"])),
        ("Likely false cuts", len(c1a["likely_false_cuts"]), len(c1b["likely_false_cuts"])),
        ("Static shots (%)", c2a["static_pct"], c2b["static_pct"]),
        ("Movement variety", c2a["movement_variety"], c2b["movement_variety"]),
        ("Avg composition words", c2a["avg_composition_words"], c2b["avg_composition_words"]),
        ("Thin compositions", len(c2a["thin_compositions"]), len(c2b["thin_compositions"])),
        ("Audio coverage (%)", c3a["audio_coverage_pct"], c3b["audio_coverage_pct"]),
        ("Misaligned audio", len(c3a["possibly_misaligned_audio"]), len(c3b["possibly_misaligned_audio"])),
        ("Audio dumps", len(c3a["possible_audio_dumps"]), len(c3b["possible_audio_dumps"])),
    ]

    for label, a, b in rows:
        print(f"  {label:<40s} {str(a):>12s} {str(b):>12s}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Shot analysis quality checker (aligned to evaluation criteria)"
    )
    parser.add_argument("json_path", help="Path to analysis JSON file")
    parser.add_argument("--video", default=None, help="Path to video (for duration check)")
    parser.add_argument("--compare", default=None, help="Path to second JSON for comparison")
    args = parser.parse_args()

    with open(args.json_path) as f:
        data = json.load(f)

    video_duration = None
    if args.video:
        video_duration = get_video_duration(args.video)

    results = analyze_json(data, video_duration)
    print_report(results)

    if args.compare:
        with open(args.compare) as f:
            data_b = json.load(f)
        results_b = analyze_json(data_b, video_duration)

        label_a = args.json_path.split("/")[-1]
        label_b = args.compare.split("/")[-1]
        compare_analyses(results, results_b, label_a, label_b)


if __name__ == "__main__":
    main()