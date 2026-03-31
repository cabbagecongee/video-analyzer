# Shot Analysis Pipeline

A multi-stage pipeline for accurate shot-by-shot video analysis. Combines deterministic computer vision (PySceneDetect, ffmpeg, Whisper) with chunked LLM analysis (Gemini) to produce precise, structured shot data.

---

## Table of Contents

- [Setup](#setup)
- [Quick Start](#quick-start)
- [Output Structure](#output-structure)
- [Pipeline Architecture](#pipeline-architecture)
  - [Stage 0: Video Profiler](#stage-0-video-profiler)
  - [Stage 1: Cut Detection](#stage-1-cut-detection)
  - [Stage 2: Timestamp Refinement](#stage-2-timestamp-refinement)
  - [Stage 3: Audio Transcription](#stage-3-audio-transcription)
  - [Stage 4: Visual Analysis](#stage-4-visual-analysis)
  - [Stage 5: Audio Alignment](#stage-5-audio-alignment)
  - [Stage 6: Stitch & Validate](#stage-6-stitch--validate)
- [Utilities](#utilities)
  - [Quality Checker](#quality-checker)
  - [Rerun Weak Shots](#rerun-weak-shots)
  - [Viewer](#viewer)
  - [Naive Baseline](#naive-baseline)
- [Design Decisions](#design-decisions)
- [Tradeoffs](#tradeoffs)
- [Known Limitations](#known-limitations)
- [File Reference](#file-reference)

---

## Setup

**Requirements:** Python 3.10+, ffmpeg

```bash
# 1. Create and activate virtual environment
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

# 2. Install ffmpeg (required for all video processing)
brew install ffmpeg       # macOS
sudo apt install ffmpeg   # Ubuntu/Debian

# 3. Install Python dependencies
pip install -r requirements_new.txt

# 4. Set your Gemini API key
echo "GEMINI_API_KEY=your_key_here" > .env
```

---

## Quick Start

```bash
# Full pipeline (recommended)
python analyze.py videos/myvideo.mp4

# Skip slow steps for a quick draft
python analyze.py videos/myvideo.mp4 --skip-refinement --skip-whisper

# Automatically re-analyze shots with thin/unknown descriptions after the pipeline
python analyze.py videos/myvideo.mp4 --rerun-weak

# Raise the word threshold for more aggressive rerun (default: 10)
python analyze.py videos/myvideo.mp4 --rerun-weak --rerun-min-words 20

# Improve weak shots manually after a run
python rerun_weak_shots.py analysis/myvideo/myvideo_analysis.json videos/myvideo.mp4

# View results in browser
open viewer.html   # then drag in the video + JSON
```

---

## Output Structure

Every run saves results into a per-video subfolder:

```
analysis/
└── myvideo/
    ├── myvideo_analysis.json   # Full shot-by-shot analysis
    └── myvideo_quality.txt     # Auto-generated quality report
```

### JSON Schema

```json
{
  "shots": [
    {
      "id": 1,
      "startTime": "00:00.000",
      "endTime": "00:02.500",
      "cut": "hard cut",
      "shotType": "wide shot",
      "angle": "eye-level",
      "lens": "normal",
      "focus": "deep focus — everything sharp from subject to background",
      "movement": "slow push in",
      "composition": "A man in a blue shirt stands with his back to camera...",
      "graphicsOverlays": "None",
      "audio": "VO/Dialogue: \"Nobody has left their desk all day.\" | [upbeat hip-hop]"
    }
  ],
  "pipeline_metadata": {
    "video_type": "commercial",
    "pace": "fast",
    "duration": 47.4
  },
  "usage": {
    "pipeline_time_seconds": 142.3,
    "whisper_model": "base",
    "gemini_model": "gemini-2.5-pro"
  }
}
```

---

## Pipeline Architecture

```
Input Video
    │
    ▼
┌─────────────────────────────────┐
│ Stage 0: Video Profiler         │  ffprobe metadata + quick scene scan
│ Classify video, set parameters  │  → video_type, pace, adaptive params
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Stage 1: Cut Detection          │  PySceneDetect (Content + Adaptive)
│ Find all edit points            │  + ffmpeg scene filter, merged
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Stage 2: Timestamp Refinement   │  High-FPS frame extraction
│ Sub-frame precision boundaries  │  + pixel comparison at each cut
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Stage 3: Audio Transcription    │  Whisper with word-level timestamps
│ Word-level speech-to-text       │  on 16kHz mono extracted audio
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Stage 4: Visual Analysis        │  Gemini on individual shots/batches
│ Per-shot description via Gemini │  Keyframe mode or video clip mode
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Stage 5: Audio Alignment        │  Map Whisper words into shot windows
│ VO/dialogue → correct shots     │  using word-level timestamps
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Stage 6: Stitch & Validate      │  Fix continuity, detect gaps/overlaps,
│ Final JSON output               │  validate all required fields
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Quality Check (auto-run)        │  Objective metrics against 3 criteria:
│ analysis/<name>_quality.txt     │  boundaries, motion, audio accuracy
└─────────────────────────────────┘
```

---

### Stage 0: Video Profiler

**File:** `pipeline/profiler.py`

Analyzes the video before any processing begins to determine optimal parameters for every downstream stage.

**What it does:**

1. Runs `ffprobe` to extract: duration, native FPS, resolution, codec, file size, whether audio exists.
2. Samples the first 10 seconds with ffmpeg's scene detection filter to estimate editing pace:
   - `slow` — < 0.5 cuts/sec
   - `medium` — 0.5–2.0 cuts/sec
   - `fast` — > 2.0 cuts/sec
3. Classifies the video type using heuristics:
   - Short + medium/fast pace → `commercial`
   - Short + slow pace → `cinematic_short`
   - Any duration + fast pace → `music_video`
   - Otherwise → `general`
4. Generates adaptive parameters based on the classification:

| Parameter | Fast pace | Medium pace | Slow pace |
|---|---|---|---|
| Scene threshold | 20.0 | 27.0 | 30.0 |
| Adaptive threshold | 2.5 | 3.0 | 3.5 |
| Gemini FPS | 8 | 4 | 2 |
| Batch size | 3–5 shots | 5–8 shots | 8 shots |

---

### Stage 1: Cut Detection

**File:** `pipeline/cut_detection.py`

Finds every shot boundary in the video using two independent algorithms, then merges their results.

**Why two algorithms?** Different detector types catch different cuts. PySceneDetect is more sensitive to rapid cuts; ffmpeg's filter has different false-positive characteristics. The union of both is more complete than either alone.

**Algorithm:**

1. **PySceneDetect** runs two detectors simultaneously:
   - `ContentDetector` — compares HSV histograms between frames
   - `AdaptiveDetector` — uses rolling average to handle lighting changes
2. **ffmpeg scene filter** (`select='gt(scene,threshold)'`) provides a second opinion
3. **Merge** — cut points within 150ms of each other are deduplicated (PySceneDetect's timestamp preferred). Unique ffmpeg cuts are added.
4. **Convert to shots** — cut timestamps become `{start_seconds, end_seconds, duration}` dicts, skipping any gap under 10ms.

---

### Stage 2: Timestamp Refinement

**File:** `pipeline/refine_timestamps.py`

Scene detectors report cuts at keyframe boundaries, which can be up to several frames off. This stage refines each cut to the exact frame where the visual change occurs.

**Algorithm (per cut):**

1. Extract frames at 30fps in a ±250ms window around the rough cut time.
2. Compare consecutive frame pairs using mean absolute pixel difference (normalized 0–1).
3. The frame pair with the highest difference is the actual cut location.
4. The refined cut time is the midpoint between those two frames.
5. Shot boundaries are updated so each shot's end = next shot's start (no gaps).

**Result:** Timestamps accurate to ±1 frame (~33ms at 30fps).

---

### Stage 3: Audio Transcription

**File:** `pipeline/audio.py`

Extracts the audio track and transcribes it with word-level timestamps using OpenAI Whisper.

**Why Whisper instead of Gemini for audio?** Whisper returns a timestamp for every individual word. This allows us to map dialogue to the exact shot it occurs in (Stage 5). Asking Gemini to transcribe a full video in one pass produces text without reliable word timing, causing voiceover to be attached to the wrong shots.

**Process:**

1. **Extract audio** — ffmpeg extracts to 16kHz mono PCM WAV (optimal format for Whisper).
2. **Transcribe** — Whisper runs with `word_timestamps=True`, producing `{word, start, end}` for every spoken word.
3. An `initial_prompt` of domain-relevant terms (brand names, technical jargon) is built from the video filename to prevent Whisper from mishearing specific words.

**Available Whisper model sizes:**

| Model | Speed | Accuracy |
|---|---|---|
| `tiny` | Fastest | Lowest |
| `base` | Fast | Good (default) |
| `small` | Moderate | Better |
| `medium` | Slow | High |
| `large` | Slowest | Highest |

---

### Stage 4: Visual Analysis

**File:** `pipeline/visual_analysis.py`

Uses Gemini to describe the visual content of each shot. Rather than sending the entire video at once (which degrades quality on longer videos), shots are grouped into small batches and each batch gets a focused Gemini call.

**Two analysis modes:**

- **Keyframe mode** — Extracts 3 JPEG frames (start, middle, end) per shot and sends them as images. Used for short shots (<1s) where motion analysis is less critical. Lower cost.
- **Clip mode** — Extracts an MP4 clip spanning the batch, uploads to the Gemini Files API, and sends the video. Better for capturing camera movement and transitions. Used for all multi-shot batches.

**Retry logic:** If a batch call fails (API error, malformed JSON), each shot in the batch is retried individually with keyframe mode. If that also fails after 2 attempts, a placeholder is inserted.

**What Gemini is asked to produce per shot:**

| Field | Description |
|---|---|
| `cut` | Transition type into this shot (hard cut, dissolve, match cut, J-cut, L-cut, etc.) |
| `shotType` | Camera framing (extreme wide, wide, medium, close-up, extreme close-up, insert, etc.) |
| `angle` | Camera angle (eye-level, low, high, dutch, bird's eye, worm's eye, POV, etc.) |
| `lens` | Apparent focal length (ultra-wide, wide, normal, telephoto, macro) |
| `focus` | Focus description and depth of field |
| `movement` | Exact camera movement — static only if truly locked off |
| `composition` | 3–5 sentences: framing, subjects, props, lighting, color, mood |
| `graphicsOverlays` | Verbatim text, logos, lower thirds, motion graphics, VFX |

**Context hints** are prepended to each prompt based on the video type from Stage 0. For example, commercial videos get: *"Pay special attention to brand logos, product shots, text overlays, and call-to-action graphics."*

---

### Stage 5: Audio Alignment

**File:** `pipeline/stitcher.py` — `align_audio_to_shots()`

Merges the three data sources into a unified shot list.

For each shot, the `audio` field is built from:
1. **Whisper transcript** — any word whose duration overlaps ≥50% with the shot window is included as dialogue/VO
2. **Gemini's visual analysis** — music, SFX, and ambient sound observed visually (these are intentionally kept separate from dialogue)

Combined format: `VO/Dialogue: "exact words" | [music/SFX description]`

---

### Stage 6: Stitch & Validate

**File:** `pipeline/stitcher.py` — `build_final_output()`

**Continuity fix:**
- Forces first shot to start at `00:00.000`
- Forces last shot to end at video duration
- Sets each shot's end = next shot's start to close sub-millisecond gaps

**Validation checks (printed as warnings):**
- First shot doesn't start at 0
- Last shot doesn't cover the full video
- Any gap or overlap > 50ms between consecutive shots
- Shots shorter than 40ms (likely false cuts)
- Shots longer than 30s (likely missed cuts)
- Required fields that are `"unknown"` or missing

**Final output** is packaged with the shot array, pipeline metadata (video type, pace, duration), and usage stats (processing time, models used).

---

## Utilities

### Quality Checker

**File:** `check_quality.py`

Runs automatically after every pipeline or naive run, saving a report alongside the JSON. Can also be run standalone:

```bash
python check_quality.py analysis/myvideo/myvideo_analysis.json --video videos/myvideo.mp4
```

Evaluates three criteria:

**Criterion 1 — Shot boundaries line up with real cuts**
- Total shots, coverage percentage, gaps and overlaps
- Flags likely false cuts (<80ms) and missed cuts (>20s)

**Criterion 2 — Motion / angle / shot type correct**
- Movement type distribution with bar chart
- Warns if >60% of shots are labeled "static" (common over-reporting issue)
- Composition word counts — flags thin descriptions (<10 words)
- Shot type and angle distributions for spot-checking

**Criterion 3 — Audio accuracy**
- Coverage percentage (shots with audio vs. total)
- Flags shots where word density is too high for their duration (misaligned audio)
- Lists all audio per shot for manual review

---

### Rerun Weak Shots

**File:** `rerun_weak_shots.py`

After reviewing the quality report, use this to selectively re-analyze shots that need improvement without rerunning the full pipeline.

```bash
# Rerun shots with thin compositions or unknown fields (default: <10 words)
python rerun_weak_shots.py analysis/myvideo/myvideo_analysis.json videos/myvideo.mp4

# Raise the threshold to catch more shots
python rerun_weak_shots.py analysis/myvideo/myvideo_analysis.json videos/myvideo.mp4 --min-words 20

# Use a specific model
python rerun_weak_shots.py analysis/myvideo/myvideo_analysis.json videos/myvideo.mp4 --model gemini-2.5-pro
```

**What counts as "weak":**
- Composition under `--min-words` words (default: 10)
- Any of `cut`, `shotType`, `angle`, `lens`, `focus`, `movement` is `"unknown"` or empty

Updates the JSON in-place and regenerates the quality report.

---

### Viewer

Open `viewer.html` in a browser, then drag in both the video file and the analysis JSON. Shows the shot list synced to video playback.

---

### Naive Baseline

**File:** `naive.py`

A single-pass approach that uploads the entire video to Gemini and asks for all shots at once. Intentionally kept simple for comparison purposes.

```bash
python naive.py videos/myvideo.mp4 --fps 4
```

Known limitations compared to the pipeline:
- Misses fast cuts (multiple cuts within a second)
- Inaccurate timestamps, especially on longer videos
- Dialogue often attached to wrong shots
- Quality degrades significantly past ~2 minutes

---

## Design Decisions

**Cut detection is deterministic, not LLM-based.**
PySceneDetect runs at native framerate and reliably catches every hard cut, including sub-second cuts in fast-paced content. Using two detectors and merging results catches what either would miss alone. This is the single biggest quality improvement over the naive approach.

**Audio is handled by a specialist.**
Whisper produces word-level timestamps that are far more accurate than asking Gemini to transcribe a full video. The key benefit is temporal precision — by knowing exactly when each word is spoken, we can assign dialogue to the correct shot window during alignment.

**Visual analysis is chunked.**
Each shot (or small batch) gets its own Gemini call with a focused prompt. This prevents quality dilution that occurs when asking the model to describe 50+ shots in a single prompt. Short shots use the cheaper keyframe mode; longer shots and batches use video clip mode to capture motion.

**Parameters adapt to the video.**
A quick profiling pass estimates edit pace and classifies the video type before any heavy processing. A fast-paced music video needs lower scene detection thresholds, higher Gemini sampling FPS, and smaller batch sizes than a slow cinematic piece. Tuning these per-video instead of using one-size-fits-all values meaningfully improves both accuracy and cost.

**Quality is verified automatically.**
Every run produces a quality report. This makes it easy to spot problems (gaps, thin descriptions, missing audio) immediately after a run without manually reviewing the JSON.

---

## Tradeoffs

| Decision | Pro | Con |
|---|---|---|
| PySceneDetect for cuts | Deterministic, catches fast cuts | Can false-trigger on flashes; weaker on dissolves |
| Whisper for audio | Accurate word-level timestamps | Separate dependency; adds processing time; torch/numpy version sensitivity |
| Chunked Gemini calls | Better per-shot detail | More API calls = higher cost and latency |
| Adaptive profiling | Auto-tunes for video type | Heuristic classification can be wrong |
| Timestamp refinement | Sub-frame precision | Slow for videos with many cuts (~1–2s per cut) |
| Keyframe vs. clip mode | Keyframe is faster/cheaper | Clip mode needed for motion and transition detection |

---

## Known Limitations

- **Visual-audio alignment from disjoint stages** - The biggest tradeoff of a multi-stage pipeline is that visual analysis (Gemini) and audio transcription (Whisper) run independently. Audio aligns to shots reliably by timestamp overlap, but visual descriptions are matched by array position within each batch. If Gemini merges, skips, or reorders shots in its response, visual descriptions can end up attached to the wrong shots — while the audio for those shots remains correct. We mitigate this with shot_index reordering, count validation, and per-shot retry fallbacks, but it's an inherent tension between the accuracy gains of specialized stages and the coherence of a single-pass approach.
- **Dissolves and wipes** — PySceneDetect is weaker on gradual transitions. The adaptive detector helps but misses some.
- **Music lyrics** — Whisper handles speech well but music lyrics with heavy beats can be noisy or hallucinated.
- **Graphics-heavy content** — Rapid motion graphics (countdowns, VFX flashes) can trigger false cuts in scene detection.
- **torch/numpy version conflict** — `openai-whisper` uses PyTorch, which has version constraints with newer NumPy. If Whisper fails, use `--skip-whisper` and note that Gemini will handle audio instead (without word-level timing).
- **API cost** — Chunked analysis uses more Gemini API calls than a single-pass approach. Use `--skip-refinement` and `--skip-whisper` to reduce cost on budget runs.

---

## File Reference

```
analyze.py                  # Main pipeline entry point
analyze_new.py              # Updated pipeline with improved audio handling
naive.py                    # Single-pass baseline for comparison
check_quality.py            # Standalone quality checker
rerun_weak_shots.py         # Re-analyze shots with thin/unknown fields
viewer.html                 # Browser-based results viewer
requirements_new.txt        # Python dependencies
.env                        # GEMINI_API_KEY (not committed)

pipeline/
  __init__.py
  profiler.py               # Stage 0: Video metadata + adaptive params
  cut_detection.py          # Stage 1: PySceneDetect + ffmpeg cut detection
  refine_timestamps.py      # Stage 2: Sub-frame timestamp refinement
  audio.py                  # Stage 3: ffmpeg audio extraction + Whisper
  visual_analysis.py        # Stage 4: Gemini chunked visual analysis
  stitcher.py               # Stage 5+6: Audio alignment + final output

analysis/
  <video_name>/
    <video_name>_analysis.json   # Shot analysis output
    <video_name>_quality.txt     # Auto-generated quality report
```
