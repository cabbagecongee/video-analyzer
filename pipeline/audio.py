"""
Stage 3: Audio Extraction & Transcription

Extracts audio track with ffmpeg, then runs OpenAI Whisper for
word-level transcription with timestamps.
"""

import subprocess
import os
import json
from typing import Optional


def extract_audio(video_path: str, output_path: str = None) -> str:
    """
    Extract audio track from video as WAV (16kHz mono, ideal for Whisper).
    """
    if output_path is None:
        base = os.path.splitext(video_path)[0]
        output_path = f"{base}_audio.wav"

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                    # No video
        "-acodec", "pcm_s16le",   # PCM 16-bit
        "-ar", "16000",           # 16kHz sample rate
        "-ac", "1",               # Mono
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {result.stderr}")

    print(f"  Extracted audio to {output_path}")
    return output_path


def transcribe_whisper(
    audio_path: str,
    model_size: str = "base",
    language: str = "en",
    initial_prompt: Optional[str] = None,
) -> dict:
    """
    Run Whisper on the audio file. Returns segments with word-level timestamps.
    
    Uses the whisper Python package (openai-whisper).
    initial_prompt gives Whisper context to improve accuracy on domain-specific
    words (brand names, technical terms, etc.)
    """
    import whisper

    print(f"  Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)

    # Build transcription kwargs
    transcribe_kwargs = {
        "language": language,
        "word_timestamps": True,
        "verbose": False,
        "condition_on_previous_text": True,
    }

    # initial_prompt helps Whisper with brand names and domain-specific words
    # that it might otherwise mishear (e.g., "bitcoin" -> "hey coin")
    if initial_prompt:
        transcribe_kwargs["initial_prompt"] = initial_prompt

    print(f"  Transcribing audio...")
    result = model.transcribe(audio_path, **transcribe_kwargs)

    # Extract word-level data
    words = []
    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            words.append({
                "word": word_info["word"].strip(),
                "start": round(word_info["start"], 3),
                "end": round(word_info["end"], 3),
            })

    # Also keep segment-level data as fallback
    segments = []
    for segment in result.get("segments", []):
        segments.append({
            "text": segment["text"].strip(),
            "start": round(segment["start"], 3),
            "end": round(segment["end"], 3),
        })

    return {
        "full_text": result.get("text", "").strip(),
        "words": words,
        "segments": segments,
        "language": result.get("language", language),
    }


def get_words_for_shot(
    transcript: dict,
    start_time: float,
    end_time: float,
    overlap_threshold: float = 0.5,
) -> str:
    """
    Get the transcript text that falls within a shot's time window.
    A word is included if at least overlap_threshold of its duration
    falls within the shot boundary.
    """
    shot_words = []
    for word in transcript.get("words", []):
        word_start = word["start"]
        word_end = word["end"]
        word_duration = word_end - word_start

        if word_duration <= 0:
            # Point-like word, check if it's within the shot
            if start_time <= word_start <= end_time:
                shot_words.append(word["word"])
            continue

        # Calculate overlap
        overlap_start = max(word_start, start_time)
        overlap_end = min(word_end, end_time)
        overlap = max(0, overlap_end - overlap_start)

        if overlap / word_duration >= overlap_threshold:
            shot_words.append(word["word"])

    return " ".join(shot_words) if shot_words else ""


def extract_and_transcribe(
    video_path: str,
    model_size: str = "base",
    language: str = "en",
    initial_prompt: Optional[str] = None,
) -> dict:
    """
    Full audio pipeline: extract audio, transcribe with Whisper.
    initial_prompt can include brand names and terms to improve accuracy.
    """
    print("  Extracting audio track...")
    audio_path = extract_audio(video_path)

    print("  Running Whisper transcription...")
    transcript = transcribe_whisper(
        audio_path, model_size, language,
        initial_prompt=initial_prompt,
    )

    print(f"  Transcribed {len(transcript['words'])} words, "
          f"{len(transcript['segments'])} segments")

    # Clean up audio file
    try:
        os.remove(audio_path)
    except OSError:
        pass

    return transcript


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.audio <video_path> [model_size]")
        sys.exit(1)

    model = sys.argv[2] if len(sys.argv) > 2 else "base"
    transcript = extract_and_transcribe(sys.argv[1], model_size=model)
    print(f"\nFull text: {transcript['full_text'][:200]}...")
    print(f"\nFirst 10 words with timestamps:")
    for w in transcript["words"][:10]:
        print(f"  [{w['start']:.3f} - {w['end']:.3f}] {w['word']}")