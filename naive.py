"""
This is a baseline implementation that uploads a video to Gemini and asks for
a shot-by-shot analysis in a single pass. It works okay but has issues with:
- Missing fast cuts (multiple cuts within a second)
- Often misses shots or inaccurately describes them, especially on longer vids
- Inaccurate timestamps (especially with Gemini 3)
- Mishearing audio/dialogue
"""

import io
import os
import re
import json
import mimetypes
from contextlib import redirect_stdout
from google import genai
from google.genai import types
from dotenv import load_dotenv
from check_quality import analyze_json as quality_analyze, print_report as quality_print_report

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def upload_video(video_path: str) -> str:
    """Upload video to Gemini Files API and wait for processing"""
    print(f"Uploading {video_path}...")
    
    # Detect mime type (default to mp4 if unknown)
    mime_type, _ = mimetypes.guess_type(video_path)
    if not mime_type or not mime_type.startswith("video/"):
        mime_type = "video/mp4"
    
    with open(video_path, "rb") as f:
        uploaded = client.files.upload(file=f, config={"mime_type": mime_type})
    
    # Wait for file to be ready
    while True:
        info = client.files.get(name=uploaded.name)
        if info.state.name == "ACTIVE":
            print(f"Upload complete: {uploaded.uri}")
            return uploaded.uri
        if info.state.name == "FAILED":
            raise Exception(f"Upload failed: {info.error}")
        import time
        time.sleep(2)


def analyze_video(file_uri: str, use_gemini_3: bool = False, fps: int = 4) -> dict:
    """
    Analyze video with Gemini and return shot-by-shot breakdown.
    
    Args:
        file_uri: Gemini file URI from upload_video()
        use_gemini_3: If True, use Gemini 3 Pro. If False, use Gemini 2.5 Pro.
        fps: Frames per second to sample from video (default 4)
    """
    
    prompt = """Analyze this video shot by shot. For each shot, provide:
- id: Sequential number starting at 1
- startTime: Start timestamp in MM:SS.mmm format
- endTime: End timestamp in MM:SS.mmm format  
- cut: Type of cut/transition INTO this shot (hard cut, dissolve, match cut, etc.)
- shotType: Camera framing (extreme wide, wide, medium, close-up, extreme close-up, etc.)
- angle: Camera angle (eye-level, low angle, high angle, dutch, bird's eye, etc.)
- lens: Apparent focal length (wide, normal, telephoto)
- focus: What's in focus and depth of field
- movement: Camera movement (static, pan, tilt, dolly, tracking, handheld, zoom, etc.)
- composition: 3-5 sentences describing the framing, subjects, props, mood, and action
- graphicsOverlays: Any text, logos, or post-production elements (or "None")
- audio: Exact dialogue in quotes, voiceover, music, sound effects (or "None")

A "shot" is defined as any continuous footage between two edits. Every hard cut, 
dissolve, whip pan, or other transition marks the beginning of a new shot.

Output valid JSON in this exact format:
{
  "shots": [
    {
      "id": 1,
      "startTime": "00:00.000",
      "endTime": "00:02.500",
      "cut": "...",
      "shotType": "...",
      "angle": "...",
      "lens": "...",
      "focus": "...",
      "movement": "...",
      "composition": "...",
      "graphicsOverlays": "...",
      "audio": "..."
    }
  ]
}"""

    parts = [
        types.Part(file_data=types.FileData(file_uri=file_uri), video_metadata=types.VideoMetadata(fps=fps)),
        types.Part(text=prompt)
    ]
    
    if use_gemini_3:
        print("Analyzing with Gemini 3 Pro...")
        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=types.Content(parts=parts),
            config=types.GenerateContentConfig(
                max_output_tokens=65536,
                media_resolution="MEDIA_RESOLUTION_HIGH",
                thinking_config=types.ThinkingConfig(thinking_level="high")
            )
        )
    else:
        print("Analyzing with Gemini 2.5 Pro...")
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=types.Content(parts=parts),
            config=types.GenerateContentConfig(
                max_output_tokens=65536,
                media_resolution="MEDIA_RESOLUTION_HIGH",
                thinking_config=types.ThinkingConfig(thinking_budget=32768)
            )
        )
    
    # Extract JSON from response
    text = response.text
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    json_str = match.group(1).strip() if match else text.strip()
    # Remove JS-style comments (common model mistake)
    json_str = re.sub(r'//[^\n]*', '', json_str)
    json_str = re.sub(r'/\*[\s\S]*?\*/', '', json_str)
    # Remove trailing commas before } or ] (common model mistake)
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        # Print the lines around the error location
        lines = json_str.splitlines()
        err_line = e.lineno - 1
        snippet = "\n".join(lines[max(0, err_line-2):err_line+3])
        print(f"Context around error (line {e.lineno}):\n{snippet}")
        raise
    
    # Add token usage stats
    usage = getattr(response, "usage_metadata", None)
    if usage:
        result["usage"] = {
            "prompt_token_count": usage.prompt_token_count,
            "candidates_token_count": usage.candidates_token_count,
            "thoughts_token_count": getattr(usage, "thoughts_token_count", None),
            "total_token_count": usage.total_token_count,
        }
    return result


def analyze_video_file(video_path: str, use_gemini_3: bool = False, fps: int = 4) -> dict:
    """
    Full pipeline: upload video and analyze it.
    
    Args:
        video_path: Path to video file
        use_gemini_3: If True, use Gemini 3 Pro. If False, use Gemini 2.5 Pro.
        fps: Frames per second to sample from video (default 4)
    
    Returns:
        Dict with "shots" array containing shot-by-shot analysis
    """
    file_uri = upload_video(video_path)
    return analyze_video(file_uri, use_gemini_3, fps)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python naive.py <video_path> [--gemini3] [--fps N]")
        print("\nExample:")
        print("  python naive.py my_video.mp4")
        print("  python naive.py my_video.mp4 --gemini3")
        print("  python naive.py my_video.mp4 --fps 8")
        sys.exit(1)
    
    video_path = sys.argv[1]
    use_gemini_3 = "--gemini3" in sys.argv
    
    # Parse --fps argument
    fps = 4
    if "--fps" in sys.argv:
        fps_idx = sys.argv.index("--fps")
        if fps_idx + 1 < len(sys.argv):
            fps = int(sys.argv[fps_idx + 1])
    
    print(f"Using FPS: {fps}")
    result = analyze_video_file(video_path, use_gemini_3, fps)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("analysis", video_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, video_name + "_analysis.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {len(result['shots'])} shots to {output_path}")

    # Run quality check and save report
    from check_quality import get_video_duration
    video_duration = get_video_duration(video_path)
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

