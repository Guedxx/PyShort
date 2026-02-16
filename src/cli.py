import argparse
import os
import sys

from src.config import MODEL_DEFAULTS, load_config
from src.parsing import parse_ai_response
from src.providers import find_clips
from src.transcription import transcribe_video
from src.utils import get_video_duration, make_safe_filename, parse_time_str, read_srt
from src.video import clip_video


def _find_existing_srt(video_path: str) -> str | None:
    """Return path to existing SRT next to the video, or None."""
    srt_path = os.path.splitext(video_path)[0] + ".srt"
    if os.path.isfile(srt_path):
        return srt_path
    return None


def main():
    parser = argparse.ArgumentParser(
        description="YouTube Shorts Clipper - AI-powered video clipping"
    )

    ai_group = parser.add_mutually_exclusive_group()
    ai_group.add_argument("-o", "--openai", action="store_true", help="Use OpenAI")
    ai_group.add_argument("-g", "--gemini", action="store_true", help="Use Google Gemini")
    ai_group.add_argument("-l", "--ollama", action="store_true", help="Use Ollama (local)")
    ai_group.add_argument("-m", "--manual", nargs="+", metavar=("START", "END"),
                          help="Manual mode: START END [TITLE] — skip AI, clip directly")

    parser.add_argument("video", help="Path to video file")
    parser.add_argument("srt", nargs="?", default=None, help="Path to SRT subtitle file (optional for manual mode)")
    parser.add_argument("-d", "--output-dir", default=None, help="Output directory")
    parser.add_argument("--model", default=None, help="Override AI model name")
    parser.add_argument("--config", default=None, help="Path to config TOML file")
    parser.add_argument("--remove-silence", action="store_true", help="Remove silent moments from clips")
    parser.add_argument("--transcribe", action="store_true", help="Auto-transcribe video (requires openai-whisper)")

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Resolve output dir: CLI flag > config > default
    output_dir = args.output_dir or cfg.output_dir
    
    # Resolve remove_silence: CLI flag > config > default
    remove_silence = args.remove_silence or cfg.remove_silence

    # Validate inputs
    if not os.path.isfile(args.video):
        print(f"Video not found: {args.video}")
        sys.exit(1)
        
    if args.srt and not os.path.isfile(args.srt):
        print(f"SRT not found: {args.srt}")
        sys.exit(1)

    # Auto-transcription logic
    if args.transcribe and not args.srt:
        existing = _find_existing_srt(args.video)
        if existing:
            print(f"Found existing SRT: {existing}")
            args.srt = existing
        else:
            try:
                print("Auto-transcription enabled. Generating SRT from video audio...")
                srt_path = transcribe_video(args.video)
                args.srt = srt_path
            except Exception as e:
                print(f"Transcription failed: {e}")
                sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    if args.manual:
        # Manual mode: skip AI, clip directly
        if len(args.manual) < 2:
            parser.error("-m/--manual requires at least START and END times")
        start_time = args.manual[0]
        end_time = args.manual[1]
        title = args.manual[2] if len(args.manual) > 2 else "clip"
        clips = [{"start_time": start_time, "end_time": end_time, "title": title}]
        print(f"Manual mode: [{start_time} -> {end_time}] {title}")
    else:
        # AI Mode requires SRT (either provided or generated)
        if not args.srt:
            existing = _find_existing_srt(args.video)
            if existing:
                print(f"Found existing SRT: {existing}")
                args.srt = existing
            else:
                print("No SRT provided. Auto-transcribing video...")
                try:
                    args.srt = transcribe_video(args.video)
                except Exception as e:
                    print(f"Transcription failed: {e}")
                    sys.exit(1)
            
        # Resolve provider: CLI flag > config > error
        if args.openai:
            provider = "openai"
        elif args.gemini:
            provider = "gemini"
        elif args.ollama:
            provider = "ollama"
        elif cfg.provider:
            provider = cfg.provider
        else:
            print("No AI provider specified. Use -o/--openai, -g/--gemini, -l/--ollama, or set provider in config.toml")
            sys.exit(1)

        # Resolve model: --model flag > config > default for provider
        model = args.model or cfg.model or MODEL_DEFAULTS.get(provider)
        if not model:
            print(f"No model configured for provider '{provider}'")
            sys.exit(1)

        # Step 1: Read SRT
        print(f"Reading SRT: {args.srt}")
        transcript = read_srt(args.srt)
        print(f"  {len(transcript)} characters read")

        # Step 2: AI analysis
        print(f"Analyzing transcript with {provider} ({model})...")
        raw_response = find_clips(provider, transcript, model)

        clips = parse_ai_response(raw_response)
        print(f"  Found {len(clips)} clips:")
        for i, clip in enumerate(clips):
            print(f"    {i+1}. [{clip['start_time']} -> {clip['end_time']}] {clip['title']}")

    # Validate clip timestamps against video duration
    duration = get_video_duration(args.video)
    if duration > 0:
        valid_clips = []
        for clip in clips:
            end_sec = parse_time_str(clip["end_time"])
            if end_sec > duration:
                print(f"  Skipping '{clip['title']}': end time {clip['end_time']} exceeds video duration ({duration:.0f}s)")
            else:
                valid_clips.append(clip)
        clips = valid_clips

    # Step 3: Clip and process each segment
    results = []
    for i, clip in enumerate(clips):
        safe_name = make_safe_filename(clip.get("title", f"clip_{i}"))
        out_path = os.path.join(output_dir, f"{safe_name}.mp4")

        print(f"\nClipping {i+1}/{len(clips)}: {clip['title']}")
        print(f"  {clip['start_time']} -> {clip['end_time']}")

        success = clip_video(args.video, args.srt, clip, out_path, remove_silence=remove_silence)
        results.append({
            "title": clip["title"],
            "file": out_path,
            "status": "success" if success else "failed",
        })

    # Summary
    print("\n" + "=" * 50)
    print("RESULTS:")
    print("=" * 50)
    for r in results:
        icon = "+" if r["status"] == "success" else "x"
        size = ""
        if r["status"] == "success" and os.path.isfile(r["file"]):
            mb = os.path.getsize(r["file"]) / (1024 * 1024)
            size = f" ({mb:.1f} MB)"
        print(f"  {icon} {r['title']}{size}")
        print(f"    -> {r['file']}")

    succeeded = sum(1 for r in results if r["status"] == "success")
    print(f"\n{succeeded}/{len(results)} clips created in {output_dir}/")
