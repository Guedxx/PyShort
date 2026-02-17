import argparse
import json
import os
import sys

from src.config import MODEL_DEFAULTS, load_config
from src.parsing import parse_ai_response
from src.providers import find_clips
from src.transcription import transcribe_video
from src.utils import get_video_duration, make_safe_filename, parse_time_str, read_srt
from src.video import clip_video

REQUIRED_CLIP_KEYS = ("start_time", "end_time", "title")
CUTS_CACHE_FILENAME = "cuts.json"


def _find_existing_srt(video_path: str) -> str | None:
    """Return path to existing SRT next to the video, or None."""
    srt_path = os.path.splitext(video_path)[0] + ".srt"
    if os.path.isfile(srt_path):
        return srt_path
    return None


def _exit_with_error(message: str, code: int = 1) -> None:
    print(message)
    raise SystemExit(code)


def _parse_timestamp(value: object, label: str) -> float:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty timestamp string")
    try:
        return parse_time_str(value)
    except (AttributeError, TypeError, ValueError):
        raise ValueError(f"{label} '{value}' is not parseable")


def _build_manual_clip(args_manual: list[str]) -> dict[str, str]:
    start_time = args_manual[0]
    end_time = args_manual[1]
    title = " ".join(args_manual[2:]) if len(args_manual) > 2 else "clip"

    start_sec = _parse_timestamp(start_time, "Manual start_time")
    end_sec = _parse_timestamp(end_time, "Manual end_time")
    if start_sec >= end_sec:
        raise ValueError(
            f"Manual start time must be before end time (got {start_time} >= {end_time})"
        )

    return {"start_time": start_time, "end_time": end_time, "title": title}


def _cuts_cache_path(video_path: str) -> str:
    video_dir = os.path.dirname(os.path.abspath(video_path)) or "."
    return os.path.join(video_dir, CUTS_CACHE_FILENAME)


def _load_cached_ai_response(cache_path: str) -> str | None:
    if not os.path.isfile(cache_path):
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: Failed to read cuts cache '{cache_path}': {exc}")
        return None

    if isinstance(payload, dict) and isinstance(payload.get("response"), str):
        return payload["response"]

    if isinstance(payload, (dict, list)):
        # Backward compatibility: allow direct JSON payloads without {"response": "..."}
        return json.dumps(payload)

    print(f"Warning: Invalid cuts cache format in '{cache_path}'.")
    return None


def _save_cached_ai_response(cache_path: str, response: str) -> None:
    try:
        with open(cache_path, "w", encoding="utf-8") as file_handle:
            json.dump({"response": response}, file_handle, ensure_ascii=False, indent=2)
    except OSError as exc:
        print(f"Warning: Failed to write cuts cache '{cache_path}': {exc}")
        return

    print(f"Saved cuts cache: {cache_path}")


def _request_ai_response(provider: str, transcript: str, model: str) -> str:
    print(f"Analyzing transcript with {provider} ({model})...")
    try:
        return find_clips(provider, transcript, model)
    except SystemExit as exc:
        _exit_with_error(
            f"AI provider failed before clip parsing (exit code: {exc.code})."
        )
    except Exception as exc:
        _exit_with_error(f"AI provider failed before clip parsing: {exc}")


def _validate_ai_clips(clips: object) -> list[dict[str, str]]:
    if not isinstance(clips, list):
        _exit_with_error("AI response parsing failed: expected a list of clips.")

    validated: list[dict[str, str]] = []
    for i, clip in enumerate(clips, start=1):
        if not isinstance(clip, dict):
            print(f"  Skipping AI clip {i}: expected object, got {type(clip).__name__}")
            continue

        missing = [k for k in REQUIRED_CLIP_KEYS if k not in clip]
        if missing:
            print(f"  Skipping AI clip {i}: missing keys {', '.join(missing)}")
            continue

        start_time = clip.get("start_time")
        end_time = clip.get("end_time")
        title = str(clip.get("title", "")).strip() or f"clip_{i}"

        try:
            start_sec = _parse_timestamp(start_time, f"AI clip {i} start_time")
            end_sec = _parse_timestamp(end_time, f"AI clip {i} end_time")
        except ValueError as exc:
            print(f"  Skipping AI clip {i}: {exc}")
            continue

        if start_sec >= end_sec:
            print(
                f"  Skipping AI clip {i}: start_time must be before end_time "
                f"(got {start_time} >= {end_time})"
            )
            continue

        validated.append(
            {
                "start_time": str(start_time).strip(),
                "end_time": str(end_time).strip(),
                "title": title,
            }
        )

    return validated


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
        try:
            manual_clip = _build_manual_clip(args.manual)
        except ValueError as exc:
            _exit_with_error(f"Invalid manual clip: {exc}")
        clips = [manual_clip]
        print(
            "Manual mode: "
            f"[{manual_clip['start_time']} -> {manual_clip['end_time']}] {manual_clip['title']}"
        )
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

        # Step 2: AI analysis (cache-aware)
        cache_path = _cuts_cache_path(args.video)
        cached_response = _load_cached_ai_response(cache_path)
        used_cache = cached_response is not None

        if used_cache:
            print(f"Using cached cuts: {cache_path}")
            raw_response = cached_response
        else:
            raw_response = _request_ai_response(provider, transcript, model)

        try:
            parsed = parse_ai_response(raw_response)
        except SystemExit:
            if used_cache:
                print("Cached cuts are invalid. Regenerating with AI provider...")
                raw_response = _request_ai_response(provider, transcript, model)
                used_cache = False
                try:
                    parsed = parse_ai_response(raw_response)
                except SystemExit:
                    _exit_with_error("AI response parsing failed.")
                except Exception as exc:
                    _exit_with_error(f"AI response parsing failed: {exc}")
            else:
                _exit_with_error("AI response parsing failed.")
        except Exception as exc:
            if used_cache:
                print(f"Cached cuts parsing failed: {exc}. Regenerating with AI provider...")
                raw_response = _request_ai_response(provider, transcript, model)
                used_cache = False
                try:
                    parsed = parse_ai_response(raw_response)
                except SystemExit:
                    _exit_with_error("AI response parsing failed.")
                except Exception as parse_exc:
                    _exit_with_error(f"AI response parsing failed: {parse_exc}")
            else:
                _exit_with_error(f"AI response parsing failed: {exc}")

        if not used_cache:
            _save_cached_ai_response(cache_path, raw_response)

        clips = _validate_ai_clips(parsed)
        if not clips:
            _exit_with_error("No valid AI clips found after validation.")

        print(f"  Found {len(clips)} valid clips:")
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

    if not clips:
        _exit_with_error("No valid clips to process after validation and duration checks.")

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
