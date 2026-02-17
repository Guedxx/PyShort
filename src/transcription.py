import ast
import os

# Monkey patch ast.Num for compatibility with newer Python versions (3.12+)
# where ast.Num was removed, but libraries like Triton/Whisper might still use it.
if not hasattr(ast, "Num"):
    ast.Num = ast.Constant

import torch
import whisper


def _resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA (GPU) not available. Falling back to CPU for transcription.")
        return "cpu"
    return device


def _chunk_text(words: list[dict]) -> str:
    return "".join(word.get("word", "") for word in words).strip()


def _append_chunk_block(blocks: list[str], index: int, chunk: list[dict]) -> int:
    start = chunk[0]["start"]
    end = chunk[-1]["end"]
    blocks.append(create_srt_block(index, start, end, _chunk_text(chunk)))
    return index + 1


def transcribe_video(video_path: str, model_size: str = "medium", device: str = "cuda") -> str:
    """
    Transcribes the audio from the given video file using openai-whisper.
    Generates an SRT file with the same name as the video (e.g. video.mp4 -> video.srt).
    Returns the path to the generated SRT file.
    """
    device = _resolve_device(device)

    print(f"Loading Whisper model '{model_size}' on {device}...")
    try:
        model = whisper.load_model(model_size, device=device)
    except Exception as exc:
        raise RuntimeError(f"Failed to load Whisper model '{model_size}' on {device}: {exc}") from exc

    try:
        print(f"Transcribing {video_path}...")
        result = model.transcribe(video_path, fp16=(device == "cuda"), word_timestamps=True)
    except Exception as exc:
        raise RuntimeError(f"Whisper transcription failed for '{video_path}': {exc}") from exc
    finally:
        del model
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Detected language '{result['language']}'")

    srt_blocks: list[str] = []
    subtitle_index = 1
    current_chunk: list[dict] = []

    for segment in result["segments"]:
        words = segment.get("words", [])
        if not words:
            srt_blocks.append(
                create_srt_block(subtitle_index, segment["start"], segment["end"], segment["text"].strip())
            )
            subtitle_index += 1
            continue

        for word_info in words:
            current_chunk.append(word_info)
            if len(current_chunk) >= 4:
                subtitle_index = _append_chunk_block(srt_blocks, subtitle_index, current_chunk)
                current_chunk = []

        if current_chunk:
            subtitle_index = _append_chunk_block(srt_blocks, subtitle_index, current_chunk)
            current_chunk = []

    base_name = os.path.splitext(video_path)[0]
    srt_path = f"{base_name}.srt"

    with open(srt_path, "w", encoding="utf-8") as file_handle:
        file_handle.write("".join(srt_blocks))

    print(f"Transcription saved to: {srt_path}")
    return srt_path


def create_srt_block(index: int, start: float, end: float, text: str) -> str:
    start_fmt = format_timestamp(start)
    end_fmt = format_timestamp(end)
    return f"{index}\n{start_fmt} --> {end_fmt}\n{text}\n\n"


def format_timestamp(seconds: float) -> str:
    """Formats seconds into SRT timestamp format: HH:MM:SS,mmm"""
    total_seconds = int(seconds)
    milliseconds = int((seconds - total_seconds) * 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"
