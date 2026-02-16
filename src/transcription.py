import os
import datetime
import ast

# Monkey patch ast.Num for compatibility with newer Python versions (3.12+)
# where ast.Num was removed, but libraries like Triton/Whisper might still use it.
if not hasattr(ast, "Num"):
    ast.Num = ast.Constant

import torch
import whisper

def transcribe_video(video_path: str, model_size: str = "medium", device: str = "cuda") -> str:
    """
    Transcribes the audio from the given video file using openai-whisper.
    Generates an SRT file with the same name as the video (e.g. video.mp4 -> video.srt).
    Returns the path to the generated SRT file.
    """
    
    # Check device availability
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA (GPU) not available. Falling back to CPU for transcription.")
        device = "cpu"
    
    print(f"Loading Whisper model '{model_size}' on {device}...")
    model = whisper.load_model(model_size, device=device)
    
    print(f"Transcribing {video_path}...")
    # fp16=False allows running on CPU or older GPUs without errors
    # word_timestamps=True is required for word-level segmentation
    result = model.transcribe(video_path, fp16=(device == "cuda"), word_timestamps=True)

    # Free Whisper model from VRAM
    del model
    torch.cuda.empty_cache()

    # Detect language (it's in the result["language"])
    print(f"Detected language '{result['language']}'")

    # Generate SRT content with max 4 words per line
    srt_content = ""
    subtitle_index = 1
    
    current_chunk = []
    
    for segment in result["segments"]:
        words = segment.get("words", [])
        
        # Fallback if no words found (some models/languages might not support it perfectly)
        if not words:
            # Treat the whole segment as one chunk if short enough? 
            # Or just split by text. Let's just use the segment logic if words are missing.
            # But with word_timestamps=True, words should be there.
            text_words = segment["text"].strip().split()
            if len(text_words) <= 4:
                 srt_content += create_srt_block(subtitle_index, segment["start"], segment["end"], segment["text"].strip())
                 subtitle_index += 1
                 continue
            # If long segment and no words, we can't easily split time. Fallback to standard segment.
            srt_content += create_srt_block(subtitle_index, segment["start"], segment["end"], segment["text"].strip())
            subtitle_index += 1
            continue

        for word_info in words:
            current_chunk.append(word_info)
            
            if len(current_chunk) >= 4:
                # Flush chunk
                start = current_chunk[0]["start"]
                end = current_chunk[-1]["end"]
                text = "".join([w["word"] for w in current_chunk]).strip()
                
                srt_content += create_srt_block(subtitle_index, start, end, text)
                subtitle_index += 1
                current_chunk = []
        
        # If segment ends, we might want to flush the chunk or keep it for next segment?
        # A sentence might span across Whisper segments? 
        # Ideally yes, but Whisper segments usually break on pauses/sentences.
        # Flushing at segment end is safer for accurate timing.
        if current_chunk:
             start = current_chunk[0]["start"]
             end = current_chunk[-1]["end"]
             text = "".join([w["word"] for w in current_chunk]).strip()
             
             srt_content += create_srt_block(subtitle_index, start, end, text)
             subtitle_index += 1
             current_chunk = []

    # Save to SRT file
    base_name = os.path.splitext(video_path)[0]
    srt_path = f"{base_name}.srt"
    
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
        
    print(f"Transcription saved to: {srt_path}")
    return srt_path

def create_srt_block(index: int, start: float, end: float, text: str) -> str:
    start_fmt = format_timestamp(start)
    end_fmt = format_timestamp(end)
    return f"{index}\n{start_fmt} --> {end_fmt}\n{text}\n\n"

def format_timestamp(seconds: float) -> str:
    """Formats seconds into SRT timestamp format: HH:MM:SS,mmm"""
    td = datetime.timedelta(seconds=seconds)
    # timedelta string is usually H:MM:SS.micros, roughly.
    # We need strictly HH:MM:SS,mmm
    
    # Calculate hours, minutes, seconds, milliseconds
    total_seconds = int(seconds)
    milliseconds = int((seconds - total_seconds) * 1000)
    
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"
