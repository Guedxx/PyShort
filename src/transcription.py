import os
import datetime
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
    result = model.transcribe(video_path, fp16=(device == "cuda"))
    
    # Detect language (it's in the result["language"])
    print(f"Detected language '{result['language']}'")

    # Generate SRT content
    srt_content = ""
    for i, segment in enumerate(result["segments"]):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"].strip()
        
        srt_content += f"{i + 1}\n"
        srt_content += f"{start} --> {end}\n"
        srt_content += f"{text}\n\n"

    # Save to SRT file
    base_name = os.path.splitext(video_path)[0]
    srt_path = f"{base_name}.srt"
    
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
        
    print(f"Transcription saved to: {srt_path}")
    return srt_path

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
