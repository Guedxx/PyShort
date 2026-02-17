# Short-Maker

**Short-Maker** is an AI-powered tool that automatically identifies and clips the most engaging segments from long-form videos to create YouTube Shorts. It leverages LLMs (OpenAI or Google Gemini) to analyze video transcripts and FFmpeg for high-quality video processing.

## Features

- **AI-Powered Clipping**: Uses LLMs (OpenAI GPT or Google Gemini) to analyze subtitles and find viral-worthy hooks and stories.
- **Auto-Transcription**: Automatically generates SRT subtitles using OpenAI Whisper if no subtitle file is provided.
- **Smart Framing**: Detects faces in the video to automatically crop landscape (16:9) video into vertical (9:16) format, keeping the speaker in focus.
- **Silence Removal**: Optionally removes silent moments from the clips for a faster pace.
- **Hardware Acceleration**: Supports VAAPI for fast hardware-accelerated encoding on Linux (with CPU fallback).
- **Burn-in Subtitles**: Adds stylized subtitles and titles to the output video.
- **Manual Mode**: Allows manually specifying start and end times for clips.

## Prerequisites

- **Python 3.11+**
- **FFmpeg**: Required for video processing.
  - On Linux (Ubuntu/Debian): `sudo apt install ffmpeg`
- **API Keys**: You need an API key for either OpenAI (`OPENAI_API_KEY`) or Gemini (`GEMINI_API_KEY`).

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/short-maker.git
   cd short-maker
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -e .[all]
   ```
   Or install specific provider dependencies:
   ```bash
   pip install -e .[openai]  # For OpenAI
   pip install -e .[gemini]  # For Google Gemini
   ```

4. **Configuration:**
   
   Create a `.env` file in the project root with your API keys:
   ```env
   OPENAI_API_KEY=your_openai_key
   GEMINI_API_KEY=your_gemini_key
   ```

   Optional: create `.env.local` for machine-specific overrides. Runtime loads env files in this order:
   1. `.env`
   2. `.env.local`

   Existing shell environment variables are not overridden by these files.
   
   (Optional) Copy `config.example.toml` to `config.toml` to set default preferences:
   ```bash
   cp config.example.toml config.toml
   ```

## Usage

Basic usage requires a video file. If no SRT file is provided, it can auto-transcribe (requires Whisper).

### 1. Auto-Transcribe & Clip (AI Mode)
Let the AI transcribe the video, analyze it, and create clips.
```bash
short-maker video.mp4 --transcribe --gemini
```

### 2. Clip with Existing Subtitles
If you already have an SRT file:
```bash
short-maker video.mp4 subtitles.srt --gemini
```

### 3. Manual Clipping
Clip a specific segment without AI analysis.
Format: `START END [TITLE]`
```bash
short-maker video.mp4 --manual 00:05:30 00:06:15 "My Custom Clip"
```

### 4. Advanced Options
- **Remove Silence**: `--remove-silence`
- **Specify Output Directory**: `-d /path/to/output`
- **Select Model**: `--model gpt-4-turbo`

```bash
short-maker video.mp4 --transcribe --openai --model gpt-4-turbo --remove-silence -d ./my_shorts
```

## Configuration (config.toml)

You can define defaults in `config.toml`:

```toml
[ai]
provider = "gemini"              # "openai", "gemini", or "ollama"
model = "gemini-3-flash-preview" # Model name for the selected provider

[output]
dir = "./shorts_clips"
remove_silence = false
```

## How It Works

1.  **Transcription**: If no SRT is provided, OpenAI Whisper generates one from the video audio.
2.  **Analysis**: The LLM reads the transcript and identifies 3-5 engaging segments (15-60s) with strong hooks.
3.  **Face Detection**: The tool analyzes frames to find the speaker's face ensuring they stay centered in the vertical crop.
4.  **Processing**: FFmpeg crops the video, burns in subtitles/titles, removes silence (if requested), and speeds up the video slightly (1.2x) for better retention.
