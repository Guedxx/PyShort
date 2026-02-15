import subprocess


def escape_srt_path(path: str) -> str:
    """Escape special chars for FFmpeg subtitles filter."""
    path = path.replace("\\", "\\\\\\\\")
    path = path.replace(":", "\\\\:")
    path = path.replace("'", "\\\\'")
    path = path.replace("[", "\\\\[")
    path = path.replace("]", "\\\\]")
    return path


def escape_drawtext(text: str) -> str:
    """Escape special chars for FFmpeg drawtext filter."""
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "\u2019")
    text = text.replace(":", "\\:")
    text = text.replace(";", "\\;")
    return text


def clip_video(
    video_path: str,
    srt_path: str,
    clip: dict,
    output_path: str,
) -> bool:
    srt_escaped = escape_srt_path(srt_path)
    title_escaped = escape_drawtext(clip.get("title", ""))

    force_style = (
        "FontName=Arial,"
        "FontSize=12,"
        "Bold=1,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"
        "Outline=3,"
        "Shadow=0,"
        "Alignment=2,"
        "MarginV=50"
    )

    vf_base = (
        f"split=2[bg][fg];"
        f"[bg]scale=-2:2560,crop=1440:2560:(iw-1440)/2:0,"
        f"gblur=sigma=40[bg_out];"
        f"[fg]scale=2160:-2,crop=1440:ih:(iw-1440)/2:0[fg_out];"
        f"[bg_out][fg_out]overlay=0:(H-h)/2,"
        f"drawtext=text='{title_escaped}':"
        f"fontfile=/usr/share/fonts/TTF/Arialbd.TTF:"
        f"fontsize=90:fontcolor=white:"
        f"borderw=4:bordercolor=black:"
        f"x=(w-text_w)/2:y=200,"
        f"subtitles={srt_escaped}:fontsdir=/usr/share/fonts/TTF/:force_style='{force_style}',"
        f"setpts=PTS/1.2"
    )

    vf_vaapi = vf_base + ",format=nv12,hwupload"

    vaapi_cmd = [
        "ffmpeg", "-y",
        "-init_hw_device", "vaapi=va:/dev/dri/renderD128",
        "-filter_hw_device", "va",
        "-i", video_path,
        "-ss", clip["start_time"],
        "-to", clip["end_time"],
        "-vf", vf_vaapi,
        "-c:v", "h264_vaapi",
        "-qp", "23",
        "-af", "atempo=1.2",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]

    cpu_cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ss", clip["start_time"],
        "-to", clip["end_time"],
        "-vf", vf_base,
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "fast",
        "-af", "atempo=1.2",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]

    print(f"  Running FFmpeg (VAAPI)...")
    result = subprocess.run(vaapi_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  VAAPI encoding failed, falling back to CPU...")
        result = subprocess.run(cpu_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  FFmpeg error:\n{result.stderr[-500:]}")
        return False

    return True
