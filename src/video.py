import os
import re
import subprocess
from typing import List, Tuple

from src.framing import detect_primary_face_x
from src.utils import parse_time_str

VAAPI_DEVICE = "/dev/dri/renderD128"
ENCODE_SPEED = 1.2
MIN_INTERVAL_SECONDS = 0.05
DEFAULT_FORCE_STYLE = (
    "FontName=Arial,"
    "FontSize=12,"
    "Bold=1,"
    "PrimaryColour=&H00FFFFFF,"
    "OutlineColour=&H00000000,"
    "Outline=1,"
    "Shadow=0,"
    "MarginV=62"
)
FONT_CANDIDATES = [
    "/usr/share/fonts/TTF/Arialbd.TTF",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def escape_srt_path(path: str) -> str:
    """Escape special chars for FFmpeg subtitles/font path arguments."""
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


def _resolve_font_file() -> str | None:
    override = os.getenv("SHORT_MAKER_FONT_FILE")
    candidates = [override] if override else []
    candidates.extend(FONT_CANDIDATES)

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def _font_spec(font_file: str | None) -> str:
    if font_file:
        return f"fontfile={escape_srt_path(font_file)}:"
    return "font=Sans:"


def _build_subtitle_filter(srt_path: str | None, fonts_dir: str | None) -> str:
    if not srt_path:
        return ""

    parts = [f"subtitles={escape_srt_path(srt_path)}"]
    if fonts_dir:
        parts.append(f"fontsdir={escape_srt_path(fonts_dir)}")
    parts.append(f"force_style='{DEFAULT_FORCE_STYLE}'")
    return ":".join(parts) + ","


def _split_title_lines(title: str) -> tuple[str, str | None]:
    words = title.split()
    if len(words) > 4:
        mid = (len(words) + 1) // 2
        return escape_drawtext(" ".join(words[:mid])), escape_drawtext(" ".join(words[mid:]))
    return escape_drawtext(title), None


def _crop_x_from_face(face_x: float, scaled_width: int = 2160, crop_width: int = 1440) -> int:
    face_center_px = face_x * scaled_width
    crop_x = face_center_px - (crop_width / 2)
    crop_x = max(0, min(scaled_width - crop_width, crop_x))
    return int(crop_x)


def _build_visual_filter(
    crop_x: int,
    title_line1: str,
    title_line2: str | None,
    srt_path: str | None,
    font_file: str | None,
    fonts_dir: str | None,
) -> str:
    subtitle_filter = _build_subtitle_filter(srt_path, fonts_dir)
    font_spec = _font_spec(font_file)
    cta_text = escape_drawtext("Watch Full Video Here \u25BC")

    title_line2_filter = (
        f"drawtext=text='{title_line2}':"
        f"{font_spec}"
        f"fontsize=90:fontcolor=white:"
        f"borderw=10:bordercolor=black:"
        f"x=(w-text_w)/2:y=310,"
        if title_line2
        else ""
    )

    cta_filter = (
        f"drawtext=text='{cta_text}':"
        f"{font_spec}"
        f"fontsize=30:fontcolor=red:"
        f"borderw=3:bordercolor=white:"
        f"alpha='if(lt(mod(t,1),0.5),1,0)':"
        f"x=(w-text_w)/2-20:y=h-310"
    )

    return (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale=-2:2560,crop=1440:2560:(iw-1440)/2:0,"
        f"gblur=sigma=40[bg_out];"
        f"[fg]scale=2160:-2,crop=1440:ih:{crop_x}:0[fg_out];"
        f"[bg_out][fg_out]overlay=0:(H-h)/2,"
        f"drawtext=text='{title_line1}':"
        f"{font_spec}"
        f"fontsize=90:fontcolor=white:"
        f"borderw=10:bordercolor=black:"
        f"x=(w-text_w)/2:y=200,"
        f"{title_line2_filter}"
        f"{subtitle_filter}"
        f"{cta_filter}[v_visual]"
    )


def _build_normal_filter_complex(
    visual_filter: str,
    start_seconds: float,
    speed: float = ENCODE_SPEED,
) -> tuple[str, str, str]:
    filter_complex = (
        f"{visual_filter};"
        f"[v_visual]setpts=(PTS-{start_seconds}/TB)/{speed}[outv];"
        f"[0:a]asetpts=PTS-{start_seconds}/TB,atempo={speed}[outa]"
    )
    return filter_complex, "[outv]", "[outa]"


def _format_time(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _build_silence_filter_complex(
    visual_filter: str,
    intervals: List[Tuple[float, float]],
    start_seconds: float,
    speed: float = ENCODE_SPEED,
) -> tuple[str, str, str]:
    if not intervals:
        raise ValueError("Silence filter requested without intervals")

    count = len(intervals)
    v_sources = [f"[v_src{i}]" for i in range(count)]
    a_sources = [f"[a_src{i}]" for i in range(count)]

    parts = [
        visual_filter,
        f"[v_visual]split={count}{''.join(v_sources)}",
        f"[0:a]asplit={count}{''.join(a_sources)}",
    ]

    concat_inputs = []
    for i, (seg_start, seg_end) in enumerate(intervals):
        abs_start = start_seconds + seg_start
        abs_end = start_seconds + seg_end

        parts.append(
            f"{v_sources[i]}trim=start={_format_time(abs_start)}:end={_format_time(abs_end)},"
            f"setpts=PTS-STARTPTS[v{i}]"
        )
        parts.append(
            f"{a_sources[i]}atrim=start={_format_time(abs_start)}:end={_format_time(abs_end)},"
            f"asetpts=PTS-STARTPTS[a{i}]"
        )
        concat_inputs.append(f"[v{i}][a{i}]")

    parts.append(f"{''.join(concat_inputs)}concat=n={count}:v=1:a=1[v_cat][a_cat]")
    parts.append(f"[v_cat]setpts=PTS/{speed}[outv]")
    parts.append(f"[a_cat]atempo={speed}[outa]")

    return ";".join(parts), "[outv]", "[outa]"


def _build_vaapi_filter_complex(filter_complex: str, video_map: str) -> tuple[str, str]:
    return f"{filter_complex};{video_map}format=nv12,hwupload[outv_hw]", "[outv_hw]"


def _build_cpu_cmd(
    video_path: str,
    clip: dict,
    output_path: str,
    filter_complex: str,
    video_map: str,
    audio_map: str,
) -> List[str]:
    return [
        "ffmpeg",
        "-y",
        "-ss",
        clip["start_time"],
        "-to",
        clip["end_time"],
        "-copyts",
        "-i",
        video_path,
        "-filter_complex",
        filter_complex,
        "-map",
        video_map,
        "-map",
        audio_map,
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-preset",
        "fast",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        output_path,
    ]


def _build_vaapi_cmd(
    video_path: str,
    clip: dict,
    output_path: str,
    filter_complex: str,
    video_map: str,
    audio_map: str,
) -> List[str]:
    return [
        "ffmpeg",
        "-y",
        "-init_hw_device",
        f"vaapi=va:{VAAPI_DEVICE}",
        "-filter_hw_device",
        "va",
        "-ss",
        clip["start_time"],
        "-to",
        clip["end_time"],
        "-copyts",
        "-i",
        video_path,
        "-filter_complex",
        filter_complex,
        "-map",
        video_map,
        "-map",
        audio_map,
        "-c:v",
        "h264_vaapi",
        "-qp",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        output_path,
    ]


def _vaapi_available() -> bool:
    disabled = os.getenv("SHORT_MAKER_DISABLE_VAAPI", "").strip().lower()
    if disabled in {"1", "true", "yes"}:
        return False
    return os.path.exists(VAAPI_DEVICE) and os.access(VAAPI_DEVICE, os.R_OK | os.W_OK)


def detect_silence_intervals(
    video_path: str,
    start_time: str,
    end_time: str,
    db_threshold: int = -30,
    min_duration: float = 0.5,
) -> List[Tuple[float, float]]:
    """
    Detect non-silent intervals in a video segment.
    Returns intervals relative to the clip start (0-based).
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        start_time,
        "-to",
        end_time,
        "-i",
        video_path,
        "-af",
        f"silencedetect=noise={db_threshold}dB:d={min_duration}",
        "-f",
        "null",
        "-",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stderr or ""

    silence_starts: List[float] = []
    silence_ends: List[float] = []

    for line in output.splitlines():
        if "silence_start:" in line:
            match = re.search(r"silence_start:\s*([0-9]+(?:\.[0-9]+)?)", line)
            if match:
                silence_starts.append(float(match.group(1)))
        elif "silence_end:" in line:
            match = re.search(r"silence_end:\s*([0-9]+(?:\.[0-9]+)?)", line)
            if match:
                silence_ends.append(float(match.group(1)))

    seg_start = parse_time_str(start_time)
    seg_end = parse_time_str(end_time)
    duration = max(0.0, seg_end - seg_start)

    if duration == 0:
        return []

    keep_segments: List[Tuple[float, float]] = []
    current_time = 0.0

    for index, s_start in enumerate(silence_starts):
        s_end = silence_ends[index] if index < len(silence_ends) else duration
        s_start = max(0.0, min(duration, s_start))
        s_end = max(0.0, min(duration, s_end))

        if s_start > current_time:
            keep_segments.append((current_time, s_start))
        current_time = max(current_time, s_end)

    if current_time < duration:
        keep_segments.append((current_time, duration))

    return [
        (start, end)
        for start, end in keep_segments
        if (end - start) >= MIN_INTERVAL_SECONDS
    ]


def _is_full_duration_interval(intervals: List[Tuple[float, float]], duration: float) -> bool:
    if len(intervals) != 1:
        return False
    start, end = intervals[0]
    return start <= 0.05 and end >= duration - 0.05


def clip_video(
    video_path: str,
    srt_path: str | None,
    clip: dict,
    output_path: str,
    remove_silence: bool = False,
) -> bool:
    if "start_time" not in clip or "end_time" not in clip:
        print("  Invalid clip payload: missing start_time/end_time.")
        return False

    try:
        start_seconds = parse_time_str(clip["start_time"])
        end_seconds = parse_time_str(clip["end_time"])
    except ValueError as exc:
        print(f"  Invalid clip timestamp: {exc}")
        return False

    if end_seconds <= start_seconds:
        print("  Invalid clip range: end_time must be greater than start_time.")
        return False

    title_line1, title_line2 = _split_title_lines(clip.get("title", ""))

    print(f"  Analyzing frame at {clip['start_time']} for face...")
    face_x = detect_primary_face_x(video_path, clip["start_time"])
    crop_x = _crop_x_from_face(face_x)
    print(f"  Face x: {face_x:.2f}, Crop x: {crop_x}")

    font_file = _resolve_font_file()
    fonts_dir = os.path.dirname(font_file) if font_file else None
    if not font_file:
        print("  Font file not found in known paths. Using ffmpeg default font.")

    visual_filter = _build_visual_filter(
        crop_x=crop_x,
        title_line1=title_line1,
        title_line2=title_line2,
        srt_path=srt_path,
        font_file=font_file,
        fonts_dir=fonts_dir,
    )

    intervals: List[Tuple[float, float]] = []
    if remove_silence:
        print("  Detecting silence...")
        intervals = detect_silence_intervals(video_path, clip["start_time"], clip["end_time"])
        clip_duration = end_seconds - start_seconds
        if not intervals or _is_full_duration_interval(intervals, clip_duration):
            print("  No significant silence detected.")
            intervals = []

    if intervals:
        print(f"  Removing silence: {len(intervals)} clean segments.")
        filter_complex, map_v, map_a = _build_silence_filter_complex(
            visual_filter=visual_filter,
            intervals=intervals,
            start_seconds=start_seconds,
        )
    else:
        filter_complex, map_v, map_a = _build_normal_filter_complex(
            visual_filter=visual_filter,
            start_seconds=start_seconds,
        )

    cpu_cmd = _build_cpu_cmd(
        video_path=video_path,
        clip=clip,
        output_path=output_path,
        filter_complex=filter_complex,
        video_map=map_v,
        audio_map=map_a,
    )

    result = None
    if _vaapi_available():
        vaapi_filter, vaapi_map_v = _build_vaapi_filter_complex(filter_complex, map_v)
        vaapi_cmd = _build_vaapi_cmd(
            video_path=video_path,
            clip=clip,
            output_path=output_path,
            filter_complex=vaapi_filter,
            video_map=vaapi_map_v,
            audio_map=map_a,
        )

        print("  Running FFmpeg (VAAPI)...")
        result = subprocess.run(vaapi_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("  VAAPI encoding failed, falling back to CPU...")
    else:
        print("  VAAPI device unavailable. Using CPU encoding.")

    if result is None or result.returncode != 0:
        print("  Running FFmpeg (CPU)...")
        result = subprocess.run(cpu_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  FFmpeg error:\n{(result.stderr or '')[-500:]}")
        return False

    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        print("  FFmpeg produced an empty output file.")
        return False

    return True
