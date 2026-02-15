import re
import subprocess
from typing import List, Tuple
from src.utils import parse_time_str
from src.framing import detect_primary_face_x


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


def detect_silence_intervals(
    video_path: str,
    start_time: str,
    end_time: str,
    db_threshold: int = -30,
    min_duration: float = 0.5
) -> List[Tuple[float, float]]:
    """
    Detect non-silent intervals in the video segment.
    Returns a list of (start, end) tuples relative to the segment start (0-based).
    """
    cmd = [
        "ffmpeg", "-y",
        "-ss", start_time,
        "-to", end_time,
        "-i", video_path,
        "-af", f"silencedetect=noise={db_threshold}dB:d={min_duration}",
        "-f", "null", "-",
    ]
    
    # Run ffmpeg and capture stderr (where silencedetect writes)
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stderr
    
    silence_starts = []
    silence_ends = []
    
    for line in output.splitlines():
        if "silence_start:" in line:
            match = re.search(r"silence_start: (\d+\.?\d*)", line)
            if match:
                silence_starts.append(float(match.group(1)))
        elif "silence_end:" in line:
            match = re.search(r"silence_end: (\d+\.?\d*)", line)
            if match:
                silence_ends.append(float(match.group(1)))
                
    # Calculate total duration of the segment
    # ffmpeg stderr usually contains "Duration: ..." but better to calculate from args
    seg_start = parse_time_str(start_time)
    seg_end = parse_time_str(end_time)
    duration = seg_end - seg_start
    
    # Construct non-silent intervals
    keep_segments = []
    current_time = 0.0
    
    # Logic:
    # If silence starts > current_time, add [current_time, silence_start]
    # Update current_time to silence_end
    
    period_count = len(silence_ends) # Usually starts match ends, but sometimes silence at end
    
    for i in range(len(silence_starts)):
        s_start = silence_starts[i]
        
        # If silence starts after the current position, there is audio in between
        if s_start > current_time:
            keep_segments.append((current_time, s_start))
            
        # Move current position to the end of this silence
        if i < len(silence_ends):
            current_time = silence_ends[i]
        else:
            # Silence detected at end (or ongoing), so we skip to end?
            # Usually silence_start without silence_end means silence until EOF
            current_time = duration
            
    # Add final segment if remaining
    if current_time < duration:
        keep_segments.append((current_time, duration))
        
    # Sanity check: filter out extremely short segments?
    # For now, return as is.
    return keep_segments


def clip_video(
    video_path: str,
    srt_path: str,
    clip: dict,
    output_path: str,
    remove_silence: bool = False,
) -> bool:
    srt_escaped = escape_srt_path(srt_path)
    title_escaped = escape_drawtext(clip.get("title", ""))

    force_style = (
        "FontName=Arial,"
        "FontSize=12,"
        "Bold=1,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"
        "Outline=1,"
        "Shadow=0,"
        "Alignment=2,"
        "MarginV=50"
    )

    start_seconds = parse_time_str(clip["start_time"])
    
    # Calculate face center
    print(f"  Analyzing frame at {clip['start_time']} for face...")
    face_x = detect_primary_face_x(video_path, clip["start_time"])
    
    # Calculate crop x offset
    # The [fg] scale is 2160:-2, so width is 2160.
    # We want a 1440 width crop.
    # Center of face in pixels: face_x * 2160
    # Left edge of crop: Center - (1440 / 2)
    # Valid range for x: 0 to (2160 - 1440) = 720
    
    scaled_width = 2160
    crop_width = 1440
    face_center_px = face_x * scaled_width
    crop_x = face_center_px - (crop_width / 2)
    
    # Clamp
    crop_x = max(0, min(scaled_width - crop_width, crop_x))
    crop_x_str = f"{int(crop_x)}"
    
    print(f"  Face x: {face_x:.2f}, Crop x: {crop_x_str}")

    # Base filter chain: split -> [bg] processing, [fg] processing -> overlay -> drawtext -> subtitles
    # Note: We do NOT include setpts here yet if removing silence
    
    filter_core = (
        f"split=2[bg][fg];"
        f"[bg]scale=-2:2560,crop=1440:2560:(iw-1440)/2:0,"
        f"gblur=sigma=40[bg_out];"
        f"[fg]scale=2160:-2,crop=1440:ih:{crop_x_str}:0[fg_out];"
        f"[bg_out][fg_out]overlay=0:(H-h)/2,"
        f"drawtext=text='{title_escaped}':"
        f"fontfile=/usr/share/fonts/TTF/Arialbd.TTF:"
        f"fontsize=90:fontcolor=white:"
        f"borderw=10:bordercolor=black:"
        f"x=(w-text_w)/2:y=200,"
        f"subtitles={srt_escaped}:fontsdir=/usr/share/fonts/TTF/:force_style='{force_style}',"
        f"drawtext=text='Watch Full Video Here \u25BC':"
        f"fontfile=/usr/share/fonts/TTF/Arialbd.TTF:"
        f"fontsize=30:fontcolor=red:"
        f"x=(w-text_w)/2-20:y=h-310"
    )
    
    intervals = []
    if remove_silence:
        print("  Detecting silence...")
        intervals = detect_silence_intervals(
            video_path, clip["start_time"], clip["end_time"]
        )
        # If no silence found intervals will be [(0, duration)], which is effectively same as no silence removal
        if len(intervals) <= 1 and (not intervals or (intervals[0][0] == 0 and intervals[0][1] >= parse_time_str(clip["end_time"]) - start_seconds - 0.1)):
            print("  No significant silence detected.")
            intervals = [] # Fallback to normal mode

    if intervals:
        print(f"  Removing silence: {len(intervals)} clean segments.")
        # Complex filter construction
        # [0:v] -> filter_core -> [v_processed]
        # [v_processed] trim/concat [v_out]
        # [0:a] atrim/concat [a_out]
        
        # We append [v_base] to filter_core
        filter_str = filter_core + "[v_base_pre];"
        
        # Split video and audio for multiple trims
        count = len(intervals)
        v_sources = [f"[v_src{i}]" for i in range(count)]
        a_sources = [f"[a_src{i}]" for i in range(count)]
        
        filter_str += f"[v_base_pre]split={count}{''.join(v_sources)};"
        filter_str += f"[0:a]asplit={count}{''.join(a_sources)};"
        
        concat_inputs = ""
        
        for i, (seg_start, seg_end) in enumerate(intervals):
            # seg_start/seg_end are relative to clip start (0)
            # But we are using input seeking + -copyts, so timestamps start at start_seconds
            
            abs_start = start_seconds + seg_start
            abs_end = start_seconds + seg_end
            
            # Video trim
            filter_str += f"{v_sources[i]}trim=start={abs_start}:end={abs_end},setpts=PTS-STARTPTS[v{i}];"
            
            # Audio trim
            filter_str += f"{a_sources[i]}atrim=start={abs_start}:end={abs_end},asetpts=PTS-STARTPTS[a{i}];"
            
            # Interleave inputs for concat: [v0][a0][v1][a1]...
            concat_inputs += f"[v{i}][a{i}]"
            
        # Concatenate
        filter_str += f"{concat_inputs}concat=n={len(intervals)}:v=1:a=1[v_cat][a_cat];"
        
        # Final speedup (1.2x)
        # Verify: concat resets timestamps to 0, so just PTS/1.2 is fine
        filter_str += f"[v_cat]setpts=PTS/1.2[outv];"
        filter_str += f"[a_cat]atempo=1.2[outa]"
        
        # Output mapping names
        map_v = "[outv]"
        map_a = "[outa]"
        
    else:
        # Standard processing
        # Add timestamp correction to the core filter
        vf_base = filter_core + f",setpts=(PTS-{start_seconds}/TB)/1.2[outv]"
        
        # Audio filter
        af_base = f"asetpts=PTS-START/TB,atempo=1.2".replace("START", str(start_seconds))
        
        filter_str = vf_base  # vf argument only takes video filters usually, but we are using -filter_complex?
        # Wait, the original code used -vf and -af separately.
        # We need to adapt. 
        # For simplicity, if NO silence removal, we keep using -vf and -af as before for minimal risk.
        pass

    if intervals:
        # Use -filter_complex
        cmd = [
            "ffmpeg", "-y",
            "-init_hw_device", "vaapi=va:/dev/dri/renderD128",
            "-filter_hw_device", "va",
            "-ss", clip["start_time"],
            "-copyts",
            "-i", video_path,
            "-to", clip["end_time"],
            "-filter_complex", filter_str,
            "-map", map_v,
            "-map", map_a,
            "-c:v", "h264_vaapi",
            "-qp", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            output_path,
        ]
        
        # Fallback CPU cmd for complex filter
        # Need to adjust filter for CPU (remove hwupload if present? wait, vf_base creates [outv] on CPU if we don't upload?)
        # My filter_core ends with overlay... then trim.
        # So it is CPU based.
        # BUT for VAAPI, we need to upload to GPU *after* all processing? 
        # Or standard VAAPI flow: upload -> process -> download?
        # In original code: `vf_vaapi = vf_base + ",format=nv12,hwupload"`
        # So `vf_base` was software filters.
        # Here filter_str is doing software trims.
        # So we should upload [outv] to VAAPI at the end?
        # `[outv]format=nv12,hwupload[outv_hw]`?
        pass 
        
        # Let's fix the VAAPI complex filter string
        # The [outv] from concat is software AVFrame.
        # We need to append hwupload.
        final_v_map = "[outv_hw]"
        filter_str_vaapi = filter_str + f";[outv]format=nv12,hwupload[outv_hw]"
        
        vaapi_cmd = [
            "ffmpeg", "-y",
            "-init_hw_device", "vaapi=va:/dev/dri/renderD128",
            "-filter_hw_device", "va",
            "-ss", clip["start_time"],
            "-copyts",
            "-i", video_path,
            "-to", clip["end_time"],
            "-filter_complex", filter_str_vaapi,
            "-map", final_v_map,
            "-map", map_a,
            "-c:v", "h264_vaapi",
            "-qp", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            output_path,
        ]
        
        cpu_cmd = [
            "ffmpeg", "-y",
            "-ss", clip["start_time"],
            "-copyts",
            "-i", video_path,
            "-to", clip["end_time"],
            "-filter_complex", filter_str,
            "-map", map_v,
            "-map", map_a,
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "fast",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            output_path,
        ]

    else:
        # ORIGINAL LOGIC
        # We need to reconstruction vf_base and af_filter
        vf_base = (
            f"split=2[bg][fg];"
            f"[bg]scale=-2:2560,crop=1440:2560:(iw-1440)/2:0,"
            f"gblur=sigma=40[bg_out];"
            f"[fg]scale=2160:-2,crop=1440:ih:{crop_x_str}:0[fg_out];"
            f"[bg_out][fg_out]overlay=0:(H-h)/2,"
            f"drawtext=text='{title_escaped}':"
            f"fontfile=/usr/share/fonts/TTF/Arialbd.TTF:"
            f"fontsize=90:fontcolor=white:"
            f"borderw=4:bordercolor=black:"
            f"x=(w-text_w)/2:y=200,"
            f"subtitles={srt_escaped}:fontsdir=/usr/share/fonts/TTF/:force_style='{force_style}',"
            f"setpts=(PTS-{start_seconds}/TB)/1.2"
        )

        vf_vaapi = vf_base + ",format=nv12,hwupload"
        af_filter = f"asetpts=PTS-START/TB,atempo=1.2".replace("START", str(start_seconds))

        vaapi_cmd = [
            "ffmpeg", "-y",
            "-init_hw_device", "vaapi=va:/dev/dri/renderD128",
            "-filter_hw_device", "va",
            "-ss", clip["start_time"],
            "-copyts",
            "-i", video_path,
            "-to", clip["end_time"],
            "-vf", vf_vaapi,
            "-c:v", "h264_vaapi",
            "-qp", "23",
            "-af", af_filter,
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            output_path,
        ]

        cpu_cmd = [
            "ffmpeg", "-y",
            "-ss", clip["start_time"],
            "-copyts",
            "-i", video_path,
            "-to", clip["end_time"],
            "-vf", vf_base,
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "fast",
            "-af", af_filter,
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
        print(f"  FFmpeg error:\n{(result.stderr or '')[-500:]}")
        return False

    return True
