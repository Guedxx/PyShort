import re


def read_srt(srt_path: str) -> str:
    with open(srt_path, "r", encoding="utf-8") as f:
        return f.read()


def make_safe_filename(title: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", title)
    return safe[:50]


def parse_time_str(time_str: str) -> float:
    """Convert HH:MM:SS or MM:SS or SS to seconds."""
    parts = time_str.strip().split(":")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds
