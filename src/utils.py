import re


def read_srt(srt_path: str) -> str:
    with open(srt_path, "r", encoding="utf-8") as f:
        return f.read()


def make_safe_filename(title: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", title)
    return safe[:50]
