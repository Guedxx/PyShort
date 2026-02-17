import json
import re
import sys

MIN_CLIP_DURATION_SECONDS = 15
MAX_CLIP_DURATION_SECONDS = 60
TIMESTAMP_PATTERN = re.compile(r"^(\d+):([0-5]\d):([0-5]\d)$")
REQUIRED_CLIP_FIELDS = ("start_time", "end_time", "title")


def _parse_json_payload(text: str) -> dict | list:
    """Parse JSON from a raw model response, including markdown-wrapped JSON."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as direct_error:
        code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        for block in code_blocks:
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue

        decoder = json.JSONDecoder()
        for match in re.finditer(r"[\[{]", text):
            try:
                payload, _ = decoder.raw_decode(text[match.start() :])
                return payload
            except json.JSONDecodeError:
                continue

        raise ValueError(f"Invalid JSON: {direct_error.msg}") from direct_error


def _extract_clips(payload: dict | list) -> list[dict]:
    if isinstance(payload, dict):
        if "clips" not in payload:
            raise ValueError("Top-level JSON object must include a 'clips' key.")
        clips = payload["clips"]
    elif isinstance(payload, list):
        clips = payload
    else:
        raise ValueError("Response JSON must be an object with 'clips' or a list of clip objects.")

    if not isinstance(clips, list):
        raise ValueError("'clips' must be a JSON array.")
    if not clips:
        raise ValueError("No clips found in response; provide at least one clip.")

    return clips


def _to_seconds(timestamp: str, *, clip_index: int, field: str) -> int:
    value = timestamp.strip()
    match = TIMESTAMP_PATTERN.fullmatch(value)
    if not match:
        raise ValueError(
            f"Clip {clip_index} has invalid {field} '{timestamp}'; expected HH:MM:SS."
        )

    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    return (hours * 3600) + (minutes * 60) + seconds


def _normalize_timestamp(timestamp: str) -> str:
    hours, minutes, seconds = map(int, timestamp.strip().split(":"))
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _normalize_clip(clip: dict, clip_index: int) -> dict:
    if not isinstance(clip, dict):
        raise ValueError(f"Clip {clip_index} must be a JSON object.")

    missing_fields = [field for field in REQUIRED_CLIP_FIELDS if field not in clip]
    if missing_fields:
        fields = ", ".join(missing_fields)
        raise ValueError(f"Clip {clip_index} missing required field(s): {fields}.")

    normalized: dict[str, str] = {}
    for field in REQUIRED_CLIP_FIELDS:
        value = clip[field]
        if not isinstance(value, str):
            raise ValueError(f"Clip {clip_index} field '{field}' must be a string.")
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"Clip {clip_index} field '{field}' cannot be empty.")
        normalized[field] = stripped

    # `reason` is optional: keep compatibility with model outputs that omit it.
    if "reason" in clip:
        reason_value = clip["reason"]
        if not isinstance(reason_value, str):
            raise ValueError(f"Clip {clip_index} field 'reason' must be a string.")
        normalized["reason"] = reason_value.strip()
    else:
        normalized["reason"] = ""

    start_seconds = _to_seconds(
        normalized["start_time"], clip_index=clip_index, field="start_time"
    )
    end_seconds = _to_seconds(
        normalized["end_time"], clip_index=clip_index, field="end_time"
    )

    if start_seconds >= end_seconds:
        raise ValueError(
            f"Clip {clip_index} has invalid time range: start_time must be before end_time."
        )

    clip_duration = end_seconds - start_seconds
    if not MIN_CLIP_DURATION_SECONDS <= clip_duration <= MAX_CLIP_DURATION_SECONDS:
        raise ValueError(
            f"Clip {clip_index} duration is {clip_duration}s; expected "
            f"{MIN_CLIP_DURATION_SECONDS}-{MAX_CLIP_DURATION_SECONDS}s."
        )

    normalized["start_time"] = _normalize_timestamp(normalized["start_time"])
    normalized["end_time"] = _normalize_timestamp(normalized["end_time"])
    return normalized


def parse_ai_response(text: str) -> list[dict]:
    try:
        payload = _parse_json_payload(text)
        clips = _extract_clips(payload)
        return [_normalize_clip(clip, index + 1) for index, clip in enumerate(clips)]
    except ValueError as error:
        print(f"Failed to parse AI response: {error}")
        sys.exit(1)
