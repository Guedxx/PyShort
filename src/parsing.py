import json
import re
import sys


def parse_ai_response(text: str) -> list[dict]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            data = json.loads(match.group())
        else:
            print(f"Failed to parse AI response:\n{text}")
            sys.exit(1)

    clips = data.get("clips", data if isinstance(data, list) else None)
    if not clips:
        print(f"No clips found in response:\n{data}")
        sys.exit(1)

    return clips
