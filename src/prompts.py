SYSTEM_PROMPT = (
    "You are an expert video editor and social media strategist. "
    "Analyze video transcripts and identify the most viral-worthy, "
    "engaging segments for YouTube Shorts. Respond with valid JSON only."
)

USER_PROMPT_TEMPLATE = """Analyze this SRT transcript and identify the most engaging 15-60 second segments for YouTube Shorts.

SRT Transcript:
{transcript}

Return ONLY valid JSON in this exact format:
{{"clips":[{{"start_time":"HH:MM:SS","end_time":"HH:MM:SS","title":"Short descriptive title","reason":"Why this clip is engaging"}}]}}

Rules:
- Each clip MUST be between 15 and 60 seconds long
- Identify 3-5 of the most engaging moments
- Prefer segments with strong hooks, surprising statements, emotional peaks, or self-contained stories
- start_time and end_time MUST use HH:MM:SS format
- Make sure the titles are short and engaging
- Make sure the sentences and logic a cohesive for a self-contained clip
- Return ONLY the JSON object, nothing else"""
