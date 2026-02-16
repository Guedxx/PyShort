import os
import sys

from src.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


def find_clips_openai(transcript: str, model: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        print("Install openai: pip install openai")
        sys.exit(1)

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(transcript=transcript)},
        ],
    )
    return response.choices[0].message.content.strip()


def find_clips_gemini(transcript: str, model: str) -> str:
    try:
        from google import genai
    except ImportError:
        print("Install google-genai: pip install google-genai")
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY environment variable")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TEMPLATE.format(transcript=transcript)}",
    )
    return response.text.strip()


def find_clips(provider: str, transcript: str, model: str) -> str:
    if provider == "openai":
        return find_clips_openai(transcript, model)
    elif provider == "gemini":
        return find_clips_gemini(transcript, model)
    elif provider == "ollama":
        return find_clips_ollama(transcript, model)


def find_clips_ollama(transcript: str, model: str) -> str:
    try:
        import ollama
    except ImportError:
        print("Install ollama: pip install ollama")
        sys.exit(1)

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(transcript=transcript)},
            ],
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"Ollama error: {e}")
        sys.exit(1)
    else:
        print(f"Unknown provider: {provider}")
        sys.exit(1)
