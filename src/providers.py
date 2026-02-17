import os
import sys

from src.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


def _exit_with_error(message: str) -> None:
    """Print a user-facing provider error and terminate."""
    print(message)
    sys.exit(1)


def find_clips_openai(transcript: str, model: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        _exit_with_error("OpenAI provider unavailable. Install dependency: pip install openai")

    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(transcript=transcript)},
            ],
        )
        content = response.choices[0].message.content
    except Exception as exc:
        _exit_with_error(f"OpenAI request failed: {exc}")

    if not content:
        _exit_with_error("OpenAI returned an empty response.")
    return content.strip()


def find_clips_gemini(transcript: str, model: str) -> str:
    try:
        from google import genai
    except ImportError:
        _exit_with_error("Gemini provider unavailable. Install dependency: pip install google-genai")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        _exit_with_error("Gemini provider requires GEMINI_API_KEY to be set.")

    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=model,
            contents=f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TEMPLATE.format(transcript=transcript)}",
        )
        content = response.text
    except Exception as exc:
        _exit_with_error(f"Gemini request failed: {exc}")

    if not content:
        _exit_with_error("Gemini returned an empty response.")
    return content.strip()


def find_clips(provider: str, transcript: str, model: str) -> str:
    provider_handlers = {
        "openai": find_clips_openai,
        "gemini": find_clips_gemini,
        "ollama": find_clips_ollama,
    }
    handler = provider_handlers.get(provider)
    if handler is None:
        valid_providers = ", ".join(provider_handlers.keys())
        _exit_with_error(f"Unknown provider '{provider}'. Expected one of: {valid_providers}")
    return handler(transcript, model)


def find_clips_ollama(transcript: str, model: str) -> str:
    try:
        import ollama
    except ImportError:
        _exit_with_error("Ollama provider unavailable. Install dependency: pip install ollama")

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(transcript=transcript)},
            ],
        )
        content = response["message"]["content"]
    except Exception as exc:
        _exit_with_error(f"Ollama request failed: {exc}")

    if not content:
        _exit_with_error("Ollama returned an empty response.")
    return content.strip()
