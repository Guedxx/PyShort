import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

MODEL_DEFAULTS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-3-flash-preview",
    "ollama": "llama3",
}

CONFIG_SEARCH_PATHS = [
    Path("config.toml"),
    Path.home() / ".config" / "short-maker" / "config.toml",
]

ENV_SEARCH_PATHS = [
    Path(".env"),
    Path(".env.local"),
]


@dataclass
class Config:
    provider: str | None = None
    model: str | None = None
    output_dir: str = "./shorts_clips"
    remove_silence: bool = False


def load_dotenv() -> None:
    """Load env vars from .env files if python-dotenv is available.

    Loading order is deterministic:
    1) .env
    2) .env.local

    Existing process environment variables are never overridden.
    """
    try:
        from dotenv import dotenv_values
    except ImportError:
        return

    merged: dict[str, str] = {}
    for env_path in ENV_SEARCH_PATHS:
        if not env_path.exists():
            continue
        values = dotenv_values(env_path)
        for key, value in values.items():
            if value is not None:
                merged[key] = value

    for key, value in merged.items():
        os.environ.setdefault(key, value)


def load_config(path: str | None = None) -> Config:
    """Load config from TOML file.

    Search order: explicit path > ./config.toml > ~/.config/short-maker/config.toml
    Missing config file is not an error (defaults are used).
    """
    load_dotenv()

    config_path = None
    if path:
        config_path = Path(path)
        if not config_path.exists():
            print(f"Config file not found: {path}")
            sys.exit(1)
    else:
        for candidate in CONFIG_SEARCH_PATHS:
            if candidate.exists():
                config_path = candidate
                break

    if config_path is None:
        return Config()

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    ai = data.get("ai", {})
    output = data.get("output", {})

    return Config(
        provider=ai.get("provider"),
        model=ai.get("model"),
        output_dir=output.get("dir", Config.output_dir),
        remove_silence=output.get("remove_silence", False),
    )
