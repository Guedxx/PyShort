import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

MODEL_DEFAULTS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-3-flash-preview",
}

CONFIG_SEARCH_PATHS = [
    Path("config.toml"),
    Path.home() / ".config" / "short-maker" / "config.toml",
]


@dataclass
class Config:
    provider: str | None = None
    model: str | None = None
    output_dir: str = "./shorts_clips"
    remove_silence: bool = False


def load_dotenv() -> None:
    """Load .env.local if python-dotenv is available."""
    try:
        from dotenv import load_dotenv as _load
        env_path = Path(".env.local")
        if env_path.exists():
            _load(env_path)
    except ImportError:
        pass


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
