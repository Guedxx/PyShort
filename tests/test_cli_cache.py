import importlib
import json
import sys
import types
from types import SimpleNamespace

import pytest


@pytest.fixture
def cli_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", types.ModuleType("cv2"))
    sys.modules.pop("src.cli", None)
    sys.modules.pop("src.video", None)
    sys.modules.pop("src.framing", None)
    return importlib.import_module("src.cli")


def _configure_common(monkeypatch, cli_module, tmp_path):
    cfg = SimpleNamespace(
        provider=None,
        model=None,
        output_dir=str(tmp_path / "clips"),
        remove_silence=False,
    )
    monkeypatch.setattr(cli_module, "load_config", lambda _path: cfg)
    monkeypatch.setattr(cli_module, "get_video_duration", lambda _path: 120.0)
    monkeypatch.setattr(cli_module, "clip_video", lambda *_args, **_kwargs: True)


def test_uses_existing_cuts_cache_in_video_folder(monkeypatch, tmp_path, cli_module):
    _configure_common(monkeypatch, cli_module, tmp_path)

    video_path = tmp_path / "input.mp4"
    srt_path = tmp_path / "input.srt"
    cuts_path = tmp_path / "cuts.json"
    video_path.write_text("", encoding="utf-8")
    srt_path.write_text("srt", encoding="utf-8")

    cached_response = '{"clips":[{"start_time":"00:00:10","end_time":"00:00:30","title":"From cache"}]}'
    cuts_path.write_text(json.dumps({"response": cached_response}), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        ["short-maker", str(video_path), str(srt_path), "--openai"],
    )
    monkeypatch.setattr(cli_module, "read_srt", lambda _path: "transcript")
    monkeypatch.setattr(
        cli_module,
        "find_clips",
        lambda *_args: pytest.fail("find_clips should not be called when cache exists"),
    )

    seen = {}

    def _parse(raw):
        seen["raw"] = raw
        return [{"start_time": "00:00:10", "end_time": "00:00:30", "title": "From cache"}]

    monkeypatch.setattr(cli_module, "parse_ai_response", _parse)

    cli_module.main()

    assert seen["raw"] == cached_response


def test_writes_cuts_cache_after_provider_response(monkeypatch, tmp_path, cli_module):
    _configure_common(monkeypatch, cli_module, tmp_path)

    video_path = tmp_path / "input.mp4"
    srt_path = tmp_path / "input.srt"
    cuts_path = tmp_path / "cuts.json"
    video_path.write_text("", encoding="utf-8")
    srt_path.write_text("srt", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        ["short-maker", str(video_path), str(srt_path), "--openai"],
    )
    monkeypatch.setattr(cli_module, "read_srt", lambda _path: "transcript")

    raw_response = '{"clips":[{"start_time":"00:00:15","end_time":"00:00:35","title":"From model"}]}'
    monkeypatch.setattr(cli_module, "find_clips", lambda *_args: raw_response)
    monkeypatch.setattr(
        cli_module,
        "parse_ai_response",
        lambda _raw: [{"start_time": "00:00:15", "end_time": "00:00:35", "title": "From model"}],
    )

    cli_module.main()

    assert cuts_path.exists()
    payload = json.loads(cuts_path.read_text(encoding="utf-8"))
    assert payload["response"] == raw_response
