import importlib
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
    monkeypatch.setattr(
        cli_module.os.path,
        "isfile",
        lambda path: path in {"video.mp4", "captions.srt"},
    )
    monkeypatch.setattr(
        cli_module,
        "clip_video",
        lambda *args, **kwargs: pytest.fail("clip_video should not run in guardrail tests"),
    )


def test_manual_mode_rejects_unparseable_timestamp(monkeypatch, capsys, tmp_path, cli_module):
    _configure_common(monkeypatch, cli_module, tmp_path)
    monkeypatch.setattr(sys, "argv", ["short-maker", "video.mp4", "-m", "bad", "00:00:10"])
    monkeypatch.setattr(
        cli_module,
        "get_video_duration",
        lambda _path: pytest.fail("duration probing should not happen for invalid manual input"),
    )

    with pytest.raises(SystemExit) as exc:
        cli_module.main()

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "Invalid manual clip" in out
    assert "not parseable" in out


def test_manual_mode_rejects_start_not_before_end(monkeypatch, capsys, tmp_path, cli_module):
    _configure_common(monkeypatch, cli_module, tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["short-maker", "video.mp4", "-m", "00:00:15", "00:00:05"],
    )
    monkeypatch.setattr(
        cli_module,
        "get_video_duration",
        lambda _path: pytest.fail("duration probing should not happen for invalid manual input"),
    )

    with pytest.raises(SystemExit) as exc:
        cli_module.main()

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "start time must be before end time" in out


def test_ai_parse_errors_exit_cleanly(monkeypatch, capsys, tmp_path, cli_module):
    _configure_common(monkeypatch, cli_module, tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["short-maker", "video.mp4", "captions.srt", "--openai"],
    )
    monkeypatch.setattr(cli_module, "read_srt", lambda _path: "transcript")
    monkeypatch.setattr(cli_module, "find_clips", lambda *_args: None)
    monkeypatch.setattr(
        cli_module,
        "get_video_duration",
        lambda _path: pytest.fail("duration probing should not happen when parsing fails"),
    )

    with pytest.raises(SystemExit) as exc:
        cli_module.main()

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "AI response parsing failed" in out


def test_zero_valid_clips_exits_before_processing(monkeypatch, capsys, tmp_path, cli_module):
    _configure_common(monkeypatch, cli_module, tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["short-maker", "video.mp4", "captions.srt", "--openai"],
    )
    monkeypatch.setattr(cli_module, "read_srt", lambda _path: "transcript")
    monkeypatch.setattr(cli_module, "find_clips", lambda *_args: '{"clips": []}')
    monkeypatch.setattr(
        cli_module,
        "parse_ai_response",
        lambda _raw: [
            {"start_time": "00:00:01", "end_time": "00:00:10", "title": "Too Long"}
        ],
    )
    monkeypatch.setattr(cli_module, "get_video_duration", lambda _path: 5.0)

    with pytest.raises(SystemExit) as exc:
        cli_module.main()

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "exceeds video duration" in out
    assert "No valid clips to process after validation and duration checks." in out
