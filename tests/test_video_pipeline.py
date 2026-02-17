from src import video


class _Result:
    def __init__(self, returncode: int = 0, stderr: str = "", stdout: str = ""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout


def _mock_output_ok(monkeypatch):
    monkeypatch.setattr(video.os.path, "isfile", lambda _: True)
    monkeypatch.setattr(video.os.path, "getsize", lambda _: 1024)


def _base_clip() -> dict:
    return {"start_time": "00:00", "end_time": "00:05", "title": "hello from tests"}


def test_clip_video_uses_cpu_when_vaapi_unavailable(monkeypatch):
    commands = []

    monkeypatch.setattr(video, "detect_primary_face_x", lambda *_: 0.5)
    monkeypatch.setattr(video, "_resolve_font_file", lambda: None)
    monkeypatch.setattr(video, "_vaapi_available", lambda: False)
    monkeypatch.setattr(video.subprocess, "run", lambda cmd, **_: commands.append(cmd) or _Result(0))
    _mock_output_ok(monkeypatch)

    ok = video.clip_video("input.mp4", None, _base_clip(), "out.mp4", remove_silence=False)

    assert ok is True
    assert len(commands) == 1
    assert "libx264" in commands[0]
    assert "-filter_complex" in commands[0]
    filter_str = commands[0][commands[0].index("-filter_complex") + 1]
    assert "concat=n=" not in filter_str


def test_clip_video_falls_back_to_cpu_after_vaapi_failure(monkeypatch):
    commands = []

    def fake_run(cmd, **_):
        commands.append(cmd)
        if "h264_vaapi" in cmd:
            return _Result(1, stderr="vaapi failure")
        return _Result(0)

    monkeypatch.setattr(video, "detect_primary_face_x", lambda *_: 0.5)
    monkeypatch.setattr(video, "_resolve_font_file", lambda: None)
    monkeypatch.setattr(video, "_vaapi_available", lambda: True)
    monkeypatch.setattr(video.subprocess, "run", fake_run)
    _mock_output_ok(monkeypatch)

    ok = video.clip_video("input.mp4", None, _base_clip(), "out.mp4", remove_silence=False)

    assert ok is True
    assert len(commands) == 2
    assert "h264_vaapi" in commands[0]
    assert "libx264" in commands[1]


def test_clip_video_uses_silence_pipeline_when_intervals_exist(monkeypatch):
    commands = []

    monkeypatch.setattr(video, "detect_primary_face_x", lambda *_: 0.5)
    monkeypatch.setattr(video, "_resolve_font_file", lambda: None)
    monkeypatch.setattr(video, "_vaapi_available", lambda: False)
    monkeypatch.setattr(video, "detect_silence_intervals", lambda *_: [(0.0, 1.0), (1.5, 3.0)])
    monkeypatch.setattr(video.subprocess, "run", lambda cmd, **_: commands.append(cmd) or _Result(0))
    _mock_output_ok(monkeypatch)

    ok = video.clip_video("input.mp4", None, _base_clip(), "out.mp4", remove_silence=True)

    assert ok is True
    filter_str = commands[0][commands[0].index("-filter_complex") + 1]
    assert "concat=n=2:v=1:a=1" in filter_str
    assert "[v_visual]split=2" in filter_str


def test_clip_video_falls_back_to_normal_pipeline_without_real_silence(monkeypatch):
    commands = []

    monkeypatch.setattr(video, "detect_primary_face_x", lambda *_: 0.5)
    monkeypatch.setattr(video, "_resolve_font_file", lambda: None)
    monkeypatch.setattr(video, "_vaapi_available", lambda: False)
    monkeypatch.setattr(video, "detect_silence_intervals", lambda *_: [(0.0, 5.0)])
    monkeypatch.setattr(video.subprocess, "run", lambda cmd, **_: commands.append(cmd) or _Result(0))
    _mock_output_ok(monkeypatch)

    ok = video.clip_video("input.mp4", None, _base_clip(), "out.mp4", remove_silence=True)

    assert ok is True
    filter_str = commands[0][commands[0].index("-filter_complex") + 1]
    assert "concat=n=" not in filter_str


def test_detect_silence_intervals_builds_non_silent_segments(monkeypatch):
    stderr = """
    [silencedetect @ 0x0] silence_start: 1.0
    [silencedetect @ 0x0] silence_end: 2.0 | silence_duration: 1.0
    [silencedetect @ 0x0] silence_start: 3.5
    [silencedetect @ 0x0] silence_end: 4.0 | silence_duration: 0.5
    """

    monkeypatch.setattr(video.subprocess, "run", lambda *_, **__: _Result(0, stderr=stderr))

    intervals = video.detect_silence_intervals(
        "input.mp4",
        start_time="00:00",
        end_time="00:06",
    )

    assert intervals == [(0.0, 1.0), (2.0, 3.5), (4.0, 6.0)]
