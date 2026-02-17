import pytest

from src.parsing import parse_ai_response


def test_parse_ai_response_normalizes_valid_clip():
    raw = """
    {
      "clips": [
        {
          "start_time": " 0:00:15 ",
          "end_time": "00:00:45",
          "title": " Great Hook ",
          "reason": " Strong opening statement "
        }
      ]
    }
    """

    clips = parse_ai_response(raw)

    assert clips == [
        {
            "start_time": "00:00:15",
            "end_time": "00:00:45",
            "title": "Great Hook",
            "reason": "Strong opening statement",
        }
    ]


def test_parse_ai_response_rejects_invalid_json(capsys):
    with pytest.raises(SystemExit) as error:
        parse_ai_response("not json at all")

    assert error.value.code == 1
    captured = capsys.readouterr()
    assert "Failed to parse AI response: Invalid JSON:" in captured.out


def test_parse_ai_response_accepts_missing_reason():
    raw = """
    {
      "clips": [
        {
          "start_time": "00:00:10",
          "end_time": "00:00:40",
          "title": "Missing reason"
        }
      ]
    }
    """

    clips = parse_ai_response(raw)
    assert clips == [
        {
            "start_time": "00:00:10",
            "end_time": "00:00:40",
            "title": "Missing reason",
            "reason": "",
        }
    ]


def test_parse_ai_response_rejects_bad_timestamp_format(capsys):
    raw = """
    {
      "clips": [
        {
          "start_time": "00:61:10",
          "end_time": "00:01:40",
          "title": "Bad timestamp",
          "reason": "Minutes cannot be 61"
        }
      ]
    }
    """

    with pytest.raises(SystemExit) as error:
        parse_ai_response(raw)

    assert error.value.code == 1
    captured = capsys.readouterr()
    assert "invalid start_time '00:61:10'; expected HH:MM:SS" in captured.out


def test_parse_ai_response_rejects_out_of_bounds_duration(capsys):
    raw = """
    {
      "clips": [
        {
          "start_time": "00:00:10",
          "end_time": "00:00:70",
          "title": "Too long",
          "reason": "Duration is outside allowed range"
        }
      ]
    }
    """

    with pytest.raises(SystemExit) as error:
        parse_ai_response(raw)

    assert error.value.code == 1
    captured = capsys.readouterr()
    assert "invalid end_time '00:00:70'; expected HH:MM:SS" in captured.out


def test_parse_ai_response_rejects_start_after_end(capsys):
    raw = """
    {
      "clips": [
        {
          "start_time": "00:00:35",
          "end_time": "00:00:20",
          "title": "Backwards",
          "reason": "Start is after end"
        }
      ]
    }
    """

    with pytest.raises(SystemExit) as error:
        parse_ai_response(raw)

    assert error.value.code == 1
    captured = capsys.readouterr()
    assert "start_time must be before end_time" in captured.out


def test_parse_ai_response_rejects_duration_too_short(capsys):
    raw = """
    {
      "clips": [
        {
          "start_time": "00:00:10",
          "end_time": "00:00:20",
          "title": "Too short",
          "reason": "Only 10 seconds"
        }
      ]
    }
    """

    with pytest.raises(SystemExit) as error:
        parse_ai_response(raw)

    assert error.value.code == 1
    captured = capsys.readouterr()
    assert "duration is 10s; expected 15-60s" in captured.out
