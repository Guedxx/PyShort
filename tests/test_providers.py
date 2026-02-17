import pytest

from src import providers


def _unexpected_handler(name: str):
    def _handler(*_args, **_kwargs):
        pytest.fail(f"Unexpected call to {name}")

    return _handler


@pytest.mark.parametrize(
    ("provider_name", "handler_name"),
    [
        ("openai", "find_clips_openai"),
        ("gemini", "find_clips_gemini"),
        ("ollama", "find_clips_ollama"),
    ],
)
def test_find_clips_routes_known_providers(monkeypatch, provider_name, handler_name):
    for name in ("find_clips_openai", "find_clips_gemini", "find_clips_ollama"):
        monkeypatch.setattr(providers, name, _unexpected_handler(name))

    expected = f"{provider_name}-response"
    called = {}

    def selected_handler(transcript, model):
        called["args"] = (transcript, model)
        return expected

    monkeypatch.setattr(providers, handler_name, selected_handler)

    result = providers.find_clips(provider_name, "some transcript", "test-model")

    assert result == expected
    assert called["args"] == ("some transcript", "test-model")


def test_find_clips_invalid_provider_exits_with_clear_message(capsys):
    with pytest.raises(SystemExit) as exc:
        providers.find_clips("invalid-provider", "some transcript", "test-model")

    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "Unknown provider 'invalid-provider'" in output
    assert "openai" in output
    assert "gemini" in output
    assert "ollama" in output
