from types import SimpleNamespace

from app.services.llm import (
    LLMProviderError,
    _coerce_content_to_text,
    _extract_message_text,
    _map_provider_exception,
)


def test_coerce_content_list_parts_to_text():
    content = [
        {"type": "text", "text": "Bonjour "},
        {"text": "ChatFleet"},
        " !",
    ]

    assert _coerce_content_to_text(content) == "Bonjour ChatFleet !"


def test_extract_qwen_reasoning_without_answer_metrics():
    choice = SimpleNamespace(
        finish_reason="length",
        message=SimpleNamespace(content="", reasoning_content="Thinking Process..."),
    )
    response = SimpleNamespace(
        usage=SimpleNamespace(completion_tokens=200),
    )

    text, metrics = _extract_message_text(choice, response)

    assert text == ""
    assert metrics["finish_reason"] == "length"
    assert metrics["reasoning_len"] > 0
    assert metrics["completion_tokens"] == 200


def test_extract_inline_thinking_without_final_answer_metrics():
    choice = SimpleNamespace(
        finish_reason="length",
        message=SimpleNamespace(content="<think>", reasoning_content=None),
    )
    response = SimpleNamespace(
        usage=SimpleNamespace(completion_tokens=1),
    )

    text, metrics = _extract_message_text(choice, response)

    assert text == ""
    assert metrics["reasoning_len"] > 0
    assert metrics["completion_tokens"] == 1


def test_extract_inline_thinking_keeps_final_answer():
    choice = SimpleNamespace(
        finish_reason="stop",
        message=SimpleNamespace(content="<think>Reasoning</think>\nFinal answer.", reasoning_content=None),
    )
    response = SimpleNamespace(
        usage=SimpleNamespace(completion_tokens=12),
    )

    text, metrics = _extract_message_text(choice, response)

    assert text == "Final answer."
    assert metrics["reasoning_len"] > 0
    assert metrics["content_len"] == len("Final answer.")


def test_map_timeout_to_actionable_error():
    exc = TimeoutError("request timed out")

    mapped = _map_provider_exception(exc)

    assert isinstance(mapped, LLMProviderError)
    assert mapped.code == "LLM_TIMEOUT"
    assert "trop de temps" in mapped.user_message


def test_map_context_limit_to_actionable_error():
    exc = RuntimeError("request exceeds the available context size")

    mapped = _map_provider_exception(exc)

    assert mapped.code == "LLM_CONTEXT_LIMIT"
    assert "fenêtre de contexte" in mapped.user_message
