from __future__ import annotations

from collections.abc import Mapping, Sequence

from .types import ChatResponse


def chat_response(
    text: str,
    response: object,
    *,
    fallback_model: str | None = None,
) -> ChatResponse:
    model = response_model(response)
    return ChatResponse(
        text=text,
        model=model if model is not None else fallback_model,
        usage=response_usage(response),
        request_id=response_request_id(response),
        finish_reason=response_finish_reason(response),
        raw=response,
    )


def response_model(response: object) -> str | None:
    model = getattr(response, "model", None)
    if isinstance(model, str):
        return model
    if isinstance(response, Mapping):
        value = response.get("model")
        if isinstance(value, str):
            return value
    return None


def response_finish_reason(response: object) -> str | None:
    choice = _response_choice(response)
    if choice is None:
        return None
    finish_reason = getattr(choice, "finish_reason", None)
    if isinstance(finish_reason, str):
        return finish_reason
    if isinstance(choice, Mapping):
        value = choice.get("finish_reason")
        if isinstance(value, str):
            return value
    return None


def response_text(response: object, *, provider: str) -> str:
    choice = _response_choice(response)
    if choice is None:
        raise TypeError(f"{provider} response must contain choices[0].message.")
    message = _choice_message(choice)

    if isinstance(message, str):
        return message
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    raise TypeError(f"{provider} response message must be a string or contain string content.")


def response_usage(response: object) -> Mapping[str, object] | None:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, Mapping):
        usage = response.get("usage")
    return _usage_mapping(usage)


def response_request_id(response: object) -> str | None:
    for key in ("_request_id", "request_id", "id"):
        value = getattr(response, key, None)
        if isinstance(value, str):
            return value
    if isinstance(response, Mapping):
        for key in ("_request_id", "request_id", "id"):
            value = response.get(key)
            if isinstance(value, str):
                return value
    return None


def _response_choice(response: object) -> object | None:
    choices = getattr(response, "choices", None)
    if isinstance(response, Mapping):
        choices = response.get("choices")
    if isinstance(choices, str) or not isinstance(choices, Sequence):
        return None
    try:
        return choices[0]
    except (IndexError, TypeError):
        return None


def _choice_message(choice: object) -> object:
    message = getattr(choice, "message", None)
    if isinstance(choice, Mapping):
        message = choice.get("message")
    return message


def _usage_mapping(usage: object) -> Mapping[str, object] | None:
    if usage is None:
        return None
    if isinstance(usage, Mapping):
        return {str(key): value for key, value in usage.items()}

    for method_name in ("model_dump", "dict"):
        method = getattr(usage, method_name, None)
        if callable(method):
            try:
                payload = method()
            except TypeError:
                continue
            if isinstance(payload, Mapping):
                return {str(key): value for key, value in payload.items()}

    attrs: dict[str, object] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
        "cost",
        "total_cost",
    ):
        value = getattr(usage, key, None)
        if value is not None:
            attrs[key] = value
    if attrs:
        return attrs

    raw_attrs = getattr(usage, "__dict__", None)
    if isinstance(raw_attrs, Mapping):
        return {str(key): value for key, value in raw_attrs.items() if not str(key).startswith("_")}
    return None
