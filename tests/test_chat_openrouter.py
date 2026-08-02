import asyncio
import os
import sys
import types
import unittest
from unittest.mock import patch

from _chat_fakes import chat_model as _chat_model
from _chat_fakes import models_requests as _models_requests

from anytrain.chat import Chat
from anytrain.chat.config import (
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_BASE_URL_ENV,
    OPENROUTER_CANDIDATE_LIMIT_ENV,
    OPENROUTER_MODELS_ENV,
)


class OpenRouterChatTest(unittest.TestCase):
    def test_openrouter_backend_uses_openai_client_without_extra_body_for_single_model(self):
        calls = []
        catalog = [_chat_model("openai/gpt-4o-mini", prompt=0.00000015, completion=0.0000006)]

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs)
                message = types.SimpleNamespace(content="done")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice], model="openai/gpt-4o-mini")

        class FakeOpenAI:
            def __init__(self, *, api_key, base_url):
                calls.append({"api_key": api_key, "base_url": base_url})
                self.chat = types.SimpleNamespace(
                    completions=FakeCompletions(),
                )

        fake_openai = types.SimpleNamespace(OpenAI=FakeOpenAI)
        env = {
            OPENROUTER_BASE_URL_ENV: "https://openrouter.ai/api/v1",
            OPENROUTER_MODELS_ENV: "openai/gpt-4o-mini",
            OPENROUTER_API_KEY_ENV: "secret",
            OPENROUTER_CANDIDATE_LIMIT_ENV: "1",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(
                sys.modules,
                {"openai": fake_openai, "requests": _models_requests(catalog)},
            ),
            self.assertLogs("anytrain.chat", level="INFO") as logs,
        ):
            response = Chat("openrouter")("hello")

        self.assertEqual(response, "done")
        self.assertEqual(
            calls[0],
            {
                "api_key": "secret",
                "base_url": "https://openrouter.ai/api/v1",
            },
        )
        self.assertEqual(
            calls[1],
            {
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )
        self.assertTrue(
            any("openrouter using model=openai/gpt-4o-mini" in message for message in logs.output)
        )

    def test_openrouter_chat_passes_image_url_content_parts(self):
        calls = []
        catalog = [
            _chat_model(
                "openai/vision",
                prompt=0.00000015,
                completion=0.0000006,
                inputs=("text", "image"),
            )
        ]

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs)
                message = types.SimpleNamespace(content="done")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice], model="openai/vision")

        class FakeOpenAI:
            def __init__(self, *, api_key, base_url):
                self.chat = types.SimpleNamespace(
                    completions=FakeCompletions(),
                )

        fake_openai = types.SimpleNamespace(OpenAI=FakeOpenAI)
        env = {
            OPENROUTER_BASE_URL_ENV: "https://openrouter.ai/api/v1",
            OPENROUTER_MODELS_ENV: "openai/vision",
            OPENROUTER_API_KEY_ENV: "secret",
            OPENROUTER_CANDIDATE_LIMIT_ENV: "1",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(
                sys.modules,
                {"openai": fake_openai, "requests": _models_requests(catalog)},
            ),
        ):
            response = Chat("openrouter")("describe this", images=["data:image/png;base64,abc"])

        self.assertEqual(response, "done")
        self.assertEqual(
            calls[0],
            {
                "model": "openai/vision",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe this"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,abc"},
                            },
                        ],
                    }
                ],
                "stream": False,
            },
        )

    def test_openrouter_backend_passes_fallback_models(self):
        calls = []
        catalog = [
            _chat_model("openai/gpt-4o-mini", prompt=0.00000015, completion=0.0000006),
            _chat_model("meta-llama/llama-3.2-3b-instruct:free", prompt=0, completion=0),
            _chat_model("google/gemini-flash-1.5", prompt=0.000000075, completion=0.0000003),
        ]

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs)
                message = types.SimpleNamespace(content="done")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(
                    choices=[choice],
                    model="google/gemini-flash-1.5",
                )

        class FakeOpenAI:
            def __init__(self, *, api_key, base_url):
                self.chat = types.SimpleNamespace(
                    completions=FakeCompletions(),
                )

        fake_openai = types.SimpleNamespace(OpenAI=FakeOpenAI)
        env = {
            OPENROUTER_BASE_URL_ENV: "https://openrouter.ai/api/v1",
            OPENROUTER_MODELS_ENV: (
                "openai/gpt-4o-mini,meta-llama/llama-3.2-3b-instruct:free,google/gemini-flash-1.5"
            ),
            OPENROUTER_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(
                sys.modules,
                {"openai": fake_openai, "requests": _models_requests(catalog)},
            ),
            self.assertLogs("anytrain.chat", level="INFO") as logs,
        ):
            client = Chat("openrouter")
            response = client("hello")

        self.assertEqual(response, "done")
        self.assertEqual(
            client.models,
            (
                "meta-llama/llama-3.2-3b-instruct:free",
                "google/gemini-flash-1.5",
                "openai/gpt-4o-mini",
            ),
        )
        self.assertEqual(
            calls[0],
            {
                "model": "meta-llama/llama-3.2-3b-instruct:free",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
                "extra_body": {
                    "models": [
                        "google/gemini-flash-1.5",
                        "openai/gpt-4o-mini",
                    ]
                },
            },
        )
        self.assertTrue(
            any(
                "openrouter using model=google/gemini-flash-1.5" in message
                for message in logs.output
            )
        )

    def test_openrouter_async_call_passes_fallback_models(self):
        calls = []
        catalog = [
            _chat_model("openai/gpt-4o-mini", prompt=0.00000015, completion=0.0000006),
            _chat_model("meta-llama/llama-3.2-3b-instruct:free", prompt=0, completion=0),
            _chat_model("google/gemini-flash-1.5", prompt=0.000000075, completion=0.0000003),
        ]

        class FakeCompletions:
            def create(self, **kwargs):
                raise AssertionError("sync client should not handle async_call")

        class FakeAsyncCompletions:
            async def create(self, **kwargs):
                calls.append(kwargs)
                message = types.SimpleNamespace(content="done")
                choice = types.SimpleNamespace(message=message, finish_reason="stop")
                return types.SimpleNamespace(
                    choices=[choice],
                    model="google/gemini-flash-1.5",
                )

        class FakeOpenAI:
            def __init__(self, *, api_key, base_url):
                self.chat = types.SimpleNamespace(
                    completions=FakeCompletions(),
                )

        class FakeAsyncOpenAI:
            def __init__(self, *, api_key, base_url):
                self.chat = types.SimpleNamespace(
                    completions=FakeAsyncCompletions(),
                )

        async def run_client():
            client = Chat("openrouter")
            return await client.async_call("hello")

        fake_openai = types.SimpleNamespace(OpenAI=FakeOpenAI, AsyncOpenAI=FakeAsyncOpenAI)
        env = {
            OPENROUTER_BASE_URL_ENV: "https://openrouter.ai/api/v1",
            OPENROUTER_MODELS_ENV: (
                "openai/gpt-4o-mini,meta-llama/llama-3.2-3b-instruct:free,google/gemini-flash-1.5"
            ),
            OPENROUTER_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(
                sys.modules,
                {"openai": fake_openai, "requests": _models_requests(catalog)},
            ),
            self.assertLogs("anytrain.chat", level="INFO") as logs,
        ):
            response = asyncio.run(run_client())

        self.assertEqual(response, "done")
        self.assertEqual(
            calls,
            [
                {
                    "model": "meta-llama/llama-3.2-3b-instruct:free",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                    "extra_body": {
                        "models": [
                            "google/gemini-flash-1.5",
                            "openai/gpt-4o-mini",
                        ]
                    },
                }
            ],
        )
        self.assertTrue(
            any(
                "openrouter using model=google/gemini-flash-1.5 finish_reason=stop" in message
                for message in logs.output
            )
        )

    def test_openrouter_backend_requires_requests_extra_for_selection(self):
        env = {
            OPENROUTER_BASE_URL_ENV: "https://openrouter.ai/api/v1",
            OPENROUTER_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"requests": None}),
            self.assertRaisesRegex(ImportError, r"anytrain\.chat"),
        ):
            Chat("openrouter")

    def test_openrouter_backend_requires_chat_extra(self):
        catalog = [_chat_model("openai/gpt-4o-mini", prompt=0.00000015, completion=0.0000006)]
        env = {
            OPENROUTER_BASE_URL_ENV: "https://openrouter.ai/api/v1",
            OPENROUTER_MODELS_ENV: "openai/gpt-4o-mini",
            OPENROUTER_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(
                sys.modules,
                {"openai": None, "requests": _models_requests(catalog)},
            ),
            self.assertRaisesRegex(ImportError, r"anytrain\.chat"),
        ):
            Chat("openrouter")("hello")

    def test_openrouter_image_generation_is_not_supported(self):
        catalog = [_chat_model("openai/gpt-4o-mini", prompt=0.00000015, completion=0.0000006)]

        class FakeOpenAI:
            def __init__(self, *, api_key, base_url):
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(),
                )

        fake_openai = types.SimpleNamespace(OpenAI=FakeOpenAI)
        env = {
            OPENROUTER_BASE_URL_ENV: "https://openrouter.ai/api/v1",
            OPENROUTER_MODELS_ENV: "openai/gpt-4o-mini",
            OPENROUTER_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(
                sys.modules,
                {"openai": fake_openai, "requests": _models_requests(catalog)},
            ),
        ):
            client = Chat("openrouter")
            with self.assertRaises(NotImplementedError):
                client.image("draw a cat")
