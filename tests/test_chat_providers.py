import asyncio
import base64
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from anytrain.chat import Chat
from anytrain.chat._provider import (
    DEEPSEEK_REASONING_EFFORT,
    DEEPSEEK_SYSTEM_PROMPT,
    GLM_MAX_TOKENS,
    GLM_TEMPERATURE,
)
from anytrain.chat.config import (
    DEEPSEEK_API_KEY_ENV,
    DEEPSEEK_BASE_URL_ENV,
    DEEPSEEK_MODEL_ENV,
    GLM_API_KEY_ENV,
    GLM_BASE_URL_ENV,
    GLM_MODEL_ENV,
)


class ChatProviderTest(unittest.TestCase):
    def test_deepseek_backend_uses_openai_client(self):
        calls = []

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs)
                message = types.SimpleNamespace(content="done")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        class FakeOpenAI:
            def __init__(self, *, api_key, base_url):
                calls.append({"api_key": api_key, "base_url": base_url})
                self.chat = types.SimpleNamespace(
                    completions=FakeCompletions(),
                )

        fake_openai = types.SimpleNamespace(OpenAI=FakeOpenAI)
        env = {
            DEEPSEEK_BASE_URL_ENV: "https://example.test/deepseek",
            DEEPSEEK_MODEL_ENV: "deepseek-v4-pro",
            DEEPSEEK_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"openai": fake_openai}),
        ):
            response = Chat("deepseek")("hello")

        self.assertEqual(response, "done")
        self.assertEqual(
            calls[0],
            {
                "api_key": "secret",
                "base_url": "https://example.test/deepseek",
            },
        )
        self.assertEqual(
            calls[1],
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    {"role": "system", "content": DEEPSEEK_SYSTEM_PROMPT},
                    {"role": "user", "content": "hello"},
                ],
                "stream": False,
                "reasoning_effort": DEEPSEEK_REASONING_EFFORT,
                "extra_body": {"thinking": {"type": "enabled"}},
            },
        )

    def test_deepseek_chat_rejects_image_inputs(self):
        class FakeCompletions:
            def create(self, **kwargs):
                raise AssertionError("DeepSeek should reject image inputs before request")

        class FakeOpenAI:
            def __init__(self, *, api_key, base_url):
                self.chat = types.SimpleNamespace(
                    completions=FakeCompletions(),
                )

        fake_openai = types.SimpleNamespace(OpenAI=FakeOpenAI)
        env = {
            DEEPSEEK_BASE_URL_ENV: "https://example.test/deepseek",
            DEEPSEEK_MODEL_ENV: "deepseek-v4-pro",
            DEEPSEEK_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"openai": fake_openai}),
        ):
            client = Chat("deepseek")
            with self.assertRaises(NotImplementedError):
                client("describe this", images=["https://example.test/cat.png"])

    def test_chat_logs_request_lifecycle_with_elapsed_and_finish_reason(self):
        class FakeCompletions:
            def create(self, **kwargs):
                message = types.SimpleNamespace(content="done")
                choice = types.SimpleNamespace(message=message, finish_reason="stop")
                return types.SimpleNamespace(choices=[choice], model=kwargs["model"])

        class FakeOpenAI:
            def __init__(self, *, api_key, base_url):
                self.chat = types.SimpleNamespace(
                    completions=FakeCompletions(),
                )

        fake_openai = types.SimpleNamespace(OpenAI=FakeOpenAI)
        env = {
            DEEPSEEK_BASE_URL_ENV: "https://example.test/deepseek",
            DEEPSEEK_MODEL_ENV: "deepseek-v4-pro",
            DEEPSEEK_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"openai": fake_openai}),
            self.assertLogs("anytrain.chat", level="INFO") as logs,
        ):
            response = Chat("deepseek")("hello")

        self.assertEqual(response, "done")
        self.assertTrue(
            any(
                "chat request started provider=DeepSeek model=deepseek-v4-pro messages=2" in message
                for message in logs.output
            )
        )
        self.assertTrue(
            any(
                "chat request stopped provider=DeepSeek model=deepseek-v4-pro "
                "used_model=deepseek-v4-pro finish_reason=stop elapsed_s=" in message
                for message in logs.output
            )
        )
        output = "\n".join(logs.output)
        self.assertIn("started_at=", output)
        self.assertIn("stopped_at=", output)
        self.assertNotIn("secret", output)
        self.assertNotIn("hello", output)
        self.assertNotIn("done", output)

    def test_deepseek_reuses_openai_client(self):
        client_init_count = 0
        create_messages = []

        class FakeCompletions:
            def create(self, **kwargs):
                create_messages.append(kwargs["messages"])
                message = types.SimpleNamespace(content=kwargs["messages"][-1]["content"])
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        class FakeOpenAI:
            def __init__(self, *, api_key, base_url):
                nonlocal client_init_count
                client_init_count += 1
                self.chat = types.SimpleNamespace(
                    completions=FakeCompletions(),
                )

        fake_openai = types.SimpleNamespace(OpenAI=FakeOpenAI)
        env = {
            DEEPSEEK_BASE_URL_ENV: "https://example.test/deepseek",
            DEEPSEEK_MODEL_ENV: "deepseek-v4-pro",
            DEEPSEEK_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"openai": fake_openai}),
        ):
            client = Chat("deepseek")
            self.assertEqual(client("first"), "first")
            self.assertEqual(client("second"), "second")

        self.assertEqual(client_init_count, 1)
        self.assertEqual(
            create_messages,
            [
                [
                    {"role": "system", "content": DEEPSEEK_SYSTEM_PROMPT},
                    {"role": "user", "content": "first"},
                ],
                [
                    {"role": "system", "content": DEEPSEEK_SYSTEM_PROMPT},
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "first"},
                    {"role": "user", "content": "second"},
                ],
            ],
        )

    def test_deepseek_async_call_uses_async_openai_client(self):
        sync_init = []
        async_init = []
        async_calls = []

        class FakeCompletions:
            def create(self, **kwargs):
                raise AssertionError("sync client should not handle async_call")

        class FakeAsyncCompletions:
            async def create(self, **kwargs):
                async_calls.append(kwargs)
                content = kwargs["messages"][-1]["content"]
                message = types.SimpleNamespace(content=content)
                choice = types.SimpleNamespace(message=message, finish_reason="stop")
                return types.SimpleNamespace(choices=[choice], model=kwargs["model"])

        class FakeOpenAI:
            def __init__(self, *, api_key, base_url):
                sync_init.append({"api_key": api_key, "base_url": base_url})
                self.chat = types.SimpleNamespace(
                    completions=FakeCompletions(),
                )

        class FakeAsyncOpenAI:
            def __init__(self, *, api_key, base_url):
                async_init.append({"api_key": api_key, "base_url": base_url})
                self.chat = types.SimpleNamespace(
                    completions=FakeAsyncCompletions(),
                )

        async def run_client():
            client = Chat("deepseek")
            self.assertEqual(await client.async_call("first"), "first")
            self.assertEqual(await client.acall("second"), "second")

        fake_openai = types.SimpleNamespace(OpenAI=FakeOpenAI, AsyncOpenAI=FakeAsyncOpenAI)
        env = {
            DEEPSEEK_BASE_URL_ENV: "https://example.test/deepseek",
            DEEPSEEK_MODEL_ENV: "deepseek-v4-pro",
            DEEPSEEK_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"openai": fake_openai}),
        ):
            asyncio.run(run_client())

        self.assertEqual(
            sync_init,
            [{"api_key": "secret", "base_url": "https://example.test/deepseek"}],
        )
        self.assertEqual(
            async_init,
            [{"api_key": "secret", "base_url": "https://example.test/deepseek"}],
        )
        self.assertEqual(
            [call["messages"] for call in async_calls],
            [
                [
                    {"role": "system", "content": DEEPSEEK_SYSTEM_PROMPT},
                    {"role": "user", "content": "first"},
                ],
                [
                    {"role": "system", "content": DEEPSEEK_SYSTEM_PROMPT},
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "first"},
                    {"role": "user", "content": "second"},
                ],
            ],
        )

    def test_deepseek_refresh_starts_new_context(self):
        create_messages = []

        class FakeCompletions:
            def create(self, **kwargs):
                create_messages.append(kwargs["messages"])
                message = types.SimpleNamespace(content=kwargs["messages"][-1]["content"])
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        class FakeOpenAI:
            def __init__(self, *, api_key, base_url):
                self.chat = types.SimpleNamespace(
                    completions=FakeCompletions(),
                )

        fake_openai = types.SimpleNamespace(OpenAI=FakeOpenAI)
        env = {
            DEEPSEEK_BASE_URL_ENV: "https://example.test/deepseek",
            DEEPSEEK_MODEL_ENV: "deepseek-v4-pro",
            DEEPSEEK_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"openai": fake_openai}),
        ):
            client = Chat("deepseek")
            self.assertEqual(client("first"), "first")
            self.assertEqual(client("second", refresh=True), "second")
            client.refresh()
            self.assertEqual(client("third"), "third")

        self.assertEqual(
            create_messages,
            [
                [
                    {"role": "system", "content": DEEPSEEK_SYSTEM_PROMPT},
                    {"role": "user", "content": "first"},
                ],
                [
                    {"role": "system", "content": DEEPSEEK_SYSTEM_PROMPT},
                    {"role": "user", "content": "second"},
                ],
                [
                    {"role": "system", "content": DEEPSEEK_SYSTEM_PROMPT},
                    {"role": "user", "content": "third"},
                ],
            ],
        )

    def test_deepseek_backend_requires_chat_extra(self):
        env = {
            DEEPSEEK_BASE_URL_ENV: "https://example.test/deepseek",
            DEEPSEEK_MODEL_ENV: "deepseek-v4-pro",
            DEEPSEEK_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"openai": None}),
            self.assertRaisesRegex(ImportError, r"anytrain\.chat"),
        ):
            Chat("deepseek")("hello")

    def test_glm_backend_uses_zai_client(self):
        calls = []

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs)
                message = types.SimpleNamespace(content="done")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        class FakeZhipuAiClient:
            def __init__(self, *, api_key, base_url):
                calls.append({"api_key": api_key, "base_url": base_url})
                self.chat = types.SimpleNamespace(
                    completions=FakeCompletions(),
                )

        fake_zai = types.SimpleNamespace(ZhipuAiClient=FakeZhipuAiClient)
        env = {
            GLM_BASE_URL_ENV: "https://example.test/api/paas/v4",
            GLM_MODEL_ENV: "glm-5.2",
            GLM_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"zai": fake_zai}),
        ):
            response = Chat("glm")("hello")

        self.assertEqual(response, "done")
        self.assertEqual(
            calls[0],
            {
                "api_key": "secret",
                "base_url": "https://example.test/api/paas/v4",
            },
        )
        self.assertEqual(
            calls[1],
            {
                "model": "glm-5.2",
                "messages": [{"role": "user", "content": "hello"}],
                "thinking": {"type": "enabled"},
                "max_tokens": GLM_MAX_TOKENS,
                "temperature": GLM_TEMPERATURE,
            },
        )

    def test_glm_chat_passes_image_url_content_parts(self):
        calls = []

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs)
                message = types.SimpleNamespace(content="done")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        class FakeZhipuAiClient:
            def __init__(self, *, api_key, base_url):
                self.chat = types.SimpleNamespace(
                    completions=FakeCompletions(),
                )

        fake_zai = types.SimpleNamespace(ZhipuAiClient=FakeZhipuAiClient)
        env = {
            GLM_BASE_URL_ENV: "https://example.test/api/paas/v4",
            GLM_MODEL_ENV: "glm-5.2",
            GLM_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"zai": fake_zai}),
        ):
            response = Chat("glm")("describe this", images=["https://example.test/cat.png"])

        self.assertEqual(response, "done")
        self.assertEqual(
            calls[0]["messages"],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.test/cat.png"},
                        },
                    ],
                }
            ],
        )

    def test_glm_async_call_and_acall_pass_image_url_content_parts(self):
        create_messages = []

        class FakeCompletions:
            def create(self, **kwargs):
                create_messages.append(kwargs["messages"])
                message = types.SimpleNamespace(content="done")
                choice = types.SimpleNamespace(message=message, finish_reason="stop")
                return types.SimpleNamespace(choices=[choice], model=kwargs["model"])

        class FakeZhipuAiClient:
            def __init__(self, *, api_key, base_url):
                self.chat = types.SimpleNamespace(
                    completions=FakeCompletions(),
                )

        async def run_client():
            client = Chat("glm")
            self.assertEqual(
                await client.async_call("first", images=["https://example.test/one.png"]),
                "done",
            )
            self.assertEqual(
                await client.acall("second", images=["data:image/png;base64,abc"], refresh=True),
                "done",
            )

        fake_zai = types.SimpleNamespace(ZhipuAiClient=FakeZhipuAiClient)
        env = {
            GLM_BASE_URL_ENV: "https://example.test/api/paas/v4",
            GLM_MODEL_ENV: "glm-5.2",
            GLM_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"zai": fake_zai}),
        ):
            asyncio.run(run_client())

        self.assertEqual(
            create_messages,
            [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "first"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.test/one.png"},
                            },
                        ],
                    }
                ],
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "second"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,abc"},
                            },
                        ],
                    }
                ],
            ],
        )

    def test_local_image_input_is_converted_to_data_url(self):
        calls = []
        image_bytes = b"\x89PNG\r\n\x1a\n"

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs)
                message = types.SimpleNamespace(content="done")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        class FakeZhipuAiClient:
            def __init__(self, *, api_key, base_url):
                self.chat = types.SimpleNamespace(
                    completions=FakeCompletions(),
                )

        fake_zai = types.SimpleNamespace(ZhipuAiClient=FakeZhipuAiClient)
        env = {
            GLM_BASE_URL_ENV: "https://example.test/api/paas/v4",
            GLM_MODEL_ENV: "glm-5.2",
            GLM_API_KEY_ENV: "secret",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "pixel.png"
            image_path.write_bytes(image_bytes)
            with (
                patch.dict(os.environ, env, clear=True),
                patch.dict(sys.modules, {"zai": fake_zai}),
            ):
                response = Chat("glm")("describe this", images=[image_path])

        expected_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
        self.assertEqual(response, "done")
        self.assertEqual(
            calls[0]["messages"][0]["content"],
            [
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": expected_url}},
            ],
        )

    def test_glm_reuses_context_until_refresh(self):
        create_messages = []

        class FakeCompletions:
            def create(self, **kwargs):
                create_messages.append(kwargs["messages"])
                message = types.SimpleNamespace(content=kwargs["messages"][-1]["content"])
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        class FakeZhipuAiClient:
            def __init__(self, *, api_key, base_url):
                self.chat = types.SimpleNamespace(
                    completions=FakeCompletions(),
                )

        fake_zai = types.SimpleNamespace(ZhipuAiClient=FakeZhipuAiClient)
        env = {
            GLM_BASE_URL_ENV: "https://example.test/api/paas/v4",
            GLM_MODEL_ENV: "glm-5.2",
            GLM_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"zai": fake_zai}),
        ):
            client = Chat("glm")
            self.assertEqual(client("first"), "first")
            self.assertEqual(client("second"), "second")
            self.assertEqual(client("third", refresh=True), "third")

        self.assertEqual(
            create_messages,
            [
                [{"role": "user", "content": "first"}],
                [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "first"},
                    {"role": "user", "content": "second"},
                ],
                [{"role": "user", "content": "third"}],
            ],
        )

    def test_glm_async_call_uses_sync_client_thread_fallback(self):
        create_messages = []

        class FakeCompletions:
            def create(self, **kwargs):
                create_messages.append(kwargs["messages"])
                content = kwargs["messages"][-1]["content"]
                message = types.SimpleNamespace(content=content)
                choice = types.SimpleNamespace(message=message, finish_reason="stop")
                return types.SimpleNamespace(choices=[choice], model=kwargs["model"])

        class FakeZhipuAiClient:
            def __init__(self, *, api_key, base_url):
                self.chat = types.SimpleNamespace(
                    completions=FakeCompletions(),
                )

        async def run_client():
            client = Chat("glm")
            self.assertEqual(await client.async_call("first"), "first")
            self.assertEqual(await client.acall("second"), "second")

        fake_zai = types.SimpleNamespace(ZhipuAiClient=FakeZhipuAiClient)
        env = {
            GLM_BASE_URL_ENV: "https://example.test/api/paas/v4",
            GLM_MODEL_ENV: "glm-5.2",
            GLM_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"zai": fake_zai}),
        ):
            asyncio.run(run_client())

        self.assertEqual(
            create_messages,
            [
                [{"role": "user", "content": "first"}],
                [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "first"},
                    {"role": "user", "content": "second"},
                ],
            ],
        )

    def test_glm_backend_requires_chat_extra(self):
        env = {
            GLM_BASE_URL_ENV: "https://example.test/api/paas/v4",
            GLM_MODEL_ENV: "glm-5.2",
            GLM_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"zai": None}),
            self.assertRaisesRegex(ImportError, r"anytrain\.chat"),
        ):
            Chat("glm")("hello")
