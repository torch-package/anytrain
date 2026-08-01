import asyncio
import base64
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import anytrain.chat as chat_api
from anytrain import __all__ as anytrain_exports
from anytrain.chat import Chat, ChatResponse
from anytrain.chat._openrouter import resolve_models
from anytrain.chat._provider import (
    DEEPSEEK_REASONING_EFFORT,
    DEEPSEEK_SYSTEM_PROMPT,
    GLM_IMAGE_MODEL,
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
    OPENROUTER_ALLOW_CUSTOM_BASE_URL_ENV,
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_BASE_URL_ENV,
    OPENROUTER_CANDIDATE_LIMIT_ENV,
    OPENROUTER_DEFAULT_BASE_URL,
    OPENROUTER_MODELS_ENV,
)


def _chat_model(model_id, *, prompt, completion, discount=0.0, outputs=("text",), inputs=("text",)):
    return {
        "id": model_id,
        "architecture": {
            "input_modalities": list(inputs),
            "output_modalities": list(outputs),
        },
        "supported_parameters": ["temperature"],
        "pricing": {
            "prompt": str(prompt),
            "completion": str(completion),
            "discount": discount,
        },
    }


def _models_requests(catalog):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": catalog}

    class FakeRequests:
        @staticmethod
        def get(**kwargs):
            FakeRequests.calls.append(kwargs)
            return FakeResponse()

    FakeRequests.calls = []
    return FakeRequests


class ChatTest(unittest.TestCase):
    def test_package_root_does_not_export_chat_client(self):
        self.assertNotIn("Chat", anytrain_exports)

    def test_chat_module_only_exports_chat_client(self):
        self.assertEqual(
            chat_api.__all__,
            [
                "Chat",
                "ChatConfig",
                "ChatResponse",
                "ImageGeneration",
                "ImageOutput",
                "ModelType",
                "config_from_env",
            ],
        )

    def test_model_type_uses_lowercase_string_values(self):
        self.assertEqual(chat_api.ModelType.DEEPSEEK.value, "deepseek")
        self.assertEqual(chat_api.ModelType.GLM.value, "glm")
        self.assertEqual(chat_api.ModelType.OPENROUTER.value, "openrouter")

    def test_chat_request_returns_response_metadata_and_call_returns_text(self):
        class AttrUsage:
            input_tokens = 3
            output_tokens = 4

        class FakeCompletions:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                message = types.SimpleNamespace(content=f"done-{self.calls}")
                choice = types.SimpleNamespace(message=message, finish_reason="stop")
                return types.SimpleNamespace(
                    choices=[choice],
                    model=kwargs["model"],
                    usage=AttrUsage(),
                    _request_id=f"req-{self.calls}",
                )

        completions = FakeCompletions()

        class FakeOpenAI:
            def __init__(self, *, api_key, base_url):
                self.chat = types.SimpleNamespace(completions=completions)

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
            detailed = client.request("hello")
            self.assertIs(client.last_response, detailed)
            plain = client("again")

        self.assertIsInstance(detailed, ChatResponse)
        self.assertEqual(detailed.text, "done-1")
        self.assertEqual(detailed.model, "deepseek-v4-pro")
        self.assertEqual(detailed.usage, {"input_tokens": 3, "output_tokens": 4})
        self.assertEqual(detailed.request_id, "req-1")
        self.assertEqual(detailed.finish_reason, "stop")

        self.assertEqual(plain, "done-2")
        self.assertIsInstance(client.last_response, ChatResponse)
        self.assertEqual(client.last_response.text, "done-2")
        self.assertEqual(client.last_usage, {"input_tokens": 3, "output_tokens": 4})
        self.assertEqual(client.last_request_id, "req-2")
        self.assertEqual(client.last_model, "deepseek-v4-pro")
        self.assertEqual(client.last_finish_reason, "stop")

    def test_config_from_env_uses_deepseek_envs(self):
        env = {
            DEEPSEEK_BASE_URL_ENV: "https://example.test/deepseek",
            DEEPSEEK_MODEL_ENV: "deepseek-chat",
            DEEPSEEK_API_KEY_ENV: "secret",
        }
        with patch.dict(os.environ, env, clear=True):
            config = chat_api.config_from_env("deepseek")

        self.assertEqual(config.base_url, "https://example.test/deepseek")
        self.assertEqual(config.model, "deepseek-chat")
        self.assertEqual(config.models, ("deepseek-chat",))
        self.assertNotIn("secret", repr(config))

    def test_config_from_env_uses_glm_envs(self):
        env = {
            GLM_BASE_URL_ENV: "https://example.test/api/paas/v4",
            GLM_MODEL_ENV: "glm-5.2",
            GLM_API_KEY_ENV: "secret",
        }
        with patch.dict(os.environ, env, clear=True):
            config = chat_api.config_from_env(chat_api.ModelType.GLM)

        self.assertEqual(config.base_url, "https://example.test/api/paas/v4")
        self.assertEqual(config.model, "glm-5.2")
        self.assertEqual(config.models, ("glm-5.2",))

    def test_config_from_env_loads_openrouter_without_selecting_models(self):
        env = {
            OPENROUTER_API_KEY_ENV: "secret",
            OPENROUTER_MODELS_ENV: "openai/gpt-4o-mini,meta-llama/llama-3.2-3b-instruct:free",
            OPENROUTER_CANDIDATE_LIMIT_ENV: "3",
        }
        with patch.dict(os.environ, env, clear=True):
            config = chat_api.config_from_env("openrouter")

        self.assertEqual(config.base_url, OPENROUTER_DEFAULT_BASE_URL)
        self.assertEqual(config.model, "")
        self.assertEqual(config.models, ())
        self.assertEqual(
            config.allowlist,
            ("openai/gpt-4o-mini", "meta-llama/llama-3.2-3b-instruct:free"),
        )
        self.assertEqual(config.candidate_limit, 3)
        self.assertNotIn("secret", repr(config))

    def test_openrouter_models_reject_empty_entries(self):
        env = {
            OPENROUTER_BASE_URL_ENV: "https://openrouter.ai/api/v1",
            OPENROUTER_MODELS_ENV: "openai/gpt-4o-mini,,google/gemini-flash-1.5",
            OPENROUTER_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            self.assertRaisesRegex(ValueError, OPENROUTER_MODELS_ENV),
        ):
            chat_api.config_from_env(chat_api.ModelType.OPENROUTER)

    def test_openrouter_custom_base_url_requires_explicit_opt_in(self):
        env = {
            OPENROUTER_BASE_URL_ENV: "https://evil.test/api/v1",
            OPENROUTER_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            self.assertRaisesRegex(ValueError, OPENROUTER_ALLOW_CUSTOM_BASE_URL_ENV),
        ):
            chat_api.config_from_env(chat_api.ModelType.OPENROUTER)

    def test_openrouter_custom_base_url_accepts_explicit_opt_in(self):
        env = {
            OPENROUTER_BASE_URL_ENV: "https://proxy.test/api/v1",
            OPENROUTER_ALLOW_CUSTOM_BASE_URL_ENV: "1",
            OPENROUTER_API_KEY_ENV: "secret",
        }
        with patch.dict(os.environ, env, clear=True):
            config = chat_api.config_from_env(chat_api.ModelType.OPENROUTER)

        self.assertEqual(config.base_url, "https://proxy.test/api/v1")

    def test_openrouter_custom_base_url_must_use_https_even_when_opted_in(self):
        env = {
            OPENROUTER_BASE_URL_ENV: "http://proxy.test/api/v1",
            OPENROUTER_ALLOW_CUSTOM_BASE_URL_ENV: "1",
            OPENROUTER_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            self.assertRaisesRegex(ValueError, "https"),
        ):
            chat_api.config_from_env(chat_api.ModelType.OPENROUTER)

    def test_openrouter_resolve_models_prefers_free_then_discount_then_price(self):
        catalog = [
            _chat_model("paid/expensive", prompt=0.000002, completion=0.000004),
            _chat_model("paid/cheap", prompt=0.0000001, completion=0.0000002),
            _chat_model(
                "paid/discounted",
                prompt=0.000001,
                completion=0.000002,
                discount=0.5,
            ),
            _chat_model("vendor/model:free", prompt=0, completion=0),
            _chat_model("openrouter/free", prompt=0, completion=0),
            _chat_model(
                "audio/model",
                prompt=0,
                completion=0,
                outputs=("text", "audio"),
            ),
            _chat_model("openrouter/auto", prompt=-1, completion=-1),
        ]
        config = chat_api.ChatConfig(
            base_url="https://openrouter.ai/api/v1",
            model="",
            api_key="secret",
            models=(),
            candidate_limit=4,
        )
        with (
            patch.dict(sys.modules, {"requests": _models_requests(catalog)}),
            self.assertLogs("anytrain.chat", level="INFO") as logs,
        ):
            resolved = resolve_models(config)

        self.assertEqual(
            resolved.models,
            (
                "openrouter/free",
                "vendor/model:free",
                "paid/discounted",
                "paid/cheap",
            ),
        )
        self.assertEqual(resolved.model, "openrouter/free")
        self.assertTrue(any("openrouter selected candidates=" in message for message in logs.output))

    def test_openrouter_resolve_models_ranks_allowlist_from_live_catalog(self):
        catalog = [
            _chat_model("openai/gpt-4o-mini", prompt=0.00000015, completion=0.0000006),
            _chat_model("meta-llama/llama-3.2-3b-instruct:free", prompt=0, completion=0),
            _chat_model("google/gemini-flash-1.5", prompt=0.000000075, completion=0.0000003),
        ]
        config = chat_api.ChatConfig(
            base_url="https://openrouter.ai/api/v1",
            model="",
            api_key="secret",
            models=(),
            allowlist=(
                "openai/gpt-4o-mini",
                "meta-llama/llama-3.2-3b-instruct:free",
                "google/gemini-flash-1.5",
            ),
            candidate_limit=5,
        )
        with patch.dict(sys.modules, {"requests": _models_requests(catalog)}):
            resolved = resolve_models(config)

        self.assertEqual(
            resolved.models,
            (
                "meta-llama/llama-3.2-3b-instruct:free",
                "google/gemini-flash-1.5",
                "openai/gpt-4o-mini",
            ),
        )

    def test_openrouter_resolve_models_keeps_text_image_input_models(self):
        catalog = [
            _chat_model("openai/text-only", prompt=0.00000015, completion=0.0000006),
            _chat_model(
                "openai/vision",
                prompt=0.0000002,
                completion=0.0000008,
                inputs=("text", "image"),
            ),
        ]
        config = chat_api.ChatConfig(
            base_url="https://openrouter.ai/api/v1",
            model="",
            api_key="secret",
            models=(),
            allowlist=("openai/vision",),
            candidate_limit=5,
        )
        with patch.dict(sys.modules, {"requests": _models_requests(catalog)}):
            resolved = resolve_models(config)

        self.assertEqual(resolved.models, ("openai/vision",))
        self.assertEqual(resolved.model, "openai/vision")

    def test_missing_env_fails_explicitly(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(ValueError, DEEPSEEK_BASE_URL_ENV),
        ):
            chat_api.config_from_env(chat_api.ModelType.DEEPSEEK)

    def test_empty_env_fails_explicitly(self):
        env = {
            DEEPSEEK_BASE_URL_ENV: "",
            DEEPSEEK_MODEL_ENV: "deepseek-chat",
            DEEPSEEK_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            self.assertRaisesRegex(ValueError, DEEPSEEK_BASE_URL_ENV),
        ):
            chat_api.config_from_env(chat_api.ModelType.DEEPSEEK)

    def test_unknown_model_type_fails(self):
        with self.assertRaises(ValueError):
            chat_api.config_from_env("unknown")

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
                "chat request started provider=DeepSeek model=deepseek-v4-pro messages=2"
                in message
                for message in logs.output
            )
        )
        self.assertTrue(
            any(
                "chat request stopped provider=DeepSeek model=deepseek-v4-pro "
                "used_model=deepseek-v4-pro finish_reason=stop elapsed_s="
                in message
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

    def test_chat_rejects_empty_prompt_before_backend(self):
        class FakeZhipuAiClient:
            def __init__(self, *, api_key, base_url):
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(),
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
            with self.assertRaisesRegex(ValueError, "prompt"):
                client("")
            with self.assertRaisesRegex(ValueError, "images"):
                client("hello", images=[])

    def test_glm_image_generation_uses_sync_http_api(self):
        calls = []

        class FakeResponse:
            def raise_for_status(self):
                calls.append("raise_for_status")

            def json(self):
                return {
                    "created": 123,
                    "data": [
                        {
                            "url": "https://example.test/image.png",
                            "revised_prompt": "cat",
                        }
                    ],
                }

        class FakeRequests:
            @staticmethod
            def post(**kwargs):
                calls.append(kwargs)
                return FakeResponse()

        class FakeZhipuAiClient:
            def __init__(self, *, api_key, base_url):
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(),
                )

        fake_zai = types.SimpleNamespace(ZhipuAiClient=FakeZhipuAiClient)
        env = {
            GLM_BASE_URL_ENV: "https://example.test/api/paas/v4/",
            GLM_MODEL_ENV: "glm-5.2",
            GLM_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"requests": FakeRequests, "zai": fake_zai}),
        ):
            result = Chat("glm").image("draw a cat", size="1024x1024", timeout=3.0)

        self.assertEqual(
            calls[0],
            {
                "url": "https://example.test/api/paas/v4/images/generations",
                "json": {
                    "model": GLM_IMAGE_MODEL,
                    "prompt": "draw a cat",
                    "size": "1024x1024",
                },
                "headers": {
                    "Authorization": "Bearer secret",
                    "Content-Type": "application/json",
                },
                "timeout": 3.0,
            },
        )
        self.assertEqual(calls[1], "raise_for_status")
        self.assertEqual(result.model, GLM_IMAGE_MODEL)
        self.assertEqual(result.prompt, "draw a cat")
        self.assertEqual(result.size, "1024x1024")
        self.assertEqual(result.url, "https://example.test/image.png")
        self.assertIsNone(result.b64_json)
        self.assertEqual(result.data[0].revised_prompt, "cat")
        self.assertEqual(result.raw["created"], 123)

    def test_glm_image_generation_accepts_custom_model_and_b64_output(self):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"data": [{"b64_json": "encoded"}]}

        class FakeRequests:
            @staticmethod
            def post(**kwargs):
                return FakeResponse()

        class FakeZhipuAiClient:
            def __init__(self, *, api_key, base_url):
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(),
                )

        fake_zai = types.SimpleNamespace(ZhipuAiClient=FakeZhipuAiClient)
        env = {
            GLM_BASE_URL_ENV: "https://example.test/api/paas/v4",
            GLM_MODEL_ENV: "glm-5.2",
            GLM_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"requests": FakeRequests, "zai": fake_zai}),
        ):
            result = Chat("glm").image("draw a cat", model="glm-image-test")

        self.assertEqual(result.model, "glm-image-test")
        self.assertIsNone(result.url)
        self.assertEqual(result.b64_json, "encoded")

    def test_deepseek_image_generation_is_not_supported(self):
        class FakeOpenAI:
            def __init__(self, *, api_key, base_url):
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(),
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
                client.image("draw a cat")

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
                "openai/gpt-4o-mini,meta-llama/llama-3.2-3b-instruct:free,"
                "google/gemini-flash-1.5"
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
                "openai/gpt-4o-mini,meta-llama/llama-3.2-3b-instruct:free,"
                "google/gemini-flash-1.5"
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
                "openrouter using model=google/gemini-flash-1.5 finish_reason=stop"
                in message
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

    def test_image_generation_rejects_empty_fields_before_http(self):
        class FakeZhipuAiClient:
            def __init__(self, *, api_key, base_url):
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(),
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
            with self.assertRaisesRegex(ValueError, "prompt"):
                client.image("")
            with self.assertRaisesRegex(ValueError, "size"):
                client.image("draw a cat", size="")
            with self.assertRaisesRegex(ValueError, "model"):
                client.image("draw a cat", model="")

    def test_glm_image_generation_rejects_malformed_response(self):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"data": [{"revised_prompt": "cat"}]}

        class FakeRequests:
            @staticmethod
            def post(**kwargs):
                return FakeResponse()

        class FakeZhipuAiClient:
            def __init__(self, *, api_key, base_url):
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(),
                )

        fake_zai = types.SimpleNamespace(ZhipuAiClient=FakeZhipuAiClient)
        env = {
            GLM_BASE_URL_ENV: "https://example.test/api/paas/v4",
            GLM_MODEL_ENV: "glm-5.2",
            GLM_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"requests": FakeRequests, "zai": fake_zai}),
            self.assertRaisesRegex(TypeError, "url or b64_json"),
        ):
            Chat("glm").image("draw a cat")

    def test_glm_image_generation_requires_requests_extra(self):
        class FakeZhipuAiClient:
            def __init__(self, *, api_key, base_url):
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(),
                )

        fake_zai = types.SimpleNamespace(ZhipuAiClient=FakeZhipuAiClient)
        env = {
            GLM_BASE_URL_ENV: "https://example.test/api/paas/v4",
            GLM_MODEL_ENV: "glm-5.2",
            GLM_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"requests": None, "zai": fake_zai}),
            self.assertRaisesRegex(ImportError, r"anytrain\.chat"),
        ):
            Chat("glm").image("draw a cat")


if __name__ == "__main__":
    unittest.main()
