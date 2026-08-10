import os
import sys
import types
import unittest
from unittest.mock import patch

import anytrain.chat as chat_api
from anytrain import __all__ as anytrain_exports
from anytrain.chat import Chat, ChatResponse
from anytrain.chat.config import (
    DEEPSEEK_API_KEY_ENV,
    DEEPSEEK_BASE_URL_ENV,
    DEEPSEEK_MODEL_ENV,
    GLM_API_KEY_ENV,
    GLM_BASE_URL_ENV,
    GLM_MODEL_ENV,
)


class ChatApiTest(unittest.TestCase):
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
        assert client.last_response is not None
        self.assertEqual(client.last_response.text, "done-2")
        self.assertEqual(client.last_usage, {"input_tokens": 3, "output_tokens": 4})
        self.assertEqual(client.last_request_id, "req-2")
        self.assertEqual(client.last_model, "deepseek-v4-pro")
        self.assertEqual(client.last_finish_reason, "stop")

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
