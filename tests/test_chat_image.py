import os
import sys
import types
import unittest
from unittest.mock import patch

from anytrain.chat import Chat
from anytrain.chat._provider import GLM_IMAGE_MODEL
from anytrain.chat.config import (
    DEEPSEEK_API_KEY_ENV,
    DEEPSEEK_BASE_URL_ENV,
    DEEPSEEK_MODEL_ENV,
    GLM_API_KEY_ENV,
    GLM_BASE_URL_ENV,
    GLM_MODEL_ENV,
)


class ChatImageGenerationTest(unittest.TestCase):
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
