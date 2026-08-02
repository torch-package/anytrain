import os
import sys
import unittest
from unittest.mock import patch

from _chat_fakes import chat_model as _chat_model
from _chat_fakes import models_requests as _models_requests

import anytrain.chat as chat_api
from anytrain.chat._openrouter import resolve_models
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


class ChatConfigTest(unittest.TestCase):
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
        self.assertTrue(
            any("openrouter selected candidates=" in message for message in logs.output)
        )

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
