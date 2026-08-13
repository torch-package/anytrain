from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch

from anytrain.tts import (
    TTSAcousticEngine,
    TTSEngineCapabilities,
    TTSInferenceConfig,
    TTSOptions,
    TTSTransformerEngine,
)
from anytrain.tts._environment import QWEN_TORCH_ENVIRONMENT, VLLM_OMNI_ENVIRONMENT
from anytrain.tts._omni import generate_audio, load_omni
from anytrain.tts.cosyvoice import CosyVoiceTTSConfig
from anytrain.tts.moss import MossTTS, MossTTSConfig
from anytrain.tts.moss._omni import build_prompts as build_moss_prompts
from anytrain.tts.qwen import QwenCustomVoiceTTS, QwenCustomVoiceTTSConfig
from anytrain.tts.qwen._omni import QwenOmniPromptBuilder


def _installed(versions: dict[str, str]):
    def lookup(distribution: str) -> str:
        try:
            return versions[distribution]
        except KeyError:
            from importlib.metadata import PackageNotFoundError

            raise PackageNotFoundError(distribution) from None

    return lookup


@pytest.mark.parametrize(
    ("contract", "versions"),
    (
        (
            QWEN_TORCH_ENVIRONMENT,
            {"qwen-tts": "0.1.1", "transformers": "4.57.3"},
        ),
        (
            VLLM_OMNI_ENVIRONMENT,
            {
                "transformers": "5.5.3",
                "vllm": "0.26.0",
                "vllm-omni": "0.26.0",
            },
        ),
    ),
)
def test_tts_environment_contracts_accept_only_their_backend_family(
    contract: Any,
    versions: dict[str, str],
) -> None:
    with patch("anytrain.tts._environment.version", side_effect=_installed(versions)):
        contract.validate()


@pytest.mark.parametrize(
    ("contract", "versions"),
    (
        (
            QWEN_TORCH_ENVIRONMENT,
            {
                "qwen-tts": "0.1.1",
                "transformers": "4.57.3",
                "vllm": "0.26.0",
            },
        ),
        (
            VLLM_OMNI_ENVIRONMENT,
            {
                "qwen-tts": "0.1.1",
                "transformers": "5.5.3",
                "vllm": "0.26.0",
                "vllm-omni": "0.26.0",
            },
        ),
    ),
)
def test_tts_environment_contracts_reject_mixed_qwen_and_vllm_families(
    contract: Any,
    versions: dict[str, str],
) -> None:
    with (
        patch("anytrain.tts._environment.version", side_effect=_installed(versions)),
        pytest.raises(RuntimeError, match="separate environments and process boundaries"),
    ):
        contract.validate()


@pytest.mark.parametrize(
    ("contract", "versions", "message", "error"),
    (
        (
            QWEN_TORCH_ENVIRONMENT,
            {"qwen-tts": "0.1.1", "transformers": "5.5.3"},
            "transformers==4.57.3",
            RuntimeError,
        ),
        (
            VLLM_OMNI_ENVIRONMENT,
            {"transformers": "5.5.3", "vllm": "0.26.0"},
            "vllm-omni",
            ImportError,
        ),
    ),
)
def test_tts_environment_contracts_report_version_and_missing_dependencies(
    contract: Any,
    versions: dict[str, str],
    message: str,
    error: type[Exception],
) -> None:
    with (
        patch("anytrain.tts._environment.version", side_effect=_installed(versions)),
        pytest.raises(error, match=message),
    ):
        contract.validate()


def test_engine_config_validates_capabilities_and_excludes_concurrency_from_hash() -> None:
    capabilities = TTSEngineCapabilities(
        transformer_engines=frozenset(
            (TTSTransformerEngine.TORCH, TTSTransformerEngine.VLLM_OMNI)
        ),
        acoustic_engines=frozenset((TTSAcousticEngine.TORCH,)),
    )
    first = TTSInferenceConfig(
        transformer_engine=TTSTransformerEngine.VLLM_OMNI,
        max_concurrency=1,
    )
    replica = TTSInferenceConfig(
        transformer_engine=TTSTransformerEngine.VLLM_OMNI,
        max_concurrency=8,
    )

    capabilities.validate(first, "backend")
    assert first.hash_dict() == replica.hash_dict()
    with pytest.raises(ValueError, match="acoustic engine 'tensorrt'"):
        capabilities.validate(
            TTSInferenceConfig(acoustic_engine=TTSAcousticEngine.TENSORRT),
            "backend",
        )
    with pytest.raises(ValueError, match="requires a vLLM engine"):
        capabilities.validate(TTSInferenceConfig(max_concurrency=2), "backend")


def test_backend_engine_capability_matrices_reject_false_acceleration_paths() -> None:
    with pytest.raises(ValueError, match="vllm_omni"):
        CosyVoiceTTSConfig(
            inference=TTSInferenceConfig(
                transformer_engine=TTSTransformerEngine.VLLM_OMNI
            )
        )
    with pytest.raises(ValueError, match="tensorrt"):
        QwenCustomVoiceTTSConfig(
            inference=TTSInferenceConfig(acoustic_engine=TTSAcousticEngine.TENSORRT)
        )
    with pytest.raises(ValueError, match="tensorrt"):
        MossTTSConfig(
            inference=TTSInferenceConfig(acoustic_engine=TTSAcousticEngine.TENSORRT)
        )


def test_cosyvoice_typed_config_maps_real_upstream_flags() -> None:
    config = CosyVoiceTTSConfig(
        inference=TTSInferenceConfig(
            transformer_engine=TTSTransformerEngine.VLLM,
            acoustic_engine=TTSAcousticEngine.TENSORRT,
            max_concurrency=4,
        ),
        load_kwargs={"fp16": False},
    )

    assert config.resolved_inference().max_concurrency == 4
    assert config.hash_dict()["inference"] == {
        "transformer_engine": "vllm",
        "acoustic_engine": "tensorrt",
    }
    with pytest.raises(ValueError, match="conflicts"):
        CosyVoiceTTSConfig(
            inference=TTSInferenceConfig(
                transformer_engine=TTSTransformerEngine.VLLM
            ),
            load_kwargs={"load_vllm": False},
        )
    with pytest.raises(ValueError, match="trt_concurrent conflicts"):
        CosyVoiceTTSConfig(
            inference=TTSInferenceConfig(
                acoustic_engine=TTSAcousticEngine.TENSORRT,
                max_concurrency=4,
            ),
            load_kwargs={"trt_concurrent": 2},
        )


class _FakeOmni:
    def __init__(self) -> None:
        self.default_sampling_params_list = [
            SimpleNamespace(max_tokens=2048, temperature=0.9, top_p=1.0, seed=0),
            SimpleNamespace(max_tokens=65536, temperature=0.0, top_p=1.0, seed=0),
        ]
        self.calls: list[tuple[object, object, bool]] = []

    def generate(
        self,
        prompts: object,
        sampling_params_list: object = None,
        *,
        use_tqdm: bool = False,
    ) -> list[object]:
        self.calls.append((prompts, sampling_params_list, use_tqdm))
        count = len(cast(Any, prompts))
        outputs = [
            SimpleNamespace(
                request_id=f"{index}_request",
                multimodal_output={
                    "audio": torch.tensor(
                        [float(index * 2 + 1), float(index * 2 + 2)]
                    ),
                    "sr": 16000,
                },
            )
            for index in range(count)
        ]
        if count > 1:
            outputs[0].multimodal_output = {
                "audio": [torch.tensor([1.0]), torch.tensor([2.0])],
                "sr": torch.tensor(16000),
            }
        return list(reversed(outputs))


def test_omni_output_normalization_preserves_request_order_and_sampling() -> None:
    engine = _FakeOmni()
    outputs = generate_audio(
        cast(Any, engine),
        [{"prompt_token_ids": [0]}, {"prompt_token_ids": [0]}],
        options=TTSOptions(max_new_tokens=17, temperature=0.7, top_p=0.8, seed=7),
        sample_rate=24000,
        backend="fake",
    )

    assert torch.equal(outputs[0].waveform, torch.tensor([[1.0, 2.0]]))
    assert torch.equal(outputs[1].waveform, torch.tensor([[3.0, 4.0]]))
    assert [output.sample_rate for output in outputs] == [16000, 16000]
    params = cast(Any, engine.calls[0][1])
    assert params[0].max_tokens == 17
    assert params[0].temperature == 0.7
    assert params[0].top_p == 0.8
    assert params[0].seed == 7


def test_omni_max_concurrency_chunks_submissions_without_changing_order() -> None:
    engine = _FakeOmni()
    outputs = generate_audio(
        cast(Any, engine),
        [{"prompt_token_ids": [0]} for _ in range(5)],
        options=TTSOptions(),
        sample_rate=24000,
        backend="fake",
        max_concurrency=2,
    )

    assert [len(cast(Any, call[0])) for call in engine.calls] == [2, 2, 1]
    assert len(outputs) == 5


def test_omni_rejects_unknown_runtime_options() -> None:
    with pytest.raises(TypeError, match="unknown"):
        generate_audio(
            cast(Any, _FakeOmni()),
            [{"prompt_token_ids": [0]}],
            options=TTSOptions(extra={"unknown": True}),
            sample_rate=24000,
            backend="fake",
        )


class _FakeQwenEstimator:
    calls: list[dict[str, Any]] = []

    @staticmethod
    def estimate_prompt_len_from_additional_information(
        additional_information: dict[str, Any] | None,
        **kwargs: object,
    ) -> int:
        assert additional_information is not None
        _FakeQwenEstimator.calls.append(additional_information)
        tokenize = cast(Any, kwargs["tokenize_prompt"])
        assert tokenize("text") == [1, 2]
        return 3


class _FakeQwenTokenizer:
    def __call__(self, value: str, *, padding: bool) -> dict[str, list[int]]:
        assert value == "text"
        assert padding is False
        return {"input_ids": [1, 2]}


def test_qwen_omni_builder_uses_exact_placeholder_contract() -> None:
    _FakeQwenEstimator.calls.clear()
    builder = QwenOmniPromptBuilder(
        tokenizer=_FakeQwenTokenizer(),
        talker_config=SimpleNamespace(
            spk_id={"vivian": 0},
            codec_language_id={"english": 1},
            spk_is_dialect={},
        ),
        estimator=cast(Any, _FakeQwenEstimator),
    )

    prompts = builder.build(
        ["hello"],
        ["Vivian"],
        ["English"],
        ["happy"],
        TTSOptions(max_new_tokens=19),
    )

    assert prompts[0]["prompt_token_ids"] == [0, 0, 0]
    info = cast(dict[str, list[object]], prompts[0]["additional_information"])
    assert info == {
        "task_type": ["CustomVoice"],
        "text": ["hello"],
        "language": ["English"],
        "speaker": ["Vivian"],
        "instruct": ["happy"],
        "max_new_tokens": [19],
    }


class _FakeMossProcessor:
    model_config = SimpleNamespace(sampling_rate=24000)

    def __init__(self) -> None:
        self.calls = 0

    def build_user_message(self, **kwargs: object) -> object:
        return kwargs

    def __call__(self, value: object, **kwargs: object) -> object:
        assert kwargs == {"mode": "generation"}
        assert len(cast(Any, value)) == 1
        self.calls += 1
        offset = self.calls * 100
        return {
            "input_ids": torch.tensor(
                [[[self.calls * 10 + 1, offset + 1, offset + 2],
                  [self.calls * 10 + 2, offset + 3, offset + 4]]]
            )
        }

    def decode(self, value: object) -> object:
        return value


def test_moss_omni_builder_keeps_text_and_reference_code_axes() -> None:
    prompts = build_moss_prompts(
        cast(Any, _FakeMossProcessor()),
        ["hello", "world"],
        TTSOptions(language="English"),
        [None, None],
    )

    assert prompts[0]["prompt_token_ids"] == [11, 12]
    assert prompts[1]["prompt_token_ids"] == [21, 22]
    first_info = cast(dict[str, dict[str, torch.Tensor]], prompts[0]["additional_information"])
    assert torch.equal(
        first_info["codes"]["ref"],
        torch.tensor([[101, 102], [103, 104]]),
    )


def test_vllm_omni_backends_use_real_engine_objects() -> None:
    qwen_engine = _FakeOmni()
    qwen_builder = QwenOmniPromptBuilder(
        tokenizer=_FakeQwenTokenizer(),
        talker_config=SimpleNamespace(spk_id={"vivian": 0}),
        estimator=cast(Any, _FakeQwenEstimator),
    )
    qwen = QwenCustomVoiceTTS(
        model=qwen_engine,
        config=QwenCustomVoiceTTSConfig(
            speaker="vivian",
            inference=TTSInferenceConfig(
                transformer_engine=TTSTransformerEngine.VLLM_OMNI
            ),
        ),
        model_loaded_from="vllm_omni",
        omni_prompt_builder=qwen_builder,
    )
    qwen_output = qwen.synthesize("hello", TTSOptions(seed=7))
    assert not isinstance(qwen_output, list)
    assert qwen_output.meta["transformer_engine"] == "vllm_omni"

    moss = MossTTS(
        model=_FakeOmni(),
        processor=cast(Any, _FakeMossProcessor()),
        config=MossTTSConfig(
            inference=TTSInferenceConfig(
                transformer_engine=TTSTransformerEngine.VLLM_OMNI
            )
        ),
        model_loaded_from="vllm_omni",
    )
    moss_outputs = moss.synthesize(["hello", "world"])
    assert isinstance(moss_outputs, list)
    assert len(moss_outputs) == 2


def test_backend_hashes_keep_engine_but_ignore_concurrency() -> None:
    first = QwenCustomVoiceTTS(
        model=object(),
        config=QwenCustomVoiceTTSConfig(
            inference=TTSInferenceConfig(
                transformer_engine=TTSTransformerEngine.VLLM_OMNI,
                max_concurrency=1,
            ),
            load_kwargs={"cache_dir": "first", "device_map": "cuda:0"},
        ),
    )
    replica = QwenCustomVoiceTTS(
        model=object(),
        config=QwenCustomVoiceTTSConfig(
            inference=TTSInferenceConfig(
                transformer_engine=TTSTransformerEngine.VLLM_OMNI,
                max_concurrency=8,
            ),
            load_kwargs={"cache_dir": "second", "device_map": "cuda:1"},
        ),
    )
    torch_backend = QwenCustomVoiceTTS(model=object())
    moss_first = MossTTSConfig(
        inference=TTSInferenceConfig(
            transformer_engine=TTSTransformerEngine.VLLM_OMNI,
            max_concurrency=1,
        ),
        load_kwargs={"cache_dir": "first", "devices": "0"},
    )
    moss_replica = MossTTSConfig(
        inference=TTSInferenceConfig(
            transformer_engine=TTSTransformerEngine.VLLM_OMNI,
            max_concurrency=8,
        ),
        load_kwargs={"cache_dir": "second", "devices": "1"},
    )

    assert first.config_hash() == replica.config_hash()
    assert first.config_hash() != torch_backend.config_hash()
    assert moss_first.hash_dict() == moss_replica.hash_dict()


def test_qwen_omni_speaker_validation_is_case_insensitive() -> None:
    engine = _FakeOmni()
    qwen = QwenCustomVoiceTTS(
        model=engine,
        config=QwenCustomVoiceTTSConfig(
            speaker="Vivian",
            inference=TTSInferenceConfig(
                transformer_engine=TTSTransformerEngine.VLLM_OMNI
            ),
        ),
        model_loaded_from="vllm_omni",
        omni_prompt_builder=QwenOmniPromptBuilder(
            tokenizer=_FakeQwenTokenizer(),
            talker_config=SimpleNamespace(spk_id={"vivian": 0}),
            estimator=cast(Any, _FakeQwenEstimator),
        ),
    )

    output = qwen.synthesize("hello")
    assert not isinstance(output, list)


def test_qwen_from_pretrained_constructs_omni_and_exact_prompt_builder() -> None:
    engine = _FakeOmni()
    builder = QwenOmniPromptBuilder(
        tokenizer=_FakeQwenTokenizer(),
        talker_config=SimpleNamespace(spk_id={"vivian": 0}),
        estimator=cast(Any, _FakeQwenEstimator),
    )
    with (
        patch("anytrain.tts.qwen.tts.load_omni", return_value=engine) as load,
        patch.object(
            QwenOmniPromptBuilder,
            "from_pretrained",
            return_value=builder,
        ) as prompt_builder,
    ):
        backend = QwenCustomVoiceTTS.from_pretrained(
            "fake-qwen",
            inference=TTSInferenceConfig(
                transformer_engine=TTSTransformerEngine.VLLM_OMNI
            ),
            omni_kwargs={"stage_init_timeout": 600},
            cache_dir="storage/hf",
            revision="revision",
            local_files_only=True,
        )

    assert backend.model is engine
    assert backend.model_loaded_from == "vllm_omni"
    load.assert_called_once_with(
        "fake-qwen",
        {
            "stage_init_timeout": 600,
            "revision": "revision",
            "dtype": "bfloat16",
        },
    )
    prompt_builder.assert_called_once_with(
        "fake-qwen",
        cache_dir="storage/hf",
        revision="revision",
        local_files_only=True,
    )


def test_moss_from_pretrained_constructs_omni_and_processor_only() -> None:
    engine = _FakeOmni()
    processor = _FakeMossProcessor()
    with (
        patch("anytrain.tts.moss.tts.load_omni", return_value=engine) as load,
        patch(
            "anytrain.tts.moss.tts.load_pretrained_processor",
            return_value=processor,
        ) as load_processor,
    ):
        backend = MossTTS.from_pretrained(
            "fake-moss",
            codec_model="fake-codec",
            inference=TTSInferenceConfig(
                transformer_engine=TTSTransformerEngine.VLLM_OMNI
            ),
            omni_kwargs={"stage_init_timeout": 600},
            cache_dir="storage/hf",
            revision="revision",
            local_files_only=True,
        )

    assert backend.model is engine
    assert backend.model_loaded_from == "vllm_omni"
    assert backend.config.trust_remote_code is True
    load.assert_called_once_with(
        "fake-moss",
        {
            "stage_init_timeout": 600,
            "revision": "revision",
        },
    )
    load_processor.assert_called_once_with(
        "fake-moss",
        cache_dir="storage/hf",
        codec_model="fake-codec",
        local_files_only=True,
        revision="revision",
        trust_remote_code=True,
    )


def test_load_omni_reports_missing_optional_dependency() -> None:
    with (
        patch("anytrain.tts._omni.VLLM_OMNI_ENVIRONMENT"),
        patch.dict("sys.modules", {"vllm_omni": None}),
        pytest.raises(ImportError, match="vllm-omni"),
    ):
        load_omni("fake", {})


def test_load_omni_validates_the_environment_before_import() -> None:
    with (
        patch("anytrain.tts._omni.VLLM_OMNI_ENVIRONMENT") as environment,
        pytest.raises(RuntimeError, match="separate environments"),
    ):
        environment.validate.side_effect = RuntimeError("use separate environments")
        load_omni("fake", {})
