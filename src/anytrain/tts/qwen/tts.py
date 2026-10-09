"""Qwen3-TTS CustomVoice inference adapter."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict, Unpack, cast, overload

import numpy as np
import torch
from torch import Tensor

from anytrain.tts import (
    AudioReference,
    EncodedAudioReference,
    PathAudioReference,
    TTSAcousticEngine,
    TTSCapabilities,
    TTSEngineCapabilities,
    TTSInferenceConfig,
    TTSOptions,
    TTSOutput,
    TTSTransformerEngine,
    TTSVoiceMode,
    WaveformReference,
    validate_text,
    waveform_duration,
)
from anytrain.tts._common import config_hash, hashable_mapping, positive_int, validate_mapping
from anytrain.tts._omni import OmniEngine, generate_audio, load_omni
from anytrain.tts.environments import QWEN_TORCH_ENVIRONMENT

from ._omni import QwenOmniPromptBuilder

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_SAMPLE_RATE = 24000
_PLACEMENT_LOAD_KWARGS = frozenset(
    (
        "batch_timeout",
        "cache_dir",
        "deploy_config",
        "device",
        "device_map",
        "devices",
        "gpu_memory_utilization",
        "init_timeout",
        "log_stats",
        "max_num_batched_tokens",
        "max_num_seqs",
        "stage_init_timeout",
        "local_files_only",
    )
)
ENGINE_CAPABILITIES = TTSEngineCapabilities(
    transformer_engines=frozenset(
        (TTSTransformerEngine.TORCH, TTSTransformerEngine.VLLM_OMNI)
    ),
    acoustic_engines=frozenset((TTSAcousticEngine.TORCH,)),
)


class QwenLoadKwargs(TypedDict, total=False):
    attn_implementation: object
    device: object
    device_map: object
    dtype: object
    torch_dtype: object


@dataclass(frozen=True)
class QwenCustomVoiceTTSConfig:
    model: str = DEFAULT_MODEL
    sample_rate: int = DEFAULT_SAMPLE_RATE
    language: str = "Auto"
    speaker: str | None = None
    instruct: str | None = None
    inference: TTSInferenceConfig = field(default_factory=TTSInferenceConfig)
    load_kwargs: Mapping[str, object] = field(default_factory=dict)
    runtime_kwargs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must be non-empty.")
        positive_int(self.sample_rate, "sample_rate")
        if not isinstance(self.language, str) or not self.language:
            raise ValueError("language must be a non-empty string.")
        if self.speaker is not None and not self.speaker:
            raise ValueError("speaker must be non-empty when set.")
        validate_mapping(self.load_kwargs, "load_kwargs")
        validate_mapping(self.runtime_kwargs, "runtime_kwargs")
        ENGINE_CAPABILITIES.validate(self.inference, "Qwen3-TTS")

    def options(self) -> TTSOptions:
        kwargs: dict[str, object] = {
            "language": self.language,
            "sample_rate": self.sample_rate,
            "extra": self.runtime_kwargs,
        }
        if self.speaker is not None:
            kwargs["speaker"] = self.speaker
        return TTSOptions(**cast(dict[str, Any], kwargs))

    def hash_dict(self) -> dict[str, object]:
        semantic_load_kwargs = {
            key: value
            for key, value in self.load_kwargs.items()
            if key not in _PLACEMENT_LOAD_KWARGS
        }
        return {
            "model": self.model,
            "sample_rate": self.sample_rate,
            "language": self.language,
            "speaker": self.speaker,
            "instruct": self.instruct,
            "inference": self.inference.hash_dict(),
            "load_kwargs": hashable_mapping(semantic_load_kwargs),
            "runtime_kwargs": hashable_mapping(self.runtime_kwargs),
        }


@dataclass
class QwenCustomVoiceTTS:
    model: Any
    config: QwenCustomVoiceTTSConfig = field(default_factory=QwenCustomVoiceTTSConfig)
    model_loaded_from: str = "custom"
    omni_prompt_builder: QwenOmniPromptBuilder | None = None

    name: str = "qwen3-tts-customvoice"

    @classmethod
    def from_pretrained(
        cls,
        model: str | Path = DEFAULT_MODEL,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        language: str = "Auto",
        speaker: str | None = None,
        instruct: str | None = None,
        inference: TTSInferenceConfig | None = None,
        omni_kwargs: Mapping[str, object] | None = None,
        cache_dir: str | Path | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        runtime_kwargs: Mapping[str, object] | None = None,
        **load_kwargs: Unpack[QwenLoadKwargs],
    ) -> QwenCustomVoiceTTS:
        selected = inference or TTSInferenceConfig()
        ENGINE_CAPABILITIES.validate(selected, "Qwen3-TTS")
        if not isinstance(local_files_only, bool):
            raise TypeError("local_files_only must be a bool.")
        if selected.transformer_engine is TTSTransformerEngine.VLLM_OMNI:
            if load_kwargs:
                raise ValueError(
                    "Qwen3-TTS PyTorch load kwargs cannot be used with vLLM-Omni; "
                    "pass engine options through omni_kwargs."
                )
            runtime = dict(runtime_kwargs or {})
            engine_kwargs = dict(omni_kwargs or {})
            if revision is not None:
                configured_revision = engine_kwargs.get("revision")
                if configured_revision is not None and configured_revision != revision:
                    raise ValueError(
                        "Qwen3-TTS revision conflicts with omni_kwargs['revision']."
                    )
                engine_kwargs["revision"] = revision
            _validate_qwen_dtype(engine_kwargs.get("dtype"))
            engine_kwargs.setdefault("dtype", "bfloat16")
            config = QwenCustomVoiceTTSConfig(
                model=str(model),
                sample_rate=sample_rate,
                language=language,
                speaker=speaker,
                instruct=instruct,
                inference=selected,
                load_kwargs=engine_kwargs,
                runtime_kwargs=runtime,
            )
            loaded = load_omni(str(model), engine_kwargs)
            builder = QwenOmniPromptBuilder.from_pretrained(
                str(model),
                cache_dir=None if cache_dir is None else str(cache_dir),
                revision=revision,
                local_files_only=local_files_only,
            )
            return cls(
                loaded,
                config=config,
                model_loaded_from="vllm_omni",
                omni_prompt_builder=builder,
            )
        if omni_kwargs:
            raise ValueError("omni_kwargs require transformer_engine='vllm_omni'.")
        effective_load_kwargs: dict[str, object] = dict(load_kwargs)
        _validate_qwen_dtype(
            effective_load_kwargs.get(
                "torch_dtype",
                effective_load_kwargs.get("dtype"),
            )
        )
        effective_load_kwargs.setdefault("torch_dtype", torch.bfloat16)
        if cache_dir is not None:
            effective_load_kwargs["cache_dir"] = str(cache_dir)
        if revision is not None:
            effective_load_kwargs["revision"] = revision
        if local_files_only:
            effective_load_kwargs["local_files_only"] = True
        QWEN_TORCH_ENVIRONMENT.validate()
        try:
            from qwen_tts import Qwen3TTSModel  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise ImportError(
                "anytrain.tts.qwen requires Qwen3-TTS. "
                "Install the PyTorch backend with "
                "`python -m pip install 'anytrain[qwen-tts]'`."
            ) from exc

        runtime = dict(runtime_kwargs or {})
        config = QwenCustomVoiceTTSConfig(
            model=str(model),
            sample_rate=sample_rate,
            language=language,
            speaker=speaker,
            instruct=instruct,
            inference=selected,
            load_kwargs=effective_load_kwargs,
            runtime_kwargs=runtime,
        )
        try:
            loaded = Qwen3TTSModel.from_pretrained(str(model), **effective_load_kwargs)
        except Exception as exc:
            raise RuntimeError(f"failed to load Qwen3-TTS checkpoint {model!s}") from exc
        return cls(loaded, config=config, model_loaded_from="qwen_tts")

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def config_hash(self) -> str:
        return config_hash(self.config.hash_dict())

    @property
    def capabilities(self) -> TTSCapabilities:
        speakers = self.supported_speakers()
        return TTSCapabilities(
            voice_modes=frozenset((TTSVoiceMode.SPEAKER_ID,)),
            speaker_ids=speakers or None,
            engines=ENGINE_CAPABILITIES,
        )

    def supported_speakers(self) -> tuple[str, ...]:
        if self.omni_prompt_builder is not None:
            return self.omni_prompt_builder.supported_speakers()
        getter = getattr(self.model, "get_supported_speakers", None)
        if not callable(getter):
            return ()
        speakers = tuple(cast(Iterable[str], getter()))
        if any(not isinstance(speaker, str) or not speaker for speaker in speakers):
            raise RuntimeError("Qwen3-TTS returned invalid supported speaker ids.")
        return speakers

    @overload
    def synthesize(
        self,
        text: str,
        options: TTSOptions | None = None,
        *,
        reference_audio: AudioReference | None = None,
        reference_audio_path: PathAudioReference | None = None,
    ) -> TTSOutput: ...

    @overload
    def synthesize(
        self,
        text: Sequence[str],
        options: TTSOptions | None = None,
        *,
        reference_audios: Sequence[AudioReference] | None = None,
        reference_audio_paths: Sequence[PathAudioReference] | None = None,
    ) -> list[TTSOutput]: ...

    def synthesize(
        self,
        text: str | Sequence[str],
        options: TTSOptions | None = None,
        *,
        reference_audio: AudioReference | None = None,
        reference_audios: Sequence[AudioReference] | None = None,
        reference_audio_path: PathAudioReference | None = None,
        reference_audio_paths: Sequence[PathAudioReference] | None = None,
    ) -> TTSOutput | list[TTSOutput]:
        if any(
            value is not None
            for value in (
                reference_audio,
                reference_audios,
                reference_audio_path,
                reference_audio_paths,
            )
        ):
            raise ValueError("Qwen CustomVoice TTS does not accept reference audio.")
        self._validate_options(options)
        merged = self._options(options)
        if merged.speaker is None:
            raise ValueError("Qwen CustomVoice TTS requires a speaker id.")
        return self._synthesize_custom_voice(
            text,
            speakers=merged.speaker,
            languages=merged.language or self.config.language,
            instructs=self.config.instruct,
            options=merged,
        )

    def synthesize_custom_voice(
        self,
        text: str | Sequence[str],
        *,
        speakers: str | Sequence[str] | None = None,
        languages: str | Sequence[str] | None = None,
        instructs: str | Sequence[str] | None = None,
        options: TTSOptions | None = None,
    ) -> TTSOutput | list[TTSOutput]:
        self._validate_options(options)
        return self._synthesize_custom_voice(
            text,
            speakers=speakers,
            languages=languages,
            instructs=instructs,
            options=self._options(options),
        )

    def _synthesize_custom_voice(
        self,
        text: str | Sequence[str],
        *,
        speakers: str | Sequence[str] | None,
        languages: str | Sequence[str] | None,
        instructs: str | Sequence[str] | None,
        options: TTSOptions,
    ) -> TTSOutput | list[TTSOutput]:
        single = isinstance(text, str)
        texts = [validate_text(text)] if single else _text_batch(text)
        resolved_speakers = _value_batch(
            speakers if speakers is not None else self.config.speaker,
            count=len(texts),
            name="speakers",
        )
        resolved_languages = _value_batch(
            languages if languages is not None else self.config.language,
            count=len(texts),
            name="languages",
        )
        resolved_instructs = _optional_value_batch(
            instructs if instructs is not None else self.config.instruct,
            count=len(texts),
            name="instructs",
        )
        self._validate_speakers(resolved_speakers)

        if self.config.inference.transformer_engine is TTSTransformerEngine.VLLM_OMNI:
            if self.omni_prompt_builder is None:
                raise RuntimeError("Qwen3-TTS vLLM-Omni prompt builder is not loaded.")
            prompts = self.omni_prompt_builder.build(
                texts,
                resolved_speakers,
                resolved_languages,
                resolved_instructs,
                options,
            )
            outputs = generate_audio(
                cast(OmniEngine, self.model),
                prompts,
                options=options,
                sample_rate=self.config.sample_rate,
                backend=self.name,
                max_concurrency=self.config.inference.max_concurrency,
                meta=[
                    {"speaker": speaker, "language": language}
                    for speaker, language in zip(
                        resolved_speakers, resolved_languages, strict=True
                    )
                ],
            )
        else:
            runtime = _runtime_kwargs(options)
            generated = self.model.generate_custom_voice(
                text=text if single else texts,
                language=resolved_languages[0] if single else resolved_languages,
                speaker=resolved_speakers[0] if single else resolved_speakers,
                instruct=None
                if resolved_instructs is None
                else resolved_instructs[0]
                if single
                else resolved_instructs,
                **runtime,
            )
            outputs = _outputs(
                generated,
                speakers=resolved_speakers,
                languages=resolved_languages,
                default_sample_rate=self.config.sample_rate,
            )
        return outputs[0] if single else outputs

    def _options(self, options: TTSOptions | None) -> TTSOptions:
        return self.config.options().merged(options)

    def _validate_options(self, options: TTSOptions | None) -> None:
        if options is None:
            return
        if (
            options.seed is not None
            and self.config.inference.transformer_engine is TTSTransformerEngine.TORCH
        ):
            raise ValueError("Qwen CustomVoice TTS does not support seed.")
        if options.sample_rate is not None:
            raise ValueError("Qwen CustomVoice TTS does not support sample_rate overrides.")

    def _validate_speakers(self, speakers: Sequence[str]) -> None:
        supported = self.supported_speakers()
        if not supported:
            return
        supported_lower = {speaker.lower() for speaker in supported}
        missing = sorted(
            speaker for speaker in set(speakers) if speaker.lower() not in supported_lower
        )
        if missing:
            available = ", ".join(supported)
            raise ValueError(
                f"unsupported Qwen speaker ids: {', '.join(missing)}; "
                f"available speakers: {available}."
            )


def _validate_qwen_dtype(value: object | None) -> None:
    if value is torch.float16 or (
        isinstance(value, str) and value.lower() in {"float16", "fp16", "half"}
    ):
        raise ValueError("Qwen3-TTS inference uses bfloat16; float16 is unsupported.")


def _text_batch(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("text batch must be a sequence of strings.")
    texts = [validate_text(text) for text in value]
    if not texts:
        raise ValueError("text batch must not be empty.")
    return texts


def _value_batch(
    value: str | Sequence[str] | None,
    *,
    count: int,
    name: str,
) -> list[str]:
    if value is None:
        raise ValueError(f"Qwen CustomVoice TTS requires {name}.")
    if isinstance(value, str):
        values = [value] * count
    elif isinstance(value, Sequence):
        values = list(value)
    else:
        raise TypeError(f"{name} must be a string or sequence of strings.")
    if len(values) != count:
        raise ValueError(f"{name} length must match text batch length.")
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{name} must contain non-empty strings.")
    return values


def _optional_value_batch(
    value: str | Sequence[str] | None,
    *,
    count: int,
    name: str,
) -> list[str] | None:
    if value is None:
        return None
    return _value_batch(value, count=count, name=name)


def _runtime_kwargs(options: TTSOptions) -> dict[str, object]:
    runtime = dict(options.extra)
    if options.max_new_tokens is not None:
        runtime["max_new_tokens"] = options.max_new_tokens
    if options.temperature is not None:
        runtime["temperature"] = options.temperature
    if options.top_p is not None:
        runtime["top_p"] = options.top_p
    return runtime


def _outputs(
    generated: object,
    *,
    speakers: Sequence[str],
    languages: Sequence[str],
    default_sample_rate: int,
) -> list[TTSOutput]:
    waveforms, sample_rate = _generated_parts(generated, default_sample_rate)
    values = (
        [waveforms]
        if _is_waveform_value(waveforms)
        else list(cast(Iterable[object], waveforms))
    )
    if len(values) != len(speakers):
        raise RuntimeError("Qwen3-TTS output count does not match text batch length.")
    return [
        _output(value, sample_rate, speaker=speaker, language=language)
        for value, speaker, language in zip(values, speakers, languages, strict=False)
    ]


def _generated_parts(
    generated: object,
    default_sample_rate: int,
) -> tuple[object, int]:
    if isinstance(generated, tuple) and len(generated) == 2:
        waveforms, sample_rate = generated
        positive_int(sample_rate, "sample_rate")
        return waveforms, int(sample_rate)
    return generated, default_sample_rate


def _is_waveform_value(value: object) -> bool:
    if isinstance(value, Tensor):
        return True
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _output(value: object, sample_rate: int, *, speaker: str, language: str) -> TTSOutput:
    waveform = _waveform(value)
    return TTSOutput(
        waveform=waveform,
        sample_rate=sample_rate,
        duration=waveform_duration(waveform, sample_rate),
        meta={
            "backend": DEFAULT_MODEL,
            "speaker": speaker,
            "language": language,
        },
    )


def _waveform(value: object) -> Tensor:
    waveform = torch.as_tensor(value, dtype=torch.float32).detach().cpu().contiguous()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise RuntimeError("Qwen3-TTS returned an invalid waveform shape.")
    if not torch.isfinite(waveform).all():
        raise RuntimeError("Qwen3-TTS returned a non-finite waveform.")
    return waveform


class QwenVoiceCloneTTS:
    """Qwen3-TTS Base ICL (voice-clone) inference adapter.

    Wraps ``qwen_tts.Qwen3TTSModel.generate_voice_clone``: the reference is
    one audio clip plus its verbatim transcript (``prompt_text``), mirroring
    the CosyVoice zero-shot interface so producers stay engine-agnostic.
    The Base family does not take speaker ids; a reference is required.

    Transformer engines: ``TORCH`` (in-process qwen_tts) and ``VLLM_OMNI``
    (offline vLLM-Omni engine; ``omni_kwargs`` carries the deploy config,
    e.g. the dual-GPU talker/Code2Wav split).
    """

    VOICE_CLONE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    sample_rate = DEFAULT_SAMPLE_RATE

    def __init__(self, model: Any) -> None:
        is_omni = hasattr(model, "generate") and not hasattr(model, "generate_voice_clone")
        if not hasattr(model, "generate_voice_clone") and not is_omni:
            raise TypeError(
                "QwenVoiceCloneTTS requires a Qwen3TTSModel or vLLM-Omni backend."
            )
        self.model = model
        self.omni_prompt_builder: Any | None = None

    @classmethod
    def from_pretrained(
        cls,
        model: str | Path | None = None,
        *,
        inference: TTSInferenceConfig | None = None,
        omni_kwargs: Mapping[str, object] | None = None,
        dtype: Any = torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
        device_map: str | Any = "cuda:0",
        **load_options: Any,
    ) -> QwenVoiceCloneTTS:
        selected = inference or TTSInferenceConfig()
        ENGINE_CAPABILITIES.validate(selected, "Qwen3-TTS voice clone")
        if selected.transformer_engine is TTSTransformerEngine.VLLM_OMNI:
            source = str(model or cls.VOICE_CLONE_MODEL)
            engine = load_omni(source, dict(omni_kwargs or {}))
            instance = cls(engine)
            instance.omni_prompt_builder = (
                _load_voice_clone_prompt_builder(
                    source,
                    cache_dir=load_options.get("cache_dir"),
                    revision=load_options.get("revision"),
                    local_files_only=bool(load_options.get("local_files_only", False)),
                )
            )
            return instance
        QWEN_TORCH_ENVIRONMENT.validate()
        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise ImportError("QwenVoiceCloneTTS requires the qwen-tts package.") from exc
        backend = Qwen3TTSModel.from_pretrained(
            model or cls.VOICE_CLONE_MODEL,
            dtype=dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
            **load_options,
        )
        return cls(backend)

    def capabilities(self) -> TTSCapabilities:
        return TTSCapabilities(
            voice_modes=frozenset((TTSVoiceMode.REFERENCE_AUDIO,)),
            engines=ENGINE_CAPABILITIES,
        )

    @torch.no_grad()
    def synthesize(
        self,
        text: str | Sequence[str],
        options: TTSOptions | None = None,
        *,
        reference_audio: AudioReference | None = None,
        reference_audios: Sequence[AudioReference] | None = None,
        reference_audio_path: PathAudioReference | None = None,
        reference_audio_paths: Sequence[PathAudioReference] | None = None,
    ) -> TTSOutput | list[TTSOutput]:
        resolved = TTSOptions().merged(options)
        single = isinstance(text, str)
        texts = [validate_text(text)] if single else list(text)
        references = _clone_reference_batch(
            reference_audio=reference_audio,
            reference_audios=reference_audios,
            reference_audio_path=reference_audio_path,
            reference_audio_paths=reference_audio_paths,
            count=len(texts),
            single=single,
        )
        prompt_texts = resolved.prompt_text
        if not isinstance(prompt_texts, str) and (
            not isinstance(prompt_texts, Sequence) or len(prompt_texts) != len(texts)
        ):
            raise ValueError("Qwen ICL synthesis requires one verbatim prompt_text per text.")
        prompt_text_list = (
            [prompt_texts] * len(texts) if isinstance(prompt_texts, str) else list(prompt_texts)
        )
        languages = (
            [_clone_language(resolved.language)] * len(texts)
            if resolved.language is not None
            else None
        )
        if self.omni_prompt_builder is not None:
            if self.omni_prompt_builder is None:
                raise RuntimeError("Qwen3-TTS vLLM-Omni clone prompt builder is not loaded.")
            ref_payloads = [
                _clone_reference_data_uri(reference) for reference in references
            ]
            prompts = self.omni_prompt_builder.build(
                texts,
                ref_payloads,
                prompt_text_list,
                languages,
                resolved,
            )
            outputs = generate_audio(
                cast(OmniEngine, self.model),
                prompts,
                options=resolved,
                sample_rate=self.sample_rate,
                backend=self.name,
                prompt_extra_keys=frozenset({"ref_audio", "ref_text"}),
                meta=[{"language": language} for language in languages],
            )
            return outputs[0] if single else outputs
        if len(texts) == 1:
            output = self._synthesize_one(
                texts[0], references[0], prompt_text_list[0], resolved.language
            )
            return output
        # Batched: one generate_voice_clone call with list inputs lets the
        # backend batch the talker decode, which is the whole point of the
        # vLLM-free scaling route.
        ref_payloads = [_clone_reference_payload(reference) for reference in references]
        wavs, sample_rate = self.model.generate_voice_clone(
            text=texts,
            language=_clone_language(resolved.language),
            ref_audio=ref_payloads,
            ref_text=prompt_text_list,
        )
        outputs = []
        for wav in wavs:
            waveform = _waveform(wav)
            outputs.append(
                TTSOutput(
                    waveform=waveform,
                    sample_rate=int(sample_rate),
                    duration=waveform_duration(waveform, sample_rate),
                    meta={
                        "backend": self.VOICE_CLONE_MODEL,
                        "voice_mode": TTSVoiceMode.REFERENCE_AUDIO.value,
                        "language": resolved.language,
                    },
                )
            )
        return outputs

    name = "qwen_voice_clone"

    def _synthesize_one(
        self,
        text: str,
        reference: AudioReference,
        prompt_text: str,
        language: str | None,
    ) -> TTSOutput:
        wavs, sample_rate = self.model.generate_voice_clone(
            text=text,
            language=_clone_language(language),
            ref_audio=_clone_reference_payload(reference),
            ref_text=prompt_text,
        )
        if not wavs:
            raise RuntimeError("Qwen3-TTS voice clone returned no waveform.")
        waveform = _waveform(wavs[0])
        return TTSOutput(
            waveform=waveform,
            sample_rate=int(sample_rate),
            duration=waveform_duration(waveform, sample_rate),
            meta={
                "backend": self.VOICE_CLONE_MODEL,
                "voice_mode": TTSVoiceMode.REFERENCE_AUDIO.value,
                "language": language,
            },
        )


def _load_voice_clone_prompt_builder(
    model: str,
    *,
    cache_dir: object = None,
    revision: object = None,
    local_files_only: bool = False,
) -> Any:
    """Load the Base-task prompt builder from the vLLM-Omni install."""

    from ._omni import QwenOmniVoiceClonePromptBuilder

    return QwenOmniVoiceClonePromptBuilder.from_pretrained(
        model,
        cache_dir=cache_dir,  # type: ignore[arg-type]
        revision=revision,  # type: ignore[arg-type]
        local_files_only=local_files_only,
    )


def _clone_language(language: str | None) -> str:
    """Normalize language codes and names to the full names qwen-tts expects."""

    value = str(language or "").strip().lower()
    if value in {"zh", "zh-cn", "chinese"}:
        return "Chinese"
    if value in {"en", "english"}:
        return "English"
    raise ValueError(f"Qwen ICL synthesis requires language 'zh'/'en', got {language!r}.")


def _mono_float32(waveform) -> np.ndarray:
    import numpy as np

    array = np.asarray(waveform, dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=0)
    return array.reshape(-1)


def _clone_reference_payload(reference: AudioReference) -> object:
    """Torch-backend payload: 1-D float32 mono (the qwen_tts downmix branch
    mutates its (wav, sr) tuple in place for multichannel input, so the mono
    form also avoids that crash)."""
    if isinstance(reference, (str, Path)):
        return str(reference)
    if isinstance(reference, WaveformReference):
        return (_mono_float32(reference.waveform), int(reference.sample_rate))
    if isinstance(reference, EncodedAudioReference):
        raise ValueError("Qwen ICL references accept a path or waveform, not encoded bytes.")
    raise TypeError(f"Unsupported Qwen ICL reference type: {type(reference)!r}.")


def _clone_reference_data_uri(reference: AudioReference) -> str:
    """Omni-backend payload: wav data URI (the vLLM-Omni prompt builder
    natively accepts data:audio URIs)."""
    import base64
    import io
    import wave

    if isinstance(reference, (str, Path)):
        return str(reference)
    if isinstance(reference, WaveformReference):
        pcm = (_mono_float32(reference.waveform) * 32767.0).astype("<i2").tobytes()
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(int(reference.sample_rate))
            handle.writeframes(pcm)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:audio/wav;base64,{encoded}"
    if isinstance(reference, EncodedAudioReference):
        raise ValueError("Qwen ICL references accept a path or waveform, not encoded bytes.")
    raise TypeError(f"Unsupported Qwen ICL reference type: {type(reference)!r}.")


def _clone_reference_batch(
    *,
    reference_audio: AudioReference | None,
    reference_audios: Sequence[AudioReference] | None,
    reference_audio_path: PathAudioReference | None,
    reference_audio_paths: Sequence[PathAudioReference] | None,
    count: int,
    single: bool,
) -> list[AudioReference]:
    if single:
        reference: AudioReference | None = reference_audio
        if reference is None and reference_audio_path is not None:
            reference = reference_audio_path
        if reference is None and reference_audios is not None:
            if len(reference_audios) != 1:
                raise ValueError("Single-text ICL synthesis takes exactly one reference.")
            reference = reference_audios[0]
        if reference is None and reference_audio_paths is not None:
            if len(reference_audio_paths) != 1:
                raise ValueError("Single-text ICL synthesis takes exactly one reference.")
            reference = reference_audio_paths[0]
        if reference is None:
            raise ValueError("Qwen ICL synthesis requires reference audio.")
        return [reference]
    references = reference_audios
    if references is None and reference_audio_paths is not None:
        references = list(reference_audio_paths)
    if references is None and reference_audio is not None:
        references = [reference_audio] * count
    if references is None or len(references) != count:
        raise ValueError("Batched ICL synthesis requires one reference per text.")
    return list(references)
