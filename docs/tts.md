# TTS

`anytrain.tts` provides small inference adapters for text-to-speech backends. It
does not own datasets, waveform caches, manifest files, training steps, or a
default CLI.

## Inference engines

Engine selection is an AnyTrain model-runtime concern. `TTSInferenceConfig`
separates the autoregressive transformer engine from the downstream acoustic
engine because TensorRT is not a universal TTS capability:

| Backend | Transformer engines | Acoustic engines |
| --- | --- | --- |
| CosyVoice 2/3 | `torch`, embedded `vllm` | `torch`, `tensorrt` |
| Qwen3-TTS CustomVoice | `torch`, `vllm_omni` | `torch` |
| MOSS-TTS v1.5 | `torch`, `vllm_omni` | `torch` |

Unsupported combinations fail while building the configuration. `vllm` and
`vllm_omni` are distinct values: CosyVoice owns an embedded upstream vLLM
integration, while Qwen3-TTS and MOSS-TTS use the multi-stage vLLM-Omni
pipeline. AnyTrain never interprets either value as a boolean request to keep
running the PyTorch backend.

`max_concurrency` bounds requests submitted together and CosyVoice's upstream
TRT pool. It is an execution setting and is excluded from `config_hash`;
engine selection remains in the hash because different engines may change
numerical output. Dataset batching, OOM splitting, materialized stores and
worker scheduling remain in AnyDataset. Model ids, revisions, prompt choices
and production profiles remain in workspace or the calling project.

Install the optional vLLM-Omni runtime in a dedicated CUDA environment:

```bash
python -m pip install 'anytrain[tts-vllm-omni]'
```

The extra installs matching `vllm` and `vllm-omni` 0.26 release lines plus
`transformers>=5.5.3`. vLLM-Omni does not declare the core vLLM wheel itself,
and their request and pipeline contracts evolve together.

Qwen3-TTS's PyTorch wrapper is a different environment contract. The pinned
upstream package requires `transformers==4.57.3`, so install it in its own
environment:

```bash
python -m pip install 'anytrain[qwen-tts]'
```

Do not install `anytrain[qwen-tts,tts-vllm-omni]` together. The dependency
constraints are intentionally unsatisfiable: a caller that needs both engines
must run them behind separate process boundaries. AnyTrain validates the
selected backend's installed distributions before model construction and
reports missing, mismatched, or mutually incompatible runtimes. It does not
patch Transformers APIs or fall back to a different engine.

Install the MOSS backend dependencies from the anytrain repository root:

```bash
python -m pip install 'anytrain[moss-tts]'
```

The optional TTS runtimes are intentionally separate environment families. AnyTrain
publishes their inspectable distribution contracts through
`anytrain.tts.environments` so a workspace installer can validate the same rules
without importing a backend:

| id | validated distribution family | boundary |
| --- | --- | --- |
| `qwen_torch` | `qwen-tts==0.1.1`, `transformers==4.57.3` | Qwen3-TTS CustomVoice PyTorch |
| `moss_torch` | Torch/TorchAudio 2.9, TorchCodec 0.8, `transformers==5.0.0` | MOSS-TTS v1.5 PyTorch |
| `cosyvoice_vllm` | Torch/TorchAudio 2.8, `transformers==4.57.1`, `vllm==0.11.0`, ORT GPU 1.20.2 | CosyVoice embedded vLLM |
| `vllm_omni` | `vllm`/`vllm-omni` 0.26 release line, `transformers>=5.5.3` | Qwen3-TTS or MOSS vLLM-Omni |

Use `environment_contract("moss_torch")` or the read-only
`ENVIRONMENT_CONTRACTS` mapping to inspect requirements and conflicts. Calling
`contract.validate()` checks the current interpreter; passing a mapping such as
`contract.validate({"transformers": "5.0.0", ...})` validates a recorded inventory.
The MOSS and CosyVoice rows describe the verified runtime inventories, not a
promise that arbitrary CUDA wheels or an unpinned upstream checkout are portable.

CosyVoice 2/3 uses the official source tree because the upstream runtime also
depends on its Matcha-TTS submodule. Keep a production checkout pinned by the
calling project, initialize submodules, install its requirements, and make the
checkout importable before constructing the backend:

```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /path/to/CosyVoice
python -m pip install -r /path/to/CosyVoice/requirements.txt
export PYTHONPATH=/path/to/CosyVoice:$PYTHONPATH
```

For the embedded-vLLM path, install the validated AnyTrain runtime first:

```bash
python -m pip install 'anytrain[cosyvoice-vllm]'
```

The source checkout and Matcha-TTS revision remain workspace-owned because they
are upstream source boundaries rather than Python distribution contracts.

The stable public boundary is:

```python
import torch

from anytrain.tts import EncodedAudioReference, TTSOptions, WaveformReference
from anytrain.tts.moss import MossTTS

tts = MossTTS.from_pretrained(
    "OpenMOSS-Team/MOSS-TTS-v1.5",
    trust_remote_code=True,
    torch_dtype="bfloat16",
)
audio = tts.synthesize("hello")
batch_audio = tts.synthesize(["hello", "你好"])
```

MOSS vLLM-Omni uses the same public facade and the upstream processor to build
the aligned text/reference-code prompt:

```python
from anytrain.tts import TTSInferenceConfig, TTSTransformerEngine

tts = MossTTS.from_pretrained(
    "OpenMOSS-Team/MOSS-TTS-v1.5",
    trust_remote_code=True,
    inference=TTSInferenceConfig(
        transformer_engine=TTSTransformerEngine.VLLM_OMNI,
        max_concurrency=4,
    ),
    omni_kwargs={"deploy_config": "moss_tts"},
)
```

This path is implemented against vLLM-Omni's native MOSS pipeline, but MOSS
v1.5 still requires a real GPU smoke before production use; the matching
upstream end-to-end test is currently skipped for vLLM-Omni issue #4643.
Realtime MOSS is a distinct checkpoint/pipeline capability and is not exposed
as a TensorRT or generic `rt=True` switch on the v1.5 adapter.

`synthesize()` is the high-level `text -> waveform` entry point.
`TTSOutput.waveform` is a float tensor with shape `[channels, time]`.
Passing a single string returns one `TTSOutput`; passing a sequence of strings
returns `list[TTSOutput]`. Batch synthesis shares one `TTSOptions` object across
all texts and keeps each variable-length waveform as a separate output item.

`protocol.py` owns the stable reference/capability/options/output/backend contract. Numeric controls are
strict host values: sample rates and token limits are positive integers, `temperature` is finite positive,
`top_p` is finite in `(0, 1]`, and bool/string coercions are rejected. `TTSOutput` requires a non-empty
`[channels, time]` Tensor, finite positive duration, positive integer sample rate, and string-keyed metadata.
`_common.py` owns only shared validation and deterministic config hashing; provider support policy remains in
each backend adapter.

Every loaded backend exposes `capabilities`. `TTSVoiceMode.SPEAKER_ID` means a
caller can persist one speaker id and reuse it across texts and languages;
`TTSVoiceMode.REFERENCE_AUDIO` means it can reuse one audio reference. The
optional `speaker_ids` tuple lists known ids when the backend can enumerate
them. `unsupported_speaker_ids()` returns the requested ids that are known to
be unsupported, or `None` when a speaker-id backend cannot enumerate its ids.
Dataset code must choose and persist the condition; the backend does not own
source scheduling or speaker assignment.

The MOSS adapter intentionally targets only MOSS-TTS v1.5. It loads the
Hugging Face remote-code model and processor, then runs the v1.5 processor
generation path. Keyword arguments passed to `from_pretrained()` are load-time
options, generation settings belong in `TTSOptions` or `runtime_kwargs`, and
reference audio is passed through the canonical `reference_audio` /
`reference_audios` parameters:

```python
tts = MossTTS.from_pretrained(
    "OpenMOSS-Team/MOSS-TTS-v1.5",
    trust_remote_code=True,
    codec_model="OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
)

audio = tts.synthesize(
    "你好",
    TTSOptions(seed=7),
    reference_audio="assets/audio/zh_1.wav",
)

waveform_prompt = tts.synthesize(
    "你好",
    reference_audio=WaveformReference(
        waveform=torch.zeros(1, 24000),
        sample_rate=24000,
    ),
)
encoded_prompt = tts.synthesize(
    "你好",
    reference_audio=EncodedAudioReference(flac_bytes, ".flac"),
)
```

`WaveformReference` and `EncodedAudioReference` stay in memory at the public
boundary. A path-only backend such as MOSS materializes them into a private
temporary directory for the complete processor/generate/decode call and removes
the files afterward. `reference_audio_path` / `reference_audio_paths` remain as
deprecated path-only compatibility aliases; do not pass an alias together with
its canonical parameter.

The v1.5 adapter returns audio in memory and does not support legacy
`prompt_audio_path`, reference audio as a runtime kwarg,
`output_audio_path`, `speaker`, `model.inference()`, or `tokenize -> generate ->
decode` compatibility paths.

When merging `TTSOptions`, explicitly passed fields override defaults even when
the value is `None`; omitted fields inherit the backend config.

The Qwen adapter targets the CustomVoice checkpoint and uses speaker ids instead
of prompt audio:

```python
from anytrain.tts import TTSOptions
from anytrain.tts.qwen import QwenCustomVoiceTTS

tts = QwenCustomVoiceTTS.from_pretrained(device="cuda", torch_dtype="bfloat16")
audio = tts.synthesize(
    "hello",
    TTSOptions(speaker="Vivian", language="English"),
)
batch_audio = tts.synthesize_custom_voice(
    ["hello", "你好"],
    speakers=["Vivian", "Ryan"],
    languages=["English", "Chinese"],
)
```

The accelerated path constructs the exact vLLM-Omni CustomVoice placeholder
length instead of forwarding a `load_vllm` flag to the Hugging Face wrapper:

```python
from anytrain.tts import TTSInferenceConfig, TTSTransformerEngine

tts = QwenCustomVoiceTTS.from_pretrained(
    speaker="Vivian",
    inference=TTSInferenceConfig(
        transformer_engine=TTSTransformerEngine.VLLM_OMNI,
        max_concurrency=8,
    ),
    omni_kwargs={"stage_init_timeout": 600},
)
```

Qwen inference remains `bfloat16` by project policy. PyTorch-only load knobs
such as `torch_dtype` and `attn_implementation` are rejected on this path;
vLLM-Omni engine/deploy knobs belong in `omni_kwargs`.

Cost-aware planning can load only the matching Hugging Face processor and count
the exact CustomVoice assistant-prompt tokens used by the upstream inference
path:

```python
from anytrain.tts.qwen import QwenCustomVoicePromptEstimator

prompt_cost = QwenCustomVoicePromptEstimator(
    revision="0c0e3051f131929182e2c023b9537f8b1c68adfe",
    cache_dir="storage/huggingface",
    local_files_only=True,
)
tokens = prompt_cost.estimate("hello")
```

The estimator is lazy and pickle-safe. It calls
`AutoProcessor.from_pretrained(..., fix_mistral_regex=True)` without loading the
TTS model, so it uses the Transformers processor already present in either
selected backend environment and does not require the `qwen-tts` distribution.
It then applies the same
`<|im_start|>assistant\n...<|im_end|>\n<|im_start|>assistant\n` protocol as
Qwen3-TTS CustomVoice. Its `identity` includes only the logical model id, pinned
model revision, and prompt protocol. Cache paths and `local_files_only` are
physical loading policy and do not change semantic identity. The orchestration
layer must bind the estimator and synthesis backend to the same logical model
and revision; a resolved local cache path never becomes semantic identity.

CosyVoice 2 and 3 share one adapter and the same `TTSOptions` contract. A
reference plus `prompt_text` selects zero-shot synthesis; a reference without
prompt text selects cross-lingual synthesis; `instruction` selects instruct2.
SFT checkpoints can instead use `speaker` without a reference:

```python
from anytrain.tts import (
    TTSAcousticEngine,
    TTSInferenceConfig,
    TTSOptions,
    TTSTransformerEngine,
)
from anytrain.tts.cosyvoice import CosyVoiceTTS

tts = CosyVoiceTTS.from_pretrained(
    version="v3",
    device="cuda:0",
    inference=TTSInferenceConfig(
        transformer_engine=TTSTransformerEngine.VLLM,
        acoustic_engine=TTSAcousticEngine.TENSORRT,
        max_concurrency=4,
    ),
    runtime_kwargs={"speed": 1.0, "text_frontend": True},
)
audio = tts.synthesize(
    "目标文本",
    TTSOptions(prompt_text="参考音频转写"),
    reference_audio="assets/reference.wav",
)
batch_audio = tts.synthesize(
    ["目标一", "目标二"],
    TTSOptions(prompt_text=["参考一", "参考二"]),
    reference_audios=["assets/one.wav", "assets/two.wav"],
)
```

The default model ids are `FunAudioLLM/CosyVoice2-0.5B` and
`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`. Path references are passed directly to
the upstream frontend. In-memory waveform references are converted to mono and
materialized as request-scoped WAV files; encoded references retain their
original bytes and suffix. Temporary files remain available until the upstream
generator is fully consumed. Upstream `tts_speech` chunks are concatenated, and
output remains one variable-length `TTSOutput` per input. For CosyVoice3,
AnyTrain adds the required `You are a helpful assistant.<|endofprompt|>` marker
internally; callers continue to pass only the reference transcript. CosyVoice2
prompt text is unchanged. Upstream exposes scalar `inference_*` methods rather
than an end-to-end `list[str]` API. With `load_vllm=True` and
`max_concurrency=N`, AnyTrain submits up to `N` scalar requests concurrently so
the upstream vLLM request queue can dynamically batch the LLM stage; results are
returned in input order. Flow and HiFT decoding remain per-request, so this is
not full-pipeline native batching. `max_concurrency` is an execution setting and
does not change `config_hash`; values greater than one require a vLLM engine.
The legacy `load_vllm`, `load_trt`, `trt_concurrent` and `max_concurrency`
arguments remain accepted for existing callers, but conflicting legacy and
typed engine selections fail explicitly.
AnyTrain publishes the CosyVoice model registration through the
`vllm.general_plugins` entry-point group so vLLM's spawned engine-core process
loads the same bias-aware model class as the client. Install or refresh the
AnyTrain package after changing this integration; adding `src` to `PYTHONPATH`
alone does not update installed entry-point metadata. CUDA visibility and
multi-process device assignment belong to the caller's orchestration layer. The
adapter consumes its process-local device and does not rewrite
`CUDA_VISIBLE_DEVICES` or patch child-process startup.

Spark-TTS is exposed as a reference-audio backend. It does not create or persist
speaker ids. The caller supplies one transcript aligned with each reference
audio through `TTSOptions.prompt_text`:

```python
from anytrain.tts import TTSOptions
from anytrain.tts.spark import SparkTTS

tts = SparkTTS.from_pretrained(
    "SparkAudio/Spark-TTS-0.5B",
    device="cuda:0",
)
audio = tts.synthesize(
    "目标文本",
    TTSOptions(prompt_text="参考音频转写"),
    reference_audio="assets/reference.wav",
)
batch_audio = tts.synthesize(
    ["目标一", "目标二"],
    TTSOptions(prompt_text=["参考一", "参考二"]),
    reference_audios=["assets/one.wav", "assets/two.wav"],
)
```

The upstream Spark runtime currently exposes single-item inference. The
anytrain batch API preserves aligned variable-length outputs but executes those
items sequentially; it must not be reported as native model batching. Treat
this backend as smoke/probe-only and do not use it for long-running production
materialization until the upstream runtime provides native batched inference.

MOSS is split by actual lifecycle ownership: `_config.py` owns serializable semantic configuration,
`_loading.py` owns lazy Hugging Face construction, `_runtime.py` owns generation and request-scoped reference
materialization, and `_output.py` owns processor decode normalization. Processor sample-rate/duration metadata
is validated with the same strict numeric contract as the public output rather than coerced with `int()` or
`float()`. Qwen, CosyVoice, and Spark keep their upstream inference state machines intact because their native
batching, voice conditioning, and output protocols differ materially.

## Verification

```bash
python -m pytest \
  tests/test_tts_engines.py tests/test_tts_moss.py tests/test_qwen_tts.py \
  tests/test_cosyvoice_tts.py tests/test_spark_tts.py
```
