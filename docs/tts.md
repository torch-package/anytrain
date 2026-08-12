# TTS

`anytrain.tts` provides small inference adapters for text-to-speech backends. It
does not own datasets, waveform caches, manifest files, training steps, or a
default CLI.

Install the MOSS backend dependencies from the anytrain repository root:

```bash
python -m pip install "torchaudio>=2.0" transformers
```

Install the Qwen3-TTS CustomVoice backend when speaker-id based synthesis is
needed:

```bash
python -m pip install git+https://github.com/QwenLM/Qwen3-TTS.git
```

CosyVoice 2/3 uses the official source tree because the upstream runtime also
depends on its Matcha-TTS submodule. Keep a production checkout pinned by the
calling project, initialize submodules, install its requirements, and make the
checkout importable before constructing the backend:

```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /path/to/CosyVoice
python -m pip install -r /path/to/CosyVoice/requirements.txt
export PYTHONPATH=/path/to/CosyVoice:$PYTHONPATH
```

The stable public boundary is:

```python
import torch

from anytrain.tts import EncodedAudioReference, TTSOptions, WaveformReference
from anytrain.tts.moss import MossTTS

tts = MossTTS.from_pretrained(
    "OpenMOSS-Team/MOSS-TTS-v1.5",
    trust_remote_code=True,
    torch_dtype="auto",
)
audio = tts.synthesize("hello")
batch_audio = tts.synthesize(["hello", "你好"])
```

`synthesize()` is the high-level `text -> waveform` entry point.
`TTSOutput.waveform` is a float tensor with shape `[channels, time]`.
Passing a single string returns one `TTSOutput`; passing a sequence of strings
returns `list[TTSOutput]`. Batch synthesis shares one `TTSOptions` object across
all texts and keeps each variable-length waveform as a separate output item.

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

tts = QwenCustomVoiceTTS.from_pretrained(device="cuda")
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
TTS model, then applies the same
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
from anytrain.tts import TTSOptions
from anytrain.tts.cosyvoice import CosyVoiceTTS

tts = CosyVoiceTTS.from_pretrained(
    version="v3",
    device="cuda:0",
    load_vllm=True,
    max_concurrency=4,
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
does not change `config_hash`; values greater than one require `load_vllm=True`.
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
