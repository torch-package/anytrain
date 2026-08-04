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
