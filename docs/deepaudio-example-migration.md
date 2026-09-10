# Deepaudio Example Package Migration Design

## 当前状态

迁移已落地：`anytrain.example` 顶层导出 `color_your_night()` 和 `vctk()`，资源路径与 lazy
decoder helper 放在 `anytrain.example.resources` / `anytrain.example.audio`，speech FLAC 与
music MP3 都已进入 package data。下文的实施顺序保留为设计记录，不表示这些阶段仍未完成。

## 目标

把 `deepaudio/src/deepaudio/example` 迁到 `anytrain/src/anytrain/example`，提供随包分发的小音频 assets 和直接可 import 的读取 helper，用于 examples、单元测试和下游 smoke test。

这个迁移的重点不是 `deepaudio/demo` 或 notebooks，而是源码包里的 `deepaudio.example`：

```text
deepaudio/src/deepaudio/example/
  __init__.py
  README.md
  music.py
  speech.py
  musdb.py
  assets/
    music/Atlus Sound Team-Color Your Night.mp3
    speech/p225_001_mic1.flac
```

第一版目标：

- 在 `src/anytrain/example/` 提供稳定的 package resource API。
- 在 `src/anytrain/example/assets/` 放 deepaudio example 现有的 speech / music 小音频文件。
- helper 返回 `(waveform, sample_rate)`，其中 `waveform` 是 `torch.Tensor`，默认 shape 为 `(channels, time)`。
- 测试可以不依赖外部数据集、私有路径、网络下载或预训练权重。
- assets 必须被 wheel/sdist 打包，不能只在 editable install 下可用。

## 非目标

第一版不迁这些内容：

- `deepaudio/demo/**` 和 `deepaudio/notebooks/**` 的 notebook / 展示页面 / 试听资产。
- `deepaudio.example.musdb_hq()` 的真实 MUSDB-HQ 数据集读取入口。它依赖 `MUSDBHQ_DATASET_DIR`，不适合作为随包 example。
- `deepaudio.utils.io` 整套 IOHandler、保存逻辑和 torchcodec 绑定。
- DynaCodec、DAC、SCNet、stable-codec、wrapper、zoo 或预训练权重。
- 训练入口、Hydra 配置、data module 或 batch schema。

## 现状盘点

### `deepaudio.example.music`

当前提供：

```python
def color_your_night(start_seconds: float | None = 0, duration: float | None = None):
    return load(_COLOR_YOUR_NIGHT_PATH, start_seconds=start_seconds, duration=duration)
```

asset：

- `assets/music/Atlus Sound Team-Color Your Night.mp3`
- 当前大小约 8.9 MB
- MP3 44.1 kHz joint stereo

迁移注意：

- 文件名包含空格，内部 API 应暴露稳定 path helper，避免用户手写路径。
- MP3 读取需要额外 decoder，不能假设 Python 标准库可读。
- 这个文件比 speech fixture 大很多，迁移时需要确认仓库体积和来源说明可以接受。

### `deepaudio.example.speech`

当前提供：

```python
def vctk(...):
    return load(_SPEECH_PATH, start_seconds=0, duration=None)
```

asset：

- `assets/speech/p225_001_mic1.flac`
- 当前大小约 104 KB
- FLAC 48 kHz mono，约 98k samples

迁移注意：

- 当前 `vctk()` 暴露了 `frame_offset`、`num_frames`、`backend` 等参数，但没有真正传入 `load()`。
- docstring 错写成了 `Color Your Night.mp3`。
- 迁移时不保留这种无效参数；只保留能真正支持的读取参数。

### `deepaudio.example.musdb`

当前提供：

```python
def musdb_hq(..., sources=["bass", "drums", "other", "vocals", "mixture"]):
    root = f"{MUSDBHQ_DATASET_DIR}/musdb18hq/train"
    ...
```

迁移结论：

- 不进入 `anytrain.example` 第一版。
- 如果下游需要真实数据集 fixture，应由 `anydataset` 或具体音频项目提供。

## 目标结构

推荐新增：

```text
src/anytrain/example/
  __init__.py
  README.md
  resources.py
  audio.py
  assets/
    speech/
      p225_001_mic1.flac
    music/
      color_your_night.mp3
tests/
  test_example_assets.py
```

第一版按 speech / music 两类都迁移，保证测试能覆盖不同采样率、声道数和编码格式。若后续确认 MP3 不适合进入基础包，再降级为只保留 speech fixture。

## 公开 API

`anytrain.example` 顶层只导出两个固定音频读取函数：

```python
from anytrain.example import color_your_night, vctk
```

资源路径、枚举和 fallback 读取 helper 放在 `anytrain.example.resources` /
`anytrain.example.audio` 内部使用，不从包顶层导出。

### Path API

Path API 不需要任何 audio decoder，可作为最稳的测试入口：

```python
from pathlib import Path

from anytrain.example.resources import vctk_path

path: Path = vctk_path()
assert path.exists()
```

推荐实现：

```python
from importlib.resources import files
from pathlib import Path


def example_audio_path(name: ExampleAudio | str) -> Path:
    resource = files("anytrain.example").joinpath("assets", ...)
    return Path(resource)
```

注意：`importlib.resources.files()` 返回的是 `Traversable`。当资源来自 zip wheel 时不一定是普通文件路径。第一版如果只支持普通文件系统安装，应在文档里说明；更稳的实现可以用 `importlib.resources.as_file()`，但这会让 API 更适合 context manager。

为了测试和下游读取简单，第一版建议：

- `*_path()` 返回 `Path`，要求资源位于普通 filesystem wheel / editable install。
- 后续如要支持 zipped package，再增加 `open_example_audio()` 或 context manager API。

### Load API

读取逻辑抽成一个 helper，固定资源函数只负责传入对应 path：

```python
def load_example_audio(
    path: Path,
    *,
    start_seconds: float = 0.0,
    duration: float | None = None,
) -> tuple[Tensor, int]:
    ...


def vctk(
    *,
    start_seconds: float = 0.0,
    duration: float | None = None,
) -> tuple[Tensor, int]:
    ...


def color_your_night(
    *,
    start_seconds: float = 0.0,
    duration: float | None = None,
) -> tuple[Tensor, int]:
    ...
```

行为约定：

- `vctk()` 等固定资源函数内部调用 `load_example_audio(vctk_path(), ...)`。
- 返回 `(waveform, sample_rate)`。
- `waveform` 使用 `float32`，shape 为 `(channels, time)`。
- `start_seconds` 必须非负。
- `duration=None` 表示读取到文件尾。
- 如果 `start_seconds + duration` 超过音频长度，直接抛出 `ValueError`，不静默裁剪。
- 不做隐式 resample，不做隐式 mono/stereo 转换；需要这些处理时由测试或下游显式完成。

## Decoder 依赖策略

`anytrain` 当前 core 依赖没有 `torchcodec` / `torchaudio` / `soundfile`。为了不让 example asset 读取扩大基础依赖，读取 helper 采用 fallback 策略：

1. Path API 是 core 能力，不需要额外依赖。
2. Load API 先尝试 `torchcodec`。
3. 如果 `torchcodec` import 失败或读取失败，使用 `warnings.warn(..., RuntimeWarning)` 说明正在 fallback 到 `torchaudio`。
4. 再尝试 `torchaudio`。
5. 如果 `torchaudio` 也 import 失败或读取失败，抛出明确错误，并包含两个 backend 的失败原因和安装提示。

推荐内部结构：

```python
def load_example_audio(
    path: Path,
    *,
    start_seconds: float = 0.0,
    duration: float | None = None,
) -> tuple[Tensor, int]:
    torchcodec_error: Exception | None = None
    try:
        return _load_audio_with_torchcodec(path, start_seconds=start_seconds, duration=duration)
    except Exception as exc:
        torchcodec_error = exc
        warnings.warn(
            "torchcodec failed to read example audio; falling back to torchaudio. "
            f"torchcodec error: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    try:
        return _load_audio_with_torchaudio(path, start_seconds=start_seconds, duration=duration)
    except Exception as torchaudio_error:
        raise RuntimeError(
            "Failed to read example audio with torchcodec and torchaudio. "
            "Install audio dependencies with `python -m pip install torchaudio torchcodec`. "
            f"torchcodec error: {torchcodec_error}; torchaudio error: {torchaudio_error}"
        ) from torchaudio_error
```

实现细节：

- `_load_audio_with_torchcodec()` 和 `_load_audio_with_torchaudio()` 放在 `audio.py` 里，保持 IDE 友好的显式函数。
- fallback warning 只在需要 fallback 时出现；torchcodec 成功时不 warning。
- `torchaudio` backend 可以先整段读取 example 音频，再按 `start_seconds` / `duration` 在内存中切片；这些随包 assets 很小，优先换取更少的 API 兼容风险。
- 两个 backend 都要返回同一种 shape / dtype，不把 backend 差异暴露给调用方。
- 参数非法或时间范围越界应直接抛出 `ValueError`，不触发 fallback warning。
- `load_example_audio()` 不做静默兼容；参数非法、范围越界或两个 backend 都失败时直接抛错。

`pyproject.toml` 建议新增 optional extra：

```toml
[project.optional-dependencies]
audio = ["torchcodec>=0.2", "torchaudio>=2.0"]
```

第一版不要把 decoder 放进默认 dependencies。测试里涉及实际读取的用例如果两个 backend 都缺，应 skip optional read case；但如果只缺 `torchcodec` 且有 `torchaudio`，应断言会 warning 后成功读取。

## 资产选择

### Speech fixture

迁移：

```text
src/anytrain/example/assets/speech/p225_001_mic1.flac
```

理由：

- 文件小，适合放进 wheel 和单元测试。
- 语音样本覆盖 speech evaluator、codec loss、spectral transform 等常见 smoke test。
- 不依赖真实数据集路径。

### Music fixture

迁移：

```text
src/anytrain/example/assets/music/color_your_night.mp3
```

迁移时建议重命名为无空格文件名：

```text
color_your_night.mp3
```

理由：

- 保持 Python API 稳定，不暴露原始长文件名。
- 避免 shell / docs / 测试路径中处理空格。

注意：

- 文件约 8.9 MB，会明显增加仓库和 wheel 体积。
- MP3 decoder 依赖更容易受环境影响。

如果后续确认不迁 MP3，`color_your_night()` 不应导出假接口；避免用户 import 成功但调用失败。

## 打包配置

`anytrain` 使用 `pyproject.toml` + setuptools。需要显式加入 package data：

```toml
[tool.setuptools.package-data]
"anytrain.example" = [
    "README.md",
    "assets/speech/*.flac",
    "assets/music/*.mp3",
]
```

第一版推荐保持 assets 在 `anytrain/example/assets/` 普通目录，避免 namespace 复杂度。

验收必须包含：

```bash
python -m build
python -m pip install dist/anytrain-*.whl
python - <<'PY'
from anytrain.example.resources import vctk_path
print(vctk_path().exists())
PY
```

如果本地没有 `build`，可以先用 `pip install -e .` 做开发期 smoke test，但不能替代 wheel package-data 验收。

## 测试策略

### `tests/test_example_assets.py`

基础测试不依赖 decoder：

- `from anytrain.example import color_your_night, vctk`
- `from anytrain.example.resources import vctk_path`
- `vctk_path().is_file()`
- 文件后缀是 `.flac`
- 文件大小大于一个很小阈值，例如 `> 1024`

如果启用 `audio` extra，再测真实读取：

- `waveform, sample_rate = vctk(duration=0.1)`
- `waveform.ndim == 2`
- `sample_rate == 48000`
- `waveform.shape[-1] > 0`
- `waveform.dtype == torch.float32`

fallback 测试：

- mock `_load_audio_with_torchcodec()` 抛错，确认 `load_example_audio()` 会 warning 并调用 torchaudio helper。
- mock 两个 backend 都抛错，确认最终错误包含安装提示。

缺 decoder 时不要静默通过 `vctk()`；测试应只 skip optional read case。

### 下游 example 使用

后续仓库级训练示例可以从 `anytrain.example` 读取真实短音频：

```python
from anytrain.example import vctk

waveform, sample_rate = vctk(duration=1.0)
```

例如：

- codec reconstruction smoke test 用 `vctk(duration=1.0)` 做单样本重建。
- spectral loss / codec loss 测试用真实 waveform 替代纯随机噪声。
- evaluator speech 测试可以只用 path API，避免强制 ASR 依赖。

## 实施顺序

P0: 设计冻结

- 更新本设计文档，明确迁移对象是 `src/deepaudio/example`。
- 确认第一版迁移 speech + music 两个 assets，或记录不迁 MP3 的原因。

P1: `anytrain.example` 最小包

- 新增 `src/anytrain/example/__init__.py`。
- 新增 `src/anytrain/example/resources.py`，提供 enum / path / list API。
- 新增 `src/anytrain/example/audio.py`，提供 `load_example_audio()` 和固定资源读取函数。
- 复制 `p225_001_mic1.flac` 到 `src/anytrain/example/assets/speech/`。
- 复制并重命名 `Atlus Sound Team-Color Your Night.mp3` 到 `src/anytrain/example/assets/music/color_your_night.mp3`。
- 更新 `pyproject.toml` package-data。
- 新增 `tests/test_example_assets.py`。

P2: optional audio decoder

- 新增 `audio` extra。
- 实现 `_load_audio_with_torchcodec()` 和 `_load_audio_with_torchaudio()`。
- 实现 `load_example_audio()` 的 torchcodec -> torchaudio fallback 和 warning。
- 对 `duration` / `start_seconds` 增加错误路径测试。

P3: 使用真实 example 音频改造训练示例

- 把后续 `examples/codec_reconstruction.py` 或相关 tests 从合成 waveform 改成读取 `anytrain.example.vctk(duration=...)`。
- 保持训练示例仍然不依赖 deepaudio、真实数据集和网络下载。

## 验收标准

第一版完成时应满足：

- `from anytrain.example import color_your_night, vctk` 成功。
- `anytrain.example.__all__ == ["color_your_night", "vctk"]`。
- `from anytrain.example.resources import vctk_path` 成功，且 `vctk_path().is_file()` 在 editable install 和 wheel install 下都为真。
- 基础测试不需要网络、不需要真实数据集、不需要 deepaudio。
- `import anytrain` 不主动导入 `anytrain.example` 或 audio decoder。
- `import anytrain.example` 不主动导入 decoder；只有调用 `vctk()` 时才解析 decoder。
- torchcodec 失败且 torchaudio 可用时，`vctk()` warning 后成功读取。
- 两个 decoder 都失败时，`vctk()` 抛出带安装提示的错误。
- 如果迁移 music asset，公开文件名和 API 不包含空格路径细节。
