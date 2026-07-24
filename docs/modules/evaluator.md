# `anytrain.evaluator` Design

## 定位

`anytrain.evaluator` 是训练期 metric/evaluator 组件层。它面向下游 LightningModule 的 `validation_step()`、`test_step()` 或需要监控的 `training_step()`，负责把 prediction/target 转成可记录指标。

## 当前结构

源码结构：

```text
src/anytrain/evaluator/
  __init__.py
  _validation.py
  abc.py
  group.py
  speech/
    __init__.py
    _torch.py
    asr.py
    audio.py
    evaluator.py
    utmos.py
  text/
    __init__.py
    _chinese.py
    _fallback.py
    evaluator.py
    normalization.py
    scores.py
```

当前公开导出：

- `EvaluatorABC`
- `MetricValue`
- `MetricDict`
- `EvaluatorGroup`

optional 子模块导出：

- `anytrain.evaluator.text.TextComparisonEvaluator`
- `anytrain.evaluator.text.TextInput`
- `anytrain.evaluator.text.TextNormalizationConfig`
- `anytrain.evaluator.speech.SpeechEvaluator`
- `anytrain.evaluator.speech.WhisperASREvaluator`
- `anytrain.evaluator.speech.UTMOSEvaluator`

## 接口

core evaluator 的调用形态：

```python
metrics = evaluator(prediction, target)
```

返回值约定：

```python
dict[str, float | torch.Tensor]
```

core 组件：

- `EvaluatorABC`：继承 `torch.nn.Module` 的无状态 evaluator 抽象基类，子类实现 `evaluate()`。
- `MetricValue`：只允许 Python `float` 或 0 维 `torch.Tensor`。
- `MetricDict`：`dict[str, MetricValue]`，不再额外维护 metric map / mapping 类型。
- `EvaluatorGroup`：组合多个返回 metric mapping 的 `nn.Module`，内部用 `nn.ModuleDict` 注册子 evaluator，处理 key 校验并代理可选的状态生命周期。

`EvaluatorGroup` 接收 evaluator mapping，并通过 `evaluators` 属性注册子模块：

```python
evaluator = EvaluatorGroup(
    {
        "error": ErrorEvaluator(),
        "quality": QualityEvaluator(),
    }
)
metrics = evaluator(prediction, target)
```

输出 key 会包含 evaluator 名称和 metric key，例如：

```text
error/mae
quality/snr
```

metric dict 必须是一层 `dict[str, MetricValue]`。metric key 和 evaluator name 不允许包含 separator，默认是 `/`。evaluator name 还需要满足 `nn.ModuleDict` 的原生命名规则。

无状态 evaluator 继承 `EvaluatorABC`，只实现单步 `evaluate()`：

```python
class BatchMAE(EvaluatorABC):
    def evaluate(self, prediction, target):
        return {"mae": (prediction - target).abs().mean()}
```

`EvaluatorABC.__call__()` 和 `EvaluatorGroup.forward()` 会调用 `evaluate()`，校验并返回合并后的 metric dict，不写入内部状态。

`EvaluatorABC` 默认不假设 metric 能按 batch scalar 合并。`update()`、`compute()`、`reset()`
是 stateful evaluator 的扩展点；没有实现状态的子类调用这些方法时会明确抛出
`NotImplementedError`。需要状态的子类必须自己实现完整生命周期，并按指标定义保存充分状态。
例如 corpus BLEU、WER 和 chrF 不能通过等权平均 batch scalar 得到。

即时调用的返回值校验规则：

- 返回值必须是非空 `dict`。
- key 必须是非空字符串。
- key 不能包含 `EvaluatorABC.metric_key_separator`，默认是 `/`。
- value 必须是 Python `float` 或 0 维 `torch.Tensor`。
- Python `int`、`bool` 和非 0 维 tensor 会直接抛错。

`EvaluatorGroup.compute()` 合并 stateful evaluator 的结果时也执行同一套 schema 校验；直接
调用某个自定义 evaluator 的 `compute()` 则由该实现自己负责返回值校验，基类不会自动包装。

`EvaluatorGroup.update/compute/reset` 会先确认所有子 evaluator 都实现了完整状态生命周期，
再逐个代理；因此混入无状态 evaluator 时会在修改任何子状态前抛出 `NotImplementedError`，
不会留下部分更新，也不会静默跳过或伪造 running metric。

`TextComparisonEvaluator` 实现了正确的 corpus 状态生命周期：

```python
evaluator.update(prediction_text_batch, target_text_batch)
metrics = evaluator.compute()
evaluator.reset()
```

它在 `update()` 时保存规范化后的 Python 文本，不保存 batch metric 或 GPU tensor；`compute()`
对所有已保存文本一次计算 corpus BLEU、WER 和 chrF，因此结果不受 batch 大小或切分方式影响。
如果 `torch.distributed` 已初始化，所有 rank 必须共同调用 `compute()`；实现会通过
`all_gather_object` 汇总各 rank 的文本 corpus，使每个 rank 得到相同的全局指标。`reset()` 只清空
当前 rank 的本地文本状态。某个 rank 可以没有本地样本，只要聚合后的全局 corpus 非空；所有
rank 都为空时 `compute()` 会明确抛出 `ValueError`。

`EvaluatorGroup` 接受任意 `nn.Module`，只要即时调用返回合法的 metric mapping；这允许下游
直接接入结构兼容的第三方 metric module。plain callable 不会被接受，因为组合器需要通过
`nn.ModuleDict` 正确注册设备和状态。调用 group 的 stateful 生命周期时，每个子模块仍必须
完整实现 `update()`、`compute()` 和 `reset()`。

## 依赖分层

core evaluator 只依赖默认依赖，提供通用 ABC 和轻量组合器。

当前 optional 子模块：

- `anytrain.evaluator.text`：文本生成 evaluator。
- `anytrain.evaluator.speech`：ASR、UTMOS 和 speech-to-text reference metric evaluator。

规划中的 optional 子模块：

- `anytrain.evaluator.metrics`：torchmetrics-backed 通用分类/回归指标。
- `anytrain.evaluator.audio`：codec、speech、audio quality evaluator。

optional 子模块缺依赖时应抛出清晰错误，提示安装对应 extra。

## Optional Text/Speech Evaluator

`anytrain.evaluator.text` 提供 `TextComparisonEvaluator`，比较 prediction/reference text，并返回：

- `bleu`：0-100 区间 sacreBLEU smoothed effective-order BLEU。
- `wer`：jiwer word error rate。
- `chrf`：0-100 区间 sacreBLEU chrF。

默认文本归一化会 strip、collapse whitespace、remove punctuation，并把中文繁体归一到简体；
`lowercase` 仍需显式打开。
安装 `jiwer` 和 `sacrebleu`（`python -m pip install jiwer sacrebleu`）可使用 sacreBLEU/jiwer backend。缺少
`sacrebleu` 或 `jiwer` 时仍可构造 text evaluator，但会显式 warning 并使用
`text._fallback` 的轻量实现。

`anytrain.evaluator.speech` 提供：

- `SpeechEvaluator`：组合 `WhisperASREvaluator` 和 `UTMOSEvaluator`，一次返回 `bleu`、`wer`、`chrf`、`utmos`。
- `WhisperASREvaluator`：默认用 `openai-whisper` 的 `large-v3` 转写 audio；通过 allowlisted `model_name`、`device`、`download_root`、`load_options` 选择官方 Whisper 配置，不接受本地 checkpoint path，也不开放任意 ASR backend/model 注入；再复用 text evaluator 输出 `bleu`、`wer`、`chrf`。
  短音频 batch 会在兼容选项下使用 Whisper 的 batched 30 秒 mel decode；长音频或需要完整 transcribe 调度的选项会回退到逐条 `model.transcribe()`。
- `UTMOSEvaluator`：默认用 `torch.hub` 加载 `tarepan/SpeechMOS` 的 `utmos22_strong`，也可显式注入 UTMOS backend；输出 `utmos`。

`anytrain.evaluator.speech` 顶层只导出 evaluator。ASR 的 Whisper loader 是私有实现；
UTMOS backend 细节从 `anytrain.evaluator.speech.utmos` 显式导入。

默认 backend 都是延迟加载：构造 evaluator 不会下载或加载模型，第一次 `evaluate()` /
`transcribe()` 才会访问模型依赖。需要真实 speech evaluator 时安装 `jiwer`、`openai-whisper`、`sacrebleu`、`soundfile` 和 `torchaudio`（`python -m pip install jiwer openai-whisper sacrebleu soundfile torchaudio`）。
需要控制 checkpoint、cache 或 device 时，优先通过系统环境变量或 evaluator 的显式配置项传入。
如果未设置，anytrain 会把 `HF_HOME`、`TORCH_HOME`、`ANYTRAIN_WHISPER_ROOT`
默认到 `${ANYTRAIN_HOME:-~/.anytrain}` 下。
默认 torch backend 会在调用前将模型 `requires_grad_(False)`、`eval()`，并在
`torch.inference_mode()` 内执行模型前向或转写。

## 边界

`evaluator` 不做：

- 不解释 batch schema。
- 不决定 validation/test step 的运行方式。
- 不直接操作 Trainer。
- 不把领域 metric 放进 core import。
- 不隐藏第三方 metric backend 的状态生命周期。
- 不为任意 batch scalar 猜测聚合规则或 batch 权重。

## 测试策略

当前覆盖：

- 单个 evaluator 返回 dict。
- 多 evaluator 组合和 key 校验。
- 无状态 `EvaluatorABC` 的 `update/compute/reset` 明确错误。
- text corpus metric 在不等 batch 切分下与直接整 corpus 计算一致。
- text evaluator 的 `update/compute/reset`、group 代理和分布式 corpus 聚合。
- 错误返回类型、空 metrics、非 0 维 Tensor、Python 非 float 标量和非 ABC evaluator。

后续实现 metrics/audio evaluator 后，需要补充 optional 缺依赖时的错误信息。

当前 optional 覆盖：

- text evaluator 的单条、batch、normalization、空 reference、长度不一致和 metric key。
- speech evaluator 的默认 backend 延迟构造、ASR 配置项、ASR reference 校验、text metric 透传、UTMOS batch mean 和真实 backend 输出规范化。
