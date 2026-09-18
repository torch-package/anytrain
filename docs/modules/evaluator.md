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
  types.py
  weighted.py
  speech/
    __init__.py
    _config.py
    _torch.py
    _whisper.py
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
  translation/
    __init__.py
    comet.py
```

当前公开导出：

- `EvaluatorABC`
- `MetricValue`
- `MetricDict`
- `EvaluatorGroup`
- `weighted`：逐行 values 与 observation counts 的通用加权指标契约。

`MetricValue` 和 `MetricDict` 的定义归 `types.py` 所有，并从 package root 重导出。内部实现
直接从 `types.py` 引用类型，不通过 `abc.py` 的偶然 import 形成隐藏入口。metric/evaluator
名称的公共校验规则集中在 `_validation.py`。

optional 子模块导出：

- `anytrain.evaluator.text.TextComparisonEvaluator`
- `anytrain.evaluator.text.TextInput`
- `anytrain.evaluator.text.TextNormalizationConfig`
- `anytrain.evaluator.speech.SpeechEvaluator`
- `anytrain.evaluator.speech.SpeechEvaluatorConfig`
- `anytrain.evaluator.speech.SpeechEvaluatorLoadOptions`
- `anytrain.evaluator.speech.WhisperASREvaluator`
- `anytrain.evaluator.speech.WhisperSemanticConfig`
- `anytrain.evaluator.speech.WhisperLoadOptions`
- `anytrain.evaluator.speech.UTMOSEvaluator`
- `anytrain.evaluator.speech.UTMOSSemanticConfig`
- `anytrain.evaluator.speech.UTMOSLoadOptions`
- `anytrain.evaluator.translation.COMET_PROTOCOL`
- `anytrain.evaluator.translation.CometBackend`
- `anytrain.evaluator.translation.CometEvaluator`
- `anytrain.evaluator.translation.CometLoadOptions`
- `anytrain.evaluator.translation.CometResult`
- `anytrain.evaluator.translation.CometSample`
- `anytrain.evaluator.translation.CometSemanticConfig`

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
- `weighted.Metric`：保存 shape 对齐的逐行 values/weights；weights 只能是有限、非负的整数
  observation count。`reduced()` 返回加权均值与精确 count；`weighted.reduce()` 批量归约指标，并让
  共享同一 weights tensor 的指标只同步读取一次 count。

整数值的浮点 weights（包括 FP16/BF16）同样可用；校验后用 int64 精确累计 count，
总数必须在 `1` 到 `2**53 - 1` 之间，超出范围会在整数转换前抛出 `ValueError`。
该范围保护 count 的精确表示，不承诺浮点加权均值没有舍入误差。

加权乘加至少使用 FP32，FP64 values 则保持 FP64，避免 FP16/BF16 的大 observation count 溢出或量化。
归约会先把 count 为 0 的位置清零，因此 padding 位置即使含 NaN/Inf 也不会污染有效均值；正权重位置的
非有限值仍会保留并由下游显式暴露。Lightning epoch/DDP 生命周期由
`anytrain.lightning.validation.log()` 负责，不放进这个纯数值契约。

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
例如 corpus BLEU、WER、chrF 和 CER 不能通过等权平均 batch scalar 得到。

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

`EvaluatorABC`、`EvaluatorGroup` 和 optional evaluator 共用同一个 `MetricDict` schema。
`EvaluatorGroup` 仍保持普通 `nn.Module`，不伪装成只实现 `evaluate()` 的
`EvaluatorABC`；它负责组合和 stateful lifecycle 预检，ABC 负责单 evaluator 的调用边界。

`TextComparisonEvaluator` 实现了正确的 corpus 状态生命周期：

```python
evaluator.update(prediction_text_batch, target_text_batch)
metrics = evaluator.compute()
evaluator.reset()
```

它在 `update()` 时保存规范化后的 Python 文本，不保存 batch metric 或 GPU tensor；`compute()`
对所有已保存文本一次计算 corpus BLEU、WER、chrF/chrF++ 和 CER，因此结果不受 batch 大小或切分方式影响。
如果 `torch.distributed` 已初始化，所有 rank 必须共同调用 `compute()`；实现会通过
`gather_object` 把各 rank 的文本 corpus 汇集到 rank 0，只由 rank 0 计算四个全局指标，再把统一
metric dict 或统一错误广播给所有 rank。`evaluate()` 仍是 rank-local 的无状态调用，不参与该聚合；
`reset()` 只清空当前 rank 的本地文本状态。某个 rank 可以没有本地样本，只要聚合后的全局 corpus
非空；各 rank 的 normalization、BLEU 和 chrF 配置必须一致。所有 rank 都为空或配置不一致时，
所有 rank 会收到包含原始错误类型和消息的统一 `RuntimeError`。

`EvaluatorGroup` 接受任意 `nn.Module`，只要即时调用返回合法的 metric mapping；这允许下游
直接接入结构兼容的第三方 metric module。plain callable 不会被接受，因为组合器需要通过
`nn.ModuleDict` 正确注册设备和状态。调用 group 的 stateful 生命周期时，每个子模块仍必须
完整实现 `update()`、`compute()` 和 `reset()`。

## 依赖分层

core evaluator 只依赖默认依赖，提供通用 ABC 和轻量组合器。

当前 optional 子模块：

- `anytrain.evaluator.text`：文本生成 evaluator。
- `anytrain.evaluator.speech`：ASR、UTMOS 和 speech-to-text reference metric evaluator。
- `anytrain.evaluator.translation`：reference-based COMET 翻译质量 evaluator。

规划中的 optional 子模块：

- `anytrain.evaluator.metrics`：torchmetrics-backed 通用分类/回归指标。
- `anytrain.evaluator.audio`：codec、speech、audio quality evaluator。

optional 子模块缺依赖时应抛出清晰错误，提示安装对应 extra。

`text/_chinese.py` 是静态繁简字符映射数据，不承担 evaluator 控制逻辑；其文件长度不作为
拆分依据。speech 的 semantic config、load options 和 backend adapter 使用现有分层，core
metric schema 不反向依赖这些 optional 实现。

## Optional Text/Speech Evaluator

`anytrain.evaluator.text` 提供 `TextComparisonEvaluator`，比较 prediction/reference text，并返回：

- `bleu`：0-100 区间 sacreBLEU smoothed effective-order BLEU；为兼容已有调用，`bleu_tokenize`
  默认 `none`。翻译评测应显式选择 tokenizer：中文目标使用 `zh`，其他标准空格语言通常使用 `13a`；
  `none` 仅用于已经按空格预分词的文本。
- `wer`：jiwer word error rate。
- `chrf`：0-100 区间 sacreBLEU chrF；为兼容已有调用，默认 character order 6、word order 0、
  beta 2。翻译评测使用 chrF++ 时显式设置 `chrf_word_order=2`。
- `cer`：jiwer character error rate（以 reference character 数为 corpus 分母）。

当整个 corpus 的 reference 都为空时，CER 沿用 jiwer 的定义返回插入字符数；prediction 和
reference 都为空时返回 `0.0`。因此该边界结果不应被解释为比例。

默认文本归一化会 strip、collapse whitespace、remove punctuation，并把中文繁体归一到简体；
`lowercase` 仍需显式打开。
`metric_signatures()` 返回当前 backend 的 one-reference sacreBLEU BLEU/chrF signature，发布结果时应和
`bleu_tokenize`、chrF beta 等 profile 参数一起持久化。不同 tokenizer/profile 的分数不得横向比较。
安装 `jiwer` 和 `sacrebleu`（`python -m pip install jiwer sacrebleu`）可使用 sacreBLEU/jiwer backend。缺少
`sacrebleu` 或 `jiwer` 时仍可构造 text evaluator，但会显式 warning 并使用
`text._fallback` 的轻量实现。

`anytrain.evaluator.speech` 提供：

- `SpeechEvaluator`：组合 `WhisperASREvaluator` 和 `UTMOSEvaluator`，一次返回 `bleu`、`wer`、`chrf`、`cer`、`utmos`。
- `WhisperASREvaluator`：默认用 `openai-whisper` 的 `large-v3` 转写 audio；通过 allowlisted `model_name`、`device`、`download_root`、`load_options` 选择官方 Whisper 配置，不接受本地 checkpoint path，也不开放任意 ASR backend/model 注入；再复用 text evaluator 输出 `bleu`、`wer`、`chrf`、`cer`。
  短音频 batch 会在兼容选项下使用 Whisper 的 batched 30 秒 mel decode；长音频或需要完整 transcribe 调度的选项会回退到逐条 `model.transcribe()`。
- `UTMOSEvaluator`：可用 `torch.hub` 加载 `tarepan/SpeechMOS` 的 `utmos22_strong`，也可显式注入 UTMOS backend；输出 `utmos`。远端 TorchHub 代码加载需要显式传入 `allow_remote_code=True`。

`anytrain.evaluator.translation` 提供 `CometEvaluator`，使用 `source`、模型 `hypothesis` 和人工
`reference` 三元组计算 reference-based COMET。它同时支持即时 `evaluate()`、逐 segment 的
`score()` 以及跨 batch 的 `update()/compute()/reset()`；`compute()` 在 distributed 已初始化时先
把各 rank 的三元组 gather 到 rank 0，由 rank 0 执行一次完整 corpus 的 batched inference 和
system-level aggregation，再把结果或统一错误广播给所有 rank；不能平均各 batch 的 system score。
所有 rank 必须共同调用 `compute()`。`score()` 和 `evaluate()` 仍是 rank-local，调用各自 rank 的
backend 一次，不参与该 distributed 聚合。
COMET 是重量级 optional backend：安装 `anytrain[translation-eval]`，并通过
`CometSemanticConfig` 固定 checkpoint model/revision/weights SHA-256、backend version、文本
normalization 和 batch size。构造 evaluator 不会加载模型；真实 checkpoint 只在首次评分时校验摘要并加载。

```python
from anytrain.evaluator.translation import (
    CometLoadOptions,
    CometSemanticConfig,
)

config = CometSemanticConfig(
    model="Unbabel/wmt22-comet-da",
    revision="<40-character-model-commit>",
    weights_sha256="<64-character-checkpoint-sha256>",
    backend_version="<installed-unbabel-comet-version>",
)
evaluator = config.evaluator(
    CometLoadOptions(checkpoint="/models/wmt22-comet-da.ckpt", gpus=1),
)
score = evaluator("source", "system output", "human translation")
```

COMET checkpoint 和版本不能使用 `latest` 或未固定的 model alias；不同 scorer identity 的绝对分数
不得直接放入同一排名。缺少人工 source transcript 时，上层 benchmark 应将该 case 标记为 ineligible，
不能用 source audio 的 ASR 派生文本静默替代。

用于持久化 filter/评测 lineage 时，不应从运行对象的 cache/device 参数手写 identity。共享层提供三组
frozen semantic config：

```python
from anytrain.evaluator.speech import (
    SpeechEvaluatorConfig,
    SpeechEvaluatorLoadOptions,
    UTMOSLoadOptions,
    WhisperLoadOptions,
)

config = SpeechEvaluatorConfig()
evaluator = config.evaluator(
    SpeechEvaluatorLoadOptions(
        whisper=WhisperLoadOptions(device="cuda:0", local_files_only=True),
        utmos=UTMOSLoadOptions(
            device="cuda:0",
            allow_remote_code=True,
            local_files_only=True,
        ),
    )
)
content_id = config.semantic_id()
```

`WhisperSemanticConfig` 固定官方 checkpoint SHA-256、`openai-whisper==20250625`、默认 decode
options、文本 normalization、BLEU smoothing 和 AnyTrain decoder/metric protocol。运行时会核对已安装
Whisper 版本及其官方 model URL 中的 artifact digest；不匹配时明确失败。`UTMOSSemanticConfig` 默认固定
`tarepan/SpeechMOS` commit `ed25eacbfa42b99156c36ebec67a733b5dbb9b79`、TorchHub entrypoint 和
`utmos22_strong_step7459_v1.pt` weights SHA-256
`38aa51ab79e2a4e09a1449758a4b37e9cbb2e8235a49662a732d33a9ba1e9bff`。backend 在加载后验证
TorchHub cache 中的 checkpoint digest；组合 `SpeechEvaluatorConfig` 的 identity 递归包含两者。

device、download/cache root、remote-code opt-in、offline policy 和 TorchHub cache controls 只存在于
对应 `LoadOptions`，不进入 semantic identity。config 构造 evaluator 仍是 lazy 的：只有第一次转写或评分
才加载模型。`local_files_only=True` 会在调用第三方 loader 前检查 pinned Whisper checkpoint，或
TorchHub repository/weights cache；缺失时抛出 `FileNotFoundError`，不发起隐式下载。

直接注入自定义 UTMOS backend 仍可用于测试和研究，但共享层不会为未知 backend 猜测 identity；调用
`semantic_id()` 会明确报错。需要 durable identity 时使用 pinned `UTMOSSemanticConfig` 构造内置 backend。

`anytrain.evaluator.speech` 顶层只导出 evaluator 及其 semantic/load config。ASR 的 Whisper loader 是私有实现；
UTMOS backend 细节从 `anytrain.evaluator.speech.utmos` 显式导入。

默认 backend 都是延迟加载：构造 evaluator 不会下载或加载模型，第一次 `evaluate()` /
`transcribe()` 才会访问模型依赖。需要真实 speech evaluator 时安装 `anytrain[speech]`；该 extra 固定
`openai-whisper==20250625`，其余音频和文本指标依赖由项目版本约束管理。
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
- translation evaluator 的 semantic config、custom backend、batch/corpus aggregation、root-only
  distributed inference、结果/错误广播和 optional dependency lazy load。

模块验收命令：

```bash
python -m ruff check src/anytrain/evaluator tests/test_evaluator.py \
  tests/test_evaluator_text.py tests/test_evaluator_speech.py \
  tests/test_evaluator_translation.py
basedpyright --pythonpath /path/to/python src/anytrain/evaluator \
  tests/test_evaluator.py tests/test_evaluator_text.py tests/test_evaluator_speech.py \
  tests/test_evaluator_translation.py
python -m pytest -q tests/test_evaluator.py tests/test_evaluator_text.py \
  tests/test_evaluator_speech.py tests/test_evaluator_translation.py
```
