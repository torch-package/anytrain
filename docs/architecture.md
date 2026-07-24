# anytrain Architecture

## 定位

`anytrain` 是给用户编写 PyTorch/Lightning 训练代码时使用的组件库，重点服务普通 `LightningModule` 的实现。它提供下游 `LightningModule` 里可能用到的可组合积木，例如 logging mixin、debug callback、loss/evaluator 组合器、optimizer/scheduler helper 和 task-agnostic `torch.nn.Module`。

它的设计重点是帮助用户写自己的训练模块，而不是接管训练工程入口。下游项目自己决定配置系统、对象装配、运行目录、checkpoint resume 和 `Trainer.fit()` 调用；`anytrain` 不提供默认 CLI/app 层，也不要求下游继承自定义 `LightningModule` 基类。

核心目标：

- 提供写 `LightningModule` 时常用的 runtime helper。
- 提供可显式组合进训练模块的 loss/evaluator/logging/framework 组件。
- 提供 optimizer/scheduler 和 task-agnostic module 积木。
- 通过 optional dependencies 提供领域组件，而不是把所有重依赖压进 core。
- 保持入口和配置装配由用户项目显式拥有。

## 模块准入

新增能力先按对外契约归类，而不是按当前调用项目归类：

- `core`：跨任务复用、只依赖默认依赖，并直接服务训练模块实现。
- `framework`：跨项目复用的训练范式，必须保持可组合，不生成完整任务或训练入口。
- `domain adapter`：封装 codec、TTS、speech 等领域 backend，放显式 optional 子模块。
- `runtime utility`：chat、tokenizer、example 等训练工程辅助能力，不能进入根包或 core import。

新模块必须同时给出公开输入输出、依赖 extra、缺依赖错误、最小 CPU 测试和文档入口。只服务
单个实验、绑定私有 batch schema 或需要 workspace 路径的逻辑不进入 `anytrain`。

## 分层

### Core

默认安装必须可用：

- `anytrain.lightning`：LightningModule logging mixin、debug callback 和性能观测 callback。
- `anytrain.lightning.perf`：训练效率观测 helper，包括参数量、FLOPs、硬件峰值算力和 MFU。
- `anytrain.loss`：通用 loss 接口和组合器。
- `anytrain.evaluator`：通用 evaluator 接口和组合器。
- `anytrain.optim`：optimizer、scheduler 和 LLM/Muon helper。
- `anytrain.module`：少量稳定的 task-agnostic `torch.nn.Module` 积木。
- `anytrain.module.dynamic_conv`、`anytrain.module.quantization`、`anytrain.module.qwen`：显式子模块组件入口。
- `anytrain.module.idspace`：local/global token id 映射和 block embedding 路由。

Lightning、Torch 和 einops 是 core 依赖。`module` 顶层只导出少量稳定组件；dynamic conv、
quantization、Qwen 和 idspace 从子模块显式导入。只有 Qwen3 builder 需要 `module` extra。Muon 是否可用取决于当前
PyTorch 是否提供 `torch.optim.Muon`，不影响其余 core API。

### Optional General

当前已实现的组件不绑定具体领域，但需要额外依赖：

- `chat`：环境变量驱动的大模型调用入口，用于训练工程里的实验总结、日志解释和 LLM 辅助评估原型。
- `tokenizer`：基于 Hugging Face `tokenizers` 的 codec frame BPE。
- third-party logger backend：wandb、mlflow 等。

规划中的 optional general 组件：

- metrics evaluator：基于 torchmetrics 的通用分类/回归指标。

### Optional Domain

这些领域组件由 `anytrain` 提供，并从对应子模块显式导入。需要第三方 backend 的组件安装
对应 extra；只依赖 core torch 的 temporal/spectral loss 不额外要求 audio extra：

- temporal/spectral audio loss：torch backend 使用 core；torchaudio backend 安装 `audio` extra。
- codec wrapper：DAC、LongCat、UniCodec 分别使用 `dac`、`longcat`、`unicodec` backend
  extra；`audio` extra 只提供 torchcodec/torchaudio，不会安装 codec 模型。Stable Codec 的
  torch pin 与 core 冲突，按 `docs/stable-codec.md` 使用独立兼容环境。
- text evaluator：安装 `text` extra 使用 sacreBLEU/jiwer backend；无 extra 时有显式 warning
  的轻量 fallback。
- speech evaluator：安装 `speech` extra 使用 Whisper、UTMOS 和 speech text metrics。
- TTS：公共协议保留轻量，MOSS adapter 安装 `moss-tts` extra。

### Optional Framework

当前提供的 optional framework：

- flow matching。
- GAN adversarial objective、feature matching、WGAN-GP 和 DAC discriminator preset。

只有跨项目复用需求明确后再考虑：

- masked autoencoder。
- GAN manual optimization helper。
- 其他跨项目复用的训练范式。

这些不进入 core，避免 `anytrain` 的默认依赖和默认心智负担膨胀。

## 运行边界

`anytrain` 不定义固定运行链路。一个下游入口通常会自己完成：

1. 读取项目配置，或直接用 Python 构造对象。
2. 设置随机种子、matmul precision 等运行环境。
3. 创建下游 `LightningModule` 和 data module / dataloader。
4. 创建 `lightning.pytorch.Trainer`，设置 `default_root_dir`、logger、callback 和 resume 参数。
5. 调用 `trainer.fit(...)`。

这些步骤可以用普通 Python、Hydra、argparse、pydantic 或项目自己的配置系统实现。`anytrain` 只要求组件作为普通 Python 对象显式传入下游 `LightningModule`，不假设配置树形状，也不自动实例化对象。

## Lightning 接口

下游 pl module 直接继承 Lightning 原生基类：

```python
from lightning import pytorch as pl


class MyPLModule(pl.LightningModule):
    ...
```

`anytrain` 不接管组件注入。复杂项目可以把多个子模块、损失、优化器工厂、scheduler 工厂、辅助头或 evaluator 直接声明为 `pl_module` 参数，也可以完全在代码中构造。

更完整的 Lightning 设计见 `docs/lightning.md`。逐模块设计文档见 `docs/modules/index.md`。

## 模块定位

### `lightning`

提供 LightningModule logging mixin 和 callback。logger backend 由下游或 Lightning 原生配置负责创建；第三方 logger backend 是 optional。

### `lightning.perf`

提供 task-agnostic 的训练效率观测能力。`anytrain.lightning.perf` 可以统计参数量、估算代表性 forward FLOPs、查表推断硬件峰值算力，并通过 Lightning callback 记录 step time 和 MFU。下游项目仍负责提供代表性输入、训练 step FLOPs 估计口径，以及 tokens/frames 等任务数据量日志。

### `loss`

提供训练 step 中可直接使用的 loss 组件。core 里保留通用组合器；audio/text/speech 等领域 loss 通过 optional 子模块提供。GAN adversarial training 属于 `framework.gan`，不放在 `loss` 下。

### `evaluator`

提供 validation/test/training step 中可直接使用的 metric/evaluator 组件。core 里保留接口和组合器；codec/text/speech 等领域 evaluator 通过 optional 子模块提供。

### `optim`

提供 optimizer/scheduler helper。下游仍在自己的 `configure_optimizers()` 里显式调用 helper，并按 Lightning 原生格式返回。

### `module`

提供下游 LightningModule 可显式组合的 task-agnostic `torch.nn.Module` 积木。`anytrain.module` 顶层只放 ADT、DiT 这类少量稳定组件；dynamic conv、quantization、Qwen 和 idspace 从对应子模块导入。`einops` 是默认依赖，用于保持动态层 shape 变换可读；需要其它额外依赖的组件通过 `module` extra 暴露，不进入 package root import。

### `module.idspace` / `tokenizer`

`anytrain.module.idspace` 是 core 的 local/global token id 映射和 embedding 路由。`tokenizer` 是 optional
general 算法层；CodecBPE 需要 `tokenizer` extra，不进入 package root import。

### `codec` / `tts`

`codec` 顶层只提供轻量 protocol；具体 audio codec wrapper 从 `anytrain.codec.*` 子模块导入。`tts` 的协议和输出类型保持轻量，具体 backend adapter 按依赖 extra 隔离。两者都不接管数据集、训练 step 或模型组合。

### `chat`

提供环境变量驱动的大模型调用入口，通常依赖 `chat` extra。它负责把显式 prompt 和实例内消息上下文发给指定 provider，不接管 prompt 模板、任务 schema、provider cache 观测或训练流程。

### `framework`

提供跨项目复用的训练范式，作为 optional/experimental 层，不进入 core。这里的 `framework` 是研究范式组件层，例如 flow matching、MAE 或 GAN helper；不是训练工程入口。

## 包结构

```text
src/anytrain/
  lightning/
    callback/
    mixin/
    perf/
  loss/
  evaluator/
  optim/
  module/
    dynamic_conv/
    idspace/
    quantization/
  chat/
  tokenizer/
  codec/
  tts/
  framework/
  example/
```

`lightning`、`lightning.perf`、`loss`、`evaluator`、`optim`、`module` 和 `module.idspace` 是核心体验；`chat`、
`framework` 和领域组件按依赖拆分为 optional 子模块。

## 边界

`anytrain` 提供组件，但不做这些事：

- 替用户写完整 task module。
- 提供默认训练 CLI、app 层或配置装配协议。
- 替用户规定 batch schema。
- 把 optimizer/scheduler 作为顶层硬配置自动注入。
- 把模型 zoo 或预训练下载逻辑放进 core。
- 把所有 audio/text/speech/chat/framework 依赖放进默认安装。
- 静默兼容缺失依赖；optional 子模块应明确说明实际缺失的包，并给出可直接执行的安装命令。
