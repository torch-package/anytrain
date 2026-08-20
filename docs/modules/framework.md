# `anytrain.framework` Design

## 定位

`anytrain.framework` 是 optional/experimental 研究框架层，用于沉淀跨项目复用的训练范式组件，例如 flow matching、masked autoencoder 或 GAN helper。

它不是 core 层，也不替下游项目生成完整 LightningModule。不要把这里的 `framework` 和训练工程入口混在一起：入口由下游项目自己维护，`anytrain.framework` 是可选训练范式积木。

## 当前状态

当前已有：

- `framework.flow_matching`：Facebook `flow_matching` 的可组合再封装。
- `framework.gan`：adversarial training loss、feature matching、WGAN-GP 和 DAC discriminator preset。
- `framework.rl`：DPO、token GRPO、continuous Flow-GRPO、neighbor GRPO 和最小 rollout
  tensor contract；不拥有 rollout、reward、dataset 或下游 batch schema。

`framework.flow_matching` 的设计见 [`docs/flow-matching-design.md`](../flow-matching-design.md)。它作为 Facebook `flow_matching` 的可组合再封装，提供 source、time sampler、objective 和 sampler 等积木，而不是完整任务框架。

`ContinuousVelocityObjective` 和 `DiscreteGeneralizedKLObjective` 固定返回 0-D Tensor，并按
concrete type 注册推荐的 `OutputObservation("loss", reduction=observation.Reduction.Mean)`；自定义 continuous loss 只观察其已归约
输出，不重算或猜测 per-item 统计。

`framework.gan` 的设计见 [`docs/gan-framework-design.md`](../gan-framework-design.md)。当前提供轻量 manual optimization helper，但不接管完整训练循环；音频只提供 DAC 风格 discriminator preset，不接管 codec 任务训练。

`framework.rl` 的设计见 [`docs/rl-framework-design.md`](../rl-framework-design.md)。它只消费
下游已计算的 log-prob、reward 和 mask tensor；RL-specific batch record、采样与评分流程
留在下游项目。

## 当前结构

```text
src/anytrain/framework/
  __init__.py
  _validation.py
  flow_matching/
  gan/
    __init__.py
    _output.py
    criterion.py
    feature.py
    manual.py
    module.py
    penalty.py
    audio/
      __init__.py
      dac.py
    types.py
  rl/
    __init__.py
    dpo.py
    grpo.py
    continuous_grpo.py
    neighbor_grpo.py
    advantage.py
    logprob.py
    types.py
```

`anytrain.framework.__init__` 不默认导出 framework 子模块。用户显式导入：

```python
from anytrain.framework.gan import GAN, Loss, Preset, Reduction
from anytrain.framework.flow_matching import (
    ContinuousFlowRuntime,
    ContinuousVelocityObjective,
    DiscreteFlowRuntime,
    DiscreteGeneralizedKLObjective,
)
from anytrain.framework.rl import (
    ContinuousGRPOLoss,
    DPOLoss,
    GRPOLoss,
    NeighborGRPOLoss,
)
```

`_validation.py` 是 framework 内部 host scalar contract 的唯一 owner。flow source/time/sampler、
GAN loss 权重和 RL loss 超参数都在公开入口拒绝 bool、Tensor、字符串与非有限实数；算法文件只保留
各自的范围约束。`validate`、`reference_free`、sampler logging/trajectory 开关严格要求 bool，避免
配置中的 `0`、`1` 或字符串被静默解释。

flow matching、GAN 和 RL 的训练生命周期、张量布局及数值目标不同，因此保持三个独立子模块；
不抽取跨算法基类，也不按文件长度拆散 tensor math。RL 的 `validate=False` 仍是显式 trusted path，
只跳过 batch tensor 的有限性与 mask 检查，不放宽公开控制项的类型。

## GAN

`framework.gan` 放 adversarial training 需要的通用组件和轻量 manual optimization helper，但不接管完整训练循环、batch schema 或 generator forward。

当前提供：

- `Loss`：持有 discriminator，提供 `discriminator_loss(fake, real)` 和 `generator_loss(fake, real=None)`。
- `GAN`：`StrEnum`，当前支持 `GAN.Hinge`、`GAN.LSGAN`、`GAN.WGAN`。
- `Loss.from_preset(Preset.DAC, **kwargs)`：创建 DAC 风格 MPD / MRD discriminator；`kwargs` 转发给 `DACDiscriminator`。
- `anytrain.framework.gan.audio.DACDiscriminator`：可直接使用的音频判别器。
- `discriminator_step()` / `generator_step()`：不绑定 Lightning 的 manual optimization helper；D step 默认合并 real/fake batch，G step 自动临时冻结 discriminator 参数。
- `generator_step(..., eval_discriminator=True)`：在 G step 期间额外临时切换 discriminator 到 eval mode，防止 BatchNorm 等运行统计更新，并恢复原状态。
- `freeze_parameters()`：保留原始 `requires_grad` 状态的上下文管理器，可选临时切换 eval mode。
- `should_apply_gradient_penalty()`：按 discriminator step 间隔实现 lazy GP。
- 私有 `_LogitCriterion` / `_FeatureMatching` / `_GradientPenalty`：作为 `Loss` 的内部实现，不作为稳定 public API。

示例：

```python
from anytrain.framework.gan import GAN, Loss, Preset, Reduction

loss_fn = Loss(
    discriminator=discriminator,
    gan=GAN.Hinge,
    reduction=Reduction.Mean,
    feature_weight=1.0,
)

d_loss = loss_fn.discriminator_loss(fake, real)
g_loss = loss_fn.generator_loss(fake, real)
```

性能选项是显式 opt-in，不改变现有默认行为：

```python
d_loss = loss_fn.discriminator_loss(fake, real, combine_inputs=True)
g_loss = generator_step(
    loss_fn, opt_g, self.manual_backward,
    lambda: rec_loss + gan_weight * loss_fn.generator_loss(fake, real),
)
gp_interval = 4
d_loss = discriminator_step(
    loss_fn, fake, real, opt_d, self.manual_backward,
    gradient_penalty=should_apply_gradient_penalty(d_step, interval=gp_interval),
    gradient_penalty_scale=gp_interval,
)
```

`combine_inputs=True` 要求判别器保持 batch 维度，并把 real 放在拼接 batch 的前半段；
含 BatchNorm 或依赖 batch 统计的判别器应保持默认 `False`。lazy GP 只控制本次是否计算；
要保持期望 regularization 权重，应在执行 step 使用 `gradient_penalty_scale=interval`。没有正的
`gp_weight` 时不会凭空启用 GP。

两个入口都返回 0-D Tensor。`Loss.forward(..., objective=...)` 是它们共享的 scalar
内核，兼容方法仍保留，因而已有 manual optimization 接线只需删除 tuple 解包。

DAC preset：

```python
loss_fn = Loss.from_preset(Preset.DAC, feature_weight=1.0, in_channels=1)
```

判别器 forward 需要直接返回 `Sequence[Sequence[Tensor]]`。每个 branch 的最后一个
tensor 是 logits，前面 tensor 是 feature maps；简单判别器返回 `[[logits]]`。

边界：

- `discriminator_loss()` 内部会 detach fake，避免 D step 回传到 generator。
- `generator_loss()` 不 detach fake，让 generator 可以收到 adversarial 和 feature matching 梯度。
- `Loss` 不自动 freeze discriminator 参数；`generator_step()` helper 会在 G objective 期间完成 freeze/unfreeze。
- `generator_step()` 会在 generator forward/backward 期间自动 freeze discriminator，并在退出后恢复每个参数原来的 `requires_grad` 状态。
- `GAN.WGAN` 默认启用 `gp_weight=10.0` 的 gradient penalty；其他 GAN 默认不启用。显式 `gp_weight=0.0` 可以关闭。
- WGAN-GP 要求 discriminator 不包含 BatchNorm 或其他跨样本耦合层，否则一次 `autograd.grad` 不能得到独立的 per-sample gradient norm。
- `reduction` 公开入口支持 `Reduction` enum 或 torch 风格字符串；内部统一使用 `Reduction`，默认 `Reduction.Mean`。
- DAC preset 只提供判别器和 loss 计算，不接管 codec 模型、batch schema 或重建 loss 组合；其默认 periods、FFT、bands 和 channel dimensions 保持第三方实现原值。

`GAN Loss`、`DPOLoss`、`GRPOLoss`、`ContinuousGRPOLoss` 和 `NeighborGRPOLoss` 都在
具体 module 定义旁注册推荐 observation。总 loss 从 scalar forward output 采集；本次 forward
才存在的分量和统计量通过 `ForwardEvent` 发出，不保存 `last_*` tensor，也不再通过返回值携带
logging payload。挂载 `anytrain.lightning.ObservationCallback` 后即可自动发现这些声明。

每次 forward 必须覆盖该 objective 的完整 reduction domain。已经归约的 scalar 不能在不同
shard、rank 或 microbatch 的有效 count 不同时事后精确合并；这种场景必须在 objective 内获得
全局 normalizer，或由下游在进入 objective 前构造完整 reduction domain。

## 适合放入的内容

- 可跨多个项目复用的训练目标或采样调度。
- 可组合进下游 LightningModule 的 helper。
- 明确依赖 extra 和测试边界的研究范式模块。
- 与 `loss`、`evaluator` 可组合但不强耦合的组件。

## 不适合放入的内容

- 完整任务工程模板。
- 要求下游继承特定 LightningModule 基类的大框架。
- 项目私有 batch schema。
- 项目私有模型 zoo。
- 数据集适配逻辑。
- 只在单个下游实验中使用的一次性代码。

## 依赖策略

framework 子模块默认不进入 core import。每个子模块需要声明自己的 extra 或安装说明，缺依赖时应直接报错并提示安装方式。

示例边界：

- `framework.flow_matching` 可以依赖 flow matching 所需的数值 helper，但不应依赖 audio/text 特定组件。
- `framework.gan` 可以提供 manual optimization helper，但具体 discriminator/generator 架构由下游项目提供。
- `framework.mae` 可以提供 mask 生成和重建损失辅助，但不接管完整数据 pipeline。

## 测试策略

每个 framework 子模块至少需要：

- 一个最小 CPU 单测，验证核心数学或状态转换。
- 一个与下游 LightningModule 可组合的 smoke test。
- optional 依赖缺失时的错误路径测试。
- 文档示例和测试配置保持一致。

当前覆盖：

- `framework.flow_matching` 的 continuous/discrete 最小路径。
- `framework.gan` 的常见公式、判别器输出契约、feature matching detach、D/G 梯度边界、WGAN-GP 和 DAC preset smoke test。
- `framework.rl` 的 DPO、GRPO、continuous / neighbor GRPO 数学、mask reduction、KL、
  group selection、非有限值边界和 policy-only 梯度路径。
- framework 共享 host scalar contract 的 bool/type/finiteness/range 边界，以及 RL trusted path。
