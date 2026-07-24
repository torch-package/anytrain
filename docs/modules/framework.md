# `anytrain.framework` Design

## 定位

`anytrain.framework` 是 optional/experimental 研究框架层，用于沉淀跨项目复用的训练范式组件，例如 flow matching、masked autoencoder 或 GAN helper。

它不是 core 层，也不替下游项目生成完整 LightningModule。不要把这里的 `framework` 和训练工程入口混在一起：入口由下游项目自己维护，`anytrain.framework` 是可选训练范式积木。

## 当前状态

当前已有：

- `framework.flow_matching`：Facebook `flow_matching` 的可组合再封装。
- `framework.gan`：adversarial training loss、feature matching、WGAN-GP 和 DAC discriminator preset。

`framework.flow_matching` 的设计见 [`docs/flow-matching-design.md`](../flow-matching-design.md)。它作为 Facebook `flow_matching` 的可组合再封装，提供 source、time sampler、objective 和 sampler 等积木，而不是完整任务框架。

`framework.gan` 的设计见 [`docs/gan-framework-design.md`](../gan-framework-design.md)。当前不提供完整 manual optimization helper；音频只提供 DAC 风格 discriminator preset，不接管 codec 任务训练。

## 当前结构

```text
src/anytrain/framework/
  __init__.py
  flow_matching/
  gan/
    __init__.py
    _output.py
    criterion.py
    feature.py
    module.py
    penalty.py
    audio/
      __init__.py
      dac.py
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
```

## GAN

`framework.gan` 放 adversarial training 需要的通用组件，但不接管 manual optimization、batch schema 或 generator forward。

当前提供：

- `Loss`：持有 discriminator，提供 `discriminator_loss(fake, real)` 和 `generator_loss(fake, real=None)`。
- `GAN`：`StrEnum`，当前支持 `GAN.Hinge`、`GAN.LSGAN`、`GAN.WGAN`。
- `Loss.from_preset(Preset.DAC, **kwargs)`：创建 DAC 风格 MPD / MRD discriminator；`kwargs` 转发给 `DACDiscriminator`。
- `anytrain.framework.gan.audio.DACDiscriminator`：可直接使用的音频判别器。
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

d_loss, d_details = loss_fn.discriminator_loss(fake, real)
g_loss, g_details = loss_fn.generator_loss(fake, real)
```

DAC preset：

```python
loss_fn = Loss.from_preset(Preset.DAC, feature_weight=1.0, in_channels=1)
```

判别器 forward 需要直接返回 `Sequence[Sequence[Tensor]]`。每个 branch 的最后一个
tensor 是 logits，前面 tensor 是 feature maps；简单判别器返回 `[[logits]]`。

边界：

- `discriminator_loss()` 内部会 detach fake，避免 D step 回传到 generator。
- `generator_loss()` 不 detach fake，让 generator 可以收到 adversarial 和 feature matching 梯度。
- `framework.gan` 不自动 freeze discriminator 参数；下游 LightningModule 或后续 helper 负责训练调度。
- `GAN.WGAN` 默认启用 `gp_weight=10.0` 的 gradient penalty；其他 GAN 默认不启用。显式 `gp_weight=0.0` 可以关闭。
- `reduction` 公开入口支持 `Reduction` enum 或 torch 风格字符串；内部统一使用 `Reduction`，默认 `Reduction.Mean`。
- DAC preset 只提供判别器和 loss 计算，不接管 codec 模型、batch schema 或重建 loss 组合。

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
