# GAN Framework Design

## 目标

`anytrain.framework.gan` 提供 adversarial training 中可直接组合进下游
`LightningModule` 的最小公共组件。它不属于 `anytrain.loss`，因为 GAN 这里处理
的是训练范式：同一个判别器会参与 discriminator step、generator step、feature
matching 和可选 gradient penalty。

第一版的根 public API 只暴露：

```python
from anytrain.framework.gan import GAN, Loss, Preset, Reduction
```

`Loss` 持有 discriminator，提供：

```python
d_loss = loss_fn.discriminator_loss(fake, real)
g_loss = loss_fn.generator_loss(fake, real)
```

返回约定沿用 scalar-only loss 层：

- 每个入口固定返回一个 0-D `torch.Tensor`。
- `discriminator_loss()` / `generator_loss()` 是调用统一
  `forward(..., objective=...)` 的兼容 wrapper。
- logging-only 分量不进入返回值，通过 module observation registry 声明和发出。

## 非目标

第一版不做：

- 完整 `LightningModule`、训练入口、配置系统或默认 app 层。
- 自动创建 generator / discriminator optimizer。
- 自动 freeze / unfreeze discriminator 参数。
- 判别器更新频率、warmup、EMA、混合精度策略。
- 完整音频 codec 模型或预训练权重管理。
- 和 `deepaudio` API 的逐字段兼容。

这些训练流程相关逻辑如需复用，后续继续放在 `anytrain.framework.gan` 的 helper
中，不混进 `loss`。

## deepaudio 参考

参考源码：

- `deepaudio/src/deepaudio/loss/gan/mixin.py`
- `deepaudio/src/deepaudio/loss/gan/dac.py`
- `deepaudio/src/deepaudio/loss/gan/types.py`
- `deepaudio/src/deepaudio/protocol/loss/gan.py`
- `deepaudio/src/deepaudio/lightning/gan.py`

保留的思路：

- `real_loss` / `fake_loss` / `adv_loss` 三个公式入口。
- `discriminator_loss(fake, real)` 和 `generator_loss(fake, real)` 两条训练目标入口。
- feature matching 使用 fake feature 和 detached real feature 做 L1。
- 多判别器输出按 branch 聚合，最后一层作为 logits，前面层作为 feature maps。

调整点：

- 不保留 `GANMixin` 继承树，改成组合式 `Loss`。
- DAC MPD / MSD / MRD 判别器放在 `framework.gan.audio`，根包只 lazy preset，不默认导入音频依赖。
- 不提供 output adapter；要求下游判别器 forward 直接返回统一结构。
- `GAN` 和 `Reduction` 使用 `StrEnum` + `auto()`，保留精简 enum 接口。
- `reduction` 公开入口兼容 torch 风格字符串，内部统一使用 `Reduction`。
- WGAN 默认启用 GP，其他 GAN 默认不启用。

## 包结构

```text
src/anytrain/framework/gan/
  __init__.py
  types.py
  _output.py
  criterion.py
  feature.py
  penalty.py
  audio/
    __init__.py
    dac.py
  module.py
tests/
  test_framework_gan.py
```

职责：

- `types.py`：`GAN`、`Preset`、`Reduction` 等公开 enum。
- `_output.py`：判别器输出结构校验和 logits/features 拆分。
- `criterion.py`：私有 `_LogitCriterion`，实现 Hinge / LSGAN / WGAN 公式。
- `feature.py`：私有 `_FeatureMatching`。
- `penalty.py`：私有 `_GradientPenalty`。
- `audio/dac.py`：DAC 风格 MPD / MSD / MRD 判别器。
- `module.py`：public `Loss` 组合器。
- `__init__.py`：只导出 `GAN`、`Loss`、`Preset` 和 `Reduction`。

`anytrain.framework.__init__` 不默认导入 `framework.gan`。用户显式使用：

```python
from anytrain.framework.gan import GAN, Loss, Preset, Reduction
```

## Public API

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

也可以用内置 preset 创建 discriminator：

```python
loss_fn = Loss.from_preset(
    Preset.DAC,
    feature_weight=1.0,
    in_channels=1,
)
```

构造参数：

- `discriminator`：`torch.nn.Module`。
- `gan`：`GAN | str`，默认 `GAN.Hinge`。
- `reduction`：`Reduction | str`，默认 `Reduction.Mean`；公开入口接受 `"mean"` / `"sum"`，内部统一转成 enum。
- `feature_weight`：generator step 中 feature matching 权重，默认 `0.0`。
- `gp_weight`：gradient penalty 权重。`None` 时，`GAN.WGAN` 默认 `10.0`，其他
  GAN 默认 `0.0`；显式传 `0.0` 可以关闭 WGAN 默认 GP。

`Loss.from_preset(Preset.DAC, **kwargs)` 会把额外 `kwargs` 转发给
`DACDiscriminator`。preset 分发层不展开具体判别器参数，避免后续新增 preset 时让
`Loss.from_preset()` 的签名持续膨胀。

`Loss` 在定义旁注册以下推荐 observation：

- `OutputObservation("discriminator/loss")`：D objective 的总 scalar。
- `ForwardEvent("discriminator/components")`：`real`、`fake`。
- `ForwardEvent("discriminator/gradient_penalty")`：启用 GP 时的 `gp`。
- `OutputObservation("generator/loss")`：G objective 的总 scalar。
- `ForwardEvent("generator/components")`：`adv`。
- `ForwardEvent("generator/feature_matching")`：启用 feature matching 时的 `feature`。
- `StateObservation("weights")`：只读记录 `feature_weight` 和解析后的 `gp_weight`。

events 中的 tensor 在发出时 detach，只用于日志，不保存到 module 属性；
weights collector 只读取 module 已有配置，不缓存 forward tensor。未挂
`ObservationCallback` 时，第一次 training + grad-enabled 调用会提示推荐观测入口。

## GAN 类型和公式

```python
from enum import StrEnum, auto


class GAN(StrEnum):
    Hinge = auto()
    LSGAN = auto()
    WGAN = auto()
```

公式：

```python
Hinge:
  fake = relu(1 + fake_logits).mean()
  real = relu(1 - real_logits).mean()
  adv = -fake_logits.mean()

LSGAN:
  fake = (fake_logits ** 2).mean()
  real = ((1 - real_logits) ** 2).mean()
  adv = ((1 - fake_logits) ** 2).mean()

WGAN:
  fake = fake_logits.mean()
  real = -real_logits.mean()
  adv = -fake_logits.mean()
```

每个 branch 内部先按 tensor 元素求均值，再按 `reduction` 聚合 branch：

- `Reduction.Mean` / `"mean"`：不同 branch 平均，默认行为。
- `Reduction.Sum` / `"sum"`：不同 branch 求和，适合复刻某些 neural codec vocoder 的尺度。

## 判别器输出契约

`framework.gan` 不再猜判别器输出，也不提供 adapter。用户需要让 discriminator
的 `forward()` 返回：

```python
Sequence[Sequence[Tensor]]
```

每个内层 sequence 是一个 branch：

- 最后一个 tensor 是 logits。
- 前面的 tensor 是 feature maps。

简单单判别器：

```python
class D(nn.Module):
    def forward(self, x):
        logits = self.net(x)
        return [[logits]]
```

带 feature matching 的多 branch 判别器：

```python
class D(nn.Module):
    def forward(self, x):
        return [
            [feat_a1, feat_a2, logits_a],
            [feat_b1, feat_b2, logits_b],
        ]
```

校验规则：

- 至少一个 branch。
- 每个 branch 至少包含 logits。
- 所有 item 必须是 tensor。
- 直接返回 tensor 会报错，并提示简单判别器应返回 `[[logits]]`。
- feature matching 要求 fake / real 的 branch 数、feature 数和 feature shape 一致。

## Feature Matching

`feature_weight > 0` 时，`Loss.generator_loss(fake, real)` 会额外计算 feature
matching：

1. 用当前 discriminator 计算 fake output，保留 fake 到 generator 的梯度。
2. 在 `torch.no_grad()` 下计算 real output。
3. 用 L1 计算 fake features 和 detached real features。
4. 返回 `adv + feature_weight * feature`。

约束：

- `feature_weight > 0` 时必须传 `real`。
- 判别器输出没有 feature maps 时直接抛 `ValueError`。
- `framework.gan` 不自动修改 discriminator 参数的 `requires_grad`。

## Gradient Penalty

GP 是 `Loss` 的内部能力，不作为 public helper 暴露。

默认行为：

- `gan=GAN.WGAN` 且 `gp_weight is None`：`gp_weight = 10.0`。
- 其他 GAN 且 `gp_weight is None`：`gp_weight = 0.0`。
- 显式 `gp_weight=0.0`：关闭 GP。
- 显式 `gp_weight > 0`：启用 GP。

GP 对 fake / real 插值并 `requires_grad_(True)`，再通过同一个 discriminator 得到
logits。每个 branch 的 logits flatten 后按样本求均值，再按 `reduction` 聚合
branch，最终得到每个样本一个 score。

启用 GP 后：

```python
loss = real_loss + fake_loss + gp_weight * gp
```

`real_loss`、`fake_loss` 和 `gp` 作为 observation event 发出；只有加权后的 `loss`
进入训练返回值。

## Lightning 接线边界

`Loss` 不接管 manual optimization。下游仍然显式写 Lightning 原生训练循环：

```python
class CodecModule(pl.LightningModule):
    automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        real = batch.waveform
        fake = self.model(real)

        d_loss = self.gan.discriminator_loss(fake, real)
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        rec_loss = self.rec_loss(fake, real)
        g_gan_loss = self.gan.generator_loss(fake, real)
        g_loss = rec_loss + self.gan_weight * g_gan_loss

        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()
```

这里的 `self.gan_weight` 属于任务目标组合权重，不属于 `Loss` 内部。重建
loss、commitment loss、adversarial loss 如何组合，应由下游 task module 或
`LossGroup` 显式决定。

每次 D/G forward 必须覆盖本次 objective 的完整 reduction domain。不同 shard、rank
或 microbatch 的有效 count 不同时，事后平均多个已归约 scalar 不能恢复精确结果；调用方
必须提供完整 domain，或在 loss 内实现所需的全局 normalizer。

后续如果多个项目复用同一套 manual optimization 模式，可以新增：

```text
src/anytrain/framework/gan/
  manual.py
```

它只提供 optimizer step helper、freeze context、D/G update frequency 和日志 key
组装，不拥有 batch schema 或模型 forward。

## 音频判别器迁移策略

`Loss.from_preset(Preset.DAC)` 内置了一个 DAC 风格 MPD / MSD / MRD 判别器：

- waveform 输入形状为 `(batch, channels, time)`。
- forward 返回 `Sequence[Sequence[Tensor]]`，天然符合 `Loss` 的判别器契约。
- MPD 使用 period 分支；MSD 只有传 `sample_rates` 时才启用；MRD 使用多 STFT resolution。
- Conv stack 直接用 `nn.Conv1d/2d + weight_norm + LeakyReLU` 实现，不依赖 deepaudio 私有 `ConvBlock` / `ActType`。
- `torchaudio.functional.resample` 只在启用 MSD 且需要重采样时 lazy import。

直接使用判别器时：

```python
from anytrain.framework.gan.audio import DACDiscriminator

discriminator = DACDiscriminator(in_channels=1)
```

后续如果要补 codec-specific 判别器权重、默认 task wiring 或更多音频结构，继续放
`framework.gan.audio`，不进入根包默认 import。

## 测试策略

当前最小测试覆盖：

- Hinge / LSGAN / WGAN 数值。
- 默认 `Reduction.Mean` 和显式 `Reduction.Sum`。
- 直接 tensor output 报错，并提示 `[[logits]]`。
- `discriminator_loss()` 不给 fake 输入产生梯度。
- `generator_loss()` 给 fake 输入产生梯度。
- feature matching 对 real features detach，fake features 保留梯度。
- feature matching 缺少 `real`、没有 feature maps 或结构不匹配时抛错。
- WGAN 默认启用 GP，非 WGAN 默认不启用 GP。
- D/G objective 固定返回 0-D Tensor，component diagnostics 通过 registry event 发出。
- `Loss.from_preset(Preset.DAC)` 的最小 forward / D loss / G loss smoke。

## 后续

- 根据真实下游需求决定是否新增 `manual.py`。
- 根据跨项目复用情况决定是否扩展更多 audio discriminator preset。
