# Quantization Migration Design

## 目标

把 `deepaudio` 中不绑定音频任务的量化组件迁入 `anytrain.module`，作为下游 LightningModule 可显式组合的 `torch.nn.Module` 积木。

第一版迁移重点：

- Finite Scalar Quantization, FSQ。
- Embedding-table nearest-neighbor Vector Quantization, VQ。
- Residual Vector Quantization, RVQ。
- 量化输出 dataclass 和轻量 protocol。
- 最小线性投影能力，用于 input dim 和 codebook dim 不一致的情况。

迁移后的模块应保持 task-agnostic，不解释 batch schema，不依赖 codec/audio/zoo，不接管训练 step。对象实例化由下游项目自己的入口或配置系统负责。

## 非目标

这些内容不进入 `anytrain` 第一版迁移：

- `deepaudio.model.dynacodec` 里的完整 codec 模型。
- `deepaudio.protocol.model.codec` 这类 audio codec 协议。
- `deepaudio.zoo.stable_codec`、DAC/Encodec wrapper、LM codebook pattern。
- 数据集、data module、task、wrapper、预训练权重下载逻辑。
- `deepaudio.module.projector` 的 MoE/ExpertRegistry 整套体系。

如果后续确实需要动态 projector，应该先在 `anytrain.module` 单独设计 projector 边界，再让 quantizer 显式依赖它。

## 源代码盘点

迁移候选：

- `deepaudio/src/deepaudio/protocol/module/vector_quantizer.py`
- `deepaudio/src/deepaudio/module/vector_quantizer/__init__.py`
- `deepaudio/src/deepaudio/module/vector_quantizer/types.py`
- `deepaudio/src/deepaudio/module/vector_quantizer/config.py`
- `deepaudio/src/deepaudio/module/vector_quantizer/finite_scalar.py`
- `deepaudio/src/deepaudio/module/vector_quantizer/vanilla.py`
- `deepaudio/src/deepaudio/module/vector_quantizer/residual.py`

当前直接依赖：

- `torch`
- `einops`
- `deepaudio.module.projector.Projector`
- `deepaudio.module.projector.ExpertType`
- `deepaudio.protocol.module.VQEncodeOutput`
- `deepaudio.protocol.module.VQLoss`

`torch` 和 `einops` 已经是 `anytrain` 默认依赖。`Projector` 依赖链偏大，第一版不整体迁移，只在 quantizer 内提供 `nn.Identity` / `nn.Linear` 级别的投影。

## 目标位置

推荐放在：

```text
src/anytrain/module/quantization/
  __init__.py
  output.py
  protocol.py
  types.py
  projection.py
  finite_scalar.py
  embedding.py
  residual.py
```

公开导入：

```python
from anytrain.module.quantization import (
    FSQConfig,
    EmbeddingVectorQuantizer,
    FiniteScalarQuantizer,
    GVQConfig,
    GroupedVectorQuantizer,
    QuantizationLoss,
    QuantizeOutput,
    QuantizerType,
    RVQConfig,
    ResidualVectorQuantizer,
    VQConfig,
)
```

`anytrain.module.__init__` 可以在实现稳定后导出常用类，但 `import anytrain` 不应导入量化组件。

## 统一接口

### 输出对象

统一使用一个输出 dataclass，避免 deepaudio 里 `loss` / `vq_loss`、`codes` / `codebook_vectors` 混用：

```python
@dataclass(eq=False)
class QuantizationLoss:
    commitment: Tensor
    codebook: Tensor


@dataclass(eq=False)
class QuantizeOutput:
    quantized_latents: Tensor
    indices: Tensor
    codebook_vectors: Tensor | None = None
    latents: Tensor | None = None
    loss: QuantizationLoss | None = None
```

shape 约定：

- latent-like tensor 的 feature 维放最后一维。
- leading dimensions 视为 batch/time/spatial 维，并在流程中保持对齐。
- `quantized_latents` 是投影回 input dim 后的张量。
- `codebook_vectors` 是投影前的 codebook 空间向量。

### Quantizer protocol

第一版 protocol 只描述稳定方法，不把训练细节藏进 protocol：

```python
class QuantizerProtocol(Protocol):
    num_codebooks: int
    codebook_size: int
    codebook_dim: int
    input_dim: int

    @property
    def codebook_sizes(self) -> tuple[int, ...]: ...
    def forward(self, latents: Tensor) -> QuantizeOutput: ...
    def quantize(self, latents: Tensor) -> QuantizeOutput: ...
    def latents_to_codebook_vectors(self, latents: Tensor) -> Tensor: ...
    def codebook_vectors_to_indices(self, codebook_vectors: Tensor) -> Tensor: ...
    def indices_to_codebook_vectors(self, indices: Tensor) -> Tensor: ...
    def project_codebook_vectors(self, codebook_vectors: Tensor) -> Tensor: ...
```

避免继续使用 `hidden_size`，统一叫 `input_dim`。

`forward()` 应委托给 `quantize()`。训练代码可以按 PyTorch 习惯调用 `quantizer(x)`，需要语义更明确时也可以调用 `quantizer.quantize(x)`。

FSQ 还需要额外 helpers，但不放进基础 protocol：

```python
def indices_to_levels(indices: Tensor) -> Tensor: ...
def levels_to_indices(levels: Tensor) -> Tensor: ...
def indices_to_level_probs(indices: Tensor, tau: float = 1.0) -> Tensor: ...
def level_logits_mask(logits: Tensor) -> Tensor: ...
```

## 关键设计决策

### 1. 包名使用 `quantization`

`deepaudio` 当前目录名是 `vector_quantizer`，但 `anytrain` 里建议用 `anytrain.module.quantization`。原因是 FSQ 严格来说不是基于 embedding table 的 vector quantizer，`quantization` 更适合作为 umbrella module。

第一版不提供 `anytrain.module.vector_quantizer` 兼容别名，除非下游已有明确迁移压力。

### 2. 投影只做最小闭环

deepaudio 的 `Projector` 会引入 ExpertRegistry、MoE、Router 等额外设计。量化迁移第一版只支持：

- input dim 等于 codebook dim 时使用 `nn.Identity`。
- input dim 不等于 codebook dim 时使用 `nn.Linear`。

后续如需更复杂投影，用显式 `projection_factory` 或独立 `anytrain.module.projector` 承接，不在这次迁移里混入。

### 3. FSQ 的 `indices` 语义要修正

deepaudio 当前 FSQ 的 `codes_to_indices()` 返回的是 per-dimension levels，而不是 flat codebook index；但 protocol 文档说返回 flat indices。迁移后不再公开 `codes_to_indices()`，改用不会混淆整数 id 和连续向量的 `codebook_vectors_to_indices()`。

`anytrain` 第一版应统一为：

- `indices`: shape `(...)`，和普通 VQ 一样表示单个 codebook 的 flat integer id。
- `codebook_vectors`: shape `(..., codebook_dim)`，和普通 VQ 的末维接口对齐。
- `codebook_sizes`: 单 codebook 返回 `(codebook_size,)`；residual / multi-stage
  quantizer 返回每个 stage 的 flat id 空间大小。
- `levels`: 每个 scalar code dimension 的 level 数。
- `codebook_size`: `prod(levels)`，即 flat id 的取值范围是 `[0, codebook_size - 1]`。
- `indices_to_levels(indices)`: flat id 转 per-dimension level indices。
- `levels_to_indices(levels)`: per-dimension level indices 转 flat id。
- 如训练任务需要 per-dimension label smoothing，调用 `indices_to_level_probs()` 或 `indices_to_levels()`。

这样 VQ/RVQ/FSQ 在 `forward().indices` 上保持一致。

FSQ 推荐尽量使用 odd `levels`，这样每个 scalar grid 都能严格以 0 为中心。even `levels` 不作为错误处理，但 `FSQConfig` 会发出 warning；当前实现会继续采用 zero-friendly offset grid，使 0 仍然是可选码点，而不是切到严格对称但不含 0 的网格。

`default_fsq_levels()` 和 `FSQConfig` 的默认值使用 odd-only presets，优先保持对称网格；这些 presets 的 `prod(levels)` 接近但不保证精确等于 `2**power`。

`bound_scale` 控制 projected latents 进入 `tanh` 前的尺度，默认 `1.0` 保持原始行为；训练中如果 projected latents 过早贴到边界、导致 `tanh` 梯度变小，可以适当调大这个值。

### 4. VQ 类名要避免协议冲突

deepaudio 当前实现类叫 `VectorQuantizer`，容易和 protocol 名称混在一起。迁移后：

- 单 codebook protocol 叫 `QuantizerProtocol`；支持 `num_active_codebooks` 的 residual quantizer 额外实现 `ResidualQuantizerProtocol`。
- embedding table + nearest-neighbor 实现叫 `EmbeddingVectorQuantizer`。
- `VQConfig` 保留，表示向量量化的常用配置缩写。

### 5. GVQ 作为 learned product codebook

`GroupedVectorQuantizer` 和普通 VQ 使用同一个外部接口，但内部把 codebook 拆成多个 group 做笛卡尔积。例如：

```python
GVQConfig(
    input_dim=128,
    group_sizes=(90, 90),
    codebook_dim=128,
    group_dims=(64, 64),
)
```

对外：

- `codebook_size == 90 * 90 == 8100`
- `indices` shape 为 `(...)`
- `codebook_vectors` shape 为 `(..., 128)`

内部：

- `indices_to_group_indices(indices)` 得到 `(..., 2)`
- 每个 group 各自做 nearest-neighbor，搜索复杂度约为 `90 + 90`，而不是 `8100`
- `group_indices_to_indices(group_indices)` 把 group id 合并回 flat id

这让 FSQ、GVQ、VQ 的接口保持一致：FSQ 是 fixed scalar product grid，GVQ 是 learned product codebook，VQ 是 learned flat table。

### 6. VQ 的训练分支要先修正再迁移

deepaudio 当前 VQ 实现有几处不能照搬的问题：

- `VQEncodeOutput(..., vq_loss=vq_loss)` 和 dataclass 字段 `loss` 不一致。
- 非 EMA 训练时也更新 `self.code_counter`，但 `code_counter` 只在 EMA 模式下创建。
- eval 模式下 `vq_loss` 可能未定义。
- `VQConfig.expert_type` 缺少 dataclass 类型标注。

迁移时应把这些作为接口修正，而不是做兼容分支。

VQ 的 l2-normalized nearest-neighbor lookup 保留为默认行为，但要用显式配置字段表示，例如 `normalize_lookup: bool = True`，避免后续读代码时把它误判为标准欧氏 VQ。lookup 比较默认临时升到 FP32（`lookup_in_float32=True`），输出仍保持 embedding/projector 的 dtype。

### 7. RVQ 先支持统一 codebook dim

RVQ 可以在多个 VQ 上累加 residual。第一版建议优先支持所有 quantizer 使用同一个 `codebook_dim`，这样 `codebook_vectors` 可以稳定 stack 成 `(..., num_codebooks, codebook_dim)`。

异构 `codebook_dim` 可以后续支持，但不能静默 cat 成一个模糊张量。需要支持时再把返回结构设计清楚。

RVQ `forward()` 的临时截断参数不叫 `num_quantizers`，改叫 `num_active_codebooks`，避免和配置里的 `num_codebooks` 混淆。

### 8. Buffer 使用统一注册入口

FSQ 的 basis、levels、half levels、offsets，VQ EMA 的 counter/sum 都通过
`anytrain._buffer.register_buffer` 注册。PyTorch 2.5 及以上使用 `nn.Buffer`，旧版本回退到
`Module.register_buffer`，两条路径保持 `persistent`、设备迁移和 state dict 语义一致。

## 分阶段计划

### P0: 固定设计和验收标准

- 增加本设计文档。
- 在 `todo.md` 加迁移任务。
- 明确第一版不迁移 audio codec/model/zoo。

### P1: 搭建 quantization 包和 FSQ

- 新增 `anytrain.module.quantization.output`。
- 新增 `types.py`，提供 `QuantizerType`。
- 新增 `projection.py`，只提供 identity/linear 投影 helper。
- 迁移并整理 `FiniteScalarQuantizer`。
- 将 FSQ 配置字段从 `levels_per_codebook` 收敛为 `levels`。
- 修正 FSQ `indices` 为 flat ids，并保留 levels 转换 helper。
- 增加 FSQ 单测。

验收：

- forward 输出 shape 正确。
- `indices -> codebook_vectors -> project_codebook_vectors` 可回到 decoder 可消费的 latent shape。
- `codebook_vectors_to_indices(indices_to_codebook_vectors(indices))` round-trip。
- 不需要 deepaudio import。

### P2: 迁移 vanilla VQ

- 迁移 `VQConfig` 和 `EmbeddingVectorQuantizer`。
- 修正 `loss` 字段命名。
- 修正 EMA 和非 EMA 训练分支。
- eval 模式返回 `loss=None`。
- 增加 `normalize_lookup: bool = True`，显式保留 deepaudio 当前的 l2-normalized distance 行为。
- 增加 `lookup_in_float32: bool = True`，避免低精度 lookup 在大维度或 L2 距离下误选。

验收：

- 非 EMA 模式有 commitment/codebook loss，梯度能流向 encoder 和 codebook。
- EMA 模式更新 counter/sum，codebook weight 不走梯度。
- eval 模式 forward 不报错，输出 finite。
- `indices_to_codebook_vectors()` 和 `latents_to_codebook_vectors()` shape 稳定。

### P3: 迁移 RVQ

- 迁移 `RVQConfig` 和 `ResidualVectorQuantizer`。
- `forward()` 返回 `QuantizeOutput`，而不是 tuple。
- 支持 `num_active_codebooks` 控制推理时使用的 codebook 数。
- 训练 dropout 只在 train 模式启用。
- 暂时限制统一 `codebook_dim`，或在配置校验中明确报错。

验收：

- 单 quantizer RVQ 和 vanilla VQ 形状一致。
- 多 codebook 输出 `indices` shape 为 `(..., num_active_codebooks)`。
- residual 累加输出 finite。
- dropout 训练分支可控，eval 不随机截断。

### P4: 文档和公开导出

- 更新 `docs/modules/module.md`，把 quantization 加入 task-agnostic module 列表。
- 更新 `docs/modules/index.md` 链接本迁移文档。
- 在 `anytrain.module.quantization.__init__` 导出稳定 API。
- 评估是否在 `anytrain.module.__init__` 导出最常用类。

验收：

- `from anytrain.module.quantization import FiniteScalarQuantizer` 成功。
- `python -m pytest tests/test_module_quantization*.py` 通过。
- `ruff check src/anytrain tests` 通过。

YAML 配置示例：

```yaml
quantizer:
  config:
    input_dim: 128
    levels: [9, 7, 7, 7, 7, 3]
    bound_scale: 1.0
```

## 测试设计

建议新增测试文件：

```text
tests/test_module_quantization_fsq.py
tests/test_module_quantization_vq.py
tests/test_module_quantization_rvq.py
```

核心用例：

- plain-config friendly dataclass 字段稳定，无函数对象默认值。
- invalid config 早失败，例如非正 codebook size、空 levels、dim 不匹配。
- forward 支持 `(..., input_dim)`，至少覆盖 2D 和 3D latent。
- round-trip helper 在 CPU 上可稳定运行。
- 训练分支 loss 为 scalar Tensor。
- backward 后关键参数梯度 finite。
- import smoke 不触发 deepaudio/audio/zoo 依赖。

## 风险和处理

### 行为和 deepaudio 不完全兼容

FSQ `indices` 语义会从 per-dimension levels 修正为 flat ids。这是刻意的接口收敛。需要旧行为时通过 `indices_to_levels()` 显式拿 levels。

### Projector 能力变少

第一版只支持线性投影。这样可以先保证量化核心可用，也避免把 deepaudio 的 MoE/projector 依赖链一次性搬入 `anytrain`。复杂投影后续单独设计。

### RVQ 异构 codebook dim

异构 codebook dim 会让 `codebook_vectors` 返回结构不稳定。第一版先报错或只在内部使用，不静默返回拼接结果。

### 训练稳定性功能

EMA VQ 支持第一次训练 forward 的分布式 k-means 初始化和基于 EMA count 的 dead code
recycling；分别由 `kmeans_init` / `kmeans_iters` 和 `dead_code_threshold` 控制。distributed
初始化后，k-means、recycling 和每次 EMA 更新都保持 rank collective 对齐，counts/sums 以
FP32 保存。所有 rank 仍必须使用相同的 train/eval mode 和 active stage 数。RVQ 的
per-vector dropout 允许各 rank 使用不同的 local mask；本地空 stage 以零统计参与
collective，全局空 stage 不更新 EMA state/codebook。含可训练参数的 stage 可能在单个 rank
完全未使用，因此下游 PyTorch DDP 需要设置 `find_unused_parameters=True`。

## 最小使用示例

```python
import torch

from anytrain.module.quantization import FSQConfig, FiniteScalarQuantizer

quantizer = FiniteScalarQuantizer(
    FSQConfig(
        input_dim=128,
        levels=(9, 7, 7, 7, 7, 3),
        bound_scale=1.0,
    )
)

x = torch.randn(2, 16, 128)
output = quantizer(x)

assert output.quantized_latents.shape == x.shape
assert output.indices.shape == (2, 16)
```
