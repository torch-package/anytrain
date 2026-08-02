# Flow Matching Design

## 当前状态

第一版 source、time sampler、continuous/discrete runtime、objective、sampler、optional 依赖
边界和 CPU 测试已经实现。稳定入口从 `anytrain.framework.flow_matching` 显式导入；
`anytrain.framework` 根模块保持空导出。下文的“实现计划”保留为设计决策记录，不表示这些
阶段仍未完成。

## 目标

`anytrain.framework.flow_matching` 作为 Facebook `flow_matching` 的轻量再封装，提供可组合进下游 `LightningModule` 的训练目标和采样 helper。

核心目标：

- 复用 Facebook `flow_matching` 的 path、loss 和 solver，不重写概率路径数学。
- 保留 deepaudio 中 `ContinuousFlowMatching` / `DiscreteFlowMatching` 的易用入口。
- 把 source distribution、time sampler、path、objective、solver 拆成可替换组件。
- 不绑定 audio/text/codec 任务语义，不解释 batch schema，不生成完整 `LightningModule`。
- optional 依赖不进入 `anytrain` 默认 import。

## 非目标

第一版不做：

- 完整 diffusion/flow 模型 zoo。
- 下游训练入口、配置系统、data module 或 task step。
- audio codec、text encoder、condition encoder 等领域组件。
- 自定义 probability path 数学实现，除非只是包一层已有 Facebook path。
- 和 deepaudio API 的逐字段兼容。

## deepaudio 参考

参考源码：

- `deepaudio/src/deepaudio/framework/flow_matching/abc.py`
- `deepaudio/src/deepaudio/framework/flow_matching/continuous.py`
- `deepaudio/src/deepaudio/framework/flow_matching/discrete.py`
- `deepaudio/src/deepaudio/task/sound_flow/pl_module.py`

可保留的思路：

- 连续空间默认使用 `CondOTProbPath`，目标是预测 velocity，loss 为 MSE。
- 离散空间默认使用 `MixtureDiscreteProbPath` 和 `MixturePathGeneralizedKL`。
- source distribution 支持连续的 gaussian/uniform、离散的 uniform/mask。
- sampling 通过 Facebook solver 完成，而不是自己写积分器。

需要调整的地方：

- deepaudio 的 helper 同时持有 path、source、loss、solver，并直接调用 task model；`anytrain` 应拆成可组合组件。
- deepaudio 依赖 `deepaudio.protocol.model.FlowModel`；`anytrain` 应只依赖 `torch.nn.Module` 或轻量 protocol。
- 连续和离散模型调用方式不统一；`anytrain` 第一版统一默认 `model(x_t, t, **extras)`，特殊模型用显式 adapter。
- 默认参数不要直接使用 `CondOTProbPath()` 这类对象实例，避免共享可变状态；构造函数用 `None` 后内部创建。
- `SoundFlow` 里的 codec/text 逻辑属于下游任务，不迁入。

## 目标包结构

推荐第一版放在：

```text
src/anytrain/framework/flow_matching/
  __init__.py
  _deps.py
  types.py
  source.py
  time.py
  objective.py
  sampler.py
  continuous.py
  discrete.py
```

职责：

- `_deps.py`：集中导入 Facebook `flow_matching`，缺依赖时抛清晰错误。
- `types.py`：轻量 protocol、dataclass 和 enum。
- `source.py`：source distribution 采样组件。
- `time.py`：训练和采样的时间网格组件。
- `objective.py`：连续 velocity matching、离散 generalized KL 等训练目标。
- `sampler.py`：对 Facebook solver 的薄封装，统一返回 final/intermediates。
- `continuous.py` / `discrete.py`：连续和离散 runtime，统一持有训练路径与推理配置。

公开导入第一版只从 `anytrain.framework.flow_matching` 导出稳定对象；`anytrain.framework.__init__` 不默认导入该子模块。

## 组合模型

公开接口分成两层：runtime 持有 `path/source/time_sampler/sampler/call_model`，objective 只负责把 runtime 的训练样本转换成 loss。底层组件仍可单独替换。

### Source

source 只负责从 target 或 shape 生成 `x_0`：

```python
class Source(Protocol):
    def sample_like(self, x_1: Tensor) -> Tensor: ...
```

建议实现：

- `GaussianSource`
- `UniformSource`
- `UniformTokenSource(vocab_size: int)`
- `MaskTokenSource(mask_id: int)`

连续 source 返回和 `x_1` 同 shape/device/dtype 的 tensor。离散 source 返回 `torch.long` token tensor；mask source 的 `mask_id` 必须显式传入，且必须落在 `[0, vocab_size)`（与 Facebook discrete backend 的 gather/one_hot 一致）。
连续 runtime 也要求显式传入的 `x_0` 与 `x_1` device/dtype 一致，并在进入 path
前把采样时间转换到 `x_1.dtype`，避免插值结果被默认 float32 静默提升。

### Time Sampler

time sampler 只负责给 batch 生成时间：

```python
class TimeSampler(Protocol):
    def sample(self, batch_size: int, device: torch.device) -> Tensor: ...
```

默认实现：

- 连续 runtime 默认使用 `UniformTimeSampler(t_min=0.0, t_max=1.0)`，对应标准 flow matching 对时间积分的均匀 Monte Carlo 估计。
- 离散 runtime 默认使用 `UniformTimeSampler(t_min=0.0, t_max=1.0 - 1e-3)`，和离散 Euler sampler 的 `[0, 1 - eps]` 端点保持一致，并避开 generalized KL 在 `t=1` 的奇点。
- `LogitNormalTimeSampler` 保留为显式实验选项。它会改变 objective 的时间权重，不作为未配套 importance weighting 或实验依据时的通用默认值。
- `t_min` / `t_max` 描述训练和采样约定的时间区间。不要默认把 `t_min` 设成非零值，否则推理时起点会变成未知的 `x_eps`。

### Objective

objective 负责把 `x_0`、`x_1`、`t` 和模型预测变成 scalar loss：

```python
loss = objective(model, x_1, x_0=None, **extras)
```

连续 runtime：

- path：`CondOTProbPath`
- source：`GaussianSource`
- time：`UniformTimeSampler`
- loss：`mse_loss(model(x_t, t, **extras), dx_t)`

离散 runtime：

- path：`MixtureDiscreteProbPath(PolynomialConvexScheduler(n=2.0))`
- source：`UniformTokenSource` 或 `MaskTokenSource`
- time：`UniformTimeSampler(t_min=0.0, t_max=1.0 - 1e-3)`
- loss：`MixturePathGeneralizedKL`
- `x_0` / `x_1` 必须是 `torch.long` token tensor，不做静默 dtype 转换。
- 模型输出 logits，objective 内部不自动做 sampling 或 argmax。

`objective` 返回必须是 scalar Tensor。额外日志不在第一版塞进返回值；后续如果需要，可加 `FlowLossOutput(loss, details)`，但不能影响 `backward()` 主路径。

continuous 和 discrete objective 都只接收对应 runtime，避免再次传入 path、source、time sampler 或 model caller。

### Model Caller

默认模型调用约定为：

```python
prediction = model(x_t, t, **extras)
```

如果下游模型需要 `model(x=..., t=...)`、condition 预处理或多个输入，可以显式传入：

```python
def call_model(model: nn.Module, x_t: Tensor, t: Tensor, extras: Mapping[str, object]) -> Tensor:
    return model(x=x_t, t=t, **extras)
```

这样保留组合自由度，但不把 batch schema 写进 `anytrain`。

### Solver

runtime 通过 sampler 完成 sampling：

```python
output = runtime.sample(model, x_0, **extras)
```

sampler 只保存 method、NFE、step 数等数值求解策略；path、vocab size 和 model caller 由 runtime 在调用时传入，不在多个对象中重复保存。

建议统一返回：

```python
@dataclass(eq=False)
class FlowSampleOutput:
    final: Tensor
    states: Tensor | None = None
    time_grid: Tensor | None = None
```

连续 sampler 默认封装 `ODESolver`，默认参数：

- `method="midpoint"`
- `nfe=20`
- `num_steps=10`
- `return_intermediates=False`

离散 sampler 默认封装 `MixtureDiscreteEulerSolver`，内部把 logits 转成概率：

```python
prob = model(x_t, t, **extras).softmax(dim=-1)
```

离散 sampler 的默认时间网格是 `[0, 1 - eps]`，和离散训练 objective 保持同一个有效时间区间。`vocab_size` 表示目标 token 空间大小；mask token 是否属于模型输入 embedding，由下游模型自己处理。

## Runtime / objective

continuous 和 discrete 使用同一套调用方式：

```python
continuous = ContinuousFlowRuntime(
    sampler=ODESampler(),
)
continuous_objective = ContinuousVelocityObjective(continuous)

continuous_loss = continuous_objective(model, x_1, condition=condition)
continuous_x_0 = continuous.source_like(x_1)
continuous_sample = continuous.sample(model, continuous_x_0, condition=condition)

discrete = DiscreteFlowRuntime(
    vocab_size=1025,
    source=MaskTokenSource(mask_id=1024),
)
discrete_objective = DiscreteGeneralizedKLObjective(discrete)

discrete_loss = discrete_objective(model, tokens, condition=condition)
discrete_x_0 = discrete.source_like(tokens)
discrete_sample = discrete.sample(model, discrete_x_0, condition=condition)
```

runtime 不继承或替代下游 `LightningModule`。下游需要 mask 或其它 reduction 时，可以直接使用 `runtime.training_sample()` 实现任务 objective。

## Flow-GRPO stochastic step

RectifiedFlowSDE 提供 Flow-GRPO 风格的 reverse denoising Gaussian transition：
下游 sampler 在已获得 velocity 后调用 step_with_log_prob(x_t, t, next_t, velocity)，
得到 next_x、transition mean、std 和每条样本的 log-prob。它适合在线 RL rollout
或 PPO/GRPO update 阶段重算 log-prob，不负责 prompt、reward、pipeline patch 或
LightningModule 训练循环。

时间约定为 0 < next_t < t < 1，即从更 noisy 的 latent 向更 clean 的 latent 走一步。
默认 log_prob_reduction="mean" 对 latent 维求均值，贴近 Flow-GRPO 官方实现的高维
latent 稳定训练做法；需要严格联合 Gaussian log-prob 时可显式使用 "sum"。

论文引用：Liu et al., “Flow-GRPO: Training Flow Matching Models via Online RL”,
arXiv:2505.05470。该 helper 对应论文中 ODE-to-SDE transition、transition
log-prob 和等方差 Gaussian KL 的可复用张量计算。

## 依赖策略

`flow_matching` 作为 optional framework 依赖，不进入默认安装。建议在 `pyproject.toml` 增加：

```toml
[project.optional-dependencies]
flow = ["flow_matching"]
```

依赖行为：

- `import anytrain` 不导入 `flow_matching`。
- `import anytrain.framework` 不导入 `flow_matching`。
- `import anytrain.framework.flow_matching` 如果缺依赖，抛出带安装提示的 `ImportError`。
- 单测要覆盖缺依赖错误路径，但不影响 core 测试。

## 第一版 API 草案

```python
from anytrain.framework.flow_matching import (
    ContinuousFlowRuntime,
    ContinuousVelocityObjective,
    DiscreteFlowRuntime,
    DiscreteGeneralizedKLObjective,
    FlowSampleOutput,
    GaussianSource,
    LogitNormalTimeSampler,
    MaskTokenSource,
    ODESampler,
    UniformTimeSampler,
    UniformTokenSource,
)
```

命名约定：

- 组合器用 `FlowRuntime`，表示一套训练路径和推理配置。
- 训练目标用 `Objective`，避免和 `torch.nn.Module.loss` 或 `anytrain.loss` 混淆。
- sampling helper 用 `Sampler`，不叫 `Solver`；`Solver` 特指 Facebook 原始 solver。
- `nfe` 可以保留，因为 flow matching 论文和 solver API 都常用这个名字。

## 实现计划

### P0: 设计冻结

- 增加本设计文档。
- 在 `docs/modules/framework.md` 和 docs index 加链接。
- 在根 `todo.md` 的 Framework 章节记录尚未完成的任务。

### P1: 最小包结构和依赖边界

- 新增 `flow_matching` 包和 `_deps.py`。
- 新增 source/time/model caller 基础组件。
- 新增 import smoke：core import 不触发 optional 依赖。
- 新增缺依赖错误测试。

### P2: Continuous 最小闭环

- 实现 `ContinuousVelocityObjective`。
- 实现 `ODESampler`。
- 实现 continuous `ContinuousFlowRuntime`、`ContinuousVelocityObjective` 和 `ODESampler` 分层。
- 增加 CPU toy model 测试：loss finite、backward finite、sample shape 正确。

### P3: Discrete 最小闭环

- 实现 `DiscreteGeneralizedKLObjective`。
- 实现 `DiscreteEulerSampler`。
- 实现 `DiscreteFlowRuntime`。
- 增加 CPU toy token model 测试：uniform/mask source、loss finite、sample shape 正确。

### P4: 文档示例

- 在 `docs/modules/framework.md` 补最小使用示例链接。
- 如有真实下游需求，再补 examples；examples 只展示普通 `LightningModule` 如何组合 helper，不提供完整 app。

## 验收标准

- 不需要 deepaudio import。
- 不依赖 audio/text/codec 包。
- `import anytrain` 和 `import anytrain.framework` 不触发 `flow_matching` import。
- 连续和离散目标都能在 CPU 上用 toy model 跑通 forward/backward。
- continuous/discrete runtime 与 objective 使用同一套调用约定。
- 缺 optional 依赖时错误信息包含安装方式。
- 文档示例和测试使用同一套调用约定。

## 风险和处理

### Facebook API 变化

把所有 Facebook `flow_matching` 导入集中到 `_deps.py`。如果上游 API 改名，修正面集中，不影响用户侧组合接口。

### 过度封装

第一版只封 source、time、runtime、objective 和 sampler，不新增 registry、配置装配或 Lightning 基类。真实项目需要复杂配置时，由下游项目自行选择 Hydra/pydantic/普通 Python。

### 离散 mask token 语义

Facebook discrete backend 对 `x_t` 做 `gather` / `one_hot`，因此 `MaskTokenSource.mask_id`
必须落在 `[0, vocab_size)`。mask 若是额外吸收态，应把 `vocab_size` 扩到包含该 id（例如
`vocab_size=1025`、`mask_id=1024`），并由模型输出对应维度的 logits。`DiscreteFlowRuntime`
在构造时拒绝越界 `mask_id`。

### 额外日志需求

第一版 objective 只返回 scalar loss。需要 path sample、target velocity 或 per-token KL 时，后续再增加显式 debug output，不让训练主路径先复杂化。
