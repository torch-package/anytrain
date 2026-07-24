# `anytrain.optim` Design

## 定位

`anytrain.optim` 提供 optimizer / scheduler 构造 helper。它不接管训练流程，也不要求下游继承特殊 LightningModule 基类；下游仍然在自己的 `configure_optimizers()` 里显式调用这些 helper。

## 当前结构

源码结构：

```text
src/anytrain/optim/
  __init__.py
  _llm_config.py
  _params.py
  adamw.py
  compose.py
  llm.py
  muon.py
  options.py
  rules.py
  scheduler/
    __init__.py
    _shape.py
    compose.py
    curve.py
    presets.py
    types.py
```

当前公开导出：

- `CompositeOptimizer`
- `ExcludedModules`
- `LRScaleRule`
- `LRScaleRules`
- `create_adamw_optimizer`
- `split_adamw_decay_params`
- `create_muon_adamw_optimizer`
- `muon_available`
- `split_muon_params`
- `create_scheduler`
- `create_llm_optimizer`
- `create_llm_lightning_optimizers`

高级对象和 options 类型保留在子模块：

- `anytrain.optim.adamw`: `create_adamw_optimizer`
- `anytrain.optim.options`: `AdamWOptions`、`MuonOptions`、`MuonAdamWOptions`、`OptimizerOptions`、`MuonAdjustLRFn`
- `anytrain.optim.muon`: `create_muon_adamw_optimizer`
- `anytrain.optim.scheduler`: `CurveShape`、`Phase`、`Schedule`、`PhaseLike`、`make_scheduler_config`、`make_named_scheduler_config`、`create_scheduler_from_config`
- `anytrain.optim.llm`: `OptimizationConfig`、`create_optimizer_from_config`、`create_lightning_optimizers_from_config`

`anytrain` 的最低依赖是 `torch>=2.4`。Muon 是条件能力：`muon_available()` 在当前
PyTorch 提供 `torch.optim.Muon` 时返回 `True`。旧版 PyTorch 仍可使用 AdamW、scheduler、
参数分组等 API，但创建 Muon optimizer 会直接抛出带版本说明的 `RuntimeError`。

AdamW / Muon 的高级入口接收 typed dict options，不再提供 dataclass config，也不重复校验
torch 原生参数。`anytrain` 只校验自己定义的边界：参数分组、排除 module、lr scale rules、
composite optimizer 参数互斥和 scheduler DSL。`betas`、`eps`、`momentum`、`adjust_lr_fn` 等
optimizer 参数由对应的 torch optimizer 暴露错误。

## LR Scale Rules

需要给某个 module 子树设置相对学习率时，传 `lr_scale_rules`。`name` 对应
`module.named_modules()` 的路径，`lr_scale` 必须大于 0；未命中参数使用 `1.0`：

```python
optimizer = create_adamw_optimizer(
    model,
    {"lr": 3e-4},
    lr_scale_rules=[
        {"name": "encoder", "lr_scale": 0.5},
        {"name": "encoder.layers.23", "lr_scale": 1.0},
        {"name": "lm_head", "lr_scale": 2.0},
    ],
)
```

如果一个参数被多个规则命中，使用最长 module path 对应的规则；同样具体的规则给同一个
shared parameter 分配不同 `lr_scale` 时会抛错。

`lr_scale_rules` 是通用参数分组逻辑，AdamW、Muon+AdamW 和 LLM preset helper 都使用同一套规则。
LLM preset 会把规则透传给底层 optimizer。

## AdamW

`create_adamw_optimizer(module, {"lr": ..., "weight_decay": ...})` 会按常见 AdamW 规则拆分 decay / no-decay 参数组：

- decay：非 embedding / norm / 显式排除 module 中，名称为 `weight` 且维度不小于 2 的参数。
- no-decay：bias、norm、embedding，以及用户显式排除的 module。

`optim` 不按名字猜 output head。需要排除 head 时，显式传 module 对象：

```python
optimizer = create_adamw_optimizer(model, {"lr": 3e-4}, excluded_modules=(model.lm_head,))
```

## Muon

调用 Muon 入口前可以显式检查环境：

```python
from anytrain.optim import muon_available

if not muon_available():
    raise RuntimeError("Muon requires a newer Python/PyTorch environment.")
```

`split_muon_params(module)` 默认将 hidden 2D weight 放入 Muon，其余参数留给 AdamW。默认排除：

- embedding。
- norm。
- bias 和非 `weight` 参数。
- 非 2D weight。
- `excluded_modules` 传入的 module 及其子 module。

因此 head 不再因为名字包含 `head` 自动排除；如果要让 head 走 AdamW，传入具体 module：

```python
optimizer = create_muon_adamw_optimizer(
    model,
    muon={"lr": 3e-4},
    adamw={"lr": 3e-4},
    excluded_modules=(model.lm_head,),
)
```

`lr_scale_rules` 同样会作用到 Muon 和 AdamW fallback 子 optimizer 的参数组。

`create_muon_adamw_optimizer(module, muon={...}, adamw={...}, ...)` 返回 `CompositeOptimizer`：

- `"muon"` 子 optimizer 处理 Muon 参数。
- `"adamw"` 子 optimizer 处理其余参数，并继续按 AdamW 规则拆分 decay / no-decay。

`MuonOptions.adjust_lr_fn` 默认是 `MuonAdjustLRFn.MATCH_RMS_ADAMW`，用于和 AdamW 常用 lr/wd 配置对齐。PyTorch 原生 `torch.optim.Muon` 默认不设置 `adjust_lr_fn`；这里显式改成 LLM 训练更常用的对齐口径。为了方便 YAML/JSON 配置，`adjust_lr_fn` 也接受字符串 `"original"` 和 `"match_rms_adamw"`。高级 options 中显式传入 `adjust_lr_fn=None` 时，会保留 PyTorch 原生行为。

## Scheduler

顶层 `create_scheduler(optimizer, schedule=...)` 提供命名的 step-level scheduler。常用模式是：

```python
create_scheduler(
    optimizer,
    schedule="warmup_cosine",
    warmup_steps=1000,
    total_steps=100000,
    min_lr_ratio=0.1,
)
```

可用的 `schedule` 是：

- `constant`: 保持学习率不变。
- `warmup_cosine`: 线性 warmup 后做 cosine decay。
- `wsd`: warmup + stable + cosine decay。

如果需要完整 phase DSL，继续用 `anytrain.optim.scheduler` 子模块。每个 `Phase` 表示一个连续 phase：

```python
Phase(
    shape="linear",          # 也可以用 CurveShape.LINEAR
    duration_steps=1000,    # -1 表示无限尾段，只能用于最后一个 constant phase
    start_lr_ratio=0.0,     # None 表示接上一个 phase 的 end_lr_ratio
    end_lr_ratio=1.0,
)
```

`shape` 会规范化为 `CurveShape`，可选值是 `CONSTANT`、`LINEAR`、`COSINE`。
内部实现拆成三层：`curve` 定义单段曲线，`compose` 负责把 phase 串成 step-level lambda，
`presets` 把命名 schedule 转成 phase 列表。`Phase` / `Schedule` 是对外领域名。

多个 phase 通过 `Schedule(phases=(...))` 串起来。旧的 warmup + cosine 可以写成：

```python
Schedule(
    phases=(
        Phase("linear", duration_steps=1000, start_lr_ratio=0.0, end_lr_ratio=1.0),
        Phase("cosine", duration_steps=99000, end_lr_ratio=0.1),
    )
)
```

WSD 可以写成：

```python
Schedule(
    phases=(
        Phase("linear", duration_steps=1000, start_lr_ratio=0.0, end_lr_ratio=1.0),
        Phase("constant", duration_steps=90000, end_lr_ratio=1.0),
        Phase("cosine", duration_steps=10000, end_lr_ratio=0.1),
    )
)
```

`create_scheduler_from_config(optimizer, config)` 返回 `torch.optim.lr_scheduler.LambdaLR`。所有 phase 都由自己的 `duration_steps` 定义长度；有限 phase 全部结束后会保持最后一个 `end_lr_ratio`。

`duration_steps=-1` 是无限尾段。它会让后续 phase 不可达，因此只能出现在最后一个 phase，并且只支持 `constant`。

如果只需要默认 ratio，可以用简便接口：

```python
make_scheduler_config(
    ("linear", 1000),
    ("constant", 90000),
    ("cosine", 10000),
)
```

## CompositeOptimizer

`CompositeOptimizer` 把多个真实 optimizer 暴露成一个 `torch.optim.Optimizer`：

- `param_groups` 引用子 optimizer 的真实 param group，因此 scheduler 修改 lr 时会同步影响子 optimizer。
- `step()` 依次调用子 optimizer。
- `zero_grad()` 依次清空子 optimizer。
- `state_dict()` / `load_state_dict()` 按 optimizer 名称保存和恢复。
- 子 optimizer 不能共享参数，否则会在构造时抛错，避免同一参数被 step 两次。
- 构造后不支持直接对 composite 调用 `add_param_group()`；新参数组必须先加到真实子 optimizer 上。

第一版 closure 只执行一次，并把结果作为 composite `step()` 返回值；子 optimizer 的 `step()` 不再接收 closure，避免重复执行 forward/backward。

## LLM Helper

`create_llm_optimizer(module, preset=..., optimizer=..., lr=..., ...)` 用于按扁平参数创建 LLM 常用 optimizer。`preset` 负责默认值，`optimizer` 只决定用 AdamW 还是 Muon+AdamW。

```python
optimizer = create_llm_optimizer(
    model,
    preset="pretrain",
    optimizer="muon",
    excluded_modules=(model.lm_head,),
)
```

LLM 预设下 AdamW 默认 `weight_decay=0.01`，Muon 默认 `weight_decay=0.0`。Muon 和 AdamW fallback
共用同一套 `lr`，顶层 `weight_decay` override 只作用于 AdamW options。需要局部调整学习率时使用
`lr_scale_rules`；需要控制 Muon 专属超参或 Muon weight decay 时，直接使用 `anytrain.optim.muon` 或
`OptimizationConfig(optimizer_options={"muon": ..., "adamw": ...})`。

`create_llm_lightning_optimizers()` 直接返回 Lightning `configure_optimizers()` 可用的 dict：

```python
return create_llm_lightning_optimizers(
    self.model,
    preset="sft",
    schedule="warmup_cosine",
    warmup_steps=1000,
    total_steps=100000,
)
```

如果要保留配置对象装配，可以直接用 `anytrain.optim.llm` 子模块：

```python
config = OptimizationConfig.from_preset(
    "pretrain",
    optimizer="muon",
)
optimizer = create_optimizer_from_config(model, config)
```

子模块里的 `from_preset()` 仍然支持 phase 列表：

```python
config = OptimizationConfig.from_preset(
    "pretrain",
    optimizer="muon",
    scheduler=[("linear", 1000), ("cosine", 10000)],
)
```

高级构造还可以用 `create_optimizer_from_config()` 和 `create_lightning_optimizers_from_config()`，它们保留了完整的 config 对象边界。

## 边界

`optim` 不做：

- 不解释 batch schema。
- 不替下游选择模型结构。
- 不作为顶层硬字段自动注入。
- 不提供要求继承的 LightningModule 基类。
- 不隐藏多 optimizer 或 scheduler 的返回结构；下游仍可直接按 Lightning 原生写法返回。
