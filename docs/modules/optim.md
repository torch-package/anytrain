# `anytrain.optim` Design

## 定位

`anytrain.optim` 提供 optimizer / scheduler 构造 helper。它不接管训练流程，也不要求下游继承特殊 LightningModule 基类；下游仍然在自己的 `configure_optimizers()` 里显式调用这些 helper。

## 当前结构

源码结构：

```text
src/anytrain/optim/
  __init__.py
  _params.py
  adamw.py
  compose.py
  llm.py
  lora_muon.py
  muon.py
  options.py
  peft.py
  rules.py
  scheduler/
    __init__.py
```

当前公开导出：

- `CompositeOptimizer`
- `ExcludedModules`
- `LRScaleRule`
- `LRScaleRules`
- `LoRAFactorPair`
- `LoRAMuon`
- `create_adamw_optimizer`
- `split_adamw_decay_params`
- `create_muon_adamw_optimizer`
- `muon_available`
- `split_muon_params`
- `create_scheduler`
- `create_llm_optimizer`

高级对象和 options 类型保留在子模块：

- `anytrain.optim.adamw`: `create_adamw_optimizer`
- `anytrain.optim.options`: `AdamWOptions`、`MuonOptions`、`MuonAdamWOptions`、`OptimizerOptions`、`MuonAdjustLRFn`
- `anytrain.optim.muon`: `create_muon_adamw_optimizer`
- `anytrain.optim.lora_muon`: `LoRAFactorPair`、`LoRAMuon`
- `anytrain.optim.peft`: `lora_factor_pairs`、`create_lora_muon_optimizer`、`create_lora_muon_adamw_optimizer`
- `anytrain.optim.scheduler`: `create_scheduler`
- `anytrain.optim.llm`: `create_optimizer`

`anytrain` 的最低依赖是 `torch>=2.4`。Muon 是条件能力：`muon_available()` 在当前
PyTorch 提供 `torch.optim.Muon` 时返回 `True`。旧版 PyTorch 仍可使用 AdamW、scheduler、
参数分组等 API，但创建 Muon optimizer 会直接抛出带版本说明的 `RuntimeError`。

AdamW / Muon 的高级入口接收 typed dict options，不再提供 dataclass config，也不重复校验
torch 原生参数。`anytrain` 只校验自己定义的边界：参数分组、排除 module、lr scale rules、
composite optimizer 参数互斥和命名 scheduler 参数。`betas`、`eps`、`momentum`、`adjust_lr_fn` 等
optimizer 参数由对应的 torch optimizer 暴露错误。scheduler 只保留命名 preset，不提供独立
phase 配置层。

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

## LoRA-Muon

`LoRAMuon` 是 [LoRA-Muon: Spectral Steepest Descent on the Low-Rank Manifold](https://arxiv.org/abs/2606.12921)
Algorithm 1 的 paper-faithful 参考实现。核心 optimizer 不依赖 PEFT，也可以直接接收显式因子对：

```python
from anytrain.optim import LoRAFactorPair, LoRAMuon

optimizer = LoRAMuon(
    [
        LoRAFactorPair(
            left=layer.lora_B["default"].weight,   # [out_features, rank]
            right=layer.lora_A["default"].weight,  # [rank, in_features]
            scale=layer.scaling["default"],
        )
    ],
    lr=0.1,
    momentum=0.95,
    weight_decay=0.0,
)
```

每一对的有效增量是 `scale * left @ right`。`scale` 会被精确吸收到论文的
`A = sqrt(scale) * left`、`B = sqrt(scale) * right.T` 参数化中；它不是额外的学习率缩放。
optimizer 为每对因子保存一个一阶动量，使用精确 SVD matrix sign 和 Gram 矩阵 EVD
inverse root，低精度参数的几何计算暂时提升到 float32。

安装 `peft` extra 后，使用基于类型的 PEFT 入口，不需要按参数名扫描：

```bash
python -m pip install -e ".[peft]"
```

```python
from peft import LoraConfig, get_peft_model

from anytrain.optim.peft import create_lora_muon_optimizer

model = get_peft_model(
    backbone,
    LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        init_lora_weights="pissa",
    ),
)
optimizer = create_lora_muon_optimizer(
    model,
    lr=0.1,
    momentum=0.95,
    weight_decay=0.0,
)
```

`anytrain.optim.peft` 支持 PEFT `>=0.17.1,<0.21`。Python 3.9 会由包解析器选择仍支持
该 Python 版本的 PEFT 0.17.1；Python 3.10+ 可以使用 0.20.x。`adapter_names=None`
默认收集每个 `LoraLayer` 当前 active adapter，也可以显式传多个 adapter name。

- `create_lora_muon_optimizer()` 要求模型所有 trainable parameter 都属于选中的 LoRA A/B。
- `create_lora_muon_adamw_optimizer()` 用于混合训练：LoRA 因子走 `LoRAMuon`，其余 trainable
  参数走 AdamW，并以 `CompositeOptimizer` 暴露；没有剩余参数时退回纯 `LoRAMuon`。
- 存在 trainable bias、`modules_to_save` 或其它非 LoRA 参数时，应使用混合入口或自行调用
  `lora_factor_pairs()` 再组装，避免参数被静默遗漏。

这是实验性参考实现，不是 `torch.optim.Muon` 的别名。两个因子
在构造和每次更新时，paper 参数化中的 `left` 和 `right.T` 都必须保持数值满列秩；标准 LoRA 的一侧零初始化位于 rank-deficient
边界，因此 PEFT 默认的 `init_lora_weights=True` 会明确报错。对零初始化的阻尼扩展尚未实现；
PiSSA、OLoRA 等初始化只有在最终 A/B 都数值满秩时才可用。

PEFT 自动收集首版只支持 active、enabled、unmerged 的官方 vanilla `Linear` adapter；
DoRA/QALoRA/aLoRA/VeLoRA 等 `lora_variant`、LoRA bias、Embedding、卷积、MHA、
parameter/MoE adapter、tensor-parallel 和量化 wrapper 均明确拒绝。LoRA dropout 作用在
adapter 输入上，loss 对参数仍只通过 `scale * B @ A` 依赖，因此可以使用。收集时会拒绝
重复参数和共享 storage。optimizer 构造后不要切换 active adapter，也不要调用 PEFT 的
enable/disable、merge/unmerge、`set_scale()` 或 `scale_layer()`；adapter 集合、运行状态和
scale 都是 optimizer 契约的一部分，需要改变时应重建 optimizer。

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

`warmup_cosine` 要求 `total_steps > warmup_steps`。`wsd` 要求显式提供
`stable_steps` 和 `decay_steps`；如果同时给出 `total_steps`，它必须等于三段步数之和。
三个 preset 都返回 step-level `LambdaLR`，训练预算之外保持最终学习率比例。

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

`create_llm_optimizer(module, preset=..., optimizer=..., lr=..., ...)` 用于按扁平参数创建 LLM 常用 optimizer。`preset` 负责默认值，`optimizer` 决定用 AdamW 还是 Muon 系路径。

```python
optimizer = create_llm_optimizer(
    model,
    preset="pretrain",
    optimizer="muon",
    excluded_modules=(model.lm_head,),
)
```

`optimizer="adamw"` 始终对全部 trainable 参数走 AdamW（含 PEFT LoRA 因子）。

`optimizer="muon"` 按模型内容自动路由：

- 模型含 vanilla PEFT `Linear` LoRA 时，调用
  `create_lora_muon_adamw_optimizer()`：LoRA A/B 走 `LoRAMuon`，其余 trainable 走 AdamW；
  若没有剩余参数则退回纯 `LoRAMuon`。
- 否则走现有 `create_muon_adamw_optimizer()`（Muon + AdamW）。
- 未安装 `peft` 时按无 LoRA 处理；一旦检测到 PEFT Linear 并进入混合入口，构造期校验错误
  （零初始化、不支持 variant、参数漏覆盖等）会直接抛出，不静默回退。

LLM 预设下 AdamW 默认 `weight_decay=0.01`，Muon / LoRA-Muon 默认 `weight_decay=0.0`。
Muon 系路径与 AdamW fallback 共用同一套 `lr`，顶层 `weight_decay` override 只作用于 AdamW
options。需要局部调整学习率时使用 `lr_scale_rules`；需要控制 Muon 专属超参或 Muon weight
decay 时，直接使用 `create_muon_adamw_optimizer()` 或 `create_lora_muon_adamw_optimizer()`。

Lightning 返回结构由下游显式组装：

```python
optimizer = create_llm_optimizer(
    model,
    preset="sft",
)
scheduler = create_scheduler(
    optimizer,
    schedule="warmup_cosine",
    warmup_steps=1000,
    total_steps=100000,
)
return {
    "optimizer": optimizer,
    "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
}
```

## 边界

`optim` 不做：

- 不解释 batch schema。
- 不替下游选择模型结构。
- 不作为顶层硬字段自动注入。
- 不提供要求继承的 LightningModule 基类。
- 不隐藏多 optimizer 或 scheduler 的返回结构；下游仍可直接按 Lightning 原生写法返回。
