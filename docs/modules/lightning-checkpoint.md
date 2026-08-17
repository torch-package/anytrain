# `anytrain.lightning.checkpoint` Design

## 定位

`anytrain.lightning.checkpoint` 提供普通 PyTorch module 和单文件 Lightning checkpoint 之间可复用的
序列化、结构校验与严格恢复 primitive，并提供训练 checkpoint series 的离线评估编排。下游项目继续
拥有 Trainer resume、模型版本迁移、单 checkpoint 评估流程以及业务 metadata 的解释。

## 公开接口

- `canonical_value()`、`stable_json()`、`canonical_sha256()`：把稳定 contract 规范化、序列化并
  计算摘要。
- `first_difference()`：按稳定路径返回两个 contract 的首个差异。
- `state_dict_schema()`：记录 parameter/buffer 名称、形状和 schema 摘要。
- `load_checkpoint()`：从路径读取一个完整 Lightning payload，校验 root 和
  `state_dict`，按需调用 module 的 `on_load_checkpoint()`，再以 `strict=True` 恢复 state。
- `LightningCheckpoint`：loader 返回的字符串键 payload 类型。
- `discover_step_checkpoints()`：对一个目录中的 durable `step-<digits>.ckpt` 文件取一次快照，
  排除 `last.ckpt` 与临时文件，并按数值 step 排序。
- `evaluate_checkpoints()`：把快照中的每个 `StepCheckpoint` 依次交给下游 evaluator；异常直接传播，
  因而默认 fail-fast。

```python
from anytrain.lightning.checkpoint import load_checkpoint

checkpoint = load_checkpoint(module, "outputs/step=100.ckpt")
global_step = checkpoint.get("global_step")
```

loader 固定把 checkpoint 映射到 CPU，并显式使用 `weights_only=False`，以保留 Lightning hook
所需的完整 payload。因此调用方只应加载可信 checkpoint。root 与 `state_dict` 必须是字符串键
mapping；module hook 默认在严格 state 恢复前调用，没有该 hook 的普通 `nn.Module` 也可直接使用。
不需要 lifecycle hook 时可显式传入 `call_module_hook=False`。
`module` 必须是 `torch.nn.Module`，`call_module_hook` 必须是严格的 `bool`；这两个控制参数
在访问文件前校验，错误配置不会被路径错误或 Python 真值转换掩盖。

源码保持三个 owner：`contract.py` 负责 canonical value、difference 和 state-dict schema，
`loading.py` 只负责可信单文件 Lightning payload 的严格加载，`evaluation.py` 负责 checkpoint series
发现和 evaluator 调度。

## 边界

- 不猜测 latest checkpoint，不监听持续训练中的目录，不维护项目运行目录，也不接管
  `Trainer.fit(ckpt_path=...)`。
- 不定义 evaluator 的进程模型、输出目录、幂等 marker、重试或任务语义；这些由下游的单 checkpoint
  evaluator 负责。
- 不解释 task contract、`global_step`、optimizer/callback state 或其它项目 metadata。
- 不降级为非严格 state 恢复，不提供旧格式 alias 或隐式迁移。
- 当前 loader 面向普通单文件 full checkpoint；FSDP、DeepSpeed 等分片或目录格式由对应
  Lightning strategy 恢复。

## 验收

```bash
python -m ruff check src/anytrain/lightning/checkpoint tests/test_lightning_checkpoint.py
basedpyright --pythonpath /path/to/python src/anytrain/lightning/checkpoint \
  tests/test_lightning_checkpoint.py
python -m pytest -q tests/test_lightning_checkpoint.py
```
