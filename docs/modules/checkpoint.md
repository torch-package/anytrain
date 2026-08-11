# `anytrain.checkpoint` Design

## 定位

`anytrain.checkpoint` 提供普通 PyTorch module 和单文件 Lightning checkpoint 之间可复用的
序列化、结构校验与严格恢复 primitive。下游项目继续拥有 checkpoint 路径选择、Trainer resume、
模型版本迁移以及业务 metadata 的解释。

## 公开接口

- `canonical_value()`、`stable_json()`、`canonical_sha256()`：把稳定 contract 规范化、序列化并
  计算摘要。
- `first_difference()`：按稳定路径返回两个 contract 的首个差异。
- `state_dict_schema()`：记录 parameter/buffer 名称、形状和 schema 摘要。
- `load_lightning_checkpoint()`：从路径读取一个完整 Lightning payload，校验 root 和
  `state_dict`，按需调用 module 的 `on_load_checkpoint()`，再以 `strict=True` 恢复 state。
- `LightningCheckpoint`：loader 返回的字符串键 payload 类型。

```python
from anytrain.checkpoint import load_lightning_checkpoint

checkpoint = load_lightning_checkpoint(module, "outputs/step=100.ckpt")
global_step = checkpoint.get("global_step")
```

loader 固定把 checkpoint 映射到 CPU，并显式使用 `weights_only=False`，以保留 Lightning hook
所需的完整 payload。因此调用方只应加载可信 checkpoint。root 与 `state_dict` 必须是字符串键
mapping；module hook 默认在严格 state 恢复前调用，没有该 hook 的普通 `nn.Module` 也可直接使用。
不需要 lifecycle hook 时可显式传入 `call_module_hook=False`。

## 边界

- 不发现最新 checkpoint，不维护运行目录，也不接管 `Trainer.fit(ckpt_path=...)`。
- 不解释 task contract、`global_step`、optimizer/callback state 或其它项目 metadata。
- 不降级为非严格 state 恢复，不提供旧格式 alias 或隐式迁移。
- 当前 loader 面向普通单文件 full checkpoint；FSDP、DeepSpeed 等分片或目录格式由对应
  Lightning strategy 恢复。
