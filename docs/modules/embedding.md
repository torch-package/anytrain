# `anytrain.module.embedding`

该模块提供与任务无关的共享 embedding 参数拓扑：

- `EmbeddingBank` 只注册一次 canonical owner，并通过 logical alias 表达共享关系；
- `EmbeddingView` 以持久化 row map 把一个方向的 local vocabulary 路由到 owner；
- 可选 prediction map 让输入 rows 与输出 tied/untied rows 独立配置。

view 不注册 bank。模型根节点应注册 bank 一次，再把 view 交给输入和输出接口，避免
重复的 state-dict 路径和 optimizer 参数。owner 只需要是 `nn.Module`，并可提供
`rows(ids)` 以支持 factorized embedding；codec、special token 和生成 grammar 由上层负责。

```python
from anytrain.module.embedding import EmbeddingBank, EmbeddingView

bank = EmbeddingBank(embedding_dim=1024)
bank.add_owner("shared", nn.Embedding(1024, 1024))
```
