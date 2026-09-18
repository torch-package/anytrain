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

## Owner contract

新 owner 可以实现 `embedding_contract_state() -> Mapping[str, object]`，显式返回完整的
结构身份；`add_owner(..., state=...)` 提供的 metadata 覆盖同名默认值。bank 每次生成
contract 时调用 provider，并用既有 canonical JSON 值域校验返回值；provider 必须返回
mapping，其内部异常直接传播。此路径不会读取 `weight` 或猜测量化器属性。

没有 provider 的 owner 继续通过 `_embedding_contract.py` 中的 legacy adapter 生成
原有身份，包括 dense 和 factorized owner 的历史字段及摘要。已有 checkpoint 不需要迁移。
新 owner 应由自身描述结构，不再向 legacy adapter 增加 family 分支；不引入 provider registry。
