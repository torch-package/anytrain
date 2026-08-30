# `anytrain.module.token_space`

`token_space.Layout` 管理命名且不重叠的 global token-id 区间，并提供 local/global 转换。
它不保存 tokenizer special-id，不组合 tokenizer，也不定义 output head 或生成 grammar。

```python
from anytrain.module.token_space import Layout

layout = Layout(text=(0, 32000), audio=(32000, 33024))
global_ids = layout.to_global("audio", local_ids)
local_ids = layout.to_local(global_ids)
```

区间可以留出 gap；gap 的具体用途由调用方维护。跨 block 的 label 转换会显式报错，
`ignore` 参数可用于保留 loss ignore index。
