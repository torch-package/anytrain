# `anytrain.module.idspace` Design

## 定位

`anytrain.module.idspace` 只管理统一 token id space 里的命名连续区间，并提供按区间路由的
embedding wrapper。它不训练 tokenizer，不组合 tokenizer，不保存 pad/eos/bos 等业务 token
别名，不定义 batch schema，也不替下游组合跨 block 输出 head。

当前公开对象：

- `Layout`
- `Embedding`

## Layout

`Layout` 是下游 tokenizer 或 codec local id 到统一 global id 的映射。它由一组不重叠
block 构成，每个 block 用字符串命名，并用半开区间 `(start, end)` 描述：

```python
from anytrain.module.idspace import Layout

layout = Layout(
    text=(0, 32000),
    audio=(32000, 33024),
)
```

`Layout` 不知道 special/control token。不同模态如果需要 pad/eos/bos，应由对应 tokenizer、
codec 或下游配置保存 local id，再通过 block 映射到 global id：

```python
text_pad = layout.to_global("text", torch.tensor([text_tokenizer.pad_token_id]))[0]
audio_pad = layout.to_global("audio", torch.tensor([audio_pad_local_id]))[0]
```

这样每个 concrete global id 必须且只能属于一个 block。跨模态共享某个 concrete special id
不是 `Layout` 的目标；如果下游要把多个模态的 pad 当成同一种逻辑角色，应在 batch/schema
层用多个 id 共同表达。

`to_global(name, ids)` 需要显式 block 名，因为 local id 本身没有全局语义。
`to_local(ids)` 不需要 block 名；两个转换入口只接受 `torch.Tensor`，并要求传入的所有
global id 属于同一个 block，否则报错。

```python
text_ids = layout.to_global("text", torch.tensor([0, 1, 2]))
local_ids = layout.to_local(text_ids)
```

`to_local(ids, ignore=pad_id)` 会让等于 `ignore` 的位置原样保留，并且不参与 block
推断和校验。非 ignore 的 id 仍然必须属于同一个 block：

```python
labels = layout.to_local(text_global_ids, ignore=-100)
```

`block_name_for_id(token_id)` 可用于查询单个 global id 所属 block。未知 id、跨 block 的
`to_local()` 输入和非法 block 定义都会显式报错。

## Embedding

`Embedding` 消费 `Layout`，并要求调用方为每个 block 显式传入一个
`nn.Embedding`：

```python
import torch
from torch import nn
from anytrain.module.idspace import Embedding, Layout

layout = Layout(text=(0, 32000), audio=(32000, 33024))
embed = Embedding(
    layout,
    text=nn.Embedding(32000, 1024),
    audio=nn.Embedding(1024, 512),
    adapters={"audio": nn.Linear(512, 1024, bias=False)},
)

input_ids = layout.to_global("text", local_text_ids)
h = embed(input_ids)
```

构造器只做结构绑定：block 名必须和 `Layout` 完全一致；每个 `nn.Embedding` 的
`num_embeddings` 必须等于对应 block 大小。它不默认初始化缺失 block，不推断或缓存
dense global weight，也不提前跑 embedding/adapter probe。

`forward()` 接收 global id tensor，按 block mask 路由到对应 embedding，再应用同名 adapter。
如果同一次 forward 命中的多个 block 输出维不一致，会直接报错。未知 global id 和空输入
也会显式报错；输入 dtype 错误交给 PyTorch embedding 自身暴露。

没有 adapter、所有 block embedding 输出维、dtype 和 device 一致，并且当前不记录参数梯度时，
`forward()` 使用不做逐 block host 同步的快速路径。训练可学习 embedding 时仍使用通用路由路径，
不会为本次未命中的 block materialize 新梯度；配合 optimizer 常用的
`zero_grad(set_to_none=True)`，未命中参数可保持 `grad is None`，避免 AdamW 对它做 weight
decay。如果调用方保留了已有的零梯度，这一层不会替 optimizer 清理。未知 id 只在所有 block
路由结束后同步一次并显式报错。带 adapter 或异构输出时也使用通用路径，且只要求本次实际命中
的 block 输出维一致。

`Embedding` 不提供输出 head。下游如果要做 tied head，直接读取对应 block
embedding 的 `weight`，并显式使用 `layout` 做 label 的 global/local 转换：

```python
import torch.nn.functional as F

logits = F.linear(hidden, embed.embeddings["text"].weight)
labels = layout.to_local(text_global_ids)
```

adapter 只属于 embedding forward 的投影路径，不参与输出 head。需要跨 block 组合 logits
时，也由具体任务在下游显式组织。

## 更新策略

`Embedding` 不内置 optimizer policy。下游直接按真实子模块分组：

```python
optimizer = torch.optim.AdamW(
    [
        {"params": embed.embeddings["text"].parameters(), "lr": 0.0},
        {"params": embed.embeddings["audio"].parameters(), "lr": 1e-4},
        {"params": embed.adapters.parameters(), "lr": 1e-4},
    ]
)
```

## 非目标

当前不做：

- 不训练 tokenizer。
- 不读取或迁移已有 tokenizer/model vocab。
- 不保存 pad/eos/bos 等 token alias。
- 不定义多模态 batch schema。
- 不自动 interleave 不同模态序列。
- 不组合多个 tokenizer。
- 不自动修改下游模型结构或 config。
- 不静默处理未知 id。
- 不组合跨 block 输出 head；下游需要按任务把多个 block head 的 logits 自行组织起来。
