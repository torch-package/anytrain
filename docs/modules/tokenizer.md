# `anytrain.tokenizer` Design

## 定位

`anytrain.tokenizer` 放 tokenizer 算法组件。统一 token id space 和 token embedding/head
已独立到 [`anytrain.module.idspace`](idspace.md)。

当前公开对象：

- `CodecBPE`
- `EvalStats`

`tokenizer` 是 optional general 模块。安装依赖：

```bash
python -m pip install tokenizers
```

`import anytrain.tokenizer` 保持轻量；构造、加载或训练 `CodecBPE` 时才导入
Hugging Face `tokenizers`，缺失时直接提示安装 `tokenizer` extra。

## CodecBPE

`CodecBPE` 用于在离散 codec frame sequence 上训练和评估 BPE。训练和语料入口只接受
2D 语义的 frame 序列：单 codebook 也写作 `[[id], ...]`，多 codebook 写作
`[[codebook_0_id, codebook_1_id, ...], ...]`。它内部把 frame 按 `codebook_sizes`
mixed-radix 编码成统一的 int sequence，再训练 compact BPE vocab，并持有
`tokenizers.models.BPE` 执行分词。

它支持：

- `train(corpus, codebook_sizes=..., ..., max_frames=1_000_000_000)`
- `encode(frames)`
- `decode(token_ids)`
- `evaluate(corpus, show_progress=True)`
- `save_pretrained(path)` / `from_pretrained(path)`

`path` 是 artifact 目录；`save_pretrained(path)` 只写出 `codec_bpe.json`，
`from_pretrained(path)` 只从该目录读取 `codec_bpe.json`。

基础用法：

```python
from anytrain.tokenizer import CodecBPE

def corpus():
    return [[[1], [2], [1], [2], [3]], [[1], [2], [3]]]

bpe = CodecBPE.train(
    corpus,
    codebook_sizes=(8192,),
    vocab_size=8,
    max_frames=1_000_000_000,
)
token_ids = bpe.encode([[1], [2], [1], [2], [3]])
frames = bpe.decode(token_ids)
stats = bpe.evaluate([[[1], [2], [1], [2], [3]]], show_progress=False)
compression_ratio = stats["compression_ratio"]
vocab_size = bpe.vocab_size
```

多 codebook codec frame 用同一个入口：

```python
def multi_codebook_corpus():
    return [[[1, 4], [2, 7], [1, 4], [2, 7]]]

bpe = CodecBPE.train(
    multi_codebook_corpus,
    codebook_sizes=(4, 16),
    vocab_size=8,
)
token_ids = bpe.encode([[1, 4], [2, 7]])
frames = bpe.decode(token_ids)
```

Python sequence 入口会返回 Python list。空序列、未知 frame / token id 一律报错。
下游如果需要把 BPE token 级张量展开到原始 frame 粒度，应当用 `decode(token_ids)`
返回的 frame 数量自行构造 repeat counts；`CodecBPE` 不再持有 tensor 展开入口。

`CodecBPE` 的 BPE token id 是内部 compact vocab index。原始 codec frame
只通过 `decode()` 还原，不保证和 BPE token id 相同。
encode 唯一走 `tokenizers.models.BPE.tokenize`。训练入口使用
`tokenizers.trainers.BpeTrainer`。当前暴露的训练参数按 `BpeTrainer` 命名和含义对齐：
`vocab_size`、`min_frequency`、`show_progress`、`max_token_length`。`vocab_size` 包含
alphabet，因此如果 corpus 中不同 frame 数量更多，最终 compact vocab 会保留完整 alphabet
并可能大于传入值。`special_tokens`、`limit_alphabet` 和 `initial_alphabet` 这类文本
tokenizer 参数暂不暴露，因为 `CodecBPE` 的每个 token 都必须能无损还原成 codec frame。
`max_frames` 是 `CodecBPE` 自己的语料安全上限，不是 `BpeTrainer` 参数。它统计送入
训练的原始 codec frame，而不是训练后才产生的 BPE token；默认在累计达到
`1_000_000_000` frames 后停止，只接受正整数，传入 `None` 可取消限制。每条 sequence
始终完整保留；frame 数跨 sequence 累计，并在一条 sequence 后达到或超过上限时停止，
不会继续读取下一条。因此实际训练 frame 数可能高于 `max_frames`，超出量小于触发停止的
最后一条 sequence 的 frame 数，但不会引入人为的 sequence 分界。
`show_progress=True` 且 `max_frames` 为整数时，会在交互式终端里额外按已读取的原始
codec frame 显示相对 frame 上限的百分比，并在语料读取结束后报告实际 frame 和 sequence
数。非交互式 job 不渲染动态进度条，每 60 秒向 stderr 写一条 corpus、alphabet 或 eval
静态进度，trainer 阶段开始和结束也会留下静态消息。语料可以在上限前耗尽，此时进度停在
实际比例；触发上限的完整 sequence 仍会全部送入训练，但进度最多显示 100%。
`max_frames=None` 时不接管 `BpeTrainer` 的默认进度。trainer
开始和完成会分别输出静态日志，便于在不渲染动态进度条的 job 日志中确认训练阶段。
能由 Unicode 私用区完整表达时，单 codebook 的 alphabet 直接由 `codebook_sizes` 构造，训练
只扫描 corpus 一遍；合法但未在训练语料出现的 code id 仍可编码。完整 alphabet 会计入
`vocab_size`，因此最终 vocab 至少为单 codebook size。`show_progress=True` 时会明确提示跳过
alphabet scan。超过私用区容量的单 codebook 会回退到 observed alphabet scan。多 codebook 仍会
两遍扫描 corpus：第一遍收集完整 observed frame alphabet 并校验 frame，第二遍交给
`BpeTrainer` 学习 merges。`max_frames` 在每次扫描中独立应用，因此两次都消费相同的完整
sequence 前缀。所有 `train()` corpus 都必须是 callable，并在每次调用时返回新的 frame
sequence iterable；能直接构造完整 alphabet 的单 codebook 调用一次，超出 Unicode 私用区
容量的单 codebook fallback 和多 codebook 都调用两次。
`evaluate()` 也支持 `show_progress`，用于评估大语料压缩率和 BPE token 使用分布时显示独立进度。
它返回扁平的 `EvalStats`：压缩指标里 `original_frames` 是 codec frame 数，
`encoded_tokens` 是 BPE token 数；同时包含 token 出现次数直方图、top-k token、实际使用
token 数、vocab 覆盖率、entropy，以及 vocab / eval 使用视角的 token 展开长度分布。
`top_token_counts` 每项是 `{token_id, count, frequency, length}`。长度分布是 dense
tuple，下标就是 BPE token 展开后的原始 frame 数，例如 `used_token_length_counts[3]`
表示长度为 3 的 BPE token 在 eval corpus 中出现了多少次；tuple 的最大下标就是当前统计中
最大的 BPE 展开长度。`evaluate()` 不返回完整 token-id 到 count 或 length 的大 dict；具体
bpe id 的频率明细只保留 `top_token_counts`，默认 `top_k=100`。
长度定义为一个 BPE token 通过 `decode()` 还原后覆盖的原始 codec frame 数。
Python sequence 入口中，frame 长度必须等于 `len(codebook_sizes)`，每个 code id
必须在对应 book size 范围内。单 codebook 也必须通过 `[id]` 表达，不保留 1D
unit sequence 入口。

`tokenizers` 不进入 package root import 链；只有使用 `CodecBPE` 需要构造底层 BPE model 时才要求安装。

## 边界

`tokenizer` 不做：

- 不定义 shared special token 或多模态 global id space。
- 不管理 embedding/head。
- 不解释多模态 batch schema。
- 不自动处理下游模型的 padding side、label mask 或 loss。
