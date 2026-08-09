# `anytrain.module.stream`

`anytrain.module.stream` 是模型输入侧的 named 1-D stream 组合层。它只描述离散
codes、时间轴、embedding 对齐和 flatten 布局，不规定 stream 的语义名称，也不实现
codec inventory、tokenizer grammar 或生成时的预测依赖关系。

## 四个阶段

输入侧接口按逻辑顺序分成四层。前两步只建立和绑定离散布局，最后一步才进入连续
embedding 空间：

```text
Expr  --compile_layout(timelines)-->  Layout
Layout + Codes/Features --bind-->     Bound
Bound  --Materializer-->              EmbeddingBatch
Layout + carrier Codes --bind_targets--> TargetBatch
```

- `Expr` 是无 payload、无 tensor、无 `nn.Module` 的声明。`Stream` 是叶子，`Align`
  是保留 carrier 时间轴的关系，`Flatten` 是序列化结构。
- `compile_layout` 只读取 `Timeline`，编译长度、mask、owner、source index、control
  位置和 Align 几何。这是 Align 的离散阶段，不执行 pooling、插值或 embedding。
- `bind` 把 `Codes` 或 `Features` 绑定到已编译布局，可以分批绑定。它只校验名字、
  shape、lengths、device 和 frame rate。
- `Materializer` 才执行离散 embedder、连续 Align operator、merge 和 flatten gather，
  返回一个 `[B, target_time, D]` 的 `EmbeddingBatch`。因此调用方可以先持有
  `Layout + Codes/Features`，到模型前向边界再 materialize。
- `bind_targets` 是输出侧的局部离散标签视图。它只接受 root layout 的 carrier
  streams；Align 的 source stream 如果没有出现在 carrier lineage 中，不会成为标签。

## 基础类型

```python
Timeline(lengths: int[B], size: int, frame_rate: float | None = None)
Codes(name, values: int[B, T, ...], lengths: int[B], frame_rate=None)
Features(name, values: float[B, T, D], lengths: int[B], frame_rate=None)
```

`frame_rate` 可以是非整数。`size` 是 batch 内的 padded 时间长度，`lengths` 是每个
样本的有效长度。Align 的离散几何按 source/target 的有效长度比例编译；固定长度
stream 可以显式选择 `BROADCAST`，由 operator 先聚合再广播。结构化 codes（例如
`[B,T,Q]` 的 RVQ）原样交给该 stream 的 embedder，layout 不知道 `Q` 的含义。

## Align：离散几何与连续变换

```python
Align(
    name="b_with_c",
    carrier="b",
    source="c",
    spec=AlignSpec(AlignMethod.LINEAR, AlignMerge.ADD),
    operator="default",
)
```

Align 的输出时间轴和离散 lineage 永远来自 `carrier`。`source` 只提供 conditioning
feature；因此 `Align("bc", "b", "c", ...)` 可以放进 `Flatten`，但 flat position 的
owner 仍然是 `b`。Align 的两个操作层是分开的：

- `AlignLayout`（挂在 `CompiledNode.alignment` 上）保存 `indices`、`weights` 或
  `starts/ends`，只表示离散位置关系。
- `AlignOperator` 接收 carrier/source 的 `Features` 和 `AlignLayout`，执行连续变换。
  `StandardAlign` 提供 `IDENTITY`、`NEAREST`、`LINEAR`、`POOL`、`BROADCAST` 和
  `ADD`/`CONCAT`/`SOURCE`；`ATTENTION` 或投影类逻辑通过注册自定义 operator 实现。

离散 code 不会被 linear 插值。若需要在输出侧解释 hard alignment，可以使用布局中的
source index；经过 linear、pool、add 或 projection 的连续结果不能反推可靠离散 label。

Align operand 必须是 timeline expression（`Stream` 或另一个 `Align`），不能直接把
`Flatten` 当作时间轴。反方向的嵌套是允许的：`Flatten` 的 segment 可以包含 Align 或
另一个 Flatten。

## Flatten

```python
Flatten(
    "input",
    (
        StreamMajor(("a", Align("bc", "b", "c", spec))),
        FrameMajor(("q0", "q1")),
    ),
)
```

`StreamMajor` 按 stream 顺序拼接有效帧；`FrameMajor` 按时间帧交错 stream，要求每个
child 逐样本长度和 frame rate 相同，因此可直接描述 RVQ：

```text
q0[0], q1[0], q2[0], q0[1], q1[1], q2[1], ...
```

嵌套 Flatten 会保留每个叶子 stream 的 `owner` 和 `source_index`。同一 DAG 节点可以
在 flatten order 中出现多次，每次都会产生位置，但 Materializer 只计算一次该节点的
embedding 或 Align 结果。

## Control token

`ControlTokens` 只在 Flatten segment 中声明 symbolic name，例如 `"frame:start"`。
编译后 `Layout.control_names` 给出本布局的局部 namespace，control position 的
`owner == CONTROL_OWNER`，且不会伪装成 stream code。

Materializer 通过两项外部配置把 symbolic name 变成 embedding：

```python
Materializer(
    control_embedding=embedding,
    control_ids={"frame:start": 17, "frame:end": 18},
)
```

这样 control 的全局 embedding row 不会因为另一个布局的声明顺序改变。没有 control
position 时不需要提供 control embedding；布局含 control 而缺少映射时会明确报错。

## 输出标签

```python
targets = bind_targets(
    layout,
    {"a": a_codes, "b": b_codes},
    predicted_controls={"frame:start"},
)
```

`TargetBatch.values` 是 `[B,T]` 的 local ids，`owner`/`source_index` 提供回到 carrier
stream 的路由，`predict_mask` 决定哪些位置参与预测。control label 使用
`Layout.control_names` 的局部 id；需要 global vocabulary 时由调用方或
`anytrain.module.idspace` 再做映射。默认 control 不参与预测，只有显式加入
`predicted_controls` 才打开。

## S2S 集成边界

推荐的 source-audio 输入链路是：

```text
local Codes/Features + Timeline
    -> compile_layout / bind
    -> Bound
    -> caller-owned Materializer
    -> EmbeddingBatch
    -> S2S EmbeddingSpan
```

`anytrain` 到 `EmbeddingBatch` 为止。S2S 只把这个连续 span 与自己的 text/control token
span 组成 backbone sequence；它不读取 `Layout` 的 stream 名、flatten order 或 Align
lineage。反过来，S2S 的 output tokenizer、global token ID、BOA/EOA/stream marker、合法
next-token grammar、parser 和 waveform decode 也不进入 `anytrain.module.stream`。

`ControlTokens` 有意停留在 layout 的局部 namespace：它们若位于 materialized audio span
内部，由 `Materializer` 直接变成 embedding；若是 S2S sequence 或 output grammar 的控制
token，则由 S2S 以自己的 global ID 组成离散 token span。这两个 namespace 不应隐式合并。

## 迁移边界

现有 tokenizer 可以按下面的方式映射到本模块，但不会被本模块吞并：

| 现有职责 | stream 表达 | 仍由上层负责 |
| --- | --- | --- |
| codebook-major 序列，每个 codebook 前有 marker | 每个 codebook 一个 `StreamMajor` segment，加 `segment_start` control | marker 的 global id、vocab range、tokenizer encode/decode |
| RVQ frame-major 展开 | 一个 `FrameMajor((q0, q1, ...))` | codebook inventory 和各 codebook 的 id space |
| fixed/global stream 融入 frame stream | `Align(..., AlignMethod.BROADCAST, ...)`，source 作为 conditioning | codec-specific pooling/projector 参数 |
| token offsets、marker range、grammar variant、合法 next token | 不进入 `Layout` | S2S tokenizer grammar / output parser / generation schedule |

因此这层可以替换现有代码里的“时间轴排列和 embedding 组合”部分，但不能无损替换
完整 audio tokenizer。迁移时先把各 stream 的 local codes 和 `Timeline` 交给
`compile_layout`/`bind`，输入侧由 `Materializer` 交付 model-facing embedding；需要生成的
离散输出仍由 S2S id-space、grammar 和 parser 维护 global token contract。

## 非目标

本模块暂不实现：

- semantic/acoustic/global 等固定 stream 角色或 codec 专用名字；
- 把输入 alignment 和输出生成 dependency 合并为一个 DAG；
- output schedule、parser、先预测什么再预测什么的约束；
- 从连续 aligned embedding 反推出 source codes；
- 对离散 code id 做线性插值。

输出依赖可以在上层依据 `Layout` 的 lineage 和 target namespace 另行声明，不应混入
输入侧 `Expr`。
