# anytrain

`anytrain` 是一个面向 PyTorch/Lightning 的训练组件库。它提供编写普通
`LightningModule` 时常用的 loss、evaluator、optimizer helper、`torch.nn.Module`
积木、logging mixin 和 debug callback；训练入口、配置系统、运行目录和
`Trainer.fit()` 启动由下游项目自己定义。

核心边界：`anytrain` 帮用户写好自己的训练模块，但不接管工程入口。下游仍然直接继承
Lightning 原生 `LightningModule`，自己实现 batch 解释、训练 step、optimizer/scheduler
和启动脚本。

## 安装

开发环境推荐在项目自己的虚拟环境中安装：

```bash
python -m pip install -e ".[test]"
```

按需安装可选依赖：

```bash
python -m pip install "wandb>=0.16" transformers "tokenizers>=0.20"
python -m pip install -e ".[muon]"
python -m pip install -e ".[peft]"
```

当前包要求 Python `>=3.12`，核心依赖包括 `torch>=2.4` 和 `lightning>=2.0`。原生 Muon
通过 `muon` extra 安装，该 extra 将 PyTorch 约束提升为 `torch>=2.9`；未安装时 AdamW、
scheduler 和参数分组等其余 core API 仍可用。可以用 `anytrain.optim.muon_available()` 查询
当前环境能力。

## Quick Start

下游项目通常只需要写自己的 Lightning module 和启动脚本。`anytrain` 组件通过普通
Python import 显式组合进去：

```python
from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial

import torch
import torch.nn.functional as F
from lightning import pytorch as pl
from torch.optim import Optimizer

from anytrain.lightning import DebugCallback


class RegressionPLModule(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        lr: float = 0.0003,
        optimizer: Callable[[Iterable[torch.nn.Parameter]], Optimizer] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.optimizer = optimizer

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.mse_loss(pred, y)
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        if self.optimizer is not None:
            return self.optimizer(self.parameters())
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def train(data_module: pl.LightningDataModule) -> None:
    module = RegressionPLModule(
        model=torch.nn.Linear(4, 1),
        optimizer=partial(torch.optim.AdamW, lr=0.0003),
    )
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices="auto",
        default_root_dir="outputs/my_project/debug",
        callbacks=[DebugCallback()],
    )
    trainer.fit(module, datamodule=data_module)
```

`DebugCallback` 只在调用方显式加入 callback 列表时启用。它会在每次 backward 后扫描参数和
梯度；正式运行不需要该检查时，应从 callback 列表中移除。

optimizer / scheduler 由下游 `pl_module.configure_optimizers()` 自己创建；多 optimizer、多
模块参数组等复杂逻辑直接按 Lightning 原生写法返回。配置文件如果需要 YAML/JSON/Hydra
支持，也应在下游项目入口里完成对象装配。

仓库内可运行 smoke 示例放在 `examples/`；`anytrain.example` 只提供打包后的示例资源和
读取 helper，不提供训练入口：

```bash
PYTHONPATH=src python examples/tiny_regression.py
```

## 组件

| 模块 | 作用 |
| --- | --- |
| `anytrain.lightning` | Lightning logging mixin、task-agnostic background materialization、长驻服务生命周期、module/loss/gradient logging、debug 和性能观测 callback。 |
| `anytrain.lightning.checkpoint` | canonical contract、state-dict schema 与普通单文件 Lightning checkpoint 的严格恢复。 |
| `anytrain.observation` | 普通 module 的 concrete-type registry、State/Output/Event 来源、Curve/Distribution/Profile/Bins/Heatmap 值契约，以及 step/EMA 状态。 |
| `anytrain.lightning.schedule` | 统一 step/unit LR schedule spec、unit progress、unit LR runtime 和按 unit 停止训练。 |
| `anytrain.lightning.perf` | 参数量、FLOPs、硬件峰值算力和 MFU 计算 helper。 |
| `anytrain.stats` | 纯 tensor 统计 helper，例如按 time bucket 聚合 loss。 |
| `anytrain.loss` | 通用 loss 接口、loss 组合器和 loss balancer。 |
| `anytrain.evaluator` | 通用 evaluator 接口、组合器，以及 text/speech evaluator 子模块。 |
| `anytrain.optim` | AdamW/Muon/LoRA-Muon 参数分组、scheduler helper 和 LLM optimizer helper。 |
| `anytrain.optim.peft` | 依赖 `peft` extra，从 Hugging Face PEFT Linear adapter 构造 LoRA-Muon。 |
| `anytrain.module` | 少量稳定的 task-agnostic `torch.nn.Module` 积木，例如 ADT 和 DiT。 |
| `anytrain.module.moe.conv` | 动态卷积和 router 组件。 |
| `anytrain.module.quantization` | FSQ/VQ/GVQ/RVQ/AGRVQ quantizer。 |
| `anytrain.module.qwen` | Qwen3 复用 helper 和 experimental MTP codebook predictor。 |
| `anytrain.module.token_space` | 多块 local/global token id 映射。 |
| `anytrain.module.embedding` | 共享 embedding owner 与 directional view。 |
| `anytrain.module.stream` | named 1D stream 的声明式 layout、payload 绑定、embedding materialization 与 target routing。 |
| `anytrain.audio` | waveform tokenizer / detokenizer / codec 与 frame BPE 的 optional 包入口。 |
| `anytrain.audio.bpe` | 依赖 `bpe` extra 的 codec frame BPE。 |
| `anytrain.audio.codec` | audio tokenizer / detokenizer capability view、静态 preset spec、frame / semantic-acoustic / semantic-global protocol、feature helper 和共享 backend 的 lazy registry；具体 backend 从 `anytrain.audio.codec.*` 子模块导入。 |
| `anytrain.tts` | TTS 公共协议和可选 backend adapter。 |
| `anytrain.translation` | dataset-independent 批量翻译协议、pinned Qwen semantic spec、lazy factory 和 tokenizer-only token estimator。 |
| `anytrain.chat` | 可选 LLM provider 调用入口。 |
| `anytrain.s2s` | 可选 speech-language 语义生成、token scoring 与领域日志契约；不包含具体模型。 |
| `anytrain.framework` | flow matching、GAN 与 DPO/GRPO 等可选研究训练范式组件。 |
| `anytrain.example` | 随包分发的短音频 smoke fixtures 与 lazy audio loader；训练示例仍在仓库 `examples/`。 |

更细的模块边界见 `docs/modules/index.md`。

## 项目边界

- 数据和 canonical sample 由 `anydataset` 或下游项目负责，`anytrain` 不内置数据适配规则；
  `anytrain.s2s` 只定义跨多个 provider 复用的模型侧 batch 契约。
- 下游项目负责具体任务语义、`training_step`、batch 解释和模型组合。
- 配置组合和对象实例化由下游选择，可以用普通 Python、Hydra、argparse、pydantic 或其它项目内约定。
- `anytrain.lightning` 是核心依赖层；Lightning 不是 optional integration。
- `lightning.checkpoint`、`lightning.schedule`、`lightning.perf`、`loss`、`evaluator`、`optim`、`module.token_space`、`module.embedding`、`translation`、`framework` 是下游训练模块可显式组合的组件，按 core/optional 子模块拆分依赖。

明确不做：

- 不提供 `AnyTrainModule` 这类魔法继承基类。
- 不提供默认训练 CLI 或配置装配入口。
- 不把 optimizer / scheduler / batch schema 做成隐藏注入协议。
- 不把项目私有 task、model zoo 或数据适配规则放进 core。

## 开发

常用检查使用顶层 dev-env 约定的 `py312`：

```bash
PY312=/Users/zhuyin/miniconda3/envs/py312/bin/python
"$PY312" -m pytest
"$PY312" -m ruff check .
"$PY312" -m basedpyright --pythonpath "$PY312"
```

仓库不使用 basedpyright baseline：源码、示例和测试诊断都会让检查失败，不通过基线豁免。
修复诊断后运行同一命令确认通过。

Lightning 边界见 `docs/lightning.md`，组件分层见 `docs/components.md`，总体架构见
`docs/architecture.md`，未实现需求见 `todo.md`。
