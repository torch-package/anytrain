# anytrain

## 上层约定

- 本仓库同时受 `../../AGENTS.md` 的顶层开发约定约束。开始任何任务前，先读顶层
  `AGENTS.md`，再读本文件以及下方引用的 `docs/` 文档。
- 涉及本地 Python、虚拟环境、ruff、basedpyright 或 pytest 时，先读
  `../../docs/dev-env.md`。本仓库本地开发和验证默认使用顶层约定的 `py312`
  环境，不使用系统 Python，也不写裸 `python` / `python3` 命令。
- 本文件是 anytrain 的特化约定；如果顶层通用训练工程组织和本文件的组件库边界冲突，
  以本文件的 `项目边界` 为准，因为 anytrain 是 `third_party/` 下的通用训练组件库，
  不是具体训练工程。
- 顶层约定中的小步改动、先最小闭环、显式暴露错误、避免无关重构、实验文档可提交但
  大型训练制品不进入 Git，以及跨模块先读文档等要求在本仓库默认全部适用。

这是从 `deepaudio.lightning` 和训练组件中抽出的 PyTorch/Lightning 训练代码组件库，不再绑定 audio。它重点服务用户自己写普通 `LightningModule` 时会用到的组件，不框架化用户自己的入口、配置系统或 LightningModule。

## 本地开发与验证

- 默认 Python 解释器是
  `/Users/zhuyin/miniconda3/envs/py312/bin/python`；如果先手动
  `conda activate py312`，也要确保命令实际来自该环境。
- 依赖和检查参数以 `pyproject.toml` 为事实来源；不要临时改用 pyright、mypy
  或自造配置。
- 常规验证命令：

```bash
/Users/zhuyin/miniconda3/envs/py312/bin/python -m pytest
/Users/zhuyin/miniconda3/envs/py312/bin/python -m ruff check .
/Users/zhuyin/miniconda3/envs/py312/bin/python -m basedpyright --pythonpath /Users/zhuyin/miniconda3/envs/py312/bin/python
```

## 项目边界

- 数据、canonical sample 和 batch schema 由外部 `anydataset` 或下游项目负责；`anytrain` 不内置数据集适配规则。
- 具体 task 语义、`training_step`、batch 解释和模型组合由下游用户实现。
- 训练入口、配置装配、运行目录和 `fit()` 调用由下游项目自己定义；`anytrain` 不提供默认 app 层。
- `lightning` 负责 task-agnostic 的 LightningModule logging mixin 和训练调试 callback；Lightning 是核心依赖，不是 optional integration。
- 公开接口优先围绕 `pl_module`，不要把单一 `model` 写成硬边界。
- 下游 `pl_module` 直接继承 Lightning 原生 `LightningModule`；不要提供或要求继承 `AnyTrainModule` 这类魔法基类。
- `lightning`、`lightning.schedule`、`lightning.perf`、`loss`、`evaluator`、`optim`、`module` 和 `module.idspace` 构成默认依赖下的核心体验。
- `chat`、`tts`、`framework` 和领域组件按依赖拆分为 optional 子模块；audio codec / BPE 收在 `audio` 下。
- package root 保持轻量，不从 `anytrain` 根包导出跨模块快捷别名。

## 当前模块

- `lightning`：logging mixin、schedule runtime、checkpoint 和训练调试 callback。
- `loss`、`evaluator`：通用接口、组合器及按领域拆分的实现。
- `optim`：optimizer 参数分组、scheduler 和 LLM/Muon helper。
- `module`：task-agnostic `nn.Module` 积木；`module.idspace` 存 token id space。
- `audio`：waveform tokenizer / detokenizer / codec 组合，以及 frame-sequence BPE。
- `chat`：optional general 组件。
- `tts`：optional domain adapter。
- `framework`：flow matching、GAN 等 optional/experimental 训练范式组件。

## 代码组织

- 文件按单一职责和功能归属拆分。
- 当一组文件形成明确、可命名、可独立导入的功能域、公开 API 边界或依赖边界时，将其组织为子包。
- 文件数量只作为目录可读性的信号，不设置按文件数拆子包的机械阈值；同一内聚功能域可以保持扁平，不能只因文件较多而增加包层级。
- 子包入口保持显式且轻量；通过 `__init__.py` 暴露稳定接口，不为缩短 import 路径向上层或 package root 扩散快捷别名。

## 明确避免

- 不迁移 `datasets/`、`data_module/`、具体 `task/`、`wrapper/`、`zoo/`、`_pretrained/`。
- 不把 optimizer / scheduler / batch schema 做成隐藏注入协议。
- 不把 audio codec、source separation、text-to-audio 等任务 step 放进 core。
- 不把 `anydataset` 的数据集适配规则复制进来；数据依赖留给下游项目。
- 不把需要领域依赖的 loss/evaluator 放进 core；放 optional 子模块。

Lightning 边界见 `docs/lightning.md`，组件分层见 `docs/components.md`，正式架构见 `docs/architecture.md`，逐模块设计见 `docs/modules/index.md`，未实现需求见 `todo.md`。
