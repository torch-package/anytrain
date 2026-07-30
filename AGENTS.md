# anytrain

这是从 `deepaudio.lightning` 和训练组件中抽出的 PyTorch/Lightning 训练代码组件库，不再绑定 audio。它重点服务用户自己写普通 `LightningModule` 时会用到的组件，不框架化用户自己的入口、配置系统或 LightningModule。

## 项目边界

- 数据、canonical sample 和 batch schema 由外部 `anydataset` 或下游项目负责；`anytrain` 不内置数据集适配规则。
- 具体 task 语义、`training_step`、batch 解释和模型组合由下游用户实现。
- 训练入口、配置装配、运行目录和 `fit()` 调用由下游项目自己定义；`anytrain` 不提供默认 app 层。
- `lightning` 负责 task-agnostic 的 LightningModule logging mixin 和训练调试 callback；Lightning 是核心依赖，不是 optional integration。
- 公开接口优先围绕 `pl_module`，不要把单一 `model` 写成硬边界。
- 下游 `pl_module` 直接继承 Lightning 原生 `LightningModule`；不要提供或要求继承 `AnyTrainModule` 这类魔法基类。
- `lightning`、`lightning.perf`、`loss`、`evaluator`、`optim`、`module` 和 `module.idspace` 构成默认依赖下的核心体验。
- `chat`、`tokenizer`、`codec`、`tts`、`framework` 和领域组件按依赖拆分为 optional 子模块。
- package root 保持轻量，不从 `anytrain` 根包导出跨模块快捷别名。

## 当前模块

- `lightning`：logging mixin、checkpoint 和训练调试 callback。
- `loss`、`evaluator`：通用接口、组合器及按领域拆分的实现。
- `optim`：optimizer 参数分组、scheduler 和 LLM/Muon helper。
- `module`：task-agnostic `nn.Module` 积木；`module.idspace` 存 token id space。
- `chat`、`tokenizer`：optional general 组件。
- `codec`、`tts`：optional domain adapter。
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
