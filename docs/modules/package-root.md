# `anytrain` Package Root Design

## 定位

`anytrain.__init__` 是包的轻量空入口，不触发训练依赖初始化。它不承担训练入口、组件注册或 optional dependency 加载。

## 当前实现

当前不做根包快捷导出：

```python
import anytrain

assert anytrain.__all__ == []
```

对应源码：

- `src/anytrain/__init__.py`

顶层入口分为三类：

- `anytrain.<package>`：按职责显式导入，例如 `anytrain.lightning`、`anytrain.loss`、
  `anytrain.translation` 和 `anytrain.framework.rl`。
- `anytrain.stats` / `anytrain.env` / `anytrain.utils`：没有必要扩成目录的小型公开 primitive，
  分别拥有 tensor statistics、optional backend cache root 和 tensor utility 契约。
- `_audio.py` / `_buffer.py` / `_compat.py` / `_semantic.py`：跨模块私有基础设施，不从根包导出。

`examples/` 是可直接运行的训练示例；`anytrain.example` 只拥有随 wheel 打包的短音频资源与
lazy decoder helper。二者不互相 re-export，也不把训练入口带进安装包 public root。

## 设计原则

- 根包 import 应保持轻量，不隐式导入 Lightning、torchmetrics、plot/audio/text 等重依赖，也不导出跨模块快捷别名。
- 训练入口由下游项目自己维护，根包不提供快捷启动函数。
- 用户需要 module observation 契约时显式使用 `anytrain.observation`。
- 用户需要 checkpoint contract 或严格单文件 Lightning 恢复时显式使用
  `anytrain.lightning.checkpoint`。
- 用户需要 Lightning helper 时显式使用 `anytrain.lightning`。
- 用户需要 loss/evaluator/framework 或 optional backend 时显式使用对应子模块。
- 顶层 package 的 `__all__` 必须无重复且每个名称可解析；根包保持空，不做跨模块聚合。

## 边界

根包不做：

- 不自动注册项目组件。
- 不暴露全量子模块对象。
- 不提供 `run_train` 快捷别名，避免 `import anytrain` 触发 Lightning 相关依赖路径或暗示默认 app 层。
- 不承诺 optional 子模块在默认安装下可导入。
- 不从 `anytrain.example` 暴露 `examples/` 中的 runnable training entry。

## 后续演进

可考虑增加 `__version__`，但不应把训练入口、optional backend 或跨模块 helper 放到根包导出里。
