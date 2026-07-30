# `anytrain.chat` Design

## 定位

`anytrain.chat` 提供环境变量驱动的轻量有状态大模型调用入口，用于训练工程里的实验总结、日志解释和 LLM 辅助评估原型。它不是训练入口、配置系统或 provider SDK 封装层；下游项目仍然负责决定何时调用、如何记录结果和如何处理失败。

当前公开接口：

```python
from anytrain import chat

response = chat.Chat("deepseek")("summarize this training curve")
```

`anytrain.chat` 导出 `Chat`、公共配置、provider 枚举和返回类型。环境变量名与 provider 默认参数仍是模块实现细节，不进入 `__all__`。

当前 `__all__`：

- `Chat`
- `ChatConfig`
- `ImageGeneration`
- `ImageOutput`
- `ModelType`
- `config_from_env`

同一个 `Chat` 实例会保留已成功请求的 user/assistant 消息，后续请求会把历史消息作为上下文继续发送。需要开启新上下文时，可以调用 `client.refresh()`，或在单次请求中传入 `refresh=True`：

```python
client = chat.Chat("deepseek")
client("summarize epoch 1")
client("compare it with epoch 2")
client.refresh()
client("start a new summary")
client("ignore previous context", refresh=True)
```

## 当前实现

当前模块实现稳定外部契约、实例内消息上下文、环境变量解析、DeepSeek OpenAI-compatible SDK 调用、GLM SDK 调用、OpenRouter OpenAI-compatible SDK 调用和 GLM 图片生成调用。缺少对应 optional 依赖时会明确抛出 `ImportError`，避免静默 fallback。

支持的 `model_type`：

- `deepseek`
- `glm`
- `openrouter`

环境变量：

- `DEEPSEEK_BASE_URL`
- `DEEPSEEK_MODEL`
- `DEEPSEEK_API_KEY`
- `GLM_BASE_URL`
- `GLM_MODEL`
- `GLM_API_KEY`
- `OPENROUTER_API_KEY`（必填）
- `OPENROUTER_BASE_URL`（可选，默认 `https://openrouter.ai/api/v1`）
- `OPENROUTER_MODELS`（可选白名单；未设置时从直播 catalog 自动选）
- `OPENROUTER_CANDIDATE_LIMIT`（可选，默认 `4`；主模型 + 最多 3 个 OpenRouter fallback）

DeepSeek / GLM 变量缺失或为空时，`Chat(...)` 会直接抛出 `ValueError`。OpenRouter 在 `OPENROUTER_API_KEY` 缺失、白名单为空串、或候选 limit 非法时同样直接抛错。

`openrouter` 面向低成本、低智能要求的对话任务（例如给任务生成 prompt），不是 DeepSeek/GLM 的平替智能后端。构造 `Chat("openrouter")` 时会请求 `{base_url}/models`，按直播定价选择候选：

1. 过滤为文本 chat 模型（输入/输出含 `text`、输出不含 `audio`、支持 `temperature`；排除负价格 sentinel）
2. 排序：免费（prompt+completion 为 0、`discount >= 1` 或 id 以 `:free` 结尾）→ 折扣（`discount > 0`）→ 其余按 `prompt+completion` 升序
3. 同价时优先 `openrouter/free`
4. 取前 `OPENROUTER_CANDIDATE_LIMIT` 个作为候选；请求时主模型用第一个，其余最多 3 个放入 `extra_body.models`

若设置了 `OPENROUTER_MODELS`，只在该白名单内做上述实时排序；白名单中不存在于 catalog 的 id 会直接报错。选型结果与每次成功响应使用的 `model` 都会通过 `logging.getLogger("anytrain.chat")` 打 INFO。

## Optional Extra

provider 请求实现放在 `chat` extra 后面，不进入默认依赖：

```bash
python -m pip install openai requests zai-sdk
```

当前 extra 依赖 `openai`、`requests` 和 `zai-sdk`。根包 `import anytrain` 不导入 chat backend，也不导出 `Chat`。

DeepSeek 调用使用：

- `from openai import OpenAI`
- `OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)`
- system prompt: `You are a helpful assistant`
- `stream=False`
- `reasoning_effort="high"`
- `extra_body={"thinking": {"type": "enabled"}}`

GLM 调用使用：

- `from zai import ZhipuAiClient`
- `ZhipuAiClient(api_key=GLM_API_KEY, base_url=GLM_BASE_URL)`
- `thinking={"type": "enabled"}`
- `max_tokens=65536`
- `temperature=1.0`

`GLM_BASE_URL` 应填写 SDK 根路径，例如 `https://open.bigmodel.cn/api/paas/v4`；`zai-sdk` 会自行拼接 `/chat/completions`。

OpenRouter 调用使用：

- 构造时用 `requests` 拉取 `/models` 并完成实时选型
- `from openai import OpenAI`
- `OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)`
- 无强制 system prompt
- `stream=False`
- 主模型为选型后的第一个候选；其余候选放入 `extra_body={"models": [...]}`

GLM 图片生成使用同步 HTTP 调用：

```python
image = chat.Chat("glm").image(
    "一只可爱的小猫咪，坐在阳光明媚的窗台上，背景是蓝天白云.",
    size="1280x1280",
)
```

- 默认模型：`glm-image`
- 默认尺寸：`1280x1280`
- URL：`GLM_BASE_URL` 拼接 `/images/generations`
- 返回值包含 `data`、`raw`，并提供首张图片的 `url` 和 `b64_json` 快捷属性

DeepSeek 与 OpenRouter 当前不支持图片生成，调用 `Chat("deepseek").image(...)` 或
`Chat("openrouter").image(...)` 会抛出 `NotImplementedError`。

## 边界

`anytrain.chat` 不做：

- 不接管 prompt 模板、任务 schema、provider cache 观测或结果记录。
- 不在缺失环境变量时静默换 provider。
- 不把 API key 写入 repr、日志或异常文本。
- 不在 package root 导出 `Chat`。
- 不为图片生成维护额外对话上下文；图片生成是独立请求。

后续扩展 provider 参数时，应保持 `Chat(model_type)(prompt, refresh=False) -> str` 的顶层契约，provider-specific 参数再通过明确配置扩展。
