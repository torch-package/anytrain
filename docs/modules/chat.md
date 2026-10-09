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

`types.py` 是公开 provider/value/response 类型 owner；`config.py` 只负责凭据、endpoint 与模型候选
配置和环境解析。`ChatConfig` 在 client/catalog 请求前校验非空 endpoint/API key、primary-first unique
model tuple、allowlist 和正整数 candidate limit。`chat.py` 拥有实例消息历史与 sync/async 串行化，
四个文本入口共享同一 prompt/refresh 校验；图片 timeout 要求 `None` 或有限正 host real。

同一个 `Chat` 实例会保留已成功请求的 user/assistant 消息，后续请求会把历史消息作为上下文继续发送。需要开启新上下文时，可以调用 `client.refresh()`，或在单次请求中传入 `refresh=True`：

```python
client = chat.Chat("deepseek")
client("summarize epoch 1")
client("compare it with epoch 2")
client.refresh()
client("start a new summary")
client("ignore previous context", refresh=True)
```

异步调用保留同样的有状态上下文语义：

```python
client = await chat.Chat.async_create("deepseek")
response = await client.async_call("summarize this training curve")
response = await client.acall("compare it with the previous answer")
```

`async_call()` 和 `acall()` 会复用同一个异步 provider client；OpenAI-compatible provider
使用 SDK 的 async client，GLM 当前通过线程回退包住同步 SDK 调用。`async_image()` 同样提供
线程回退的异步图片入口。因为 `Chat` 实例持有会话上下文，同一实例内的异步文本请求会串行执行，
避免并发更新消息历史。

## 当前实现

当前模块实现稳定外部契约、实例内消息上下文、环境变量解析、DeepSeek OpenAI-compatible SDK 调用、GLM SDK 调用、OpenRouter OpenAI-compatible SDK 调用、Zoom DevMate `llm_proxy` OpenAI-compatible SDK 调用和 GLM 图片生成调用。缺少对应 optional 依赖时会明确抛出 `ImportError`，避免静默 fallback。

支持的 `model_type`：

- `deepseek`
- `glm`
- `openrouter`
- `llm_proxy`（Zoom DevMate OpenAI-compatible gateway；pi `llm-proxy` 同款）

环境变量：

- `DEEPSEEK_BASE_URL`
- `DEEPSEEK_MODEL`
- `DEEPSEEK_API_KEY`
- `GLM_BASE_URL`
- `GLM_MODEL`
- `GLM_API_KEY`
- `OPENROUTER_API_KEY`（必填）
- `OPENROUTER_BASE_URL`（可选，默认 `https://openrouter.ai/api/v1`）
- `OPENROUTER_ALLOW_CUSTOM_BASE_URL`（可选；自定义 OpenRouter host 时必须显式设为 `1`）
- `OPENROUTER_MODELS`（可选白名单；未设置时从直播 catalog 自动选）
- `OPENROUTER_CANDIDATE_LIMIT`（可选，默认 `4`；主模型 + 最多 3 个 OpenRouter fallback）
- `LLM_PROXY_API_KEY`（必填；未设时可回退 `DEVMATE_API_KEY`）
- `LLM_PROXY_BASE_URL`（可选，默认 `https://devmate.zoom.us/api/v1/llm-proxy/v1`）
- `LLM_PROXY_MODEL`（可选，默认 `google_gemma_4_31b_it_dev_mate`）

DeepSeek / GLM 变量缺失或为空时，`Chat(...)` 会直接抛出 `ValueError`。OpenRouter 在 `OPENROUTER_API_KEY` 缺失、白名单为空串、或候选 limit 非法时同样直接抛错。
如果 `OPENROUTER_BASE_URL` 指向非官方 host，必须使用 HTTPS 并设置
`OPENROUTER_ALLOW_CUSTOM_BASE_URL=1`；否则不会把 API key 发给该 endpoint。
`llm_proxy` 在 API key 缺失、或显式把 `LLM_PROXY_BASE_URL` / `LLM_PROXY_MODEL` 设为空串时抛错；未设置 URL/模型时使用上述默认值。

`openrouter` 面向低成本、低智能要求的对话任务（例如给任务生成 prompt），不是 DeepSeek/GLM 的平替智能后端。构造 `Chat("openrouter")` 时会请求 `{base_url}/models`，按直播定价选择候选：

1. 过滤为文本 chat 模型（输入/输出含 `text`、输出不含 `audio`、支持 `temperature`；排除负价格 sentinel）
2. 排序：免费（prompt+completion 为 0、`discount >= 1` 或 id 以 `:free` 结尾）→ 折扣（`discount > 0`）→ 其余按 `prompt+completion` 升序
3. 同价时优先 `openrouter/free`
4. 取前 `OPENROUTER_CANDIDATE_LIMIT` 个作为候选；请求时主模型用第一个，其余最多 3 个放入 `extra_body.models`

若设置了 `OPENROUTER_MODELS`，只在该白名单内做上述实时排序；白名单中不存在于 catalog 的 id 会直接报错。选型结果与每次成功响应使用的 `model` 都会通过 `logging.getLogger("anytrain.chat")` 打 INFO。

文本请求会通过 `logging.getLogger("anytrain.chat")` 记录请求生命周期：

- `chat request started`：provider、主模型、发送时的消息条数和 `started_at`。
- `chat request stopped`：provider、主模型、provider 返回的实际模型、`finish_reason`、`elapsed_s`、`stopped_at` 和响应字符数。
- `chat request failed` / `chat request cancelled`：provider、主模型、停止前 `elapsed_s` 和 `stopped_at`；普通异常会带 `exc_info=True`。

日志不会记录 prompt 内容、响应正文或 API key；Python logging formatter 仍可负责输出具体墙钟时间。

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

LLM proxy（DevMate）调用使用：

- `from openai import OpenAI`
- `OpenAI(api_key=LLM_PROXY_API_KEY, base_url=LLM_PROXY_BASE_URL)`
- 无强制 system prompt
- `stream=False`
- 不走 OpenRouter catalog 选型；模型由 `LLM_PROXY_MODEL`（或默认 Gemma）固定
- 不支持 image input / image generation

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

`_request.py` 保留 provider payload、同步/异步调用和生命周期日志的完整状态机，`_response.py` 只做
SDK/dict 响应归一化，`_image.py` 只做 GLM image HTTP 协议，`_openrouter.py` 只做 live catalog 过滤与
排序。OpenRouter pricing 接受 JSON 数字字符串，但拒绝 NaN/Infinity；这是外部 wire-format 解析，
不同于 public Python API 的严格 host-value 契约。

## 验收

```bash
python -m pytest \
  tests/test_chat_api.py tests/test_chat_config.py tests/test_chat_image.py \
  tests/test_chat_openrouter.py tests/test_chat_providers.py
```
