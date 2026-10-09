# OpenAI 兼容接口

`serve --package` 在模型包包含 `tokenizer.json` 时启用 `/v1` 兼容路由；Chat Completions 还需要包内聊天模板。接口直接调用本地推理引擎，不需要 OpenAI 账号或 API key。

响应格式参考官方 [Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) 与 [Completions](https://developers.openai.com/api/reference/resources/completions/methods/create)，但兼容范围是明确限定的。

工具调用、文本流式和服务长度配置仍待实施，见 [独立方案](../plans/serving/agent-api.md)。扩展按实际使用场景选择，以下只说明当前能力。

## 路由

| 方法与路径 | 行为 |
|---|---|
| `GET /v1/models` | 返回当前 engine 的单个模型，ID 为内部 ModelId 的十进制字符串 |
| `POST /v1/chat/completions` | 用包内模板编码 messages，返回 `chat.completion` |
| `POST /v1/completions` | 编码单个字符串 prompt，返回 `text_completion` |

CLI 样例模型名为 `"1"`，调用方应先查询模型列表。请求未加载的模型返回 404。包内没有 tokenizer 时上述路由不存在，原生 token API 仍可使用。

## 请求参数

| 参数 | 约定 |
|---|---|
| `model` | 必填字符串，必须匹配 `/v1/models` |
| `messages` | 仅 chat，1..256 条；role 为 system/user/assistant，content 为字符串 |
| `prompt` | 仅 completions，非空字符串 |
| `max_tokens` | 正整数，默认 16 |
| `max_completion_tokens` | 仅 chat，与 `max_tokens` 互斥 |
| `temperature` | 0..2；省略时由模型生成配置与模式预设解析，0 为 greedy |
| `seed` | 非负 u64；省略时使用解析后的默认值 |
| `n` | 仅支持 1 |
| `stream` | 仅支持 false |
| `top_p` | (0,1]；省略时使用解析后的默认值 |
| `top_k` | 非负整数，0 禁用过滤；省略时使用模型默认 |
| `min_p` | 0..1；省略时使用解析后的默认值 |
| `presence_penalty` | -2..2；省略时使用解析后的默认值 |
| `repetition_penalty` | 正有限数；省略时使用解析后的默认值 |
| `enable_thinking` | 布尔值，控制模板与受支持模型的模式预设 |

可选字段允许为 `null`，按未提供处理。服务已读取包内 `generation_config.json`，并从 generation config / model config 解析 EOS；识别到的模型还会应用模式预设，显式请求字段最后覆盖。模型包配置了 EOS 时会参与停止判断，未配置时不凭空增加 EOS。请求级 `eos_token` 覆盖使用原生 API。

输入与请求输出上限合计不能超过模型上下文，运行时另有输入与资源预算；目前没有统一的部署 `--max-model-len` 入口。不要把当前输出默认 16 当作上下文上限，也不要假设省略采样参数一定得到 greedy。

不支持的字段（如 `tools`、`stop`、`response_format`、`logprobs`）、多模态 content 与批量 prompt 会被拒绝，不会静默忽略。Responses、Embeddings、鉴权、服务端存储与兼容 SSE 均未实现；需要流式 token ID 时可使用原生 SSE。

## 响应与错误

成功响应包含单个 choice、创建时间和 `usage`。prompt token 数包含实际聊天模板，completion token 数来自完成结果，`finish_reason` 为 `length` 或 `stop`。引擎取消、超时或失败返回错误，不包装成成功结果。

```json
{
  "error": {
    "message": "model is not served; see /v1/models",
    "type": "invalid_request_error",
    "param": null,
    "code": "model_not_found"
  }
}
```

schema 或输入错误返回 400，模型不存在 404，body 超限 413，容量不足 429，已知未实现选项 501，backend 失败 503，无效 Content-Type 保留 415。body 上限为 2 MiB。

请求 ID 来自 `RuntimeHandle` 的单调分配器：显式原生请求会推进计数器，路由重建不会重置；混用两套接口时可用 `RuntimeHandle::allocate_request_id` 分配 ID，避免手动 ID 冲突。文本编码与结果解码分别运行在有界 preparation / delivery 池中，兼容层继承原生准入、背压和断连取消机制。
