# OpenAI 兼容接口

`serve --package` 在模型包包含 `tokenizer.json` 时启用 `/v1` 兼容路由；Chat Completions 还需要包内聊天模板。接口直接调用本地推理引擎，不需要 OpenAI 账号或 API key。

响应格式参考官方 [Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) 与 [Completions](https://developers.openai.com/api/reference/resources/completions/methods/create)，但兼容范围是明确限定的。

首批已交付长度预算、有效配置读回、文本流式、typed 工具历史与基础 parser。生成约束（`required`、指定 function、`strict` schema）仍待实施，见 [独立方案](../plans/serving/agent-api.md)。扩展按实际使用场景选择，以下只说明当前能力。

## 路由

| 方法与路径 | 行为 |
|---|---|
| `GET /v1/models` | 返回当前 engine 的单个模型，ID 为生效的部署名 |
| `POST /v1/chat/completions` | 用包内模板编码 messages，返回 `chat.completion` 或 SSE |
| `POST /v1/completions` | 编码单个字符串 prompt，返回 `text_completion` 或 SSE |

部署名默认是内部 ModelId 的十进制字符串，可用 `--served-model-name` 覆盖；样例模型默认名为 `"1"`。请求未加载的模型返回 404。包内没有 tokenizer 时上述路由不存在，原生 token API 仍可使用。

## 请求参数

| 参数 | 约定 |
|---|---|
| `model` | 必填字符串，必须匹配 `/v1/models` 返回的部署名 |
| `messages` | 仅 chat，1..256 条；role 为 system/user/assistant/tool，见下节 |
| `prompt` | 仅 completions，非空字符串 |
| `max_tokens` | 正整数，默认 16 |
| `max_completion_tokens` | 仅 chat，与 `max_tokens` 互斥 |
| `temperature` | 0..2；省略时由模型生成配置与模式预设解析，0 为 greedy |
| `seed` | 非负 u64；省略时使用解析后的默认值 |
| `n` | 仅支持 1 |
| `stream` | 布尔值；`true` 返回 SSE，见「流式」 |
| `stream_options` | 仅 `stream=true` 时接受；`include_usage` 追加终末 usage |
| `top_p` | (0,1]；省略时使用解析后的默认值 |
| `top_k` | 非负整数，0 禁用过滤；省略时使用模型默认 |
| `min_p` | 0..1；省略时使用解析后的默认值 |
| `presence_penalty` | -2..2；省略时使用解析后的默认值 |
| `repetition_penalty` | 正有限数；省略时使用解析后的默认值 |
| `enable_thinking` | 布尔值，控制模板与受支持模型的模式预设 |
| `tools` | 仅 chat，至多 128 个 function 工具，名称唯一 |
| `tool_choice` | `"none"` 或 `"auto"`；其他模式返回 501 |
| `parallel_tool_calls` | 仅 `true`；`false` 需要生成约束，返回 501 |

可选字段允许为 `null`，按未提供处理。服务已读取包内 `generation_config.json`，并从 generation config / model config 解析 EOS；识别到的模型还会应用模式预设，显式请求字段最后覆盖。模型包配置了 EOS 时会参与停止判断，未配置时不凭空增加 EOS。请求级 `eos_token` 覆盖使用原生 API。

### 消息与工具历史

`content` 可以是字符串、`null`，或 `{"type":"text","text":...}` 组成的列表；其他 content part 类型返回 501。`null` 按空文本处理，这是只有工具调用的 assistant 消息的常见形式。

assistant 消息可以带 `tool_calls`，每项为 `{"id":..., "type":"function", "function":{"name":..., "arguments":...}}`，`arguments` 是该调用收到或产生的原始 JSON 文本。`tool` 消息必须带 `tool_call_id`，并关联到同一段历史中更早的一次 assistant 调用。id 重复、关联不存在的历史、或把 `tool_calls` / `tool_call_id` 放错 role 都返回 400。历史可以包含本次未声明的工具调用，本次声明集合只约束新产生的调用。

### 长度预算

`--max-model-len` 是单序列总上下文（prompt 与 output 合计），`--max-output-tokens` 是服务输出 cap，`--max-num-seqs` 是并发序列 cap。三者与输入 cap 分别表达，实际生效值和来源可从 `/native/v1/runtime` 的 `lengths` 字段读回。准入检查 `P <= input cap`、`G <= output cap`、`P + G <= L`；显式输出预算超过服务 cap 返回 400，不会静默削短 prompt，显式配置超出模型上限则启动失败。省略输出预算时使用有界默认值，再受服务 cap 限制。

不要把当前输出默认 16 当作上下文上限，也不要假设省略采样参数一定得到 greedy。

不支持的字段（如 `stop`、`response_format`、`logprobs`）与多模态 content 会被拒绝，不会静默忽略。Responses、Embeddings、鉴权、服务端存储均未实现。

## 响应与错误

成功响应包含单个 choice、创建时间和 `usage`。prompt token 数包含实际聊天模板，completion token 数来自完成结果，`finish_reason` 为 `length`、`stop` 或 `tool_calls`。

模型输出按包内模板绑定的方言解析：`Qwen3-VL` 使用 `<tool_call>{"name":...,"arguments":{...}}</tool_call>`，`Qwen3.8-27B` 使用 `<tool_call><function=F><parameter=K>v</parameter></function></tool_call>`。解析不修补模型输出：未闭合的块被扣留，未能解码或未声明的名称留在 `content` 中作为文本，不会伪造调用或改写名称。声明了工具且模型确实调用时，choice 形如：

```json
{
  "index": 0,
  "message": {
    "role": "assistant",
    "content": null,
    "tool_calls": [
      {
        "id": "call_1_0",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
      }
    ]
  },
  "finish_reason": "tool_calls",
  "logprobs": null
}
```

调用 id 由请求 ID 与序号组成，流式与完整返回一致。开启 thinking 时 reasoning 文本单独放在 `reasoning_content`，不计入 `content`。引擎因长度上限停止时 `finish_reason` 仍为 `length`，即使停止前已产生完整调用；调用方应按终末原因决定是否执行。

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

## 流式

`stream=true` 返回 `text/event-stream`，每行 `data:` 一个 JSON chunk，末尾为 `data: [DONE]`。首个 chunk 发布 `delta.role`，随后是 `delta.content`（completions 为 `delta.text`），工具调用以 `delta.tool_calls` 发布，最后是一个带 `finish_reason` 的 chunk；`stream_options.include_usage` 会在终末原因与 `[DONE]` 之间追加一个 `choices` 为空、带 `usage` 的 chunk。

已发布的 `id`、`index` 与 arguments 前缀不会被回写。`function-parameter` 方言需要完整块才能确定参数，因此一个调用在块闭合时整段发布；JSON 方言同样按块发布，不承诺逐 token 的 arguments。把全部 `delta.content` 与 `delta.reasoning_content` 依次拼接，与非流式返回的 `content` 与 `reasoning_content` 相同。

文本流式要求包内 tokenizer 使用 byte-level decoder，才能保证增量解码的前缀稳定；其他 decoder（例如按空格连接 token）会让分段解码与整体解码不一致，这类包请求 `stream=true` 返回 501，而不是发布随后需要更正的字符。

## 实现说明

请求 ID 来自 `RuntimeHandle` 的单调分配器：显式原生请求会推进计数器，路由重建不会重置；混用两套接口时可用 `RuntimeHandle::allocate_request_id` 分配 ID，避免手动 ID 冲突。文本编码与结果解码分别运行在有界 preparation / delivery 池中：流式路径同样把每个已提交 token 送进 delivery 池做增量 detokenize 与解析，兼容层继承原生准入、背压和断连取消机制。
