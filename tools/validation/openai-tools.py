"""Validate none/auto tool round trips and append-only deltas with the OpenAI Python SDK.

Run against a separately started server with a tool-capable model. The SDK is a validation
dependency only; this tool neither starts a server nor executes model-selected external code.
"""

import argparse
import json
from pathlib import Path

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather and report code for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }
]
REPORT_CODE = "WX_PARIS_21"


def complete(client, request):
    response = client.chat.completions.create(**request)
    choice = response.choices[0]
    return {
        "content": choice.message.content or "",
        "reasoning": getattr(choice.message, "reasoning_content", None) or "",
        "calls": [call.model_dump() for call in choice.message.tool_calls or []],
        "finish_reason": choice.finish_reason,
        "usage": response.usage.model_dump(),
    }


def streamed(client, request):
    result = {"content": "", "reasoning": "", "calls": [], "argument_deltas": 0}
    calls = {}
    finish_reason = None
    usage = None
    with client.chat.completions.create(
        **request, stream=True, stream_options={"include_usage": True}
    ) as response:
        for chunk in response:
            if chunk.usage is not None:
                assert not chunk.choices, chunk.model_dump()
                usage = chunk.usage.model_dump()
            for choice in chunk.choices:
                assert finish_reason is None, "delta received after terminal choice"
                delta = choice.delta
                result["content"] += delta.content or ""
                result["reasoning"] += getattr(delta, "reasoning_content", None) or ""
                for call in delta.tool_calls or []:
                    if call.index not in calls:
                        assert call.index == len(calls), "tool indices are not contiguous"
                        assert call.id and call.type == "function" and call.function.name
                        calls[call.index] = {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.function.name, "arguments": ""},
                        }
                    current = calls[call.index]
                    assert call.id is None or call.id == current["id"]
                    assert call.type is None or call.type == current["type"]
                    assert (
                        call.function.name is None
                        or call.function.name == current["function"]["name"]
                    )
                    arguments = call.function.arguments or ""
                    current["function"]["arguments"] += arguments
                    result["argument_deltas"] += bool(arguments)
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason
    assert finish_reason is not None, "stream has no successful terminal choice"
    assert usage is not None, "stream omitted requested terminal usage"
    result.update(calls=list(calls.values()), finish_reason=finish_reason, usage=usage)
    return result


def compare(left, right):
    for field in ("content", "reasoning", "finish_reason", "usage"):
        assert left[field] == right[field], (field, left[field], right[field])
    assert [call["function"] for call in left["calls"]] == [
        call["function"] for call in right["calls"]
    ], (left["calls"], right["calls"])


def validate(client, model, max_tokens, enable_thinking):
    messages = [
        {
            "role": "user",
            "content": (
                "Use get_weather to look up the weather in Paris. "
                "After the tool result, repeat its report_code exactly."
            ),
        }
    ]
    options = {
        "model": model,
        "temperature": 0,
        "seed": 0,
        "max_completion_tokens": max_tokens,
        "extra_body": {"enable_thinking": enable_thinking},
    }
    request = {**options, "messages": messages, "tools": TOOLS, "tool_choice": "auto"}
    first = complete(client, request)
    first_stream = streamed(client, request)
    compare(first, first_stream)
    assert first["finish_reason"] == "tool_calls", first
    assert len(first["calls"]) == 1, first
    call = first["calls"][0]
    assert call["function"]["name"] == "get_weather", call
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}, call
    assert first_stream["argument_deltas"] > 1, first_stream

    history = [
        *messages,
        {"role": "assistant", "content": first["content"] or None, "tool_calls": first["calls"]},
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "content": json.dumps(
                {"city": "Paris", "temperature_c": 21, "report_code": REPORT_CODE}
            ),
        },
    ]
    final_request = {**options, "messages": history, "tools": TOOLS, "tool_choice": "auto"}
    final = complete(client, final_request)
    final_stream = streamed(client, final_request)
    compare(final, final_stream)
    assert final["finish_reason"] == "stop" and not final["calls"], final
    assert REPORT_CODE in final["content"], final

    none_request = {
        **options,
        "messages": [{"role": "user", "content": "Reply with exactly READY."}],
        "tools": TOOLS,
        "tool_choice": "none",
    }
    none = complete(client, none_request)
    none_stream = streamed(client, none_request)
    compare(none, none_stream)
    assert none["finish_reason"] == "stop" and not none["calls"], none
    return {
        "model": model,
        "enable_thinking": enable_thinking,
        "auto": first,
        "auto_stream": first_stream,
        "result": final,
        "none": none,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Server URL including /v1")
    parser.add_argument("--model", help="Defaults to the only served model")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    from openai import OpenAI

    with OpenAI(base_url=args.base_url, api_key="local", max_retries=0, timeout=120) as client:
        models = client.models.list().data
        if not args.model and len(models) != 1:
            parser.error("--model is required when the server lists multiple models")
        result = validate(client, args.model or models[0].id, args.max_tokens, args.enable_thinking)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        f"none/auto SDK round trip passed: {result['model']}; "
        f"arguments deltas={result['auto_stream']['argument_deltas']}"
    )


if __name__ == "__main__":
    main()
