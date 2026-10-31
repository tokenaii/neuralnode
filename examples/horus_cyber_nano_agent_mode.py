"""Horus Cyber Nano 1.0 BETA Agent Mode example."""

import json
import os

import neuralnode as nn


def get_service_status(target: str) -> dict[str, str]:
    """Example local tool executed by the application, not by the model."""
    return {"target": target, "status": "healthy"}


def main() -> None:
    provider = nn.HorusCyberNanoProvider(
        api_key=os.environ["HORUS_CYBER_NANO_API_KEY"],
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_service_status",
                "description": "Read the health status of an approved service.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string"},
                    },
                    "required": ["target"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    messages = [
        {"role": "user", "content": "Check the gateway status and summarize the result."},
    ]

    first = provider.chat(messages, tools=tools, tool_choice="required")
    if not first["tool_calls"]:
        print(first["content"])
        return

    messages.append(
        {
            "role": "assistant",
            "content": first["content"] or None,
            "tool_calls": first["tool_calls"],
        }
    )
    for tool_call in first["tool_calls"]:
        arguments = json.loads(tool_call["function"]["arguments"])
        result = get_service_status(**arguments)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(result),
            }
        )

    final = provider.chat(messages)
    print(final["content"])


if __name__ == "__main__":
    main()
