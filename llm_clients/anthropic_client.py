"""
Anthropic (Claude) LLM client implementation.

Uses the official `anthropic` Python SDK.  Install it with:
    pip install anthropic
"""

import json
from typing import List

import anthropic

from llm_clients.base_client import BaseLLMClient, LLMConfig, Message


class AnthropicClient(BaseLLMClient):
    """
    LLM client backed by Anthropic's Claude models.

    Usage
    -----
    config = LLMConfig(model="claude-opus-4-6", max_tokens=2048)
    client = AnthropicClient(config, api_key="sk-ant-...")
    reply  = client.complete([Message(role="user", content="Hello!")])
    """

    def __init__(self, config: LLMConfig, api_key: str | None = None) -> None:
        super().__init__(config)
        # api_key=None lets the SDK fall back to the ANTHROPIC_API_KEY env var.
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, messages: List[Message]) -> str:
        """Return the model's plain-text reply."""
        formatted = [{"role": m.role, "content": m.content} for m in messages]

        kwargs = dict(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=formatted,
        )
        if self.config.system_prompt:
            kwargs["system"] = self.config.system_prompt

        response = self._client.messages.create(**kwargs)
        return response.content[0].text

    def complete_json(self, messages: List[Message], schema: dict) -> dict:
        """
        Ask Claude to respond with JSON matching *schema*.

        The method appends an instruction to the last user message and then
        parses the response back into a Python dict.
        """
        schema_str = json.dumps(schema, indent=2)
        json_instruction = (
            f"\n\nRespond ONLY with valid JSON that matches this schema:\n{schema_str}"
            "\nDo not include any explanation or markdown code fences."
        )

        augmented = list(messages)
        if augmented and augmented[-1].role == "user":
            augmented[-1] = Message(
                role="user",
                content=augmented[-1].content + json_instruction,
            )

        raw = self.complete(augmented)

        # Strip accidental markdown fences if the model adds them anyway.
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            raw = raw.rsplit("```", 1)[0]

        return json.loads(raw)
