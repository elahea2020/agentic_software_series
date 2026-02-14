"""
Summarize Tool — one atomic unit of summarization.

Given a piece of text (which may already be a partial/chunk summary), this
tool produces a concise summary using the LLM client provided at construction
time.

This tool is intentionally small so that the SummarizeAgent can call it
multiple times (once per chunk, then once for the final merge).
"""

from __future__ import annotations

from typing import List

from pydantic import Field

from llm_clients.base_client import BaseLLMClient, Message
from tools.base_tool import BaseTool, ToolInput, ToolOutput


# ---------------------------------------------------------------------------
# Input / Output schemas
# ---------------------------------------------------------------------------


class SummarizeToolInput(ToolInput):
    """Input for a single summarization call."""

    text: str = Field(..., description="The text to summarize.")
    is_chunk: bool = Field(
        default=False,
        description=(
            "True when this text is one chunk of a larger document. "
            "The tool adjusts its instructions accordingly."
        ),
    )


class SummarizeToolOutput(ToolOutput):
    """Output of a single summarization call."""

    summary: str = Field(..., description="Concise summary of the input text.")
    key_takeaways: List[str] = Field(
        ..., description="Bullet-point key takeaways extracted from the text."
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

_CHUNK_PROMPT = """\
You are a precise summarization assistant.
You will receive ONE CHUNK from a larger document.
Your job is to:
1. Write a concise summary of this chunk only.
2. Extract up to 5 key takeaways from this chunk.
Keep your response focused on what is in this chunk alone.\
"""

_FULL_PROMPT = """\
You are a precise summarization assistant.
You will receive a complete piece of text.
Your job is to:
1. Write a concise, coherent summary.
2. Extract up to 7 key takeaways — the most important insights a reader should remember.
Be objective and thorough.\
"""

_MERGE_PROMPT = """\
You are a precise summarization assistant.
You will receive several chunk summaries that together cover a large document.
Your job is to:
1. Write a single, unified, coherent summary of the entire document.
2. Extract up to 7 key takeaways — the most important insights from the whole document.
Do not mention that the text was split into chunks.\
"""


class SummarizeTool(BaseTool[SummarizeToolInput, SummarizeToolOutput]):
    """
    Atomic summarization tool.

    Requires an LLM client to generate the summary.  The agent passes its own
    client in so the tool stays stateless and reusable.
    """

    prompt: str = _FULL_PROMPT
    input_type = SummarizeToolInput
    output_type = SummarizeToolOutput

    def __init__(self, llm_client: BaseLLMClient) -> None:
        self._llm = llm_client

    def run(self, tool_input: SummarizeToolInput) -> SummarizeToolOutput:
        system = _CHUNK_PROMPT if tool_input.is_chunk else _FULL_PROMPT

        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "key_takeaways": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["summary", "key_takeaways"],
        }

        messages = [
            Message(role="user", content=tool_input.text),
        ]

        # Temporarily override system prompt for this call.
        original_system = self._llm.config.system_prompt
        self._llm.config.system_prompt = system

        result = self._llm.complete_json(messages, schema)

        self._llm.config.system_prompt = original_system

        return SummarizeToolOutput(
            summary=result["summary"],
            key_takeaways=result["key_takeaways"],
        )


class MergeSummariesTool(BaseTool[SummarizeToolInput, SummarizeToolOutput]):
    """
    Merges multiple chunk summaries into one final summary.

    Reuses the same input/output schema as SummarizeTool but applies a
    dedicated merge prompt.
    """

    prompt: str = _MERGE_PROMPT
    input_type = SummarizeToolInput
    output_type = SummarizeToolOutput

    def __init__(self, llm_client: BaseLLMClient) -> None:
        self._llm = llm_client

    def run(self, tool_input: SummarizeToolInput) -> SummarizeToolOutput:
        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "key_takeaways": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["summary", "key_takeaways"],
        }

        messages = [Message(role="user", content=tool_input.text)]

        original_system = self._llm.config.system_prompt
        self._llm.config.system_prompt = _MERGE_PROMPT

        result = self._llm.complete_json(messages, schema)

        self._llm.config.system_prompt = original_system

        return SummarizeToolOutput(
            summary=result["summary"],
            key_takeaways=result["key_takeaways"],
        )
