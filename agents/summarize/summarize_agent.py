"""
Summarize Agent — entry point for the summarization pipeline.

Design
------
The agent decides whether the input text is short enough to summarize in a
single LLM call, or whether it must be split into overlapping chunks first.

  SHORT text  ──► SummarizeTool ──► SummarizeOutput
  LONG  text  ──► chunk ──► SummarizeTool × N ──► MergeSummariesTool ──► SummarizeOutput

Chunk threshold and overlap are configurable so the agent can be tuned to
any model's context window.

Structured Output
-----------------
SummarizeOutput contains:
  - summary             : str          — the final summary
  - key_takeaways       : List[str]    — bullet-point insights
  - status              : str          — "success" | "failed"
  - original_content_size : int        — character count of the raw input
"""

from __future__ import annotations

import textwrap
from enum import Enum
from typing import List

from pydantic import Field

from agents.base_agent import AgentInput, AgentOutput, BaseAgent
from llm_clients.base_client import BaseLLMClient
from tools.summarize_tool import MergeSummariesTool, SummarizeTool, SummarizeToolInput


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------


class SummarizeStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Input / Output
# ---------------------------------------------------------------------------


class SummarizeInput(AgentInput):
    """Input to the Summarize Agent."""

    text: str = Field(..., description="The text to summarize. Any length is accepted.")


class SummarizeOutput(AgentOutput):
    """Structured output returned by the Summarize Agent."""

    summary: str = Field(..., description="Coherent summary of the input text.")
    key_takeaways: List[str] = Field(
        ..., description="Key insights the reader should remember."
    )
    status: SummarizeStatus = Field(
        ..., description="Whether summarization succeeded or failed."
    )
    original_content_size: int = Field(
        ..., description="Character count of the original input text."
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

# Default chunk size in characters. ~3 000 chars ≈ ~750 tokens — safely under
# most models' context windows even when the system prompt is included.
DEFAULT_CHUNK_SIZE: int = 3_000
DEFAULT_CHUNK_OVERLAP: int = 200  # overlap keeps context across chunk boundaries


class SummarizeAgent(BaseAgent[SummarizeInput, SummarizeOutput]):
    """
    Agent that summarizes text of any length.

    Parameters
    ----------
    llm_client : BaseLLMClient
        The language model used for both chunk summaries and the final merge.
    chunk_size : int
        Maximum characters per chunk when the input is too long.
    chunk_overlap : int
        Characters shared between consecutive chunks to avoid losing context
        at boundaries.

    Example
    -------
    from llm_clients import AnthropicClient
    from llm_clients.base_client import LLMConfig
    from agents.summarize import SummarizeAgent, SummarizeInput

    client = AnthropicClient(LLMConfig(model="claude-opus-4-6"))
    agent  = SummarizeAgent(client)
    result = agent.run(SummarizeInput(text="Very long article..."))
    print(result.summary)
    print(result.key_takeaways)
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        super().__init__(llm_client)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._summarize_tool = SummarizeTool(llm_client)
        self._merge_tool = MergeSummariesTool(llm_client)
        self.tools = [self._summarize_tool, self._merge_tool]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, agent_input: SummarizeInput) -> SummarizeOutput:
        """
        Summarize *agent_input.text*, chunking automatically when necessary.

        Returns a SummarizeOutput with status="success" on success or
        status="failed" with error details in `summary` if an exception occurs.
        """
        text = agent_input.text
        original_size = len(text)

        try:
            if self._needs_chunking(text):
                result = self._summarize_chunked(text)
            else:
                result = self._summarize_direct(text)

            return SummarizeOutput(
                summary=result.summary,
                key_takeaways=result.key_takeaways,
                status=SummarizeStatus.SUCCESS,
                original_content_size=original_size,
            )

        except Exception as exc:  # noqa: BLE001
            return SummarizeOutput(
                summary=f"Summarization failed: {exc}",
                key_takeaways=[],
                status=SummarizeStatus.FAILED,
                original_content_size=original_size,
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _needs_chunking(self, text: str) -> bool:
        """Return True when the text exceeds one chunk's worth of characters."""
        return len(text) > self.chunk_size

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split *text* into overlapping chunks.

        Each chunk is at most `chunk_size` characters long. Consecutive chunks
        share `chunk_overlap` characters so that sentences cut at a boundary
        still appear in full in at least one chunk.
        """
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.chunk_overlap  # slide back by overlap
            if start >= len(text):
                break
        return chunks

    def _summarize_direct(self, text: str):
        """Summarize a short text in a single LLM call."""
        return self._summarize_tool.run(
            SummarizeToolInput(text=text, is_chunk=False)
        )

    def _summarize_chunked(self, text: str):
        """
        Chunk the text, summarize each chunk, then merge all chunk summaries
        into one final summary.
        """
        chunks = self._chunk_text(text)

        chunk_summaries: List[str] = []
        for i, chunk in enumerate(chunks, start=1):
            chunk_result = self._summarize_tool.run(
                SummarizeToolInput(text=chunk, is_chunk=True)
            )
            chunk_summaries.append(
                f"--- Chunk {i} of {len(chunks)} ---\n{chunk_result.summary}"
            )

        combined = "\n\n".join(chunk_summaries)
        merged = self._merge_tool.run(
            SummarizeToolInput(text=combined, is_chunk=False)
        )
        return merged
