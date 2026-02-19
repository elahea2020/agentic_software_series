"""
Base LLM Client interface.

All LLM providers (Anthropic, OpenAI, etc.) implement this interface so that
agents and tools are decoupled from any specific provider. Swapping the
underlying model is a one-line change in your agent.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "user" | "assistant" | "system"
    content: str


@dataclass
class LLMConfig:
    """Configuration shared across all LLM clients."""

    model: str
    max_tokens: int = 4096
    temperature: float = 0.3
    system_prompt: Optional[str] = None


@dataclass
class ToolCall:
    """A single tool call requested by the model."""

    tool_use_id: str
    tool_name: str
    tool_input: dict


@dataclass
class LLMResponse:
    """Structured response from complete_with_tools()."""

    text: str                   # assistant text content (may be empty)
    tool_calls: List[ToolCall]  # tool calls requested (may be empty)
    stop_reason: str            # "end_turn" | "tool_use"
    raw_content: Any            # provider-specific raw content for re-insertion into history


class BaseLLMClient(ABC):
    """
    Abstract interface for LLM providers.

    Subclasses implement `complete()` and `complete_structured()` for their
    specific provider SDK, keeping agents 100 % provider-agnostic.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @abstractmethod
    def complete(self, messages: List[Message]) -> str:
        """
        Send a list of messages to the model and return the assistant reply as
        a plain string.
        """

    @abstractmethod
    def complete_json(self, messages: List[Message], schema: dict) -> dict:
        """
        Send messages and instruct the model to reply with JSON that conforms
        to *schema*.  Returns the parsed dict.
        """

    def complete_with_tools(self, messages: List[dict], tools: List[dict]) -> LLMResponse:
        """
        Send messages with tool definitions and return a structured response.

        *messages* uses the provider's native format (list of dicts), because
        tool-use conversations may contain multi-block content that cannot be
        represented as a plain string.

        *tools* is a list of tool schemas in the provider's native format.

        Returns an LLMResponse with the assistant's text, any tool calls
        requested, and the raw content block for re-insertion into the history.

        Raises NotImplementedError if the subclass does not support tool calling.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support tool calling. "
            "Override complete_with_tools() to add support."
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model})"
