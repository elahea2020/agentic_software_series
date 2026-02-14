"""
Base LLM Client interface.

All LLM providers (Anthropic, OpenAI, etc.) implement this interface so that
agents and tools are decoupled from any specific provider. Swapping the
underlying model is a one-line change in your agent.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model})"
