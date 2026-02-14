"""
Base Agent class.

An Agent orchestrates one or more Tools using an LLM Client to complete a
goal.  The base class wires up the client and provides a common interface so
that all agents in the series follow the same pattern.

Pattern
-------
1. Receive a high-level goal (AgentInput).
2. Decide which tool(s) to call and in what order.
3. Return a structured result (AgentOutput).
"""

from abc import ABC, abstractmethod
from typing import Generic, List, Type, TypeVar

from pydantic import BaseModel

from llm_clients.base_client import BaseLLMClient
from tools.base_tool import BaseTool


class AgentInput(BaseModel):
    """Base class for all agent inputs."""


class AgentOutput(BaseModel):
    """Base class for all agent outputs."""


InputT = TypeVar("InputT", bound=AgentInput)
OutputT = TypeVar("OutputT", bound=AgentOutput)


class BaseAgent(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for all agents.

    Subclasses must implement `run()` and declare which tools they use via the
    `tools` list.

    Attributes
    ----------
    llm_client : BaseLLMClient
        The language model client used to reason and generate text.
    tools : List[BaseTool]
        The tools this agent is allowed to invoke.
    """

    tools: List[BaseTool] = []

    def __init__(self, llm_client: BaseLLMClient) -> None:
        self.llm_client = llm_client

    @abstractmethod
    def run(self, agent_input: InputT) -> OutputT:
        """Execute the agent's goal and return a structured output."""

    def __repr__(self) -> str:
        tool_names = [t.__class__.__name__ for t in self.tools]
        return (
            f"{self.__class__.__name__}("
            f"client={self.llm_client}, tools={tool_names})"
        )
