"""
Base Tool architecture for AI agents.

Every tool in this framework follows a clean contract:
  - prompt:  The instruction that tells the LLM what this tool does and how to use it.
  - input:   A Pydantic model that defines the data the tool accepts.
  - output:  A Pydantic model that defines the data the tool returns.
  - run():   The method that executes the tool logic.

This structure makes tools self-documenting, type-safe, and easy to compose
inside agents.
"""

from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar

from pydantic import BaseModel


class ToolInput(BaseModel):
    """Base class for all tool inputs. Extend this to define your tool's input schema."""


class ToolOutput(BaseModel):
    """Base class for all tool outputs. Extend this to define your tool's output schema."""


InputT = TypeVar("InputT", bound=ToolInput)
OutputT = TypeVar("OutputT", bound=ToolOutput)


class BaseTool(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for all tools.

    Subclasses must define:
      - `prompt`      (str)            : Instructions for the LLM.
      - `input_type`  (Type[InputT])   : The Pydantic input model class.
      - `output_type` (Type[OutputT])  : The Pydantic output model class.
      - `run()`       (method)         : The execution logic.

    Example
    -------
    class MyTool(BaseTool[MyInput, MyOutput]):
        prompt = "You are a tool that does X given Y."
        input_type = MyInput
        output_type = MyOutput

        def run(self, tool_input: MyInput) -> MyOutput:
            ...
    """

    prompt: str
    input_type: Type[InputT]
    output_type: Type[OutputT]

    @abstractmethod
    def run(self, tool_input: InputT) -> OutputT:
        """Execute the tool and return a structured output."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(input={self.input_type.__name__}, output={self.output_type.__name__})"
