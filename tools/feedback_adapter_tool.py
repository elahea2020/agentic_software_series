"""
FeedbackAdapterTool — analyse recent workouts and user feedback to adapt the training plan.

Loads the user's last 5 sessions from disk and uses the LLM to recommend
intensity adjustments and personalised next-session suggestions.
"""

import json
from typing import List, Literal

from pydantic import Field

from agents.gym_trainer.user_profile import load_sessions
from llm_clients.base_client import BaseLLMClient, Message
from tools.base_tool import BaseTool, ToolInput, ToolOutput

_SYSTEM_PROMPT = """\
You are an expert personal trainer analysing a client's recent workout history and
feedback. Determine whether the training intensity should increase, decrease, or
stay the same. Provide specific, actionable recommendations for the next sessions
and an encouraging message to keep the client motivated."""


class FeedbackAdapterInput(ToolInput):
    user_id: str = Field(description="The user's unique identifier.")
    user_feedback: str = Field(
        description=(
            "The user's free-text feedback about their recent workouts, "
            "e.g. 'workouts feel too easy', 'I'm always sore for days', 'loving the progress'."
        )
    )


class FeedbackAdapterOutput(ToolOutput):
    intensity_adjustment: Literal["increase", "decrease", "maintain"] = Field(
        description="Whether to increase, decrease, or maintain current training intensity."
    )
    adjusted_recommendations: List[str] = Field(
        description="Specific changes to make to future workouts."
    )
    motivation_message: str = Field(description="An encouraging personalised message.")
    next_workout_suggestions: List[str] = Field(
        description="Concrete suggestions for the next 1-2 workout sessions."
    )


class FeedbackAdapterTool(BaseTool[FeedbackAdapterInput, FeedbackAdapterOutput]):
    """Analyse recent workout history and user feedback to adapt the training plan."""

    prompt: str = (
        "Analyse the user's recent workout history and feedback to adapt their training plan. "
        "Requires: user_id and user_feedback (free text describing how recent workouts felt). "
        "Returns an intensity adjustment direction, specific recommendations, a motivational "
        "message, and suggestions for the next sessions."
    )
    input_type = FeedbackAdapterInput
    output_type = FeedbackAdapterOutput

    def __init__(self, llm_client: BaseLLMClient, data_dir: str = "data") -> None:
        self._llm = llm_client
        self._data_dir = data_dir

    def run(self, tool_input: FeedbackAdapterInput) -> FeedbackAdapterOutput:
        recent_sessions = load_sessions(tool_input.user_id, self._data_dir)[-5:]

        history_str = (
            json.dumps(recent_sessions, indent=2)
            if recent_sessions
            else "No sessions logged yet."
        )

        user_message = (
            f"User feedback: {tool_input.user_feedback}\n\n"
            f"Recent workout history (last {len(recent_sessions)} sessions):\n{history_str}"
        )

        schema = FeedbackAdapterOutput.model_json_schema()

        original_system = self._llm.config.system_prompt
        self._llm.config.system_prompt = _SYSTEM_PROMPT
        try:
            raw = self._llm.complete_json(
                [Message(role="user", content=user_message)], schema
            )
        finally:
            self._llm.config.system_prompt = original_system

        return FeedbackAdapterOutput(**raw)