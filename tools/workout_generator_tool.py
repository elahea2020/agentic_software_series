"""
WorkoutGeneratorTool — generate a personalised workout plan using the LLM.

The tool receives the user's profile attributes and workout preferences, then
asks Claude to return a fully structured plan with warmup, main exercises, and
cooldown.
"""

from typing import List, Literal

from pydantic import BaseModel, Field

from llm_clients.base_client import BaseLLMClient, LLMConfig, Message
from tools.base_tool import BaseTool, ToolInput, ToolOutput

_SYSTEM_PROMPT = """\
You are an expert personal trainer. Generate a safe, effective, and well-structured
workout plan tailored to the user's fitness level, goals, available equipment, and
any injuries or physical limitations. Always include a warmup, main exercises, and
cooldown. Be specific with sets, reps, and rest periods. Provide clear instructions."""


class WorkoutGeneratorInput(ToolInput):
    fitness_level: Literal["beginner", "intermediate", "advanced"] = Field(
        description="User's current fitness level."
    )
    goals: List[str] = Field(description="Fitness goals, e.g. ['build muscle', 'lose weight'].")
    equipment: List[str] = Field(
        description="Available equipment, e.g. ['dumbbells', 'bodyweight only']."
    )
    injuries: List[str] = Field(
        description="Injuries or limitations to work around. Empty list if none."
    )
    focus_area: Literal["full_body", "upper_body", "lower_body", "cardio", "core"] = Field(
        description="Primary muscle group or training style for this session."
    )
    duration_minutes: int = Field(
        default=45, description="Target total workout duration in minutes."
    )


class WarmupExercise(BaseModel):
    exercise: str
    duration_seconds: int
    instructions: str


class MainExercise(BaseModel):
    exercise: str
    sets: int
    reps: str = Field(description="e.g. '10', '8-12', or '30 seconds'")
    rest_seconds: int
    instructions: str


class WorkoutGeneratorOutput(ToolOutput):
    workout_name: str
    warmup: List[WarmupExercise]
    exercises: List[MainExercise]
    cooldown: List[WarmupExercise]
    coach_notes: str


class WorkoutGeneratorTool(BaseTool[WorkoutGeneratorInput, WorkoutGeneratorOutput]):
    """Generate a personalised workout plan tailored to the user's profile and session preferences."""

    prompt: str = (
        "Generate a complete, personalised workout plan. "
        "Requires: fitness_level, goals, equipment, injuries, focus_area, and duration_minutes. "
        "Returns a structured plan with warmup, main exercises (sets/reps/rest), cooldown, and coaching notes."
    )
    input_type = WorkoutGeneratorInput
    output_type = WorkoutGeneratorOutput

    def __init__(self, llm_client: BaseLLMClient) -> None:
        self._llm = llm_client

    def run(self, tool_input: WorkoutGeneratorInput) -> WorkoutGeneratorOutput:
        user_message = (
            f"Create a {tool_input.duration_minutes}-minute {tool_input.focus_area.replace('_', ' ')} workout.\n"
            f"Fitness level: {tool_input.fitness_level}\n"
            f"Goals: {', '.join(tool_input.goals)}\n"
            f"Equipment: {', '.join(tool_input.equipment)}\n"
            f"Injuries/limitations: {', '.join(tool_input.injuries) if tool_input.injuries else 'none'}"
        )

        schema = WorkoutGeneratorOutput.model_json_schema()

        # Temporarily override the system prompt for this structured call.
        original_system = self._llm.config.system_prompt
        self._llm.config.system_prompt = _SYSTEM_PROMPT
        try:
            raw = self._llm.complete_json(
                [Message(role="user", content=user_message)], schema
            )
        finally:
            self._llm.config.system_prompt = original_system

        return WorkoutGeneratorOutput(**raw)