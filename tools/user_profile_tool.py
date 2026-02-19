"""
UserProfileTool — save or load a gym user's profile from disk.

No LLM call is made; this tool is pure file I/O so the agent always has
accurate, persistent data about the user.
"""

from typing import Literal, List, Optional

from pydantic import Field

from agents.gym_trainer.user_profile import (
    UserProfile,
    load_profile,
    save_profile,
)
from tools.base_tool import BaseTool, ToolInput, ToolOutput


class UserProfileToolInput(ToolInput):
    action: Literal["save", "load"] = Field(
        description="'save' to create/update the profile, 'load' to retrieve it."
    )
    user_id: str = Field(description="Unique identifier for the user (e.g. username).")
    # Fields required only when action == "save"
    name: Optional[str] = Field(default=None, description="User's full name.")
    age: Optional[int] = Field(default=None, description="User's age in years.")
    fitness_level: Optional[Literal["beginner", "intermediate", "advanced"]] = Field(
        default=None, description="Current fitness level."
    )
    goals: Optional[List[str]] = Field(
        default=None,
        description="Fitness goals, e.g. ['lose weight', 'build muscle'].",
    )
    equipment: Optional[List[str]] = Field(
        default=None,
        description="Available equipment, e.g. ['dumbbells', 'bodyweight only'].",
    )
    injuries: Optional[List[str]] = Field(
        default=None,
        description="Injuries or physical limitations, e.g. ['lower back']. Use [] if none.",
    )
    sessions_per_week: Optional[int] = Field(
        default=None, description="How many workout sessions per week the user plans."
    )


class UserProfileToolOutput(ToolOutput):
    success: bool
    message: str
    profile: Optional[dict] = None


class UserProfileTool(BaseTool[UserProfileToolInput, UserProfileToolOutput]):
    """Save or load a user's fitness profile from persistent JSON storage."""

    prompt: str = (
        "Save or retrieve a user's fitness profile. "
        "Use action='save' with all profile fields to create or update a profile. "
        "Use action='load' with just the user_id to retrieve an existing profile."
    )
    input_type = UserProfileToolInput
    output_type = UserProfileToolOutput

    def __init__(self, data_dir: str = "data") -> None:
        self._data_dir = data_dir

    def run(self, tool_input: UserProfileToolInput) -> UserProfileToolOutput:
        if tool_input.action == "load":
            try:
                profile = load_profile(tool_input.user_id, self._data_dir)
                return UserProfileToolOutput(
                    success=True,
                    message=f"Profile loaded for user '{tool_input.user_id}'.",
                    profile=profile.model_dump(),
                )
            except FileNotFoundError:
                return UserProfileToolOutput(
                    success=False,
                    message=f"No profile found for user '{tool_input.user_id}'. Please save one first.",
                )

        # action == "save"
        missing = [
            field
            for field in ("name", "age", "fitness_level", "goals", "equipment", "sessions_per_week")
            if getattr(tool_input, field) is None
        ]
        if missing:
            return UserProfileToolOutput(
                success=False,
                message=f"Cannot save profile — missing required fields: {', '.join(missing)}.",
            )

        profile = UserProfile(
            user_id=tool_input.user_id,
            name=tool_input.name,
            age=tool_input.age,
            fitness_level=tool_input.fitness_level,
            goals=tool_input.goals,
            equipment=tool_input.equipment,
            injuries=tool_input.injuries or [],
            sessions_per_week=tool_input.sessions_per_week,
        )
        save_profile(profile, self._data_dir)
        return UserProfileToolOutput(
            success=True,
            message=f"Profile saved for user '{tool_input.user_id}'.",
            profile=profile.model_dump(),
        )