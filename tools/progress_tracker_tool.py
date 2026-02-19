"""
ProgressTrackerTool — log a completed workout session and summarise progress.

File I/O (pure Python) handles persistence; the LLM generates a human-readable
progress summary and identifies achievements.
"""

import json
from datetime import datetime, timedelta
from typing import List, Optional

from pydantic import Field

from agents.gym_trainer.user_profile import WorkoutSession, load_sessions, save_session
from llm_clients.base_client import BaseLLMClient, Message
from tools.base_tool import BaseTool, ToolInput, ToolOutput

_SYSTEM_PROMPT = """\
You are an encouraging personal trainer reviewing a client's workout history.
Summarise their recent progress in 2-3 motivating sentences and identify any
noteworthy achievements (e.g. consistency streak, improving difficulty tolerance,
reaching session milestones). Be specific and positive."""


class ProgressTrackerInput(ToolInput):
    user_id: str = Field(description="The user's unique identifier.")
    focus_area: str = Field(description="Muscle group or training style for this session.")
    duration_minutes: int = Field(description="How long the session lasted.")
    exercises_completed: List[dict] = Field(
        description=(
            "Exercises done this session. Each dict should have: "
            "name (str), sets_done (int), reps_done (str), weight_used (str, optional)."
        )
    )
    energy_level: int = Field(ge=1, le=5, description="Energy level 1 (exhausted) – 5 (great).")
    difficulty_rating: int = Field(
        ge=1, le=5, description="Difficulty 1 (too easy) – 5 (too hard)."
    )
    notes: Optional[str] = Field(default=None, description="Any extra notes about the session.")


class ProgressTrackerOutput(ToolOutput):
    total_sessions: int
    streak_days: int
    progress_summary: str
    achievements: List[str]


def _compute_streak(sessions: List[dict]) -> int:
    """Return the current consecutive-day streak (today counts if there's a session)."""
    if not sessions:
        return 0
    dates = sorted(
        {datetime.fromisoformat(s["date"]).date() for s in sessions},
        reverse=True,
    )
    today = datetime.utcnow().date()
    streak = 0
    expected = today
    for d in dates:
        if d == expected:
            streak += 1
            expected -= timedelta(days=1)
        elif d < expected:
            break
    return streak


class ProgressTrackerTool(BaseTool[ProgressTrackerInput, ProgressTrackerOutput]):
    """Log a completed workout session and return a progress summary with achievements."""

    prompt: str = (
        "Log a completed workout session and summarise the user's overall progress. "
        "Requires: user_id, focus_area, duration_minutes, exercises_completed, "
        "energy_level (1-5), difficulty_rating (1-5), and optional notes. "
        "Returns total sessions, current streak, a progress summary, and achievements."
    )
    input_type = ProgressTrackerInput
    output_type = ProgressTrackerOutput

    def __init__(self, llm_client: BaseLLMClient, data_dir: str = "data") -> None:
        self._llm = llm_client
        self._data_dir = data_dir

    def run(self, tool_input: ProgressTrackerInput) -> ProgressTrackerOutput:
        # Persist the new session.
        session = WorkoutSession(
            focus_area=tool_input.focus_area,
            duration_minutes=tool_input.duration_minutes,
            exercises_completed=tool_input.exercises_completed,
            energy_level=tool_input.energy_level,
            difficulty_rating=tool_input.difficulty_rating,
            notes=tool_input.notes,
        )
        save_session(tool_input.user_id, session, self._data_dir)

        # Reload full history for analysis.
        all_sessions = load_sessions(tool_input.user_id, self._data_dir)
        total = len(all_sessions)
        streak = _compute_streak(all_sessions)

        # Ask the LLM for a human-readable summary and achievements.
        recent = all_sessions[-5:]
        history_str = json.dumps(recent, indent=2)
        user_message = (
            f"The user just completed session #{total}.\n"
            f"Streak: {streak} consecutive day(s).\n\n"
            f"Recent sessions (last {len(recent)}):\n{history_str}\n\n"
            "Write a short motivating progress summary and list specific achievements."
        )

        schema = {
            "type": "object",
            "properties": {
                "progress_summary": {"type": "string"},
                "achievements": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["progress_summary", "achievements"],
        }

        original_system = self._llm.config.system_prompt
        self._llm.config.system_prompt = _SYSTEM_PROMPT
        try:
            raw = self._llm.complete_json(
                [Message(role="user", content=user_message)], schema
            )
        finally:
            self._llm.config.system_prompt = original_system

        return ProgressTrackerOutput(
            total_sessions=total,
            streak_days=streak,
            progress_summary=raw["progress_summary"],
            achievements=raw["achievements"],
        )