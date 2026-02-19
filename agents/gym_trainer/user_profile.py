"""
User profile and workout session data models with JSON file persistence.

Profiles are stored as:  data/profiles/{user_id}.json
Progress history is stored as:  data/progress/{user_id}_progress.json
"""

import json
import os
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """A gym user's personal profile used to tailor workouts."""

    user_id: str
    name: str
    age: int
    fitness_level: Literal["beginner", "intermediate", "advanced"]
    goals: List[str] = Field(
        description="e.g. ['lose weight', 'build muscle', 'improve endurance']"
    )
    equipment: List[str] = Field(
        description="e.g. ['dumbbells', 'barbell', 'resistance bands', 'bodyweight only']"
    )
    injuries: List[str] = Field(
        default_factory=list,
        description="Any injuries or physical limitations, e.g. ['lower back', 'left knee']",
    )
    sessions_per_week: int
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class WorkoutSession(BaseModel):
    """A single completed workout session logged by the user."""

    date: str = Field(default_factory=lambda: datetime.utcnow().date().isoformat())
    focus_area: str
    duration_minutes: int
    exercises_completed: List[dict] = Field(
        description="e.g. [{'name': 'Push-ups', 'sets_done': 3, 'reps_done': '12', 'weight_used': 'bodyweight'}]"
    )
    energy_level: int = Field(ge=1, le=5, description="1 = exhausted, 5 = energised")
    difficulty_rating: int = Field(ge=1, le=5, description="1 = too easy, 5 = too hard")
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def _profiles_dir(data_dir: str) -> str:
    path = os.path.join(data_dir, "profiles")
    os.makedirs(path, exist_ok=True)
    return path


def _progress_dir(data_dir: str) -> str:
    path = os.path.join(data_dir, "progress")
    os.makedirs(path, exist_ok=True)
    return path


def save_profile(profile: UserProfile, data_dir: str) -> None:
    """Persist a UserProfile to disk."""
    profile.updated_at = datetime.utcnow().isoformat()
    filepath = os.path.join(_profiles_dir(data_dir), f"{profile.user_id}.json")
    with open(filepath, "w") as f:
        json.dump(profile.model_dump(), f, indent=2)


def load_profile(user_id: str, data_dir: str) -> UserProfile:
    """Load a UserProfile from disk. Raises FileNotFoundError if not found."""
    filepath = os.path.join(_profiles_dir(data_dir), f"{user_id}.json")
    with open(filepath) as f:
        return UserProfile(**json.load(f))


def save_session(user_id: str, session: WorkoutSession, data_dir: str) -> None:
    """Append a WorkoutSession to the user's progress history."""
    filepath = os.path.join(_progress_dir(data_dir), f"{user_id}_progress.json")
    sessions = load_sessions(user_id, data_dir)
    sessions.append(session.model_dump())
    with open(filepath, "w") as f:
        json.dump(sessions, f, indent=2)


def load_sessions(user_id: str, data_dir: str) -> List[dict]:
    """Load all logged workout sessions for a user. Returns [] if none exist."""
    filepath = os.path.join(_progress_dir(data_dir), f"{user_id}_progress.json")
    if not os.path.exists(filepath):
        return []
    with open(filepath) as f:
        return json.load(f)