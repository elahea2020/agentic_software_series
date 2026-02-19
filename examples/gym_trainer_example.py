"""
Gym Trainer Agent — interactive example.

Run from the project root:
    python examples/gym_trainer_example.py

The agent will greet you and guide you through:
  1. Setting up your fitness profile
  2. Generating a personalised workout plan
  3. Logging completed sessions
  4. Getting feedback-based training adaptations

Type 'quit' or 'exit' to end the session.
"""

import sys
import os

# Allow imports from the project root when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.settings import settings
from llm_clients import AnthropicClient
from llm_clients.base_client import LLMConfig
from agents.gym_trainer.gym_trainer_agent import GymTrainerAgent


def main() -> None:
    config = LLMConfig(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        temperature=0.7,
    )

    client = AnthropicClient(config, api_key=settings.anthropic_api_key)

    # Data files are stored relative to the project root.
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

    agent = GymTrainerAgent(client, data_dir=data_dir)
    agent.chat()


if __name__ == "__main__":
    main()