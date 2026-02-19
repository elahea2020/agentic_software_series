"""
GymTrainerAgent — a conversational AI personal trainer.

The agent runs an interactive chat loop using Anthropic's native tool-calling API.
It decides on its own which tools to invoke based on what the user says, executing
them transparently and weaving the results into a natural conversation.

Tools available
---------------
- UserProfileTool      : save or load a user's fitness profile
- WorkoutGeneratorTool : generate a personalised workout plan
- ProgressTrackerTool  : log a completed session and summarise progress
- FeedbackAdapterTool  : adapt future workouts based on user feedback

Usage
-----
    config = LLMConfig(model="claude-sonnet-4-6", max_tokens=4096, temperature=0.7)
    client = AnthropicClient(config)
    agent  = GymTrainerAgent(client, data_dir="data")
    agent.chat()
"""

import json
from typing import List

from pydantic import Field

from agents.base_agent import AgentInput, AgentOutput, BaseAgent
from llm_clients.base_client import BaseLLMClient, LLMResponse, ToolCall
from tools.feedback_adapter_tool import FeedbackAdapterTool
from tools.progress_tracker_tool import ProgressTrackerTool
from tools.user_profile_tool import UserProfileTool
from tools.workout_generator_tool import WorkoutGeneratorTool

_SYSTEM_PROMPT = """\
You are Coach AI, a friendly, motivating, and knowledgeable personal gym trainer.
You help users achieve their fitness goals through personalised coaching.

You have four tools at your disposal:
- UserProfileTool      : save or load a user's fitness profile (always do this first)
- WorkoutGeneratorTool : create a tailored workout plan
- ProgressTrackerTool  : log a completed workout and track progress
- FeedbackAdapterTool  : adapt the training plan based on user feedback

Guidelines
----------
1. Always ask for the user's name/user_id before calling any tool.
2. Before generating a workout, load the user's profile (action="load").
   If no profile exists, collect: name, age, fitness level, goals, equipment,
   injuries, and sessions per week — then save it (action="save").
3. When logging a session, ask for: exercises done (name, sets, reps, weight),
   energy level (1-5), difficulty rating (1-5), and optional notes.
4. Be encouraging, specific, and concise. Use the user's name when you know it.
5. After tool calls, explain results in plain, motivating language — don't just
   dump raw JSON at the user."""


class GymTrainerInput(AgentInput):
    data_dir: str = Field(default="data", description="Directory for profile and progress files.")


class GymTrainerOutput(AgentOutput):
    status: str = "completed"


class GymTrainerAgent(BaseAgent[GymTrainerInput, GymTrainerOutput]):
    """
    Conversational gym trainer agent with native Anthropic tool calling.

    Call `chat()` to start an interactive session, or `run()` to use the
    standard BaseAgent interface (which delegates to `chat()`).
    """

    def __init__(self, llm_client: BaseLLMClient, data_dir: str = "data") -> None:
        super().__init__(llm_client)
        self._data_dir = data_dir
        self._history: List[dict] = []  # Anthropic-format conversation history

        self.tools = [
            UserProfileTool(data_dir),
            WorkoutGeneratorTool(llm_client),
            ProgressTrackerTool(llm_client, data_dir),
            FeedbackAdapterTool(llm_client, data_dir),
        ]
        self._tool_map = {t.__class__.__name__: t for t in self.tools}

    # ------------------------------------------------------------------
    # Tool schema conversion
    # ------------------------------------------------------------------

    def _get_tool_schemas(self) -> List[dict]:
        """Convert BaseTool instances into Anthropic tool schema dicts."""
        schemas = []
        for tool in self.tools:
            input_schema = tool.input_type.model_json_schema()
            input_schema.pop("title", None)  # Anthropic doesn't need the Pydantic title
            schemas.append(
                {
                    "name": tool.__class__.__name__,
                    "description": tool.prompt,
                    "input_schema": input_schema,
                }
            )
        return schemas

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool(self, tool_call: ToolCall) -> str:
        """Look up the tool by name, run it, and return the result as JSON."""
        tool = self._tool_map.get(tool_call.tool_name)
        if tool is None:
            return json.dumps({"error": f"Unknown tool: {tool_call.tool_name}"})
        try:
            tool_input = tool.input_type(**tool_call.tool_input)
            result = tool.run(tool_input)
            return result.model_dump_json()
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    # ------------------------------------------------------------------
    # Agentic loop
    # ------------------------------------------------------------------

    def _run_agentic_loop(self, tool_schemas: List[dict]) -> None:
        """
        Inner loop: keep calling the LLM until it stops requesting tools.
        Prints any assistant text and shows which tools are being used.
        """
        while True:
            response: LLMResponse = self.llm_client.complete_with_tools(
                self._history, tool_schemas
            )

            if response.text:
                print(f"\nCoach AI: {response.text}\n")

            if response.stop_reason == "end_turn":
                # Append the assistant's final message and wait for user input.
                self._history.append({"role": "assistant", "content": response.raw_content})
                break

            if response.stop_reason == "tool_use":
                # Append the assistant turn that contains the tool_use blocks.
                self._history.append({"role": "assistant", "content": response.raw_content})

                # Execute every requested tool and collect results.
                tool_results = []
                for tc in response.tool_calls:
                    print(f"  [Using {tc.tool_name}...]")
                    result_json = self._execute_tool(tc)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tc.tool_use_id,
                            "content": result_json,
                        }
                    )

                # Feed results back as a user turn and continue the loop.
                self._history.append({"role": "user", "content": tool_results})

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(self) -> None:
        """Start an interactive conversation with the gym trainer."""
        # Apply the trainer system prompt for the duration of the session.
        original_system = self.llm_client.config.system_prompt
        self.llm_client.config.system_prompt = _SYSTEM_PROMPT

        tool_schemas = self._get_tool_schemas()

        print("\n" + "=" * 60)
        print("  Welcome to Coach AI — Your Personal Gym Trainer")
        print("  Type 'quit' or 'exit' to end the session.")
        print("=" * 60)

        # Kick off with a greeting from the agent.
        self._history.append({"role": "user", "content": "Hello, I'm ready to start."})
        self._run_agentic_loop(tool_schemas)

        try:
            while True:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit", "bye"):
                    print(
                        "\nCoach AI: Great work today! Keep pushing — "
                        "every session counts. See you next time!\n"
                    )
                    break

                self._history.append({"role": "user", "content": user_input})
                self._run_agentic_loop(tool_schemas)
        finally:
            self.llm_client.config.system_prompt = original_system

    def run(self, agent_input: GymTrainerInput) -> GymTrainerOutput:
        """Standard BaseAgent entry point — delegates to the interactive chat loop."""
        self._data_dir = agent_input.data_dir
        self.chat()
        return GymTrainerOutput()