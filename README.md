
# Agentic Software Series

> **A practical, beginner-friendly series on building AI agents and agentic systems.**

This repository contains every tutorial, agent, and building block from the series.
Each concept is implemented with a clean, consistent architecture so you can read
any tutorial in isolation and still understand how the pieces fit together.

---

## Core Concepts

| Concept | Where | What it teaches |
|---------|-------|-----------------|
| **LLM Clients** | `llm_clients/` | How to talk to language models in a provider-agnostic way |
| **Tools** | `tools/` | Reusable, self-describing units of capability |
| **Agents** | `agents/` | Orchestrators that combine tools + an LLM to complete a goal |

---

## Architecture

```
agentic_software_series/
├── llm_clients/
│   ├── base_client.py       # Abstract interface (BaseLLMClient, LLMConfig, Message)
│   └── anthropic_client.py  # Anthropic / Claude implementation
│
├── tools/
│   ├── base_tool.py              # Abstract BaseTool with prompt / input / output contract
│   ├── summarize_tool.py         # SummarizeTool + MergeSummariesTool
│   ├── user_profile_tool.py      # Save / load a user's fitness profile
│   ├── workout_generator_tool.py # Generate personalised workout plans
│   ├── progress_tracker_tool.py  # Log sessions and track progress
│   └── feedback_adapter_tool.py  # Adapt training based on feedback
│
├── agents/
│   ├── base_agent.py        # Abstract BaseAgent
│   ├── summarize/
│   │   └── summarize_agent.py      # SummarizeAgent (auto-chunking + structured output)
│   └── gym_trainer/
│       ├── user_profile.py         # UserProfile + WorkoutSession models + JSON helpers
│       └── gym_trainer_agent.py    # Conversational agent with native tool calling
│
├── data/                    # Auto-created at runtime
│   ├── profiles/            # {user_id}.json
│   └── progress/            # {user_id}_progress.json
│
└── examples/
    ├── summarize_example.py        # Runnable end-to-end summarize example
    └── gym_trainer_example.py      # Interactive gym trainer chat session
```

---

## Tool Architecture

Every tool follows the same clean contract:

```python
class MyTool(BaseTool[MyInput, MyOutput]):
    prompt     = "Instructions the LLM uses to understand this tool."
    input_type  = MyInput   # Pydantic model — what the tool accepts
    output_type = MyOutput  # Pydantic model — what the tool returns

    def run(self, tool_input: MyInput) -> MyOutput:
        ...
```

This makes tools:
- **Self-documenting** — `prompt` explains the tool's purpose in plain English.
- **Type-safe** — inputs and outputs are validated by Pydantic.
- **Composable** — agents can call multiple tools in any order.

---

## Tutorial 1 — Summarize Agent

`agents/summarize/summarize_agent.py`

The first agent in the series teaches you how to build a real-world agent that
handles inputs of **any length**.

### What it does

1. **Receives** any amount of text as input.
2. **Decides** whether the text fits in a single LLM call or needs chunking.
3. **Chunks** long text with configurable size and overlap, summarizes each chunk,
   then merges the chunk summaries into one coherent final summary.
4. **Returns** a structured output:

```python
SummarizeOutput(
    summary              = "...",          # Full coherent summary
    key_takeaways        = ["...", "..."], # Bullet-point insights
    status               = "success",      # "success" | "failed"
    original_content_size = 42_000,        # Character count of raw input
)
```

### Quick start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
python examples/summarize_example.py
```

### Usage in code

```python
from llm_clients import AnthropicClient
from llm_clients.base_client import LLMConfig
from agents.summarize import SummarizeAgent, SummarizeInput

client = AnthropicClient(LLMConfig(model="claude-opus-4-6"))
agent  = SummarizeAgent(client)

result = agent.run(SummarizeInput(text="Your article or document goes here..."))

print("Status :", result.status)
print("Size   :", result.original_content_size, "chars")
print("Summary:", result.summary)
for point in result.key_takeaways:
    print(" •", point)
```

---

## Tutorial 2 — Gym Trainer Agent

`agents/gym_trainer/gym_trainer_agent.py`

The second agent introduces **native tool calling** and **conversational agents**.
Instead of running a fixed pipeline, the agent holds a free-form chat with the user
and autonomously decides which tools to call based on what the user says.

### What it does

The agent acts as a personal gym trainer ("Coach AI") and supports four capabilities,
each backed by a dedicated tool:

| Tool | Capability |
|------|------------|
| `UserProfileTool` | Save or load a user's fitness profile (file I/O, no LLM) |
| `WorkoutGeneratorTool` | Generate a personalised workout plan with warmup, exercises, and cooldown |
| `ProgressTrackerTool` | Log a completed session, compute streaks, and summarise progress |
| `FeedbackAdapterTool` | Analyse recent sessions + user feedback and adapt the training intensity |

### How the agentic loop works

```
User message
    ↓
LLM decides what to do (complete_with_tools)
    ├── stop_reason == "end_turn"  → print reply, wait for next message
    └── stop_reason == "tool_use"  → execute tool(s), feed results back, repeat
```

This is the **ReAct pattern** (Reasoning + Acting): the model reasons about what
to do, calls a tool, sees the result, and reasons again until it has a final answer.

### New infrastructure

Two new building blocks were added to support tool calling:

```python
# llm_clients/base_client.py
@dataclass
class ToolCall:
    tool_use_id: str   # ID assigned by the model
    tool_name:   str   # Name of the tool to call
    tool_input:  dict  # Arguments to pass

@dataclass
class LLMResponse:
    text:        str            # Assistant's text (may be empty mid-loop)
    tool_calls:  List[ToolCall] # Tools the model wants to call
    stop_reason: str            # "end_turn" | "tool_use"
    raw_content: Any            # Provider-native blocks for re-insertion into history
```

`AnthropicClient.complete_with_tools()` uses Anthropic's tool-use API and returns
an `LLMResponse`. The agent appends `raw_content` back into the conversation history
so the model has full context on every turn.

### Quick start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
python examples/gym_trainer_example.py
```

### Sample conversation

```
============================================================
  Welcome to Coach AI — Your Personal Gym Trainer
  Type 'quit' or 'exit' to end the session.
============================================================

Coach AI: Hi! I'm Coach AI, your personal gym trainer. What's your name?

You: Hi, I'm Alex. I want to build muscle and lose some weight.

Coach AI: Great to meet you, Alex! Let me set up your profile.
          What's your age and current fitness level?
  ...
  [Using UserProfileTool...]

Coach AI: Profile saved! Ready for your first workout?

You: Yes! Upper body, 45 minutes please.

  [Using WorkoutGeneratorTool...]

Coach AI: Here's your Upper Body Strength workout:

  Warmup (10 min): Arm circles, Band pull-aparts, ...
  Main:   Push-ups 3×12, Dumbbell rows 3×10, ...
  Cooldown: Chest stretch, Lat stretch, ...
```

### Data persistence

Profiles and progress are stored as plain JSON files — no database required:

```
data/
├── profiles/
│   └── alex.json          # UserProfile fields
└── progress/
    └── alex_progress.json # List of WorkoutSession records
```

---

## Running the Examples

```bash
# 1. Clone the repo
git clone https://github.com/elahea2020/agentic_software_series.git
cd agentic_software_series

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 4a. Run the summarize example
python examples/summarize_example.py

# 4b. Run the interactive gym trainer
python examples/gym_trainer_example.py
```

---

## Contributing / Following Along

Each tutorial adds new agents and tools to this repo.  Follow the series to
build up your understanding step by step — from a single tool call all the way
to fully autonomous multi-agent pipelines.
