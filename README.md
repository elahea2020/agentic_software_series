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
│   ├── base_tool.py         # Abstract BaseTool with prompt / input / output contract
│   └── summarize_tool.py    # SummarizeTool + MergeSummariesTool
│
├── agents/
│   ├── base_agent.py        # Abstract BaseAgent
│   └── summarize/
│       └── summarize_agent.py  # SummarizeAgent (auto-chunking + structured output)
│
└── examples/
    └── summarize_example.py    # Runnable end-to-end example
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

## Running the Examples

```bash
# 1. Clone the repo
git clone https://github.com/elahea2020/agentic_software_series.git
cd agentic_software_series

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 4. Run the summarize example
python examples/summarize_example.py
```

---

## Contributing / Following Along

Each tutorial adds new agents and tools to this repo.  Follow the series to
build up your understanding step by step — from a single tool call all the way
to fully autonomous multi-agent pipelines.
