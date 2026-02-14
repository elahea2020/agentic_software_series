"""
Summarize Agent — end-to-end usage example.

Run:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY="sk-ant-..."
    python examples/summarize_example.py

The script demonstrates two scenarios:
  1. SHORT text  — fits in a single LLM call (no chunking needed).
  2. LONG  text  — automatically chunked, summarized in pieces, then merged.
"""

import sys
import os

# Allow imports from the project root when running directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_clients import AnthropicClient
from llm_clients.base_client import LLMConfig
from agents.summarize import SummarizeAgent, SummarizeInput

# ---------------------------------------------------------------------------
# Sample texts
# ---------------------------------------------------------------------------

SHORT_TEXT = """\
Artificial intelligence is transforming the way we work and live. Machine
learning models can now recognise images, translate languages, and even write
code. Researchers are investigating how these systems can be made safer and
more aligned with human values. The challenge is not only technical but also
ethical: who decides what an AI should optimise for, and how do we ensure it
benefits everyone?\
"""

# Simulate a long document by repeating a paragraph many times.
LONG_TEXT = (
    "The history of computing spans more than a century. "
    "Early mechanical computers gave way to vacuum-tube machines in the 1940s. "
    "Transistors replaced vacuum tubes in the 1950s, making computers smaller and "
    "more reliable. The invention of the integrated circuit in 1958 enabled the "
    "microprocessor revolution of the 1970s, which brought personal computers to "
    "homes and offices around the world. The internet connected these machines "
    "globally in the 1990s, and the smartphone put them in every pocket by the 2010s. "
    "Today, artificial intelligence and cloud computing are the defining trends, "
    "with large language models capable of generating human-quality text, images, "
    "and code. The pace of change shows no sign of slowing down. "
) * 20  # ~3 200 words — well above the default chunk threshold


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def print_result(label: str, result) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)
    print(f"  Status               : {result.status}")
    print(f"  Original content size: {result.original_content_size:,} chars")
    print(f"\n  Summary\n  {'-'*55}")
    print(f"  {result.summary}\n")
    print(f"  Key Takeaways\n  {'-'*55}")
    for i, point in enumerate(result.key_takeaways, start=1):
        print(f"  {i}. {point}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    config = LLMConfig(
        model="claude-opus-4-6",
        max_tokens=2048,
        temperature=0.3,
    )
    client = AnthropicClient(config)

    # chunk_size=3000 means texts ≤ 3 000 chars are handled in one call.
    agent = SummarizeAgent(client, chunk_size=3_000, chunk_overlap=200)

    print("\n=== Summarize Agent Demo ===")

    # --- Scenario 1: short text ---
    short_result = agent.run(SummarizeInput(text=SHORT_TEXT))
    print_result("Scenario 1 — SHORT text (direct, no chunking)", short_result)

    # --- Scenario 2: long text ---
    long_result = agent.run(SummarizeInput(text=LONG_TEXT))
    print_result("Scenario 2 — LONG text (auto-chunked then merged)", long_result)


if __name__ == "__main__":
    main()
