"""Shared factory for the four domain sub-agents.

Every domain agent has the same shape: a small model, a SemanticToolset
scoped to one registry category, and the gateway's error-normalization
safety net wired in. Only the name/description/instruction differ per
domain, so those are the only things each sub_agents/<domain>.py needs to
supply.
"""

from __future__ import annotations

import os

from google.adk.agents.llm_agent import Agent

from ..services.gateway import on_tool_error_callback
from ..tool_registry.semantic_toolset import SemanticToolset

_SUB_AGENT_MODEL = os.getenv("SUB_AGENT_MODEL", "gemini-2.5-flash")
_TOP_K = int(os.getenv("TOOL_RETRIEVAL_TOP_K", "5"))

_CONFIRMATION_PROTOCOL = """
Sensitive tools that move money or finalize a dispute outcome will respond
with status="confirmation_required" (and a pending_args payload) the first
time you call them instead of executing. When that happens: summarize the
exact action in plain language (amounts, currency, recipient/dispute id) and
ask the user to explicitly confirm. Only call that same tool again, with
confirm=true, after the user clearly approves in their next message. Never
set confirm=true on your own initiative.
""".strip()


def build_domain_agent(*, name: str, description: str, category: str, instruction: str) -> Agent:
    full_instruction = f"{instruction.strip()}\n\n{_CONFIRMATION_PROTOCOL}"
    return Agent(
        name=name,
        model=_SUB_AGENT_MODEL,
        description=description,
        instruction=full_instruction,
        tools=[SemanticToolset(categories=[category], top_k=_TOP_K)],
        on_tool_error_callback=on_tool_error_callback,
    )
