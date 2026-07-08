"""The Orchestrator / Supervisor Agent -- the root of the agent tree.

Routing here is deliberately *not* semantic-tool-retrieval based: with only
four domains plus two direct tools, an LLM can pick correctly among ~6
named options every time, so ADK's built-in sub-agent transfer (a cheap,
reliable mechanism) is the right tool for this layer. The semantic retrieval
spine (tool_registry/semantic_toolset.py) is reserved for where it's actually
needed -- inside each domain agent, where the real action-tool count is large
and growing.

RAG (query_knowledge_base) and System Search
(search_available_tools/get_recent_activity/get_request_status) are attached
directly here as plain tools (per the task's framing of them as
general-purpose, cross-cutting tools rather than agents), so the
orchestrator can also directly answer meta/doc questions.
"""

from __future__ import annotations

import os

from google.adk.agents.llm_agent import Agent

from .services.gateway import on_tool_error_callback
from .services.rag_pipeline import query_knowledge_base
from .services.system_search import get_recent_activity, get_request_status, search_available_tools
from .sub_agents import disputes_agent, invoicing_agent, payments_agent, reports_agent

_ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "gemini-2.5-flash")

_INSTRUCTION = """
You are the PayPal Assistant -- a conversational front-end over a PayPal-like
account (invoicing, payments, disputes, reporting) plus product
documentation and the assistant's own capabilities.

Routing:
- Invoicing requests (create/send/update/cancel/remind an invoice) -> transfer to invoicing_agent.
- Payment requests (send/request/refund a payment, balance, payouts, payment methods) -> transfer to payments_agent.
- Dispute requests (lookup, respond, evidence, escalate, resolve, appeal) -> transfer to disputes_agent.
- Reporting/analytics questions (sales volume, transaction history, fees, top customers) -> transfer to reports_agent.
- "How does X work?" / policy questions (e.g. dispute evidence requirements, fee
  structure, invoice statuses) -> call query_knowledge_base yourself.
- "What can you do?" / "what tools exist for X?" -> call search_available_tools yourself.
- "What's the status of my last request / that action?" -> call get_recent_activity
  or get_request_status yourself.

Don't guess which specialist to use if the request is ambiguous -- ask a short
clarifying question first. Never fabricate invoice/payment/dispute ids,
amounts, or report numbers; only state what a tool actually returned.
"""


def build_orchestrator() -> Agent:
    return Agent(
        name="paypal_orchestrator",
        model=_ORCHESTRATOR_MODEL,
        description=(
            "Routes PayPal-related requests to the right specialist agent and "
            "answers documentation/system questions directly."
        ),
        instruction=_INSTRUCTION,
        sub_agents=[invoicing_agent, payments_agent, disputes_agent, reports_agent],
        tools=[query_knowledge_base, search_available_tools, get_recent_activity, get_request_status],
        on_tool_error_callback=on_tool_error_callback,
    )
