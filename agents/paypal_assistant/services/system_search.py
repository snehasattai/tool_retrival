"""System Search Tool.

Answers the two meta-questions the task calls out explicitly:
  - "What tools are available for managing invoices?" -> search_available_tools
  - "What's the status of my last request?"            -> get_recent_activity /
                                                           get_request_status

`search_available_tools` queries the same shared Tool Registry the domain
agents' SemanticToolsets use, but scoped to only the four real PayPal
categories -- so introspection answers ("here's what I can do") never leak
one of the synthetic decoy/other-SaaS entries used to stress-test retrieval
at 500+ scale.

`get_recent_activity` / `get_request_status` read from the same run log every
gateway-wrapped tool call writes to (services/run_log.py), giving a live view
into "what actually happened" without a separate logging pipeline.
"""

from __future__ import annotations

from ..tool_registry import index as tool_registry_index
from .gateway import gateway_tool
from .run_log import run_log

_REAL_CATEGORIES = ["invoicing", "payments", "disputes", "reports"]


@gateway_tool(idempotent=False)
def search_available_tools(query: str, top_k: int = 8) -> dict:
    """Search this system's own capabilities for tools matching a description, e.g. "tools for managing invoices".

    Use this when the user asks what the assistant can do, rather than asking
    it to actually do something.
    """
    specs = tool_registry_index.search(query, top_k=top_k, categories=_REAL_CATEGORIES)
    return {
        "tools": [
            {"name": s.name, "category": s.category, "description": s.description}
            for s in specs
            if s.is_real
        ]
    }


@gateway_tool(idempotent=False)
def get_recent_activity(limit: int = 5) -> dict:
    """List the most recent tool calls made in this session, with their status."""
    records = run_log.last(limit)
    return {
        "activity": [
            {
                "call_id": r.call_id,
                "tool": r.tool_name,
                "status": r.result.get("status"),
                "timestamp": r.timestamp,
            }
            for r in records
        ]
    }


@gateway_tool(idempotent=False)
def get_request_status(call_id: str) -> dict:
    """Look up the full status/result of a specific past tool call by its call_id."""
    record = run_log.get(call_id)
    if record is None:
        raise LookupError(f"no record found for call_id '{call_id}'")
    return {
        "call_id": record.call_id,
        "tool": record.tool_name,
        "status": record.result.get("status"),
        "result": record.result,
        "timestamp": record.timestamp,
    }
