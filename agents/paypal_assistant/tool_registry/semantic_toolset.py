"""SemanticToolset -- the "Semantic Tool Retrieval" spine from the architecture diagram.

An ADK `BaseToolset` whose `get_tools()` is called fresh on every LLM turn.
Instead of statically listing every tool a domain agent might ever call, it:

  1. Reads the query that started the current invocation off `readonly_context`.
  2. Embeds it and searches the shared Tool Registry (tool_registry/index.py),
     scoped to this agent's category (or categories).
  3. Returns only the top-k matching tools as real ADK `FunctionTool`s.

This is what keeps each domain agent's visible tool count small and constant
(top_k, e.g. 5) regardless of whether the registry backing it holds 13 tools
or 13,000 -- the mechanism that makes the design hold up at 500+ tools per
the task's core challenge. A hard category filter (not just semantic
similarity) is layered underneath it so that even a highly-ranked but
wrong-domain tool (see the "stripe_create_invoice" decoy in
tool_registry/synthetic_specs.py) can never leak into an agent it shouldn't.

Falls back to the category's full tool list (no embedding call) when there is
no query yet, e.g. the very first turn of a session.
"""

from __future__ import annotations

import logging
from typing import Optional

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.function_tool import FunctionTool

from . import index as tool_registry_index

logger = logging.getLogger("google_adk.semantic_toolset")


class SemanticToolset(BaseToolset):
    def __init__(self, categories: list[str], top_k: int = 5, **kwargs):
        super().__init__(**kwargs)
        self._categories = categories
        self._top_k = top_k

    async def get_tools(self, readonly_context: Optional[ReadonlyContext] = None) -> list[BaseTool]:
        query = self._extract_query(readonly_context)

        if query:
            specs = tool_registry_index.search(query, top_k=self._top_k, categories=self._categories)
            logger.info(
                "semantic_toolset categories=%s query=%r -> %s",
                self._categories, query, [s.name for s in specs],
            )
        else:
            specs = tool_registry_index.by_categories(self._categories)

        tools: list[BaseTool] = []
        seen: set[str] = set()
        for spec in specs:
            if spec.func is None or spec.name in seen:
                continue
            seen.add(spec.name)
            tools.append(FunctionTool(spec.func))
        return tools

    @staticmethod
    def _extract_query(readonly_context: Optional[ReadonlyContext]) -> Optional[str]:
        if readonly_context is None or readonly_context.user_content is None:
            return None
        parts = readonly_context.user_content.parts or []
        texts = [p.text for p in parts if getattr(p, "text", None)]
        return " ".join(texts) if texts else None
