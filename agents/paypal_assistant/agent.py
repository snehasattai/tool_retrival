"""ADK entrypoint. `adk web agents` / `adk run agents/paypal_assistant`
discover this module, preferring `app` over `root_agent` if both are present.

`app` (rather than a bare `root_agent`) is what lets us turn on Gemini
context caching via `ContextCacheConfig` -- that config only exists on `App`,
not on `Agent`, and applies to every LLM agent under this root uniformly.

Caching mainly pays off for `paypal_orchestrator` itself: its tool list
(`query_knowledge_base`, `search_available_tools`, `get_recent_activity`,
`get_request_status`) is fixed, so its system-instruction+tools prefix -- and
the growing prior-turns prefix -- stays byte-identical turn to turn, which is
exactly what a cache fingerprint match requires (see
GeminiContextCacheManager._generate_cache_fingerprint, which hashes system
instruction + tool declarations + the cached-so-far contents).

The four domain sub-agents (invoicing/payments/disputes/reports) are
different: each turn's `SemanticToolset.get_tools()` retrieves a fresh top-k
tool set scoped to that turn's query (tool_registry/semantic_toolset.py), so
their tool declarations legitimately change turn to turn. That changes the
fingerprint, so the cache manager just skips creating a cache for those turns
(fingerprint-only bookkeeping, no wasted API call) -- caching is harmless but
mostly inert there, not a bug. `root_agent` is kept exported too since it's
the simpler object and nothing here needs it to go through `App`.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# Explicit, idempotent load from the repo root .env -- works regardless of
# whether this module is imported by `adk web`, `adk run`, pytest, or a
# standalone script, and never overrides a variable already set in the
# environment (override=False).
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

from google.adk.agents.context_cache_config import ContextCacheConfig  # noqa: E402
from google.adk.apps import App  # noqa: E402

from .orchestrator import build_orchestrator  # noqa: E402

root_agent = build_orchestrator()

app = App(
    name="paypal_assistant",
    root_agent=root_agent,
    context_cache_config=ContextCacheConfig(),
)
