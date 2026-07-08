# PayPal Assistant — a scalable agentic system

A chat agent that operates a mocked 50-call PayPal-style API (invoicing,
payments, disputes, reports) built on **Google's Agent Development Kit
(ADK)**. Ask it things like:

- "Send an invoice for $50 to bob@example.com"
- "What was my total sales volume last month?"
- "Is there a dispute open from user_123?"

The design problem this project is really about: LLM tool-calling accuracy
drops as you bind more tools to a single call. This system is built so
that going from 50 tools to 500+ doesn't degrade routing accuracy — no
LLM call is ever shown more than a handful of tools, no matter how large
the underlying catalog gets.

## How it works, briefly

```
Chat → Orchestrator (picks 1 of 4 domains: invoicing / payments / disputes / reports)
     → Domain agent (only sees tools retrieved for *this* message, via
       embedding search over a shared tool registry — never the full list)
     → Execution gateway (validation, retries, idempotency, confirmation
       for sensitive actions like sending money)
     → Mocked PayPal backend
```

Two extra tools sit directly on the orchestrator: a **RAG tool** for
product/doc questions ("how do disputes work?") and a **system-search
tool** for meta questions ("what tools exist for invoicing?", "what's the
status of my last request?").

The tool registry is seeded with 510 tools — the 50 real ones plus 460
synthetic "decoy" tools from other SaaS products — specifically to prove
retrieval still finds the right tool at scale, not just at 50.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # add your Gemini API key
python scripts/seed_tool_registry.py   # one-time: index the tool registry

./scripts/run_web.sh        # → http://localhost:8000, pick "paypal_assistant"
# or:
adk run agents/paypal_assistant "What was my total sales volume last month?"
```

## Project layout

```
agents/paypal_assistant/
  orchestrator.py           # root agent: routes to a domain, owns RAG + system-search
  sub_agents/                # invoicing / payments / disputes / reports agents
  tool_registry/              # embedding-based tool search (the scaling mechanism)
  services/
    paypal_backend.py         # 50 mocked PayPal functions
    gateway.py                 # validation, retries, idempotency, confirmation gate
    rag_pipeline.py            # RAG tool
    system_search.py           # system-introspection tool
knowledge_base/               # docs the RAG tool searches over
eval/                          # tool-selection accuracy checks
scripts/
  seed_tool_registry.py        # index/re-index the tool registry
  run_web.sh                    # launches adk web with LangSmith tracing wired up
```

## Why ADK

ADK has a `BaseToolset` extension point that gets re-evaluated on every
turn — exactly the hook needed to decide "which tools should this specific
message see" dynamically, instead of binding a fixed list per agent. That
single mechanism is what keeps this system's accuracy flat as the tool
catalog grows.
