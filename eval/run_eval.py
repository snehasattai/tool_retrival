#!/usr/bin/env python3
"""Tool-selection accuracy eval for the semantic retrieval spine.

This is the "Eval harness / Tool-selection accuracy checks" box in the
architecture diagram. It measures retrieval quality directly against the
Tool Registry (embeddings + Chroma) -- no LLM calls, no agent runs, no API
quota needed beyond the (one-time, already-paid) embedding calls -- so it's
fast and cheap to run in CI on every change to tool descriptions.

For each labeled (query, expected_tool) pair it reports three numbers:

  - "unscoped, no rerank": raw dense retrieval against the FULL registry (510
    tools: 50 real + 460 decoys from other SaaS) -- the hard version of the
    problem, and the baseline the other two columns are measured against.
  - "scoped, no rerank" (production): category filter + plain dense ranking
    -- what each domain sub-agent's SemanticToolset actually uses by default.
  - "scoped + rerank" (experimental, off by default): category filter + the
    IDF-weighted lexical rerank in tool_registry/index.py::search(rerank=True).
    This column exists specifically to show *why* it's off: it net-hurts
    top-1 accuracy here (three previously-correct picks get reranked below a
    wrong one) even though it fixes the one case it targets. Kept in the
    harness so re-enabling it is a measured decision, not a guess, if the
    registry grows into a shape where it might actually help.

Per config, four metrics:

  - top-1:        expected_tool is the very first result.
  - recall@5:      expected_tool appears anywhere in the top TOP_K.
  - mrr:           mean of 1/rank(expected_tool) over all cases (0 if not
                    found within TOP_K). Graded, unlike the two hit-or-miss
                    metrics above -- it's what actually separates "rank 2"
                    from "rank 5" when both still count as a recall@5 hit,
                    and separates "still found, just not #1" from "gone
                    entirely" when both count against top-1. This is the
                    metric that would show *how much* a change like rerank
                    helps or hurts, not just whether it crosses the top-1/
                    top-5 line.
  - category_precision@5 (unscoped only): of the 5 tools the FULL-registry
                    dense search returns with no category filter, what
                    fraction actually belong to the query's expected
                    category. Not printed for the two scoped configs since
                    the hard category filter makes it trivially 100% there
                    by construction (see tool_registry/index.py::search) --
                    it exists to quantify, with a number instead of one
                    hand-picked decoy example, exactly how much cross-domain
                    noise (stripe_*, slack_*, ...) that filter is cutting out.

Usage:
    python eval/run_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(_REPO_ROOT / ".env")

from agents.paypal_assistant.tool_registry import index as tool_registry_index  # noqa: E402

TOP_K = 5


def _rank_of(expected: str, names: list[str]) -> int | None:
    return names.index(expected) + 1 if expected in names else None


def _mrr(ranks: list[int | None]) -> float:
    return sum(1 / r if r else 0 for r in ranks) / len(ranks)


def run() -> None:
    cases = json.loads((_REPO_ROOT / "eval" / "tool_selection_eval.json").read_text())
    tool_registry_index.get_collection()

    hits1 = {"unscoped": 0, "scoped_norerank": 0, "scoped_rerank": 0}
    hits5 = {"unscoped": 0, "scoped_norerank": 0, "scoped_rerank": 0}
    all_ranks: dict[str, list[int | None]] = {"unscoped": [], "scoped_norerank": [], "scoped_rerank": []}
    category_precision_total = 0.0
    rows = []

    for case in cases:
        query, category, expected = case["query"], case["category"], case["expected_tool"]

        unscoped_specs = tool_registry_index.search(query, top_k=TOP_K)
        unscoped = [s.name for s in unscoped_specs]
        scoped_norerank = [
            s.name for s in tool_registry_index.search(query, top_k=TOP_K, categories=[category])
        ]
        scoped_rerank = [
            s.name for s in tool_registry_index.search(query, top_k=TOP_K, categories=[category], rerank=True)
        ]

        ranks = {
            "unscoped": _rank_of(expected, unscoped),
            "scoped_norerank": _rank_of(expected, scoped_norerank),
            "scoped_rerank": _rank_of(expected, scoped_rerank),
        }
        for key, r in ranks.items():
            hits1[key] += int(r == 1)
            hits5[key] += int(r is not None)
            all_ranks[key].append(r)

        category_precision_total += sum(1 for s in unscoped_specs if s.category == category) / len(unscoped_specs)

        rows.append((query, expected, ranks))

    n = len(cases)
    print(f"{'query':<58} {'expected_tool':<28} {'unscoped#':<10} {'scoped#':<8} {'reranked#':<9}")
    print("-" * 118)
    for query, expected, ranks in rows:
        print(
            f"{query[:56]:<58} {expected:<28} "
            f"{str(ranks['unscoped']):<10} {str(ranks['scoped_norerank']):<8} {str(ranks['scoped_rerank']):<9}"
        )

    print(f"\n=== Tool-selection accuracy (n={n}) ===")
    print(
        f"Unscoped, no rerank  (full 510-tool registry):         "
        f"top-1={hits1['unscoped']/n:.0%}  recall@{TOP_K}={hits5['unscoped']/n:.0%}  "
        f"mrr={_mrr(all_ranks['unscoped']):.3f}  category_precision@{TOP_K}={category_precision_total/n:.0%}"
    )
    print(
        f"Scoped,   no rerank  (production default):             "
        f"top-1={hits1['scoped_norerank']/n:.0%}  recall@{TOP_K}={hits5['scoped_norerank']/n:.0%}  "
        f"mrr={_mrr(all_ranks['scoped_norerank']):.3f}"
    )
    print(
        f"Scoped, + rerank     (experimental, rerank=True):      "
        f"top-1={hits1['scoped_rerank']/n:.0%}  recall@{TOP_K}={hits5['scoped_rerank']/n:.0%}  "
        f"mrr={_mrr(all_ranks['scoped_rerank']):.3f}"
    )
    if hits1["scoped_rerank"] < hits1["scoped_norerank"]:
        print("\n-> rerank=True underperforms the no-rerank default on this eval set; left off by default.")


if __name__ == "__main__":
    run()
