#!/usr/bin/env python3
"""Sync the persistent Tool Registry vector index to the current tool specs.

Run once before first use:
    python scripts/seed_tool_registry.py

Safe to re-run any time specs.py/synthetic_specs.py change -- it only embeds
tools that are new or whose description changed since the last sync (see
tool_registry/index.py::sync_registry); unchanged tools cost zero embedding
calls. Pass --rebuild to wipe and re-embed everything from scratch (e.g.
after switching embedding models/dimensionality).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from agents.paypal_assistant.tool_registry import index as tool_registry_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="force re-embedding of all tool specs")
    args = parser.parse_args()

    total = tool_registry_index.registry_size()
    real = tool_registry_index.real_tool_count()
    print(f"Syncing tool registry: {total} total tool specs ({real} real PayPal tools, {total - real} synthetic decoys)")

    t0 = time.time()
    collection = tool_registry_index.get_collection(force_rebuild=args.rebuild)
    stats = tool_registry_index.last_sync_stats
    print(
        f"Collection '{collection.name}' now has {collection.count()} vectors "
        f"({time.time() - t0:.1f}s) -- embedded {stats['embedded']} (new/changed), "
        f"deleted {stats['deleted']}, skipped {stats['unchanged']} (unchanged)"
    )


if __name__ == "__main__":
    main()
