"""In-memory run log: every tool invocation the gateway processes gets recorded here.

This backs two things in the architecture diagram:
  - "System DB / Logs" -> queried by the System Search Tool to answer
    "what's the status of my last request?"
  - A cheap source of ground truth for the Observability layer without needing
    an external tracing backend wired up.

In production this would be a real table (Postgres) or a Redis stream, keyed
by session/user, with TTL/retention policy. Kept in-memory here since the demo
runs as a single process per `adk web` / `adk run` session.
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolCallRecord:
    call_id: str
    tool_name: str
    args: dict[str, Any]
    result: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class RunLog:
    def __init__(self, max_records: int = 2000):
        self._lock = threading.Lock()
        self._records: list[ToolCallRecord] = []
        self._max_records = max_records
        self._counter = itertools.count(1)

    def record(self, tool_name: str, args: dict[str, Any], result: dict[str, Any], call_id: str) -> ToolCallRecord:
        rec = ToolCallRecord(call_id=call_id, tool_name=tool_name, args=dict(args), result=result)
        with self._lock:
            self._records.append(rec)
            if len(self._records) > self._max_records:
                self._records.pop(0)
        return rec

    def last(self, n: int = 1, tool_name: Optional[str] = None) -> list[ToolCallRecord]:
        with self._lock:
            records = self._records if tool_name is None else [r for r in self._records if r.tool_name == tool_name]
            return list(reversed(records[-n:]))

    def get(self, call_id: str) -> Optional[ToolCallRecord]:
        with self._lock:
            for rec in reversed(self._records):
                if rec.call_id == call_id:
                    return rec
        return None

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = len(self._records)
            errors = sum(1 for r in self._records if r.result.get("status") == "error")
            pending = sum(1 for r in self._records if r.result.get("status") == "confirmation_required")
        return {"total_calls": total, "errors": errors, "pending_confirmations": pending}


# Process-wide singleton -- every gateway-wrapped tool call writes here.
run_log = RunLog()
