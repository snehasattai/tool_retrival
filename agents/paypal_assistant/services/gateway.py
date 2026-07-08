"""Tool Execution Gateway.

Every mocked backend call (PayPal APIs, RAG lookups, system-search) is wrapped
by `@gateway_tool` before being handed to ADK as a `FunctionTool`. This is the
single choke point in the diagram between "domain agents" and the concrete
backends, and it is responsible for the things you don't want 500 separate
tool implementations each re-inventing:

  - Schema validation:      required-argument / type checks before the call.
  - Idempotency:             identical calls (same tool + same args) within a
                              short window are deduped, so an LLM retry (or a
                              flaky client re-send) can't double-charge a payment.
  - Retries:                 transient upstream errors get retried with backoff;
                              business/validation errors do not (retrying a
                              "invoice not found" error can't ever succeed).
  - Error normalization:     every call returns a consistent envelope
                              ({"status": "success"|"error"|"confirmation_required", ...})
                              instead of raising, so the calling LLM always gets
                              something it can reason about and relay to the user.
  - Human-in-the-loop gate:  tools marked `sensitive=True` (moves money, accepts
                              a dispute claim, etc.) refuse to execute unless
                              called with confirm=True, which the agent's
                              instructions require it to only do after the user
                              has explicitly approved the specific action.

Using a decorator (rather than, say, ADK's `before_tool_callback`) keeps the
policy declarative and colocated with each tool definition, and keeps the
mocked business logic functions themselves trivial and easy to unit test in
isolation (see tests/test_gateway.py).
"""

from __future__ import annotations

import functools
import hashlib
import json
import random
import time
import uuid
from typing import Any, Callable

from .run_log import run_log

_IDEMPOTENCY_TTL_SECONDS = 60
_idempotency_cache: dict[str, tuple[float, dict[str, Any]]] = {}


class TransientBackendError(Exception):
    """Simulated transient failure (rate limit, network blip) -- safe to retry."""


def _idempotency_key(tool_name: str, kwargs: dict[str, Any]) -> str:
    payload = json.dumps({k: v for k, v in kwargs.items() if k != "confirm"}, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{tool_name}:{digest}"


def _validate_required_args(func: Callable, kwargs: dict[str, Any]) -> str | None:
    """Best-effort schema check: flags missing/empty required (no-default) params."""
    import inspect

    sig = inspect.signature(func)
    for name, param in sig.parameters.items():
        if name in ("self", "tool_context"):
            continue
        if param.default is inspect._empty and name not in kwargs:
            return f"Missing required argument: '{name}'"
        if name in kwargs and kwargs[name] in (None, ""):
            if param.default is inspect._empty:
                return f"Argument '{name}' cannot be empty"
    return None


def gateway_tool(
    *,
    sensitive: bool = False,
    idempotent: bool = True,
    failure_rate: float = 0.0,
    max_attempts: int = 3,
):
    """Decorator implementing the Tool Execution Gateway policy for one tool function.

    Args:
      sensitive: if True, the call is refused unless invoked with confirm=True.
        The wrapped function must declare a `confirm: bool = False` parameter
        so it is visible to the LLM in the generated tool schema.
      idempotent: dedupe identical calls within a short TTL window.
      failure_rate: probability [0, 1) of injecting a simulated transient
        upstream failure -- makes the retry path exercisable in a demo/test
        without needing a real flaky backend.
      max_attempts: retry budget for transient failures.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(**kwargs: Any) -> dict[str, Any]:
            tool_name = func.__name__
            call_id = uuid.uuid4().hex[:8]

            validation_error = _validate_required_args(func, kwargs)
            if validation_error:
                envelope = {"status": "error", "error_code": "VALIDATION_ERROR", "message": validation_error}
                run_log.record(tool_name, kwargs, envelope, call_id)
                return envelope

            if sensitive and not kwargs.get("confirm", False):
                envelope = {
                    "status": "confirmation_required",
                    "action": tool_name,
                    "pending_args": {k: v for k, v in kwargs.items() if k != "confirm"},
                    "message": (
                        f"'{tool_name}' changes money movement or a dispute outcome and requires "
                        "explicit user confirmation. Summarize exactly what will happen (amounts, "
                        "recipient, etc.) and ask the user to confirm. Only call this tool again "
                        "with confirm=true after they explicitly say yes."
                    ),
                }
                run_log.record(tool_name, kwargs, envelope, call_id)
                return envelope

            idem_key = _idempotency_key(tool_name, kwargs) if idempotent else None
            if idem_key and idem_key in _idempotency_cache:
                cached_at, cached_envelope = _idempotency_cache[idem_key]
                if time.time() - cached_at < _IDEMPOTENCY_TTL_SECONDS:
                    replay = dict(cached_envelope)
                    replay["idempotent_replay"] = True
                    run_log.record(tool_name, kwargs, replay, call_id)
                    return replay

            last_error: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    if failure_rate and random.random() < failure_rate:
                        raise TransientBackendError(f"{tool_name}: simulated transient upstream error")
                    call_kwargs = {k: v for k, v in kwargs.items() if k != "confirm"}
                    data = func(**call_kwargs)
                    envelope = {"status": "success", "data": data}
                    if idem_key:
                        _idempotency_cache[idem_key] = (time.time(), envelope)
                    run_log.record(tool_name, kwargs, envelope, call_id)
                    return envelope
                except TransientBackendError as exc:
                    last_error = exc
                    if attempt < max_attempts:
                        time.sleep(0.05 * attempt)
                        continue
                except LookupError as exc:
                    envelope = {"status": "error", "error_code": "NOT_FOUND", "message": str(exc)}
                    run_log.record(tool_name, kwargs, envelope, call_id)
                    return envelope
                except ValueError as exc:
                    envelope = {"status": "error", "error_code": "VALIDATION_ERROR", "message": str(exc)}
                    run_log.record(tool_name, kwargs, envelope, call_id)
                    return envelope

            envelope = {
                "status": "error",
                "error_code": "UPSTREAM_UNAVAILABLE",
                "message": f"'{tool_name}' failed after {max_attempts} attempts: {last_error}",
            }
            run_log.record(tool_name, kwargs, envelope, call_id)
            return envelope

        return wrapper

    return decorator


def on_tool_error_callback(tool, args, tool_context, error) -> dict[str, Any]:
    """Agent-level safety net (registered as `on_tool_error_callback` on every
    LlmAgent). Catches anything that slips past a tool's own gateway decorator
    (e.g. a bug raising an unexpected exception type) and still returns a
    normalized envelope instead of surfacing a raw traceback to the model.
    """
    tool_name = getattr(tool, "name", "unknown_tool")
    envelope = {
        "status": "error",
        "error_code": "INTERNAL_ERROR",
        "message": f"'{tool_name}' raised an unexpected error: {error}",
    }
    run_log.record(tool_name, args or {}, envelope, uuid.uuid4().hex[:8])
    return envelope
