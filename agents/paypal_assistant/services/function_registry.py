"""The one piece of the tool system that has to stay in Python: a map from
tool name to the actual callable that executes it.

Auto-built by introspecting paypal_backend.py rather than hand-maintained --
every function wrapped with `@gateway_tool(...)` gets a `__wrapped__`
attribute (set by `functools.wraps`), which is how gateway-wrapped business
functions are told apart from helpers/imports in that module. Add a new real
tool by writing the function; it appears here automatically, nothing else to
update.

This is deliberately the *only* thing that lives in code once
tool_registry/catalog_db.py holds the descriptive metadata (name, category,
description) as actual data. A SQLite row can say a tool named "send_payment"
exists in the "payments" category -- it can't hold the function itself.
Keeping this registry intentionally dumb (name -> callable, nothing else)
means the code side of the system never needs to know about categories,
descriptions, or retrieval at all.
"""

from __future__ import annotations

import inspect
from typing import Callable

from . import paypal_backend as pb

FUNCTION_REGISTRY: dict[str, Callable] = {
    name: func
    for name, func in inspect.getmembers(pb, inspect.isfunction)
    if hasattr(func, "__wrapped__")
}
