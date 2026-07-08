"""Canonical tool specs for the real, callable PayPal-mock API surface (50 tools).

Each spec pairs a category (used for scoping which domain sub-agent may bind
the tool) with a description pulled from the actual gateway-wrapped
function's own docstring, so there is exactly one place the tool's behavior
is documented.

This is the bootstrap source `tool_registry/index.py::sync_registry()` reads
on every sync to refresh the SQLite catalog (name/category/description/is_real)
-- it is NOT read at query time, and the `ToolSpec` objects built here do NOT
carry a live callable (`func` is always None). The actual name -> callable
binding used at retrieval time is resolved independently, from the catalog,
via services/function_registry.py -- see index.py's `_spec_from_catalog_row`.
Binding `func` here too would be dead weight: nothing reads it off these
particular objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ..services import paypal_backend as pb


@dataclass(frozen=True)
class ToolSpec:
    name: str
    category: str
    description: str
    func: Optional[Callable] = None
    is_real: bool = True


_CATEGORY_MAP: dict[str, list[str]] = {
    "invoicing": [
        "create_invoice", "send_invoice", "get_invoice", "list_invoices", "cancel_invoice",
        "remind_invoice", "update_invoice", "delete_draft_invoice", "mark_invoice_paid",
        "generate_invoice_qr_code", "list_overdue_invoices", "duplicate_invoice", "get_invoice_pdf_link",
    ],
    "payments": [
        "send_payment", "request_payment", "get_payment", "list_payments", "cancel_payment",
        "refund_payment", "get_balance", "add_funds", "withdraw_funds", "create_payout_batch",
        "get_payout_batch_status", "list_payment_methods", "verify_recipient",
    ],
    "disputes": [
        "list_disputes", "get_dispute", "list_disputes_by_customer", "get_dispute_status",
        "open_dispute", "respond_to_dispute", "get_dispute_evidence_requirements",
        "upload_dispute_evidence", "accept_dispute_claim", "escalate_dispute_to_claim",
        "close_dispute", "appeal_dispute_decision",
    ],
    "reports": [
        "get_sales_summary", "get_total_sales_volume", "get_transaction_history", "get_top_customers",
        "get_fee_summary", "get_refund_summary", "get_dispute_rate", "export_report_csv",
        "get_monthly_recurring_revenue", "get_currency_breakdown", "get_chargeback_summary",
        "get_average_transaction_value",
    ],
}


def real_tool_specs() -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    for category, fn_names in _CATEGORY_MAP.items():
        for fn_name in fn_names:
            # getattr()-ing here is deliberate even though `func` isn't kept:
            # it means a renamed/deleted function fails immediately with a
            # plain AttributeError on import, the same way a typo would --
            # before sync_registry()'s own _validate_function_bindings check
            # ever gets a chance to run.
            func = getattr(pb, fn_name)
            specs.append(
                ToolSpec(
                    name=fn_name,
                    category=category,
                    description=(func.__doc__ or fn_name).strip(),
                    func=None,
                    is_real=True,
                )
            )
    return specs


REAL_TOOL_SPECS = real_tool_specs()
