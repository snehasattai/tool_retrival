"""Mocked PayPal backend.

A self-contained in-memory "PayPal" -- no real network calls, no real
credentials -- standing in for the 50+ call Postman collection described in
the task. Every public function here is the *business logic* for one PayPal
API call (create_invoice, send_payment, get_dispute, get_sales_summary, ...)
and is registered as a real, callable ADK tool (see sub_agents/*.py).

Each function is wrapped with `@gateway_tool` (see services/gateway.py) which
supplies retries, idempotency, error normalization, and the sensitive-action
confirmation gate uniformly -- the functions below only contain the actual
domain logic, nothing about resilience or safety.

State lives in one process-wide `_DB` object and is seeded with a few months
of synthetic activity on import, so report queries ("total sales last month")
and lookups ("dispute from user_123") return realistic, non-empty answers out
of the box.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from .gateway import gateway_tool

_CURRENCY = "USD"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class _DB:
    invoices: dict[str, dict[str, Any]] = field(default_factory=dict)
    payments: dict[str, dict[str, Any]] = field(default_factory=dict)
    disputes: dict[str, dict[str, Any]] = field(default_factory=dict)
    payout_batches: dict[str, dict[str, Any]] = field(default_factory=dict)
    balance: dict[str, float] = field(default_factory=lambda: {"USD": 12450.32, "EUR": 860.10})
    transactions: list[dict[str, Any]] = field(default_factory=list)


_db = _DB()


def _seed() -> None:
    """Populate a few months of synthetic activity so reports/lookups are meaningful."""
    random.seed(7)
    customers = [
        ("user_101", "alice@example.com"),
        ("user_123", "bob@example.com"),
        ("user_207", "carol@example.com"),
        ("user_309", "dave@example.com"),
        ("user_411", "erin@example.com"),
    ]
    today = _now()

    # Transactions ledger for reports, spread over the last 90 days.
    for i in range(140):
        days_ago = random.randint(0, 90)
        ts = today - timedelta(days=days_ago, hours=random.randint(0, 23))
        _, email = random.choice(customers)
        amount = round(random.uniform(15, 900), 2)
        kind = random.choices(["sale", "refund", "fee"], weights=[0.8, 0.1, 0.1])[0]
        _db.transactions.append(
            {
                "id": _new_id("TXN"),
                "type": kind,
                "counterparty": email,
                "amount": amount if kind != "refund" else -amount,
                "currency": _CURRENCY,
                "timestamp": _iso(ts),
            }
        )

    # A handful of named invoices/payments/disputes so demo queries resolve.
    inv1 = _new_id("INV")
    _db.invoices[inv1] = {
        "id": inv1,
        "recipient": "bob@example.com",
        "amount": 250.00,
        "currency": _CURRENCY,
        "memo": "Consulting - March",
        "status": "overdue",
        "due_date": _iso(today - timedelta(days=5)),
        "created_at": _iso(today - timedelta(days=20)),
    }

    pay1 = _new_id("PAY")
    _db.payments[pay1] = {
        "id": pay1,
        "sender": "me@business.com",
        "recipient": "bob@example.com",
        "amount": 90.00,
        "currency": _CURRENCY,
        "status": "completed",
        "note": "Refund for order #4471",
        "created_at": _iso(today - timedelta(days=10)),
    }

    dsp1 = _new_id("DSP")
    _db.disputes[dsp1] = {
        "id": dsp1,
        "transaction_id": pay1,
        "user_id": "user_123",
        "customer_email": "bob@example.com",
        "reason": "item_not_received",
        "amount": 90.00,
        "currency": _CURRENCY,
        "status": "open",
        "created_at": _iso(today - timedelta(days=3)),
        "resolution": None,
    }


_seed()


# --------------------------------------------------------------------------
# Invoicing (13 tools)
# --------------------------------------------------------------------------

@gateway_tool()
def create_invoice(recipient_email: str, amount: float, currency: str = "USD", memo: str = "") -> dict:
    """Create a draft invoice billed to recipient_email for the given amount."""
    if amount <= 0:
        raise ValueError("amount must be positive")
    inv_id = _new_id("INV")
    invoice = {
        "id": inv_id,
        "recipient": recipient_email,
        "amount": round(amount, 2),
        "currency": currency,
        "memo": memo,
        "status": "draft",
        "due_date": _iso(_now() + timedelta(days=30)),
        "created_at": _iso(_now()),
    }
    _db.invoices[inv_id] = invoice
    return invoice


@gateway_tool()
def send_invoice(invoice_id: str) -> dict:
    """Send a draft invoice to its recipient, marking it as sent."""
    inv = _db.invoices[invoice_id]
    inv["status"] = "sent"
    return inv


@gateway_tool()
def get_invoice(invoice_id: str) -> dict:
    """Fetch a single invoice by id."""
    return _db.invoices[invoice_id]


@gateway_tool()
def list_invoices(status: Optional[str] = None, limit: int = 20) -> dict:
    """List invoices, optionally filtered by status (draft/sent/paid/overdue/cancelled)."""
    items = list(_db.invoices.values())
    if status:
        items = [i for i in items if i["status"] == status]
    return {"invoices": items[:limit], "count": len(items)}


@gateway_tool()
def cancel_invoice(invoice_id: str) -> dict:
    """Cancel a draft or sent invoice."""
    inv = _db.invoices[invoice_id]
    inv["status"] = "cancelled"
    return inv


@gateway_tool()
def remind_invoice(invoice_id: str) -> dict:
    """Send a payment reminder for an overdue or outstanding invoice."""
    inv = _db.invoices[invoice_id]
    return {"invoice_id": invoice_id, "reminder_sent_to": inv["recipient"], "status": inv["status"]}


@gateway_tool()
def update_invoice(invoice_id: str, amount: Optional[float] = None, memo: Optional[str] = None) -> dict:
    """Update the amount and/or memo of a draft invoice."""
    inv = _db.invoices[invoice_id]
    if inv["status"] != "draft":
        raise ValueError("only draft invoices can be edited")
    if amount is not None:
        inv["amount"] = round(amount, 2)
    if memo is not None:
        inv["memo"] = memo
    return inv


@gateway_tool()
def delete_draft_invoice(invoice_id: str) -> dict:
    """Permanently delete a draft invoice (cannot be undone)."""
    inv = _db.invoices[invoice_id]
    if inv["status"] != "draft":
        raise ValueError("only draft invoices can be deleted")
    del _db.invoices[invoice_id]
    return {"deleted": invoice_id}


@gateway_tool()
def mark_invoice_paid(invoice_id: str) -> dict:
    """Manually mark an invoice as paid (e.g. paid by cash/check outside PayPal)."""
    inv = _db.invoices[invoice_id]
    inv["status"] = "paid"
    return inv


@gateway_tool()
def generate_invoice_qr_code(invoice_id: str) -> dict:
    """Generate a scannable QR code link for an invoice's payment page."""
    _db.invoices[invoice_id]
    return {"invoice_id": invoice_id, "qr_code_url": f"https://paypal.mock/qr/{invoice_id}.png"}


@gateway_tool()
def list_overdue_invoices() -> dict:
    """List all invoices currently overdue."""
    items = [i for i in _db.invoices.values() if i["status"] == "overdue"]
    return {"invoices": items, "count": len(items)}


@gateway_tool()
def duplicate_invoice(invoice_id: str) -> dict:
    """Create a new draft invoice copied from an existing one."""
    src = _db.invoices[invoice_id]
    new_id = _new_id("INV")
    copy = {**src, "id": new_id, "status": "draft", "created_at": _iso(_now())}
    _db.invoices[new_id] = copy
    return copy


@gateway_tool()
def get_invoice_pdf_link(invoice_id: str) -> dict:
    """Get a downloadable PDF link for an invoice."""
    _db.invoices[invoice_id]
    return {"invoice_id": invoice_id, "pdf_url": f"https://paypal.mock/pdf/{invoice_id}.pdf"}


# --------------------------------------------------------------------------
# Payments (13 tools) -- money-movement actions are `sensitive=True`
# --------------------------------------------------------------------------

@gateway_tool(sensitive=True, failure_rate=0.05)
def send_payment(recipient_email: str, amount: float, currency: str = "USD", note: str = "", confirm: bool = False) -> dict:
    """Send a payment to recipient_email. Moves real money -- requires user confirmation.

    Set confirm=true only after the user has explicitly approved the exact
    amount and recipient.
    """
    if amount <= 0:
        raise ValueError("amount must be positive")
    if _db.balance.get(currency, 0) < amount:
        raise ValueError(f"insufficient {currency} balance")
    pay_id = _new_id("PAY")
    _db.balance[currency] = round(_db.balance.get(currency, 0) - amount, 2)
    payment = {
        "id": pay_id,
        "sender": "me@business.com",
        "recipient": recipient_email,
        "amount": round(amount, 2),
        "currency": currency,
        "note": note,
        "status": "completed",
        "created_at": _iso(_now()),
    }
    _db.payments[pay_id] = payment
    return payment


@gateway_tool()
def request_payment(payer_email: str, amount: float, currency: str = "USD", note: str = "") -> dict:
    """Request a payment from payer_email (does not move money by itself)."""
    if amount <= 0:
        raise ValueError("amount must be positive")
    pay_id = _new_id("REQ")
    request = {
        "id": pay_id,
        "payer": payer_email,
        "amount": round(amount, 2),
        "currency": currency,
        "note": note,
        "status": "requested",
        "created_at": _iso(_now()),
    }
    _db.payments[pay_id] = request
    return request


@gateway_tool()
def get_payment(payment_id: str) -> dict:
    """Fetch a single payment by id."""
    return _db.payments[payment_id]


@gateway_tool()
def list_payments(status: Optional[str] = None, limit: int = 20) -> dict:
    """List payments, optionally filtered by status."""
    items = list(_db.payments.values())
    if status:
        items = [p for p in items if p["status"] == status]
    return {"payments": items[:limit], "count": len(items)}


@gateway_tool()
def cancel_payment(payment_id: str) -> dict:
    """Cancel a pending payment request."""
    pay = _db.payments[payment_id]
    if pay["status"] not in ("requested", "pending"):
        raise ValueError("only pending/requested payments can be cancelled")
    pay["status"] = "cancelled"
    return pay


@gateway_tool(sensitive=True)
def refund_payment(payment_id: str, amount: Optional[float] = None, confirm: bool = False) -> dict:
    """Refund a completed payment, fully or partially. Requires user confirmation."""
    pay = _db.payments[payment_id]
    refund_amount = amount if amount is not None else pay["amount"]
    if refund_amount > pay["amount"]:
        raise ValueError("refund amount cannot exceed original payment amount")
    pay["status"] = "refunded"
    pay["refunded_amount"] = round(refund_amount, 2)
    return pay


@gateway_tool()
def get_balance(currency: str = "USD") -> dict:
    """Get the current available balance for a currency."""
    return {"currency": currency, "balance": _db.balance.get(currency, 0.0)}


@gateway_tool(sensitive=True)
def add_funds(amount: float, currency: str = "USD", source: str = "bank", confirm: bool = False) -> dict:
    """Add funds from a linked bank/card into the PayPal balance. Requires confirmation."""
    if amount <= 0:
        raise ValueError("amount must be positive")
    _db.balance[currency] = round(_db.balance.get(currency, 0) + amount, 2)
    return {"currency": currency, "added": amount, "new_balance": _db.balance[currency], "source": source}


@gateway_tool(sensitive=True)
def withdraw_funds(amount: float, currency: str = "USD", destination: str = "bank", confirm: bool = False) -> dict:
    """Withdraw funds from the PayPal balance to a linked bank account. Requires confirmation."""
    if _db.balance.get(currency, 0) < amount:
        raise ValueError(f"insufficient {currency} balance")
    _db.balance[currency] = round(_db.balance.get(currency, 0) - amount, 2)
    return {"currency": currency, "withdrawn": amount, "new_balance": _db.balance[currency], "destination": destination}


@gateway_tool(sensitive=True)
def create_payout_batch(recipient_emails: list[str], amount_each: float, currency: str = "USD", confirm: bool = False) -> dict:
    """Send the same payout amount to a list of recipients in one batch. Requires confirmation."""
    if amount_each <= 0:
        raise ValueError("amount_each must be positive")
    batch_id = _new_id("BATCH")
    batch = {
        "id": batch_id,
        "recipients": recipient_emails,
        "amount_each": amount_each,
        "currency": currency,
        "status": "processing",
        "created_at": _iso(_now()),
    }
    _db.payout_batches[batch_id] = batch
    return batch


@gateway_tool()
def get_payout_batch_status(batch_id: str) -> dict:
    """Check the processing status of a payout batch."""
    return _db.payout_batches[batch_id]


@gateway_tool()
def list_payment_methods() -> dict:
    """List linked payment methods (bank accounts, cards)."""
    return {"methods": [
        {"type": "bank_account", "last4": "4321", "primary": True},
        {"type": "card", "brand": "visa", "last4": "0007", "primary": False},
    ]}


@gateway_tool()
def verify_recipient(email: str) -> dict:
    """Check whether an email address is a valid, verified PayPal recipient."""
    return {"email": email, "verified": True, "account_type": "personal"}


# --------------------------------------------------------------------------
# Disputes (12 tools) -- outcome-altering actions are `sensitive=True`
# --------------------------------------------------------------------------

@gateway_tool()
def list_disputes(status: Optional[str] = None, limit: int = 20) -> dict:
    """List disputes, optionally filtered by status (open/under_review/resolved)."""
    items = list(_db.disputes.values())
    if status:
        items = [d for d in items if d["status"] == status]
    return {"disputes": items[:limit], "count": len(items)}


@gateway_tool()
def get_dispute(dispute_id: str) -> dict:
    """Fetch full details for a single dispute."""
    return _db.disputes[dispute_id]


@gateway_tool()
def list_disputes_by_customer(user_id: str) -> dict:
    """Find all disputes filed by a given customer/user id."""
    items = [d for d in _db.disputes.values() if d["user_id"] == user_id]
    return {"disputes": items, "count": len(items)}


@gateway_tool()
def get_dispute_status(dispute_id: str) -> dict:
    """Quick status-only lookup for a dispute (lighter than get_dispute)."""
    d = _db.disputes[dispute_id]
    return {"dispute_id": dispute_id, "status": d["status"]}


@gateway_tool()
def open_dispute(transaction_id: str, reason: str, amount: float) -> dict:
    """File a new dispute against a transaction."""
    dsp_id = _new_id("DSP")
    dispute = {
        "id": dsp_id,
        "transaction_id": transaction_id,
        "user_id": None,
        "reason": reason,
        "amount": amount,
        "currency": _CURRENCY,
        "status": "open",
        "created_at": _iso(_now()),
        "resolution": None,
    }
    _db.disputes[dsp_id] = dispute
    return dispute


@gateway_tool()
def respond_to_dispute(dispute_id: str, message: str) -> dict:
    """Send a message/response on an open dispute."""
    d = _db.disputes[dispute_id]
    d["status"] = "under_review"
    return {"dispute_id": dispute_id, "status": d["status"], "message_sent": message}


@gateway_tool()
def get_dispute_evidence_requirements(dispute_id: str) -> dict:
    """Get the list of evidence PayPal requires to respond to a dispute."""
    _db.disputes[dispute_id]
    return {"dispute_id": dispute_id, "required_evidence": ["proof_of_shipment", "proof_of_delivery", "invoice_copy"]}


@gateway_tool()
def upload_dispute_evidence(dispute_id: str, evidence_url: str) -> dict:
    """Attach an evidence document to a dispute."""
    d = _db.disputes[dispute_id]
    d.setdefault("evidence", []).append(evidence_url)
    return {"dispute_id": dispute_id, "evidence_count": len(d["evidence"])}


@gateway_tool(sensitive=True)
def accept_dispute_claim(dispute_id: str, confirm: bool = False) -> dict:
    """Accept the claimant's side of a dispute, refunding them. Requires confirmation."""
    d = _db.disputes[dispute_id]
    d["status"] = "resolved"
    d["resolution"] = "accepted_refunded"
    return d


@gateway_tool()
def escalate_dispute_to_claim(dispute_id: str) -> dict:
    """Escalate an unresolved dispute into a formal PayPal claim."""
    d = _db.disputes[dispute_id]
    d["status"] = "escalated_claim"
    return d


@gateway_tool()
def close_dispute(dispute_id: str, resolution: str) -> dict:
    """Close a dispute with a given resolution note (e.g. resolved directly with buyer)."""
    d = _db.disputes[dispute_id]
    d["status"] = "resolved"
    d["resolution"] = resolution
    return d


@gateway_tool()
def appeal_dispute_decision(dispute_id: str, message: str) -> dict:
    """File an appeal against a resolved dispute's decision."""
    d = _db.disputes[dispute_id]
    if d["status"] != "resolved":
        raise ValueError("only resolved disputes can be appealed")
    d["status"] = "appealed"
    return {"dispute_id": dispute_id, "status": d["status"], "appeal_message": message}


# --------------------------------------------------------------------------
# Reports (12 tools)
# --------------------------------------------------------------------------

def _period_cutoff(period: str) -> datetime:
    days = {"last_week": 7, "last_month": 30, "last_quarter": 90, "last_year": 365}.get(period, 30)
    return _now() - timedelta(days=days)


def _txns_in_period(period: str) -> list[dict]:
    cutoff = _period_cutoff(period)
    return [t for t in _db.transactions if datetime.fromisoformat(t["timestamp"]) >= cutoff]


@gateway_tool()
def get_sales_summary(period: str = "last_month") -> dict:
    """Summarize sales (count and gross volume) for a period (last_week/last_month/last_quarter/last_year)."""
    txns = [t for t in _txns_in_period(period) if t["type"] == "sale"]
    total = round(sum(t["amount"] for t in txns), 2)
    return {"period": period, "sale_count": len(txns), "gross_volume": total, "currency": _CURRENCY}


@gateway_tool()
def get_total_sales_volume(period: str = "last_month") -> dict:
    """Get just the total sales volume figure for a period."""
    txns = [t for t in _txns_in_period(period) if t["type"] == "sale"]
    return {"period": period, "total_sales_volume": round(sum(t["amount"] for t in txns), 2), "currency": _CURRENCY}


@gateway_tool()
def get_transaction_history(period: str = "last_month", limit: int = 50) -> dict:
    """List raw transactions for a period, most recent first."""
    txns = sorted(_txns_in_period(period), key=lambda t: t["timestamp"], reverse=True)
    return {"period": period, "transactions": txns[:limit], "count": len(txns)}


@gateway_tool()
def get_top_customers(period: str = "last_month", limit: int = 5) -> dict:
    """Rank customers by gross sale volume for a period."""
    txns = [t for t in _txns_in_period(period) if t["type"] == "sale"]
    totals: dict[str, float] = {}
    for t in txns:
        totals[t["counterparty"]] = totals.get(t["counterparty"], 0) + t["amount"]
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    return {"period": period, "top_customers": [{"customer": c, "total": round(v, 2)} for c, v in ranked]}


@gateway_tool()
def get_fee_summary(period: str = "last_month") -> dict:
    """Summarize PayPal fees charged for a period."""
    txns = [t for t in _txns_in_period(period) if t["type"] == "fee"]
    return {"period": period, "total_fees": round(sum(abs(t["amount"]) for t in txns), 2), "currency": _CURRENCY}


@gateway_tool()
def get_refund_summary(period: str = "last_month") -> dict:
    """Summarize refunds issued for a period."""
    txns = [t for t in _txns_in_period(period) if t["type"] == "refund"]
    return {"period": period, "refund_count": len(txns), "total_refunded": round(sum(abs(t["amount"]) for t in txns), 2)}


@gateway_tool()
def get_dispute_rate(period: str = "last_month") -> dict:
    """Compute disputes-per-100-transactions for a period."""
    txns = _txns_in_period(period)
    cutoff = _period_cutoff(period)
    disputes = [d for d in _db.disputes.values() if datetime.fromisoformat(d["created_at"]) >= cutoff]
    rate = round((len(disputes) / len(txns)) * 100, 3) if txns else 0.0
    return {"period": period, "dispute_count": len(disputes), "transaction_count": len(txns), "dispute_rate_pct": rate}


@gateway_tool()
def export_report_csv(period: str = "last_month", report_type: str = "sales") -> dict:
    """Generate a downloadable CSV export link for a report."""
    return {"period": period, "report_type": report_type, "download_url": f"https://paypal.mock/reports/{report_type}-{period}.csv"}


@gateway_tool()
def get_monthly_recurring_revenue() -> dict:
    """Estimate monthly recurring revenue from subscription-like recent sales."""
    txns = [t for t in _txns_in_period("last_month") if t["type"] == "sale"]
    return {"estimated_mrr": round(sum(t["amount"] for t in txns) * 0.35, 2), "currency": _CURRENCY}


@gateway_tool()
def get_currency_breakdown(period: str = "last_month") -> dict:
    """Break down sales volume by currency for a period."""
    txns = [t for t in _txns_in_period(period) if t["type"] == "sale"]
    breakdown: dict[str, float] = {}
    for t in txns:
        breakdown[t["currency"]] = breakdown.get(t["currency"], 0) + t["amount"]
    return {"period": period, "breakdown": {k: round(v, 2) for k, v in breakdown.items()}}


@gateway_tool()
def get_chargeback_summary(period: str = "last_month") -> dict:
    """Summarize chargebacks (bank-initiated disputes) for a period."""
    cutoff = _period_cutoff(period)
    chargebacks = [
        d for d in _db.disputes.values()
        if d["reason"] == "chargeback" and datetime.fromisoformat(d["created_at"]) >= cutoff
    ]
    return {"period": period, "chargeback_count": len(chargebacks)}


@gateway_tool()
def get_average_transaction_value(period: str = "last_month") -> dict:
    """Compute average transaction value for a period."""
    txns = [t for t in _txns_in_period(period) if t["type"] == "sale"]
    avg = round(sum(t["amount"] for t in txns) / len(txns), 2) if txns else 0.0
    return {"period": period, "average_transaction_value": avg, "currency": _CURRENCY}
