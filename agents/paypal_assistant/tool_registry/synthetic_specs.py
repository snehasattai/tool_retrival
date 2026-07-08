"""Synthetic decoy tool specs.

The task explicitly asks the design to hold up at 500+ tools across various
services, not just PayPal's 50. Real integrations for 450 other SaaS APIs are
obviously out of scope for a demo, so this module generates plausible,
non-executable tool specs (name + category + description only, no backing
function) for ~20 other common SaaS products.

These decoys serve one purpose: populate the shared Chroma tool registry so
the semantic retrieval spine (tool_registry/semantic_toolset.py) has to prove
itself at real scale -- including a handful of *deliberately confusable*
entries (e.g. "stripe_create_invoice", "quickbooks_create_invoice") that sit
right next to PayPal's real create_invoice/send_invoice tools in embedding
space. See eval/tool_selection_eval.json + eval/run_eval.py for the precision
numbers this produces.

Domain sub-agents never bind these -- they are filtered out by the hard
category boundary each sub-agent's SemanticToolset enforces (see
sub_agents/*.py), so a decoy can never actually be invoked. They exist purely
to stress-test ranking quality within the registry.
"""

from __future__ import annotations

import os

from .specs import ToolSpec

_SERVICE_NOUNS: dict[str, list[str]] = {
    "slack": ["message", "channel", "user_status", "reminder", "file_upload", "channel_topic",
              "user_group", "workflow", "emoji_reaction", "pinned_message"],
    "stripe": ["charge", "customer", "subscription", "invoice", "refund", "payment_intent",
               "coupon", "payout", "dispute", "webhook_endpoint"],
    "salesforce": ["lead", "opportunity", "contact", "account", "case", "task",
                   "campaign", "report", "quote", "contract"],
    "jira": ["issue", "sprint", "epic", "board", "comment", "label",
             "worklog", "project", "filter", "transition"],
    "zendesk": ["ticket", "macro", "trigger", "sla_policy", "satisfaction_rating", "organization",
                "view", "article", "brand", "group"],
    "hubspot": ["contact", "deal", "company", "email_campaign", "workflow", "form",
                "list", "meeting", "ticket", "property"],
    "github": ["repository", "pull_request", "issue", "branch", "release", "webhook",
               "workflow_run", "label", "milestone", "commit_status"],
    "shopify": ["product", "order", "customer", "discount", "collection", "inventory_item",
                "fulfillment", "cart", "gift_card", "shipping_rate"],
    "twilio": ["sms", "call", "phone_number", "verification", "conference", "recording",
               "voicemail", "whatsapp_message", "fax", "sip_domain"],
    "mailchimp": ["campaign", "audience", "subscriber", "template", "automation", "segment",
                  "landing_page", "report", "tag", "ab_test"],
    "asana": ["task", "project", "team", "portfolio", "goal", "section",
              "tag", "custom_field", "attachment", "workspace"],
    "notion": ["page", "database", "block", "comment", "user", "template",
               "property", "view", "workspace_member", "integration"],
    "dropbox": ["file", "folder", "shared_link", "team_member", "paper_doc", "backup",
                "file_request", "space_usage", "event", "webhook"],
    "calendly": ["event_type", "scheduled_event", "invitee", "availability", "webhook_subscription",
                 "organization_membership", "routing_form", "user", "team", "poll"],
    "docusign": ["envelope", "template", "signer", "document", "recipient", "brand",
                 "webhook", "account", "power_form", "custom_field"],
    "quickbooks": ["invoice", "bill", "expense", "customer", "vendor", "journal_entry",
                   "estimate", "payment", "tax_rate", "report"],
    "xero": ["invoice", "bill", "contact", "bank_transaction", "purchase_order", "credit_note",
             "budget", "tracking_category", "payment", "report"],
    "freshdesk": ["ticket", "contact", "company", "canned_response", "sla_policy", "agent",
                  "group", "solution_article", "survey", "time_entry"],
    "intercom": ["conversation", "contact", "company", "article", "tag", "segment",
                 "campaign", "note", "team", "admin"],
    "trello": ["card", "board", "list", "label", "checklist", "member",
               "attachment", "power_up", "webhook", "organization"],
}

_VERBS = ["create", "get", "list", "update", "delete", "search", "cancel"]


def generate_synthetic_specs(target_count: int | None = None) -> list[ToolSpec]:
    target_count = target_count or int(os.getenv("DECOY_TOOL_COUNT", "460"))
    specs: list[ToolSpec] = []
    for service, nouns in _SERVICE_NOUNS.items():
        for noun in nouns:
            for verb in _VERBS:
                if len(specs) >= target_count:
                    return specs
                name = f"{service}_{verb}_{noun}"
                description = f"{verb.capitalize()} a {noun.replace('_', ' ')} in {service.capitalize()}."
                specs.append(
                    ToolSpec(
                        name=name,
                        category=service,
                        description=description,
                        func=None,
                        is_real=False,
                    )
                )
    return specs


SYNTHETIC_TOOL_SPECS = generate_synthetic_specs()
