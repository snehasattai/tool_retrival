"""Optional OpenTelemetry console tracing for non-`adk web` entrypoints.

`adk web` / `adk api_server` already auto-configure OpenTelemetry on startup
(google.adk.telemetry.setup.maybe_set_otel_providers): they add a SQLite span
exporter that backs the built-in "Traces" tab in the ADK dev UI, and will
also add an OTLP exporter for free if you set the standard
OTEL_EXPORTER_OTLP_ENDPOINT env var before launching -- pointing at a local
Jaeger/Phoenix/Langfuse collector, or Google Cloud Trace. None of that needs
any code in this repo.

`adk run` (plain terminal chat) and standalone scripts (eval harness, tests)
don't go through that startup path, so this module gives them a lightweight
opt-in equivalent: print spans to the console. Call `configure_console_tracing()`
once, early, before running the agent.
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

_configured = False


def configure_console_tracing(service_name: str = "paypal-adk-assistant") -> None:
    global _configured
    if _configured:
        return
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _configured = True
