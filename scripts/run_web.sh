#!/usr/bin/env bash
# Launch `adk web` with every variable in .env exported into the real process
# environment first.
#
# Why this script exists instead of just running `adk web agents` directly:
# ADK's own OpenTelemetry setup (which auto-wires OTEL_EXPORTER_OTLP_ENDPOINT,
# e.g. for LangSmith -- see README section 11) runs during server startup,
# while ADK's per-agent .env loading can happen later (lazily, per agent).
# Exporting .env into the shell before `adk web` even starts sidesteps that
# ordering entirely -- the vars are already real env vars from process launch.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

exec adk web agents "$@"
