#!/usr/bin/env bash
# Canonical entry point; the historical runner remains for server compatibility.
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_ROOT}/run_unsb_dosc_breast.sh" "$@"
