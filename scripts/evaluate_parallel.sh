#!/usr/bin/env bash
set -euo pipefail
exec python3 -m mteb_eval.evaluate_parallel "$@"
