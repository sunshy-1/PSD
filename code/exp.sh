#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec "${PSD_PYTHON:-python3}" -B scripts/bootstrap.py "$@"
