#!/usr/bin/env bash
# Python launcher for rebuttal runs: the venv's cuDNN sublibraries only load
# when their directory is on LD_LIBRARY_PATH (conv otherwise fails).
here="$(cd "$(dirname "$0")/.." && pwd)"
export LD_LIBRARY_PATH="$here/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$here/src"
exec "$here/.venv/bin/python" -W ignore "$@"
