#!/bin/sh
# Start the Jev decision service (CPU-only, loopback only).
#   ./run.sh                       -> http://127.0.0.1:61021
#   JEV_PORT=61022 ./run.sh
#   JEV_GGUF=/path/to/model.gguf ./run.sh
#   JEV_WORKERS=4 ./run.sh
set -eu
cd "$(dirname "$0")"
exec python3 service.py "$@"
