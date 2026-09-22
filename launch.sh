#!/bin/sh
set -eu
if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
    python3 /app/download.py
fi
exec python3 -u /app/server.py
