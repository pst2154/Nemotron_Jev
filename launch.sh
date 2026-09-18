#!/bin/sh
set -eu
if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
    python /app/download.py
fi
exec python -u /app/server.py
