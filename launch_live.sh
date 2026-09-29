#!/bin/bash
# Launch the interview copilot in live mode (loopback capture).
# Usage: ./launch_live.sh [device_index]
#   device_index: optional, default auto-detect (hint "Loopback"/"Стерео микшер")

set -e
cd "$(dirname "$0")"
VENV=".venv/Scripts/python.exe"
DEVICE="${1:-}"

echo "=== Interview Copilot — Live Mode ==="
echo "Device: ${DEVICE:-auto (hint: Loopback/Stereo Mix)}"
echo "LLM server: http://localhost:8080 (must be running)"
echo "Press Ctrl+C to stop."
echo

if [ -n "$DEVICE" ]; then
    $VENV -c "
import sys
sys.argv = ['main', 'live']
from app.config import PipelineConfig
cfg = PipelineConfig()
cfg.loopback_device = int('$DEVICE')
from app.main import Pipeline, _run_live_with_overlay
pipe = Pipeline(cfg)
_run_live_with_overlay(pipe)
"
else
    $VENV -m app.main live
fi
