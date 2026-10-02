#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
. scripts/lib.sh

for agent in $AGENTS; do stop_process "$agent agent" "$(agent_entry "$agent")"; done
stop_process translator 'src/translator.ts'
stop_process worker 'src/worker.ts'
stop_process laya 'services/laya/server.py'
stop_process mlx-audio 'mlx_audio.server|services/mlx-audio/run.sh'
stop_process voicechat 'services/voicechat/server.py'
stop_process omni 'services/omni/server.py'
rm -f "$RUN"/*.pid

if [ "${1:-}" = --all ]; then
  load_env
  docker compose stop
  "$LMS" unload "$LLM_MODEL" 2>/dev/null && echo "unloaded $LLM_MODEL"
fi
exit 0
