#!/usr/bin/env bash
# Stop the native services started by scripts/meet.sh (or by hand).
#   scripts/stop.sh          agents, translator, worker, mlx-audio, laya, voicechat
#   scripts/stop.sh --all    also docker compose stop and unload the LM Studio model
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

stop() { # NAME PATTERN
  if pkill -f "$2"; then echo "stopped $1"; else echo "$1 not running"; fi
}
stop agent 'src/agent.ts'
stop s2s-agent 'src/s2s.ts'
stop voicechat 'services/voicechat/server.py|services/voicechat/run.sh'
stop translator 'src/translator.ts'
stop worker 'src/worker.ts'
stop mlx-audio 'mlx_audio.server|services/mlx-audio/run.sh'
stop laya 'services/laya/server.py'
rm -f .run/*.pid

if [ "${1:-}" = --all ]; then
  set -a; . ./.env; set +a
  docker compose stop
  "$HOME/.lmstudio/bin/lms" unload "$LLM_MODEL" 2>/dev/null && echo "unloaded $LLM_MODEL"
fi
