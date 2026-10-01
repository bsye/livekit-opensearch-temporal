#!/usr/bin/env bash
# Start the whole local stack (skipping anything already running) and open a Meet
# session in a new room, which the voice agent joins automatically.
#
#   scripts/meet.sh [room] [identity]      defaults: meet-<HHMMSS>, $USER
#   NO_OPEN=1 scripts/meet.sh              print the link instead of opening the browser
#
# Native processes run detached; logs and pids are in .run/. Stop with scripts/stop.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
ROOM="${1:-meet-$(date +%H%M%S)}"
IDENTITY="${2:-$USER}"
RUN="$ROOT/.run"
LMS="$HOME/.lmstudio/bin/lms"
mkdir -p "$RUN"
set -a; . ./.env; set +a

step() { printf '\n\033[1m▸ %s\033[0m\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# wait_for DESCRIPTION TIMEOUT_SECONDS COMMAND...
wait_for() {
  local desc=$1 timeout=$2; shift 2
  for ((i = 0; i < timeout * 2; i++)); do
    "$@" >/dev/null 2>&1 && return 0
    sleep 0.5
  done
  die "$desc not ready after ${timeout}s (logs in .run/)"
}

# start NAME PATTERN COMMAND...: run COMMAND detached unless a process matching PATTERN is up
start() {
  local name=$1 pattern=$2; shift 2
  if pgrep -f "$pattern" >/dev/null; then
    echo "  $name already running"
    return
  fi
  nohup "$@" >"$RUN/$name.log" 2>&1 &
  echo $! >"$RUN/$name.pid"
  echo "  $name started (.run/$name.log)"
}

step "Checking prerequisites"
docker info >/dev/null 2>&1 || die "Docker (OrbStack) is not running"
command -v lk >/dev/null || die "LiveKit CLI missing: brew install livekit-cli"
[ -x "$LMS" ] || die "LM Studio CLI not found at $LMS"
[ -d node_modules ] || npm install
echo "  ok"

step "Infrastructure (docker compose: LiveKit, Caddy, Redis, egress, ingress, Temporal)"
docker compose up -d --quiet-pull 2>&1 | grep -vE 'Running|Started|Healthy|Waiting' || true
wait_for "LiveKit" 60 curl -sf https://livekit.localhost
namespace_ready() {
  [ "$(docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' "$(docker compose ps -aq temporal-namespace)")" = "exited 0" ]
}
wait_for "Temporal namespace" 120 namespace_ready
echo "  LiveKit and Temporal ready"

step "LLM (LM Studio, $LLM_MODEL)"
"$LMS" server start >/dev/null 2>&1 || true
if ! "$LMS" ps 2>/dev/null | grep -q "$LLM_MODEL"; then
  echo "  loading model..."
  "$LMS" load "$LLM_MODEL" --identifier "$LLM_MODEL" -y >/dev/null
fi
wait_for "LM Studio" 60 sh -c "curl -sf '$LLM_BASE_URL/models' | grep -q '$LLM_MODEL'"
echo "  ready"

step "Native services"
start mlx-audio 'mlx_audio.server' services/mlx-audio/run.sh
start laya 'services/laya/server.py' services/laya/run.sh
start worker 'src/worker.ts' npm run worker -w livekit-temporal
start translator 'src/translator.ts' npm run translator -w livekit-temporal
start agent 'src/agent.ts' npm run dev -w voice-agent

# A process started earlier (by us or by hand) has no fresh log to check, so fall back to probes
wait_for "mlx-audio (STT + TTS)" 180 curl -sf "$SPEECH_BASE_URL/models"
wait_for "laya (action gate)" 120 curl -sf "$LAYA_BASE_URL/health"
wait_for "translator" 60 curl -sf "http://localhost:${TRANSLATOR_PORT:-3100}/healthz"
if [ -f "$RUN/worker.pid" ] && kill -0 "$(cat "$RUN/worker.pid")" 2>/dev/null; then
  wait_for "Temporal worker" 60 grep -q "state: 'RUNNING'" "$RUN/worker.log"
fi
if [ -f "$RUN/agent.pid" ] && kill -0 "$(cat "$RUN/agent.pid")" 2>/dev/null; then
  wait_for "voice agent" 60 grep -q 'registered worker' "$RUN/agent.log"
fi
if [ -f "$RUN/mlx-audio.pid" ] && kill -0 "$(cat "$RUN/mlx-audio.pid")" 2>/dev/null; then
  wait_for "mlx-audio model warm-up" 180 grep -q 'mlx-audio ready' "$RUN/mlx-audio.log"
fi
echo "  all ready"

step "Meeting"
token=$(lk token create --join --room "$ROOM" --identity "$IDENTITY" --valid-for 24h 2>/dev/null |
  awk '/Access token:/ {print $3}')
[ -n "$token" ] || die "could not create a token"
url="https://meet.livekit.io/custom?liveKitUrl=wss://livekit.localhost&token=$token"
temporal="http://localhost:8233/namespaces/default/workflows?query=RoomName%3D%22$ROOM%22"

echo "  room:      $ROOM (as $IDENTITY)"
echo "  meet:      $url"
echo "  temporal:  $temporal"
echo "  logs:      .run/*.log    stop: scripts/stop.sh"
[ "${NO_OPEN:-}" = 1 ] || open "$url"
