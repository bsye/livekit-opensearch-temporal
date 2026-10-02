#!/usr/bin/env bash
# Starts whatever isn't running yet and opens a meeting the agent joins.
#   npm start [-- room [identity]]          defaults: meet-<HHMMSS>, $USER
#   AGENT=cascade|s2s|omni npm start         skip the agent menu (no menu outside a terminal: cascade)
#   NO_OPEN=1 npm start                      print the link instead of opening the browser
# Native processes run detached, logs in .run/. Stop with `npm stop`.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
. scripts/lib.sh

ROOM="${1:-meet-$(date +%H%M%S)}"
IDENTITY="${2:-$USER}"

step "Checking prerequisites"
[ -f .env ] && [ -d node_modules ] || die "run \`npm run setup\` first"
docker info >/dev/null 2>&1 || die "Docker (OrbStack) is not running"
command -v lk >/dev/null || die "LiveKit CLI missing: run \`npm run setup\`"
load_env
echo "  ok"

if [ -z "${AGENT:-}" ]; then
  if interactive; then AGENT=$(choose agent) || exit 130; else AGENT=cascade; fi
fi
agent_entry "$AGENT" >/dev/null || die "AGENT must be one of: $AGENTS"

step "Infrastructure (docker compose)"
docker compose up -d --quiet-pull 2>&1 | grep -vE 'Running|Started|Healthy|Waiting|Created' || true
wait_for "LiveKit" 60 curl -sf https://livekit.localhost
wait_for "Temporal namespace" 120 temporal_ready
echo "  LiveKit and Temporal ready"

if [ "$AGENT" != omni ]; then
  step "LLM (LM Studio, $LLM_MODEL)"
  "$LMS" server start >/dev/null 2>&1 || true
  "$LMS" ps 2>/dev/null | grep -q "$LLM_MODEL" || "$LMS" load "$LLM_MODEL" --identifier "$LLM_MODEL" -y >/dev/null
  wait_for "LM Studio" 60 sh -c "curl -sf '$LLM_BASE_URL/models' | grep -q '$LLM_MODEL'"
  echo "  ready"
fi

step "Native services"
for other in $AGENTS; do
  # one agent per room: any other kind would join it too
  [ "$other" = "$AGENT" ] || stop_process "$other agent" "$(agent_entry "$other")" >/dev/null
done
start laya 'services/laya/server.py' services/laya/run.sh
case "$AGENT" in
  cascade) start mlx-audio 'mlx_audio.server' services/mlx-audio/run.sh ;;
  s2s) start voicechat 'services/voicechat/server.py' services/voicechat/run.sh ;;
  omni)
    start mlx-audio 'mlx_audio.server' services/mlx-audio/run.sh
    start omni 'services/omni/server.py' services/omni/run.sh ;;
esac
start worker 'src/worker.ts' npm start -w @voice/worker
start translator 'src/translator.ts' npm start -w @voice/translator
start agent "$(agent_entry "$AGENT")" npm run "start:$AGENT" -w @voice/agent

wait_for "laya" 120 curl -sf "$LAYA_BASE_URL/health"
case "$AGENT" in
  cascade)
    wait_for "mlx-audio" 180 curl -sf "$SPEECH_BASE_URL/models"
    wait_for_log mlx-audio "mlx-audio warm-up" 180 'mlx-audio ready' ;;
  s2s)
    wait_for "voicechat" 300 curl -sf "http://localhost:$(port_of "$VOICECHAT_URL")/health"
    echo "  prefilling the agent prompt (about a minute the first time each day)..."
    npm run -s warm:s2s -w @voice/agent ;;
  omni)
    wait_for "mlx-audio" 180 curl -sf "$SPEECH_BASE_URL/models"
    wait_for "omni" 300 curl -sf "http://localhost:$(port_of "$OMNI_BASE_URL")/health" ;;
esac
wait_for "translator" 60 curl -sf "http://localhost:$TRANSLATOR_PORT/healthz"
wait_for_log worker "Temporal worker" 60 "state: 'RUNNING'"
wait_for_log agent "voice agent" 60 'registered worker'
echo "  all ready"

step "Meeting"
token=$(lk token create --join --room "$ROOM" --identity "$IDENTITY" --valid-for 24h 2>/dev/null | awk '/Access token:/ {print $3}')
[ -n "$token" ] || die "could not create a token"
url="https://meet.livekit.io/custom?liveKitUrl=wss://livekit.localhost&token=$token"
echo "  agent:     $AGENT"
echo "  room:      $ROOM (as $IDENTITY)"
echo "  meet:      $url"
echo "  temporal:  http://localhost:8233/namespaces/$TEMPORAL_NAMESPACE/workflows?query=RoomName%3D%22$ROOM%22"
echo "  logs:      .run/*.log    stop: npm stop"
[ "${NO_OPEN:-}" = 1 ] || open "$url"
