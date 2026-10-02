# Sourced by scripts/start.sh and scripts/stop.sh.
RUN="$ROOT/.run"
LMS="$HOME/.lmstudio/bin/lms"
AGENTS="cascade s2s omni"
mkdir -p "$RUN"

step() { printf '\n\033[1m▸ %s\033[0m\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
load_env() { set -a; . "$ROOT/.env"; set +a; }

agent_entry() {
  case "$1" in
    cascade) echo 'src/cascade/agent.ts' ;;
    s2s) echo 'src/s2s/agent.ts' ;;
    omni) echo 'src/omni/agent.ts' ;;
    *) return 1 ;;
  esac
}

interactive() { [ -t 0 ] && [ -t 2 ]; }

# choose agent|models: arrow-key picker (scripts/choose.ts), prints the answer
choose() { (cd "$ROOT" && node --import tsx scripts/choose.ts "$1"); }

port_of() { local p="${1##*:}"; echo "${p%%/*}"; }

# wait_for DESCRIPTION TIMEOUT_SECONDS COMMAND...
wait_for() {
  local desc=$1 timeout=$2; shift 2
  for ((i = 0; i < timeout * 2; i++)); do
    "$@" >/dev/null 2>&1 && return 0
    sleep 0.5
  done
  die "$desc not ready after ${timeout}s (logs in .run/)"
}

# wait_for_log NAME DESCRIPTION TIMEOUT PATTERN: only for processes this script started (fresh log)
wait_for_log() {
  local pid="$RUN/$1.pid"
  [ -f "$pid" ] && kill -0 "$(cat "$pid")" 2>/dev/null || return 0
  wait_for "$2" "$3" grep -q "$4" "$RUN/$1.log"
}

temporal_ready() {
  [ "$(docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' "$(docker compose ps -aq temporal-namespace)")" = "exited 0" ]
}

# start NAME PATTERN COMMAND...: run COMMAND detached unless a process matching PATTERN is up
start() {
  local name=$1 pattern=$2; shift 2
  if pgrep -f "$pattern" >/dev/null; then
    echo "  $name already running"
    rm -f "$RUN/$name.pid"
    return
  fi
  nohup "$@" >"$RUN/$name.log" 2>&1 &
  echo $! >"$RUN/$name.pid"
  echo "  $name started (.run/$name.log)"
}

stop_process() { # NAME PATTERN
  if pkill -f "$2"; then echo "stopped $1"; else echo "$1 not running"; fi
}
