set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a; . "$ROOT/.env"; set +a
PYTHON="${PYTHON:-$(command -v python3.11 || echo /opt/homebrew/bin/python3.11)}"

ensure_venv() {
  [ -x .venv/bin/python ] || "$PYTHON" -m venv .venv
  if ! cmp -s requirements.txt .venv/requirements.installed; then
    .venv/bin/pip install -q -r requirements.txt
    cp requirements.txt .venv/requirements.installed
  fi
}

port_of() {
  local p="${1##*:}"
  echo "${p%%/*}"
}
