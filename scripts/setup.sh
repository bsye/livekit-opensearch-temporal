#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
. scripts/lib.sh

WITH_S2S=0
WITH_OMNI=0
ASKED=0
for arg in "$@"; do
  case "$arg" in
    --s2s) WITH_S2S=1; ASKED=1 ;;
    --omni) WITH_OMNI=1; ASKED=1 ;;
    *) die "unknown option $arg" ;;
  esac
done

[ "$(uname -sm)" = "Darwin arm64" ] || echo "warning: built for Apple Silicon (MLX); other platforms won't run the model sidecars"
command -v brew >/dev/null || die "Homebrew is required: https://brew.sh"

step "Tools (Homebrew)"
brew_install() {
  if command -v "${2:-$1}" >/dev/null; then echo "  $1 ok"; else brew install "$1"; fi
}
brew_install node
brew_install python@3.11 python3.11
brew_install livekit-cli lk
if ! command -v docker >/dev/null; then brew install --cask orbstack; fi
echo "  docker ok"
if [ ! -x "$LMS" ]; then
  [ -d "/Applications/LM Studio.app" ] || brew install --cask lm-studio
  die "open LM Studio once (it installs its \`lms\` CLI), then re-run \`npm run setup\`"
fi
echo "  lm studio ok"

step "Environment (.env)"
[ -f .env ] || cp .env.example .env
while IFS= read -r line; do
  case "$line" in '' | \#*) continue ;; esac
  grep -q "^${line%%=*}=" .env || echo "$line" >>.env
done <.env.example
if grep -q '^LIVEKIT_API_SECRET=$' .env; then
  sed -i '' "s/^LIVEKIT_API_SECRET=$/LIVEKIT_API_SECRET=$(openssl rand -hex 32)/" .env
  echo "  generated LIVEKIT_API_SECRET"
fi
load_env
echo "  ok"

step "Node packages"
npm install --no-audit --no-fund

if [ "$ASKED" = 0 ] && interactive; then
  extra=$(choose models) || exit 130
  case " $extra " in *" s2s "*) WITH_S2S=1 ;; esac
  case " $extra " in *" omni "*) WITH_OMNI=1 ;; esac
fi

step "Python sidecars (venvs + models)"
prefetch() {
  local service=$1; shift
  (cd "services/$service" && . ../lib.sh && export HF_HUB_DISABLE_PROGRESS_BARS=1 && ensure_venv &&
    for repo in "$@"; do .venv/bin/python -c "from huggingface_hub import snapshot_download as d; d('$repo')" >/dev/null; done)
  echo "  $service ok"
}
prefetch laya "$LAYA_MODEL"
prefetch mlx-audio "$STT_MODEL" "$TTS_MODEL"
if [ "$WITH_S2S" = 1 ]; then prefetch voicechat "$VOICECHAT_MODEL"; fi
if [ "$WITH_OMNI" = 1 ]; then prefetch omni "$OMNI_MODEL"; fi

step "LLM ($LLM_MODEL)"
"$LMS" ls 2>/dev/null | grep -q "${LLM_MODEL#*/}" || "$LMS" get "$LLM_MODEL" -y

step "Containers"
open -ga OrbStack 2>/dev/null || true
wait_for "Docker" 60 docker info
docker compose up -d --quiet-pull 2>&1 | grep -vE 'Running|Started|Healthy|Waiting|Created' || true
CA="data/caddy/pki/authorities/local/root.crt"
wait_for "Caddy's local CA" 60 test -f "$CA"

step "Trust Caddy's local CA (for wss://livekit.localhost in the browser)"
fingerprint=$(openssl x509 -noout -fingerprint -sha256 -in "$CA" | cut -d= -f2 | tr -d :)
if security find-certificate -a -Z /Library/Keychains/System.keychain 2>/dev/null | grep -q "$fingerprint"; then
  echo "  already trusted"
else
  echo "  adding it to the System keychain (asks for your password)"
  sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain "$ROOT/$CA"
fi

step "Done"
echo "  npm start      open a meeting with the agent"
echo "  npm stop       stop it (npm stop -- --all also stops the containers)"
