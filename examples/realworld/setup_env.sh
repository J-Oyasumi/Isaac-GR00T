#!/usr/bin/env bash
# One-shot env setup for realworld GR00T finetuning on a dGPU server.
# Assumes: repo already cloned, run from anywhere inside the repo, sudo for apt.
set -euo pipefail

CYAN='\033[1;36m'; YELLOW='\033[1;33m'; GREEN='\033[1;32m'; RED='\033[1;31m'; NC='\033[0m'
log()   { echo -e "${CYAN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[setup]${NC} $*"; }
ok()    { echo -e "${GREEN}[setup]${NC} $*"; }
fail()  { echo -e "${RED}[setup]${NC} $*" >&2; exit 1; }

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
log "repo root: $REPO_ROOT"

# Sudo only when not root
SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

# ── system deps: ffmpeg (torchcodec runtime), libaio-dev (deepspeed async I/O)
log "installing system deps (ffmpeg, libaio-dev)"
$SUDO apt-get update -qq
$SUDO apt-get install -y --no-install-recommends ffmpeg libaio-dev

# ── uv
if ! command -v uv &>/dev/null; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
log "uv: $(uv --version)"

# ── uv sync — flash-attn wheel (~395MB) frequently times out, retry up to 3x.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
log "running uv sync --all-extras (UV_HTTP_TIMEOUT=$UV_HTTP_TIMEOUT)"
synced=0
for attempt in 1 2 3; do
    if uv sync --all-extras; then
        synced=1; break
    fi
    warn "uv sync attempt $attempt failed; retrying in 10s"
    sleep 10
done
[ "$synced" = 1 ] || fail "uv sync failed after 3 attempts"

# ── editable install of gr00t package itself
log "installing gr00t in editable mode"
uv pip install -e .

# ── verify the dataset is present (137MB, not in git — must be transferred separately)
DATASET="$REPO_ROOT/examples/realworld/realworld_lerobot"
if [ ! -d "$DATASET/data" ] || [ ! -f "$DATASET/meta/modality.json" ] || [ ! -f "$DATASET/meta/stats.json" ]; then
    warn "dataset not found at $DATASET"
    warn "  transfer it from the source machine, e.g.:"
    warn "    rsync -avh --info=progress2 source:/path/to/realworld_lerobot/ $DATASET/"
else
    ok "dataset present: $(du -sh "$DATASET" | cut -f1) at $DATASET"
fi

# ── wandb credentials reminder
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    warn "wandb not logged in. before running train.sh:  uv run wandb login"
fi

ok "setup complete. activate venv with:  source .venv/bin/activate"
