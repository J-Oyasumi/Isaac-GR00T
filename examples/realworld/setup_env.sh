#!/usr/bin/env bash
# One-shot env setup for realworld GR00T finetuning on a dGPU server.
# Assumes: repo cloned, uv already installed, no sudo.
# ffmpeg runtime libs come from a conda env (default: ffmpeg-libs);
# override with: CONDA_FFMPEG_PREFIX=/path/to/envs/<name>
set -euo pipefail

CYAN='\033[1;36m'; YELLOW='\033[1;33m'; GREEN='\033[1;32m'; RED='\033[1;31m'; NC='\033[0m'
log()   { echo -e "${CYAN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[setup]${NC} $*"; }
ok()    { echo -e "${GREEN}[setup]${NC} $*"; }
fail()  { echo -e "${RED}[setup]${NC} $*" >&2; exit 1; }

# Resolve REPO_ROOT — same logic as train.sh (SLURM-safe, no hard git dep).
if [ -z "${REPO_ROOT:-}" ]; then
    if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "$SLURM_SUBMIT_DIR/examples/realworld" ]; then
        REPO_ROOT="$SLURM_SUBMIT_DIR"
    else
        _src_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" || _src_dir=""
        if [ -n "$_src_dir" ] && _gt="$(git -C "$_src_dir" rev-parse --show-toplevel 2>/dev/null)"; then
            REPO_ROOT="$_gt"
        elif [ -n "$_src_dir" ] && [ -d "$_src_dir/../../examples/realworld" ]; then
            REPO_ROOT="$(cd "$_src_dir/../.." && pwd)"
        fi
    fi
fi
[ -n "${REPO_ROOT:-}" ] && [ -d "$REPO_ROOT/examples/realworld" ] \
    || fail "cannot resolve REPO_ROOT. cd to the repo root first, or export REPO_ROOT=/path/to/Isaac-GR00T"
cd "$REPO_ROOT"
log "repo root: $REPO_ROOT"

# ── locate ffmpeg-libs conda env (no apt, no sudo)
FFMPEG_ENV_NAME="${FFMPEG_ENV_NAME:-ffmpeg-libs}"
if [ -z "${CONDA_FFMPEG_PREFIX:-}" ]; then
    if command -v conda &>/dev/null; then
        CONDA_FFMPEG_PREFIX="$(conda info --base)/envs/$FFMPEG_ENV_NAME"
    elif [ -d "$HOME/miniconda3/envs/$FFMPEG_ENV_NAME" ]; then
        CONDA_FFMPEG_PREFIX="$HOME/miniconda3/envs/$FFMPEG_ENV_NAME"
    elif [ -d "$HOME/anaconda3/envs/$FFMPEG_ENV_NAME" ]; then
        CONDA_FFMPEG_PREFIX="$HOME/anaconda3/envs/$FFMPEG_ENV_NAME"
    else
        fail "couldn't auto-locate conda env '$FFMPEG_ENV_NAME'. set CONDA_FFMPEG_PREFIX=/path/to/envs/$FFMPEG_ENV_NAME"
    fi
fi
[ -d "$CONDA_FFMPEG_PREFIX/lib" ] || fail "no $CONDA_FFMPEG_PREFIX/lib"
ls "$CONDA_FFMPEG_PREFIX/lib"/libavformat.so* >/dev/null 2>&1 \
    || fail "$CONDA_FFMPEG_PREFIX/lib doesn't contain libavformat — wrong env?"
log "ffmpeg conda env: $CONDA_FFMPEG_PREFIX"

# ── uv
command -v uv &>/dev/null || fail "uv not found in PATH"
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

# ── persist conda-ffmpeg path so train.sh can pick it up
echo "$CONDA_FFMPEG_PREFIX" > "$REPO_ROOT/examples/realworld/.ffmpeg_prefix"
log "wrote examples/realworld/.ffmpeg_prefix"

# ── dataset presence check (137MB, not in git — fetch from gdrive)
DATASET="$REPO_ROOT/examples/realworld/realworld_lerobot"
if [ ! -d "$DATASET/data" ] || [ ! -f "$DATASET/meta/modality.json" ] || [ ! -f "$DATASET/meta/stats.json" ]; then
    warn "dataset not found at $DATASET"
    warn "  fetch it from Google Drive:"
    warn "    rclone copy gdrive:Projects/PointAction/realworld_lerobot_gr00t/ $DATASET/ --transfers 8 --progress"
else
    ok "dataset present: $(du -sh "$DATASET" | cut -f1) at $DATASET"
fi

# ── wandb credentials reminder
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    warn "wandb not logged in. before running train.sh:  uv run wandb login"
fi

ok "setup complete. activate venv with:  source .venv/bin/activate"
