#!/usr/bin/env bash
# Launch GR00T N1.7 finetuning on the realworld (xarm7) dataset.
# Override defaults via env vars, e.g.:
#   NUM_GPUS=4 GLOBAL_BATCH_SIZE=64 MAX_STEPS=20000 bash examples/realworld/train.sh
set -euo pipefail

CYAN='\033[1;36m'; YELLOW='\033[1;33m'; GREEN='\033[1;32m'; RED='\033[1;31m'; NC='\033[0m'
log()  { echo -e "${CYAN}[train]${NC} $*"; }
warn() { echo -e "${YELLOW}[train]${NC} $*"; }
ok()   { echo -e "${GREEN}[train]${NC} $*"; }
fail() { echo -e "${RED}[train]${NC} $*" >&2; exit 1; }

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

DATASET="examples/realworld/realworld_lerobot"
CONFIG="examples/realworld/realworld_config.py"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/realworld_finetune}"
EXP_NAME="${EXP_NAME:-realworld_$(date +%Y%m%d_%H%M%S)}"
WANDB_PROJECT="${WANDB_PROJECT:-realworld-gr00t}"

# defaults — finetune.sh reads these via env
export NUM_GPUS="${NUM_GPUS:-2}"
export USE_WANDB="${USE_WANDB:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export MAX_STEPS="${MAX_STEPS:-10000}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# ── pre-flight checks
[ -d "$REPO_ROOT/.venv" ] || fail "no .venv. run: bash examples/realworld/setup_env.sh"
[ -d "$DATASET/data" ]    || fail "dataset missing: $DATASET/data"
[ -f "$DATASET/meta/modality.json" ] || fail "missing $DATASET/meta/modality.json"
[ -f "$DATASET/meta/stats.json" ]    || fail "missing $DATASET/meta/stats.json — regenerate stats first"
[ -f "$CONFIG" ]          || fail "missing modality config: $CONFIG"

if [ "$USE_WANDB" = 1 ]; then
    if [ -z "${WANDB_API_KEY:-}" ] && ! grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
        fail "USE_WANDB=1 but wandb not logged in. run:  uv run wandb login   (or set WANDB_API_KEY)"
    fi
fi

# activate venv
# shellcheck disable=SC1091
source .venv/bin/activate

# ── debug print: dataset + run config
log "== run config =="
log "  base_model        = $BASE_MODEL"
log "  dataset           = $DATASET ($(du -sh "$DATASET" | cut -f1))"
log "  modality_config   = $CONFIG"
log "  embodiment_tag    = NEW_EMBODIMENT"
log "  output_dir        = $OUTPUT_DIR"
log "  experiment_name   = $EXP_NAME"
log "  wandb_project     = $WANDB_PROJECT (USE_WANDB=$USE_WANDB)"
log "  num_gpus          = $NUM_GPUS"
log "  global_batch_size = $GLOBAL_BATCH_SIZE  (per-GPU = $((GLOBAL_BATCH_SIZE / NUM_GPUS)))"
log "  max_steps         = $MAX_STEPS  save_steps=$SAVE_STEPS"
log "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# quick GPU sanity print
python - <<'PY'
import torch
from rich import print as rprint
rprint(f"[cyan][train][/cyan] torch={torch.__version__} cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    rprint(f"[cyan][train][/cyan]   gpu[{i}] = {torch.cuda.get_device_name(i)}")
PY

mkdir -p "$OUTPUT_DIR"
ok "launching finetune"

exec bash examples/finetune.sh \
    --base-model-path "$BASE_MODEL" \
    --dataset-path    "$DATASET" \
    --modality-config-path "$CONFIG" \
    --embodiment-tag  NEW_EMBODIMENT \
    --output-dir      "$OUTPUT_DIR" \
    --experiment-name "$EXP_NAME" \
    --wandb-project   "$WANDB_PROJECT" \
    "$@"
