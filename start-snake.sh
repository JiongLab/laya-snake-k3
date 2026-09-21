#!/bin/sh
set -eu
APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
LAYA_K3_HOME=${LAYA_K3_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/laya-snake-k3}
export LAYA_K3_HOME
if [ ! -x "$LAYA_K3_HOME/venv/bin/python" ]; then
    echo "Run laya-snake-k3-setup first (source checkout: sh $APP_DIR/setup.sh)." >&2
    exit 1
fi
export USE_TF=0 USE_TORCH=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
export PYTHONPATH="$APP_DIR"
exec "$LAYA_K3_HOME/venv/bin/python" -m snake_k3 --width 12 --height 8 --max-speed --backend spacemit "$@"
