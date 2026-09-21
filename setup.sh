#!/bin/sh
set -eu
APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MODEL_SOURCE=
SKIP_MODELS=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --help|-h)
            echo 'Usage: setup.sh [--model-source /path/to/models] [--skip-models]'
            echo 'Creates a per-user venv; downloads pinned models unless a verified local source is supplied.'
            echo 'Set LAYA_K3_HOME to choose another user-owned data directory.'
            exit 0 ;;
        --model-source) MODEL_SOURCE=${2:?Missing models directory}; shift 2 ;;
        --skip-models) SKIP_MODELS=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [ "$(id -u)" -eq 0 ]; then
    echo 'Run setup as your desktop user, not with sudo.' >&2
    exit 1
fi
if [ "$(uname -m)" != riscv64 ]; then
    echo 'This runtime is validated only on SpaceMIT K3 / Bianbu 4.0.5 riscv64.' >&2
    exit 1
fi
python3 -c 'import sys; assert sys.version_info[:2] == (3,14), "Python 3.14 is required by this Bianbu release"'
for entry in 'python3-torch=2.9.1+dfsg-1~exp1ubuntu2' 'python3-spacemit-ort=2.0.6' 'spacemit-onnxruntime=2.0.6' 'spacemit-tcm=3.0.0+5'; do
    package=${entry%%=*}
    want=${entry#*=}
    got=$(dpkg-query -W -f='${Version}' "$package" 2>/dev/null || true)
    if [ "$got" != "$want" ]; then
        echo "Expected $entry, found '$got'. See docs/REPRODUCE.md for the pinned apt command." >&2
        exit 1
    fi
done
LAYA_K3_HOME=${LAYA_K3_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/laya-snake-k3}
export LAYA_K3_HOME
export PYTHONNOUSERSITE=1
unset PYTHONPATH
mkdir -p "$LAYA_K3_HOME"
python3 -m venv --system-site-packages "$LAYA_K3_HOME/venv"
"$LAYA_K3_HOME/venv/bin/python" -m pip install --no-index --no-deps \
    --find-links "$APP_DIR/wheelhouse" -r "$APP_DIR/requirements-runtime.txt"
# Match the provider selection used by onnx_backend; Bianbu also ships a
# generic ORT in another directory, which is not the runtime we execute.
PYTHONPATH=/usr/lib/python3.14/dist-packages "$LAYA_K3_HOME/venv/bin/python" -m pip check
if [ "$SKIP_MODELS" -eq 0 ]; then
    if [ -n "$MODEL_SOURCE" ]; then
        python3 "$APP_DIR/scripts/models.py" --directory "$LAYA_K3_HOME/models" --source "$MODEL_SOURCE"
    else
        python3 "$APP_DIR/scripts/models.py" --directory "$LAYA_K3_HOME/models"
    fi
fi
echo "Setup complete: $LAYA_K3_HOME"
echo "Start: $APP_DIR/start-snake.sh"
