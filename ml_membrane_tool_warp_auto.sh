#!/bin/bash
# ml_membrane_tool_warp_auto.sh — run one of the membrane-branch PYTHON tools
# inside a conda env that has numpy + mrcfile.
#
#   bash ml_membrane_tool_warp_auto.sh <script.py> [args…]
#
# The tools (ml_fit_virions.py, ml_pick_surfaces.py, ml_explore_membrane.py)
# are plain python, but tomogration's own venv holds only PySide6 — so a stage
# with base "python3" would run them against an interpreter with no numpy. This
# wrapper is the one place that knows which env they belong to.
#
#   MB_CONDA_ENV   env to activate                    (default: membrainseg)
#   MB_TOOL_DRYRUN 1 = print the command and stop
#
# Never `source activate`: sourcing a missing file aborts a non-interactive
# shell with no message. Same idiom as the other membrane wrappers.
set -e

SCRIPT="${1:?Usage: $0 <script.py> [args…]}"
shift
ENV_NAME="${MB_CONDA_ENV:-membrainseg}"

# The tool is EITHER a .py shipped with the app, or a console command the env
# provides (surforama is a command, not a file). A path is checked now; a bare
# name can only be checked after the env is active, since that is what puts it
# on PATH — so that check lives at the bottom.
case "$SCRIPT" in
    */*|*.py) [ -f "$SCRIPT" ] || {
        echo "ERROR: tool not found: $SCRIPT"
        echo "       If this is a viewer shipped separately (tomoview.py), it"
        echo "       may not have synced to this machine — check ~/bin too."
        exit 1; } ;;
esac

echo "==================================================================="
echo "$(basename "$SCRIPT")  ·  $(date)"
echo "env: $ENV_NAME"
echo "==================================================================="

module load miniconda/latest 2>/dev/null || echo "WARNING: could not load miniconda/latest"
command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not on PATH."; exit 1; }
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME" || {
    echo "ERROR: could not activate '$ENV_NAME'."
    echo "       The membrane python tools need numpy + mrcfile."
    exit 1; }

python3 -c "import numpy, mrcfile" 2>/dev/null || {
    echo "ERROR: '$ENV_NAME' has no numpy/mrcfile — wrong env for these tools."
    exit 1; }

if [ -f "$SCRIPT" ]; then
    RUN=(python3 "$SCRIPT")
elif command -v "$SCRIPT" >/dev/null 2>&1; then
    RUN=("$SCRIPT")            # a console command the env provides
else
    echo "ERROR: '$SCRIPT' is neither a file nor a command in '$ENV_NAME'."
    exit 1
fi

if [ "${MB_TOOL_DRYRUN:-}" = "1" ]; then
    echo "DRY RUN: ${RUN[*]} $*"
    exit 0
fi
exec "${RUN[@]}" "$@"
