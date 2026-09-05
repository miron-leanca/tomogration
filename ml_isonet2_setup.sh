#!/bin/bash
# ml_isonet2_setup.sh — one-time install of IsoNet 2 on THIS cluster machine.
#
#   bash ml_isonet2_setup.sh [install_parent_dir]
#
# IsoNet 1 is a cluster MODULE here (module load isonet/0.3) and is owned by
# the admins. IsoNet 2 is not packaged as a module, so the upstream install.sh
# builds it in place from its own isonet2_environment.yml.
#
#   upstream: https://github.com/IsoNet-cryoET/IsoNet2
#
# NOTE the env is a PREFIX, not a name: install.sh does `conda env create -p
# <repo>/build/conda_env`, so there is no 'isonet2_environment' to activate by
# name and `conda run -n` will never find it. Everything here — and any wrapper
# built on it later — addresses it by path. (Verified against a real install on
# 2026-08-15: 341 packages, pytorch 2.1.1/cu121, python 3.10, IsoNet2 2.0.1b0.)
# install.sh also curls the isoapp GUI AppImage into build/.
#
# SAFE TO RUN WHILE A JOB IS TRAINING. This script is CPU + disk only: it
# clones, solves and downloads packages. It deliberately does NOT touch a GPU
# (see the smoke test at the end, which is opt-in) — allocating VRAM next to a
# running isonet.py refine can OOM it hours into training.
#
# HARD RULES it keeps (spec §2):
#   * NEVER installs into membrainseg / membrainpick — their dependency pins
#     conflict (napari/PyQt/scipy) and merging them breaks membrain-pick.
#   * NEVER modifies the isonet/0.3 module; IsoNet 1 keeps working unchanged.
#     The two are independent: IsoNet 1 = module, IsoNet 2 = conda env.
#
# Requirements upstream states: Linux 64-bit, NVIDIA GPU (Ampere+ recommended,
# CUDA compute capability >= 3.5), CUDA >= 11.8 (driver >= 520.61.05), ~24 GB
# VRAM recommended. Checked below where they are checkable without a GPU.
set -e

PARENT="${1:-/ceph/users/$USER/EMDatasets/processing_scripts}"
REPO="$PARENT/IsoNet2"
ENV_PREFIX="$REPO/build/conda_env"

echo "==================================================================="
echo "IsoNet 2 setup  ·  $(date)"
echo "install parent: $PARENT"
echo "conda env:      $ENV_PREFIX  (prefix, not a name)"
echo "==================================================================="

# ---- 1. what this machine offers ------------------------------------------
# Reported, not enforced: the install itself is CPU-only, so a login node with
# no GPU can legitimately build the env for a GPU node to use later.
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "--- GPUs on $(hostname) ---"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version \
               --format=csv,noheader || true
    DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    case "$DRV" in
        [0-9]*) MAJ="${DRV%%.*}"
                [ "$MAJ" -ge 520 ] 2>/dev/null \
                    || echo "WARNING: driver $DRV < 520.61.05 — IsoNet 2 needs CUDA >= 11.8." ;;
    esac
    echo "NOTE: a job may be using these GPUs right now — this script will not."
else
    echo "No nvidia-smi here (fine: the install is CPU-only)."
fi

# ---- 2. conda (never `source activate`: sourcing a missing file aborts) ----
module load miniconda/latest 2>/dev/null || echo "WARNING: could not load miniconda/latest"
command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not on PATH."; exit 1; }
eval "$(conda shell.bash hook)"
echo "conda: $(conda --version)   envs: $(conda config --show envs_dirs 2>/dev/null | tail -1)"

if [ -x "$ENV_PREFIX/bin/python" ]; then
    echo "SKIP: env already built at $ENV_PREFIX"
    HAVE_ENV=1
else
    HAVE_ENV=0
fi

# ---- 3. the source tree ----------------------------------------------------
mkdir -p "$PARENT"
if [ -d "$REPO/.git" ]; then
    echo "SKIP: $REPO already cloned (git pull it yourself if you want a newer one)."
else
    command -v git >/dev/null 2>&1 || { echo "ERROR: git not on PATH."; exit 1; }
    echo "Cloning IsoNet2 -> $REPO"
    git clone https://github.com/IsoNet-cryoET/IsoNet2.git "$REPO"
fi
[ -f "$REPO/install.sh" ] || { echo "ERROR: no install.sh in $REPO — upstream layout changed."; exit 1; }

# ---- 4. build the env (their script, their yml — we do not second-guess it) -
if [ "$HAVE_ENV" = "1" ]; then
    echo "Not re-running install.sh: the env exists. Delete it first to rebuild:"
    echo "    rm -rf $ENV_PREFIX"
else
    echo "Running upstream install.sh (this solves + downloads ~3 GB; CPU only)…"
    # install.sh sources its own bashrc at the end, which can `return` in a
    # sourced context — don't let that abort us before the report below.
    ( cd "$REPO" && bash install.sh ) || echo "install.sh exited non-zero — checking what it left behind…"
fi

# ---- 5. report what is actually there, and address it by PATH --------------
echo "==================================================================="
echo "Installed tree : $REPO"
echo "Shell setup    : source $REPO/isonet2.bashrc"
echo "Env prefix     : $ENV_PREFIX"
echo "Activate       : conda activate $ENV_PREFIX      # -p/path, NOT -n/name"
echo "One-shot       : conda run -p $ENV_PREFIX <cmd>"
if [ -x "$ENV_PREFIX/bin/python" ]; then
    # Ask the METADATA, not the module: the wheel is 'IsoNet2' but the import
    # name is not, so `import isonet2` fails on a perfectly good install.
    echo "version        : $(conda run -p "$ENV_PREFIX" python -c \
        'import sys;from importlib.metadata import version;print("IsoNet2", version("IsoNet2"), "· python", sys.version.split()[0])' \
        2>/dev/null || echo '(version metadata not found)')"
    echo "--- subcommands this build actually offers ---"
    conda run -p "$ENV_PREFIX" isonet.py --help 2>&1 | sed -n '1,40p' \
        || echo "(isonet.py not on the env PATH; source the bashrc in a fresh shell)"
else
    echo "ERROR: no python at $ENV_PREFIX — read the install.sh output above."
fi
[ -f "$REPO/build/isoapp-1.0.0.AppImage" ] \
    && echo "GUI (theirs)   : $REPO/build/isoapp-1.0.0.AppImage  (not used by tomogration)"
echo
echo "GPU smoke test — run this ONLY when no training job is using the GPUs:"
echo "    conda run -p $ENV_PREFIX python -c \\"
echo "      'import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'"
echo
echo "IsoNet 1 (module isonet/0.3) is untouched and still works."
echo "Finished: $(date)"
echo "==================================================================="
