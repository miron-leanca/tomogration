#!/bin/bash
# ml_isonet1_predict_warp_auto.sh
#
# Apply a trained IsoNet 1 model to tomograms (isonet.py predict) — the
# "predict many" half of train-once-predict-many. The model path is a config
# field INDEPENDENT of the data path (a reused config whose data_directory
# silently pointed at the wrong dataset caused a real bug once — both resolved
# paths are echoed up front, and the tomogram count is checked against the
# project's tilt-series count so a mismatch is loud, not silent).
#
# OUTPUTS ARE WEDGE-RESTORED: for PICKING COORDINATES ONLY. The PROVENANCE.json
# written here tags them so deconvolution refuses them as input and any future
# extraction path can refuse them as source. Particle extraction for averaging
# must always reference the original reconstruction.
#
# Usage:
#   bash ml_isonet1_predict_warp_auto.sh <tomo_dir> <output_dir>
#
# <tomo_dir>    tomograms to correct (the ORIGINAL reconstruction)
# <output_dir>  corrected tomograms land here (e.g. membrane/isonet_corrected)
#
# Knobs via environment (the tomogration GUI sets these):
#   ISO_MODEL      trained model .h5 (REQUIRED), e.g.
#                  membrane/isonet/results/model_iter30.h5
#   ISO_MODULE     lmod module                       (default: isonet/0.3)
#   ISO_GPU        --gpuID comma list                (default: 0)
#   ISO_PIXEL_SIZE star pixel size Å/px              (default: 12.56)
#   ISO_TOMO_LIST  space/comma list of stems; empty = ALL tomograms
#   ISO_CUBE       --cube_size                       (default: tool default 64)
#   ISO_CROP       --crop_size (raise if patchy artifacts) (default: tool default 96)
#   ISO_FORCE      1 = redo existing outputs
#
# Flags verified against isonet.py 0.3 -h (2026-08-15).
set -e

TOMO_DIR="${1:?Usage: $0 <tomo_dir> <output_dir>}"
OUT_DIR="${2:?Usage: $0 <tomo_dir> <output_dir>}"

ISO_MOD="${ISO_MODULE:-isonet/0.3}"
MODEL="${ISO_MODEL:-}"
GPU="${ISO_GPU:-0}"
PIX="${ISO_PIXEL_SIZE:-12.56}"
FORCE="${ISO_FORCE:-0}"
TOMO_LIST="$(printf '%s' "${ISO_TOMO_LIST:-}" | tr ',' ' ')"

[ -d "$TOMO_DIR" ] || { echo "ERROR: tomo dir not found: $TOMO_DIR"; exit 1; }
[ -n "$MODEL" ] || { echo "ERROR: ISO_MODEL (trained .h5) is required."; exit 1; }
[ -f "$MODEL" ] || { echo "ERROR: model not found: $MODEL"; exit 1; }
# Same refusal as training: the model was trained against the ORIGINAL
# reconstruction (with IsoNet's own deconv in the star) — predicting on an
# externally processed variant mixes pipelines and mislabels provenance.
if [ -f "$TOMO_DIR/PROVENANCE.json" ]; then
    if grep -Eqi '"variant"[^,}]*(denoise|wedge|isonet|deconv)' "$TOMO_DIR/PROVENANCE.json"; then
        echo "ERROR: $TOMO_DIR is a processed variant (see its PROVENANCE.json)."
        echo "Predict on the ORIGINAL reconstruction."
        exit 1
    fi
fi

ABS_TOMO_DIR="$(cd "$TOMO_DIR" && pwd)"
ABS_MODEL="$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")"
PROJECT_ROOT="$(pwd)"
mkdir -p "$OUT_DIR"

echo "==================================================================="
echo "IsoNet 1 predict  ·  $(date)"
echo "data:   $ABS_TOMO_DIR"
echo "model:  $ABS_MODEL"
echo "output: $OUT_DIR    gpu: $GPU"
echo "==================================================================="

module load "$ISO_MOD" 2>/dev/null || { echo "ERROR: could not module load $ISO_MOD"; exit 1; }
command -v isonet.py >/dev/null 2>&1 || { echo "ERROR: isonet.py not on PATH after $ISO_MOD"; exit 1; }

# Select tomograms ('_'-anchored tune list; typo = hard error).
IN="$OUT_DIR/input_tomos"
mkdir -p "$IN"

# Link dot-free, exactly as the training wrapper does: IsoNet 0.3 parses a
# star's micrograph name with split('.')[0] in places and splitext() in others,
# so Warp's 'Position003_12.56Apx.mrc' becomes two different names inside the
# tool. Training died on that (2026-08-15); the corrected volumes here inherit
# the sanitised name ('12p56Apx'), keeping the series prefix intact.
link_tomo() {
    local b stem safe
    b="$(basename "$1")"
    stem="${b%.mrc}"
    safe="${stem//./p}"
    ln -sf "$1" "$IN/$safe.mrc"
    [ "$safe" = "$stem" ] || echo "  linked $stem -> $safe (IsoNet mis-parses dots)"
}

rm -f "$IN"/*.mrc 2>/dev/null || true
if [ -n "$TOMO_LIST" ]; then
    n=0
    for stem in $TOMO_LIST; do
        for f in "$ABS_TOMO_DIR/$stem.mrc" "$ABS_TOMO_DIR/${stem}_"*.mrc; do
            [ -e "$f" ] || continue
            link_tomo "$f"; n=$((n+1))
        done
    done
    [ "$n" -gt 0 ] || { echo "ERROR: ISO_TOMO_LIST ($TOMO_LIST) matched nothing."; exit 1; }
else
    ls "$ABS_TOMO_DIR"/*.mrc >/dev/null 2>&1 || { echo "ERROR: no .mrc in $TOMO_DIR"; exit 1; }
    for f in "$ABS_TOMO_DIR"/*.mrc; do link_tomo "$f"; done
fi
N_SEL="$(ls "$IN" | wc -l | tr -d ' ')"

# Spec §3.2: warn when the job's tomogram count disagrees with what
# tomogration expects (the project's tilt-series count).
N_TS="$(ls "$PROJECT_ROOT"/tomostar/*.tomostar 2>/dev/null | wc -l | tr -d ' ')"
echo "$N_SEL tomogram(s) selected for correction (project has $N_TS tilt series)."
if [ -z "$TOMO_LIST" ] && [ "$N_TS" -gt 0 ] && [ "$N_SEL" -ne "$N_TS" ]; then
    echo "WARNING: correcting $N_SEL tomograms but the project has $N_TS tilt"
    echo "         series — check the data path (a reused config pointing at"
    echo "         the wrong dataset caused exactly this once before)."
fi

# Skip logic: predict writes into OUT_DIR/corrected/; done = every selected
# tomogram already has a corrected output ('_'-anchored name test).
CORR="$OUT_DIR/corrected"
if [ "$FORCE" != "1" ] && [ -d "$CORR" ]; then
    missing=0
    for f in "$IN"/*.mrc; do
        b="$(basename "$f" .mrc)"
        if ! { ls "$CORR/$b".* >/dev/null 2>&1 || ls "$CORR/${b}"_* >/dev/null 2>&1; }; then
            missing=$((missing+1))
        fi
    done
    if [ "$missing" -eq 0 ]; then
        echo "SKIP: all $N_SEL tomogram(s) already corrected in $CORR"
        exit 0
    fi
fi

# A minimal star over the selected set (predict reads tomograms via the star).
cd "$OUT_DIR"
isonet.py prepare_star input_tomos --output_star predict.star --pixel_size "$PIX"

PR_FLAGS=(--gpuID "$GPU" --output_dir corrected)
[ -n "${ISO_CUBE:-}" ] && PR_FLAGS+=(--cube_size "$ISO_CUBE")
[ -n "${ISO_CROP:-}" ] && PR_FLAGS+=(--crop_size "$ISO_CROP")
isonet.py predict predict.star "$ABS_MODEL" "${PR_FLAGS[@]}"

N_OUT="$(ls corrected/*.mrc 2>/dev/null | wc -l | tr -d ' ')"
# Provenance is written ONLY when there is real output — a failed run must not
# leave a wedge_restored tag on an empty folder.
[ "$N_OUT" -gt 0 ] || { echo "ERROR: predict produced no output."; exit 1; }
cat > PROVENANCE.json <<EOF
{
  "variant": "wedge_restored",
  "tool": "isonet.py 0.3 predict",
  "source_dir": "$ABS_TOMO_DIR",
  "model": "$ABS_MODEL",
  "params": {"pixel_size": "$PIX", "cube": "${ISO_CUBE:-default}",
             "crop": "${ISO_CROP:-default}"},
  "tomo_list": "$([ -n "$TOMO_LIST" ] && echo "$TOMO_LIST" || echo "ALL")",
  "extraction_allowed": false,
  "note": "missing-wedge restored: PICKING COORDINATES ONLY — never extraction input",
  "date": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF

echo "==================================================================="
echo "done: $N_OUT corrected tomogram(s) in $OUT_DIR/corrected/"
echo "These are WEDGE-RESTORED — pick coordinates on them, extract from the"
echo "original reconstruction (enforced via PROVENANCE.json)."
echo "Finished: $(date)"
echo "==================================================================="
