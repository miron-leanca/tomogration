#!/bin/bash
# ml_membrain_segment_warp_auto.sh
#
# Membrane segmentation with membrain-seg (env membrainseg, GPU). Runs
# `membrain segment` per tomogram; ~4 min/tomogram with 8-fold TTA on one GPU,
# so the loop prints a real progress estimate.
#
# --store-probabilities is ON unless explicitly disabled (MB_NO_PROBS=1):
# without the *_scores.mrc score map, every re-threshold costs a full GPU
# re-run — the score map is the whole point of the downstream tuning loop.
#
# Usage:
#   bash ml_membrain_segment_warp_auto.sh <input_tomo_dir> <output_dir>
#
# <input_tomo_dir>  tomograms to segment (original or deconvolved — segmentation
#                   is a COORDINATE path, so processed variants are fine here)
# <output_dir>      where segmentations + score maps land (created if missing;
#                   the tool's own default ./predictions is never relied on)
#
# Knobs via environment (the tomogration GUI sets these):
#   MB_CONDA_ENV    conda env with membrain-seg          (default: membrainseg)
#   MB_CKPT         path to the trained model .ckpt      (REQUIRED)
#   MB_GPU          CUDA_VISIBLE_DEVICES value           (default: 0)
#   MB_TOMO_LIST    space/comma list of series stems; empty = ALL (Batch).
#                   Name 1–3 stems to Tune.
#   MB_PIXEL_SIZE   --in-pixel-size Å/px                 (default: 12.56)
#   MB_OUT_PIXEL    --out-pixel-size Å/px                (default: blank = tool
#                   default 10; the tool says it should normally stay at 10)
#   MB_WINDOW       --sliding-window-size                (default: blank = tool
#                   default 160; smaller = less GPU but WORSE results)
#   MB_SEG_THRES    --segmentation-threshold             (default: blank = 0.0)
#   MB_NO_RESCALE   1 = drop --rescale-patches (on by default here: our
#                   tomograms are 12.56 Å/px, the model expects 10)
#   MB_NO_TTA       1 = --no-test-time-augmentation (faster, slightly worse)
#   MB_UNCERTAINTY  1 = --store-uncertainty-map (needs TTA on)
#   MB_NO_PROBS     1 = do NOT store score maps. STRONGLY discouraged — every
#                   re-threshold then needs a full GPU re-run.
#   MB_STORE_CC     1 = --store-connected-components (usually done later, as
#                   its own sweepable step)
#   MB_FORCE        1 = redo existing outputs
#   MB_EXTRA_ARGS   appended verbatim
#
# Flags verified against membrain-seg 0.0.10 --help (membrainseg env,
# 2026-08-14). Re-verify after env updates:
#   conda activate membrainseg && membrain segment --help
set -e

INPUT_DIR="${1:?Usage: $0 <input_tomo_dir> <output_dir>}"
OUT_DIR="${2:?Usage: $0 <input_tomo_dir> <output_dir>}"

MB_ENV="${MB_CONDA_ENV:-membrainseg}"
CKPT="${MB_CKPT:-}"
GPU="${MB_GPU:-0}"
PIX="${MB_PIXEL_SIZE:-12.56}"
FORCE="${MB_FORCE:-0}"
TOMO_LIST="$(printf '%s' "${MB_TOMO_LIST:-}" | tr ',' ' ')"

[ -d "$INPUT_DIR" ] || { echo "ERROR: input dir not found: $INPUT_DIR"; exit 1; }
[ -n "$CKPT" ] || { echo "ERROR: MB_CKPT (model checkpoint) is required."; exit 1; }
[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }

OPT_FLAGS=()
[ "${MB_NO_RESCALE:-}" = "1" ] || OPT_FLAGS+=(--rescale-patches)
[ -n "${MB_OUT_PIXEL:-}" ] && OPT_FLAGS+=(--out-pixel-size "$MB_OUT_PIXEL")
[ -n "${MB_WINDOW:-}" ]    && OPT_FLAGS+=(--sliding-window-size "$MB_WINDOW")
[ -n "${MB_SEG_THRES:-}" ] && OPT_FLAGS+=(--segmentation-threshold "$MB_SEG_THRES")
[ "${MB_NO_TTA:-}" = "1" ] && OPT_FLAGS+=(--no-test-time-augmentation)
[ "${MB_UNCERTAINTY:-}" = "1" ] && OPT_FLAGS+=(--store-uncertainty-map)
[ "${MB_STORE_CC:-}" = "1" ] && OPT_FLAGS+=(--store-connected-components)
if [ "${MB_NO_PROBS:-}" = "1" ]; then
    echo "WARNING: score maps DISABLED (MB_NO_PROBS=1) — every re-threshold"
    echo "         will need a full GPU re-run. You almost never want this."
else
    OPT_FLAGS+=(--store-probabilities)
fi
if [ "${MB_UNCERTAINTY:-}" = "1" ] && [ "${MB_NO_TTA:-}" = "1" ]; then
    echo "ERROR: --store-uncertainty-map requires test-time augmentation"
    echo "       (unset MB_NO_TTA or unset MB_UNCERTAINTY)."
    exit 1
fi

echo "==================================================================="
echo "MemBrain segment  ·  $(date)"
echo "input:  $INPUT_DIR"
echo "output: $OUT_DIR"
echo "ckpt:   $CKPT"
echo "GPU:    $GPU    in-pixel-size: $PIX"
echo "mode:   $([ -n "$TOMO_LIST" ] && echo "TUNE ($TOMO_LIST)" || echo "BATCH (all tomograms)")"
echo "==================================================================="

module load miniconda/latest 2>/dev/null || echo "WARNING: could not load miniconda/latest"
if ! conda activate "$MB_ENV" 2>/dev/null; then
    echo "ERROR: could not 'conda activate $MB_ENV'."
    echo "Expected at /ceph/users/haq21239/.conda/envs/membrainseg"
    exit 1
fi
command -v membrain >/dev/null 2>&1 || {
    echo "ERROR: membrain not found in env '$MB_ENV'."; exit 1; }

mkdir -p "$OUT_DIR"

# One output folder = one input variant. Score maps from a deconv run must not
# make a later run on the ORIGINAL reconstruction "skip", and PROVENANCE.json
# is one record per folder — mixing variants makes it lie about every earlier
# file (the exact confusion spec §5 documents happening once already).
if [ -f "$OUT_DIR/PROVENANCE.json" ] && [ "$FORCE" != "1" ]; then
    PREV_SRC="$(grep -o '"source_dir"[[:space:]]*:[[:space:]]*"[^"]*"' \
        "$OUT_DIR/PROVENANCE.json" | head -1 | sed 's/.*: *"//; s/"$//' || true)"
    if [ -n "$PREV_SRC" ] && [ "$PREV_SRC" != "$INPUT_DIR" ]; then
        echo "ERROR: $OUT_DIR already holds segmentations from a DIFFERENT input:"
        echo "         previous: $PREV_SRC"
        echo "         now:      $INPUT_DIR"
        echo "Use a separate output folder per input variant (e.g. "
        echo "membrane/segment_deconv vs membrane/segment_raw), or MB_FORCE=1"
        echo "to overwrite deliberately."
        exit 1
    fi
fi

# Record the upstream variant (raw vs deconvolved) — §5 provenance chain.
SRC_VARIANT="original"
if [ -f "$INPUT_DIR/PROVENANCE.json" ]; then
    SRC_VARIANT="$(grep -o '"variant"[[:space:]]*:[[:space:]]*"[^"]*"' \
        "$INPUT_DIR/PROVENANCE.json" | head -1 | sed 's/.*: *"//; s/"$//' || true)"
    SRC_VARIANT="${SRC_VARIANT:-unknown}"
fi

# Tune globs are '_'-anchored (Position_1 must not select Position_10), and an
# all-typo tune list is an ERROR, not a silent 0-tomogram "success".
TOMOS=()
if [ -n "$TOMO_LIST" ]; then
    for stem in $TOMO_LIST; do
        hits=()
        for f in "$INPUT_DIR/$stem.mrc" "$INPUT_DIR/${stem}_"*.mrc; do
            [ -e "$f" ] && hits+=("$f")
        done
        if [ ${#hits[@]} -gt 0 ]; then TOMOS+=("${hits[@]}")
        else echo "WARN: no ${stem}.mrc / ${stem}_*.mrc in $INPUT_DIR — skipped."; fi
    done
    [ ${#TOMOS[@]} -gt 0 ] || {
        echo "ERROR: nothing in MB_TOMO_LIST ($TOMO_LIST) matched $INPUT_DIR"; exit 1; }
else
    TOMOS=("$INPUT_DIR"/*.mrc)
    [ -e "${TOMOS[0]}" ] || { echo "ERROR: no .mrc in $INPUT_DIR"; exit 1; }
fi
N=${#TOMOS[@]}
echo "$N tomogram(s) selected  (~$((N * 4)) min with TTA on one GPU)."

ok=0; skipped=0; failed=0; k=0
for TOMO in "${TOMOS[@]}"; do
    k=$((k+1))
    name="$(basename "$TOMO" .mrc)"
    # Already segmented? A score map is the completion marker — except in
    # MB_NO_PROBS=1 mode, which writes none: there the segmentation itself
    # must count, or resume-after-crash re-segments everything from scratch.
    # Globs are '_'-anchored right after the name: a bare ${name}* prefix let
    # Position_1 claim Position_10's outputs and skip forever.
    if [ "${MB_NO_PROBS:-}" = "1" ]; then
        done_test() { [ -f "$OUT_DIR/${name}_segmented.mrc" ] ||
                      ls "$OUT_DIR/${name}"_*_segmented.mrc >/dev/null 2>&1; }
    else
        done_test() { [ -f "$OUT_DIR/${name}_scores.mrc" ] ||
                      ls "$OUT_DIR/${name}"_*_scores.mrc >/dev/null 2>&1; }
    fi
    if [ "$FORCE" != "1" ] && done_test; then
        echo "SKIP: output exists for $name"; skipped=$((skipped+1)); continue
    fi
    echo "[$k/$N] $name  (~$(( (N - k + 1) * 4 )) min remaining)"
    if CUDA_VISIBLE_DEVICES="$GPU" membrain segment \
            --tomogram-path "$TOMO" \
            --ckpt-path "$CKPT" \
            --out-folder "$OUT_DIR" \
            --in-pixel-size "$PIX" \
            "${OPT_FLAGS[@]}" ${MB_EXTRA_ARGS:-}; then
        ok=$((ok+1))
    else
        echo "ERROR: segmentation failed for $name"
        # A partial score map or segmentation left by a mid-write crash would
        # satisfy the skip test forever — remove this tomogram's outputs so
        # the re-run actually re-runs.
        rm -f "$OUT_DIR/${name}_scores.mrc" "$OUT_DIR/${name}"_*_scores.mrc \
              "$OUT_DIR/${name}_segmented.mrc" \
              "$OUT_DIR/${name}"_*_segmented.mrc \
              "$OUT_DIR/${name}_uncertainty.mrc" \
              "$OUT_DIR/${name}"_*_uncertainty.mrc 2>/dev/null || true
        failed=$((failed+1))
    fi
done

# Written only when this run produced something (or confirmed existing output)
# — a zero-output run must not clobber the real record.
if [ $((ok + skipped)) -gt 0 ]; then
    cat > "$OUT_DIR/PROVENANCE.json" <<EOF
{
  "variant": "segmentation",
  "tool": "membrain segment (membrain-seg)",
  "source_dir": "$INPUT_DIR",
  "source_variant": "$SRC_VARIANT",
  "params": {"ckpt": "$CKPT", "in_pixel_size": "$PIX",
             "tta": "$([ "${MB_NO_TTA:-}" = "1" ] && echo off || echo on)",
             "scores": "$([ "${MB_NO_PROBS:-}" = "1" ] && echo off || echo on)"},
  "tomo_list": "$([ -n "$TOMO_LIST" ] && echo "$TOMO_LIST" || echo "ALL")",
  "extraction_allowed": false,
  "note": "segmentations + score maps; coordinates only, never extraction input",
  "date": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF
fi

echo "==================================================================="
echo "done: $ok segmented, $skipped skipped (existing), $failed failed."
echo "Score maps (*_scores.mrc) are in $OUT_DIR — threshold them cheaply with"
echo "the 'Threshold sweep' step; no GPU needed there."
echo "Finished: $(date)"
echo "==================================================================="
[ "$failed" -eq 0 ] || exit 1
