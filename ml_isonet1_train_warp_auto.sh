#!/bin/bash
# ml_isonet1_train_warp_auto.sh
#
# IsoNet 1 TRAINING chain: prepare_star -> (deconv) -> make_mask -> extract ->
# refine, in one resumable run. Train on 1–5 tomograms (IsoNet's own docs:
# "Usually 1-5 tomograms are sufficient"), then apply the trained model to the
# whole dataset with ml_isonet1_predict_warp_auto.sh — train once, predict many.
#
# IsoNet is a CLUSTER MODULE here (module load isonet), not a conda env.
# All IsoNet outputs live inside <work_dir> (star, deconv/, mask/, subtomo/,
# results/), so the run is self-contained and versionable.
#
# Per-tomogram defocus: prepare_star writes ONE --defocus into every row, but
# defocus is per tilt series. After prepare_star this wrapper rewrites the
# star's _rlnDefocus per row from warp_tiltseries/<series>.xml (µm ×10000 → Å)
# via ml_isonet_star_defocus.py — echoed per series, never guessed.
#
# Usage:
#   bash ml_isonet1_train_warp_auto.sh <tomo_dir> <work_dir>
#
# <tomo_dir>   reconstructed tomograms (the ORIGINAL reconstruction — IsoNet
#              has its own CTF deconvolution; do not feed it pre-deconvolved
#              volumes)
# <work_dir>   IsoNet working dir (created; e.g. membrane/isonet)
#
# Knobs via environment (the tomogration GUI sets these):
#   ISO_MODULE       lmod module                        (default: isonet/0.3)
#   ISO_TOMO_LIST    space/comma list of series stems to TRAIN on. Strongly
#                    recommended (1–5). Empty = every tomogram in tomo_dir.
#   ISO_GPU          --gpuID comma list for refine      (default: 0)
#   ISO_PIXEL_SIZE   star pixel size Å/px               (default: 12.56)
#   ISO_XML_DIR      Warp per-series .xml dir           (default: warp_tiltseries)
#   ISO_SUBTOMOS     subtomograms per tomogram          (default: 100)
#   ISO_CUBE         --cube_size (divisible by 8)       (default: 64)
#   ISO_ITER         refine --iterations                (default: 30)
#   ISO_NO_DECONV    1 = skip IsoNet's CTF deconvolution step
#   ISO_SNRFALLOFF   deconv --snrfalloff                (default: tool default 1.0)
#   ISO_DECONVSTRENGTH deconv --deconvstrength          (default: tool default 1.0)
#   ISO_DENSITY_PCT  make_mask --density_percentage     (default: tool default 50)
#   ISO_STD_PCT      make_mask --std_percentage         (default: tool default 50)
#   ISO_ZCROP        make_mask --z_crop, e.g. 0.2       (default: unset)
#   ISO_PRETRAINED   refine --pretrained_model .h5 to start from (default: none)
#   ISO_CONTINUE     refine --continue_from <json> to resume an interrupted run
#   ISO_NCPU         cpu count for deconv/preprocessing (default: 8)
#   ISO_FORCE        1 = redo steps whose output already exists
#   ISO_EXTRA_REFINE appended verbatim to refine
#
# Two IsoNet 0.3 behaviours this wrapper works around (both hit on 2026-08-15):
#   * a DOT in a tomogram filename (Warp's _12.56Apx tag) is parsed
#     inconsistently inside the tool — the training set is linked dot-free.
#   * refine catches its own exceptions, logs the traceback and EXITS 0. The
#     final iteration's model file is checked explicitly; missing = exit 1.
#
# Every step skips itself if its output already exists (resume-friendly);
# ISO_FORCE=1 redoes. Flags verified against isonet.py 0.3 -h (2026-08-15).
set -e

TOMO_DIR="${1:?Usage: $0 <tomo_dir> <work_dir>}"
WORK_DIR="${2:?Usage: $0 <tomo_dir> <work_dir>}"

ISO_MOD="${ISO_MODULE:-isonet/0.3}"
GPU="${ISO_GPU:-0}"
PIX="${ISO_PIXEL_SIZE:-12.56}"
XML_DIR="${ISO_XML_DIR:-warp_tiltseries}"
NSUB="${ISO_SUBTOMOS:-100}"
CUBE="${ISO_CUBE:-64}"
ITER="${ISO_ITER:-30}"
NCPU="${ISO_NCPU:-8}"
FORCE="${ISO_FORCE:-0}"
TOMO_LIST="$(printf '%s' "${ISO_TOMO_LIST:-}" | tr ',' ' ')"
HELPER_DIR="$(cd "$(dirname "$0")" && pwd)"

[ -d "$TOMO_DIR" ] || { echo "ERROR: tomo dir not found: $TOMO_DIR"; exit 1; }
[ -d "$XML_DIR" ]  || { echo "ERROR: xml dir not found: $XML_DIR"; exit 1; }
# IsoNet deconvolves internally — refuse tagged pre-processed inputs outright.
if [ -f "$TOMO_DIR/PROVENANCE.json" ]; then
    if grep -Eqi '"variant"[^,}]*(denoise|wedge|isonet|deconv)' "$TOMO_DIR/PROVENANCE.json"; then
        echo "ERROR: $TOMO_DIR is a processed variant (see its PROVENANCE.json)."
        echo "IsoNet must train on the ORIGINAL reconstruction — it deconvolves itself."
        exit 1
    fi
fi

ABS_TOMO_DIR="$(cd "$TOMO_DIR" && pwd)"
ABS_XML_DIR="$(cd "$XML_DIR" && pwd)"
mkdir -p "$WORK_DIR"

echo "==================================================================="
echo "IsoNet 1 training chain  ·  $(date)"
echo "tomograms: $TOMO_DIR    work dir: $WORK_DIR"
echo "mode: $([ -n "$TOMO_LIST" ] && echo "TRAIN ON ($TOMO_LIST)" || echo "ALL tomograms — consider 1–5 (ISO_TOMO_LIST)")"
echo "gpu: $GPU   pixel: $PIX Å   subtomos: $NSUB   cube: $CUBE   iters: $ITER"
echo "==================================================================="

module load "$ISO_MOD" 2>/dev/null || { echo "ERROR: could not module load $ISO_MOD"; exit 1; }
command -v isonet.py >/dev/null 2>&1 || { echo "ERROR: isonet.py not on PATH after $ISO_MOD"; exit 1; }

# The training set: link the chosen tomograms into work_dir/input_tomos so
# prepare_star sees exactly that folder ('_'-anchored, typo = hard error).
IN="$WORK_DIR/input_tomos"
mkdir -p "$IN"

# HARD REQUIREMENT: the linked name must contain NO dot before .mrc.
# IsoNet 0.3 derives the per-iteration volume name from the star's micrograph
# name with split('.')[0] in one place and splitext() in another, so Warp's
# pixel-size tag makes the two disagree:
#     Position003_12.56Apx.mrc  ->  writes ..._12.56Apx_iter00.mrc
#                               ->  reads  ..._12_iter00.mrc   (FileNotFound)
# refine then dies inside iteration 1 (observed 2026-08-15). We own these
# names — they are our symlinks — so link dot-free and the whole chain agrees.
# The series prefix is untouched, so ISO_TOMO_LIST stems and the defocus
# injector's '_'-anchored matching still work.
link_tomo() {
    local b stem safe
    b="$(basename "$1")"
    stem="${b%.mrc}"
    safe="${stem//./p}"
    ln -sf "$1" "$IN/$safe.mrc"
    [ "$safe" = "$stem" ] || echo "  linked $stem -> $safe (IsoNet mis-parses dots)"
}

# Clear stale links first: a changed ISO_TOMO_LIST must not leave the previous
# run's tomograms in the training set (with ISO_FORCE=1 the regenerated star
# would silently include both).
rm -f "$IN"/*.mrc 2>/dev/null || true
if [ -n "$TOMO_LIST" ]; then
    n=0
    for stem in $TOMO_LIST; do
        found=0
        for f in "$ABS_TOMO_DIR/$stem.mrc" "$ABS_TOMO_DIR/${stem}_"*.mrc; do
            [ -e "$f" ] || continue
            link_tomo "$f"
            found=1; n=$((n+1))
        done
        [ "$found" = "1" ] || echo "WARN: no ${stem}.mrc / ${stem}_*.mrc in $TOMO_DIR"
    done
    [ "$n" -gt 0 ] || { echo "ERROR: ISO_TOMO_LIST ($TOMO_LIST) matched nothing."; exit 1; }
else
    ls "$ABS_TOMO_DIR"/*.mrc >/dev/null 2>&1 || { echo "ERROR: no .mrc in $TOMO_DIR"; exit 1; }
    for f in "$ABS_TOMO_DIR"/*.mrc; do link_tomo "$f"; done
fi
echo "$(ls "$IN" | wc -l | tr -d ' ') tomogram(s) linked for training."

cd "$WORK_DIR"
STAR="tomograms.star"

# Resume tests, applied per STEP. "The folder has files in it" was the old test
# and it is not the same question: a work dir left by a different tomo list (or
# by the pre-2026-08-15 dotted names) is full of files that belong to OTHER
# tomograms, so every step skipped itself and the chain marched on to fail
# deeper in. A step is done only when EVERY currently-linked tomogram has its
# output there.  (Prefix match: IsoNet writes <name>.mrc here, <name>_mask.mrc
# there.)  Paths are relative — we have cd'd into the work dir.
have_all() {                    # $1 = folder that must hold one file per tomo
    local d="$1" f b
    [ -d "$d" ] || return 1
    for f in input_tomos/*.mrc; do
        b="$(basename "$f" .mrc)"
        ls "$d/$b"*.mrc >/dev/null 2>&1 || return 1
    done
    return 0
}
covers_all() {                  # $1 = star that must name every tomo
    local f b
    [ -f "$1" ] || return 1
    for f in input_tomos/*.mrc; do
        b="$(basename "$f" .mrc)"
        grep -q -- "$b" "$1" || return 1
    done
    return 0
}

# The other half of a correct resume: the SETTINGS a folder was computed with.
# have_all only knows which tomograms are covered — deconv/ holding all five at
# strength 1.0 looks identical to all five at 1.3, so lowering the strength and
# re-running would skip deconv and train the model on the OLD volumes. Nothing
# in the log would say so. Each folder carries a stamp of the values that made
# it, checked before it is reused.
DECONV_STAMP="strength=${ISO_DECONVSTRENGTH:-default} falloff=${ISO_SNRFALLOFF:-default}"
MASK_STAMP="density=${ISO_DENSITY_PCT:-default} std=${ISO_STD_PCT:-default} zcrop=${ISO_ZCROP:-default}"
SUBTOMO_STAMP="cube=$CUBE subtomos=$NSUB"
STAMP_FILE=".tomogration_settings"

stamp_check() {                 # $1 = folder (or '.'), $2 = the stamp now
    local f="$1/$STAMP_FILE" old
    [ -f "$f" ] || return 0     # written by an older run: nothing to compare
    old="$(cat "$f")"
    [ "$old" = "$2" ] && return 0
    echo "ERROR: $WORK_DIR/$1 was computed with different settings."
    echo "         then: $old"
    echo "          now: $2"
    echo "       These decide what the model TRAINS ON, so resuming here would"
    echo "       train on the OLD ones while the log showed your new values."
    echo "       Either ISO_FORCE=1 (redo the chain and retrain), or point the"
    echo "       work folder somewhere fresh to keep both variants side by side."
    exit 1
}
stamp_write() { printf '%s\n' "$2" > "$1/$STAMP_FILE"; }

# 1. prepare_star (+ per-series defocus injection)
if [ -f "$STAR" ] && [ "$FORCE" != "1" ]; then
    echo "SKIP: $STAR exists (resume). "
    N_LINKED="$(ls input_tomos/*.mrc 2>/dev/null | wc -l | tr -d ' ')"
    N_ROWS="$(grep -c 'input_tomos/' "$STAR" 2>/dev/null || true)"
    # Rows pointing at files that are no longer linked are FATAL, not a warning:
    # the star wins on resume, so IsoNet would read a path that does not exist
    # and fail several minutes in. This fires when the tomo list changed, and
    # for every star written before the dot-free linking above.
    GONE=0
    while read -r ref; do
        [ -z "$ref" ] && continue
        [ -e "$ref" ] || { echo "         missing: $ref"; GONE=$((GONE+1)); }
    done < <(grep -o 'input_tomos/[^[:space:]]*\.mrc' "$STAR" | sort -u)
    if [ "$GONE" -gt 0 ]; then
        echo "ERROR: $STAR references $GONE tomogram(s) that are not linked now."
        echo "       The star is from an earlier run with a different tomo list,"
        echo "       or from before the dot-free linking (names ending .56Apx)."
        echo "       Fix:  rm $WORK_DIR/$STAR     and run this step again."
        echo "       (deconv/ mask/ subtomo.star then recompute whatever the new"
        echo "       selection is missing; ISO_FORCE=1 redoes all of it instead.)"
        exit 1
    fi
    if [ "$N_LINKED" != "$N_ROWS" ]; then
        echo "WARNING: the existing star lists $N_ROWS tomogram(s) but the tomo"
        echo "         list now selects $N_LINKED — the star WINS on resume."
        echo "         Delete $WORK_DIR/$STAR (or ISO_FORCE=1) to re-prepare."
    fi
else
    isonet.py prepare_star input_tomos --output_star "$STAR" \
        --pixel_size "$PIX" --number_subtomos "$NSUB"
    python3 "$HELPER_DIR/ml_isonet_star_defocus.py" "$STAR" "$ABS_XML_DIR"
fi

# 2. IsoNet's own CTF deconvolution (recommended; skip for phase-plate data)
if [ "${ISO_NO_DECONV:-}" != "1" ]; then
    if have_all deconv && [ "$FORCE" != "1" ]; then
        stamp_check deconv "$DECONV_STAMP"
        echo "SKIP: deconv/ already covers every selected tomogram"
    else
        DC_FLAGS=(--deconv_folder deconv --ncpu "$NCPU")
        [ -n "${ISO_SNRFALLOFF:-}" ]     && DC_FLAGS+=(--snrfalloff "$ISO_SNRFALLOFF")
        [ -n "${ISO_DECONVSTRENGTH:-}" ] && DC_FLAGS+=(--deconvstrength "$ISO_DECONVSTRENGTH")
        isonet.py deconv "$STAR" "${DC_FLAGS[@]}"
        stamp_write deconv "$DECONV_STAMP"
    fi
fi

# 3. make_mask (uses the deconv tomo automatically when present in the star)
if have_all mask && [ "$FORCE" != "1" ]; then
    stamp_check mask "$MASK_STAMP"
    echo "SKIP: mask/ already covers every selected tomogram"
else
    MM_FLAGS=(--mask_folder mask)
    [ -n "${ISO_DENSITY_PCT:-}" ] && MM_FLAGS+=(--density_percentage "$ISO_DENSITY_PCT")
    [ -n "${ISO_STD_PCT:-}" ]     && MM_FLAGS+=(--std_percentage "$ISO_STD_PCT")
    [ -n "${ISO_ZCROP:-}" ]       && MM_FLAGS+=(--z_crop "$ISO_ZCROP")
    isonet.py make_mask "$STAR" "${MM_FLAGS[@]}"
    stamp_write mask "$MASK_STAMP"
fi

# 4. extract subtomograms
if covers_all subtomo.star && [ "$FORCE" != "1" ]; then
    stamp_check subtomo "$SUBTOMO_STAMP"
    echo "SKIP: subtomo.star already covers every selected tomogram"
else
    isonet.py extract "$STAR" --subtomo_folder subtomo --subtomo_star subtomo.star \
        --cube_size "$CUBE"
    stamp_write subtomo "$SUBTOMO_STAMP"
fi

# 5. refine (the GPU training loop; hours)
RF_FLAGS=(--gpuID "$GPU" --iterations "$ITER" --result_dir results)
[ -n "${ISO_PRETRAINED:-}" ] && RF_FLAGS+=(--pretrained_model "$ISO_PRETRAINED")
[ -n "${ISO_CONTINUE:-}" ]   && RF_FLAGS+=(--continue_from "$ISO_CONTINUE")
# "results/ has a model in it" is NOT the same question as "is it trained".
# IsoNet writes model_iter00.h5 (the untrained initial weights) before the first
# iteration, so a crashed run leaves one behind — and this skip then declared the
# training done and exited 0 without training anything (seen 2026-08-15, twice).
# Only the FINAL iteration's model means done.
FINAL_MODEL="results/model_iter$(printf '%02d' "$ITER").h5"
RAN_REFINE=0
if [ -f "$FINAL_MODEL" ] && [ "$FORCE" != "1" ] && [ -z "${ISO_CONTINUE:-}" ]; then
    echo "SKIP: $FINAL_MODEL is already trained (ISO_FORCE=1 retrains)."
else
    if ls results/model_iter*.h5 >/dev/null 2>&1 && [ "$FORCE" != "1" ] \
            && [ -z "${ISO_CONTINUE:-}" ]; then
        echo "results/ holds model(s) but NOT $FINAL_MODEL — that training never"
        echo "finished. Restarting refine from scratch; to resume the interrupted"
        echo "run instead, stop and set ISO_CONTINUE to its results/*.json."
    fi
    isonet.py refine subtomo.star "${RF_FLAGS[@]}" ${ISO_EXTRA_REFINE:-}
    RAN_REFINE=1
fi

# Did refine actually TRAIN? `set -e` cannot answer that: IsoNet catches its own
# exceptions, logs the traceback at ERROR level and still exits 0. A run that
# died inside iteration 1 therefore looked identical to a finished one, right
# down to a model file (observed 2026-08-15). The final iteration's model is the
# only honest evidence.
if [ "$RAN_REFINE" = "1" ] && [ ! -f "$FINAL_MODEL" ]; then
    echo "==================================================================="
    echo "ERROR: refine did not reach iteration $ITER — $FINAL_MODEL missing."
    echo "       IsoNet exits 0 even when it crashes, so the traceback in the"
    echo "       log above IS the failure. Models actually written:"
    ls -t results/model_iter*.h5 2>/dev/null | head -5 | sed 's/^/         /' \
        || echo "         (none)"
    echo "       A \"No such file or directory: 'results/<name>_iterNN.mrc'\""
    echo "       traceback is the dot-in-the-filename bug — this wrapper now"
    echo "       links tomograms dot-free, so delete $WORK_DIR/$STAR and re-run."
    echo "       No PROVENANCE.json is written: nothing here is a usable model."
    echo "==================================================================="
    exit 1
fi

# Report the FINAL model, not the newest by mtime — a stub written after a
# restart would otherwise be the one the log points the predict step at.
if [ -f "$FINAL_MODEL" ]; then
    LAST_MODEL="$FINAL_MODEL"
else
    LAST_MODEL="$(ls -t results/model_iter*.h5 2>/dev/null | head -1 || true)"
fi
cat > PROVENANCE.json <<EOF
{
  "variant": "isonet_training",
  "tool": "isonet.py 0.3 (prepare_star/deconv/make_mask/extract/refine)",
  "source_dir": "$ABS_TOMO_DIR",
  "params": {"pixel_size": "$PIX", "subtomos": "$NSUB", "cube": "$CUBE",
             "iterations": "$ITER",
             "deconv": "$([ "${ISO_NO_DECONV:-}" = "1" ] && echo off || echo on)"},
  "tomo_list": "$([ -n "$TOMO_LIST" ] && echo "$TOMO_LIST" || echo "ALL")",
  "extraction_allowed": false,
  "date": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF

echo "==================================================================="
echo "IsoNet training chain done."
if [ -n "$LAST_MODEL" ]; then
    echo "Newest model: $WORK_DIR/$LAST_MODEL"
    echo "Apply it to the WHOLE dataset with the 'IsoNet: predict' step"
    echo "(the model is reusable — likely across EML45/EML46 too)."
else
    echo "WARNING: no results/model_iter*.h5 found — check the refine log above."
fi
echo "Finished: $(date)"
echo "==================================================================="
[ -n "$LAST_MODEL" ] || exit 1
