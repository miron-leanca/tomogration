#!/bin/bash
# ml_isonet2_predict_warp_auto.sh
#
# Apply a trained IsoNet 2 model to tomograms (isonet.py predict) — the
# "predict many" half of train-once-predict-many. The model path is a config
# field INDEPENDENT of the data path (a reused config whose data dir silently
# pointed at the wrong dataset caused a real bug once — both resolved paths are
# echoed up front, and the tomogram count is checked against the project's
# tilt-series count so a mismatch is loud, not silent).
#
# OUTPUTS ARE WEDGE-RESTORED / DENOISED: for PICKING COORDINATES ONLY. The
# PROVENANCE.json written here tags them so deconvolution refuses them as input
# and any future extraction path can refuse them as source. Particle extraction
# for averaging must always reference the original reconstruction.
#
# ⚠ LONG FLAGS ONLY: IsoNet 2's python-fire CLI gives -h to --highpassnyquist
# and reuses -p/-s/-d for different things per subcommand.
#
# Usage:
#   bash ml_isonet2_predict_warp_auto.sh <tomo_dir> <output_dir>
#
# <tomo_dir>    tomograms to correct (the ORIGINAL reconstruction)
# <output_dir>  corrected tomograms land here (e.g. membrane/isonet2_corrected)
#
# Knobs via environment (the tomogration GUI sets these):
#   ISO2_MODEL      trained checkpoint .pt (REQUIRED), e.g.
#                   membrane/isonet2/isonet_maps/<name>.pt
#   ISO2_ENV        conda PREFIX of the IsoNet 2 env (see the setup script)
#   ISO2_GPU        --gpuID comma list               (default: 0)
#   ISO2_PIXEL_SIZE star pixel size Å/px, or 'auto'  (default: 12.56)
#   ISO2_TOMO_LIST  space/comma list of stems; empty = ALL tomograms
#   ISO2_INPUT_COL  --input_column                   (default: rlnTomoName)
#                   NOT the tool default. predict defaults to
#                   rlnDeconvTomoName, but the star this wrapper builds holds
#                   RAW tomograms — prepare_star fills the deconv column with
#                   the literal string 'None', and predict then tries to open a
#                   file called 'None':
#                       FileNotFoundError: [Errno 2] … : 'None'
#                   (observed 2026-08-16). rlnTomoName is also what an
#                   isonet2-n2n model trained on: refine reads the raw halves,
#                   deconv/ only feeds mask generation.
#   ISO2_PADDING    --padding_factor, raise against tile seams (default 1.5)
#   ISO2_PREFIX     --output_prefix on each written .mrc
#   ISO2_APPLY_MW   --apply_mw_x1 True|False         (default: tool default True)
#   ISO2_XML_DIR    Warp per-series .xml dir         (default: warp_tiltseries)
#   ISO2_FORCE      1 = redo existing outputs
#
# Flags verified against isonet2_helps.txt captured from IsoNet2 2.0.1b0
# (2026-08-15).
set -e

TOMO_DIR="${1:?Usage: $0 <tomo_dir> <output_dir>}"
OUT_DIR="${2:?Usage: $0 <tomo_dir> <output_dir>}"

ENV_PREFIX="${ISO2_ENV:-/ceph/users/$USER/EMDatasets/processing_scripts/IsoNet2/build/conda_env}"
MODEL="${ISO2_MODEL:-}"
INPUT_COL="${ISO2_INPUT_COL:-rlnTomoName}"
GPU="${ISO2_GPU:-0}"
PIX="${ISO2_PIXEL_SIZE:-12.56}"
XML_DIR="${ISO2_XML_DIR:-warp_tiltseries}"
FORCE="${ISO2_FORCE:-0}"
TOMO_LIST="$(printf '%s' "${ISO2_TOMO_LIST:-}" | tr ',' ' ')"
HELPER_DIR="$(cd "$(dirname "$0")" && pwd)"

[ -d "$TOMO_DIR" ] || { echo "ERROR: tomo dir not found: $TOMO_DIR"; exit 1; }
[ -n "$MODEL" ] || { echo "ERROR: ISO2_MODEL (trained .pt) is required."; exit 1; }
[ -f "$MODEL" ] || { echo "ERROR: model not found: $MODEL"; exit 1; }
case "$MODEL" in
    *.pt) ;;
    *.h5) echo "ERROR: $MODEL is an IsoNet 1 model (.h5). IsoNet 2 predicts from"
          echo "       its own .pt checkpoints — use the 'IsoNet: predict' step"
          echo "       for .h5 models."; exit 1 ;;
    *)    echo "WARNING: $MODEL is not a .pt checkpoint — continuing anyway." ;;
esac
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

# Which columns predict will READ is decided by the model's method, not by us.
# Their predict_row:
#     if network.method in ['regular','isonet2']:
#         tomo_paths = [row.get(input_column) or row.rlnTomoName]
#     else:
#         tomo_paths = [row.rlnTomoReconstructedTomogramHalf1,
#                       row.rlnTomoReconstructedTomogramHalf2]
# So an n2n model reads the HALVES and ignores --input_column entirely, and a
# star built from --full alone gives it two 'None's:
#     FileNotFoundError: [Errno 2] … : 'None'    (observed 2026-08-16)
# IsoNet names its checkpoints network_<method>_<arch>_<cube>_…pt, so the method
# is readable off the file; the pre-flight check below is the real authority.
IS_N2N=0
case "$(basename "$MODEL")" in *n2n*) IS_N2N=1 ;; esac

EVEN_DIR=""; ODD_DIR=""
if [ -n "${ISO2_EVEN_DIR:-}" ] || [ -n "${ISO2_ODD_DIR:-}" ]; then
    [ -n "${ISO2_EVEN_DIR:-}" ] && [ -n "${ISO2_ODD_DIR:-}" ] || {
        echo "ERROR: set BOTH ISO2_EVEN_DIR and ISO2_ODD_DIR, or neither."; exit 1; }
    [ -d "$ISO2_EVEN_DIR" ] || { echo "ERROR: even dir not found: $ISO2_EVEN_DIR"; exit 1; }
    [ -d "$ISO2_ODD_DIR" ]  || { echo "ERROR: odd dir not found: $ISO2_ODD_DIR"; exit 1; }
    EVEN_DIR="$(cd "$ISO2_EVEN_DIR" && pwd)"
    ODD_DIR="$(cd "$ISO2_ODD_DIR" && pwd)"
fi
if [ "$IS_N2N" = "1" ] && [ -z "$EVEN_DIR" ]; then
    echo "ERROR: $(basename "$MODEL") is a noise2noise (n2n) model, and IsoNet 2"
    echo "       predicts with n2n models from the EVEN/ODD HALVES — it reads"
    echo "       rlnTomoReconstructedTomogramHalf1/2 and ignores --input_column."
    echo "       Set ISO2_EVEN_DIR and ISO2_ODD_DIR (e.g. the reconstruct job's"
    echo "       reconstruction/even and reconstruction/odd)."
    echo "       Only series WITH halves can be corrected by this model; to"
    echo "       correct every series from full tomograms alone, train a"
    echo "       single-map model instead (ISO2_METHOD=isonet2)."
    exit 1
fi

ABS_TOMO_DIR="$(cd "$TOMO_DIR" && pwd)"
ABS_MODEL="$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")"
ABS_XML_DIR="$([ -d "$XML_DIR" ] && cd "$XML_DIR" && pwd || true)"
PROJECT_ROOT="$(pwd)"
mkdir -p "$OUT_DIR"

echo "==================================================================="
echo "IsoNet 2 predict  ·  $(date)"
echo "data:   $ABS_TOMO_DIR"
echo "model:  $ABS_MODEL"
echo "output: $OUT_DIR    gpu: $GPU"
echo "env:    $ENV_PREFIX"
echo "==================================================================="

module load miniconda/latest 2>/dev/null || echo "WARNING: could not load miniconda/latest"
command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not on PATH."; exit 1; }
eval "$(conda shell.bash hook)"
[ -x "$ENV_PREFIX/bin/python" ] || {
    echo "ERROR: no IsoNet 2 env at $ENV_PREFIX"
    echo "       Install it first:  bash ml_isonet2_setup.sh"; exit 1; }
conda activate "$ENV_PREFIX"
command -v isonet.py >/dev/null 2>&1 || {
    echo "ERROR: isonet.py not on PATH after activating $ENV_PREFIX"; exit 1; }

# Headless plotting — predict's --save_slices draws through matplotlib, and
# this env ships Qt6/PySide6. Without this the process can die at teardown with
# "ICE default IO error handler doing an exit() … errno = 32" and a non-zero
# status AFTER writing every corrected tomogram (seen in refine, 2026-08-16).
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen

# Select tomograms ('_'-anchored tune list; typo = hard error). Dot-free link
# names for the same reason as training: the names IsoNet sees are our own
# symlinks, so keep Warp's pixel-size dot out of them and the corrected volumes
# inherit the sanitised name with the series prefix intact.
IN="$OUT_DIR/input_tomos"
mkdir -p "$IN"
link_tomo() {                   # $1 = src file, $2 = dest dir
    local b stem safe
    b="$(basename "$1")"
    stem="${b%.mrc}"
    safe="${stem//./p}"
    ln -sf "$1" "$2/$safe.mrc"
}

select_into() {                 # $1 = source dir, $2 = dest dir, $3 = label
    local src="$1" dst="$2" label="$3" stem f n=0
    mkdir -p "$dst"
    rm -f "$dst"/*.mrc 2>/dev/null || true
    if [ -n "$TOMO_LIST" ]; then
        for stem in $TOMO_LIST; do
            for f in "$src/$stem.mrc" "$src/${stem}_"*.mrc; do
                [ -e "$f" ] || continue
                link_tomo "$f" "$dst"; n=$((n+1))
            done
        done
        [ "$n" -gt 0 ] || { echo "ERROR: ISO2_TOMO_LIST ($TOMO_LIST) matched nothing in $src."; exit 1; }
    else
        ls "$src"/*.mrc >/dev/null 2>&1 || { echo "ERROR: no .mrc in $src ($label)"; exit 1; }
        for f in "$src"/*.mrc; do link_tomo "$f" "$dst"; n=$((n+1)); done
    fi
}

select_into "$ABS_TOMO_DIR" "$IN" "full"
if [ -n "$EVEN_DIR" ]; then
    select_into "$EVEN_DIR" "$OUT_DIR/input_even" "even half"
    select_into "$ODD_DIR"  "$OUT_DIR/input_odd"  "odd half"
    N_EVEN=$(ls "$OUT_DIR/input_even"/*.mrc 2>/dev/null | wc -l | tr -d ' ')
    N_ODD=$(ls "$OUT_DIR/input_odd"/*.mrc 2>/dev/null | wc -l | tr -d ' ')
    [ "$N_EVEN" = "$N_ODD" ] || {
        echo "ERROR: $N_EVEN even vs $N_ODD odd tomograms — halves must pair up."; exit 1; }
    echo "$N_EVEN even/odd pair(s) linked — an n2n model corrects from these."
fi
N_SEL="$(ls "$IN" | wc -l | tr -d ' ')"

# Spec §3.2: warn when the job's tomogram count disagrees with what tomogration
# expects (the project's tilt-series count).
N_TS="$(ls "$PROJECT_ROOT"/tomostar/*.tomostar 2>/dev/null | wc -l | tr -d ' ')"
echo "$N_SEL tomogram(s) selected for correction (project has $N_TS tilt series)."
if [ -z "$TOMO_LIST" ] && [ "$N_TS" -gt 0 ] && [ "$N_SEL" -ne "$N_TS" ]; then
    echo "WARNING: correcting $N_SEL tomograms but the project has $N_TS tilt"
    echo "         series — check the data path (a reused config pointing at"
    echo "         the wrong dataset caused exactly this once before)."
fi

# The name a corrected volume ENDS UP with. IsoNet writes
#     <output_prefix>_<method>_<arch>_<the name we linked>.mrc
# which is unusable downstream twice over: the series stem is no longer at the
# front, so '<stem>_*.mrc' globs (MB_TOMO_LIST, the 📂 picker) match nothing;
# and our dot-free link spells the pixel size '12p56Apx', so the variant
# registry cannot read it. Everything after this step matches on the series
# stem and reads Å/px out of the filename, so the outputs are renamed back to
# the ORIGINAL series name with an _isonet2 tag.
orig_of_link() {                # $1 = input_tomos/<safe>.mrc -> original stem
    local tgt
    tgt="$(readlink -f "$1" 2>/dev/null || true)"
    [ -n "$tgt" ] || tgt="$1"
    basename "$tgt" .mrc
}

# Skip logic: done = every selected tomogram already has its corrected output,
# tested against the FINAL name (the rename below), not IsoNet's interim one.
CORR="$OUT_DIR/corrected"
if [ "$FORCE" != "1" ] && [ -d "$CORR" ]; then
    missing=0
    for f in "$IN"/*.mrc; do
        orig="$(orig_of_link "$f")"
        [ -e "$CORR/${orig}_isonet2.mrc" ] || missing=$((missing+1))
    done
    if [ "$missing" -eq 0 ]; then
        echo "SKIP: all $N_SEL tomogram(s) already corrected in $CORR"
        exit 0
    fi
fi

# A star over the selected set (predict reads tomograms via the star). The
# defocus matters here too: an isonet2 model with CTF handling reads it from
# these rows, so inject per series rather than leaving prepare_star's placeholder.
cd "$OUT_DIR"
PS_FLAGS=(--star_name predict.star --pixel_size "$PIX" --cs "${ISO2_CS:-2.7}"
          --voltage "${ISO2_KV:-300}" --ac "${ISO2_AC:-0.1}")
if [ -n "$EVEN_DIR" ]; then
    PS_FLAGS+=(--even input_even --odd input_odd)
else
    PS_FLAGS+=(--full input_tomos)
fi
isonet.py prepare_star "${PS_FLAGS[@]}"
[ -f predict.star ] || { echo "ERROR: prepare_star wrote no predict.star."; exit 1; }
if [ -n "$ABS_XML_DIR" ]; then
    python3 "$HELPER_DIR/ml_isonet_star_defocus.py" predict.star "$ABS_XML_DIR"
else
    echo "WARNING: no $XML_DIR — defocus stays at prepare_star's placeholder."
fi

# The column predict will read must name real files. prepare_star writes the
# STRING 'None' into columns it has nothing for, so a wrong column here does not
# fail as a missing column — it fails, minutes later, trying to open a file
# called 'None'. Check it now, against the star we just built.
CHECK_COLS="$INPUT_COL"
[ "$IS_N2N" = "1" ] && CHECK_COLS="rlnTomoReconstructedTomogramHalf1 rlnTomoReconstructedTomogramHalf2"
python3 - $CHECK_COLS <<'PYEOF' || exit 1
import re, sys
lines = open("predict.star").read().splitlines()
hdr, ncol = {}, 0
for ln in lines:
    m = re.match(r"\s*(_rln\w+)\s+#(\d+)", ln)
    if m:
        hdr[m.group(1)] = int(m.group(2)) - 1
        ncol = max(ncol, int(m.group(2)))
for name in sys.argv[1:]:
    col = "_" + name.lstrip("_")
    if col not in hdr:
        sys.exit(f"ERROR: predict.star has no {col} column.\n"
                 f"       It has: {' '.join(sorted(hdr))}")
    bad = 0
    for ln in lines:
        f = ln.split()
        if len(f) >= ncol and not ln.lstrip().startswith(("_", "#", "data_", "loop_")):
            if f[hdr[col]] in ("None", "none", ""):
                bad += 1
    if bad:
        sys.exit(f"ERROR: {bad} row(s) of {col} in predict.star hold 'None', "
                 f"not a path.\n"
                 f"       prepare_star fills columns it has no data for with "
                 f"the literal string 'None', and predict opens whatever is "
                 f"there — as a filename.\n"
                 f"       Half columns empty => this model needs "
                 f"ISO2_EVEN_DIR/ISO2_ODD_DIR. rlnDeconvTomoName empty => use "
                 f"ISO2_INPUT_COL=rlnTomoName (this wrapper never deconvolves).")
    print(f"predict will read {col}: paths present in every row.")
PYEOF

PR_FLAGS=(--output_dir corrected --gpuID "$GPU" --input_column "$INPUT_COL")
[ -n "${ISO2_PADDING:-}" ]   && PR_FLAGS+=(--padding_factor "$ISO2_PADDING")
[ -n "${ISO2_PREFIX:-}" ]    && PR_FLAGS+=(--output_prefix "$ISO2_PREFIX")
[ -n "${ISO2_APPLY_MW:-}" ]  && PR_FLAGS+=(--apply_mw_x1 "$ISO2_APPLY_MW")
# Same rule as training: the exit status is not the evidence, the output is.
RC=0
isonet.py predict predict.star "$ABS_MODEL" "${PR_FLAGS[@]}" || RC=$?

# Restore series-leading names before anything downstream sees this folder.
# Matched by the LONGEST link stem the corrected name ends with, the same
# '_'-safe rule the defocus injector uses, so Position1 never claims
# Position10's output.
RENAMED=0
for out in corrected/*.mrc; do
    [ -e "$out" ] || continue
    ob="$(basename "$out" .mrc)"
    best=""; best_link=""
    for link in input_tomos/*.mrc; do
        [ -e "$link" ] || continue
        s="$(basename "$link" .mrc)"
        case "$ob" in
            *"$s") [ "${#s}" -gt "${#best}" ] && { best="$s"; best_link="$link"; } ;;
        esac
    done
    [ -n "$best" ] || continue
    orig="$(orig_of_link "$best_link")"
    new="corrected/${orig}_isonet2.mrc"
    [ "$out" = "$new" ] && continue
    if [ -e "$new" ]; then
        echo "WARN: $new already exists — leaving $(basename "$out") as it is."
        continue
    fi
    mv "$out" "$new" && RENAMED=$((RENAMED+1))
done
[ "$RENAMED" = "0" ] || echo "renamed $RENAMED output(s) to <series>_isonet2.mrc " \
    "(IsoNet writes <method>_<arch>_<link> first, which no downstream stem match can read)."

N_OUT="$(ls corrected/*.mrc 2>/dev/null | wc -l | tr -d ' ')"
if [ "$RC" != "0" ] && [ "$N_OUT" -gt 0 ]; then
    echo "NOTE: predict exited $RC but wrote $N_OUT corrected tomogram(s) —"
    echo "      usually the matplotlib/X11 teardown after the last slice is"
    echo "      saved. Check the log for a real traceback before trusting them."
fi
# Provenance is written ONLY when there is real output — a failed run must not
# leave a wedge_restored tag on an empty folder.
[ "$N_OUT" -gt 0 ] || { echo "ERROR: predict produced no output."; exit 1; }
cat > PROVENANCE.json <<EOF
{
  "variant": "wedge_restored",
  "tool": "IsoNet2 2.0.1b0 predict",
  "source_dir": "$ABS_TOMO_DIR",
  "model": "$ABS_MODEL",
  "params": {"pixel_size": "$PIX", "padding": "${ISO2_PADDING:-default}",
             "input_column": "${ISO2_INPUT_COL:-default}",
             "apply_mw_x1": "${ISO2_APPLY_MW:-default}"},
  "tomo_list": "$([ -n "$TOMO_LIST" ] && echo "$TOMO_LIST" || echo "ALL")",
  "extraction_allowed": false,
  "note": "missing-wedge restored / denoised: PICKING COORDINATES ONLY — never extraction input",
  "date": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF

echo "==================================================================="
echo "IsoNet 2 predict done: $N_OUT corrected tomogram(s) in $OUT_DIR/corrected"
echo "PICKING/SEGMENTATION ONLY — extraction must use the original reconstruction."
echo "Finished: $(date)"
echo "==================================================================="
