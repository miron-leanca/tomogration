#!/bin/bash
# ml_isonet2_train_warp_auto.sh
#
# IsoNet 2 TRAINING chain: prepare_star -> (deconv) -> make_mask -> refine,
# in one resumable run. Train on 1–5 tomograms ("Usually 1-5 tomograms are
# sufficient" — their own prepare_star help), then apply the model to the whole
# dataset with ml_isonet2_predict_warp_auto.sh.
#
# IsoNet 2 is NOT the isonet/0.3 cluster module. It is a conda PREFIX env built
# by ml_isonet2_setup.sh at <IsoNet2>/build/conda_env — there is no named env,
# so `conda activate -n isonet2_environment` finds nothing. Both live side by
# side: IsoNet 1 = module, IsoNet 2 = prefix env, neither disturbs the other.
#
# What changed from IsoNet 1 (isonet.py 0.3):
#   * NO extract step — subtomogram extraction folded into refine.
#   * refine trains EPOCHS (default 50), not iterations, and writes .pt
#     checkpoints (not .h5).
#   * refine --method picks isonet2 (single map) or isonet2-n2n (noise2noise
#     on even/odd halves). 'auto' reads the star's own columns — but it can
#     only do that when the star is UNAMBIGUOUS. With halves we also ask for
#     --create_average, which adds averaged full volumes for deconv and
#     make_mask to work on, and refine then sees both kinds and refuses:
#         "Both full and half tomograms are present in the star file.
#          Please specify method as either 'isonet2' or 'isonet2-n2n'."
#     (observed 2026-08-15, four minutes into a run). So when halves are
#     given this wrapper resolves auto -> isonet2-n2n itself and says so.
#   * The chain is STAR-DRIVEN: each step writes its result back into
#     tomograms.star (rlnDeconvTomoName, rlnMaskName), so the star — not a
#     folder convention — is the state of the run.
#
# ⚠ LONG FLAGS ONLY, everywhere below. IsoNet 2's CLI is python-fire and its
# short flags collide across subcommands and with help:
#     -h  is --highpassnyquist in denoise/deconv/refine, NOT help
#     -p  is --pixel_size / --phaseflipped / --patch_size / --padding_factor
#     -d  is --defocus / --deconvstrength / --density_percentage
#     -s  is --star_name / --snrfalloff / --std_percentage / --save_slices
# A short flag here would silently mean something else in the next subcommand.
#
# Usage:
#   bash ml_isonet2_train_warp_auto.sh <tomo_dir> <work_dir>
#
# <tomo_dir>   reconstructed tomograms (the ORIGINAL reconstruction — IsoNet
#              deconvolves internally; do not feed pre-deconvolved volumes)
# <work_dir>   IsoNet 2 project dir (created; e.g. membrane/isonet2)
#
# Knobs via environment (the tomogration GUI sets these):
#   ISO2_ENV        conda PREFIX of the IsoNet 2 env
#                   (default: /ceph/users/$USER/EMDatasets/processing_scripts/
#                    IsoNet2/build/conda_env)
#   ISO2_TOMO_LIST  space/comma list of series stems to TRAIN on (1–5).
#                   Empty = every tomogram in tomo_dir.
#   ISO2_EVEN_DIR   even-half tomograms  ┐ both set = noise2noise (isonet2-n2n)
#   ISO2_ODD_DIR    odd-half tomograms   ┘ via WarpTools --halfmap_frames
#   ISO2_GPU        --gpuID comma list                 (default: 0)
#   ISO2_PIXEL_SIZE --pixel_size Å/px, or 'auto'       (default: 12.56)
#   ISO2_XML_DIR    Warp per-series .xml dir           (default: warp_tiltseries)
#   ISO2_SUBTOMOS   --number_subtomos, or 'auto'       (default: auto)
#   ISO2_CUBE       --cube_size                        (default: tool default 96)
#   ISO2_EPOCHS     refine --epochs                    (default: tool default 50)
#   ISO2_ARCH       --arch unet-small|unet-medium|unet-large|scunet-fast
#   ISO2_BATCH      --batch_size                       (default: auto = 2×GPUs)
#   ISO2_METHOD     refine --method auto|isonet2|isonet2-n2n  (default: auto)
#   ISO2_CTF_MODE   --CTF_mode None|phase_only|network|wiener (default: None)
#   ISO2_LOSS       --loss_func L2|Huber|L1|FSC        (default: tool default L2)
#   ISO2_BFACTOR    --bfactor (0 for cellular; 200–300 isolated samples)
#   ISO2_NO_DECONV  1 = skip the deconv step (phase-plate data, or CTF_mode
#                   network/wiener which corrects inside the network instead)
#   ISO2_SNRFALLOFF   deconv --snrfalloff              (default: tool default 1)
#   ISO2_DECONVSTRENGTH deconv --deconvstrength        (default: tool default 1)
#   ISO2_DENSITY_PCT  make_mask --density_percentage   (default: tool default 50)
#   ISO2_STD_PCT      make_mask --std_percentage       (default: tool default 50)
#   ISO2_ZCROP        make_mask --z_crop               (default: tool default 0.2)
#   ISO2_PRETRAINED   refine --pretrained_model <.pt> to fine-tune from
#   ISO2_NCPUS        --ncpus                          (default: 16)
#   ISO2_TILT_MIN / ISO2_TILT_MAX  prepare_star tilt range (default: ∓60)
#   ISO2_KV / ISO2_CS / ISO2_AC    scope constants (default: 300 / 2.7 / 0.1)
#   ISO2_FORCE      1 = redo steps whose output already exists
#   ISO2_EXTRA_REFINE  appended verbatim to refine
#
# Every step skips itself if its output already covers the selected tomograms
# (resume-friendly); ISO2_FORCE=1 redoes. Flags verified against
# isonet2_helps.txt captured from IsoNet2 2.0.1b0 on 2026-08-15.
set -e

TOMO_DIR="${1:?Usage: $0 <tomo_dir> <work_dir>}"
WORK_DIR="${2:?Usage: $0 <tomo_dir> <work_dir>}"

ENV_PREFIX="${ISO2_ENV:-/ceph/users/$USER/EMDatasets/processing_scripts/IsoNet2/build/conda_env}"
GPU="${ISO2_GPU:-0}"
PIX="${ISO2_PIXEL_SIZE:-12.56}"
XML_DIR="${ISO2_XML_DIR:-warp_tiltseries}"
NSUB="${ISO2_SUBTOMOS:-auto}"
NCPUS="${ISO2_NCPUS:-16}"
METHOD="${ISO2_METHOD:-auto}"
FORCE="${ISO2_FORCE:-0}"
TOMO_LIST="$(printf '%s' "${ISO2_TOMO_LIST:-}" | tr ',' ' ')"
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
# Halves are optional; resolve them only if BOTH are given (one alone is a
# half-configured n2n run, which prepare_star would accept and quietly ignore).
EVEN_DIR=""; ODD_DIR=""
if [ -n "${ISO2_EVEN_DIR:-}" ] || [ -n "${ISO2_ODD_DIR:-}" ]; then
    [ -n "${ISO2_EVEN_DIR:-}" ] && [ -n "${ISO2_ODD_DIR:-}" ] || {
        echo "ERROR: set BOTH ISO2_EVEN_DIR and ISO2_ODD_DIR for noise2noise,"
        echo "       or neither for single-map training."; exit 1; }
    [ -d "$ISO2_EVEN_DIR" ] || { echo "ERROR: even dir not found: $ISO2_EVEN_DIR"; exit 1; }
    [ -d "$ISO2_ODD_DIR" ]  || { echo "ERROR: odd dir not found: $ISO2_ODD_DIR"; exit 1; }
    EVEN_DIR="$(cd "$ISO2_EVEN_DIR" && pwd)"
    ODD_DIR="$(cd "$ISO2_ODD_DIR" && pwd)"
fi
mkdir -p "$WORK_DIR"

# Resolve 'auto' HERE rather than leaving it to refine. With halves the star
# carries both the pairs and the averaged full volumes, and refine will not
# choose between them — it raises ValueError after deconv and make_mask have
# already run (~4 minutes in, or far longer on a real dataset). Halves are only
# ever supplied deliberately, and their docs call n2n the better denoiser, so
# that is the honest resolution of 'auto'.
METHOD_WHY=""
if [ -n "$EVEN_DIR" ] && [ "$METHOD" = "auto" ]; then
    METHOD="isonet2-n2n"
    METHOD_WHY="  (auto -> isonet2-n2n: halves given; refine will not pick between"
    METHOD_WHY="$METHOD_WHY the pairs and the averaged full volumes)"
fi

echo "==================================================================="
echo "IsoNet 2 training chain  ·  $(date)"
echo "tomograms: $TOMO_DIR    work dir: $WORK_DIR"
echo "mode: $([ -n "$TOMO_LIST" ] && echo "TRAIN ON ($TOMO_LIST)" || echo "ALL tomograms — consider 1–5 (ISO2_TOMO_LIST)")"
echo "pairing: $([ -n "$EVEN_DIR" ] && echo "EVEN/ODD halves — noise2noise available" || echo "single map (no halves)")"
echo "gpu: $GPU   pixel: $PIX   subtomos: $NSUB   epochs: ${ISO2_EPOCHS:-tool default}"
echo "method: $METHOD$METHOD_WHY"
echo "env: $ENV_PREFIX"
echo "==================================================================="

# ---- the env (a PREFIX; never `source activate`, never -n) -----------------
module load miniconda/latest 2>/dev/null || echo "WARNING: could not load miniconda/latest"
command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not on PATH."; exit 1; }
eval "$(conda shell.bash hook)"
[ -x "$ENV_PREFIX/bin/python" ] || {
    echo "ERROR: no IsoNet 2 env at $ENV_PREFIX"
    echo "       Install it first:  bash ml_isonet2_setup.sh"; exit 1; }
conda activate "$ENV_PREFIX"
command -v isonet.py >/dev/null 2>&1 || {
    echo "ERROR: isonet.py not on PATH after activating $ENV_PREFIX"; exit 1; }

# Headless plotting. refine saves preview slices and power spectra through
# matplotlib, and this env ships Qt6/PySide6 — so matplotlib picks an
# X-connecting backend, and when that connection drops the process dies with
#     ICE default IO error handler doing an exit(), pid = …, errno = 32
# and exit 1 AFTER a complete, successful training run (observed 2026-08-16,
# at epoch 50 of 50). Nothing here needs a display.
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen

# ---- the training set ------------------------------------------------------
# Link the chosen tomograms into work_dir/input_tomos so prepare_star --full
# sees exactly that folder ('_'-anchored matching, typo = hard error).
#
# Dot-free link names, as for IsoNet 1: Warp's pixel-size tag (…_12.56Apx.mrc)
# broke IsoNet 1's refine outright, and the names we hand IsoNet 2 are our own
# symlinks, so there is nothing to lose by keeping them boring. The series
# prefix is untouched, so ISO2_TOMO_LIST stems and the defocus injector's
# '_'-anchored matching still work.
link_into() {                   # $1 = src file, $2 = dest dir
    local b stem safe
    b="$(basename "$1")"
    stem="${b%.mrc}"
    safe="${stem//./p}"
    ln -sf "$1" "$2/$safe.mrc"
}
select_into() {                 # $1 = source dir, $2 = dest dir, $3 = label
    local src="$1" dst="$2" label="$3" stem f n=0 found
    mkdir -p "$dst"
    rm -f "$dst"/*.mrc 2>/dev/null || true
    if [ -n "$TOMO_LIST" ]; then
        for stem in $TOMO_LIST; do
            found=0
            for f in "$src/$stem.mrc" "$src/${stem}_"*.mrc; do
                [ -e "$f" ] || continue
                link_into "$f" "$dst"; found=1; n=$((n+1))
            done
            [ "$found" = "1" ] || echo "WARN: no ${stem}.mrc / ${stem}_*.mrc in $src ($label)"
        done
        [ "$n" -gt 0 ] || { echo "ERROR: ISO2_TOMO_LIST ($TOMO_LIST) matched nothing in $src."; exit 1; }
    else
        ls "$src"/*.mrc >/dev/null 2>&1 || { echo "ERROR: no .mrc in $src ($label)"; exit 1; }
        for f in "$src"/*.mrc; do link_into "$f" "$dst"; n=$((n+1)); done
    fi
    echo "$n tomogram(s) linked from $label."
}

select_into "$ABS_TOMO_DIR" "$WORK_DIR/input_tomos" "full"
if [ -z "$EVEN_DIR" ]; then
    # A single-map run in a work dir that last held a noise2noise run: the old
    # input_even/ and input_odd/ links still resolve, so a stale star's half
    # references look healthy and the run proceeds on the PREVIOUS run's data.
    for half in input_even input_odd; do
        if ls "$WORK_DIR/$half"/*.mrc >/dev/null 2>&1; then
            echo "clearing $half/ — this run has no halves, and leaving them"
            echo "  would let an earlier noise2noise star resolve against them."
            rm -f "$WORK_DIR/$half"/*.mrc
        fi
    done
fi
if [ -n "$EVEN_DIR" ]; then
    select_into "$EVEN_DIR" "$WORK_DIR/input_even" "even half"
    select_into "$ODD_DIR"  "$WORK_DIR/input_odd"  "odd half"
    # A half missing for one series would silently drop it from n2n training.
    N_FULL=$(ls "$WORK_DIR/input_tomos"/*.mrc 2>/dev/null | wc -l | tr -d ' ')
    N_EVEN=$(ls "$WORK_DIR/input_even"/*.mrc 2>/dev/null | wc -l | tr -d ' ')
    N_ODD=$(ls "$WORK_DIR/input_odd"/*.mrc 2>/dev/null | wc -l | tr -d ' ')
    [ "$N_EVEN" = "$N_ODD" ] || {
        echo "ERROR: $N_EVEN even vs $N_ODD odd tomograms — halves must pair up."; exit 1; }
    [ "$N_EVEN" = "$N_FULL" ] || echo "WARNING: $N_FULL full but $N_EVEN half pairs selected."
fi

cd "$WORK_DIR"
STAR="tomograms.star"

# ---- resume tests, applied per STEP ---------------------------------------
# "The folder has files in it" is the wrong question: a work dir left by a
# different tomo list is full of files belonging to OTHER tomograms, and every
# step would skip itself and the chain march on to fail deeper in. A step is
# done only when EVERY currently-linked tomogram has its output there.
have_all() {                    # $1 = folder that must hold one file per tomo
    local d="$1" f b
    [ -d "$d" ] || return 1
    for f in input_tomos/*.mrc; do
        b="$(basename "$f" .mrc)"
        ls "$d/$b"*.mrc >/dev/null 2>&1 || return 1
    done
    return 0
}

# The other half of a correct resume: the SETTINGS a folder was computed with.
# deconv/ holding all five at strength 1.0 looks identical to all five at 1.3,
# so lowering the strength and re-running would skip deconv and train on the
# OLD volumes, with nothing in the log to say so.
# The STAR's own stamp. Every step below reads the star, not input_tomos, so a
# star left by a different selection makes the whole chain run on the previous
# run's volumes — and report success. Seen 2026-08-16: a 12.56 Å single-map run
# on five named tomograms resumed a bin4 noise2noise star and spent half an
# hour deconvolving the OLD 6.28 Å volumes, three of which were not even in the
# requested list. Pixel size and pairing are not visible in the file names, so
# they go in the stamp; the names are checked separately, below.
STAR_STAMP="pixel=$PIX pairing=$([ -n "$EVEN_DIR" ] && echo n2n || echo single)"
DECONV_STAMP="strength=${ISO2_DECONVSTRENGTH:-default} falloff=${ISO2_SNRFALLOFF:-default}"
MASK_STAMP="density=${ISO2_DENSITY_PCT:-default} std=${ISO2_STD_PCT:-default} zcrop=${ISO2_ZCROP:-default}"
STAMP_FILE=".tomogration_settings"

stamp_check() {                 # $1 = folder, $2 = the stamp now
    local f="$1/$STAMP_FILE" old
    [ -f "$f" ] || return 0     # written by an older run: nothing to compare
    old="$(cat "$f")"
    [ "$old" = "$2" ] && return 0
    echo "ERROR: $WORK_DIR/$1 was computed with different settings."
    echo "         then: $old"
    echo "          now: $2"
    echo "       These decide what the model TRAINS ON, so resuming here would"
    echo "       train on the OLD ones while the log showed your new values."
    echo "       Either ISO2_FORCE=1 (redo the chain and retrain), or point the"
    echo "       work folder somewhere fresh to keep both variants side by side."
    exit 1
}
stamp_write() { printf '%s\n' "$2" > "$1/$STAMP_FILE"; }

# ---- 1. prepare_star (+ per-series defocus injection) ---------------------
if [ -f "$STAR" ] && [ "$FORCE" != "1" ]; then
    echo "SKIP: $STAR exists (resume)."
    # 1. Does the star still point at files that exist? Checked over EVERY
    #    folder the star can reference, not just input_* — averaged_tomos/ is
    #    where a noise2noise run's full volumes live, and leaving it out is how
    #    a stale star passed this test.
    GONE=0
    while read -r ref; do
        [ -z "$ref" ] && continue
        [ -e "$ref" ] || { echo "         missing: $ref"; GONE=$((GONE+1)); }
    done < <(grep -oE '(input_[a-z]*|averaged_tomos|deconv|mask)/[^[:space:]]*\.mrc' \
             "$STAR" | sort -u)
    if [ "$GONE" -gt 0 ]; then
        echo "ERROR: $STAR references $GONE tomogram(s) that are not there now."
        echo "       Fix:  rm $WORK_DIR/$STAR     and run this step again."
        exit 1
    fi
    # 2. Does it cover what THIS run selected? Existing-but-different is the
    #    dangerous case: every step reads the star, so a star describing other
    #    tomograms runs the whole chain on them and exits 0.
    MISSING=0
    for f in input_tomos/*.mrc; do
        b="$(basename "$f" .mrc)"
        grep -q -- "$b" "$STAR" || { echo "         not in the star: $b"; MISSING=$((MISSING+1)); }
    done
    if [ "$MISSING" -gt 0 ]; then
        echo "ERROR: $MISSING selected tomogram(s) do not appear in $STAR."
        echo "       This star was written by a DIFFERENT selection, and every"
        echo "       step below reads the star — so the run would process that"
        echo "       earlier selection's volumes and report success."
        echo "       It lists:"
        grep -oE '(input_[a-z]*|averaged_tomos)/[^[:space:]]*\.mrc' "$STAR" \
            | sed 's/^/         /' | sort -u | head -8
        echo "       Fix: use a work dir per variant (membrane/isonet2_bin8,"
        echo "            membrane/isonet2_bin4), or rm $WORK_DIR/$STAR."
        exit 1
    fi
    # 3. Pixel size and pairing do not show up in a file name.
    stamp_check . "$STAR_STAMP"
else
    PS_FLAGS=(--star_name "$STAR" --pixel_size "$PIX" --number_subtomos "$NSUB"
              --cs "${ISO2_CS:-2.7}" --voltage "${ISO2_KV:-300}" --ac "${ISO2_AC:-0.1}"
              --tilt_min "${ISO2_TILT_MIN:--60}" --tilt_max "${ISO2_TILT_MAX:-60}")
    if [ -n "$EVEN_DIR" ]; then
        # Their help: use even/odd for paired data, full otherwise. --full is
        # passed too so single-map refine and predict still have a column to
        # read, and --create_average keeps the full/half volumes consistent.
        PS_FLAGS+=(--even input_even --odd input_odd --create_average True)
    else
        PS_FLAGS+=(--full input_tomos)
    fi
    isonet.py prepare_star "${PS_FLAGS[@]}"
    [ -f "$STAR" ] || { echo "ERROR: prepare_star wrote no $STAR."; exit 1; }
    stamp_write . "$STAR_STAMP"
    # --defocus takes a LIST, but a list binds values to ROW ORDER; matching by
    # NAME cannot pair a defocus with the wrong tomogram. µm → Å, echoed per
    # series, never guessed.
    python3 "$HELPER_DIR/ml_isonet_star_defocus.py" "$STAR" "$ABS_XML_DIR"
fi

# ---- 2. CTF deconvolution (skip for phase-plate, or network/wiener CTF) ----
if [ "${ISO2_NO_DECONV:-}" != "1" ]; then
    if have_all deconv && [ "$FORCE" != "1" ]; then
        stamp_check deconv "$DECONV_STAMP"
        echo "SKIP: deconv/ already covers every selected tomogram"
    else
        DC_FLAGS=(--output_dir deconv --ncpus "$NCPUS")
        [ -n "${ISO2_SNRFALLOFF:-}" ]       && DC_FLAGS+=(--snrfalloff "$ISO2_SNRFALLOFF")
        [ -n "${ISO2_DECONVSTRENGTH:-}" ]   && DC_FLAGS+=(--deconvstrength "$ISO2_DECONVSTRENGTH")
        [ -n "${ISO2_HIGHPASS:-}" ]         && DC_FLAGS+=(--highpassnyquist "$ISO2_HIGHPASS")
        isonet.py deconv "$STAR" "${DC_FLAGS[@]}"
        stamp_write deconv "$DECONV_STAMP"
    fi
else
    echo "deconv: SKIPPED (ISO2_NO_DECONV=1)"
fi

# ---- 3. make_mask ----------------------------------------------------------
# Reads rlnDeconvTomoName when the star has it, else falls back to the raw
# tomogram — their default input_column already does this, so no branch here.
if have_all mask && [ "$FORCE" != "1" ]; then
    stamp_check mask "$MASK_STAMP"
    echo "SKIP: mask/ already covers every selected tomogram"
else
    MM_FLAGS=(--output_dir mask)
    [ -n "${ISO2_DENSITY_PCT:-}" ] && MM_FLAGS+=(--density_percentage "$ISO2_DENSITY_PCT")
    [ -n "${ISO2_STD_PCT:-}" ]     && MM_FLAGS+=(--std_percentage "$ISO2_STD_PCT")
    [ -n "${ISO2_ZCROP:-}" ]       && MM_FLAGS+=(--z_crop "$ISO2_ZCROP")
    isonet.py make_mask "$STAR" "${MM_FLAGS[@]}"
    stamp_write mask "$MASK_STAMP"
fi

# ---- 4. refine (the GPU training loop; hours) ------------------------------
RESULTS="isonet_maps"
RF_FLAGS=(--output_dir "$RESULTS" --gpuID "$GPU" --ncpus "$NCPUS" --method "$METHOD")
# WHICH volumes a single-map model actually trains on. refine defaults to
# rlnDeconvTomoName, but the predict step builds its star from RAW tomograms
# (it does not deconvolve), so a model trained on deconvolved volumes would be
# applied to undeconvolved ones — different statistics, quietly worse output.
# Default to rlnTomoName so training sees exactly what prediction will feed it;
# deconv still runs, because make_mask wants it. n2n models ignore this
# entirely (they read the half columns), so it is only set when it matters.
if [ "$METHOD" != "isonet2-n2n" ]; then
    RF_FLAGS+=(--input_column "${ISO2_REFINE_INPUT_COL:-rlnTomoName}")
fi
[ -n "${ISO2_EPOCHS:-}" ]     && RF_FLAGS+=(--epochs "$ISO2_EPOCHS")
[ -n "${ISO2_CUBE:-}" ]       && RF_FLAGS+=(--cube_size "$ISO2_CUBE")
[ -n "${ISO2_ARCH:-}" ]       && RF_FLAGS+=(--arch "$ISO2_ARCH")
[ -n "${ISO2_BATCH:-}" ]      && RF_FLAGS+=(--batch_size "$ISO2_BATCH")
[ -n "${ISO2_LOSS:-}" ]       && RF_FLAGS+=(--loss_func "$ISO2_LOSS")
[ -n "${ISO2_CTF_MODE:-}" ]   && RF_FLAGS+=(--CTF_mode "$ISO2_CTF_MODE")
[ -n "${ISO2_BFACTOR:-}" ]    && RF_FLAGS+=(--bfactor "$ISO2_BFACTOR")
[ -n "${ISO2_PRETRAINED:-}" ] && RF_FLAGS+=(--pretrained_model "$ISO2_PRETRAINED")

# "results/ has a checkpoint in it" is NOT the same question as "is it trained":
# save_interval writes checkpoints DURING training, so a crashed run leaves some
# behind. IsoNet 1 exits 0 even when it dies mid-iteration (seen twice on
# 2026-08-15) — assume the same here and judge by what appears AFTER this run
# starts, which needs no guess about checkpoint naming.
STARTED="$(mktemp -u .refine_started.XXXX)"; : > "$STARTED"
RAN_REFINE=0
if [ -d "$RESULTS" ] && ls "$RESULTS"/*.pt >/dev/null 2>&1 && [ "$FORCE" != "1" ]; then
    echo "SKIP: $RESULTS/ already holds trained checkpoint(s) (ISO2_FORCE=1 retrains):"
    ls -t "$RESULTS"/*.pt | head -3 | sed 's/^/         /'
else
    # NOT `set -e`'s job to judge this. refine can die at teardown with a
    # non-zero status after a complete run (the X11/ICE errno 32 above), and it
    # can also exit 0 after catching its own exception. Neither status is
    # evidence; the checkpoints on disk are. Capture the code, then look.
    RC=0
    isonet.py refine "$STAR" "${RF_FLAGS[@]}" ${ISO2_EXTRA_REFINE:-} || RC=$?
    RAN_REFINE=1
fi

if [ "$RAN_REFINE" = "1" ]; then
    FRESH="$(find "$RESULTS" -name '*.pt' -newer "$STARTED" 2>/dev/null | wc -l | tr -d ' ')"
    if [ "$FRESH" = "0" ]; then
        echo "==================================================================="
        echo "ERROR: refine wrote no new .pt checkpoint into $WORK_DIR/$RESULTS"
        echo "       (its exit status was $RC). Files it DID write since this run"
        echo "       started, if any:"
        find "$RESULTS" -newer "$STARTED" -type f 2>/dev/null | head -10 \
            | sed 's/^/         /' || true
        rm -f "$STARTED"
        echo "       Nothing here is a usable model, so no PROVENANCE.json is"
        echo "       written."
        echo "==================================================================="
        exit 1
    fi
    echo "refine wrote $FRESH new checkpoint(s)."
    if [ "$RC" != "0" ]; then
        echo "NOTE: refine exited $RC but the training COMPLETED — the"
        echo "      checkpoints above were written. A non-zero status here is"
        echo "      usually the matplotlib/X11 teardown (ICE … errno 32) after"
        echo "      the last preview is saved; this run set MPLBACKEND=Agg to"
        echo "      avoid it. Check the log above for a real traceback before"
        echo "      trusting the model."
    fi
fi
rm -f "$STARTED"

LAST_MODEL="$(ls -t "$RESULTS"/*.pt 2>/dev/null | head -1 || true)"

cat > PROVENANCE.json <<EOF
{
  "variant": "isonet2_training",
  "tool": "IsoNet2 2.0.1b0 (prepare_star/deconv/make_mask/refine)",
  "source_dir": "$ABS_TOMO_DIR",
  "params": {"pixel_size": "$PIX", "subtomos": "$NSUB",
             "epochs": "${ISO2_EPOCHS:-default}", "cube": "${ISO2_CUBE:-default}",
             "arch": "${ISO2_ARCH:-default}", "method": "$METHOD",
             "CTF_mode": "${ISO2_CTF_MODE:-default}",
             "deconv": "$([ "${ISO2_NO_DECONV:-}" = "1" ] && echo off || echo on)",
             "halves": "$([ -n "$EVEN_DIR" ] && echo "$EVEN_DIR + $ODD_DIR" || echo none)"},
  "tomo_list": "$([ -n "$TOMO_LIST" ] && echo "$TOMO_LIST" || echo "ALL")",
  "model": "$LAST_MODEL",
  "extraction_allowed": false,
  "note": "training project; the model applies to picking/segmentation volumes only",
  "date": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF

echo "==================================================================="
echo "IsoNet 2 training chain done."
echo "Newest checkpoint: $WORK_DIR/$LAST_MODEL"
echo "Apply it to the WHOLE dataset with the 'IsoNet 2: predict' step"
echo "(the model is reusable — likely across EML45/EML46 too)."
echo "Finished: $(date)"
echo "==================================================================="
