#!/bin/bash
# ml_relion4_handoff_warp_auto.sh
#
# Hand exported subtomograms (from WarpTools ts_export_particles, 3D / no --2d) to
# RELION 4 for 3D classification. This encodes the invariants that repeatedly bite
# the Warp -> RELION handoff:
#
#   1. LAUNCH RELION FROM THE EXPORT DIR. ts_export_particles writes the subtomo image
#      paths RELATIVE to its --output_processing dir. RELION must run FROM that dir or
#      every path is wrong ("file does not exist"). This script cd's there.
#   2. RELION 4 does NOT auto-resize the reference. It is rescaled/reboxed here with
#      relion_image_handler to match the particles' pixel size + box.
#   3. Never launch inside another RELION version's project — stale pipeline files are
#      parked first.
#   4. MPI = (#GPUs + 1): one non-GPU leader + one worker per GPU.
#
# Usage:
#   bash ml_relion4_handoff_warp_auto.sh <project_dir> <particles.star> [--execute]
#     <project_dir>    the ts_export_particles --output_processing dir (RELION project root)
#     <particles.star> particles star, RELATIVE to <project_dir> (e.g. matching.star)
#   Default is a DRY RUN (prints the plan + the relion command, runs nothing). Add
#   --execute to pre-scale the reference, clean the project, and submit Class3D.
#
# Knobs via environment (the tomogration GUI sets these):
#   RELION_MODULE   module load name for RELION 4       (default: relion/4.0.1)
#   REF_MAP         reference .mrc (e.g. your EMDB map)  (REQUIRED)
#   REF_ANGPIX      pixel size (Å) of REF_MAP            (REQUIRED)
#   OUTPUT_ANGPIX   particles' pixel size (Å)            (must equal the export value)
#   BOX             particles' box size (px)             (must equal the export value)
#   DIAMETER        particle diameter (Å) for the mask   (REQUIRED)
#   SYMMETRY        point group for classification       (default: C1 — symmetrise at Refine3D)
#   NCLASSES        number of 3D classes (K)             (default: 4)
#   GPUS            comma list of GPU ids                (default: 0,1,2,3; MPI = n+1)
#   ITER            classification iterations            (default: 25)
#   INI_LOWPASS     initial reference low-pass (Å)       (default: 45)
#   RELION_EXTRA    appended verbatim to relion_refine_mpi
#
# Why RELION 4 and not 5 on this VM class: RELION 5 ships as a container whose CUDA
# runtime can outrun the host driver (535.x / CUDA 12.2) -> GPU dies with error-code
# 35. RELION 4 is native -> GPU works. If the driver is later bumped, revisit v5.
set -e

EXECUTE=0
ARGS=()
for a in "$@"; do
    if [ "$a" = "--execute" ]; then EXECUTE=1; else ARGS+=("$a"); fi
done
PROJECT_DIR="${ARGS[0]:?Usage: $0 <project_dir> <particles.star> [--execute]}"
PARTICLES="${ARGS[1]:?Usage: $0 <project_dir> <particles.star> [--execute]}"

RELION_MODULE="${RELION_MODULE:-relion/4.0.1}"
SYMMETRY="${SYMMETRY:-C1}"
NCLASSES="${NCLASSES:-4}"
GPUS="${GPUS:-0,1,2,3}"
ITER="${ITER:-25}"
INI_LOWPASS="${INI_LOWPASS:-45}"

mode=$([ $EXECUTE -eq 1 ] && echo EXECUTE || echo DRY-RUN)
echo "==================================================================="
echo "ml_relion4_handoff_warp_auto    [$mode]"
echo "Project dir:   $PROJECT_DIR"
echo "Particles:     $PARTICLES  (relative to project dir)"
echo "RELION module: $RELION_MODULE    Symmetry: $SYMMETRY    Classes: $NCLASSES"
echo "GPUs:          $GPUS"
echo "Started:       $(date)"
echo "==================================================================="

# ---- 1. sanity on inputs -----------------------------------------------------
[ -d "$PROJECT_DIR" ] || { echo "ERROR: project dir not found: $PROJECT_DIR"; exit 1; }
STAR="$PROJECT_DIR/$PARTICLES"
[ -f "$STAR" ] || { echo "ERROR: particles star not found: $STAR"; exit 1; }
[ -s "$STAR" ] || { echo "ERROR: particles star is EMPTY: $STAR (export produced nothing)"; exit 1; }

for v in REF_MAP REF_ANGPIX OUTPUT_ANGPIX BOX DIAMETER; do
    if [ -z "${!v}" ]; then
        echo "ERROR: $v is required (set it in the GUI / environment)."; exit 1
    fi
done
[ -f "$REF_MAP" ] || { echo "ERROR: REF_MAP not found: $REF_MAP"; exit 1; }

# Assert the launch-root invariant: the FIRST subtomo image path in the star must
# resolve from PROJECT_DIR. If not, we'd be launching from the wrong place.
first_img=$(grep -oE '[^[:space:]]+\.mrc' "$STAR" | head -1 || true)
if [ -n "$first_img" ]; then
    case "$first_img" in
        /*) probe="$first_img" ;;                 # absolute (relative_output_paths off)
        *)  probe="$PROJECT_DIR/$first_img" ;;
    esac
    if [ ! -f "$probe" ]; then
        echo "ERROR: launch-root mismatch — '$first_img' does not resolve from $PROJECT_DIR."
        echo "       Re-export with --output_processing = $PROJECT_DIR and --relative_output_paths,"
        echo "       or point <project_dir> at the dir those paths are relative to."
        exit 1
    fi
    echo "launch-root OK: '$first_img' resolves under $PROJECT_DIR"
fi
n_particles=$(grep -cE '\.mrc' "$STAR" || true)
echo "particles star: ~$n_particles image rows"

# ---- environment -------------------------------------------------------------
module load "$RELION_MODULE" 2>/dev/null || { echo "ERROR: could not 'module load $RELION_MODULE'"; exit 1; }
command -v relion_refine_mpi >/dev/null 2>&1 || { echo "ERROR: relion_refine_mpi not on PATH after module load."; exit 1; }

# ---- 2. pre-scale the reference (RELION 4 has no auto-resize) -----------------
REF_SCALED="$PROJECT_DIR/ref_${OUTPUT_ANGPIX}apx_box${BOX}.mrc"
IMG_HANDLER=(relion_image_handler --i "$REF_MAP" --angpix "$REF_ANGPIX"
             --rescale_angpix "$OUTPUT_ANGPIX" --new_box "$BOX" --o "$REF_SCALED")
echo "-------------------------------------------------------------------"
echo "Reference prep (rescale $REF_ANGPIX -> $OUTPUT_ANGPIX Å/px, box $BOX):"
echo "  ${IMG_HANDLER[*]}"

# ---- 3. MPI sizing -----------------------------------------------------------
n_gpu=$(printf '%s' "$GPUS" | tr ',' '\n' | grep -c '[0-9]')
[ "$n_gpu" -ge 1 ] || n_gpu=1
MPI=$((n_gpu + 1))
OUT="$PROJECT_DIR/Class3D/job001/run"

# ---- 4. the Class3D command --------------------------------------------------
# NOTE: relion_refine_mpi is launched FROM the project dir; paths below are relative.
REFINE=(mpirun -n "$MPI" relion_refine_mpi
        --i "$PARTICLES"
        --ref "$(basename "$REF_SCALED")"
        --o "Class3D/job001/run"
        --ini_high "$INI_LOWPASS"
        --pad 2 --ctf
        --iter "$ITER" --tau2_fudge 4
        --K "$NCLASSES"
        --sym "$SYMMETRY"
        --particle_diameter "$DIAMETER"
        --oversampling 1 --healpix_order 2 --offset_range 5 --offset_step 2
        --dont_combine_weights_via_disc --pool 3 --j 4
        --gpu "$GPUS"
        $RELION_EXTRA)
echo "-------------------------------------------------------------------"
echo "Class3D (MPI $MPI = $n_gpu GPU + 1 leader), run FROM $PROJECT_DIR:"
echo "  ${REFINE[*]}"
echo "-------------------------------------------------------------------"

if [ "$EXECUTE" -eq 0 ]; then
    echo "DRY-RUN — nothing changed. Review the two commands above, then re-run with --execute."
    echo "(Tune iter / K / healpix / diameter to your particle before a long run.)"
    exit 0
fi

# ---- execute: reference, project hygiene, then submit ------------------------
"${IMG_HANDLER[@]}"
echo "Wrote scaled reference: $REF_SCALED"

# Park stale cross-version pipeline state so RELION doesn't read another run's nodes.
PARK="$PROJECT_DIR/_parked_$(date +%Y%m%d_%H%M%S)"
for name in default_pipeline.star .Nodes .TMP_runfiles; do
    if [ -e "$PROJECT_DIR/$name" ]; then
        mkdir -p "$PARK"; mv "$PROJECT_DIR/$name" "$PARK/"; echo "parked $name"
    fi
done
for f in "$PROJECT_DIR"/.gui_*job.star; do
    [ -e "$f" ] || continue
    mkdir -p "$PARK"; mv "$f" "$PARK/"
done

mkdir -p "$PROJECT_DIR/Class3D/job001"
echo "Launching Class3D from $PROJECT_DIR …"
( cd "$PROJECT_DIR" && "${REFINE[@]}" )
status=$?
echo "==================================================================="
if [ "$status" -eq 0 ]; then
    echo "RELION 4 Class3D finished OK. Outputs under $OUT*."
else
    echo "relion_refine_mpi exited $status — see the messages above."
fi
echo "Finished: $(date)"
echo "==================================================================="
exit "$status"
