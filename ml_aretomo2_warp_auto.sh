#!/bin/bash
# ml_aretomo2_warp_auto.sh
#
# Runs AreTomo2 on all tilt stacks produced by WarpTools ts_stack.
# Integrated with warp_auto.py's expected project structure.
#
# Usage:
#   bash ml_aretomo2_warp_auto.sh <input_dir> <output_dir> [<gpu>] [<angpix>]
#
# Arguments:
#   input_dir    Directory containing <position>/<position>.st + .rawtlt
#                (typically warp_tiltseries/tiltstack/)
#   output_dir   Where to write mrc/, aln/, proj/, Imod/ subdirectories
#                (typically selected/aretomo_output/ or aretomo_output-vN/)
#   gpu          (optional) GPU ID to use. Default: 0
#   angpix       (optional) Unbinned pixel size in Angstroms. Default: 1.57
#
# AreTomo2 parameters are read from environment variables, all optional,
# with defaults preserving the legacy behaviour:
#   ARETOMO_ALIGNZ   (default 670)        Alignment volume height, unbinned px
#   ARETOMO_VOLZ     (default 3088)       Output volume height, unbinned px
#   ARETOMO_OUTBIN   (default 8)          Output binning
#   ARETOMO_TILTCOR  (default 0)          Tilt-offset correction (0/1)
#   ARETOMO_FLIPVOLZ (default 1)          Flip Z for Warp handedness (0/1)
#   ARETOMO_WBP      (default 1)          Weighted back projection (0/1)
#   ARETOMO_DARKTOL  (default 0.000001)   Dark-frame rejection threshold
#   ARETOMO_TILTAXIS (default "")         If set, pass to -TiltAxis
#   ARETOMO_PATCH    (default "")         If set, pass to -Patch, e.g. "4 4"
#   ARETOMO_ALIGN    (default 1)          Perform alignment (0/1)
#   ARETOMO_RECON    (default 1)          Reconstruct (0/1)
#   ARETOMO_BIN      (default /ceph/groups/structbio/Programs/AreTomo2/AreTomo2)
#
# Output layout (inside <output_dir>/):
#   mrc/              reconstructed tomograms (handedness-flipped)
#   aln/              .aln alignment files
#   proj/             projection .mrc files
#   Imod/             per-tilt-series Imod folders with .xf / .tlt
#   PARAMETERS.txt    record of the parameters used for this run

# Do NOT use set -e here: we want to continue processing other tilt series
# even if one fails.

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

if [ -z "$1" ] || [ -z "$2" ]; then
    cat <<EOF
Usage: bash $0 <input_dir> <output_dir> [<gpu>] [<angpix>]

Arguments:
    input_dir   Directory containing <position>/<position>.st + .rawtlt
    output_dir  Root for AreTomo outputs (mrc/, aln/, proj/, Imod/ go here)
    gpu         GPU ID (default 0)
    angpix      Pixel size in Angstroms (default 1.57)

AreTomo parameters are tuned via environment variables; see the header of
this script for the full list (ARETOMO_ALIGNZ, ARETOMO_VOLZ, etc.)
EOF
    exit 1
fi

INPUT_DIR="$1"
BASE_OUT="$2"
GPU="${3:-0}"
ANGPIX="${4:-1.57}"

# AreTomo parameters from environment with defaults
ALIGNZ="${ARETOMO_ALIGNZ:-670}"
VOLZ="${ARETOMO_VOLZ:-3088}"
OUTBIN="${ARETOMO_OUTBIN:-8}"
DARKTOL="${ARETOMO_DARKTOL:-0.000001}"
TILTCOR="${ARETOMO_TILTCOR:-0}"
FLIPVOLZ="${ARETOMO_FLIPVOLZ:-1}"
WBP="${ARETOMO_WBP:-1}"
TILTAXIS="${ARETOMO_TILTAXIS:-}"
PATCH="${ARETOMO_PATCH:-}"
ALIGN="${ARETOMO_ALIGN:-1}"
RECON="${ARETOMO_RECON:-1}"
# RECON=0 must actually DO something (it was echoed and audited but never passed
# to AreTomo): -VolZ 0 is AreTomo2's align-only mode — .xf/.tlt still written.
if [ "$RECON" = "0" ]; then
    VOLZ=0
fi
ARETOMO2_BIN="${ARETOMO_BIN:-/ceph/groups/structbio/Programs/AreTomo2/AreTomo2}"
# Parallelism: spread tilt series across these GPUs (space/comma list), N jobs each.
# Each AreTomo job uses ONE GPU; ARETOMO_GPUS empty falls back to the single positional GPU.
GPU_LIST="${ARETOMO_GPUS:-}"
JOBS_PER_GPU="${ARETOMO_JOBS_PER_GPU:-1}"
[ "$JOBS_PER_GPU" -ge 1 ] 2>/dev/null || JOBS_PER_GPU=1

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

echo "========================================"
echo "ml_aretomo2_warp_auto"
echo "Input:      $INPUT_DIR"
echo "Output:     $BASE_OUT"
echo "GPU:        $GPU"
echo "Apix:       $ANGPIX"
echo "AlignZ:     $ALIGNZ"
echo "VolZ:       $VOLZ"
echo "OutBin:     $OUTBIN"
echo "DarkTol:    $DARKTOL"
echo "TiltCor:    $TILTCOR"
echo "FlipVolZ:   $FLIPVOLZ"
echo "Wbp:        $WBP"
[ -n "$TILTAXIS" ] && echo "TiltAxis:   $TILTAXIS"
[ -n "$PATCH" ]    && echo "Patch:      $PATCH"
echo "Align:      $ALIGN"
echo "Recon:      $RECON"
echo "Binary:     $ARETOMO2_BIN"
echo "Started:    $(date)"
echo "========================================"

module load cuda/12.5.1 2>/dev/null || echo "WARNING: could not load cuda/12.5.1"
module load imod 2>/dev/null || echo "WARNING: could not load imod"

# --- CUDA runtime setup -------------------------------------------------------
# AreTomo2 needs the CUDA runtime libs (libcufft.so.11, libcudart.so.12, ...) AND
# the REAL NVIDIA driver lib (libcuda.so.1). Two traps on this cluster:
#   1. A cuda module sets PATH/CUDA_HOME but the toolkit lib dir is not always on
#      LD_LIBRARY_PATH -> "libcufft.so.11: cannot open shared object file".
#   2. The toolkit lib dir ALSO ships a *stub* libcuda.so.1 (-> stubs/libcuda.so).
#      If that dir is on LD_LIBRARY_PATH the loader picks the stub over the real
#      driver and EVERY gpu reports "Error: GPU N is invalid, skip".
# Fix: pick a lib dir that has libcufft.so.11 (ARETOMO_CUDA_LIB first, else the
# nvcc/CUDA_HOME toolkit lib64), build a STUB-FREE symlink farm of it (everything
# except libcuda*), put only the farm on LD_LIBRARY_PATH, and scrub any inherited
# stub dir — so cuFFT/cudart come from the toolkit but libcuda.so.1 resolves to the
# real driver via the system ld cache.

# Candidate runtime dirs, in priority order.
_cuda_candidates=()
[ -n "$ARETOMO_CUDA_LIB" ] && _cuda_candidates+=("$ARETOMO_CUDA_LIB")
if command -v nvcc >/dev/null 2>&1; then
    _root=$(dirname "$(dirname "$(command -v nvcc)")")
    _cuda_candidates+=("$_root/lib64" "$_root/lib")
fi
for _cudaroot in "$CUDA_HOME" "$CUDA_PATH" "$CUDA_ROOT" "$CUDADIR" "$CUDA_INSTALL_DIR"; do
    [ -n "$_cudaroot" ] && _cuda_candidates+=("$_cudaroot/lib64" "$_cudaroot/lib")
done

_cuda_lib=""
for _d in "${_cuda_candidates[@]}"; do
    [ -e "$_d/libcufft.so.11" ] && { _cuda_lib="$_d"; break; }
done

if [ -z "$_cuda_lib" ]; then
    echo "ERROR: could not find libcufft.so.11 (the CUDA-12 runtime) anywhere."
    echo "       Provide a CUDA-12 runtime in your own space and point ARETOMO_CUDA_LIB at it:"
    echo "         module load miniconda/latest"
    echo "         conda create -y -n cuda12rt -c nvidia cuda-libraries=12.5 cuda-version=12.5"
    echo "         ARETOMO_CUDA_LIB=/ceph/users/<you>/.conda/envs/cuda12rt/lib"
    exit 1
fi

# Build a stub-free symlink farm of the chosen runtime dir (exclude libcuda*).
_cuda_farm=$(mktemp -d "${TMPDIR:-/tmp}/aretomo_cuda.XXXXXX") || {
    echo "ERROR: could not create a temp dir for the CUDA lib farm"; exit 1; }
for _f in "$_cuda_lib"/*.so*; do
    [ -e "$_f" ] || continue
    case "$(basename "$_f")" in libcuda.so*) continue ;; esac
    ln -s "$_f" "$_cuda_farm/" 2>/dev/null
done

# Scrub any inherited LD_LIBRARY_PATH entry that exposes a STUB libcuda (a
# libcuda.so* resolving into a stubs/ dir), then prepend the stub-free farm.
_clean=""
_IFS_save="$IFS"; IFS=':'
for _d in $LD_LIBRARY_PATH; do
    [ -z "$_d" ] && continue
    _t=""
    [ -e "$_d/libcuda.so.1" ] && _t=$(readlink -f "$_d/libcuda.so.1" 2>/dev/null)
    [ -z "$_t" ] && [ -e "$_d/libcuda.so" ] && _t=$(readlink -f "$_d/libcuda.so" 2>/dev/null)
    case "$_t" in
        */stubs/*) echo "Dropping CUDA stub dir from LD_LIBRARY_PATH: $_d"; continue ;;
    esac
    _clean="${_clean:+$_clean:}$_d"
done
IFS="$_IFS_save"
export LD_LIBRARY_PATH="$_cuda_farm${_clean:+:$_clean}"

# One EXIT trap cleans every temp dir we may create (stub-free CUDA farm, the
# non-setgid binary copy, and the parallel results dir).
RESDIR=""
trap 'rm -rf "$_cuda_farm" ${_bindir:+"$_bindir"} ${RESDIR:+"$RESDIR"}' EXIT

echo "CUDA runtime libs: $_cuda_lib (stub-free farm: $_cuda_farm)"
echo "libcuda.so.1 will resolve to the real driver via the system ld cache."

# An EMPTY CUDA_VISIBLE_DEVICES hides ALL GPUs (it does NOT mean "all") — CUDA then
# reports 0 devices and AreTomo2 fails with "Error: GPU N is invalid, skip". If the
# var is set but empty (often inherited from the launching shell), unset it so the
# GPUs become visible. A deliberately-set value (e.g. "0,1") is left untouched.
if [ -n "${CUDA_VISIBLE_DEVICES+x}" ] && [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    unset CUDA_VISIBLE_DEVICES
    echo "Unset empty CUDA_VISIBLE_DEVICES (it was masking every GPU)"
fi

if [ ! -x "$ARETOMO2_BIN" ]; then
    echo "ERROR: AreTomo2 binary not found or not executable: $ARETOMO2_BIN"
    exit 1
fi

# If the binary is setuid/setgid (the shared /ceph/groups/.../AreTomo2 is -rwxrwsr-x),
# the dynamic loader runs in secure-execution mode and IGNORES LD_LIBRARY_PATH, so it
# can't find libcufft no matter what we set. Transparently run a private, non-setgid
# copy (a plain cp drops the s-bits) so LD_LIBRARY_PATH is honoured. This means the
# user can leave ARETOMO_BIN pointed at the canonical shared binary.
if [ -u "$ARETOMO2_BIN" ] || [ -g "$ARETOMO2_BIN" ]; then
    # _bindir is cleaned by the combined EXIT trap set in the CUDA setup above.
    _bindir=$(mktemp -d "${TMPDIR:-/tmp}/aretomo_bin.XXXXXX") || {
        echo "ERROR: could not create a temp dir for a non-setgid AreTomo2 copy"; exit 1; }
    if cp "$ARETOMO2_BIN" "$_bindir/AreTomo2" && chmod u+rwx,u-s,g-s "$_bindir/AreTomo2"; then
        echo "Note: $ARETOMO2_BIN is setuid/setgid (loader would ignore LD_LIBRARY_PATH);"
        echo "      running a non-setgid copy instead: $_bindir/AreTomo2"
        ARETOMO2_BIN="$_bindir/AreTomo2"
    else
        echo "WARNING: could not make a non-setgid copy; AreTomo2 may fail to load libcufft."
    fi
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory not found: $INPUT_DIR"
    exit 1
fi

mkdir -p "$BASE_OUT/mrc" "$BASE_OUT/aln" "$BASE_OUT/proj" "$BASE_OUT/Imod"

# Write a PARAMETERS.txt record into the output directory for this run, so the
# user can always match a versioned output folder back to the parameters that
# produced it.
PARAMS_FILE="$BASE_OUT/PARAMETERS.txt"
{
    echo "# AreTomo2 run parameters"
    echo "# Generated by ml_aretomo2_warp_auto.sh at $(date)"
    echo ""
    echo "input_dir     = $INPUT_DIR"
    echo "output_dir    = $BASE_OUT"
    echo "gpu           = $GPU"
    echo "gpus          = ${GPU_LIST:-$GPU}"
    echo "jobs_per_gpu  = $JOBS_PER_GPU"
    echo "angpix        = $ANGPIX"
    echo "alignz        = $ALIGNZ"
    echo "volz          = $VOLZ"
    echo "outbin        = $OUTBIN"
    echo "darktol       = $DARKTOL"
    echo "tiltcor       = $TILTCOR"
    echo "flipvolz      = $FLIPVOLZ"
    echo "wbp           = $WBP"
    [ -n "$TILTAXIS" ] && echo "tiltaxis      = $TILTAXIS"
    [ -n "$PATCH" ]    && echo "patch         = $PATCH"
    echo "align         = $ALIGN"
    echo "recon         = $RECON"
    echo "binary        = $ARETOMO2_BIN"
} > "$PARAMS_FILE"
echo "Parameter record written to: $PARAMS_FILE"
echo ""

# -----------------------------------------------------------------------------
# Collect tilt series folders
# -----------------------------------------------------------------------------

TILT_DIRS=()
for d in "$INPUT_DIR"/*/; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    if [ -f "${d}${name}.st" ]; then
        TILT_DIRS+=("$d")
    fi
done

if [ ${#TILT_DIRS[@]} -eq 0 ]; then
    echo "ERROR: No tilt series found in $INPUT_DIR"
    echo "       Expected subdirectories like <Position>/ containing <Position>.st"
    exit 1
fi

echo "Found ${#TILT_DIRS[@]} tilt series to process"

# -----------------------------------------------------------------------------
# Per-series worker: aligns + reconstructs ONE tilt series on ONE GPU, then sorts
# its outputs. All output goes to stdout (the caller decides where: the terminal
# for a single GPU, a per-series log file when running in parallel). Name-scoped
# moves make concurrent series safe. Returns 0=ok, 1=fail, 3=skip.
# -----------------------------------------------------------------------------
process_one_series() {
    local TILT_DIR="$1" GPU="$2"
    local name INMRC ANGFILE OUTMRC IMOD_OUTDIR TMP_MRC SRC
    local EXTRA_FLAGS=()
    name=$(basename "$TILT_DIR" | sed 's:/$::')

    INMRC="$TILT_DIR/${name}.st"
    ANGFILE="$TILT_DIR/${name}.rawtlt"
    OUTMRC="$BASE_OUT/mrc/${name}.mrc"

    [ -f "$INMRC" ]   || { echo "SKIP: missing $INMRC"; return 3; }
    [ -f "$ANGFILE" ] || { echo "SKIP: missing $ANGFILE"; return 3; }
    if [ -f "$OUTMRC" ] && [ -s "$OUTMRC" ]; then
        echo "SKIP: already reconstructed at $OUTMRC"; return 3
    fi

    [ -n "$TILTAXIS" ] && EXTRA_FLAGS+=(-TiltAxis $TILTAXIS)
    [ -n "$PATCH" ]    && EXTRA_FLAGS+=(-Patch $PATCH)

    echo "=== $name on GPU $GPU @ $(date) ==="
    if ! "$ARETOMO2_BIN" \
        AreTomo \
        -InMrc "$INMRC" \
        -OutMrc "$OUTMRC" \
        -AngFile "$ANGFILE" \
        -gpu "$GPU" \
        -AlignZ "$ALIGNZ" \
        -VolZ "$VOLZ" \
        -OutBin "$OUTBIN" \
        -TiltCor "$TILTCOR" \
        -OutXF 1 \
        -OutImod 1 \
        -Wbp "$WBP" \
        -PixSize "$ANGPIX" \
        -DarkTol "$DARKTOL" \
        -FlipVolZ "$FLIPVOLZ" \
        -Align "$ALIGN" \
        "${EXTRA_FLAGS[@]}"; then
        echo "ERROR: AreTomo2 failed for $name"
        # AreTomo writes -OutMrc incrementally: a crash leaves a truncated,
        # non-empty file the skip test above would call "already reconstructed"
        # on every rerun — remove it so the series gets retried.
        rm -f "$OUTMRC"
        return 1
    fi

    # Move auxiliary outputs (name-scoped -> safe with other series running)
    for SRC in "." "$INPUT_DIR" "$INPUT_DIR/$name" "$BASE_OUT/mrc"; do
        [ -d "$SRC" ] || continue
        find "$SRC" -maxdepth 1 -type f -name "${name}*.aln" \
            -exec mv -v {} "$BASE_OUT/aln/" \; 2>/dev/null || true
        find "$SRC" -maxdepth 1 -type f -name "${name}_proj*.mrc" \
            -exec mv -v {} "$BASE_OUT/proj/" \; 2>/dev/null || true
        find "$SRC" -maxdepth 1 -type d -name "${name}_Imod" \
            -exec mv -v {} "$BASE_OUT/Imod/" \; 2>/dev/null || true
    done

    # Copy the rawtlt into the Imod folder as .tlt so ts_import_alignments works
    IMOD_OUTDIR="$BASE_OUT/Imod/${name}_Imod"
    if [ -d "$IMOD_OUTDIR" ] && [ ! -f "$IMOD_OUTDIR/${name}.tlt" ]; then
        cp "$ANGFILE" "$IMOD_OUTDIR/${name}.tlt"
    fi

    # Handedness flip
    if [ -f "$OUTMRC" ]; then
        echo "Flipping handedness with trimvol..."
        TMP_MRC="${OUTMRC%.mrc}_tmp.mrc"
        if trimvol -yz "$OUTMRC" "$TMP_MRC"; then
            mv "$TMP_MRC" "$OUTMRC"
        else
            echo "WARNING: trimvol -yz failed on $OUTMRC"
        fi
    fi

    echo "DONE: $name"
    return 0
}

# -----------------------------------------------------------------------------
# Build the GPU pool and process the series (parallel when >1 worker)
# -----------------------------------------------------------------------------
GPUS=()
if [ -n "$GPU_LIST" ]; then
    read -r -a GPUS <<< "${GPU_LIST//,/ }"
fi
[ ${#GPUS[@]} -eq 0 ] && GPUS=("$GPU")

# One worker per (GPU x JOBS_PER_GPU). Each worker handles a slice of the series
# sequentially, pinned to its GPU; all workers run concurrently.
WORKER_GPUS=()
for ((_j = 0; _j < JOBS_PER_GPU; _j++)); do
    for _g in "${GPUS[@]}"; do WORKER_GPUS+=("$_g"); done
done
NW=${#WORKER_GPUS[@]}

processed=0
skipped=0
failed=0

if [ "$NW" -le 1 ]; then
    # ---- Single GPU: run sequentially, stream live to the terminal ----
    echo "Mode: sequential on GPU ${WORKER_GPUS[0]:-$GPU}"
    echo ""
    for TILT_DIR in "${TILT_DIRS[@]}"; do
        name=$(basename "$TILT_DIR" | sed 's:/$::')
        echo "----- Processing: $name -----"
        process_one_series "$TILT_DIR" "${WORKER_GPUS[0]:-$GPU}"
        case $? in
            0) processed=$((processed + 1)) ;;
            3) skipped=$((skipped + 1)) ;;
            *) failed=$((failed + 1)) ;;
        esac
        echo ""
    done
else
    # ---- Parallel: ${#GPUS[@]} GPU(s) x ${JOBS_PER_GPU} job(s) = $NW concurrent ----
    echo "Mode: parallel — ${#GPUS[@]} GPU(s) [${GPUS[*]}] x ${JOBS_PER_GPU} job(s)/GPU = $NW concurrent"
    echo "Per-series logs: $BASE_OUT/logs/<name>.log"
    echo ""
    mkdir -p "$BASE_OUT/logs"
    RESDIR=$(mktemp -d "${TMPDIR:-/tmp}/aretomo_res.XXXXXX")

    # Worker: process its assigned series sequentially, each to its own log.
    _worker() {
        local gpu="$1"; shift
        local d nm rc
        for d in "$@"; do
            nm=$(basename "$d" | sed 's:/$::')
            echo "[GPU $gpu] >> start  $nm"
            process_one_series "$d" "$gpu" > "$BASE_OUT/logs/$nm.log" 2>&1
            rc=$?
            if [ "$rc" -eq 0 ]; then
                echo "ok"   > "$RESDIR/$nm"; echo "[GPU $gpu] OK     $nm"
            elif [ "$rc" -eq 3 ]; then
                echo "skip" > "$RESDIR/$nm"
                echo "[GPU $gpu] SKIP   $nm ($(grep -m1 '^SKIP' "$BASE_OUT/logs/$nm.log"))"
            else
                echo "fail" > "$RESDIR/$nm"
                echo "[GPU $gpu] FAILED $nm  -- see $BASE_OUT/logs/$nm.log"
                tail -n 12 "$BASE_OUT/logs/$nm.log" | sed 's/^/    | /'
            fi
        done
    }

    # Kill outstanding jobs if the user terminates the run.
    trap 'kill $(jobs -p) 2>/dev/null' INT TERM

    # Round-robin the series across workers (worker w gets indices w, w+NW, ...).
    for ((w = 0; w < NW; w++)); do
        chunk=()
        for ((i = w; i < ${#TILT_DIRS[@]}; i += NW)); do chunk+=("${TILT_DIRS[$i]}"); done
        [ ${#chunk[@]} -eq 0 ] && continue
        _worker "${WORKER_GPUS[$w]}" "${chunk[@]}" &
    done
    wait
    trap - INT TERM

    # Tally per-series results.
    for f in "$RESDIR"/*; do
        [ -e "$f" ] || continue
        case "$(cat "$f")" in
            ok)   processed=$((processed + 1)) ;;
            skip) skipped=$((skipped + 1)) ;;
            fail) failed=$((failed + 1)) ;;
        esac
    done
    rm -rf "$RESDIR"; RESDIR=""
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo "========================================"
echo "Summary:"
echo "  Processed successfully: $processed"
echo "  Skipped:                $skipped"
echo "  Failed:                 $failed"
echo "Finished: $(date)"
echo "========================================"

if [ "$failed" -gt 0 ]; then
    exit 2
fi
exit 0
