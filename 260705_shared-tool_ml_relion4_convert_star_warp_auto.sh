#!/bin/bash
# ml_relion4_convert_star_warp_auto.sh
#
# Convert the particle STAR that WarpTools ts_export_particles writes into the
# format RELION 4 expects, and (optionally) build a de-novo initial reference
# from a random subset of particles. This is the step BETWEEN ts_export_particles
# and the RELION 4 Class3D handoff — its outputs (matching_conv.star +
# random_subset_ref.mrc) feed straight into ml_relion4_handoff_warp_auto.sh.
#
# It does two things that repeatedly bite the Warp -> RELION 4 handoff:
#
#   1. REWRITE THE PARTICLE PATHS. ts_export_particles writes image paths relative
#      to its --output_processing dir (e.g. 'subtomo/Position001_....mrc'). This
#      rewrites that prefix to an ABSOLUTE path (PARTICLEDIR) so relion_convert_star
#      / RELION can find every subtomogram regardless of where it's launched.
#   2. STRIP THE STAR HEADER when sampling particles for the initial model. A RELION
#      star has a data_optics block + a data_particles loop_ header before the data
#      rows; the reference star must keep that whole header, then a random N rows.
#      The header boundary is AUTO-DETECTED (was hardcoded head -n 33 / tail -n +35
#      in the original hand-written script — auto-detect makes it dataset-agnostic).
#
# Adapted from a hand-written RELION-4 conversion script (obv73998), generalised so
# the paths / pixel size / subset size are set from the tomogration GUI.
#
# Usage:
#   bash ml_relion4_convert_star_warp_auto.sh <project_dir> <starfile> [--execute]
#     <project_dir>  the ts_export_particles --output_processing dir (RELION project root)
#     <starfile>     Warp export star, RELATIVE to <project_dir> (e.g. matching.star)
#   Default is a DRY RUN (prints the plan + the relion commands, changes nothing).
#   Add --execute to actually convert and reconstruct.
#
# Knobs via environment (the tomogration GUI sets these):
#   RELION_MODULE  module load name for RELION 4      (default: relion/4.0.1)
#   PARTICLEDIR    absolute path to the subtomo dir,
#                  MUST end in '/'                     (default: <project_dir>/subtomo/)
#   PATH_MATCH     path prefix in the Warp star to
#                  replace with PARTICLEDIR            (default: subtomo/)
#   CS             spherical aberration (mm)           (default: 2.7)
#   Q0            amplitude contrast                   (default: 0.07)
#   NREF           particles for the initial reference (default: 1000)
#   MAKE_REF       1 = also build random_subset_ref.mrc via relion_reconstruct,
#                  0 = only convert the star           (default: 1)
#   HEADER_LINES   override the auto-detected header line count (blank = auto)
#   DATA_START     override the auto-detected first data-row line (blank = auto)
#   REF_OUT        initial reference filename          (default: random_subset_ref.mrc)

EXECUTE=0
ARGS=()
for a in "$@"; do
    if [ "$a" = "--execute" ]; then EXECUTE=1; else ARGS+=("$a"); fi
done
PROJECT_DIR="${ARGS[0]:?Usage: $0 <project_dir> <starfile> [--execute]}"
STARFILE="${ARGS[1]:?Usage: $0 <project_dir> <starfile> [--execute]}"

RELION_MODULE="${RELION_MODULE:-relion/4.0.1}"
PATH_MATCH="${PATH_MATCH:-subtomo/}"
CS="${CS:-2.7}"
Q0="${Q0:-0.07}"
NREF="${NREF:-1000}"
MAKE_REF="${MAKE_REF:-1}"
REF_OUT="${REF_OUT:-random_subset_ref.mrc}"

mode=$([ $EXECUTE -eq 1 ] && echo EXECUTE || echo DRY-RUN)

# ---- 1. sanity on inputs -----------------------------------------------------
[ -d "$PROJECT_DIR" ] || { echo "ERROR: project dir not found: $PROJECT_DIR"; exit 1; }
# Everything runs FROM the project dir (the launch-root invariant): the Warp star's
# image paths are relative to it, and RELION resolves them from the cwd.
PROJECT_DIR=$(cd "$PROJECT_DIR" && pwd)
STAR="$PROJECT_DIR/$STARFILE"
[ -f "$STAR" ] || { echo "ERROR: starfile not found: $STAR"; exit 1; }
[ -s "$STAR" ] || { echo "ERROR: starfile is EMPTY: $STAR (export produced nothing)"; exit 1; }

# PARTICLEDIR defaults to the absolute subtomo/ under the project dir; enforce the
# trailing slash the sed rewrite depends on.
PARTICLEDIR="${PARTICLEDIR:-$PROJECT_DIR/subtomo/}"
case "$PARTICLEDIR" in */) ;; *) PARTICLEDIR="$PARTICLEDIR/" ;; esac

# fname = starfile without its .star extension (RELION outputs are named off it).
base="${STARFILE##*/}"
stem="${base%.star}"
FNAME="$PROJECT_DIR/$stem"
CONV="${FNAME}_conv.star"
REFSTAR="${FNAME}_conv_random${NREF}.star"

echo "==================================================================="
echo "ml_relion4_convert_star_warp_auto    [$mode]"
echo "Project dir:   $PROJECT_DIR"
echo "Warp star:     $STARFILE"
echo "Particle dir:  $PARTICLEDIR   (replaces '$PATH_MATCH' in the star)"
echo "RELION module: $RELION_MODULE    Cs: $CS    Q0: $Q0"
echo "Converted:     ${CONV##*/}"
[ "$MAKE_REF" = "1" ] && echo "Reference:     ${REF_OUT}  (from $NREF random particles)"
echo "Started:       $(date)"
echo "==================================================================="

if ! grep -q "$PATH_MATCH" "$STAR"; then
    echo "WARNING: '$PATH_MATCH' not found in $STARFILE — the path rewrite will be a"
    echo "         no-op. Check how ts_export_particles wrote the image paths and set"
    echo "         PATH_MATCH to the actual prefix (e.g. 'subtomo/')."
fi

if [ "$EXECUTE" -eq 0 ]; then
    echo "Planned steps (run FROM $PROJECT_DIR):"
    echo "  1. cp $STARFILE temp.star"
    echo "  2. sed -i 's|$PATH_MATCH|$PARTICLEDIR|g' temp.star"
    echo "  3. relion_convert_star --i temp.star --o ${CONV##*/} --Cs $CS --Q0 $Q0"
    [ "$MAKE_REF" = "1" ] && {
    echo "  4. split header + $NREF random particle rows -> ${REFSTAR##*/}"
    echo "  5. relion_reconstruct --i ${REFSTAR##*/} --o $REF_OUT --3d_rot --ctf"; }
    echo "DRY-RUN — nothing changed. Re-run with --execute to convert."
    exit 0
fi

# ---- environment -------------------------------------------------------------
module load "$RELION_MODULE" 2>/dev/null || { echo "ERROR: could not 'module load $RELION_MODULE'"; exit 1; }
command -v relion_convert_star >/dev/null 2>&1 || { echo "ERROR: relion_convert_star not on PATH after module load."; exit 1; }
shopt -s extglob

cd "$PROJECT_DIR" || { echo "ERROR: could not cd into $PROJECT_DIR"; exit 1; }

# ---- 2. rewrite paths + convert ----------------------------------------------
cp "$STARFILE" temp.star
sed -i "s|$PATH_MATCH|$PARTICLEDIR|g" temp.star
echo "Rewrote '$PATH_MATCH' -> '$PARTICLEDIR' in the particle paths."

relion_convert_star --i temp.star --o "$CONV" --Cs "$CS" --Q0 "$Q0" \
    || { echo "ERROR: relion_convert_star failed."; rm -f temp.star; exit 1; }
rm -f temp.star
echo "Wrote converted star: $CONV"

if [ "$MAKE_REF" != "1" ]; then
    echo "MAKE_REF=0 — skipping the initial reference."
    echo "Finished: $(date)"
    exit 0
fi

# ---- 3. split the star header from the data rows -----------------------------
# The header is every line up to and including the LAST '_rln...' label (the last
# such label is the last data_particles column, since data_particles is the final
# block). Data rows start at the first non-blank line after that. Overridable.
if [ -n "$HEADER_LINES" ] && [ -n "$DATA_START" ]; then
    hdr="$HEADER_LINES"; dstart="$DATA_START"
else
    hdr=$(grep -nE '^[[:space:]]*_rln' "$CONV" | tail -1 | cut -d: -f1)
    if [ -z "$hdr" ]; then
        echo "ERROR: no '_rln' labels found in $CONV — not a RELION star?"; exit 1
    fi
    dstart=$(awk -v h="$hdr" 'NR>h && NF>0 {print NR; exit}' "$CONV")
    [ -n "$dstart" ] || { echo "ERROR: no particle data rows after the header."; exit 1; }
fi
ndata=$(tail -n +"$dstart" "$CONV" | grep -c '[^[:space:]]')
echo "Header = $hdr lines; data starts at line $dstart; ~$ndata particle rows."
if [ "$ndata" -lt "$NREF" ]; then
    echo "NOTE: only ~$ndata particles (< NREF=$NREF) — using all of them."
fi

head -n "$hdr" "$CONV" > "$REFSTAR"
tail -n +"$dstart" "$CONV" | shuf -n "$NREF" >> "$REFSTAR"
echo "Wrote reference subset star: $REFSTAR"

# ---- 4. reconstruct the initial reference ------------------------------------
relion_reconstruct --i "$REFSTAR" --o "$REF_OUT" --3d_rot --ctf \
    || { echo "ERROR: relion_reconstruct failed."; exit 1; }
echo "==================================================================="
echo "Wrote initial reference: $PROJECT_DIR/$REF_OUT"
echo "Use it as REF_MAP in the 'RELION 4: Class3D handoff' step (REF_ANGPIX ="
echo "the export OUTPUT_ANGPIX, since this reference is already at particle scale)."
echo "Finished: $(date)"
echo "==================================================================="
