#!/bin/bash
# ml_m_setup_warp_auto.sh — build an M population from the tilt series that carry
# enough particles, then print the first refinement command.
#
# THE THRESHOLD IS NOT A CRASH WORKAROUND
#   Two earlier versions of this header were wrong about MCore's
#   IndexOutOfRangeException, and both wrong explanations cost real time:
#     * "MCore fails above ~145 series"  — wrong. An artefact of a bisection whose
#       every round was a silent no-op (missing --ignore_unmatched).
#     * "corrupt M metadata, fixed by a reset" — wrong. Every run that "succeeded"
#       after the reset had been given --iter 0, which performs no refinement.
#
#   THE ACTUAL CAUSE, found on EML45: two tilt series out of 290 had never been
#   CTF-estimated (CTFResolutionEstimate="0"). MPA refinement builds a PER-TILT
#   array from the CTF and runs off the end of the empty one. Re-running ts_ctf on
#   those two series fixed it outright, at the full 290, with no threshold at all.
#
#   So if MCore throws IndexOutOfRangeException, run the CTF pre-flight FIRST:
#       python3 ml_m_check_ctf.py <project>
#   (or the 'M: pre-flight — CTF completeness' card). Do not shrink the population
#   hoping the crash goes away — shrinking it just changes the odds of including a
#   bad series, which is exactly what made the first explanation look plausible.
#
#   The threshold remains useful for its own reason: series with a handful of
#   particles cannot constrain a 6x4 warp grid, so dropping them costs almost no
#   particles and removes the least-determined terms from the joint fit.
#
#   EML45 example (threshold: series kept / particles kept):
#       20 -> 276 / 99.4%     50 -> 229 / 93.3%     80 -> 167 / 78.1%
#      100 -> 124 / 63.5%    120 ->  81 / 45.9%
#
# Usage:
#   MIN_PARTICLES=80 \
#   SPECIES_HALF1=... SPECIES_HALF2=... SPECIES_MASK=... SPECIES_PARTICLES=... \
#   SPECIES_DIAMETER=150 [SPECIES_SYM=C1] [SPECIES_ANGPIX=1.57] [POP_NAME=...] \
#   bash ml_m_setup_warp_auto.sh <project_dir> [--execute]
#
#   Dry run by default: shows how many series/particles survive and what it would
#   run. --execute builds the population.
#
# Everything lands in <project>/m_<POP_NAME>/ so it never collides with a previous
# attempt. Existing populations are left alone.

set -u

WARP_LAUNCH="${WARP_LAUNCH:-module load miniconda/latest && conda activate warp}"
if ! command -v MTools >/dev/null 2>&1 && [ -z "${_M_SETUP_ENV:-}" ]; then
    export _M_SETUP_ENV=1
    _r=$(printf '%q' "$0"); for _a in "$@"; do _r="$_r $(printf '%q' "$_a")"; done
    exec bash -lc "$WARP_LAUNCH && exec bash $_r"
fi

ROOT="${1:-}"
EXECUTE=0
for a in "$@"; do [ "$a" = "--execute" ] && EXECUTE=1; done
[ -z "$ROOT" ] || [ ! -d "$ROOT" ] && {
    echo "usage: bash ml_m_setup_warp_auto.sh <project_dir> [--execute]" >&2; exit 2; }
cd "$ROOT" || exit 2

MIN_PARTICLES="${MIN_PARTICLES:-80}"
SETTINGS="${SETTINGS:-warp_tiltseries.settings}"
SYM="${SPECIES_SYM:-C1}"
POP_NAME="${POP_NAME:-min${MIN_PARTICLES}}"
GPUS="${GPUS:-1 2 3}"
PORT="${PORT:-14372}"
DIR="m_${POP_NAME}"

for v in SPECIES_HALF1 SPECIES_HALF2 SPECIES_MASK SPECIES_PARTICLES SPECIES_DIAMETER; do
    eval "val=\${$v:-}"
    [ -z "$val" ] && { echo "ERROR: $v is not set (see this script's header)." >&2; exit 2; }
done
_miss=0
for f in "$SPECIES_HALF1" "$SPECIES_HALF2" "$SPECIES_MASK" "$SPECIES_PARTICLES" "$SETTINGS"; do
    [ -e "$f" ] || { echo "ERROR: not found: $f" >&2; _miss=1; }
done
[ "$_miss" -eq 1 ] && exit 2

# ---- which series clear the threshold -------------------------------------
# Particle rows name their subtomogram as .../subtomo/<Series>/<Series>_NNN_*.mrc,
# so the second-to-last path component is the tilt series.
# mktemp, not a fixed /tmp name: on a shared node another user's identically
# named file is unwritable — and the redirection failing would leave THEIR stale
# series list to be read below.
LIST=$(mktemp "${TMPDIR:-/tmp}/m_series_${POP_NAME}.XXXXXX")
awk -v T="$MIN_PARTICLES" '
    /\.mrc/ { split($0, a, "/"); n[a[length(a)-1]]++ }
    END { for (p in n) if (n[p] >= T) print p }
' "$SPECIES_PARTICLES" | sort > "$LIST"

KEPT=$(wc -l < "$LIST")
read TOTSER TOTPART KEPTPART <<EOF
$(awk -v T="$MIN_PARTICLES" '
    /\.mrc/ { split($0, a, "/"); n[a[length(a)-1]]++ }
    END { s=0; t=0; k=0
          for (p in n) { t += n[p]; k++; if (n[p] >= T) s += n[p] }
          print k, t, s }
' "$SPECIES_PARTICLES")
EOF

echo "==================================================================="
echo "ml_m_setup_warp_auto    [$([ $EXECUTE -eq 1 ] && echo EXECUTE || echo DRY-RUN)]"
echo "Project        : $(pwd)"
echo "Threshold      : >= $MIN_PARTICLES particles per series"
echo "Series kept    : $KEPT of $TOTSER"
echo "Particles kept : $KEPTPART of $TOTPART ($(awk -v a=$KEPTPART -v b=$TOTPART 'BEGIN{printf "%.1f", 100*a/b}')%)"
echo "Population     : $DIR/${POP_NAME}.population"
echo "==================================================================="
[ "$KEPT" -eq 0 ] && { echo "No series clear the threshold — lower MIN_PARTICLES." >&2; exit 2; }

# A series that was never CTF-estimated crashes MPA refinement with
# IndexOutOfRangeException and never names itself. Catch it here rather than an
# hour into MCore. Advisory only — the check is cheap and the fix is ts_ctf.
_CHECK="$(dirname "$0")/ml_m_check_ctf.py"
if [ -f "$_CHECK" ]; then
    if ! python3 "$_CHECK" . --quiet >/tmp/m_ctf_check_$$.txt 2>&1; then
        echo
        sed 's/^/   /' /tmp/m_ctf_check_$$.txt
        echo
        echo "!! Fix the CTF above BEFORE building this population — a data source"
        echo "   caches tilt-series metadata, so it would bake the fault in."
        rm -f /tmp/m_ctf_check_$$.txt
        exit 1
    fi
    rm -f /tmp/m_ctf_check_$$.txt
    echo "CTF pre-flight: all tilt series carry a CTF estimate."
fi

# tilt series that have particles but no .tomostar would break create_source
_gone=0
while IFS= read -r s; do
    [ -e "tomostar/$s.tomostar" ] || { echo "!! no tomostar for $s" >&2; _gone=1; }
done < "$LIST"
[ "$_gone" -eq 1 ] && echo "   (those series will be skipped by create_source)"

FILES="/tmp/m_files_${POP_NAME}.txt"
sed 's|^|tomostar/|; s|$|.tomostar|' "$LIST" > "$FILES"

ANGPIX_ARG=""
[ -n "${SPECIES_ANGPIX:-}" ] && ANGPIX_ARG="--angpix_resample $SPECIES_ANGPIX"

REFINE="MCore --population $DIR/${POP_NAME}.population --min_particles $MIN_PARTICLES \\
    --refine_imagewarp 6x4 --refine_particles --ctf_defocus --ctf_defocusexhaustive \\
    --devicelist $GPUS --perdevice_refine 2 --port $PORT"

if [ "$EXECUTE" -ne 1 ]; then
    echo "Would run:"
    echo "  1. MTools create_population --directory $DIR --name $POP_NAME"
    echo "  2. MTools create_source --name $POP_NAME --population $DIR/${POP_NAME}.population \\"
    echo "         --processing_settings $SETTINGS --files $FILES"
    echo "  3. MTools create_species --population $DIR/${POP_NAME}.population --name spike \\"
    echo "         --diameter $SPECIES_DIAMETER --sym $SYM --temporal_samples 1 \\"
    echo "         --half1 $SPECIES_HALF1 --half2 $SPECIES_HALF2 \\"
    echo "         --mask $SPECIES_MASK --particles_relion $SPECIES_PARTICLES \\"
    echo "         $ANGPIX_ARG --lowpass 10 --ignore_unmatched"
    echo
    echo "Series list: $LIST"
    echo "DRY-RUN — nothing built. Add --execute to build it."
    exit 0
fi

# a stale MCore holds port 14300 and kills the NEXT run at startup
pkill -x -u "$(id -un)" MCore  >/dev/null 2>&1
pkill -x -u "$(id -un)" WarpWorker >/dev/null 2>&1
sleep 2

[ -e "$DIR/${POP_NAME}.population" ] && {
    echo "ERROR: $DIR/${POP_NAME}.population already exists." >&2
    echo "  create_population LOADS an existing population (and fails if anything it" >&2
    echo "  references has moved). Use a different POP_NAME, or move $DIR aside." >&2
    exit 2; }

echo "1/3 create_population..."
MTools create_population --directory "$DIR" --name "$POP_NAME" || exit 1
echo "2/3 create_source ($KEPT series)..."
MTools create_source --name "$POP_NAME" --population "$DIR/${POP_NAME}.population" \
    --processing_settings "$SETTINGS" --files "$FILES" || exit 1
echo "3/3 create_species..."
# --ignore_unmatched is REQUIRED here, not optional. The particle star covers every
# tilt series, but this population deliberately contains only those above the
# threshold — so the particles of the excluded series have nowhere to go. Without
# the flag create_species REFUSES to build ("N particles couldn't be matched to data
# source"), creates an empty species dir, and MCore then reports 0/0 species with no
# error of its own. The unmatched particles are exactly the ones we chose to drop.
MTools create_species --population "$DIR/${POP_NAME}.population" --name spike \
    --diameter "$SPECIES_DIAMETER" --sym "$SYM" --temporal_samples 1 \
    --half1 "$SPECIES_HALF1" --half2 "$SPECIES_HALF2" --mask "$SPECIES_MASK" \
    --particles_relion "$SPECIES_PARTICLES" $ANGPIX_ARG --lowpass 10 \
    --ignore_unmatched 2>&1 | tee "$DIR/create_species.log"
[ "${PIPESTATUS[0]}" -ne 0 ] && { echo "create_species FAILED (see $DIR/create_species.log)"; exit 1; }

# create_species can exit 0 having written NOTHING, so verify the artefact exists.
if ! ls "$DIR"/species/*/*.species >/dev/null 2>&1; then
    echo
    echo "ERROR: no .species file was written — the species is empty." >&2
    echo "Last lines of $DIR/create_species.log:" >&2
    tail -8 "$DIR/create_species.log" | sed 's/^/    /' >&2
    exit 1
fi

echo
echo "==================================================================="
echo "Population ready: $DIR/${POP_NAME}.population"
echo "==================================================================="
echo "Check it imported (no refinement, reports the starting resolution):"
echo "    MCore --population $DIR/${POP_NAME}.population --iter 0 --port $PORT"
echo
echo "Then round 1:"
echo "    $REFINE"
echo
echo "If IndexOutOfRangeException returns, do NOT shrink the population — run"
echo "    python3 ml_m_check_ctf.py ."
echo "A tilt series with no CTF estimate is the known cause, and it never names"
echo "itself in the exception."
