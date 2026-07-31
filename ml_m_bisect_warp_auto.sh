#!/bin/bash
# ml_m_bisect_warp_auto.sh — find the tilt series that crashes MCore.
#
# ###########################################################################
# # TRY ml_m_check_ctf.py FIRST. This script is the LAST resort.            #
# #                                                                         #
# # The known cause of "IndexOutOfRangeException ... b__19(Int32 t)" is a   #
# # tilt series that was never CTF-estimated: MPA refinement builds a       #
# # per-tilt array from the CTF and runs off the end of the empty one.      #
# # ml_m_check_ctf.py finds those in SECONDS by reading the .xml metadata,  #
# # where this script needs ~9 rounds of GPU time to reach the same answer. #
# #                                                                         #
# #     python3 ml_m_check_ctf.py <project>                                 #
# #                                                                         #
# # On EML45 that was 2 series out of 290; re-running ts_ctf on them fixed  #
# # the crash outright. Only bisect if the CTF pre-flight says PASS and     #
# # MCore still dies.                                                       #
# ###########################################################################
#
# THE PROBLEM
#   MCore can die inside PerformMultiParticleRefinement with a bare
#   "IndexOutOfRangeException ... b__19(Int32 t)" and NO indication of which tilt
#   series it was processing. The exception is thrown in a worker, the series name
#   never reaches stdout, and with several workers the order is not even
#   deterministic. Manually halving a 290-series population is ~9 rounds of fiddly
#   setup done by hand.
#
# WHAT THIS DOES
#   Binary-searches the series list. Each round it builds a THROWAWAY population
#   containing only the candidate subset, attaches the same species, and runs a
#   minimal refinement (--refine_particles only — the bisection already showed the
#   crash does not depend on which parameters are refined). Crash => the culprit is
#   in this half; clean => it is in the other half. Repeats until one series is left.
#
#   Everything is created under <project>/m_bisect/ and can be deleted afterwards.
#   Your real population is never touched.
#
# Usage:
#   SPECIES_HALF1=... SPECIES_HALF2=... SPECIES_MASK=... SPECIES_PARTICLES=... \
#   SPECIES_DIAMETER=150 \
#   bash ml_m_bisect_warp_auto.sh <project_dir> [--execute]
#
#   Required env (same values you gave 'M: create species'):
#     SPECIES_HALF1/HALF2   unfiltered half maps
#     SPECIES_MASK          binary mask
#     SPECIES_PARTICLES     RELION run_data.star
#     SPECIES_DIAMETER      particle diameter in A
#   Optional:
#     SPECIES_SYM           default C1
#     SPECIES_ANGPIX        --angpix_resample (default: leave to M)
#     SETTINGS              default warp_tiltseries.settings
#     GPUS                  default "0 1 2 3"
#     SERIES_LIST           file with one tilt-series name per line (default: all
#                           *.tomostar found under tomostar/)
#
#   Default is a DRY RUN that shows the plan and the first command. --execute runs it.

set -u

# ---- make MTools/MCore reachable ------------------------------------------
# They live in the user's conda env, and a plain `bash script.sh` has neither that
# env nor lmod's `module` shell function (it is only defined in a LOGIN shell). So
# if MTools is missing, re-exec ONCE through `bash -l` with the launcher applied.
# Override with WARP_LAUNCH if your env is named differently.
WARP_LAUNCH="${WARP_LAUNCH:-module load miniconda/latest && conda activate warp}"
if ! command -v MTools >/dev/null 2>&1 && [ -z "${_M_BISECT_ENV:-}" ]; then
    export _M_BISECT_ENV=1
    _relaunch=$(printf '%q' "$0")
    for _a in "$@"; do _relaunch="$_relaunch $(printf '%q' "$_a")"; done
    exec bash -lc "$WARP_LAUNCH && exec bash $_relaunch"
fi
if ! command -v MTools >/dev/null 2>&1; then
    echo "ERROR: MTools is not on PATH, even after:" >&2
    echo "    $WARP_LAUNCH" >&2
    echo "Run this from a shell where the warp env is active, or set WARP_LAUNCH." >&2
    exit 2
fi

ROOT="${1:-}"
EXECUTE=0
for a in "$@"; do [ "$a" = "--execute" ] && EXECUTE=1; done
[ -z "$ROOT" ] || [ ! -d "$ROOT" ] && {
    echo "usage: bash ml_m_bisect_warp_auto.sh <project_dir> [--execute]" >&2; exit 2; }
cd "$ROOT" || exit 2

SETTINGS="${SETTINGS:-warp_tiltseries.settings}"
GPUS="${GPUS:-0 1 2 3}"
SYM="${SPECIES_SYM:-C1}"
WORK="m_bisect"

for v in SPECIES_HALF1 SPECIES_HALF2 SPECIES_MASK SPECIES_PARTICLES SPECIES_DIAMETER; do
    eval "val=\${$v:-}"
    [ -z "$val" ] && { echo "ERROR: $v is not set (see the header of this script)." >&2; exit 2; }
done

# ---- the candidate list ---------------------------------------------------
# (a read loop, not mapfile — mapfile is bash 4+ and this must also run where the
#  script is developed/tested)
ALL=()
if [ -n "${SERIES_LIST:-}" ] && [ -f "${SERIES_LIST}" ]; then
    while IFS= read -r line; do
        [ -n "$line" ] && ALL+=("$line")
    done < "$SERIES_LIST"
else
    while IFS= read -r line; do
        [ -n "$line" ] && ALL+=("${line%.tomostar}")
    done < <(ls -1 tomostar/*.tomostar 2>/dev/null | xargs -n1 basename 2>/dev/null)
fi
N=${#ALL[@]}
[ "$N" -eq 0 ] && { echo "ERROR: no tilt series found under tomostar/." >&2; exit 2; }

echo "==================================================================="
echo "ml_m_bisect_warp_auto    [$([ $EXECUTE -eq 1 ] && echo EXECUTE || echo DRY-RUN)]"
echo "Project : $(pwd)"
echo "Series  : $N"
echo "Species : $SPECIES_HALF1"
echo "          $SPECIES_PARTICLES  (diameter $SPECIES_DIAMETER, sym $SYM)"
echo "Rounds  : ~$(python3 -c "import math;print(math.ceil(math.log2($N)))" 2>/dev/null || echo 9) (binary search)"
echo "Workdir : $WORK/  (throwaway; your real population is untouched)"
echo "==================================================================="

# ---- fail fast on missing inputs (cheaper than discovering it mid-round) ---
_missing=0
for f in "$SPECIES_HALF1" "$SPECIES_HALF2" "$SPECIES_MASK" "$SPECIES_PARTICLES" "$SETTINGS"; do
    [ -e "$f" ] || { echo "ERROR: not found: $f" >&2; _missing=1; }
done
[ "$_missing" -eq 1 ] && { echo "Fix the paths above (they are relative to $(pwd))." >&2; exit 2; }

# ---- run MCore against a subset; return 0 = clean, 1 = crashed ------------
test_subset() {
    local tag="$1"; shift
    local -a members=("$@")
    local dir="$WORK/$tag"
    rm -rf "$dir"; mkdir -p "$dir"

    # a file list for create_source --files (paths relative to the project root)
    local flist="$dir/series.txt"
    : > "$flist"
    for s in "${members[@]}"; do echo "tomostar/$s.tomostar" >> "$flist"; done

    echo "    building population for ${#members[@]} series..." >&2
    # NEVER swallow setup output — a silent "setup failed" is useless. Every step
    # logs, and a failure prints the tail of its own log.
    if ! MTools create_population --directory "$dir" --name t >"$dir/population.log" 2>&1; then
        echo "    !! create_population FAILED:" >&2
        tail -15 "$dir/population.log" | sed 's/^/       /' >&2
        return 2
    fi
    if ! MTools create_source --name t --population "$dir/t.population" \
            --processing_settings "$SETTINGS" --files "$flist" \
            >"$dir/source.log" 2>&1; then
        echo "    !! create_source FAILED:" >&2
        tail -20 "$dir/source.log" | sed 's/^/       /' >&2
        if grep -qiE "unrecognized|unknown option|not a valid|required" "$dir/source.log"; then
            echo "       ^ if --files is rejected, MTools cannot subset this way;" >&2
            echo "         tell Claude and use the subset-tomostar-dir approach." >&2
        fi
        return 2
    fi
    local angpix=()
    [ -n "${SPECIES_ANGPIX:-}" ] && angpix=(--angpix_resample "$SPECIES_ANGPIX")
    MTools create_species --population "$dir/t.population" --name s \
        --diameter "$SPECIES_DIAMETER" --sym "$SYM" --temporal_samples 1 \
        --half1 "$SPECIES_HALF1" --half2 "$SPECIES_HALF2" --mask "$SPECIES_MASK" \
        --particles_relion "$SPECIES_PARTICLES" "${angpix[@]}" --lowpass 10 \
        --ignore_unmatched \
        >"$dir/species.log" 2>&1 || {
            echo "    !! create_species FAILED:" >&2
            tail -20 "$dir/species.log" | sed 's/^/       /' >&2
            return 2; }
    # A SUBSET population always has unmatched particles, and WITHOUT
    # --ignore_unmatched create_species refuses, EXITS 0, and leaves an empty species
    # dir. MCore then has nothing to refine and also exits 0 — which this script
    # scored as "this half is clean". That is how an entire 9-round bisection
    # returned nine false negatives and a wrong conclusion. Verify the artefact.
    if ! ls "$dir"/species/*/*.species >/dev/null 2>&1; then
        echo "    !! no .species was written — the species is EMPTY, so a refinement" >&2
        echo "       here would exit 0 without doing anything (a false 'clean')." >&2
        tail -5 "$dir/species.log" | sed 's/^/       /' >&2
        return 2
    fi

    # A crashed MCore leaves an orphan holding its REST port, and the NEXT MCore
    # then dies at startup with "address already in use" — which this script would
    # have scored as "this half crashed". That single mistake invalidated a whole
    # bisection once. So: reap orphans, and give every round its own port.
    pkill -x -u "$(id -un)" MCore >/dev/null 2>&1
    pkill -x -u "$(id -un)" WarpWorker >/dev/null 2>&1
    sleep 2
    local port=$(( 14300 + (RANDOM % 400) ))
    echo "    refining... (port $port)" >&2
    MCore --population "$dir/t.population" --min_particles 20 --refine_particles \
        --devicelist $GPUS --perdevice_refine 1 --port "$port" \
        >"$dir/mcore.log" 2>&1
    local rc=$?
    # Distinguish THE crash we are hunting from any other failure. Treating every
    # non-zero exit as "the bug" produced a false confirmation once: a single-series
    # population aborted with SIGABRT (134) for an unrelated reason and was reported
    # as the culprit. Return 1 only for the index crash; 3 = failed differently.
    # "0/0" species means MCore did nothing at all — never score that as clean.
    if grep -qE "^0/0|Gathering intermediate results.*\n?0/0" "$dir/mcore.log" \
       && ! grep -qE "[0-9]+\.[0-9]+ (A|Å)" "$dir/mcore.log"; then
        echo "    !! MCore refined 0 species (no resolution reported) — not a valid" >&2
        echo "       test. Check $dir/mcore.log." >&2
        return 2
    fi
    if grep -q "IndexOutOfRangeException" "$dir/mcore.log"; then
        echo "    -> IndexOutOfRangeException (the bug we are hunting)" >&2
        return 1
    fi
    if grep -q "address already in use" "$dir/mcore.log"; then
        echo "    !! MCore could not start: its REST port was still held by an" >&2
        echo "       orphaned process. This is NOT a data problem. Kill them with:" >&2
        echo "         pkill -u \$USER -f MCore; pkill -u \$USER -f WarpWorker" >&2
        return 2
    fi
    if [ $rc -ne 0 ]; then
        echo "    -> exit $rc, but NOT the index crash. First error line:" >&2
        grep -m1 -iE "exception|error|terminate called|cuFFT" "$dir/mcore.log" \
            | sed 's/^/       /' >&2
        echo "       full log: $dir/mcore.log" >&2
        return 3
    fi
    return 0
}

if [ "$EXECUTE" -ne 1 ]; then
    echo
    echo "DRY-RUN. Round 1 would test these ${#ALL[@]} series as one half:"
    printf '    %s\n' "${ALL[@]:0:5}" ; echo "    ..."
    echo
    echo "Re-run with --execute to start the search. Each round builds a small"
    echo "population + species (a few minutes) and runs a minimal refinement."
    exit 0
fi

# ---- confirm the FULL set actually reproduces the crash -------------------
echo
echo "Round 0 — confirming the full set still crashes..."
test_subset all "${ALL[@]}"
case $? in
    0) echo "The full set refined CLEANLY. Nothing to bisect — the crash is not in"
       echo "the series list (try your real population again, or reset M)."; exit 0 ;;
    2) echo "Setup failed before refinement — fix that first (see the log above)."; exit 1 ;;
esac
echo "  confirmed: crashes."

# ---- binary search --------------------------------------------------------
CAND=("${ALL[@]}")
ROUND=1
while [ ${#CAND[@]} -gt 1 ]; do
    HALF=$(( ${#CAND[@]} / 2 ))
    LEFT=("${CAND[@]:0:$HALF}")
    RIGHT=("${CAND[@]:$HALF}")
    echo
    echo "Round $ROUND — ${#CAND[@]} candidates; testing first ${#LEFT[@]}"
    echo "    (${LEFT[0]} … ${LEFT[$((${#LEFT[@]}-1))]})"
    test_subset "r$ROUND" "${LEFT[@]}"
    case $? in
        1) echo "  -> crashed: culprit is in THIS half"; CAND=("${LEFT[@]}") ;;
        0) echo "  -> clean: culprit is in the OTHER half";  CAND=("${RIGHT[@]}") ;;
        2) echo "  setup failed; aborting."; exit 1 ;;
        3) echo "  !! this half failed for a DIFFERENT reason (see above)."
           echo "     The search cannot continue reliably — that failure would be"
           echo "     mistaken for the bug. Investigate it first."; exit 1 ;;
    esac
    ROUND=$((ROUND+1))
done

# ---- VERIFY. The search only ever tests one half and INFERS the other, so a
# ---- crash that depends on the NUMBER of series (not on any single one) makes
# ---- every round report "clean" and the walk lands on the last element purely by
# ---- construction. Testing the survivor on its own is what tells the two apart.
echo
echo "Verifying — testing ${CAND[0]} on its own..."
test_subset verify "${CAND[0]}"
VERDICT=$?
echo
echo "==================================================================="
if [ "$VERDICT" -eq 1 ]; then
    echo "CULPRIT CONFIRMED: ${CAND[0]}"
    echo "  (it throws the SAME IndexOutOfRangeException on its own)"
elif [ "$VERDICT" -eq 3 ]; then
    echo "UNCLEAR: ${CAND[0]} fails on its own, but with a DIFFERENT error than the"
    echo "  one we are hunting (see above). It is still the odd one out — its"
    echo "  neighbours refined cleanly alone — so excluding it is worth trying, but"
    echo "  this is not proof it causes the IndexOutOfRangeException."
elif [ "$VERDICT" -eq 0 ]; then
    echo "NOT CONFIRMED: ${CAND[0]} refines CLEANLY on its own."
    echo
    echo "  Every round reported 'clean', so the search walked to the last series"
    echo "  by construction. That means NO SINGLE SERIES is bad — the crash depends"
    echo "  on the SIZE of the population (only the full set failed; every subset"
    echo "  passed). This looks like a limit inside MCore, not bad data."
    echo
    echo "  WORKAROUND: refine in two populations of ~half the series each. You keep"
    echo "  all the data and all the particles; you only lose the cross-series"
    echo "  consistency of a single joint refinement."
    echo
    echo "  To find the threshold, re-run with SERIES_LIST files of 150, 200, 250"
    echo "  series and see where it starts failing."
else
    echo "INCONCLUSIVE: setup failed while verifying ${CAND[0]} (see $WORK/verify/)."
fi
echo "==================================================================="
[ "$VERDICT" -eq 1 ] || [ "$VERDICT" -eq 3 ] || exit 0
echo "Exclude it and re-run your real refinement. Quickest way:"
echo "    move tomostar/${CAND[0]}.tomostar aside, then rebuild the M population"
echo "    (create_population -> create_source -> create_species)."
echo
echo "NOTE there may be MORE than one bad series — if the next run still crashes,"
echo "re-run this script with the culprit already removed."
echo "Logs for every round are under $WORK/."
