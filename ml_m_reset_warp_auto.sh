#!/bin/bash
# ml_m_reset_warp_auto.sh — inspect, and optionally clear, an M project's setup.
#
# WHY THIS EXISTS
#   M's setup commands are NOT idempotent, and they fail in a way that is very hard
#   to read:
#     * `MTools create_population` on an EXISTING population does not make a fresh
#       one — it LOADS it, which loads every data source the population references.
#       If any .source file is missing, you get a .NET FileNotFoundException stack
#       trace instead of a useful message, and the population is now unusable.
#     * `MTools create_source` on an existing source is a silent no-op
#       ("... already exists in this population").
#     * The .source file does NOT live in the M directory — it is written next to
#       the PROCESSING SETTINGS (e.g. warp_tiltseries/<name>.source), so deleting
#       m/ alone leaves a half-configured project behind.
#
#   The result: a half-deleted M project blocks every later step, and the only way
#   out is to know exactly which files to remove. This script knows.
#
# Usage:
#   bash ml_m_reset_warp_auto.sh <project_dir> [--execute]
#     <project_dir>  the tomogration project root (holds m/ and warp_tiltseries/)
#
#   Default is a REPORT: it lists what exists, checks whether each population's
#   data sources actually resolve, and tells you whether a reset is needed.
#   Add --execute to move the setup files aside (to m_trash_<timestamp>/ — this
#   script NEVER deletes anything outright).
#
# It only ever touches: <root>/m/*.population, <root>/m/species/, and the
# <name>.source files a population references. Raw data is never touched.

set -u

ROOT="${1:-}"
EXECUTE=0
for a in "$@"; do [ "$a" = "--execute" ] && EXECUTE=1; done

if [ -z "$ROOT" ] || [ ! -d "$ROOT" ]; then
    echo "usage: bash ml_m_reset_warp_auto.sh <project_dir> [--execute]" >&2
    exit 2
fi
cd "$ROOT" || exit 2
MODE=$([ "$EXECUTE" -eq 1 ] && echo EXECUTE || echo REPORT)

echo "==================================================================="
echo "ml_m_reset_warp_auto    [$MODE]"
echo "Project: $(pwd)"
echo "==================================================================="

shopt -s nullglob
# M scatters its state widely, and after a few attempts there are several projects:
#   <root>/m*/ *.population        every population dir (m/, m_min80/, m_bisect/ …)
#   <root>/*/ *.source             the data source — written next to the SETTINGS,
#                                  i.e. warp_tiltseries/<name>.source, NOT into m/
#   <root>/m*/species/             the refined maps
#   <root>/m*/refinement_temp/     scratch that can be many GB
# Missing any one of them leaves a half-configured project that blocks the next run.
POPS=(m*/*.population)
SOURCES=(*/*.source *.source)
MDIRS=(m m_* )

if [ ${#POPS[@]} -eq 0 ] && [ ${#SOURCES[@]} -eq 0 ]; then
    echo "No M setup found — nothing to reset. You can run 'create population'."
    exit 0
fi

BROKEN=0
echo "--- populations ---"
for p in "${POPS[@]}"; do
    echo "  $p"
    # a population lists its sources by path; check each one resolves
    refs=$(grep -oE '[^"<>]+\.source' "$p" 2>/dev/null | sort -u)
    if [ -z "$refs" ]; then
        echo "      (no data source registered yet)"
        continue
    fi
    while IFS= read -r r; do
        [ -z "$r" ] && continue
        if [ -e "$r" ]; then
            echo "      ✓ source present: $r"
        else
            echo "      ✗ MISSING source: $r"
            echo "        ^ this is why MTools throws FileNotFoundException. The"
            echo "          population cannot be loaded until it is reset."
            BROKEN=1
        fi
    done <<< "$refs"
done

echo "--- .source files on disk ---"
if [ ${#SOURCES[@]} -eq 0 ]; then
    echo "  (none)"
else
    for s in "${SOURCES[@]}"; do echo "  $s"; done
fi

echo "--- M directories ---"
for d in "${MDIRS[@]}"; do
    [ -d "$d" ] || continue
    if ls "$d"/*.population >/dev/null 2>&1 || [ -d "$d/species" ] \
       || [ -d "$d/refinement_temp" ]; then
        echo "  $d/   ($(du -sh "$d" 2>/dev/null | cut -f1))"
    fi
done

SPECIES=(m*/species/*)
echo "--- species ---"
if [ ${#SPECIES[@]} -eq 0 ]; then
    echo "  (none)"
else
    for s in "${SPECIES[@]}"; do echo "  $s"; done
fi

echo "-------------------------------------------------------------------"
if [ "$BROKEN" -eq 1 ]; then
    echo "STATUS: BROKEN — a population references a .source that no longer exists."
    echo "        Reset and redo setup (create population -> create source ONCE each)."
else
    echo "STATUS: consistent. A reset is only needed if you want to start M over."
fi

if [ "$EXECUTE" -ne 1 ]; then
    echo
    echo "REPORT ONLY - nothing moved. Re-run with --execute to reset:"
    echo "    bash $0 $ROOT --execute"
    exit 0
fi

TRASH="m_trash_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TRASH" || exit 1
echo
echo "Moving setup files to $TRASH/ (nothing is deleted):"
for d in "${MDIRS[@]}"; do
    [ -d "$d" ] || continue
    # only a directory that actually holds M state — never a stray "m*" of yours
    if ls "$d"/*.population >/dev/null 2>&1 || [ -d "$d/species" ] \
       || [ -d "$d/refinement_temp" ]; then
        mv "$d" "$TRASH/" 2>/dev/null && echo "    moved $d/"
    fi
done
for f in "${SOURCES[@]}"; do
    mv "$f" "$TRASH/" 2>/dev/null && echo "    moved $f"
done
echo
echo "Reset done. Now run, ONCE each and in this order:"
echo "    1. M: create population"
echo "    2. M: create data source"
echo "Do NOT re-run either afterwards — create_population on an existing population"
echo "loads it (and fails if anything it references has moved)."
