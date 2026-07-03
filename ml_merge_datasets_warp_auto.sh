#!/bin/bash
# ml_merge_datasets_warp_auto.sh
#
# Merge the three EML45 grids into one combined tomogration dataset with a single
# continuous Position numbering, then build a small 5-per-grid optimisation subset.
#
#   grid7  20260619…grid7   already renamed+sorted, Position001..124  -> kept as-is
#   grid6  20260620…grid6   renamed Position001..084 (unsorted)        -> RENUMBER +124 -> 125..208
#   grid5  20260621…grid5   raw Tomo5 Position_NNN[_bs] (+overrides)    -> RENAME start 208 -> 209..328
#
# All renamed .eer + .mdoc are MOVED into  <BASE>/combined/{frames,mdocs}; the shared
# gain is COPIED into combined/gains; combined/selected/ gets 5 series COPIED from each
# grid (frames+mdocs+gains) to optimise reconstructions on.
#
# Usage:
#   bash ml_merge_datasets_warp_auto.sh             # DRY RUN — prints the plan, no changes
#   bash ml_merge_datasets_warp_auto.sh --execute   # actually do it
#
# Safe to dry-run as often as you like. mv within the same ceph mount is instant
# (metadata only); the only slow part is copying the ~15 selected series.
set -euo pipefail

BASE=/ceph/users/haq21239/EMDatasets/EML45
G7="$BASE/20260619_ML_EML45_OC43_grid7"
G6="$BASE/20260620_ML_EML45_grid6_OC43"
G5="$BASE/20260621_ML_EML45_grid5_OC43"
COMBINED="$BASE/combined"
GAIN_NAME="20260522_114322_EER_GainReference.gain"
N_SELECT=5
RENAME_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ml_batch_rename_eer_mdoc_mrc_warp_auto.sh"

EXECUTE=0
[ "${1:-}" = "--execute" ] && EXECUTE=1
MODE=$([ "$EXECUTE" = 1 ] && echo "EXECUTE" || echo "DRY-RUN")

shopt -s nullglob
say() { echo "$*"; }
hr()  { echo "-------------------------------------------------------------------"; }

echo "==================================================================="
echo "ml_merge_datasets_warp_auto    [$MODE]"
echo "  combined target : $COMBINED"
echo "==================================================================="

# --- sanity ---------------------------------------------------------------
for d in "$G7" "$G6" "$G5"; do
    [ -d "$d" ] || { echo "ERROR: missing grid dir: $d"; exit 1; }
done
[ -f "$RENAME_SCRIPT" ] || { echo "ERROR: rename script not found: $RENAME_SCRIPT"; exit 1; }
if [ -d "$COMBINED/frames" ] && [ -n "$(find "$COMBINED/frames" -maxdepth 1 -name '*.eer' -print -quit)" ]; then
    echo "ERROR: $COMBINED/frames already has .eer — refusing to merge twice."
    echo "       Remove/rename combined/ first if you want to redo this."
    exit 1
fi

# --- offsets --------------------------------------------------------------
g7_max=$(grep -vh '^#' "$G7/listfile_Position.txt" 2>/dev/null | awk '{print $NF}' \
         | sed 's/Position//;s/\.mdoc//' | grep -E '^[0-9]+$' | sort -n | tail -1)
g7_max=${g7_max:-124}
g6_mdocs=( "$G6"/Position[0-9][0-9][0-9].mdoc )
g6_count=${#g6_mdocs[@]}
g6_offset=$g7_max
g5_start=$((g7_max + g6_count))

say "grid7 max assigned position : $g7_max   (grid7 kept as Position001..$g7_max)"
say "grid6 series                : $g6_count -> renumber +$g6_offset -> Position$(printf '%03d' $((1+g6_offset)))..Position$(printf '%03d' $((g6_count+g6_offset)))"
say "grid5 rename start_number    : $g5_start -> Position$(printf '%03d' $((g5_start+1)))..(+ grid5 series)"
hr

mkrun() { [ "$EXECUTE" = 1 ] && mkdir -p "$@" || echo "   [dry] mkdir -p $*"; }
mkrun "$COMBINED/frames" "$COMBINED/mdocs" "$COMBINED/gains"

# ==========================================================================
# 1. grid7 — move already-renamed frames/ + mdocs/ into combined (kept 001..124)
# ==========================================================================
hr; say "[1] grid7 -> combined  (move frames/ + mdocs/, keep numbering)"
g7_eer=$(find "$G7/frames" -maxdepth 1 -name '*.eer' | wc -l)
g7_md=$(find "$G7/mdocs" -maxdepth 1 -name 'Position[0-9][0-9][0-9].mdoc' | wc -l)
say "   move $g7_eer .eer  +  $g7_md .mdoc"
if [ "$EXECUTE" = 1 ]; then
    find "$G7/frames" -maxdepth 1 -name '*.eer' -exec mv -t "$COMBINED/frames" {} +
    find "$G7/mdocs"  -maxdepth 1 -name 'Position[0-9][0-9][0-9].mdoc' -exec mv -t "$COMBINED/mdocs" {} +
fi

# ==========================================================================
# 2. grid6 — renumber +offset and move into combined
# ==========================================================================
hr; say "[2] grid6 -> combined  (renumber +$g6_offset, move)"
shown=0
for md in "${g6_mdocs[@]}"; do
    base=$(basename "$md" .mdoc)                      # Position001
    n=$((10#${base#Position}))
    new=$(printf "Position%03d" $((n + g6_offset)))   # Position125
    eers=( "$G6/${base}"_*.eer )
    if [ "$shown" -lt 3 ]; then
        say "   $base -> $new   (${#eers[@]} .eer)"; shown=$((shown+1))
    fi
    if [ "$EXECUTE" = 1 ]; then
        for f in "${eers[@]}"; do
            bn=$(basename "$f")
            mv "$f" "$COMBINED/frames/${new}${bn#"$base"}"
        done
        sed -i "s/${base}/${new}/g" "$md"
        mv "$md" "$COMBINED/mdocs/${new}.mdoc"
    fi
done
say "   … ($g6_count series renumbered $(printf '%03d' $((1+g6_offset)))..$(printf '%03d' $((g6_count+g6_offset))))"

# ==========================================================================
# 3. grid5 — quarantine overrides, rename (start=$g5_start), move into combined
# ==========================================================================
hr; say "[3] grid5 -> combined  (quarantine overrides, rename start=$g5_start, move)"
g5_over=$(find "$G5" -maxdepth 1 -name '*_override.mdoc' | wc -l)
g5_reg=$(find "$G5" -maxdepth 1 -name '*.mdoc' ! -name '*override*' | wc -l)
say "   quarantine $g5_over *_override.mdoc -> grid5/_overrides_quarantined/"
say "   rename     $g5_reg regular series  -> Position$(printf '%03d' $((g5_start+1)))..Position$(printf '%03d' $((g5_start+g5_reg)))"
if [ "$EXECUTE" = 1 ]; then
    mkdir -p "$G5/_overrides_quarantined"
    find "$G5" -maxdepth 1 -name '*_override.mdoc' -exec mv -t "$G5/_overrides_quarantined" {} +
    bash "$RENAME_SCRIPT" "$G5" Position "$g5_start"
    find "$G5" -maxdepth 1 -name 'Position[0-9][0-9][0-9]_*.eer' -exec mv -t "$COMBINED/frames" {} +
    find "$G5" -maxdepth 1 -name 'Position[0-9][0-9][0-9].mdoc'  -exec mv -t "$COMBINED/mdocs" {} +
fi

# ==========================================================================
# 4. gains — copy the shared gain (+ grid7 reciprocal/original if present)
# ==========================================================================
hr; say "[4] gains -> combined/gains  (copy)"
for g in "$G7/gains/$GAIN_NAME" "$G7/gains/gain_reciprocal.mrc" "$G7/gains/original_gain.mrc"; do
    [ -f "$g" ] || continue
    say "   cp $(basename "$g")"
    [ "$EXECUTE" = 1 ] && cp -n "$g" "$COMBINED/gains/"
done

# ==========================================================================
# 5. selected/ — copy first N existing series from each grid's number range
# ==========================================================================
hr; say "[5] combined/selected/  (copy ${N_SELECT} series from each grid)"
mkrun "$COMBINED/selected/frames" "$COMBINED/selected/mdocs" "$COMBINED/selected/gains"

pick_range() {   # $1=lo $2=hi  -> echo first N existing PositionNNN in [lo,hi] from combined/mdocs
    local lo=$1 hi=$2 c=0 num base
    for md in "$COMBINED"/mdocs/Position[0-9][0-9][0-9].mdoc; do
        [ -e "$md" ] || continue
        base=$(basename "$md" .mdoc); num=$((10#${base#Position}))
        if [ "$num" -ge "$lo" ] && [ "$num" -le "$hi" ]; then
            echo "$base"; c=$((c+1)); [ "$c" -ge "$N_SELECT" ] && break
        fi
    done
}

copy_selected() {  # $1=label $2=lo $3=hi
    local label=$1 lo=$2 hi=$3 base ne
    if [ "$EXECUTE" = 1 ]; then
        local picks; picks=$(pick_range "$lo" "$hi")
    else
        # dry run: combined/mdocs doesn't exist yet — just announce the intended range
        say "   $label: first $N_SELECT of Position$(printf '%03d' "$lo")..Position$(printf '%03d' "$hi")"
        return
    fi
    for base in $picks; do
        ne=$(find "$COMBINED/frames" -maxdepth 1 -name "${base}_*.eer" | wc -l)
        say "   $label: $base ($ne .eer)"
        cp "$COMBINED/mdocs/${base}.mdoc" "$COMBINED/selected/mdocs/"
        find "$COMBINED/frames" -maxdepth 1 -name "${base}_*.eer" -exec cp -t "$COMBINED/selected/frames" {} +
    done
}
copy_selected "grid7" 1 "$g7_max"
copy_selected "grid6" $((1+g6_offset)) $((g6_count+g6_offset))
copy_selected "grid5" $((g5_start+1)) $((g5_start+g5_reg))
if [ "$EXECUTE" = 1 ]; then
    cp -n "$COMBINED/gains/"* "$COMBINED/selected/gains/" 2>/dev/null || true
fi

hr
if [ "$EXECUTE" = 1 ]; then
    echo "DONE. Combined dataset: $COMBINED"
    echo "  frames: $(find "$COMBINED/frames" -maxdepth 1 -name '*.eer' | wc -l) .eer"
    echo "  mdocs : $(find "$COMBINED/mdocs"  -maxdepth 1 -name '*.mdoc' | wc -l) .mdoc"
    echo "  selected: $(find "$COMBINED/selected/mdocs" -maxdepth 1 -name '*.mdoc' 2>/dev/null | wc -l) series"
    echo "Point tomogration's project root at $COMBINED (or .../combined/selected to optimise)."
    echo "grid7's old warp_*/aretomo_output* are left behind (orphaned) — delete when happy."
else
    echo "DRY-RUN complete — nothing was changed."
    echo "Re-run with --execute to perform the merge:"
    echo "    bash $0 --execute"
fi
