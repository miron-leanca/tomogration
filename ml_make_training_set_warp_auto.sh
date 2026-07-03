#!/bin/bash
# ml_make_training_set_warp_auto.sh
#
# Build a FRESH miss-alignment training subset (combined/selected/) from a chosen
# set of combined Position### series. The existing selected/ is ARCHIVED (moved
# aside, never deleted), then a clean selected/ is filled with only the requested
# positions' frames + mdocs (+ gain, thumbnails, listfile subset) so you can start
# training over on a hand-picked, high-quality set.
#
# Usage:
#   bash ml_make_training_set_warp_auto.sh <combined_dir> <pos> [<pos> ...] [--execute]
#     <pos>   combined Position number (e.g. 42) or zero-padded name (Position042).
#   Default = DRY-RUN (prints the plan); add --execute to actually move/copy.
#
# Example (the 15 "great" series):
#   bash ml_make_training_set_warp_auto.sh \
#     /ceph/users/haq21239/EMDatasets/EML45/combined \
#     42 46 48 49 85 131 161 185 190 206 266 267 270 273 280
#   (review, then re-run with --execute appended)
set -e

EXECUTE=0
ARGS=()
for a in "$@"; do
    if [ "$a" = "--execute" ]; then EXECUTE=1; else ARGS+=("$a"); fi
done
COMBINED="${ARGS[0]:?Usage: $0 <combined_dir> <pos> ... [--execute]}"
POSITIONS=("${ARGS[@]:1}")
[ ${#POSITIONS[@]} -ge 1 ] || { echo "Give at least one position number."; exit 1; }

FR="$COMBINED/frames"
MD="$COMBINED/mdocs"
TH="$COMBINED/Thumbnails"
SEL="$COMBINED/selected"
[ -d "$MD" ] || { echo "ERROR: $MD missing — is '$COMBINED' the merged project?"; exit 1; }

mode=$([ $EXECUTE -eq 1 ] && echo EXECUTE || echo DRY-RUN)
echo "==================================================================="
echo "ml_make_training_set_warp_auto    [$mode]"
echo "combined: $COMBINED"
echo "positions (${#POSITIONS[@]}): ${POSITIONS[*]}"
echo "==================================================================="

# --- 1. archive any existing selected/ -----------------------------------------
if [ -d "$SEL" ]; then
    STAMP=$(date +%Y%m%d_%H%M%S)
    ARCH="${SEL}_archived_${STAMP}"
    echo "existing selected/  ->  $(basename "$ARCH")   (archived, not deleted)"
    [ $EXECUTE -eq 1 ] && mv "$SEL" "$ARCH"
fi
echo "create: $SEL/{frames,mdocs}"
if [ $EXECUTE -eq 1 ]; then
    mkdir -p "$SEL/frames" "$SEL/mdocs"
fi

# --- 2. copy gain(s) -----------------------------------------------------------
gain_done=0
for g in "$COMBINED"/*.gain "$COMBINED"/gain/*.gain; do
    [ -f "$g" ] || continue
    echo "gain: $(basename "$g")  ->  selected/"
    [ $EXECUTE -eq 1 ] && cp -n "$g" "$SEL/"
    gain_done=1
done
[ $gain_done -eq 0 ] && echo "NOTE: no *.gain found under $COMBINED (copy it into selected/ yourself)."

# --- 3. copy each position's frames + mdoc (+ thumbnail, listfile line) ---------
LF="$COMBINED/listfile_Position.txt"
SEL_LF="$SEL/listfile_Position.txt"
[ $EXECUTE -eq 1 ] && [ -f "$LF" ] && \
    echo "# training-set subset (ml_make_training_set_warp_auto.sh)" > "$SEL_LF"

ok=0; miss=0; eer_total=0
for p in "${POSITIONS[@]}"; do
    num=$(printf '%s' "$p" | sed 's/Position//')
    case "$num" in ''|*[!0-9]*) echo "  skip bad position '$p'"; continue;; esac
    pos=$(printf 'Position%03d' "$num")
    mdoc="$MD/$pos.mdoc"
    if [ ! -f "$mdoc" ]; then
        echo "  MISS: $pos has no mdoc in $MD — skipped"; miss=$((miss+1)); continue
    fi
    # count eer for this series (frames/ first, then combined root)
    n_eer=$(ls "$FR/${pos}_"*.eer 2>/dev/null | wc -l | tr -d ' ')
    [ "$n_eer" -eq 0 ] && n_eer=$(ls "$COMBINED/${pos}_"*.eer 2>/dev/null | wc -l | tr -d ' ')
    echo "  $pos : mdoc + $n_eer eer$([ -f "$TH/$pos.mrc" ] && echo ' + thumbnail')"
    if [ $EXECUTE -eq 1 ]; then
        cp "$mdoc" "$SEL/mdocs/"
        for src in "$FR/${pos}_"*.eer "$COMBINED/${pos}_"*.eer; do
            [ -f "$src" ] && cp "$src" "$SEL/frames/"
        done
        [ -f "$TH/$pos.mrc" ] && { mkdir -p "$SEL/Thumbnails"; cp "$TH/$pos.mrc" "$SEL/Thumbnails/"; }
        [ -f "$LF" ] && grep -E "[[:space:]]$pos$" "$LF" >> "$SEL_LF" 2>/dev/null || true
    fi
    ok=$((ok+1)); eer_total=$((eer_total+n_eer))
done

echo "==================================================================="
echo "series ready: $ok    missing: $miss    total eer: $eer_total"
if [ $EXECUTE -eq 0 ]; then
    echo "DRY-RUN — nothing changed. Re-run with --execute to build selected/."
else
    echo "Done. Fresh training set at: $SEL"
    echo "Next: point tomogration at $SEL and run ts_import → ts_stack → miss-alignment."
fi
exit 0
