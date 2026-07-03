#!/bin/bash
# ml_batch_remake_mdocs_warp_auto.sh
#
# Removes excluded tilt entries from .mdoc files and renumbers the remaining
# ZValue blocks contiguously. Fixes mdoc date formatting at the end.
#
# Usage:
#   bash ml_batch_remake_mdocs_warp_auto.sh <mdocs_dir> <exclusion_list> <conv_key> [<rootname>]
#
# Arguments:
#   mdocs_dir       Directory containing <rootname>NNN.mdoc files
#   exclusion_list  Text file, one line per position with excluded tilts,
#                   format: "<PositionName>\t<tilt1,tilt2,...>," (tilts in IMOD order)
#   conv_key        new_imod_conv_key.txt mapping IMOD->acquisition order
#   rootname        (optional) File prefix. Default: Position
#
# What it does for each position listed in exclusion_list:
#   1. Translates IMOD tilt numbers to acquisition order via conv_key
#   2. Removes the corresponding ZValue blocks from the .mdoc
#   3. Renumbers remaining ZValue blocks contiguously starting from 0
#
# Then fixes mdoc date formatting:
#   - Shortens 4-digit year to 2 digits (e.g. 2026 -> 26)
#   - Swaps day/year so the format becomes yy-mmm-dd (required by Warp)

set -e

# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    cat <<EOF
Usage: bash $0 <mdocs_dir> <exclusion_list> <conv_key> [<rootname>]

Example:
    bash $0 selected/mdocs selected/exclusion_list.txt selected/new_imod_conv_key.txt
EOF
    exit 1
fi

WORKDIR="$1"
EXCLFILE="$2"
CONV_KEY="$3"
ROOTNAME="${4:-Position}"

if [ ! -d "$WORKDIR" ]; then
    echo "ERROR: Mdoc directory not found: $WORKDIR"
    exit 1
fi
if [ ! -f "$EXCLFILE" ]; then
    echo "ERROR: Exclusion list not found: $EXCLFILE"
    exit 1
fi
if [ ! -f "$CONV_KEY" ]; then
    echo "ERROR: Conversion key not found: $CONV_KEY"
    exit 1
fi

echo "========================================"
echo "ml_batch_remake_mdocs_warp_auto"
echo "Mdocs dir:  $WORKDIR"
echo "Exclusion:  $EXCLFILE"
echo "Conv key:   $CONV_KEY"
echo "Rootname:   $ROOTNAME"
echo "Started:    $(date)"
echo "========================================"

cd "$WORKDIR"

# -----------------------------------------------------------------------------
# Read exclusion list into an associative array: position -> tilts
# -----------------------------------------------------------------------------

declare -A EXCLUDED
position_count=0
while IFS=$'\t ' read -r pos tilts_csv || [ -n "$pos" ]; do
    # Skip blank lines and comments
    [ -z "$pos" ] && continue
    [[ "$pos" =~ ^# ]] && continue
    # Strip trailing comma if any
    tilts_csv="${tilts_csv%,}"
    EXCLUDED["$pos"]="$tilts_csv"
    position_count=$((position_count + 1))
done < "$EXCLFILE"

echo "Loaded exclusions for $position_count positions"

# -----------------------------------------------------------------------------
# Process each mdoc that has exclusions listed
# -----------------------------------------------------------------------------

processed=0
for pos in "${!EXCLUDED[@]}"; do
    mdoc="${pos}.mdoc"
    tilts_csv="${EXCLUDED[$pos]}"

    if [ ! -f "$mdoc" ]; then
        echo "SKIP: $mdoc not found"
        continue
    fi

    echo ""
    echo "Processing $pos - excluding IMOD tilts: $tilts_csv"

    # Translate each IMOD tilt number to acquisition order via conv_key
    for imod_tilt in $(echo "$tilts_csv" | tr ',' '\n'); do
        [ -z "$imod_tilt" ] && continue
        # Grab line <imod_tilt> from the conversion key = acquisition order
        acq=$(sed -n "${imod_tilt}p" "$CONV_KEY")
        if [ -z "$acq" ]; then
            echo "  WARNING: no mapping for IMOD tilt $imod_tilt in conv key"
            continue
        fi
        acq_padded=$(printf "%03d" "$acq")
        marker="${pos}_${acq_padded}"
        echo "  IMOD $imod_tilt -> acq $acq (marker: $marker)"

        # Remove the surrounding 19 lines before + 7 lines after the marker
        awk -v marker="$marker" '
            $0 ~ marker { for(x=NR-19; x<=NR+7; x++) d[x]; }
            { a[NR]=$0 }
            END { for(i=1; i<=NR; i++) if(!(i in d)) print a[i] }
        ' "$mdoc" > "${mdoc}.tmp"
        mv "${mdoc}.tmp" "$mdoc"
    done

    # Renumber remaining ZValue blocks contiguously starting at 0
    sed -i '/ZValue/c\__WARP_AUTO_PLACEHOLDER__' "$mdoc"
    end=$(grep -c "__WARP_AUTO_PLACEHOLDER__" "$mdoc")

    for ((q=1; q<=end; q++)); do
        value=$((q - 1))
        awk -v old="__WARP_AUTO_PLACEHOLDER__" -v new="[ZValue = $value]" \
            '!x{x=sub(old,new)}7' "$mdoc" > "${mdoc}.tmp"
        mv "${mdoc}.tmp" "$mdoc"
    done

    echo "  $mdoc: renumbered $end ZValue blocks"
    processed=$((processed + 1))
done

# -----------------------------------------------------------------------------
# Fix date formatting (yy-mmm-dd required by Warp)
# -----------------------------------------------------------------------------

echo ""
echo "Fixing mdoc date formatting..."
current_year=$(date +%Y)
prev_year=$((current_year - 1))

for year in "$current_year" "$prev_year"; do
    short="${year: -2}"
    sed -i "s/-${year}/-${short}/g" *.mdoc 2>/dev/null || true
done

# Swap day and year: dd-mmm-yy -> yy-mmm-dd
sed -i -E 's/([0-9]{2})-([A-Z][a-z]{2})-([0-9]{2})/\3-\2-\1/g' *.mdoc

echo ""
echo "========================================"
echo "Summary:"
echo "  Mdocs with exclusions processed: $processed"
echo "  Date format: yy-mmm-dd (Warp-compatible)"
echo "Finished: $(date)"
echo "========================================"

exit 0
