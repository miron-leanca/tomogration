#!/bin/bash
# ml_add_thumbnails_warp_auto.sh
#
# Copy each source grid's Tomo5 Thumbnails into a MERGED tomogration project,
# RENAMED to the combined Position### numbering, and build a combined
# listfile_Position.txt (original Tomo5 name -> Position###). The Tilt Inspector
# then finds the montages (so "Open ALL in 3dmod" works at the raw stage) AND shows
# the "was <Tomo5>" conversion note via the reverse mapping.
#
# Usage:
#   bash ml_add_thumbnails_warp_auto.sh <combined_dir> <grid:offset> [<grid:offset> ...] [--execute]
#     <grid:offset>  source grid dir : number ADDED to that grid's listfile renamed
#                    number to get the combined number. Use 'auto' to detect from the
#                    grid's own listfile (<=124 -> +124, else +0). The FIRST grid that
#                    kept the combined numbering uses :0.
#   Default is a DRY-RUN (prints the plan, changes nothing); add --execute to copy.
#
# Example (this EML45 dataset):
#   bash ml_add_thumbnails_warp_auto.sh \
#     /ceph/users/haq21239/EMDatasets/EML45/combined \
#     /ceph/users/haq21239/EMDatasets/EML45/20260619_ML_EML45_OC43_grid7:0 \
#     /ceph/users/haq21239/EMDatasets/EML45/20260620_ML_EML45_grid6_OC43:auto \
#     /ceph/users/haq21239/EMDatasets/EML45/20260621_ML_EML45_grid5_OC43:0
#   (review, then re-run with --execute)
set -e

EXECUTE=0
ARGS=()
for a in "$@"; do
    if [ "$a" = "--execute" ]; then EXECUTE=1; else ARGS+=("$a"); fi
done
COMBINED="${ARGS[0]:?Usage: $0 <combined_dir> <grid:offset> ... [--execute]}"
GRIDS=("${ARGS[@]:1}")
[ ${#GRIDS[@]} -ge 1 ] || { echo "Need at least one <grid:offset> argument."; exit 1; }

COMB_THUMBS="$COMBINED/Thumbnails"
COMB_MDOCS="$COMBINED/mdocs"
COMB_LIST="$COMBINED/listfile_Position.txt"
[ -d "$COMB_MDOCS" ] || { echo "ERROR: $COMB_MDOCS missing — is '$COMBINED' the merged project?"; exit 1; }

mode=$([ $EXECUTE -eq 1 ] && echo EXECUTE || echo DRY-RUN)
echo "==================================================================="
echo "ml_add_thumbnails_warp_auto    [$mode]"
echo "combined: $COMBINED"
echo "==================================================================="

if [ $EXECUTE -eq 1 ]; then
    mkdir -p "$COMB_THUMBS"
    [ -f "$COMB_LIST" ] || \
        echo "# combined Tomo5 -> Position mapping (ml_add_thumbnails_warp_auto.sh)" > "$COMB_LIST"
fi

total_copy=0; total_skip=0
for spec in "${GRIDS[@]}"; do
    GDIR="${spec%%:*}"; OFF="${spec##*:}"
    LF="$GDIR/listfile_Position.txt"
    TH="$GDIR/Thumbnails"
    echo "-------------------------------------------------------------------"
    echo "grid: $GDIR"
    [ -f "$LF" ] || { echo "  WARN: no listfile ($LF) — skipping this grid"; continue; }
    [ -d "$TH" ] || { echo "  WARN: no Thumbnails/ ($TH) — skipping this grid"; continue; }
    if [ "$OFF" = "auto" ]; then
        gmax=$(grep -v '^#' "$LF" | awk '{print $2}' | sed 's/Position//' \
               | grep -E '^[0-9]+$' | sort -n | tail -1)
        if [ -n "$gmax" ] && [ "$gmax" -le 124 ]; then OFF=124; else OFF=0; fi
        echo "  auto offset = $OFF   (this grid's listfile max = ${gmax:-?})"
    else
        echo "  offset = $OFF"
    fi
    cnt=0; skp=0; shown=0
    while read -r orig ren _rest || [ -n "$orig" ]; do
        case "$orig" in ''|\#*) continue;; esac
        [ -n "$ren" ] || continue
        num=$(printf '%s' "$ren" | sed 's/Position//')
        case "$num" in ''|*[!0-9]*) continue;; esac
        final=$((10#$num + OFF))
        fpos=$(printf 'Position%03d' "$final")
        src="$TH/$orig.mrc"
        dst="$COMB_THUMBS/$fpos.mrc"
        # only map series that actually made it into the combined project, and never
        # clobber an existing thumbnail.
        if [ ! -f "$src" ] || [ ! -f "$COMB_MDOCS/$fpos.mdoc" ] || [ -f "$dst" ]; then
            skp=$((skp+1)); continue
        fi
        if [ $EXECUTE -eq 1 ]; then
            cp "$src" "$dst"
            printf '%s %s\n' "$orig" "$fpos" >> "$COMB_LIST"
        elif [ $shown -lt 4 ]; then
            echo "    $orig.mrc  ->  Thumbnails/$fpos.mrc   (listfile: $orig $fpos)"
            shown=$((shown+1))
        fi
        cnt=$((cnt+1))
    done < "$LF"
    echo "  $([ $EXECUTE -eq 1 ] && echo copied || echo would-copy): $cnt    skipped: $skp"
    total_copy=$((total_copy+cnt)); total_skip=$((total_skip+skp))
done

echo "==================================================================="
echo "TOTAL $([ $EXECUTE -eq 1 ] && echo copied || echo would-copy): $total_copy    skipped: $total_skip"
if [ $EXECUTE -eq 0 ]; then
    echo "DRY-RUN — nothing changed. If the plan looks right, re-run with --execute."
else
    echo "Done. Thumbnails -> $COMB_THUMBS,  mapping -> $COMB_LIST."
    echo "In tomogration (root = $COMBINED) the Tilt Inspector now shows montages in"
    echo "3dmod and the 'Position### ⟵ was <Tomo5>' conversion note."
fi
exit 0
