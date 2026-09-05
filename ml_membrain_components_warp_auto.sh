#!/bin/bash
# ml_membrain_components_warp_auto.sh
#
# Connected components of thresholded membrane segmentations
# (membrain components). CPU. Sweepable over --connected-component-thres:
# each swept value writes into its own cc<voxels>/ subfolder, and the two
# numbers the tool reports per volume ("Found N components", "Relabeled to M")
# are parsed into components_metrics.tsv — a flat Found-curve across the sweep
# means fragmentation, a steep drop means debris; that distinction is the
# whole point of sweeping.
#
# KNOWN TRAP (spec §3.5): the tool does NOT create --out-folder — it computes
# the whole result and then dies on the write. mkdir -p happens here, first.
#
# Usage:
#   bash ml_membrain_components_warp_auto.sh <seg_dir> <output_base_dir>
#
# <seg_dir>          thresholded segmentations (membrane/thresholds), or any
#                    folder of binary membrane volumes
# <output_base_dir>  base dir; each swept threshold writes cc<voxels>/
#
# Knobs via environment (the tomogration GUI sets these):
#   MB_CONDA_ENV   conda env with membrain-seg           (default: membrainseg)
#   MB_CC_THRES    space/comma list of voxel-count cutoffs (default: "50")
#                  components smaller than this are removed; sweep it
#   MB_PATTERN     input filename glob                   (default: "*.mrc")
#   MB_TOMO_LIST   space/comma list of series stems; empty = ALL
#   MB_FORCE       1 = redo existing outputs
#   MB_EXTRA_ARGS  appended verbatim
#
# Flags verified against membrain-seg 0.0.10 --help (2026-08-14).
# pipefail matters here: the tool runs through `| tee` to capture its
# Found/Relabeled lines, and without pipefail the if would test tee's status.
set -eo pipefail
trap 'rm -f "${LOG:-}"' EXIT   # the tee log is a mktemp; never leak it

SEG_DIR="${1:?Usage: $0 <seg_dir> <output_base_dir>}"
OUT_BASE="${2:?Usage: $0 <seg_dir> <output_base_dir>}"

MB_ENV="${MB_CONDA_ENV:-membrainseg}"
FORCE="${MB_FORCE:-0}"
PATTERN="${MB_PATTERN:-*.mrc}"
TOMO_LIST="$(printf '%s' "${MB_TOMO_LIST:-}" | tr ',' ' ')"
PIX="${MB_PIXEL_SIZE:-12.56}"

# Cutoffs accept the SAME physical spellings as the sweep card, so a value
# tuned there pastes here unchanged: bare '50' = voxels at THIS run's pixel
# size, '50@12.56' = 50 voxels at 12.56 Å/px rescaled to MB_PIXEL_SIZE, and
# '100nm3' = a physical volume. Everything resolves to integer voxels (the
# unit the tool takes and the cc<voxels>/ folder is named by); the raw
# spelling is echoed beside the resolved value so nothing is silent.
resolve_cc() {  # $1 = token; prints integer voxels or nothing on parse error
    awk -v t="$1" -v vx="$PIX" 'BEGIN {
        if (t ~ /^[0-9]+(\.[0-9]+)?nm3$/) {
            sub(/nm3$/, "", t)
            v = t / ((vx / 10.0) ^ 3)
        } else if (t ~ /^[0-9]+(\.[0-9]+)?@[0-9]+(\.[0-9]+)?$/) {
            split(t, a, "@")
            v = a[1] * ((a[2] / vx) ^ 3)
        } else if (t ~ /^[0-9]+$/) {
            v = t
        } else { exit 1 }
        printf "%d", (v < 1 ? 1 : v + 0.5)
    }'
}
CC_LIST=""
for tok in $(printf '%s' "${MB_CC_THRES:-50}" | tr ',' ' '); do
    v="$(resolve_cc "$tok" || true)"
    [ -n "$v" ] || { echo "ERROR: cannot parse cutoff '$tok' (use voxels, "\
"'N@apix' or 'Xnm3')."; exit 1; }
    [ "$tok" = "$v" ] || echo "cutoff: $tok -> $v voxels at $PIX Å/px"
    CC_LIST="$CC_LIST $v"
done
CC_LIST="${CC_LIST# }"

[ -d "$SEG_DIR" ] || { echo "ERROR: segmentation dir not found: $SEG_DIR"; exit 1; }

echo "==================================================================="
echo "MemBrain components  ·  $(date)"
echo "input:  $SEG_DIR  (pattern: $PATTERN)"
echo "output: $OUT_BASE/cc<voxels>/"
echo "sweep:  component cutoff [$CC_LIST] voxels"
echo "==================================================================="

module load miniconda/latest 2>/dev/null || echo "WARNING: could not load miniconda/latest"
if ! conda activate "$MB_ENV" 2>/dev/null; then
    echo "ERROR: could not 'conda activate $MB_ENV'."; exit 1
fi
command -v membrain >/dev/null 2>&1 || {
    echo "ERROR: membrain not found in env '$MB_ENV'."; exit 1; }

# Selection: batch glob first, then (in tune mode) filter per stem with an
# '_'-anchored match so Position_1 cannot also select Position_10.
ALL=("$SEG_DIR"/$PATTERN)
[ -e "${ALL[0]}" ] || { echo "ERROR: no $PATTERN in $SEG_DIR"; exit 1; }
SEGS=()
if [ -n "$TOMO_LIST" ]; then
    for f in "${ALL[@]}"; do
        base="$(basename "$f")"
        for stem in $TOMO_LIST; do
            case "$base" in
                "$stem".*|"$stem"_*) SEGS+=("$f"); break;;
            esac
        done
    done
    [ ${#SEGS[@]} -gt 0 ] || {
        echo "ERROR: nothing in MB_TOMO_LIST ($TOMO_LIST) matched $PATTERN "\
"in $SEG_DIR"; exit 1; }
else
    SEGS=("${ALL[@]}")
fi
echo "${#SEGS[@]} segmentation(s) selected."

ok=0; skipped=0; failed=0
for CC in $CC_LIST; do
    # Per-VALUE counters: the provenance gate below must judge THIS cc folder,
    # not the whole run — a cutoff that produced nothing was getting a
    # PROVENANCE.json because an earlier cutoff succeeded.
    ok_v=0; skipped_v=0
    VDIR="$OUT_BASE/cc${CC}"
    mkdir -p "$VDIR"          # the tool does NOT create this — trap from §3.5
    METRICS="$VDIR/components_metrics.tsv"
    [ -f "$METRICS" ] || printf 'file\tcc_thres\tfound\trelabeled\n' > "$METRICS"
    echo "--- cutoff ${CC} voxels -> $VDIR"
    for SEG in "${SEGS[@]}"; do
        name="$(basename "$SEG" .mrc)"
        # '_'/'.'-anchored existence check (a bare prefix let Position_1
        # claim Position_10's output).
        if [ "$FORCE" != "1" ] && { [ -f "$VDIR/${name}.mrc" ] ||
                ls "$VDIR/${name}"_*.mrc >/dev/null 2>&1; }; then
            echo "SKIP: exists for $name (cc$CC)"
            skipped=$((skipped+1)); skipped_v=$((skipped_v+1)); continue
        fi
        LOG="$(mktemp)"
        if membrain components \
                --segmentation-path "$SEG" \
                --connected-component-thres "$CC" \
                --out-folder "$VDIR" ${MB_EXTRA_ARGS:-} 2>&1 | tee "$LOG"; then
            ok=$((ok+1)); ok_v=$((ok_v+1))
            # "Found N" / "Relabeled to M" — the tune-mode metrics. Parsed
            # tolerantly (first integer after each keyword), blank if absent.
            found="$(grep -io 'found[^0-9]*[0-9]\+' "$LOG" | head -1 | grep -o '[0-9]\+' | head -1 || true)"
            relab="$(grep -io 'relabel[^0-9]*[0-9]\+' "$LOG" | head -1 | grep -o '[0-9]\+' | head -1 || true)"
            # One row per volume: drop any stale row for this name first, so
            # an MB_FORCE re-run updates instead of accumulating duplicates.
            grep -v "^${name}	" "$METRICS" > "$METRICS.tmp" 2>/dev/null || true
            mv "$METRICS.tmp" "$METRICS"
            printf '%s\t%s\t%s\t%s\n' "$name" "$CC" "${found:-?}" "${relab:-?}" >> "$METRICS"
            echo "metrics: $name cc=$CC found=${found:-?} relabeled=${relab:-?}"
        else
            echo "ERROR: components failed for $name (cc$CC)"
            # A truncated label volume would satisfy the existence check and
            # mark this tomogram complete forever — remove the attempt.
            rm -f "$VDIR/${name}.mrc" "$VDIR/${name}"_*.mrc 2>/dev/null || true
            failed=$((failed+1))
        fi
        rm -f "$LOG"
    done
    if [ $((ok_v + skipped_v)) -gt 0 ]; then
        cat > "$VDIR/PROVENANCE.json" <<EOF
{
  "variant": "connected_components",
  "tool": "membrain components",
  "source_dir": "$SEG_DIR",
  "params": {"connected_component_thres": "$CC", "pattern": "$PATTERN"},
  "tomo_list": "$([ -n "$TOMO_LIST" ] && echo "$TOMO_LIST" || echo "ALL")",
  "extraction_allowed": false,
  "date": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF
    fi
done

echo "==================================================================="
echo "done: $ok volumes labelled, $skipped skipped, $failed failed."
echo "Per-sweep metrics: $OUT_BASE/cc*/components_metrics.tsv"
echo "Size histogram / single-label extraction:"
echo "  python ml_membrain_label_tools.py histogram <labels.mrc>"
echo "  python ml_membrain_label_tools.py extract <labels.mrc> <label> <out.mrc>"
echo "Finished: $(date)"
echo "==================================================================="
[ "$failed" -eq 0 ] || exit 1
