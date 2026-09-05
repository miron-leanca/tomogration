#!/bin/bash
# ml_membrain_thresholds_warp_auto.sh
#
# Threshold membrane score maps (membrain thresholds). CPU, seconds per map —
# THIS is the main tuning loop: segment once on GPU, then re-threshold the
# *_scores.mrc as often as needed for free.
#
# Two gotchas this wrapper encodes (from the module spec + real --help):
#   * --thresholds repeats once per value (NOT a comma list).
#   * the tool's default --out-folder is ./predictions and does NOT follow the
#     scoremap's folder — it is ALWAYS passed explicitly here, or outputs from
#     different runs silently collide in one folder.
# Output names normalise the threshold to one decimal: -3 -> ..._threshold_-3.0.mrc
# (downstream steps must construct names that way, not from the raw input).
#
# Usage:
#   bash ml_membrain_thresholds_warp_auto.sh <scores_dir> <output_dir>
#
# Knobs via environment (the tomogration GUI sets these):
#   MB_CONDA_ENV    conda env with membrain-seg          (default: membrainseg)
#   MB_THRESHOLDS   space/comma list of thresholds       (default: "0.0")
#                   e.g. "-1.5 -0.5 0.0 0.5" — the whole point is the sweep
#   MB_TOMO_LIST    space/comma list of series stems; empty = ALL score maps
#   MB_FORCE        1 = redo existing outputs
#   MB_EXTRA_ARGS   appended verbatim
#
# Flags verified against membrain-seg 0.0.10 --help (2026-08-14).
set -e

SCORES_DIR="${1:?Usage: $0 <scores_dir> <output_dir>}"
OUT_DIR="${2:?Usage: $0 <scores_dir> <output_dir>}"

MB_ENV="${MB_CONDA_ENV:-membrainseg}"
FORCE="${MB_FORCE:-0}"
THRESHOLDS="$(printf '%s' "${MB_THRESHOLDS:-0.0}" | tr ',' ' ')"
TOMO_LIST="$(printf '%s' "${MB_TOMO_LIST:-}" | tr ',' ' ')"

[ -d "$SCORES_DIR" ] || { echo "ERROR: scores dir not found: $SCORES_DIR"; exit 1; }
# Token count, not raw string: a commas/whitespace-only value passes [ -n ].
read -ra _THR_TOKENS <<< "$THRESHOLDS"
[ ${#_THR_TOKENS[@]} -gt 0 ] || { echo "ERROR: MB_THRESHOLDS is empty."; exit 1; }

# The tool's one-decimal output-name normalisation: -3 -> -3.0, 0.5 -> 0.5.
norm_thr() { awk -v t="$1" 'BEGIN{printf "%g", t+0; if (t+0 == int(t+0)) printf ".0"}'; }

echo "==================================================================="
echo "MemBrain thresholds  ·  $(date)"
echo "scores: $SCORES_DIR"
echo "output: $OUT_DIR"
echo "thresholds: $THRESHOLDS"
echo "==================================================================="

module load miniconda/latest 2>/dev/null || echo "WARNING: could not load miniconda/latest"
if ! conda activate "$MB_ENV" 2>/dev/null; then
    echo "ERROR: could not 'conda activate $MB_ENV'."; exit 1
fi
command -v membrain >/dev/null 2>&1 || {
    echo "ERROR: membrain not found in env '$MB_ENV'."; exit 1; }

mkdir -p "$OUT_DIR"

MAPS=()
if [ -n "$TOMO_LIST" ]; then
    for stem in $TOMO_LIST; do
        # '_'-anchored so Position_1 cannot also select Position_10.
        hits=()
        for f in "$SCORES_DIR/${stem}_scores.mrc" "$SCORES_DIR/${stem}_"*_scores.mrc; do
            [ -e "$f" ] && hits+=("$f")
        done
        if [ ${#hits[@]} -gt 0 ]; then MAPS+=("${hits[@]}")
        else echo "WARN: no ${stem}_*_scores.mrc in $SCORES_DIR — skipped."; fi
    done
    [ ${#MAPS[@]} -gt 0 ] || {
        echo "ERROR: nothing in MB_TOMO_LIST ($TOMO_LIST) matched *_scores.mrc "\
"in $SCORES_DIR"; exit 1; }
else
    MAPS=("$SCORES_DIR"/*_scores.mrc)
    [ -e "${MAPS[0]}" ] || { echo "ERROR: no *_scores.mrc in $SCORES_DIR "\
"(run 'Segment' with score maps on first)"; exit 1; }
fi
echo "${#MAPS[@]} score map(s) selected."

ok=0; skipped=0; failed=0
for MAP in "${MAPS[@]}"; do
    name="$(basename "$MAP" .mrc)"
    # Per-VALUE completeness: run the tool with exactly the thresholds whose
    # output is missing. Checking only the first value made "extend the sweep"
    # (the advertised tuning loop!) a silent no-op: the old first output
    # existed, the map was skipped, and the new values never got generated.
    missing=()
    for t in "${_THR_TOKENS[@]}"; do
        out="$OUT_DIR/${name}_threshold_$(norm_thr "$t").mrc"
        if [ "$FORCE" = "1" ] || [ ! -f "$out" ]; then missing+=("$t"); fi
    done
    if [ ${#missing[@]} -eq 0 ]; then
        echo "SKIP: $name already has all ${#_THR_TOKENS[@]} threshold output(s)"
        skipped=$((skipped+1)); continue
    fi
    THR_FLAGS=()
    for t in "${missing[@]}"; do THR_FLAGS+=(--thresholds "$t"); done
    echo "--- $name  (${#missing[@]}/${#_THR_TOKENS[@]} value(s) to generate: ${missing[*]})"
    if membrain thresholds \
            --scoremap-path "$MAP" \
            --out-folder "$OUT_DIR" \
            "${THR_FLAGS[@]}" ${MB_EXTRA_ARGS:-}; then
        ok=$((ok+1))
    else
        echo "ERROR: thresholds failed for $name"
        # Remove exactly the outputs this attempt was generating: a truncated
        # volume written before the crash would pass the bare-existence check
        # and mark its value complete forever.
        for t in "${missing[@]}"; do
            rm -f "$OUT_DIR/${name}_threshold_$(norm_thr "$t").mrc" \
                2>/dev/null || true
        done
        failed=$((failed+1))
    fi
done

if [ $((ok + skipped)) -gt 0 ]; then
    cat > "$OUT_DIR/PROVENANCE.json" <<EOF
{
  "variant": "thresholded_segmentation",
  "tool": "membrain thresholds",
  "source_dir": "$SCORES_DIR",
  "params": {"thresholds": "$THRESHOLDS"},
  "tomo_list": "$([ -n "$TOMO_LIST" ] && echo "$TOMO_LIST" || echo "ALL")",
  "extraction_allowed": false,
  "date": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF
fi

echo "==================================================================="
echo "done: $ok maps thresholded, $skipped skipped, $failed failed."
echo "Finished: $(date)"
echo "==================================================================="
[ "$failed" -eq 0 ] || exit 1
