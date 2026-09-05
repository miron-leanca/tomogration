#!/bin/bash
# ml_membrain_mesh_warp_auto.sh
#
# Convert membrane segmentations to meshes (membrain_pick convert_file, env
# membrainpick — NOT membrainseg; the two envs conflict and must never merge).
# Pairs each segmentation with its tomogram by series stem and converts one
# (tomogram, membrane) pair at a time.
#
# Encoded traps (module spec §3.6 + real --help, membrain-pick 0.0.9):
#   * --input-pixel-size is ALWAYS passed explicitly — MRC headers have been
#     unreliable here.
#   * --only-largest-component defaults ON in the tool. For a components
#     volume that silently meshes the biggest object — often a merged CLUSTER,
#     not a virion. Default OFF here; opt in with MB_ONLY_LARGEST=1.
#   * sibling command flag drift: convert_file takes --tomogram-path but
#     convert_mb_folder takes --tomo-path. This wrapper uses convert_file only.
#   * whether --step-size / --step-numbers are pixels or Ångströms is
#     UNRESOLVED — do not trust the defaults blindly; verify one mesh visually
#     (spec says do not guess in code, so this wrapper does not).
#
# Usage:
#   bash ml_membrain_mesh_warp_auto.sh <seg_dir> <tomo_dir> <output_dir>
#
# <seg_dir>   membrane segmentations to mesh (thresholded, components output,
#             or a single-label extract)
# <tomo_dir>  the tomograms the segmentations came from (stem-matched)
# <output_dir> where mesh containers land
#
# Knobs via environment (the tomogration GUI sets these):
#   MB_CONDA_ENV     conda env with membrain-pick       (default: membrainpick)
#   MB_PIXEL_SIZE    --input-pixel-size Å/px            (default: 12.56)
#   MB_PATTERN       segmentation filename glob         (default: "*.mrc")
#   MB_TOMO_LIST     space/comma list of series stems; empty = ALL
#   MB_STEP_SIZE     --step-size                        (default: blank = 2.5)
#   MB_STEP_LO       lower --step-numbers               (default: blank = -10)
#   MB_STEP_HI       upper --step-numbers               (default: blank = 10)
#   MB_BARY_AREA     --barycentric-area                 (default: 400 — note
#                    the tool's help TEXT claims 1.0 but its real default is
#                    400.0; passed explicitly so there is no ambiguity)
#   MB_SMOOTHING     --mesh-smoothing                   (default: 1000)
#   MB_ONLY_LARGEST  1 = keep the tool's only-largest-component behaviour
#   MB_FORCE         1 = redo existing outputs
#   MB_EXTRA_ARGS    appended verbatim
#
# Flags verified against membrain-pick 0.0.9 --help (2026-08-14).
set -e

SEG_DIR="${1:?Usage: $0 <seg_dir> <tomo_dir> <output_dir>}"
TOMO_DIR="${2:?Usage: $0 <seg_dir> <tomo_dir> <output_dir>}"
OUT_DIR="${3:?Usage: $0 <seg_dir> <tomo_dir> <output_dir>}"

MB_ENV="${MB_CONDA_ENV:-membrainpick}"
PIX="${MB_PIXEL_SIZE:-12.56}"
PATTERN="${MB_PATTERN:-*.mrc}"
BARY="${MB_BARY_AREA:-400}"
SMOOTH="${MB_SMOOTHING:-1000}"
FORCE="${MB_FORCE:-0}"
TOMO_LIST="$(printf '%s' "${MB_TOMO_LIST:-}" | tr ',' ' ')"

[ -d "$SEG_DIR" ]  || { echo "ERROR: segmentation dir not found: $SEG_DIR"; exit 1; }
[ -d "$TOMO_DIR" ] || { echo "ERROR: tomogram dir not found: $TOMO_DIR"; exit 1; }

OPT_FLAGS=(--input-pixel-size "$PIX" --barycentric-area "$BARY"
           --mesh-smoothing "$SMOOTH")
[ -n "${MB_STEP_SIZE:-}" ] && OPT_FLAGS+=(--step-size "$MB_STEP_SIZE")
if [ -n "${MB_STEP_LO:-}" ] || [ -n "${MB_STEP_HI:-}" ]; then
    OPT_FLAGS+=(--step-numbers "${MB_STEP_LO:--10}" --step-numbers "${MB_STEP_HI:-10}")
fi
if [ "${MB_ONLY_LARGEST:-}" = "1" ]; then
    OPT_FLAGS+=(--only-largest-component)
else
    OPT_FLAGS+=(--no-only-largest-component)
fi

echo "==================================================================="
echo "MemBrain-pick mesh conversion  ·  $(date)"
echo "segs:   $SEG_DIR  (pattern: $PATTERN)"
echo "tomos:  $TOMO_DIR"
echo "output: $OUT_DIR    pixel size: $PIX Å/px"
echo "only-largest-component: $([ "${MB_ONLY_LARGEST:-}" = "1" ] && echo ON || echo OFF)"
echo "==================================================================="

module load miniconda/latest 2>/dev/null || echo "WARNING: could not load miniconda/latest"
if ! conda activate "$MB_ENV" 2>/dev/null; then
    echo "ERROR: could not 'conda activate $MB_ENV' (membrain-pick lives in its"
    echo "OWN env — pins scipy<1.12.1 / numpy 1.26.4 / napari 0.5.6; do not mix"
    echo "with membrainseg)."
    exit 1
fi
command -v membrain_pick >/dev/null 2>&1 || {
    echo "ERROR: membrain_pick not found in env '$MB_ENV'."; exit 1; }

mkdir -p "$OUT_DIR"

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

# Pair a segmentation with its tomogram: the LONGEST tomogram basename that
# prefixes the seg name (followed by '_', '.' or end). A first-underscore
# stem truncated PACEtomo-style names (Position_1 -> "Position"), and taking
# the first glob hit could silently mesh against the WRONG tomogram with a
# mismatched pixel size.
tomo_for() {  # $1 = seg basename (no .mrc); prints tomogram path or nothing
    local name="$1" best="" bestlen=0 t tb
    for t in "$TOMO_DIR"/*.mrc; do
        [ -e "$t" ] || break
        tb="$(basename "$t" .mrc)"
        case "$name" in
            "$tb"|"$tb"_*|"$tb".*)
                if [ ${#tb} -gt $bestlen ]; then best="$t"; bestlen=${#tb}; fi;;
        esac
    done
    printf '%s' "$best"
}

# Same variant-mixing refusal as the segment wrapper: meshes from a DIFFERENT
# segmentation folder silently interleaving here is the §5 confusion again.
if [ -f "$OUT_DIR/PROVENANCE.json" ] && [ "$FORCE" != "1" ]; then
    PREV_SRC="$(grep -o '"source_dir"[[:space:]]*:[[:space:]]*"[^"]*"' \
        "$OUT_DIR/PROVENANCE.json" | head -1 | sed 's/.*: *"//; s/"$//' || true)"
    if [ -n "$PREV_SRC" ] && [ "$PREV_SRC" != "$SEG_DIR" ]; then
        echo "ERROR: $OUT_DIR already holds meshes from a DIFFERENT segmentation:"
        echo "         previous: $PREV_SRC"
        echo "         now:      $SEG_DIR"
        echo "Use one output folder per segmentation variant, or MB_FORCE=1."
        exit 1
    fi
fi

ok=0; skipped=0; failed=0; unpaired=0
for SEG in "${SEGS[@]}"; do
    name="$(basename "$SEG" .mrc)"
    TOMO="$(tomo_for "$name")"
    if [ -z "$TOMO" ]; then
        echo "WARN: no tomogram in $TOMO_DIR prefixes $name — skipped."
        unpaired=$((unpaired+1)); continue
    fi
    # One SUBFOLDER per segmentation: derived names in this pipeline extend
    # each other (X vs X_threshold_-1.0), so any name-glob skip test matched a
    # DIFFERENT segmentation's containers — and the failure cleanup deleted
    # them. A folder per seg makes both exact by construction.
    SEG_OUT="$OUT_DIR/$name"
    if [ "$FORCE" != "1" ] && ls "$SEG_OUT"/* >/dev/null 2>&1; then
        echo "SKIP: mesh output exists for $name"; skipped=$((skipped+1)); continue
    fi
    mkdir -p "$SEG_OUT"
    echo "--- $name  (tomo: $(basename "$TOMO"))"
    if membrain_pick convert_file \
            --tomogram-path "$TOMO" \
            --mb-path "$SEG" \
            --out-folder "$SEG_OUT" \
            "${OPT_FLAGS[@]}" ${MB_EXTRA_ARGS:-}; then
        ok=$((ok+1))
    else
        echo "ERROR: mesh conversion failed for $name"
        # Remove the partial subfolder, or the skip check treats the crashed
        # series as done forever. Exactly this seg's folder — nothing shared.
        rm -rf "$SEG_OUT"
        failed=$((failed+1))
    fi
done

if [ $((ok + skipped)) -gt 0 ]; then
    cat > "$OUT_DIR/PROVENANCE.json" <<EOF
{
  "variant": "membrane_mesh",
  "tool": "membrain_pick convert_file",
  "source_dir": "$SEG_DIR",
  "tomo_dir": "$TOMO_DIR",
  "params": {"input_pixel_size": "$PIX", "barycentric_area": "$BARY",
             "mesh_smoothing": "$SMOOTH",
             "step_size": "${MB_STEP_SIZE:-tool default}",
             "step_numbers": "${MB_STEP_LO:--10}..${MB_STEP_HI:-10}$([ -z "${MB_STEP_LO:-}${MB_STEP_HI:-}" ] && echo ' (tool default)')",
             "pattern": "$PATTERN",
             "only_largest_component": "${MB_ONLY_LARGEST:-0}"},
  "tomo_list": "$([ -n "$TOMO_LIST" ] && echo "$TOMO_LIST" || echo "ALL")",
  "extraction_allowed": false,
  "date": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF
fi

echo "==================================================================="
echo "done: $ok meshed, $skipped skipped, $unpaired unpaired, $failed failed."
echo "Finished: $(date)"
echo "==================================================================="
[ "$failed" -eq 0 ] || exit 1
# Nothing meshed AND nothing already there = the run achieved nothing
# (all inputs unpaired) — report it as the failure it is.
if [ "$ok" -eq 0 ] && [ "$skipped" -eq 0 ]; then
    echo "ERROR: no segmentation could be paired with a tomogram."
    exit 1
fi
