#!/bin/bash
# ml_membrain_deconv_warp_auto.sh
#
# CTF-deconvolve reconstructed tomograms for membrane segmentation
# (tomo_preprocessing deconvolve, from the membrain-seg env). This boosts
# low-frequency membrane contrast BEFORE `membrain segment` / IsoNet — it is a
# PICKING aid: never extract particles for averaging from deconvolved volumes
# (tomogration enforces this via the PROVENANCE.json written here).
#
# The defocus (--df) is PER TILT SERIES and is parsed automatically from
# warp_tiltseries/<series>.xml  <Param Name="Defocus" Value="..."/>, which Warp
# stores in MICROMETRES — converted here to Ångströms (×10000). Each parsed
# value is echoed per series so it can be sanity-checked in the log; a series
# whose defocus cannot be read is SKIPPED with a warning, never guessed.
#
# Usage:
#   bash ml_membrain_deconv_warp_auto.sh <input_recon_dir> <output_base_dir>
#
# <input_recon_dir>   dir of reconstructed <series>*.mrc (warp_tiltseries/reconstruction)
# <output_base_dir>   base output dir; each parameter variant writes into its own
#                     subfolder  <base>/s<strength>_f<falloff>/  with a
#                     PROVENANCE.json audit record (variant, params, source).
#
# Knobs via environment (the tomogration GUI sets these):
#   MB_CONDA_ENV   conda env with membrain-seg + tomo_preprocessing (default: membrainseg)
#   MB_XML_DIR     dir of Warp per-series .xml for defocus     (default: warp_tiltseries)
#   MB_TOMO_LIST   space/comma list of series stems (e.g. "Position003 Position017").
#                  Empty = ALL tomograms in the input dir (Batch mode). Naming
#                  1–3 = Tune mode: sweep parameters cheaply, then batch the winner.
#   MB_STRENGTH    deconvolution strength value OR space-separated sweep list
#                  (default: 1.0). Multiple values -> one output subfolder each.
#   MB_FALLOFF     falloff value or sweep list                 (default: 1.0)
#   MB_KV          acceleration voltage kV                     (default: 300)
#   MB_CS          spherical aberration mm                     (default: 2.7)
#   MB_AMPCON      amplitude contrast                          (default: 0.07)
#   MB_HP          highpass fraction                           (default: 0.02)
#   MB_PIXEL_SIZE  Å/px passed as --pixel-size; blank = MRC header (the tool's
#                  --help warns a wrong header causes severe errors)
#   MB_SKIP_LOWPASS 1 = pass --skip-lowpass (tool says "not recommended")
#   MB_FORCE       1 = redo existing outputs (default: skip already-done files)
#   MB_EXTRA_ARGS  appended verbatim to tomo_preprocessing deconvolve
#
# Flags verified against the installed tool's --help output (membrainseg env,
# 2026-08-14). Re-verify after any env update — flag names HAVE changed between
# versions of these tools:
#   conda activate membrainseg && tomo_preprocessing deconvolve --help
set -e

INPUT_DIR="${1:?Usage: $0 <input_recon_dir> <output_base_dir>}"
OUT_BASE="${2:?Usage: $0 <input_recon_dir> <output_base_dir>}"

MB_ENV="${MB_CONDA_ENV:-membrainseg}"
XML_DIR="${MB_XML_DIR:-warp_tiltseries}"
KV="${MB_KV:-300}"
CS="${MB_CS:-2.7}"
AMPCON="${MB_AMPCON:-0.07}"
HP="${MB_HP:-0.02}"
PIXSIZE="${MB_PIXEL_SIZE:-}"
FORCE="${MB_FORCE:-0}"
OPT_FLAGS=()
[ -n "$PIXSIZE" ] && OPT_FLAGS+=(--pixel-size "$PIXSIZE")
[ "${MB_SKIP_LOWPASS:-}" = "1" ] && OPT_FLAGS+=(--skip-lowpass)
STRENGTHS="${MB_STRENGTH:-1.0}"
FALLOFFS="${MB_FALLOFF:-1.0}"
# Accept comma or space separated lists everywhere.
STRENGTHS="$(printf '%s' "$STRENGTHS" | tr ',' ' ')"
FALLOFFS="$(printf '%s' "$FALLOFFS" | tr ',' ' ')"
TOMO_LIST="$(printf '%s' "${MB_TOMO_LIST:-}" | tr ',' ' ')"

[ -d "$INPUT_DIR" ] || { echo "ERROR: input dir not found: $INPUT_DIR"; exit 1; }
[ -d "$XML_DIR" ]   || { echo "ERROR: xml dir not found: $XML_DIR (need Warp per-series .xml for --df)"; exit 1; }

# HARD RULE: never deconvolve an already-denoised / wedge-restored volume — it
# degrades it. Any PROVENANCE.json in the input dir naming such a variant stops
# the run outright.
if [ -f "$INPUT_DIR/PROVENANCE.json" ]; then
    if grep -Eqi '"variant"[^,}]*(denoise|wedge|isonet|deconv)' "$INPUT_DIR/PROVENANCE.json"; then
        echo "ERROR: $INPUT_DIR is tagged as a processed variant:"
        grep -o '"variant"[^,}]*' "$INPUT_DIR/PROVENANCE.json" || true
        echo "Deconvolving denoised/wedge-restored (or re-deconvolving) data degrades it."
        echo "Point input at the ORIGINAL reconstruction dir."
        exit 1
    fi
fi

# Defocus in µm from the Warp per-series xml -> Ångströms. Echo what was read.
defocus_A() {  # $1 = series stem; prints integer Å or nothing
    local xml="$XML_DIR/$1.xml" raw=""
    [ -f "$xml" ] || return 0
    raw="$(grep -o '<Param Name="Defocus" Value="[^"]*"' "$xml" 2>/dev/null \
           | head -1 | sed 's/.*Value="//; s/"$//' || true)"
    [ -n "$raw" ] || return 0
    awk -v d="$raw" 'BEGIN { if (d+0 == d) printf "%.0f", d * 10000 }'
}

# Series stem for a tomogram filename: the LONGEST xml basename that prefixes
# it (followed by '_', '.' or end). A naive ${name%%_*} truncates PACEtomo-
# style stems containing underscores (Position_1 -> "Position"), and a naive
# prefix match lets Position_1 claim Position_10's files.
stem_for() {  # $1 = tomogram basename (no .mrc); prints stem or nothing
    local name="$1" best="" b x
    for x in "$XML_DIR"/*.xml; do
        [ -e "$x" ] || break
        b="$(basename "$x" .xml)"
        case "$name" in
            "$b"|"$b"_*|"$b".*) [ ${#b} -gt ${#best} ] && best="$b";;
        esac
    done
    printf '%s' "$best"
}

echo "==================================================================="
echo "MemBrain deconvolve  ·  $(date)"
echo "input:  $INPUT_DIR    xml: $XML_DIR"
echo "output: $OUT_BASE/s<strength>_f<falloff>/"
echo "sweep:  strength [$STRENGTHS]  falloff [$FALLOFFS]"
echo "mode:   $([ -n "$TOMO_LIST" ] && echo "TUNE ($TOMO_LIST)" || echo "BATCH (all tomograms)")"
echo "==================================================================="

module load miniconda/latest 2>/dev/null || echo "WARNING: could not load miniconda/latest"

# Activate the membrain-seg env (same idiom as the miss-alignment wrapper —
# NEVER `source activate`: sourcing a missing file aborts a non-interactive
# shell and no || can catch it).
if ! conda activate "$MB_ENV" 2>/dev/null; then
    echo "ERROR: could not 'conda activate $MB_ENV'."
    echo "The membrane envs live on shared storage (see the module spec):"
    echo "  /ceph/users/haq21239/.conda/envs/membrainseg   (membrain-seg + tomo_preprocessing)"
    echo "Keep membrainseg / membrainpick / isonet SEPARATE — their deps conflict."
    exit 1
fi

command -v tomo_preprocessing >/dev/null 2>&1 || {
    echo "ERROR: tomo_preprocessing not found in env '$MB_ENV'."; exit 1; }

# Collect the input tomograms (Tune list, or everything in the dir). Tune
# globs are '_'-anchored so Position_1 cannot also select Position_10..._19,
# and an all-typo tune list is an ERROR, not a silent 0-tomogram success.
TOMOS=()
if [ -n "$TOMO_LIST" ]; then
    for stem in $TOMO_LIST; do
        hits=()
        for f in "$INPUT_DIR/$stem.mrc" "$INPUT_DIR/${stem}_"*.mrc; do
            [ -e "$f" ] && hits+=("$f")
        done
        if [ ${#hits[@]} -gt 0 ]; then
            TOMOS+=("${hits[@]}")
        else
            echo "WARN: no ${stem}.mrc / ${stem}_*.mrc in $INPUT_DIR — skipped."
        fi
    done
    [ ${#TOMOS[@]} -gt 0 ] || {
        echo "ERROR: nothing in MB_TOMO_LIST ($TOMO_LIST) matched $INPUT_DIR"; exit 1; }
else
    TOMOS=("$INPUT_DIR"/*.mrc)
    [ -e "${TOMOS[0]}" ] || { echo "ERROR: no .mrc in $INPUT_DIR"; exit 1; }
fi
echo "${#TOMOS[@]} tomogram(s) selected."

ok=0; skipped=0; failed=0; nodf=0
for SV in $STRENGTHS; do
  for FV in $FALLOFFS; do
    VDIR="$OUT_BASE/s${SV}_f${FV}"
    mkdir -p "$VDIR"
    echo "--- variant strength=$SV falloff=$FV -> $VDIR"
    for TOMO in "${TOMOS[@]}"; do
        name="$(basename "$TOMO" .mrc)"
        # Longest xml-stem prefix (underscore-safe; see stem_for above).
        stem="$(stem_for "$name")"
        if [ -z "$stem" ]; then
            echo "WARN: $name — no matching <stem>.xml in $XML_DIR — SKIPPED."
            nodf=$((nodf+1)); continue
        fi
        OUT="$VDIR/${name}_deconv.mrc"
        if [ -f "$OUT" ] && [ "$FORCE" != "1" ]; then
            echo "SKIP: exists $OUT"; skipped=$((skipped+1)); continue
        fi
        DF="$(defocus_A "$stem")"
        if [ -z "$DF" ]; then
            echo "WARN: $name — no defocus in $XML_DIR/$stem.xml — SKIPPED (never guessed)."
            nodf=$((nodf+1)); continue
        fi
        awk -v d="$DF" -v s="$stem" 'BEGIN { printf "%s: defocus %.2f um -> --df %d A\n", s, d/10000, d }'
        if tomo_preprocessing deconvolve \
                --input "$TOMO" --output "$OUT" \
                --df "$DF" --kv "$KV" --cs "$CS" \
                --strength "$SV" --falloff "$FV" \
                --ampcon "$AMPCON" --hp-fraction "$HP" \
                "${OPT_FLAGS[@]}" ${MB_EXTRA_ARGS:-}; then
            ok=$((ok+1))
        else
            echo "ERROR: deconvolve failed for $name (variant s$SV f$FV)"
            rm -f "$OUT"       # never leave a truncated volume a re-run would skip
            failed=$((failed+1))
        fi
    done
    # Audit record: what made this folder, from what, with which knobs.
    # Only when the folder actually holds output — a zero-output run must not
    # clobber the record of the run that produced the existing files.
    if ls "$VDIR"/*_deconv.mrc >/dev/null 2>&1; then
        cat > "$VDIR/PROVENANCE.json" <<EOF
{
  "variant": "deconvolved",
  "tool": "tomo_preprocessing deconvolve",
  "source_dir": "$INPUT_DIR",
  "xml_dir": "$XML_DIR",
  "params": {"strength": "$SV", "falloff": "$FV", "kv": "$KV", "cs": "$CS",
             "ampcon": "$AMPCON", "hp_fraction": "$HP"},
  "tomo_list": "$([ -n "$TOMO_LIST" ] && echo "$TOMO_LIST" || echo "ALL")",
  "extraction_allowed": false,
  "note": "picking/segmentation aid; particle extraction must use the original reconstruction",
  "date": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF
    fi
  done
done

echo "==================================================================="
echo "done: $ok deconvolved, $skipped skipped (existing), $nodf no-defocus, $failed failed."
echo "Finished: $(date)"
echo "==================================================================="
[ "$failed" -eq 0 ] || exit 1
# A run that produced nothing and skipped nothing (every series lacked a
# readable defocus) is a failure, not a success with an empty folder.
if [ "$ok" -eq 0 ] && [ "$skipped" -eq 0 ]; then
    echo "ERROR: no tomogram was deconvolved (all lacked a readable defocus)."
    exit 1
fi
