#!/bin/bash
# ml_missalignment_warp_auto.sh
#
# Run the miss-alignment deep-learning tilt-series aligner (warpem/miss-alignment)
# on a Warp tilt-series project. It REFINES an existing coarse alignment — it does
# NOT align raw stacks. The docs are explicit: "miss-alignment starts from an
# initially coarse aligned dataset." So you MUST coarse-align first, e.g.
#   WarpTools ts_etomo_patches --settings warp_tiltseries.settings --angpix 10 \
#       --patch_size 2000 --initial_axis <nominal_axis_deg> --do_axis_search
#   WarpTools ts_autolevel      --settings warp_tiltseries.settings --angpix 10 --patch_size 2000
# (or AreTomo → ts_import_alignments → change_selection --select). Running on RAW,
# unaligned series gives a garbage, featureless tomogram. miss-alignment then reads
# and refines the per-series .xml in place, so AFTERWARDS there is no ts_import step
# — go straight to ts_ctf / ts_reconstruct.
#
# Usage:
#   bash ml_missalignment_warp_auto.sh <config.yaml> <input_dir>
#
# <input_dir> is the Warp tilt-series dir (warp_tiltseries) holding <series>.xml +
# tiltstack/<series>/<series>.st  (run ts_import + ts_stack first, same as AreTomo).
#
# Knobs via environment (the tomogration GUI sets these):
#   MA_MODE               'train' (default) or 'infer'. infer = reuse a finished run's
#                         models to align a NEW/LARGER dataset WITHOUT retraining; runs
#                         'miss-alignment infer', seeds an inference config (data_directory
#                         + model_run_directory), and uses MA_INFER_DEVICES.
#   MA_MODEL_RUN_DIR      (infer only) finished training run holding iter1/.. iterN/model.ckpt
#   MA_INFER_DEVICES      (infer only) GPUs for alignment, comma list   (default: 0,1,2,3)
#   MA_CONDA_ENV          conda env with miss-alignment   (default: miss-alignment)
#   MA_TRAINING_DEVICES   --training-devices              (default: 0)
#                         KEEP A SINGLE GPU. >1 GPU makes torch spawn one trainer per
#                         GPU and they race to wipe the shared reconstruction pool dir
#                         (FileNotFoundError on a partition_*_worker_*.pickle during
#                         datamodule setup). Add GPUs to MA_RECON_DEVICES for speed.
#   MA_RECON_DEVICES      --reconstruction-devices        (default: 0,0,0)
#                         This is the speed knob (recon feeds the pool, the bottleneck).
#   MA_DATALOADERS        --dataloaders-per-trainer       (default: 5)
#   MA_START_ITER         --start-at-iteration            (default: 0)
#   MA_PREPARE_STACKS     --prepare-stacks <Å/px>         (default: 10.0; empty = skip)
#   MA_EXTRA_ARGS         appended verbatim to the miss-alignment command
#
# NOTE: this deliberately does NOT load a cuda module — the miss-alignment conda
# env ships its own CUDA 12.9 runtime, and loading a cluster cuda module would put
# its stub libcuda.so on LD_LIBRARY_PATH and make every GPU read as "invalid"
# (the same trap AreTomo hit). The system NVIDIA driver supplies libcuda.so.1.
set -e

CONFIG="${1:?Usage: $0 <config.yaml> <input_dir>}"
INPUT_DIR="${2:?Usage: $0 <config.yaml> <input_dir>}"

MA_ENV="${MA_CONDA_ENV:-miss-alignment}"
MODE="${MA_MODE:-train}"
TRAIN_DEV="${MA_TRAINING_DEVICES:-0}"
RECON_DEV="${MA_RECON_DEVICES:-0,0,0}"
# Sanitize GPU lists — miss-alignment's int() parser chokes on a stray leading/
# trailing/double comma (e.g. ',2,3' → int('')). Squeeze and trim commas.
_clean_devs() { printf '%s' "$1" | tr -s ',' ',' | sed 's/^,//; s/,$//'; }
TRAIN_DEV="$(_clean_devs "$TRAIN_DEV")"
RECON_DEV="$(_clean_devs "$RECON_DEV")"
INFER_DEV="$(_clean_devs "${MA_INFER_DEVICES:-0,1,2,3}")"
MODEL_RUN_DIR="${MA_MODEL_RUN_DIR:-}"
DLOADERS="${MA_DATALOADERS:-5}"
START_ITER="${MA_START_ITER:-0}"
PREP="${MA_PREPARE_STACKS:-10.0}"
# Reconstruction pool is split into (training_devices × dataloaders) partitions, and
# each partition must hold >= 2 × config batch_size. Default 2000 keeps 4 GPUs × 5
# dataloaders × batch 32 valid (2000/20 = 100 >= 64). Blank = tool default (1000).
POOL="${MA_POOL_SIZE:-2000}"

echo "========================================"
echo "ml_missalignment_warp_auto    [MODE: $MODE]"
echo "Config:        $CONFIG"
echo "Input dir:     $INPUT_DIR"
echo "Conda env:     $MA_ENV"
if [ "$MODE" = "infer" ]; then
    echo "Model run dir: ${MODEL_RUN_DIR:-<UNSET — set MA_MODEL_RUN_DIR>}"
    echo "Infer devices: $INFER_DEV    Start iter: $START_ITER    Prepare stacks: ${PREP:-<skip>}"
else
    echo "Train devices: $TRAIN_DEV    Recon devices: $RECON_DEV"
    echo "Dataloaders:   $DLOADERS    Start iter: $START_ITER    Prepare stacks: ${PREP:-<skip>}    Pool size: ${POOL:-<default>}"
fi
echo "Started:       $(date)"
echo "----------------------------------------"
echo "REMINDER: miss-alignment REFINES a coarse alignment; it does NOT align raw stacks."
echo "If '$INPUT_DIR' has not been coarse-aligned first (AreTomo→ts_import_alignments, or"
echo "WarpTools ts_etomo_patches + ts_autolevel), the tomograms will come out featureless."
echo "========================================"

module load miniconda/latest 2>/dev/null || echo "WARNING: could not load miniconda/latest"

# Activate the miss-alignment env (its own torch/CUDA stack; NOT the warp env).
if ! conda activate "$MA_ENV" 2>/dev/null; then
    echo "ERROR: could not 'conda activate $MA_ENV'."
    echo "Create the env once (per the miss-alignment README):"
    echo "  conda create -n $MA_ENV -c conda-forge python=3.11 cuda-toolkit=12.9 -y"
    echo "  conda activate $MA_ENV"
    echo "  python -m pip install torch==2.8.0 numpy"
    echo "  python -m pip install torch-projectors --index-url https://warpem.github.io/torch-projectors/cu129/simple/"
    echo "  python -m pip install 'git+https://github.com/warpem/miss-alignment.git'"
    echo "    (or, from a local checkout:  cd /path/to/miss-alignment && pip install -e .)"
    exit 1
fi

if ! command -v miss-alignment >/dev/null 2>&1; then
    echo "ERROR: 'miss-alignment' is not on PATH inside env '$MA_ENV'."
    echo "The env exists but the miss-alignment PACKAGE isn't installed in it. With the"
    echo "env active, install it FROM ITS SOURCE (not from your data folder):"
    echo "  python -m pip install 'git+https://github.com/warpem/miss-alignment.git'"
    echo "  # or from a local checkout:  cd /path/to/miss-alignment && pip install -e ."
    echo "Then check:  miss-alignment --help"
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: input dir not found: $INPUT_DIR  (run ts_import + ts_stack first)."
    exit 1
fi

# An EMPTY CUDA_VISIBLE_DEVICES hides ALL GPUs; unset it if inherited empty.
if [ -n "${CUDA_VISIBLE_DEVICES+x}" ] && [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    unset CUDA_VISIBLE_DEVICES
    echo "Unset empty CUDA_VISIBLE_DEVICES (it was masking every GPU)"
fi

# Infer mode needs a finished training run to load models from.
if [ "$MODE" = "infer" ]; then
    if [ -z "$MODEL_RUN_DIR" ]; then
        echo "ERROR: MA_MODE=infer needs MA_MODEL_RUN_DIR — the finished training run that"
        echo "holds iter1/model.ckpt … iterN/model.ckpt. It is unset."
        exit 1
    fi
    if ! ls "$MODEL_RUN_DIR"/iter*/model.ckpt >/dev/null 2>&1; then
        echo "ERROR: no iter*/model.ckpt under MA_MODEL_RUN_DIR='$MODEL_RUN_DIR'."
        echo "Point it at the directory that CONTAINS the iter1/ … iterN/ folders from training."
        exit 1
    fi
    n_models=$(ls -d "$MODEL_RUN_DIR"/iter*/model.ckpt 2>/dev/null | wc -l | tr -d ' ')
    echo "infer: found $n_models trained model checkpoint(s) under $MODEL_RUN_DIR."
fi

# Seed a config from the installed package template if the user has none yet, then
# stop so they can review it (the data path and devices must be set deliberately).
if [ ! -f "$CONFIG" ] && [ "$MODE" = "infer" ]; then
    # INFERENCE config: reuse a finished run's models on this dataset. Different shape
    # from the training config (data_directory + model_run_directory; no training/
    # shift_generation sections).
    _abs_input="$(cd "$INPUT_DIR" && pwd)"
    echo "Inference config '$CONFIG' not found; writing a starter inference config…"
    cat > "$CONFIG" <<EOF
general:
  # MissAlignment INFERENCE: reuse a finished run's models to align THIS dataset. No training.
  # For iteration N the model at <model_run_directory>/iterN/model.ckpt is loaded and applied.
  data_directory: ${_abs_input}/        # this dataset's warp_tiltseries (must already hold a coarse alignment)
  model_run_directory: ${MODEL_RUN_DIR}        # finished training run holding iter1/.. iterN/model.ckpt
  apply_ctf: False
  iteration_settings:                   # MUST match the training run's settings; length <= number of models
    - { downsample: 3, alignment: anchoring }
    - { downsample: 2, alignment: anchoring }
    - { downsample: 1, alignment: global }
    - { downsample: 1, alignment: global }
    - { downsample: 1, alignment: [3, 3] }
    - { downsample: 1, alignment: [3, 3] }
  seed: 45132

tilt_series_alignment:
  patch_size: 96      # same as training
  patch_overlap: 0.1
  batch_size: 32
EOF
    echo ">> REVIEW '$CONFIG' before running again:"
    echo "     - data_directory   = ABSOLUTE path of '$INPUT_DIR'"
    echo "     - model_run_directory = your finished training run (the dir with iterN/model.ckpt)"
    echo "     - iteration_settings MUST match the training run AND not exceed $n_models model(s)"
    echo "   Then run this step again."
    exit 2
fi

if [ ! -f "$CONFIG" ]; then
    echo "Config '$CONFIG' not found; trying to seed it from the installed package…"
    TEMPLATE=$(python -c "import os,miss_alignment as m; p=os.path.join(os.path.dirname(m.__file__),'config_template.yaml'); print(p if os.path.exists(p) else '')" 2>/dev/null || true)
    if [ -n "$TEMPLATE" ] && [ -f "$TEMPLATE" ]; then
        cp "$TEMPLATE" "$CONFIG"
        echo "Wrote a starter config to: $CONFIG"
    else
        echo "Package ships no template; writing the standard miss-alignment config"
        echo "(docs/config_template.yaml) with training_directory pre-filled."
        _abs_input="$(cd "$INPUT_DIR" && pwd)"
        cat > "$CONFIG" <<EOF
general:
  # MissAlignment iteratively trains models and realigns the tilt-series
  training_directory: ${_abs_input}/   # Warp warp_tiltseries dir (XMLs + tilt stacks)
  apply_ctf: False                # leave False; enabling CTF doubles processing time with no alignment benefit
  iteration_settings:             # one entry per iteration (length = number of iterations)
    # alignment modes: "global" (single pass), "anchoring" (iterative), "spline" (coarse-to-fine), [N, N] (local warp grid)
    # Speed-tuned: 6 iterations (stock miss-alignment uses 8 — for the slower, slightly
    # higher-quality run add two more "- { downsample: 1, alignment: [3, 3] }" lines).
    - { downsample: 3, alignment: anchoring }
    - { downsample: 2, alignment: anchoring }
    - { downsample: 1, alignment: global }
    - { downsample: 1, alignment: global }
    - { downsample: 1, alignment: [3, 3] }
    - { downsample: 1, alignment: [3, 3] }
  seed: 45132

model_training:
  model_architecture: 'default'
  model_checkpoint: null          # only used as initialization for iteration 0
  loss_margin: 0.5
  learning_rate: 1.0e-3
  weight_decay: 1.0e-4            # set to 0 to disable AdamW weight decay
  max_epochs_per_iteration: 18    # speed-tuned (stock = 30); the contrastive loss plateaus early on a clean set
  warmup_steps: 500
  multistep_lr_scheduler:
    milestones: [5, 15]
    gamma: 0.5

data_loading:
  batch_size: 32
  patch_size: 96
  steps_per_epoch: 1000

# synthetic shift generation for contrastive training (units = pixels)
shift_generation:
  trajectory_probability: .5
  trajectory_max_shift: 10.0
  jitter_probability: .5
  jitter_max_std: 2.0
  outlier_probability: .5
  outlier_max_shift: 20.0
  fracture_probability: .5
  fracture_max_shift: 20.0

tilt_series_alignment:
  patch_size: 96
  patch_overlap: 0.1
  batch_size: 32
EOF
    fi
    echo ">> EDIT '$CONFIG' before running:"
    echo "     - set general.training_directory to the ABSOLUTE path of '$INPUT_DIR'"
    echo "     - review iterations / downsample / alignment mode / shift settings"
    echo "   Then run this step again."
    exit 2
fi

# ---------------------------------------------------------------------------
# PATH SYNC — the caller (tomogration / your command line) is the SOURCE OF TRUTH.
#
# miss-alignment reads WHICH DATA to process from the YAML's general.data_directory
# (infer) / general.training_directory (train), NOT from the input_dir we pass. So a
# config copied from another project silently processes THAT project's data — you only
# notice when the series count is wrong (e.g. "Preparing stacks 249/290" on a 72-series
# project), and it will happily overwrite the OTHER dataset's .xml files.
#
# So: rewrite those keys in place to match what we were actually given. Comments and
# formatting are preserved (targeted line edit, not a YAML re-dump), the original is
# backed up once, and every change is logged loudly.
_ABS_INPUT="$(cd "$INPUT_DIR" && pwd)"
_ABS_MODEL=""
[ -n "$MODEL_RUN_DIR" ] && [ -d "$MODEL_RUN_DIR" ] && _ABS_MODEL="$(cd "$MODEL_RUN_DIR" && pwd)"
SYNC_MSG=$(python - "$CONFIG" "$MODE" "$_ABS_INPUT" "$_ABS_MODEL" <<'PYEOF' 2>/dev/null || true
import sys, re, os, shutil, datetime
cfg_path, mode, abs_input, abs_model = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
try:
    text = open(cfg_path, errors="replace").read()
except OSError:
    sys.exit(0)

want = {}
if mode == "infer":
    want["data_directory"] = abs_input + "/"
    if abs_model:
        want["model_run_directory"] = abs_model + "/"
else:
    want["training_directory"] = abs_input + "/"

changed, out = [], text
for key, newval in want.items():
    # key: <value>   [# trailing comment]   — replace only <value>
    pat = re.compile(r'^(?P<i>[ \t]*)(?P<k>%s)[ \t]*:[ \t]*(?P<v>[^#\n]*?)[ \t]*(?P<c>#.*)?$'
                     % re.escape(key), re.M)
    m = pat.search(out)
    if not m:
        continue
    old = (m.group("v") or "").strip()
    if old.rstrip("/") == newval.rstrip("/"):
        continue
    rep = "%s%s: %s%s" % (m.group("i"), m.group("k"), newval,
                          ("   " + m.group("c")) if m.group("c") else "")
    out = out[:m.start()] + rep + out[m.end():]
    changed.append("%s: %s  ->  %s" % (key, old or "(empty)", newval))

if changed:
    bak = "%s.bak_%s" % (cfg_path, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    shutil.copy2(cfg_path, bak)
    open(cfg_path, "w").write(out)
    print("BACKUP %s" % os.path.basename(bak))
    for c in changed:
        print(c)
PYEOF
)
if [ -n "$SYNC_MSG" ]; then
    echo "-------------------------------------------------------------------"
    echo "CONFIG PATH SYNC — '$CONFIG' pointed somewhere else; corrected to match this job:"
    while IFS= read -r line; do echo "   $line"; done <<< "$SYNC_MSG"
    echo "   (the YAML decides which data is processed, so it must match input_dir)"
    echo "-------------------------------------------------------------------"
else
    echo "config paths already match this job (data dir: $_ABS_INPUT)"
fi

# Pre-flight: catch a structurally broken config (missing/misplaced keys) HERE with a
# clear message, instead of a deep KeyError traceback from inside miss-alignment. The
# commonest break is hand-editing: dropping general.seed or putting iteration_settings
# at the top level instead of nested under general:.
CFG_CHECK=$(python - "$CONFIG" "$MODE" <<'PYEOF' 2>/dev/null || true
import sys, yaml
mode = sys.argv[2] if len(sys.argv) > 2 else "train"
try:
    cfg = yaml.safe_load(open(sys.argv[1])) or {}
except Exception as e:
    print("YAML did not parse: %s" % e); sys.exit(0)
if not isinstance(cfg, dict):
    print("top level is not a mapping"); sys.exit(0)
problems = []
g = cfg.get("general")
if not isinstance(g, dict):
    problems.append("the 'general:' section is missing or malformed")
    g = {}
if mode == "infer":
    req_general = ("data_directory", "model_run_directory", "iteration_settings")
    req_sections = ("tilt_series_alignment",)
    misplaced = ("iteration_settings", "data_directory", "model_run_directory")
else:
    req_general = ("training_directory", "seed", "iteration_settings")
    req_sections = ("model_training", "data_loading")
    misplaced = ("iteration_settings", "seed", "apply_ctf")
for k in req_general:
    if k not in g:
        problems.append("general.%s is missing" % k)
for k in misplaced:                          # belong UNDER general
    if k in cfg and k not in g:
        problems.append("'%s:' is at the TOP LEVEL — indent it under 'general:'" % k)
for sec in req_sections:
    if sec not in cfg:
        problems.append("the top-level '%s:' section is missing" % sec)
for p in problems:
    print(p)
PYEOF
)
if [ -n "$CFG_CHECK" ]; then
    echo "ERROR: '$CONFIG' is not a complete miss-alignment config:"
    while IFS= read -r line; do echo "   - $line"; done <<< "$CFG_CHECK"
    echo ""
    echo "miss-alignment needs the FULL config structure (nesting matters: iteration_settings"
    echo "and seed live UNDER 'general:'). Fastest fix: move '$CONFIG' aside and re-run this"
    echo "step to reseed a complete, speed-tuned template, then re-apply just the VALUES you"
    echo "want to change (don't delete whole sections)."
    exit 3
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Reduce CUDA fragmentation between the training and alignment phases (the alignment
# workers OOM'd with "reserved but unallocated" memory) — recommended by torch itself.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Pre-flight: miss-alignment reads EVERY <series>.xml in the dir directly — it does NOT
# honour Warp's 'deselected' flag. A series that failed coarse alignment has
# VolumeDimensionsAngstrom = [0,0,0] and kills an alignment worker
# ("... has zero values in 'VolumeDimensionsAngstrom'"). Move any such un-aligned XMLs
# aside so the run processes only the good series. Reversible: they go to _no_alignment/.
PARKED=$(python - "$INPUT_DIR" <<'PYEOF' 2>/dev/null || true
import sys, glob, re, os, shutil
d = sys.argv[1]; park = os.path.join(d, "_no_alignment"); moved = []
for x in sorted(glob.glob(os.path.join(d, "*.xml"))):
    try:
        t = open(x, errors="replace").read()
    except OSError:
        continue
    m = re.search(r'VolumeDimensionsAngstrom[^\d\-]*(-?[\d.]+)[,\s]+(-?[\d.]+)[,\s]+(-?[\d.]+)', t)
    if m and all(float(v) == 0 for v in m.groups()):
        os.makedirs(park, exist_ok=True)
        shutil.move(x, os.path.join(park, os.path.basename(x)))
        moved.append(os.path.basename(x)[:-4])
print(" ".join(moved))
PYEOF
)
if [ -n "$PARKED" ]; then
    echo "Parked un-aligned tilt series (zero VolumeDimensionsAngstrom) -> $INPUT_DIR/_no_alignment/:"
    echo "  $PARKED"
    echo "  (These failed coarse alignment; miss-alignment can't refine them. To recover:"
    echo "   re-align them with AreTomo, then move their .xml back and re-run.)"
fi

PREP_ARG=()
[ -n "$PREP" ] && PREP_ARG=(--prepare-stacks "$PREP")
POOL_ARG=()
[ -n "$POOL" ] && POOL_ARG=(--pool-size "$POOL")

# The CLI changed shape between releases: newer builds are a Typer group with
# 'train' / 'infer' SUBCOMMANDS; older builds are a single flat 'miss-alignment
# [OPTIONS]' with no subcommands (and no inference at all). Detect which we have so
# this works on both — a subcommand probe exits 0 only if that subcommand exists.
_has_subcmd() { miss-alignment "$1" --help >/dev/null 2>&1; }

set +e
if [ "$MODE" = "infer" ]; then
    if ! _has_subcmd infer; then
        echo "ERROR: your installed miss-alignment has NO 'infer' subcommand — it's an"
        echo "older, flat-CLI build (Usage: miss-alignment [OPTIONS]); inference to reuse a"
        echo "trained model was added later. Upgrade JUST the package (leave the torch/CUDA"
        echo "stack alone) in this env:"
        echo "  conda activate $MA_ENV"
        echo "  python -m pip install -U --force-reinstall --no-deps 'git+https://github.com/warpem/miss-alignment.git'"
        echo "  miss-alignment infer --help    # confirm the subcommand now exists"
        echo "Then re-run this step. (After upgrading, TRAINING also uses 'miss-alignment"
        echo "train …' — this wrapper detects that automatically.)"
        status=2
    else
        # Inference: load each iteration's trained model and apply it; no training, so no
        # multi-trainer pool-dir race. Alignment uses all visible GPUs (CUDA_VISIBLE_DEVICES).
        echo "Running miss-alignment infer  (CUDA_VISIBLE_DEVICES=$INFER_DEV)…"
        CUDA_VISIBLE_DEVICES="$INFER_DEV" miss-alignment infer \
            --config-file "$CONFIG" \
            --start-at-iteration "$START_ITER" \
            "${PREP_ARG[@]}" \
            $MA_EXTRA_ARGS
        status=$?
    fi
else
    # Newer builds train via 'miss-alignment train …'; older flat builds run it directly.
    if _has_subcmd train; then
        echo "Running miss-alignment train…"
        TRAIN_CMD=(miss-alignment train)
    else
        echo "Running miss-alignment (flat CLI — no subcommand)…"
        TRAIN_CMD=(miss-alignment)
    fi
    "${TRAIN_CMD[@]}" \
        --config-file "$CONFIG" \
        --training-devices "$TRAIN_DEV" \
        --reconstruction-devices "$RECON_DEV" \
        --dataloaders-per-trainer "$DLOADERS" \
        --start-at-iteration "$START_ITER" \
        "${POOL_ARG[@]}" \
        "${PREP_ARG[@]}" \
        $MA_EXTRA_ARGS
    status=$?
fi
set -e

echo "========================================"
if [ "$status" -eq 0 ]; then
    echo "miss-alignment finished OK."
    echo "Aligned parameters are written back into the Warp .xml files under"
    echo "$INPUT_DIR — go straight to ts_ctf / ts_reconstruct."
    echo "(No ts_import_alignments needed; that step is only for the AreTomo path.)"
else
    echo "miss-alignment exited with status $status — see the messages above."
fi
echo "Finished: $(date)"
echo "========================================"
exit "$status"
