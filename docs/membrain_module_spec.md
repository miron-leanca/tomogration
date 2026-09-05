# tomogration2 — MemBrain + IsoNet module

**Brief for Claude Code. Draft 1.** Build iteratively, smallest useful thing first.
Do not build everything in this document at once.

---

## 0. Context

`tomogration2` is an existing node-graph automation layer that drives a cryo-ET
pipeline (WarpTools 2.0 → miss-alignment → RELION 4.0.1). It already has:

- a node graph where each node is a pipeline stage with a config panel
- per-node badges showing progress (`72/72` style counts)
- a JSON-driven docs panel (`tomogration_docs.json`)
- shell wrappers for external tools (e.g. `ml_missalignment_warp_auto.sh`)

This spec adds a **membrane segmentation and picking branch** that starts from
reconstructed tomograms and ends with particle coordinates + orientation priors
that feed back into RELION.

Active dataset: `EML46 / OC43-3mM-disacch`, 72 tilt series,
project root `/ceph/users/haq21239/EMDatasets/EML46/OC43-3mM-disacch/`.
Working tomograms are 12.56 Å/px. Target particles are coronavirus spike
(~200 Å tall, ~100–150 Å spacing) and HE (~50 Å).

---

## 1. Design principle: tune on few, then batch

Every stage must run in one of two modes, selected per node:

| Mode | Input set | Purpose |
|---|---|---|
| **Tune** | 1–3 named tomograms | sweep parameters, compare visually, pick values |
| **Batch** | all 72 | apply the values chosen in Tune |

Tune mode must support **parameter sweeps**: given a list of values for one
parameter, run the stage once per value into separate output folders, collect
the numeric result of each (e.g. component counts), and present them as a table
plus a one-click "open all in viewer" action.

This is the single most important feature. The current manual workflow is
"run, look, adjust, re-run" and that loop is where all the time goes.

---

## 2. Environments and machine constraints

Three separate conda envs on shared storage `/ceph/users/haq21239/.conda/envs`.
They have **conflicting dependencies** and must never be merged:

| Env | Contains | Notes |
|---|---|---|
| `membrainseg` | membrain-seg, tomo_preprocessing, napari 0.8 (PyQt6) | segmentation + deconvolution |
| `membrainpick` | membrain-pick, surforama, napari (PyQt5) | pins `scipy<1.12.1`, `numpy 1.26.4` — installing napari upgrades these and breaks it. Pin `napari==0.5.6`. |
| `isonet` | IsoNet or IsoNet2 | to be created; own torch/tensorflow stack |

Node configs must carry an explicit **env name** field. Wrappers activate the
named env; never assume the caller's env.

**GUI constraint.** `rfi-structbio-workstation-01` can run Qt apps.
`rfi-els-workstation-03` cannot (missing system `libxcb-cursor0`) but has free
GPUs. Nodes must be taggable as *compute* or *interactive*, and the UI should
warn when an interactive node is dispatched to a machine without a working
display. All paths are on `/ceph` so no file transfer is needed between them.

---

## 3. Stage nodes — build in this order

### 3.1 Deconvolve  *(build first — highest value, lowest risk)*

Wraps `tomo_preprocessing deconvolve` (env `membrainseg`).

```
tomo_preprocessing deconvolve \
  --input <tomo>.mrc --output <tomo>_deconv.mrc \
  --df <angstroms> --kv 300 --cs 2.7 \
  --strength 1.0 --falloff 1.0 --ampcon 0.07 --hp-fraction 0.02
```

**Critical:** `--df` is per tilt series and must be read automatically from
`warp_tiltseries/<TS>.xml`, field `<Param Name="Defocus" Value="..."/>`, which
is in **micrometres** — multiply by 10000. Never hardcode. Surface the parsed
value per series in the node so it can be sanity-checked.

`--strength` and `--falloff` are the contrast knobs and should be sweepable in
Tune mode.

Node must refuse to run on an input already marked as denoised (see §3.2);
deconvolving denoised data degrades it.

### 3.2 IsoNet — offer **both** versions as selectable backends

One node, a `version` selector, two wrapper scripts. They are not interchangeable
in their inputs, so the config panel must change with the selection.

**IsoNet 1** (`isonet.py`) — modules: `prepare_star`, `deconv`, `make_mask`,
`extract`, `refine`, `predict`. Iterative; separate manual steps. Works from
plain tomograms, no even/odd needed. Designed for −60 to +60° tilt range.
Mask generation matters when the sample is sparse, and needs decent contrast —
so run after deconvolution.

**IsoNet 2** (`isonet.py refine`) — modules: `prepare_star`, `deconv`,
`make_mask`, `denoise`, `refine`, `predict`. Rewritten in PyTorch. Combines
missing-wedge correction, Noise2Noise denoising and learned CTF correction in a
single optimisation loop, so subtomogram extraction and CTF deconvolution are no
longer separate steps and the whole refinement can run as one command. Roughly
10× faster than IsoNet1 with better output. Prefers **even/odd paired tomograms**
(splittable by frame or tilt) but retains IsoNet1's non-paired path.

Implementation notes:
- The even/odd requirement means a **new upstream node**: reconstruct half-tomograms
  from the existing WarpTools stage. Check whether `ts_reconstruct` can emit halves;
  if not, this must be flagged to the user rather than silently skipped.
- Train once, predict many: the trained model is reusable across a dataset, and
  probably across EML45↔EML46 given their similarity. Model path must be a config
  field independent of the data path — **this exact separation caused a bug before**,
  where a reused YAML's `data_directory` silently pointed at the wrong dataset.
  Show both resolved paths prominently in the node, and warn when the tomogram
  count seen by the job disagrees with the count tomogration expects.
- Output tomograms must be **tagged as wedge-restored** and excluded from any
  downstream particle-extraction path. They are for picking coordinates only.

### 3.3 Segment

Wraps `membrain segment` (env `membrainseg`, GPU).

```
CUDA_VISIBLE_DEVICES=<n> membrain segment \
  --tomogram-path <tomo>.mrc --ckpt-path <model>.ckpt \
  --rescale-patches --in-pixel-size 12.56 \
  --test-time-augmentation --store-probabilities --out-folder <dir>
```

`--store-probabilities` must be **on by default and hard to turn off** — without
the `_scores.mrc` score map, every re-threshold requires a full GPU re-run.
Expose `--store-uncertainty-map` too (note: the flag is `--store-uncertainty-map`,
not `--store-uncertainty`).

Runtime ~4 min/tomogram with 8-fold TTA on one GPU. Show a real progress estimate.

### 3.4 Threshold sweep

Wraps `membrain thresholds`. Cheap, CPU, seconds — this is the main tuning loop.

```
membrain thresholds --scoremap-path <stem>_scores.mrc \
  --thresholds -1.0 --thresholds -2.0 ... --out-folder <dir>
```

- Values repeat as separate flags.
- Output filenames always normalise to one decimal: `-3` → `..._threshold_-3.0.mrc`.
  Downstream nodes must construct the name that way, not from the raw input string.
- Default `--out-folder` is `./predictions` and does **not** follow the scoremap's
  folder. Always pass it explicitly or outputs silently land beside a different run's
  files with colliding names.

### 3.5 Connected components

Wraps `membrain components`.

```
membrain components --segmentation-path <seg>.mrc \
  --connected-component-thres <voxels> --out-folder <dir>
```

- **The tool does not create `--out-folder`.** `mkdir -p` first or it computes the
  whole result and then dies on the write.
- Parse and store the two reported numbers (`Found N` / `Relabeled to M`) as node
  metrics. In Tune mode plot them across the swept parameter — a flat curve means
  fragmentation, a steep drop means debris, and that distinction is the whole point
  of the sweep.
- Add a **component size histogram** action: read the output label volume, report
  voxel count per label, sorted. Needed to spot merged virions (a chain of 3–4 is
  ~4× a single shell) and to pick a representative single virion for testing.
- Add a **extract single label** action: write `label == N` as a binary mask, for
  feeding a single membrane to the mesh stage.

### 3.6 Mesh conversion

Wraps `membrain_pick convert_file` / `convert_mb_folder` (env `membrainpick`).

```
membrain_pick convert_file \
  --tomogram-path <tomo>.mrc --mb-path <seg>.mrc --out-folder <dir> \
  --input-pixel-size 12.56 --step-size <f> --step-numbers <lo> --step-numbers <hi> \
  --barycentric-area 400 --mesh-smoothing 1000
```

- Always pass `--input-pixel-size` explicitly; headers have been unreliable here.
- `--only-largest-component` defaults **on**. For a components volume this silently
  picks the biggest object, which is often a merged cluster, not a virion. Default
  it off in the node and make the choice explicit.
- **Unresolved:** whether `--step-size` / `--step-numbers` are in pixels or Ångströms.
  Do not guess in code. Add a config note and let the user verify visually.

### 3.7 Surforama / annotate / train / predict  *(defer)*

`membrain_pick surforama --h5-path <container>.h5`, then `train`, `predict`,
`mean_shift`, `assign_angles`. Interactive-only, workstation-01.
Do not build until the earlier stages are stable.

---

## 4. Viewer integration

Replace the current manual `tomoview` invocation. A viewer action on any node
should assemble the argument list itself:

- **Compare thresholds:** open the tomogram + every thresholded segmentation from
  a sweep as separate labels layers, named by threshold value.
- **Compare deconvolution settings:** tomogram variants as image layers.
- **Score map heatmap:** open `_scores.mrc` as a float layer with contrast limits
  preset near the score distribution, so dragging the lower limit sweeps the
  threshold live. Report the value under the cursor. This can replace most
  threshold sweeps entirely.
- **Components:** open the label volume with per-label colours, plus the size
  histogram from §3.5 side by side.

Reference implementation exists at `/ceph/users/haq21239/bin/tomoview.py` —
reads MRC via `mrcfile`, auto-scales volumes of differing shape onto the
tomogram grid, routes integer volumes to Labels layers and float volumes to
Image layers. Reuse or absorb it.

---

## 5. Provenance and RELION handoff

Each output must record which upstream variant produced it: raw vs deconvolved
vs IsoNet-corrected, threshold value, component size cutoff, model checkpoint.
Two runs producing `Position003_..._scores.mrc` in different folders have already
been confused once.

**Hard rule to enforce in code:** coordinates may derive from processed tomograms;
particle *extraction* for averaging must always reference the original
reconstruction. Any export node that would extract from a deconvolved,
denoised or wedge-restored volume must refuse and say why.

Final handoff writes a RELION-compatible `.star` with coordinates and the
Euler angles from `assign_angles`, matching the existing
`ts_export_particles` / optimisation-set convention already in tomogration.
`_rlnTomoName` must match the `.tomostar` stem exactly.

---

## 6. Explicitly out of scope for draft 1

- **Geometric surface fitting** (fitting spheres/ellipsoids to virions instead of
  chasing voxel-perfect segmentation). This may replace a large part of §3.4–3.6
  if adopted. Pending a decision — do not design around it, but avoid hard-coding
  assumptions that would block it.
- Mosaic / napari-lasso-3d manual splitting of merged virions.
- Cap recovery for the missing wedge.

---

## 7. General instruction

Do not invent command-line flags. Every wrapper must be written against the real
`--help` output of the installed version, and the module should surface a
"check tool version" action that captures `--help` per stage into the docs panel.
Flag names have already changed between versions of these tools
(`--store-uncertainty` vs `--store-uncertainty-map`,
`--input-tomogram` vs `--input`).
