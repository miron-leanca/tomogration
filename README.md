<p align="center">
  <img src="Tomogration-icon.png" width="128" alt="Tomogration logo">
</p>

<h1 align="center">Tomogration</h1>

<p align="center"><b>A GUI controller for the cryo-electron tomography preprocessing pipeline.</b></p>

Tomogration is a desktop app (PySide6) that drives a full cryo-ET workflow — from
raw movies to picked particles — by orchestrating the standard external tools
([WarpTools 2.0](https://github.com/warpem/warp),
[AreTomo2](https://github.com/czimaginginstitute/AreTomo2),
[IMOD](https://bio3d.colorado.edu/imod/), and the deep-learning aligner
[miss-alignment](https://github.com/warpem/miss-alignment)). Every step is a
button with a form; the app builds the exact shell command, shows it to you
(editable), runs it, streams the log, and tracks outputs — so you get the
reproducibility of the command line with the convenience of a GUI.

![The Tomogration main window](docs/screenshot.png)

> **Scope:** raw data preparation → template matching → particle export → the
> **RELION 4 loop** (Class3D handoff, class selection, re-extraction) → the
> **M refinement group** (population/source/species setup, MCore runs, weights,
> version tracking). The heavy lifting still happens in RELION's and M's own
> engines — Tomogration builds, launches, and audits those runs, it doesn't
> reimplement classification or refinement.

Tomogration is a **wrapper/controller**: it doesn't reimplement any cryo-ET
algorithm, it runs the real tools and manages the bookkeeping around them
(file naming, versioned outputs, tilt exclusions, mdoc repair, run history).

---

## Why it exists

The WarpTools tilt-series pipeline is powerful but is a long sequence of
command-line steps, each with fiddly interdependencies (pixel sizes that must
match, alignment that must be imported before CTF, tilt-axis quirks from Tomo5
mdocs, GPU/driver traps, versioned AreTomo folders…). Tomogration encodes those
dependencies and the hard-won gotchas so you — and your collaborators — don't
rediscover them each time. The command it runs is always visible and editable,
so nothing is hidden and you can still drop to the terminal any time.

---

## What the interface gives you

- **A pipeline laid out as clickable steps**, grouped (Data prep → Gain → Tilt
  series → Alignment → CTF → Reconstruct → Pick), each with a status dot.
- **A parameter form per step** with inline help, and a **live command box** that
  shows exactly what will run. Edit the form *or* the command directly.
- **Your parameter edits persist** — per step, across restarts — until you change
  them, a newer input version supersedes them, or you hit **Reset defaults**.
- **A queue** to chain steps, a streaming **terminal**, and a **directory
  overview** that colour-codes inputs/outputs and opens folders on click.
- **Open outputs in one click** — a 📂 (file manager) and a **3dmod** button next
  to each step's output, so you never hand-type `module load 3dmod; 3dmod *.mrc`.
- **A Tilt Inspector** to review series, exclude bad tilts/whole series, and open
  them in 3dmod. Sorting files the coarse per-series `Position*.mrc` tilt stacks
  into `mrcs-tiltseries-coarse/` and the inspector opens them from there, so the
  project root stays a dozen folders instead of thousands of MRCs (a project
  sorted by an older build is tidied on **Init dirs** or on opening the
  inspector — nothing is overwritten, and only root-level `Position*.mrc` moves).
- **A `Project` button** beside the root box, for switching project on the fly.
  It lists your bookmarked folders and the projects inside each one, ticking the
  one you're in. A bookmark can be a project root *or* a folder holding several
  — both are listed. Bookmarks live in `~/.tomogration.json` (`project_dirs`)
  and are editable from the menu, one path per line.
- **A Processing History** window: a flowchart of every run, its non-default
  parameters, and clickable input/output files (openable with 3dmod/gedit).
- **A workflow-graph canvas** (the default view): every run is a **job card**
  with its own `jobs/J###` output dir. Cards can be queued, forked, wired into
  downstream jobs, dragged into branches, cleared, or deleted; work done
  outside the app (pick sets, RELION jobs) is discovered on disk and offered for
  adoption as cards. The canvas view is
  `[ pinned pipeline outline | canvas | full-height terminal ]`:
  - the **pipeline outline** is pinned down the left window edge (the default
    pipeline as an always-visible reference — click a step to show its work,
    double-click to open it, drag it onto the canvas to add a job); job types
    are filed under collapsible **categories** (the pipeline groups), each
    with a rolled-up job count;
  - the canvas itself is striped into **category bands** (dividing lines +
    captions per pipeline group), so Warp processing, RELION, M and the
    membrane/IsoNet branch read as separate regions and new cards pop into
    their category's band;
  - each card has a **❯ pop-out panel** with three tabs — **Builder** (the
    job's parameter form and run/queue buttons), **Details** (what the job
    does and what every parameter means), **Outputs** (where its files are).
    Panels stay open, track their job live, several can be open at once, and
    each is **tethered to its card** — it follows the card through pans,
    zooms and drags (move a panel and your chosen offset is kept);
  - **Outputs rows drag into input fields** of any open builder — file vs
    folder and relative vs absolute are converted to what the field wants;
  - the **terminal** runs the full height on the right and has tabs: the app's
    run log plus any number of scratch shell tabs (＋). The widths you drag
    the outline/canvas/terminal splitters to are remembered and become the
    default.

---

## The pipeline, step by step

| Group | Step | What it does |
|---|---|---|
| **1. Data prep** | Rename / Sort | Rename raw EER + mdocs to `Position###`, sort into the project layout (`frames/`, `mdocs/`, `gains/`, `mrcs-tiltseries-coarse/`) |
| | Inspect tilt stacks | Review series, mark bad tilts / whole series for exclusion |
| | Remake mdocs | Apply tilt exclusions and renumber mdoc ZValue blocks |
| **2. Gain** | gain convert / reciprocal | Convert the `.gain` reference to the `.mrc` Warp expects |
| **3. Frame series** | create settings / fs_motion_and_ctf | Motion-correct + estimate per-movie CTF |
| **4. Tilt series** | ts_import / ts_stack | Build `.tomostar` tilt series and the aligned tilt stacks |
| **5. Alignment** | **Align with AreTomo2** | Coarse fiducial-less alignment (writes IMOD `.xf`/`.tlt`) |
| | **ts_import_alignments** | Pull AreTomo's alignment into the Warp `.xml` |
| | **sync selection** | Deselect series that have no alignment |
| | **miss-alignment (train)** | Train a DL model on this set to *refine* the coarse alignment |
| | **miss-alignment (infer)** | *Reuse* that trained model to align a larger set, no retraining |
| **6. CTF** | ts_defocus_hand / ts_ctf | Fix defocus handedness, estimate tilt-series CTF |
| **7. Reconstruct** | ts_reconstruct | Back-project aligned, CTF-corrected tilts into tomograms |
| **8. Pick** | ts_template_match / threshold | Template-match particles and threshold the picks |
| **9. Export** | ts_export_particles | Extract particles into a RELION project (3D subtomos for RELION 4, or 2D for RELION 5) |
| **10. RELION 4** | RELION 4: convert STAR + init ref | Convert the Warp export star to RELION 4 format (rewrite particle paths), and build a de-novo initial reference from a random particle subset |
| | Merge optics groups | Collapse per-export optics groups so RELION counts particles once, not per group |
| | Check particles exist | Verify every particle image the star names is on disk (and optionally prune the missing) |
| | RELION 4: Class3D handoff | Pre-scale the reference, check the launch-root invariant, and submit 3D classification |
| | Select good class → picks | Turn a Class3D/Select result back into per-tomogram Warp pick stars |
| | Re-extraction = ts_export_particles from a RELION star | Build the Export card downstream of a Subset-selection card: Warp reads the RELION star directly, applies the refined shifts itself, and cuts the same particles at a finer pixel size |
| | Verify re-extraction | Cross-check the re-extracted set against the selection (counts, scale, recentring) |
| **11. M refinement** | create population / source / species, mask | Set up an M project from the RELION results (population, data source, species with half-maps and mask) |
| | M: refine (MCore) | Run MCore refinements — image/volume warp, CTF, defocus — one enabled thing at a time |
| | estimate weights / resample trajectories | Per-series then per-tilt exposure weights; finer temporal pose sampling |
| | CTF pre-flight, version index, reset, kill orphans | Guard against the IndexOutOfRange CTF trap, label and browse refinement versions, and recover a wedged M setup |

Left-panel documentation for every step (what/why/parameters/pitfalls) lives in
`tomogration_docs.json` and is shown in-app.

### Alignment: AreTomo **then** miss-alignment (not either/or)

This is the most misunderstood part of the workflow, so it's enforced in the app:

- **AreTomo2** does the coarse, fiducial-less alignment and (crucially) **solves
  the tilt-axis angle**. Its `.xf`/`.tlt` are pulled into Warp with
  `ts_import_alignments`.
- **miss-alignment is a *refiner*, not an alternative** — its own docs say it
  *"starts from an initially coarse aligned dataset."* Run on raw stacks it produces
  a featureless tomogram. Tomogration therefore **blocks** the train step unless a
  coarse AreTomo alignment exists, and on a fresh run it auto-prepends the
  AreTomo→import→select chain so the trained model starts from the right place.
- **Train once, infer everywhere.** Train a model on a small, clean subset
  (`miss-alignment (train)`), then apply it to your full dataset with
  `miss-alignment (infer)` — no retraining. Infer reuses the saved
  `iterN/model.ckpt` checkpoints.

miss-alignment needs its **own conda env** (it ships its own CUDA/torch stack —
*not* the Warp env):

```bash
conda create -n miss-alignment -c conda-forge python=3.11 cuda-toolkit=12.9 -y
conda activate miss-alignment
python -m pip install torch==2.8.0 numpy
python -m pip install torch-projectors --index-url https://warpem.github.io/torch-projectors/cu129/simple/
python -m pip install "git+https://github.com/warpem/miss-alignment.git"
```

---

## Requirements

**On the workstation (Linux):**
- A CUDA GPU + NVIDIA driver, with the external tools installed and loadable
  (typically via `module load` / conda): **WarpTools 2.0**, **AreTomo2**,
  **IMOD/3dmod**, and (optional) **miss-alignment** in its own conda env.
- Python 3.10+ for the GUI. `install.sh` builds an isolated `.venv` with PySide6
  (Debian/Ubuntu block system `pip` under PEP 668, so a venv is required).
- Qt 6.5+ needs `libxcb-cursor0`; `fetch_xcb_libs.sh` fetches it without sudo if
  it's missing.

Tomogration is developed on macOS but **only runs on Linux** — on the Mac it is
only syntax-checked, never launched.

---

## Install & launch

Copy this whole folder to the workstation, then from inside it:

```bash
bash install.sh      # one-time: builds .venv, fetches Qt libs, registers the launcher
bash tomogration.sh  # launch (or use the Applications menu → Science → Tomogration)
```

`install.sh` is idempotent and self-healing (it rebuilds the venv if a different
machine's Python broke it, skips work already done, and self-tests). **Run it once
per workstation** — the `.venv` may live on shared storage, but each machine's
`$HOME` launcher entry is local.

---

## What every file is

```
tomogration_app.py            The PySide6 application: main window, job-card canvas,
                              parameter forms, queue, streaming terminal, inspectors.
tomogration_core.py           Tiny shared helpers (script paths, tilt-range expansion,
                              progress-line keys). Bottom of the import DAG.
tomogration_stages.py         The data-driven pipeline: the STAGES list, per-stage
                              params/docs/validators, and build_command (the pure
                              command assembler the command box is seeded from).
tomogration_project.py        ProjectState — all filesystem inspection/bookkeeping
                              (status, history, exclusions, groups, mdoc repair).
tomogration_jobs.py           The job model: the .tomogration_jobs.json store, the
                              canvas DAG layout, and discovery of outside work.
tomogration_relion_handoff_skeleton.py  Shared skeleton for the RELION handoff stages.
tomogration_docs.json         Per-step documentation shown in the left panel.
tests/                        The no-GPU test suite (python3 tests/run_all.py) —
                              stubs PySide6 and exercises the pure logic anywhere.
docs/                         Screenshot + the printable pipeline reference.
tomogration.sh                Self-locating launcher (finds the venv, sets Qt paths, runs the app).
tomogration.desktop           XDG desktop-entry template (install.sh fills in the path).
install.sh                    Builds the venv, fetches Qt libs, registers the launcher.
fetch_xcb_libs.sh             Fetches libxcb-cursor0 without sudo when the wheel lacks it.
Tomogration-icon.png          Launcher icon.
tomogration_brief.md          Design brief / internal reference for the pipeline.
LICENSE                       Apache-2.0 (this wrapper). External tools keep their own licenses.
CONTRIBUTING.md               How to contribute + how to test without a GPU/display.

Companion scripts the app shells out to (keep them beside tomogration_app.py —
their paths are resolved automatically):

ml_batch_rename_eer_mdoc_mrc_warp_auto.sh   Rename raw EER/mdoc/mrc to Position### and sort.
ml_imodtowarpkey_generator_warp_auto.py     Build the IMOD→acquisition-order conversion key.
ml_batch_remake_mdocs_warp_auto.sh          Apply tilt exclusions + renumber mdoc ZValues.
ml_aretomo2_warp_auto.sh                    Parallel AreTomo2 farm (handles the CUDA/setgid traps).
ml_missalignment_warp_auto.sh               miss-alignment wrapper (train + infer modes).
ml_cryolo_to_warp_picks_auto.py             Convert crYOLO coordinate files to Warp pick stars.
ml_relion4_convert_star_warp_auto.sh        Convert the Warp export star to RELION 4 (path rewrite) + build an initial reference.
ml_relion4_handoff_warp_auto.sh             RELION 4 Class3D handoff (reference prep + launch-root guard + submit).
ml_relion4_merge_optics.py                  Collapse per-export optics groups in a RELION star.
ml_relion4_select_picks.py                  Class3D/Select result → per-tomogram Warp pick stars.
ml_star_check_particles.py                  Verify (and optionally prune) particles a star names.
ml_verify_reextract.py                      Cross-check a re-extraction against its selection.
ml_m_check_ctf.py                           M pre-flight: find series whose CTF would crash MCore.
ml_m_index_versions.py                      Label and index M species/refinement versions.
ml_m_setup_warp_auto.sh                     Manual M population/source setup helper (not called by the GUI).
ml_m_bisect_warp_auto.sh                    Manual bisection of an MCore IndexOutOfRange crash (not called by the GUI).
ml_m_reset_warp_auto.sh                     Move a wedged M setup aside (trash, not delete) for a clean restart.
ml_add_thumbnails_warp_auto.sh              Carry Tomo5 thumbnails into a merged project, renamed.
ml_make_training_set_warp_auto.sh           Build a fresh miss-alignment training subset (selected/).
ml_merge_datasets_warp_auto.sh              Merge multiple grids into one continuously-numbered project.
```

---

## How settings persist

- **Per-step parameters** you edit are saved and restored automatically, in
  `.tomogration_params.json` **in the project root** (so they follow the dataset
  across VMs on shared storage; older per-machine `~/.tomogration.json` edits are
  migrated once). A step's **Reset defaults** button discards its saved edits; a
  newer input version (e.g. a fresh AreTomo folder) automatically supersedes the
  stored value for that field.
- **`warp_launch`** — how `WarpTools` is invoked on your cluster (e.g.
  `module load warp && WarpTools`) — is set once from the Tools menu (stored
  per-machine in `~/.tomogration.json`).
- **Jobs and the workflow graph** live in `.tomogration_jobs.json` in the project
  root; **processing history** in `.tomogration_history.json` beside it. Existing
  outputs are archived (never overwritten) on re-run of the steps that support it.

---

## Troubleshooting (the traps this app was built to survive)

- **AreTomo "GPU is invalid" / `libcufft.so.11` not found.** The shared AreTomo2
  binary is often setgid (the loader then ignores `LD_LIBRARY_PATH`) and cluster
  CUDA modules ship a *stub* `libcuda`. `ml_aretomo2_warp_auto.sh` runs a
  non-setgid copy from a stub-free library farm to avoid both.
- **`ts_import_alignments`: "Could not find PositionNNN.xf".** AreTomo auto-versions
  its output folder; import from the folder that actually holds the `.xf` (the app
  tracks the newest one). If a run was killed, alignments may be partial — deselect
  the unaligned series with the **sync selection** helper.
- **miss-alignment made a featureless tomogram.** It refines; it doesn't align raw
  stacks. Coarse-align with AreTomo first (the app enforces this).
- **`ts_reconstruct` "stuck at 0/15".** You asked for native pixel size — a full
  native tomogram is ~260× a 10 Å one. Reconstruct at `--angpix 10`.
- **`ts_template_match` "reconstruction at the desired resolution was not found".**
  `--tomo_angpix` must equal a `ts_reconstruct --angpix` you already ran.
- **"tilt axis … Tomo5 mdoc files are known to provide incorrect values".** Advisory,
  printed for *every* Tomo5 mdoc — the tilt-*axis* angle (~-174°) is not your tilt
  range, and AreTomo refines it anyway.
- **A GPU is unexpectedly slow.** Orphaned processes from a killed run may still hold
  it. `nvidia-smi` shows only *your* PIDs, so judge by GPU-Util %, and clear
  leftovers with `pkill -u $USER -f <that run's tmp binary hash>`.
- **RELION: "file does not exist" for every subtomogram.** You launched RELION from
  the wrong directory. `ts_export_particles` writes paths relative to
  `--output_processing`; launch RELION from *that* directory (the RELION 4 step does
  this and checks it for you).
- **RELION 5 GPU dies with "error-code 35".** The RELION 5 container's CUDA runtime is
  newer than the host driver. Use RELION 4 (GPU-native) until the driver is bumped —
  which is exactly why the handoff targets RELION 4.

---

## Contributing

Bug reports and focused PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
The golden rule: it's developed on macOS but runs on Linux, so keep
`python3 -m py_compile tomogration_app.py` and `bash -n *.sh` green, and never
commit microscopy data.

## License

[Apache-2.0](LICENSE) © 2026 Miron Leanca and The Rosalind Franklin Institute. Tomogration orchestrates
external tools (WarpTools, AreTomo2, IMOD, miss-alignment, RELION/M) that carry
their own licenses — see the note at the bottom of `LICENSE`.

## Acknowledgements

Tomogration stands on the shoulders of the tools it drives — thanks to the Warp,
AreTomo, IMOD, and miss-alignment developers.

## Re-extracting a RELION selection (bin4 → bin2 → bin1)

There is one route, and it is the export card itself. Right-click the
Subset-selection card (or the found-on-disk RELION job) → **Re-extract with
Warp** / **Build downstream ▸ ts_export_particles**. The card arrives with:

* `input_star` = `Select/jobNNN/particles.star` — Warp reads the star's
  `rlnCoordinateX/Y/Z`, **subtracts the refined `rlnOriginX/Y/ZAngst` itself**
  (divided by the star's own `rlnImagePixelSize`), and copies the refined Euler
  angles into the output star;
* `coords_angpix` = the star's `rlnImagePixelSize`, read from the file (6.28 for
  particles first extracted at bin4). The run is refused if it disagrees;
* `normalized_coords` OFF and the pick-star folder/pattern dropped — a RELION star
  holds pixel coordinates, and declaring them 0-1 fractions multiplies every one
  by the tomogram width.

You choose `output_angpix`, `box` and `diameter` (halve the pixel size → double the
box). Then run **RELION 4: convert STAR + init ref** as after any export; its
`random_subset_ref.mrc` is now an *oriented* average (the angles came along), so it
should look like the particle rather than a blob. `ml_verify_reextract.py` pairs the
two stars and checks `new = (coord − origin/apx) × (apx_old/apx_new)`.

Same extraction code as the crYOLO route (`--input_directory` of normalised pick
stars + `--normalized_coords`): identical subtomogram reconstruction, identical
output coordinates (`position_Å / output_angpix`), no pre-rotation on either route.
The differences are only the coordinate source (fractions × tomogram size vs pixels
× `coords_angpix`), the shifts (none vs the refined origins), and the angle columns
(zeros vs refined). The pick-star converter (`ml_relion4_select_picks.py`, modes
A/B/C) is retired; old cards remain readable.
