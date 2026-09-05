# tomogration — Claude Code Project Brief

> Context dump + build spec for rewriting `warp_auto.py` into a three-panel
> cryo-ET pipeline controller. Feed this whole file to Claude Code as the
> project brief (e.g. drop it in as `CLAUDE.md` or paste at session start).
> Compiled from prior Claude chats on the EML40/EML45 tomography work.

---

## 1. Who / what / where

- **User:** Miron, postdoc, Rosalind Franklin Institute (structural biology, cryo-ET).
- **HPC:** ceph cluster, username `haq21239`. GPUs historically V100 (verify per node).
- **Current project root (screenshot):** `/ceph/users/haq21239/EMDatasets/EML45/selected`
- **Scripts dir:** `/ceph/users/haq21239/EMDatasets/processing_scripts/`
- **Prior project:** `EML40` grid 9 — most accumulated parameter knowledge comes from here.
- **Acquisition:** EER format, **apix 1.57 Å/px**, **~3.5 e⁻/Å²/tilt**.
- **Existing tool:** `warp_auto.py` — Python/**tkinter** GUI + companion `ml_*.sh` bash scripts.
  Saves/loads state to `warp_auto.project.json`.

## 2. The pipeline (this is the spine of the new center panel)

Tools: **WarpTools 2.0** (motion/CTF/export) → **AreTomo2** (alignment/recon) →
**IMOD** (inspect/fiducials) → **RELION 5** (`--tomo`) → **M / MCore** (multi-particle refine).

Stage order (each must be independently runnable, each shows its output):

1. **Data prep** — batch rename .eer/.mdoc; generate IMOD→Warp key; remake mdocs (remove bad tilts).
2. **Gain** — convert `.gain`→`.mrc`; compute reciprocal gain (Linux WarpTools).
3. **Frameseries** — `create_settings (fs)` → `fs_motion_and_ctf`.
4. **Tiltseries** — `create_settings (ts)` → `ts_import` → `ts_stack`.
5. **Alignment** — AreTomo2 (`.xf`) → `ts_import_alignments` → sync selection vs alignments.
6. **CTF** — `ts_defocus_hand` (check handedness) → `ts_ctf` (per-tilt refine).
7. **Reconstruct** — `ts_reconstruct` → tomograms.
8. **Pick** — `ts_template_match` → `threshold_picks` (or manual/IMOD, or crYOLO).
9. **Export** — `ts_export_particles` → RELION-compatible `optimisation_set.star`.
10. **Average** — RELION `--tomo` 3D refine → **M** multi-particle refine → final map.

> **Honest scope line:** stages 1–9 are WarpTools/AreTomo CLI the GUI can own directly.
> Stage 10 (RELION/M) has its own pipeliner — the GUI should **launch and hand off**,
> not reimplement averaging. Draw the boundary on the center axis (see §5).

## 3. Established parameters (defaults to pre-fill in the new UI)

These are the values we converged on; treat as defaults, all overridable.

**fs_motion_and_ctf**
- motion grid `1x1x3` (temporal dim ~ frame count), CTF grid `2x2x1`
- m_range 500→10 Å, m_bfac −500

**ts_reconstruct** ⚠ (see gotcha 4.1)
- safe command: `--device_list <N> --perdevice 1 --dont_invert`
- do **NOT** combine `--perdevice 2` with `--deconv` on V100 → SIGABRT (exit 134).
- `--deconv` / `--dont_invert` are dataset-specific, off by default in official guide.

**ts_template_match** (apoferritin example values — adapt per target)
- `--tomo_angpix 10 --subdivisions 3 --template_diameter <Å> --symmetry <e.g. O> --whiten --check_hand 2`

**threshold_picks** — `--minimum 3` (scores normalised to bg mean/SD, comparable across tomos).

**ts_export_particles** (RELION 5 path)
- `--output_angpix 4 --box 64 --diameter <Å> --2d --normalized_coords --relative_output_paths`
- `--2d` = 2D image series (RELION 5 preferred); omit for 3D subtomo volumes.
- choose output_angpix so Nyquist sits just below feature resolution.

## 4. Hard-won gotchas — DO NOT regress these

1. **ts_reconstruct SIGABRT (exit 134, `WorkerConsole.SetFileOutput`)** = `--perdevice 2` +
   `--deconv` cuFFT collision on V100. Keep perdevice=1 when deconv on.
2. **IMOD can't read Warp float16 MRC** (mode 12). Export `WARP_FORCE_MRC_FLOAT32=1`
   before any `3dmod` session.
3. **mdoc integrity on quarantine:** when an `.eer` is moved to `frames/bad/`, the matching
   ZValue block MUST be removed from the mdoc and remaining blocks renumbered, or `ts_import`
   fails with "failed to parse specific tilts". Shared helper: `remove_mdoc_zvalue_block()`.
   Keep the "Repair mdocs" retroactive fixer.
4. **bash `set -e` + `((counter++))`** evaluates to 0 on first increment and silently exits.
   Use `counter=$((counter+1))` or `((++counter))` in companion scripts.
5. **exclusion_list.txt format:** manual tilt-number exclusions above
   `# --- Auto-excluded during processing ---`; remake_mdocs must stop parsing at that header.
6. **AreTomo versioning:** versioned output folders (`aretomo_output/`, `-v2/`…), each with a
   `PARAMETERS.txt` audit record. Generalise this pattern for the iteration queue (§5).

## 5. The redesign — what to build

Three-panel resizable window (use real splitters):

```
┌──────────────┬───────────────────────────┬─────────────────┐
│ LEFT          │ CENTER                     │ RIGHT            │
│ Theory /      │ Vertical pipeline axis     │ Live terminal    │
│ flowchart     │ (tilt series → tomogram    │ (stdout/stderr   │
│ + docs for    │  → picked → averaged)      │  stream, colour  │
│ the selected  │ Per-step: command preview, │  coded, with a   │
│ step          │ editable params, sliders,  │  TERMINATE)      │
│               │ checkboxes, RUN, output    │                  │
└──────────────┴───────────────────────────┴─────────────────┘
```

**Center panel (the controller):**
- Vertical node-per-stage layout following §2. Each node: status dot (grey/green/orange/red),
  RUN button, "show output" expander.
- Click a node → its parameter form appears: **sliders** for bounded numerics
  (B-factor, threshold, binning, tilt-angle range), **checkboxes** for flags
  (`--deconv`, `--dont_invert`, `--whiten`, `--2d`), free-text for paths/patterns/grids
  (`2x2x1` can't be a slider — be honest about which params get which widget).
- **Single source of truth:** an editable command-line box at the bottom of each form that
  stays in sync with the widgets. Whatever is in that box is exactly what runs. Show it
  BEFORE running. Each param carries an inline description + valid range + effect.
- **Raw-file affordances:** Edit / replace / inspect buttons for `.settings`, mdocs,
  `exclusion_list.txt`, star files (reuse the existing Positions Inspector + Repair mdocs).
- Draw a visible boundary after stage 9: stages 10 = "launch RELION / M" handoff buttons.

**Iteration queue (key requirement):**
- For a given stage + dataset, define N parameter variants (e.g. tilt-angle ranges, binning),
  enqueue them, run **sequentially on one GPU**, each writing a versioned output folder +
  `PARAMETERS.txt` (generalise the AreTomo pattern). Show queue progress in the right panel.

**Left panel (theory/docs):**
- Per-selected-step: what the step does, why, parameter ranges + effects, typical pitfalls.
  Static rich content keyed to the center node. Pull text from the protocol doc we built
  (WarpTools guide + Helena Watson RFI adaptations).

**Right panel (terminal):**
- Stream subprocess stdout/stderr live, colour-coded (green=success, orange=warn, red=crash),
  collapse repetitive progress lines, keep the Ctrl-C-equivalent TERMINATE that kills the job
  but not the GUI. Preserve auto-recovery for `fs_motion_and_ctf` only (ts_* crashes are
  resource/GPU, not bad-file).

**ChimeraX (low priority, only if trivial):** a button that runs `chimerax <volume.mrc>`.
Nothing more. Do not spend effort here.

## 6. Engineering recommendations (challenges — read before coding)

1. **Reconsider tkinter.** The three-panel + splitters + embedded live terminal + flowchart +
   real sliders is fighting tkinter the whole way. **PySide6/PyQt6** gives `QSplitter`,
   `QPlainTextEdit` terminal, `QProcess` (clean live stdout + kill), native sliders/checks,
   and `QGraphicsView`/`QWebEngineView` for the flowchart — far less custom plumbing.
   **De-risking move:** the backend (ProjectState, mdoc repair, subprocess/recovery, AreTomo
   versioning) is framework-agnostic Python — keep it, rebuild only the view layer in Qt.
   If staying in tkinter is non-negotiable, it's still doable but the terminal and flowchart
   are the expensive parts.
2. **Don't oversell stage 10.** Averaging lives in RELION/M with their own job management.
   Build to the export handoff; launch the externals; don't reimplement refinement.
3. **Command box is authoritative.** Two-way sync between widgets and the command string is a
   classic source of bugs — pick the command string as the final authority and regenerate it
   from widgets, letting manual edits win and flagging when they desync.

## 7. Verdict

Realistically buildable? **Yes.** It's a desktop GUI wrapping CLI tools + a sequential job
queue + a static docs panel. No novel engineering. The only real decisions are framework
(recommend Qt), the STA handoff boundary, and which params deserve sliders vs text.
