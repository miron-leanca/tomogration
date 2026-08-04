#!/usr/bin/env python3
# Part of tomogration2 — split out of the single-file app so the pieces can be
# edited (and tested) independently. Import graph is a strict DAG:
#     core  ->  stages  ->  jobs        (project depends only on core)
# tomogration_app.py imports all of them and holds the Qt window.

"""The pipeline itself: STAGES (add a stage = add a dict), the pure command
assembler, and the per-stage path/label maps. No Qt, no filesystem side effects —
so build_command and the validators stay unit-testable."""

import os
import re

from tomogration_core import _pkg_script

_APX_TAG_RE = re.compile(r"([0-9]*\.?[0-9]+)Apx")


def _validate_export(v):
    """Warnings for ts_export_particles. Four traps, worst first.

    EVERY applicable warning is returned, not just the first. These traps combine —
    a run once went out with the coordinate scale 4x wrong AND the star written into
    the previous round's folder, and only the second was reported because it was
    checked first.

    1. COORD SCALE vs the pick filenames. ml_relion4_select_picks names its output
       <stem>_<apx>Apx_<suffix>.star where <apx> is the pixel size the coordinates
       are IN. If coords_angpix disagrees with that tag, every particle is extracted
       at the wrong distance from the origin — silently, with no error.
    2. COORD MODE. --normalized_coords (0-1 fractions) and --coords_angpix (pixels
       at a stated Å/px) are mutually exclusive, and exactly one is needed.
    3. BOX vs DIAMETER. box is the container; diameter is where the particle is
       assumed to end, and everything outside it is the SOLVENT used for background
       normalisation. If diameter crowds the box there's no solvent shell left.
    4. RELION launch root — output_star must sit inside output_processing.
    """
    out = []
    norm = bool(v.get("normalized_coords"))
    capx = str(v.get("coords_angpix", "")).strip()

    # 1. The pick filenames state their own pixel size — believe them.
    tag = _APX_TAG_RE.search(str(v.get("input_pattern", "") or ""))
    if tag and capx and not norm:
        try:
            want, got = float(tag.group(1)), float(capx)
        except ValueError:
            want = got = None
        if want and got and abs(want - got) > 1e-6:
            out.append(
                f"⚠ input_pattern says the coordinates are in {want:g} Å/px "
                f"(*{tag.group(1)}Apx*) but coords_angpix is {got:g}. Every particle "
                f"would be extracted {got / want:.3g}× too far from the origin, with no "
                f"error — set coords_angpix to {want:g}.")

    if norm and capx:
        out.append(
            "⚠ normalized_coords is ON and coords_angpix is set — they are mutually "
            "exclusive. Normalised coords are 0-1 fractions and carry no pixel size: "
            "clear coords_angpix, or turn normalized_coords OFF if coords are pixels.")
    if not norm and not capx:
        out.append(
            "⚠ normalized_coords is OFF and coords_angpix is blank — Warp will not "
            "know the scale of the input coordinates. Set coords_angpix to the Å/px "
            "the pick coords are in (e.g. 6.28), or turn normalized_coords ON if "
            "they are 0-1 fractions.")
    try:
        box_a = float(v.get("box") or 0) * float(str(v.get("output_angpix", "")).strip() or 0)
        diam = float(str(v.get("diameter", "")).strip() or 0)
    except ValueError:
        box_a = diam = 0.0
    if box_a and diam:
        if diam >= box_a:
            out.append(
                f"⚠ diameter ({diam:g} Å) is not smaller than the box "
                f"({box_a:g} Å = {v.get('box')} px × {v.get('output_angpix')} Å). The "
                f"particle would fill the whole box, leaving no solvent for background "
                f"normalisation. Lower diameter, or raise box.")
        elif diam > 0.8 * box_a:
            out.append(
                f"⚠ diameter ({diam:g} Å) is {100 * diam / box_a:.0f}% of the box "
                f"({box_a:g} Å). Little solvent left for background normalisation — "
                f"aim for ≤80% (ideally 50-70%). Set diameter to the TRUE particle "
                f"size, or raise box.")
    op = str(v.get("output_processing", "") or "").rstrip("/")
    os_ = str(v.get("output_star", "") or "")
    if op and os_ and not os_.startswith(op + "/"):
        out.append(
            f"⚠ output_star is written to '{os_.rsplit('/', 1)[0]}' but the subtomos go "
            f"to '{op}'. The star must live INSIDE output_processing or RELION cannot "
            f"resolve the subtomo paths — and this OVERWRITES whatever star is already "
            f"in that other folder.")
    return "\n".join(out)


# ===========================================================================
# Stage definitions  (declarative — extend the pipeline by adding a dict)
#
# Schema (copy ts_reconstruct as the template):
#   group/id/label : center-axis grouping + node identity
#   base           : command head. Empty for env-only/positional commands.
#   params[]:
#     name         : control label + key
#     kind         : "text" | "slider_int" | "check" | "env" | "env_int"
#     flag         : CLI flag (e.g. "--perdevice"). None = positional value.
#                    For env/env_int kinds, flag is the VAR name (VAR=value).
#     default/min/max/step  : widget config
#     help         : inline description + range + effect (shown under control)
#   validate(values)->str   : optional live red warning under the form
#   docs           : what / range / effect / pitfall  (LEFT panel)
#   status(ps)->(truthy,label) : drives the node status dot via file existence
#   aretomo / auto_recover / sync_helper : special-behaviour flags
#
# Param-widget honesty (brief §5): sliders only for bounded numerics (B-factor,
# threshold, binning, perdevice); checkboxes for boolean flags; everything else
# (paths, patterns, grids like 2x2x1, env 0/1 toggles) stays free-text. The
# editable command box at the bottom of each form is the single source of truth.
# ===========================================================================
STAGES = [
    # ---------------- 1. Data prep ----------------
    {
        "group": "1. Data prep", "id": "rename", "label": "Rename .eer/.mdoc",
        "base": f'bash {_pkg_script("ml_batch_rename_eer_mdoc_mrc_warp_auto.sh")}',
        "clean_overrides": True,
        "params": [
            {"name": "source_dir", "kind": "text", "flag": None,
             "default": ".",
             "help": "Folder with the raw Tomo5 .eer + .mdoc (+ .mrc) TOGETHER, "
                     "relative to the project root. '.' = the project root itself "
                     "(use this when the root IS your acquisition folder). Run this "
                     "BEFORE Sort files — rename needs .eer and .mdoc in one folder."},
            {"name": "rootname", "kind": "text", "flag": None,
             "default": "Position", "help": "Prefix for renamed files (e.g. Position, TS)."},
            {"name": "start_number", "kind": "text", "flag": None,
             "default": "0", "help": "Starting counter (0 -> first output is 001)."},
        ],
        "docs": {
            "what": "Renames Tomo5 beam-shift names (Position_1_2) to Warp form "
                    "(Position001). Writes listfile_<root>.txt mapping old->new "
                    "and fixes mdoc dates to yy-mmm-dd (required by Warp).",
            "range": "n/a",
            "effect": "listfile is required later to translate IMOD-order exclusion tables.",
            "pitfall": "ORDER: rename FIRST (on the raw folder, .eer+.mdoc together), "
                       "THEN Sort files to split into frames/ + mdocs/. Renames IN "
                       "PLACE — work on a copy. Identical Tomo5 *_override.mdoc are "
                       "auto-moved to mdocs/bad/ before renaming (they would consume "
                       "position numbers). No 'selected/' folder is needed.",
        },
        "status": lambda ps: ps.status_listfile(),
    },
    {
        "group": "1. Data prep", "id": "imod_warp_key", "label": "IMOD→Warp key",
        "base": f'python {_pkg_script("ml_imodtowarpkey_generator_warp_auto.py")}',
        "params": [
            {"name": "input_mdoc", "kind": "text", "flag": None,
             "default": "mdocs/Position001.mdoc",
             "help": "A COMPLETE reference mdoc (all tilts present)."},
            {"name": "output_key", "kind": "text", "flag": None,
             "default": "new_imod_conv_key.txt",
             "help": "Where to write the IMOD->acquisition-order key."},
        ],
        "docs": {
            "what": "Builds the IMOD->Warp tilt-number key. mdoc ZValue blocks "
                    "are in dose-symmetric acquisition order; 3dmod shows tilts "
                    "in angle order. The key translates between them.",
            "range": "n/a",
            "effect": "Used by remake_mdocs to map IMOD-order exclusions to acq order.",
            "pitfall": "Use a mdoc with EVERY tilt present, or the mapping is wrong.",
        },
        "status": lambda ps: ps.status_conv_key(),
    },
    {
        "group": "1. Data prep", "id": "inspect_select",
        "label": "Inspect tilt stacks",
        "tool": "inspector",
        "docs": {
            "what": "Visually review each tilt series (thumbnail grid + 3dmod) and "
                    "mark bad tilts (IMOD #, ranges OK e.g. 1,4-12,48) or whole "
                    "series for exclusion before remaking the mdocs.",
            "range": "n/a",
            "effect": "Tilt exclusions feed exclusion_list.txt (Remake mdocs applies "
                      "them); excluded whole series move to mdocs/bad/ + frames/bad/. "
                      "You can also hand-edit exclusion_list.txt: 'PositionNNN<tab>1,4-12' "
                      "trims those tilts; a bare 'PositionNNN' line drops the whole series.",
            "pitfall": "Do this BEFORE Remake mdocs. Already-excluded series are "
                       "flagged ✗EXCLUDED. Tip: 'Open ALL in 3dmod + list' opens "
                       "every series in one 3dmod next to a compact type-in list.",
        },
        "status": lambda ps: ps.status_exclusions(),
    },
    {
        "group": "1. Data prep", "id": "remake_mdocs", "label": "Remake mdocs",
        "base": f'bash {_pkg_script("ml_batch_remake_mdocs_warp_auto.sh")}',
        # The script cd's into mdocs/ then reads these by name, so pass them
        # absolute or they're looked up inside mdocs/ and not found.
        "abs_paths": ["mdocs_dir", "exclusion_list", "conv_key"],
        # Expand tilt ranges (1-3) in exclusion_list.txt first — the script
        # can't parse ranges (sed chokes on "1-3p").
        "normalize_exclusions": True,
        "params": [
            {"name": "mdocs_dir", "kind": "text", "flag": None,
             "default": "mdocs", "help": "Dir with <root>NNN.mdoc files."},
            {"name": "exclusion_list", "kind": "text", "flag": None,
             "default": "exclusion_list.txt",
             "help": "Manual tilt exclusions (IMOD order), above the auto header."},
            {"name": "conv_key", "kind": "text", "flag": None,
             "default": "new_imod_conv_key.txt", "help": "IMOD->acq key file."},
            {"name": "rootname", "kind": "text", "flag": None,
             "default": "Position", "help": "File prefix (optional)."},
        ],
        "docs": {
            "what": "Removes excluded tilts from mdocs and renumbers remaining "
                    "ZValue blocks contiguously, then fixes mdoc date format.",
            "range": "n/a",
            "effect": "Quarantining a tilt without fixing the mdoc breaks ts_import.",
            "pitfall": "exclusion_list.txt: manual tilt numbers go ABOVE the "
                       "auto-excluded header; the script stops parsing at it. A bare "
                       "'PositionNNN' line (no tilts) is applied on enqueue by "
                       "quarantining that whole series (mdoc/.eer → bad/). "
                       "For files already moved to frames/bad/, use 'Repair mdocs'.",
        },
        "status": lambda ps: ps.status_mdocs(),
    },

    # ---------------- 2. Gain ----------------
    {
        "group": "2. Gain", "id": "gain_convert", "label": "gain .gain→.mrc",
        "base": "module load eman && e2proc3d.py",
        "params": [
            {"name": "in_gain", "kind": "text", "flag": None,
             "default": "gains/original.gain", "help": "Input .gain reference."},
            {"name": "out_mrc", "kind": "text", "flag": None,
             "default": "gains/original_gain.mrc", "help": "Output .mrc gain."},
        ],
        "docs": {
            "what": "Converts a camera .gain reference to .mrc via EMAN2.",
            "range": "n/a",
            "effect": "Produces the .mrc that the reciprocal step inverts.",
            "pitfall": "Needs `module load eman`. This is the NON-reciprocal gain.",
        },
        "status": lambda ps: ps.status_gain_original(),
    },
    {
        "group": "2. Gain", "id": "gain_reciprocal", "label": "reciprocal gain",
        "base": "module load eman && e2proc2d.py",
        "params": [
            {"name": "in_mrc", "kind": "text", "flag": None,
             "default": "gains/original_gain.mrc", "help": "Input .mrc gain."},
            {"name": "out_reciprocal", "kind": "text", "flag": None,
             "default": "gains/gain_reciprocal.mrc", "help": "Output reciprocal gain."},
            {"name": "reciprocal", "kind": "check", "flag": "--process math.reciprocal",
             "default": True, "help": "Take the per-pixel reciprocal."},
        ],
        "docs": {
            "what": "Computes the reciprocal gain that Linux WarpTools expects.",
            "range": "n/a",
            "effect": "create_settings --gain_path must point at THIS file.",
            "pitfall": "Linux WarpTools wants the reciprocal; using the plain gain "
                       "double-applies the correction.",
        },
        "status": lambda ps: ps.status_gain_reciprocal(),
    },

    # ---------------- 3. Frameseries ----------------
    {
        "group": "3. Frameseries", "id": "create_settings_fs",
        "label": "create_settings (fs)",
        "base": "WarpTools create_settings",
        "params": [
            {"name": "folder_data", "kind": "text", "flag": "--folder_data",
             "default": "frames", "help": "Raw .eer folder."},
            {"name": "folder_processing", "kind": "text", "flag": "--folder_processing",
             "default": "warp_frameseries", "help": "Processing output folder."},
            {"name": "output", "kind": "text", "flag": "--output",
             "default": "warp_frameseries.settings", "help": "Settings file to write."},
            {"name": "extension", "kind": "text", "flag": "--extension",
             "default": "*.eer", "help": "Input file glob."},
            {"name": "angpix", "kind": "text", "flag": "--angpix",
             "default": "1.57", "help": "Pixel size (Å/px). This dataset: 1.57."},
            {"name": "gain_path", "kind": "text", "flag": "--gain_path",
             "default": "gains/gain_reciprocal.mrc",
             "help": "RECIPROCAL gain (Linux WarpTools)."},
            {"name": "exposure", "kind": "text", "flag": "--exposure",
             "default": "3.5", "help": "Dose per TILT (e/Å², not per frame)."},
            {"name": "eer_ngroups", "kind": "text", "flag": "--eer_ngroups",
             "default": "10", "help": "EER frame groups."},
        ],
        "docs": {
            "what": "Writes the Warp frame-series .settings file.",
            "range": "apix 1.57; dose 3.5 e/Å²/tilt; eer_ngroups 10.",
            "effect": "Every downstream fs_* step reads these settings.",
            "pitfall": "Point gain_path at the RECIPROCAL gain; dose is per tilt.",
        },
        "status": lambda ps: ps.status_fs_settings(),
    },
    {
        "group": "3. Frameseries", "id": "fs_motion_and_ctf",
        "label": "fs_motion_and_ctf",
        "base": "WarpTools fs_motion_and_ctf",
        "auto_recover": True,
        "group_scope": "fs",
        "params": [
            {"name": "settings", "kind": "text", "flag": "--settings",
             "default": "warp_frameseries.settings", "help": "fs .settings file."},
            {"name": "m_grid", "kind": "text", "flag": "--m_grid",
             "default": "1x1x3", "help": "Motion grid XxYxT; temporal ≈ frame count."},
            {"name": "c_grid", "kind": "text", "flag": "--c_grid",
             "default": "2x2x1", "help": "CTF grid XxYxT (not slider-able)."},
            {"name": "m_range_min", "kind": "text", "flag": "--m_range_min",
             "default": "500", "help": "Motion fit low-res bound (Å)."},
            {"name": "m_range_max", "kind": "text", "flag": "--m_range_max",
             "default": "10", "help": "Motion fit high-res bound (Å)."},
            {"name": "m_bfac", "kind": "slider_int", "flag": "--m_bfac",
             "default": -500, "min": -1000, "max": 0, "step": 50,
             "help": "Motion B-factor; more negative = stronger low-pass."},
            {"name": "c_range_max", "kind": "text", "flag": "--c_range_max",
             "default": "7", "help": "CTF fit max resolution (Å)."},
            {"name": "c_defocus_max", "kind": "text", "flag": "--c_defocus_max",
             "default": "8", "help": "Max defocus to search (µm)."},
            {"name": "out_averages", "kind": "check", "flag": "--out_averages",
             "default": True, "help": "Write aligned averages. REQUIRED — ts_import "
             "needs them ('no aligned average result' error if off)."},
            {"name": "out_average_halves", "kind": "check", "flag": "--out_average_halves",
             "default": True, "help": "Write odd/even half-averages (for Noise2Noise "
             "denoising)."},
            {"name": "device_list", "kind": "text", "flag": "--device_list", "gpu_sep": " ",
             "default": "0", "help": "GPU id(s), e.g. 0 or '0 1'."},
            {"name": "perdevice", "kind": "slider_int", "flag": "--perdevice",
             "default": 2, "min": 1, "max": 4, "step": 1,
             "help": "Workers per GPU (no deconv here, so 2 is fine)."},
        ],
        "docs": {
            "what": "Per-frame-series motion correction + CTF estimation.",
            "range": "motion grid 1x1x3; CTF grid 2x2x1; m_range 500→10 Å; m_bfac −500.",
            "effect": "Finer grids model more local motion/CTF, at higher cost.",
            "pitfall": "Auto-recovery is ON: a cuFFT crash on a bad .eer is "
                       "quarantined to frames/bad/, its mdoc ZValue block removed, "
                       "logged to exclusion_list.txt, and the run retried. Only "
                       "fs_* recovers — ts_* crashes are GPU/resource, not bad files.",
        },
        "status": lambda ps: ps.status_fs_motion_ctf(),
    },

    # ---------------- 4. Tilt series ----------------
    {
        "group": "4. Tilt series", "id": "create_settings_ts",
        "label": "create_settings (ts)",
        "base": "WarpTools create_settings",
        "params": [
            {"name": "folder_data", "kind": "text", "flag": "--folder_data",
             "default": "tomostar", "help": "tomostar folder."},
            {"name": "folder_processing", "kind": "text", "flag": "--folder_processing",
             "default": "warp_tiltseries", "help": "Processing output folder."},
            {"name": "output", "kind": "text", "flag": "--output",
             "default": "warp_tiltseries.settings", "help": "Settings file to write."},
            {"name": "extension", "kind": "text", "flag": "--extension",
             "default": "*.tomostar", "help": "Input glob."},
            {"name": "angpix", "kind": "text", "flag": "--angpix",
             "default": "1.57", "help": "Pixel size (Å/px)."},
            {"name": "gain_path", "kind": "text", "flag": "--gain_path",
             "default": "gains/gain_reciprocal.mrc", "help": "Reciprocal gain."},
            {"name": "exposure", "kind": "text", "flag": "--exposure",
             "default": "3.5", "help": "Dose per tilt (e/Å²)."},
            {"name": "tomo_dimensions", "kind": "text", "flag": "--tomo_dimensions",
             "default": "4096x4096x3088", "help": "Tomogram dims XxYxZ (unbinned)."},
        ],
        "docs": {
            "what": "Writes the Warp tilt-series .settings file.",
            "range": "tomo_dimensions XxYxZ; apix 1.57.",
            "effect": "Every ts_* step reads these settings.",
            "pitfall": "Z dimension must exceed the lamella thickness.",
        },
        "status": lambda ps: ps.status_ts_settings(),
    },
    {
        "group": "4. Tilt series", "id": "ts_import", "label": "ts_import",
        "base": "WarpTools ts_import",
        "params": [
            {"name": "mdocs", "kind": "text", "flag": "--mdocs",
             "default": "mdocs", "help": "Mdocs folder."},
            {"name": "frameseries", "kind": "text", "flag": "--frameseries",
             "default": "warp_frameseries", "help": "Frameseries processing folder."},
            {"name": "tilt_exposure", "kind": "text", "flag": "--tilt_exposure",
             "default": "3.5", "help": "Dose per tilt (e/Å²)."},
            {"name": "min_intensity", "kind": "text", "flag": "--min_intensity",
             "default": "0", "help": "Min intensity filter."},
            {"name": "dont_invert", "kind": "check", "flag": "--dont_invert",
             "default": True, "help": "Keep tilt polarity as-is (dataset-specific)."},
            {"name": "output", "kind": "text", "flag": "--output",
             "default": "tomostar", "help": "tomostar output folder."},
            {"name": "override_axis", "kind": "text", "flag": "--override_axis",
             "default": "", "help": "Tilt-AXIS rotation (deg) — the IN-PLANE angle of the "
             "tilt axis, NOT the stage tilt range. Blank = use the mdoc value (fine: it's "
             "only the STARTING guess; AreTomo then searches & refines it). Set a number "
             "only if you know the correct axis (e.g. from AreTomo's solved .aln)."},
        ],
        "docs": {
            "what": "Builds .tomostar files by pairing mdocs with frameseries.",
            "range": "n/a",
            "effect": "tomostar is the unit AreTomo and ts_* operate on.",
            "pitfall": "Fails with 'failed to parse specific tilts' when a "
                       "quarantined .eer left a stale mdoc ZValue block — run "
                       "'Repair mdocs' first (brief gotcha §4.3). The 'tilt axis angle … "
                       "Tomo5 mdoc files are known to provide incorrect values' message is "
                       "ADVISORY — printed for every Tomo5 mdoc, not a detected error; the "
                       "axis is the in-plane rotation (e.g. ~-174°), and AreTomo refines it, "
                       "so blank is usually right. Override only with a known-good value.",
        },
        "status": lambda ps: ps.status_tomostar(),
    },
    {
        "group": "4. Tilt series", "id": "ts_stack", "label": "ts_stack",
        "base": "WarpTools ts_stack",
        "group_scope": "ts",
        "params": [
            {"name": "settings", "kind": "text", "flag": "--settings",
             "default": "warp_tiltseries.settings", "help": "ts .settings file."},
            {"name": "angpix", "kind": "text", "flag": "--angpix",
             "default": "", "help": "Output pixel size (Å/px). Blank = native."},
            {"name": "device_list", "kind": "text", "flag": "--device_list", "gpu_sep": " ",
             "default": "0", "help": "GPU id(s)."},
            {"name": "perdevice", "kind": "slider_int", "flag": "--perdevice",
             "default": 2, "min": 1, "max": 4, "step": 1, "help": "Workers per GPU."},
        ],
        "docs": {
            "what": "Builds aligned tilt stacks (.st) per tilt series for AreTomo.",
            "range": "n/a",
            "effect": "Produces warp_tiltseries/tiltstack/<Position>/<Position>.st.",
            "pitfall": "Run ts_import first; a missing mdoc entry stalls a stack.",
        },
        "status": lambda ps: ps.status_ts_stacks(),
    },

    # ---------------- 5. Alignment ----------------
    {
        "group": "5. Alignment", "id": "aretomo", "label": "Align with AreTomo2",
        "base": "bash",
        "aretomo": True,
        "params": [
            {"name": "script", "kind": "text", "flag": None,
             "default": _pkg_script("ml_aretomo2_warp_auto.sh"),
             "help": "AreTomo2 wrapper script (shipped with the app)."},
            {"name": "input_dir", "kind": "text", "flag": None,
             "default": "warp_tiltseries/tiltstack", "help": "Folder of <Pos>/<Pos>.st."},
            {"name": "output_dir", "kind": "text", "flag": None,
             "default": "aretomo_output",
             "help": "Versioned output folder (auto-bumped to -vN; PARAMETERS.txt written here)."},
            {"name": "gpu", "kind": "text", "flag": None,
             "default": "0", "help": "Fallback SINGLE GPU id, used only if the 'GPUs' "
             "list below is left blank. A single id here = sequential on one GPU. To "
             "use several GPUs, fill the GPUs field instead (do NOT put '0 1 2 3' here — "
             "extra tokens here shift the positional args and corrupt angpix)."},
            {"name": "angpix", "kind": "text", "flag": None,
             "default": "1.57", "help": "Input pixel size (Å/px). 1.57 for this data."},
            {"name": "ARETOMO_GPUS", "kind": "env", "flag": "ARETOMO_GPUS", "gpu_sep": " ",
             "default": "0 1 2 3",
             "help": "GPUs to spread tilt series across (space- or comma-separated). Each "
             "series runs on ONE GPU; with N GPUs, N series align at once (~N× faster). "
             "Blank = use the single 'gpu' field above (sequential)."},
            {"name": "ARETOMO_JOBS_PER_GPU", "kind": "env_int", "flag": "ARETOMO_JOBS_PER_GPU",
             "default": 1, "min": 1, "max": 4, "step": 1,
             "help": "Concurrent AreTomo jobs PER GPU. 1 is safe; a 32 GB V100 can usually "
             "fit 2 at bin 8. Total concurrency = (#GPUs) × this."},
            {"name": "ARETOMO_ALIGNZ", "kind": "env", "flag": "ARETOMO_ALIGNZ",
             "default": "670", "help": "Alignment Z (unbinned px) ≈ lamella thickness."},
            {"name": "ARETOMO_VOLZ", "kind": "env", "flag": "ARETOMO_VOLZ",
             "default": "3088", "help": "Output Z height; must exceed lamella thickness. "
             "Set 0 for ALIGNMENT-ONLY (skips the slow tomogram, still writes the .xf you "
             "import) — the fast way to align a whole dataset for ts_import / miss-alignment."},
            {"name": "ARETOMO_OUTBIN", "kind": "env_int", "flag": "ARETOMO_OUTBIN",
             "default": 8, "min": 1, "max": 16, "step": 1,
             "help": "Output binning. 8 → 12.56 Å/px at 1.57 input."},
            {"name": "ARETOMO_DARKTOL", "kind": "env", "flag": "ARETOMO_DARKTOL",
             "default": "0.000001", "help": "Dark-frame tol; ~0 disables (pre-curated tilts)."},
            {"name": "ARETOMO_TILTCOR", "kind": "env", "flag": "ARETOMO_TILTCOR",
             "default": "0", "help": "Tilt-offset correction 0/1. Usually 0 for lamellae."},
            {"name": "ARETOMO_FLIPVOLZ", "kind": "env", "flag": "ARETOMO_FLIPVOLZ",
             "default": "1", "help": "Flip handedness for Warp 0/1. Usually 1."},
            {"name": "ARETOMO_WBP", "kind": "env", "flag": "ARETOMO_WBP",
             "default": "1", "help": "Weighted back projection 0/1."},
            {"name": "ARETOMO_TILTAXIS", "kind": "env", "flag": "ARETOMO_TILTAXIS",
             "default": "", "help": "Tilt-axis (deg). Blank = AreTomo searches."},
            {"name": "ARETOMO_PATCH", "kind": "env", "flag": "ARETOMO_PATCH",
             "default": "", "help": "Patch align e.g. '4 4'. Blank = skip patch tracking."},
            {"name": "ARETOMO_ALIGN", "kind": "env", "flag": "ARETOMO_ALIGN",
             "default": "1", "help": "1 = align+recon, 0 = reconstruct only."},
            {"name": "ARETOMO_BIN", "kind": "env", "flag": "ARETOMO_BIN",
             "default": "/ceph/groups/structbio/Programs/AreTomo2/AreTomo2",
             "help": "AreTomo2 executable path."},
            {"name": "ARETOMO_CUDA_LIB", "kind": "env", "flag": "ARETOMO_CUDA_LIB",
             "default": "",
             "help": "Optional. Dir holding libcufft.so.11 + the CUDA-12 runtime. Normally "
             "BLANK — the wrapper uses the cuda module's libs (via a stub-free symlink "
             "farm so the real driver is found). Set it only to override with your own "
             "CUDA-12 runtime, e.g. /ceph/users/<you>/.conda/envs/cuda12rt/lib."},
        ],
        "docs": {
            "what": "Marker-free tilt-series alignment (+optional recon). Writes "
                    ".xf/.tlt per series under <output>/Imod/.",
            "range": "ALIGNZ ≈ lamella thickness; OUTBIN 8 → 12.56 Å/px; patch 4×4–6×6.",
            "effect": "Patch tracking improves local alignment but is slower. GPUs runs "
                      "tilt series in parallel (one per GPU); '0 1 2 3' is ~4× faster than one.",
            "pitfall": "Each run writes a NEW versioned folder (aretomo_output, "
                       "-v2, …) with a PARAMETERS.txt audit. IMOD can't read Warp "
                       "float16 MRC — export WARP_FORCE_MRC_FLOAT32=1 before 3dmod "
                       "(brief gotcha §4.2).",
        },
        "status": lambda ps: ps.status_aretomo_xf(),
    },
    {
        # miss-alignment TRAIN: train a model on THIS set (which must already hold a
        # coarse AreTomo alignment) and refine it. Enforced order: AreTomo ->
        # ts_import_alignments -> select -> THIS -> ts_ctf. On a fresh run the GUI
        # auto-prepends the AreTomo import + select; on raw stacks it's blocked outright.
        "group": "5. Alignment", "id": "miss_align",
        "label": "miss-alignment (train)",
        "base": "bash",
        "requires_coarse_alignment": True,
        "fixed_env": {"MA_MODE": "train"},
        "params": [
            {"name": "script", "kind": "text", "flag": None,
             "default": _pkg_script("ml_missalignment_warp_auto.sh"),
             "help": "miss-alignment wrapper script (shipped with the app)."},
            {"name": "config", "kind": "text", "flag": None,
             "default": "missalignment_config.yaml",
             "help": "TRAIN YAML config (relative to root). If missing, the wrapper seeds "
             "a training template and stops so you can review it, then re-run."},
            {"name": "input_dir", "kind": "text", "flag": None,
             "default": "warp_tiltseries",
             "help": "Warp tilt-series dir with <series>.xml + tiltstack/<series>/"
             "<series>.st (run ts_import + ts_stack first, same prereqs as AreTomo)."},
            {"name": "MA_TRAINING_DEVICES", "kind": "env", "flag": "MA_TRAINING_DEVICES",
             "default": "0", "help": "--training-devices. KEEP THIS A SINGLE GPU (e.g. "
             "'0'). >1 makes torch spawn one trainer per GPU and they race to wipe the "
             "shared pool dir → FileNotFoundError on a partition_*.pickle. Scale speed "
             "with RECON devices + dataloaders instead."},
            {"name": "MA_RECON_DEVICES", "kind": "env", "flag": "MA_RECON_DEVICES", "gpu_sep": ",",
             "default": "0,0,0", "help": "--reconstruction-devices: this is where you add "
             "GPUs for speed (recon feeds the pool and is the bottleneck). e.g. '0,1,2,3' "
             "or repeat an id to stack workers on it ('0,0,0')."},
            {"name": "MA_DATALOADERS", "kind": "env_int", "flag": "MA_DATALOADERS",
             "default": 5, "min": 1, "max": 16, "step": 1,
             "help": "--dataloaders-per-trainer. The recon pool is split into "
             "(training_devices × this) partitions, each needing ≥ 2×batch_size; "
             "if it errors, raise Pool size or lower this."},
            {"name": "MA_POOL_SIZE", "kind": "env_int", "flag": "MA_POOL_SIZE",
             "default": 2000, "min": 500, "max": 8000, "step": 100,
             "help": "--pool-size: subtomogram reconstructions cached in the temp "
             "pool. Must be ≥ 2×batch_size×training_devices×dataloaders (2000 keeps "
             "4 GPU × 5 loaders × batch 32 valid). Type the exact number."},
            {"name": "MA_START_ITER", "kind": "env_int", "flag": "MA_START_ITER",
             "default": 0, "min": 0, "max": 20, "step": 1,
             "help": "--start-at-iteration (resume from the HIGHEST existing iterN)."},
            {"name": "MA_PREPARE_STACKS", "kind": "env", "flag": "MA_PREPARE_STACKS",
             "default": "10.0", "help": "--prepare-stacks pixel size (Å/px) for the "
             "reconstruction patches. Blank = skip stack preparation."},
            {"name": "MA_CONDA_ENV", "kind": "env", "flag": "MA_CONDA_ENV",
             "default": "miss-alignment",
             "help": "conda env that has miss-alignment installed (its own CUDA 12.9 / "
             "torch stack — NOT the warp env)."},
        ],
        "docs": {
            "what": "Deep-learning REFINEMENT of an EXISTING alignment by TRAINING a model "
                    "on this set (warpem/miss-alignment). Does NOT align raw stacks — the "
                    "docs state 'miss-alignment starts from an initially coarse aligned "
                    "dataset', so coarse-align FIRST (AreTomo → ts_import_alignments).",
            "range": "training/recon devices, prepare-stacks Å/px, start-iteration.",
            "effect": "Trains a 3D CNN to score reconstruction quality, then gradient-"
                      "optimises the shifts against it, writing the refined alignment back "
                      "into the .xml. The trained iterN/model.ckpt can then be REUSED via "
                      "the 'miss-alignment (infer)' step on a larger set.",
            "pitfall": "MUST have a coarse prior alignment first — on RAW stacks it yields "
                       "a featureless tomogram (the GUI blocks this). Single TRAINING GPU "
                       "only. First run seeds a config and STOPS for review. Re-train if the "
                       "prior alignment changed.",
        },
        "status": None,
    },
    {
        # miss-alignment INFER: REUSE a finished model to align a new/larger set WITHOUT
        # training. No coarse-align auto-chain (you coarse-align the big set yourself and
        # deselect the unaligned); needs MA_MODEL_RUN_DIR = the training run's iterN dir.
        "group": "5. Alignment", "id": "miss_align_infer",
        "label": "miss-alignment (infer — reuse model)",
        "base": "bash",
        "fixed_env": {"MA_MODE": "infer"},
        "params": [
            {"name": "script", "kind": "text", "flag": None,
             "default": _pkg_script("ml_missalignment_warp_auto.sh"),
             "help": "miss-alignment wrapper script (shipped with the app)."},
            {"name": "config", "kind": "text", "flag": None,
             "default": "missalignment_infer_config.yaml",
             "help": "INFER YAML config (relative to root). If missing, the wrapper seeds "
             "an inference template (data_directory + model_run_directory) and stops for "
             "review. iteration_settings MUST match the training run."},
            {"name": "input_dir", "kind": "text", "flag": None,
             "default": "warp_tiltseries",
             "help": "This dataset's warp_tiltseries — already coarse-aligned + imported, "
             "with the unaligned series deselected."},
            {"name": "MA_MODEL_RUN_DIR", "kind": "env", "flag": "MA_MODEL_RUN_DIR",
             "default": "", "help": "REQUIRED: the finished TRAINING run dir holding "
             "iter1/model.ckpt … iterN/model.ckpt (e.g. <selected>/warp_tiltseries)."},
            {"name": "MA_INFER_DEVICES", "kind": "env", "flag": "MA_INFER_DEVICES", "gpu_sep": ",",
             "default": "0,1,2,3", "help": "GPUs for alignment (CUDA_VISIBLE_DEVICES). "
             "Inference has no training race, so use all the idle cards (check util%)."},
            {"name": "MA_START_ITER", "kind": "env_int", "flag": "MA_START_ITER",
             "default": 0, "min": 0, "max": 20, "step": 1,
             "help": "--start-at-iteration (resume inference from iteration N)."},
            {"name": "MA_PREPARE_STACKS", "kind": "env", "flag": "MA_PREPARE_STACKS",
             "default": "10.0", "help": "--prepare-stacks pixel size (Å/px). MUST equal the "
             "resolution you TRAINED at (e.g. 12.56) — the model only works at its scale."},
            {"name": "MA_CONDA_ENV", "kind": "env", "flag": "MA_CONDA_ENV",
             "default": "miss-alignment",
             "help": "conda env with miss-alignment installed."},
        ],
        "docs": {
            "what": "Reuse a model trained by 'miss-alignment (train)' to align THIS "
                    "(usually larger) dataset with NO retraining — runs 'miss-alignment "
                    "infer', loading iterN/model.ckpt for each iteration.",
            "range": "model_run_directory, infer devices, prepare-stacks Å/px.",
            "effect": "Applies the trained models to refine the coarse alignment already "
                      "in this set's .xml. Much faster than training. Alignment uses all "
                      "visible GPUs (no training-worker race).",
            "pitfall": "This set must ALREADY be coarse-aligned (AreTomo → import → "
                       "deselect unaligned) — infer refines, it does not align from scratch. "
                       "prepare-stacks and the config's iteration_settings MUST match the "
                       "training run, and length ≤ number of iterN/model.ckpt.",
        },
        "status": None,
    },
    {
        "group": "5. Alignment", "id": "ts_import_alignments",
        "label": "ts_import_alignments",
        "base": "WarpTools ts_import_alignments",
        "params": [
            {"name": "settings", "kind": "text", "flag": "--settings",
             "default": "warp_tiltseries.settings", "help": "ts .settings file."},
            {"name": "alignments", "kind": "text", "flag": "--alignments",
             "default": "aretomo_output/Imod/", "help": "AreTomo Imod/ folder."},
            {"name": "alignment_angpix", "kind": "text", "flag": "--alignment_angpix",
             "default": "1.57", "help": "Pixel size AreTomo aligned at (1.57)."},
        ],
        "docs": {
            "what": "Imports AreTomo .xf/.tlt alignments back into Warp.",
            "range": "n/a",
            "effect": "ts_ctf / ts_reconstruct use these alignments.",
            "pitfall": "alignment_angpix must match the AreTomo INPUT pixel size, "
                       "not the binned output (1.57 here).",
        },
        "status": lambda ps: ps.status_alignments_imported(),
    },
    {
        "group": "5. Alignment", "id": "sync_selection",
        "label": "sync selection ↔ alignments",
        "base": "WarpTools change_selection",
        "sync_helper": True,
        "params": [
            {"name": "settings", "kind": "text", "flag": "--settings",
             "default": "warp_tiltseries.settings", "help": "ts .settings file."},
            {"name": "mode", "kind": "choice", "flag": None,
             "choices": [
                 ("Deselect", "--deselect"),
                 ("Select (re-enable)", "--select"),
                 ("Null (reset to unset)", "--null"),
                 ("Invert", "--invert"),
             ],
             "default": "--deselect",
             "help": "WarpTools accepts EXACTLY ONE. Deselect = drop the listed series; "
                     "Select = re-enable them. Without --input_data it applies to ALL."},
            {"name": "input_data", "kind": "text", "flag": "--input_data",
             "default": "", "help": "One tomostar to (de)select. Use the button to "
             "fill a chained command for ALL unaligned tomostars. Blank = all series."},
        ],
        "docs": {
            "what": "Deselects tilt series with no AreTomo alignment so "
                    "ts_reconstruct won't crash trying to reconstruct them.",
            "range": "n/a",
            "effect": "Reconstruct only operates on selected, aligned series.",
            "pitfall": "Reversible — re-run with mode 'Select' to re-enable. WarpTools "
                       "errors ('Choose exactly 1 of the options') if no mode is given. "
                       "The dot is green only when every tomostar is aligned.",
        },
        "status": lambda ps: ps.status_selection_sync(),
    },

    # ---------------- 6. CTF ----------------
    {
        "group": "6. CTF", "id": "ts_defocus_hand", "label": "ts_defocus_hand",
        "base": "WarpTools ts_defocus_hand",
        "params": [
            {"name": "settings", "kind": "text", "flag": "--settings",
             "default": "warp_tiltseries.settings", "help": "ts .settings file."},
            {"name": "mode", "kind": "choice", "flag": None,
             "choices": [
                 ("Check handedness only (no change)", "--check"),
                 ("Set: flip", "--set_flip"),
                 ("Set: no-flip", "--set_noflip"),
                 ("Set: auto (apply the checked result)", "--set_auto"),
                 ("Set: switch (toggle current)", "--set_switch"),
             ],
             "default": "--check",
             "help": "WarpTools accepts EXACTLY ONE mode. Run 'Check' first; if it "
                     "reports a negative average correlation, switch to 'Set: flip' "
                     "(or 'Set: auto') and Run again."},
        ],
        "validate": lambda v: (
            "⚠ Run 'Check handedness only' first; pick a Set option only after it "
            "reports a negative correlation."
            if v.get("mode") not in (None, "--check") else ""),
        "docs": {
            "what": "Checks (and optionally flips) defocus handedness. Exactly one "
                    "mode runs per invocation.",
            "range": "n/a",
            "effect": "Wrong handedness inverts the CTF and ruins refinement.",
            "pitfall": "Check first; only set flip/auto on a confirmed negative "
                       "correlation. Passing --check together with a --set_ option "
                       "errors ('Choose exactly 1 of the options').",
        },
        "status": None,  # no distinct file output
    },
    {
        "group": "6. CTF", "id": "ts_ctf", "label": "ts_ctf",
        "base": "WarpTools ts_ctf",
        "group_scope": "ts",
        "params": [
            {"name": "settings", "kind": "text", "flag": "--settings",
             "default": "warp_tiltseries.settings", "help": "ts .settings file."},
            {"name": "range_high", "kind": "text", "flag": "--range_high",
             "default": "7", "help": "CTF fit max resolution (Å)."},
            {"name": "defocus_max", "kind": "text", "flag": "--defocus_max",
             "default": "8", "help": "Max defocus to search (µm)."},
            {"name": "device_list", "kind": "text", "flag": "--device_list", "gpu_sep": " ",
             "default": "0", "help": "GPU id(s)."},
            {"name": "perdevice", "kind": "slider_int", "flag": "--perdevice",
             "default": 2, "min": 1, "max": 4, "step": 1, "help": "Workers per GPU."},
        ],
        "docs": {
            "what": "Per-tilt CTF refinement across each series.",
            "range": "range_high ~7 Å; defocus_max ~8 µm.",
            "effect": "Better per-tilt CTF improves reconstruction + averaging.",
            "pitfall": "Run ts_defocus_hand first to fix handedness.",
        },
        "status": lambda ps: ps.status_ts_ctf(),
    },

    # ---------------- 7. Reconstruct ----------------
    {
        # ---------- REFERENCE STAGE / GPU GUARD ----------
        "group": "7. Reconstruct", "id": "ts_reconstruct", "label": "ts_reconstruct",
        "base": "WarpTools ts_reconstruct",
        "group_scope": "ts",
        # Warp writes float16 MRC by default; force float32 so IMOD/3dmod/Dynamo can
        # read the tomograms (and ts_reconstruct itself needs it set in the env).
        "env_export": {"WARP_FORCE_MRC_FLOAT32": "1"},
        "params": [
            {"name": "settings", "kind": "text", "flag": "--settings",
             "default": "warp_tiltseries.settings", "help": "ts .settings file."},
            {"name": "angpix", "kind": "text", "flag": "--angpix",
             "default": "10", "help": "OUTPUT tomogram pixel size (Å/px). 10 = a normal "
             "viewable/pickable tomogram. DO NOT use native (1.57) for full tomograms: "
             "the volume scales as (10/1.57)³ ≈ 260×, so each is tens of GB and ~40 min. "
             "Particles get reconstructed at fine res later by ts_export_particles."},
            {"name": "device_list", "kind": "text", "flag": "--device_list", "gpu_sep": " ",
             "default": "0", "help": "GPU id(s), e.g. 0 or '0 1'. Pick GPUs whose "
             "nvidia-smi GPU-Util is ~0% — low memory-used alone does NOT mean free."},
            {"name": "perdevice", "kind": "slider_int", "flag": "--perdevice",
             "default": 1, "min": 1, "max": 4, "step": 1,
             "help": "Workers per GPU. KEEP AT 1 when --deconv is on (V100 cuFFT crash)."},
            {"name": "deconv", "kind": "check", "flag": "--deconv",
             "default": False, "help": "Deconvolve for visual contrast (not for STA)."},
            {"name": "dont_invert", "kind": "check", "flag": "--dont_invert",
             "default": True, "help": "Skip contrast inversion (dataset-specific)."},
            {"name": "dont_overwrite", "kind": "check", "flag": "--dont_overwrite",
             "default": False,
             "help": "Skip tilt series that already have a tomogram instead of "
             "rebuilding them. OFF means the tomograms already in "
             "<processing>/reconstruction/ are REPLACED — and once M has changed the "
             "alignments they were made from, they cannot be rebuilt. Turn ON to "
             "resume an interrupted run, or to protect an existing set."},
        ],
        # Tomograms land in <processing folder>/reconstruction/, which is named
        # inside the .settings file, NOT in jobs/<id>. Without this the details pane
        # reported an empty job folder and the files looked lost.
        "settings_param": "settings",
        "output_subdirs": ["reconstruction"],
        # NOT a blanket "this overwrites" warning: it would fire on every build,
        # including the first run into an empty folder, and a warning that is always
        # on is a warning nobody reads. The real check looks at the directory (see
        # _confirm_overwrite) and only speaks when there is something to lose.
        "validate": lambda v: (
            "⚠ perdevice > 1 with --deconv crashes on V100 (SIGABRT exit 134). "
            "Set perdevice 1 — or, if EML45 is NOT V100, re-test before overriding."
            if v.get("perdevice", 1) > 1 and v.get("deconv") else ""),
        "docs": {
            "what": "Back-projects aligned, CTF-corrected tilts into 3D tomograms.",
            "range": "angpix ~10 for viewable tomograms; perdevice 1-2; deconv off for averaging.",
            "effect": "deconv boosts low-freq contrast; dont_invert flips densities.",
            "pitfall": "angpix native (1.57 / blank) makes tens-of-GB tomograms (~260× a "
                       "10 Å one) — it looks 'stuck at 0/15' but is just grinding; use ~10. "
                       "perdevice 2 + deconv = SIGABRT on V100 (cuFFT collision). Float16 MRC "
                       "output needs WARP_FORCE_MRC_FLOAT32=1 to open in IMOD/3dmod.",
        },
        "status": lambda ps: ps.status_warp_tomograms(),
    },

    # ---------------- 8. Pick ----------------
    {
        "group": "8. Pick", "id": "ts_template_match", "label": "ts_template_match",
        "base": "WarpTools ts_template_match",
        "group_scope": "ts",
        "params": [
            {"name": "settings", "kind": "text", "flag": "--settings",
             "default": "warp_tiltseries.settings", "help": "ts .settings file."},
            {"name": "tomo_angpix", "kind": "text", "flag": "--tomo_angpix",
             "default": "10", "help": "Matching pixel size (Å). MUST equal a ts_reconstruct "
             "--angpix you have ALREADY run — matching reuses that full tomogram "
             "(warp_tiltseries/reconstruction/<pos>_<angpix>Apx.mrc). Mismatch → "
             "'A reconstruction at the desired resolution was not found'. 8-12 typical."},
            {"name": "template_emdb", "kind": "text", "flag": "--template_emdb",
             "default": "", "help": "EMDB code to fetch + use as the template, e.g. 70905. "
             "Set EITHER this OR template_path (not both)."},
            {"name": "template_path", "kind": "text", "flag": "--template_path",
             "default": "", "help": "Path to a local template .mrc. Set EITHER this OR "
             "template_emdb (not both)."},
            {"name": "override_suffix", "kind": "text", "flag": "--override_suffix",
             "default": "", "help": "Overrides the STAR suffix (normally derived from the "
             "template name) so this pick set gets its OWN name: files become "
             "warp_tiltseries/matching/<pos>_<tomo_angpix>Apx<suffix>.star. INCLUDE A LEADING "
             "UNDERSCORE if you want one (e.g. '_run2'; without it the suffix abuts 'Apx'). "
             "Use a different suffix per run to keep parallel pick sets side by side — "
             "threshold_picks / export then pick a set via --in_suffix (this is how you fork "
             "picking). Blank = default template-derived name."},
            {"name": "subdivisions", "kind": "slider_int", "flag": "--subdivisions",
             "default": 3, "min": 1, "max": 6, "step": 1,
             "help": "Angular subdivisions of the search (finer = more orientations, slower)."},
            {"name": "template_diameter", "kind": "text", "flag": "--template_diameter",
             "default": "", "help": "Particle diameter (Å)."},
            {"name": "symmetry", "kind": "text", "flag": "--symmetry",
             "default": "C1", "help": "Point group, e.g. O, D2, C1."},
            {"name": "whiten", "kind": "check", "flag": "--whiten",
             "default": True, "help": "Spectral whitening; helps with good alignments."},
            {"name": "optimize_poses", "kind": "check", "flag": "--optimize_poses",
             "default": False, "help": "Locally refine each hit's orientation/position after "
             "the coarse search (better picks, a bit slower). ON in the reference workflow."},
            {"name": "check_hand", "kind": "slider_int", "flag": "--check_hand",
             "default": 2, "min": 0, "max": 2, "step": 1,
             "help": "2 = verify geometry/handedness during matching."},
            {"name": "npeaks", "kind": "slider_int", "flag": "--npeaks",
             "default": 2000, "min": 100, "max": 50000, "step": 500,
             "help": "Max peaks SAVED per tilt series. This is a HARD CAP — if a series "
             "actually has more particles you'll silently keep only the top-scoring 2000. "
             "For crowded samples raise it (you can tell you're capped when every series "
             "returns exactly this many). Costs disk, not match time."},
            {"name": "peak_distance", "kind": "text", "flag": "--peak_distance",
             "default": "", "help": "Minimum spacing between peaks in Å. Blank = the template "
             "diameter. Lower it (e.g. 30) for tightly-packed particles so neighbours aren't "
             "suppressed; raise it to avoid double-picking one particle."},
            {"name": "max_missing_tilts", "kind": "slider_int", "flag": "--max_missing_tilts",
             "default": 2, "min": -1, "max": 20, "step": 1,
             "help": "Drop positions not covered by at least this many tilts. -1 disables "
             "culling (keep everything, e.g. thin/edge regions); default 2."},
            {"name": "subvolume_size", "kind": "slider_int", "flag": "--subvolume_size",
             "default": 192, "min": 48, "max": 512, "step": 16,
             "help": "Local matching TILE size, in TOMOGRAM pixels (at tomo_angpix, NOT raw "
             "pixels). Just needs to comfortably exceed the template — 192 does so hugely. "
             "It is NOT the particle box (that's export --box). Reduce only if you hit GPU "
             "OOM or want speed; keep it even (FFT-friendly)."},
            {"name": "device_list", "kind": "text", "flag": "--device_list", "gpu_sep": " ",
             "default": "", "help": "GPU id(s), space-separated e.g. '2 3'. BLANK = ALL "
             "GPUs (Warp's default — it WILL grab 0/1). Set this to the idle cards (check "
             "nvidia-smi util%) to leave others' jobs alone. Or prefix CUDA_VISIBLE_DEVICES=2,3."},
            {"name": "perdevice", "kind": "slider_int", "flag": "--perdevice",
             "default": 1, "min": 1, "max": 4, "step": 1,
             "help": "Worker processes per GPU (raise only on big-memory cards)."},
        ],
        "validate": lambda v: (
            "⚠ Set EXACTLY ONE of template_emdb / template_path — matching needs a template."
            if bool(str(v.get("template_emdb", "")).strip())
            == bool(str(v.get("template_path", "")).strip())
            else "⚠ check_hand does NOT work with override_suffix: Warp reads the handedness "
            "test back under the DEFAULT template name and dies ('Could not find "
            "…_emd_XXXXX.star', all items fail). Set check_hand 0 for suffixed/forked runs — "
            "determine handedness ONCE without a suffix, then reuse check_hand 0."
            if str(v.get("override_suffix", "")).strip() and int(v.get("check_hand") or 0) > 0
            else ""),
        "docs": {
            "what": "CTF-aware template matching to locate particles "
                    "(apoferritin example values — adapt per target).",
            "range": "tomo_angpix 8-12; subdivisions 3-4; check_hand 2 (0 with a suffix).",
            "effect": "Lower tomo_angpix + higher subdivisions = finer, MUCH slower. With "
                      "--optimize_poses, coarser subdivisions (3-4) suffice — local refinement "
                      "recovers the precision.",
            "pitfall": "tomo_angpix MUST match a ts_reconstruct --angpix you already ran "
                       "(matching reuses that full tomogram) — else 'A reconstruction at the "
                       "desired resolution was not found' and every series fails. check_hand>0 "
                       "is INCOMPATIBLE with override_suffix (handedness readback uses the "
                       "default template name → 'Could not find …_emd_XXXXX.star'): set "
                       "check_hand 0 for suffixed runs. Defaults to ALL GPUs — set --device_list "
                       "(e.g. '2 3') to avoid disturbing others on 0/1. Scores are "
                       "background-normalised, so a threshold is comparable across tomograms.",
        },
        "status": lambda ps: ps.status_template_matches(),
    },
    {
        "group": "8. Pick", "id": "threshold_picks", "label": "threshold_picks",
        "base": "WarpTools threshold_picks",
        "params": [
            {"name": "settings", "kind": "text", "flag": "--settings",
             "default": "warp_tiltseries.settings", "help": "ts .settings file."},
            {"name": "in_suffix", "kind": "text", "flag": "--in_suffix",
             "default": "", "help": "Suffix of the template-match star files to threshold."},
            {"name": "out_suffix", "kind": "text", "flag": "--out_suffix",
             "default": "clean", "help": "Suffix for thresholded output star files."},
            {"name": "minimum", "kind": "slider_int", "flag": "--minimum",
             "default": 3, "min": 0, "max": 10, "step": 1,
             "help": "Min normalised score (≈ σ above background). 3 is a good start."},
        ],
        "docs": {
            "what": "Keeps picks above a normalised score threshold; writes "
                    "*<out_suffix>.star.",
            "range": "minimum ~3 (σ above background).",
            "effect": "Higher minimum = fewer, cleaner picks.",
            "pitfall": "Scores compare across tomograms thanks to bg normalisation, "
                       "so one minimum works project-wide.",
        },
        "status": lambda ps: ps.status_thresholded(),
    },

    # ---------------- 9. Export ----------------
    {
        "group": "9. Export", "id": "ts_export_particles",
        "label": "ts_export_particles",
        "base": "WarpTools ts_export_particles",
        "group_scope": "ts",
        "output_params": ["output_processing"],
        "params": [
            {"name": "settings", "kind": "text", "flag": "--settings",
             "default": "warp_tiltseries.settings", "help": "ts .settings file."},
            {"name": "input_directory", "kind": "text", "flag": "--input_directory",
             "default": "warp_tiltseries/matching",
             "help": "Where the thresholded pick stars live."},
            {"name": "input_pattern", "kind": "text", "flag": "--input_pattern",
             "default": "*clean.star", "help": "Glob for thresholded pick star files."},
            {"name": "output_star", "kind": "text", "flag": "--output_star",
             "default": "relion4/{jobid}/matching.star",
             "help": "Output star path. Put it INSIDE the RELION project dir "
             "(output_processing) — RELION must later be launched from that dir. "
             "{jobid} resolves to this job's id so runs never collide."},
            {"name": "output_processing", "kind": "text", "flag": "--output_processing",
             "default": "relion4/{jobid}",
             "help": "RELION project/export dir. The subtomo image paths in the star are "
             "written RELATIVE to this, so you MUST launch RELION from here (the recurring "
             "'file does not exist' bug is launching from the wrong dir). Default "
             "relion4/{jobid} gives each export its own dir; Build-downstream from a "
             "pick-set card instead names it after the pick set (relion4/<tag>)."},
            {"name": "output_angpix", "kind": "text", "flag": "--output_angpix",
             "default": "4", "help": "Export pixel size (Å). Choose so Nyquist sits just "
             "below feature resolution."},
            {"name": "box", "kind": "slider_int", "flag": "--box",
             "default": 64, "min": 32, "max": 256, "step": 8,
             "help": "Box size (px) — the CONTAINER: how much field of view is cut out. "
             "Its physical size is box × output_angpix Å. Aim for ~1.5-2× the particle "
             "diameter: enough padding for the particle plus alignment shifts, no more "
             "(cost scales with box³ in every downstream RELION job). If you re-extract "
             "at half the pixel size WITHOUT recentring, double the box to keep the same "
             "field; recentred picks (converter MODE C) need less padding."},
            {"name": "diameter", "kind": "text", "flag": "--diameter",
             "default": "",
             "help": "TRUE particle diameter (Å) — NOT the box. This is where the particle "
             "is assumed to end: everything outside this sphere is treated as SOLVENT and "
             "used to estimate the background for normalising each subtomogram. So it must "
             "be comfortably smaller than the box (≤80%, ideally 50-70%) or there is no "
             "solvent shell left to measure. Overstating it corrupts the normalisation."},
            {"name": "relion_format", "kind": "choice", "flag": None,
             "choices": [
                 ("3D subtomograms (RELION 4)", "--3d"),
                 ("2D image series (RELION 5)", "--2d"),
             ],
             "default": "--3d",
             "help": "RELION 4 uses 3D subtomos (--3d). RELION 5 --tomo uses the 2D "
             "image series (--2d). WarpTools needs exactly one of these — pick to match "
             "the RELION you'll hand off to."},
            {"name": "normalized_coords", "kind": "check", "flag": "--normalized_coords",
             "default": True,
             "help": "ON = the input pick star's coords are 0-1 FRACTIONS of the tomogram "
             "(what Warp template matching and the crYOLO converter write). OFF = they are "
             "pixels, and you MUST set coords_angpix to say at which pixel size. Getting "
             "this wrong silently extracts at the wrong places → a noise volume. Check your "
             "pick star: values 0-1 = ON; values in the hundreds = OFF + coords_angpix."},
            {"name": "coords_angpix", "kind": "text", "flag": "--coords_angpix",
             "default": "",
             "help": "Pixel size (Å/px) the INPUT coordinates are expressed in. Use with "
             "normalized_coords OFF — e.g. 6.28 for coords taken from a bin4 matching.star. "
             "Leave BLANK when normalized_coords is ON. (Mutually exclusive with it.)"},
            {"name": "relative_output_paths", "kind": "check",
             "flag": "--relative_output_paths", "default": True,
             "help": "Write relative paths into the star (portable projects). Keep ON — "
             "the RELION-launch-from-output_processing rule depends on it."},
            {"name": "device_list", "kind": "text", "flag": "--device_list", "gpu_sep": " ",
             "default": "", "help": "GPU id(s), space-separated (e.g. '0 1 2 3'). Blank = "
             "all GPUs in the system. Pick GPUs whose nvidia-smi GPU-Util is ~0%."},
            {"name": "perdevice", "kind": "slider_int", "flag": "--perdevice",
             "default": 1, "min": 1, "max": 8, "step": 1,
             "help": "Processes per GPU. 1 is safe; raise only if GPU memory allows."},
        ],
        "validate": _validate_export,
        "docs": {
            "what": "Extracts CTF-corrected particles into a RELION project dir — a "
                    "particles star (+ optimisation_set.star for RELION 5).",
            "range": "box 64-128; output_angpix 3-5 for most targets.",
            "effect": "3D subtomos (--3d) = RELION 4; --2d = RELION 5 --tomo. Paths are "
                      "relative to output_processing — that dir IS the RELION project root.",
            "pitfall": "LAUNCH RELION FROM output_processing, or every subtomo path is "
                       "wrong ('file does not exist'). RELION 4 does NOT auto-resize the "
                       "reference — pre-scale it (see 'RELION 4: Class3D'). Then continue "
                       "in the 'RELION 4' steps below, or hand off to RELION's own GUI.",
        },
        "status": lambda ps: ps.status_exported(),
    },

    # ---------------- 10. RELION 4 handoff (subtomo averaging) ----------------
    {
        # Bridge between ts_export_particles and the Class3D handoff: convert the Warp
        # export star to RELION 4 format (relion_convert_star), rewriting the relative
        # 'subtomo/' particle paths to absolute so RELION finds every subtomogram, and
        # optionally build a de-novo initial reference from a random particle subset
        # (relion_reconstruct). Driven by ml_relion4_convert_star_warp_auto.sh.
        "group": "10. RELION 4", "id": "relion4_convert",
        "label": "RELION 4: convert STAR + init ref",
        "base": "bash",
        "output_params": ["project_dir"],
        "params": [
            {"name": "script", "kind": "text", "flag": None,
             "default": _pkg_script("ml_relion4_convert_star_warp_auto.sh"),
             "help": "STAR conversion + initial-reference wrapper (shipped with the app)."},
            {"name": "project_dir", "kind": "text", "flag": None,
             "default": "relion4/warp",
             "help": "The RELION project dir = ts_export_particles' output_processing. "
             "The script runs relion FROM here (the launch-root invariant)."},
            {"name": "starfile", "kind": "text", "flag": None,
             "default": "matching.star",
             "help": "Warp export star, RELATIVE to project_dir (e.g. matching.star)."},
            {"name": "RELION_MODULE", "kind": "env", "flag": "RELION_MODULE",
             "default": "relion/4.0.1", "help": "module load name for RELION 4 on your cluster."},
            {"name": "PARTICLEDIR", "kind": "env", "flag": "PARTICLEDIR",
             "default": "", "help": "Absolute path to the exported subtomo/ dir (the "
             "'subtomo/' prefix in the star is rewritten to this). Blank = "
             "<project_dir>/subtomo/. A trailing slash is enforced."},
            {"name": "PATH_MATCH", "kind": "env", "flag": "PATH_MATCH",
             "default": "subtomo/", "help": "Path prefix in the Warp star to replace with "
             "PARTICLEDIR. Change only if export wrote a different prefix."},
            {"name": "CS", "kind": "env", "flag": "CS",
             "default": "2.7", "help": "Spherical aberration (mm) for relion_convert_star."},
            {"name": "Q0", "kind": "env", "flag": "Q0",
             "default": "0.07", "help": "Amplitude contrast for relion_convert_star."},
            {"name": "NREF", "kind": "env_int", "flag": "NREF",
             "default": 1000, "min": 100, "max": 5000, "step": 100,
             "help": "Random particles used to reconstruct the initial reference."},
            {"name": "MAKE_REF", "kind": "env", "flag": "MAKE_REF",
             "default": "1", "help": "1 = also build random_subset_ref.mrc (relion_reconstruct); "
             "0 = only convert the star."},
            {"name": "execute", "kind": "check", "flag": "--execute",
             "default": False, "help": "OFF = dry run (prints the plan + relion commands, "
             "runs nothing). Turn ON to actually convert + reconstruct."},
        ],
        "docs": {
            "what": "Converts the Warp ts_export_particles star to RELION 4 format and builds "
                    "a de-novo initial reference. Rewrites the relative 'subtomo/' particle "
                    "paths to absolute (so RELION finds every subtomogram), runs "
                    "relion_convert_star, then samples NREF random particles and "
                    "relion_reconstructs random_subset_ref.mrc.",
            "range": "NREF 500-2000; Cs 2.7 mm, Q0 0.07 (300 kV cryo defaults).",
            "effect": "Writes <star>_conv.star (the RELION 4 particles) and, unless MAKE_REF=0, "
                      "random_subset_ref.mrc — a ready-to-use reference for Class3D (no external "
                      "EMDB map needed). Defaults to a DRY RUN — tick EXECUTE to run.",
            "pitfall": "Runs FROM project_dir (paths are relative to it). PATH_MATCH must match "
                       "how export wrote the paths ('subtomo/' by default) or the rewrite is a "
                       "no-op and RELION can't find the particles. The header split is "
                       "auto-detected from the star's '_rln' labels (replaces the old hardcoded "
                       "head -n 33 / tail -n +35).",
        },
        "status": None,
    },
    {
        # A RELION job that was run OUTSIDE tomogration (Refine3D / Class3D / Select),
        # adopted from the Found-on-disk drawer so it can act as a parent in the DAG.
        # It has no command of its own — the work is already done. Its whole purpose is
        # to be a node you can "Build downstream from", which is how a finished RELION
        # refinement feeds M (mask -> species) without you copying three long paths by
        # hand. Adoption records the ACTUAL filenames it found (RELION's differ between
        # a converged run and a mid-iteration one), so the children are filled in
        # correctly rather than guessed.
        "group": "10. RELION 4", "id": "relion4_result",
        "label": "RELION 4: result (adopted)",
        "base": "",
        "tool": "adopted",
        "params": [
            {"name": "job_dir", "kind": "text", "flag": None, "default": "",
             "help": "The RELION job folder this card stands for."},
            {"name": "job_type", "kind": "text", "flag": None, "default": "",
             "help": "Refine3D / Class3D / Select."},
            {"name": "data_star", "kind": "text", "flag": None, "default": "",
             "help": "Its particle star (run_itNNN_data.star / particles.star)."},
            {"name": "half1", "kind": "text", "flag": None, "default": "",
             "help": "Unfiltered half-map 1, if this job produced one."},
            {"name": "half2", "kind": "text", "flag": None, "default": "",
             "help": "Unfiltered half-map 2, if this job produced one."},
            {"name": "class_map", "kind": "text", "flag": None, "default": "",
             "help": "A class/consensus map, used as the input for mask creation."},
        ],
        "docs": {
            "what": "A finished RELION job adopted from disk so the workflow graph knows "
                    "about it. Nothing runs — this is a handle on work already done.",
            "range": "Adopt Refine3D results to start M; adopt Select results to "
                     "re-extract.",
            "effect": "Right-click ▸ Build downstream to create the next step with its "
                      "half maps / particle star already filled in.",
            "pitfall": "Re-running this card does nothing (it has no command). If the "
                       "RELION job changes on disk, adopt it again.",
        },
        "status": None,
    },
    {
        # Pre-flight for any RELION run: verify every particle the star lists actually
        # exists on disk. RELION reads the star, not the disk, so a star listing a
        # subtomogram that was never written runs fine for an iteration or two and THEN
        # dies mid-expectation ("Cannot read file ... It does not exist"), wasting the
        # whole run. Driven by ml_star_check_particles.py; can prune the star in place
        # of a re-export when only a scattering of edge particles is missing.
        "group": "10. RELION 4", "id": "relion4_check_star",
        "label": "RELION 4: check particles exist",
        "base": "python3",
        "params": [
            {"name": "script", "kind": "text", "flag": None,
             "default": _pkg_script("ml_star_check_particles.py"),
             "help": "Particle-existence checker (shipped with the app)."},
            {"name": "star", "kind": "text", "flag": None,
             "default": "",
             "help": "The particle STAR you are about to refine (e.g. "
             "relion4/<project>/matching_conv.star). REQUIRED."},
            {"name": "root", "kind": "text", "flag": "--root",
             "default": "",
             "help": "Only for stars with RELATIVE particle paths: the dir to resolve "
             "them against (usually the RELION project dir). Blank = the star's own "
             "folder, then the working dir. Absolute paths need nothing here."},
            {"name": "fix", "kind": "check", "flag": "--fix",
             "default": False,
             "help": "Write a pruned star keeping ONLY particles whose file exists. OFF "
             "= report only. Prune when a scattering of particles is missing (edge-"
             "clipped); if whole tomograms are missing, re-run the export instead."},
            {"name": "out", "kind": "text", "flag": "--out",
             "default": "",
             "help": "Where to write the pruned star (default: <star>_present.star). "
             "Keep it inside the RELION project dir."},
            {"name": "ref", "kind": "text", "flag": "--ref",
             "default": "",
             "help": "Optional: also check the REFERENCE map you will refine against — "
             "that it exists, and that its box + pixel size match the particles. RELION 4 "
             "does NOT rescale a reference, and a wrong path fails at startup ('Cannot "
             "read file ...'), because RELION resolves relative paths from the dir it was "
             "LAUNCHED in. The relion4_convert step writes a ready-made one at particle "
             "scale: <project>/random_subset_ref.mrc."},
        ],
        "validate": lambda v: (
            "⚠ Point 'star' at the particle STAR you will refine (the one RELION reads)."
            if not str(v.get("star", "")).strip() else ""),
        "docs": {
            "what": "Checks every _rlnImageName in a particle STAR against the files on "
                    "disk, reports how many are missing and which tomograms they are in, "
                    "and can write a pruned STAR containing only the particles that exist.",
            "range": "Run it after every export, before any Refine3D/Class3D.",
            "effect": "Report only by default. With 'fix' it writes <star>_present.star, "
                      "preserving the optics block and all formatting — only the missing "
                      "particle rows are dropped.",
            "pitfall": "RELION validates particle files lazily, so a bad star fails HOURS "
                       "into a refinement, not at the start. If whole tomograms are "
                       "missing the export was interrupted or failed — re-export rather "
                       "than prune, or you silently throw away good data.",
        },
        "status": None,
    },
    {
        # The other half of the pre-flight: relion4_check_star proves the particle FILES
        # exist; this proves they were cut in the RIGHT PLACES. Every coordinate mistake
        # in the RELION->Warp loop is silent (wrong --coords_angpix, normalised coords
        # exported without --normalized_coords, a missed recentring) — Warp writes a
        # well-formed star either way and you only find out hours later when the map is
        # a blob. Driven by ml_verify_reextract.py.
        "group": "10. RELION 4", "id": "relion4_verify_reextract",
        "label": "RELION 4: verify re-extraction",
        "base": "python3",
        "params": [
            {"name": "script", "kind": "text", "flag": None,
             "default": _pkg_script("ml_verify_reextract.py"),
             "help": "Re-extraction coordinate verifier (shipped with the app)."},
            {"name": "source_star", "kind": "text", "flag": None,
             "default": "",
             "help": "The RELION star you re-extracted FROM (e.g. "
             "Select/job009/particles.star). REQUIRED."},
            {"name": "new_star", "kind": "text", "flag": None,
             "default": "",
             "help": "The matching.star the export WROTE (e.g. "
             "relion4/<project>/matching.star). REQUIRED."},
            {"name": "no_recenter", "kind": "check", "flag": "--no-recenter",
             "default": False,
             "help": "Tick ONLY if the converter ran with recentring disabled. Leave OFF "
             "when MODE C applied the refined origins (the default), or every particle "
             "will look off by its refinement shift."},
        ],
        "validate": lambda v: (
            "⚠ Give both stars: the RELION star you re-extracted from, and the "
            "matching.star the export wrote."
            if not (str(v.get("source_star", "")).strip()
                    and str(v.get("new_star", "")).strip()) else ""),
        "docs": {
            "what": "Pairs particles per tomogram between the RELION star you re-extracted "
                    "from and the matching.star the export wrote, and checks the "
                    "coordinates agree after recentring and rescaling.",
            "range": "max error < 1 px = PASS. < 5 px = rounding, probably fine. "
                     "Anything larger = the particles were cut in the wrong places.",
            "effect": "Read-only report. On failure it diagnoses the likely cause from the "
                      "coordinate ratio (2x/0.5x = an angpix mistake; collapsed toward "
                      "zero = normalised coords exported without --normalized_coords).",
            "pitfall": "Run this BEFORE any refinement. A wrong extraction is completely "
                       "silent — RELION refines it happily for hours and yields noise. "
                       "Pair with 'check particles exist' (files present) — this one "
                       "answers the different question of whether they are the RIGHT files.",
        },
        "status": None,
    },
    {
        # Extends the pipeline past export into a RELION 4 Class3D handoff, driven by
        # ml_relion4_handoff_warp_auto.sh. Encodes the invariants that repeatedly bite:
        # launch-from-export-dir, v4 reference pre-scale, clean project dir, nGPU+1 MPI.
        "group": "10. RELION 4", "id": "relion4_class3d",
        "label": "RELION 4: Class3D handoff",
        "base": "bash",
        # project_dir is the RELION PROJECT ROOT — it holds matching_conv.star,
        # subtomo/ and every previous job, none of which this touches. What it
        # actually writes is Class3D/job001/, and that name is HARDCODED in the
        # handoff script, so a second Class3D overwrites the first. Declaring the
        # subfolder puts the overwrite warning on the thing genuinely at risk
        # instead of on the container.
        "output_params": ["project_dir"],
        "output_subdir_param": "project_dir",
        "output_subdirs": ["Class3D/job001"],
        "params": [
            {"name": "script", "kind": "text", "flag": None,
             "default": _pkg_script("ml_relion4_handoff_warp_auto.sh"),
             "help": "RELION 4 handoff wrapper (shipped with the app)."},
            {"name": "project_dir", "kind": "text", "flag": None,
             "default": "relion4/warp",
             "help": "The RELION project dir = ts_export_particles' output_processing. "
             "The script runs relion FROM here (the launch-root invariant)."},
            {"name": "particles", "kind": "text", "flag": None,
             "default": "matching.star",
             "help": "Particles star, RELATIVE to project_dir (e.g. matching.star)."},
            {"name": "RELION_MODULE", "kind": "env", "flag": "RELION_MODULE",
             "default": "relion/4.0.1", "help": "module load name for RELION 4 on your cluster."},
            {"name": "REF_MAP", "kind": "env", "flag": "REF_MAP",
             "default": "", "help": "Reference map (.mrc) — e.g. the EMDB map you template-"
             "matched with. RELION 4 does NOT auto-resize; the script rescales it to match."},
            {"name": "REF_ANGPIX", "kind": "env", "flag": "REF_ANGPIX",
             "default": "", "help": "Pixel size (Å) of REF_MAP (from its header/EMDB page)."},
            {"name": "OUTPUT_ANGPIX", "kind": "env", "flag": "OUTPUT_ANGPIX",
             "default": "4", "help": "Must equal the export output_angpix (particles' Å/px)."},
            {"name": "BOX", "kind": "env_int", "flag": "BOX",
             "default": 64, "min": 32, "max": 256, "step": 8,
             "help": "Must equal the export box size (px)."},
            {"name": "DIAMETER", "kind": "env", "flag": "DIAMETER",
             "default": "", "help": "Particle diameter (Å) for the mask."},
            {"name": "SYMMETRY", "kind": "env", "flag": "SYMMETRY",
             "default": "C1", "help": "Classify in C1; symmetrise only at Refine3D."},
            {"name": "NCLASSES", "kind": "env_int", "flag": "NCLASSES",
             "default": 4, "min": 1, "max": 12, "step": 1, "help": "Number of 3D classes (K)."},
            {"name": "GPUS", "kind": "env", "flag": "GPUS", "gpu_sep": ",",
             "default": "0,1,2,3", "help": "GPU ids for RELION (idle ones — check util%). "
             "Any separator; the script converts to RELION's colon form (0:1:2:3) so each "
             "MPI follower gets its OWN GPU — a space/comma list makes RELION pile all "
             "followers onto GPU 0. MPI is set to (#GPUs + 1) automatically."},
            {"name": "execute", "kind": "check", "flag": "--execute",
             "default": False, "help": "OFF = dry run (prints the plan + relion command, "
             "runs nothing). Turn ON to actually submit Class3D."},
        ],
        "docs": {
            "what": "Hands the exported subtomograms to RELION 4 for 3D classification — "
                    "asserts the export is complete, checks the launch-root path invariant, "
                    "pre-scales the reference, cleans the project dir, and submits Class3D.",
            "range": "NCLASSES 3-6; ini-lowpass 45 Å; MPI = #GPUs + 1.",
            "effect": "Runs relion_refine_mpi from project_dir. Defaults to a DRY RUN — tick "
                      "EXECUTE to launch. RELION 4 is the GPU-native path on this VM class "
                      "(RELION 5's container CUDA can outrun the host driver → GPU error 35).",
            "pitfall": "Class3D/job001 is a FIXED name — running this twice in the same project_dir overwrites the previous classification (the particle star and subtomos are untouched). Use a different project_dir to keep both. REF must be pre-scaled to OUTPUT_ANGPIX + BOX (the script does it via "
                       "relion_image_handler). OUTPUT_ANGPIX/BOX MUST match the export. Never "
                       "launch inside another RELION version's project (the script parks stale "
                       "pipeline files first).",
        },
        "status": None,
    },
    {
        # Select the "good" 3D class(es) from a RELION Class3D result and turn them
        # back into Warp pick stars, so ts_export_particles can RE-EXTRACT just those
        # particles at a finer pixel size (bin4 -> bin2) for a higher-res refine.
        # Driven by ml_relion4_select_picks.py. The key trick: DON'T convert RELION
        # coordinates (that mangles them -> garbage volume) — RELION never moves a
        # particle during classification, it only labels it, and keeps the original
        # subtomo path in _rlnImageName, which is an exact key back to the Warp pick.
        # Two ways in (fill ONE): MODE A filters the per-tomogram pick stars you fed
        # to Export; MODE B rebuilds them from matching.star + the reconstructions.
        "group": "10. RELION 4", "id": "relion4_select_picks",
        "label": "RELION 4: select good class → picks",
        "base": "python3",
        "output_params": ["out_dir"],
        "params": [
            {"name": "script", "kind": "text", "flag": None,
             "default": _pkg_script("ml_relion4_select_picks.py"),
             "help": "Class-selection → Warp-picks tool (shipped with the app)."},
            {"name": "class_star", "kind": "text", "flag": None,
             "default": "",
             "help": "The RELION classification result to select from: the "
             "run_itNNN_data.star inside your Class3D job (e.g. "
             "relion4/warp/Class3D/job005/run_it025_data.star). REQUIRED."},
            {"name": "classes", "kind": "text", "flag": "--classes",
             "default": "",
             "help": "The good class number(s) — the ones that look like your particle "
             "in the RELION display. Type one (e.g. 3) or several (e.g. 1,3). LEAVE "
             "BLANK until you know which class is good; the tool prints the class "
             "populations on a dry run to help you decide. REQUIRED before EXECUTE."},
            # ---- MODE A: filter existing per-tomogram pick stars ----
            {"name": "picks_dir", "kind": "text", "flag": "--picks-dir",
             "default": "",
             "help": "MODE A (simplest, coords untouched): the dir of per-tomogram pick "
             "stars you fed to Export (e.g. warp_tiltseries/matching_cryolo). Fill this "
             "OR the MODE B fields below — not both."},
            {"name": "pattern", "kind": "text", "flag": "--pattern",
             "default": "*_cryolo.star",
             "help": "MODE A: filename glob inside picks_dir (default *_cryolo.star)."},
            # ---- MODE B: rebuild from matching.star + reconstructions ----
            {"name": "from_matching", "kind": "text", "flag": "--from-matching",
             "default": "",
             "help": "MODE B (use if the pick stars are gone): the matching.star that "
             "fed RELION (e.g. relion4/cryolo_combined/matching.star). Needs recon_dir "
             "too. Fill this OR MODE A — not both."},
            {"name": "recon_dir", "kind": "text", "flag": "--recon-dir",
             "default": "",
             "help": "MODE B: the reconstruction dir (PositionNNN_<apx>Apx.mrc) — used to "
             "re-normalise the coordinates. Usually warp_tiltseries/reconstruction."},
            {"name": "apx", "kind": "text", "flag": "--apx",
             "default": "12.56",
             "help": "MODE B: the angpix tag in the reconstruction filenames "
             "(PositionNNN_<apx>Apx.mrc). Default 12.56."},
            # ---- common ----
            {"name": "out_dir", "kind": "text", "flag": "--out-dir",
             "default": "warp_tiltseries/matching_good",
             "help": "Where to write the filtered pick stars. Point ts_export_particles' "
             "input_directory here for the bin2 re-extraction."},
            {"name": "suffix", "kind": "text", "flag": "--suffix",
             "default": "good",
             "help": "Tag added to output filenames (default 'good'); the export "
             "input_pattern is then '*_good.star'."},
            {"name": "execute", "kind": "check", "flag": "--execute",
             "default": False,
             "help": "OFF = dry run (prints class populations + per-tomogram counts, "
             "writes nothing). Turn ON to actually write the filtered pick stars."},
        ],
        "validate": lambda v: (
            # Required positional — see the note on relion4_to_warp.
            "⚠ No classification star. This card needs the Class3D run_itNNN_data.star "
            "to read class assignments from."
            if not str(v.get("class_star", "")).strip() else
            "⚠ Type the good class number(s) into 'classes' (e.g. 3, or 1,3). Run a dry "
            "run first (EXECUTE off) to see the class populations."
            if not str(v.get("classes", "")).strip() else
            "⚠ Fill MODE A (picks_dir) OR MODE B (from_matching), not both."
            if str(v.get("picks_dir", "")).strip() and str(v.get("from_matching", "")).strip() else
            "⚠ Fill either picks_dir (MODE A) or from_matching + recon_dir (MODE B)."
            if not str(v.get("picks_dir", "")).strip() and not str(v.get("from_matching", "")).strip() else
            "⚠ MODE B needs recon_dir (the reconstructions) to re-normalise coordinates."
            if str(v.get("from_matching", "")).strip() and not str(v.get("recon_dir", "")).strip() else
            "⚠ Not the classification star. Use the run_itNNN_data.star from your Class3D "
            "job (has _rlnClassNumber), not matching.star."
            if str(v.get("class_star", "")).strip().endswith("matching.star") else ""),
        "docs": {
            "what": "Selects the good 3D class(es) from a RELION Class3D result and writes "
                    "Warp pick stars containing only those particles, so ts_export_particles "
                    "can re-extract them at a finer pixel size for a higher-resolution refine.",
            "range": "classes: whichever class(es) look right in RELION. MODE A if you kept "
                     "the pick stars that fed Export; MODE B otherwise.",
            "effect": "Writes filtered *_<suffix>.star pick stars to out_dir. No coordinate "
                      "conversion — matches particles by their subtomo filename, so the "
                      "coordinates stay exactly what Warp already used. Defaults to a DRY RUN.",
            "pitfall": "Feeding RELION's OWN coordinates back to Warp is what produces a "
                       "garbage volume — this tool avoids that entirely. After it runs, "
                       "re-extract with ts_export_particles: --input_directory <out_dir> "
                       "--input_pattern '*_good.star' --output_angpix <finer> --box <bigger>. "
                       "Keep --normalized_coords for MODE B (and for MODE A if your original "
                       "export used it). Verify ONE tomogram before trusting all.",
        },
        "status": None,
    },
    {
        # The RELION 4 -> Warp particle converter the user actually wants: take a
        # RELION particles.star that is ALREADY the final set you want (typically the
        # Subset-selection output — the good class with duplicate particles removed,
        # the same star fed to Refine3D) and re-extract EXACTLY those particles in Warp
        # at a finer pixel size. No class filtering (--keep-all): every particle in the
        # input is kept, matched to its Warp pick by the subtomo filename in
        # _rlnImageName, and re-extracted with Warp's own coordinates untouched. Same
        # engine as 'select good class' (ml_relion4_select_picks.py) but keep-all.
        "group": "10. RELION 4", "id": "relion4_to_warp",
        "label": "RELION 4 → Warp: re-extract particles",
        "base": "python3",
        "output_params": ["out_dir"],
        "params": [
            {"name": "script", "kind": "text", "flag": None,
             "default": _pkg_script("ml_relion4_select_picks.py"),
             "help": "RELION→Warp pick converter (shipped with the app)."},
            {"name": "particles_star", "kind": "text", "flag": None,
             "default": "",
             "help": "The RELION particles.star to re-extract. Use your SUBSET-SELECTION "
             "output — the good class with duplicate particles removed (the same star you "
             "fed to Refine3D), e.g. Select/job009/particles.star. NOT the raw Class3D "
             "run_itNNN_data.star (that still has every class + duplicates). REQUIRED."},
            {"name": "keep_all", "kind": "check", "flag": "--keep-all",
             "default": True,
             "help": "Re-extract EVERY particle in the input star (no class filtering). "
             "Leave ON — your Subset-selection output is already exactly the particles you "
             "want. (Turn off only if you meant to use the 'select good class' card.)"},
            # ---- MODE C (recommended): coords straight from the RELION star ----
            {"name": "relion_coords", "kind": "check", "flag": "--relion-coords",
             "default": True,
             "help": "MODE C (RECOMMENDED — leave ON). Take coordinates from the RELION "
             "star itself: they are already in a known Å/px, so there is no normalising, "
             "no reconstruction dimensions and no pick stars involved. Also RECENTRES on "
             "the refined origins and carries the refined angles. When ON, picks_dir / "
             "from_matching below are IGNORED. Export then needs --coords_angpix and NO "
             "--normalized_coords (the card prints the exact flags)."},
            {"name": "coords_angpix", "kind": "text", "flag": "--coords-angpix",
             "default": "",
             "help": "MODE C: override the coords' pixel size (Å/px). Blank = read it from "
             "the star (_rlnImagePixelSize), which is what you want."},
            {"name": "no_recenter", "kind": "check", "flag": "--no-recenter",
             "default": False,
             "help": "MODE C: tick to NOT apply the refined _rlnOriginXYZAngst shifts. "
             "Leave OFF — applying them re-extracts on the centre the refinement found "
             "(skipping it re-extracts on the original pick, mis-centred by tens of Å)."},
            {"name": "no_keep_angles", "kind": "check", "flag": "--no-keep-angles",
             "default": False,
             "help": "MODE C: tick to NOT carry the refined Euler angles into the picks. "
             "Leave OFF — carrying them gives the new refinement its orientation priors "
             "instead of restarting from scratch."},
            # ---- MODE A: filter existing per-tomogram pick stars ----
            {"name": "picks_dir", "kind": "text", "skip_if": lambda v: bool(v.get("relion_coords")),
             "flag": "--picks-dir",
             "default": "",
             "help": "MODE A (simplest, coords untouched): the dir of per-tomogram pick "
             "stars you fed to Export (e.g. warp_tiltseries/matching_cryolo). Fill this "
             "OR the MODE B fields below — not both."},
            {"name": "pattern", "kind": "text", "skip_if": lambda v: bool(v.get("relion_coords")),
             "flag": "--pattern",
             "default": "*_cryolo.star",
             "help": "MODE A: filename glob inside picks_dir (default *_cryolo.star)."},
            # ---- MODE B: rebuild from matching.star + reconstructions ----
            {"name": "from_matching", "kind": "text", "skip_if": lambda v: bool(v.get("relion_coords")),
             "flag": "--from-matching",
             "default": "",
             "help": "MODE B (use if the pick stars are gone): the matching.star that "
             "fed RELION (e.g. relion4/cryolo_combined/matching.star). Needs recon_dir "
             "too. Fill this OR MODE A — not both."},
            {"name": "recon_dir", "kind": "text", "skip_if": lambda v: bool(v.get("relion_coords")),
             "flag": "--recon-dir",
             "default": "",
             "help": "MODE B: the reconstruction dir (PositionNNN_<apx>Apx.mrc) — used to "
             "re-normalise coordinates. Usually warp_tiltseries/reconstruction."},
            {"name": "apx", "kind": "text", "skip_if": lambda v: bool(v.get("relion_coords")),
             "flag": "--apx",
             "default": "12.56",
             "help": "MODE B: the angpix tag in the reconstruction filenames "
             "(PositionNNN_<apx>Apx.mrc). Default 12.56."},
            # ---- common ----
            {"name": "out_dir", "kind": "text", "flag": "--out-dir",
             "default": "warp_tiltseries/matching_reextract",
             "help": "Where to write the pick stars. Point ts_export_particles' "
             "input_directory here for the finer-bin re-extraction."},
            {"name": "suffix", "kind": "text", "flag": "--suffix",
             "default": "reextract",
             "help": "Tag added to output filenames (default 'reextract'); the export "
             "input_pattern is then '*_reextract.star'."},
            {"name": "execute", "kind": "check", "flag": "--execute",
             "default": False,
             "help": "OFF = dry run (prints how many particles matched, writes nothing). "
             "Turn ON to actually write the pick stars."},
        ],
        "validate": lambda v: (
            # The star is a REQUIRED POSITIONAL argument. Blank means the script dies
            # with "the following arguments are required: class_star" after the job
            # has already been queued and run — which is exactly what a card created
            # with template defaults does.
            "⚠ No particle star. This card needs the RELION star to re-extract from "
            "(e.g. Select/job019/particles.star). Build it downstream of a Subset "
            "selection card and it is filled in for you."
            if not str(v.get("particles_star", "")).strip() else
            "⚠ Keep 'keep_all' ON here — this card re-extracts every particle in the input "
            "star. To pick specific classes instead, use the 'select good class' card."
            if not v.get("keep_all") else
            "ℹ MODE C: coords come from the RELION star. Export MUST use --coords_angpix "
            "(the run prints the value) and must NOT use --normalized_coords. Remember to "
            "DOUBLE the box when you halve output_angpix, to keep the same field of view."
            if v.get("relion_coords") else
            "⚠ Fill MODE A (picks_dir) OR MODE B (from_matching), not both."
            if str(v.get("picks_dir", "")).strip() and str(v.get("from_matching", "")).strip() else
            "⚠ Fill either picks_dir (MODE A) or from_matching + recon_dir (MODE B) — or "
            "tick relion_coords for MODE C (recommended)."
            if not str(v.get("picks_dir", "")).strip() and not str(v.get("from_matching", "")).strip() else
            "⚠ MODE B needs recon_dir (the reconstructions) to re-normalise coordinates."
            if str(v.get("from_matching", "")).strip() and not str(v.get("recon_dir", "")).strip() else ""),
        "docs": {
            "what": "Converts a finished RELION particle set (typically the Subset-selection "
                    "output — good class, duplicates removed) into Warp pick stars, so "
                    "ts_export_particles re-extracts exactly those particles at a finer "
                    "pixel size for a higher-resolution refine.",
            "range": "Point it at Select/jobNNN/particles.star. MODE A if you kept the pick "
                     "stars that fed Export; MODE B otherwise.",
            "effect": "Writes *_reextract.star pick stars to out_dir. No coordinate "
                      "conversion and no class filtering — every particle in the input is "
                      "matched to its Warp pick by subtomo filename and kept. Dry run by default.",
            "pitfall": "Use the Subset-selection output, not the raw Class3D result — the "
                       "latter still contains all classes and duplicate picks. After it runs, "
                       "re-extract with ts_export_particles: --input_directory <out_dir> "
                       "--input_pattern '*_reextract.star' --output_angpix <finer> --box "
                       "<bigger>. Keep --normalized_coords for MODE B (and for MODE A if your "
                       "original export used it). Verify ONE tomogram before trusting all.",
        },
        "status": None,
    },
    # ================= 11. M — multi-particle refinement =====================
    # M sits at the FAR END of the pipeline: it takes a finished RELION refinement
    # and re-refines the tilt-series geometry, particle poses and CTF *together*,
    # against the raw data. For tilt series it usually beats the initial AreTomo /
    # IMOD alignment noticeably. Terminology is deliberately sociological:
    #     Population = the project        Data Source = a set of tilt series
    #     Species    = one map being refined (+ its particle metadata)
    # Setup runs once (population -> source -> mask -> species); MCore is then run
    # REPEATEDLY, adding one refinement parameter at a time. That iterate-and-compare
    # loop is exactly what job cards are for — fork a card, add a flag, compare.
    {
        # THE pre-flight for M, and the first thing to run when MCore throws
        # IndexOutOfRangeException. A tilt series whose CTF estimation never
        # succeeded keeps its .xml, imports into a data source and reconstructs
        # normally — but MPA refinement builds a per-tilt array from the CTF and
        # walks off the end of the empty one. The exception is raised inside a
        # worker, so the series name never reaches stdout.
        #
        # `MCore --iter 0` succeeds anyway, because iter 0 performs no refinement
        # and never touches that array. "iter 0 fine, real refinement crashes" is
        # this bug's signature; on EML45 it was 2 series out of 290, and it was
        # misread for days as a mask / population-size / flag problem.
        "group": "11. M refinement", "id": "m_check_ctf",
        "label": "M: pre-flight — CTF completeness",
        "base": "python3",
        "params": [
            {"name": "script", "kind": "text", "flag": None,
             "default": _pkg_script("ml_m_check_ctf.py"),
             "help": "CTF completeness checker (shipped with the app)."},
            {"name": "project_dir", "kind": "text", "flag": None,
             "default": ".",
             "help": "Project root (holds warp_tiltseries/ and tomostar/)."},
            {"name": "processing", "kind": "text", "flag": "--processing",
             "default": "warp_tiltseries",
             "help": "Tilt-series processing folder holding the per-series .xml "
             "metadata, relative to the project root."},
            {"name": "tomostar", "kind": "text", "flag": "--tomostar",
             "default": "tomostar",
             "help": "Folder holding the .tomostar files, relative to the project "
             "root. Used to report each offender's tilt count and to build the "
             "ts_ctf command that fixes them."},
            {"name": "write_list", "kind": "text", "flag": "--write-list",
             "default": "",
             "help": "Optional: write the offending series' tomostar paths to this "
             "file, one per line, ready for `ts_ctf --input_data`. Blank = don't."},
        ],
        "docs": {
            "what": "Reads every tilt series' .xml and reports which ones carry no "
                    "CTF estimate (CTFResolutionEstimate absent, empty or 0). Prints "
                    "the exact ts_ctf command to fix the ones it finds.",
            "range": "Run before M setup, and again after any re-run of ts_ctf. "
                     "It must say PASS before you build a population.",
            "effect": "Read-only. Exits non-zero when any series is unestimated, so "
                      "it also works as a gate in a script.",
            "pitfall": "Fixing the CTF is not enough on its own — a data source "
                       "CACHES the tilt-series metadata when it is created, so an "
                       "existing population will not see the new estimates. Re-run "
                       "ts_ctf, then reset M and rebuild population → source → "
                       "species.",
        },
        "status": None,
    },
    {
        # M commits every refinement round as a new species version in a folder with
        # an opaque random name ("Hhhxx2to", "-2-ij4-g") — no index, no order, and
        # nothing inside saying which run made it. After a few rounds you cannot tell
        # which folder was the good one. Renaming them breaks the species (the
        # .species file refers to version names internally), so this labels them
        # instead: it matches each folder's write time against the job store and
        # writes an inert text file inside.
        "group": "11. M refinement", "id": "m_index_versions",
        "label": "M: label version folders",
        "base": "python3",
        "params": [
            {"name": "script", "kind": "text", "flag": None,
             "default": _pkg_script("ml_m_index_versions.py"),
             "help": "Species-version indexer (shipped with the app)."},
            {"name": "project_dir", "kind": "text", "flag": None,
             "default": ".",
             "help": "Project root (holds m*/ and .tomogration_jobs.json)."},
            {"name": "species", "kind": "text", "flag": "--species",
             "default": "",
             "help": "Only index species whose folder name contains this. Blank = "
             "all of them."},
            {"name": "write", "kind": "check", "flag": "--write",
             "default": False,
             "help": "OFF = print a chronological table only. Turn ON to also write "
             "_tomogration_version.txt inside each version folder, so the label "
             "travels with the data and survives this app. M never reads that file."},
        ],
        "docs": {
            "what": "Lists every m*/species/*/versions/* folder in the order it was "
                    "written, with its size, the tomogration job that produced it, "
                    "that job's resolution, and the refinement flags that made it "
                    "different from the round before.",
            "range": "Run it after any batch of MCore rounds, or whenever you are "
                     "looking at the versions folder wondering which is which.",
            "effect": "Read-only unless 'write' is ticked, and even then it only adds "
                      "a new text file — nothing existing is modified.",
            "pitfall": "Do NOT rename the folders yourself. The .species file and the "
                       "population refer to version names internally, so a rename "
                       "breaks the species silently. Runs launched from a terminal "
                       "instead of tomogration have no job to match and are listed "
                       "with their timestamp only.",
        },
        "status": None,
    },
    {
        # Inspect, and optionally wipe, M's setup. M's setup commands are NOT
        # idempotent and fail unreadably when half-reset: create_population on an
        # EXISTING population LOADS it (and every .source it references), so one
        # missing .source turns every later M command into a .NET
        # FileNotFoundException. M also scatters its state across m*/ dirs AND a
        # .source written next to the tilt-series settings, so deleting m/ by hand
        # leaves the project half-configured. Driven by ml_m_reset_warp_auto.sh.
        #
        # There were once TWO cards for this — "check / reset setup" and
        # "reset / wipe M project" — with the SAME id, so the canvas drew two
        # identical ghosts and the stage lookup could only ever reach the first.
        # One card, one id: report is the default, --execute does the wipe.
        "group": "11. M refinement", "id": "m_reset",
        "label": "M: check / reset setup",
        "base": "bash",
        "params": [
            {"name": "script", "kind": "text", "flag": None,
             "default": _pkg_script("ml_m_reset_warp_auto.sh"),
             "help": "M setup inspector / reset (shipped with the app)."},
            {"name": "project_dir", "kind": "text", "flag": None,
             "default": ".",
             "help": "Project root (holds m*/ and warp_tiltseries/). '.' = this project."},
            {"name": "execute", "kind": "check", "flag": "--execute",
             "default": False,
             "help": "OFF = REPORT: lists every population, .source and M directory it "
             "found, and checks whether each population's data source still resolves, "
             "so you can see whether M is in a broken state. Turn ON to MOVE the whole "
             "setup aside to m_trash_<timestamp>/ and start clean. Nothing is ever "
             "deleted."},
        ],
        "docs": {
            "what": "Finds every M artefact (all m*/ population dirs, their species and "
                    "refinement_temp, and the <name>.source files beside the settings), "
                    "reports whether each population's data sources still resolve, and "
                    "with EXECUTE moves them into a timestamped m_trash_<date>/ folder.",
            "range": "Run the report whenever an M command throws FileNotFoundException; "
                     "--execute when you want to start M over.",
            "effect": "MOVES, never deletes — everything stays in m_trash_<date>/ so a "
                      "mistake costs nothing. Raw data is never touched.",
            "pitfall": "The .source file is NOT in m/ — MTools writes it next to the "
                       "processing settings (warp_tiltseries/<name>.source), which is "
                       "why deleting m/ by hand leaves M half-configured. After a reset "
                       "run create_population then create_source ONCE each, in that "
                       "order: create_population on an existing population loads it "
                       "instead of replacing it.",
        },
        "status": None,
    },
    {
        # A crashed MCore leaves WarpWorker processes alive; they keep GPU memory AND
        # the REST port, so the NEXT run either dies at startup with "address already
        # in use" or behaves erratically. This has caused several false diagnoses in
        # this project, so make the cleanup a one-click card rather than folklore.
        "group": "11. M refinement", "id": "m_kill_orphans",
        "label": "M: kill stale Warp/M processes",
        "base": ("bash -lc 'set +e; "
                 # -x matches the EXECUTABLE NAME, not the command line. With -f this
                 # command matched ITSELF (its own text contains \"MCore\") and sent
                 # itself SIGTERM — it died instantly with exit 15 and cleaned nothing.
                 "echo \"--- your Warp/M processes ---\"; "
                 "pgrep -a -x -u $USER MCore; pgrep -a -x -u $USER WarpWorker; "
                 "pgrep -x -u $USER MCore >/dev/null || pgrep -x -u $USER WarpWorker "
                 ">/dev/null || echo \"  (none)\"; "
                 "pkill -x -u $USER MCore; pkill -x -u $USER WarpWorker; sleep 2; "
                 "pkill -9 -x -u $USER MCore; pkill -9 -x -u $USER WarpWorker; sleep 1; "
                 "echo; echo \"--- after ---\"; "
                 "pgrep -a -x -u $USER MCore || pgrep -a -x -u $USER WarpWorker || "
                 "echo \"  all clear\"; "
                 "echo; echo \"--- REST ports ---\"; "
                 "ss -tlnp 2>/dev/null | grep -E \":143[0-9][0-9]\" || echo \"  free\"; "
                 "echo; echo \"--- GPUs (other users included) ---\"; "
                 "nvidia-smi --query-compute-apps=pid,process_name,used_memory "
                 "--format=csv 2>/dev/null; exit 0'"),
        "params": [],
        "docs": {
            "what": "Kills YOUR leftover MCore/WarpWorker processes, then shows the "
                    "REST port and what is still on the GPUs.",
            "range": "Run after any MCore crash, before the next attempt.",
            "effect": "Only ever touches processes owned by you — another user's jobs "
                      "are listed by nvidia-smi but never signalled.",
            "pitfall": "MCore's REST port (14300) is a fixed global default, so on a "
                       "shared machine a colleague's run collides with yours even after "
                       "cleanup. Always pass a private --port as well (e.g. 14372).",
        },
        "status": None,
    },
    {
        "group": "11. M refinement", "id": "m_create_population",
        "label": "M: create population",
        "base": "MTools create_population",
        "params": [
            {"name": "directory", "kind": "text", "flag": "--directory",
             "default": "m", "help": "Folder to hold the M project (created if absent)."},
            {"name": "name", "kind": "text", "flag": "--name",
             "default": "",
             "help": "Population name — becomes <directory>/<name>.population, which "
             "every later M step points at. Use something short and dataset-specific "
             "(e.g. EML45_spike). REQUIRED."},
        ],
        "validate": lambda v: (
            "⚠ Give the population a name — the rest of M refers to "
            "<directory>/<name>.population."
            if not str(v.get("name", "")).strip() else
            "ℹ RUN THIS ONCE. On an EXISTING population this does not start fresh — it "
            "LOADS it, and fails with a .NET FileNotFoundException if any data source "
            "it references has moved. To start over use 'M: check / reset setup'."),
        "output_params": ["directory"],
        "docs": {
            "what": "Creates the M project (a 'Population'). Run once per dataset.",
            "range": "directory 'm'; name = anything short and specific.",
            "effect": "Writes m/<name>.population — the handle every later M step needs.",
            "pitfall": "Everything downstream references this exact path. Renaming it "
                       "later means editing every M job.",
        },
        "status": lambda ps: ps.status_m_population(),
    },
    {
        "group": "11. M refinement", "id": "m_create_source",
        "label": "M: create data source",
        "base": "MTools create_source",
        "params": [
            {"name": "name", "kind": "text", "flag": "--name",
             "default": "", "help": "Data-source name (usually the same as the "
             "population name). REQUIRED."},
            {"name": "population", "kind": "text", "flag": "--population",
             "default": "m/{name}.population",
             "help": "The .population file from 'create population'."},
            {"name": "processing_settings", "kind": "text",
             "flag": "--processing_settings",
             "default": "warp_tiltseries.settings",
             "help": "Your tilt-series settings file — this is how M finds the tilt "
             "series, their alignments and CTF metadata."},
        ],
        "validate": lambda v: (
            "⚠ Give the data source a name."
            if not str(v.get("name", "")).strip() else
            "ℹ RUN THIS ONCE. Re-running is a no-op ('already exists in this "
            "population'). NOTE the .source file is written next to the SETTINGS "
            "(warp_tiltseries/<name>.source), not into m/ — delete it by hand and the "
            "population breaks. Use 'M: check / reset setup' instead."),
        # The .source is NOT beside the settings and NOT in m/ — MTools writes it
        # into the processing folder named inside the .settings file, which no
        # parameter here spells out. Find it by name instead of guessing.
        "output_find": "{name}.source",
        "output_params": ["population"],
        "docs": {
            "what": "Registers your tilt series (and all their Warp metadata) with the "
                    "population as a Data Source.",
            "range": "One source per settings file; a population may hold several.",
            "effect": "M now knows which tilt series it is allowed to refine against.",
            "pitfall": "Point it at the TILT-SERIES settings, not the frame-series one.",
        },
        "status": lambda ps: ps.status_m_source(),
    },
    {
        "group": "11. M refinement", "id": "m_mask_create",
        "label": "M: create mask (RELION)",
        "base": "relion_mask_create",
        "params": [
            {"name": "relion_module", "kind": "module", "flag": None,
             "default": "relion/4.0.1",
             "help": "lmod module providing relion_mask_create. Everything on this "
             "cluster is behind a module, so without this the command is simply "
             "'not found'. Blank = assume RELION is already on PATH."},
            {"name": "i", "kind": "text", "flag": "--i",
             "default": "",
             "help": "Input map to threshold — normally your RELION Refine3D result "
             "(e.g. relion4/<proj>/Refine3D/jobNNN/run_class001.mrc). REQUIRED."},
            {"name": "o", "kind": "text", "flag": "--o",
             "default": "m/mask.mrc", "help": "Output binary mask."},
            {"name": "ini_threshold", "kind": "text", "flag": "--ini_threshold",
             "default": "0.04",
             "help": "Density threshold that becomes the mask edge. Check it in 3dmod: "
             "the mask should hug the particle with no floating specks. Too low = the "
             "mask swallows solvent noise; too high = it clips the particle."},
        ],
        "validate": lambda v: ("⚠ Point --i at the map to make a mask from (your "
                               "Refine3D class/half map)."
                               if not str(v.get("i", "")).strip() else ""),
        "output_params": ["o"],
        "docs": {
            "what": "Makes the binary mask M needs around the particle. M expands it and "
                    "adds a soft edge itself during refinement.",
            "range": "ini_threshold ~0.01-0.05 depending on map contrast.",
            "effect": "Writes a binary mask at the SAME box/pixel size as the input map.",
            "pitfall": "Do NOT pre-soften this mask — M does that. Always eyeball it in "
                       "3dmod before refining; a bad mask quietly wrecks the refinement.",
        },
        "status": lambda ps: ps.status_m_mask(),
    },
    {
        "group": "11. M refinement", "id": "m_create_species",
        "label": "M: create species",
        "base": "MTools create_species",
        "params": [
            {"name": "population", "kind": "text", "flag": "--population",
             "default": "", "help": "The .population file. REQUIRED."},
            {"name": "name", "kind": "text", "flag": "--name",
             "default": "",
             "help": "Species name (e.g. spike). REQUIRED. NOTE create_species ADDS a "
             "species — it never replaces one. Running it twice leaves BOTH in the "
             "population, MCore then refines both and prints a resolution line for "
             "each, and if they share a name you cannot tell which is which. To "
             "REPLACE a species, reset M and rebuild. To COMPARE two masks/references, "
             "give them DIFFERENT names on purpose."},
            {"name": "diameter", "kind": "text", "flag": "--diameter",
             "default": "", "help": "Particle diameter in Å — the TRUE particle size, "
             "same number you used for extraction."},
            {"name": "sym", "kind": "text", "flag": "--sym",
             "default": "C1",
             "help": "Point-group symmetry (C1, C3, D2, O…). Only impose symmetry you "
             "are confident in — a wrong one caps resolution and can't be undone later."},
            {"name": "temporal_samples", "kind": "text", "flag": "--temporal_samples",
             "default": "1",
             "help": "Pose samples through the tilt series. START AT 1. Raise it later "
             "with 'resample trajectories' once you have resolution to spend."},
            {"name": "half1", "kind": "text", "flag": "--half1",
             "default": "",
             "help": "UNFILTERED half-map 1 from RELION (run_half1_class001_unfil.mrc)."},
            {"name": "half2", "kind": "text", "flag": "--half2",
             "default": "",
             "help": "UNFILTERED half-map 2 (run_half2_class001_unfil.mrc). The two "
             "halves must be the independent ones — that is what keeps FSC honest."},
            {"name": "mask", "kind": "text", "flag": "--mask",
             "default": "m/mask.mrc", "help": "Binary mask from the previous step."},
            {"name": "particles_relion", "kind": "text", "flag": "--particles_relion",
             "default": "",
             "help": "The RELION run_data.star holding the refined particle poses "
             "(e.g. .../Refine3D/jobNNN/run_data.star)."},
            {"name": "ignore_unmatched", "kind": "check", "flag": "--ignore_unmatched",
             "default": False,
             "help": "Proceed when the particle star references tilt series the data "
             "source does not contain. REQUIRED whenever the population is a SUBSET of "
             "your series (e.g. filtered by particle count) — otherwise create_species "
             "refuses to build, writes an EMPTY species folder, and MCore then reports "
             "'0/0' species with no error of its own. Only tick it when you know the "
             "unmatched particles are ones you meant to leave out."},
            {"name": "angpix_resample", "kind": "text", "flag": "--angpix_resample",
             "default": "",
             "help": "Pixel size (Å) M should refine at. Set it FINER than your "
             "extraction — that headroom is the point of M. Cannot go below the raw "
             "detector pixel size."},
            {"name": "lowpass", "kind": "text", "flag": "--lowpass",
             "default": "10",
             "help": "Initial low-pass (Å) applied to the reference, to stop the first "
             "iterations chasing noise."},
        ],
        "validate": lambda v: (
            "⚠ Give the species a name." if not str(v.get("name", "")).strip() else
            "⚠ Both UNFILTERED half-maps are required (half1 + half2)."
            if not (str(v.get("half1", "")).strip() and str(v.get("half2", "")).strip()) else
            "⚠ particles_relion must point at the Refine3D run_data.star (the refined poses)."
            if not str(v.get("particles_relion", "")).strip() else
            "⚠ Set diameter to the TRUE particle size in Å."
            if not str(v.get("diameter", "")).strip() else
            "ℹ half1/half2 must be the UNFILTERED maps (…_unfil.mrc), or the FSC is "
            "meaningless. Symmetry is baked in here — impose only what you're sure of."
            if "unfil" not in str(v.get("half1", "")) else ""),
        "output_params": ["population"],
        "docs": {
            "what": "Defines the thing being refined: the map, its mask, its symmetry and "
                    "the refined particle poses from RELION.",
            "range": "temporal_samples 1 to start; lowpass 10 Å; angpix_resample finer "
                     "than your extraction pixel size.",
            "effect": "Writes m/species/<name>_<hash>/<name>.species. NOTE the random "
                      "hash in that path — later steps need the full path, so copy it "
                      "from this job's output folder.",
            "pitfall": "Use the UNFILTERED half maps. Wrong symmetry or a bad mask here "
                       "silently limits everything downstream. And create_species ADDS "
                       "rather than replaces: re-running it (e.g. after remaking a mask) "
                       "leaves the old species in the population too, so MCore refines "
                       "both and reports two resolutions. Reset M to replace one.",
        },
        "status": lambda ps: ps.status_m_species(),
    },
    {
        # MCore — the refinement engine. Run REPEATEDLY, adding ONE parameter per
        # round and checking the resolution. Flags verified against `MCore --help`
        # (2.0.0): note the GPU flag is --devicelist, NOT --device_list as the
        # WarpTools commands use.
        "group": "11. M refinement", "id": "m_core",
        "label": "M: refine (MCore)",
        "base": "MCore",
        "params": [
            {"name": "population", "kind": "text", "flag": "--population",
             "default": "", "help": "The .population file. REQUIRED."},
            {"name": "iter", "kind": "text", "flag": "--iter",
             "default": "",
             "help": "Refinement sub-iterations (MCore's own default is 3). Set to 0 "
             "for a CHECK RUN: imports everything, refines nothing, reports the "
             "starting resolution. Blank = MCore's default."},
            {"name": "min_particles", "kind": "text", "flag": "--min_particles",
             "default": "20",
             "help": "Skip tilt series with fewer than N particles in view. THE "
             "DEFAULT IN MCORE IS 1, which lets a series with a single particle into "
             "the refinement — far too little to constrain an image-warp grid plus "
             "stage angles, and a known cause of 'IndexOutOfRangeException' during "
             "refinement. 20+ is a safe floor; raise it if crashes persist."},
            {"name": "refine_imagewarp", "kind": "text", "flag": "--refine_imagewarp",
             "default": "",
             "help": "Refine 2D image warp on an XxY grid, e.g. 6x4. The main "
             "tilt-series deformation correction and the usual first thing to enable. "
             "Blank = don't refine. A finer grid needs more particles per series."},
            {"name": "refine_particles", "kind": "check", "flag": "--refine_particles",
             "default": False, "help": "Refine per-particle poses."},
            {"name": "ctf_defocus", "kind": "check", "flag": "--ctf_defocus",
             "default": False, "help": "Refine per-particle defocus (local search)."},
            {"name": "ctf_defocusexhaustive", "kind": "check",
             "flag": "--ctf_defocusexhaustive", "default": False,
             "help": "Exhaustive defocus grid search in the FIRST sub-iteration. Only "
             "works together with ctf_defocus. Use on the first refinement only — "
             "afterwards the estimates are close and this is wasted time."},
            {"name": "refine_stageangles", "kind": "check", "flag": "--refine_stageangles",
             "default": False,
             "help": "Refine stage angles (tilt series only). Introduce AFTER image "
             "warp + particles + defocus are working — roughly round 4."},
            {"name": "refine_mag", "kind": "check", "flag": "--refine_mag",
             "default": False,
             "help": "Refine anisotropic magnification. Late-stage (round 5+)."},
            {"name": "ctf_cs", "kind": "check", "flag": "--ctf_cs",
             "default": False,
             "help": "Refine spherical aberration (also a proxy for pixel size). Late."},
            {"name": "ctf_zernike3", "kind": "check", "flag": "--ctf_zernike3",
             "default": False,
             "help": "Refine 3rd-order Zernike (beam tilt, trefoil). Fast. Late."},
            {"name": "ctf_zernike5", "kind": "check", "flag": "--ctf_zernike5",
             "default": False, "help": "Refine 5th-order Zernike. Fast. Very late."},
            {"name": "ctf_zernike2", "kind": "check", "flag": "--ctf_zernike2",
             "default": False, "help": "Refine 2nd-order Zernike. SLOW."},
            {"name": "ctf_zernike4", "kind": "check", "flag": "--ctf_zernike4",
             "default": False, "help": "Refine 4th-order Zernike. SLOW."},
            {"name": "ctf_phase", "kind": "check", "flag": "--ctf_phase",
             "default": False, "help": "Refine phase shift — PHASE PLATE data only."},
            {"name": "refine_volumewarp", "kind": "text", "flag": "--refine_volumewarp",
             "default": "",
             "help": "Volume warp on an XxYxZxT grid (tilt series only), e.g. "
             "'4x6x1x41'. Models deformation through the specimen AND time. Many "
             "parameters — only with strong data, and late."},
            {"name": "refine_tiltmovies", "kind": "check", "flag": "--refine_tiltmovies",
             "default": False,
             "help": "Refine the alignments of the tilt MOVIES (needs the raw movie "
             "frames still present)."},
            {"name": "first_iteration_fraction", "kind": "text",
             "flag": "--first_iteration_fraction", "default": "",
             "help": "Fraction of available resolution used for alignment in the first "
             "sub-iteration, rising to 1.0 by the last. Lower (e.g. 0.5) is a gentler "
             "start when the reference is poor. Blank = MCore default (1)."},
            {"name": "weight_threshold", "kind": "text", "flag": "--weight_threshold",
             "default": "",
             "help": "Refine each tilt up to the resolution where exposure weighting "
             "falls to this value. Blank = MCore default (0.05)."},
            {"name": "ctf_minresolution", "kind": "text", "flag": "--ctf_minresolution",
             "default": "",
             "help": "Only use species at least this good (Å) for CTF refinement. "
             "Blank = MCore default (8)."},
            {"name": "ctf_batch", "kind": "text", "flag": "--ctf_batch",
             "default": "",
             "help": "CTF refinement batch size. Lower = less GPU memory, higher = "
             "faster. Blank = MCore default (32). Drop this first on out-of-memory."},
            {"name": "cpu_memory", "kind": "check", "flag": "--cpu_memory",
             "default": False,
             "help": "Hold particle images in CPU RAM instead of GPU. Slower, but the "
             "way out if the GPUs run out of memory."},
            {"name": "port", "kind": "text", "flag": "--port",
             "default": "",
             "help": "REST API port (MCore's default is 14300). MCore FAILS TO START "
             "if the port is taken — 'Failed to bind to address ... address already in "
             "use' — which happens whenever a previous MCore crashed and left an "
             "orphaned process holding it. Set a different port (e.g. 14350) to dodge a "
             "stale one, or -1 to disable the API. Better: kill the orphans first "
             "(pkill -u $USER -f MCore; pkill -u $USER -f WarpWorker)."},
            {"name": "devicelist", "kind": "text", "flag": "--devicelist",
             "gpu_sep": " ", "default": "",
             "help": "GPU ids, SPACE-separated (e.g. '1 2 3'). Blank = all GPUs. NOTE "
             "MCore spells this --devicelist, unlike WarpTools' --device_list."},
            {"name": "perdevice_refine", "kind": "slider_int",
             "flag": "--perdevice_refine", "default": 1, "min": 1, "max": 8, "step": 1,
             "help": "Refinement processes per GPU. Raise for utilisation if memory "
             "allows; drop to 1 on out-of-memory or CUFFT errors."},
            {"name": "perdevice_preprocess", "kind": "text",
             "flag": "--perdevice_preprocess", "default": "",
             "help": "Processes per GPU for map PRE-processing. Blank = same as "
             "perdevice_refine."},
            {"name": "perdevice_postprocess", "kind": "text",
             "flag": "--perdevice_postprocess", "default": "",
             "help": "Processes per GPU for map POST-processing. Blank = same as "
             "perdevice_refine."},
        ],
        "validate": lambda v: (
            "ℹ CHECK RUN (--iter 0): nothing is refined. Confirm the reported "
            "resolution looks sane, then clear 'iter' and enable ONE thing at a time."
            if str(v.get("iter", "")).strip() == "0" else
            "⚠ ctf_defocusexhaustive only works together with ctf_defocus — tick that too."
            if v.get("ctf_defocusexhaustive") and not v.get("ctf_defocus") else
            "⚠ min_particles is 1 (or blank): a series with a single particle cannot "
            "constrain an image-warp grid, and M crashes with IndexOutOfRangeException. "
            "Set it to 20 or more."
            if (str(v.get("refine_imagewarp", "")).strip()
                and str(v.get("min_particles", "")).strip() in ("", "0", "1")) else
            "⚠ Nothing to refine — enable at least one refine_*/ctf_* option, or set "
            "iter=0 for a check run."
            if not any(v.get(k) for k in (
                "refine_particles", "refine_stageangles", "refine_mag", "ctf_defocus",
                "ctf_cs", "ctf_zernike3", "ctf_zernike5", "ctf_zernike2",
                "ctf_zernike4", "ctf_phase", "refine_tiltmovies"))
            and not str(v.get("refine_imagewarp", "")).strip()
            and not str(v.get("refine_volumewarp", "")).strip() else
            "⚠ Several parameters at once. M's guidance is ONE AT A TIME, checking the "
            "resolution each round — stage angles ~round 4, magnification/Cs/Zernike "
            "~round 5+. Enabling them early is a common cause of both overfitting and "
            "refinement crashes."
            if sum(bool(v.get(k)) for k in (
                "refine_particles", "refine_stageangles", "refine_mag", "ctf_defocus",
                "ctf_cs", "ctf_zernike3", "ctf_zernike5", "ctf_zernike2",
                "ctf_zernike4")) >= 4 else ""),
        "output_params": ["population"],
        "docs": {
            "what": "The refinement engine. Run it repeatedly, adding one parameter per "
                    "round and watching the resolution.",
            "range": "Round 1: imagewarp 6x4 + particles + ctf_defocus + "
                     "ctf_defocusexhaustive. Round 2: same without exhaustive. "
                     "Round 3+: add stageangles, then mag/cs/zernike3.",
            "effect": "Refines tilt-series geometry, poses and CTF jointly against the "
                      "raw data; writes an updated map + FSC into the species folder.",
            "pitfall": "min_particles defaults to 1 in MCore — sparse series then crash "
                       "the refinement (IndexOutOfRangeException). Add parameters ONE "
                       "at a time: the resolution number can improve while the map gets "
                       "worse. Fork this card per round so you can compare and roll back.",
        },
        "status": lambda ps: ps.status_m_refined(),
    },
    {
        "group": "11. M refinement", "id": "m_estimate_weights",
        "label": "M: estimate weights",
        "base": "EstimateWeights",
        "params": [
            {"name": "population", "kind": "text", "flag": "--population",
             "default": "", "help": "The .population file. REQUIRED."},
            {"name": "source", "kind": "text", "flag": "--source",
             "default": "", "help": "Data-source name (from 'create data source')."},
            {"name": "resolve", "kind": "choice", "flag": None,
             "choices": [("Per tilt series (--resolve_items)", "--resolve_items"),
                         ("Per tilt, averaged over series (--resolve_frames)",
                          "--resolve_frames")],
             "default": "--resolve_items",
             "help": "Which exposure weights to estimate. Do PER-SERIES first, run MCore "
             "once, then per-tilt. They are separate passes, not alternatives."},
        ],
        "validate": lambda v: ("⚠ Give the source name."
                               if not str(v.get("source", "")).strip() else
                               "ℹ Follow this with a plain MCore run (no refine flags) so "
                               "the new weights are actually applied."),
        "output_params": ["population"],
        "docs": {
            "what": "Estimates exposure/dose weights, either per tilt series or per tilt "
                    "averaged across series.",
            "range": "Run --resolve_items first, then --resolve_frames.",
            "effect": "Writes weights into the population; the NEXT MCore run uses them.",
            "pitfall": "This does not refine anything on its own — always follow it with "
                       "an MCore run or nothing changes.",
        },
        "status": None,
    },
    {
        "group": "11. M refinement", "id": "m_resample_trajectories",
        "label": "M: resample trajectories",
        "base": "MTools resample_trajectories",
        "params": [
            {"name": "population", "kind": "text", "flag": "--population",
             "default": "", "help": "The .population file. REQUIRED."},
            {"name": "species", "kind": "text", "flag": "--species",
             "default": "",
             "help": "FULL path to the .species file, including its random hash — e.g. "
             "m/species/spike_797f75c2/spike.species. Find it with:  ls m/species/*/*.species"},
            {"name": "samples", "kind": "text", "flag": "--samples",
             "default": "2",
             "help": "Temporal pose samples per tilt series. Go 1 → 2 only once the map "
             "is good; more samples need more signal per particle."},
        ],
        "validate": lambda v: (
            "⚠ Give the FULL .species path (it contains a random hash). "
            "Find it with:  ls m/species/*/*.species"
            if not str(v.get("species", "")).strip() else ""),
        "output_params": ["population"],
        "docs": {
            "what": "Increases how finely M models particle pose CHANGE through the tilt "
                    "series (beam-induced motion).",
            "range": "samples 1 → 2. Only worth it late, on good data.",
            "effect": "Re-samples the trajectories; the next MCore run refines them.",
            "pitfall": "More samples = more parameters per particle = easier overfitting. "
                       "Raise only when the resolution has plateaued.",
        },
        "status": None,
    },
]

def _h(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_docs_html(doc):
    """Rich left-panel HTML for a stage doc dict (from tomogration_docs.json):
    title → what → why → parameter table → pitfalls (red) → good-output QC →
    refs as clickable links."""
    h = [f"<h2 style='color:#eaeaea;margin:0 0 8px 0;'>{_h(doc.get('title', ''))}</h2>"]
    for key, head, col in (("what", "WHAT", "#9ec5ff"), ("why", "WHY", "#9ec5ff")):
        if doc.get(key):
            h.append(f"<p style='margin:6px 0;'><b style='color:{col};'>{head}</b><br>"
                     f"{_h(doc[key])}</p>")
    params = doc.get("params") or []
    if params:
        h.append("<p style='margin:8px 0 2px;'><b style='color:#9ec5ff;'>PARAMETERS</b></p>")
        h.append("<table cellspacing='0' cellpadding='3' width='100%' "
                 "style='border-collapse:collapse;font-size:11px;'>")
        h.append("<tr style='color:#9a9a9a;'><th align='left'>name</th>"
                 "<th align='left'>flag</th><th align='left'>default</th>"
                 "<th align='left'>range</th><th align='left'>effect</th></tr>")
        for i, p in enumerate(params):
            bg = "#202020" if i % 2 else "#262626"
            h.append(
                f"<tr style='background:{bg};'>"
                f"<td valign='top'><code style='color:#cfe;'>{_h(p.get('name', ''))}</code></td>"
                f"<td valign='top'><code style='color:#cc9;'>{_h(p.get('flag', ''))}</code></td>"
                f"<td valign='top'>{_h(p.get('default', ''))}</td>"
                f"<td valign='top' style='color:#9a9a9a;'>{_h(p.get('range', ''))}</td>"
                f"<td valign='top'>{_h(p.get('effect', ''))}</td></tr>")
        h.append("</table>")
    if doc.get("pitfalls"):
        h.append(f"<p style='margin:8px 0;'><b style='color:#e24b4a;'>PITFALLS</b><br>"
                 f"<span style='color:#f0a0a0;'>{_h(doc['pitfalls'])}</span></p>")
    if doc.get("qc"):
        h.append(f"<p style='margin:8px 0;'><b style='color:#27ae60;'>GOOD OUTPUT "
                 f"LOOKS LIKE</b><br><span style='color:#b8e0c0;'>{_h(doc['qc'])}"
                 f"</span></p>")
    refs = doc.get("refs") or []
    if refs:
        h.append("<p style='margin:8px 0 2px;'><b style='color:#9ec5ff;'>REFS</b></p>"
                 "<ul style='margin:2px 0 2px 16px;padding:0;'>")
        for r in refs:
            h.append(f"<li><a href='{_h(r)}' style='color:#7fb4ff;'>{_h(r)}</a></li>")
        h.append("</ul>")
    return "".join(h)


def render_inline_docs_html(spec):
    """Fallback left-panel HTML for stages not covered by tomogration_docs.json:
    render the stage's own inline docs dict (what/range/effect/pitfall)."""
    d = spec.get("docs", {})
    h = [f"<h2 style='color:#eaeaea;margin:0 0 8px 0;'>{_h(spec['label'])}</h2>"]
    for key, head, col in (("what", "WHAT", "#9ec5ff"), ("range", "RANGE", "#9a9a9a"),
                           ("effect", "EFFECT", "#9a9a9a")):
        if d.get(key) and d.get(key) != "n/a":
            h.append(f"<p style='margin:6px 0;'><b style='color:{col};'>{head}</b><br>"
                     f"{_h(d[key])}</p>")
    if d.get("pitfall"):
        h.append(f"<p style='margin:8px 0;'><b style='color:#e24b4a;'>PITFALL</b><br>"
                 f"<span style='color:#f0a0a0;'>{_h(d['pitfall'])}</span></p>")
    return "".join(h)


def stage_defaults(spec):
    """{param_name: default} for a stage — the values the widgets start at."""
    return {p["name"]: p.get("default") for p in spec.get("params", [])}


def _norm_gpu(s, sep):
    """Normalise a GPU-id list to the separator the target tool wants, so the user
    can type either '0 1 2 3' or '0,1,2,3' anywhere and it comes out correct
    (WarpTools/AreTomo need spaces; RELION/miss-alignment need commas). No-op when
    the param has no gpu_sep hint."""
    if not sep or not s:
        return s
    return sep.join(t for t in re.split(r"[ ,]+", s.strip()) if t)


# Warp 2.0 ships these alongside WarpTools in the same conda env; they all need the
# user's module-load + conda-activate prefix in front of them.
WARP_SUITE = ("WarpTools", "MTools", "MCore", "EstimateWeights",
              "Noise2Map", "Noise2Mic", "Frankenmap")


def _subst_siblings(text, values):
    """Resolve {field} references against the OTHER fields of the same form.

    Stage defaults use {placeholders} so one field can follow another — e.g. the M
    data source's population defaults to "m/{name}.population". Without this, the
    placeholder went to the tool VERBATIM: MTools was handed the literal path
    "m/{name}.population" and cheerfully created a brand-new population file called
    "{name}.population", so the data source was registered against the wrong project.

    {jobid} is deliberately left alone — it is resolved later, per job instance, by
    build_job_command.
    """
    def rep(m):
        key = m.group(1)
        if key == "jobid" or key not in values:
            return m.group(0)
        return str(values.get(key) or "")
    return re.sub(r"\{(\w+)\}", rep, text)


def build_command(spec, values, warp_cmd=None, group_inputs=None):
    """Pure command assembler (no Qt) — the single source of truth the editable
    command box is seeded from. env/env_int params become a VAR=value prefix;
    checks emit their flag when truthy; flagged params emit 'flag value';
    flag=None params emit their value positionally, in declaration order.

    base = (env prefix) + spec.base + (flags/positionals). If warp_cmd is given,
    a leading 'WarpTools' in the base is replaced by it (so the user's module
    load / conda activate / path runs before every WarpTools subcommand)."""
    env_parts, body_parts, modules = [], [], []
    for p in spec.get("params", []):
        # A param can declare itself irrelevant for the current settings (e.g. the
        # MODE A/B fields once MODE C is on). Emitting args the tool ignores makes the
        # command box lie about what will happen, so drop them entirely.
        skip = p.get("skip_if")
        if skip is not None:
            try:
                if skip(values):
                    continue
            except Exception:
                pass
        v = values.get(p["name"])
        kind = p["kind"]
        if kind == "module":
            # Every tool on this cluster lives behind an lmod module, so a bare
            # `relion_mask_create` is "command not found". A module param becomes a
            # `module load <x> &&` PREFIX and is never passed as an argument.
            mv = str(v or "").strip()
            if mv:
                modules.append(mv)
            continue
        if kind in ("env", "env_int"):
            s = _subst_siblings(_norm_gpu(str(v).strip(), p.get("gpu_sep")), values)
            if s != "":
                env_parts.append(f"{p['flag']}='{s}'" if any(c in s for c in " \t")
                                 else f"{p['flag']}={s}")
        elif kind == "check":
            if v:
                body_parts.append(p["flag"])
        else:
            s = _subst_siblings(_norm_gpu(str(v).strip(), p.get("gpu_sep")), values)
            if s != "":
                body_parts.append(f"{p['flag']} {s}" if p.get("flag") else s)
    # Active tilt-series group: restrict this step to the subset via --input_data
    # (unless the user already typed an --input_data into the params).
    scope = spec.get("group_scope")
    if (group_inputs and scope and group_inputs.get(scope)
            and not any(part.startswith("--input_data") for part in body_parts)):
        body_parts.append(f"--input_data {group_inputs[scope]}")
    # Always-on VAR=value the stage bakes in (e.g. MA_MODE=infer) — kept out of the
    # form so there's no confusing editable field for a value that must not change.
    for k, v in (spec.get("fixed_env") or {}).items():
        env_parts.insert(0, f"{k}={v}")
    base = spec.get("base", "")
    # The Warp 2.0 suite ships several executables inside the SAME conda env, and
    # warp_cmd is the user's full launcher ("module load … && conda activate warp &&
    # WarpTools"). For the other tools we reuse everything up to that trailing
    # "WarpTools" and append their own name — otherwise MTools/MCore/EstimateWeights
    # run outside the env and die with "command not found".
    if warp_cmd and base:
        head = base.split(None, 1)[0]
        if head == "WarpTools":
            base = warp_cmd + base[len("WarpTools"):]
        elif head in WARP_SUITE:
            wc = warp_cmd.rstrip()
            prefix = wc[:-len("WarpTools")].rstrip() if wc.endswith("WarpTools") else ""
            if prefix:
                base = prefix + " " + base
    if modules and base:
        base = " && ".join(f"module load {m}" for m in modules) + " && " + base
    segs = []
    if env_parts:
        segs.append(" ".join(env_parts))
    if base:
        segs.append(base)
    segs.extend(body_parts)
    cmd = " ".join(segs)
    # Vars that must be EXPORTED into the shell before the tool runs. A bare
    # "VAR=val cmd" prefix only applies to the first word, which for WarpTools is
    # `module` (base = "module load … && conda activate … && WarpTools …"), so the
    # var would never reach WarpTools. `export VAR=val && …` puts it in the
    # environment for the whole chain. (ts_reconstruct needs WARP_FORCE_MRC_FLOAT32=1
    # so its tomograms are float32 and IMOD/3dmod can read them.)
    exports = spec.get("env_export")
    if exports:
        ex = " ".join(f"{k}={v}" for k, v in exports.items())
        cmd = f"export {ex} && {cmd}"
    return cmd


# Primary output directory per stage (relative to the project root) for the
# per-step "open output" button + iteration dropdown. "aretomo" = the versioned
# AreTomo folders. Stages absent here (tool/action steps) get no output controls.
STAGE_OUTPUTS = {
    "rename": ".", "imod_warp_key": ".", "remake_mdocs": "mdocs",
    "gain_convert": "gains", "gain_reciprocal": "gains",
    "create_settings_fs": "warp_frameseries", "fs_motion_and_ctf": "warp_frameseries",
    "create_settings_ts": "warp_tiltseries", "ts_import": "tomostar",
    "ts_stack": "warp_tiltseries/tiltstack", "aretomo": "aretomo",
    "miss_align": "warp_tiltseries", "miss_align_infer": "warp_tiltseries",
    "ts_import_alignments": "warp_tiltseries", "ts_ctf": "warp_tiltseries",
    "ts_reconstruct": "warp_tiltseries/reconstruction",
    "ts_template_match": "warp_tiltseries/matching",
    "threshold_picks": "warp_tiltseries/matching", "ts_export_particles": "relion4",
    "relion4_convert": "relion4", "relion4_class3d": "relion4",
    "relion4_select_picks": "warp_tiltseries/matching_good",
    "relion4_check_star": "relion4",
    "relion4_verify_reextract": "relion4",
    "relion4_result": "relion4",
    "m_reset": "m",
    "m_create_population": "m", "m_create_source": "m", "m_mask_create": "m",
    "m_create_species": "m", "m_core": "m", "m_estimate_weights": "m",
    "m_resample_trajectories": "m", "m_reset": ".", "m_kill_orphans": ".",
    "relion4_to_warp": "warp_tiltseries/matching_reextract",
}

# Which mockup column each stage group belongs to (the three job-list panels).
COLUMN_OF_GROUP = {
    "1. Data prep": "curation", "2. Gain": "curation",
    "3. Frameseries": "stackprep", "4. Tilt series": "stackprep",
    "5. Alignment": "alignrecon", "6. CTF": "alignrecon",
    "7. Reconstruct": "alignrecon", "8. Pick": "alignrecon",
    "9. Export": "alignrecon", "10. RELION 4": "alignrecon",
    "11. M refinement": "alignrecon",
}
COLUMN_TITLES = {
    "curation": "Tilt curation", "stackprep": "Stack preparation",
    "alignrecon": "Alignment & Reconstruction",
}

# Directory-overview schematic: the key project dirs to draw, in pipeline order.
KEY_DIRS = [
    ("frames", "frames/"), ("mdocs", "mdocs/"), ("gains", "gains/"),
    ("Thumbnails", "Thumbnails/"),
    ("warp_frameseries", "warp_frameseries/"),
    ("tomostar", "tomostar/"),
    ("warp_tiltseries", "warp_tiltseries/"),
    ("warp_tiltseries/tiltstack", "…/tiltstack/"),
    ("aretomo_output", "aretomo_output*/"),
    ("warp_tiltseries/reconstruction", "…/reconstruction/"),
    ("warp_tiltseries/matching", "…/matching/"),
    ("relion4/warp", "relion4/ (RELION project)"),
]

# Per-stage (inputs, outputs) as dir paths relative to the project root — drives
# the directory-overview highlighting (blue=input, green=output, grey=other).
STAGE_IO = {
    "rename":               (["."], ["mdocs", "frames"]),
    "imod_warp_key":        (["mdocs"], ["."]),
    "inspect_select":       (["Thumbnails", "mdocs"], ["mdocs"]),
    "remake_mdocs":         (["mdocs"], ["mdocs"]),
    "gain_convert":         (["gains"], ["gains"]),
    "gain_reciprocal":      (["gains"], ["gains"]),
    "create_settings_fs":   (["frames", "gains"], ["warp_frameseries"]),
    "fs_motion_and_ctf":    (["frames", "warp_frameseries"], ["warp_frameseries"]),
    "create_settings_ts":   (["mdocs"], ["warp_tiltseries"]),
    "ts_import":            (["mdocs", "warp_frameseries"], ["tomostar", "warp_tiltseries"]),
    "ts_stack":             (["warp_tiltseries", "tomostar"], ["warp_tiltseries/tiltstack"]),
    "aretomo":              (["warp_tiltseries/tiltstack"], ["aretomo_output"]),
    "miss_align":           (["warp_tiltseries", "warp_tiltseries/tiltstack"], ["warp_tiltseries"]),
    "miss_align_infer":     (["warp_tiltseries", "warp_tiltseries/tiltstack"], ["warp_tiltseries"]),
    "ts_import_alignments": (["aretomo_output", "warp_tiltseries"], ["warp_tiltseries"]),
    "sync_selection":       (["warp_tiltseries"], ["warp_tiltseries"]),
    "ts_defocus_hand":      (["warp_tiltseries"], ["warp_tiltseries"]),
    "ts_ctf":               (["warp_tiltseries", "warp_tiltseries/tiltstack"], ["warp_tiltseries"]),
    "ts_reconstruct":       (["warp_tiltseries"], ["warp_tiltseries/reconstruction"]),
    "ts_template_match":    (["warp_tiltseries/reconstruction"], ["warp_tiltseries/matching"]),
    "threshold_picks":      (["warp_tiltseries/matching"], ["warp_tiltseries/matching"]),
    "ts_export_particles":  (["warp_tiltseries", "warp_tiltseries/matching"], ["relion4/warp"]),
    "relion4_convert":      (["relion4/warp"], ["relion4/warp"]),
    "relion4_class3d":      (["relion4/warp"], ["relion4/warp"]),
    "relion4_result":       (["relion4/warp"], ["relion4/warp"]),
    "relion4_select_picks": (["relion4/warp", "warp_tiltseries/matching"],
                             ["warp_tiltseries/matching_good"]),
    "relion4_to_warp":      (["relion4/warp", "warp_tiltseries/matching"],
                             ["warp_tiltseries/matching_reextract"]),
    "m_reset":              (["m", "warp_tiltseries"], ["m"]),
    "m_create_population":  ([], ["m"]),
    "m_create_source":      (["warp_tiltseries", "m"], ["m"]),
    "m_mask_create":        (["relion4/warp"], ["m"]),
    "m_create_species":     (["relion4/warp", "m"], ["m"]),
    "m_core":               (["m", "warp_tiltseries"], ["m"]),
    "m_estimate_weights":   (["m"], ["m"]),
    "m_resample_trajectories": (["m"], ["m"]),
    "m_reset":              (["m"], ["."]),
}

# Generic filename patterns each key directory is searched for — shown in the
# Job details INPUTS/OUTPUTS lists so the user knows WHICH files a step consumes
# or produces in each folder (not just the folder name). Keyed by the same rel
# dirs used in STAGE_IO.
DIR_FILE_HINTS = {
    "frames": "*.eer  (raw movies)",
    "mdocs": "*.mdoc  (per-series)",
    "gains": "*.gain / gain reference",
    "Thumbnails": "*.mrc  (per-series montage)",
    "warp_frameseries": "*.xml  (per-movie metadata)",
    "tomostar": "*.tomostar  (per-series)",
    "warp_tiltseries": "*.xml  (per-series metadata)",
    "warp_tiltseries/tiltstack": "*.st + *.rawtlt  (aligned stacks)",
    "aretomo_output": "Imod/*.xf  (alignments)",
    "warp_tiltseries/reconstruction": "*_<angpix>Apx.mrc  (tomograms)",
    "warp_tiltseries/matching": "*_<suffix>.star  (pick lists)",
    "warp_tiltseries/matching_good": "*_good.star  (class-filtered picks)",
    "warp_tiltseries/matching_reextract": "*_reextract.star  (re-extraction picks)",
    "relion4/warp": "*.star + subtomo/*.mrc",
    "m": "*.population + species/<name>_<hash>/  (M project)",
    ".": "(project root)",
}

# Retired: the app used to archive ts_reconstruct/ts_template_match outputs to
# <dir>.bak_<ts> on re-run so nothing was lost. That surprised users (their
# terminal-made picks got moved aside) and the job model supersedes it — each job
# writes its OWN jobs/J### dir, so variants coexist without shuffling shared dirs.
# Kept empty (not deleted) so _record_history_start stays a no-op archiver.
ARCHIVE_ON_RERUN = set()

# Back-half stages that produce a self-contained product and are worth running as
# JOB instances (own dir, forkable, interconnectable) rather than overwriting the
# shared trunk. ▶ Run offers to build these as a job. NOTE ts_template_match is NOT
# here: it reads the SHARED reconstructions and its --override_suffix already keeps
# pick sets distinct, so it runs on the trunk and is surfaced as a card by the
# discover/adopt flow (a per-job dir would hide the reconstructions from it).
JOB_STAGES = {"ts_ctf", "ts_reconstruct", "threshold_picks",
              "ts_export_particles", "relion4_convert", "relion4_class3d"}

# WarpTools stages that must run on the TRUNK (no --output_processing): they read
# shared products (reconstructions) that only exist in the trunk processing dir, and
# distinguish their own output by suffix. Wiring them to a job dir makes WarpTools
# look for those shared inputs in the empty job dir ("reconstruction not found").
TRUNK_STAGES = {"ts_template_match"}

# File-open routing for the Processing-History detail view.
THREEDMOD_EXTS = {".mrc", ".mrcs", ".st", ".ali", ".rec", ".preali", ".mod", ".map"}
TEXT_EXTS = {".txt", ".star", ".xml", ".mdoc", ".settings", ".yaml", ".yml",
             ".log", ".csv", ".com", ".json", ".tlt", ".rawtlt", ".xf", ".aln"}


# ===========================================================================
# JOB MODEL (Phase 1) — a CryoSPARC-style DAG of job INSTANCES.
#
# The three-column view treats each STAGE as a singleton whose result lives at
# one conventional path (STAGE_OUTPUTS). The card view instead models each RUN
# as a job instance with its OWN processing directory, wired to upstream jobs by
# named input slots. This is possible because every WarpTools command accepts
# --input_processing / --output_processing (they live in BaseCommand): a job
# READS its parent's processing dir and WRITES its own, sharing no mutable state.
# Verified on the VM 2026-07-10 — a branched `ts_ctf --output_processing
# warp_tiltseries_b` left the trunk XML byte-identical (md5 OK), kept all upstream
# alignment metadata (size 430,737 -> 431,020, not a stripped rewrite), and a
# `ts_reconstruct --input_processing warp_tiltseries_b` read it back and
# reconstructed. Non-WarpTools wrappers (aretomo, miss_align, relion4_*) take
# explicit in/out dirs instead, so they get no processing-dir flags here.
#
# Store: .tomogration_jobs.json in the project root:
#     {"seq": <int>, "jobs": {"J1": {..job..}, "J2": {...}}}
# This whole layer is PURE (stdlib only, no Qt) so it unit-tests off the VM; the
# GUI wiring (dispatch, the canvas) sits on top of it in the Tomogration class.
# ===========================================================================
