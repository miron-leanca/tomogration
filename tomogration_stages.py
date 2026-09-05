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
    """Warnings for ts_export_particles. EVERY applicable one is returned.

    Two input routes, with different traps:

    RELION-star route (input_star set) — re-extracting a RELION selection.
      Warp reads the star's rlnCoordinateX/Y/Z, subtracts the refined
      rlnOriginX/Y/ZAngst itself (divided by the star's own rlnImagePixelSize)
      and then scales by coords_angpix. So coords_angpix is REQUIRED and must
      be the star's pixel size; the pick-star fields and normalized_coords are
      dropped from the command. (Verified against WarpTools source, 2026-09-02;
      this is the route that produced EML46's good bin1 re-extraction.)
    Pick-star route (input_directory + input_pattern) — a first extraction.
      1. COORD MODE: normalized_coords (0-1 fractions, what template matching
         and the crYOLO converter write) XOR coords_angpix (pixels at a stated
         Å/px). Exactly one, or Warp refuses / extracts from the wrong places.
      2. COORD SCALE vs the filename's <apx>Apx tag when the picks are pixels.
    Both routes:
      3. BOX vs DIAMETER — diameter is where the particle is assumed to end,
         everything outside it is the SOLVENT used for background normalisation.
      4. output_star must sit inside output_processing (RELION launch root).
    """
    out = []
    norm = bool(v.get("normalized_coords"))
    capx = str(v.get("coords_angpix", "")).strip()
    istar = str(v.get("input_star", "")).strip()
    idir = str(v.get("input_directory", "")).strip()

    if istar:
        if idir:
            out.append("ℹ Using the RELION star; the pick-star folder and pattern "
                       "are IGNORED and will not be put on the command line.")
        if not capx:
            out.append(
                "⚠ coords_angpix is REQUIRED with a RELION star: Warp multiplies the "
                "star's rlnCoordinateX/Y/Z by it. Set it to the star's optics "
                "rlnImagePixelSize — 6.28 for particles first extracted at bin4, "
                "3.14 at bin2. Building this card downstream of the Subset-selection "
                "card reads it from the star for you.")
        if norm:
            out.append("ℹ '0-1 fractions' is IGNORED with a RELION star and will not "
                       "be sent: a RELION star holds PIXEL coordinates, and declaring "
                       "them fractions multiplies every one by the tomogram width.")
    elif not idir:
        out.append("⚠ NO INPUT set. Give Warp the RELION star to re-extract, or the "
                   "pick-star folder + pattern.")
    else:
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
                    f"(*{tag.group(1)}Apx*) but coords_angpix is {got:g}. Every "
                    f"particle would be extracted {got / want:.3g}× too far from the "
                    f"origin, with no error — set coords_angpix to {want:g}.")
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
             "title": "Raw data folder",
             "default": ".",
             "help": "Folder with the raw Tomo5 .eer + .mdoc (+ .mrc) TOGETHER, "
                     "relative to the project root. '.' = the project root itself "
                     "(use this when the root IS your acquisition folder). Run this "
                     "BEFORE Sort files — rename needs .eer and .mdoc in one folder."},
            {"name": "rootname", "kind": "text", "flag": None,
             "title": "Series prefix",
             "default": "Position", "help": "Prefix for renamed files (e.g. Position, TS)."},
            {"name": "start_number", "kind": "text", "flag": None,
             "title": "First number",
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
             "title": "Reference mdoc",
             "default": "mdocs/Position001.mdoc",
             "help": "A COMPLETE reference mdoc (all tilts present)."},
            {"name": "output_key", "kind": "text", "flag": None,
             "title": "Output key file",
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
             "title": "Mdoc folder",
             "default": "mdocs", "help": "Dir with <root>NNN.mdoc files."},
            {"name": "exclusion_list", "kind": "text", "flag": None,
             "title": "Exclusion list",
             "default": "exclusion_list.txt",
             "help": "Manual tilt exclusions (IMOD order), above the auto header."},
            {"name": "conv_key", "kind": "text", "flag": None,
             "title": "IMOD→Warp key",
             "default": "new_imod_conv_key.txt", "help": "IMOD->acq key file."},
            {"name": "rootname", "kind": "text", "flag": None,
             "title": "Series prefix",
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
        "output_params": ["out_mrc"],
        "params": [
            {"name": "in_gain", "kind": "text", "flag": None,
             "title": "Input gain (.gain)",
             "default": "gains/original.gain", "help": "Input .gain reference."},
            {"name": "out_mrc", "kind": "text", "flag": None,
             "title": "Output gain (.mrc)",
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
        "output_params": ["out_reciprocal"],
        "params": [
            {"name": "in_mrc", "kind": "text", "flag": None,
             "title": "Input gain (.mrc)",
             "default": "gains/original_gain.mrc", "help": "Input .mrc gain."},
            {"name": "out_reciprocal", "kind": "text", "flag": None,
             "title": "Output reciprocal gain",
             "default": "gains/gain_reciprocal.mrc", "help": "Output reciprocal gain."},
            {"name": "reciprocal", "kind": "check", "flag": "--process math.reciprocal",
             "title": "Take reciprocal",
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
             "title": "Raw data folder",
             "default": "frames", "help": "Raw .eer folder."},
            {"name": "folder_processing", "kind": "text", "flag": "--folder_processing",
             "title": "Processing folder",
             "default": "warp_frameseries", "help": "Processing output folder."},
            {"name": "output", "kind": "text", "flag": "--output",
             "title": "Settings file (output)",
             "default": "warp_frameseries.settings", "help": "Settings file to write."},
            {"name": "extension", "kind": "text", "flag": "--extension",
             "title": "File pattern",
             "default": "*.eer", "help": "Input file glob."},
            {"name": "angpix", "kind": "text", "flag": "--angpix",
             "title": "Pixel size (Å/px)",
             "default": "1.57", "help": "Pixel size (Å/px). This dataset: 1.57."},
            {"name": "gain_path", "kind": "text", "flag": "--gain_path",
             "title": "Gain reference",
             "default": "gains/gain_reciprocal.mrc",
             "help": "RECIPROCAL gain (Linux WarpTools)."},
            {"name": "exposure", "kind": "text", "flag": "--exposure",
             "title": "Dose per tilt (e⁻/Å²)",
             "default": "3.5", "help": "Dose per TILT (e/Å², not per frame)."},
            {"name": "eer_ngroups", "kind": "text", "flag": "--eer_ngroups",
             "title": "EER frame groups",
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
             "title": "Settings file",
             "default": "warp_frameseries.settings", "help": "fs .settings file."},
            {"name": "m_grid", "kind": "text", "flag": "--m_grid",
             "title": "Motion grid",
             "default": "1x1x3", "help": "Motion grid XxYxT; temporal ≈ frame count."},
            {"name": "c_grid", "kind": "text", "flag": "--c_grid",
             "title": "CTF grid",
             "default": "2x2x1", "help": "CTF grid XxYxT (not slider-able)."},
            {"name": "m_range_min", "kind": "text", "flag": "--m_range_min",
             "title": "Motion low-res limit (Å)",
             "default": "500", "help": "Motion fit low-res bound (Å)."},
            {"name": "m_range_max", "kind": "text", "flag": "--m_range_max",
             "title": "Motion high-res limit (Å)",
             "default": "10", "help": "Motion fit high-res bound (Å)."},
            {"name": "m_bfac", "kind": "slider_int", "flag": "--m_bfac",
             "title": "Motion B-factor",
             "default": -500, "min": -1000, "max": 0, "step": 50,
             "help": "Motion B-factor; more negative = stronger low-pass."},
            {"name": "c_range_max", "kind": "text", "flag": "--c_range_max",
             "title": "CTF max resolution (Å)",
             "default": "7", "help": "CTF fit max resolution (Å)."},
            {"name": "c_defocus_max", "kind": "text", "flag": "--c_defocus_max",
             "title": "Max defocus (µm)",
             "default": "8", "help": "Max defocus to search (µm)."},
            {"name": "out_averages", "kind": "check", "flag": "--out_averages",
             "title": "Write averages",
             "default": True, "help": "Write aligned averages. REQUIRED — ts_import "
             "needs them ('no aligned average result' error if off)."},
            {"name": "out_average_halves", "kind": "check", "flag": "--out_average_halves",
             "title": "Write half-averages",
             "default": True, "help": "Write odd/even half-averages (for Noise2Noise "
             "denoising)."},
            {"name": "device_list", "kind": "text", "flag": "--device_list", "gpu_sep": " ",
             "title": "GPUs",
             "default": "0", "help": "GPU id(s), e.g. 0 or '0 1'."},
            {"name": "perdevice", "kind": "slider_int", "flag": "--perdevice",
             "title": "Workers per GPU",
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
             "title": "Tomostar folder",
             "default": "tomostar", "help": "tomostar folder."},
            {"name": "folder_processing", "kind": "text", "flag": "--folder_processing",
             "title": "Processing folder",
             "default": "warp_tiltseries", "help": "Processing output folder."},
            {"name": "output", "kind": "text", "flag": "--output",
             "title": "Settings file (output)",
             "default": "warp_tiltseries.settings", "help": "Settings file to write."},
            {"name": "extension", "kind": "text", "flag": "--extension",
             "title": "File pattern",
             "default": "*.tomostar", "help": "Input glob."},
            {"name": "angpix", "kind": "text", "flag": "--angpix",
             "title": "Pixel size (Å/px)",
             "default": "1.57", "help": "Pixel size (Å/px)."},
            {"name": "gain_path", "kind": "text", "flag": "--gain_path",
             "title": "Gain reference",
             "default": "gains/gain_reciprocal.mrc", "help": "Reciprocal gain."},
            {"name": "exposure", "kind": "text", "flag": "--exposure",
             "title": "Dose per tilt (e⁻/Å²)",
             "default": "3.5", "help": "Dose per tilt (e/Å²)."},
            {"name": "tomo_dimensions", "kind": "text", "flag": "--tomo_dimensions",
             "title": "Tomogram dimensions",
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
        "output_params": ["output"],
        "params": [
            {"name": "mdocs", "kind": "text", "flag": "--mdocs",
             "title": "Mdoc folder",
             "default": "mdocs", "help": "Mdocs folder."},
            {"name": "frameseries", "kind": "text", "flag": "--frameseries",
             "title": "Frameseries folder",
             "default": "warp_frameseries", "help": "Frameseries processing folder."},
            {"name": "tilt_exposure", "kind": "text", "flag": "--tilt_exposure",
             "title": "Dose per tilt (e⁻/Å²)",
             "default": "3.5", "help": "Dose per tilt (e/Å²)."},
            {"name": "min_intensity", "kind": "text", "flag": "--min_intensity",
             "title": "Min intensity",
             "default": "0", "help": "Min intensity filter."},
            {"name": "dont_invert", "kind": "check", "flag": "--dont_invert",
             "title": "Keep tilt polarity",
             "default": True, "help": "Keep tilt polarity as-is (dataset-specific)."},
            {"name": "output", "kind": "text", "flag": "--output",
             "title": "Tomostar folder (output)",
             "default": "tomostar", "help": "tomostar output folder."},
            {"name": "override_axis", "kind": "text", "flag": "--override_axis",
             "title": "Tilt-axis angle (°)",
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
             "title": "Settings file",
             "default": "warp_tiltseries.settings", "help": "ts .settings file."},
            {"name": "angpix", "kind": "text", "flag": "--angpix",
             "title": "Output pixel size (Å/px)",
             "default": "", "help": "Output pixel size (Å/px). Blank = native."},
            {"name": "device_list", "kind": "text", "flag": "--device_list", "gpu_sep": " ",
             "title": "GPUs",
             "default": "0", "help": "GPU id(s)."},
            {"name": "perdevice", "kind": "slider_int", "flag": "--perdevice",
             "title": "Workers per GPU",
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
        "output_params": ["output_dir"],
        # The wrapper exits 2 when SOME series failed and the rest
        # aligned. That is a result with casualties, not a failure: 76 of
        # 81 usable alignments were marking the card red and dequeuing
        # everything behind it.
        "partial_exit": 2,
        "aretomo": True,
        "params": [
            {"name": "script", "kind": "text", "flag": None,
             "title": "Wrapper script",
             "default": _pkg_script("ml_aretomo2_warp_auto.sh"),
             "help": "AreTomo2 wrapper script (shipped with the app)."},
            {"name": "input_dir", "kind": "text", "flag": None,
             "title": "Tilt-stack folder",
             "default": "warp_tiltseries/tiltstack", "help": "Folder of <Pos>/<Pos>.st."},
            {"name": "output_dir", "kind": "text", "flag": None,
             "title": "Output folder",
             "default": "jobs/{jobid}",
             "help": "This job's own folder. Alignments land in <output>/Imod/ and "
                     "the run's PARAMETERS.txt beside them, so two AreTomo runs "
                     "never share a folder and the card names where its results "
                     "actually are. Point it at aretomo_output to use the older "
                     "shared, auto-versioned layout instead."},
            {"name": "gpu", "kind": "text", "flag": None,
             "title": "Fallback GPU",
             "default": "0", "help": "Fallback SINGLE GPU id, used only if the 'GPUs' "
             "list below is left blank. A single id here = sequential on one GPU. To "
             "use several GPUs, fill the GPUs field instead (do NOT put '0 1 2 3' here — "
             "extra tokens here shift the positional args and corrupt angpix)."},
            {"name": "angpix", "kind": "text", "flag": None,
             "title": "Pixel size (Å/px)",
             "default": "1.57", "help": "Input pixel size (Å/px) of the tilt "
             "stacks. Filled from the mdoc when one is readable."},
            {"name": "ARETOMO_GPUS", "kind": "env", "flag": "ARETOMO_GPUS", "gpu_sep": " ",
             "title": "GPUs",
             "default": "0 1 2 3",
             "help": "GPUs to spread tilt series across (space- or comma-separated). Each "
             "series runs on ONE GPU; with N GPUs, N series align at once (~N× faster). "
             "Blank = use the single 'gpu' field above (sequential)."},
            {"name": "ARETOMO_JOBS_PER_GPU", "kind": "env_int", "flag": "ARETOMO_JOBS_PER_GPU",
             "title": "Jobs per GPU",
             "default": 1, "min": 1, "max": 4, "step": 1,
             "help": "Concurrent AreTomo jobs PER GPU. 1 is safe; a 32 GB V100 can usually "
             "fit 2 at bin 8. Total concurrency = (#GPUs) × this."},
            {"name": "ARETOMO_ALIGNZ", "kind": "env", "flag": "ARETOMO_ALIGNZ",
             "title": "Alignment Z (px)",
             "default": "670", "help": "Alignment Z (unbinned px) ≈ lamella thickness."},
            {"name": "ARETOMO_VOLZ", "kind": "env", "flag": "ARETOMO_VOLZ",
             "title": "Volume Z (px)",
             "default": "3088", "help": "Output Z height; must exceed lamella thickness. "
             "Set 0 for ALIGNMENT-ONLY (skips the slow tomogram, still writes the .xf you "
             "import) — the fast way to align a whole dataset for ts_import / miss-alignment."},
            {"name": "ARETOMO_OUTBIN", "kind": "env_int", "flag": "ARETOMO_OUTBIN",
             "title": "Output binning",
             "default": 8, "min": 1, "max": 16, "step": 1,
             "help": "Output binning. 8 → 12.56 Å/px at 1.57 input."},
            {"name": "ARETOMO_DARKTOL", "kind": "env", "flag": "ARETOMO_DARKTOL",
             "title": "Dark-frame tolerance",
             "default": "0.000001", "help": "Dark-frame tol; ~0 disables (pre-curated tilts)."},
            {"name": "ARETOMO_TILTCOR", "kind": "env", "flag": "ARETOMO_TILTCOR",
             "title": "Tilt-offset correction",
             "default": "0", "help": "Tilt-offset correction 0/1. Usually 0 for lamellae."},
            {"name": "ARETOMO_FLIPVOLZ", "kind": "env", "flag": "ARETOMO_FLIPVOLZ",
             "title": "Flip volume Z",
             "default": "1", "help": "Flip handedness for Warp 0/1. Usually 1."},
            {"name": "ARETOMO_WBP", "kind": "env", "flag": "ARETOMO_WBP",
             "title": "Weighted back-projection",
             "default": "1", "help": "Weighted back projection 0/1."},
            {"name": "ARETOMO_TILTAXIS", "kind": "env", "flag": "ARETOMO_TILTAXIS",
             "title": "Tilt-axis angle (°)",
             "default": "", "help": "Tilt-axis (deg). Blank = AreTomo searches."},
            {"name": "ARETOMO_PATCH", "kind": "env", "flag": "ARETOMO_PATCH",
             "title": "Patch tracking",
             "default": "", "help": "Patch align e.g. '4 4'. Blank = skip patch tracking."},
            {"name": "ARETOMO_ALIGN", "kind": "env", "flag": "ARETOMO_ALIGN",
             "title": "Align (vs recon-only)",
             "default": "1", "help": "1 = align+recon, 0 = reconstruct only."},
            {"name": "ARETOMO_BIN", "kind": "env", "flag": "ARETOMO_BIN",
             "title": "AreTomo2 executable",
             "default": "/ceph/groups/structbio/Programs/AreTomo2/AreTomo2",
             "help": "AreTomo2 executable path."},
            {"name": "ARETOMO_CUDA_LIB", "kind": "env", "flag": "ARETOMO_CUDA_LIB",
             "title": "CUDA library dir",
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
            "pitfall": "Each job writes into its OWN jobs/<id>/ with a "
                       "PARAMETERS.txt audit. IMOD can't read Warp "
                       "float16 MRC — export WARP_FORCE_MRC_FLOAT32=1 before 3dmod "
                       "(brief gotcha §4.2).",
        },
        "status": lambda ps: ps.status_aretomo_xf(),
    },
    {
        "group": "5. Alignment", "id": "ts_import_alignments",
        "label": "ts_import_alignments",
        "base": "WarpTools ts_import_alignments",
        "params": [
            {"name": "settings", "kind": "text", "flag": "--settings",
             "title": "Settings file",
             "default": "warp_tiltseries.settings", "help": "ts .settings file."},
            {"name": "alignments", "kind": "text", "flag": "--alignments",
             "title": "AreTomo Imod folder",
             "default": "aretomo_output/Imod/",
             "help": "AreTomo Imod/ folder — filled from the newest AreTomo run, "
                     "job folders included."},
            {"name": "alignment_angpix", "kind": "text", "flag": "--alignment_angpix",
             "title": "Alignment pixel size (Å/px)",
             "default": "1.57", "help": "Pixel size AreTomo aligned at (1.57)."},
        ],
        "docs": {
            "what": "Imports AreTomo .xf/.tlt alignments back into Warp.",
            "range": "n/a",
            "effect": "ts_ctf / ts_reconstruct use these alignments.",
            "pitfall": "Run this AFTER AreTomo and BEFORE miss-alignment. miss-alignment writes its refined geometry "
                       "straight into the Warp XMLs, so there are no .xf files to import afterwards -- and a failed "
                       "import DESELECTS every series, which Warp persists, leaving ts_ctf/ts_reconstruct to "
                       "silently process nothing. alignment_angpix is the pixel size of the STACKS AreTomo "
                       "aligned, not the binned output.",
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
             "title": "Settings file",
             "default": "warp_tiltseries.settings", "help": "ts .settings file."},
            {"name": "mode", "kind": "choice", "flag": None,
             "title": "Action",
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
             "title": "Tomostar to change",
             "default": "", "help": "One tomostar to (de)select. Use the button to "
             "fill a chained command for ALL unaligned tomostars. Blank = all series."},
        ],
        "validate": lambda v: (
            "⚠ input_data is blank: this would apply to EVERY tilt series "
            "(a bare Deselect/Invert disables the whole dataset). Use the sync "
            "button to target the unaligned series, or name one tomostar."
            if (v.get("mode") in ("--deselect", "--invert")
                and not str(v.get("input_data", "")).strip()) else ""),
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
             "title": "Wrapper script",
             "default": _pkg_script("ml_missalignment_warp_auto.sh"),
             "help": "miss-alignment wrapper script (shipped with the app)."},
            {"name": "config", "kind": "text", "flag": None,
             "title": "Training config (YAML)",
             "default": "missalignment_config.yaml",
             "help": "TRAIN YAML config (relative to root). If missing, the wrapper seeds "
             "a training template and stops so you can review it, then re-run."},
            {"name": "input_dir", "kind": "text", "flag": None,
             "title": "Warp tilt-series folder",
             "default": "warp_tiltseries",
             "help": "Warp tilt-series dir with <series>.xml + tiltstack/<series>/"
             "<series>.st (run ts_import + ts_stack first, same prereqs as AreTomo)."},
            {"name": "MA_TRAINING_DEVICES", "kind": "env", "flag": "MA_TRAINING_DEVICES",
             "title": "Training GPU",
             "default": "0", "help": "--training-devices. KEEP THIS A SINGLE GPU (e.g. "
             "'0'). >1 makes torch spawn one trainer per GPU and they race to wipe the "
             "shared pool dir → FileNotFoundError on a partition_*.pickle. Scale speed "
             "with RECON devices + dataloaders instead."},
            {"name": "MA_RECON_DEVICES", "kind": "env", "flag": "MA_RECON_DEVICES", "gpu_sep": ",",
             "title": "Reconstruction GPUs",
             "default": "0,0,0", "help": "--reconstruction-devices: this is where you add "
             "GPUs for speed (recon feeds the pool and is the bottleneck). e.g. '0,1,2,3' "
             "or repeat an id to stack workers on it ('0,0,0')."},
            {"name": "MA_DATALOADERS", "kind": "env_int", "flag": "MA_DATALOADERS",
             "title": "Dataloaders per trainer",
             "default": 5, "min": 1, "max": 16, "step": 1,
             "help": "--dataloaders-per-trainer. The recon pool is split into "
             "(training_devices × this) partitions, each needing ≥ 2×batch_size; "
             "if it errors, raise Pool size or lower this."},
            {"name": "MA_POOL_SIZE", "kind": "env_int", "flag": "MA_POOL_SIZE",
             "title": "Pool size",
             "default": 2000, "min": 500, "max": 8000, "step": 100,
             "help": "--pool-size: subtomogram reconstructions cached in the temp "
             "pool. Must be ≥ 2×batch_size×training_devices×dataloaders (2000 keeps "
             "4 GPU × 5 loaders × batch 32 valid). Type the exact number."},
            {"name": "MA_START_ITER", "kind": "env_int", "flag": "MA_START_ITER",
             "title": "Start at iteration",
             "default": 0, "min": 0, "max": 20, "step": 1,
             "help": "--start-at-iteration (resume from the HIGHEST existing iterN)."},
            {"name": "MA_PREPARE_STACKS", "kind": "env", "flag": "MA_PREPARE_STACKS",
             "title": "Stack prep (Å/px)",
             "default": "10.0", "help": "--prepare-stacks pixel size (Å/px) for the "
             "reconstruction patches. Blank = skip stack preparation."},
            {"name": "MA_CONDA_ENV", "kind": "env", "flag": "MA_CONDA_ENV",
             "title": "Conda env",
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
        # infer REFINES the coarse alignment in the target XMLs exactly as
        # training does -- the docs say so outright ("miss-alignment starts
        # from an initially coarse aligned dataset"). Only train carried the
        # gate, so an infer run on a dataset whose AreTomo alignments had
        # never been imported went ahead and refined nothing, silently.
        "requires_coarse_alignment": True,
        "fixed_env": {"MA_MODE": "infer"},
        "params": [
            {"name": "script", "kind": "text", "flag": None,
             "title": "Wrapper script",
             "default": _pkg_script("ml_missalignment_warp_auto.sh"),
             "help": "miss-alignment wrapper script (shipped with the app)."},
            {"name": "config", "kind": "text", "flag": None,
             "title": "Inference config (YAML)",
             "default": "missalignment_infer_config.yaml",
             "help": "INFER YAML config (relative to root). If missing, the wrapper seeds "
             "an inference template (data_directory + model_run_directory) and stops for "
             "review. iteration_settings MUST match the training run."},
            {"name": "input_dir", "kind": "text", "flag": None,
             "title": "Warp tilt-series folder",
             "default": "warp_tiltseries",
             "help": "This dataset's warp_tiltseries — already coarse-aligned + imported, "
             "with the unaligned series deselected."},
            {"name": "MA_MODEL_RUN_DIR", "kind": "env", "flag": "MA_MODEL_RUN_DIR",
             "title": "Trained model run",
             "default": "", "help": "REQUIRED: the finished TRAINING run dir holding "
             "iter1/model.ckpt … iterN/model.ckpt (e.g. <selected>/warp_tiltseries)."},
            {"name": "MA_INFER_DEVICES", "kind": "env", "flag": "MA_INFER_DEVICES", "gpu_sep": ",",
             "title": "GPUs",
             "default": "0,1,2,3", "help": "GPUs for alignment (CUDA_VISIBLE_DEVICES). "
             "Inference has no training race, so use all the idle cards (check util%)."},
            {"name": "MA_START_ITER", "kind": "env_int", "flag": "MA_START_ITER",
             "title": "Start at iteration",
             "default": 0, "min": 0, "max": 20, "step": 1,
             "help": "--start-at-iteration (resume inference from iteration N)."},
            {"name": "MA_PREPARE_STACKS", "kind": "env", "flag": "MA_PREPARE_STACKS",
             "title": "Stack prep (Å/px)",
             "default": "10.0", "help": "--prepare-stacks pixel size (Å/px). MUST equal the "
             "resolution you TRAINED at (e.g. 12.56) — the model only works at its scale."},
            {"name": "MA_CONDA_ENV", "kind": "env", "flag": "MA_CONDA_ENV",
             "title": "Conda env",
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

    # ---------------- 6. CTF ----------------
    {
        "group": "6. CTF", "id": "ts_defocus_hand", "label": "ts_defocus_hand",
        "base": "WarpTools ts_defocus_hand",
        "params": [
            {"name": "settings", "kind": "text", "flag": "--settings",
             "title": "Settings file",
             "default": "warp_tiltseries.settings", "help": "ts .settings file."},
            {"name": "mode", "kind": "choice", "flag": None,
             "title": "Action",
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
             "title": "Settings file",
             "default": "warp_tiltseries.settings", "help": "ts .settings file."},
            {"name": "range_high", "kind": "text", "flag": "--range_high",
             "title": "Max resolution (Å)",
             "default": "7", "help": "CTF fit max resolution (Å)."},
            {"name": "defocus_max", "kind": "text", "flag": "--defocus_max",
             "title": "Max defocus (µm)",
             "default": "8", "help": "Max defocus to search (µm)."},
            {"name": "device_list", "kind": "text", "flag": "--device_list", "gpu_sep": " ",
             "title": "GPUs",
             "default": "0", "help": "GPU id(s)."},
            {"name": "perdevice", "kind": "slider_int", "flag": "--perdevice",
             "title": "Workers per GPU",
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
             "title": "Settings file",
             "default": "warp_tiltseries.settings", "help": "ts .settings file."},
            {"name": "angpix", "kind": "text", "flag": "--angpix",
             "title": "Output pixel size (Å/px)",
             "default": "10", "help": "OUTPUT tomogram pixel size (Å/px). 10 = a normal "
             "viewable/pickable tomogram. DO NOT use native (1.57) for full tomograms: "
             "the volume scales as (10/1.57)³ ≈ 260×, so each is tens of GB and ~40 min. "
             "Particles get reconstructed at fine res later by ts_export_particles."},
            {"name": "input_data", "kind": "text", "flag": "--input_data",
             "title": "Only these tilt series",
             "default": "",
             "help": "BLANK = every series in the .settings file (72 here). "
             "Name 2–3 to test a setting cheaply before committing the whole "
             "dataset: 'tomostar/Position003.tomostar "
             "tomostar/Position004.tomostar'. Paths are relative to the "
             "PROJECT ROOT, so they include tomostar/. A .txt file with one "
             "path per line works too. This does NOT edit the .settings file "
             "— every other step still sees all 72. (The Groups manager is "
             "the same mechanism with a checkbox list, applied to every "
             "WarpTools step at once.)"},
            {"name": "output_processing", "kind": "text", "flag": "--output_processing",
             "title": "Output folder",
             "default": "",
             "help": "Where this run writes — tomograms land in "
             "<here>/reconstruction/. BLANK = this job's own jobs/J<id>/, "
             "wired automatically. Name it to run VARIANTS of the same tilt "
             "series side by side (a bin4 thread and a bin8 thread, each "
             "keeping its own tomograms) instead of one overwriting the "
             "other. {jobid} expands to this job's id, e.g. "
             "jobs/{jobid}/reconstruction_bin4. Reading still comes from the "
             ".settings processing folder (or the parent job), so only the "
             "OUTPUT moves."},
            {"name": "device_list", "kind": "text", "flag": "--device_list", "gpu_sep": " ",
             "title": "GPUs",
             "default": "0", "help": "GPU id(s), e.g. 0 or '0 1'. Pick GPUs whose "
             "nvidia-smi GPU-Util is ~0% — low memory-used alone does NOT mean free."},
            {"name": "perdevice", "kind": "slider_int", "flag": "--perdevice",
             "title": "Workers per GPU",
             "default": 1, "min": 1, "max": 4, "step": 1,
             "help": "Workers per GPU. KEEP AT 1 when --deconv is on (V100 cuFFT crash)."},
            {"name": "deconv", "kind": "check", "flag": "--deconv",
             "title": "Deconvolve",
             "default": False, "help": "Deconvolve for visual contrast (not for STA). "
             "Also deconvolves every half-tomogram you asked for."},
            # The three deconv knobs are dropped from the command when Deconvolve
            # is off — Warp ignores them there, and a command box showing flags
            # that do nothing is a command box that lies.
            {"name": "deconv_strength", "kind": "text", "flag": "--deconv_strength",
             "title": "Deconv strength",
             "default": "", "skip_if": lambda v: not v.get("deconv"),
             "help": "Blank = Warp default 1. THE contrast knob; only used "
             "when Deconvolve is on."},
            {"name": "deconv_falloff", "kind": "text", "flag": "--deconv_falloff",
             "title": "Deconv falloff",
             "default": "", "skip_if": lambda v: not v.get("deconv"),
             "help": "Blank = Warp default 1. Only used when Deconvolve is on."},
            {"name": "deconv_highpass", "kind": "text", "flag": "--deconv_highpass",
             "title": "Deconv high-pass (Å)",
             "default": "", "skip_if": lambda v: not v.get("deconv"),
             "help": "Blank = Warp default 300 Å. Only used when Deconvolve "
             "is on."},
            {"name": "halfmap_frames", "kind": "check", "flag": "--halfmap_frames",
             "title": "Half-tomograms (frames)",
             "default": False,
             "help": "Also reconstruct two half-tomograms from frame halves — "
             "the even/odd input IsoNet 2 / Noise2Noise denoising wants. "
             "REQUIRES the frameseries motion step to have run with "
             "--average_halves; without that this fails, and the frameseries "
             "step must be re-run first."},
            {"name": "halfmap_tilts", "kind": "check", "flag": "--halfmap_tilts",
             "title": "Half-tomograms (tilts)",
             "default": False,
             "help": "Half-tomograms from tilt halves instead — no frameseries "
             "re-run needed, but Warp's own help says it doesn't work quite as "
             "well as frame halves."},
            {"name": "dont_overwrite", "kind": "check", "flag": "--dont_overwrite",
             "title": "Keep existing tomograms",
             "default": False,
             "help": "Skip tilt series that already have a tomogram instead of "
             "rebuilding them. OFF means the tomograms already in "
             "<processing>/reconstruction/ are REPLACED — and once M has changed the "
             "alignments they were made from, they cannot be rebuilt. Turn ON to "
             "resume an interrupted run, or to protect an existing set. (Giving "
             "each variant its own Output folder above is the cleaner way to "
             "keep two sets.)"},
            {"name": "subvolume_size", "kind": "text", "flag": "--subvolume_size",
             "title": "Sub-volume size",
             "default": "",
             "help": "Blank = Warp default 64. Reconstruction runs locally in "
             "sub-volumes of this size (pixels)."},
            {"name": "subvolume_padding", "kind": "text", "flag": "--subvolume_padding",
             "title": "Sub-volume padding",
             "default": "",
             "help": "Blank = Warp default 3. Padding factor against aliasing "
             "at sub-volume borders — raise if you see a grid of seams."},
            {"name": "keep_full_voxels", "kind": "check", "flag": "--keep_full_voxels",
             "title": "Mask partial voxels",
             "default": False,
             "help": "Masks out voxels not covered by every tilt image (large "
             "sample shifts). Warp's help is explicit: DON'T use this if you "
             "intend to run template matching."},
            {"name": "dont_invert", "kind": "check", "flag": "--dont_invert",
             "title": "Don't invert contrast",
             "default": False,
             "help": "Leave OFF for cryo data: contrast inversion is what "
             "template matching expects when density is dark in the original "
             "images. ON only for already-inverted input."},
            {"name": "dont_normalize", "kind": "check", "flag": "--dont_normalize",
             "title": "Don't normalize tilts",
             "default": False,
             "help": "Skips tilt-image normalisation. Leave OFF unless you "
             "are reproducing an external pipeline that expects raw scaling."},
            {"name": "dont_mask", "kind": "check", "flag": "--dont_mask",
             "title": "Don't mask tilts",
             "default": False,
             "help": "Skips per-tilt masking; masked areas would otherwise be "
             "filled with Gaussian noise. Leave OFF unless a mask is making "
             "things worse."},
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
            if v.get("perdevice", 1) > 1 and v.get("deconv") else
            # The deconv knobs are dropped from the command when Deconvolve is
            # off, so a typed strength would silently do nothing.
            "⚠ Deconv strength/falloff/high-pass are set but Deconvolve is OFF "
            "— they are dropped from the command. Tick Deconvolve, or clear them."
            if (not v.get("deconv")
                and any(str(v.get(k, "")).strip()
                        for k in ("deconv_strength", "deconv_falloff",
                                  "deconv_highpass"))) else
            "⚠ 'Mask partial voxels' is ON: Warp's own help says don't use it "
            "if you intend to run template matching on these tomograms."
            if v.get("keep_full_voxels") else ""),
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
             "title": "Settings file",
             "default": "warp_tiltseries.settings", "help": "ts .settings file."},
            {"name": "tomo_angpix", "kind": "text", "flag": "--tomo_angpix",
             "title": "Matching pixel size (Å/px)",
             "default": "10", "help": "Matching pixel size (Å). MUST equal a ts_reconstruct "
             "--angpix you have ALREADY run — matching reuses that full tomogram "
             "(warp_tiltseries/reconstruction/<pos>_<angpix>Apx.mrc). Mismatch → "
             "'A reconstruction at the desired resolution was not found'. 8-12 typical."},
            {"name": "template_emdb", "kind": "text", "flag": "--template_emdb",
             "title": "Template (EMDB code)",
             "default": "", "help": "EMDB code to fetch + use as the template, e.g. 70905. "
             "Set EITHER this OR template_path (not both)."},
            {"name": "template_path", "kind": "text", "flag": "--template_path",
             "title": "Template (local .mrc)",
             "default": "", "help": "Path to a local template .mrc. Set EITHER this OR "
             "template_emdb (not both)."},
            {"name": "override_suffix", "kind": "text", "flag": "--override_suffix",
             "title": "Pick-set name",
             "default": "", "help": "Overrides the STAR suffix (normally derived from the "
             "template name) so this pick set gets its OWN name: files become "
             "warp_tiltseries/matching/<pos>_<tomo_angpix>Apx<suffix>.star. INCLUDE A LEADING "
             "UNDERSCORE if you want one (e.g. '_run2'; without it the suffix abuts 'Apx'). "
             "Use a different suffix per run to keep parallel pick sets side by side — "
             "threshold_picks / export then pick a set via --in_suffix (this is how you fork "
             "picking). Blank = default template-derived name."},
            {"name": "subdivisions", "kind": "slider_int", "flag": "--subdivisions",
             "title": "Angular subdivisions",
             "default": 3, "min": 1, "max": 6, "step": 1,
             "help": "Angular subdivisions of the search (finer = more orientations, slower)."},
            {"name": "template_diameter", "kind": "text", "flag": "--template_diameter",
             "title": "Particle diameter (Å)",
             "default": "", "help": "Particle diameter (Å)."},
            {"name": "symmetry", "kind": "text", "flag": "--symmetry",
             "title": "Symmetry",
             "default": "C1", "help": "Point group, e.g. O, D2, C1."},
            {"name": "whiten", "kind": "check", "flag": "--whiten",
             "title": "Spectral whitening",
             "default": True, "help": "Spectral whitening; helps with good alignments."},
            {"name": "optimize_poses", "kind": "check", "flag": "--optimize_poses",
             "title": "Refine hit poses",
             "default": False, "help": "Locally refine each hit's orientation/position after "
             "the coarse search (better picks, a bit slower). ON in the reference workflow."},
            {"name": "check_hand", "kind": "slider_int", "flag": "--check_hand",
             "title": "Handedness check",
             "default": 2, "min": 0, "max": 2, "step": 1,
             "help": "2 = verify geometry/handedness during matching."},
            {"name": "npeaks", "kind": "slider_int", "flag": "--npeaks",
             "title": "Max peaks per series",
             "default": 2000, "min": 100, "max": 50000, "step": 500,
             "help": "Max peaks SAVED per tilt series. This is a HARD CAP — if a series "
             "actually has more particles you'll silently keep only the top-scoring 2000. "
             "For crowded samples raise it (you can tell you're capped when every series "
             "returns exactly this many). Costs disk, not match time."},
            {"name": "peak_distance", "kind": "text", "flag": "--peak_distance",
             "title": "Min peak spacing (Å)",
             "default": "", "help": "Minimum spacing between peaks in Å. Blank = the template "
             "diameter. Lower it (e.g. 30) for tightly-packed particles so neighbours aren't "
             "suppressed; raise it to avoid double-picking one particle."},
            {"name": "max_missing_tilts", "kind": "slider_int", "flag": "--max_missing_tilts",
             "title": "Max missing tilts",
             "default": 2, "min": -1, "max": 20, "step": 1,
             "help": "Drop positions not covered by at least this many tilts. -1 disables "
             "culling (keep everything, e.g. thin/edge regions); default 2."},
            {"name": "subvolume_size", "kind": "slider_int", "flag": "--subvolume_size",
             "title": "Tile size (px)",
             "default": 192, "min": 48, "max": 512, "step": 16,
             "help": "Local matching TILE size, in TOMOGRAM pixels (at tomo_angpix, NOT raw "
             "pixels). Just needs to comfortably exceed the template — 192 does so hugely. "
             "It is NOT the particle box (that's export --box). Reduce only if you hit GPU "
             "OOM or want speed; keep it even (FFT-friendly)."},
            {"name": "device_list", "kind": "text", "flag": "--device_list", "gpu_sep": " ",
             "title": "GPUs",
             "default": "", "help": "GPU id(s), space-separated e.g. '2 3'. BLANK = ALL "
             "GPUs (Warp's default — it WILL grab 0/1). Set this to the idle cards (check "
             "nvidia-smi util%) to leave others' jobs alone. Or prefix CUDA_VISIBLE_DEVICES=2,3."},
            {"name": "perdevice", "kind": "slider_int", "flag": "--perdevice",
             "title": "Workers per GPU",
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
             "title": "Settings file",
             "default": "warp_tiltseries.settings", "help": "ts .settings file."},
            {"name": "in_suffix", "kind": "text", "flag": "--in_suffix",
             "title": "Input suffix",
             "default": "", "help": "Suffix of the template-match star files to threshold."},
            {"name": "out_suffix", "kind": "text", "flag": "--out_suffix",
             "title": "Output suffix",
             "default": "clean", "help": "Suffix for thresholded output star files."},
            {"name": "minimum", "kind": "slider_int", "flag": "--minimum",
             "title": "Min score to keep",
             "default": 3, "min": 0, "max": 10, "step": 1,
             "help": "The cut applied to _rlnAutopickFigureOfMerit — whatever "
             "the file that made the picks put there. It is NOT a fraction and "
             "NOT a percentile.\n"
             "• Template matching writes a score normalised against the "
             "background, so the number reads as 'standard deviations above "
             "noise'. 3 keeps confident hits, 2 is permissive, 5 is strict, "
             "and it means the same thing in every tomogram.\n"
             "• crYOLO .coords carry NO score, so the converter writes a "
             "constant 1.0 for every particle. Thresholding those can only "
             "keep all of them or none — set the confidence in crYOLO at "
             "prediction time instead."},
        ],
        "validate": lambda v: (
            "⚠ These look like crYOLO picks, and crYOLO .coords carry no "
            "score — every particle was written with the same figure of "
            "merit (1.0). This step can only keep all of them or none of "
            "them; it cannot rank them. Filter at crYOLO's own confidence "
            "threshold when predicting instead."
            if "cryolo" in str(v.get("in_suffix", "")).lower() else ""),
        "docs": {
            "what": "Keeps picks whose figure of merit is at least `minimum`; "
                    "writes *<out_suffix>.star.",
            "range": "3 for template matching. There is no useful value for "
                     "crYOLO picks — their score is constant.",
            "effect": "Higher minimum = fewer, cleaner picks.",
            "pitfall": "The number only means something if the picks carry a "
                       "VARYING score. Template matching normalises against "
                       "background, so one minimum works project-wide; crYOLO "
                       "COORDS have no score at all, and thresholding them "
                       "looks like it worked while doing nothing.",
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
             "title": "Settings file",
             "default": "warp_tiltseries.settings", "help": "ts .settings file."},
            {"name": "input_star", "kind": "text", "flag": "--input_star",
             "title": "RELION star (re-extract a selection)",
             "default": "",
             "help": "RE-EXTRACTION ROUTE. Hand Warp a RELION particles.star directly — "
             "normally the Subset-selection output (Select/jobNNN/particles.star, the "
             "good class with duplicates removed). Warp reads its rlnCoordinateX/Y/Z, "
             "SUBTRACTS the refined rlnOriginX/Y/ZAngst shifts itself and scales by "
             "coords_angpix, so each particle is cut on its refined centre with no "
             "pick stars and no hand conversion. Set coords_angpix to the star's own "
             "pixel size (its optics rlnImagePixelSize — read for you when this card is "
             "built downstream of the selection). The refined Euler angles are copied "
             "into the output star, so random_subset_ref.mrc from the convert step is "
             "an ORIENTED average and should look like the particle, not a blob.\n"
             "Leave BLANK for a first extraction from pick stars (folder + pattern "
             "below). The two routes are mutually exclusive: with a star set, the "
             "pick-star fields and '0-1 fractions' are dropped from the command."},
            # Dropped entirely once a RELION star is given: Warp takes one
            # input route or the other, and leaving these populated emitted
            # BOTH on the same command line. A warning was not enough — the
            # card's own defaults filled them in, so the command ran with both.
            {"name": "input_directory", "kind": "text", "flag": "--input_directory",
             "skip_if": lambda v: bool(str(v.get("input_star", "")).strip()),
             "title": "Pick-star folder",
             "default": "warp_tiltseries/matching",
             "help": "Where the thresholded pick stars live."},
            {"name": "input_pattern", "kind": "text", "flag": "--input_pattern",
             "skip_if": lambda v: bool(str(v.get("input_star", "")).strip()),
             "title": "Pick-star pattern",
             "default": "*clean.star", "help": "Glob for thresholded pick star files."},
            {"name": "output_star", "kind": "text", "flag": "--output_star",
             "title": "Output star",
             "default": "relion4/{jobid}/matching.star",
             "help": "Output star path. Put it INSIDE the RELION project dir "
             "(output_processing) — RELION must later be launched from that dir. "
             "{jobid} resolves to this job's id so runs never collide."},
            {"name": "output_processing", "kind": "text", "flag": "--output_processing",
             "title": "RELION project folder",
             "default": "relion4/{jobid}",
             "help": "RELION project/export dir. The subtomo image paths in the star are "
             "written RELATIVE to this, so you MUST launch RELION from here (the recurring "
             "'file does not exist' bug is launching from the wrong dir). Default "
             "relion4/{jobid} gives each export its own dir; Build-downstream from a "
             "pick-set card instead names it after the pick set (relion4/<tag>)."},
            {"name": "output_angpix", "kind": "text", "flag": "--output_angpix",
             "title": "Export pixel size (Å/px)",
             "default": "4", "help": "Export pixel size (Å). Choose so Nyquist sits just "
             "below feature resolution."},
            {"name": "box", "kind": "slider_int", "flag": "--box",
             "title": "Box size (px)",
             "default": 64, "min": 32, "max": 256, "step": 8,
             "help": "Box size (px) — the CONTAINER: how much field of view is cut out. "
             "Its physical size is box × output_angpix Å. Aim for ~1.5-2× the particle "
             "diameter: enough padding for the particle plus alignment shifts, no more "
             "(cost scales with box³ in every downstream RELION job). If you re-extract "
             "at half the pixel size WITHOUT recentring, double the box to keep the same "
             "field; an export from a RELION star is already centred on the refined "
             "particle centre and needs less padding."},
            {"name": "diameter", "kind": "text", "flag": "--diameter",
             "title": "Particle diameter (Å)",
             "default": "",
             "help": "TRUE particle diameter (Å) — NOT the box. This is where the particle "
             "is assumed to end: everything outside this sphere is treated as SOLVENT and "
             "used to estimate the background for normalising each subtomogram. So it must "
             "be comfortably smaller than the box (≤80%, ideally 50-70%) or there is no "
             "solvent shell left to measure. Overstating it corrupts the normalisation."},
            {"name": "relion_format", "kind": "choice", "flag": None,
             "title": "RELION format",
             "choices": [
                 ("3D subtomograms (RELION 4)", "--3d"),
                 ("2D image series (RELION 5)", "--2d"),
             ],
             "default": "--3d",
             "help": "RELION 4 uses 3D subtomos (--3d). RELION 5 --tomo uses the 2D "
             "image series (--2d). WarpTools needs exactly one of these — pick to match "
             "the RELION you'll hand off to."},
            # Never sent with a RELION star: such a star holds PIXEL coordinates,
            # and declaring them 0-1 fractions makes Warp multiply by the tomogram
            # width (EML46 J74, 2026-09-02: coordinates up to 4,114,010 in a 4096 px
            # volume, every particle cut from outside the tomogram).
            {"name": "normalized_coords", "kind": "check", "flag": "--normalized_coords",
             "skip_if": lambda v: bool(str(v.get("input_star", "")).strip()),
             "title": "Coords are 0–1 fractions",
             "default": True,
             "help": "PICK-STAR ROUTE ONLY. ON = the pick stars' coords are 0-1 FRACTIONS "
             "of the tomogram (what Warp template matching and the crYOLO converter "
             "write). OFF = they are pixels, and coords_angpix must say at which pixel "
             "size. Check a pick star: values 0-1 = ON; values in the hundreds = OFF + "
             "coords_angpix. Ignored (not sent) when a RELION star is given above."},
            # KEPT on the RELION-star route: Warp requires exactly one of
            # --normalized_coords / --coords_angpix on every route, and for a RELION
            # star the right value is the star's own pixel size. Dropping it here
            # (as an earlier version did) made Warp refuse the run.
            {"name": "coords_angpix", "kind": "text", "flag": "--coords_angpix",
             "title": "Coords pixel size (Å/px)",
             "default": "",
             "help": "Pixel size (Å/px) the INPUT coordinates are expressed in.\n"
             "RELION star: REQUIRED — the star's optics rlnImagePixelSize, i.e. the "
             "export pixel size of the extraction RELION refined (6.28 for bin4, 3.14 "
             "for bin2). Filled in from the star when the card is built downstream of "
             "a selection; the run is refused if it disagrees with the star.\n"
             "Pick stars in pixels: the Å/px they were written at (the <apx>Apx tag in "
             "the filename). Leave BLANK when '0-1 fractions' is ON."},
            {"name": "relative_output_paths", "kind": "check",
             "title": "Relative paths",
             "flag": "--relative_output_paths", "default": True,
             "help": "Write relative paths into the star (portable projects). Keep ON — "
             "the RELION-launch-from-output_processing rule depends on it."},
            {"name": "device_list", "kind": "text", "flag": "--device_list", "gpu_sep": " ",
             "title": "GPUs",
             "default": "", "help": "GPU id(s), space-separated (e.g. '0 1 2 3'). Blank = "
             "all GPUs in the system. Pick GPUs whose nvidia-smi GPU-Util is ~0%."},
            {"name": "perdevice", "kind": "slider_int", "flag": "--perdevice",
             "title": "Workers per GPU",
             "default": 1, "min": 1, "max": 8, "step": 1,
             "help": "Processes per GPU. 1 is safe; raise only if GPU memory allows."},
        ],
        "validate": _validate_export,
        "docs": {
            "what": "Extracts CTF-corrected particles into a RELION project dir — a "
                    "particles star (+ optimisation_set.star for RELION 5). Two inputs, "
                    "same extraction: a folder of pick stars (first extraction), or a "
                    "RELION star (re-extraction of a selection at a finer pixel size).",
            "range": "box 64-128; output_angpix 3-5 for most targets. Re-extracting at "
                     "half the pixel size: double the box to keep the same field of view.",
            "effect": "3D subtomos (--3d) = RELION 4; --2d = RELION 5 --tomo. Paths are "
                      "relative to output_processing — that dir IS the RELION project "
                      "root. RELION-star route: Warp applies the refined rlnOrigin*Angst "
                      "shifts itself and copies the refined angles into the output star.",
            "pitfall": "LAUNCH RELION FROM output_processing, or every subtomo path is "
                       "wrong ('file does not exist'). RELION 4 does NOT auto-resize the "
                       "reference — pre-scale it (see 'RELION 4: Class3D'). Re-extraction: "
                       "coords_angpix must be the RELION star's own pixel size and "
                       "'0-1 fractions' must be OFF — the card enforces both.",
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
        "validate": lambda v: (
            "\u26a0 'relion4' is the container that holds one folder per export, "
            "not a project folder. Point this at the export's OWN folder "
            "(relion4/<something>) or the star will not be found."
            if str(v.get("project_dir", "")).strip().strip("/") == "relion4"
            else ""),
        "output_params": ["project_dir"],
        "params": [
            {"name": "script", "kind": "text", "flag": None,
             "title": "Wrapper script",
             "default": _pkg_script("ml_relion4_convert_star_warp_auto.sh"),
             "help": "STAR conversion + initial-reference wrapper (shipped with the app)."},
            {"name": "project_dir", "kind": "text", "flag": None,
             "title": "RELION project folder",
             "default": "relion4",
             "help": "The folder your 'Extract particles (\u2192 RELION)' job wrote "
             "into \u2014 the one holding the export star and a subtomo/ folder of "
             "particle images. It is that job's own 'RELION project folder' "
             "value, e.g. relion4/cryolo_isonetmodel_J48; the export card's "
             "Details \u25b8 OUTPUTS names it.\n\n"
             "Everything runs FROM this folder, because the particle image "
             "paths inside the star are written relative to it. Run from "
             "anywhere else and RELION reports 'file does not exist' for every "
             "particle \u2014 that is the mistake this field exists to prevent, "
             "and you must launch RELION itself from here later for the same "
             "reason.\n\n"
             "NOT the bare relion4/ container: that holds one folder per "
             "export, so pointing at it finds no star. The reliable way to "
             "fill it in is right-click the export card \u25b8 Build downstream "
             "\u25b8 RELION 4: convert STAR, which copies it across for you."},
            {"name": "starfile", "kind": "text", "flag": None,
             "title": "Export star",
             "default": "matching.star",
             "help": "The Warp export star, named RELATIVE to the project "
             "folder above \u2014 just 'matching.star', not a path. The export "
             "job wrote it directly inside that folder."},
            {"name": "RELION_MODULE", "kind": "env", "flag": "RELION_MODULE",
             "title": "RELION module",
             "default": "relion/4.0.1", "help": "module load name for RELION 4 on your cluster."},
            {"name": "PARTICLEDIR", "kind": "env", "flag": "PARTICLEDIR",
             "title": "Subtomo folder (absolute)",
             "default": "", "help": "Absolute path to the exported subtomo/ dir (the "
             "'subtomo/' prefix in the star is rewritten to this). Blank = "
             "<project_dir>/subtomo/. A trailing slash is enforced."},
            {"name": "PATH_MATCH", "kind": "env", "flag": "PATH_MATCH",
             "title": "Path prefix to replace",
             "default": "subtomo/", "help": "Path prefix in the Warp star to replace with "
             "PARTICLEDIR. Change only if export wrote a different prefix."},
            {"name": "CS", "kind": "env", "flag": "CS",
             "title": "Spherical aberration (mm)",
             "default": "2.7", "help": "Spherical aberration (mm) for relion_convert_star."},
            {"name": "Q0", "kind": "env", "flag": "Q0",
             "title": "Amplitude contrast",
             "default": "0.07", "help": "Amplitude contrast for relion_convert_star."},
            {"name": "NREF", "kind": "env_int", "flag": "NREF",
             "title": "Particles for reference",
             "default": 1000, "min": 100, "max": 5000, "step": 100,
             "help": "Random particles used to reconstruct the initial reference."},
            {"name": "MAKE_REF", "kind": "env", "flag": "MAKE_REF",
             "title": "Build random reference",
             "default": "1", "help": "1 = also build random_subset_ref.mrc (relion_reconstruct); "
             "0 = only convert the star."},
            {"name": "execute", "kind": "check", "flag": "--execute",
             "title": "Execute (not dry-run)",
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
             "title": "RELION job folder",
             "help": "The RELION job folder this card stands for."},
            {"name": "job_type", "kind": "text", "flag": None, "default": "",
             "title": "Job type",
             "help": "Refine3D / Class3D / Select."},
            {"name": "data_star", "kind": "text", "flag": None, "default": "",
             "title": "Particle star",
             "help": "Its particle star (run_itNNN_data.star / particles.star)."},
            {"name": "half1", "kind": "text", "flag": None, "default": "",
             "title": "Half-map 1",
             "help": "Unfiltered half-map 1, if this job produced one."},
            {"name": "half2", "kind": "text", "flag": None, "default": "",
             "title": "Half-map 2",
             "help": "Unfiltered half-map 2, if this job produced one."},
            {"name": "class_map", "kind": "text", "flag": None, "default": "",
             "title": "Class map",
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
        # RELION estimates the initial noise spectrum PER OPTICS GROUP, sampling up
        # to 1000 particles from each. After M refines spherical aberration per tilt
        # series (MCore --ctf_cs) every series carries its own Cs, so every series
        # becomes its own optics group. On EML45: 267 groups, and RELION reported
        # "Estimating initial noise spectra from 266000 particles" — 5.9 hours
        # before iteration 1. A pre-M export of the same data has 2 groups and
        # reports 1000.
        "group": "10. RELION 4", "id": "relion4_merge_optics",
        "label": "RELION 4: merge optics groups",
        "base": "python3",
        "params": [
            {"name": "script", "kind": "text", "flag": None,
             "title": "Wrapper script",
             "default": _pkg_script("ml_relion4_merge_optics.py"),
             "help": "Optics-group merger (shipped with the app)."},
            {"name": "star", "kind": "text", "flag": None,
             "title": "Input star",
             "default": "",
             "help": "The converted star to collapse, e.g. "
             "relion4/<set>/matching_conv.star. REQUIRED. Never modified — a new "
             "file is written beside it."},
            {"name": "out", "kind": "text", "flag": "-o",
             "title": "Output star",
             "default": "",
             "help": "Output star. Blank = <input>_1optics.star next to the input."},
            {"name": "cs", "kind": "text", "flag": "--cs",
             "title": "Spherical aberration (mm)",
             "default": "2.7",
             "help": "Spherical aberration (mm) for the merged group. Cs is a "
             "property of the MICROSCOPE, so one value is correct; the per-series "
             "values come from M having refined it. 2.7 is the usual Krios/Talos "
             "figure — check yours before setting anything else."},
            {"name": "report", "kind": "check", "flag": "--report",
             "title": "Report only",
             "default": True,
             "help": "ON = only COUNT the groups and say how much noise estimation "
             "they will cost; writes nothing. Turn OFF to write the collapsed star."},
        ],
        "validate": lambda v: (
            "⚠ Point 'star' at the converted star to collapse (matching_conv.star)."
            if not str(v.get("star", "")).strip() else
            "ℹ REPORT only — nothing is written. Turn 'report' OFF to write the "
            "collapsed star."
            if v.get("report") else ""),
        "docs": {
            "what": "Collapses a star's optics groups into one, so RELION estimates "
                    "the initial noise spectrum once instead of once per group.",
            "range": "Use before 3D classification when the group count is in the "
                     "hundreds. Check with 'report' first.",
            "effect": "Writes a NEW star; the input is untouched. Every particle is "
                      "reassigned to group 1, and the kept parameters are those "
                      "shared by most groups (Cs overridable).",
            "pitfall": "The per-series groups are REAL — M refined Cs per tilt "
                       "series, and that precision is worth keeping for a FINAL "
                       "refinement. Collapse for CLASSIFICATION, where it only "
                       "costs time: feed the collapsed star to Class3D and the "
                       "original to Refine3D.",
        },
        "status": None,
    },
    {
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
             "title": "Wrapper script",
             "default": _pkg_script("ml_star_check_particles.py"),
             "help": "Particle-existence checker (shipped with the app)."},
            {"name": "star", "kind": "text", "flag": None,
             "title": "Particle star",
             "default": "",
             "help": "The particle STAR you are about to refine (e.g. "
             "relion4/<project>/matching_conv.star). REQUIRED."},
            {"name": "root", "kind": "text", "flag": "--root",
             "title": "Resolve paths against",
             "default": "",
             "help": "Only for stars with RELATIVE particle paths: the dir to resolve "
             "them against (usually the RELION project dir). Blank = the star's own "
             "folder, then the working dir. Absolute paths need nothing here."},
            {"name": "fix", "kind": "check", "flag": "--fix",
             "title": "Write pruned star",
             "default": False,
             "help": "Write a pruned star keeping ONLY particles whose file exists. OFF "
             "= report only. Prune when a scattering of particles is missing (edge-"
             "clipped); if whole tomograms are missing, re-run the export instead."},
            {"name": "out", "kind": "text", "flag": "--out",
             "title": "Pruned star (output)",
             "default": "",
             "help": "Where to write the pruned star (default: <star>_present.star). "
             "Keep it inside the RELION project dir."},
            {"name": "ref", "kind": "text", "flag": "--ref",
             "title": "Reference map to check",
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
             "title": "Wrapper script",
             "default": _pkg_script("ml_verify_reextract.py"),
             "help": "Re-extraction coordinate verifier (shipped with the app)."},
            {"name": "source_star", "kind": "text", "flag": None,
             "title": "Source star (RELION)",
             "default": "",
             "help": "The RELION star you re-extracted FROM — the export card's "
             "input_star (e.g. Select/job029/particles.star). REQUIRED."},
            {"name": "new_star", "kind": "text", "flag": None,
             "title": "New star (export)",
             "default": "",
             "help": "The matching.star the export WROTE (e.g. "
             "relion4/<project>/matching.star). REQUIRED."},
            {"name": "no_recenter", "kind": "check", "flag": "--no-recenter",
             "title": "Shifts were NOT applied",
             "default": False,
             "help": "Leave OFF for an export made from the RELION star: Warp "
             "subtracts the refined rlnOrigin*Angst shifts itself, so the check "
             "expects new = (coord − origin/apx) × (apx_old/apx_new). Tick only if "
             "the source star had its origins zeroed (or has none), otherwise every "
             "particle looks off by its refinement shift."},
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
        # RETIRED 2026-09-02. Re-extraction no longer goes through pick stars:
        # ts_export_particles takes the RELION star directly (input_star), and
        # Warp subtracts the refined rlnOrigin*Angst shifts itself. The MODE A/B/C
        # converter that used to sit here produced three coordinate conventions
        # and a silent failure for each. This entry remains only so cards made
        # before then keep their row, details and recorded command; it is not in
        # the palette, the outline, the template rail or any downstream menu.
        "group": "10. RELION 4", "id": "relion4_to_warp",
        "label": "RELION 4 → Warp: pick-star converter (retired)",
        "legacy": True,
        "base": "python3",
        "output_params": ["out_dir"],
        "params": [
            {"name": "script", "kind": "text", "flag": None,
             "title": "Wrapper script",
             "default": _pkg_script("ml_relion4_select_picks.py"),
             "help": "Retired pick-star converter (kept for old cards)."},
            {"name": "particles_star", "kind": "text", "flag": None,
             "title": "RELION particles star", "default": "",
             "help": "The RELION star this card converted. To re-extract it now, "
             "build ts_export_particles downstream of the selection card instead."},
            {"name": "keep_all", "kind": "check", "flag": "--keep-all",
             "title": "Keep all particles", "default": True,
             "help": "Every particle in the star (no class filtering)."},
            {"name": "out_dir", "kind": "text", "flag": "--out-dir",
             "title": "Output folder", "default": "warp_tiltseries/matching_reextract",
             "help": "Where the pick stars were written."},
            {"name": "suffix", "kind": "text", "flag": "--suffix",
             "title": "Output suffix", "default": "reextract",
             "help": "Tag in the pick-star filenames."},
            {"name": "execute", "kind": "check", "flag": "--execute",
             "title": "Execute (not dry-run)", "default": False,
             "help": "OFF = dry run."},
        ],
        "validate": lambda v: (
            "⚠ RETIRED. Re-extract by building ts_export_particles downstream of the "
            "Subset-selection card: it takes the RELION star directly (input_star + "
            "coords_angpix) and Warp applies the refined shifts itself. This converter "
            "wrote pick stars in one of three coordinate conventions and is kept only "
            "so old cards can be read."),
        "docs": {
            "what": "Retired. Converted a RELION particle star into per-tomogram Warp "
                    "pick stars for a second export.",
            "range": "Do not use. Build ts_export_particles downstream of the selection.",
            "effect": "Old cards keep their recorded command and outputs.",
            "pitfall": "Its three modes wrote three coordinate conventions (0-1 "
                       "fractions vs pixels at two pixel sizes) and the wrong export flag "
                       "for any of them extracted noise with exit 0. The direct route has "
                       "one convention: the star's own pixel size.",
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
             "title": "Wrapper script",
             "default": _pkg_script("ml_m_check_ctf.py"),
             "help": "CTF completeness checker (shipped with the app)."},
            {"name": "project_dir", "kind": "text", "flag": None,
             "title": "Project root",
             "default": ".",
             "help": "Project root (holds warp_tiltseries/ and tomostar/)."},
            {"name": "processing", "kind": "text", "flag": "--processing",
             "title": "Processing folder",
             "default": "warp_tiltseries",
             "help": "Tilt-series processing folder holding the per-series .xml "
             "metadata, relative to the project root."},
            {"name": "tomostar", "kind": "text", "flag": "--tomostar",
             "title": "Tomostar folder",
             "default": "tomostar",
             "help": "Folder holding the .tomostar files, relative to the project "
             "root. Used to report each offender's tilt count and to build the "
             "ts_ctf command that fixes them."},
            {"name": "write_list", "kind": "text", "flag": "--write-list",
             "title": "Write offender list to",
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
             "title": "Wrapper script",
             "default": _pkg_script("ml_m_index_versions.py"),
             "help": "Species-version indexer (shipped with the app)."},
            {"name": "project_dir", "kind": "text", "flag": None,
             "title": "Project root",
             "default": ".",
             "help": "Project root (holds m*/ and .tomogration_jobs.json)."},
            {"name": "species", "kind": "text", "flag": "--species",
             "title": "Species filter",
             "default": "",
             "help": "Only index species whose folder name contains this. Blank = "
             "all of them."},
            {"name": "write", "kind": "check", "flag": "--write",
             "title": "Write version files",
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
             "title": "Wrapper script",
             "default": _pkg_script("ml_m_reset_warp_auto.sh"),
             "help": "M setup inspector / reset (shipped with the app)."},
            {"name": "project_dir", "kind": "text", "flag": None,
             "title": "Project root",
             "default": ".",
             "help": "Project root (holds m*/ and warp_tiltseries/). '.' = this project."},
            {"name": "execute", "kind": "check", "flag": "--execute",
             "title": "Execute (not report-only)",
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
             "title": "M project folder",
             "default": "m", "help": "Folder to hold the M project (created if absent)."},
            {"name": "name", "kind": "text", "flag": "--name",
             "title": "Population name",
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
             "title": "Source name",
             "default": "", "help": "Data-source name (usually the same as the "
             "population name). REQUIRED."},
            {"name": "population", "kind": "text", "flag": "--population",
             "title": "Population file",
             "default": "m/{name}.population",
             "help": "The .population file from 'create population'."},
            {"name": "processing_settings", "kind": "text",
             "title": "Tilt-series settings",
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
             "title": "RELION module",
             "default": "relion/4.0.1",
             "help": "lmod module providing relion_mask_create. Everything on this "
             "cluster is behind a module, so without this the command is simply "
             "'not found'. Blank = assume RELION is already on PATH."},
            {"name": "i", "kind": "text", "flag": "--i",
             "title": "Input map",
             "default": "",
             "help": "Input map to threshold — normally your RELION Refine3D result "
             "(e.g. relion4/<proj>/Refine3D/jobNNN/run_class001.mrc). REQUIRED."},
            {"name": "o", "kind": "text", "flag": "--o",
             "title": "Output mask",
             "default": "m/mask.mrc", "help": "Output binary mask."},
            {"name": "ini_threshold", "kind": "text", "flag": "--ini_threshold",
             "title": "Density threshold",
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
             "title": "Population file",
             "default": "", "help": "The .population file. REQUIRED."},
            {"name": "name", "kind": "text", "flag": "--name",
             "title": "Species name",
             "default": "",
             "help": "Species name (e.g. spike). REQUIRED. NOTE create_species ADDS a "
             "species — it never replaces one. Running it twice leaves BOTH in the "
             "population, MCore then refines both and prints a resolution line for "
             "each, and if they share a name you cannot tell which is which. To "
             "REPLACE a species, reset M and rebuild. To COMPARE two masks/references, "
             "give them DIFFERENT names on purpose."},
            {"name": "diameter", "kind": "text", "flag": "--diameter",
             "title": "Particle diameter (Å)",
             "default": "", "help": "Particle diameter in Å — the TRUE particle size, "
             "same number you used for extraction."},
            {"name": "sym", "kind": "text", "flag": "--sym",
             "title": "Symmetry",
             "default": "C1",
             "help": "Point-group symmetry (C1, C3, D2, O…). Only impose symmetry you "
             "are confident in — a wrong one caps resolution and can't be undone later."},
            {"name": "temporal_samples", "kind": "text", "flag": "--temporal_samples",
             "title": "Temporal samples",
             "default": "1",
             "help": "Pose samples through the tilt series. START AT 1. Raise it later "
             "with 'resample trajectories' once you have resolution to spend."},
            {"name": "half1", "kind": "text", "flag": "--half1",
             "title": "Half-map 1 (unfiltered)",
             "default": "",
             "help": "UNFILTERED half-map 1 from RELION (run_half1_class001_unfil.mrc)."},
            {"name": "half2", "kind": "text", "flag": "--half2",
             "title": "Half-map 2 (unfiltered)",
             "default": "",
             "help": "UNFILTERED half-map 2 (run_half2_class001_unfil.mrc). The two "
             "halves must be the independent ones — that is what keeps FSC honest."},
            {"name": "mask", "kind": "text", "flag": "--mask",
             "title": "Mask",
             "default": "m/mask.mrc", "help": "Binary mask from the previous step."},
            {"name": "particles_relion", "kind": "text", "flag": "--particles_relion",
             "title": "RELION particle star",
             "default": "",
             "help": "The RELION run_data.star holding the refined particle poses "
             "(e.g. .../Refine3D/jobNNN/run_data.star)."},
            {"name": "ignore_unmatched", "kind": "check", "flag": "--ignore_unmatched",
             "title": "Ignore unmatched series",
             "default": False,
             "help": "Proceed when the particle star references tilt series the data "
             "source does not contain. REQUIRED whenever the population is a SUBSET of "
             "your series (e.g. filtered by particle count) — otherwise create_species "
             "refuses to build, writes an EMPTY species folder, and MCore then reports "
             "'0/0' species with no error of its own. Only tick it when you know the "
             "unmatched particles are ones you meant to leave out."},
            {"name": "angpix_resample", "kind": "text", "flag": "--angpix_resample",
             "title": "Refine at (Å/px)",
             "default": "",
             "help": "Pixel size (Å) M should refine at. Set it FINER than your "
             "extraction — that headroom is the point of M. Cannot go below the raw "
             "detector pixel size."},
            {"name": "lowpass", "kind": "text", "flag": "--lowpass",
             "title": "Initial low-pass (Å)",
             "default": "10",
             "help": "Initial low-pass (Å) applied to the reference, to stop the first "
             "iterations chasing noise."},
        ],
        "validate": lambda v: (
            "⚠ Set population to the .population file (REQUIRED)."
            if not str(v.get("population", "")).strip() else
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
             "title": "Population file",
             "default": "", "help": "The .population file. REQUIRED."},
            {"name": "iter", "kind": "text", "flag": "--iter",
             "title": "Sub-iterations",
             "default": "",
             "help": "Refinement sub-iterations (MCore's own default is 3). Set to 0 "
             "for a CHECK RUN: imports everything, refines nothing, reports the "
             "starting resolution. Blank = MCore's default."},
            {"name": "min_particles", "kind": "text", "flag": "--min_particles",
             "title": "Min particles per series",
             "default": "20",
             "help": "Skip tilt series with fewer than N particles in view. THE "
             "DEFAULT IN MCORE IS 1, which lets a series with a single particle into "
             "the refinement — far too little to constrain an image-warp grid plus "
             "stage angles, and a known cause of 'IndexOutOfRangeException' during "
             "refinement. 20+ is a safe floor; raise it if crashes persist."},
            {"name": "refine_imagewarp", "kind": "text", "flag": "--refine_imagewarp",
             "title": "Image warp grid",
             "default": "",
             "help": "Refine 2D image warp on an XxY grid, e.g. 6x4. The main "
             "tilt-series deformation correction and the usual first thing to enable. "
             "Blank = don't refine. A finer grid needs more particles per series."},
            {"name": "refine_particles", "kind": "check", "flag": "--refine_particles",
             "title": "Refine particle poses",
             "default": False, "help": "Refine per-particle poses."},
            {"name": "ctf_defocus", "kind": "check", "flag": "--ctf_defocus",
             "title": "Refine defocus",
             "default": False, "help": "Refine per-particle defocus (local search)."},
            {"name": "ctf_defocusexhaustive", "kind": "check",
             "title": "Exhaustive defocus search",
             "flag": "--ctf_defocusexhaustive", "default": False,
             "help": "Exhaustive defocus grid search in the FIRST sub-iteration. Only "
             "works together with ctf_defocus. Use on the first refinement only — "
             "afterwards the estimates are close and this is wasted time."},
            {"name": "refine_stageangles", "kind": "check", "flag": "--refine_stageangles",
             "title": "Refine stage angles",
             "default": False,
             "help": "Refine stage angles (tilt series only). Introduce AFTER image "
             "warp + particles + defocus are working — roughly round 4."},
            {"name": "refine_mag", "kind": "check", "flag": "--refine_mag",
             "title": "Refine magnification",
             "default": False,
             "help": "Refine anisotropic magnification. Late-stage (round 5+)."},
            {"name": "ctf_cs", "kind": "check", "flag": "--ctf_cs",
             "title": "Refine Cs",
             "default": False,
             "help": "Refine spherical aberration (also a proxy for pixel size). Late."},
            {"name": "ctf_zernike3", "kind": "check", "flag": "--ctf_zernike3",
             "title": "Zernike 3rd order",
             "default": False,
             "help": "Refine 3rd-order Zernike (beam tilt, trefoil). Fast. Late."},
            {"name": "ctf_zernike5", "kind": "check", "flag": "--ctf_zernike5",
             "title": "Zernike 5th order",
             "default": False, "help": "Refine 5th-order Zernike. Fast. Very late."},
            {"name": "ctf_zernike2", "kind": "check", "flag": "--ctf_zernike2",
             "title": "Zernike 2nd order (slow)",
             "default": False, "help": "Refine 2nd-order Zernike. SLOW."},
            {"name": "ctf_zernike4", "kind": "check", "flag": "--ctf_zernike4",
             "title": "Zernike 4th order (slow)",
             "default": False, "help": "Refine 4th-order Zernike. SLOW."},
            {"name": "ctf_phase", "kind": "check", "flag": "--ctf_phase",
             "title": "Refine phase shift",
             "default": False, "help": "Refine phase shift — PHASE PLATE data only."},
            {"name": "refine_volumewarp", "kind": "text", "flag": "--refine_volumewarp",
             "title": "Volume warp grid",
             "default": "",
             "help": "Volume warp on an XxYxZxT grid (tilt series only), e.g. "
             "'4x6x1x41'. Models deformation through the specimen AND time. Many "
             "parameters — only with strong data, and late."},
            {"name": "refine_tiltmovies", "kind": "check", "flag": "--refine_tiltmovies",
             "title": "Refine tilt movies",
             "default": False,
             "help": "Refine the alignments of the tilt MOVIES (needs the raw movie "
             "frames still present)."},
            {"name": "first_iteration_fraction", "kind": "text",
             "title": "First-iteration fraction",
             "flag": "--first_iteration_fraction", "default": "",
             "help": "Fraction of available resolution used for alignment in the first "
             "sub-iteration, rising to 1.0 by the last. Lower (e.g. 0.5) is a gentler "
             "start when the reference is poor. Blank = MCore default (1)."},
            {"name": "weight_threshold", "kind": "text", "flag": "--weight_threshold",
             "title": "Weight threshold",
             "default": "",
             "help": "Refine each tilt up to the resolution where exposure weighting "
             "falls to this value. Blank = MCore default (0.05)."},
            {"name": "ctf_minresolution", "kind": "text", "flag": "--ctf_minresolution",
             "title": "CTF min resolution (Å)",
             "default": "",
             "help": "Only use species at least this good (Å) for CTF refinement. "
             "Blank = MCore default (8)."},
            {"name": "ctf_batch", "kind": "text", "flag": "--ctf_batch",
             "title": "CTF batch size",
             "default": "",
             "help": "CTF refinement batch size. Lower = less GPU memory, higher = "
             "faster. Blank = MCore default (32). Drop this first on out-of-memory."},
            {"name": "cpu_memory", "kind": "check", "flag": "--cpu_memory",
             "title": "Particles in CPU RAM",
             "default": False,
             "help": "Hold particle images in CPU RAM instead of GPU. Slower, but the "
             "way out if the GPUs run out of memory."},
            {"name": "port", "kind": "text", "flag": "--port",
             "title": "REST API port",
             "default": "",
             "help": "REST API port (MCore's default is 14300). MCore FAILS TO START "
             "if the port is taken — 'Failed to bind to address ... address already in "
             "use' — which happens whenever a previous MCore crashed and left an "
             "orphaned process holding it. Set a different port (e.g. 14350) to dodge a "
             "stale one, or -1 to disable the API. Better: kill the orphans first "
             "(pkill -u $USER -f MCore; pkill -u $USER -f WarpWorker)."},
            {"name": "devicelist", "kind": "text", "flag": "--devicelist",
             "title": "GPUs",
             "gpu_sep": " ", "default": "",
             "help": "GPU ids, SPACE-separated (e.g. '1 2 3'). Blank = all GPUs. NOTE "
             "MCore spells this --devicelist, unlike WarpTools' --device_list."},
            {"name": "perdevice_refine", "kind": "slider_int",
             "title": "Refine workers per GPU",
             "flag": "--perdevice_refine", "default": 1, "min": 1, "max": 8, "step": 1,
             "help": "Refinement processes per GPU. Raise for utilisation if memory "
             "allows; drop to 1 on out-of-memory or CUFFT errors."},
            {"name": "perdevice_preprocess", "kind": "text",
             "title": "Pre-process workers per GPU",
             "flag": "--perdevice_preprocess", "default": "",
             "help": "Processes per GPU for map PRE-processing. Blank = same as "
             "perdevice_refine."},
            {"name": "perdevice_postprocess", "kind": "text",
             "title": "Post-process workers per GPU",
             "flag": "--perdevice_postprocess", "default": "",
             "help": "Processes per GPU for map POST-processing. Blank = same as "
             "perdevice_refine."},
        ],
        "validate": lambda v: (
            "⚠ Set population to the .population file (REQUIRED)."
            if not str(v.get("population", "")).strip() else
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
             "title": "Population file",
             "default": "", "help": "The .population file. REQUIRED."},
            {"name": "source", "kind": "text", "flag": "--source",
             "title": "Source name",
             "default": "", "help": "Data-source name (from 'create data source')."},
            {"name": "resolve", "kind": "choice", "flag": None,
             "title": "Weight granularity",
             "choices": [("Per tilt series (--resolve_items)", "--resolve_items"),
                         ("Per tilt, averaged over series (--resolve_frames)",
                          "--resolve_frames")],
             "default": "--resolve_items",
             "help": "Which exposure weights to estimate. Do PER-SERIES first, run MCore "
             "once, then per-tilt. They are separate passes, not alternatives."},
        ],
        "validate": lambda v: ("⚠ Set population to the .population file (REQUIRED)."
                               if not str(v.get("population", "")).strip() else
                               "⚠ Give the source name."
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
             "title": "Population file",
             "default": "", "help": "The .population file. REQUIRED."},
            {"name": "species", "kind": "text", "flag": "--species",
             "title": "Species file (full path)",
             "default": "",
             "help": "FULL path to the .species file, including its random hash — e.g. "
             "m/species/spike_797f75c2/spike.species. Find it with:  ls m/species/*/*.species"},
            {"name": "samples", "kind": "text", "flag": "--samples",
             "title": "Temporal samples",
             "default": "2",
             "help": "Temporal pose samples per tilt series. Go 1 → 2 only once the map "
             "is good; more samples need more signal per particle."},
        ],
        "validate": lambda v: (
            "⚠ Set population to the .population file (REQUIRED)."
            if not str(v.get("population", "")).strip() else
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
    {
        # MEMBRANE BRANCH (MemBrain + IsoNet module, spec draft 1). Starts from
        # reconstructed tomograms, ends in coordinates + priors for RELION.
        # Deconvolve is deliberately FIRST (highest value, lowest risk): it
        # boosts membrane contrast before membrain-seg / IsoNet. Its outputs are
        # a PICKING aid only — PROVENANCE.json tags them so extraction paths can
        # refuse them (§5 hard rule).
        "group": "12. Membrane", "id": "mb_deconv",
        "label": "Deconvolve (membrane contrast)",
        "base": "bash",
        "env_name": "membrainseg",      # conda env the wrapper activates
        "node_kind": "compute",         # no display needed; fine on els-03
        "params": [
            {"name": "script", "kind": "text", "flag": None, "title": "Wrapper script",
             "default": _pkg_script("ml_membrain_deconv_warp_auto.sh"),
             "help": "Deconvolution wrapper (shipped with the app). Wraps "
             "tomo_preprocessing deconvolve from the membrain-seg env."},
            {"name": "input_dir", "kind": "text", "flag": None,
             "title": "Tomogram folder",
             "default": "warp_tiltseries/reconstruction",
             "help": "Reconstructed tomograms. Must be the ORIGINAL "
             "reconstruction — the wrapper refuses dirs whose PROVENANCE.json "
             "says denoised/wedge-restored/deconvolved (double-processing "
             "degrades them)."},
            {"name": "output_dir", "kind": "text", "flag": None,
             "title": "Output folder",
             "default": "membrane/deconv",
             "help": "Output base. Each parameter variant writes its own "
             "s<strength>_f<falloff>/ subfolder with a PROVENANCE.json audit."},
            {"name": "MB_TOMO_LIST", "kind": "env", "flag": "MB_TOMO_LIST",
             "pick_from": "input_dir",
             "title": "Only these tomograms",
             "default": "",
             "help": "Blank = ALL tomograms (Batch mode). Name 1–3 stems "
             "('Position003 Position017') to sweep parameters on a few "
             "cheaply, compare in the viewer, then batch the winning value."},
            {"name": "MB_STRENGTH", "kind": "env", "flag": "MB_STRENGTH",
             "title": "Strength",
             "default": "1.0",
             "help": "THE contrast knob. A space-separated list sweeps "
             "('0.5 1.0 1.5' → one output folder per value)."},
            {"name": "MB_FALLOFF", "kind": "env", "flag": "MB_FALLOFF",
             "title": "Falloff",
             "default": "1.0",
             "help": "The other contrast knob. Also sweepable as a space-"
             "separated list."},
            {"name": "MB_PIXEL_SIZE", "kind": "env", "flag": "MB_PIXEL_SIZE",
             "title": "Pixel size (Å/px)",
             "default": "",
             "help": "Blank = read from the MRC header. The tool's own --help "
             "warns a wrong header causes severe errors — set explicitly "
             "(working tomograms here are 12.56) if headers have been "
             "unreliable."},
            {"name": "MB_KV", "kind": "env", "flag": "MB_KV", "default": "300",
             "title": "Voltage (kV)",
             "help": "Acceleration voltage of the TEM."},
            {"name": "MB_CS", "kind": "env", "flag": "MB_CS", "default": "2.7",
             "title": "Spherical aberration (mm)",
             "help": "Cs of the scope; 2.7 for the usual Krios optics."},
            {"name": "MB_AMPCON", "kind": "env", "flag": "MB_AMPCON",
             "title": "Amplitude contrast",
             "default": "0.07", "help": "Fraction between 0 and 1."},
            {"name": "MB_HP", "kind": "env", "flag": "MB_HP", "default": "0.02",
             "title": "Highpass fraction",
             "help": "Fraction of Nyquist cut off on the low end (it gets "
             "boosted the most)."},
            {"name": "MB_SKIP_LOWPASS", "kind": "env", "flag": "MB_SKIP_LOWPASS",
             "title": "Skip low-pass",
             "default": "",
             "help": "1 = pass --skip-lowpass: keep information beyond the "
             "first CTF zero. The tool marks this 'not recommended' — leave "
             "blank unless you know why."},
            {"name": "MB_XML_DIR", "kind": "env", "flag": "MB_XML_DIR",
             "title": "Warp XML folder",
             "default": "warp_tiltseries",
             "help": "Per-series .xml with the defocus. --df is parsed PER "
             "SERIES (Warp stores µm; converted ×10000 to Å and echoed in the "
             "log for sanity-checking). A series without a readable defocus "
             "is SKIPPED, never guessed."},
            {"name": "MB_CONDA_ENV", "kind": "env", "flag": "MB_CONDA_ENV",
             "title": "Conda env",
             "default": "membrainseg",
             "help": "Env with membrain-seg + tomo_preprocessing. Keep the "
             "three membrane envs separate (membrainseg / membrainpick / "
             "isonet) — their dependencies conflict."},
            {"name": "MB_FORCE", "kind": "env", "flag": "MB_FORCE",
             "title": "Redo existing",
             "default": "0",
             "help": "1 = recompute outputs that already exist (default "
             "skips them)."},
        ],
        "validate": lambda v: (
            "⚠ input_dir looks like a PROCESSED variant (denoised/IsoNet/"
            "deconvolved). Deconvolve only ORIGINAL reconstructions — "
            "double-processing degrades them."
            if any(t in str(v.get("input_dir", "")).lower()
                   for t in ("denois", "isonet", "deconv", "wedge")) else
            "⚠ Sweeping multiple values across ALL tomograms. Narrow it first: put "
            "1–3 stems in MB_TOMO_LIST, pick the winner visually, then batch "
            "ONE value."
            if (not str(v.get("MB_TOMO_LIST", "")).strip()
                and (len(str(v.get("MB_STRENGTH", "")).split()) > 1
                     or len(str(v.get("MB_FALLOFF", "")).split()) > 1))
            else ""),
        "output_params": ["output_dir"],
        "docs": {
            "what": "CTF-deconvolve reconstructed tomograms to boost membrane "
                    "contrast before membrain-seg / IsoNet (tomo_preprocessing "
                    "deconvolve, env membrainseg). The --df defocus is parsed "
                    "automatically per tilt series from warp_tiltseries/"
                    "<series>.xml (Warp stores µm → converted to Å ×10000) and "
                    "echoed per series in the log.",
            "range": "strength / falloff ≈ 0.5–1.5 are the knobs worth "
                     "sweeping; kv/cs/ampcon are scope constants.",
            "effect": "Stronger deconvolution = more low-frequency membrane "
                      "contrast but more halo. Sweep on 1–3 tomograms "
                      "(MB_TOMO_LIST), compare in the viewer, batch the winner.",
            "pitfall": "NEVER deconvolve denoised or wedge-restored volumes — "
                       "the wrapper refuses dirs tagged so in PROVENANCE.json. "
                       "Outputs are for SEGMENTATION/PICKING only: particle "
                       "extraction must reference the original reconstruction "
                       "(each output folder is tagged extraction_allowed: "
                       "false). Flags verified against the installed tool's "
                       "--help (membrainseg env, 2026-08-14); re-check after "
                       "any env update — these tools have renamed flags "
                       "before.",
        },
        "status": lambda ps: ps.status_mb_deconv(),
    },
    {
        # IsoNet 1 (§3.2): missing-wedge restoration. TRAIN on 1–5 tomograms
        # (IsoNet's own recommendation), then apply everywhere with the
        # predict step. IsoNet is a CLUSTER MODULE (module load isonet) —
        # NOT a conda env; the spec's "create an env" note predates this.
        # IsoNet 2 is not installed (isonet/latest == 0.3, no `denoise`
        # subcommand); if it ever lands, it becomes a sibling backend stage.
        "group": "12. Membrane", "id": "mb_isonet_train",
        "label": "IsoNet: train (1–5 tomos)",
        "base": "bash",
        "env_name": "isonet-module",
        "node_kind": "compute",
        "params": [
            {"name": "script", "kind": "text", "flag": None, "title": "Wrapper script",
             "default": _pkg_script("ml_isonet1_train_warp_auto.sh"),
             "help": "IsoNet 1 training chain (shipped with the app): "
             "prepare_star → deconv → make_mask → extract → refine, "
             "resumable, all inside the work dir."},
            {"name": "input_dir", "kind": "text", "flag": None,
             "title": "Tomogram folder",
             "default": "warp_tiltseries/reconstruction",
             "help": "The ORIGINAL reconstruction. IsoNet deconvolves "
             "internally — the wrapper refuses dirs tagged as processed "
             "variants."},
            {"name": "work_dir", "kind": "text", "flag": None,
             "title": "IsoNet work folder",
             "default": "membrane/isonet",
             "help": "Self-contained IsoNet project dir: tomograms.star, "
             "deconv/, mask/, subtomo/, results/ (the trained models)."},
            {"name": "ISO_TOMO_LIST", "kind": "env", "flag": "ISO_TOMO_LIST",
             "pick_from": "input_dir",
             "title": "Train on these (1–5)",
             "default": "",
             "help": "WHICH tomograms to train on, named by series stem "
             "('Position003 Position017') — 1–5 of them is a COUNT, not the "
             "first five by name: pick representative ones (IsoNet's own docs: "
             "'Usually 1-5 tomograms are sufficient'). 📂 Pick… lists the "
             ".mrc actually in the tomogram folder above. Blank trains on "
             "everything — slow and rarely better."},
            {"name": "ISO_GPU", "kind": "env", "flag": "ISO_GPU", "gpu_sep": ",",
             "title": "GPUs",
             "default": "0",
             "help": "Comma list for refine (e.g. 0,1,2,3). Batch size "
             "defaults to 2× GPU count."},
            {"name": "ISO_PIXEL_SIZE", "kind": "env", "flag": "ISO_PIXEL_SIZE",
             "title": "Pixel size (Å/px)",
             "default": "12.56",
             "help": "Star pixel size. IsoNet recommends ~10 Å/px working "
             "tomograms (target Z resolution ~30 Å); 12.56 is fine."},
            {"name": "ISO_SUBTOMOS", "kind": "env", "flag": "ISO_SUBTOMOS",
             "title": "Subtomos per tomogram",
             "default": "100",
             "help": "Training subtomograms extracted per tomogram. Lower it "
             "when masks exclude a lot of area."},
            {"name": "ISO_ITER", "kind": "env", "flag": "ISO_ITER",
             "title": "Training iterations",
             "default": "30",
             "help": "refine --iterations. Noise injection ramps at "
             "iterations 11/16/21/26 (tool defaults), so fewer than ~15 "
             "skips most of the denoising schedule."},
            {"name": "ISO_CUBE", "kind": "env", "flag": "ISO_CUBE",
             "title": "Cube size",
             "default": "64",
             "help": "Training cube edge, divisible by 8 (extracted subtomo "
             "= cube + 16)."},
            {"name": "ISO_NO_DECONV", "kind": "env", "flag": "ISO_NO_DECONV",
             "title": "Skip IsoNet deconv",
             "default": "",
             "help": "1 = skip IsoNet's internal CTF deconvolution (only for "
             "phase-plate data, per the tool docs)."},
            {"name": "ISO_DECONVSTRENGTH", "kind": "env",
             "flag": "ISO_DECONVSTRENGTH",
             "title": "Deconv strength",
             "default": "",
             "help": "deconv --deconvstrength; blank = tool default 1.0. Same "
             "knob as the Deconvolve step's Strength, but applied INSIDE "
             "IsoNet to the training input — it shapes what the model learns, "
             "so changing it means training again in a FRESH work folder, not "
             "re-running this one."},
            {"name": "ISO_SNRFALLOFF", "kind": "env", "flag": "ISO_SNRFALLOFF",
             "title": "Deconv falloff",
             "default": "",
             "help": "deconv --snrfalloff; blank = tool default 1.0. The "
             "other contrast knob (≈0.5–1.5 is the useful span)."},
            {"name": "ISO_DENSITY_PCT", "kind": "env", "flag": "ISO_DENSITY_PCT",
             "title": "Mask density %",
             "default": "",
             "help": "make_mask --density_percentage; blank = tool default "
             "50. Raise to keep only the densest voxels — where subtomograms "
             "get sampled from."},
            {"name": "ISO_STD_PCT", "kind": "env", "flag": "ISO_STD_PCT",
             "title": "Mask std %",
             "default": "",
             "help": "make_mask --std_percentage; blank = tool default 50. "
             "The local-variance half of the same mask."},
            {"name": "ISO_ZCROP", "kind": "env", "flag": "ISO_ZCROP",
             "title": "Mask Z-crop",
             "default": "",
             "help": "e.g. 0.2 masks the top and bottom 20% along Z — "
             "useful when the lamella floats mid-volume."},
            {"name": "ISO_PRETRAINED", "kind": "env", "flag": "ISO_PRETRAINED",
             "title": "Start from model",
             "default": "",
             "help": "A previous .h5 to fine-tune from (e.g. EML45's model "
             "for EML46 — the datasets are similar enough to transfer)."},
            {"name": "ISO_CONTINUE", "kind": "env", "flag": "ISO_CONTINUE",
             "title": "Resume from (json)",
             "default": "",
             "help": "The per-iteration .json refine writes — resumes an "
             "interrupted training instead of restarting."},
            {"name": "ISO_XML_DIR", "kind": "env", "flag": "ISO_XML_DIR",
             "title": "Warp XML folder",
             "default": "warp_tiltseries",
             "help": "Per-series defocus source: prepare_star writes ONE "
             "defocus for all rows, so the wrapper rewrites _rlnDefocus per "
             "tomogram from these xml (µm→Å, echoed, never guessed)."},
            {"name": "ISO_NCPU", "kind": "env", "flag": "ISO_NCPU",
             "title": "CPUs (deconv/mask)",
             "default": "",
             "help": "Blank = 8. Only the CPU steps use it; refine is GPU."},
            {"name": "ISO_MODULE", "kind": "env", "flag": "ISO_MODULE",
             "title": "Cluster module",
             "default": "isonet/0.3",
             "help": "lmod module (isonet is a module here, not a conda "
             "env). isonet/latest currently resolves to 0.3 too."},
            {"name": "ISO_FORCE", "kind": "env", "flag": "ISO_FORCE",
             "title": "Redo existing",
             "default": "0",
             "help": "1 = redo chain steps whose output already exists "
             "(default resumes). Each step now resumes per TOMOGRAM, so a "
             "changed tomo list recomputes what it needs without this; set it "
             "when you change a deconv/mask VALUE, which existing outputs "
             "cannot reveal."},
            {"name": "ISO_EXTRA_REFINE", "kind": "env", "flag": "ISO_EXTRA_REFINE",
             "title": "Extra refine flags",
             "default": "",
             "help": "Appended verbatim to `isonet.py refine` — an escape "
             "hatch for flags this form does not model (e.g. "
             "--noise_level_ramp). Wrong flags fail the run outright."},
        ],
        "validate": lambda v: (
            "⚠ Training on ALL tomograms — IsoNet's docs say 1–5 are "
            "sufficient. Put 1–5 representative stems in 'Train on'."
            if not str(v.get("ISO_TOMO_LIST", "")).strip() else
            "⚠ Input looks like a PROCESSED variant — IsoNet must train on "
            "the ORIGINAL reconstruction (it deconvolves internally)."
            if any(t in str(v.get("input_dir", "")).lower()
                   for t in ("denois", "isonet", "deconv", "wedge"))
            else ""),
        "output_params": ["work_dir"],
        "docs": {
            "what": "Train an IsoNet 1 missing-wedge model on a few "
                    "representative tomograms (isonet.py: prepare_star → "
                    "deconv → make_mask → extract → refine). Per-series "
                    "defocus is injected into the star automatically from "
                    "the Warp xml.",
            "range": "1–5 training tomograms; 30 iterations; cube 64; "
                     "subtomos ~100 (fewer if masks are tight).",
            "effect": "Produces results/model_iter*.h5 — reusable across "
                      "the dataset and likely across EML45↔EML46. Resumable: "
                      "each chain step skips if its output exists.",
            "pitfall": "Train on the ORIGINAL reconstruction only (own "
                       "deconv inside). The refine step is hours on GPU. "
                       "Flags verified against isonet.py 0.3 -h "
                       "(2026-08-15).",
        },
        "status": lambda ps: ps.status_mb_isonet_train(),
    },
    {
        "group": "12. Membrane", "id": "mb_isonet_predict",
        "label": "IsoNet: predict (apply model)",
        "base": "bash",
        "env_name": "isonet-module",
        "node_kind": "compute",
        "params": [
            {"name": "script", "kind": "text", "flag": None, "title": "Wrapper script",
             "default": _pkg_script("ml_isonet1_predict_warp_auto.sh"),
             "help": "Applies a trained model to tomograms (isonet.py "
             "predict) — the 'predict many' half."},
            {"name": "input_dir", "kind": "text", "flag": None,
             "title": "Tomogram folder",
             "default": "warp_tiltseries/reconstruction",
             "help": "Tomograms to correct — the ORIGINAL reconstruction."},
            {"name": "output_dir", "kind": "text", "flag": None,
             "title": "Output folder",
             "default": "membrane/isonet_corrected",
             "help": "Corrected tomograms land in <here>/corrected/, tagged "
             "wedge-restored: PICKING ONLY, never extraction input."},
            {"name": "ISO_MODEL", "kind": "env", "flag": "ISO_MODEL",
             "title": "Trained model (.h5)",
             "default": "",
             "help": "REQUIRED. e.g. membrane/isonet/results/model_iter30.h5. "
             "Deliberately independent of the data path — a reused config "
             "whose data dir pointed at the wrong dataset caused a real bug "
             "once; both resolved paths are echoed at the top of the log, "
             "and the run warns when the tomogram count doesn't match the "
             "project's tilt-series count."},
            {"name": "ISO_TOMO_LIST", "kind": "env", "flag": "ISO_TOMO_LIST",
             "pick_from": "input_dir",
             "title": "Only these tomograms",
             "default": "",
             "help": "Blank = ALL (the normal predict-many run). Name stems "
             "to test the model on a subset first."},
            {"name": "ISO_GPU", "kind": "env", "flag": "ISO_GPU", "gpu_sep": ",",
             "title": "GPUs",
             "default": "0", "help": "Comma list for prediction."},
            {"name": "ISO_PIXEL_SIZE", "kind": "env", "flag": "ISO_PIXEL_SIZE",
             "title": "Pixel size (Å/px)",
             "default": "12.56", "help": "Must match the training pixel size."},
            {"name": "ISO_CUBE", "kind": "env", "flag": "ISO_CUBE",
             "title": "Cube size",
             "default": "",
             "help": "Blank = tool default 64. Match the CUBE THE MODEL WAS "
             "TRAINED WITH — predicting with a different one is a silent "
             "quality loss, not an error."},
            {"name": "ISO_CROP", "kind": "env", "flag": "ISO_CROP",
             "title": "Crop size",
             "default": "",
             "help": "Blank = tool default 96. Raise if the corrected "
             "volumes show patchy artifacts."},
            {"name": "ISO_MODULE", "kind": "env", "flag": "ISO_MODULE",
             "title": "Cluster module",
             "default": "isonet/0.3", "help": "lmod module name."},
            {"name": "ISO_FORCE", "kind": "env", "flag": "ISO_FORCE",
             "title": "Redo existing",
             "default": "0", "help": "1 = re-predict even when every "
             "selected tomogram already has a corrected output."},
        ],
        "validate": lambda v: (
            "⚠ Trained model (.h5) is REQUIRED — train first (IsoNet: train), "
            "then point this at results/model_iter<N>.h5."
            if not str(v.get("ISO_MODEL", "")).strip() else ""),
        "output_params": ["output_dir"],
        "docs": {
            "what": "Apply a trained IsoNet model across the dataset "
                    "(isonet.py predict). Train once, predict many — the "
                    "model transfers across similar datasets (EML45↔EML46).",
            "range": "GPU count scales speed; crop 96 → larger only against "
                     "patch artifacts.",
            "effect": "Writes corrected/ tomograms with the missing wedge "
                      "restored — membranes continuous top to bottom, ideal "
                      "for segmentation and picking.",
            "pitfall": "WEDGE-RESTORED VOLUMES ARE FOR COORDINATES ONLY — "
                       "extraction for averaging must use the original "
                       "reconstruction (PROVENANCE.json enforces the refusal "
                       "downstream). Flags verified against isonet.py 0.3 -h "
                       "(2026-08-15).",
        },
        "status": lambda ps: ps.status_mb_isonet_predict(),
    },
    {
        # IsoNet 2 (§3.2's second backend). SIBLING stages, not a version
        # selector inside the IsoNet 1 node: the spec asked for one node whose
        # panel changes with the version, but the params are static per stage
        # here, and the two versions barely share a knob (epochs vs iterations,
        # .pt vs .h5, no extract step, method auto-detects n2n). Two stages
        # state that honestly; the labels carry the version.
        #
        # NOT the isonet/0.3 module: IsoNet 2 is a conda PREFIX env built by
        # ml_isonet2_setup.sh. `conda activate -n <name>` finds nothing.
        "group": "12. Membrane", "id": "mb_isonet2_train",
        "label": "IsoNet 2: train (1–5 tomos)",
        "base": "bash",
        "env_name": "isonet2-prefix",
        "node_kind": "compute",
        "params": [
            {"name": "script", "kind": "text", "flag": None, "title": "Wrapper script",
             "default": _pkg_script("ml_isonet2_train_warp_auto.sh"),
             "help": "IsoNet 2 training chain (shipped with the app): "
             "prepare_star → deconv → make_mask → refine, resumable, all "
             "inside the work dir. No extract step — IsoNet 2 folded "
             "subtomogram extraction into refine."},
            {"name": "input_dir", "kind": "text", "flag": None,
             "title": "Tomogram folder",
             "default": "warp_tiltseries/reconstruction",
             "help": "The ORIGINAL reconstruction. IsoNet deconvolves "
             "internally — the wrapper refuses dirs tagged as processed "
             "variants."},
            {"name": "work_dir", "kind": "text", "flag": None,
             "title": "IsoNet 2 work folder",
             "default": "membrane/isonet2",
             "help": "Self-contained project dir: tomograms.star, deconv/, "
             "mask/, isonet_maps/ (the .pt checkpoints). Kept separate from "
             "membrane/isonet so both versions' results coexist."},
            {"name": "ISO2_TOMO_LIST", "kind": "env", "flag": "ISO2_TOMO_LIST",
             "pick_from": "input_dir",
             "title": "Train on these (1–5)",
             "default": "",
             "help": "WHICH tomograms to train on, named by series stem — 1–5 "
             "of them is a COUNT, not the first five by name ('Usually 1-5 "
             "tomograms are sufficient', their prepare_star help). 📂 Pick… "
             "lists the .mrc in the folder above. Blank trains on everything."},
            {"name": "ISO2_GPU", "kind": "env", "flag": "ISO2_GPU", "gpu_sep": ",",
             "title": "GPUs",
             "default": "0",
             "help": "--gpuID comma list (e.g. 0,1,2,3). Batch size defaults "
             "to 2× GPU count (4 on a single GPU)."},
            {"name": "ISO2_EVEN_DIR", "kind": "env", "flag": "ISO2_EVEN_DIR",
             "title": "Even halves (optional)",
             "default": "",
             "help": "Even half-tomograms for noise2noise. Set BOTH halves or "
             "neither. Half-tomograms come from ts_reconstruct "
             "--halfmap_frames (needs the frameseries step re-run with "
             "--average_halves) and land in <that job's output>/"
             "reconstruction/even/ — e.g. jobs/J16_reconstruct-full-tomograms/reconstruction/even. Blank "
             "= single-map training, which is fully supported."},
            {"name": "ISO2_ODD_DIR", "kind": "env", "flag": "ISO2_ODD_DIR",
             "title": "Odd halves (optional)",
             "default": "",
             "help": "Odd half-tomograms (…/reconstruction/odd/). The wrapper "
             "refuses one without the other — a half-configured pair would "
             "silently train single-map."},
            {"name": "ISO2_PIXEL_SIZE", "kind": "env", "flag": "ISO2_PIXEL_SIZE",
             "title": "Pixel size (Å/px)",
             "default": "12.56",
             "help": "Or 'auto' to read the mrc header. IsoNet wants ~10 Å/px "
             "working tomograms (target Z resolution ~30 Å); 12.56 is fine."},
            {"name": "ISO2_SUBTOMOS", "kind": "env", "flag": "ISO2_SUBTOMOS",
             "title": "Subtomos per tomogram",
             "default": "auto",
             "help": "prepare_star --number_subtomos. 'auto' lets IsoNet 2 "
             "size the training set itself; a number overrides it (lower when "
             "masks exclude a lot of area)."},
            {"name": "ISO2_EPOCHS", "kind": "env", "flag": "ISO2_EPOCHS",
             "title": "Epochs",
             "default": "",
             "help": "refine --epochs. Blank = tool default 50. IsoNet 2 "
             "trains EPOCHS, not IsoNet 1's iterations — the two numbers are "
             "not comparable."},
            {"name": "ISO2_ARCH", "kind": "env", "flag": "ISO2_ARCH",
             "title": "Architecture",
             "default": "",
             "help": "Blank = tool default unet-medium. Also unet-small, "
             "unet-large, scunet-fast. Bigger = more VRAM."},
            {"name": "ISO2_CUBE", "kind": "env", "flag": "ISO2_CUBE",
             "title": "Cube size",
             "default": "",
             "help": "Blank = tool default 96. Must divide by the network's "
             "downsampling factors, so stick to multiples of 8."},
            {"name": "ISO2_METHOD", "kind": "env", "flag": "ISO2_METHOD",
             "title": "Method",
             "default": "auto",
             "help": "auto, isonet2 (single map), isonet2-n2n (noise2noise on "
             "halves). With halves the wrapper resolves auto → isonet2-n2n "
             "ITSELF: the star then holds both the pairs and the averaged "
             "full volumes, and refine refuses to choose — it raises after "
             "deconv and make_mask have already run. Set isonet2 explicitly "
             "to train single-map on the averaged volumes instead."},
            {"name": "ISO2_REFINE_INPUT_COL", "kind": "env",
             "flag": "ISO2_REFINE_INPUT_COL",
             "title": "Train on column",
             "default": "rlnTomoName",
             "help": "Which volumes a SINGLE-MAP model trains on. IsoNet's own "
             "default is rlnDeconvTomoName, but the predict step builds its "
             "star from RAW tomograms and never deconvolves — so a model "
             "trained on deconvolved volumes would be applied to "
             "undeconvolved ones. rlnTomoName keeps training and prediction "
             "on the same footing; deconv still runs, because make_mask wants "
             "it. Ignored by noise2noise models, which read the halves."},
            {"name": "ISO2_CTF_MODE", "kind": "env", "flag": "ISO2_CTF_MODE",
             "title": "CTF mode",
             "default": "",
             "help": "Blank = tool default None. phase_only / network / "
             "wiener move CTF correction INTO the network — with network or "
             "wiener, consider skipping the deconv step below."},
            {"name": "ISO2_NO_DECONV", "kind": "env", "flag": "ISO2_NO_DECONV",
             "title": "Skip deconv step",
             "default": "",
             "help": "1 = skip CTF deconvolution (phase-plate data, or when "
             "CTF mode already corrects inside the network)."},
            {"name": "ISO2_ZCROP", "kind": "env", "flag": "ISO2_ZCROP",
             "title": "Mask Z-crop",
             "default": "",
             "help": "Blank = tool default 0.2 (crops the top and bottom "
             "10% each) — useful when the lamella floats mid-volume."},
            {"name": "ISO2_BFACTOR", "kind": "env", "flag": "ISO2_BFACTOR",
             "title": "B-factor",
             "default": "",
             "help": "Blank = tool default 0, which is what they recommend "
             "for CELLULAR tomograms. 200–300 is for isolated samples."},
            {"name": "ISO2_PRETRAINED", "kind": "env", "flag": "ISO2_PRETRAINED",
             "title": "Start from model",
             "default": "",
             "help": "A previous .pt to fine-tune from. Its method, arch, "
             "cube_size and CTF_mode are loaded WITH it, overriding the "
             "settings above."},
            {"name": "ISO2_XML_DIR", "kind": "env", "flag": "ISO2_XML_DIR",
             "title": "Warp XML folder",
             "default": "warp_tiltseries",
             "help": "Per-series defocus source. IsoNet 2's --defocus takes a "
             "list, but a list binds values to star ROW ORDER — the wrapper "
             "matches by NAME instead (µm→Å, echoed, never guessed)."},
            {"name": "ISO2_ENV", "kind": "env", "flag": "ISO2_ENV",
             "title": "Conda env prefix",
             "default": "",
             "help": "Blank = <processing_scripts>/IsoNet2/build/conda_env "
             "for your user. A PREFIX PATH, not a name: upstream's install.sh "
             "builds the env inside its own tree, so `conda activate -n "
             "isonet2_environment` finds nothing. Install with "
             "ml_isonet2_setup.sh."},
            {"name": "ISO2_FORCE", "kind": "env", "flag": "ISO2_FORCE",
             "title": "Redo existing",
             "default": "0",
             "help": "1 = redo chain steps whose output already exists "
             "(default resumes per tomogram)."},
        ],
        "validate": lambda v: (
            "⚠ Training on ALL tomograms — 1–5 representative ones are what "
            "IsoNet recommends. Put 1–5 stems in 'Train on'."
            if not str(v.get("ISO2_TOMO_LIST", "")).strip() else
            "⚠ Set BOTH even and odd half folders, or neither. One alone "
            "trains single-map while looking like a noise2noise run."
            if (bool(str(v.get("ISO2_EVEN_DIR", "")).strip())
                != bool(str(v.get("ISO2_ODD_DIR", "")).strip())) else
            "⚠ Method isonet2-n2n needs even/odd half tomograms — give both "
            "half folders, or leave the method on auto."
            if (str(v.get("ISO2_METHOD", "")).strip() == "isonet2-n2n"
                and not str(v.get("ISO2_EVEN_DIR", "")).strip()) else
            "⚠ Input looks like a PROCESSED variant — IsoNet must train on "
            "the ORIGINAL reconstruction (it deconvolves internally)."
            if any(t in str(v.get("input_dir", "")).lower()
                   for t in ("denois", "isonet", "deconv", "wedge"))
            else ""),
        "output_params": ["work_dir"],
        "docs": {
            "what": "Train an IsoNet 2 model (PyTorch rewrite) on a few "
                    "representative tomograms: prepare_star → deconv → "
                    "make_mask → refine. Missing-wedge correction, denoising "
                    "and CTF handling happen in ONE optimisation loop, so "
                    "there is no separate extract step.",
            "range": "1–5 training tomograms; epochs 50, cube 96, "
                     "unet-medium are the tool's defaults.",
            "effect": "Writes isonet_maps/*.pt — reusable across the dataset "
                      "and likely across EML45↔EML46. Resumable: each chain "
                      "step skips when it already covers every selected "
                      "tomogram, and refuses to reuse a folder computed with "
                      "different deconv/mask values.",
            "pitfall": "Train on the ORIGINAL reconstruction only. Checkpoints "
                       "are .pt — the IsoNet 1 predict step wants .h5 and the "
                       "two never mix. Flags verified against IsoNet2 2.0.1b0 "
                       "(isonet2_helps.txt, 2026-08-15); its CLI is "
                       "python-fire, where -h means --highpassnyquist, so the "
                       "wrapper only ever uses long flags.",
        },
        "status": lambda ps: ps.status_mb_isonet2_train(),
    },
    {
        "group": "12. Membrane", "id": "mb_isonet2_predict",
        "label": "IsoNet 2: predict (apply model)",
        "base": "bash",
        "env_name": "isonet2-prefix",
        "node_kind": "compute",
        "params": [
            {"name": "script", "kind": "text", "flag": None, "title": "Wrapper script",
             "default": _pkg_script("ml_isonet2_predict_warp_auto.sh"),
             "help": "Applies a trained IsoNet 2 checkpoint to tomograms "
             "(isonet.py predict) — the 'predict many' half."},
            {"name": "input_dir", "kind": "text", "flag": None,
             "title": "Tomogram folder",
             "default": "warp_tiltseries/reconstruction",
             "help": "Tomograms to correct — the ORIGINAL reconstruction."},
            {"name": "output_dir", "kind": "text", "flag": None,
             "title": "Output folder",
             "default": "membrane/isonet2_corrected",
             "help": "Corrected tomograms land in <here>/corrected/, tagged "
             "wedge-restored: PICKING ONLY, never extraction input."},
            {"name": "ISO2_MODEL", "kind": "env", "flag": "ISO2_MODEL",
             "title": "Trained model (.pt)",
             "default": "",
             "help": "REQUIRED. e.g. membrane/isonet2/isonet_maps/<name>.pt. "
             "Deliberately independent of the data path — a reused config "
             "whose data dir pointed at the wrong dataset caused a real bug "
             "once; both resolved paths are echoed at the top of the log. An "
             "IsoNet 1 .h5 is rejected outright."},
            {"name": "ISO2_TOMO_LIST", "kind": "env", "flag": "ISO2_TOMO_LIST",
             "pick_from": "input_dir",
             "title": "Only these tomograms",
             "default": "",
             "help": "Blank = ALL (the normal predict-many run). Name stems "
             "to test the model on a subset first."},
            {"name": "ISO2_EVEN_DIR", "kind": "env", "flag": "ISO2_EVEN_DIR",
             "title": "Even halves (n2n models)",
             "default": "",
             "help": "REQUIRED for a noise2noise model: IsoNet predicts with "
             "n2n models FROM THE HALVES (it reads "
             "rlnTomoReconstructedTomogramHalf1/2 and ignores the input "
             "column). Same folder the training used, e.g. "
             "jobs/J16_reconstruct-full-tomograms/reconstruction/even. Only series that HAVE halves can "
             "be corrected by such a model. Leave blank for a single-map "
             "(isonet2) model."},
            {"name": "ISO2_ODD_DIR", "kind": "env", "flag": "ISO2_ODD_DIR",
             "title": "Odd halves (n2n models)",
             "default": "",
             "help": "The other half (…/reconstruction/odd). Both or neither."},
            {"name": "ISO2_GPU", "kind": "env", "flag": "ISO2_GPU", "gpu_sep": ",",
             "title": "GPUs",
             "default": "0", "help": "--gpuID comma list for prediction."},
            {"name": "ISO2_PIXEL_SIZE", "kind": "env", "flag": "ISO2_PIXEL_SIZE",
             "title": "Pixel size (Å/px)",
             "default": "12.56", "help": "Must match the training pixel size."},
            {"name": "ISO2_INPUT_COL", "kind": "env", "flag": "ISO2_INPUT_COL",
             "title": "Star input column",
             "default": "rlnTomoName",
             "help": "NOT the tool's own default (rlnDeconvTomoName): this "
             "step builds its star from RAW tomograms, and prepare_star fills "
             "the deconv column with the literal string 'None' — predict then "
             "tries to open a file called 'None'. rlnTomoName is also what an "
             "isonet2-n2n model trained on (refine reads the raw halves; "
             "deconv only feeds mask generation). The wrapper checks the "
             "column holds real paths before starting."},
            {"name": "ISO2_PADDING", "kind": "env", "flag": "ISO2_PADDING",
             "title": "Padding factor",
             "default": "",
             "help": "Blank = tool default 1.5. Raise if the corrected "
             "volumes show tile seams; costs compute."},
            {"name": "ISO2_XML_DIR", "kind": "env", "flag": "ISO2_XML_DIR",
             "title": "Warp XML folder",
             "default": "warp_tiltseries",
             "help": "Per-series defocus for the prediction star — a model "
             "with CTF handling reads it from these rows, so it is injected "
             "by name (µm→Å) rather than left at prepare_star's placeholder."},
            {"name": "ISO2_ENV", "kind": "env", "flag": "ISO2_ENV",
             "title": "Conda env prefix",
             "default": "",
             "help": "Blank = <processing_scripts>/IsoNet2/build/conda_env "
             "for your user. A PREFIX PATH, not an env name."},
            {"name": "ISO2_FORCE", "kind": "env", "flag": "ISO2_FORCE",
             "title": "Redo existing",
             "default": "0", "help": "1 = re-predict even when every "
             "selected tomogram already has a corrected output."},
        ],
        "validate": lambda v: (
            "⚠ Trained model (.pt) is REQUIRED — train first (IsoNet 2: "
            "train), then point this at isonet_maps/<name>.pt."
            if not str(v.get("ISO2_MODEL", "")).strip() else
            "⚠ That is an IsoNet 1 model (.h5). Use the 'IsoNet: predict' "
            "step for it — IsoNet 2 predicts only from its own .pt."
            if str(v.get("ISO2_MODEL", "")).strip().endswith(".h5") else
            # n2n models read the halves, not the full tomogram. Without them
            # predict opens a file called 'None' after the star is built.
            "⚠ This is an n2n model — IsoNet predicts with it FROM THE HALVES. "
            "Set both half folders (e.g. jobs/J16_…/reconstruction/even and "
            "/odd), or use a single-map (isonet2) model to correct from full "
            "tomograms."
            if ("n2n" in str(v.get("ISO2_MODEL", ""))
                and not str(v.get("ISO2_EVEN_DIR", "")).strip()) else
            "⚠ Set BOTH half folders, or neither."
            if (bool(str(v.get("ISO2_EVEN_DIR", "")).strip())
                != bool(str(v.get("ISO2_ODD_DIR", "")).strip())) else ""),
        "output_params": ["output_dir"],
        "docs": {
            "what": "Apply a trained IsoNet 2 checkpoint across the dataset "
                    "(isonet.py predict). Train once, predict many.",
            "range": "GPU count scales speed; padding 1.5 → larger only "
                     "against tile seams.",
            "effect": "Writes corrected/ tomograms with the missing wedge "
                      "restored and (for n2n models) denoised — membranes "
                      "continuous top to bottom, ideal for segmentation.",
            "pitfall": "CORRECTED VOLUMES ARE FOR COORDINATES ONLY — "
                       "extraction for averaging must use the original "
                       "reconstruction (PROVENANCE.json enforces the refusal "
                       "downstream). Flags verified against IsoNet2 2.0.1b0 "
                       "(2026-08-15).",
        },
        "status": lambda ps: ps.status_mb_isonet2_predict(),
    },
    {
        # GPU segmentation. Score maps are the load-bearing output: with
        # *_scores.mrc on disk every re-threshold is a free CPU pass; without
        # them it is a 4-min/tomogram GPU re-run. Hence stored by default and
        # only disableable through a deliberately awkward env knob.
        "group": "12. Membrane", "id": "mb_segment",
        "label": "Segment membranes",
        "base": "bash",
        "env_name": "membrainseg",
        "node_kind": "compute",
        "params": [
            {"name": "script", "kind": "text", "flag": None, "title": "Wrapper script",
             "default": _pkg_script("ml_membrain_segment_warp_auto.sh"),
             "help": "Segmentation wrapper (shipped with the app). Wraps "
             "membrain segment from the membrain-seg env; ~4 min/tomogram "
             "with TTA on one GPU."},
            {"name": "input_dir", "kind": "text", "flag": None,
             "title": "Tomogram folder",
             "default": "membrane/deconv/s1.0_f1.0",
             "help": "Tomograms to segment — usually a deconvolved variant "
             "(better contrast); the original reconstruction also works. "
             "Segmentation is a COORDINATE path, so processed inputs are "
             "fine here."},
            {"name": "output_dir", "kind": "text", "flag": None,
             "title": "Output folder",
             "default": "membrane/segment",
             "help": "Segmentations + *_scores.mrc land here (always passed "
             "explicitly — the tool's own default ./predictions would drop "
             "them next to whatever you last ran). ONE folder per input "
             "variant: the wrapper refuses to mix (e.g. use "
             "membrane/segment_raw for the original reconstruction)."},
            {"name": "MB_CKPT", "kind": "env", "flag": "MB_CKPT",
             "title": "Model checkpoint",
             "default": "",
             "help": "The membrain-seg model to run. REQUIRED, and NOT "
             "something you train first: membrain segment is inference-only "
             "and ships no weights, so it needs a checkpoint pointed at it. "
             "Use the official pretrained release — on this cluster that is "
             "<processing_scripts>/membrain-seg/MemBrain_seg_v10_beta.ckpt. "
             "Your own fine-tuned model goes here instead only if the "
             "pretrained one underperforms on this sample."},
            {"name": "MB_TOMO_LIST", "kind": "env", "flag": "MB_TOMO_LIST",
             "pick_from": "input_dir",
             "title": "Only these tomograms",
             "default": "",
             "help": "Blank = every tomogram in the folder. Name series stems to "
             "subset first."},
            {"name": "MB_GPU", "kind": "env", "flag": "MB_GPU", "gpu_sep": ",",
             "title": "GPU",
             "default": "0",
             "help": "CUDA_VISIBLE_DEVICES value — which GPU runs the model."},
            {"name": "MB_PIXEL_SIZE", "kind": "env", "flag": "MB_PIXEL_SIZE",
             "title": "Pixel size (Å/px)",
             "default": "12.56",
             "help": "Input tomogram pixel size; with patch rescaling on, "
             "patches are rescaled to the model's 10 Å/px on the fly."},
            {"name": "MB_WINDOW", "kind": "env", "flag": "MB_WINDOW",
             "title": "Sliding window",
             "default": "",
             "help": "Blank = tool default 160. Smaller uses less GPU memory "
             "but the tool warns results get WORSE — lower only on OOM."},
            {"name": "MB_NO_TTA", "kind": "env", "flag": "MB_NO_TTA",
             "title": "Disable TTA",
             "default": "",
             "help": "1 = skip 8-fold test-time augmentation: faster, "
             "slightly worse segmentations."},
            {"name": "MB_UNCERTAINTY", "kind": "env", "flag": "MB_UNCERTAINTY",
             "title": "Uncertainty map",
             "default": "",
             "help": "1 = also store a TTA-variance uncertainty map (needs "
             "TTA on). Note the flag is --store-uncertainty-map — older docs "
             "call it --store-uncertainty, which no longer exists."},
            {"name": "MB_NO_PROBS", "kind": "env", "flag": "MB_NO_PROBS",
             "title": "Disable score maps",
             "default": "",
             "help": "1 = do NOT store *_scores.mrc. STRONGLY discouraged: "
             "without score maps every re-threshold is a full GPU re-run "
             "instead of a free CPU pass."},
            {"name": "MB_CONDA_ENV", "kind": "env", "flag": "MB_CONDA_ENV",
             "title": "Conda env",
             "default": "membrainseg",
             "help": "Env with membrain-seg. Never merge the membrane envs."},
            {"name": "MB_FORCE", "kind": "env", "flag": "MB_FORCE",
             "title": "Redo existing",
             "default": "0",
             "help": "1 = re-segment tomograms whose score map already "
             "exists."},
        ],
        "validate": lambda v: (
            "⚠ Model checkpoint (MB_CKPT) is REQUIRED — the wrapper refuses "
            "to start without one."
            if not str(v.get("MB_CKPT", "")).strip() else
            "⚠ Score maps disabled (MB_NO_PROBS=1): every re-threshold will "
            "cost a full GPU re-run. You almost never want this."
            if str(v.get("MB_NO_PROBS", "")).strip() == "1" else
            "⚠ Uncertainty map needs TTA — unset 'Disable TTA' or drop the "
            "uncertainty map."
            if (str(v.get("MB_UNCERTAINTY", "")).strip() == "1"
                and str(v.get("MB_NO_TTA", "")).strip() == "1")
            else ""),
        "output_params": ["output_dir"],
        "docs": {
            "what": "Segment membranes with membrain-seg (GPU). Writes a "
                    "segmentation plus a *_scores.mrc score map per tomogram; "
                    "the score map is what the whole tuning loop downstream "
                    "feeds on.",
            "range": "sliding window 128–160 (lower only on OOM); "
                     "segmentation threshold usually swept LATER on the score "
                     "map, not here.",
            "effect": "TTA on = ~4 min/tomogram/GPU, slightly better maps. "
                      "Try 1–3 tomograms, then run all 72.",
            "pitfall": "Score maps ON is the default and should stay on — "
                       "*_scores.mrc makes re-thresholding free. Flags "
                       "verified against membrain-seg 0.0.10 (2026-08-14); "
                       "--store-uncertainty-map is the current spelling.",
        },
        "status": lambda ps: ps.status_mb_segment(),
    },
    {
        # The main tuning loop: CPU, seconds per map, zero GPU.
        "group": "12. Membrane", "id": "mb_thresholds",
        "label": "Threshold sweep",
        "base": "bash",
        "env_name": "membrainseg",
        "node_kind": "compute",
        "params": [
            {"name": "script", "kind": "text", "flag": None, "title": "Wrapper script",
             "default": _pkg_script("ml_membrain_thresholds_warp_auto.sh"),
             "help": "Threshold wrapper (shipped with the app). Wraps "
             "membrain thresholds — CPU, seconds per score map."},
            {"name": "input_dir", "kind": "text", "flag": None,
             "title": "Score-map folder",
             "default": "membrane/segment",
             "help": "Folder of *_scores.mrc from the Segment step."},
            {"name": "output_dir", "kind": "text", "flag": None,
             "title": "Output folder",
             "default": "membrane/thresholds",
             "help": "ALWAYS passed explicitly: the tool's default "
             "./predictions does not follow the scoremap's folder, so "
             "outputs would silently land beside a different run's files."},
            {"name": "MB_THRESHOLDS", "kind": "env", "flag": "MB_THRESHOLDS",
             "title": "Thresholds",
             "default": "0.0",
             "help": "Space-separated sweep list, e.g. '-1.5 -0.5 0.0 0.5'. "
             "Each value becomes its own --thresholds flag (the tool takes "
             "repeated flags, not a comma list). Scores are normalised to "
             "the background mean/SD, so values are comparable across "
             "tomograms."},
            {"name": "MB_TOMO_LIST", "kind": "env", "flag": "MB_TOMO_LIST",
             "pick_from": "input_dir",
             "title": "Only these tomograms",
             "default": "",
             "help": "Blank = every score map. Name stems to threshold a "
             "subset."},
            {"name": "MB_CONDA_ENV", "kind": "env", "flag": "MB_CONDA_ENV",
             "title": "Conda env",
             "default": "membrainseg", "help": "Env with membrain-seg."},
            {"name": "MB_FORCE", "kind": "env", "flag": "MB_FORCE",
             "title": "Redo existing",
             "default": "0", "help": "1 = re-threshold maps whose first "
             "output already exists."},
        ],
        "validate": lambda v: (
            "⚠ Thresholds list is empty — give at least one value "
            "(e.g. '0.0', or a sweep '-1.5 -0.5 0.0 0.5')."
            if not str(v.get("MB_THRESHOLDS", "")).strip() else ""),
        "output_params": ["output_dir"],
        "docs": {
            "what": "Threshold membrane score maps into binary segmentations "
                    "(membrain thresholds). Cheap CPU pass — THE tuning loop: "
                    "segment once on GPU, re-threshold as often as needed for "
                    "free.",
            "range": "sweep -1.5 … +0.5 to start; scores are background-"
                     "normalised so one good value usually generalises.",
            "effect": "Lower threshold = more membrane kept (and more noise). "
                      "Output names normalise the value to one decimal: -3 → "
                      "..._threshold_-3.0.mrc — downstream steps must build "
                      "names that way.",
            "pitfall": "Never rely on the tool's default out-folder "
                       "(./predictions) — this stage always passes it "
                       "explicitly. Flags verified against membrain-seg "
                       "0.0.10 (2026-08-14).",
        },
        "status": lambda ps: ps.status_mb_thresholds(),
    },
    {
        "group": "12. Membrane", "id": "mb_components",
        "label": "Connected components",
        "base": "bash",
        "env_name": "membrainseg",
        "node_kind": "compute",
        "params": [
            {"name": "script", "kind": "text", "flag": None, "title": "Wrapper script",
             "default": _pkg_script("ml_membrain_components_warp_auto.sh"),
             "help": "Components wrapper (shipped with the app). Wraps "
             "membrain components and parses its Found/Relabeled counts "
             "into components_metrics.tsv."},
            {"name": "input_dir", "kind": "text", "flag": None,
             "title": "Segmentation folder",
             "default": "membrane/thresholds",
             "help": "Thresholded segmentations (or any binary membrane "
             "volumes) to label."},
            {"name": "output_dir", "kind": "text", "flag": None,
             "title": "Output folder",
             "default": "membrane/components",
             "help": "Base dir; each swept cutoff writes its own cc<voxels>/ "
             "subfolder. The wrapper mkdir -p's it first — the tool computes "
             "everything and then DIES if the folder is missing."},
            {"name": "MB_CC_THRES", "kind": "env", "flag": "MB_CC_THRES",
             "title": "Min component size",
             "default": "50",
             "help": "Components smaller than this are removed. Same physical "
             "spellings as the sweep card, so a tuned value pastes here "
             "unchanged: bare voxels ('50'), '50@12.56' (voxels at that Å/px, "
             "rescaled to this run's pixel size), or '100nm3'. Space-"
             "separated list sweeps; on a few tomograms read the Found/Relabeled "
             "curve — flat = fragmentation, steep drop = debris, and that "
             "distinction is the whole point of the sweep."},
            {"name": "MB_PATTERN", "kind": "env", "flag": "MB_PATTERN",
             "title": "Input pattern",
             "default": "",
             "help": "Filename glob within the input folder (blank = *.mrc). "
             "E.g. *_threshold_-1.0.mrc to label exactly one sweep variant — "
             "note thresholds normalise to one decimal in filenames."},
            {"name": "MB_TOMO_LIST", "kind": "env", "flag": "MB_TOMO_LIST",
             "pick_from": "input_dir",
             "title": "Only these tomograms",
             "default": "", "help": "Blank = every tomogram in the folder. Name series stems to do just those (📂 Pick… lists them)."},
            {"name": "MB_PIXEL_SIZE", "kind": "env", "flag": "MB_PIXEL_SIZE",
             "title": "Pixel size (Å/px)",
             "default": "12.56",
             "help": "Resolves physical cutoffs ('50@12.56', '100nm3') to "
             "voxels for THIS run's volumes. Bare voxel counts ignore it."},
            {"name": "MB_CONDA_ENV", "kind": "env", "flag": "MB_CONDA_ENV",
             "title": "Conda env",
             "default": "membrainseg", "help": "Env with membrain-seg."},
            {"name": "MB_FORCE", "kind": "env", "flag": "MB_FORCE",
             "title": "Redo existing",
             "default": "0", "help": "1 = relabel volumes with existing "
             "outputs."},
        ],
        "validate": lambda v: (
            "⚠ Min-size list is empty — give at least one cutoff "
            "(e.g. '50', '50@12.56', '100nm3', or a sweep '20 50 100 200')."
            if not str(v.get("MB_CC_THRES", "")).strip() else ""),
        "output_params": ["output_dir"],
        "docs": {
            "what": "Label connected components of the thresholded membranes "
                    "(membrain components), sweep the min-size cutoff, and "
                    "record Found/Relabeled per volume into "
                    "components_metrics.tsv.",
            "range": "cutoff 20–200 voxels; sweep and read the curve.",
            "effect": "Bigger cutoff removes debris; too big deletes real "
                      "membranes. A label ~4× the median voxel count is "
                      "likely 3–4 MERGED virions — check with "
                      "ml_membrain_label_tools.py histogram, and pull one "
                      "clean virion out with its extract command for mesh "
                      "testing.",
            "pitfall": "The tool does NOT create --out-folder (the wrapper "
                       "does). Flags verified against membrain-seg 0.0.10 "
                       "(2026-08-14).",
        },
        "status": lambda ps: ps.status_mb_components(),
    },
    {
        # First membrainpick-env stage. From here on the branch runs in a
        # DIFFERENT conda env than segmentation — the two conflict.
        "group": "12. Membrane", "id": "mb_mesh",
        "label": "Convert to meshes",
        "base": "bash",
        "env_name": "membrainpick",
        "node_kind": "compute",
        "params": [
            {"name": "script", "kind": "text", "flag": None, "title": "Wrapper script",
             "default": _pkg_script("ml_membrain_mesh_warp_auto.sh"),
             "help": "Mesh wrapper (shipped with the app). Pairs each "
             "segmentation with its tomogram by stem and runs membrain_pick "
             "convert_file per pair."},
            {"name": "input_dir", "kind": "text", "flag": None,
             "title": "Segmentation folder",
             "default": "membrane/thresholds",
             "help": "Membranes to mesh: a thresholds output, a components "
             "cc<voxels>/ folder, or a single-label extract."},
            {"name": "tomo_dir", "kind": "text", "flag": None,
             "title": "Tomogram folder",
             "default": "membrane/deconv/s1.0_f1.0",
             "help": "The tomograms the segmentations came from (stem-"
             "matched; values are projected onto the mesh from here)."},
            {"name": "output_dir", "kind": "text", "flag": None,
             "title": "Output folder",
             "default": "membrane/mesh",
             "help": "Mesh containers land here (later opened with "
             "surforama)."},
            {"name": "MB_PIXEL_SIZE", "kind": "env", "flag": "MB_PIXEL_SIZE",
             "title": "Pixel size (Å/px)",
             "default": "12.56",
             "help": "ALWAYS passed explicitly (--input-pixel-size): MRC "
             "headers have been unreliable here."},
            {"name": "MB_ONLY_LARGEST", "kind": "env", "flag": "MB_ONLY_LARGEST",
             "title": "Only largest component",
             "default": "",
             "help": "The tool's default is ON, which silently meshes the "
             "biggest object — often a merged CLUSTER, not a virion. This "
             "stage defaults it OFF; set 1 only when the input really holds "
             "one object of interest."},
            {"name": "MB_STEP_SIZE", "kind": "env", "flag": "MB_STEP_SIZE",
             "title": "Normal step size",
             "default": "",
             "help": "Blank = tool default 2.5. UNRESOLVED whether this is "
             "pixels or Ångströms — do not trust blindly; convert one "
             "membrane and check the projection visually."},
            {"name": "MB_STEP_LO", "kind": "env", "flag": "MB_STEP_LO",
             "title": "Normal steps (low)",
             "default": "",
             "help": "Lower --step-numbers bound (blank = -10). Same "
             "unresolved units as step size."},
            {"name": "MB_STEP_HI", "kind": "env", "flag": "MB_STEP_HI",
             "title": "Normal steps (high)",
             "default": "",
             "help": "Upper --step-numbers bound (blank = 10)."},
            {"name": "MB_BARY_AREA", "kind": "env", "flag": "MB_BARY_AREA",
             "title": "Barycentric area",
             "default": "400",
             "help": "Mesh density. Passed explicitly because the tool's "
             "help TEXT claims default 1.0 while its actual default is "
             "400.0 — pinning it here removes the ambiguity."},
            {"name": "MB_SMOOTHING", "kind": "env", "flag": "MB_SMOOTHING",
             "title": "Mesh smoothing",
             "default": "1000", "help": "Smoothing factor for the mesh."},
            {"name": "MB_PATTERN", "kind": "env", "flag": "MB_PATTERN",
             "title": "Input pattern",
             "default": "",
             "help": "Filename glob within the segmentation folder (blank = "
             "*.mrc). E.g. *_threshold_-1.0.mrc to mesh exactly one sweep "
             "variant instead of every threshold of every series — note the "
             "one-decimal normalisation in threshold filenames."},
            {"name": "MB_TOMO_LIST", "kind": "env", "flag": "MB_TOMO_LIST",
             "pick_from": "input_dir",
             "title": "Only these tomograms",
             "default": "", "help": "Blank = every tomogram in the folder. Name series stems to do just those (📂 Pick… lists them)."},
            {"name": "MB_CONDA_ENV", "kind": "env", "flag": "MB_CONDA_ENV",
             "title": "Conda env",
             "default": "membrainpick",
             "help": "membrain-pick's OWN env (pins scipy<1.12.1, numpy "
             "1.26.4, napari 0.5.6). Installing anything napari-ish here "
             "breaks it; never merge with membrainseg."},
            {"name": "MB_FORCE", "kind": "env", "flag": "MB_FORCE",
             "title": "Redo existing",
             "default": "0", "help": "1 = re-mesh membranes with existing "
             "outputs."},
        ],
        "validate": lambda v: (
            "⚠ Only-largest-component is ON: for a components volume this "
            "meshes the biggest object, which is often a MERGED cluster. "
            "Extract the label you want first (ml_membrain_label_tools.py "
            "extract) or switch it off."
            if str(v.get("MB_ONLY_LARGEST", "")).strip() == "1" else ""),
        "output_params": ["output_dir"],
        "docs": {
            "what": "Convert membrane segmentations to meshes with projected "
                    "tomogram values (membrain_pick convert_file, env "
                    "membrainpick — the branch switches conda envs here).",
            "range": "barycentric area ~400, smoothing ~1000 to start; step "
                     "size/numbers only after a visual check.",
            "effect": "Meshes + projections feed surforama annotation and "
                      "the pick/angle assignment steps (deferred until this "
                      "far is stable).",
            "pitfall": "Sibling flag drift: convert_file takes "
                       "--tomogram-path, convert_mb_folder takes --tomo-path "
                       "— this stage uses convert_file only. Step units "
                       "(px vs Å) are UNRESOLVED — verify visually, the "
                       "code does not guess. Flags verified against "
                       "membrain-pick 0.0.9 (2026-08-14).",
        },
        "status": lambda ps: ps.status_mb_mesh(),
    },
    {
        # Trap 1. The prerequisite for Fit virions: the density-support gate
        # needs a SIGN, and the wrong one is silent — every real virion fails
        # and bulk ice passes. Measured off the segmentation, never declared.
        "group": "12. Membrane", "id": "mb_polarity",
        "label": "Variant polarity (measure)",
        "base": "bash",
        "env_name": "membrainseg",
        "node_kind": "compute",
        "params": [
            {"name": "wrapper", "kind": "text", "flag": None,
             "title": "Env wrapper",
             "default": _pkg_script("ml_membrane_tool_warp_auto.sh"),
             "help": "Activates the conda env, then runs the tool."},
            {"name": "script", "kind": "text", "flag": None, "title": "Tool",
             "default": _pkg_script("ml_variant_polarity.py"),
             "help": "Samples the tomogram at the voxels the segmentation "
             "already calls membrane and compares their median with the whole "
             "volume's. No guessing, no flag to get wrong."},
            {"name": "tomogram", "kind": "text", "flag": None,
             "title": "Tomogram",
             "default": "warp_tiltseries/reconstruction/Position003_12.56Apx.mrc",
             "help": "One volume from the variant you are measuring. Polarity "
             "is a property of the VARIANT, so any one of its tomograms will "
             "do."},
            {"name": "segmentation", "kind": "text", "flag": None,
             "title": "Segmentation",
             "default": "membrane/segment/Position003_12.56Apx_segmented.mrc",
             "help": "A segmentation on the SAME grid — it is what says which "
             "voxels are membrane, so this card cannot run before one exists. "
             "Run 'Membrane: segment' on one tomogram of this variant first; "
             "any of its outputs works."},
            {"name": "out", "kind": "text", "flag": "--out",
             "title": "Output folder",
             "default": "jobs/{jobid}",
             "help": "Where polarity.json lands — the medians, the voxel "
             "count and the verdict. The LIVE value downstream reads is the "
             "one in the variant registry; this is the record of how it was "
             "measured, so a later disagreement can be traced."},
            {"name": "root", "kind": "text", "flag": "--root",
             "title": "Project root",
             "default": ".",
             "help": "Where the result is recorded, so the other cards can "
             "read it back instead of asking you again."},
            {"name": "variant_path", "kind": "text", "flag": "--variant-path",
             "title": "Variant folder",
             "default": "",
             "help": "The folder this polarity belongs to, e.g. "
             "jobs/J31-bin8-isonet2model/corrected. Blank = derive it from the "
             "tomogram path. Recorded against the variant, and a later run "
             "that disagrees says so rather than overwriting quietly."},
            {"name": "MB_CONDA_ENV", "kind": "env", "flag": "MB_CONDA_ENV",
             "title": "Conda env",
             "default": "membrainseg", "help": "Needs numpy + mrcfile."},
        ],
        "validate": lambda v: (
            "⚠ No segmentation named. This card measures polarity AT the "
            "voxels a segmentation calls membrane, so run 'Membrane: segment' "
            "on this variant first."
            if not str(v.get("segmentation", "")).strip() else
            "⚠ The tomogram and the segmentation look like different variants. "
            "Measuring one variant's polarity against another's segmentation "
            "records a sign for the wrong volumes."
            if (str(v.get("segmentation", "")).strip()
                and str(v.get("tomogram", "")).strip()
                and _series_of(v["tomogram"]) != _series_of(v["segmentation"]))
            else ""),
        "output_params": [],
        "docs": {
            "what": "Measures whether membranes are darker or brighter than "
                    "their surroundings, and records it against the variant.",
            "range": "Seconds. Strided sampling, so tomogram size barely "
                     "matters.",
            "effect": "Fit virions stops disabling its density gate, and the "
                      "parameter search stops warning about it.",
            "pitfall": "Run it once per VARIANT, not once per tomogram. A "
                       "polarity measured on the raw reconstruction says "
                       "nothing about an IsoNet-corrected one — that is "
                       "exactly the pair that differs.",
        },
        "status": lambda ps: ps.status_mb_polarity(),
    },
    {
        # Measured BEFORE any fitting, across the whole dataset. The prototype
        # learned its radius from whichever tomogram ran first, so tomogram B
        # inherited tomogram A's answer — and if A happened to hold a cluster
        # of merged virions, every later tomogram was measured against a radius
        # that no single virion had.
        "group": "12. Membrane", "id": "mb_population",
        "label": "Population radius + volume (whole dataset)",
        "base": "bash",
        "env_name": "membrainseg",
        "node_kind": "compute",
        "params": [
            {"name": "wrapper", "kind": "text", "flag": None,
             "title": "Env wrapper", "default": _pkg_script("ml_membrane_tool_warp_auto.sh"),
             "help": "Activates the conda env, then runs the tool."},
            {"name": "script", "kind": "text", "flag": None, "title": "Tool",
             "default": _pkg_script("ml_population_stats.py"),
             "help": "Measures every tomogram INDEPENDENTLY, then pools. It "
             "reuses the fitter's own fit_sphere/coverage/radius_away_from_gap, "
             "so 'a whole virion' cannot drift between the two steps."},
            {"name": "components", "kind": "text", "flag": "--components",
             "title": "Components folder",
             "default": "membrane/components/cc50",
             "help": "A FOLDER of labelled components — the cc<voxels> folder "
             "from the Components step, not a single file. The point is to "
             "measure across many tomograms at once."},
            {"name": "out_folder", "kind": "text", "flag": "--out-folder",
             "title": "Output folder", "default": "jobs/{jobid}",
             "help": "population.json lands here: pooled radius and volume, "
             "the per-tomogram breakdown, and every individual measurement so "
             "the spread can be re-examined without re-running. {jobid} keeps "
             "each run in its own folder — a fixed path means the second "
             "measurement silently overwrites the first, and two jobs claim "
             "the same directory."},
            {"name": "min_size", "kind": "text", "flag": "--min-size",
             "title": "Min component size", "default": "1000@12.56",
             "help": "PHYSICAL: '1000@12.56' means 1000 voxels at 12.56 Å/px. "
             "A bare voxel count means an 8× different object at bin4."},
            {"name": "diam_range", "kind": "text", "flag": "--diam-range",
             "title": "Plausible diameter (nm)", "default": "60 160",
             "help": "Two numbers. Only components inside this window are "
             "measured — wide enough not to prejudge the answer, narrow "
             "enough to exclude ice chunks and long membrane sheets."},
            {"name": "tomo_list", "kind": "text", "flag": "--tomo-list",
             "title": "Specific files only", "default": "",
             "help": "Blank = every component volume in the folder, which is "
             "what you want. Name stems only to re-measure a subset."},
            {"name": "MB_CONDA_ENV", "kind": "env", "flag": "MB_CONDA_ENV",
             "title": "Conda env", "default": "membrainseg",
             "help": "Needs numpy + mrcfile; the wrapper checks and refuses."},
        ],
        "output_params": ["out_folder"],
        "docs": {
            "what": "Measures the virion population — radius AND volume — "
                    "over the whole dataset, before any per-tomogram fitting.",
            "range": "Run it on as many tomograms as you have segmented. The "
                     "pooled value steadies with every one added.",
            "effect": "population.json, which Fit virions then reads instead "
                      "of learning a radius from one tomogram.",
            "pitfall": "Pooling uses the MEDIAN, not the mean: a single "
                       "tomogram full of merged blobs drags a mean and barely "
                       "moves a median. Check the MAD — if it is much above "
                       "10%, the components are probably still merged, and "
                       "Split clusters is the next step.",
        },
        "status": lambda ps: ps.status_mb_population(),
    },
    {
        # The measured population turned into a JUDGEMENT on each component:
        # far bigger than one virion = probably a fused cluster (with an
        # estimate of how many are inside), far smaller = probably debris or a
        # partial shell. Nothing is deleted — the verdicts point Split
        # clusters at its work and the outliers_only volume points napari at
        # what to eyeball.
        "group": "12. Membrane", "id": "mb_size_qc",
        "label": "Virion size QC (outliers)",
        "base": "bash",
        "env_name": "membrainseg",
        "node_kind": "compute",
        "params": [
            {"name": "wrapper", "kind": "text", "flag": None,
             "title": "Env wrapper",
             "default": _pkg_script("ml_membrane_tool_warp_auto.sh"),
             "help": "Runs the tool inside the membrainseg env (numpy + "
             "mrcfile)."},
            {"name": "script", "kind": "text", "flag": None, "title": "Tool",
             "default": _pkg_script("ml_virion_qc.py"),
             "help": "Size QC tool (shipped with the app)."},
            {"name": "components", "kind": "text", "flag": "--components",
             "title": "Components (file or folder)",
             "default": "",
             "help": "The labelled components to judge — one volume, or a "
             "folder to QC every tomogram. Build downstream from the "
             "components (or split) card and this fills itself."},
            {"name": "population", "kind": "text", "flag": "--population",
             "title": "Population stats",
             "default": "",
             "help": "population.json from the radius+volume card. REQUIRED "
             "— QC is only meaningful against a MEASURED population; the "
             "tool refuses to guess."},
            {"name": "angpix", "kind": "text", "flag": "--angpix",
             "title": "Pixel size (Å/px)", "default": "12.56",
             "help": "For voxel→nm³. 0 = trust the mrc header (risky — "
             "headers have lied here before)."},
            {"name": "small_frac", "kind": "text", "flag": "--small-frac",
             "title": "Debris below (× virion)", "default": "0.35",
             "help": "Components under this fraction of one virion volume "
             "are flagged too_small (debris / partial shells)."},
            {"name": "big_frac", "kind": "text", "flag": "--big-frac",
             "title": "Cluster above (× virion)", "default": "1.7",
             "help": "Components over this multiple of one virion volume are "
             "flagged too_big, with k ≈ how many virions are fused (round of "
             "the ratio). 1.7 leaves room for genuinely large singles; drop "
             "toward 1.4 if the population spread is tight."},
            {"name": "sample", "kind": "text", "flag": "--sample",
             "title": "Inspect shortlist", "default": "10",
             "help": "How many outliers per verdict go on the inspection "
             "shortlist (inspect.txt), largest first. 0 = every outlier."},
            {"name": "tomogram", "kind": "text", "flag": "--tomogram",
             "title": "Tomogram folder", "default": "",
             "help": "The greyscale volumes the components came from — "
             "recorded so 🔍 View in napari opens outliers OVER the "
             "tomogram. Carried automatically by Build downstream."},
            {"name": "out_folder", "kind": "text", "flag": "--out-folder",
             "title": "Output folder", "default": "jobs/{jobid}",
             "help": "size_qc.json/.csv (every component's verdict), "
             "outliers_only.mrc (OK labels zeroed — open it over the "
             "tomogram and everything visible is suspect), inspect.txt (the "
             "shortlist)."},
            {"name": "MB_CONDA_ENV", "kind": "env", "flag": "MB_CONDA_ENV",
             "title": "Conda env", "default": "membrainseg",
             "help": "Needs numpy + mrcfile; the wrapper checks and refuses."},
        ],
        "validate": lambda v: (
            "⚠ Point 'Components' at the labelled volume(s) to judge — build "
            "downstream from the components card to fill it automatically."
            if not str(v.get("components", "")).strip() else
            "⚠ 'Population stats' is REQUIRED — run the Population radius + "
            "volume card first and point this at its population.json."
            if not str(v.get("population", "")).strip() else ""),
        "output_params": ["out_folder"],
        "docs": {
            "what": "Judges every connected component against the measured "
                    "population: too big = likely a fused cluster of k "
                    "virions, too small = likely debris. Writes verdicts "
                    "(json/csv), an outliers-only label volume for napari, "
                    "and an inspection shortlist.",
            "range": "small_frac ~0.35, big_frac ~1.4–1.7 of one virion "
                     "volume; thresholds are population-relative so any "
                     "binning works unchanged.",
            "effect": "Tells you HOW MUCH of the dataset Split clusters must "
                      "rescue (the 'extra virions in clusters' count) before "
                      "you commit to the full-set run.",
            "pitfall": "A judgement aid, not a filter — nothing is removed. "
                       "Verdicts inherit the population's quality: if the "
                       "population was measured on still-merged components, "
                       "run Split clusters and re-measure first. 🔍 View in "
                       "napari opens outliers_only.mrc over the tomogram; "
                       "label values match the report, so napari's label "
                       "picker reads the same numbers.",
        },
        "status": lambda ps: ps.status_mb_size_qc(),
    },
    {
        # Connected-component labelling fuses virions that touch, and a fused
        # pair is not a big virion — it is two, with a volume that corrupts
        # every population statistic it enters.
        "group": "12. Membrane", "id": "mb_split_clusters",
        "label": "Split merged virions",
        "base": "bash",
        "env_name": "membrainseg",
        "node_kind": "compute",
        "params": [
            {"name": "wrapper", "kind": "text", "flag": None,
             "title": "Env wrapper", "default": _pkg_script("ml_membrane_tool_warp_auto.sh"),
             "help": "Activates the conda env, then runs the tool."},
            {"name": "script", "kind": "text", "flag": None, "title": "Tool",
             "default": _pkg_script("ml_split_clusters.py"),
             "help": "RANSAC with the radius FIXED to the measured population "
             "value, so only the centre is unknown. Greedy: take the best "
             "sphere, remove its voxels, look again."},
            {"name": "components", "kind": "text", "flag": "--components",
             "title": "Components volume or folder",
             "default": "membrane/components/cc50",
             "help": "A folder re-labels every volume in it; a single .mrc "
             "does just that one."},
            {"name": "population", "kind": "text", "flag": "--population",
             "title": "Population store", "default": "",
             "help": "REQUIRED — the population job's own folder, e.g. "
             "jobs/J43_population-radius-volume/population.json — the population.json FILE, not its folder. Right-click that job and Build "
             "downstream to have it filled in. The radius comes from here: "
             "the point is that this tomogram is split against the "
             "dataset's measured virion, not against its own largest blob."},
            {"name": "out_folder", "kind": "text", "flag": "--out-folder",
             "title": "Output folder", "default": "jobs/{jobid}",
             "help": "A re-labelled *_split.mrc per volume. The input is never "
             "touched, so a bad split can be discarded by deleting this "
             "folder."},
            {"name": "merge_ratio", "kind": "text", "flag": "--merge-ratio",
             "title": "Suspect above (× virion radius)", "default": "1.4",
             "help": "A DISTANCE test: only components that reach this many "
             "times a virion's radius from their own centre are examined. One "
             "virion reaches 1.0×, a touching pair about 1.8×. Everything "
             "below passes through untouched — a component that is already "
             "virion-sized cannot be improved by splitting it."},
            {"name": "tol_mad", "kind": "text", "flag": "--tol-mad",
             "title": "Shell tolerance (× MAD)", "default": "2.0",
             "help": "How far a voxel may sit from the sphere and still count. "
             "In MADs of the measured radius, so it scales with how uniform "
             "this dataset's virions actually are."},
            {"name": "min_coverage", "kind": "text", "flag": "--min-coverage",
             "title": "Min shell coverage", "default": "0.30",
             "help": "A candidate must be covered over at least this fraction "
             "of directions. Raise it to split less."},
            {"name": "min_separation", "kind": "text", "flag": "--min-separation",
             "title": "Min centre separation (× radius)", "default": "1.5",
             "help": "Two virions must have centres at least this many radii "
             "apart. Touching virions sit about 2 apart, so 1.5 keeps every "
             "real merged pair while refusing the same shell fitted twice, "
             "slightly offset — which is how a blob reaching only 1.7 radii "
             "was once reported as three whole virions."},
            {"name": "MB_CONDA_ENV", "kind": "env", "flag": "MB_CONDA_ENV",
             "title": "Conda env", "default": "membrainseg",
             "help": "Needs numpy + mrcfile; the wrapper checks and refuses."},
        ],
        "validate": lambda v: (
            "⚠ No population store set. The radius MUST come from a measured "
            "population — without one there is nothing to split against."
            if not str(v.get("population", "")).strip() else ""),
        "output_params": ["out_folder"],
        "docs": {
            "what": "Separates virions that connected-component labelling "
                    "fused into one blob.",
            "range": "suspect above 1.4 radii to start. Raise it, or raise "
                     "min coverage, if anything looks over-split.",
            "effect": "A re-labelled components volume where a touching pair "
                      "counts as two virions instead of one giant one.",
            "pitfall": "OVER-SPLITTING IS THE REAL RISK, and it is invisible: "
                       "a missed split loses one particle, but an invented one "
                       "adds a virion that never existed to every count, "
                       "diameter and volume downstream. So a candidate must "
                       "also be HOLLOW — coverage alone cannot tell a virion "
                       "from a solid lump, since a sphere drawn inside any "
                       "dense object passes through material in every "
                       "direction. Compare the counts before and after.",
        },
        "status": lambda ps: ps.status_mb_split(),
    },
    {
        # A3 (cards spec draft 2). The python tools all run through one wrapper
        # because tomogration's own venv has PySide6 and nothing else — a stage
        # with base "python3" would hand them an interpreter with no numpy.
        "group": "12. Membrane", "id": "mb_fit_virions",
        "label": "Fit virions (locate + measure)",
        "base": "bash",
        "env_name": "membrainseg",
        "node_kind": "compute",
        "params": [
            {"name": "wrapper", "kind": "text", "flag": None,
             "title": "Env wrapper",
             "default": _pkg_script("ml_membrane_tool_warp_auto.sh"),
             "help": "Activates the conda env, then runs the tool."},
            {"name": "script", "kind": "text", "flag": None, "title": "Tool",
             "default": _pkg_script("ml_fit_virions.py"),
             "help": "Two-pass fitter: a free fit finds whole virions and "
             "learns the population radius, then fixed-radius centre-only "
             "fits rescue partial arcs. Eight gates reject the rest."},
            {"name": "components", "kind": "text", "flag": "--components",
             "title": "Components volume or folder",
             "default": "membrane/components/cc50",
             "help": "A cc<voxels> FOLDER fits every tomogram in it, each into "
             "its own subfolder plus a combined fits.json. A single .mrc fits "
             "just that one."},
            {"name": "tomogram", "kind": "text", "flag": "--tomogram",
             "title": "Tomogram or folder",
             "default": "warp_tiltseries/reconstruction",
             "help": "The volumes the density-support gate reads. Give the "
             "matching FOLDER when components is a folder — each components "
             "volume is paired with the longest tomogram name that prefixes "
             "it, so a mismatched pixel size cannot slip through. Must be the "
             "same grid as the components."},
            {"name": "polarity", "kind": "choice", "flag": "--polarity",
             "title": "Membrane polarity",
             "default": "unknown",
             "choices": [("Measure it first (gate disabled)", "unknown"),
                         ("Dark membranes", "dark"),
                         ("Bright membranes", "bright")],
             "help": "NEVER guess this. The density gate keeps surfaces whose "
             "voxels sit in the membrane-like tercile; with the sign wrong it "
             "rejects every real virion and accepts bulk ice. Measure it with "
             "the Variant polarity step, which reads it off the segmentation."},
            {"name": "out_folder", "kind": "text", "flag": "--out-folder",
             "title": "Output folder",
             "default": "jobs/{jobid}",
             "help": "fits.json (centres, radii, volumes, recall, gate "
             "settings) + surfaces.npz, and one subfolder per tomogram when "
             "fitting a whole folder. {jobid} keeps each run in its own "
             "folder — a fixed path means a re-run silently overwrites the "
             "fits you were comparing it against."},
            {"name": "min_size", "kind": "text", "flag": "--min-size",
             "title": "Min component size",
             "default": "1000@12.56",
             "help": "PHYSICAL: '520nm3', or '1000@12.56' meaning 1000 voxels "
             "at 12.56 Å/px. A bare voxel count means an 8× different object "
             "between bin4 and bin8."},
            {"name": "population", "kind": "text", "flag": "--population",
             "title": "Population store",
             "default": "",
             "help": "population.json to read the radius from and add to. "
             "Blank = learn it from this tomogram alone, which is what the "
             "prototype did with 3 virions — a stored population measured "
             "across several tomograms is far steadier."},
            {"name": "write_masks", "kind": "check", "flag": "--write-masks",
             "title": "Write fit masks",
             "default": True,
             "help": "One shell .mrc per accepted virion, for the viewer. "
             "Needed by the 'open in viewer' action."},
            {"name": "MB_CONDA_ENV", "kind": "env", "flag": "MB_CONDA_ENV",
             "title": "Conda env",
             "default": "membrainseg",
             "help": "Needs numpy + mrcfile; the wrapper checks and refuses."},
        ],
        "validate": lambda v: (
            "⚠ Polarity is unmeasured, so the density-support gate is DISABLED "
            "— every surface passes it. Run 'Variant polarity' first."
            if str(v.get("polarity", "")).strip() in ("", "unknown") else ""),
        "output_params": ["out_folder"],
        "docs": {
            "what": "Finds virions in a components volume and measures each "
                    "one: centre, radius, volume, density support.",
            "range": "min size 1000@12.56 to start; gates default to the "
                     "prototype's values.",
            "effect": "fits.json holds the accepted virions and the RECALL — "
                      "the fraction of segmented voxels they account for.",
            "pitfall": "Volume comes from the CONSTRAINED radius, never a free "
                       "ellipsoid: the missing wedge smears membranes along z, "
                       "and an ellipsoid follows that smear, inflating volume "
                       "by ~50%. Polarity must be measured, not guessed.",
        },
        "status": lambda ps: ps.status_mb_fits(),
    },
    {
        # A4, Oversample mode.
        "group": "12. Membrane", "id": "mb_pick_surfaces",
        "label": "Pick surfaces (oversample)",
        "base": "bash",
        "env_name": "membrainseg",
        "node_kind": "compute",
        "params": [
            {"name": "wrapper", "kind": "text", "flag": None,
             "title": "Env wrapper",
             "default": _pkg_script("ml_membrane_tool_warp_auto.sh"),
             "help": "Activates the conda env, then runs the tool."},
            {"name": "script", "kind": "text", "flag": None, "title": "Tool",
             "default": _pkg_script("ml_pick_surfaces.py"),
             "help": "Samples the accepted surfaces and writes a RELION star "
             "with per-site normals, Euler angles and theta."},
            {"name": "fits", "kind": "text", "flag": "--fits",
             "title": "Fits (fits.json)",
             "default": "",
             "help": "REQUIRED — the fits.json a Fit virions job wrote, e.g. "
             "jobs/J45_fit-virions/fits.json. Build downstream from that card "
             "and this fills itself. A BATCH fits.json works too: one star "
             "per tomogram, each stem named automatically."},
            {"name": "tomostar", "kind": "text", "flag": "--tomostar",
             "title": "Series stem",
             "default": "",
             "help": "Single-tomogram fits only. Goes into _rlnTomoName and "
             "must match the .tomostar stem EXACTLY — Warp matches particles "
             "to tilt series on that exact string. A batch fits.json names "
             "string, and a near-miss exports nothing at all."},
            {"name": "out", "kind": "text", "flag": "--out",
             "title": "Output star",
             "default": "membrane/picks/{jobid}_oversample.star",
             "help": "One row per site. {jobid} keeps runs apart."},
            {"name": "spacing", "kind": "text", "flag": "--spacing",
             "title": "Site spacing (Å)",
             "default": "50",
             "help": "Sites scale with surface AREA, so halving this "
             "quadruples the count."},
            {"name": "shells", "kind": "text", "flag": "--shells",
             "title": "Shells (Å)",
             "default": "0,50,-50",
             "help": "Radius offsets. These overlap BY DESIGN — recall first, "
             "duplicate removal after the first refinement, never before."},
            {"name": "random_psi", "kind": "check", "flag": "--random-psi",
             "title": "Randomise psi",
             "default": False,
             "help": "A surface normal fixes two Euler angles and says nothing "
             "about the in-plane one. Randomising avoids a spurious common "
             "orientation in the first classification; 0 is easier to read."},
            {"name": "MB_CONDA_ENV", "kind": "env", "flag": "MB_CONDA_ENV",
             "title": "Conda env",
             "default": "membrainseg", "help": "Needs numpy."},
        ],
        "validate": lambda v: (
            "⚠ Fits is REQUIRED — point it at a Fit virions job's fits.json "
            "(e.g. jobs/J45_fit-virions/fits.json), or build this card "
            "downstream from that job."
            if not str(v.get("fits", "")).strip() else
            "⚠ Series stem is blank. Fine for a BATCH fits.json (stems are "
            "named automatically); for a single-tomogram fits it becomes "
            "_rlnTomoName and the tool will refuse without it."
            if not str(v.get("tomostar", "")).strip() else ""),
        "output_params": ["out"],
        "docs": {
            "what": "Oversampled picking: sites at ~50 Å over every accepted "
                    "surface, plus shells inside and outside it.",
            "range": "spacing 50 Å; shells 0,50,-50.",
            "effect": "A star with position, outward normal, Euler angles and "
                      "THETA — the angle to the wedge axis.",
            "pitfall": "Theta cannot be reconstructed later, and detection "
                       "efficiency depends on it, so any spatial statistic "
                       "computed without it is biased unrecoverably. Psi is "
                       "undetermined by a normal — refine it, never trust it.",
        },
        "status": lambda ps: ps.status_mb_picks(),
    },
    {
        # Part C. --plan only for now: it costs the sweep without running it.
        "group": "12. Membrane", "id": "mb_explore",
        "label": "Parameter search",
        "base": "bash",
        "env_name": "membrainseg",
        "node_kind": "compute",
        "params": [
            {"name": "wrapper", "kind": "text", "flag": None,
             "title": "Env wrapper",
             "default": _pkg_script("ml_membrane_tool_warp_auto.sh"),
             "help": "Activates the conda env, then runs the tool."},
            {"name": "script", "kind": "text", "flag": None, "title": "Tool",
             "default": _pkg_script("ml_explore_membrane.py"),
             "help": "Sweeps variants × thresholds × cutoffs × tomograms and "
             "reports what each stage would actually have to run."},
            {"name": "root", "kind": "text", "flag": "--root",
             "title": "Project root",
             "default": ".",
             "help": "Where the relative paths below start from — your project "
             "folder, so '.' is right unless you are sweeping another "
             "dataset."},
            {"name": "out", "kind": "text", "flag": "--out",
             "title": "Output folder",
             "default": "jobs/{jobid}",
             "help": "Everything this run writes — cache/ and results.json. "
             "{jobid} becomes this job's id, so each sweep keeps its own "
             "folder and nothing lands in a shared path. The cache is keyed "
             "by content hash, so a re-run with one more threshold reuses "
             "every segmentation instead of redoing it."},
            {"name": "tomo_dir", "kind": "text", "flag": "--tomo-dir",
             "title": "Tomogram folders to compare",
             "default": "raw=warp_tiltseries/reconstruction",
             "help": "The variants to sweep, as name=path, space separated. "
             "Two is the usual start: raw=warp_tiltseries/reconstruction "
             "isonet2=jobs/J31-bin8-isonet2model/corrected. Pixel size is "
             "read from each folder's own filenames, so cutoffs resolve "
             "correctly per variant."},
            {"name": "derive_deconv", "kind": "text", "flag": "--derive-deconv",
             "title": "Also deconvolve",
             "default": "raw",
             "help": "Name one of the folders above and the sweep adds a third "
             "variant, 'deconv', derived from it — raw vs deconvolved vs "
             "IsoNet compared in one pass, without reconstructing anything. "
             "Blank = no derived variant."},
            {"name": "tomogram", "kind": "text", "flag": "--tomogram",
             "title": "Tomograms (2–3)",
             "default": "Position003 Position045",
             "help": "Series stems, space or comma separated. This card "
             "refuses more than 3 — it exists to find an operating point "
             "before committing all 72."},
            {"name": "threshold", "kind": "text", "flag": "--threshold",
             "title": "Thresholds",
             "default": "-2 0 2",
             "help": "Score thresholds to sweep, space or comma separated. "
             "Every extra value MULTIPLIES the matrix, but costs no extra "
             "segmentation — that is the point of the cache."},
            {"name": "threshold_mode", "kind": "choice", "flag": "--threshold-mode",
             "title": "Threshold mode",
             "default": "absolute",
             "choices": [("Absolute score", "absolute"),
                         ("Percentile of the score map", "percentile")],
             "help": "MemBrain scores shift with input contrast, so −2 on a "
             "deconvolved tomogram and −2 on an IsoNet one are different "
             "operating points. Percentile makes them comparable."},
            {"name": "cutoff", "kind": "text", "flag": "--cutoff",
             "title": "Size cutoffs",
             "default": "1000@12.56 10000@12.56",
             "help": "Smallest component to keep, as a PHYSICAL size. "
             "'1000@12.56' reads as 'as big as 1000 voxels are at 12.56 Å/px' "
             "— the sweep re-resolves that for each variant's own pixel size "
             "(8000 voxels at 6.28 Å/px, same physical object). '520nm3' says "
             "it as a volume instead. A bare voxel count is refused: it means "
             "a different object at every binning."},
            # The flag is --run, not --plan. With a 'Plan only' checkbox,
            # unticking it merely REMOVED --plan and the tool (which needs
            # --run to execute) planned anyway — silently, twice. The control
            # now emits the flag that does the thing.
            {"name": "run", "kind": "check", "flag": "--run",
             "title": "Run the sweep",
             "default": False,
             "help": "OFF = plan only: cost the matrix, list what is cached, "
             "write nothing. ON = actually run it, in dependency order, "
             "caching each stage so a changed threshold never re-segments. "
             "Plan first; it takes seconds and tells you the GPU cost before "
             "you spend it."},
            {"name": "ckpt", "kind": "text", "flag": "--ckpt",
             "title": "Segmentation model (.ckpt)",
             "default": "",
             "help": "REQUIRED to run (ignored when planning): the "
             "membrain-seg checkpoint, e.g. <processing_scripts>/membrain-seg/"
             "MemBrain_seg_v10_beta.ckpt. This is the one thing the whole "
             "sweep cannot start without."},
            {"name": "gpu", "kind": "text", "flag": "--gpu", "gpu_sep": ",",
             "title": "GPUs",
             "default": "0",
             "help": "Passed to each segmentation. The sweep is GPU-bound in "
             "the segment stage and CPU-bound everywhere else."},

            {"name": "MB_CONDA_ENV", "kind": "env", "flag": "MB_CONDA_ENV",
             "title": "Conda env",
             "default": "membrainseg", "help": "Needs numpy."},
        ],
        "validate": lambda v: (
            "⚠ Name at least one tomogram (2–3). This card is for finding an "
            "operating point, not for processing the dataset."
            if not str(v.get("tomogram", "")).strip() else
            # Planning needs nothing; RUNNING cannot start a single
            # segmentation without a checkpoint, and finding that out 40
            # minutes in would waste the whole sweep.
            "⚠ 'Run the sweep' is ON, but no segmentation model checkpoint "
            "is set — the first segmentation would fail. Give the .ckpt, or "
            "untick it to just cost the matrix."
            if (v.get("run") and not str(v.get("ckpt", "")).strip())
            else ""),
        "output_params": ["out"],
        "docs": {
            "what": "Costs and plans a parameter sweep over the variant "
                    "registry, caching the expensive axis.",
            "range": "2–3 tomograms; thresholds −2/0/2; two size cutoffs.",
            "effect": "Segmentation runs once per (variant, tomogram) — 12 "
                      "runs for a 72-cell matrix, not 72.",
            "pitfall": "Rank cells on virions accepted AND recall together: "
                       "the gates measure precision only, so accepting three "
                       "easy virions scores perfectly on all of them.",
        },
        "status": lambda ps: ps.status_mb_explore(),
    },
]

def param_title(p):
    """Human-readable display name for a parameter row.

    The form used to print the raw wire name (MB_STRENGTH, output_angpix) as
    the row label, which reads like debugging output. The rule now: an explicit
    'title' on the param wins; otherwise the name is prettified — env prefixes
    (MB_, MA_, TS_…) dropped, underscores to spaces, sentence case — while
    short bare acronyms (CS, KV, Q0, GPUS) stay as they are. The exact wire
    name/flag still appears in the help line, so nothing technical is hidden."""
    t = p.get("title")
    if t:
        return t
    name = str(p.get("name", ""))
    toks = [t for t in name.split("_") if t]
    if not toks:
        return name
    if name.isupper():
        if len(toks) == 1:
            return name                        # bare acronym: CS, Q0, GPUS
        if len(toks[0]) <= 3:
            toks = toks[1:]                    # env prefix: MB_, MA_, TS_…
        if len(toks) == 1 and len(toks[0]) <= 4:
            return toks[0]                     # MB_KV -> KV, MB_CS -> CS
    s = " ".join(t.lower() for t in toks)
    return s[:1].upper() + s[1:]


_PATHISH_NAMES = {
    "settings", "star", "particles", "mask", "half1", "half2", "alignments",
    "mdocs", "frameseries", "tomostar", "processing", "config", "class_map",
    "data_star", "source_star", "new_star", "class_star", "particles_star",
    "starfile", "template_path", "gain_path", "in_gain", "in_mrc", "out_mrc",
    "input_mdoc", "conv_key", "exclusion_list", "i", "o", "ref",
}
_PATHISH_SUFFIXES = ("_dir", "_path", "_folder", "_star", "_map", "_list",
                     "_key", "dir", "folder", "model", "ckpt")


def param_is_pathish(p):
    """Should this field get the inventory Browse… button?

    Explicit p['browse'] (True/False) always wins. Otherwise: text/env params
    whose name or default reads as a file-system location — except the shipped
    'script'/'wrapper' fields (they point INTO the app, not the project) and
    fields that already carry the tomogram picker (pick_from)."""
    if "browse" in p:
        return bool(p["browse"])
    if p.get("kind") not in ("text", "env"):
        return False
    if p.get("pick_from"):
        return False
    name = str(p.get("name", ""))
    n = name.lower()
    if n in ("script", "wrapper"):
        return False
    if n in _PATHISH_NAMES or n.endswith(_PATHISH_SUFFIXES):
        return True
    d = str(p.get("default", "") or "")
    return "/" in d


def param_wire_name(p):
    """The technical identity shown in the help line: the CLI flag when the
    param has one, $VAR for env knobs, or the raw name for positionals."""
    kind = p.get("kind", "")
    flag = p.get("flag")
    if kind in ("env", "env_int"):
        return f"${flag or p.get('name', '')}"
    if flag:
        return str(flag)
    return str(p.get("name", ""))


def stage_tool_line(spec):
    """What actually runs — shown under the form title so the human label
    (e.g. 'Tomogram reconstruction') never hides the tool (WarpTools
    ts_reconstruct). For bash wrapper stages, names the shipped script."""
    base = str(spec.get("base", "") or "")
    if base == "bash":
        script = next((p for p in spec.get("params", [])
                       if p.get("name") == "script"), None)
        name = os.path.basename(str((script or {}).get("default", "")).strip())
        return f"bash {name}" if name else "bash"
    return base or str(spec.get("id", ""))


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


def _series_of(path):
    """Series stem of a tomogram/segmentation path, for cross-checking that a
    polarity is measured on one variant rather than across two."""
    import os as _os
    base = _os.path.basename(str(path))
    for suffix in ("_segmented", "_scores", "_isonet2", "_isonet1", "_deconv"):
        base = base.replace(suffix, "")
    return base.rsplit(".", 1)[0]


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


def _quote_glob(s):
    """Single-quote a pattern value (*.eer, *clean.star) so `bash -lc` passes it
    to the tool verbatim. Unquoted, bash expands it against whatever sits in the
    project root — e.g. when the root IS the acquisition folder, `--extension
    *.eer` becomes hundreds of filenames and WarpTools mis-parses the command."""
    if (any(c in s for c in "*?[") and " " not in s and "\t" not in s
            and not s.startswith(("'", '"'))):
        return f"'{s}'"
    return s


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
        # A key MISSING from values is an empty value, not the string "None".
        # Jobs store the params they were saved with, so every param added to a
        # stage later is absent from every job saved before it — and
        # str(None).strip() == "None" put a literal `--output_processing None`
        # into those jobs' commands the moment a new field shipped.
        v = values.get(p["name"])
        if v is None:
            v = ""
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
                s = _quote_glob(s)
                # A POSITIONAL value containing whitespace must be one shell
                # word: unquoted, 'membrane/segment v2' word-splits into two
                # positionals and the wrapper silently writes into the wrong
                # folder. Flagged params are left alone — some (device_list)
                # legitimately pass space-separated token lists.
                if (not p.get("flag") and re.search(r"\s", s)
                        and not s.startswith(("'", '"'))):
                    s = "'" + s.replace("'", "'\\''") + "'"
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
    "relion4_convert": "relion4",
    "relion4_check_star": "relion4",
    "relion4_verify_reextract": "relion4",
    "relion4_result": "relion4",
    "m_reset": "m",
    "m_create_population": "m", "m_create_source": "m", "m_mask_create": "m",
    "m_create_species": "m", "m_core": "m", "m_estimate_weights": "m",
    "m_resample_trajectories": "m", "m_kill_orphans": ".",
    "relion4_to_warp": "warp_tiltseries/matching_reextract",
    "mb_deconv": "membrane/deconv",
    "mb_polarity": ".",
    "mb_isonet_train": "membrane/isonet",
    "mb_isonet_predict": "membrane/isonet_corrected",
    "mb_isonet2_train": "membrane/isonet2",
    "mb_isonet2_predict": "membrane/isonet2_corrected",
    "mb_segment": "membrane/segment",
    "mb_thresholds": "membrane/thresholds",
    "mb_components": "membrane/components",
    "mb_mesh": "membrane/mesh",
    "mb_fit_virions": "jobs",
    "mb_population": "jobs",
    "mb_size_qc": "jobs",
    "mb_split_clusters": "jobs",
    "mb_pick_surfaces": "jobs",
    "mb_explore": "jobs",
}

# Which mockup column each stage group belongs to (the three job-list panels).
COLUMN_OF_GROUP = {
    "1. Data prep": "curation", "2. Gain": "curation",
    "3. Frameseries": "stackprep", "4. Tilt series": "stackprep",
    "5. Alignment": "alignrecon", "6. CTF": "alignrecon",
    "7. Reconstruct": "alignrecon", "8. Pick": "alignrecon",
    "9. Export": "alignrecon", "10. RELION 4": "alignrecon",
    "11. M refinement": "alignrecon", "12. Membrane": "alignrecon",
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
    ("relion4", "relion4/ (RELION projects)"),
    ("membrane", "membrane/ (segmentation branch)"),
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
    "ts_export_particles":  (["warp_tiltseries", "warp_tiltseries/matching"], ["relion4"]),
    "relion4_convert":      (["relion4"], ["relion4"]),
    "relion4_result":       (["relion4"], ["relion4"]),
    "relion4_to_warp":      (["relion4", "warp_tiltseries/matching"],
                             ["warp_tiltseries/matching_reextract"]),
    "m_reset":              (["m", "warp_tiltseries"], ["m"]),
    "m_create_population":  ([], ["m"]),
    "m_create_source":      (["warp_tiltseries", "m"], ["m"]),
    "m_mask_create":        (["relion4"], ["m"]),
    "m_create_species":     (["relion4", "m"], ["m"]),
    "m_core":               (["m", "warp_tiltseries"], ["m"]),
    "m_estimate_weights":   (["m"], ["m"]),
    "m_resample_trajectories": (["m"], ["m"]),
    "mb_deconv":            (["warp_tiltseries/reconstruction",
                              "warp_tiltseries"], ["membrane/deconv"]),
    "mb_polarity":          (["warp_tiltseries/reconstruction",
                              "membrane/segment"], []),
    "mb_isonet_train":      (["warp_tiltseries/reconstruction",
                              "warp_tiltseries"], ["membrane/isonet"]),
    "mb_isonet_predict":    (["warp_tiltseries/reconstruction",
                              "membrane/isonet"],
                             ["membrane/isonet_corrected"]),
    "mb_isonet2_train":     (["warp_tiltseries/reconstruction",
                              "warp_tiltseries"], ["membrane/isonet2"]),
    "mb_isonet2_predict":   (["warp_tiltseries/reconstruction",
                              "membrane/isonet2"],
                             ["membrane/isonet2_corrected"]),
    "mb_segment":           (["membrane/deconv"], ["membrane/segment"]),
    "mb_thresholds":        (["membrane/segment"], ["membrane/thresholds"]),
    "mb_components":        (["membrane/thresholds"], ["membrane/components"]),
    "mb_mesh":              (["membrane/thresholds", "membrane/deconv"],
                             ["membrane/mesh"]),
    "mb_fit_virions":       (["membrane/components",
                              "warp_tiltseries/reconstruction"],
                             ["membrane/popfits"]),
    "mb_population":        (["membrane/components"],
                             ["membrane/population"]),
    "mb_size_qc":           (["membrane/components",
                              "membrane/population"],
                             ["membrane/size_qc"]),
    "mb_split_clusters":    (["membrane/components",
                              "membrane/population"],
                             ["membrane/split"]),
    "mb_pick_surfaces":     (["membrane/popfits"], ["membrane/picks"]),
    "mb_explore":           (["membrane/components",
                              "warp_tiltseries/reconstruction"],
                             ["membrane/explore"]),
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
    "relion4": "<export>/*.star + <export>/subtomo/*.mrc  (per-export dirs)",
    "m": "*.population + species/<name>_<hash>/  (M project)",
    "membrane": "segmentation branch (deconv/segment/thresholds/components/mesh)",
    "membrane/deconv": "s<strength>_f<falloff>/*_deconv.mrc + PROVENANCE.json",
    "membrane/isonet": "tomograms.star + results/model_iter*.h5  (IsoNet project)",
    "membrane/isonet_corrected": "corrected/*.mrc  (wedge-restored — picking ONLY)",
    "membrane/isonet2": "tomograms.star + isonet_maps/*.pt  (IsoNet 2 project)",
    "membrane/isonet2_corrected": "corrected/*.mrc  (wedge-restored — picking ONLY)",
    "membrane/segment": "*_scores.mrc + segmentations + PROVENANCE.json",
    "membrane/thresholds": "*_threshold_<t>.mrc  (one decimal, e.g. -3.0)",
    "membrane/components": "cc<voxels>/*.mrc + components_metrics.tsv",
    "membrane/mesh": "mesh containers (.h5) for surforama",
    "membrane/population": "population.json  (pooled radius + volume, per-tomogram breakdown)",
    "membrane/size_qc": "size_qc.json/.csv + outliers_only.mrc  (verdicts + napari overlay)",
    "membrane/split": "*_split.mrc  (components re-labelled, merged virions separated)",
    "membrane/popfits": "fits.json + surfaces.npz + fit_<label>.mrc  (virions)",
    "membrane/picks": "*.star  (one row per sampled surface site)",
    "membrane/explore": "results.json + cache/<stage>/<hash>/  (sweep)",
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
              "ts_export_particles", "relion4_convert"}

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
