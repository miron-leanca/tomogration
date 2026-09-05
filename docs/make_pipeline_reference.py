#!/usr/bin/env python3
"""Generate the Tomogration pipeline reference (HTML -> PDF).

One block per pipeline command: what it does, which DIRECTORIES it reads, the exact
FILENAME PATTERNS it looks for, what it writes and where, the key flags, and the
gotchas that actually bite.

The stage list, flags and defaults are read from tomogration_app.py so this can't
drift. The file-naming facts (FILE_FACTS below) are curated — they encode things
only learned by running the tools (e.g. --override_suffix renames the peak STAR but
NOT the correlation volume).

    python3 docs/make_pipeline_reference.py           # writes docs/tomogration_pipeline_reference.{html,pdf}
"""
import html
import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "tests" / "stub"))       # fake PySide6, no GUI needed

spec = importlib.util.spec_from_file_location("tomapp", REPO / "tomogration_app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

E = html.escape

# ---------------------------------------------------------------------------
# Curated per-stage file facts. reads/writes are (path-pattern, what-it-is).
# `<pos>` = PositionNNN (the series name). `<apx>` = pixel size formatted to TWO
# decimals by Warp, e.g. 10 -> "10.00", 12.56 -> "12.56".
# ---------------------------------------------------------------------------
FILE_FACTS = {
    "rename": dict(
        reads=[("<source_dir>/*.eer", "raw movies from the microscope"),
               ("<source_dir>/*.mdoc", "one per tilt series (SerialEM/Tomo5)")],
        writes=[("<source_dir>/PositionNNN_*.eer", "renamed in place"),
                ("<source_dir>/PositionNNN.mdoc", "renamed in place"),
                ("listfile_Position.txt", "map: original Tomo5 name -> PositionNNN")],
        notes=["RENAME FIRST, then Sort files. The script globs *.mdoc + matching *.eer "
               "together in ONE folder; sorting first breaks that pairing.",
               "*_override.mdoc files are quarantined to mdocs/bad/ (they would take "
               "bogus PositionNNN slots)."]),
    "imod_warp_key": dict(
        reads=[("mdocs/PositionNNN.mdoc", "one sample mdoc, for the tilt order")],
        writes=[("new_imod_conv_key.txt", "IMOD tilt-index -> mdoc-section key")],
        notes=["Feeds remake_mdocs, which uses it to translate IMOD tilt numbers."]),
    "inspect_select": dict(
        reads=[("Thumbnails/*.mrc", "per-series montage (one per tilt series)"),
               ("mdocs/*.mdoc", "to list the series")],
        writes=[("exclusion_list.txt", "manual section: `PositionNNN<tab>1,4-12`"),
                ("mdocs/bad/", "quarantined whole series (mdoc moved here)"),
                ("frames/bad/", "quarantined whole series (.eer moved here)")],
        notes=["A BARE `PositionNNN` line (no tilt numbers) = drop the WHOLE series.",
               "Tilt ranges like `4-12` are expanded before remake_mdocs (its sed can't "
               "parse ranges)."]),
    "remake_mdocs": dict(
        reads=[("mdocs/*.mdoc", "the series to rewrite"),
               ("exclusion_list.txt", "which tilts/series to drop"),
               ("new_imod_conv_key.txt", "IMOD -> mdoc section mapping")],
        writes=[("mdocs/PositionNNN.mdoc", "rewritten without the excluded tilts")],
        notes=["Needs ABSOLUTE paths (it cd's into mdocs/ and reads the lists by name)."]),
    "gain_convert": dict(
        reads=[("gains/<name>.gain", "the raw gain reference (name varies per dataset)")],
        writes=[("gains/original_gain.mrc", "gain as MRC")],
        notes=["The default `gains/original.gain` usually does NOT exist — the real gain "
               "file is auto-detected; check the name."]),
    "gain_reciprocal": dict(
        reads=[("gains/original_gain.mrc", "from gain_convert")],
        writes=[("gains/gain_reciprocal.mrc", "1/gain — this is what create_settings uses")],
        notes=[]),
    "create_settings_fs": dict(
        reads=[("frames/*.eer", "just to locate/describe the data"),
               ("gains/gain_reciprocal.mrc", "recorded as the gain path")],
        writes=[("warp_frameseries.settings", "the frameseries settings file")],
        notes=["Writes NO per-movie data — it only records paths/pixel size/exposure.",
               "--eer_ngroups IS the correct flag on this Warp 2.0 build."]),
    "fs_motion_and_ctf": dict(
        reads=[("frames/*.eer", "the raw movies"),
               ("warp_frameseries.settings", "paths, angpix, gain, exposure")],
        writes=[("warp_frameseries/<movie>.xml", "per-movie motion + CTF metadata"),
                ("warp_frameseries/average/<movie>.mrc", "aligned average (--out_averages)"),
                ("warp_frameseries/averagehalves/", "half-averages (--out_average_halves)")],
        notes=["MUST pass --out_averages (and --out_average_halves). Without them no "
               "aligned averages are written and ts_import fails for EVERY series with "
               "'does not have an aligned average result'.",
               "Idempotent: re-running skips movies that already have XML."]),
    "create_settings_ts": dict(
        reads=[("(nothing yet — tomostar/ may be empty)", "")],
        writes=[("warp_tiltseries.settings", "the tilt-series settings file")],
        notes=["Run BEFORE ts_import. Seeing '0 files found / tomostar not found' here is "
               "EXPECTED — tomostar/ doesn't exist yet."]),
    "ts_import": dict(
        reads=[("mdocs/*.mdoc", "tilt angles, order, exposure"),
               ("warp_frameseries/<movie>.xml", "the per-movie results + aligned averages")],
        writes=[("tomostar/PositionNNN.tomostar", "one per tilt series — the item list"),
                ("warp_tiltseries/PositionNNN.xml", "per-series metadata (created here)")],
        notes=["CANNOT be scoped with --input_data (only --mdocs / --pattern), which is why "
               "subset work uses a separate project folder."]),
    "ts_stack": dict(
        reads=[("tomostar/*.tomostar", "the series"),
               ("warp_frameseries/average/*.mrc", "the aligned averages")],
        writes=[("warp_tiltseries/tiltstack/PositionNNN/PositionNNN.st", "the tilt stack"),
                ("warp_tiltseries/tiltstack/PositionNNN/*.rawtlt", "tilt angles")],
        notes=[]),
    "aretomo": dict(
        reads=[("warp_tiltseries/tiltstack/*/*.st", "the tilt stacks")],
        writes=[("aretomo_output[-vN]/PositionNNN/", "per-series AreTomo output"),
                ("aretomo_output[-vN]/Imod/PositionNNN/PositionNNN.xf",
                 "the ALIGNMENT — this is the only thing Warp needs"),
                ("aretomo_output[-vN]/PARAMETERS.txt", "what was run")],
        notes=["Auto-VERSIONS: each run makes a NEW aretomo_output-vN folder. A killed run "
               "leaves a PARTIAL folder — a later version can have FEWER .xf than an "
               "earlier one. Check: for d in aretomo_output*; do echo \"$d: $(find $d -name '*.xf'|wc -l)\"; done",
               "Set ARETOMO_VOLZ=0 for ALIGNMENT-ONLY (skips the slow tomogram; you only "
               "need the .xf)."]),
    "miss_align": dict(
        reads=[("warp_tiltseries/PositionNNN.xml", "reads AND REWRITES these in place"),
               ("warp_tiltseries/tiltstack/", "the stacks")],
        writes=[("warp_tiltseries/PositionNNN.xml", "refined alignment written back in place"),
                ("warp_tiltseries/models/", "the trained model checkpoints")],
        notes=["REFINES an existing coarse alignment — it CANNOT align raw stacks. Run "
               "AreTomo -> ts_import_alignments -> select FIRST, or you get a featureless "
               "tomogram.",
               "MA_TRAINING_DEVICES must be a SINGLE GPU. Speed comes from MA_RECON_DEVICES."]),
    "miss_align_infer": dict(
        reads=[("<MA_MODEL_RUN_DIR>/iterN/model.ckpt", "models from a finished training run"),
               ("warp_tiltseries/PositionNNN.xml", "reads AND rewrites in place")],
        writes=[("warp_tiltseries/PositionNNN.xml", "refined alignment, no training")],
        notes=["The larger set still needs its OWN coarse alignment imported first."]),
    "ts_import_alignments": dict(
        reads=[("aretomo_output[-vN]/Imod/PositionNNN/PositionNNN.xf", "AreTomo's alignment"),
               ("warp_tiltseries/PositionNNN.xml", "the series to update")],
        writes=[("warp_tiltseries/PositionNNN.xml", "tilt axis + shifts written INTO the xml")],
        notes=["Points at the NEWEST aretomo_output*/Imod/ by default. If that run was killed "
               "it may be INCOMPLETE -> 'Could not find PositionNNN.xf' for most series. "
               "Type a specific version (e.g. aretomo_output-v2/Imod/) to override.",
               "A FAILED import DESELECTS every failed series. Re-enable with sync_selection "
               "-> Select."]),
    "sync_selection": dict(
        reads=[("warp_tiltseries/PositionNNN.xml", "the selection flag lives here")],
        writes=[("warp_tiltseries/PositionNNN.xml", "sets selected/deselected in place")],
        notes=["ANY failed per-item run DESELECTS that item, and downstream steps then "
               "SILENTLY SKIP it. After any partial failure, run this with --select.",
               "WarpTools needs EXACTLY ONE mode flag (--select / --deselect / ...)."]),
    "ts_defocus_hand": dict(
        reads=[("warp_tiltseries/PositionNNN.xml", "CTF fits")],
        writes=[("warp_tiltseries/PositionNNN.xml", "sets the defocus handedness")],
        notes=["Run --check first; only then --set_*. Exactly ONE mode flag."]),
    "ts_ctf": dict(
        reads=[("warp_tiltseries/PositionNNN.xml", "series metadata"),
               ("warp_tiltseries/tiltstack/*/*.st", "the stacks to fit")],
        writes=[("warp_tiltseries/PositionNNN.xml",
                 "CTF written INTO the xml: <CTF><Param Name=\"Defocus\"> (per-series mean, "
                 "um), DefocusDelta/DefocusAngle (astigmatism), and <GridCTF> (per-TILT "
                 "defocus, one Node per tilt)")],
        notes=["No new files — it EDITS the per-series xml in place."]),
    "ts_reconstruct": dict(
        reads=[("warp_tiltseries/PositionNNN.xml", "alignment + CTF"),
               ("warp_tiltseries/tiltstack/*/*.st", "the stacks")],
        writes=[("warp_tiltseries/reconstruction/PositionNNN_<apx>Apx.mrc",
                 "the full tomogram, e.g. Position042_12.56Apx.mrc")],
        notes=["<apx> is the pixel size to TWO DECIMALS: --angpix 10 -> '10.00Apx'.",
               "Leaving --angpix blank means NATIVE (1.57 A) -> ~260x the volume, tens of GB "
               "and ~40 min EACH. Use ~10 A for viewing/picking.",
               "Needs WARP_FORCE_MRC_FLOAT32=1 for IMOD/3dmod to read the output."]),
    "ts_template_match": dict(
        reads=[("warp_tiltseries/reconstruction/PositionNNN_<apx>Apx.mrc",
                "the FULL TOMOGRAM at --tomo_angpix. It REUSES this; it does not rebuild it."),
               ("warp_tiltseries/template/emd_<code>.mrc",
                "the template (auto-downloaded by --template_emdb) or your --template_path"),
               ("warp_tiltseries/PositionNNN.xml", "alignment + CTF")],
        writes=[("warp_tiltseries/matching/PositionNNN_<apx>Apx<SUFFIX>.star",
                 "THE PEAK LIST. <SUFFIX> = --override_suffix verbatim if set, else the "
                 "template-derived name (_emd_70905 / _<template stem>)"),
                ("warp_tiltseries/matching/PositionNNN_<apx>Apx<TEMPLATE>_corr.mrc",
                 "correlation volume — named after the TEMPLATE, NOT your override_suffix"),
                ("warp_tiltseries/matching/PositionNNN_<apx>Apx<TEMPLATE>_angleid.mrc",
                 "best-orientation map (same template-based naming)"),
                ("warp_tiltseries/template/emd_<code>.mrc", "the downloaded EMDB map")],
        notes=["--tomo_angpix MUST equal an --angpix you ALREADY ran ts_reconstruct at, or "
               "every series fails with 'A reconstruction at the desired resolution was not "
               "found.'",
               "--override_suffix renames ONLY the .star. The _corr.mrc / _angleid.mrc keep "
               "the TEMPLATE name — so a viewer needs -mp '*<suffix>.star' but "
               "-cvp '*<template>_corr.mrc'.",
               "--check_hand > 0 is INCOMPATIBLE with --override_suffix: the handedness test "
               "reads back the DEFAULT template-named star and dies. Use check_hand 0 with a "
               "suffix; determine handedness once WITHOUT one.",
               "--npeaks (default 2000) is a HARD per-series CAP. If every series returns "
               "exactly that many, you are capped and losing particles.",
               "--subvolume_size is the matching TILE in TOMOGRAM pixels; it is NOT the "
               "particle box and NOT a big speed lever. Cost is driven by --subdivisions "
               "(orientations)."]),
    "threshold_picks": dict(
        reads=[("<processing>/matching/PositionNNN_<in_suffix>.star",
                "the peak list from template matching")],
        writes=[("<processing>/matching/PositionNNN_<in_suffix>_<out_suffix>.star",
                 "the kept picks, written NEXT TO the input (same dir)")],
        notes=["--in_suffix is a LITERAL string, NOT a glob, and must NOT include '.star'. "
               "Warp builds the name as {item}_{in_suffix}.star. Passing '*.star' searches "
               "for 'PositionNNN_*.star.star' and fails for every series.",
               "Find the right value: ls warp_tiltseries/matching/ and take what sits between "
               "'PositionNNN_' and '.star' (e.g. 12.56Apx_emd_70905).",
               "It reads its input from the OUTPUT processing dir (it augments a matching dir "
               "in place) — which is why a job must have its parent's stars staged into its "
               "own matching/ folder.",
               "Scores are background-normalised, so one --minimum is comparable across "
               "tomograms — but NOT across runs with different settings (whitening shifts the "
               "scale)."]),
    "ts_export_particles": dict(
        reads=[("--input_directory + --input_pattern",
                "e.g. warp_tiltseries/matching + '*12.56Apx_v3-optimized_clean.star'. NOTE: "
                "the pick STARs come from --input_directory, NOT --input_processing."),
               ("warp_tiltseries/PositionNNN.xml", "alignment + CTF, to reconstruct subtomos"),
               ("warp_tiltseries/tiltstack/", "the tilt data")],
        writes=[("<output_processing>/subtomo/*.mrc",
                 "ONE 3D SUBTOMOGRAM PER PARTICLE — this is the bulk (box^3 x 4 bytes each)"),
                ("<output_star>", "the particles STAR RELION will read, e.g. relion4/<tag>/matching.star")],
        notes=["Needs EXACTLY ONE of --3d (RELION 4 subtomograms) or --2d (RELION 5 --tomo).",
               "Image paths in the STAR are RELATIVE to --output_processing, so RELION MUST be "
               "launched from that directory or every path is wrong.",
               "Size check: 578k particles at box 48 ~ 250 GB of subtomograms. Check disk "
               "before a big export."]),
    "relion4_convert": dict(
        reads=[("<project_dir>/<starfile>", "the Warp export STAR (e.g. matching.star)")],
        writes=[("<project_dir>/<stem>_conv.star",
                 "THE RELION-4-FORMAT PARTICLES STAR — this is what you feed RELION"),
                ("<project_dir>/random_subset_ref.mrc",
                 "de-novo initial reference from NREF random particles, already at the "
                 "particles' box + pixel size")],
        notes=["It rewrites the 'subtomo/' path prefix in the STAR to an ABSOLUTE path so "
               "RELION can resolve every subtomogram.",
               "DRY-RUN by default — tick execute to actually convert.",
               "In the RELION GUI, 'Input images STAR file' = the _conv.star, NOT the raw "
               "Warp matching.star."]),
    "relion4_class3d": dict(
        reads=[("<project_dir>/<particles>", "the CONVERTED star (matching_conv.star)"),
               ("REF_MAP", "your reference .mrc (EMDB map or random_subset_ref.mrc)")],
        writes=[("<project_dir>/Class3D/job001/run_itNNN_classNNN.mrc", "the class volumes"),
                ("<project_dir>/Class3D/job001/run_itNNN_data.star", "particle assignments"),
                ("<project_dir>/ref_<angpix>apx_box<box>.mrc", "the RE-SCALED reference")],
        notes=["RELION 4 does NOT auto-rescale the reference. OUTPUT_ANGPIX and BOX must match "
               "the export, and the script pre-scales REF_MAP with relion_image_handler.",
               "Runs relion_refine_mpi FROM <project_dir>. Do NOT also drive the same dir from "
               "the RELION GUI — pick one.",
               "MPI = #GPUs + 1. GPUs are colon-separated for RELION (0:1:2:3)."]),
}

# The job model (why jobs/J###/ exists) — shown as its own block.
JOB_MODEL = """
Every run started from the CARD VIEW is a <b>job</b> with its own processing directory
<code>jobs/J###/</code>. WarpTools' <code>--input_processing</code> / <code>--output_processing</code>
(available on every command) redirect where a run READS its metadata and WRITES its results, so a job
reads its parent's directory and writes its own. That is what lets two variants of the same step
coexist instead of overwriting one another.<br><br>
Bulk data (tilt stacks, raw movies) is resolved through the <code>.settings</code> file and is NOT
duplicated — a branch only costs the per-series XML (~0.4 MB each). Products that a step genuinely
creates (tomograms, subtomograms) live only in the job that made them.<br><br>
<b>Trunk runs</b> (the classic &#9654; Run) skip all this and write to the shared conventional folders
above. Both are fine; jobs are what you want when you're comparing variants.
"""

GOTCHAS = [
    ("A failed item is SILENTLY DESELECTED",
     "Any per-item failure marks that series unselected <i>in its .xml</i>, and every downstream step "
     "then skips it without saying so. After ANY partial failure run "
     "<code>WarpTools change_selection --settings warp_tiltseries.settings --select</code>."),
    ("Pixel sizes are formatted to 2 decimals",
     "<code>--angpix 10</code> produces <code>..._10.00Apx.mrc</code>. Template matching's "
     "<code>--tomo_angpix</code> must match an existing reconstruction EXACTLY."),
    ("Suffixes are how parallel pick sets coexist",
     "One matching/ directory holds many pick sets, told apart only by the suffix in the filename. "
     "<code>--override_suffix</code> names the STAR; the corr/angleid volumes keep the TEMPLATE name."),
    ("in_suffix / input_pattern are different things",
     "<code>threshold_picks --in_suffix</code> is a LITERAL string with no <code>.star</code> and no "
     "wildcard. <code>ts_export_particles --input_pattern</code> IS a glob and DOES include "
     "<code>.star</code>."),
    ("GPU list formats differ per tool",
     "WarpTools/AreTomo want SPACES (<code>0 1 2 3</code>); RELION wants COLONS "
     "(<code>0:1:2:3</code>); miss-alignment wants COMMAS (<code>0,1,2,3</code>). Tomogration "
     "normalises these for you."),
]


def card(stage):
    sid = stage["id"]
    facts = FILE_FACTS.get(sid, {})
    docs = stage.get("docs", {}) or {}
    title = app.stage_title(sid, stage.get("label", sid))
    base = stage.get("base", "") or "(no command — an in-app tool)"
    # keep only the command word(s), not the absolute script path
    base = base.replace(str(REPO) + "/", "")
    ins, outs = app.STAGE_IO.get(sid, ([], []))

    def rows(items, kind):
        if not items:
            return f'<tr><td colspan="2" class="none">(none)</td></tr>'
        out = []
        for path, what in items:
            out.append(f'<tr><td class="path">{E(path)}</td><td class="what">{E(what)}</td></tr>')
        return "".join(out)

    flags = []
    for p in stage.get("params", []):
        f = p.get("flag")
        if not f or p["name"] == "script":
            continue
        d = p.get("default")
        d = "" if d in (None, "", False) else ("on" if d is True else str(d))
        flags.append(f'<span class="flag">{E(str(f))}</span>'
                     + (f'<span class="dflt">{E(d)}</span>' if d else ""))
    notes = "".join(f"<li>{n}</li>" for n in facts.get("notes", []))
    if docs.get("pitfall") and not notes:
        notes = f"<li>{E(docs['pitfall'])}</li>"

    return f"""
<section class="card">
  <h2>{E(title)}</h2>
  <div class="cmd">{E(base)}</div>
  <p class="what-p">{E(docs.get('what', '') or docs.get('why', ''))}</p>

  <div class="io">
    <div class="col">
      <h3 class="reads">READS</h3>
      <div class="dirs">dirs: {E(', '.join(ins) or '—')}</div>
      <table>{rows(facts.get('reads', []), 'r')}</table>
    </div>
    <div class="col">
      <h3 class="writes">WRITES</h3>
      <div class="dirs">dirs: {E(', '.join(outs) or '—')}</div>
      <table>{rows(facts.get('writes', []), 'w')}</table>
    </div>
  </div>

  {'<div class="flags"><b>Flags</b> ' + " ".join(flags) + '</div>' if flags else ''}
  {'<div class="notes"><b>Watch out</b><ul>' + notes + '</ul></div>' if notes else ''}
</section>"""


CSS = """
@page { size: A4; margin: 14mm 12mm; }
* { box-sizing: border-box; }
body { font: 10.5px/1.45 -apple-system, "Helvetica Neue", Arial, sans-serif;
       color: #1a1a1a; margin: 0; }
h1 { font-size: 22px; margin: 0 0 2px; }
.sub { color: #666; font-size: 11px; margin-bottom: 14px; }
.legend { background: #f4f6f8; border: 1px solid #dde3e8; border-radius: 6px;
          padding: 8px 10px; margin-bottom: 14px; font-size: 10px; }
.legend code { background: #e6ebf0; padding: 0 3px; border-radius: 3px; }
.card { border: 1px solid #d3dae0; border-radius: 7px; padding: 9px 11px 8px;
        margin: 0 0 9px; page-break-inside: avoid; background: #fff; }
.card h2 { font-size: 13.5px; margin: 0 0 1px; color: #12304d; }
.cmd { font-family: "SF Mono", Menlo, monospace; font-size: 9.5px; color: #7a4a00;
       background: #fdf6e9; border: 1px solid #f0e2c8; border-radius: 4px;
       padding: 1px 5px; display: inline-block; margin-bottom: 4px; }
.what-p { margin: 3px 0 6px; color: #333; }
.io { display: flex; gap: 10px; }
.col { flex: 1 1 50%; min-width: 0; }
h3 { font-size: 9.5px; letter-spacing: .06em; margin: 0 0 2px; }
h3.reads  { color: #1668b0; } h3.writes { color: #1a7a45; }
.dirs { font-size: 9px; color: #667; margin-bottom: 3px; }
table { width: 100%; border-collapse: collapse; }
td { vertical-align: top; padding: 1.5px 0; border-top: 1px solid #eef1f4; }
td.path { font-family: "SF Mono", Menlo, monospace; font-size: 8.8px; color: #0b3d63;
          width: 47%; padding-right: 6px; word-break: break-all; }
td.what { font-size: 9px; color: #444; }
td.none { color: #999; font-size: 9px; }
.flags { margin-top: 6px; font-size: 9px; color: #555; }
.flag { font-family: "SF Mono", Menlo, monospace; background: #eef2f6; border-radius: 3px;
        padding: 0 3px; margin: 0 1px; font-size: 8.6px; color: #24506f; }
.dflt { color: #888; font-size: 8.4px; margin-left: 2px; }
.notes { margin-top: 6px; background: #fff6f5; border-left: 3px solid #d9534f;
         padding: 4px 8px; font-size: 9px; }
.notes ul { margin: 2px 0 0 12px; padding: 0; }
.notes li { margin-bottom: 2px; }
.group { font-size: 11px; font-weight: 700; color: #7a8794; text-transform: uppercase;
         letter-spacing: .08em; margin: 12px 0 5px; border-bottom: 1px solid #dde3e8;
         padding-bottom: 2px; page-break-after: avoid; }
.big { border: 1px solid #cfd8de; border-radius: 7px; padding: 9px 11px; margin-bottom: 10px;
       background: #f7fafc; font-size: 10px; page-break-inside: avoid; }
.big h2 { font-size: 13px; margin: 0 0 4px; color: #12304d; }
.big code { font-family: "SF Mono", Menlo, monospace; background: #e6ebf0; padding: 0 3px;
            border-radius: 3px; font-size: 9px; }
.gotcha { page-break-inside: avoid; margin-bottom: 5px; }
.gotcha b { color: #a3332f; }
"""


def build_html():
    parts = [f"<!doctype html><meta charset='utf-8'><style>{CSS}</style>",
             "<h1>Tomogration &mdash; pipeline reference</h1>",
             "<div class='sub'>Every command: what it reads, the exact filenames it looks for, "
             "what it writes and where. Generated from the app's own stage definitions.</div>",
             "<div class='legend'><b>Notation</b> &nbsp; "
             "<code>PositionNNN</code> = a tilt series &nbsp;&middot;&nbsp; "
             "<code>&lt;apx&gt;</code> = pixel size to TWO decimals (<code>--angpix 10</code> "
             "&rarr; <code>10.00Apx</code>) &nbsp;&middot;&nbsp; "
             "<code>&lt;processing&gt;</code> = <code>warp_tiltseries/</code> for a trunk run, or "
             "<code>jobs/J###/</code> for a job.</div>",
             f"<div class='big'><h2>The job model &mdash; jobs/J###/</h2>{JOB_MODEL}</div>"]

    seen = set()
    for stage in app.STAGES:
        if stage.get("legacy"):          # retired: kept for old cards, not documented
            continue
        g = stage.get("group", "")
        if g not in seen:
            seen.add(g)
            parts.append(f"<div class='group'>{E(g)}</div>")
        parts.append(card(stage))

    parts.append("<div class='group'>Cross-cutting gotchas</div><div class='big'>")
    for t, b in GOTCHAS:
        parts.append(f"<div class='gotcha'><b>{E(t)}</b> &mdash; {b}</div>")
    parts.append("</div>")
    return "\n".join(parts)


def main():
    out_html = HERE / "tomogration_pipeline_reference.html"
    out_pdf = HERE / "tomogration_pipeline_reference.pdf"
    out_html.write_text(build_html())
    print(f"wrote {out_html}")

    brave = Path("/Applications/Brave Browser.app/Contents/MacOS/Brave Browser")
    if brave.is_file():
        subprocess.run([str(brave), "--headless", "--disable-gpu", "--no-sandbox",
                        f"--print-to-pdf={out_pdf}", "--no-pdf-header-footer",
                        out_html.as_uri()], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"wrote {out_pdf}")
    else:
        print("No Chromium found — open the HTML and Print > Save as PDF.")


if __name__ == "__main__":
    main()
