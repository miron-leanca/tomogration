#!/usr/bin/env python3
"""Membrane parameter search (cards spec draft 2, Part C).

    ml_explore_membrane.py --plan            # what would run, and what is cached
    ml_explore_membrane.py --run             # actually run it
    ml_explore_membrane.py --table           # re-print the results table

Sweeps variants x thresholds x cutoffs x tomograms and reports one row per
cell, so an operating point can be CHOSEN rather than guessed. Deliberately
refuses more than --max-tomograms (3): this is the card you run before
committing 72 tomograms to a setting, and a sweep over the full dataset is the
thing it exists to avoid.

CACHE THE EXPENSIVE AXIS
------------------------
The matrix is uneven by three orders of magnitude:

    segment      ~4 min GPU   depends on (variant, tomogram, model, seg params)
    thresholds   ~seconds     + threshold
    components   ~seconds     + size cutoff
    fit + gates  ~seconds     + gate params

so 4 variants x 3 thresholds x 2 cutoffs x 3 tomograms is 72 cells but only
TWELVE segmentations. Each stage is keyed on the parameters it actually depends
on and cached by content hash, so changing a threshold never re-segments.
Getting this wrong turns a 50-minute exploration into a 5-hour one, which is
why plan_matrix() is pure and tested rather than trusted.

Every row carries RECALL as well as virions accepted. The eight gates in A3
measure precision only: accepting three obvious virions and discarding
everything else scores perfectly on all of them, so a table ranked on
acceptance alone rewards exactly the wrong thing.
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tomogration_variants as VR         # noqa: E402
from tomogration_variants import (        # noqa: E402
    parse_size, nm3_to_voxels, DARK, BRIGHT, UNKNOWN)

STAGES = ("segment", "thresholds", "components", "fit")
# The deconv wrapper's own subfolder naming, at its default 1.0/1.0.
DECONV_SUB = "s1.0_f1.0"


# ------------------------------------------------------------------- inputs
def split_list(values):
    """Flatten what a GUI form types into a list.

    The form gives one flag with everything after it ('--threshold -2 0 2'),
    people also type commas, and argparse's append action wants the flag
    repeated. Accept all three spellings rather than making the user learn
    which one this tool happens to want."""
    out = []
    for v in values or []:
        for part in str(v).replace(",", " ").split():
            if part:
                out.append(part)
    return out


def parse_tomo_dir(text):
    """'name=path' -> (name, path). A bare path is named after its folder, so
    'jobs/J31-bin8/corrected' becomes the variant 'corrected'... which is
    useless, hence the name= form is what the card writes."""
    s = str(text).strip()
    if "=" in s:
        name, path = s.split("=", 1)
        return name.strip(), path.strip().rstrip("/")
    p = s.rstrip("/")
    return (Path(p).name or p), p


def variant_info(name, path, root=".", registry=None):
    """What the sweep needs to know about one input folder.

    angpix comes from the tomogram FILENAME (Warp writes it there) and falls
    back to the registry — never to a default, because a wrong pixel size makes
    every physical size cutoff wrong by the same silent factor."""
    entry = None
    for e in (registry or []):
        if e.get("name") == name or str(e.get("path", "")).rstrip("/") == path:
            entry = e
            break
    angpix = entry.get("angpix") if entry else None
    if not angpix:
        d = Path(root) / path if not os.path.isabs(path) else Path(path)
        try:
            first = sorted(x.name for x in d.glob("*.mrc"))[:1]
        except OSError:
            first = []
        if first:
            angpix = VR.angpix_from_name(first[0])
    return {"name": name, "path": path, "angpix": angpix,
            "polarity": (entry or {}).get("polarity", UNKNOWN),
            "derived_from": None}


# ------------------------------------------------------------------- keying
def key_hash(payload):
    """Short stable hash of a stage's inputs. Sorted keys so dict order never
    changes a cache path."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def stage_keys(cell, seg_params, fit_params):
    """The four cache keys for one matrix cell, each over ONLY what that stage
    depends on. This is the whole caching argument in one function: `segment`
    must not see the threshold, or every threshold re-segments."""
    seg = {"stage": "segment", "variant": cell["variant"],
           "tomogram": cell["tomogram"], **seg_params}
    thr = {"stage": "thresholds", "parent": key_hash(seg),
           "threshold": cell["threshold"], "mode": cell.get("threshold_mode", "absolute")}
    comp = {"stage": "components", "parent": key_hash(thr),
            "cutoff": cell["cutoff"]}
    fit = {"stage": "fit", "parent": key_hash(comp),
           "polarity": cell.get("polarity", UNKNOWN), **fit_params}
    return {"segment": key_hash(seg), "thresholds": key_hash(thr),
            "components": key_hash(comp), "fit": key_hash(fit)}


def plan_matrix(variants, tomograms, thresholds, cutoffs,
                seg_params=None, fit_params=None, threshold_mode="absolute",
                polarities=None):
    """(cells, work) — every matrix cell, and the DISTINCT runs each stage needs.

    `work[stage]` is the set of distinct keys, so len() is the real cost. The
    number that matters: len(work['segment']) is variants x tomograms, NOT the
    number of cells."""
    seg_params = dict(seg_params or {})
    fit_params = dict(fit_params or {})
    polarities = dict(polarities or {})
    cells, work = [], {s: {} for s in STAGES}
    for v in variants:
        for t in tomograms:
            for thr in thresholds:
                for cut in cutoffs:
                    cell = {"variant": v, "tomogram": t, "threshold": thr,
                            "cutoff": cut, "threshold_mode": threshold_mode,
                            "polarity": polarities.get(v, UNKNOWN)}
                    cell["keys"] = stage_keys(cell, seg_params, fit_params)
                    cells.append(cell)
                    for s in STAGES:
                        work[s].setdefault(cell["keys"][s], cell)
    return cells, work


def find_volume(dirpath, stem):
    """The .mrc for one series inside a folder, or "" if it is not there yet.

    Warp writes 'Position003_12.56Apx.mrc', membrain appends its own suffixes,
    and a cache dir holds whatever the stage produced — so a path can never be
    built as f"{stem}.mrc". Matched '_'-anchored, longest first, exactly like
    the wrappers, so Position_1 never claims Position_10's volume.
    """
    d = Path(dirpath)
    if not d.is_dir():
        return ""
    exact = d / f"{stem}.mrc"
    if exact.is_file():
        return str(exact)
    hits = sorted((p for p in d.glob(f"{stem}_*.mrc")), key=lambda p: -len(p.name))
    if hits:
        return str(hits[0])
    # NO "there is only one file, use it" fallback. It resolves for ANY stem —
    # asking for Position999 returned Position003's volume, and Position_1
    # happily took Position_10's — which is how a sweep quietly reports numbers
    # for the wrong tomogram. Some wrappers nest one level (components writes
    # cc<voxels>/, deconv writes s<strength>_f<falloff>/), so look there too,
    # but always '_'-anchored on the stem.
    exact = list(d.glob(f"*/{stem}.mrc"))
    if exact:
        return str(exact[0])
    nested = sorted(d.glob(f"*/{stem}_*.mrc"), key=lambda q: -len(q.name))
    return str(nested[0]) if nested else ""



# --------------------------------------------------------- output file names
# Only these stages are renamed. segment and thresholds outputs are the INPUT
# to the next tool, which globs for its own suffixes (*_scores.mrc) — renaming
# those breaks the chain. components and fit are read only by this sweep (by
# stem-prefix match) and by the viewer, so they are safe to name properly.
RENAMED_STAGES = ("components", "fit")


def cell_tag(cell, stage):
    """The parameters that stage's output ACTUALLY depends on.

    A segmentation is shared by every threshold and cutoff of its variant, so
    tagging it with one cell's threshold would be a lie about what it is. The
    tag matches the cache key's own dependencies, stage by stage."""
    bits = [str(cell["variant"])]
    if stage in ("thresholds", "components", "fit"):
        bits.append(f"thr{cell['threshold']}")
    if stage in ("components", "fit"):
        bits.append("cut" + str(cell.get("cutoff", "")).split("@")[0])
    return "_".join(bits)


def canonical_name(path, cell, stage):
    """What a stage's output SHOULD be called, or "" if it already is.

    Every cell's files used to be named by the tool that wrote them, so a
    components volume for cut=1000 and one for cut=10000 had identical
    basenames in different folders — open both in napari and they are
    indistinguishable. The series stem stays FIRST so find_volume() keeps
    matching; everything that never varies (pixel tag, checkpoint name) goes;
    the parameters that do vary come in.
    """
    p = Path(path)
    base = re.sub(r"\.mrc$", "", p.name, flags=re.I)
    tag = cell_tag(cell, stage)
    if f"_{tag}_" in base or base.endswith(f"_{tag}"):
        return ""                                   # already canonical
    stem = cell["tomogram"]
    m = re.match(r"^fit_(\d+)$", base)
    kind = f"fit{m.group(1)}" if m else stage
    return f"{stem}_{tag}_{kind}.mrc"


def normalise_outputs(stage, cell, folder):
    """Rename a stage's .mrc outputs in place to their canonical names.

    Idempotent, and applied to CACHED folders too, so an existing sweep heals
    itself without re-running anything."""
    d = Path(folder)
    if stage not in RENAMED_STAGES or not d.is_dir():
        return 0
    n = 0
    for src in sorted(list(d.glob("*.mrc")) + list(d.glob("*/*.mrc"))):
        want = canonical_name(src, cell, stage)
        if not want:
            continue
        dst = src.with_name(want)
        if dst.exists():
            continue
        src.rename(dst)
        n += 1
    return n

def stage_command(stage, cell, root, opts):
    """The exact shell command one stage of one cell runs.

    Every stage goes through the SHIPPED wrapper, never membrain directly:
    those wrappers are the ones written against the real --help, and they
    already carry the traps (mkdir the output folder, pass --out-folder
    explicitly, the one-decimal threshold filenames). Pure, so the whole
    executor is testable without membrain installed."""
    k = cell["keys"]
    here = Path(root)
    seg_d = cache_dir(here, "segment", k["segment"])
    thr_d = cache_dir(here, "thresholds", k["thresholds"])
    comp_d = cache_dir(here, "components", k["components"])
    fit_d = cache_dir(here, "fit", k["fit"])
    env, args = {}, []

    if stage == "segment":
        env = {"MB_CONDA_ENV": opts.get("conda_env", "membrainseg"),
               "MB_CKPT": opts["ckpt"], "MB_GPU": opts.get("gpu", "0"),
               "MB_TOMO_LIST": cell["tomogram"]}
        args = [str(HERE / "ml_membrain_segment_warp_auto.sh"),
                cell["variant_path"], str(seg_d)]
    elif stage == "thresholds":
        # Percentile mode is resolved to a real score BEFORE this point, so the
        # wrapper always receives an absolute value — and the realised score is
        # recorded in the row, which is the only way two variants' thresholds
        # can be compared honestly.
        env = {"MB_CONDA_ENV": opts.get("conda_env", "membrainseg"),
               "MB_THRESHOLDS": str(cell.get("threshold_value", cell["threshold"])),
               "MB_TOMO_LIST": cell["tomogram"]}
        args = [str(HERE / "ml_membrain_thresholds_warp_auto.sh"),
                str(seg_d), str(thr_d)]
    elif stage == "components":
        env = {"MB_CONDA_ENV": opts.get("conda_env", "membrainseg"),
               "MB_CC_THRES": str(cell["cutoff_voxels"]),
               "MB_TOMO_LIST": cell["tomogram"]}
        args = [str(HERE / "ml_membrain_components_warp_auto.sh"),
                str(thr_d), str(comp_d)]
    elif stage == "fit":
        env = {"MB_CONDA_ENV": opts.get("conda_env", "membrainseg")}
        args = [str(HERE / "ml_membrane_tool_warp_auto.sh"),
                str(HERE / "ml_fit_virions.py"),
                "--components", cell.get("components_path", str(comp_d)),
                "--tomogram", cell["tomogram_path"],
                "--polarity", cell.get("polarity", UNKNOWN),
                "--min-size", str(cell["cutoff"]),
                "--out-folder", str(fit_d), "--write-masks"]
    else:
        raise ValueError(stage)

    prefix = " ".join(f"{k2}={shlex.quote(str(v))}" for k2, v in sorted(env.items()))
    body = " ".join(shlex.quote(a) for a in args)
    return f"{prefix} bash {body}" if stage != "fit" else f"{prefix} bash {body}"


def collect_metrics(fit_dir, comp_log=""):
    """One results row's numbers, from what the stages actually wrote.

    Reads A3's fits.json rather than re-deriving anything: virions accepted,
    recall, and the per-gate attrition all come from the run that made the
    decisions. Missing or half-written files give a row of None, never a
    crash — a sweep that dies on cell 41 of 54 must still print 40 rows."""
    out = {"virions_accepted": None, "recall": None, "median_residual_A": None,
           "median_support": None, "mean_diameter_nm": None,
           "mean_volume_nm3": None, "gate_attrition": {}}
    try:
        d = json.loads(Path(fit_dir, "fits.json").read_text())
    except (OSError, ValueError):
        return out
    virions = d.get("virions") or []
    out["virions_accepted"] = len(virions)
    out["recall"] = d.get("recall")
    out["gate_attrition"] = d.get("gate_attrition") or {}
    if virions:
        def med(key):
            vals = sorted(v[key] for v in virions if v.get(key) is not None)
            return vals[len(vals) // 2] if vals else None

        def mean(key):
            vals = [v[key] for v in virions if v.get(key) is not None]
            return sum(vals) / len(vals) if vals else None
        out["median_support"] = med("support")
        out["mean_diameter_nm"] = mean("diameter_nm")
        out["mean_volume_nm3"] = mean("volume_nm3")
        out["median_residual_A"] = med("residual_A")
    # membrain components reports what it kept; the wrapper echoes both lines.
    for line in (comp_log or "").splitlines():
        low = line.lower()
        if "found" in low and out.get("components_found") is None:
            for tok in line.replace(":", " ").split():
                if tok.isdigit():
                    out["components_found"] = int(tok)
                    break
        if "relabel" in low:
            for tok in line.replace(":", " ").split():
                if tok.isdigit():
                    out["components_kept"] = int(tok)
                    break
    return out


def cache_dir(root, stage, key):
    return Path(root) / "cache" / stage / key


def cached(root, stage, key, marker="DONE"):
    return (cache_dir(root, stage, key) / marker).is_file()


def mark_done(root, stage, key, payload=None, marker="DONE"):
    d = cache_dir(root, stage, key)
    d.mkdir(parents=True, exist_ok=True)
    (d / marker).write_text(json.dumps(payload or {}, indent=1))
    return d


# ------------------------------------------------------------------ ranking
def normalise(values):
    """Scale to 0..1 by the maximum. Empty or all-zero -> zeros."""
    vals = [v if isinstance(v, (int, float)) and v == v else 0.0 for v in values]
    top = max(vals) if vals else 0.0
    return [(v / top if top > 0 else 0.0) for v in vals]


def rank_rows(rows):
    """Score every row on virions accepted AND recall, and sort.

    The spec is explicit that neither alone is a valid objective: acceptance
    alone rewards a gate set that keeps three easy virions, and recall alone
    rewards keeping everything. Both columns are normalised across the sweep and
    combined with a harmonic mean, which is small unless BOTH are high. The raw
    columns stay in the table — the composite is a sort order, not evidence."""
    acc = normalise([r.get("virions_accepted", 0) for r in rows])
    rec = normalise([r.get("recall", 0.0) for r in rows])
    for r, a, c in zip(rows, acc, rec):
        r["score"] = (0.0 if (a + c) == 0 else 2 * a * c / (a + c))
    return sorted(rows, key=lambda r: -r["score"])



# ------------------------------------------------------------------ analysis
# Everything the analysis window plots, computed here so it is testable without
# Qt. Each function returns plain lists of (label, value) — the dialog only
# draws.
AXES = ("variant", "threshold", "cutoff")


def filter_rows(rows, show=None):
    """Rows matching a {'variant': {...}, 'threshold': {...}, ...} selection.

    An axis absent from `show`, or with an empty set, means "everything on that
    axis" — an empty tick-list reads as "no filter", not "no rows", because a
    dialog that blanks itself when you untick one box is useless."""
    show = show or {}
    out = []
    for r in rows:
        keep = True
        for ax in AXES:
            want = show.get(ax)
            if want and str(r.get(ax)) not in {str(w) for w in want}:
                keep = False
                break
        if keep:
            out.append(r)
    return out


def axis_values(rows, axis):
    """The distinct values a sweep actually used on one axis, in run order."""
    seen = []
    for r in rows:
        v = str(r.get(axis))
        if v not in seen:
            seen.append(v)
    return seen


def _mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and v == v]
    return sum(vals) / len(vals) if vals else None


def group_stats(rows, axis, metric):
    """[(value, mean, n)] for one metric grouped along one axis.

    Averaged ACROSS tomograms: the question the card answers is "which variant
    /threshold/cutoff", and a single tomogram's count is noise against that."""
    out = []
    for v in axis_values(rows, axis):
        sub = [r for r in rows if str(r.get(axis)) == v]
        out.append((v, _mean([r.get(metric) for r in sub]), len(sub)))
    return out


def tradeoff_points(rows):
    """[(label, recall, virions, row)] — the scatter the choice really lives in.

    Neither axis alone is a valid objective, so plotting them against each
    other is the honest picture: up-and-right is better, and a point that is
    high on one axis only is visibly not."""
    pts = []
    for r in rows:
        rec, n = r.get("recall"), r.get("virions_accepted")
        if rec is None or n is None:
            continue
        label = (f"{r.get('variant')} thr={r.get('threshold')} "
                 f"cut={r.get('cutoff')} {r.get('tomogram')}")
        pts.append((label, float(rec), float(n), r))
    return pts


def sphere_vs_ellipsoid(rows):
    """[(label, sphere_nm3, ellipsoid_nm3, ratio)] per accepted virion.

    The reported volume against the free-ellipsoid one for the SAME component.
    On real data this is the evidence for the constraint; if the ratio here is
    near 1.0 the wedge is not distorting these virions much, and if it is ~1.5
    it is exactly the effect the synthetic test predicts."""
    out = []
    for r in rows:
        for v in r.get("virions") or []:
            sph, ell = v.get("volume_nm3"), v.get("ellipsoid_volume_nm3")
            if not sph or not ell:
                continue
            out.append((f"{r.get('variant')} {r.get('tomogram')} #{v.get('label')}",
                        float(sph), float(ell), float(ell) / float(sph)))
    return out


def summarise(rows):
    """One line per axis value for every metric the window plots."""
    return {ax: {m: group_stats(rows, ax, m)
                 for m in ("virions_accepted", "recall", "mean_diameter_nm",
                           "mean_volume_nm3", "median_residual_A")}
            for ax in AXES}

# ------------------------------------------------------------------ commands
def segment_cmd(cell, out_dir, ckpt, gpu, extra=""):
    return (f"bash {shlex.quote(str(HERE / 'ml_membrain_segment_warp_auto.sh'))} "
            f"{shlex.quote(cell['variant_path'])} {shlex.quote(str(out_dir))}")


def viewer_cmd(row, viewer=None):
    """The one-click 'open in viewer' for a row: tomogram, its components, and
    the accepted fits as shells. tomoview takes the tomogram FIRST."""
    v = viewer or str(HERE / "tomoview.py")
    parts = [v, row.get("tomogram_path", ""), row.get("components_path", "")]
    fits = row.get("fit_masks") or []
    return " ".join(shlex.quote(p) for p in parts + list(fits) if p)


def compare_cmd(row_a, row_b, viewer=None):
    """Two fit sets against ONE tomogram, for 'compare two rows'."""
    v = viewer or str(HERE / "tomoview.py")
    parts = [v, row_a.get("tomogram_path", "")]
    parts += [row_a.get("components_path", "")] + list(row_a.get("fit_masks") or [])
    parts += [row_b.get("components_path", "")] + list(row_b.get("fit_masks") or [])
    return " ".join(shlex.quote(p) for p in parts if p)


# -------------------------------------------------------------------- table
COLUMNS = [("variant", "variant", "{}"), ("threshold", "thr", "{}"),
           ("cutoff", "cutoff", "{}"), ("tomogram", "tomogram", "{}"),
           ("components_found", "found", "{}"), ("components_kept", "kept", "{}"),
           ("virions_accepted", "virions", "{}"), ("recall", "recall", "{:.3f}"),
           ("median_residual_A", "resid A", "{:.0f}"),
           ("median_support", "support", "{:.2f}"),
           ("mean_diameter_nm", "diam nm", "{:.0f}"),
           ("mean_volume_nm3", "vol nm3", "{:.0f}"),
           ("score", "score", "{:.3f}")]


def format_table(rows):
    head = "  ".join(f"{h:>9}" for _, h, _ in COLUMNS)
    out = [head, "-" * len(head)]
    for r in rows:
        cells = []
        for key, _h, fmt in COLUMNS:
            v = r.get(key)
            try:
                if v is None or (isinstance(v, float) and v != v):
                    cells.append(f"{'—':>9}")   # NaN is a gate that was OFF
                    continue
                cells.append(f"{fmt.format(v):>9}")
            except (TypeError, ValueError):
                cells.append(f"{str(v):>9}")
        out.append("  ".join(cells))
    return "\n".join(out)


def attrition_table(rows):
    """How many virions each gate removed, summed over the sweep. The point of
    a sweep is often 'which gate is doing the damage', and that is invisible in
    the per-cell counts."""
    tally = {}
    for r in rows:
        for gate, n in (r.get("gate_attrition") or {}).items():
            tally[gate] = tally.get(gate, 0) + int(n)
    return sorted(tally.items(), key=lambda kv: -kv[1])


# --------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".",
                    help="project root — every relative path below is resolved "
                         "against it, and the variant registry (if any) is read "
                         "from it. Normally '.'")
    ap.add_argument("--out", default="membrane/explore",
                    help="everything this run writes: cache/, results.json")
    ap.add_argument("--tomo-dir", nargs="+", default=[], metavar="NAME=PATH",
                    help="an input tomogram folder to compare, as name=path. "
                         "Repeat or list: --tomo-dir raw=warp_tiltseries/"
                         "reconstruction isonet2=jobs/J31/corrected")
    ap.add_argument("--derive-deconv", default="",
                    help="name of a --tomo-dir to deconvolve into an extra "
                         "variant called 'deconv', so raw vs deconvolved vs "
                         "IsoNet get compared in one sweep")
    ap.add_argument("--variant", nargs="+", default=[],
                    help="registry variant names, if a registry exists")
    ap.add_argument("--tomogram", nargs="+", default=[],
                    help="series stems to sweep over (2-3)")
    ap.add_argument("--threshold", nargs="+", default=[],
                    help="score thresholds (default -2 0 2)")
    ap.add_argument("--threshold-mode", choices=["absolute", "percentile"],
                    default="absolute",
                    help="MemBrain scores shift with input contrast, so -2 on a "
                         "deconvolved tomogram and -2 on an IsoNet one are "
                         "different operating points. percentile cuts at the Nth "
                         "percentile of each score map instead.")
    ap.add_argument("--cutoff", nargs="+", default=[],
                    help="smallest component to keep, as a PHYSICAL size. "
                         "'1000@12.56' means 'as big as 1000 voxels are at "
                         "12.56 A/px' and is re-resolved for each variant's own "
                         "pixel size (8000 voxels at 6.28). '520nm3' says the "
                         "same thing as a volume. A bare voxel count is refused, "
                         "because it means different objects at different "
                         "binnings. (default 1000@12.56 10000@12.56)")
    ap.add_argument("--max-tomograms", type=int, default=3)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--ckpt", default="",
                    help="membrain-seg model checkpoint. Required with --run; "
                         "it is what the segment stage cannot start without.")
    ap.add_argument("--plan", action="store_true", help="print the plan and exit")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--table", action="store_true", help="re-print results.json")
    a = ap.parse_args(argv)

    out = Path(a.out)
    results_path = out / "results.json"

    if a.table:
        if not results_path.is_file():
            sys.exit(f"no results yet: {results_path}")
        rows = json.loads(results_path.read_text())["rows"]
        print(format_table(rank_rows(rows)))
        return 0

    reg = VR.load(a.root).get("variants", [])
    tomograms = split_list(a.tomogram)
    thresholds = split_list(a.threshold) or ["-2", "0", "2"]
    cutoffs = split_list(a.cutoff) or ["1000@12.56", "10000@12.56"]

    # Inputs: explicit folders first (what the card writes), registry names
    # second, and the whole registry only if neither was given.
    infos = []
    for spec in split_list(a.tomo_dir):
        name, path = parse_tomo_dir(spec)
        infos.append(variant_info(name, path, a.root, reg))
    for name in split_list(a.variant):
        e = next((x for x in reg if x.get("name") == name), None)
        if e is None:
            sys.exit(f"'{name}' is not in the registry under {a.root}. "
                     f"Registry has: {', '.join(sorted(x['name'] for x in reg)) or '(empty)'}\n"
                     f"Or name the folder directly: --tomo-dir {name}=<path>")
        infos.append(variant_info(name, e.get("path", ""), a.root, reg))
    if not infos and reg:
        infos = [variant_info(e["name"], e.get("path", ""), a.root, reg) for e in reg]
    if not infos:
        sys.exit("no input tomograms. Name the folders to compare, e.g.\n"
                 "    --tomo-dir raw=warp_tiltseries/reconstruction "
                 "isonet2=jobs/J31-bin8-isonet2model/corrected")

    # The derived variant: deconvolving the raw input gives a third thing to
    # compare without reconstructing anything. It is PLANNED here, and prepared
    # by the deconvolve step — one run per (source variant, tomogram).
    derived = []
    if a.derive_deconv:
        src = next((i for i in infos if i["name"] == a.derive_deconv), None)
        if src is None:
            sys.exit(f"--derive-deconv {a.derive_deconv!r} names no input. "
                     f"Have: {', '.join(i['name'] for i in infos)}")
        # ml_membrain_deconv_warp_auto.sh writes each parameter variant into
        # its own s<strength>_f<falloff>/ subfolder, so THAT is the folder the
        # sweep segments — pointing at the base gives an empty directory.
        derived.append({"name": "deconv",
                        "path": f"{str(a.out).rstrip('/')}/deconv/{DECONV_SUB}",
                        "base": f"{str(a.out).rstrip('/')}/deconv",
                        "source_path": src["path"],
                        "angpix": src["angpix"], "polarity": src["polarity"],
                        "derived_from": src["name"]})
        infos += derived

    by_name = {i["name"]: i for i in infos}
    variants = [i["name"] for i in infos]

    if not tomograms:
        sys.exit("--tomogram is required: name 2-3 series stems.")
    if len(tomograms) > a.max_tomograms:
        sys.exit(f"{len(tomograms)} tomograms — this card is for finding an "
                 f"operating point on {a.max_tomograms} or fewer. Run the "
                 f"production cards for the full dataset.")

    polarities = {n: by_name[n].get("polarity", UNKNOWN) for n in variants}

    cells, work = plan_matrix(variants, tomograms, thresholds, cutoffs,
                              threshold_mode=a.threshold_mode,
                              polarities=polarities)

    print("inputs:")
    for i in infos:
        ang = f"{i['angpix']:.2f} A/px" if i["angpix"] else "pixel size UNKNOWN"
        src = f"  <- deconvolved from {i['derived_from']}" if i["derived_from"] else ""
        print(f"  {i['name']:<12} {ang:>16}  polarity {i['polarity']:<7} "
              f"{i['path']}{src}")
    if derived:
        n_prep = len(tomograms) * len(derived)
        print(f"  (preparing the derived variant costs {n_prep} deconvolve "
              f"run(s), seconds each)")
    print()
    print(f"{len(cells)} matrix cells "
          f"({len(variants)} variants x {len(thresholds)} thresholds x "
          f"{len(cutoffs)} cutoffs x {len(tomograms)} tomograms)")
    print("distinct runs per stage, after caching:")
    for s in STAGES:
        todo = [k for k in work[s] if not cached(out, s, k)]
        print(f"  {s:<12} {len(work[s]):>4} distinct   "
              f"{len(todo):>4} to run   {len(work[s]) - len(todo):>4} cached")
    seg_n = len(work["segment"])
    print(f"\nsegmentation runs {seg_n} times, not {len(cells)} — "
          f"~{seg_n * 4} min of GPU rather than ~{len(cells) * 4}.")

    # Size cutoffs resolve per variant: a bare voxel count means an 8x
    # different object at bin4 than at bin8.
    print("\nsize cutoffs, resolved per variant:")
    for cut in cutoffs:
        try:
            _raw, nm3 = parse_size(cut)
        except ValueError as e:
            sys.exit(f"bad --cutoff {cut!r}: {e}")
        bits = []
        for n in variants:
            ang = by_name[n].get("angpix")
            bits.append(f"{n}={nm3_to_voxels(nm3, ang)}vx" if ang
                        else f"{n}=? (pixel size unknown)")
        print(f"  {cut:>14} = {nm3:8.1f} nm3   " + "  ".join(bits))

    unknown_pol = [n for n in variants if polarities.get(n) not in (DARK, BRIGHT)]
    if unknown_pol:
        print(f"\nWARNING: polarity unmeasured for {', '.join(unknown_pol)} — the "
              f"density-support gate will be DISABLED for those cells, so "
              f"every surface passes it.\n"
              f"         Fix, in order: run 'Membrane: segment' on one "
              f"tomogram of each variant (the polarity check reads which "
              f"voxels are membrane off the segmentation), then 'Variant "
              f"polarity' on that pair. Once per VARIANT, not per tomogram.")

    if not a.run:
        print("\n" + "=" * 67)
        print("PLAN ONLY — nothing above has been run, no files written.")
        print("Tick off 'Plan only' (or pass --run) to execute the sweep.")
        print("=" * 67)
        return 0

    # ------------------------------------------------------------- execute
    if not a.ckpt:
        sys.exit("--ckpt is required to run: the membrain-seg model checkpoint.")
    opts = {"ckpt": a.ckpt, "gpu": a.gpu,
            "conda_env": os.environ.get("MB_CONDA_ENV", "membrainseg")}
    out.mkdir(parents=True, exist_ok=True)

    # Resolve every per-cell path ONCE, so a stage command stays a pure
    # function of its cell and nothing re-derives a path mid-run.
    for c in cells:
        e = by_name[c["variant"]]
        c["variant_path"] = e.get("path", "")
        # NOT resolved here: a DERIVED variant's volumes do not exist yet —
        # they are made by the derive step below. Resolving now gave every
        # deconv cell an empty path and skipped all 18 of them, ten lines
        # before the files appeared. Resolved at fit time instead.
        c["cutoff_voxels"] = nm3_to_voxels(parse_size(c["cutoff"])[1],
                                           e.get("angpix") or 1.0)
        # The components VOLUME lives inside the cache dir and does not exist
        # until that stage has run, so it is resolved in the fit loop below.
        c["components_dir"] = str(cache_dir(out, "components",
                                            c["keys"]["components"]))

    failures = []

    # ---- prepare derived variants -----------------------------------------
    # Nothing else creates these. Before this, every deconv cell failed at the
    # segment stage with "input dir not found" and took its 18 rows with it.
    for info in derived:
        for stem in tomograms:
            if find_volume(info["path"], stem):
                continue
            cmd = (f"MB_CONDA_ENV={shlex.quote(opts['conda_env'])} "
                   f"MB_TOMO_LIST={shlex.quote(stem)} "
                   f"bash {shlex.quote(str(HERE / 'ml_membrain_deconv_warp_auto.sh'))} "
                   f"{shlex.quote(info['source_path'])} "
                   f"{shlex.quote(info['base'])}")
            print(f"[derive {info['name']}] {stem}\n  $ {cmd}")
            rc = subprocess.call(cmd, shell=True)
            if rc != 0 or not find_volume(info["path"], stem):
                print(f"  FAILED to derive {info['name']} for {stem} "
                      f"— its cells will show dashes")
                failures.append({"stage": "derive", "key": stem, "exit": rc,
                                 "variant": info["name"], "tomogram": stem})

    for stage in STAGES:
        # Heal names in folders that are already cached, so an existing sweep
        # gets unique filenames without re-running a single GPU minute.
        for k, c in work[stage].items():
            if cached(out, stage, k):
                normalise_outputs(stage, c, cache_dir(out, stage, k))
        todo = [(k, c) for k, c in work[stage].items() if not cached(out, stage, k)]
        print(f"\n=== {stage}: {len(todo)} to run "
              f"({len(work[stage]) - len(todo)} cached) ===")
        for i, (key, cell) in enumerate(todo, 1):
            if stage == "fit":
                # Both inputs are produced by earlier stages, so they can only
                # be resolved now. Missing means an upstream cell failed —
                # record that plainly instead of handing a directory to mrcfile
                # and printing a traceback per cell.
                comps = find_volume(cell["components_dir"], cell["tomogram"])
                if not cell.get("tomogram_path"):
                    cell["tomogram_path"] = find_volume(cell["variant_path"],
                                                        cell["tomogram"])
                if not comps or not cell["tomogram_path"]:
                    missing = ("components volume" if not comps
                               else "tomogram volume")
                    print(f"[fit {i}/{len(todo)}] SKIP {cell['variant']} "
                          f"{cell['tomogram']} — no {missing} "
                          f"(an earlier stage did not produce it)")
                    failures.append({"stage": stage, "key": key, "exit": None,
                                     "variant": cell["variant"],
                                     "tomogram": cell["tomogram"],
                                     "why": f"no {missing}"})
                    continue
                cell["components_path"] = comps
            cmd = stage_command(stage, cell, out, opts)
            print(f"[{stage} {i}/{len(todo)}] {cell['variant']} {cell['tomogram']} "
                  f"thr={cell['threshold']} cut={cell['cutoff']}")
            print(f"  $ {cmd}")
            rc = subprocess.call(cmd, shell=True)
            if rc != 0:
                # One dead cell must not cost the other 53: record it, carry on,
                # and let its row show dashes.
                print(f"  FAILED (exit {rc}) — continuing with the rest")
                failures.append({"stage": stage, "key": key, "exit": rc,
                                 "variant": cell["variant"],
                                 "tomogram": cell["tomogram"]})
                continue
            mark_done(out, stage, key, {"variant": cell["variant"],
                                        "tomogram": cell["tomogram"],
                                        "command": cmd})
            normalise_outputs(stage, cell, cache_dir(out, stage, key))

    # ------------------------------------------------------------- results
    rows = []
    for c in cells:
        fit_d = cache_dir(out, "fit", c["keys"]["fit"])
        row = {"variant": c["variant"], "tomogram": c["tomogram"],
               "threshold": c["threshold"], "cutoff": c["cutoff"],
               "cutoff_voxels": c["cutoff_voxels"],
               "tomogram_path": (c.get("tomogram_path")
                                 or find_volume(c["variant_path"], c["tomogram"])),
               # Resolved for EVERY row, not just cells whose fit ran this
               # time: a cached cell never entered the fit loop, so it fell
               # back to the cache DIRECTORY and the viewer offered a hash
               # like '751e88af5c27' as something to open.
               "components_path": (c.get("components_path")
                                   or find_volume(c["components_dir"],
                                                  c["tomogram"])),
               "fit_masks": sorted(str(p) for p in Path(fit_d).glob("*fit*.mrc")),
               "keys": c["keys"]}
        row.update(collect_metrics(fit_d))
        rows.append(row)
    rows = rank_rows(rows)
    results_path.write_text(json.dumps({"rows": rows, "failures": failures},
                                       indent=1))

    print("\n" + format_table(rows))
    att = attrition_table(rows)
    if att:
        print("\ngates, by virions removed across the sweep:")
        for gate, n in att:
            print(f"  {gate:<16} {n}")
    if failures:
        print(f"\n{len(failures)} cell-stage(s) failed; those rows show dashes:")
        for f in failures[:8]:
            print(f"  {f['stage']:<11} {f['variant']}/{f['tomogram']} exit {f['exit']}")

    # One line per row, best first, so a promising cell opens in one paste.
    viewer_sh = out / "open_in_viewer.sh"
    viewer_sh.write_text(
        "#!/bin/bash\n# Rows best-first. Run a line to open that cell in napari\n"
        "# (tomogram, its components, and every accepted fit as a shell).\n\n"
        + "".join(
            f"# {r['variant']} thr={r['threshold']} cut={r['cutoff']} "
            f"{r['tomogram']}  virions={r.get('virions_accepted')} "
            f"recall={r.get('recall')}\n{viewer_cmd(r)}\n\n" for r in rows))
    viewer_sh.chmod(0o755)
    print(f"\nresults : {results_path}")
    print(f"viewer  : {viewer_sh}  (one line per row, best first)")
    print(f"compare : ml_explore_membrane.py --table --out {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
