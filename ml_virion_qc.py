#!/usr/bin/env python3
"""Size QC: which connected components are probably NOT single virions.

    python3 ml_virion_qc.py --components <labels.mrc | dir> \
        --population membrane/population.json --angpix 12.56 \
        --out-folder jobs/J40_virion-size-qc [--tomogram <tomo.mrc|dir>]

Compares every component's voxel volume against the MEASURED population
(population.json from the radius+volume card): components far LARGER than one
virion are flagged as likely clusters (with k = how many virions they probably
contain), components far SMALLER as likely debris/partial shells. This is a
judgement AID, not a filter — nothing is deleted; the verdicts feed the
split-clusters step and your own eyes.

Per components volume, into its own subfolder (batch over a dir):
  size_qc.json      every component: voxels, volume_nm3, r_eff_A, verdict,
                    k_estimate, ratio to the population volume
  size_qc.csv       the same, spreadsheet-shaped
  outliers_only.mrc the components volume with OK labels zeroed — open this
                    over the tomogram in napari and everything visible is
                    suspect (label values are preserved, so napari's label
                    picker reads the same numbers as the report)
  inspect.txt       the sampled inspection list (--sample N per verdict;
                    0 = list every outlier), one "label verdict volume" line
                    per row — the shortlist to walk through in napari

Thresholds are population-relative (--small-frac, --big-frac), not absolute:
the same card works at any binning because everything is computed in nm³ from
--angpix. A missing/empty population.json is an ERROR, not a guess — run the
population measurement card first; that ordering is the point of the design.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import mrcfile
except ImportError:
    sys.exit("ERROR: mrcfile not available — run through "
             "ml_membrane_tool_warp_auto.sh (membrainseg env).")


def read_labels(path):
    with mrcfile.open(path, permissive=True) as m:
        d = np.asarray(m.data)
        vx = float(m.voxel_size.x) if m.voxel_size.x else 0.0
    if not np.issubdtype(d.dtype, np.integer):
        # membrain components are integer-valued but often stored float.
        d = np.rint(d).astype(np.int32)
    return d, vx


def component_volumes(labels):
    """{label: voxels} via one bincount — never a dense mask per label."""
    counts = np.bincount(labels.ravel().clip(min=0))
    return {int(l): int(counts[l]) for l in range(1, len(counts))
            if counts[l] > 0}


def qc_one(comp_path, pop, angpix, small_frac, big_frac, out_dir, sample):
    labels, hdr_vx = read_labels(comp_path)
    vx = float(angpix) if angpix else hdr_vx
    if vx <= 0:
        sys.exit(f"ERROR: no pixel size — pass --angpix (header of "
                 f"{comp_path} says {hdr_vx}).")
    nm3_per_voxel = (vx / 10.0) ** 3
    v_pop = float(pop["whole_virion_nm3"])
    r_pop_A = float(pop.get("radius_A") or 0.0)

    rows = []
    for label, voxels in sorted(component_volumes(labels).items()):
        vol = voxels * nm3_per_voxel
        ratio = vol / v_pop if v_pop > 0 else 0.0
        r_eff_A = ((3.0 * voxels / (4.0 * np.pi)) ** (1.0 / 3.0)) * vx
        if ratio > big_frac:
            verdict, k = "too_big", max(2, int(round(ratio)))
        elif ratio < small_frac:
            verdict, k = "too_small", 0
        else:
            verdict, k = "ok", 1
        rows.append({"label": label, "voxels": voxels,
                     "volume_nm3": round(vol, 1),
                     "r_eff_A": round(r_eff_A, 1),
                     "ratio_to_population": round(ratio, 2),
                     "verdict": verdict, "k_estimate": k})

    os.makedirs(out_dir, exist_ok=True)
    n_ok = sum(1 for r in rows if r["verdict"] == "ok")
    n_big = sum(1 for r in rows if r["verdict"] == "too_big")
    n_small = sum(1 for r in rows if r["verdict"] == "too_small")
    report = {"components": os.path.abspath(str(comp_path)),
              "angpix": vx,
              "population": {"whole_virion_nm3": v_pop, "radius_A": r_pop_A,
                             "n_virions": pop.get("n_virions"),
                             "tomograms": pop.get("tomograms")},
              "small_frac": small_frac, "big_frac": big_frac,
              "n_components": len(rows), "n_ok": n_ok,
              "n_too_big": n_big, "n_too_small": n_small,
              "extra_virions_in_clusters":
                  sum(r["k_estimate"] for r in rows
                      if r["verdict"] == "too_big"),
              "rows": rows}
    (Path(out_dir) / "size_qc.json").write_text(json.dumps(report, indent=1))
    with open(Path(out_dir) / "size_qc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows
                           else ["label"])
        w.writeheader()
        w.writerows(rows)

    # The napari artifact: OK labels zeroed, outlier labels preserved. MRC has
    # no int32 mode (the J44 lesson) — int16 covers any real component count,
    # and a pathological >32767-label volume falls back to float32 rather
    # than dying at the very last write.
    outliers = {r["label"] for r in rows if r["verdict"] != "ok"}
    keep = np.isin(labels, sorted(outliers))
    dtype = np.int16 if (not outliers or max(outliers) <= 32767) else np.float32
    out_vol = np.where(keep, labels, 0).astype(dtype)
    with mrcfile.new(str(Path(out_dir) / "outliers_only.mrc"),
                     overwrite=True) as m:
        m.set_data(out_vol)
        m.voxel_size = vx

    # Inspection shortlist: everything, or N largest per verdict (largest
    # first — the big clusters are what split-clusters must handle).
    def shortlist(verdict):
        vs = [r for r in rows if r["verdict"] == verdict]
        vs.sort(key=lambda r: -r["volume_nm3"])
        return vs if sample <= 0 else vs[:sample]
    lines = [f"{r['label']}\t{r['verdict']}\t{r['volume_nm3']} nm3"
             + (f"\t~{r['k_estimate']} virions" if r["verdict"] == "too_big"
                else "")
             for v in ("too_big", "too_small") for r in shortlist(v)]
    (Path(out_dir) / "inspect.txt").write_text(
        "\n".join(lines) + ("\n" if lines else ""))

    print(f"{Path(str(comp_path)).name}: {len(rows)} components — "
          f"{n_ok} ok, {n_big} likely clusters "
          f"(~{report['extra_virions_in_clusters']} virions inside), "
          f"{n_small} likely debris")
    for ln in lines[:12]:
        print("  " + ln.replace("\t", "  "))
    if len(lines) > 12:
        print(f"  … {len(lines) - 12} more in inspect.txt")
    return report


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--components", required=True,
                    help="components labels .mrc, or a dir of them")
    ap.add_argument("--population", required=True,
                    help="population.json from the radius+volume card")
    ap.add_argument("--angpix", type=float, default=0.0,
                    help="Å/px (0 = read the mrc header)")
    ap.add_argument("--small-frac", type=float, default=0.35,
                    help="below this fraction of a virion volume = debris")
    ap.add_argument("--big-frac", type=float, default=1.7,
                    help="above this multiple of a virion volume = cluster")
    ap.add_argument("--sample", type=int, default=10,
                    help="inspection shortlist per verdict (0 = all)")
    ap.add_argument("--out-folder", required=True)
    ap.add_argument("--tomogram", default="",
                    help="tomogram (or dir) the components came from — "
                    "recorded for the viewer, never read")
    a = ap.parse_args()

    try:
        pop = json.loads(Path(a.population).read_text())
    except (OSError, ValueError) as e:
        sys.exit(f"ERROR: cannot read population stats: {e}\n"
                 f"Run the 'Population radius + volume' card first — QC is "
                 f"only meaningful against a MEASURED population.")
    if not pop.get("whole_virion_nm3"):
        sys.exit("ERROR: population.json has no whole_virion_nm3 — re-run the "
                 "population measurement card.")
    if a.big_frac <= a.small_frac:
        sys.exit("ERROR: --big-frac must exceed --small-frac.")

    src = Path(a.components)
    if src.is_dir():
        vols = sorted(p for p in src.glob("*.mrc")
                      if "outliers_only" not in p.name)
        if not vols:
            sys.exit(f"ERROR: no .mrc in {src}")
        reports = []
        for p in vols:
            sub = Path(a.out_folder) / p.stem
            r = qc_one(p, pop, a.angpix, a.small_frac, a.big_frac, sub,
                       a.sample)
            r["tomogram"] = a.tomogram
            reports.append({"name": p.stem, **{k: r[k] for k in
                            ("n_components", "n_ok", "n_too_big",
                             "n_too_small", "extra_virions_in_clusters")}})
        summary = {"population": a.population, "volumes": reports,
                   "tomogram": a.tomogram}
        Path(a.out_folder, "size_qc_batch.json").write_text(
            json.dumps(summary, indent=1))
        print(f"batch: {len(reports)} volume(s) QC'd -> {a.out_folder}")
    else:
        r = qc_one(src, pop, a.angpix, a.small_frac, a.big_frac,
                   a.out_folder, a.sample)
        r["tomogram"] = a.tomogram
        (Path(a.out_folder) / "size_qc.json").write_text(
            json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
