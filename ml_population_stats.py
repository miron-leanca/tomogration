#!/usr/bin/env python3
"""Measure a dataset's virion population BEFORE any fitting uses it.

    ml_population_stats.py --components <folder> --tomograms <folder> \
                           --polarity dark --out membrane/population.json

Every tomogram is measured INDEPENDENTLY and the results are pooled at the end.
That is the whole point: ml_fit_virions, given a store, prefers the stored
radius over learning its own — so running it tomogram by tomogram would make
the second inherit the first, the third inherit that, and the "population"
would be a measurement of tomogram A wearing a dataset's name.

Here nothing is shared until every tomogram has had its say. Each contributes
only its WHOLE virions — components whose surface is complete enough to measure
a radius from (pass 1 of the fitter, same criteria) — and the population radius
is the MEDIAN across all of them, which a couple of merged blobs cannot drag.

Writes the same population.json the fitter reads, plus the per-tomogram
breakdown so a disagreeing tomogram is visible rather than averaged away.

Needs numpy + mrcfile: run from membrainseg.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ml_fit_virions as FIT              # noqa: E402  (pass 1 lives there)
from tomogration_variants import parse_size, nm3_to_voxels, UNKNOWN   # noqa: E402


def whole_virions(components_path, min_size_nm3, diam_range):
    """[(label, radius_px, voxels)] for components complete enough to measure.

    Same test as the fitter's pass 1 — not a re-implementation, because two
    definitions of 'whole virion' that drift apart would make the population
    radius describe a different set from the one it is later applied to.

    min_size is PHYSICAL (nm³) and resolved against THIS volume's own pixel
    size — resolving it once at an assumed 12.56 Å/px made the cutoff an 8×
    different object on a bin4 volume, exactly the trap the '@apix' size
    syntax exists to prevent."""
    d, vx = FIT.load_mrc(components_path)
    min_voxels = nm3_to_voxels(min_size_nm3, vx)
    shape = np.array(d.shape)
    out = []
    for l in [int(x) for x in np.unique(d[d > 0])]:
        idx = np.argwhere(d == l).astype(float)
        if len(idx) < min_voxels:
            continue
        if (idx.min(0) <= 2).any() or (idx.max(0) >= shape - 3).any():
            continue                       # touches the edge: truncated, not whole
        c0, _r = FIT.fit_sphere(idx)
        v = idx - c0
        r = np.linalg.norm(v, axis=1)
        u = v / np.maximum(r[:, None], 1e-9)
        cov, gap = FIT.coverage(u)
        R_band, band = FIT.radius_away_from_gap(r, u, gap)
        diam_nm = 2 * R_band * vx / 10.0
        resid = float(np.median(np.abs(r[band] - R_band))) * vx
        if (cov * 100 >= 58.0 and diam_range[0] <= diam_nm <= diam_range[1]
                and np.isfinite(resid) and resid <= 45.0):
            out.append((l, float(R_band), int(len(idx))))
    return out, vx


def pool(per_tomogram):
    """The dataset's population from every tomogram's independent measurement.

    Median, not mean: one tomogram full of merged blobs shifts a mean and
    barely moves a median. The spread is reported so 'is this a population or
    three different populations?' is answerable rather than assumed."""
    radii = [r for rows in per_tomogram.values() for _l, r, _n in rows]
    voxels = [n for rows in per_tomogram.values() for _l, _r, n in rows]
    if not radii:
        return {}
    med = float(np.median(radii))
    mad = float(np.median(np.abs(np.asarray(radii) - med)))
    return {"radius_px": med, "mad_px": mad, "n_virions": len(radii),
            "median_voxels": float(np.median(voxels)),
            "spread_pct": (100.0 * mad / med if med else 0.0),
            "radii_px": [float(r) for r in radii]}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--components", required=True,
                    help="folder of labelled components (a cc<voxels>/ folder)")
    ap.add_argument("--out", default="membrane/population.json")
    ap.add_argument("--out-folder", default="",
                    help="write population.json INTO this folder instead. The "
                         "job card uses this so the job owns a directory it "
                         "can be re-run over, rather than a loose file that "
                         "several jobs would silently overwrite.")
    ap.add_argument("--min-size", default="1000@12.56")
    ap.add_argument("--diam-range", type=float, nargs=2, default=[60.0, 160.0])
    ap.add_argument("--tomo-list", default="",
                    help="space/comma list of stems; blank = every component "
                         "volume in the folder")
    a = ap.parse_args(argv)
    if str(a.out_folder).strip():
        a.out = str(Path(a.out_folder.strip()) / "population.json")

    comp_dir = Path(a.components)
    if not comp_dir.is_dir():
        sys.exit(f"not a folder: {comp_dir}")
    wanted = set(str(a.tomo_list).replace(",", " ").split())
    files = sorted(comp_dir.glob("*.mrc"))
    if wanted:
        files = [f for f in files
                 if any(f.name.startswith(w + "_") or f.stem == w for w in wanted)]
    if not files:
        sys.exit(f"no component volumes in {comp_dir}")

    print(f"measuring {len(files)} tomogram(s) independently — nothing is "
          f"shared until every one has contributed\n")
    per_tomo, angpix = {}, None
    min_size_nm3 = parse_size(a.min_size)[1]
    for i, f in enumerate(files, 1):
        rows, vx = whole_virions(f, min_size_nm3, a.diam_range)
        angpix = angpix or vx
        per_tomo[f.stem] = rows
        diam = [2 * r * vx / 10.0 for _l, r, _n in rows]
        print(f"[{i:3d}/{len(files)}] {f.stem[:52]:<54} "
              f"{len(rows):3d} whole"
              + (f"   median {np.median(diam):5.1f} nm" if diam else ""))

    stats = pool(per_tomo)
    if not stats:
        sys.exit("\nNo whole virions found in any tomogram — nothing to measure.")
    vx = angpix or 12.56
    radius_A = stats["radius_px"] * vx
    print(f"\n{'=' * 62}")
    print(f"population: {stats['n_virions']} whole virions across "
          f"{sum(1 for r in per_tomo.values() if r)} tomogram(s)")
    print(f"radius    : {radius_A:.0f} A   diameter {2 * radius_A / 10:.1f} nm")
    print(f"spread    : +/- {stats['mad_px'] * vx:.0f} A "
          f"({stats['spread_pct']:.1f}% MAD) — over ~10% and these may not be "
          f"one population")
    # Volume measured PER VIRION and then pooled, not computed from the median
    # radius: a 5% spread in radius is a ~15% spread in volume (it goes as R^3),
    # so quoting one number from the median radius hides most of the variation
    # in the quantity people actually compare between conditions.
    vols = [FIT.virion_volume_nm3(r, vx) for r in stats["radii_px"]]
    v_med = float(np.median(vols))
    v_mad = float(np.median(np.abs(np.asarray(vols) - v_med)))
    print(f"volume    : {v_med:,.0f} nm3 median  +/- {v_mad:,.0f} "
          f"({100 * v_mad / v_med if v_med else 0:.1f}% MAD)")
    print(f"            range {min(vols):,.0f} to {max(vols):,.0f} nm3 "
          f"across {len(vols)} virions")
    print(f"            (volume goes as R^3, so its spread is ~3x the "
          f"radius spread — expected here, not a problem)")
    print(f"{'=' * 62}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "radius_A": radius_A,
        # Kept for the fitter, which reads this key.
        "whole_virion_nm3": FIT.virion_volume_nm3(stats["radius_px"], vx),
        # The measured distribution, which is what a comparison between
        # conditions actually needs.
        "volume_nm3_median": v_med,
        "volume_nm3_mad": v_mad,
        "volume_nm3_min": float(min(vols)),
        "volume_nm3_max": float(max(vols)),
        "diameter_nm_median": 2 * radius_A / 10.0,
        "virion_volumes_nm3": [float(v) for v in vols],
        "virion_radii_A": [float(r) * vx for r in stats["radii_px"]],
        "n_virions": stats["n_virions"],
        # The median SHELL voxel count + the pixel size it was counted at —
        # what the fitter's rescue gate needs (comparing a solid-sphere nm³
        # to a shell's voxel count was a unit mix that let debris through).
        "whole_virion_voxels": stats["median_voxels"],
        "mad_A": stats["mad_px"] * vx,
        "spread_pct": stats["spread_pct"],
        "angpix": vx,
        "measured_from": str(comp_dir),
        "tomograms": sorted(k for k, v in per_tomo.items() if v),
        "per_tomogram": {k: {"n_whole": len(v),
                             "median_radius_A": (float(np.median([r for _l, r, _n in v])) * vx
                                                 if v else None)}
                         for k, v in per_tomo.items()},
    }, indent=1))
    print(f"\nwrote {out}")
    print(f"Point the fit card's 'Population store' at it: every tomogram is "
          f"then fitted against the SAME measured radius.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
