#!/usr/bin/env python3
"""Membrane location estimation (cards spec draft 2, A3) — absorbs fitpop2.py.

    ml_fit_virions.py --components comps.mrc --tomogram tomo.mrc \\
                      --polarity dark --out-folder popfits [--write-masks]

Two passes, as in the prototype: a free fit finds whole virions and learns the
population radius, then fixed-radius centre-only fits rescue partial arcs. Eight
gates reject: angular extent, voxel ratio, prior residual, self-measured size,
self residual, density support, in-volume fraction, mutual exclusion.

WHAT CHANGED FROM fitpop2.py, and why
-------------------------------------
* VOLUME is reported, from the CONSTRAINED radius. A free ellipsoid returns
  ~1.5x elongation along the wedge axis because no data constrains it there,
  and volume goes as the product of all three axes — so free-fit volumes run
  ~50% high. That is the easiest way to publish a wrong number from this
  pipeline, so the ellipsoid is never used for volume (fitshapes.py remains the
  place to LOOK at ellipsoid fits; test_fitpop.py pins the inflation).

* POLARITY is required, not assumed. The prototype hardcoded "membranes are
  dark" in the density gate. On an inverted variant that silently selects the
  opposite of what it should: every real virion fails and phantoms in bulk ice
  pass. Pass --polarity from ml_variant_polarity.py (which measures it); the
  density gate REFUSES to run on 'unknown' rather than guess a sign.

* SIZES are physical. '--min-size 520nm3' or '1000@12.56' (voxels at a stated
  pixel size), resolved per tomogram through its own pixel size — a bare voxel
  count means an 8x different object between bin4 and bin8.

* RECALL is measured: the fraction of segmented voxels that ended up inside an
  accepted virion. The gates measure precision only, and it is trivial to score
  perfectly by accepting three virions and discarding everything else.

* POPULATION STATISTICS can be stored and reused, so the radius comes from many
  tomograms rather than the 3 virions in 1 tomogram the prototype learned from.

Needs numpy (+ mrcfile for I/O), so run it from membrainseg — not the GUI venv.
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tomogration_variants import (        # noqa: E402  (path set above)
    BRIGHT, DARK, UNKNOWN, parse_size, nm3_to_voxels)

POP_STORE = "population.json"


# --------------------------------------------------------------------- basics
def load_mrc(path):
    import mrcfile                        # noqa: PLC0415 (optional dependency)
    with mrcfile.open(str(path), permissive=True) as m:
        data = np.asarray(m.data)
        try:
            vx = float(m.voxel_size.x)
        except Exception:
            vx = 1.0
    return data, (vx if vx > 0 else 1.0)


def fib(n):
    """n roughly-uniform unit vectors (Fibonacci sphere)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.sin(phi) * np.cos(th),
                     np.sin(phi) * np.sin(th), np.cos(phi)], 1)


def fit_sphere(xyz):
    """Algebraic sphere fit -> (centre, radius)."""
    A = np.hstack([2 * xyz, np.ones((len(xyz), 1))])
    sol, *_ = np.linalg.lstsq(A, (xyz ** 2).sum(1), rcond=None)
    c = sol[:3]
    return c, np.sqrt(max(sol[3] + (c ** 2).sum(), 1e-9))


def fit_centre_fixed_radius(xyz, R, c0, iters=80):
    """Gauss-Newton on the centre alone, radius held at R."""
    c = np.asarray(c0, dtype=float).copy()
    for _ in range(iters):
        d = xyz - c
        n = np.linalg.norm(d, axis=1)
        ok = n > 1e-9
        if ok.sum() < 4:
            break
        J = -(d[ok] / n[ok, None])
        step, *_ = np.linalg.lstsq(J, -(n[ok] - R), rcond=None)
        c = c + step
        if np.linalg.norm(step) < 1e-4:
            break
    return c, np.abs(np.linalg.norm(xyz - c, axis=1) - R)


def coverage(u, n_probe=800, tol_deg=12.0):
    """(fraction of directions observed, mean |axis| of the uncovered ones)."""
    p = fib(n_probe)
    ct = np.cos(np.deg2rad(tol_deg))
    us = u[::max(1, len(u) // 15000)]
    cov = np.zeros(n_probe, bool)
    for i in range(0, n_probe, 200):
        cov[i:i + 200] = (p[i:i + 200] @ us.T).max(1) >= ct
    gap = np.abs(p[~cov]).mean(0) if (~cov).any() else np.full(3, np.nan)
    return cov.mean(), gap


def angular_extent(u):
    """Half-angle (deg) of the smallest cone containing the observed directions.
    A closed shell gives ~90; a small patch gives a small number."""
    mean = u.mean(0)
    n = np.linalg.norm(mean)
    if n < 1e-6:
        return 90.0
    cosines = u @ (mean / n)
    # 5th percentile, so a few stray voxels cannot inflate it
    return float(np.rad2deg(np.arccos(np.clip(np.percentile(cosines, 5), -1, 1))))


def radius_away_from_gap(r, u, gap):
    """Median radius measured only in directions well away from the gap axis.

    With the polar caps missing there is no data constraining the radius along
    the wedge axis, so including those directions biases it. This is the same
    reasoning that rules the ellipsoid out for volume."""
    if np.isfinite(gap).all() and np.linalg.norm(gap) > 1e-6:
        g = gap / np.linalg.norm(gap)
        band = np.abs(u @ g) < 0.5
    else:
        band = np.ones(len(u), bool)
    if band.sum() < 50:
        band = np.ones(len(u), bool)
    return float(np.median(r[band])), band


# ---------------------------------------------------- gate: density support
def density_support(tomo, seg_any, centre, R, polarity, n=1500, band=1.5):
    """(supported fraction, in-volume fraction) for the predicted surface.

    A point is supported if the segmentation has membrane within `band` voxels,
    OR the density there is membrane-like. Which tercile "membrane-like" means
    is the whole point: DARK -> lowest, BRIGHT -> highest. Guessing it inverts
    the gate, so UNKNOWN refuses (returns nan) instead of picking one."""
    if polarity not in (DARK, BRIGHT):
        return float("nan"), float("nan")
    dirs = fib(n)
    pts = centre + dirs * R
    shape = np.array(tomo.shape)
    inside = np.all((pts >= band + 1) & (pts < shape - band - 1), axis=1)
    if inside.sum() < 50:
        return float("nan"), float(inside.mean())
    p = np.round(pts[inside]).astype(int)

    off = np.array([[dz, dy, dx]
                    for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    seg_hit = np.zeros(len(p), bool)
    for o in off:
        q = p + o
        seg_hit |= seg_any[q[:, 0], q[:, 1], q[:, 2]]

    vals = tomo[p[:, 0], p[:, 1], p[:, 2]]
    sub = tomo[::4, ::4, ::4]
    if polarity == DARK:
        dens_hit = vals <= np.percentile(sub, 33)
    else:
        dens_hit = vals >= np.percentile(sub, 67)
    return float((seg_hit | dens_hit).mean()), float(inside.mean())


# ------------------------------------------------------- volume, and recall
def virion_volume_nm3(radius_px, angpix):
    """Volume of a virion of CONSTRAINED radius, in nm³.

    Deliberately a sphere. A free ellipsoid fitted to a capped shell elongates
    along the unmeasured axis by ~1.5x, and volume is the product of the three
    axes, so its volume runs ~50% high. See test_fitpop.py, which measures that
    inflation on synthetic data so nobody has to rediscover it."""
    r_nm = float(radius_px) * float(angpix) / 10.0
    return (4.0 / 3.0) * math.pi * r_nm ** 3


def ellipsoid_axes(xyz_c):
    """Semi-axes (sorted) of a free ellipsoid through centred points, or None.

    Recorded ALONGSIDE the sphere so the inflation can be seen on real data
    instead of taken on trust — it is never the reported volume. A capped shell
    smeared along z by the missing wedge makes this elongate, and volume is the
    product of all three axes, so it runs ~50% high (test_fitpop.py measures
    1.50x against 1.07x for the constrained radius)."""
    x, y, z = xyz_c.T
    D = np.stack([x * x, y * y, z * z, 2 * x * y, 2 * x * z, 2 * y * z], axis=1)
    try:
        sol, *_ = np.linalg.lstsq(D, np.ones(len(xyz_c)), rcond=None)
    except np.linalg.LinAlgError:
        return None
    A = np.array([[sol[0], sol[3], sol[4]],
                  [sol[3], sol[1], sol[5]],
                  [sol[4], sol[5], sol[2]]])
    w = np.linalg.eigvalsh(A)
    if np.any(w <= 0):
        return None                  # not a closed surface
    return np.sort(1.0 / np.sqrt(w))


def ellipsoid_volume_nm3(axes, angpix):
    """4/3 pi abc in nm³ — the number NOT to publish, kept for comparison."""
    if axes is None or not np.isfinite(axes).all():
        return None
    a = np.asarray(axes, float) * float(angpix) / 10.0
    return float((4.0 / 3.0) * math.pi * a[0] * a[1] * a[2])


def recall(labels, accepted_labels):
    """Fraction of SEGMENTED voxels that ended up inside an accepted virion.

    The eight gates measure precision only: accepting three obvious virions and
    discarding everything else scores perfectly on every one of them. Recall is
    the number that stops that, and it is why the exploration card ranks on
    both together."""
    total = int((labels > 0).sum())
    if not total:
        return float("nan")
    if not accepted_labels:
        return 0.0
    keep = np.isin(labels, np.asarray(sorted(accepted_labels)))
    return float(int(keep.sum()) / total)


# ------------------------------------------------------ population statistics
def load_population(path):
    try:
        d = json.loads(Path(path).read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_population(path, radius_A, whole_nm3, n_virions, tomograms):
    """Store what the population LOOKS like, in physical units.

    The prototype learned this from 3 virions in 1 tomogram and used it
    immediately. Stored per dataset, the production cards can consume a radius
    measured across many tomograms instead of re-learning a shaky one from
    whatever is in front of them."""
    prev = load_population(path)
    seen = sorted(set(prev.get("tomograms", [])) | set(tomograms))
    # MERGE, never replace: population.json is also written by the richer
    # population-stats card (spread, per-tomogram breakdown, the shell voxel
    # count the rescue gate needs) — a write-back from the fitter must not
    # destroy those measurements.
    out = dict(prev)
    out.update({"radius_A": float(radius_A),
                "whole_virion_nm3": float(whole_nm3),
                "n_virions": int(n_virions) + int(prev.get("n_virions", 0)),
                "tomograms": seen})
    Path(path).write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------ rendering
def render_shell(shape, centre, R, n=6000, thickness=1.5):
    vol = np.zeros(shape, np.uint8)
    dirs = fib(n)
    for o in np.linspace(-thickness, thickness, 5):
        p = np.round(centre + dirs * (R + o)).astype(int)
        ok = np.all((p >= 0) & (p < np.array(shape)), axis=1)
        p = p[ok]
        vol[p[:, 0], p[:, 1], p[:, 2]] = 1
    return vol


# ----------------------------------------------------------------------- main
def pair_tomogram(name, tomo_dir):
    """The tomogram a components volume came from, by name.

    The LONGEST tomogram basename that prefixes the components name (followed
    by '_', '.' or end) — the same rule the mesh wrapper uses, deliberately:
    a first-underscore stem truncates PACEtomo-style names (Position_1 ->
    'Position') and the first glob hit can silently pair a volume with the
    WRONG tomogram at a different pixel size. Both mistakes are quiet."""
    best, best_len = None, -1
    for t in sorted(Path(tomo_dir).glob("*.mrc")):
        tb = t.stem
        if (name == tb or name.startswith(tb + "_") or
                name.startswith(tb + ".")) and len(tb) > best_len:
            best, best_len = t, len(tb)
    return best


def volumes_in(path):
    """One volume, or every volume in a folder — sorted naturally."""
    p = Path(path)
    if p.is_dir():
        def key(f):
            return [int(t) if t.isdigit() else t.lower()
                    for t in re.split(r"(\d+)", f.name)]
        return sorted(p.glob("*.mrc"), key=key)
    return [p]


def fit_one(a, comp_path, tomo_path, out_dir):
    """Fit every virion in ONE components volume. Returns its summary."""
    os.makedirs(out_dir, exist_ok=True)
    d, vx = load_mrc(comp_path)
    tomo, _ = load_mrc(tomo_path)
    if tomo.shape != d.shape:
        sys.exit(f"shape mismatch: components {d.shape} vs tomogram {tomo.shape}")
    seg_any = d > 0
    shape = np.array(d.shape)

    min_voxels = nm3_to_voxels(parse_size(a.min_size)[1], vx)
    print(f"volume {d.shape}   {vx:.2f} A/px   polarity {a.polarity}")
    print(f"min size {a.min_size} = {min_voxels} voxels at this pixel size")
    if a.polarity == UNKNOWN:
        print("WARNING: polarity unknown — the density-support gate is DISABLED "
              "for this run. Measure it with ml_variant_polarity.py.")

    # Count every label ONCE and drop the sub-threshold ones before any dense
    # scan: argwhere rescans the whole 386MB volume per label, and most labels
    # in a raw components volume are specks below min_voxels.
    _ids, _counts = np.unique(d[d > 0], return_counts=True)
    labels = [int(l) for l, c in zip(_ids, _counts) if c >= min_voxels]
    print(f"{len(_ids)} labels ({len(labels)} above min size)\n")

    # ---------------------------------------------------------------- pass 1
    info, whole = {}, []
    for l in labels:
        idx = np.argwhere(d == l).astype(float)
        if len(idx) < min_voxels:
            continue
        edge = bool((idx.min(0) <= 2).any() or (idx.max(0) >= shape - 3).any())
        c0, _R0 = fit_sphere(idx)
        v = idx - c0
        r = np.linalg.norm(v, axis=1)
        u = v / np.maximum(r[:, None], 1e-9)
        cov, gap = coverage(u)
        ext = angular_extent(u)
        R_band, band = radius_away_from_gap(r, u, gap)
        diam = 2 * R_band * vx / 10.0
        resid = float(np.median(np.abs(r[band] - R_band))) * vx
        good = (not edge and cov * 100 >= a.whole_coverage
                and a.diam_range[0] <= diam <= a.diam_range[1]
                and np.isfinite(resid) and resid <= a.max_self_resid)
        info[l] = dict(n=len(idx), c0=c0, R=R_band, cov=cov * 100, ext=ext,
                       diam=diam, resid=resid, edge=edge, whole=good)
        if good:
            whole.append(l)
    print(f"pass 1: {len(whole)} components accepted as whole virions")

    pop_path = a.population or os.path.join(out_dir, POP_STORE)
    stored = load_population(pop_path) if a.population else {}
    if a.radius is not None:
        Rpop = a.radius / vx
        src = "supplied"
    elif stored.get("radius_A"):
        Rpop = float(stored["radius_A"]) / vx
        src = (f"stored ({stored.get('n_virions', '?')} virions across "
               f"{len(stored.get('tomograms', []))} tomogram(s))")
    elif len(whole) >= 2:
        Rpop = float(np.median([info[l]["R"] for l in whole]))
        src = f"this tomogram ({len(whole)} whole virions)"
    else:
        # An extreme threshold legitimately leaves too little to fit. That is a
        # real answer about this operating point — 0 virions — so record it and
        # exit 0, rather than failing and leaving a dash the sweep cannot rank.
        print(f"only {len(whole)} whole virion(s) found and no stored "
              f"population radius — nothing can be fitted at this operating "
              f"point. Recording a zero-virion result.")
        summary = {"components": os.path.abspath(comp_path),
                   "tomogram": os.path.abspath(tomo_path),
                   "polarity": a.polarity, "angpix": vx,
                   "min_size": a.min_size, "min_size_voxels": min_voxels,
                   "population_radius_A": None,
                   "population_source": "none — too few whole virions",
                   "accepted": 0, "recall": 0.0, "virions": [],
                   "note": "too few whole virions to learn a radius"}
        with open(os.path.join(out_dir, "fits.json"), "w") as f:
            json.dump(summary, f, indent=1)
        # The SUMMARY, not 0. A zero-virion tomogram is a real result and has
        # to appear in the batch report — returning 0 dropped it from the
        # per-tomogram list without counting it as failed either, so a run
        # where most tomograms found nothing looked like a run of only the
        # few that succeeded.
        return summary
    # The rescue gate's reference is a SHELL voxel count. When this tomogram
    # has whole virions, measure it here; else use the stored population's
    # measured shell count rescaled to this pixel size. NEVER derive it from
    # whole_virion_nm3 — that is the SOLID sphere volume, and solid-nm³
    # converted to voxels is several-fold larger than a shell's count, a unit
    # mix that let partial arcs and debris through the gate. With no measured
    # count at all the gate stands down (inf) rather than compare wrong units.
    if whole:
        whole_n = float(np.median([info[l]["n"] for l in whole]))
    elif stored.get("whole_virion_voxels"):
        s_vx = float(stored.get("angpix") or vx)
        whole_n = float(stored["whole_virion_voxels"]) * (s_vx / vx) ** 3
    else:
        whole_n = float("inf")
    print(f"population radius {Rpop*vx:.0f} A = {2*Rpop*vx/10:.0f} nm diameter"
          f"   [{src}]")

    # ---------------------------------------------------------------- pass 2
    cand = []
    for l, it in info.items():
        why = []
        if it["whole"]:
            centre, R, mode = it["c0"], it["R"], "free"
        elif it["cov"] >= a.closed_coverage:
            centre, R, mode = it["c0"], it["R"], "self"
            if not (a.diam_range[0] <= it["diam"] <= a.diam_range[1]):
                why.append(f"size {it['diam']:.0f}nm")
            elif it["resid"] > a.max_self_resid:
                why.append(f"resid {it['resid']:.0f}A (merged?)")
        else:
            if it["ext"] < a.min_extent:
                why.append(f"extent {it['ext']:.0f}deg")
                cand.append([l, it, None, None, "prior", why, np.nan, np.nan])
                continue
            if np.isfinite(whole_n) and it["n"] > a.max_voxel_ratio * whole_n:
                why.append(f"{it['n']/whole_n:.1f}x a whole virion (merged?)")
                cand.append([l, it, None, None, "prior", why, np.nan, np.nan])
                continue
            idx = np.argwhere(d == l).astype(float)
            centre, res = fit_centre_fixed_radius(idx, Rpop, it["c0"])
            R, mode = Rpop, "prior"
            rms = float(np.sqrt((res ** 2).mean())) * vx
            it["resid"] = rms
            if rms > a.max_prior_resid:
                why.append(f"resid {rms:.0f}A off the prior sphere")

        sup, ins = density_support(tomo, seg_any, centre, R, a.polarity)
        if a.polarity != UNKNOWN:
            if not np.isfinite(sup) or ins < a.min_inside:
                why.append(f"only {ins*100:.0f}% in volume")
            elif sup < a.min_support:
                why.append(f"support {sup*100:.0f}%")
        cand.append([l, it, centre, R, mode, why, sup, ins])

    # ------------------------------------------------ gate: mutual exclusion
    passing = [c for c in cand if not c[5]]
    passing.sort(key=lambda c: -(c[6] if np.isfinite(c[6]) else 0))
    accepted = []
    for c in passing:
        centre, R = c[2], c[3]
        if any(np.linalg.norm(centre - x[2]) < 0.6 * (R + x[3]) for x in accepted):
            c[5].append("overlaps")
        else:
            accepted.append(c)

    acc_ids = {c[0] for c in accepted}
    rec = recall(d, acc_ids)
    print(f"\n{'lab':>4} {'voxels':>8} {'mode':>6} {'diam nm':>8} "
          f"{'vol nm3':>10} {'suppt%':>7}  result")
    for c in sorted(cand, key=lambda c: -(c[6] if np.isfinite(c[6]) else -1)):
        l, it, centre, R, mode, why, sup, ins = c
        rad = R if R is not None else it["R"]
        s = f"{sup*100:6.0f}" if np.isfinite(sup) else "     —"
        res = "KEEP" if l in acc_ids else "reject: " + ", ".join(why)
        print(f"{l:4d} {it['n']:8d} {mode:>6} {2*rad*vx/10:8.0f} "
              f"{virion_volume_nm3(rad, vx):10.0f} {s}  {res}")

    print(f"\naccepted {len(accepted)} virions "
          f"({sum(1 for c in accepted if c[4]=='free')} whole, "
          f"{sum(1 for c in accepted if c[4]=='self')} self-measured, "
          f"{sum(1 for c in accepted if c[4]=='prior')} rescued)")
    print(f"recall {rec*100:.1f}% of segmented voxels are inside an accepted "
          f"virion")

    # ---------------------------------------------------------------- output
    rows, npz = [], {}
    for l, it, centre, R, mode, why, sup, ins in accepted:
        vol_nm3 = virion_volume_nm3(R, vx)
        # Same component, fitted freely, for the sphere-vs-ellipsoid comparison
        # in the analysis window. Never substituted for vol_nm3.
        idx_l = np.argwhere(d == l).astype(float)
        ax = ellipsoid_axes(idx_l - idx_l.mean(0)) if len(idx_l) >= 10 else None
        ell_vol = ellipsoid_volume_nm3(ax, vx)
        elong = (float(ax[2] / ax[0]) if ax is not None and ax[0] > 0 else None)
        rows.append({"label": l, "centre_px": [float(x) for x in centre],
                     "radius_A": float(R * vx), "diameter_nm": float(2 * R * vx / 10),
                     "volume_nm3": vol_nm3, "support": float(sup),
                     # The residual is the fit-quality number the sweep ranks
                     # on; without it in here the table's 'resid A' column can
                     # only ever be a dash.
                     "ellipsoid_volume_nm3": ell_vol,
                     "ellipsoid_elongation": elong,
                     "residual_A": (float(it["resid"])
                                    if it.get("resid") is not None
                                    and np.isfinite(it["resid"]) else None),
                     "mode": mode, "voxels": int(it["n"])})
        npz[f"{l}_centre"] = centre
        npz[f"{l}_radius"] = R
        npz[f"{l}_support"] = sup
        npz[f"{l}_volume_nm3"] = vol_nm3
        if a.write_masks:
            import mrcfile                # noqa: PLC0415
            with mrcfile.new(os.path.join(out_dir, f"fit_{l}.mrc"),
                             overwrite=True) as m:
                m.set_data(render_shell(d.shape, centre, R))
                m.voxel_size = vx
    np.savez(os.path.join(out_dir, "surfaces.npz"), voxel_size=vx, **npz)

    # Provenance on every artefact (spec Part D): two runs have already made
    # identically-named files in different folders and been confused.
    summary = {
        "components": os.path.abspath(comp_path),
        "tomogram": os.path.abspath(tomo_path),
        "polarity": a.polarity,
        "angpix": vx,
        "min_size": a.min_size,
        "min_size_voxels": min_voxels,
        "population_radius_A": float(Rpop * vx),
        "population_source": src,
        "gates": {"min_extent": a.min_extent, "whole_coverage": a.whole_coverage,
                  "closed_coverage": a.closed_coverage,
                  "min_support": a.min_support,
                  "max_prior_resid": a.max_prior_resid,
                  "max_voxel_ratio": a.max_voxel_ratio,
                  "max_self_resid": a.max_self_resid,
                  "min_inside": a.min_inside,
                  "diam_range": a.diam_range},
        "accepted": len(accepted),
        "recall": rec,
        "virions": rows,
    }
    with open(os.path.join(out_dir, "fits.json"), "w") as f:
        json.dump(summary, f, indent=1)
    if whole and a.population:
        save_population(pop_path,
                        radius_A=Rpop * vx,
                        whole_nm3=virion_volume_nm3(Rpop, vx),
                        n_virions=len(whole),
                        tomograms=[Path(tomo_path).stem])
    print(f"wrote {out_dir}/fits.json and surfaces.npz")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--components", required=True)
    ap.add_argument("--tomogram", required=True,
                    help="the tomogram the density-support gate reads")
    ap.add_argument("--polarity", required=True, choices=[DARK, BRIGHT, UNKNOWN],
                    help="membrane polarity of THIS variant, as measured by "
                         "ml_variant_polarity.py. Never guess it: the wrong "
                         "sign rejects every real virion and accepts ice.")
    ap.add_argument("--out-folder", default="popfits")
    ap.add_argument("--min-size", default="1000@12.56",
                    help="smallest component to consider, PHYSICAL: '520nm3' "
                         "or '1000@12.56' (voxels at a pixel size)")
    ap.add_argument("--min-extent", type=float, default=30.0,
                    help="gate 1: minimum half-angle subtended, degrees")
    ap.add_argument("--whole-coverage", type=float, default=58.0,
                    help="pass 1: coverage to call a component a whole virion "
                         "(both caps missing tops out near 66%%)")
    ap.add_argument("--closed-coverage", type=float, default=85.0)
    ap.add_argument("--min-support", type=float, default=0.55)
    ap.add_argument("--max-prior-resid", type=float, default=50.0)
    ap.add_argument("--max-voxel-ratio", type=float, default=1.15,
                    help="a RATIO of the whole-virion median, so it needs no "
                         "pixel size — a partial arc cannot hold more voxels "
                         "than a complete virion")
    ap.add_argument("--max-self-resid", type=float, default=45.0)
    ap.add_argument("--min-inside", type=float, default=0.6)
    ap.add_argument("--radius", type=float, default=None,
                    help="skip pass 1 and use this radius (Angstroms)")
    ap.add_argument("--population", default="",
                    help=f"population.json to READ the radius from (and write "
                         f"back to). Default name: {POP_STORE}")
    ap.add_argument("--diam-range", type=float, nargs=2, default=[60.0, 160.0])
    ap.add_argument("--write-masks", action="store_true")
    a = ap.parse_args(argv)
    a = ap.parse_args(argv)

    # A FOLDER for --components means fit every volume in it. The card's
    # build-downstream fills in the components job's output DIRECTORY, which
    # is what you want (72 tomograms, not one) — the single-file form still
    # works and is unchanged.
    comps = volumes_in(a.components)
    if not comps:
        sys.exit(f"no .mrc volumes in {a.components}")
    batch = len(comps) > 1 or Path(a.components).is_dir()
    tomo_dir = Path(a.tomogram)

    pairs, unpaired = [], []
    for c in comps:
        if tomo_dir.is_dir():
            t = pair_tomogram(c.stem, tomo_dir)
            if t is None:
                unpaired.append(c.name)
                continue
        else:
            t = tomo_dir
        pairs.append((c, t))
    for n in unpaired:
        print(f"WARN: no tomogram in {tomo_dir} prefixes {n} — skipped.")
    if not pairs:
        sys.exit(f"no components volume could be paired with a tomogram in "
                 f"{tomo_dir}")

    if not batch:
        c, t = pairs[0]
        fit_one(a, str(c), str(t), a.out_folder)
        print(f"wrote {a.out_folder}/fits.json and surfaces.npz")
        return 0

    # Batch. Each volume gets its OWN subfolder: fits.json and fit_*.mrc have
    # fixed names, so a shared folder would have every tomogram overwriting
    # the last and the run would end holding one tomogram's results while
    # reporting 72.
    print(f"{len(pairs)} components volume(s) to fit\n")
    os.makedirs(a.out_folder, exist_ok=True)
    every, failed = [], []
    for i, (c, t) in enumerate(pairs, 1):
        print(f"\n{'=' * 67}\n[{i}/{len(pairs)}] {c.name}  (tomo: {t.name})\n"
              f"{'=' * 67}")
        sub = os.path.join(a.out_folder, c.stem)
        try:
            s = fit_one(a, str(c), str(t), sub)
        except SystemExit as e:            # a shape mismatch on ONE volume
            print(f"ERROR: {e}")
            failed.append(c.name)
            continue
        if s:
            s["stem"] = c.stem
            every.append(s)

    # A combined fits.json so the card status, the viewer and any later
    # analysis see the whole dataset without walking 72 subfolders.
    total = sum(int(s.get("accepted") or 0) for s in every)
    rec = [s.get("recall") for s in every if s.get("recall") is not None]
    combined = {
        "components": os.path.abspath(a.components),
        "tomogram": os.path.abspath(a.tomogram),
        "polarity": a.polarity,
        "batch": True,
        "n_tomograms": len(every),
        "accepted": total,
        "recall": float(np.median(rec)) if rec else None,
        "unpaired": unpaired,
        "failed": failed,
        "per_tomogram": [{"stem": s["stem"], "accepted": s.get("accepted"),
                          "recall": s.get("recall"),
                          "angpix": s.get("angpix")} for s in every],
        "virions": [dict(v, stem=s["stem"]) for s in every
                    for v in (s.get("virions") or [])],
    }
    with open(os.path.join(a.out_folder, "fits.json"), "w") as f:
        json.dump(combined, f, indent=1)
    print(f"\n{'=' * 67}")
    print(f"done: {len(every)} fitted, {len(failed)} failed, "
          f"{len(unpaired)} unpaired — {total} virions accepted in total")
    if rec:
        print(f"median recall {np.median(rec) * 100:.1f}%")
    print(f"wrote {a.out_folder}/fits.json (combined) + one folder per tomogram")
    return 1 if (failed and not every) else 0



if __name__ == "__main__":
    sys.exit(main())
