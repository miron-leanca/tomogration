#!/usr/bin/env python
"""
fitpop2 - two-pass shape fitting with acceptance gates that actually reject.

Why this differs from the first attempt
---------------------------------------
Fixing the radius makes the residual test useless: any small fragment lies
almost perfectly on a large sphere, so low residual proves nothing. Four
independent gates are needed instead.

  1  ANGULAR EXTENT   a fragment must subtend enough of the surface for its
                      curvature to be measurable at all (default 30 deg)
  2  SELF-CONSISTENCY a component that is already closed (high coverage) has
                      measured itself; the prior must not override it
  3  DENSITY SUPPORT  the predicted surface must coincide with membrane-like
                      density in the tomogram, not run through bulk ice
  4  EXCLUSION        virions cannot interpenetrate; overlapping placements
                      are resolved in favour of the better-supported one

The prior is a SPHERE, not an ellipsoid. With the polar caps missing there is
no data constraining the radius along the wedge axis, so a free ellipsoid fit
returns a spurious ~1.5x elongation along it. The radius is therefore measured
only from directions well away from the gap axis.

Usage
-----
    fitpop2.py --components comps.mrc --tomogram tomo_deconv.mrc \\
               --out-folder popfits --write-masks

    # inspect gate decisions without writing anything
    fitpop2.py --components comps.mrc --tomogram tomo.mrc --explain
"""

import argparse
import os
import sys

import numpy as np

try:
    import mrcfile
except ImportError:
    sys.exit("mrcfile not installed")


# --------------------------------------------------------------------- basics

def load_mrc(path):
    with mrcfile.open(path, permissive=True) as m:
        data = np.asarray(m.data)
        try:
            vx = float(m.voxel_size.x)
        except Exception:
            vx = 1.0
    return data, (vx if vx > 0 else 1.0)


def fib(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.sin(phi) * np.cos(th),
                     np.sin(phi) * np.sin(th), np.cos(phi)], 1)


def fit_sphere(xyz):
    A = np.hstack([2 * xyz, np.ones((len(xyz), 1))])
    sol, *_ = np.linalg.lstsq(A, (xyz ** 2).sum(1), rcond=None)
    c = sol[:3]
    return c, np.sqrt(max(sol[3] + (c ** 2).sum(), 1e-9))


def coverage(u, n_probe=800, tol_deg=12.0):
    p = fib(n_probe)
    ct = np.cos(np.deg2rad(tol_deg))
    us = u[::max(1, len(u) // 15000)]
    cov = np.zeros(n_probe, bool)
    for i in range(0, n_probe, 200):
        cov[i:i + 200] = (p[i:i + 200] @ us.T).max(1) >= ct
    gap = np.abs(p[~cov]).mean(0) if (~cov).any() else np.full(3, np.nan)
    return cov.mean(), gap


# --------------------------------------------------------- gate 1: extent

def angular_extent(u):
    """
    Half-angle (degrees) of the smallest cone containing the observed
    directions. A closed shell gives ~90; a small patch gives a small number.
    """
    mean = u.mean(0)
    n = np.linalg.norm(mean)
    if n < 1e-6:
        return 90.0
    axis = mean / n
    cosines = u @ axis
    # 5th percentile, so a few stray voxels do not inflate it
    return float(np.rad2deg(np.arccos(np.clip(np.percentile(cosines, 5), -1, 1))))


# --------------------------------------------------- pass 2: centre-only fit

def fit_centre_fixed_radius(xyz, R, c0, iters=80):
    """Gauss-Newton on the centre of a sphere of known radius R."""
    c = c0.astype(float).copy()
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
    resid = np.abs(np.linalg.norm(xyz - c, axis=1) - R)
    return c, resid


# ------------------------------------------------- gate 3: density support

def density_support(tomo, seg_any, centre, R, n=1500, band=1.5):
    """
    Fraction of the predicted surface that is supported.

    A point counts as supported if the segmentation has membrane within `band`
    voxels of it, OR the tomogram density there is membrane-like (dark, in the
    lowest tercile of the local distribution). Points falling outside the
    volume are excluded rather than counted either way.
    """
    dirs = fib(n)
    pts = centre + dirs * R
    shape = np.array(tomo.shape)
    inside = np.all((pts >= band + 1) & (pts < shape - band - 1), axis=1)
    if inside.sum() < 50:
        return np.nan, float(inside.mean())
    p = np.round(pts[inside]).astype(int)

    # segmentation support, dilated by the band
    off = np.array([[dz, dy, dx]
                    for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    seg_hit = np.zeros(len(p), bool)
    for o in off:
        q = p + o
        seg_hit |= seg_any[q[:, 0], q[:, 1], q[:, 2]]

    # density support: membranes are dark in these tomograms
    vals = tomo[p[:, 0], p[:, 1], p[:, 2]]
    lo = np.percentile(tomo[::4, ::4, ::4], 33)
    dens_hit = vals <= lo

    return float((seg_hit | dens_hit).mean()), float(inside.mean())


# ------------------------------------------------------------------ rendering

def render_shell(shape, centre, R, n=6000, thickness=1.5):
    dirs = fib(n)
    vol = np.zeros(shape, np.uint8)
    for o in np.linspace(-thickness, thickness, 5):
        p = np.round(centre + dirs * (R + o)).astype(int)
        ok = np.all((p >= 0) & (p < np.array(shape)), axis=1)
        p = p[ok]
        vol[p[:, 0], p[:, 1], p[:, 2]] = 1
    return vol


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--components", required=True)
    ap.add_argument("--tomogram", required=True,
                    help="deconvolved tomogram, used for the density-support gate")
    ap.add_argument("--out-folder", default="popfits")
    ap.add_argument("--min-voxels", type=int, default=2000)
    ap.add_argument("--min-extent", type=float, default=30.0,
                    help="gate 1: minimum half-angle subtended, degrees")
    ap.add_argument("--whole-coverage", type=float, default=58.0,
                    help="pass 1: coverage needed to call a component a whole virion.\n"
                         "A virion with both polar caps missing tops out near 66%%.")
    ap.add_argument("--closed-coverage", type=float, default=85.0,
                    help="gate 2: above this %% the object is genuinely closed, so\n"
                         "trust its own measured size rather than the prior")
    ap.add_argument("--min-support", type=float, default=0.55,
                    help="gate 3: minimum supported fraction of the predicted surface")
    ap.add_argument("--max-prior-resid", type=float, default=50.0,
                    help="gate 1b: reject a fragment whose voxels lie this far off\n"
                         "the prior sphere. Low residual proves nothing here, but\n"
                         "high residual still disproves.")
    ap.add_argument("--max-voxel-ratio", type=float, default=1.15,
                    help="gate 1c: a partial arc cannot hold more voxels than a\n"
                         "complete virion. Multiple of the whole-virion median.")
    ap.add_argument("--max-self-resid", type=float, default=45.0,
                    help="gate 2: reject a closed object above this RMS, Angstroms;\n"
                         "merged virions fit one sphere badly and show up here")
    ap.add_argument("--min-inside", type=float, default=0.6,
                    help="reject placements mostly outside the volume")
    ap.add_argument("--radius", type=float, default=None,
                    help="skip pass 1 and use this radius (Angstroms)")
    ap.add_argument("--diam-range", type=float, nargs=2, default=[60.0, 160.0])
    ap.add_argument("--write-masks", action="store_true")
    ap.add_argument("--explain", action="store_true",
                    help="show why each component was rejected")
    args = ap.parse_args()

    os.makedirs(args.out_folder, exist_ok=True)
    d, vx = load_mrc(args.components)
    tomo, _ = load_mrc(args.tomogram)
    if tomo.shape != d.shape:
        sys.exit(f"shape mismatch: components {d.shape} vs tomogram {tomo.shape}")
    seg_any = d > 0
    shape = np.array(d.shape)
    print(f"volume {d.shape}   {vx:.2f} A/px")

    labels = [int(l) for l in np.unique(d[d > 0])]
    print(f"{len(labels)} labels, {args.min_voxels}+ voxels required\n")

    # ---------------------------------------------------------------- pass 1
    info, whole = {}, []
    for l in labels:
        idx = np.argwhere(d == l).astype(float)
        if len(idx) < args.min_voxels:
            continue
        edge = bool((idx.min(0) <= 2).any() or (idx.max(0) >= shape - 3).any())
        c0, R0 = fit_sphere(idx)
        v = idx - c0
        r = np.linalg.norm(v, axis=1)
        u = v / np.maximum(r[:, None], 1e-9)
        cov, gap = coverage(u)
        ext = angular_extent(u)

        # radius measured only away from the gap axis, so the missing caps
        # cannot bias it
        if np.isfinite(gap).all() and np.linalg.norm(gap) > 1e-6:
            g = gap / np.linalg.norm(gap)
            band = np.abs(u @ g) < 0.5
        else:                       # fully covered: no gap axis to avoid
            band = np.ones(len(u), bool)
        if band.sum() < 50:
            band = np.ones(len(u), bool)
        R_band = float(np.median(r[band]))

        diam = 2 * R_band * vx / 10.0
        resid = float(np.median(np.abs(r[band] - R_band))) * vx
        good = (not edge and cov * 100 >= args.whole_coverage
                and args.diam_range[0] <= diam <= args.diam_range[1]
                and np.isfinite(resid) and resid <= 45)
        info[l] = dict(n=len(idx), c0=c0, R=R_band, cov=cov * 100, ext=ext,
                       diam=diam, resid=resid, edge=edge, whole=good)
        if good:
            whole.append(l)

    print(f"pass 1: {len(whole)} components accepted as whole virions")
    for l in whole:
        print(f"          label {l:4d}  radius {info[l]['R']*vx:.0f} A "
              f"= {info[l]['diam']:.0f} nm   coverage {info[l]['cov']:.0f}%")

    if args.radius is not None:
        Rpop = args.radius / vx
        whole_n = float(np.median([info[l]["n"] for l in whole])) if whole \
            else float("inf")
        print(f"\nusing supplied radius {args.radius:.0f} A")
    elif len(whole) >= 2:
        Rpop = float(np.median([info[l]["R"] for l in whole]))
        whole_n = float(np.median([info[l]["n"] for l in whole]))
        spread = np.std([info[l]["R"] for l in whole]) * vx
        print(f"\npopulation radius {Rpop*vx:.0f} A "
              f"= {2*Rpop*vx/10:.0f} nm diameter   (sd {spread:.0f} A)")
        print(f"whole-virion size {whole_n:.0f} voxels; fragments above "
              f"{args.max_voxel_ratio*whole_n:.0f} treated as merged")
    else:
        sys.exit("\nfewer than 2 whole virions — supply --radius directly")

    # ---------------------------------------------------------------- pass 2
    cand = []
    for l, it in info.items():
        why = []
        if it["whole"]:
            centre, R, mode = it["c0"], it["R"], "free"
        elif it["cov"] >= args.closed_coverage:
            # gate 2: the object is closed, so it measured its own size.
            # Keep it if that size is virion-like and the fit is tight;
            # a merged pair shows up here as a high residual.
            centre, R, mode = it["c0"], it["R"], "self"
            if not (args.diam_range[0] <= it["diam"] <= args.diam_range[1]):
                why.append(f"size {it['diam']:.0f}nm")
            elif it["resid"] > args.max_self_resid:
                why.append(f"resid {it['resid']:.0f}A (merged?)")
        else:
            if it["ext"] < args.min_extent:
                why.append(f"extent {it['ext']:.0f}deg")
                cand.append((l, it, None, None, "prior", why, np.nan, np.nan))
                continue
            if it["n"] > args.max_voxel_ratio * whole_n:
                why.append(f"{it['n']/whole_n:.1f}x a whole virion (merged?)")
                cand.append((l, it, None, None, "prior", why, np.nan, np.nan))
                continue
            idx = np.argwhere(d == l).astype(float)
            centre, res = fit_centre_fixed_radius(idx, Rpop, it["c0"])
            R, mode = Rpop, "prior"
            rms = float(np.sqrt((res ** 2).mean())) * vx
            it["resid"] = rms
            if rms > args.max_prior_resid:
                why.append(f"resid {rms:.0f}A off the prior sphere")

        sup, ins = density_support(tomo, seg_any, centre, R)
        if not np.isfinite(sup) or ins < args.min_inside:
            why.append(f"only {ins*100:.0f}% in volume")
        elif sup < args.min_support:
            why.append(f"support {sup*100:.0f}%")
        cand.append((l, it, centre, R, mode, why, sup, ins))

    # ------------------------------------------------ gate 4: mutual exclusion
    passing = [c for c in cand if not c[5]]
    passing.sort(key=lambda c: -(c[6] if np.isfinite(c[6]) else 0))
    accepted = []
    for c in passing:
        l, it, centre, R, mode, why, sup, ins = c
        clash = False
        for a in accepted:
            sep = np.linalg.norm(centre - a[2])
            if sep < 0.6 * (R + a[3]):        # centres closer than 60% of summed radii
                clash = True
                break
        if clash:
            c[5].append("overlaps")
        else:
            accepted.append(c)

    acc_ids = {c[0] for c in accepted}
    print(f"\n{'lab':>4} {'voxels':>8} {'mode':>6} {'cover%':>7} {'ext°':>5} "
          f"{'diam nm':>8} {'rms A':>7} {'suppt%':>7}  result")
    for c in sorted(cand, key=lambda c: -(c[6] if np.isfinite(c[6]) else -1)):
        l, it, centre, R, mode, why, sup, ins = c
        dia = 2 * (R if R is not None else it["R"]) * vx / 10
        s = f"{sup*100:6.0f}" if np.isfinite(sup) else "     —"
        res = "KEEP" if l in acc_ids else "reject: " + ", ".join(why)
        print(f"{l:4d} {it['n']:8d} {mode:>6} {it['cov']:7.0f} {it['ext']:5.0f} "
              f"{dia:8.0f} {it['resid']:7.0f} {s}  {res}")

    print(f"\naccepted {len(accepted)} virions "
          f"({sum(1 for c in accepted if c[4]=='free')} whole, "
          f"{sum(1 for c in accepted if c[4]=='self')} self-measured, "
          f"{sum(1 for c in accepted if c[4]=='prior')} rescued)")

    npz = {}
    for l, it, centre, R, mode, why, sup, ins in accepted:
        npz[f"{l}_centre"] = centre
        npz[f"{l}_radius"] = R
        npz[f"{l}_support"] = sup
        if args.write_masks:
            with mrcfile.new(os.path.join(args.out_folder, f"fit_{l}.mrc"),
                             overwrite=True) as m:
                m.set_data(render_shell(d.shape, centre, R))
                m.voxel_size = vx
    np.savez(os.path.join(args.out_folder, "surfaces.npz"), voxel_size=vx, **npz)
    print(f"wrote {args.out_folder}/surfaces.npz")


if __name__ == "__main__":
    main()
