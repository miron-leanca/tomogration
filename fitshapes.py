#!/usr/bin/env python
"""
fitshapes - fit sphere / ellipsoid / spherical-harmonic surfaces to a segmented
membrane component, compare them, and write overlays you can inspect in tomoview.

Usage
-----
    # list candidate labels, sorted by how spherical they look
    fitshapes.py --components comps.mrc --list

    # fit one label with all three models
    fitshapes.py --components comps.mrc --label 62 --out-folder fits

    # fit several
    fitshapes.py --components comps.mrc --label 62 --label 5 --label 3

Outputs, per label, into --out-folder:
    fit_<label>_sphere.mrc      surface rendered as a shell mask, same grid as input
    fit_<label>_ellipsoid.mrc
    fit_<label>_sh<L>.mrc
    fit_<label>_points.npz      sampled positions + outward normals for each model

Inspect with:
    tomoview <tomo>.mrc fits/fit_62_*.mrc

All fits are compared the same way: from a common centre, each model predicts a
radius in every direction, and the residual is the radial deviation of the
observed membrane voxels from that prediction. That makes the numbers directly
comparable across model orders.
"""

import argparse
import os
import sys

import numpy as np

try:
    import mrcfile
except ImportError:
    sys.exit("mrcfile not installed")


# ----------------------------------------------------------------- utilities

def load_components(path):
    with mrcfile.open(path, permissive=True) as m:
        data = np.asarray(m.data)
        try:
            vx = float(m.voxel_size.x)
        except Exception:
            vx = 1.0
    if vx <= 0:
        vx = 1.0
    return data, vx


def to_spherical(xyz):
    """Return radius, and unit direction vectors."""
    r = np.linalg.norm(xyz, axis=1)
    u = xyz / np.maximum(r[:, None], 1e-9)
    return r, u


def fibonacci_directions(n):
    """n roughly-uniform unit vectors on the sphere."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(phi)], axis=1)


# ------------------------------------------------------------------- fitting

def fit_sphere(xyz):
    """Algebraic sphere fit. Returns centre, radius."""
    A = np.hstack([2 * xyz, np.ones((len(xyz), 1))])
    b = (xyz ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    centre = sol[:3]
    radius = np.sqrt(max(sol[3] + (centre ** 2).sum(), 1e-9))
    return centre, radius


def fit_ellipsoid(xyz_c):
    """
    Fit x^T A x = 1 on centred points. Returns the symmetric 3x3 A,
    or None if the solution is not a genuine ellipsoid.
    """
    x, y, z = xyz_c.T
    D = np.stack([x * x, y * y, z * z, 2 * x * y, 2 * x * z, 2 * y * z], axis=1)
    sol, *_ = np.linalg.lstsq(D, np.ones(len(xyz_c)), rcond=None)
    A = np.array([[sol[0], sol[3], sol[4]],
                  [sol[3], sol[1], sol[5]],
                  [sol[4], sol[5], sol[2]]])
    w = np.linalg.eigvalsh(A)
    if np.any(w <= 0):
        return None            # hyperboloid / degenerate ? not a closed surface
    return A


def ellipsoid_radius(A, u):
    """Radius along unit directions u for surface x^T A x = 1."""
    q = np.einsum("ij,jk,ik->i", u, A, u)
    return 1.0 / np.sqrt(np.maximum(q, 1e-12))


def real_sph_harm_basis(lmax, u):
    """
    Real spherical harmonic basis evaluated at unit vectors u.
    Returns (n_points, n_basis) and a list of (l, m) labels.
    """
    try:                                        # scipy >= 1.15
        from scipy.special import sph_harm_y
        def _Y(l, m, az, pol):
            return sph_harm_y(l, m, pol, az)
    except ImportError:                         # scipy < 1.15
        from scipy.special import sph_harm
        def _Y(l, m, az, pol):
            return sph_harm(m, l, az, pol)

    x, y, z = u.T
    theta = np.arctan2(y, x)                    # azimuth  [-pi, pi]
    phi = np.arccos(np.clip(z, -1, 1))          # polar    [0, pi]

    cols, labels = [], []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            Y = _Y(l, abs(m), theta, phi)
            if m < 0:
                v = np.sqrt(2) * (-1) ** m * Y.imag
            elif m == 0:
                v = Y.real
            else:
                v = np.sqrt(2) * (-1) ** m * Y.real
            cols.append(v)
            labels.append((l, m))
    return np.stack(cols, axis=1), labels


def fit_sh(u, r, lmax, ridge=1e-3):
    """Least-squares fit of r(direction) in a real SH basis, lightly ridged."""
    B, labels = real_sph_harm_basis(lmax, u)
    G = B.T @ B + ridge * len(r) * np.eye(B.shape[1])
    coef = np.linalg.solve(G, B.T @ r)
    return coef, labels


def sh_radius(coef, lmax, u):
    B, _ = real_sph_harm_basis(lmax, u)
    return B @ coef


# ------------------------------------------------------------------ coverage

def direction_coverage(u_obs, n_probe=2000, tol_deg=12.0):
    """
    Fraction of directions with at least one observed voxel nearby, plus the
    mean |z| of the uncovered directions (near 1.0 = the gaps are at the poles).
    """
    probe = fibonacci_directions(n_probe)
    cos_tol = np.cos(np.deg2rad(tol_deg))
    # chunked to keep memory sane
    covered = np.zeros(n_probe, dtype=bool)
    step = max(1, len(u_obs) // 20000)
    us = u_obs[::step]
    for i in range(0, n_probe, 200):
        blk = probe[i:i + 200]
        covered[i:i + 200] = (blk @ us.T).max(1) >= cos_tol
    frac = covered.mean()
    if (~covered).any():
        gap = np.abs(probe[~covered]).mean(0)   # per array-axis
    else:
        gap = np.full(3, np.nan)
    return frac, gap


# ------------------------------------------------------------------- writing

def render_shell(shape, centre, dirs, radii, thickness=1.5):
    """Paint a thin shell into an empty volume for visual overlay."""
    vol = np.zeros(shape, dtype=np.uint8)
    pts = centre + dirs * radii[:, None]
    # thicken by jittering along the normal
    offs = np.linspace(-thickness, thickness, 5)
    for o in offs:
        p = np.round(centre + dirs * (radii + o)[:, None]).astype(int)
        ok = np.all((p >= 0) & (p < np.array(shape)), axis=1)
        p = p[ok]
        vol[p[:, 0], p[:, 1], p[:, 2]] = 1
    return vol


def write_mask(path, vol, vx):
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(vol)
        m.voxel_size = vx


# ---------------------------------------------------------------------- main

def summarise_labels(d):
    shape = np.array(d.shape)
    rows = []
    for l in np.unique(d[d > 0]):
        idx = np.argwhere(d == l)
        ext = idx.max(0) - idx.min(0)
        edge = bool((idx.min(0) <= 2).any() or (idx.max(0) >= shape - 3).any())
        rows.append((int(l), len(idx), ext, ext.min() / max(ext.max(), 1), edge))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--components", required=True, help="labelled components .mrc")
    ap.add_argument("--label", type=int, action="append", default=[],
                    help="label to fit; repeat for several")
    ap.add_argument("--out-folder", default="fits")
    ap.add_argument("--lmax", type=int, action="append", default=[],
                    help="spherical harmonic orders to try (default 2 and 4)")
    ap.add_argument("--n-sample", type=int, default=6000,
                    help="points sampled on each fitted surface")
    ap.add_argument("--list", action="store_true",
                    help="just list labels sorted by sphericity and exit")
    ap.add_argument("--screen", action="store_true",
                    help="fit every label and tabulate; use this to find virions")
    ap.add_argument("--min-voxels", type=int, default=1500,
                    help="ignore labels smaller than this when screening")
    args = ap.parse_args()

    lmaxes = args.lmax or [2, 4]
    d, vx = load_components(args.components)
    print(f"volume {d.shape}   {vx:.2f} A/px\n")

    if args.screen:
        shape = np.array(d.shape)
        rows = []
        for l in np.unique(d[d > 0]):
            idx = np.argwhere(d == l).astype(float)
            if len(idx) < args.min_voxels:
                continue
            edge = bool((idx.min(0) <= 2).any() or (idx.max(0) >= shape - 3).any())
            centre, R = fit_sphere(idx)
            xyz_c = idx - centre
            r_obs, u_obs = to_spherical(xyz_c)
            frac, _ = direction_coverage(u_obs, n_probe=800)
            A = fit_ellipsoid(xyz_c)
            if A is None:
                axes = np.full(3, np.nan); rms = np.nan
            else:
                axes = np.sort(1.0 / np.sqrt(np.linalg.eigvalsh(A))) * vx
                rms = np.sqrt(((r_obs - ellipsoid_radius(A, u_obs)) ** 2).mean()) * vx
            elong = axes[2] / axes[0] if np.isfinite(axes).all() else np.nan
            rows.append((int(l), len(idx), 2 * R * vx / 10, frac * 100,
                         axes, elong, rms, edge))

        print(f"{'lab':>4} {'voxels':>8} {'diam nm':>8} {'cover%':>7} "
              f"{'ellipsoid semi-axes A':>26} {'elong':>6} {'rms A':>7}  flags")
        for l, n, dia, cov, axes, el, rms, edge in sorted(rows, key=lambda x: -x[3]):
            ax = "  ".join(f"{a:5.0f}" for a in axes) if np.isfinite(axes).all() else "   ?"
            flags = []
            if edge:
                flags.append("EDGE")
            if cov < 60:
                flags.append("low-cover")
            if np.isfinite(el) and el > 1.8:
                flags.append("elongated")
            if not (60 <= dia <= 160):
                flags.append("size?")
            good = "  <-- virion" if not flags else ""
            print(f"{l:4d} {n:8d} {dia:8.0f} {cov:7.0f} {ax:>26} "
                  f"{el:6.2f} {rms:7.1f}  {','.join(flags)}{good}")
        print("\nA real OC43 virion: 60?160 nm across, >60% coverage,")
        print("elongation under ~1.8, no edge contact.")
        print("Re-run with --label <N> on the marked ones to compare model orders.")
        return

    if args.list or not args.label:
        rows = summarise_labels(d)
        print(f"{'lab':>4} {'voxels':>8} {'extent':>16} {'ratio':>6}  edge")
        for l, n, e, r, edge in sorted(rows, key=lambda x: -x[3])[:25]:
            print(f"{l:4d} {n:8d} {str(e):>16} {r:6.2f}  {'EDGE' if edge else ''}")
        if not args.label:
            print("\nre-run with --label <N> to fit one of these")
        return

    os.makedirs(args.out_folder, exist_ok=True)
    dirs = fibonacci_directions(args.n_sample)

    for label in args.label:
        idx = np.argwhere(d == label).astype(float)
        if len(idx) < 200:
            print(f"label {label}: only {len(idx)} voxels, skipping")
            continue

        print(f"=== label {label}   {len(idx)} voxels ===")
        centre, R = fit_sphere(idx)
        xyz_c = idx - centre
        r_obs, u_obs = to_spherical(xyz_c)

        frac, gap = direction_coverage(u_obs)
        print(f"  centre {np.round(centre,1)}   sphere radius {R:.1f} px "
              f"= {R*vx:.0f} A   diameter {2*R*vx/10:.0f} nm")
        print(f"  surface coverage {frac*100:.0f}%")
        print(f"  gap direction (per array axis) {np.round(gap,2)}"
              "   a value near 1 means the gaps face that axis")

        models = {}

        # sphere
        models["sphere"] = (np.full(len(dirs), R), np.abs(r_obs - R))

        # ellipsoid
        A = fit_ellipsoid(xyz_c)
        if A is None:
            print("  ellipsoid: fit is not a closed surface, skipped")
        else:
            axes = 1.0 / np.sqrt(np.linalg.eigvalsh(A))
            models["ellipsoid"] = (ellipsoid_radius(A, dirs),
                                   np.abs(r_obs - ellipsoid_radius(A, u_obs)))
            print(f"  ellipsoid semi-axes {np.round(np.sort(axes)*vx,0)} A")

        # spherical harmonics
        for L in lmaxes:
            try:
                coef, _ = fit_sh(u_obs, r_obs, L)
                models[f"sh{L}"] = (sh_radius(coef, L, dirs),
                                    np.abs(r_obs - sh_radius(coef, L, u_obs)))
            except Exception as e:
                print(f"  sh{L}: failed ({e})")

        print(f"\n  {'model':<12}{'RMS resid':>12}{'max resid':>12}"
              f"{'r range (A)':>18}")
        for name, (r_pred, resid) in models.items():
            print(f"  {name:<12}{np.sqrt((resid**2).mean())*vx:9.1f} A"
                  f"{resid.max()*vx:9.1f} A"
                  f"{r_pred.min()*vx:9.0f} ?{r_pred.max()*vx:6.0f}")

        print("\n  lower RMS means a better fit to the measured band,")
        print("  but check the r range: a model whose radius swings wildly is")
        print("  extrapolating badly into the unmeasured caps.\n")

        out = {}
        for name, (r_pred, _) in models.items():
            vol = render_shell(d.shape, centre, dirs, r_pred)
            p = os.path.join(args.out_folder, f"fit_{label}_{name}.mrc")
            write_mask(p, vol, vx)
            out[f"{name}_points"] = centre + dirs * r_pred[:, None]
            out[f"{name}_normals"] = dirs
            print(f"  wrote {p}")

        np.savez(os.path.join(args.out_folder, f"fit_{label}_points.npz"),
                 centre=centre, voxel_size=vx, **out)
        print()


if __name__ == "__main__":
    main()
