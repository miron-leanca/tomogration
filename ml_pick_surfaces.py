#!/usr/bin/env python3
"""Oversampled surface picking (cards spec draft 2, A4 — Oversample mode).

    ml_pick_surfaces.py --fits popfits/fits.json --tomostar Position003 \\
                        --out picks/Position003_oversample.star

Samples the surfaces A3 accepted, at ~50 A spacing, plus shells at +/-50 A.
Maximises recall and lets RELION classification do the discrimination; the
output is flagged for duplicate removal after the first refinement, because
the three shells converge on the same particles by construction.

WHAT EACH SITE CARRIES, and why
-------------------------------
position          voxel coordinates on the sampled shell
outward normal    the surface normal at that site
rot / tilt / psi  RELION Euler angles putting the reference Z along the normal
theta             THE ANGLE BETWEEN THE NORMAL AND THE WEDGE AXIS

Theta is the one that cannot be reconstructed later. Detection efficiency
varies with it — a spike whose normal points along the missing-wedge axis is
systematically harder to see than one pointing across it — so any spatial
statistic computed without it is biased by an amount nobody can recover after
the fact. It is written per site, always.

PSI IS NOT DETERMINED by a surface normal: the normal fixes two of the three
Euler angles and leaves the in-plane rotation free. It is written as 0 (or
randomised with --random-psi) and must be refined, never trusted.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# RELION 4 / Warp column names for a tomogram particle set with priors.
STAR_COLUMNS = [
    "rlnTomoName", "rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ",
    "rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi",
    "rlnAngleTiltPrior", "rlnAnglePsiPrior",
]
# Ours, carried alongside so nothing needs recomputing downstream.
EXTRA_COLUMNS = ["rlnTomoParticleId", "tomogrationTheta", "tomogrationShell",
                 "tomogrationVirion"]


def fibonacci(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.sin(phi) * np.cos(th),
                     np.sin(phi) * np.sin(th), np.cos(phi)], 1)


def n_sites(radius_A, spacing_A):
    """How many sites cover a sphere of this radius at ~this spacing.

    Area per site is spacing^2, so n = 4*pi*R^2 / spacing^2. Rounded up, and at
    least 12 — below that the Fibonacci lattice stops being even and clusters
    at the poles, which would bias exactly the direction that matters."""
    if radius_A <= 0 or spacing_A <= 0:
        return 0
    return max(12, int(math.ceil(4 * math.pi * radius_A ** 2 / spacing_A ** 2)))


def euler_from_normal(n):
    """(rot, tilt, psi) in degrees putting the reference Z axis along n.

    RELION's ZYZ convention: tilt is the polar angle off Z, rot the azimuth.
    Psi is the rotation ABOUT the normal, which a surface normal says nothing
    about — it is returned as 0 and must be refined."""
    n = np.asarray(n, dtype=float)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    tilt = math.degrees(math.acos(float(np.clip(n[2], -1.0, 1.0))))
    rot = math.degrees(math.atan2(float(n[1]), float(n[0])))
    return rot, tilt, 0.0


def theta_to_wedge(n, axis=(0.0, 0.0, 1.0)):
    """Angle in degrees between a normal and the wedge axis (Z by convention).

    0 = pointing straight along the missing wedge (worst detected), 90 = lying
    in the well-sampled plane (best). Kept unfolded over 0..180 so the caller
    can decide whether the two poles are equivalent for its statistic."""
    n = np.asarray(n, dtype=float)
    a = np.asarray(axis, dtype=float)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    return math.degrees(math.acos(float(np.clip(np.dot(n, a), -1.0, 1.0))))


def sample_virion(centre_px, radius_px, angpix, spacing_A=50.0,
                  shells_A=(0.0, 50.0, -50.0), shape=None, virion_id=0):
    """Sites on one virion's surface, over every requested shell.

    Shells are radius OFFSETS in Angstroms: the surface itself, plus one
    outside and one inside. They deliberately overlap in what they will pick up
    — that is what oversampling means here — so every site records which shell
    it came from and which virion, and the set is flagged for duplicate removal
    after the first refinement."""
    centre = np.asarray(centre_px, dtype=float)
    radius_A = float(radius_px) * float(angpix)
    out = []
    for shell in shells_A:
        r_A = radius_A + float(shell)
        if r_A <= 0:
            continue
        dirs = fibonacci(n_sites(r_A, spacing_A))
        pts = centre + dirs * (r_A / float(angpix))
        for d, p in zip(dirs, pts):
            if shape is not None:
                if not (0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]
                        and 0 <= p[2] < shape[2]):
                    continue
            # centre_px (and therefore d) is in ARRAY order (z, y, x);
            # euler_from_normal and theta_to_wedge treat index 2 as Z. Passing
            # d unswapped assigned the priors with X and Z exchanged — every
            # tilt/rot prior wrong, and theta (the wedge-quality column, kept
            # precisely because the bias is unrecoverable later) measured
            # against the wrong axis.
            n_xyz = (float(d[2]), float(d[1]), float(d[0]))
            rot, tilt, psi = euler_from_normal(n_xyz)
            out.append({"x": float(p[2]), "y": float(p[1]), "z": float(p[0]),
                        "nx": float(d[2]), "ny": float(d[1]), "nz": float(d[0]),
                        "rot": rot, "tilt": tilt, "psi": psi,
                        "theta": theta_to_wedge(n_xyz),
                        "shell": float(shell), "virion": int(virion_id)})
    return out


def write_star(path, tomo_name, sites, random_psi=False, seed=0):
    """A RELION-style particle star with the surface priors filled in.

    rlnTomoName must match the .tomostar stem EXACTLY — Warp matches particles
    to tilt series on that string, and a near-miss silently exports nothing."""
    rng = np.random.default_rng(seed)
    lines = ["", "data_particles", "", "loop_"]
    cols = STAR_COLUMNS + EXTRA_COLUMNS
    for i, c in enumerate(cols, 1):
        lines.append(f"_{c} #{i}")
    for i, s in enumerate(sites, 1):
        psi = float(rng.uniform(-180, 180)) if random_psi else s["psi"]
        lines.append("\t".join([
            tomo_name,
            f"{s['x']:.2f}", f"{s['y']:.2f}", f"{s['z']:.2f}",
            f"{s['rot']:.2f}", f"{s['tilt']:.2f}", f"{psi:.2f}",
            f"{s['tilt']:.2f}", f"{psi:.2f}",
            str(i), f"{s['theta']:.2f}", f"{s['shell']:.0f}", str(s["virion"]),
        ]))
    Path(path).write_text("\n".join(lines) + "\n")
    return len(sites)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fits", required=True, help="fits.json written by A3")
    ap.add_argument("--tomostar", default="",
                    help="series stem — must match the .tomostar EXACTLY. "
                         "Required for a single-tomogram fits.json; a BATCH "
                         "fits.json names each stem itself.")
    ap.add_argument("--out", required=True,
                    help="output .star (single mode), or the folder for the "
                         "per-stem stars (batch mode / when a directory)")
    ap.add_argument("--spacing", type=float, default=50.0,
                    help="site spacing on the surface, Angstroms")
    ap.add_argument("--shells", default="0,50,-50",
                    help="radius offsets in Angstroms")
    ap.add_argument("--random-psi", action="store_true",
                    help="randomise the undetermined in-plane angle instead of "
                         "writing 0 (avoids a spurious common orientation in "
                         "the first classification)")
    a = ap.parse_args(argv)

    data = json.loads(Path(a.fits).read_text())
    virions = data.get("virions") or []
    if not virions:
        sys.exit(f"{a.fits} lists no accepted virions — nothing to pick.")
    shells = [float(s) for s in a.shells.split(",") if s.strip()]

    def sites_for(vs, angpix):
        out = []
        for v in vs:
            out += sample_virion(v["centre_px"],
                                 float(v["radius_A"]) / angpix, angpix,
                                 spacing_A=a.spacing, shells_A=shells,
                                 virion_id=v.get("label", 0))
        return out

    if data.get("batch"):
        # The combined fits.json a batch fit writes: virions carry their stem,
        # per_tomogram carries each stem's angpix. One star per stem, into a
        # folder — rlnTomoName is the stem itself.
        apx_of = {t.get("stem"): t.get("angpix")
                  for t in data.get("per_tomogram", [])}
        out_dir = Path(a.out)
        if out_dir.suffix == ".star":       # derive wiring hands a file path
            out_dir = out_dir.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        groups = {}
        for v in virions:
            groups.setdefault(v.get("stem", ""), []).append(v)
        written, skipped, sites = 0, [], []
        for stem in sorted(groups):
            apx = apx_of.get(stem)
            if not stem or not apx:
                skipped.append(stem or "(no stem)")
                continue                     # never guess a pixel size
            s = sites_for(groups[stem], float(apx))
            write_star(out_dir / f"{stem}_oversample.star", stem, s,
                       random_psi=a.random_psi)
            sites += s
            written += 1
        for stem in skipped:
            print(f"WARN: {stem}: no angpix in per_tomogram — SKIPPED, "
                  f"never guessed.")
        if not written:
            sys.exit("ERROR: no tomogram in this batch fits.json carried an "
                     "angpix — nothing written.")
        print(f"batch: {written} star(s) in {out_dir} "
              f"({len(skipped)} skipped)")
        n = len(sites)
    else:
        # Single-tomogram fits.json: angpix is REQUIRED — the old silent
        # default of 1.0 scaled every radius by the full binning factor.
        if not data.get("angpix"):
            sys.exit(f"ERROR: {a.fits} has no 'angpix' — refusing to guess "
                     f"(1.0 would mis-scale every radius). Re-run the fit, "
                     f"or point --fits at the per-tomogram fits.json.")
        if not a.tomostar:
            sys.exit("ERROR: --tomostar is required for a single-tomogram "
                     "fits.json (rlnTomoName must match the .tomostar stem "
                     "exactly).")
        angpix = float(data["angpix"])
        sites = sites_for(virions, angpix)
        out_path = Path(a.out)
        if out_path.is_dir():
            out_path = out_path / f"{a.tomostar}_oversample.star"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n = write_star(out_path, a.tomostar, sites, random_psi=a.random_psi)
        a.out = str(out_path)

    th = np.array([s["theta"] for s in sites])
    print(f"{len(virions)} virions -> {n} sites "
          f"({len(shells)} shells at {a.shells} A, ~{a.spacing:.0f} A spacing)")
    print(f"theta to the wedge axis: median {np.median(th):.0f} deg, "
          f"{(th < 30).mean()*100:.0f}% within 30 deg of the axis (the poorly "
          f"detected ones — keep this column, the bias is not recoverable later)")
    print(f"wrote {a.out}")
    print("These shells overlap BY DESIGN — remove duplicates after the first "
          "refinement, not before.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
