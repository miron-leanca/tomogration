#!/usr/bin/env python3
"""Split components that are two or more virions stuck together.

    ml_split_clusters.py --components comps.mrc --population membrane/population.json \
                         --out-folder jobs/J40

Connected-component labelling cannot separate virions whose membranes touch, so
one label becomes one "virion" with two or three times the volume. The fitter
already SEES these — it rejects them as "2.2x a whole virion (merged?)" — but
rejecting throws away real particles, and a few slip through the size gate
entirely: in this dataset's own population measurement, 13 of 624 "whole
virions" are over twice the median volume, and the largest is 4.8x it, a 140 nm
"virion" that is certainly a pair.

The split is possible because the population is now MEASURED. Every virion has
essentially the same radius (415 A, MAD 8.5%), so the only unknown per particle
is its centre — and a sphere of known radius fitted by RANSAC either finds a
dense set of surface voxels at that distance or does not. Two virions give two
such centres; one virion gives one, and the second attempt finds nothing.

OVER-SPLITTING IS THE DANGER, not under-splitting: inventing a virion adds a
particle that never existed to every downstream statistic. So a candidate is
kept only if it holds its own share of voxels AND covers enough of a sphere's
directions to be a shell rather than a patch, and the tests check a single
virion is NOT split as hard as they check a pair IS.

Needs numpy + mrcfile: run from membrainseg.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ml_fit_virions as FIT              # noqa: E402


def ransac_sphere(points, R, tol, iters=200, seed=0):
    """(centre, inlier mask) for the best fixed-radius sphere through `points`.

    The centre is the unknown; the radius is not, because the population
    measured it. Each trial seeds from 4 random points, refines the centre with
    the radius held, and scores by how many voxels land within `tol` of it."""
    rng = np.random.default_rng(seed)
    best_c, best_in = None, np.zeros(len(points), bool)
    if len(points) < 8:
        return None, best_in
    for _ in range(iters):
        pick = points[rng.choice(len(points), size=6, replace=False)]
        # Fit the trial centre to the SAMPLE ONLY. Fitting it to every point
        # was the bug: with two shells present the least-squares centre lands
        # between them and matches neither, so a real pair scored zero inliers
        # and was never split.
        c, _res = FIT.fit_centre_fixed_radius(pick, R, pick.mean(0), iters=25)
        d = np.abs(np.linalg.norm(points - c, axis=1) - R)
        inl = d <= tol
        if inl.sum() > best_in.sum():
            best_c, best_in = c, inl
    if best_c is not None and best_in.sum() >= 8:
        # Refine on the consensus set, never on everything.
        best_c, _res = FIT.fit_centre_fixed_radius(points[best_in], R, best_c)
        best_in = np.abs(np.linalg.norm(points - best_c, axis=1) - R) <= tol
    return best_c, best_in


def hollowness(all_points, centre, R, tol):
    """How empty the INSIDE is: shell voxels / voxels within the shell radius.

    Coverage alone cannot tell a hollow virion from a solid blob — put a centre
    in the middle of any dense object and a sphere of radius R will pass
    through plenty of it, in every direction. A virion is a SHELL, so almost
    nothing sits well inside it; a blob is full. This is the test that stopped
    random material being carved into virions."""
    r = np.linalg.norm(all_points - centre, axis=1)
    inside = r < (R - tol)
    shell = np.abs(r - R) <= tol
    n_shell = int(shell.sum())
    if n_shell < 8:
        return 0.0
    return float(n_shell / (n_shell + int(inside.sum())))


def shell_quality(points, centre, R):
    """(coverage fraction, angular extent deg) of the voxels around a centre.

    A patch of a bigger structure can sit at the right distance from SOME point
    — coverage is what separates a shell from a patch, and it is the test that
    stops a single virion being split into two half-virions."""
    v = points - centre
    r = np.linalg.norm(v, axis=1)
    u = v / np.maximum(r[:, None], 1e-9)
    cov, _gap = FIT.coverage(u)
    return float(cov), FIT.angular_extent(u)


def label_array(out, n_labels):
    """Labels in a dtype MRC can actually hold.

    MRC has no int32 mode, so writing one raises rather than saving — after
    all 72 volumes had been split. int16 covers any real components volume;
    uint16 doubles the headroom; float32 is the last resort and is exact for
    label counts far beyond anything a tomogram produces."""
    if n_labels <= 32767:
        return out.astype(np.int16)
    if n_labels <= 65535:
        return out.astype(np.uint16)
    return out.astype(np.float32)


def component_reach(points):
    """How far the component reaches from its own centre, in voxels.

    This is the suspicion test, and it has to be a DISTANCE. The obvious
    alternative — convert the voxel count to an equivalent solid ball and
    compare volumes — is wrong twice over: a segmented virion is a hollow
    SHELL, so its voxel count read as a solid ball gives a radius far below
    the real one, and that was being compared against the volume of a FILLED
    virion. The gate could then never fire: every merged pair sailed through
    untouched while the tool reported success.

    A single virion's voxels all sit about one radius from its centre; a
    merged pair reaches roughly twice that. Shell thickness, segmentation
    weight and voxel size do not enter into it. The 98th percentile rather
    than the maximum, so one stray voxel cannot make a component suspect."""
    return float(np.percentile(
        np.linalg.norm(points - points.mean(0), axis=1), 98))


def split_component(points, R, tol, max_spheres=4, min_frac=0.15,
                    min_coverage=0.30, min_hollow=0.55, min_sep=1.5, seed=0):
    """[(centre, inlier mask)] for the virions inside one merged component.

    Greedy: take the best sphere, remove its voxels, look again. Stops when a
    candidate is too small a share of what is left (`min_frac`), too patchy to
    be a shell (`min_coverage`), too solid to be one (`min_hollow`), or too
    close to a virion already found (`min_sep`) — every threshold here exists
    to refuse a split rather than to find one."""
    remaining = np.ones(len(points), bool)
    found = []
    for _ in range(max_spheres):
        idx = np.flatnonzero(remaining)
        if len(idx) < 8:
            break
        c, inl = ransac_sphere(points[idx], R, tol, seed=seed + len(found))
        if c is None or inl.sum() < max(8, min_frac * len(points)):
            break
        # Two virions that merely TOUCH have centres about 2 radii apart, so
        # anything closer than 1.5 is
        # the SAME shell fitted twice, slightly offset. Without this the greedy
        # pass carved one thick or noisy shell into three "virions" whose
        # centres sat almost on top of each other, and reported a component
        # reaching only 1.7 radii as three whole virions, which is not
        # geometrically possible.
        if any(np.linalg.norm(c - c0) < min_sep * R for c0, _m in found):
            break
        cov, _ext = shell_quality(points[idx][inl], c, R)
        hollow = hollowness(points, c, R, tol)
        if cov < min_coverage or hollow < min_hollow:
            break
        mask = np.zeros(len(points), bool)
        mask[idx[inl]] = True
        found.append((c, mask))
        remaining &= ~mask
    return found


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--components", required=True)
    ap.add_argument("--population", required=True,
                    help="population.json — the MEASURED radius this fits at")
    ap.add_argument("--out-folder", default="split")
    ap.add_argument("--merge-ratio", type=float, default=1.4,
                    help="a component that reaches this many times a virion "
                         "RADIUS from its own centre is a split candidate")
    ap.add_argument("--tol-mad", type=float, default=2.0,
                    help="inlier tolerance in MADs of the population radius")
    ap.add_argument("--min-coverage", type=float, default=0.30)
    ap.add_argument("--min-separation", type=float, default=1.5,
                    help="two virions must have centres at least this many "
                         "RADII apart. Touching virions sit about 2 apart, so "
                         "1.5 keeps every real merged pair and refuses the "
                         "same shell fitted twice")
    a = ap.parse_args(argv)

    pop = json.loads(Path(a.population).read_text())
    # A FOLDER splits every volume in it — the same as the fitter, and what
    # build-downstream fills in.
    vols = FIT.volumes_in(a.components)
    if not vols:
        sys.exit(f"no .mrc volumes in {a.components}")
    if len(vols) > 1:
        print(f"{len(vols)} components volume(s) to split\n")
    tot_in = tot_out = tot_split = 0
    for comp in vols:
        if len(vols) > 1:
            print(f"--- {comp.name}")
        n_in, n_out, n_split = split_one(a, str(comp), pop)
        tot_in += n_in
        tot_out += n_out
        tot_split += n_split
    if len(vols) > 1:
        print(f"\n{'=' * 62}\ndone: {len(vols)} volume(s), {tot_in} labels in, "
              f"{tot_out} out ({tot_split} component(s) split)")
    return 0


def split_one(a, comp_path, pop):
    """Split one components volume. Returns (labels in, labels out, splits)."""
    d, vx = FIT.load_mrc(comp_path)
    R = float(pop["radius_A"]) / vx
    mad = float(pop.get("mad_A", 0.08 * pop["radius_A"])) / vx
    tol = max(1.5, a.tol_mad * mad)
    med_vol = float(pop.get("volume_nm3_median")
                    or pop.get("whole_virion_nm3") or 0.0)
    print(f"population radius {pop['radius_A']:.0f} A = {R:.1f} px   "
          f"tolerance +/-{tol:.1f} px ({a.tol_mad:g} MAD)")

    labels = [int(x) for x in np.unique(d[d > 0])]
    out = np.zeros_like(d, dtype=np.int32)
    next_label, splits, kept = 1, 0, 0
    for l in labels:
        idx = np.argwhere(d == l).astype(float)
        reach = component_reach(idx) / R
        if reach < a.merge_ratio:
            out[d == l] = next_label
            next_label += 1
            kept += 1
            continue
        found = split_component(idx, R, tol, min_coverage=a.min_coverage,
                                min_sep=a.min_separation)
        if len(found) < 2:
            out[d == l] = next_label
            next_label += 1
            kept += 1
            print(f"  label {l:4d}: reaches {reach:.1f}x a virion radius, "
                  f"NOT split ({len(found)} shell found)")
            continue
        print(f"  label {l:4d}: reaches {reach:.1f}x a virion radius "
              f"-> {len(found)} virions")
        for c, mask in found:
            pts = idx[mask].astype(int)
            out[pts[:, 0], pts[:, 1], pts[:, 2]] = next_label
            next_label += 1
        splits += 1

    outdir = Path(a.out_folder)
    outdir.mkdir(parents=True, exist_ok=True)
    dst = outdir / Path(comp_path).name.replace(".mrc", "_split.mrc")
    import mrcfile                          # noqa: PLC0415
    with mrcfile.new(str(dst), overwrite=True) as m:
        m.set_data(label_array(out, next_label - 1))
        m.voxel_size = vx
    print(f"\n{len(labels)} labels in, {next_label - 1} out "
          f"({splits} component(s) split, {kept} unchanged)")
    print(f"wrote {dst}")
    return len(labels), next_label - 1, splits


if __name__ == "__main__":
    sys.exit(main())
