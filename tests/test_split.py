"""Splitting virions that connected-component labelling fused into one.

THE DANGER HERE IS OVER-SPLITTING. Failing to split loses a real particle;
inventing one adds a virion that never existed to every downstream count,
diameter and volume — and it looks exactly like a successful split. So these
tests weigh "a single virion is NOT split" and "a blob is NOT carved up" as
heavily as "a pair IS split".

Parameters come from this dataset's own measured population: radius 415 A =
33 px at 12.56 A/px, MAD 8.5%.

    python3 tests/test_split.py
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE / "stub"))
sys.path.insert(0, str(REPO))

passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}")


try:
    import numpy as np
except ImportError:                        # pragma: no cover
    print("numpy not installed — skipping")
    print("\n0 passed, 0 failed")
    sys.exit(0)


def load(mod):
    spec = importlib.util.spec_from_file_location(mod, REPO / f"{mod}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod] = m
    spec.loader.exec_module(m)
    return m


FIT = load("ml_fit_virions")
S = load("ml_split_clusters")

R, TOL = 33.0, 5.6                         # 415 A radius, 2 MADs, at 12.56 A/px
rng = np.random.default_rng(0)


def shell(centre, R=R, n=3000, cap=0.55, jitter=1.0):
    """A virion as segmentation sees it: a noisy shell with polar caps missing."""
    u = FIT.fib(n)
    u = u[np.abs(u[:, 2]) < cap]
    return np.asarray(centre, float) + u * (R + rng.normal(0, jitter, len(u)))[:, None]


def _accepts(mrcfile, dtype):
    try:
        mrcfile.mode_from_dtype(dtype)
        return True
    except ValueError:
        return False


def main():
    # ---- must NOT split -----------------------------------------------------
    one = shell([80, 80, 80])
    check("a single virion stays ONE virion",
          len(S.split_component(one, R, TOL, seed=1)) == 1)
    blob = rng.uniform([60, 60, 60], [110, 110, 110], size=(3000, 3))
    check("a formless dense blob is not carved into virions",
          len(S.split_component(blob, R, TOL, seed=4)) <= 1)
    # The hollowness test is what distinguishes those two: coverage alone
    # cannot, because a centre inside ANY dense object has material in every
    # direction at radius R.
    check("hollowness separates a shell from solid material",
          S.hollowness(one, np.array([80., 80, 80]), R, TOL) > 0.9
          and S.hollowness(blob, blob.mean(0), R, TOL) < 0.55)

    # ---- must split ---------------------------------------------------------
    a, b = np.array([70., 80, 80]), np.array([70 + 1.8 * R, 80, 80])
    two = np.vstack([shell(a), shell(b)])
    found = S.split_component(two, R, TOL, seed=2)
    check(f"a touching pair splits into two ({len(found)})", len(found) == 2)
    if len(found) == 2:
        cs = sorted([c for c, _m in found], key=lambda c: c[0])
        err = max(float(np.linalg.norm(cs[0] - a)), float(np.linalg.norm(cs[1] - b)))
        check(f"and both centres land on the real ones ({err:.1f} px)", err < 4.0)
        shares = sorted(int(m.sum()) for _c, m in found)
        check(f"each takes a real share of the voxels {shares}",
              shares[0] > 0.3 * len(two) / 2)
    chain = [np.array([60., 80, 80]), np.array([60 + 1.8 * R, 80, 80]),
             np.array([60 + 3.6 * R, 80, 80])]
    check("a chain of three splits into three",
          len(S.split_component(np.vstack([shell(c) for c in chain]),
                                R, TOL, seed=3)) == 3)

    # ---- the fit that made it possible -------------------------------------
    # Trial centres are fitted to the SAMPLE, never to every point: with two
    # shells present a least-squares centre over everything lands between them
    # and matches neither, which scored zero inliers and never split a pair.
    c, inl = S.ransac_sphere(two, R, TOL, seed=7)
    check("a RANSAC trial locks onto ONE shell, not the midpoint of both",
          c is not None and 0.25 * len(two) < inl.sum() < 0.75 * len(two))
    check("a fixed radius means only the centre is unknown",
          abs(np.median(np.linalg.norm(two[inl] - c, axis=1)) - R) < 1.5)

    # ---- the suspicion gate -------------------------------------------------
    # This is where a merged pair either gets examined or silently passes. The
    # first version compared a voxel count read as a SOLID ball against the
    # volume of a FILLED virion; a segmented virion is a hollow shell, so the
    # gate could never fire and every merged pair went through untouched while
    # the tool reported success. Both unit tests above still passed — only an
    # end-to-end run caught it. Hence these.
    check("one virion reaches about one radius from its centre",
          0.85 < S.component_reach(one) / R < 1.15)
    check("a merged pair reaches far enough to be suspected",
          S.component_reach(two) / R > 1.4)
    check("a chain of three reaches further still",
          S.component_reach(np.vstack([shell(c) for c in chain])) / R >
          S.component_reach(two) / R)
    check("the gate is a DISTANCE, so shell thickness cannot move it",
          abs(S.component_reach(shell([80, 80, 80], n=9000))
              - S.component_reach(one)) < 2.0)
    check("and one stray voxel cannot make a lone virion suspect",
          S.component_reach(np.vstack([one, [[80, 80, 200]]])) / R < 1.4)

    # ---- batch pairing ------------------------------------------------------
    # J42 failed on its first run because build-downstream prefilled the
    # components FOLDER and the fitter opened it as a file. Both tools now
    # take either, and pairing a components volume with the wrong tomogram is
    # the quiet failure worth guarding: same grid, different pixel size.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        for n in ("Position003_12.56Apx_isonet2", "Position106_12.56Apx_isonet2",
                  "Position10_12.56Apx_isonet2"):
            (t / f"{n}.mrc").write_bytes(b"")
        got = FIT.pair_tomogram(
            "Position003_12.56Apx_isonet2_scores_threshold_-1.5_components", t)
        check("a long components name pairs with its own tomogram",
              got is not None and got.stem == "Position003_12.56Apx_isonet2")
        check("Position10 does not steal Position106's volume",
              FIT.pair_tomogram("Position106_12.56Apx_isonet2_x", t).stem
              == "Position106_12.56Apx_isonet2")
        check("an unpairable name returns nothing rather than guessing",
              FIT.pair_tomogram("Position999_12.56Apx_isonet2_x", t) is None)
        check("a folder yields every volume, a file yields itself",
              len(FIT.volumes_in(t)) == 3
              and len(FIT.volumes_in(t / "Position10_12.56Apx_isonet2.mrc")) == 1)
        check("and they come back in natural order, not Position10 first",
              [f.stem.split("_")[0] for f in FIT.volumes_in(t)]
              == ["Position003", "Position10", "Position106"])

    # ---- what J44 hit on the cluster ---------------------------------------
    # 1. It split all 72 volumes and then died writing the first one: MRC has
    #    no int32 mode. The stub mrcfile enforces the real rules, so this test
    #    fails the same way the cluster did rather than passing on a lie.
    import mrcfile                                           # noqa: PLC0415
    for n, want in ((5, "int16"), (40000, "uint16"), (70000, "float32")):
        arr = S.label_array(np.zeros((4, 4, 4), np.int32), n)
        ok_dtype = True
        try:
            mrcfile.mode_from_dtype(arr.dtype)
        except ValueError:
            ok_dtype = False
        check(f"{n} labels are written as {want}, which MRC can hold",
              ok_dtype and arr.dtype.name == want)
    check("int32 is refused by the stub exactly as mrcfile refuses it",
          not _accepts(mrcfile, np.int32))

    # 2. It reported a component reaching 1.7 radii as THREE virions. Three
    #    whole virions cannot fit that close together — those were one shell
    #    fitted three times, slightly offset, which is the over-splitting that
    #    invents particles.
    thick = np.vstack([shell([80, 80, 80], R=R + d, n=4000, jitter=1.6)
                       for d in (-2.0, 0.0, 2.0)])
    got = S.split_component(thick, R, TOL, seed=11)
    check(f"one thick noisy shell is not carved into virions ({len(got)})",
          len(got) <= 1)
    check("and the reach of such a blob never justified more than one",
          S.component_reach(thick) / R < 1.4)
    near = S.split_component(two, R, TOL, min_sep=2.5, seed=2)
    check("raising the separation floor refuses a split it would have made",
          len(near) == 1)
    check("while the default still splits a genuinely merged pair",
          len(S.split_component(two, R, TOL, seed=2)) == 2)

    # The invariant that makes "3 virions in 1.7 radii" impossible to report,
    # whatever the data looks like: every pair of accepted centres is at least
    # min_sep radii apart. Checked over shapes that reach past the gate, which
    # is where the cluster run went wrong.
    shapes = {
        "a merged pair": two,
        "a chain of three": np.vstack([shell(c) for c in chain]),
        "a shell with a lobe": np.vstack([
            shell([80, 80, 80]),
            shell([80, 80 + 0.9 * R, 80], R=0.55 * R, n=1500)]),
        "random material": rng.uniform([50, 50, 50], [140, 140, 140],
                                       size=(4000, 3)),
        "a shell in noise": np.vstack([
            shell([90, 90, 90]),
            rng.uniform([50, 50, 50], [130, 130, 130], size=(1200, 3))]),
    }
    for label, pts in shapes.items():
        got = S.split_component(pts, R, TOL, seed=5)
        cs = [c for c, _m in got]
        gaps = [float(np.linalg.norm(cs[i] - cs[j]))
                for i in range(len(cs)) for j in range(i + 1, len(cs))]
        check(f"{label}: no two virions land closer than the floor "
              f"({len(got)} found)", all(g >= 1.5 * R for g in gaps))
        check(f"{label}: the count is possible for how far it reaches",
              len(got) <= 1 or S.component_reach(pts) / R >= 1.4)

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
