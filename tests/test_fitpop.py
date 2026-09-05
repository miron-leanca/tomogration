"""Membrane location estimation (A3) — the numbers that would be published.

THE POINT of these tests, on synthetic volumes where the truth is known:

  * VOLUME comes from the constrained radius. The spec calls a free-ellipsoid
    volume "the single easiest way to publish a wrong number from this
    pipeline"; this MEASURES that error on a capped shell instead of asserting
    it, so the reason for the constraint is visible rather than folklore.
  * POLARITY flips the density gate. The same virion in a dark-membrane and a
    bright-membrane volume must both pass; assuming dark on an inverted variant
    is what rejects every real virion and accepts bulk ice.
  * RECALL exists. Every gate measures precision, so accepting three virions
    and discarding everything else scores perfectly on all of them.

Needs numpy (mrcfile is NOT needed — nothing here touches a file).

    python3 tests/test_fitpop.py
"""
import importlib.util
import sys
import tempfile
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
except ImportError:                      # pragma: no cover - dev box without numpy
    print("numpy not installed — skipping A3 fitting tests")
    print("\n0 passed, 0 failed")
    sys.exit(0)


def load(mod):
    spec = importlib.util.spec_from_file_location(mod, REPO / f"{mod}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod] = m
    spec.loader.exec_module(m)
    return m


F = load("ml_fit_virions")
V = load("tomogration_variants")


def coverage_of(u):
    """coverage() returns (fraction, gap axis); the gap is what matters here."""
    return F.coverage(u)


def sphere_points(centre, R, n=4000, cap=None):
    """Points on a sphere. `cap` drops directions whose |z| exceeds it — the
    missing polar caps a 60-degree tilt range leaves behind."""
    u = F.fib(n)
    if cap is not None:
        u = u[np.abs(u[:, 2]) < cap]
    return np.asarray(centre) + u * R


def main():
    # ---- the fit recovers a known sphere ----------------------------------
    c, R = F.fit_sphere(sphere_points([40.0, 50.0, 60.0], 20.0))
    check("sphere fit recovers the centre",
          np.allclose(c, [40, 50, 60], atol=0.2))
    check("sphere fit recovers the radius", abs(R - 20.0) < 0.2)

    # A capped shell: the free fit still finds the centre, and the fixed-radius
    # fit rescues it from a fragment — the two passes of A3.
    pts = sphere_points([40.0, 50.0, 60.0], 20.0, cap=0.5)
    c2, resid = F.fit_centre_fixed_radius(pts, 20.0, np.array([38.0, 52.0, 61.0]))
    check("fixed-radius fit recovers the centre of a capped shell",
          np.allclose(c2, [40, 50, 60], atol=0.3))
    check("and its residual is small", float(np.sqrt((resid ** 2).mean())) < 0.2)

    # ---- THE volume trap ---------------------------------------------------
    # The mechanism is NOT that the fit is under-determined: a noiseless capped
    # sphere fits back to a sphere exactly (checked below), and symmetric radial
    # noise only elongates it by ~1.03x. What actually elongates real virions is
    # the MISSING WEDGE smearing the reconstruction along z, so the segmented
    # membrane genuinely lies further out there. A free ellipsoid faithfully
    # follows that smear; volume is the product of three axes, so it inherits
    # it in full. Simulate the smear and measure both estimators against it.
    def ellipsoid_axes(xyz):
        x, y, z = xyz.T
        D = np.stack([x * x, y * y, z * z, 2 * x * y, 2 * x * z, 2 * y * z], 1)
        sol, *_ = np.linalg.lstsq(D, np.ones(len(xyz)), rcond=None)
        A = np.array([[sol[0], sol[3], sol[4]],
                      [sol[3], sol[1], sol[5]],
                      [sol[4], sol[5], sol[2]]])
        w = np.linalg.eigvalsh(A)
        return None if np.any(w <= 0) else np.sort(1.0 / np.sqrt(w))

    R_TRUE, SMEAR = 20.0, 1.5
    clean = sphere_points([0.0, 0.0, 0.0], R_TRUE, n=6000, cap=0.5)
    check("a noiseless capped shell fits back to a SPHERE — under-determination "
          "is not the mechanism",
          abs(np.prod(ellipsoid_axes(clean)) / R_TRUE ** 3 - 1.0) < 0.02)

    smeared = clean.copy()
    smeared[:, 2] *= SMEAR                       # what the missing wedge does
    axes = ellipsoid_axes(smeared)
    ell_inflation = float(np.prod(axes)) / R_TRUE ** 3
    r = np.linalg.norm(smeared, axis=1)
    u = smeared / r[:, None]
    _cov, gap = coverage_of(u)
    R_band, _band = F.radius_away_from_gap(r, u, gap)
    ours_inflation = (R_band / R_TRUE) ** 3
    check(f"the gap axis is correctly identified as z (gap {np.round(gap, 2)})",
          gap[2] > gap[0] and gap[2] > gap[1])
    check(f"free-ellipsoid volume inherits the smear in full "
          f"({ell_inflation:.2f}x too large)", ell_inflation > 1.4)
    check(f"the constrained radius is nearly immune to it "
          f"({ours_inflation:.2f}x)", ours_inflation < 1.15)
    check(f"so the constraint buys a {ell_inflation / ours_inflation:.1f}x "
          f"better volume on smeared data",
          ell_inflation / ours_inflation > 1.3)
    # The constrained radius gives the honest number, in nm³.
    vol = F.virion_volume_nm3(20.0, 12.56)          # R=20 px at 12.56 A/px
    check("constrained volume is 4/3 pi R^3 in nm³",
          abs(vol - 4.0 / 3.0 * np.pi * (20 * 1.256) ** 3) < 1e-6)
    check("a bin4 fit of the same physical virion gives the same volume",
          abs(F.virion_volume_nm3(40.0, 6.28) - vol) < 1e-6)

    # ---- polarity flips the density gate -----------------------------------
    # One synthetic tomogram, membranes dark; the same volume negated. The same
    # virion must pass in both, each with its own polarity.
    shape = (60, 60, 60)
    tomo_dark = np.random.default_rng(0).normal(0, 0.05, shape).astype("f4")
    seg = np.zeros(shape, bool)
    ctr, rad = np.array([30.0, 30.0, 30.0]), 18.0
    p = np.round(ctr + F.fib(4000) * rad).astype(int)
    p = p[np.all((p >= 0) & (p < 60), axis=1)]
    tomo_dark[p[:, 0], p[:, 1], p[:, 2]] = -1.0     # membranes DARK
    seg[p[:, 0], p[:, 1], p[:, 2]] = True
    tomo_bright = -tomo_dark

    sup_d, ins_d = F.density_support(tomo_dark, seg, ctr, rad, V.DARK)
    sup_b, _ = F.density_support(tomo_bright, seg, ctr, rad, V.BRIGHT)
    check(f"dark variant: the virion is supported ({sup_d:.2f})", sup_d > 0.9)
    check(f"bright variant, with the right polarity: also supported ({sup_b:.2f})",
          sup_b > 0.9)
    check("the surface is inside the volume", ins_d > 0.9)
    # The failure the spec is most worried about is silent, so pin it: a
    # phantom in bulk ice must NOT be supported by density alone.
    empty = np.random.default_rng(1).normal(0, 0.05, shape).astype("f4")
    sup_phantom, _ = F.density_support(empty, np.zeros(shape, bool),
                                       ctr, rad, V.DARK)
    check(f"a phantom in bulk ice is not supported ({sup_phantom:.2f})",
          sup_phantom < 0.55)
    check("unknown polarity REFUSES to gate rather than guess a sign",
          all(np.isnan(x) for x in
              F.density_support(tomo_dark, seg, ctr, rad, V.UNKNOWN)))

    # ---- recall ------------------------------------------------------------
    labels = np.zeros((20, 20, 20), np.int32)
    labels[:5, :, :] = 1        # 2000 voxels
    labels[5:7, :, :] = 2       # 800
    labels[7:8, :, :] = 3       # 400
    check("recall: accepting the big component gets most of the voxels",
          abs(F.recall(labels, {1}) - 2000 / 3200) < 1e-9)
    check("recall: accepting everything is 1.0",
          abs(F.recall(labels, {1, 2, 3}) - 1.0) < 1e-9)
    check("recall: accepting nothing is 0.0, not an error",
          F.recall(labels, set()) == 0.0)
    check("recall: three good virions out of many score POORLY, which is why "
          "it is measured", F.recall(labels, {3}) < 0.15)

    # ---- population statistics survive between tomograms -------------------
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "population.json"
        F.save_population(p, radius_A=430.0, whole_nm3=333.0, n_virions=3,
                          tomograms=["Position003"])
        F.save_population(p, radius_A=440.0, whole_nm3=340.0, n_virions=4,
                          tomograms=["Position045"])
        pop = F.load_population(p)
        check("population: virions accumulate across tomograms",
              pop["n_virions"] == 7)
        check("population: and it records which tomograms contributed",
              pop["tomograms"] == ["Position003", "Position045"])

    # ---- sizes stay physical ----------------------------------------------
    check("A3 resolves its min size through the variant registry",
          V.nm3_to_voxels(V.parse_size("1000@12.56")[1], 6.28) == 8000)

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
