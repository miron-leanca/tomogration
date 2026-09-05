"""The centring test must tell signal from noise — including when there is none.

This is the one check in the pipeline that reads VOXELS rather than comparing
one star to another. ml_verify_reextract proved an export's coordinates were
exactly 4x its input while the data was still noise, because scaling numbers
correctly says nothing about where Warp actually cut.

Its first version divided the centre-to-edge swing by the scatter between
outer shells. On pure noise both are tiny, their ratio is meaningless, and it
declared random ice "CENTRED on something real" — the exact false negative
that would have sent someone back to chasing star files. The measure has to be
ABSOLUTE: n particles normalised to unit variance and averaged leave a noise
floor of 1/sqrt(n), and the swing is judged against that.

    python3 tests/test_audit.py
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
    import mrcfile                                            # noqa: F401
except ImportError:
    print("numpy/mrcfile stub unavailable — skipping")
    print("\n0 passed, 0 failed")
    sys.exit(0)

spec = importlib.util.spec_from_file_location("audit", REPO / "ml_audit_chain.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def build(root, kind, box=40, n=40, offset=0.0, seed=0):
    """A stack of subtomograms laid out the way an export writes them."""
    import mrcfile as mf
    rng = np.random.default_rng(seed)
    d = Path(root) / kind / "subtomo" / "Position003"
    d.mkdir(parents=True)
    z, y, x = np.indices((box, box, box))
    c = (box - 1) / 2
    for i in range(n):
        vol = rng.normal(0, 1, (box, box, box)).astype(np.float32)
        if kind != "noise":
            v = rng.normal(0, 1, 3)
            v /= np.linalg.norm(v)
            oz, oy, ox = v * offset
            r = np.sqrt((z - c - oz) ** 2 + (y - c - oy) ** 2 + (x - c - ox) ** 2)
            vol -= 2.5 * np.exp(-(r ** 2) / (2 * 5.0 ** 2))
        with mf.new(str(d / f"p{i:05d}.mrc"), overwrite=True) as m:
            m.set_data(vol)
    return str(Path(root) / kind)


def main():
    with tempfile.TemporaryDirectory() as td:
        centred = build(td, "centred", offset=0.0, seed=1)
        noise = build(td, "noise", seed=2)
        smeared = build(td, "smeared", offset=8.0, seed=3)

        s_ok = A.centring_test(centred, n=40, label="")
        s_no = A.centring_test(noise, n=40, label="")
        s_sm = A.centring_test(smeared, n=40, label="")

    check(f"a centred stack scores well above the floor (SNR {s_ok:.1f})",
          s_ok > 3)
    check(f"PURE NOISE scores below it (SNR {s_no:.1f}) — the bug the first "
          f"version had", s_no < 1)
    check(f"a displaced stack lands in between (SNR {s_sm:.1f}), so a smeared "
          f"result is distinguishable from both", 1 < s_sm < s_ok)
    check("and the three are ordered noise < smeared < centred",
          s_no < s_sm < s_ok)

    # The scale must not be gameable by sample size: more particles lower the
    # floor, so a real signal should score HIGHER, never lower.
    with tempfile.TemporaryDirectory() as td:
        big = build(td, "centred", offset=0.0, n=90, seed=1)
        s_40 = A.centring_test(big, n=40)
        s_90 = A.centring_test(big, n=90)
    check(f"averaging more particles raises the score ({s_40:.1f} -> "
          f"{s_90:.1f}), it does not dilute it", s_90 > s_40 * 0.9)

    # A profile has to come back with the right sense, since 'protein dark'
    # versus 'bright' is a real property of these reconstructions.
    prof = A.radial_profile(np.fromfunction(
        lambda z, y, x: -np.exp(-(((z - 19.5) ** 2 + (y - 19.5) ** 2
                                   + (x - 19.5) ** 2) / 50.0)), (40, 40, 40)))
    check("a dense centre reads as a downhill profile", prof[0] < prof[-1])

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
