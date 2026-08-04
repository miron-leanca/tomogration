"""test_merge_optics.py — collapsing a star's optics groups.

RELION estimates the initial noise spectrum PER OPTICS GROUP, sampling up to 1000
particles from each. After M refines spherical aberration per tilt series
(MCore --ctf_cs) every series carries its own Cs, so every series becomes its own
optics group — 267 of them on EML45, and 5.9 hours of noise estimation before
iteration 1. Observed exactly:

    2   groups -> "Estimating initial noise spectra from 1000 particles"
    267 groups -> "... from 266000 particles"

    python3 tests/test_merge_optics.py
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE / "stub"))
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("mo", REPO / "ml_relion4_merge_optics.py")
mo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mo)

passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}")


OPT_LABELS = """
data_optics

loop_
_rlnOpticsGroup #1
_rlnOpticsGroupName #2
_rlnSphericalAberration #3
_rlnVoltage #4
_rlnImagePixelSize #5
_rlnImageSize #6
_rlnImageDimensionality #7
"""

PAR_LABELS = """
data_particles

loop_
_rlnCoordinateX #1
_rlnImageName #2
_rlnOpticsGroup #3
"""


def star(cs_values, n_par_per_group=1):
    """A star with one optics group per Cs value, and particles pointing at each."""
    opt = "".join(f"{i+1} opticsGroup{i+1} {cs} 300.0 3.14 112 3\n"
                  for i, cs in enumerate(cs_values))
    par = ""
    for i in range(len(cs_values)):
        for k in range(n_par_per_group):
            par += f"{100+i}.5 sub/p{i}_{k}.mrc {i+1}\n"
    return OPT_LABELS + opt + PAR_LABELS + par


# ---- counting --------------------------------------------------------------
s = star(["3.935", "2.700", "2.700", "4.379"], 2)
check("counts optics groups", mo.count_groups(s)[0] == 4)
check("counts particles", mo.count_groups(s)[1] == 8)


# ---- the merge --------------------------------------------------------------
new, before, after, kept = mo.merge_optics(s, {"_rlnSphericalAberration": "2.7"})
check("reports the original group count", before == 4)
check("collapses to one group", after == 1)
n_opt, n_par = mo.count_groups(new)
check("the output really has one optics group", n_opt == 1)
check("NO particles are lost", n_par == 8)
check("Cs override applied", kept["_rlnSphericalAberration"] == "2.7")
check("group renumbered to 1", kept["_rlnOpticsGroup"] == "1")
check("group renamed", kept["_rlnOpticsGroupName"] == "opticsGroup1")

# Every particle must point at the surviving group, or RELION reads a group that
# is no longer there.
blocks = mo.parse_star(new)
par = mo.find_block(blocks, "particles")
gi = par[2].index("_rlnOpticsGroup")
check("every particle reassigned to group 1",
      all(r[gi] == "1" for r in par[3]))
check("the particle rows are otherwise untouched",
      [r[1] for r in par[3]] == [f"sub/p{i}_{k}.mrc"
                                 for i in range(4) for k in range(2)])

# The kept row is the MODE, not the first: one odd series must not define the set.
new2, _, _, kept2 = mo.merge_optics(star(["9.999", "2.700", "2.700", "2.700"]))
check("keeps the most common parameter set, not the first",
      kept2["_rlnSphericalAberration"] == "2.700")

# Values that are genuinely shared must survive untouched.
check("pixel size preserved", kept["_rlnImagePixelSize"] == "3.14")
check("box preserved", kept["_rlnImageSize"] == "112")
check("dimensionality preserved (3 = subtomograms)",
      kept["_rlnImageDimensionality"] == "3")
check("voltage preserved", kept["_rlnVoltage"] == "300.0")


# ---- structural integrity ---------------------------------------------------
# A blank line between the loop_ labels and the first row ends the block for
# RELION's parser — it would read zero particles from a file that looks fine.
lines = new.splitlines()
for i, ln in enumerate(lines):
    if ln.strip().startswith("_rln") and i + 1 < len(lines):
        nxt = lines[i + 1].strip()
        if not nxt.startswith("_rln"):
            check(f"no blank line after the last label before data ({ln.strip()})",
                  nxt != "")
            break
check("output ends with a newline", new.endswith("\n"))
check("both blocks survive", "data_optics" in new and "data_particles" in new)


# ---- already-merged input is a no-op ---------------------------------------
one = star(["2.700"], 3)
new3, before3, after3, _ = mo.merge_optics(one)
check("a single-group star still reports 1 before", before3 == 1)
check("and 1 after", after3 == 1)
check("its particles are untouched", mo.count_groups(new3)[1] == 3)


# ---- failure modes ----------------------------------------------------------
try:
    mo.merge_optics("data_particles\n\nloop_\n_rlnImageName #1\na.mrc\n")
    ok = False
except ValueError:
    ok = True
check("a star with no optics block is refused, not silently mangled", ok)

try:
    mo.merge_optics(OPT_LABELS)          # labels but zero rows
    ok = False
except ValueError:
    ok = True
check("an empty optics block is refused", ok)


# ---- end to end through main() ---------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "matching_conv.star"
    p.write_text(star([f"{2.0 + i * 0.01:.3f}" for i in range(267)], 1))
    check("fixture reproduces the real group count",
          mo.count_groups(p.read_text())[0] == 267)

    rc = mo.main([str(p), "--report"])
    check("--report exits 0", rc == 0)
    check("--report writes nothing", not (Path(tmp) / "matching_conv_1optics.star").exists())

    rc = mo.main([str(p), "--cs", "2.7"])
    out = Path(tmp) / "matching_conv_1optics.star"
    check("main() exits 0", rc == 0)
    check("main() writes the default output name", out.is_file())
    check("the input is NEVER modified", mo.count_groups(p.read_text())[0] == 267)
    n_opt, n_par = mo.count_groups(out.read_text())
    check("267 groups collapsed to 1", n_opt == 1)
    check("all 267 particles kept", n_par == 267)

    rc = mo.main([str(p), "-o", str(Path(tmp) / "custom.star")])
    check("-o honoured", (Path(tmp) / "custom.star").is_file())

    rc = mo.main([str(Path(tmp) / "nope.star")])
    check("a missing file exits 2", rc == 2)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
