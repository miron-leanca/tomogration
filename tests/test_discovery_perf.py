"""Discovery must not enumerate bulk data directories.

REGRESSION GUARD. discover_relion_jobs once used root.glob("*/*/JobType/job*"),
which makes Python list EVERY directory two levels deep — including frames/ with
tens of thousands of .eer. It runs on the UI thread on a timer, so that single
glob pattern froze the whole app for long stretches. This builds a project with a
fat frames/ dir and asserts (a) the RELION jobs are still found at all three
depths and (b) the scan never reads frames/.

    python3 tests/test_discovery_perf.py
"""
import os
import sys
import tempfile
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE / "stub"))
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("tomjobs", REPO / "tomogration_jobs.py")
jobs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jobs)

passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}")


def main():
    root = Path(tempfile.mkdtemp())
    # RELION jobs at all three depths
    for rel, star in ((("Class3D", "job005"), "run_it025_data.star"),
                      (("relion4", "Refine3D", "job010"), "run_it017_data.star"),
                      (("relion4", "warp", "Select", "job009"), "particles.star")):
        d = root.joinpath(*rel)
        d.mkdir(parents=True)
        (d / star).write_text("data_\n")
    # a fat data dir that must NEVER be enumerated
    fat = root / "frames"
    fat.mkdir()
    for i in range(300):
        (fat / f"Position{i:03d}_0001.eer").write_text("")
    (root / "warp_tiltseries").mkdir()

    # instrument os.scandir + os.listdir to see what gets read
    touched = []
    real_scandir, real_listdir = os.scandir, os.listdir

    def spy_scandir(path=".", *a, **k):
        touched.append(str(path))
        return real_scandir(path, *a, **k)

    def spy_listdir(path=".", *a, **k):
        touched.append(str(path))
        return real_listdir(path, *a, **k)

    os.scandir, os.listdir = spy_scandir, spy_listdir
    try:
        found = jobs.discover_relion_jobs(root)
    finally:
        os.scandir, os.listdir = real_scandir, real_listdir

    sfx = sorted(f["suffix"] for f in found)
    check("finds job at project root", any(s.startswith("Class3D/job005") for s in sfx))
    check("finds job one level in", any(s.startswith("Refine3D/job010") for s in sfx))
    check("finds job two levels in", any(s.startswith("Select/job009") for s in sfx))
    check("exactly the three jobs", len(found) == 3)

    read_fat = [t for t in touched if os.path.basename(t.rstrip("/")) == "frames"]
    check("NEVER enumerates frames/", not read_fat)
    check("skip-list covers the bulk dirs",
          {"frames", "mdocs", "warp_tiltseries"} <= jobs.SKIP_SCAN_DIRS)
    check("scan stays small (no data-dir walk)", len(touched) < 40)

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
