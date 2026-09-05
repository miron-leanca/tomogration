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

        # ---------------------------------------------------------------------------
    # The viewer inventory is the same class of bug, and it bit on 2026-08-19:
    # "view in napari" on a 72-tomogram mesh job froze tomogration until the
    # desktop offered to kill it. viewer_inventory listed the series with one
    # rglob and then called viewer_plan PER SERIES, each doing another rglob of
    # the same tree — 73 recursive walks plus 72 globs of the tomogram folder, on
    # the UI thread. Fine on a local disk, fatal on /ceph where every stat is a
    # network round trip. It must stay a single pass however many tomograms there
    # are.
    import pathlib as _pl
    import time as _time

    with tempfile.TemporaryDirectory() as _td:
        _r = Path(_td)
        _out = _r / "jobs/J39"
        _out.mkdir(parents=True)
        _grey = _r / "grey"
        _grey.mkdir()
        for _i in range(1, 73):
            _p = f"Position{_i:03d}"
            (_grey / f"{_p}_12.56Apx_isonet2.mrc").write_bytes(b"")
            _stem = f"{_p}_12.56Apx_isonet2_scores_threshold_-1.5_components"
            for _k in range(6):
                (_out / f"{_stem}_mesh{_k}.h5").write_bytes(b"")

        _walks = {"n": 0}
        _orig = _pl.Path.rglob

        def _counting(self, pat):
            _walks["n"] += 1
            return _orig(self, pat)

        _pl.Path.rglob = _counting
        try:
            _t = _time.perf_counter()
            _inv = jobs.viewer_inventory("mb_mesh", "jobs/J39", "grey", _r)
            _dt = _time.perf_counter() - _t
        finally:
            _pl.Path.rglob = _orig

        check("the inventory walks the output tree ONCE, not once per series "
              f"({_walks['n']} walk(s) for 72 tomograms)", _walks["n"] <= 2)
        check(f"and stays quick enough for the UI thread ({_dt * 1000:.0f} ms)",
              _dt < 1.0)
        check("all 72 tomograms are still listed", len(_inv) == 72)
        check("each series still carries its six meshes",
              all(sum(f.endswith(".h5") for f in v) == 6
                  for v in _inv.values()))
        # surforama is handed ONE .h5 whose densities were projected from the
        # tomogram when the mesh was built, so a .mrc here would be an entry
        # that cannot be opened. The tomogram folder is still read — it is what
        # names the groups — it just is not offered as something to tick.
        check("but no tomogram is offered to surforama, which cannot take one",
              all(not f.endswith(".mrc") for v in _inv.values() for f in v))
        check("the groups are still named after the tomograms",
              all(k.startswith("Position") and "_12_" not in k for k in _inv))
        check("Position10 did not collect Position106's tomogram",
              all("Position106" not in f for f in _inv.get("Position010", [])))

        print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
