"""M stages must go GREEN once their output is on disk.

Trunk runs (▶ Run) create no job record, so a stage's card is green only if its
`status` lambda finds the output. Every M stage shipped with status=None, so
create_source and create_mask completed with exit 0 and their cards stayed grey
"not built" — the workflow looked unstarted after real work had been done.

Also guards the path fact that caused the earlier breakage: MTools writes
<name>.source next to the PROCESSING SETTINGS (warp_tiltseries/), not into m/.

    python3 tests/test_m_status.py
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE / "stub"))
sys.path.insert(0, str(REPO))


def load(mod):
    spec = importlib.util.spec_from_file_location(mod, REPO / f"{mod}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod] = m
    spec.loader.exec_module(m)
    return m


st = load("tomogration_stages")
proj = load("tomogration_project")
passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}{(' — ' + detail) if detail else ''}")


def stage(sid):
    return next(s for s in st.STAGES if s["id"] == sid)


def main():
    root = Path(tempfile.mkdtemp())
    (root / "m").mkdir()
    (root / "warp_tiltseries").mkdir()
    ps = proj.ProjectState(str(root))

    # every M stage that produces a file must HAVE a status check
    for sid in ("m_create_population", "m_create_source", "m_mask_create",
                "m_create_species", "m_core"):
        check(f"{sid} has a status check", stage(sid).get("status") is not None)

    # nothing on disk yet -> all grey
    for sid in ("m_create_population", "m_create_source", "m_mask_create",
                "m_create_species"):
        ok, _ = stage(sid)["status"](ps)
        check(f"{sid} is grey on an empty project", not ok)

    # population
    (root / "m" / "EML45-spike-closed.population").write_text("x")
    ok, lab = stage("m_create_population")["status"](ps)
    check("population turns green", bool(ok), str(lab))

    # data source — lands beside the SETTINGS, not in m/
    (root / "warp_tiltseries" / "EML45-spike-closed.source").write_text("x")
    ok, lab = stage("m_create_source")["status"](ps)
    check("data source turns green (found in warp_tiltseries/)", bool(ok), str(lab))
    check("source is NOT expected inside m/",
          not (root / "m" / "EML45-spike-closed.source").exists())

    # mask
    ok, _ = stage("m_mask_create")["status"](ps)
    check("mask still grey before it exists", not ok)
    (root / "m" / "mask_closed_spike.mrc").write_text("x")
    ok, lab = stage("m_mask_create")["status"](ps)
    check("mask turns green", bool(ok), str(lab))

    # species lives under a hashed directory
    sp = root / "m" / "species" / "spike_797f75c2"
    sp.mkdir(parents=True)
    ok, _ = stage("m_create_species")["status"](ps)
    check("empty species dir does not count", not ok)
    (sp / "spike.species").write_text("x")
    ok, lab = stage("m_create_species")["status"](ps)
    check("species turns green once the .species exists", bool(ok), str(lab))

    # MCore output
    ok, _ = stage("m_core")["status"](ps)
    check("refine grey before any map", not ok)
    (sp / "spike_half1.mrc").write_text("x")
    ok, lab = stage("m_core")["status"](ps)
    check("refine turns green once a map is written", bool(ok), str(lab))

    # status checks must never walk bulk data
    fat = root / "frames"
    fat.mkdir()
    for i in range(200):
        (fat / f"P{i:03d}.eer").write_text("")
    touched = []
    real = os.listdir

    def spy(p="."):
        touched.append(str(p))
        return real(p)

    os.listdir = spy
    try:
        for sid in ("m_create_population", "m_create_source", "m_mask_create",
                    "m_create_species", "m_core"):
            stage(sid)["status"](ps)
    finally:
        os.listdir = real
    check("M status never enumerates frames/",
          not any(os.path.basename(t.rstrip("/")) == "frames" for t in touched))

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
