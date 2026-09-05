"""The project inventory model behind the input-field browser.

Shaped by the real EML46 dump (2026-08): job dirs renamed by hand
(jobs/J27-bin8-isonet2model), 17k-file subtomo dirs that must collapse, and
pixel-size tags that must NOT collapse across binnings.

    python3 tests/test_inventory.py
"""
import importlib.util
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


jb = load("tomogration_jobs")
inv = load("tomogration_inventory")

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
    # ---- name templates -----------------------------------------------------
    check("digit runs collapse to #",
          inv.name_template("Position002_12.56Apx.mrc")
          == inv.name_template("Position115_12.56Apx.mrc"))
    check("pixel-size tags stay literal (binnings are different volumes)",
          inv.name_template("Position002_12.56Apx.mrc")
          != inv.name_template("Position002_6.28Apx.mrc"))
    check("template shows the Apx tag",
          "12.56Apx" in inv.name_template("Position002_12.56Apx.mrc"))

    names = [f"Position{i:03d}_12.56Apx.mrc" for i in range(2, 74)] \
        + ["PROVENANCE.json", "notes.txt"]
    rows = inv.collapse_files(names)
    fam = [r for r in rows if "template" in r]
    single = [r for r in rows if "name" in r]
    check("a 72-file family collapses to one row",
          len(fam) == 1 and fam[0]["count"] == 72)
    check("small files stay individual rows",
          {r["name"] for r in single} == {"PROVENANCE.json", "notes.txt"})

    # ---- list_dir -----------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        d = root / "warp_tiltseries" / "reconstruction"
        d.mkdir(parents=True)
        for i in range(2, 74):
            (d / f"Position{i:03d}_12.56Apx.mrc").write_text("x")
        (d / "PROVENANCE.json").write_text("{}")
        (root / "warp_tiltseries" / "results~").mkdir()      # backup noise
        (root / "warp_tiltseries" / "__pycache__").mkdir()
        (root / "warp_tiltseries" / "logs").mkdir()

        entries, trunc = inv.list_dir(root, "warp_tiltseries")
        names = [e["name"] for e in entries]
        check("subdirs listed first", names[0] == "logs")
        check("noise dirs are hidden",
              "results~" not in names and "__pycache__" not in names)

        entries, trunc = inv.list_dir(root, "warp_tiltseries/reconstruction")
        fam = [e for e in entries if e.get("family")]
        check("the tomogram family is one collapsed row",
              len(fam) == 1 and fam[0]["count"] == 72)
        check("collapsed row carries its member files",
              "Position002_12.56Apx.mrc" in fam[0]["family"])
        check("PROVENANCE.json is its own row",
              any(e["name"] == "PROVENANCE.json" and not e["is_dir"]
                  for e in entries))
        check("rel paths are project-relative",
              fam[0]["rel"] == "warp_tiltseries/reconstruction")
        check("not truncated below the cap", trunc is False)

        big = root / "big"
        big.mkdir()
        for i in range(700):
            (big / f"f{i}.bin").write_text("x")
        _, trunc = inv.list_dir(root, "big", cap=100)
        check("a 17k-file dir stops at the cap and says so", trunc is True)

    # ---- job dir resolution (hand-renamed dirs) -----------------------------
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "jobs" / "J27-bin8-isonet2model").mkdir(parents=True)
        (root / "jobs" / "J2").mkdir()
        check("renamed job dir resolves by J-number prefix",
              inv.resolve_job_dir(root, "J27") == "jobs/J27-bin8-isonet2model")
        check("J2 does not claim J27's folder",
              inv.resolve_job_dir(root, "J2") == "jobs/J2")
        check("a missing job dir resolves to ''",
              inv.resolve_job_dir(root, "J99") == "")

    # ---- p-spelled pixel tags (12p56Apx) ------------------------------------
    check("dot-free Apx tags also stay literal",
          inv.name_template("Position002_12p56Apx.mrc")
          != inv.name_template("Position002_6p28Apx.mrc"))

    # ---- job_locations: params name the REAL product dirs -------------------
    # The EML46 store proves output_dir lies: exports live in relion4/<x>_J10,
    # IsoNet work in membrane/isonet2, predictions in ANOTHER job's dir.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "jobs" / "J10").mkdir(parents=True)
        (root / "relion4" / "cryolo_260813_J10").mkdir(parents=True)
        job = {"id": "J10", "output_dir": "jobs/J10",
               "params": {"output_processing": "relion4/cryolo_260813_J10",
                          "box": "64", "input_pattern": "*clean.star",
                          "missing": "relion4/not_there"}}
        locs = inv.job_locations(root, job)
        check("own jobs/ dir comes first", locs[0] == "jobs/J10")
        check("param-named product dir is a location too",
              "relion4/cryolo_260813_J10" in locs)
        check("non-existent and glob params are not locations",
              len(locs) == 2)

    # ---- which fields get the Browse button ---------------------------------
    st = load("tomogration_stages")
    check("input_dir is pathish",
          st.param_is_pathish({"name": "input_dir", "kind": "text"}))
    check("settings is pathish",
          st.param_is_pathish({"name": "settings", "kind": "text"}))
    check("a slashy default is pathish",
          st.param_is_pathish({"name": "whatever", "kind": "text",
                               "default": "membrane/deconv"}))
    check("script fields are NOT (they point into the app)",
          not st.param_is_pathish({"name": "script", "kind": "text",
                                   "default": "/app/ml_x.sh"}))
    check("pick_from fields keep their own picker",
          not st.param_is_pathish({"name": "MB_TOMO_LIST", "kind": "env",
                                   "pick_from": "input_dir"}))
    check("numbers are not pathish",
          not st.param_is_pathish({"name": "box", "kind": "slider_int"}))
    check("explicit browse: False wins",
          not st.param_is_pathish({"name": "input_dir", "kind": "text",
                                   "browse": False}))

    # ---- sources ------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "jobs").mkdir()
        j1 = jb.new_job(root, "ts_template_match", "picks", {})
        j2 = jb.new_job(root, "mb_isonet2_predict", "corrected", {})
        (root / "jobs" / f"{j2['id']}-bin8").mkdir()
        (root / "Class3D" / "job003").mkdir(parents=True)
        (root / "aretomo_output-v4").mkdir()
        (root / "warp_tiltseries" / "reconstruction").mkdir(parents=True)
        rows = inv.inventory_sources(root)
        kinds = [r["kind"] for r in rows]
        ids = [r["id"] for r in rows]
        check("jobs come first, newest first",
              ids[0] == j2["id"] and ids[1] == j1["id"])
        check("renamed job row resolves to the real folder",
              rows[0]["rel"] == f"jobs/{j2['id']}-bin8")
        check("RELION job dirs are rows", "Class3D/job003" in ids)
        check("AreTomo versions are rows", "aretomo_output-v4" in ids)
        check("canonical dirs present when they exist",
              "warp_tiltseries/reconstruction" in ids)
        check("project root is always the last resort",
              rows[-1]["title"] == "Project root")
        check("every row names its kind",
              all(k in ("job", "dir", "relion") for k in kinds))

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
