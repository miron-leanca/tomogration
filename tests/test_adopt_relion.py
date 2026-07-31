"""Adopting a RELION job must produce a RELION node that can feed M.

THE BUG: _adopt_orphan assumed every found-on-disk item was a template-match pick
set, so adopting a Refine3D result created a job with stage_id="ts_template_match"
— the card read "Template matching · 0 series", and "Build downstream" offered
Threshold picks / Extract particles. There was no route from a finished refinement
into M at all.

Also checks the handoff that makes M usable: the three paths M needs (two UNFILTERED
half maps + the particle star) are carried across automatically, because copying
them by hand is where a filtered half map sneaks in and silently ruins the FSC.

    python3 tests/test_adopt_relion.py
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE / "stub"))
sys.path.insert(0, str(REPO))


def load(mod, path):
    spec = importlib.util.spec_from_file_location(mod, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod] = m
    spec.loader.exec_module(m)
    return m


st = load("tomogration_stages", REPO / "tomogration_stages.py")
jobs = load("tomogration_jobs", REPO / "tomogration_jobs.py")
app = load("tomapp", REPO / "tomogration_app.py")

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}{(' — ' + detail) if detail else ''}")


def main():
    root = Path(tempfile.mkdtemp())
    jd = root / "Refine3D" / "job029"
    jd.mkdir(parents=True)
    for f in ("run_half1_class001_unfil.mrc", "run_half2_class001_unfil.mrc",
              "run_class001.mrc", "run_it005_data.star",
              "run_half1_class001.mrc"):          # a FILTERED map, must not be picked
        (jd / f).write_text("")

    orph = {"stage_id": "relion4_to_warp", "kind": "relion_job",
            "suffix": "Refine3D/job029 it5", "dir": "Refine3D/job029",
            "star": "Refine3D/job029/run_it005_data.star", "n_series": 0}

    class Stub:
        project_root = str(root)
        _adopt_relion_job = app.Tomogration._adopt_relion_job
        _adopt_orphan = app.Tomogration._adopt_orphan
        _symlink_into = staticmethod(app.Tomogration._symlink_into)
        logs = []

        def _log(self, t, lvl):
            self.logs.append(t)

        def _invalidate_orphans(self):
            pass

        def _refresh_canvas(self):
            pass

    s = Stub()
    s._adopt_orphan(orph)
    store = jobs.load_jobs(str(root))["jobs"]
    check("adoption created exactly one job", len(store) == 1)
    job = list(store.values())[0]

    check("adopted as a RELION result, NOT template matching",
          job["stage_id"] == "relion4_result", job["stage_id"])
    check("card is not labelled 'Template matching'",
          "Template matching" not in job.get("label", ""))
    check("marked completed", job["status"] == "completed")
    pr = job["params"]
    check("records the job type", pr["job_type"] == "Refine3D")
    check("carries the particle star",
          pr["data_star"].endswith("run_it005_data.star"))
    check("picks the UNFILTERED half1",
          pr["half1"].endswith("run_half1_class001_unfil.mrc"), pr["half1"])
    check("picks the UNFILTERED half2",
          pr["half2"].endswith("run_half2_class001_unfil.mrc"), pr["half2"])
    check("never picks the FILTERED map as a half map",
          not pr["half1"].endswith("run_half1_class001.mrc"))
    check("finds a class map for masking",
          pr["class_map"].endswith("run_class001.mrc"), pr["class_map"])
    check("tells the user how to reach M",
          any("Build downstream" in t for t in s.logs))

    # ---- the DAG must offer the M route ------------------------------------
    down = jobs.DOWNSTREAM.get("relion4_result", [])
    check("downstream offers M mask", "m_mask_create" in down)
    check("downstream offers M species", "m_create_species" in down)
    check("downstream still offers re-extraction", "relion4_to_warp" in down)
    check("downstream does NOT offer threshold picks", "threshold_picks" not in down)

    # ---- and the child params must arrive pre-filled ------------------------
    mask = jobs.derive_child_params("m_mask_create", "relion4_result", pr, "")
    check("mask job gets the class map as --i",
          mask.get("i", "").endswith("run_class001.mrc"), str(mask))
    sp = jobs.derive_child_params("m_create_species", "relion4_result", pr, "")
    check("species gets half1", sp.get("half1", "").endswith("_unfil.mrc"))
    check("species gets half2", sp.get("half2", "").endswith("_unfil.mrc"))
    check("species gets the particle star",
          sp.get("particles_relion", "").endswith("run_it005_data.star"))
    check("species gets the mask path", sp.get("mask") == "m/mask.mrc")
    # and those values must satisfy the species stage's own validator
    spec = next(x for x in st.STAGES if x["id"] == "m_create_species")
    vals = {p["name"]: p.get("default") for p in spec["params"]}
    vals.update(sp, population="m/x.population", name="spike", diameter="150")
    check("derived species params pass validation", spec["validate"](vals) == "",
          spec["validate"](vals))

    re_x = jobs.derive_child_params("relion4_to_warp", "relion4_result", pr, "")
    check("re-extract gets the particle star",
          re_x.get("particles_star", "").endswith("run_it005_data.star"))

    # ---- a Select job (no half maps) must still adopt, with a warning -------
    sd = root / "Select" / "job009"
    sd.mkdir(parents=True)
    (sd / "particles.star").write_text("")
    s.logs = []
    s._adopt_orphan({"kind": "relion_job", "suffix": "Select/job009",
                     "dir": "Select/job009", "star": "Select/job009/particles.star"})
    store2 = jobs.load_jobs(str(root))["jobs"]
    check("Select job also adopts", len(store2) == 2)
    check("warns that M needs half maps",
          any("half maps" in t for t in s.logs), str(s.logs))

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
