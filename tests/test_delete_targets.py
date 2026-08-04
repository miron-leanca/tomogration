"""A permanent delete must never reach raw data or shared directories.

This is the only code in tomogration that removes user files, so the rules live in
a pure function (job_delete_targets) and are asserted here rather than trusted.
The dangerous case is real: a job's declared output dir is often SHARED — several
ts_export_particles jobs write into relion4/ — and a careless wipe would take out
another job's results, or worse, frames/.

    python3 tests/test_delete_targets.py
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE / "stub"))
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("tomjobs", REPO / "tomogration_jobs.py")
jobs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jobs)

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
    for d in ("frames", "mdocs", "tomostar", "warp_tiltseries", "relion4",
              "jobs/J5", "relion4/myexport", "m_min80"):
        (root / d).mkdir(parents=True, exist_ok=True)

    # the job's OWN folder is always fair game
    t, sk = jobs.job_delete_targets(root, "J5", "ts_ctf", {}, [])
    check("job's own jobs/J5 is a target", "jobs/J5" in t, str(t))

    # a private output dir is deletable
    t, sk = jobs.job_delete_targets(root, "J5", "ts_export_particles",
                                    {"output_processing": "relion4/myexport"},
                                    ["output_processing"])
    check("private output dir is a target", "relion4/myexport" in t, str(t))

    # ...but the SHARED parent is not
    t, sk = jobs.job_delete_targets(root, "J5", "ts_export_particles",
                                    {"output_processing": "relion4"},
                                    ["output_processing"])
    check("shared relion4/ is refused", "relion4" not in t, str(t))
    check("and it says why", any("protected" in x for x in sk), str(sk))

    # raw data is unreachable, whatever a job claims
    for bad in ("frames", "mdocs", "tomostar", "warp_tiltseries", "gains", "."):
        t, sk = jobs.job_delete_targets(root, "J5", "x", {"out_dir": bad}, ["out_dir"])
        check(f"{bad!r} can never be deleted", bad not in t, str(t))

    # escaping the project is refused
    for esc in ("../../etc", "/etc"):
        t, sk = jobs.job_delete_targets(root, "J5", "x", {"out_dir": esc}, ["out_dir"])
        check(f"{esc!r} is refused", not any("etc" in x for x in t), str(t))

    # a path that does not exist is simply not listed
    t, sk = jobs.job_delete_targets(root, "J5", "x", {"out_dir": "m_nope"}, ["out_dir"])
    check("non-existent dir is not a target", "m_nope" not in t)

    # blank params are ignored
    t, sk = jobs.job_delete_targets(root, "J5", "x", {"out_dir": ""}, ["out_dir"])
    check("blank param ignored", t == ["jobs/J5"], str(t))

    # duplicates collapse
    t, sk = jobs.job_delete_targets(root, "J5", "x",
                                    {"a": "m_min80", "b": "m_min80"}, ["a", "b"])
    check("duplicate targets collapse", t.count("m_min80") == 1, str(t))

    # the protected set covers everything that matters
    for d in ("frames", "mdocs", "gains", "tomostar", "warp_tiltseries",
              "warp_frameseries", "relion4", "m", "jobs", "Refine3D", "Select"):
        check(f"{d} is in PROTECTED_DIRS", d in jobs.PROTECTED_DIRS)


    # ---- a job must never delete another job's results ----------------------
    # J89 and J98 both wrote to relion4/picks_class3_Spike-flower_v5_260804. J89
    # succeeded, J98 failed. Clearing the failed one would have deleted the
    # successful one's particles — the outcome that makes "clear it and re-run
    # with different parameters" unsafe to offer at all.
    with tempfile.TemporaryDirectory() as tmp3:
        r3 = Path(tmp3)
        shared = "relion4/picks_v5"
        (r3 / shared / "subtomo").mkdir(parents=True)
        (r3 / shared / "matching.star").write_text("x")

        good = jobs.new_job(r3, "ts_export_particles", "good",
                            {"output_processing": shared,
                             "output_star": f"{shared}/matching.star"})
        bad = jobs.new_job(r3, "ts_export_particles", "bad",
                           {"output_processing": shared,
                            "output_star": f"{shared}/matching.star"})
        (r3 / "jobs" / bad["id"]).mkdir(parents=True)
        store = jobs.load_jobs(r3)
        spec = next(x for x in jobs.STAGES if x["id"] == "ts_export_particles")

        t, sk = jobs.job_delete_targets(r3, bad["id"], "ts_export_particles",
                                        bad["params"], spec.get("output_params"),
                                        store=store)
        check("a shared output dir is NOT deletable", shared not in t)
        check("and the refusal names the other job",
              any(shared in x and good["id"] in x for x in sk))
        check("the job's own dir is still deletable",
              f"jobs/{bad['id']}" in t)

        # Without the store it cannot know, and must not pretend to: the old
        # behaviour is preserved so the guard is clearly the store's doing.
        t2, _ = jobs.job_delete_targets(r3, bad["id"], "ts_export_particles",
                                        bad["params"], spec.get("output_params"))
        check("without the store the shared dir is not refused", shared in t2)

        # A dir only THIS job writes to stays deletable — the guard must not make
        # clearing useless.
        jobs.update_job(r3, good["id"], params={"output_processing": "relion4/other",
                                                "output_star": "relion4/other/m.star"})
        (r3 / "relion4/other").mkdir(parents=True)
        t3, _ = jobs.job_delete_targets(r3, bad["id"], "ts_export_particles",
                                        bad["params"], spec.get("output_params"),
                                        store=jobs.load_jobs(r3))
        check("an unshared output dir is deletable again", shared in t3)

    # {jobid} must resolve, or a job-scoped output looks like no output at all
    with tempfile.TemporaryDirectory() as tmp4:
        r4 = Path(tmp4)
        j = jobs.new_job(r4, "ts_export_particles", "e",
                         {"output_processing": "relion4/picks_{jobid}"})
        (r4 / f"relion4/picks_{j['id']}").mkdir(parents=True)
        spec = next(x for x in jobs.STAGES if x["id"] == "ts_export_particles")
        t, _ = jobs.job_delete_targets(r4, j["id"], "ts_export_particles",
                                       j["params"], spec.get("output_params"),
                                       store=jobs.load_jobs(r4))
        check("{jobid} resolves in a delete target",
              f"relion4/picks_{j['id']}" in t)
        check("declared_output_dirs resolves it too",
              f"relion4/picks_{j['id']}"
              in jobs.declared_output_dirs(r4, jobs.load_jobs(r4)["jobs"][j["id"]]))

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
