"""Each job writes into its own folder, and the next job reads THAT folder.

The WarpTools stages get this free from --input_processing/--output_processing.
The alignment chain does not: aretomo and miss-alignment are wrapper scripts
taking explicit directories, so every hop was left pointing at the trunk. J13
wrote into the shared aretomo_output/ while its own jobs/J13_.../ stayed empty,
and J14 imported alignments from the shared folder rather than from J13.

Also pinned here: alignment_angpix. It is the pixel size of the STACKS AreTomo
aligned, because that is the unit its .xf shifts carry. It was hardcoded to
1.57 -- EML46's number -- so EML50 at 1.98 imported every shift 26% short with
no error raised anywhere.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stub"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tomogration_jobs import (derive_child_params, build_job_command,  # noqa: E402
                              job_real_outputs, latest_stage_output_dir)
from tomogration_project import ProjectState  # noqa: E402
from tomogration_stages import STAGES  # noqa: E402

passed = failed = 0


def check(name, got, want):
    global passed, failed
    if got == want:
        passed += 1
    else:
        failed += 1
        print(f"FAIL {name}: got {got!r} want {want!r}")


def spec(sid):
    return next(s for s in STAGES if s["id"] == sid)


def test_stack_to_aretomo():
    check("reads the stack job, writes its own",
          derive_child_params("aretomo", "ts_stack", {}, "jobs/J12_ts-stack"),
          {"input_dir": "jobs/J12_ts-stack/tiltstack", "output_dir": "jobs/{jobid}"})


def test_stack_to_aretomo_without_a_job_parent():
    """A trunk parent still yields a usable path, not an empty one."""
    check("trunk parent", derive_child_params("aretomo", "ts_stack", {}, ""),
          {"input_dir": "warp_tiltseries/tiltstack", "output_dir": "jobs/{jobid}"})


def test_aretomo_to_import_alignments():
    check("imports from the job",
          derive_child_params("ts_import_alignments", "aretomo", {},
                              "jobs/J13_align-with-aretomo2"),
          {"alignments": "jobs/J13_align-with-aretomo2/Imod/"})
    check("falls back to the param",
          derive_child_params("ts_import_alignments", "aretomo",
                              {"output_dir": "aretomo_output"}, ""),
          {"alignments": "aretomo_output/Imod/"})
    check("nothing to go on",
          derive_child_params("ts_import_alignments", "aretomo", {}, ""), {})


def test_aretomo_default_is_its_own_folder():
    d = dict((p["name"], p.get("default")) for p in spec("aretomo")["params"])
    check("output default", d["output_dir"], "jobs/{jobid}")


def test_jobid_resolves_in_the_command():
    """{jobid} is substituted at run time; an unresolved literal would make
    AreTomo write to a folder called '{jobid}'."""
    job = {"id": "J13", "stage_id": "aretomo",
           "output_dir": "jobs/J13_align-with-aretomo2", "inputs": {},
           "params": dict((p["name"], p.get("default"))
                          for p in spec("aretomo")["params"])}
    job["params"]["input_dir"] = "jobs/J12_ts-stack/tiltstack"
    cmd = build_job_command(spec("aretomo"), job, {"jobs": {"J13": job}})
    check("no literal token", "{jobid}" in cmd, False)
    check("writes to its own dir", "jobs/J13_align-with-aretomo2" in cmd, True)
    check("reads the stack job", "jobs/J12_ts-stack/tiltstack" in cmd, True)


def test_outputs_pane_finds_the_job_folder():
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "jobs", "J13_align-with-aretomo2", "Imod"))
    job = {"id": "J13", "stage_id": "aretomo",
           "output_dir": "jobs/J13_align-with-aretomo2",
           "params": {"output_dir": "jobs/{jobid}"}}
    check("card names its own folder", job_real_outputs(d, job, spec("aretomo")),
          [("jobs/J13_align-with-aretomo2", "")])


def project(dirs=(), files=()):
    d = tempfile.mkdtemp()
    for rel in dirs:
        os.makedirs(os.path.join(d, rel), exist_ok=True)
    for rel in files:
        full = os.path.join(d, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        open(full, "w").close()
    return d


def test_aretomo_discovery_sees_job_folders():
    """Job-scoped AreTomo writes jobs/J13_.../Imod/; a scan that only knew the
    shared layout reported 'no alignment found' for a run that had succeeded."""
    job_xf = "jobs/J13_align-with-aretomo2/Imod/Position001_Imod/Position001.xf"
    check("nested job layout",
          ProjectState(project(files=[job_xf])).latest_aretomo_imod(),
          "jobs/J13_align-with-aretomo2/Imod/")
    check("flat job layout",
          ProjectState(project(files=["jobs/J13_x/Imod/P.xf"])).latest_aretomo_imod(),
          "jobs/J13_x/Imod/")
    check("shared layout still works",
          ProjectState(project(files=["aretomo_output/Imod/P.xf"])).latest_aretomo_imod(),
          "aretomo_output/Imod/")
    check("job wins over shared",
          ProjectState(project(files=["jobs/J13_x/Imod/P.xf",
                                      "aretomo_output/Imod/P.xf"])).latest_aretomo_imod(),
          "jobs/J13_x/Imod/")
    # J13 vs J9: string ordering picks J9.
    check("numeric job order",
          ProjectState(project(files=["jobs/J9_x/Imod/P.xf",
                                      "jobs/J13_x/Imod/P.xf"])).latest_aretomo_imod(),
          "jobs/J13_x/Imod/")
    check("an Imod dir with no .xf is not an alignment",
          ProjectState(project(dirs=["jobs/J13_x/Imod"])).latest_aretomo_imod(), "")
    check("nothing at all", ProjectState(project()).latest_aretomo_imod(), "")


def test_alignment_angpix_is_not_hardcoded():
    """The one that raises no error: a wrong alignment_angpix scales every
    imported shift and the run still reports success."""
    import tomogration_app  # noqa: F401  (import-time syntax guard)
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "tomogration_app.py")).read()
    check("no literal 1.57 in the import chain",
          "--alignment_angpix 1.57" in src, False)
    check("reads the mdoc instead",
          "--alignment_angpix {angpix}" in src, True)


def test_latest_stage_output_dir_feeds_the_rebuild():
    store = {"jobs": {"J12": {"id": "J12", "stage_id": "ts_stack",
                              "status": "completed",
                              "output_dir": "jobs/J12_ts-stack"}}}
    check("stack job found", latest_stage_output_dir(store, "ts_stack"),
          "jobs/J12_ts-stack")


for fn in sorted(k for k in dict(globals()) if k.startswith("test_")):
    globals()[fn]()
print(f"{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
