"""A card must name where its files REALLY landed.

J4 converted a gain into gains/original_gain.mrc and its Outputs tab offered
`jobs/J4` — a folder only WarpTools --output_processing stages ever create.
Clicking it reported "Directory does not exist yet". Thirty wrapper stages
declared no output at all, so this was the rule, not the exception.

Also here: a FAILED job re-runs in place. J4's re-run with the corrected gain
path went in as an untracked trunk run because a failed job would not bind to
the builder, leaving a red card over a successful run.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stub"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tomogration_jobs import (job_real_outputs, job_delete_targets,  # noqa: E402
                              RERUNNABLE, PROTECTED_DIRS)
from tomogration_stages import STAGES, STAGE_OUTPUTS  # noqa: E402

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


def project(dirs=(), files=()):
    d = tempfile.mkdtemp()
    for rel in dirs:
        os.makedirs(os.path.join(d, rel), exist_ok=True)
    for rel in files:
        full = os.path.join(d, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        open(full, "w").close()
    return d


GAIN_JOB = {"id": "J4", "stage_id": "gain_convert",
            "params": {"in_gain": "gains/20260811_100618_EER_GainReference.gain",
                       "out_mrc": "gains/original_gain.mrc"}}


def test_gain_convert_points_at_gains():
    d = project(files=["gains/original_gain.mrc"])
    check("gain output", job_real_outputs(d, GAIN_JOB, spec("gain_convert")),
          [("gains", "holds original_gain.mrc")])


def test_output_named_but_not_written_yet():
    """Before it runs the parent is still the right place to look."""
    d = project(dirs=["gains"])
    check("gain pending", job_real_outputs(d, GAIN_JOB, spec("gain_convert")),
          [("gains", "original_gain.mrc is not there (yet)")])


def test_conventional_fallback():
    """STAGE_OUTPUTS already knew; it simply was not being asked."""
    d = project(dirs=["warp_frameseries"])
    check("settings fallback",
          job_real_outputs(d, {"id": "J5", "stage_id": "create_settings_fs",
                               "params": {}}, spec("create_settings_fs")),
          [("warp_frameseries", "where this step writes (by convention)")])


def test_fallback_never_claims_the_project_root():
    """'this job wrote to your whole project' is noise, not an answer."""
    d = project()
    for sid in ("imod_warp_key", "rename"):
        check(f"root not claimed ({sid})",
              job_real_outputs(d, {"id": "J1", "stage_id": sid, "params": {}},
                               spec(sid)), [])


def test_fallback_yields_to_a_real_parameter():
    """A param that resolved something specific wins over the convention."""
    d = project(files=["gains/original_gain.mrc"], dirs=["gains"])
    got = job_real_outputs(d, GAIN_JOB, spec("gain_convert"))
    check("no convention row", [n for _, n in got if "convention" in n], [])


def test_no_fallback_when_the_stage_declares_outputs():
    """A stage that DOES declare outputs and resolved none of them wrote
    somewhere unnameable; the convention would point away from the truth."""
    d = project(dirs=["m"])
    got = job_real_outputs(
        d, {"id": "J64", "stage_id": "m_core",
            "params": {"population": "/elsewhere/x.population"}}, spec("m_core"))
    check("declared stage, no guess",
          [n for _, n in got if "convention" in n], [])


def test_fallback_only_when_the_dir_exists():
    """Every row is a directory that exists — a plausible path is worse."""
    check("absent convention",
          job_real_outputs(project(), {"id": "J5",
                                       "stage_id": "create_settings_fs",
                                       "params": {}},
                           spec("create_settings_fs")), [])


def test_jobid_token_resolves():
    """declared_output_dirs resolved {jobid}; this did not, so a job-scoped
    output became a literal '{jobid}' matching nothing."""
    d = project(dirs=["jobs/J9"])
    check("jobid resolved",
          job_real_outputs(d, {"id": "J9", "stage_id": "mb_polarity",
                               "params": {"out": "jobs/{jobid}"}},
                           dict(spec("mb_polarity"), output_params=["out"])),
          [("jobs/J9", "")])


def test_declared_stages_have_the_param_they_name():
    """A typo in output_params is silent — it just resolves nothing."""
    for s in STAGES:
        names = {p["name"] for p in s.get("params", [])}
        for key in s.get("output_params") or []:
            check(f'{s["id"]}.{key} exists', key in names, True)


def test_new_declarations():
    for sid, want in (("gain_convert", ["out_mrc"]),
                      ("gain_reciprocal", ["out_reciprocal"]),
                      ("ts_import", ["output"]),
                      ("aretomo", ["output_dir"])):
        check(f"{sid} declares", spec(sid).get("output_params"), want)


def test_declaring_gains_does_not_make_it_deletable():
    """gains/ is raw input for the whole project. Declaring an output inside it
    must not put it within reach of a job delete."""
    d = project(files=["gains/original_gain.mrc"])
    targets, skipped = job_delete_targets(
        d, "J4", "gain_convert", GAIN_JOB["params"],
        output_params=spec("gain_convert")["output_params"])
    check("gains not deletable", [t for t in targets if t.startswith("gains")], [])
    check("gains is protected", "gains" in PROTECTED_DIRS, True)


def test_failed_is_rerunnable_completed_is_not():
    check("failed re-runs", "failed" in RERUNNABLE, True)
    check("building re-runs", "building" in RERUNNABLE, True)
    check("queued re-runs", "queued" in RERUNNABLE, True)
    # Re-running into a completed job would erase the record of a success.
    check("completed forks", "completed" in RERUNNABLE, False)
    # A running job's status is owned by its process.
    check("running excluded", "running" in RERUNNABLE, False)


def test_stage_outputs_paths_are_relative():
    """A fallback row is opened as project-relative; an absolute one would
    escape the project."""
    for sid, rel in STAGE_OUTPUTS.items():
        check(f"{sid} relative", os.path.isabs(str(rel)), False)


for fn in sorted(k for k in dict(globals()) if k.startswith("test_")):
    globals()[fn]()
print(f"{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
