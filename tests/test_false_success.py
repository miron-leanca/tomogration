"""Exit 0 is a claim, not evidence.

WarpTools 2.0.0 answers an unrecognised option by printing its full help and
exiting ZERO. J8 (create_settings) and J9 (ts_import) were handed
--input_processing/--output_processing, which those two subcommands do not
take. Both were recorded "completed (exit 0)" having written no
warp_tiltseries.settings and no tomostar/, and the pipeline marched on until
J12 tripped over the absence an hour later.

Two defects, both covered here: the flags must not be sent at all, and a run
that provably did nothing must be FAILED whatever the shell says.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stub"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tomogration_jobs import (false_success_reason, takes_processing_flags,  # noqa: E402
                              is_warp_stage, io_flags_for_job,
                              build_job_command, default_parent_for,
                              derive_child_params, latest_stage_output_dir)
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


# The three that rejected the flags on the VM, 2026-09-03.
REJECTERS = ("create_settings_fs", "create_settings_ts", "ts_import")
# Verified accepting them: J7 and J12 echoed input_processing/output_processing
# back in their options list.
ACCEPTERS = ("fs_motion_and_ctf", "ts_stack", "ts_ctf", "ts_reconstruct")


def test_rejecters_get_no_flags():
    store = {"jobs": {"J7": {"id": "J7", "stage_id": "fs_motion_and_ctf",
                             "output_dir": "jobs/J7_fs-motion-and-ctf"}}}
    for sid in REJECTERS:
        job = {"id": "J8", "stage_id": sid, "output_dir": "jobs/J8_x",
               "inputs": {"processing": "J7"}, "params": {}}
        check(f"{sid} predicate", takes_processing_flags(spec(sid)), False)
        check(f"{sid} no flags", io_flags_for_job(spec(sid), job, store), "")
        cmd = build_job_command(spec(sid), job, store)
        check(f"{sid} cmd clean", "_processing" in cmd, False)


def test_accepters_still_get_flags():
    """The per-job isolation this whole layer rests on must not regress."""
    store = {"jobs": {"J7": {"id": "J7", "stage_id": "fs_motion_and_ctf",
                             "output_dir": "jobs/J7_fs-motion-and-ctf"}}}
    for sid in ACCEPTERS:
        job = {"id": "J20", "stage_id": sid, "output_dir": "jobs/J20_x",
               "inputs": {"processing": "J7"}, "params": {}}
        check(f"{sid} predicate", takes_processing_flags(spec(sid)), True)
        flags = io_flags_for_job(spec(sid), job, store)
        check(f"{sid} writes own dir",
              "--output_processing jobs/J20_x" in flags, True)
        check(f"{sid} reads parent",
              "--input_processing jobs/J7_fs-motion-and-ctf" in flags, True)


def test_predicate_tracks_the_settings_flag():
    """Checked, not hand-listed, so a stage added later cannot silently rejoin
    the broken set."""
    for s in STAGES:
        if not is_warp_stage(s):
            check(f'{s["id"]} non-warp', takes_processing_flags(s), False)
            continue
        has = any(p.get("flag") == "--settings" or p.get("name") == "settings"
                  for p in s.get("params", []))
        check(f'{s["id"]} matches --settings', takes_processing_flags(s), has)


def test_no_downstream_job_reads_a_settings_job():
    """J9 was wired to read J8's processing dir. J8 has none."""
    store = {"jobs": {"J8": {"id": "J8", "stage_id": "create_settings_ts",
                             "output_dir": "jobs/J8_create-settings"}}}
    check("ts_import ignores J8", default_parent_for("ts_import", store), None)


def test_ts_import_is_aimed_at_its_parent():
    """Losing --input_processing means --frameseries must be aimed by hand;
    J7's averages went to its job dir and J9 read the empty trunk."""
    check("parent dir",
          derive_child_params("ts_import", "fs_motion_and_ctf", {},
                              "jobs/J7_fs-motion-and-ctf"),
          {"frameseries": "jobs/J7_fs-motion-and-ctf"})
    check("trunk parent",
          derive_child_params("ts_import", "fs_motion_and_ctf", {}, ""),
          {"frameseries": "warp_frameseries"})


EML50 = {"jobs": {
    "J7": {"id": "J7", "stage_id": "fs_motion_and_ctf", "status": "completed",
           "output_dir": "jobs/J7_fs-motion-and-ctf"},
    "J8": {"id": "J8", "stage_id": "create_settings_ts", "status": "completed",
           "output_dir": "jobs/J8_create-settings"},
}}


def test_latest_stage_output_dir():
    """The derive fires only when a job is CREATED; J9 already existed and kept
    warp_frameseries. So the same answer has to be available as a default."""
    check("finds J7", latest_stage_output_dir(EML50, "fs_motion_and_ctf"),
          "jobs/J7_fs-motion-and-ctf")
    check("newest wins", latest_stage_output_dir(
        {"jobs": dict(EML50["jobs"],
                      J20={"id": "J20", "stage_id": "fs_motion_and_ctf",
                           "status": "completed", "output_dir": "jobs/J20_x"})},
        "fs_motion_and_ctf"), "jobs/J20_x")
    # J10 vs J9: string ordering would pick J9.
    check("numeric order", latest_stage_output_dir(
        {"jobs": {"J9": {"id": "J9", "stage_id": "s", "status": "completed",
                         "output_dir": "a"},
                  "J10": {"id": "J10", "stage_id": "s", "status": "completed",
                          "output_dir": "b"}}}, "s"), "b")


def test_latest_stage_output_dir_declines_to_guess():
    """No usable upstream job means leave the trunk default alone."""
    for store, why in (({}, "empty store"), ({"jobs": {}}, "no jobs"),
                       (None, "no store"),
                       ({"jobs": {"J7": dict(EML50["jobs"]["J7"],
                                             status="failed")}}, "failed"),
                       ({"jobs": {"J7": dict(EML50["jobs"]["J7"],
                                             output_dir="")}}, "no dir")):
        check(f"declines ({why})",
              latest_stage_output_dir(store, "fs_motion_and_ctf"), "")


def test_the_real_warptools_line():
    for line in ("Option 'input_processing' is unknown.",
                 "Option 'output_processing' is unknown.",
                 "option 'foo' is unknown"):
        check(f"caught {line!r}", bool(false_success_reason(line)), True)


def test_ordinary_output_is_not_a_false_success():
    for line in ("4145 files found", "Running command ts_stack with:",
                 "input_processing = jobs/J9_ts-import",
                 "unknown", "Option is unknown", "", None):
        check(f"quiet {line!r}", false_success_reason(line), "")


def test_reason_names_the_cause():
    why = false_success_reason("Option 'output_processing' is unknown.")
    check("names create_settings", "create_settings" in why, True)
    check("says nothing ran", "exited 0" in why, True)


for fn in sorted(k for k in dict(globals()) if k.startswith("test_")):
    globals()[fn]()
print(f"{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
