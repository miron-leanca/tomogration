"""Answer 'did miss-alignment have anything to refine?' from the record.

miss-alignment refines the geometry already in the Warp XMLs. Run on XMLs that
never received AreTomo's alignment, it refines nothing and the tomograms come
out featureless -- with no error at any point. tomogration prepends
ts_import_alignments INSIDE the miss-alignment job's command, so there is no
separate card to look for; the evidence is the stored command line.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stub"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ml_check_align_order as m  # noqa: E402

passed = failed = 0


def check(name, got, want):
    global passed, failed
    if got == want:
        passed += 1
    else:
        failed += 1
        print(f"FAIL {name}: got {got!r} want {want!r}")


def job(jid, stage_id, cmd="", status="completed"):
    return {"id": jid, "stage_id": stage_id, "status": status, "command": cmd}


def project(*jobs):
    d = tempfile.mkdtemp()
    with open(os.path.join(d, ".tomogration_jobs.json"), "w") as fh:
        json.dump({"jobs": {j["id"]: j for j in jobs}}, fh)
    return d


IMPORT_CMD = "WarpTools ts_import_alignments --settings x && miss-alignment"


def test_prepended_import_counts():
    check("prepended", m.main([project(
        job("J13", "aretomo"), job("J14", "miss_align", IMPORT_CMD))]), 0)


def test_a_separate_import_job_counts():
    check("separate job", m.main([project(
        job("J13", "aretomo"), job("J14", "ts_import_alignments"),
        job("J15", "miss_align", "bash ma.sh"))]), 0)


def test_no_import_at_all_is_flagged():
    check("bare refine", m.main([project(
        job("J13", "aretomo"), job("J14", "miss_align", "bash ma.sh"))]), 1)


def test_infer_inherits_an_earlier_import_in_the_same_project():
    """The XMLs are shared, so an import prepended by an earlier refinement in
    this project still applies to a later one."""
    check("inherits", m.main([project(
        job("J14", "miss_align", IMPORT_CMD),
        job("J15", "miss_align_infer", "bash ma.sh infer"))]), 0)


def test_an_import_AFTER_the_refinement_does_not_count():
    """Order is the whole question -- a later import cannot have fed an
    earlier run."""
    check("too late", m.main([project(
        job("J14", "miss_align", "bash ma.sh"),
        job("J15", "ts_import_alignments"))]), 1)


def test_an_unfinished_import_does_not_count():
    for status in ("failed", "queued", "running", "building"):
        check(f"{status} import", m.main([project(
            job("J13", "ts_import_alignments", status=status),
            job("J14", "miss_align", "bash ma.sh"))]), 1)


def test_job_numbers_compare_numerically():
    """J9 import, J10 refine: string ordering would call J9 the later one."""
    check("J9 before J10", m.main([project(
        job("J9", "ts_import_alignments"),
        job("J10", "miss_align", "bash ma.sh"))]), 0)


def test_nothing_to_judge():
    check("no refiners", m.main([project(job("J1", "aretomo"))]), 0)
    check("no store", m.main([tempfile.mkdtemp()]), 2)
    check("no such dir", m.main(["/nonexistent"]), 2)


def test_missing_command_field_is_survivable():
    """Older records predate the stored command; absence is not an import."""
    d = tempfile.mkdtemp()
    with open(os.path.join(d, ".tomogration_jobs.json"), "w") as fh:
        json.dump({"jobs": {"J1": {"id": "J1", "stage_id": "miss_align",
                                   "status": "completed"}}}, fh)
    check("no command key", m.main([d]), 1)


for fn in sorted(k for k in dict(globals()) if k.startswith("test_")):
    globals()[fn]()
print(f"{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
