"""ts_import_alignments belongs BEFORE miss-alignment, never after.

It is AreTomo's hand-off: it reads .xf/.tlt and writes them into the Warp XMLs.
miss-alignment REFINES what is already in those XMLs and writes its result
straight back -- it produces no .xf at all. Running the import afterwards:

  1. replaces the refined geometry with the coarse alignment it started from,
  2. fails on every series, and a failed import marks each UNSELECTED, which
     Warp PERSISTS -- ts_ctf and ts_reconstruct then process nothing and
     report success.

This happened on EML45 (2026-06-29) and cost a full reprocessing round. The
pipeline list made it worse by showing 'Import alignments' AFTER both refine
steps, so following the list top-to-bottom was the wrong order.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stub"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tomogration_jobs import refined_since_last_import  # noqa: E402
from tomogration_jobs import io_flags_for_job  # noqa: E402
from tomogration_stages import STAGES, TRUNK_STAGES  # noqa: E402

passed = failed = 0


def check(name, got, want):
    global passed, failed
    if got == want:
        passed += 1
    else:
        failed += 1
        print(f"FAIL {name}: got {got!r} want {want!r}")


def job(jid, stage_id, status="completed"):
    return {"id": jid, "stage_id": stage_id, "status": status,
            "output_dir": f"jobs/{jid}"}


def store(*jobs):
    return {"jobs": {j["id"]: j for j in jobs}}


def test_the_list_shows_the_order_you_should_follow():
    """Reading the sidebar top-to-bottom must not walk you into the trap."""
    order = [s["id"] for s in STAGES if s["group"].startswith("5.")]
    check("import before both refiners",
          order.index("ts_import_alignments") < order.index("miss_align")
          and order.index("ts_import_alignments") < order.index("miss_align_infer"),
          True)
    check("aretomo first", order.index("aretomo"), 0)
    check("select between import and refine",
          order.index("sync_selection") > order.index("ts_import_alignments"), True)


def test_flags_a_refinement_newer_than_the_import():
    check("train", refined_since_last_import(
        store(job("J1", "ts_import_alignments"), job("J2", "miss_align"))), "J2")
    check("infer", refined_since_last_import(
        store(job("J1", "ts_import_alignments"),
              job("J2", "miss_align_infer"))), "J2")


def test_the_correct_order_is_not_flagged():
    check("import last", refined_since_last_import(
        store(job("J1", "miss_align"), job("J2", "ts_import_alignments"))), "")
    check("import only", refined_since_last_import(
        store(job("J1", "ts_import_alignments"))), "")
    check("nothing yet", refined_since_last_import({}), "")
    check("no store", refined_since_last_import(None), "")


def test_only_completed_runs_count():
    """A queued or failed refinement wrote nothing into the XMLs."""
    for status in ("queued", "failed", "running", "building"):
        check(f"{status} refinement ignored", refined_since_last_import(
            store(job("J1", "ts_import_alignments"),
                  job("J2", "miss_align", status))), "")


def test_job_numbers_compare_numerically():
    """J10 is newer than J9; string ordering says otherwise."""
    check("J10 refine after J9 import", refined_since_last_import(
        store(job("J9", "ts_import_alignments"), job("J10", "miss_align"))), "J10")
    check("J10 import after J9 refine", refined_since_last_import(
        store(job("J9", "miss_align"), job("J10", "ts_import_alignments"))), "")


def test_a_re_import_clears_the_flag():
    """AreTomo -> import -> refine -> import -> refine is legitimate: each
    import is answered by a later refinement."""
    check("newest refine wins", refined_since_last_import(
        store(job("J1", "ts_import_alignments"), job("J2", "miss_align"),
              job("J3", "ts_import_alignments"), job("J4", "miss_align"))), "J4")
    check("newest import wins", refined_since_last_import(
        store(job("J1", "ts_import_alignments"), job("J2", "miss_align"),
              job("J3", "ts_import_alignments"))), "")


def test_the_import_is_not_forced_onto_the_trunk():
    """It was, briefly, on a bad measurement.

    J18 failed 81 of 81 as a job and 5 of 81 outside one, and I read that as
    the --output_processing scoping misdirecting it. It was not: the two runs
    also used different --alignments paths, and the corrected path worked WITH
    the processing flags, writing into warp_tiltseries/ as it should. The
    confound was mine, so the change is gone.
    """
    check("import is job-scopable", "ts_import_alignments" in TRUNK_STAGES, False)
    check("selection is job-scopable", "sync_selection" in TRUNK_STAGES, False)
    spec = next(s for s in STAGES if s["id"] == "ts_import_alignments")
    job = {"id": "J18", "stage_id": "ts_import_alignments",
           "output_dir": "jobs/J18_ts-import-alignments", "inputs": {}}
    check("flags restored",
          "--output_processing jobs/J18_ts-import-alignments"
          in io_flags_for_job(spec, job, {}), True)


def test_template_match_is_still_trunk_only():
    """The one stage that genuinely must not be scoped: it reads the SHARED
    reconstructions and keeps its own output distinct by suffix."""
    check("still trunk", TRUNK_STAGES, {"ts_template_match"})


def test_the_pitfall_says_so():
    doc = next(s for s in STAGES
               if s["id"] == "ts_import_alignments")["docs"]["pitfall"]
    check("names the order", "BEFORE miss-alignment" in doc, True)
    check("names the deselect", "DESELECT" in doc, True)
    check("no stale pixel size", "1.57" in doc, False)


for fn in sorted(k for k in dict(globals()) if k.startswith("test_")):
    globals()[fn]()
print(f"{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
