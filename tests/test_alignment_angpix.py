"""The pixel size AreTomo's .xf shifts are in must never be guessed.

It was hardcoded 1.57 -- EML46's number -- in the miss-alignment prepend, with
a literal "1.57" fallback either side of the mdoc lookup. EML50 is 1.98, so
every imported shift would land 21% short. Nothing errors: ts_import_alignments
succeeds, the run completes, and the tomograms are quietly misaligned.

So: the AreTomo JOB's own angpix first (ts_stack can bin the stacks, and the
shifts are then in those pixels), the mdoc second, and "" -- refuse -- third.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stub"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tomogration_jobs import (alignment_pixel_size, latest_stage_job,  # noqa: E402
                              latest_stage_output_dir)

passed = failed = 0


def check(name, got, want):
    global passed, failed
    if got == want:
        passed += 1
    else:
        failed += 1
        print(f"FAIL {name}: got {got!r} want {want!r}")


def store(*jobs):
    return {"jobs": {j["id"]: j for j in jobs}}


def aretomo(jid="J13", status="failed", angpix="1.98"):
    return {"id": jid, "stage_id": "aretomo", "status": status,
            "output_dir": f"jobs/{jid}_aretomo_output",
            "params": {"angpix": angpix}}


def test_the_aretomo_job_wins():
    """ts_stack can bin; the mdoc's unbinned value would then be wrong."""
    check("binned stacks", alignment_pixel_size(store(aretomo(angpix="3.96")),
                                                "1.98"), "3.96")


def test_mdoc_is_the_fallback():
    """An alignment folder made outside tomogration has no job record."""
    check("mdoc used", alignment_pixel_size({}, "1.98"), "1.98")
    check("EML50 not EML46", alignment_pixel_size({}, "1.98") == "1.57", False)


def test_it_refuses_rather_than_guessing():
    for label, st, md in (("nothing at all", {}, ""),
                          ("junk mdoc", {}, "abc"),
                          ("zero", {}, "0"),
                          ("negative", {}, "-1.98"),
                          ("empty job angpix", store(aretomo(angpix="")), "")):
        check(f"refuses ({label})", alignment_pixel_size(st, md), "")


def test_a_failed_aretomo_job_still_counts():
    """J13 exited 2 with 76 of 81 aligned -- those .xf files are real, and its
    angpix is the right unit for them."""
    check("failed job used", alignment_pixel_size(store(aretomo(status="failed")),
                                                  ""), "1.98")


def test_a_queued_job_does_not_count():
    """A job that has not run has produced no shifts to describe."""
    check("queued ignored",
          alignment_pixel_size(store(aretomo(status="queued")), ""), "")


def test_newest_job_wins():
    check("J20 over J13",
          alignment_pixel_size(store(aretomo("J13", angpix="1.98"),
                                     aretomo("J20", angpix="2.64")), ""), "2.64")
    # String ordering would pick J9.
    check("numeric not lexical",
          alignment_pixel_size(store(aretomo("J9", angpix="1.0"),
                                     aretomo("J10", angpix="2.0")), ""), "2.0")


def test_latest_stage_job_shape():
    check("returns the record",
          latest_stage_job(store(aretomo()), "aretomo",
                           ("completed", "failed"))["id"], "J13")
    check("empty when none", latest_stage_job({}, "aretomo"), {})
    check("output_dir helper still works",
          latest_stage_output_dir(store(aretomo(status="completed")), "aretomo"),
          "jobs/J13_aretomo_output")


def test_no_silent_1_57_remains_in_the_app():
    """The literal that caused this must not creep back as a fallback."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "tomogration_app.py")).read()
    live = [ln for ln in src.splitlines()
            if '"1.57"' in ln and not ln.strip().startswith("#")]
    check("no literal 1.57 fallback", live, [])


for fn in sorted(k for k in dict(globals()) if k.startswith("test_")):
    globals()[fn]()
print(f"{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
