"""76 of 81 aligned is a result with casualties, not a failure.

ml_aretomo2_warp_auto exits 2 when SOME series failed, and the finalizer read
any non-zero exit as failure -- so a card went red over 76 usable alignments
and everything queued behind it was dequeued.

Only a stage that DECLARES partial_exit gets this reading: `exit 2` means
"config missing, edit it and run again" in the miss-alignment wrapper, which
really is a hard failure and must stay one.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stub"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tomogration_jobs import batch_tally_line, partial_batch_result  # noqa: E402
from tomogration_stages import STAGES  # noqa: E402

passed = failed = 0


def check(name, got, want):
    global passed, failed
    if got == want:
        passed += 1
    else:
        failed += 1
        print(f"FAIL {name}: got {got!r} want {want!r}")


ARETOMO = next(s for s in STAGES if s["id"] == "aretomo")
# The exact block ml_aretomo2_warp_auto printed for J13.
J13 = ["========================================", "Summary:",
       "  Processed successfully: 76", "  Skipped:                0",
       "  Failed:                 5", "Finished: Fri 4 Sep 13:44:08 BST 2026"]


def tally(lines):
    out = {}
    for line in lines:
        hit = batch_tally_line(line)
        if hit:
            out[hit[0]] = hit[1]
    return out


def test_reads_the_real_summary_block():
    check("J13 tally", tally(J13), {"ok": 76, "skipped": 0, "failed": 5})


def test_ignores_everything_else():
    for line in ("[GPU 0] OK Position001", "Found 81 tilt series to process",
                 "ERROR: AreTomo2 failed for Position005", "", "Failed: many"):
        check(f"not a tally: {line[:24]!r}", batch_tally_line(line), None)


def test_j13_is_completed_not_failed():
    status, note = partial_batch_result(ARETOMO, 2, tally(J13))
    check("status", status, "completed")
    check("counts named", "76 of 81 succeeded, 5 failed" in note, True)


def test_a_total_wipeout_stays_failed():
    """Nothing succeeded means no result to carry forward, whatever the code."""
    status, note = partial_batch_result(ARETOMO, 2, {"ok": 0, "failed": 81})
    check("status", status, "failed")
    check("says so", "every item failed" in note, True)


def test_only_declared_stages_get_the_reading():
    """miss_align exits 2 for a missing config -- a hard failure."""
    for sid in ("miss_align", "miss_align_infer", "ts_stack", "relion4_convert"):
        spec = next(s for s in STAGES if s["id"] == sid)
        check(f"{sid} undeclared", spec.get("partial_exit"), None)
        check(f"{sid} unaffected",
              partial_batch_result(spec, 2, {"ok": 5, "failed": 1}), (None, ""))


def test_other_exit_codes_are_untouched():
    for code in (0, 1, 6, 137):
        check(f"exit {code}", partial_batch_result(ARETOMO, code, tally(J13)),
              (None, ""))


def test_a_clean_run_never_looks_partial():
    """exit 0 with a tally of zero failures is just success."""
    check("clean", partial_batch_result(ARETOMO, 0,
                                        {"ok": 81, "failed": 0}), (None, ""))


def test_missing_tally_is_survivable():
    """The summary block may scroll past unparsed; do not crash or invent."""
    check("no tally", partial_batch_result(ARETOMO, 2, {}), ("failed",
          "every item failed (0 of 0)"))
    check("None tally", partial_batch_result(ARETOMO, 2, None)[0], "failed")
    check("no spec", partial_batch_result(None, 2, {"ok": 1}), (None, ""))


def test_skipped_counts_toward_the_total():
    status, note = partial_batch_result(
        ARETOMO, 2, {"ok": 70, "skipped": 6, "failed": 5})
    check("status", status, "completed")
    check("total includes skipped", "70 of 81 succeeded" in note, True)


for fn in sorted(k for k in dict(globals()) if k.startswith("test_")):
    globals()[fn]()
print(f"{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
