"""A run that exits 0 can still have thrown work away.

J7 (fs_motion_and_ctf, 4145 movies) exited 0 after WarpTools deselected
Position117_043 — its CTF fit diverged into a .NET OverflowException. One line,
an hour into the log, and the card went green. The tilt is simply absent from
the series that needed it, and nothing said so.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stub"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tomogration_jobs import (run_warning_key, run_warning_report,  # noqa: E402
                              run_failure_hint, RUN_WARNING_SIGNATURES)

passed = failed = 0


def check(name, got, want):
    global passed, failed
    if got == want:
        passed += 1
    else:
        failed += 1
        print(f"FAIL {name}: got {got!r} want {want!r}")


WARP_LINE = ("Failed to process /ceph/users/haq21239/EMDatasets/EML50/"
             "manually-selected/jobs/J7_fs-motion-and-ctf/"
             "Position117_043_-52.50_20260821_194051_EER.eer, "
             "marked as unselected")


def test_matches_the_real_line():
    check("warp deselect line", run_warning_key(WARP_LINE), "deselected")


def test_case_insensitive():
    check("upper", run_warning_key(WARP_LINE.upper()), "deselected")


def test_ordinary_lines_do_not_match():
    for line in ("4145 files found", "Finished processing in 01:01:20", "Done",
                 "Connected to 8 workers", "", None):
        check(f"quiet {line!r}", run_warning_key(line), None)


def test_counts_are_the_finding():
    """One deselected movie and four hundred are different findings."""
    check("one", run_warning_report({"deselected": 1})[0][1], 1)
    check("many", run_warning_report({"deselected": 412})[0][1], 412)
    check("one reads singular-safe",
          "1 item(s) deselected by WarpTools"
          in run_warning_report({"deselected": 1})[0][2], True)


def test_zero_is_not_reported():
    """'0 items deselected' is noise on a card that succeeded cleanly."""
    for counts in ({}, None, {"deselected": 0}, {"deselected": -1},
                   {"deselected": ""}, {"unknown_key": 9}):
        check(f"silent {counts!r}", run_warning_report(counts), [])


def test_report_shape():
    key, n, line, why = run_warning_report({"deselected": 3})[0]
    check("key", key, "deselected")
    check("count", n, 3)
    check("line has the number", line.startswith("3 "), True)
    check("explanation is actionable", "change_selection" in why, True)


def test_a_warning_is_not_a_failure():
    """The run succeeded. A warning must not be dressed up as a crash — and the
    failure hints must not swallow the deselect line either."""
    check("no failure hint", run_failure_hint(WARP_LINE), "")


def test_signature_table_is_well_formed():
    for w in RUN_WARNING_SIGNATURES:
        for field in ("sig", "key", "label", "why"):
            check(f"{w.get('key')}.{field}", bool(w.get(field)), True)
        # Matching lowercases the line, so an upper-case signature never fires.
        check(f"{w['key']} sig is lowercase", w["sig"], w["sig"].lower())


for fn in sorted(k for k in dict(globals()) if k.startswith("test_")):
    globals()[fn]()
print(f"{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
