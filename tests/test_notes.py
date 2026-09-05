"""A note that stores its text and paints an empty box is a lost note.

The canvas drew annotations with QGraphicsTextItem + setTextWidth. The text
round-tripped through the store perfectly -- reopening the editor showed it --
and the canvas painted a bare dashed rectangle. Every card label on the same
scene is a QGraphicsSimpleTextItem and every one of them renders, so notes are
drawn that way too. Simple items do not wrap, so the wrapping is done here,
in a pure function that can be checked without a display.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stub"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tomogration_jobs import (wrap_lines, add_note, update_note,  # noqa: E402
                              load_notes, delete_note, notes_for_canvas,
                              load_jobs)

passed = failed = 0
W = 6.0                                   # a fixed-width "font", 6px per char


def measure(s):
    return len(s) * W


def check(name, got, want):
    global passed, failed
    if got == want:
        passed += 1
    else:
        failed += 1
        print(f"FAIL {name}: got {got!r} want {want!r}")


NOTE = "Here is where I sorted the files to frames, mdocs, and gains"


def test_wraps_to_the_note_width():
    lines = wrap_lines(NOTE, 206, measure)
    check("every line fits", [l for l in lines if measure(l) > 206], [])
    check("nothing dropped", " ".join(lines), NOTE)
    check("more than one line", len(lines) > 1, True)


def test_nothing_to_draw():
    for empty in ("", "   ", "\n", None):
        check(f"empty {empty!r}", wrap_lines(empty, 206, measure), [])


def test_a_single_word_is_broken_not_overflowed():
    """A pasted path is one word and would run off the note."""
    path = "/ceph/users/haq21239/EMDatasets/EML50/manually-selected"
    lines = wrap_lines(path, 60, measure)
    check("all fit", [l for l in lines if measure(l) > 60], [])
    check("nothing lost", "".join(lines), path)


def test_explicit_newlines_are_kept():
    check("paragraphs", wrap_lines("one\n\ntwo", 206, measure), ["one", "", "two"])


def test_clipping_marks_itself():
    """A clipped note must look clipped, not merely short."""
    lines = wrap_lines(NOTE, 206, measure, max_lines=1)
    check("one line", len(lines), 1)
    check("ellipsis", lines[0].endswith("…"), True)
    check("still fits", measure(lines[0]) <= 206, True)


def test_no_ellipsis_when_it_all_fits():
    lines = wrap_lines("short", 206, measure, max_lines=4)
    check("no ellipsis", lines, ["short"])


def test_max_lines_exactly_at_the_limit():
    full = wrap_lines(NOTE, 206, measure)
    check("no clipping at exactly N", wrap_lines(NOTE, 206, measure,
                                                 max_lines=len(full)), full)


def test_a_narrow_note_still_terminates():
    """A width too small for even one character must not loop forever."""
    check("narrow", len(wrap_lines("abc", 1.0, measure)) >= 1, True)


def test_store_round_trip():
    """The half that always worked -- pinned so a paint fix cannot mask a
    storage regression later."""
    d = tempfile.mkdtemp()
    n = add_note(d, "note", 10, 20, text="first")
    check("stored", load_notes(d)[0]["text"], "first")
    update_note(d, n["id"], text=NOTE)
    check("updated", load_notes(d)[0]["text"], NOTE)
    check("survives reload", load_jobs(d)["notes"][0]["text"], NOTE)
    frames, stickies = notes_for_canvas(load_jobs(d))
    check("painted as a sticky", [s["text"] for s in stickies], [NOTE])
    check("not a frame", frames, [])
    delete_note(d, n["id"])
    check("deleted", load_notes(d), [])


def test_unknown_keys_are_not_stored():
    d = tempfile.mkdtemp()
    n = add_note(d, "note", 0, 0, text="t")
    update_note(d, n["id"], txt="typo", text="real")
    check("typo ignored", "txt" in load_notes(d)[0], False)
    check("real text kept", load_notes(d)[0]["text"], "real")


for fn in sorted(k for k in dict(globals()) if k.startswith("test_")):
    globals()[fn]()
print(f"{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
