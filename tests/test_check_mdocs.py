"""ts_import names its casualties once, in passing, and moves on.

"81/83, 2 failed" with one line per failure buried in thousands ("Failed to
parse Position017.mdoc: Value cannot be null. (Parameter 'key')") is not a
list of what to fix. ml_check_mdocs finds them without running Warp.

"Value cannot be null (Parameter 'key')" is a dictionary being handed a null
KEY, so the checks hunt exactly that: an empty key on a line, a section header
with no name, and a block with no SubFramePath (Warp keys its frame-series
lookup on that basename).
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stub"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ml_check_mdocs as m  # noqa: E402

passed = failed = 0


def check(name, got, want):
    global passed, failed
    if got == want:
        passed += 1
    else:
        failed += 1
        print(f"FAIL {name}: got {got!r} want {want!r}")


def mdoc(pos="Position017", n=6, **broken):
    head = (f"PixelSpacing = 1.98\nVoltage = 300.00\nImageFile = {pos}.mrc\n\n"
            "[T = SerialEM: Titan]\n\n")
    if broken.get("empty_header_key"):
        head += " = orphaned\n\n"
    if broken.get("keyless_section"):
        head += "[ = nothing]\n\n"
    out = []
    for i in range(n):
        ang = -60.0 + i * 2.5
        b = [f"[ZValue = {i}]", f"TiltAngle = {ang:.2f}", "Magnification = 64000"]
        if broken.get("drop_subframe") != i:
            b.append(f"SubFramePath = \\\\srv\\{pos}_{i + 1:03d}_{ang:.2f}_EER.eer")
        if broken.get("drop_tilt") == i:
            b = [x for x in b if not x.startswith("TiltAngle")]
        if broken.get("empty_key") == i:
            b.append(" = stray")
        out.append("\n".join(b) + "\n\n")
    txt = head + "".join(out)
    if broken.get("gap"):
        txt = txt.replace("[ZValue = 3]", "[ZValue = 9]")
    if broken.get("dup"):
        txt = txt.replace("[ZValue = 4]", "[ZValue = 2]")
    if broken.get("noninteger"):
        txt = txt.replace("[ZValue = 2]", "[ZValue = two]")
    if broken.get("no_blocks"):
        txt = head
    return txt


def write(txt, name="Position017.mdoc"):
    d = tempfile.mkdtemp()
    p = os.path.join(d, name)
    with open(p, "w") as fh:
        fh.write(txt)
    return p


def first(problems):
    return problems[0] if problems else ""


def test_a_clean_mdoc_is_clean():
    check("clean", m.check(write(mdoc())), [])


def test_null_key_shapes():
    """The three ways a parser gets handed an empty key."""
    check("empty key in a block",
          "empty key" in first(m.check(write(mdoc(empty_key=1)))), True)
    check("keyless section",
          "no key" in first(m.check(write(mdoc(keyless_section=True)))), True)
    check("empty key in header",
          "empty key" in first(m.check(write(mdoc(empty_header_key=True)))), True)


def test_missing_subframepath():
    """Warp keys the frame-series lookup on this basename."""
    check("no SubFramePath",
          "no SubFramePath" in first(m.check(write(mdoc(drop_subframe=2)))), True)


def test_missing_tiltangle():
    check("no TiltAngle",
          any("no TiltAngle" in p for p in m.check(write(mdoc(drop_tilt=0)))), True)


def test_numbering():
    check("gap", "not 0..5" in first(m.check(write(mdoc(gap=True)))), True)
    check("duplicate", "duplicate" in first(m.check(write(mdoc(dup=True)))), True)
    check("non-integer",
          "not an integer" in first(m.check(write(mdoc(noninteger=True)))), True)


def test_no_blocks():
    check("no blocks", "no [ZValue] blocks at all"
          in m.check(write(mdoc(no_blocks=True))), True)


def test_frames_cross_check():
    """A tilt whose .eer was quarantined leaves a block pointing at nothing."""
    p = write(mdoc())
    frames = tempfile.mkdtemp()
    probs = m.check(p, frames)
    check("all frames missing", len([x for x in probs if "frame not in" in x]), 6)
    check("no frames arg, no complaint",
          [x for x in m.check(p) if "frame not in" in x], [])


def test_unreadable_file_is_reported_not_raised():
    check("missing file", "unreadable" in first(m.check("/nonexistent/x.mdoc")),
          True)


def test_zvalue_parser():
    check("plain", m._zvalue("[ZValue = 7]"), 7)
    check("spacey", m._zvalue("  [ZValue=12]  "), 12)
    check("not a number", m._zvalue("[ZValue = two]"), None)
    check("no equals", m._zvalue("[ZValue]"), None)


def test_override_mdocs_are_skipped_by_main():
    """Tomo5 *_override.mdoc are not acquisition records; ts_import ignores
    them and so must the checker, or every project reports fake failures."""
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "Position001.mdoc"), "w") as fh:
        fh.write(mdoc("Position001"))
    with open(os.path.join(d, "Position001_override.mdoc"), "w") as fh:
        fh.write("[ = junk]\n")
    check("override ignored", m.main([d]), 0)


def test_exit_codes():
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "P.mdoc"), "w") as fh:
        fh.write(mdoc(empty_key=1))
    check("problems -> 1", m.main([d]), 1)
    check("no folder -> 2", m.main(["/nonexistent"]), 2)
    check("empty folder -> 2", m.main([tempfile.mkdtemp()]), 2)


for fn in sorted(k for k in dict(globals()) if k.startswith("test_")):
    globals()[fn]()
print(f"{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
