"""Where did the imported alignment actually land?

A --output_processing run writes into that job folder. For a step that EDITS
the shared per-series record rather than producing new data from it, that is
the difference between working and silently doing nothing: miss-alignment,
ts_ctf and ts_reconstruct all read warp_tiltseries/, so an alignment sitting in
jobs/J18_.../ is invisible to them and nothing reports it.

VolumeDimensionsAngstrom is the tell -- [0,0,0] until a series has a geometry,
which is what importing an alignment gives it. The miss-alignment wrapper
already relies on this to park un-alignable series.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stub"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ml_check_xml_alignment as m  # noqa: E402
from tomogration_project import ProjectState  # noqa: E402

passed = failed = 0


def check(name, got, want):
    global passed, failed
    if got == want:
        passed += 1
    else:
        failed += 1
        print(f"FAIL {name}: got {got!r} want {want!r}")


def xml(dims):
    if dims is None:
        return "<TiltSeries><Tilts>0</Tilts></TiltSeries>"
    return ("<TiltSeries><VolumeDimensionsAngstrom>"
            f"{dims[0]}, {dims[1]}, {dims[2]}"
            "</VolumeDimensionsAngstrom></TiltSeries>")


def folder(series):
    """series: {name: dims-or-None}"""
    d = tempfile.mkdtemp()
    for name, dims in series.items():
        with open(os.path.join(d, name), "w") as fh:
            fh.write(xml(dims))
    return d


ALIGNED = (8110.0, 8110.0, 6114.0)


def test_aligned_and_unaligned():
    d = folder({"Position001.xml": ALIGNED, "Position005.xml": (0, 0, 0)})
    check("aligned", m.aligned(os.path.join(d, "Position001.xml"))[0], True)
    check("zeros", m.aligned(os.path.join(d, "Position005.xml"))[0], False)


def test_dims_are_reported():
    d = folder({"P.xml": ALIGNED})
    check("dims", m.aligned(os.path.join(d, "P.xml"))[1], ALIGNED)


def test_a_missing_field_is_not_an_alignment():
    d = folder({"P.xml": None})
    check("no field", m.aligned(os.path.join(d, "P.xml")), (False, None))


def test_unreadable_file():
    check("missing file", m.aligned("/nonexistent/P.xml"), (False, None))


def test_negative_dimensions_still_count_as_written():
    """Only all-zero means 'never imported'; a negative is a different bug and
    must not be reported as an absent alignment."""
    d = folder({"P.xml": (-1, 8110, 6114)})
    check("negative", m.aligned(os.path.join(d, "P.xml"))[0], True)


def test_scan_counts():
    d = folder({"a.xml": ALIGNED, "b.xml": ALIGNED, "c.xml": (0, 0, 0)})
    n_ok, n_all, example = m.scan(d)
    check("aligned count", n_ok, 2)
    check("total", n_all, 3)
    check("example named", example[0], "a.xml")


def test_scan_ignores_non_xml():
    d = folder({"a.xml": ALIGNED})
    open(os.path.join(d, "notes.txt"), "w").close()
    check("only xml counted", m.scan(d)[1], 1)


def test_scan_of_a_missing_folder():
    check("missing dir", m.scan("/nonexistent"), (None, None, None))


def test_the_case_this_exists_for():
    """Trunk empty, job folder full -- the alignment went to the wrong place."""
    trunk = folder({f"P{i}.xml": (0, 0, 0) for i in range(3)})
    job = folder({f"P{i}.xml": ALIGNED for i in range(3)})
    check("flagged", m.main([trunk, job]), 1)


def test_both_populated_is_fine():
    trunk = folder({"P.xml": ALIGNED})
    check("ok", m.main([trunk]), 0)


def test_nothing_to_read():
    check("no args", m.main([]), 2)
    check("bad dir", m.main(["/nonexistent"]), 2)


# ---------------------------------------------------------------------------
# The same question, asked by the miss-alignment gate before it lets you run.
# It used to ask "does a .xf exist somewhere?", which is wrong both ways: a .xf
# that was never IMPORTED leaves the XML untouched (J17 refined nothing with a
# full aretomo_output folder sitting right there), and an alignment that arrived
# another way has no .xf to find.
# ---------------------------------------------------------------------------
def project(states):
    """states: {series: dims | None}. None = no XML written at all."""
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "tomostar"))
    os.makedirs(os.path.join(d, "warp_tiltseries"))
    for name, dims in states.items():
        open(os.path.join(d, "tomostar", name + ".tomostar"), "w").close()
        if dims is not None:
            with open(os.path.join(d, "warp_tiltseries", name + ".xml"), "w") as fh:
                fh.write(xml(dims))
    return ProjectState(d)


def test_gate_sees_the_eml50_shape():
    """76 imported, 5 AreTomo casualties."""
    states = {f"P{i:03d}": ALIGNED for i in range(76)}
    states.update({f"Q{i:03d}": (0, 0, 0) for i in range(5)})
    check("only the 5", sorted(project(states).series_without_imported_alignment()),
          [f"Q{i:03d}" for i in range(5)])


def test_gate_catches_the_j17_case():
    """AreTomo ran, .xf files exist, nothing was imported."""
    check("all flagged", len(project({f"P{i}": (0, 0, 0) for i in range(3)}
                                     ).series_without_imported_alignment()), 3)


def test_a_missing_xml_counts_as_unaligned():
    """'Not proven' is the honest answer when the record cannot be read."""
    check("no xml", project({"P1": None}).series_without_imported_alignment(), ["P1"])


def test_a_fully_imported_project_is_silent():
    check("no warning", project({"P1": ALIGNED, "P2": ALIGNED}
                                ).series_without_imported_alignment(), [])


def test_no_tomostars_means_nothing_to_judge():
    check("empty", project({}).series_without_imported_alignment(), [])
    check("no dirs", ProjectState(tempfile.mkdtemp()
                                  ).series_without_imported_alignment(), [])


def test_it_does_not_depend_on_where_aretomo_wrote():
    """The whole point: an alignment imported by hand, from any folder, counts."""
    p = project({"P1": ALIGNED})
    check("no aretomo folder needed", p.series_without_imported_alignment(), [])


for fn in sorted(k for k in dict(globals()) if k.startswith("test_")):
    globals()[fn]()
print(f"{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
