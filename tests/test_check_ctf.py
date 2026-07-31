"""test_check_ctf.py — ml_m_check_ctf.py must catch an unestimated tilt series.

This guards the root cause of the M IndexOutOfRangeException on EML45: two tilt
series out of 290 carried CTFResolutionEstimate="0" and crashed MPA refinement
without ever naming themselves. The checker is only worth having if it is
reliable, so the cases below mirror the real project exactly.
"""

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ml_m_check_ctf as chk  # noqa: E402


# The real Position001.xml, cut down to what the checker reads.
GOOD_XML = """<?xml version="1.0" encoding="utf-8"?>
<TiltSeries DataDirectory="/data/tomostar" AreAnglesInverted="False"
            ImageDimensionsAngstrom="6430.72, 6430.72"
            VolumeDimensionsAngstrom="6430.72, 6430.72, 4848.16"
            UnselectFilter="False" UnselectManual="False"
            CTFResolutionEstimate="{res}">
  <CTF>
    <Param Name="Defocus" Value="3.1" />
  </CTF>
  <GridCTFDefocus Width="1" Height="1" Depth="41">
    <Node X="0" Y="0" Z="0" Value="3.1" />
  </GridCTFDefocus>
</TiltSeries>
"""

# Position030.xml / Position240.xml: present, well-formed, imports fine — and no
# CTF was ever estimated for it.
BAD_XML = """<?xml version="1.0" encoding="utf-8"?>
<TiltSeries DataDirectory="/data/tomostar" AreAnglesInverted="False"
            ImageDimensionsAngstrom="6430.72, 6430.72"
            VolumeDimensionsAngstrom="6430.72, 6430.72, 4848.16"
            UnselectFilter="False" UnselectManual="False"
            CTFResolutionEstimate="0">
</TiltSeries>
"""

TOMOSTAR = """
data_

loop_
_wrpMovieName #1
_wrpAngleTilt #2
frames/a.eer 0.0
frames/b.eer 3.0
frames/c.eer -3.0
"""


def make_project(good=3, bad=0, tmp=None):
    """Build a project tree with `good` estimated and `bad` unestimated series."""
    root = Path(tmp)
    proc = root / "warp_tiltseries"
    tstar = root / "tomostar"
    proc.mkdir(parents=True)
    tstar.mkdir(parents=True)
    n = 0
    for i in range(good):
        n += 1
        name = f"Position{n:03d}"
        (proc / f"{name}.xml").write_text(GOOD_XML.format(res=4.5 + i * 0.1))
        (tstar / f"{name}.tomostar").write_text(TOMOSTAR)
    for _ in range(bad):
        n += 1
        name = f"Position{n:03d}"
        (proc / f"{name}.xml").write_text(BAD_XML)
        (tstar / f"{name}.tomostar").write_text(TOMOSTAR)
    return root


def run(root, *extra):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = chk.main([str(root), *extra])
    return code, buf.getvalue()


class TestCheckCtf(unittest.TestCase):

    def test_clean_project_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_project(good=5, bad=0, tmp=tmp)
            code, out = run(root)
        self.assertEqual(code, 0, out)
        self.assertIn("PASS", out)

    def test_single_unestimated_series_is_caught(self):
        """The EML45 case: a needle in an otherwise healthy project."""
        with tempfile.TemporaryDirectory() as tmp:
            root = make_project(good=288, bad=2, tmp=tmp)
            code, out = run(root)
        self.assertEqual(code, 1, out)
        self.assertIn("FAIL", out)
        self.assertIn("Position289", out)
        self.assertIn("Position290", out)
        # It must hand back a runnable fix, not just a complaint.
        self.assertIn("WarpTools ts_ctf", out)
        self.assertIn("tomostar/Position289.tomostar", out)

    def test_missing_attribute_counts_as_unestimated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_project(good=2, bad=0, tmp=tmp)
            (root / "warp_tiltseries" / "Position999.xml").write_text(
                '<?xml version="1.0"?><TiltSeries UnselectManual="False" />')
            code, out = run(root)
        self.assertEqual(code, 1, out)
        self.assertIn("Position999", out)
        self.assertIn("no CTFResolutionEstimate", out)

    def test_unparseable_xml_is_reported_not_raised(self):
        """A broken project must still produce a report."""
        with tempfile.TemporaryDirectory() as tmp:
            root = make_project(good=2, bad=0, tmp=tmp)
            (root / "warp_tiltseries" / "Position998.xml").write_text("<Tilt")
            code, out = run(root)
        self.assertEqual(code, 1, out)
        self.assertIn("Position998", out)
        self.assertIn("cannot parse", out)

    def test_write_list_is_usable_as_input_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_project(good=3, bad=2, tmp=tmp)
            listing = Path(tmp) / "bad.txt"
            code, out = run(root, "--write-list", str(listing))
            self.assertEqual(code, 1, out)
            lines = listing.read_text().split()
        self.assertEqual(lines,
                         ["tomostar/Position004.tomostar",
                          "tomostar/Position005.tomostar"])

    def test_reports_tilt_count_from_tomostar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_project(good=1, bad=1, tmp=tmp)
            code, out = run(root)
        self.assertEqual(code, 1, out)
        self.assertIn("3 tilts", out)

    def test_missing_processing_dir_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = run(Path(tmp))
        self.assertEqual(code, 2)

    def test_count_tilts_ignores_headers_and_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.tomostar"
            p.write_text(TOMOSTAR)
            self.assertEqual(chk.count_tilts(p), 3)
            self.assertIsNone(chk.count_tilts(Path(tmp) / "nope.tomostar"))


def main():
    """Run the suite and print the 'N passed, M failed' tally run_all.py parses."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCheckCtf)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    failed = len(result.failures) + len(result.errors)
    for case, _ in result.failures + result.errors:
        print(f"FAIL  {case.id().rsplit('.', 1)[-1]}")
    print(f"\n{result.testsRun - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
