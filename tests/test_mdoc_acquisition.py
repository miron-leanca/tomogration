"""Acquisition parameters come from the mdoc, not from the last dataset.

EML50 was collected at 1.98 A/px; every stage default said 1.57 (EML46's
number). Nothing errors on a wrong pixel size -- it just makes every
downstream box quietly wrong -- so this is guarded by tests, not by eyes.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stub"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tomogration_project import ProjectState  # noqa: E402

EML50 = """[ZValue = 0]
ImageSize = 4096 4096
PixelSpacing = 1.98
Voltage = 300.00
Magnification = 64000
ExposureDose = 3.49
SubFramePath = \\\\192.168.10.81\\Storage\\Position068_001_0.00_EER.eer
[ZValue = 1]
PixelSpacing = 1.98
ExposureDose = 3.49
"""

passed = failed = 0


def check(name, got, want):
    global passed, failed
    if got == want:
        passed += 1
    else:
        failed += 1
        print(f"FAIL {name}: got {got!r} want {want!r}")


def project(files, subdir="mdocs"):
    d = tempfile.mkdtemp()
    if files is not None:
        os.makedirs(os.path.join(d, subdir))
        for fn, txt in files.items():
            with open(os.path.join(d, subdir, fn), "w") as f:
                f.write(txt)
    return ProjectState(d)


def test_reads_real_mdoc():
    check("real mdoc", project({"Position068.mdoc": EML50}).mdoc_acquisition(),
          {"angpix": "1.98", "exposure": "3.49", "image_size": ["4096", "4096"]})


def test_first_zvalue_wins():
    """Values repeat per block; a later block must not overwrite the first."""
    txt = EML50 + "[ZValue = 2]\nPixelSpacing = 9.99\nExposureDose = 99\n"
    check("first block wins",
          project({"P.mdoc": txt}).mdoc_acquisition().get("angpix"), "1.98")


def test_override_mdocs_ignored():
    """Tomo5 *_override.mdoc are not acquisition records."""
    check("override skipped",
          project({"P_override.mdoc": EML50}).mdoc_acquisition(), {})


def test_bad_values_dropped():
    """A wrong default is worse than the known one: drop what won't parse."""
    for label, txt in (
        ("non-numeric", "PixelSpacing = abc\nExposureDose = xyz\n"),
        ("zero", "PixelSpacing = 0\nExposureDose = 0\n"),
        ("negative", "PixelSpacing = -1.98\nExposureDose = -3\n"),
        ("bad dims", "ImageSize = 4096\n"),
        ("non-digit dims", "ImageSize = x y\n"),
    ):
        check(f"dropped {label}", project({"P.mdoc": txt}).mdoc_acquisition(), {})


def test_partial_mdoc():
    """A pixel size without a dose is still worth having."""
    check("angpix only",
          project({"P.mdoc": "PixelSpacing = 1.98\n"}).mdoc_acquisition(),
          {"angpix": "1.98"})


def test_missing_sources():
    check("empty mdocs dir", project({}).mdoc_acquisition(), {})
    check("no mdocs dir", project(None).mdoc_acquisition(), {})


def test_gain_detection_matches_sort_rule():
    """Sorting files any *gain*.mrc into gains/ while detection looked for
    *_gain*.mrc meant 'gainref.mrc' was filed correctly and never found."""
    for names, want in (
        (["original.gain"], "gains/original.gain"),
        (["gainref.mrc"], "gains/gainref.mrc"),
        (["gain.mrc"], "gains/gain.mrc"),
        (["CountRef_gain.mrc"], "gains/CountRef_gain.mrc"),
        (["gain_reciprocal.mrc"], ""),
        (["gain_reciprocal.mrc", "gainref.mrc"], "gains/gainref.mrc"),
        (["original_gain.mrc", "original.gain"], "gains/original.gain"),
        ([], ""),
    ):
        p = project({n: "" for n in names}, subdir="gains")
        check(f"gain {names}", p.gain_source(), want)


for fn in sorted(k for k in dict(globals()) if k.startswith("test_")):
    globals()[fn]()
print(f"{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
