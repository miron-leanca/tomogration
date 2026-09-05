"""'Deselect all unaligned' must find the alignments before it judges them.

tomostars_without_alignments compared tomostar/*.tomostar against .xf files in
the newest AreTomo folder -- but looked only at the project ROOT
(aretomo_output[-vN]). AreTomo now writes into jobs/J13_aretomo_output/, so
every series looked unaligned and the helper would have built a command
deselecting the entire dataset. Deselected series are skipped by ts_ctf,
ts_reconstruct and M, silently, so the run "succeeds" having processed nothing.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stub"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tomogration_project import ProjectState  # noqa: E402

passed = failed = 0
SERIES = ("Position001", "Position002", "Position005")


def check(name, got, want):
    global passed, failed
    if got == want:
        passed += 1
    else:
        failed += 1
        print(f"FAIL {name}: got {got!r} want {want!r}")


def project(where=None, aligned=(), series=SERIES, flat=False):
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "tomostar"))
    for s in series:
        open(os.path.join(d, "tomostar", s + ".tomostar"), "w").close()
    for s in aligned:
        if flat:                       # some AreTomo builds write Imod/<s>.xf
            sub = os.path.join(d, where, "Imod")
            os.makedirs(sub, exist_ok=True)
            open(os.path.join(sub, s + ".xf"), "w").close()
        else:
            sub = os.path.join(d, where, "Imod", s + "_Imod")
            os.makedirs(sub, exist_ok=True)
            open(os.path.join(sub, s + ".xf"), "w").close()
    return ProjectState(d)


def test_job_scoped_aretomo_is_found():
    """The case that broke: AreTomo wrote to jobs/, the scan looked at root."""
    check("job-scoped", project("jobs/J13_aretomo_output",
                                ["Position001", "Position002"]
                                ).tomostars_without_alignments(),
          ["Position005"])


def test_root_scoped_aretomo_still_works():
    check("root-scoped", project("aretomo_output", ["Position001", "Position002"]
                                 ).tomostars_without_alignments(),
          ["Position005"])


def test_flat_imod_layout():
    check("Imod/<s>.xf", project("jobs/J13_aretomo_output",
                                 ["Position001", "Position002"], flat=True
                                 ).tomostars_without_alignments(),
          ["Position005"])


def test_all_unaligned_is_reported_honestly():
    """A genuinely empty AreTomo run SHOULD list everything -- the bug was
    reporting that when alignments existed, not the report itself."""
    check("nothing aligned", project("jobs/J13_aretomo_output", []
                                     ).tomostars_without_alignments(), list(SERIES))


def test_no_aretomo_at_all():
    check("no folder", project().tomostars_without_alignments(), list(SERIES))


def test_no_tomostars_means_nothing_to_judge():
    check("no tomostar dir",
          ProjectState(tempfile.mkdtemp()).tomostars_without_alignments(), [])
    check("empty tomostar dir",
          project(series=()).tomostars_without_alignments(), [])


def test_an_empty_newer_folder_does_not_mask_an_older_one():
    """A restarted AreTomo leaves an empty job dir; the alignments are in the
    previous one and must still count."""
    p = project("aretomo_output", ["Position001", "Position002"])
    os.makedirs(os.path.join(p.root, "jobs", "J20_aretomo_output", "Imod"))
    check("falls through to the real one", p.tomostars_without_alignments(),
          ["Position005"])


def test_a_dir_without_an_xf_is_not_an_alignment():
    """AreTomo makes the _Imod folder before it succeeds."""
    p = project("jobs/J13_aretomo_output", ["Position001"])
    os.makedirs(os.path.join(p.root, "jobs", "J13_aretomo_output", "Imod",
                             "Position002_Imod"))
    check("empty _Imod ignored", p.tomostars_without_alignments(),
          ["Position002", "Position005"])


for fn in sorted(k for k in dict(globals()) if k.startswith("test_")):
    globals()[fn]()
print(f"{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
