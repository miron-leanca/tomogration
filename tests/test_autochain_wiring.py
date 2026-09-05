"""Every job-building path must see the project, not just the template.

The dynamic defaults -- detected gain, the mdoc's pixel size and dose, the
newest upstream job's output dir -- all lived inside _populate_builder, so they
applied only when a HUMAN OPENED A FORM. Jobs the auto-chain built got
_effective_params (template defaults + stored edits) and none of it. That is
one bug with many faces:

    J9  imported from an empty warp_frameseries   (parent had written jobs/J7_...)
    J13 wrote to the trunk aretomo_output
    J4  imported from an empty warp_frameseries, again
    every settings card carried EML46's 1.57 A/px onto 1.98 A/px data

and it is why the fix was always "click Rebuild from controls" -- Rebuild opens
the form. Both paths now call _dynamic_overrides, and _build_job wires to its
parent the way _build_downstream always has.
"""
import importlib.util
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stub"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_spec = importlib.util.spec_from_file_location(
    "tomapp", os.path.join(ROOT, "tomogration_app.py"))
tomapp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tomapp)

from tomogration_stages import STAGES, stage_defaults  # noqa: E402
from tomogration_project import ProjectState  # noqa: E402
from tomogration_jobs import new_job, load_jobs, save_jobs  # noqa: E402

passed = failed = 0


def check(name, got, want):
    global passed, failed
    if got == want:
        passed += 1
    else:
        failed += 1
        print(f"FAIL {name}: got {got!r} want {want!r}")


def stage(sid):
    return next(s for s in STAGES if s["id"] == sid)


class Harness(tomapp.Tomogration):
    """Only the params/build plumbing -- no Qt, no __init__ chain."""
    def __init__(self, root):
        self.project_root = root
        self.project = ProjectState(root)
        self._param_store = {}
        self._pending_parent = {}
        self.current = None
        self.logs = []

    def _log(self, msg, kind="info"):
        self.logs.append(msg)

    def _stage_by_id(self, sid):
        return next((s for s in STAGES if s["id"] == sid), None)

    def _refresh_canvas(self, *a, **k):
        pass

    def _persist_param_store(self):
        pass


# EML50: 1.98 A/px, 3.49 e/A^2, a gain whose name no template could guess.
MDOC = ("[ZValue = 0]\nPixelSpacing = 1.98\nExposureDose = 3.49\n"
        "ImageSize = 4096 4096\n"
        "SubFramePath = \\\\srv\\Position001_001_0.00_EER.eer\n")


def project():
    d = tempfile.mkdtemp()
    for sub in ("mdocs", "gains", "jobs"):
        os.makedirs(os.path.join(d, sub))
    with open(os.path.join(d, "mdocs", "Position001.mdoc"), "w") as fh:
        fh.write(MDOC)
    open(os.path.join(d, "gains",
                      "20260811_100618_EER_GainReference.gain"), "w").close()
    return Harness(d)


def test_effective_params_carries_the_project():
    h = project()
    for sid, key, want in (("create_settings_fs", "angpix", "1.98"),
                           ("create_settings_ts", "angpix", "1.98"),
                           ("create_settings_fs", "exposure", "3.49"),
                           ("aretomo", "angpix", "1.98"),
                           ("gain_convert", "in_gain",
                            "gains/20260811_100618_EER_GainReference.gain")):
        check(f"{sid}.{key}", h._effective_params(stage(sid))[key], want)


def test_the_template_really_did_disagree():
    """Guards the test itself: if the templates ever ship 1.98 this proves
    nothing."""
    check("template is EML46's", stage_defaults(stage("create_settings_fs"))["angpix"],
          "1.57")
    check("gain placeholder", stage_defaults(stage("gain_convert"))["in_gain"],
          "gains/original.gain")


def test_auto_chain_wires_to_its_parent():
    """J2 -> J4, the exact case that failed: ts_import takes no
    --input_processing, so --frameseries is the only thing aiming it."""
    h = project()
    j2 = new_job(h.project_root, "fs_motion_and_ctf", "Motion + CTF",
                 h._effective_params(stage("fs_motion_and_ctf")), {})
    j4 = h._build_job("ts_import", run=False, confirm=False, parent=j2["id"])
    check("frameseries follows the parent", j4["params"]["frameseries"],
          j2["output_dir"])
    check("not the trunk", j4["params"]["frameseries"] == "warp_frameseries", False)


def test_wiring_works_while_the_parent_is_still_running():
    """J4 was built mid-run of J2. A 'completed'-only lookup would miss it, so
    the wiring reads the parent record directly."""
    h = project()
    j2 = new_job(h.project_root, "fs_motion_and_ctf", "Motion + CTF",
                 h._effective_params(stage("fs_motion_and_ctf")), {})
    store = load_jobs(h.project_root)
    store["jobs"][j2["id"]]["status"] = "running"
    save_jobs(h.project_root, store)
    j4 = h._build_job("ts_import", run=False, confirm=False, parent=j2["id"])
    check("running parent still wires", j4["params"]["frameseries"],
          j2["output_dir"])


def test_explicit_params_are_never_overwritten():
    """_build_downstream derives first and passes params in; typed form values
    arrive the same way. Neither is ours to second-guess."""
    h = project()
    j2 = new_job(h.project_root, "fs_motion_and_ctf", "Motion + CTF", {}, {})
    mine = dict(h._effective_params(stage("ts_import")), frameseries="my/own/dir")
    j4 = h._build_job("ts_import", params=mine, run=False, confirm=False,
                      parent=j2["id"])
    check("caller wins", j4["params"]["frameseries"], "my/own/dir")


def test_only_params_the_stage_actually_has_are_wired():
    """A derive may return keys for a sibling stage; writing them in would put
    junk in the record and, worse, on the command line."""
    h = project()
    j2 = new_job(h.project_root, "fs_motion_and_ctf", "Motion + CTF", {}, {})
    j4 = h._build_job("ts_import", run=False, confirm=False, parent=j2["id"])
    check("no foreign keys",
          set(j4["params"]) - {p["name"] for p in stage("ts_import")["params"]},
          set())


def test_a_trunk_parented_job_still_gets_project_values():
    h = project()
    j = h._build_job("create_settings_ts", run=False, confirm=False)
    check("angpix", j["params"]["angpix"], "1.98")
    check("exposure", j["params"]["exposure"], "3.49")


def test_no_mdoc_no_invention():
    """An empty project must fall back to the template, not to a guess."""
    h = Harness(tempfile.mkdtemp())
    check("template angpix", h._effective_params(stage("create_settings_fs"))["angpix"],
          "1.57")


for fn in sorted(k for k in dict(globals()) if k.startswith("test_")):
    globals()[fn]()
print(f"{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
