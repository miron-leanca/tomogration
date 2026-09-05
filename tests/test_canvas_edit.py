"""test_canvas_edit.py — an editable canvas: card naming, placement, re-wiring.

The auto-layout (one row per stage, forks across columns) is right for a fresh
project and wrong the moment a real one branches, and an adopted RELION job used
to arrive as a card reading "RELION selection · Select/job009 · used" — which says
neither what kind of job it was nor how big. Three pieces:

  * relion_card_text()   name the RELION job type, number and particle count
  * set_card_position()  a dragged card's position wins over the computed one
  * set_job_parent()     the DAG is editable, not merely inferred

    python3 tests/test_canvas_edit.py
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE / "stub"))
sys.path.insert(0, str(REPO))


def load(mod):
    spec = importlib.util.spec_from_file_location(mod, REPO / f"{mod}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod] = m
    spec.loader.exec_module(m)
    return m


jobs = load("tomogration_jobs")
passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}")


# ---- naming ----------------------------------------------------------------
check("Select folder -> type + number",
      jobs.relion_job_parts("Select/job009") == ("Select", "job009"))
check("nested under a project dir",
      jobs.relion_job_parts("relion4/warp/Refine3D/job017") == ("Refine3D", "job017"))
check("Class3D recognised",
      jobs.relion_job_parts("Class3D/job005") == ("Class3D", "job005"))
check("unknown folder yields no type",
      jobs.relion_job_parts("SomethingElse/job001") == ("", "job001"))
check("empty is safe", jobs.relion_job_parts("") == ("", ""))
check("None is safe", jobs.relion_job_parts(None) == ("", ""))
check("a bare job number still parses",
      jobs.relion_job_parts("job042") == ("", "job042"))

title, sub = jobs.relion_card_text(
    {"params": {"job_dir": "Select/job009", "job_type": "Select",
                "n_particles": 32656}})
check("Select is titled 'Subset selection'", title == "Subset selection")
check("subtitle carries the job number", "job009" in sub)
check("subtitle carries a thousands-separated count", "32,656 particles" in sub)

title, sub = jobs.relion_card_text(
    {"params": {"job_dir": "Refine3D/job017", "job_type": "Refine3D"}})
check("Refine3D is titled '3D refinement'", title == "3D refinement")
check("no count -> no particle text", "particles" not in sub)
check("but the job number still shows", sub == "job017")

# A job adopted by an older version has no job_type/n_particles recorded — it must
# still read sensibly rather than falling back to "RELION selection".
title, sub = jobs.relion_card_text({"params": {"source_star": "Select/job023/particles.star"},
                                    "label": "Select/job023"})
check("legacy job (no recorded type) still resolves its type",
      title == "Subset selection")
check("legacy job still resolves its number", "job023" in sub)
check("a job with no params at all does not crash",
      jobs.relion_card_text({})[0] == "RELION job")
check("a non-numeric particle count is ignored, not crashed",
      "particles" not in jobs.relion_card_text(
          {"params": {"job_dir": "Select/job009", "n_particles": "lots"}})[1])


# ---- particle counting -----------------------------------------------------
STAR = """
data_particles

loop_
_rlnCoordinateX #1
_rlnCoordinateY #2
_rlnImageName #3
100.0 200.0 subtomo/Position001/a.mrc
101.0 201.0 subtomo/Position001/b.mrc
102.0 202.0 subtomo/Position002/c.mrc
"""

with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "particles.star"
    p.write_text(STAR)
    check("counts data rows, not header lines", jobs.star_particle_count(p) == 3)
    check("a missing file returns None",
          jobs.star_particle_count(Path(tmp) / "nope.star") is None)
    (Path(tmp) / "empty.star").write_text("")
    check("an empty file counts zero",
          jobs.star_particle_count(Path(tmp) / "empty.star") == 0)
    # A star with an optics block first: the count must come from the particles
    # loop, not the two-row optics table that precedes it.
    (Path(tmp) / "optics.star").write_text(
        "data_optics\n\nloop_\n_rlnOpticsGroup #1\n_rlnImagePixelSize #2\n"
        "1 3.14\n\ndata_particles\n\nloop_\n_rlnCoordinateX #1\n"
        "1.0\n2.0\n3.0\n4.0\n")
    check("an optics block does not inflate the count",
          jobs.star_particle_count(Path(tmp) / "optics.star") == 4)
    (Path(tmp) / "capped.star").write_text(
        "data_p\n\nloop_\n_rlnX #1\n" + "\n".join(str(i) for i in range(50)))
    check("the cap bounds the read", jobs.star_particle_count(
        Path(tmp) / "capped.star", cap=10) == 10)


# ---- positions -------------------------------------------------------------
def project(tmp, jobs_list):
    store = {"seq": len(jobs_list), "jobs": {}}
    for j in jobs_list:
        store["jobs"][j["id"]] = {
            "id": j["id"], "stage_id": j["stage_id"], "label": j.get("label", ""),
            "params": j.get("params", {}), "inputs": j.get("inputs", {}),
            "output_dir": f"jobs/{j['id']}", "command": "", "status": "completed",
            "exit_code": 0, "created": "2026-08-01 10:00:00",
            "started": "2026-08-01 10:00:00", "finished": "2026-08-01 10:05:00",
            "summary": {}}
    (Path(tmp) / ".tomogration_jobs.json").write_text(json.dumps(store))
    return Path(tmp)


with tempfile.TemporaryDirectory() as tmp:
    root = project(tmp, [{"id": "J1", "stage_id": "ts_ctf"},
                         {"id": "J2", "stage_id": "ts_reconstruct"}])
    nodes, _ = jobs.canvas_layout(jobs.load_jobs(root))
    auto = {n["id"]: (n["x"], n["y"]) for n in nodes}
    check("a fresh project uses the computed layout",
          all(not n.get("moved") for n in nodes))

    jobs.set_card_position(root, "J1", 640.0, 1234.5)
    nodes, edges = jobs.canvas_layout(jobs.load_jobs(root))
    idx = {n["id"]: n for n in nodes}
    check("a stored position overrides the computed one",
          (idx["J1"]["x"], idx["J1"]["y"]) == (640.0, 1234.5))
    check("the moved card is flagged as moved", idx["J1"].get("moved") is True)
    check("other cards keep their computed position",
          (idx["J2"]["x"], idx["J2"]["y"]) == auto["J2"])
    check("position survives a round-trip through the store",
          jobs.load_jobs(root)["positions"]["J1"] == [640.0, 1234.5])

    # The template rail is FIXED FURNITURE — the reference the working canvas is
    # read against. A stored position for it is ignored rather than honoured, so it
    # cannot be dragged out of pipeline order (by a user or by a stale coordinate).
    jobs.set_card_position(root, "ghost:ts_stack", 10.0, 20.0)
    idx = {n["id"]: n for n in jobs.canvas_layout(jobs.load_jobs(root))[0]}
    check("a template card cannot be placed",
          (idx["ghost:ts_stack"]["x"], idx["ghost:ts_stack"]["y"]) != (10.0, 20.0))
    check("it stays in the rail", idx["ghost:ts_stack"]["x"] == 0)

    # A position stored before the rail existed would drop a real job onto the
    # template; it is clamped out instead.
    jobs.set_card_position(root, "J2", 5.0, 400.0)
    idx = {n["id"]: n for n in jobs.canvas_layout(jobs.load_jobs(root))[0]}
    check("a job placed inside the rail is clamped out",
          idx["J2"]["x"] == jobs.RAIL_W)
    check("its row is respected", idx["J2"]["y"] == 400.0)
    jobs.clear_card_positions(root, "J2")

    jobs.clear_card_positions(root, "J1")
    idx = {n["id"]: n for n in jobs.canvas_layout(jobs.load_jobs(root))[0]}
    check("clearing one position restores its computed place",
          (idx["J1"]["x"], idx["J1"]["y"]) == auto["J1"])
    check("and leaves the others alone",
          jobs.load_jobs(root)["positions"].get("ghost:ts_stack") == [10.0, 20.0])
    check("(stored, but ignored at layout time)",
          {n["id"]: n for n in jobs.canvas_layout(jobs.load_jobs(root))[0]}
          ["ghost:ts_stack"]["x"] == 0)

    jobs.clear_card_positions(root)
    check("clearing all empties the map", jobs.load_jobs(root)["positions"] == {})

with tempfile.TemporaryDirectory() as tmp:
    # A position left behind by a deleted job must not be applied to anything, and
    # must not stop the layout being built.
    root = project(tmp, [{"id": "J1", "stage_id": "ts_ctf"}])
    before = {n["id"]: (n["x"], n["y"])
              for n in jobs.canvas_layout(jobs.load_jobs(root))[0]}
    jobs.set_card_position(root, "J99", 500.0, 500.0)     # job no longer exists
    check("a bad coordinate is refused, not raised",
          jobs.set_card_position(root, "J1", "bad", None) is None)
    nodes, _ = jobs.canvas_layout(jobs.load_jobs(root))
    after = {n["id"]: (n["x"], n["y"]) for n in nodes}
    check("a position for a vanished job is ignored", after == before)
    check("every card still has numeric coordinates",
          all(isinstance(n["x"], (int, float)) and isinstance(n["y"], (int, float))
              for n in nodes))
    # A hand-edited store with a garbage entry must not take the canvas down.
    st = jobs.load_jobs(root)
    st["positions"]["J1"] = "not-a-pair"
    jobs.save_jobs(root, st)
    check("a malformed stored position is skipped",
          jobs.canvas_layout(jobs.load_jobs(root))[0] is not None)


# ---- re-wiring the DAG -----------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    root = project(tmp, [{"id": "J1", "stage_id": "relion4_result"},
                         {"id": "J2", "stage_id": "m_create_species"}])
    _, edges = jobs.canvas_layout(jobs.load_jobs(root))
    check("an adopted job draws no edge until it is wired",
          ("J1", "J2") not in edges)

    jobs.set_job_parent(root, "J2", "J1")
    check("the parent is recorded",
          jobs.load_jobs(root)["jobs"]["J2"]["inputs"]["processing"] == "J1")
    _, edges = jobs.canvas_layout(jobs.load_jobs(root))
    check("and the canvas now draws that edge", ("J1", "J2") in edges)

    jobs.set_job_parent(root, "J2", None)
    check("detaching removes the input",
          "processing" not in jobs.load_jobs(root)["jobs"]["J2"]["inputs"])

    check("a job cannot be its own parent",
          jobs.set_job_parent(root, "J2", "J2") is None)
    check("an unknown parent is refused",
          jobs.set_job_parent(root, "J2", "J77") is None)
    check("an unknown job is refused",
          jobs.set_job_parent(root, "J77", "J1") is None)
    check("a refused re-wire changes nothing",
          jobs.load_jobs(root)["jobs"]["J2"]["inputs"] == {})



# ---- the export traps that actually fired ----------------------------------
# A re-extraction went out with coords_angpix 12.56 while the pick files said
# 3.14Apx, AND its star aimed at the previous round's folder. Both were
# detectable; only one was reported, because the validator returned the FIRST
# warning and stopped.
st = load("tomogration_stages")
EXP = next(s for s in st.STAGES if s["id"] == "ts_export_particles")
# input_directory is set because the form always supplies one: the validator
# now also checks that EXACTLY ONE input route is chosen (a RELION star, or a
# pick-star folder), and a dict with neither is not a state the card can be in.
bad = {"input_directory": "jobs/J23/matching",
       "input_pattern": "*Spike-flower_v2.star", "coords_angpix": "12.56",
       "normalized_coords": False, "box": 80, "output_angpix": "3.14",
       "diameter": "150", "output_processing": "relion4/picks_v2",
       "output_star": "relion4/picks_v1/matching.star"}
msg = EXP["validate"](bad)
check("no Apx tag in the pattern -> no scale warning", "too far" not in msg)
check("mismatched output_star is still reported", "OVERWRITES" in msg)

bad["input_pattern"] = "*3.14Apx_picks_class3_Spike-flower_v2.star"
msg = EXP["validate"](bad)
check("a pattern stating 3.14Apx vs coords_angpix 12.56 is caught",
      "coords_angpix is 12.56" in msg and "3.14" in msg)
check("the warning quantifies the error", "4×" in msg or "4x" in msg)
check("BOTH problems are reported together",
      "coords_angpix" in msg and "OVERWRITES" in msg)

good = dict(bad, coords_angpix="3.14",
            output_star="relion4/picks_v2/matching.star")
check("a missing input route is now its own warning",
      "NO INPUT" in EXP["validate"](dict(bad, input_directory="")))
check("and with both filled in, the folder is declared ignored",
      "IGNORED" in EXP["validate"](dict(bad, input_star="s.star")))
check("a correct export produces no warning at all",
      EXP["validate"](good) == "")
check("matching tag and coords -> no scale warning",
      "too far" not in EXP["validate"](good))
# normalized_coords carries no pixel size, so the tag must not be compared to it
check("normalised coords are not compared against the Apx tag",
      "too far" not in EXP["validate"](
          dict(good, normalized_coords=True, coords_angpix="")))

# ---- an export built from a RELION result reads the star DIRECTLY -----------
# The one re-extraction route: Warp takes the RELION star, subtracts the refined
# rlnOrigin*Angst itself and scales by coords_angpix (= the star's own pixel
# size, which the app fills in from the file). No pick-star folder, no pattern,
# no hand conversion — the converter with its three modes is retired.
d = jobs.derive_child_params(
    "ts_export_particles", "relion4_result",
    {"data_star": "Select/job029/particles.star", "job_dir": "Select/job029"}, "")
check("input_star is the selection's star",
      d["input_star"] == "Select/job029/particles.star")
check("the pick-star folder and pattern are blanked, not inherited",
      d["input_directory"] == "" and d["input_pattern"] == "")
check("'0-1 fractions' is OFF for a RELION star", d["normalized_coords"] is False)
check("coords_angpix is left for the star to supply", d["coords_angpix"] == "")
check("the export dir is named after the selection",
      d["output_processing"] == "relion4/Select-job029_{jobid}")
check("output_star lands inside output_processing",
      d["output_star"].startswith(d["output_processing"] + "/"))
check("a promoted selection (source_star) derives the same way",
      jobs.derive_child_params(
          "ts_export_particles", "relion4_result",
          {"source_star": "Select/job029/particles.star",
           "job_dir": "Select/job029"}, "")["input_star"]
      == "Select/job029/particles.star")
check("with coords_angpix filled in, the derived export passes its validator",
      EXP["validate"]({**d, "coords_angpix": "6.28", "box": 192,
                       "output_angpix": "1.57", "diameter": "240"}) == "")
check("without it, the validator refuses the RELION-star route",
      "REQUIRED" in EXP["validate"]({**d, "box": 192, "output_angpix": "1.57",
                                     "diameter": "240"}))
_cmd = st.build_command(EXP, {**{q["name"]: q.get("default") for q in EXP["params"]},
                              **d, "coords_angpix": "6.28"})
check("the command carries --input_star and --coords_angpix and nothing pick-star",
      "--input_star Select/job029/particles.star" in _cmd
      and "--coords_angpix 6.28" in _cmd and "--input_directory" not in _cmd
      and "--input_pattern" not in _cmd and "--normalized_coords" not in _cmd)

# The edges have to exist or the menu never offers them.
check("export is downstream of a RELION result",
      "ts_export_particles" in jobs.DOWNSTREAM["relion4_result"])
check("the retired converters are offered nowhere",
      "relion4_to_warp" not in jobs.DOWNSTREAM
      and "relion4_select_picks" not in jobs.DOWNSTREAM
      and all("relion4_to_warp" not in v and "relion4_select_picks" not in v
              for v in jobs.DOWNSTREAM.values()))
check("verify is downstream of an export",
      "relion4_verify_reextract" in jobs.DOWNSTREAM["ts_export_particles"])
v = jobs.derive_child_params(
    "relion4_verify_reextract", "ts_export_particles",
    {"input_star": "Select/job029/particles.star",
     "output_star": "relion4/Select-job029_{jobid}/matching.star"},
    "jobs/J80_ts-export-particles")
check("verify pairs the export's source star with the star it wrote",
      v["source_star"] == "Select/job029/particles.star"
      and v["new_star"] == "relion4/Select-job029_J80_ts-export-particles/matching.star")
check("and checks WITH recentring, because Warp applied the shifts",
      v["no_recenter"] is False)


# ---- pick-set conventions must not be crossed ------------------------------
# crYOLO/Warp picks are 0-1 FRACTIONS; a RELION-derived re-extract set holds
# ABSOLUTE PIXELS at the size in its filename. The two look identical in the form,
# and the completion message for the RELION path told users to turn
# --normalized_coords ON — copied from the crYOLO path — which piles every particle
# into one corner.
d = jobs.derive_child_params(
    "ts_export_particles", "ts_template_match",
    {"override_suffix": "picks_v4", "tomo_angpix": "6.28",
     "source_star": "Select/job019/particles.star"}, "jobs/J83")
check("a RELION-derived pick set carries its pixel size",
      d["coords_angpix"] == "6.28")
check("and is explicitly NOT normalised", d["normalized_coords"] is False)
check("the derived export agrees with its own pattern",
      "too far" not in EXP["validate"](
          {**d, "box": 80, "output_angpix": "3.14", "diameter": "150"}))

# A plain template-match / crYOLO parent has no source_star: it must NOT be given
# coords_angpix, because those picks really are normalised.
d2 = jobs.derive_child_params(
    "ts_export_particles", "ts_template_match",
    {"override_suffix": "cryolo_combined", "tomo_angpix": "12.56"}, "jobs/J10")
check("a non-RELION pick set is left normalised",
      "coords_angpix" not in d2 and "normalized_coords" not in d2)


# ---- "Build downstream from this" must actually build something ------------
# It used to create NO card: it derived params, seeded the builder and stashed the
# parent for a later build. A selection card now feeds the EXPORT directly, and
# the derivation must tell a selection apart from a folder of pick stars.
sel_params = {"source_star": "Select/job019/particles.star",
              "job_dir": "Select/job019"}
d = jobs.derive_child_params("ts_export_particles", "ts_template_match",
                             sel_params, "jobs/J71")
check("a pre-migration selection card supplies the export's star",
      d.get("input_star") == "Select/job019/particles.star")
check("and blanks the pick-star route", d.get("input_directory") == "")

# A re-extract PICK SET also carries source_star, but it is a folder of pick
# stars: it keeps the pick-star derivation (pixels at the filename's Å/px).
d = jobs.derive_child_params("ts_export_particles", "ts_template_match",
                             {"source_star": "Select/job019/particles.star",
                              "override_suffix": "_picks_v5",
                              "tomo_angpix": "6.28"}, "jobs/J87")
check("a re-extract pick set still exports from its folder",
      d.get("input_directory") == "jobs/J87/matching" and "input_star" not in d)
check("at its own pixel size, as pixels",
      d.get("coords_angpix") == "6.28" and d.get("normalized_coords") is False)

# A template-match / crYOLO parent has no source_star: the pick-star route.
d = jobs.derive_child_params("ts_export_particles", "ts_template_match",
                             {"override_suffix": "cryolo", "tomo_angpix": "12.56"},
                             "jobs/J10")
check("a plain pick set exports from its matching folder",
      d.get("input_directory") == "jobs/J10/matching" and "input_star" not in d)

# The retired converter stays readable on old cards, is flagged, and refuses.
TOWARP = next(s for s in st.STAGES if s["id"] == "relion4_to_warp")
check("the pick-star converter is marked retired", TOWARP.get("legacy") is True)
check("and says so instead of running", "RETIRED" in TOWARP["validate"]({}))
check("select-good-class is gone",
      not any(s["id"] == "relion4_select_picks" for s in st.STAGES))


# ---- annotation items construct, and say they carry no data ------------------
# The canvas paints frames behind cards and notes in front; both are dashed,
# because a solid line on this canvas means "this job read that job's output".
tomapp = load("tomogration_app")


class _StrictNote(tomapp._NoteItem):
    _OWN = {"_note", "_canvas", "_press_pos"}

    def __getattr__(self, name):
        if name in _StrictNote._OWN:
            raise AttributeError(f"_NoteItem touched self.{name} before __init__ set it")
        inherited = getattr(super(), "__getattr__", None)
        if inherited is None:
            raise AttributeError(name)
        return inherited(name)


class _FakeCanvas:
    locked = False


for kind, colour in (("frame", "violet"), ("note", "amber"), ("note", "octarine")):
    n = {"id": "N1", "kind": kind, "x": 10, "y": 20, "w": 200, "h": 90,
         "text": "bin4 n2n", "colour": colour}
    try:
        _StrictNote(n, _FakeCanvas())
        ok, why = True, ""
    except Exception as e:
        ok, why = False, f"{type(e).__name__}: {e}"
    check(f"_NoteItem builds for a {kind}/{colour} {why}", ok)

check("an unknown colour cannot reach the painter without a fallback",
      "octarine" not in tomapp._NOTE_COLOURS and "amber" in tomapp._NOTE_COLOURS)
check("every colour the data layer allows has a palette entry",
      all(c in tomapp._NOTE_COLOURS for c in tomapp.NOTE_COLOURS))
_f = _StrictNote({"id": "N1", "kind": "frame", "x": 0, "y": 0, "w": 10, "h": 10,
                  "text": "", "colour": "grey"}, _FakeCanvas())
_n = _StrictNote({"id": "N2", "kind": "note", "x": 0, "y": 0, "w": 10, "h": 10,
                  "text": "", "colour": "grey"}, _FakeCanvas())
check("a frame is built as a frame and a note as a note",
      _f._note["kind"] == "frame" and _n._note["kind"] == "note")

# ---------------------------------------------------------------------------
# Notes and frames survive a round trip. These live in the SAME store file as
# the jobs, so the failures worth guarding are the quiet ones: an id that
# collides with a job id, a typo'd field becoming part of the record, and a
# frame drawn in front of the cards it is supposed to sit behind.
import tempfile                                              # noqa: E402
from pathlib import Path as _P                               # noqa: E402

with tempfile.TemporaryDirectory() as _td:
    _root = _P(_td)
    (_root / ".tomogration_jobs.json").write_text('{"seq":0,"jobs":{}}')
    _j = tomapp.new_job(_root, "mb_segment", "Segment", {})
    _n1 = tomapp.add_note(_root, "note", 10, 20, text="check polarity here")
    _fr = tomapp.add_note(_root, "frame", 0, 0, w=600, h=400, colour="blue")

    check("a note gets an N-id that cannot collide with a job id",
          _n1["id"].startswith("N") and _n1["id"] != _j["id"])
    check("ids keep counting past a deletion",
          tomapp.delete_note(_root, _n1["id"])
          and tomapp.add_note(_root, "note", 0, 0)["id"] != _n1["id"])
    check("deleting a note that is already gone says so",
          not tomapp.delete_note(_root, _n1["id"]))

    _n2 = tomapp.add_note(_root, "note", 5, 5, text="before")
    tomapp.update_note(_root, _n2["id"], text="after", colour="green",
                       id="HACKED", created="nonsense")
    _got = [n for n in tomapp.load_notes(_root) if n["id"] == _n2["id"]][0]
    check("an edit updates what it should", _got["text"] == "after"
          and _got["colour"] == "green")
    check("and silently ignores keys it should not touch",
          _got["id"] == _n2["id"] and _got["created"] != "nonsense")
    check("editing a note that does not exist returns nothing",
          tomapp.update_note(_root, "N999", text="x") is None)
    check("an unknown colour falls back rather than reaching the painter",
          tomapp.add_note(_root, "note", 0, 0, colour="octarine")["colour"]
          == "amber")

    _store = tomapp.load_jobs(_root)
    _frames, _stickies = tomapp.notes_for_canvas(_store)
    check("frames and notes are split for painting, frames behind",
          len(_frames) == 1 and _frames[0]["id"] == _fr["id"]
          and all(n["kind"] == "note" for n in _stickies))
    check("a junk entry is dropped rather than drawn in the wrong layer",
          tomapp.notes_for_canvas({"notes": ["not a dict", {"kind": "wat"}]})
          == ([], []))
    check("notes survive alongside the jobs in one store",
          _j["id"] in tomapp.load_jobs(_root)["jobs"] and len(tomapp.load_notes(_root)) >= 2)

    # A frame is 'about' whatever sits inside it, computed from geometry — so
    # dragging a card into a branch needs no bookkeeping to be captured by it.
    _nodes = [{"id": "J1", "x": 100, "y": 100, "w": 200, "h": 80},
              {"id": "J2", "x": 5000, "y": 5000, "w": 200, "h": 80}]
    _in = tomapp.cards_inside(_fr, _nodes)
    check("a frame captures the cards inside it and no others",
          "J1" in _in and "J2" not in _in)
    check("ghosts and templates are never captured",
          "G1" not in tomapp.cards_inside(
              _fr, [{"id": "G1", "x": 100, "y": 100, "w": 10, "h": 10,
                     "is_ghost": True}]))


# ---------------------------------------------------------------------------
# A note paints the text it stores. QGraphicsTextItem + setTextWidth stored and
# reopened perfectly while painting an empty box; the simple-text path used by
# every card label is what actually renders, so the note uses it too.
# ---------------------------------------------------------------------------
_painted = []


class _CapturingText:
    """Stands in for QGraphicsSimpleTextItem to capture what reached the scene."""
    def __init__(self, text="", parent=None):
        _painted.append(str(text))

    def __getattr__(self, _n):
        return lambda *a, **k: None


_real_simple = tomapp.QGraphicsSimpleTextItem
tomapp.QGraphicsSimpleTextItem = _CapturingText
try:
    _painted.clear()
    tomapp._NoteItem({"id": "N9", "kind": "note", "x": 0, "y": 0, "w": 220, "h": 96,
                   "text": "Here is where I sorted the files to frames, mdocs, "
                           "and gains", "colour": "amber"}, _FakeCanvas())
    check("a note paints its own text", bool(_painted) and "sorted" in _painted[0])
    check("the text is wrapped, not one long line", "\n" in _painted[0])
    check("every painted line fits the note",
          all(len(l) * 6 <= 220 - 14 for l in _painted[0].split("\n")))

    _painted.clear()
    tomapp._NoteItem({"id": "N10", "kind": "note", "x": 0, "y": 0, "w": 220, "h": 96,
                   "text": "", "colour": "amber"}, _FakeCanvas())
    check("an empty note paints nothing", _painted == [""])

    _painted.clear()
    long_text = " ".join(["word"] * 400)
    tomapp._NoteItem({"id": "N11", "kind": "note", "x": 0, "y": 0, "w": 220, "h": 96,
                   "text": long_text, "colour": "amber"}, _FakeCanvas())
    check("a long note is clipped to its own height",
          len(_painted[0].split("\n")) <= 96 // 14)
    check("and says it was clipped", _painted[0].endswith("\u2026"))
finally:
    tomapp.QGraphicsSimpleTextItem = _real_simple

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
