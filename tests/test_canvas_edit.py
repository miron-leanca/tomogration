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
bad = {"input_pattern": "*Spike-flower_v2.star", "coords_angpix": "12.56",
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
check("a correct export produces no warning at all",
      EXP["validate"](good) == "")
check("matching tag and coords -> no scale warning",
      "too far" not in EXP["validate"](good))
# normalized_coords carries no pixel size, so the tag must not be compared to it
check("normalised coords are not compared against the Apx tag",
      "too far" not in EXP["validate"](
          dict(good, normalized_coords=True, coords_angpix="")))

# ---- an export wired to a converter inherits every fiddly value -------------
d = jobs.derive_child_params(
    "ts_export_particles", "relion4_select_picks",
    {"out_dir": "picks_class3_Spike-flower_v2_260803", "suffix": "picks_v2",
     "coords_angpix": "3.14"}, "jobs/J74")
check("input_directory comes from the converter's out_dir",
      d["input_directory"] == "picks_class3_Spike-flower_v2_260803")
check("input_pattern is built from the converter's suffix",
      d["input_pattern"] == "*picks_v2.star")
check("coords_angpix is carried, not retyped", d["coords_angpix"] == "3.14")
check("output_star lands inside output_processing",
      d["output_star"].startswith(d["output_processing"] + "/"))
check("the derived export passes its own validator",
      EXP["validate"]({**d, "box": 80, "output_angpix": "3.14",
                       "diameter": "150"}) == "")
check("relion4_to_warp derives the same way",
      jobs.derive_child_params("ts_export_particles", "relion4_to_warp",
                               {"out_dir": "picks", "suffix": "s",
                                "coords_angpix": "6.28"})["coords_angpix"] == "6.28")
check("a converter with no coords_angpix omits it rather than guessing",
      "coords_angpix" not in jobs.derive_child_params(
          "ts_export_particles", "relion4_to_warp",
          {"out_dir": "picks", "suffix": "s"}))

# The edge has to exist or the menu never offers it — that is why the export fell
# back to "newest threshold_picks" and inherited the wrong pick set.
check("export is downstream of relion4_select_picks",
      "ts_export_particles" in jobs.DOWNSTREAM["relion4_select_picks"])
check("export is downstream of relion4_to_warp",
      "ts_export_particles" in jobs.DOWNSTREAM["relion4_to_warp"])


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
# parent for a later build. The menu promises a card, so when none appeared the
# natural move was to drag one in from the palette — which arrives blank, then
# picked up the stashed parent. That is how a re-extract ran with no particle star
# and died on its required positional argument.
TOWARP = next(s for s in st.STAGES if s["id"] == "relion4_to_warp")
SELPICK = next(s for s in st.STAGES if s["id"] == "relion4_select_picks")

sel_params = {"source_star": "Select/job019/particles.star",
              "override_suffix": "picks_v5", "tomo_angpix": "6.28"}
d = jobs.derive_child_params("relion4_to_warp", "ts_template_match",
                             sel_params, "jobs/J71")
check("a selection card supplies the converter's particle star",
      d["particles_star"] == "Select/job019/particles.star")
check("and turns MODE C on", d["relion_coords"] is True)
check("the derived converter passes its own validator",
      "No particle star" not in TOWARP["validate"]({**d, "keep_all": True}))

d = jobs.derive_child_params("relion4_select_picks", "ts_template_match",
                             sel_params, "jobs/J71")
check("select-good-class gets the star too",
      d["class_star"] == "Select/job019/particles.star")

# The blank card that actually shipped: no star at all.
check("a blank converter is rejected before it runs",
      "No particle star" in TOWARP["validate"]({"keep_all": True,
                                                "relion_coords": True}))
check("a blank class-select is rejected too",
      "No classification star" in SELPICK["validate"]({"classes": "3"}))
check("the star check fires FIRST, before the mode hints",
      TOWARP["validate"]({}).startswith("⚠ No particle star"))

# A template-match / crYOLO parent has no source_star and must not be treated as a
# selection — there is no RELION star to hand over.
check("a plain pick set supplies no particle star",
      "particles_star" not in jobs.derive_child_params(
          "relion4_to_warp", "ts_template_match",
          {"override_suffix": "cryolo", "tomo_angpix": "12.56"}, "jobs/J10"))

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
