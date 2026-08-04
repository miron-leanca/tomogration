"""Unit tests for the Phase-1 job-model layer of tomogration_app.

Runs OFF the VM: stubs PySide6 (tests/stub), loads the app file by path (its name
is export-mangled, so we glob for it), and exercises the PURE job functions
against a temp project dir. No Qt, no cluster, no data.

    python3 tests/test_jobs.py
"""
import ast
import sys
import tempfile
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE / "stub"))                      # fake PySide6 first

matches = sorted(REPO.glob("*tomogration_app.py"))
if not matches:
    print("FAIL  could not find *tomogration_app.py next to tests/")
    sys.exit(2)
spec = importlib.util.spec_from_file_location("tomapp", matches[0])
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)                                # import the real thing

passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}")


def stage(sid):
    return next(s for s in app.STAGES if s["id"] == sid)


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)

    # ---- store: empty scaffold, create, persist, monotonic ids -------------
    check("load_jobs empty", app.load_jobs(root) == {"seq": 0, "jobs": {}})

    j1 = app.new_job(root, "ts_ctf", "CTF", {"window": "512"}, inputs={"processing": None})
    check("new_job id J1", j1["id"] == "J1")
    check("new_job output_dir", j1["output_dir"] == "jobs/J1")
    check("new_job status queued", j1["status"] == "queued")
    check("jobs file written", app.jobs_path(root).is_file())

    j2 = app.new_job(root, "ts_reconstruct", "Reconstruct", {"angpix": "10"},
                     inputs={"processing": "J1"})
    check("monotonic id J2", j2["id"] == "J2")

    store = app.load_jobs(root)
    check("store seq 2", store["seq"] == 2)
    check("store has 2 jobs", len(store["jobs"]) == 2)

    app.update_job(root, "J1", status="completed", exit_code=0)
    check("update persists", app.load_jobs(root)["jobs"]["J1"]["status"] == "completed")
    check("update unknown id -> None", app.update_job(root, "J99", status="x") is None)

    # ids stay monotonic even after a deletion (no reuse) --------------------
    store = app.load_jobs(root)
    del store["jobs"]["J1"]
    app.save_jobs(root, store)
    j3 = app.new_job(root, "ts_ctf", "CTF", {})
    check("id not reused after delete", j3["id"] == "J3")

    # ---- warp detection ----------------------------------------------------
    check("ts_ctf is warp", app.is_warp_stage(stage("ts_ctf")))
    check("aretomo not warp", not app.is_warp_stage(stage("aretomo")))
    check("relion4_class3d not warp", not app.is_warp_stage(stage("relion4_class3d")))

    # ---- parent resolution -------------------------------------------------
    check("parent via processing slot",
          app.parent_job_id({"inputs": {"processing": "J2"}}) == "J2")
    check("parent via first non-null slot",
          app.parent_job_id({"inputs": {"a": None, "b": "J5"}}) == "J5")
    check("parent none when trunk",
          app.parent_job_id({"inputs": {"processing": None}}) is None)

    # ---- io flags ----------------------------------------------------------
    store = app.load_jobs(root)
    recon = store["jobs"]["J2"]
    recon["inputs"] = {"processing": "J3"}                  # live parent
    flags = app.io_flags_for_job(stage("ts_reconstruct"), recon, store)
    check("recon input_processing", "--input_processing jobs/J3" in flags)
    check("recon output_processing", "--output_processing jobs/J2" in flags)

    trunk = {"id": "J2", "inputs": {"processing": None}, "output_dir": "jobs/J2"}
    flags = app.io_flags_for_job(stage("ts_ctf"), trunk, store)
    check("trunk has output flag", "--output_processing jobs/J2" in flags)
    check("trunk has NO input flag", "--input_processing" not in flags)
    check("aretomo no io flags",
          app.io_flags_for_job(stage("aretomo"), trunk, store) == "")
    # threshold_picks: only --output_processing (inputs are staged into its own dir)
    thr = {"id": "J9", "inputs": {"processing": "J3"}, "output_dir": "jobs/J9"}
    tflags = app.io_flags_for_job(stage("threshold_picks"), thr, store)
    check("threshold has output flag", "--output_processing jobs/J9" in tflags)
    check("threshold has NO input flag (staged in place)",
          "--input_processing" not in tflags)

    # ---- build_job_command -------------------------------------------------
    cmd = app.build_job_command(stage("ts_reconstruct"), recon, store,
                                warp_cmd="module load warp && WarpTools")
    check("cmd exports float32 first", cmd.startswith("export WARP_FORCE_MRC_FLOAT32=1 &&"))
    check("cmd uses warp launch", "module load warp && WarpTools ts_reconstruct" in cmd)
    check("cmd wired output", "--output_processing jobs/J2" in cmd)
    check("cmd wired input", "--input_processing jobs/J3" in cmd)

    # a manual --output_processing must NOT be doubled -----------------------
    def fake_build(spec, params, *a, **k):
        return "WarpTools ts_ctf --output_processing custom/"
    _orig = app.build_command
    app.build_command = fake_build
    try:
        manual = {"id": "J4", "inputs": {"processing": "J3"}, "output_dir": "jobs/J4",
                  "params": {}}
        cmd2 = app.build_job_command(stage("ts_ctf"), manual, store)
    finally:
        app.build_command = _orig
    check("no double output_processing", cmd2.count("--output_processing") == 1)

    # ---- summarizers -------------------------------------------------------
    jd = root / "jobs" / "Jsum"
    jd.mkdir(parents=True)
    for i in range(3):
        (jd / f"Position{i:03d}.xml").write_text("<xml/>")
    (jd / "matching.star").write_text("data_\n")
    check("generic fallback for unknown stage",
          app.summarize_job("gain_convert", jd).get("xml") == 3)
    check("summarize missing dir -> {}", app.summarize_job("ts_ctf", root / "nope") == {})

    cnt, capped = app._count_glob(jd, "*.xml", cap=2)
    check("count_glob caps", cnt == 2 and capped is True)

    # ---- ts_ctf summarizer: parse the <CTF> scalar Defocus, not GridCTF -----
    # Minimal fixture mirroring the real VM XML: a <CTF> block with the scalar
    # Defocus, plus a <GridCTF> whose per-tilt <Node Value=...> must be IGNORED.
    def ctf_xml(defocus):
        nodes = "\n".join(f'<Node X="0" Y="0" Z="{z}" Value="9.99" />' for z in range(3))
        return (f'<CTF>\n<Param Name="DefocusDelta" Value="0.03" />\n'
                f'<Param Name="Defocus" Value="{defocus}" />\n'
                f'<Param Name="DefocusAngle" Value="61" />\n</CTF>\n'
                f'<GridCTF>\n{nodes}\n</GridCTF>\n')
    cd = root / "jobs" / "Jctf"
    cd.mkdir(parents=True)
    for i, dz in enumerate((5.0, 5.5, 6.0)):
        (cd / f"Position{i:03d}.xml").write_text(ctf_xml(dz))
    (cd / "empty.xml").write_text("<CTF></CTF>")           # no Defocus -> skipped
    summ = app.summarize_job("ts_ctf", cd)
    check("ctf counts only series with defocus", summ.get("series") == 3)
    check("ctf mean±std of scalar defocus", summ.get("defocus_um") == "5.50 ± 0.41")
    check("ctf ignores GridCTF nodes (mean not 9.99)", "9.99" not in summ.get("defocus_um", ""))

    # ---- ts_reconstruct summarizer: count tomograms + parse angpix ---------
    rd = root / "jobs" / "Jrec" / "reconstruction"
    rd.mkdir(parents=True)
    for i in range(4):
        (rd / f"Position{i:03d}_10.00Apx.mrc").write_bytes(b"\0")
    summ = app.summarize_job("ts_reconstruct", root / "jobs" / "Jrec")
    check("reconstruct counts tomograms", summ.get("tomograms") == 4)
    check("reconstruct parses angpix", summ.get("angpix") == "10.00")
    check("reconstruct empty when no reconstruction/ dir",
          app.summarize_job("ts_reconstruct", cd) == {})

# ---- friendly titles + napari path helpers (pure) -------------------------
check("stage_title friendly", app.stage_title("ts_reconstruct") == "Tomogram reconstruction")
check("stage_title fallback", app.stage_title("unknown_x", "raw") == "raw")
check("fmt_angpix 2dp", app.fmt_angpix("10") == "10.00" and app.fmt_angpix("12.56") == "12.56")
check("star suffix uses override verbatim",
      app.template_match_suffix({"override_suffix": "260712v2", "template_emdb": "70905"}) == "260712v2")
check("star suffix from emdb", app.template_match_suffix({"template_emdb": "70905"}) == "_emd_70905")
check("star suffix from path stem",
      app.template_match_suffix({"template_path": "/a/b/ribo.mrc"}) == "_ribo")
check("star suffix none when nothing set", app.template_match_suffix({}) == "")
# the CORR volume keeps the TEMPLATE suffix even when override_suffix renames the star
check("corr suffix ignores override",
      app.template_corr_suffix({"override_suffix": "260712v2", "template_emdb": "70905"}) == "_emd_70905")
check("corr suffix from path stem",
      app.template_corr_suffix({"template_path": "/a/b/ribo.mrc"}) == "_ribo")
check("every stage has a friendly title",
      all(s["id"] in app.FRIENDLY_TITLES for s in app.STAGES))

# A duplicated stage id draws TWO identical ghost cards on the canvas while every
# lookup (_stage_by_id, status, docs) silently resolves to the first — so the second
# card can never be reached. Two "M: reset project" cards shipped this way.
_ids = [s["id"] for s in app.STAGES]
_dupes = sorted({i for i in _ids if _ids.count(i) > 1})
check(f"stage ids are unique{(' — DUPLICATED: ' + ', '.join(_dupes)) if _dupes else ''}",
      not _dupes)

# The same mistake in FRIENDLY_TITLES is INVISIBLE at runtime — a repeated key in a
# dict literal just overwrites, so the map looks fine while one label is dead. Only
# the source can reveal it.
_src = ast.parse((REPO / "tomogration_jobs.py").read_text())
_title_keys = []
for _node in ast.walk(_src):
    if (isinstance(_node, ast.Assign)
            and any(getattr(t, "id", "") == "FRIENDLY_TITLES" for t in _node.targets)
            and isinstance(_node.value, ast.Dict)):
        _title_keys = [k.value for k in _node.value.keys
                       if isinstance(k, ast.Constant)]
_tdupes = sorted({k for k in _title_keys if _title_keys.count(k) > 1})
check(f"FRIENDLY_TITLES has no repeated keys"
      f"{(' — REPEATED: ' + ', '.join(_tdupes)) if _tdupes else ''}",
      bool(_title_keys) and not _tdupes)

# ---- canvas layout (Phase 2, pure) ----------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    croot = Path(tmp)
    n_stages = len(app.STAGES)

    # empty store -> one ghost per stage, chained along the trunk
    nodes, edges = app.canvas_layout(app.load_jobs(croot))
    check("empty: one node per stage", len(nodes) == n_stages)
    check("empty: all ghost", all(n["is_ghost"] for n in nodes))
    check("empty: trunk chained", len(edges) == n_stages - 1)
    check("empty: rows increase by stage order",
          [n["row"] for n in nodes] == list(range(n_stages)))

    # a CTF job + a reconstruct job wired to it -> two real nodes + a DAG edge
    ctf = app.new_job(croot, "ts_ctf", "CTF", {"window": "512"})
    app.update_job(croot, ctf["id"], status="completed",
                   summary={"series": 15, "defocus_um": "5.25 ± 0.41"})
    rec = app.new_job(croot, "ts_reconstruct", "Reconstruct", {"angpix": "10"},
                      inputs={"processing": ctf["id"]})
    nodes, edges = app.canvas_layout(app.load_jobs(croot))
    idx = {n["id"]: n for n in nodes}
    check("real ctf node present + not ghost",
          ctf["id"] in idx and not idx[ctf["id"]]["is_ghost"])
    check("real ctf carries status", idx[ctf["id"]]["status"] == "completed")
    check("real ctf carries summary", idx[ctf["id"]]["summary"].get("series") == 15)
    check("DAG edge parent->child present", (ctf["id"], rec["id"]) in edges)
    check("still one ghost for an un-run stage", f"ghost:aretomo" in idx)

    # a second CTF job (fork) -> two nodes on the ts_ctf row, side by side
    ctf2 = app.new_job(croot, "ts_ctf", "CTF wide", {"window": "1024"})
    nodes, _ = app.canvas_layout(app.load_jobs(croot))
    ctf_row = [n for n in nodes if n["stage_id"] == "ts_ctf"]
    jobs_row = [n for n in ctf_row if not n.get("is_template")]
    tmpl_row = [n for n in ctf_row if n.get("is_template")]
    check("fork: two JOB nodes on ts_ctf row", len(jobs_row) == 2)
    # The template rail is the default pipeline as a permanent reference, so it is
    # present for EVERY stage — including ones that already have real jobs. It used
    # to appear only for stages with none, so the template dissolved exactly as a
    # project got complicated.
    check("fork: the stage still shows its template card", len(tmpl_row) == 1)
    check("every stage has exactly one template card",
          len([n for n in nodes if n.get("is_template")]) == len(app.STAGES))
    check("the template sits in the rail at x=0", tmpl_row[0]["x"] == 0)
    check("real jobs start right of the rail",
          all(n["x"] >= app.RAIL_W for n in nodes if not n.get("is_template")))
    check("so no job can overlap the template",
          app.RAIL_W >= app.CARD_W)
    # The rail is its own top-to-bottom chain; it must not wire into real jobs.
    _, edges2 = app.canvas_layout(app.load_jobs(croot))
    rail = [(a, b) for a, b in edges2
            if str(a).startswith("ghost:") or str(b).startswith("ghost:")]
    check("every rail edge joins two templates",
          all(str(a).startswith("ghost:") and str(b).startswith("ghost:")
              for a, b in rail))
    check("the rail chains consecutive stages",
          ("ghost:" + app.STAGES[0]["id"], "ghost:" + app.STAGES[1]["id"]) in rail)
    check("fork: distinct columns (side by side)",
          {n["col"] for n in ctf_row} == {0, 1}
          and ctf_row[0]["x"] != ctf_row[1]["x"])

    # summary_text formatting
    check("summary_text ctf", app.summary_text({"series": 15, "defocus_um": "5.25 ± 0.41"})
          == "15 series · 5.25 ± 0.41 µm")
    check("summary_text reconstruct",
          app.summary_text({"tomograms": 290, "angpix": "10.00"}) == "290 tomo · 10.00 Å")
    check("summary_text empty -> ''", app.summary_text({}) == "")

# ---- ts_template_match new params + validate ------------------------------
tm = next(s for s in app.STAGES if s["id"] == "ts_template_match")
v = app.stage_defaults(tm)
v["template_emdb"] = "70905"; v["optimize_poses"] = True; v["override_suffix"] = "_run2"
v["peak_distance"] = "30"; v["max_missing_tilts"] = -1; v["npeaks"] = 8000
cmd = app.build_command(tm, v)
check("template_match emits --template_emdb", "--template_emdb 70905" in cmd)
check("template_match emits --optimize_poses", "--optimize_poses" in cmd)
check("template_match emits --override_suffix", "--override_suffix _run2" in cmd)
check("template_match emits --peak_distance", "--peak_distance 30" in cmd)
check("template_match emits --max_missing_tilts -1", "--max_missing_tilts -1" in cmd)
check("template_match emits --npeaks", "--npeaks 8000" in cmd)
check("template blank -> no template flags",
      "--template_emdb" not in app.build_command(tm, app.stage_defaults(tm)))
check("validate warns when neither template set", bool(tm["validate"](app.stage_defaults(tm))))
vok = app.stage_defaults(tm); vok["template_emdb"] = "70905"
check("validate ok when exactly one set", tm["validate"](vok) == "")
vboth = app.stage_defaults(tm); vboth["template_emdb"] = "70905"; vboth["template_path"] = "/x.mrc"
check("validate warns when both set", bool(tm["validate"](vboth)))
# check_hand > 0 is incompatible with override_suffix (Warp readback bug)
vhand = app.stage_defaults(tm); vhand["template_emdb"] = "70905"
vhand["override_suffix"] = "_v1"; vhand["check_hand"] = 2
check("validate warns check_hand + override_suffix", bool(tm["validate"](vhand)))
vhand0 = dict(vhand); vhand0["check_hand"] = 0
check("validate ok check_hand 0 + override_suffix", tm["validate"](vhand0) == "")

# ---- DIR_FILE_HINTS covers every STAGE_IO input/output dir -----------------
io_dirs = set()
for ins, outs in app.STAGE_IO.values():
    io_dirs.update(ins); io_dirs.update(outs)
io_dirs.discard("aretomo_output")   # dynamic versioned name, hinted separately
missing = [d for d in io_dirs if d not in app.DIR_FILE_HINTS]
check(f"every STAGE_IO dir has a file-pattern hint (missing: {missing})", not missing)

# ---- default_parent_for + delete_job (pure) -------------------------------
with tempfile.TemporaryDirectory() as tmp:
    pr = Path(tmp)
    check("no parent when store empty",
          app.default_parent_for("ts_reconstruct", app.load_jobs(pr)) is None)
    c1 = app.new_job(pr, "ts_ctf", "CTF", {})
    c2 = app.new_job(pr, "ts_ctf", "CTF2", {})
    # reconstruct's parent = newest upstream WARP job (the ts_ctf just made)
    check("parent = newest upstream warp job",
          app.default_parent_for("ts_reconstruct", app.load_jobs(pr)) == c2["id"])
    # aretomo is a wrapper (not a WarpTools stage) -> skipped as a processing parent
    app.new_job(pr, "aretomo", "AreTomo", {})
    check("wrapper stage not chosen as processing parent",
          app.default_parent_for("ts_reconstruct", app.load_jobs(pr)) == c2["id"])
    # export sits after reconstruct/threshold; with only ctf jobs, ctf is nearest warp
    check("nearest upstream warp chosen",
          app.default_parent_for("ts_export_particles", app.load_jobs(pr)) == c2["id"])
    check("delete removes the record",
          app.delete_job(pr, c1["id"]) and c1["id"] not in app.load_jobs(pr)["jobs"])
    check("delete unknown id -> False", app.delete_job(pr, "J999") is False)
    check("_jobnum orders", app._jobnum("J12") == 12 and app._jobnum("bad") == 0)

# ---- GPU-list separator normalisation (pure) ------------------------------
check("norm gpu comma->space", app._norm_gpu("0,1,2,3", " ") == "0 1 2 3")
check("norm gpu space->comma", app._norm_gpu("0 1 2 3", ",") == "0,1,2,3")
check("norm gpu mixed->comma", app._norm_gpu("0, 1 2,3", ",") == "0,1,2,3")
check("norm gpu preserves repeats", app._norm_gpu("0 0 0", ",") == "0,0,0")
check("norm gpu no sep = untouched", app._norm_gpu("0 1 2 3", None) == "0 1 2 3")
check("norm gpu empty", app._norm_gpu("", " ") == "")

# through build_command on the real stage specs
ctf = next(s for s in app.STAGES if s["id"] == "ts_ctf")
vals = app.stage_defaults(ctf)
vals["device_list"] = "0,1,2,3"                       # user typed commas
cmd = app.build_command(ctf, vals)
check("device_list normalised to spaces in cmd", "--device_list 0 1 2 3" in cmd)

rel = next(s for s in app.STAGES if s["id"] == "relion4_class3d")
vals = app.stage_defaults(rel)
vals["GPUS"] = "0 1 2 3"                               # user typed spaces
cmd = app.build_command(rel, vals)
check("GPUS normalised to commas in cmd", "GPUS=0,1,2,3" in cmd)

# --3d still emitted (regression guard from the earlier fix)
exp = next(s for s in app.STAGES if s["id"] == "ts_export_particles")
check("export still emits --3d by default",
      "--3d" in app.build_command(exp, app.stage_defaults(exp)).split())

# ---- discover_picksets + orphan cards (Phase: adopt on-disk sets) ----------
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    md = root / "warp_tiltseries" / "matching"
    md.mkdir(parents=True)
    # two pick sets by suffix, over 2 series, + non-star noise
    for series in ("Position042", "Position046"):
        (md / f"{series}_12.56Apx260712v2.star").write_text("x")
        (md / f"{series}_12.56Apx_emd_70905.star").write_text("x")
        (md / f"{series}_12.56Apx_emd_70905_corr.mrc").write_bytes(b"\0")  # not a .star
    orphs = app.discover_picksets(root, app.load_jobs(root))
    by_suffix = {o["suffix"]: o for o in orphs}
    check("discovers both suffixes", set(by_suffix) == {"260712v2", "_emd_70905"})
    check("counts series per set", by_suffix["260712v2"]["n_series"] == 2)
    check("parses angpix", by_suffix["260712v2"]["angpix"] == "12.56")
    check("ignores non-star files", all(".mrc" not in o["suffix"] for o in orphs))

    # a job already covering a suffix hides that orphan
    app.new_job(root, "ts_template_match", "M", {"override_suffix": "260712v2",
                                                 "template_emdb": "70905"})
    orphs2 = app.discover_picksets(root, app.load_jobs(root))
    check("existing job hides its suffix",
          {o["suffix"] for o in orphs2} == {"_emd_70905"})

    # orphans render as extra 'orphan' cards in the ts_template_match row
    nodes, _ = app.canvas_layout(app.load_jobs(root), orphs)
    onodes = [n for n in nodes if n.get("is_orphan")]
    check("orphan nodes created", len(onodes) == len(orphs))
    check("orphan node styling", onodes and onodes[0]["status"] == "orphan"
          and onodes[0]["stage_id"] == "ts_template_match")
    check("orphan sits beside the real job in its row",
          onodes and onodes[0]["row"] == next(n["row"] for n in nodes
                                              if n["stage_id"] == "ts_template_match"
                                              and not n.get("is_orphan"))
          and onodes[0]["col"] >= 1)
    check("canvas_layout without orphans still works",
          app.canvas_layout(app.load_jobs(root)) is not None)

# ---- downstream wiring: derive child params from a parent job (pure) -------
# the WHOLE middle is the threshold in_suffix, not just the override suffix
check("match_star_infix full middle",
      app.match_star_infix({"tomo_angpix": "12.56", "override_suffix": "_v3-optimized"})
      == "12.56Apx_v3-optimized")
check("match_star_infix from template",
      app.match_star_infix({"tomo_angpix": "10", "template_emdb": "70905"})
      == "10.00Apx_emd_70905")
check("threshold inherits in_suffix from template match",
      app.derive_child_params("threshold_picks", "ts_template_match",
                              {"tomo_angpix": "12.56", "override_suffix": "_v3-optimized"})
      == {"in_suffix": "12.56Apx_v3-optimized"})
exp_from_thr = app.derive_child_params(
    "ts_export_particles", "threshold_picks",
    {"in_suffix": "12.56Apx_v3-optimized", "out_suffix": "clean"}, "jobs/J8")
check("export reads the parent job's matching dir",
      exp_from_thr["input_directory"] == "jobs/J8/matching")
check("export pattern from threshold",
      exp_from_thr["input_pattern"] == "*12.56Apx_v3-optimized_clean.star")
check("export writes into a pick-set-named RELION dir",
      exp_from_thr["output_processing"] == "relion4/v3-optimized"
      and exp_from_thr["output_star"] == "relion4/v3-optimized/matching.star")
check("_picktag strips angpix prefix", app._picktag("12.56Apx_v3-optimized") == "v3-optimized")
check("_picktag from emd", app._picktag("10.00Apx_emd_70905") == "emd_70905")
check("export dir falls back to trunk without a parent dir",
      app.derive_child_params("ts_export_particles", "threshold_picks",
                              {"in_suffix": "x", "out_suffix": "clean"})
      .get("input_directory") == "warp_tiltseries/matching")

# convert inherits the export's RELION dir + star name; Class3D inherits the
# converted star — so the whole handoff stays in relion4/<tag>/
conv = app.derive_child_params("relion4_convert", "ts_export_particles",
                               {"output_processing": "relion4/v3-optimized",
                                "output_star": "relion4/v3-optimized/matching.star"})
check("convert inherits project dir + star",
      conv == {"project_dir": "relion4/v3-optimized", "starfile": "matching.star"})
cls = app.derive_child_params("relion4_class3d", "relion4_convert",
                              {"project_dir": "relion4/v3-optimized", "starfile": "matching.star"})
check("Class3D inherits dir + the CONVERTED star",
      cls == {"project_dir": "relion4/v3-optimized", "particles": "matching_conv.star"})
check("no derivation for unrelated pair",
      app.derive_child_params("ts_reconstruct", "ts_ctf", {}) == {})
check("DOWNSTREAM edges present",
      "threshold_picks" in app.DOWNSTREAM["ts_template_match"]
      and "ts_export_particles" in app.DOWNSTREAM["threshold_picks"])

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
