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
    # Named, not bare: the dir says what the job DID (jobs/J1_<stage-slug>),
    # matching the convention users were already applying by hand.
    check("new_job output_dir is named",
          j1["output_dir"].startswith("jobs/J1_")
          and len(j1["output_dir"]) > len("jobs/J1_"))
    # A new job is BUILDING, not queued: a card nobody has finished configuring
    # must not start on its own the moment the queue drains.
    check("new_job status building", j1["status"] == "building")
    check("a building job is not in the run queue",
          j1["id"] not in [q["id"] for q in app.queued_jobs(app.load_jobs(root))])
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
    check("relion4_convert not warp", not app.is_warp_stage(stage("relion4_convert")))

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

# A stage's status callable runs on EVERY canvas repaint. A lambda naming a
# ProjectState method that does not exist raises AttributeError the moment the
# canvas draws — never at import, and never in a test that only exercises the
# stages it knows about. Call every one of them against an empty project.
import tempfile as _tf
with _tf.TemporaryDirectory() as _td:
    _ps = app.ProjectState(Path(_td))
    _broken = []
    for _s in app.STAGES:
        _fn = _s.get("status")
        if _fn is None:
            continue
        try:
            _n, _msg = _fn(_ps)
            if not isinstance(_n, int):
                _broken.append(f"{_s['id']}: returned {type(_n).__name__}, not int")
        except Exception as _e:                                   # noqa: BLE001
            _broken.append(f"{_s['id']}: {type(_e).__name__}: {_e}")
    check(f"every stage status runs on an empty project{(' — ' + '; '.join(_broken)) if _broken else ''}",
          not _broken)
# The form label is param_title(p); the raw wire name (MB_XML_DIR,
# output_angpix) lives in the help line. Every param carries an explicit
# human title so the prettifier fallback never has to guess one.
check("every param has an explicit human title",
      all(p.get("title") for s in app.STAGES for p in s.get("params", [])))

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
    # Retired stages (legacy=True) get a row only while a project still holds
    # their cards, so an empty project shows one template per LIVE stage.
    n_stages = len([x for x in app.STAGES if not x.get("legacy")])

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
    check("every live stage has exactly one template card",
          len([n for n in nodes if n.get("is_template")]) == n_stages)
    check("the template sits in the rail at x=0", tmpl_row[0]["x"] == 0)
    # The rail carries NO job state. A template that reports 'completed' is one
    # that every status-driven path — colouring, details, the running check —
    # treats as finished work, and the eye can no longer tell template from job.
    check("no template ever reports a job status",
          all(n["status"] == "ghost" for n in nodes if n.get("is_template")))
    check("templates are never 'completed'",
          not any(n["status"] == "completed" for n in nodes if n.get("is_template")))
    check("a template says how many jobs its stage has",
          tmpl_row[0].get("n_jobs") == 2)
    check("an unworked stage reports none",
          next(n for n in nodes
               if n.get("is_template") and n["stage_id"] == "aretomo")["n_jobs"] == 0)
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

# The comma-separated counterpart. This used to be checked on relion4_class3d's
# GPUS; that stage is gone (removed 2026-08-21 — never run through tomogration),
# but the normalisation is a property of gpu_sep, not of any one card.
rel = next(s for s in app.STAGES if s["id"] == "mb_isonet_train")
vals = app.stage_defaults(rel)
vals["ISO_GPU"] = "0 1 2 3"                            # user typed spaces
cmd = app.build_command(rel, vals)
check("a comma-separated GPU list is normalised from spaces",
      "ISO_GPU=0,1,2,3" in cmd)

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
# Named for the PICK SET so it is recognisable, and suffixed with {jobid} so two
# exports of the same pick set never share a directory. Sharing one is what made
# "clear this job" able to delete a DIFFERENT job's particles — the reason
# clearing-and-re-running could not be offered safely at all.
check("export writes into a pick-set-named, job-unique RELION dir",
      exp_from_thr["output_processing"] == "relion4/v3-optimized_{jobid}"
      and exp_from_thr["output_star"]
      == "relion4/v3-optimized_{jobid}/matching.star")
check("the star still sits inside the processing dir",
      exp_from_thr["output_star"].startswith(
          exp_from_thr["output_processing"] + "/"))
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
# The relion4_convert -> relion4_class3d derivation went with that stage
# (removed 2026-08-21). Nothing downstream of convert derives params now, and
# a rule for a stage that no longer exists must not linger silently.
check("convert has no derivation left to a removed child",
      app.derive_child_params("relion4_class3d", "relion4_convert",
                              {"project_dir": "relion4/v3-optimized",
                               "starfile": "matching.star"}) == {})
check("and convert's only remaining child is merge optics",
      app.DOWNSTREAM["relion4_convert"] == ["relion4_merge_optics"])
check("no derivation for unrelated pair",
      app.derive_child_params("ts_reconstruct", "ts_ctf", {}) == {})
check("DOWNSTREAM edges present",
      "threshold_picks" in app.DOWNSTREAM["ts_template_match"]
      and "ts_export_particles" in app.DOWNSTREAM["threshold_picks"])


# ---- work that ran before the job model still gets a card -------------------
# ts_stack, aretomo, ts_import_alignments and ts_ctf were all run on the trunk or
# from the terminal: real output, no job record. Saying "output on disk" on the
# template was not enough — those steps DID run, and a pipeline showing nothing
# for them reads as one that never started.
with tempfile.TemporaryDirectory() as tmp:
    droot = Path(tmp)
    st_map = {"ts_stack": (True, "290 stacks"), "aretomo": (True, "286 .xf")}
    nodes, edges = app.canvas_layout(app.load_jobs(droot), [], st_map)
    idx = {n["id"]: n for n in nodes}

    check("a stage with on-disk output gets its own card", "disk:ts_stack" in idx)
    d = idx["disk:ts_stack"]
    check("it reads as completed", d["status"] == "completed")
    check("it carries what was found", "290 stacks" in d.get("subtitle", ""))
    check("it is flagged as discovered", d.get("is_discovered") is True)
    check("it sits in the working canvas, not the rail", d["x"] == app.RAIL_W)
    check("its stage's template is still in the rail",
          idx["ghost:ts_stack"]["x"] == 0)
    check("the template does not overlap it",
          idx["ghost:ts_stack"]["x"] + app.CARD_W <= d["x"])
    check("a stage with nothing on disk gets no such card",
          "disk:ts_ctf" not in idx)

    # It must not masquerade as a tracked job: no id in the store, and the ghost
    # menu (build a real job from this stage) is the right one for it.
    check("it is not a job record",
          "disk:ts_stack" not in app.load_jobs(droot)["jobs"])
    check("it uses the ghost menu branch", d["is_ghost"] is True)

    # Once a stage HAS jobs, the jobs are the truth — no duplicate disk card.
    app.new_job(droot, "ts_stack", "stack", {})
    idx2 = {n["id"]: n for n in app.canvas_layout(app.load_jobs(droot), [], st_map)[0]}
    check("a stage with real jobs shows no disk card", "disk:ts_stack" not in idx2)
    check("but the other discovered stage is untouched", "disk:aretomo" in idx2)

    # Hiding one works like hiding any card.
    idx3 = {n["id"]: n for n in
            app.canvas_layout(app.load_jobs(droot), [], st_map, {"disk:aretomo"})[0]}
    check("a discovered card can be hidden", "disk:aretomo" not in idx3)


# ---- store resilience: what a crash mid-write must NOT be able to destroy ----
with tempfile.TemporaryDirectory() as tmp:
    sroot = Path(tmp)
    app.new_job(sroot, "ts_ctf", "CTF", {})
    app.new_job(sroot, "ts_reconstruct", "Recon", {})

    # A corrupt store is quarantined (renamed aside), never left in place for the
    # next save to overwrite: the scaffold + one card drag used to equal total loss.
    app.jobs_path(sroot).write_text("{ this is not json")
    check("corrupt store loads as scaffold",
          app.load_jobs(sroot) == {"seq": 0, "jobs": {}})
    check("corrupt store was renamed aside",
          not app.jobs_path(sroot).is_file()
          and list(sroot.glob(".tomogration_jobs.json.corrupt_*")))

    # No .tmp droppings: save goes through temp + atomic replace.
    app.new_job(sroot, "ts_ctf", "CTF", {})
    check("atomic save leaves no temp file",
          not list(sroot.glob("*.tmp")) and app.jobs_path(sroot).is_file())

    # 'seq' inference must never re-issue a live id. A store missing 'seq' with
    # deletions (J1 gone, J5 alive) has len(jobs)=1 — a naive default would hand
    # out J2..J5 again and silently clobber J5's record.
    app.jobs_path(sroot).write_text(
        '{"jobs": {"J5": {"id": "J5", "stage_id": "ts_ctf", "status": "completed"}}}')
    check("seq inferred from max J-number", app.load_jobs(sroot)["seq"] == 5)
    jn = app.new_job(sroot, "ts_ctf", "CTF", {})
    check("new id after inference does not collide", jn["id"] == "J6")
    check("the surviving job is untouched",
          app.load_jobs(sroot)["jobs"]["J5"]["status"] == "completed")

    # An interrupted job gets a finished stamp, or versions_for_job attributes
    # every later M version folder to it forever.
    app.update_job(sroot, "J6", status="running",
                   started="2026-01-01 00:00:00", finished=None)
    stale = app.reconcile_running(sroot)
    j6 = app.load_jobs(sroot)["jobs"]["J6"]
    check("reconcile marks interrupted", "J6" in stale and j6["status"] == "failed")
    check("reconcile stamps finished", bool(j6.get("finished")))


# Every DOWNSTREAM edge must name a REAL stage on both ends. A stage insert once
# failed while the DOWNSTREAM edit that accompanied it succeeded, leaving the map
# pointing at a stage that did not exist — the menu would have offered a build that
# could never resolve.
_ids = {x["id"] for x in app.STAGES}
_bad = sorted({f"{k} -> {c}" for k, v in app.DOWNSTREAM.items() for c in v
               if k not in _ids or c not in _ids})
check(f"DOWNSTREAM names only real stages{(' — ' + ', '.join(_bad)) if _bad else ''}",
      not _bad)

# ---- hand-set job status ----------------------------------------------------
# An exit code answers "did the process return 0", which is not "did this work".
# Both failure modes are real and both happened on 2026-08-16: a run that exited
# 0 after silently reusing another run's data, and a training run that finished
# all 50 epochs and then died on an X11 teardown. So the verdict can be set by
# hand — but the machine's record must survive underneath it.
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    (root / ".tomogration_jobs.json").write_text('{"seq":0,"jobs":{}}')
    j = app.new_job(root, "mb_isonet2_train", "IsoNet 2: train", {})
    jid = j["id"]

    # A BUILDING card can be judged too: a card whose work was actually done
    # outside tomogration, or one that is a dead end, should not keep reading
    # as outstanding work. Restoring it returns it to building.
    ok, _msg = app.set_manual_status(root, jid, "completed",
                                     reason="ran it from the terminal")
    st = app.load_jobs(root)["jobs"][jid]
    check("a building job can be marked completed by hand", ok)
    check("and it says so on the card", st["status"] == "completed")
    check("with building kept underneath as what really happened",
          st["manual_status"]["was"] == "building")
    check("the note reports it honestly",
          "actually building" in app.manual_status_note(st))
    ok, _msg = app.clear_manual_status(root, jid)
    st = app.load_jobs(root)["jobs"][jid]
    check("restoring a marked building card returns it to building",
          ok and st["status"] == "building")

    app.update_job(root, jid, status="queued")
    ok, msg = app.set_manual_status(root, jid, "failed")
    check("a queued job cannot be judged (dequeue it first)",
          not ok and "Dequeue it first" in msg)

    app.update_job(root, jid, status="running")
    ok, msg = app.set_manual_status(root, jid, "failed")
    check("running job cannot be judged (kill it first)",
          not ok and "Kill it first" in msg)

    app.update_job(root, jid, status="completed", exit_code=0)
    ok, msg = app.set_manual_status(root, jid, "failed", reason="reused a stale star")
    st = app.load_jobs(root)["jobs"][jid]
    check("a completed job can be marked failed", ok)
    check("the card colour follows the mark", st["status"] == "failed")
    check("the machine's verdict survives underneath",
          st["manual_status"]["was"] == "completed"
          and st["manual_status"]["exit_code"] == 0)
    check("the exit code itself is never rewritten", st["exit_code"] == 0)
    check("the reason is kept", "stale star" in st["manual_status"]["reason"])
    check("the note says what actually happened",
          "actually completed" in app.manual_status_note(st))

    # Marking twice must not lose the original: 'was' is the MACHINE's verdict.
    app.set_manual_status(root, jid, "completed")
    st = app.load_jobs(root)["jobs"][jid]
    check("re-marking keeps the original machine verdict",
          st["manual_status"]["was"] == "completed" and st["status"] == "completed")
    app.set_manual_status(root, jid, "failed")
    ok, msg = app.clear_manual_status(root, jid)
    st = app.load_jobs(root)["jobs"][jid]
    check("restore puts back what the machine recorded",
          ok and st["status"] == "completed" and "manual_status" not in st)
    ok, msg = app.clear_manual_status(root, jid)
    check("restoring an unmarked job says so", not ok and "no hand-set" in msg)

    ok, msg = app.set_manual_status(root, jid, "queued")
    check("only completed/failed can be set by hand", not ok)

    # A fresh run supersedes the mark — otherwise the old judgement labels it.
    app.set_manual_status(root, jid, "failed")
    app.update_job(root, jid, status="running")
    st = app.load_jobs(root)["jobs"][jid]
    check("re-running drops the hand-set verdict", "manual_status" not in st)

    # The canvas must be able to SHOW that a colour is an opinion.
    app.update_job(root, jid, status="completed")
    app.set_manual_status(root, jid, "failed")
    store = app.load_jobs(root)
    nodes, _ = app.canvas_layout(store, [], {}, set())
    node = next(n for n in nodes if n.get("id") == jid)
    check("the node is flagged as hand-set for the card", node.get("manual") is True)


# ---- canvas annotations: notes and branch frames ----------------------------
# They carry NO data meaning. Nothing that RUNS may ever consult them — they
# exist so a canvas with five IsoNet variants on it can say which is which.
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    (root / ".tomogration_jobs.json").write_text('{"seq":0,"jobs":{}}')
    check("a project with no annotations reads as empty", app.load_notes(root) == [])

    f = app.add_note(root, "frame", 100, 50, "bin4 n2n", colour="violet",
                     w=600, h=400)
    n = app.add_note(root, "note", 120, 500, "polarity is dark here")
    check("ids come from their own counter, never a job's",
          f["id"] == "N1" and n["id"] == "N2")
    check("a frame gets frame defaults, a note note ones",
          (f["w"], f["h"]) == (600, 400) and (n["w"], n["h"]) == (220.0, 96.0))
    check("colour is kept", f["colour"] == "violet")
    check("an unknown colour falls back rather than reaching the painter",
          app.add_note(root, "note", 0, 0, colour="octarine")["colour"] == "amber")
    try:
        app.add_note(root, "scribble", 0, 0)
        ok = False
    except ValueError:
        ok = True
    check("an unknown KIND is refused (it would be drawn in the wrong layer)", ok)

    app.update_note(root, n["id"], x=999, text="edited", bogus="ignored")
    got = [x for x in app.load_notes(root) if x["id"] == n["id"]][0]
    check("notes move and retitle", got["x"] == 999 and got["text"] == "edited")
    check("unknown fields are dropped, not stored", "bogus" not in got)

    frames, sticky = app.notes_for_canvas(app.load_jobs(root))
    check("frames and notes are separated for painting (frames behind)",
          [x["id"] for x in frames] == ["N1"] and len(sticky) == 2)

    # A frame is 'about' whatever sits inside it, computed from geometry — so
    # dragging a card into a branch needs no bookkeeping.
    nodes = [{"id": "J1", "x": 200, "y": 100, "w": 100, "h": 60},
             {"id": "J2", "x": 2000, "y": 100, "w": 100, "h": 60},
             {"id": None, "x": 200, "y": 120, "w": 100, "h": 60, "is_ghost": True}]
    check("cards_inside finds the cards a frame covers",
          app.cards_inside(f, nodes) == ["J1"])
    check("and ignores ghosts, which are template furniture",
          "J3" not in app.cards_inside(f, nodes))

    check("deleting one reports success", app.delete_note(root, n["id"]) is True)
    check("deleting it twice does not", app.delete_note(root, n["id"]) is False)
    check("the others survive",
          {x["id"] for x in app.load_notes(root)} == {"N1", "N3"})

    # Annotations must never disturb the job model.
    j = app.new_job(root, "ts_ctf", "CTF", {})
    check("job ids are unaffected by the note counter", j["id"] == "J1")
    check("and the notes are still there after a job write",
          len(app.load_notes(root)) == 2)


# ---- drag-to-connect: the cycle guard ---------------------------------------
# set_job_parent already refused self-parenting, but a longer loop (A feeds B,
# then B wired to feed A) is just as impossible and the canvas would draw it
# forever. The gesture makes that easy to attempt by accident.
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    (root / ".tomogration_jobs.json").write_text('{"seq":0,"jobs":{}}')
    a = app.new_job(root, "ts_ctf", "A", {})["id"]
    b = app.new_job(root, "ts_reconstruct", "B", {}, inputs={"processing": a})["id"]
    c = app.new_job(root, "ts_template_match", "C", {}, inputs={"processing": b})["id"]
    store = app.load_jobs(root)
    check("feeding A from C closes the loop A->B->C",
          app.would_cycle(store, a, c) is True)
    check("feeding A from B closes the shorter loop",
          app.would_cycle(store, a, b) is True)
    check("a job cannot feed itself", app.would_cycle(store, a, a) is True)
    check("an unrelated pair is fine", app.would_cycle(store, c, a) is False)
    check("set_job_parent REFUSES the cycle rather than storing it",
          app.set_job_parent(root, a, c) is None)
    check("and the wiring is untouched after the refusal",
          (app.load_jobs(root)["jobs"][a].get("inputs") or {}) == {})
    check("a legal rewire still works",
          app.set_job_parent(root, c, a) is not None
          and app.load_jobs(root)["jobs"][c]["inputs"]["processing"] == a)

    # An already-cyclic store must TERMINATE: would_cycle is what the UI calls
    # to describe the graph, and hanging there would be worse than the cycle.
    bad = app.load_jobs(root)
    bad["jobs"][a]["inputs"] = {"processing": b}
    bad["jobs"][b]["inputs"] = {"processing": a}
    app.save_jobs(root, bad)
    check("a store that is already looped does not hang the check",
          app.would_cycle(app.load_jobs(root), c, a) is False)


# ---- membrane branch: build downstream ------------------------------------
# Every membrane step reads a folder its parent wrote. That folder is the
# PARENT JOB's own dir (jobs/J35), which no field on the form shows and the
# card's default points somewhere else entirely — so wiring it by hand meant
# asking "which folder did the last one write to?" at every step.
_d = app.derive_child_params
_seg = _d("mb_thresholds", "mb_segment", {"MB_TOMO_LIST": "Position003"}, "jobs/J35")
check("segment -> thresholds reads the segment JOB's folder, not the card default",
      _seg["input_dir"] == "jobs/J35")
check("and the child writes to its own job dir",
      _seg["output_dir"] == "jobs/{jobid}")
check("a parent restricted to one tomogram does not hand its child all 72",
      _seg["MB_TOMO_LIST"] == "Position003")
check("thresholds -> components chains the same way",
      _d("mb_components", "mb_thresholds", {}, "jobs/J36")["input_dir"] == "jobs/J36")
# Parents that nest their output must be followed INTO the subfolder.
check("isonet2 predict -> segment enters corrected/",
      _d("mb_segment", "mb_isonet2_predict", {}, "jobs/J31")["input_dir"]
      == "jobs/J31/corrected")
check("reconstruct -> deconv enters reconstruction/",
      _d("mb_deconv", "ts_reconstruct", {}, "jobs/J16")["input_dir"]
      == "jobs/J16/reconstruction")
# A deconv SWEEP writes one folder per value; the first is the only defensible
# default, and naming it makes the choice visible in the child's field.
check("deconv -> segment picks the first swept strength/falloff folder",
      _d("mb_segment", "mb_deconv", {"MB_STRENGTH": "0.5 1.0 1.5"},
         "jobs/J12")["input_dir"] == "jobs/J12/s0.5_f1.0")
# The components job root holds NO volumes — only cc<voxels>/ subfolders. Mesh
# took the root and died with "no *.mrc in jobs/J37" (2026-08-18); the rule now
# lives in one place so both children get it.
check("components -> mesh enters cc<voxels>/, not the job root",
      _d("mb_mesh", "mb_components", {"MB_CC_THRES": "50"},
         "jobs/J37")["input_dir"] == "jobs/J37/cc50")
check("components -> fit aims at the first cutoff folder, not the job root",
      _d("mb_fit_virions", "mb_components",
         {"MB_CC_THRES": "1000 10000", "input_dir": "jobs/J36"},
         "jobs/J37")["components"] == "jobs/J37/cc1000")
# THE bug this chain makes easy: everything after segmentation reads MASKS, so
# by the time you reach mesh or fit-virions the greyscale volumes are three hops
# back. Taking the immediate parent's input_dir handed the mesh a folder of
# binary thresholds and gave the fitter's density-support gate a mask to measure
# densities in — which would have run, and been meaningless.
_GREY = "jobs/J31-bin8-isonet2model/corrected"
_thr = _d("mb_thresholds", "mb_segment", {"input_dir": _GREY}, "jobs/J35")
_comp = _d("mb_components", "mb_thresholds", _thr, "jobs/J36")
check("the greyscale tomograms are carried down the chain, not re-derived",
      _comp[app.GREY_KEY] == _GREY)
check("mesh gets the GREYSCALE tomograms, not the thresholds folder",
      _d("mb_mesh", "mb_components", _comp, "jobs/J37")["tomo_dir"] == _GREY)
check("and fit-virions measures density in the greyscale volume, not a mask",
      _d("mb_fit_virions", "mb_components", _comp, "jobs/J37")["tomogram"] == _GREY)
check("an isonet predict parent contributes its CORRECTED output, not its input",
      _d("mb_segment", "mb_isonet2_predict",
         {"input_dir": "warp_tiltseries/reconstruction",
          "output_dir": "jobs/J31/corrected"}, "jobs/J31")[app.GREY_KEY]
      == "jobs/J31/corrected")
check("the breadcrumb never reaches a command line",
      app.GREY_KEY.startswith("_"))
check("fit -> pick surfaces reads fits.json",
      _d("mb_pick_surfaces", "mb_fit_virions", {}, "jobs/J38")["fits"]
      == "jobs/J38/fits.json")
# The menu must actually offer them.
for _p, _c in (("mb_segment", "mb_thresholds"), ("mb_thresholds", "mb_components"),
               ("mb_components", "mb_fit_virions"), ("mb_fit_virions", "mb_pick_surfaces")):
    check(f"{_p} offers {_c} downstream", _c in app.DOWNSTREAM.get(_p, []))
check("reconstruct still offers template matching as well as the membrane path",
      "ts_template_match" in app.DOWNSTREAM["ts_reconstruct"]
      and "mb_deconv" in app.DOWNSTREAM["ts_reconstruct"])


# The rail lists stages in STAGES order, so that order IS the suggested order of
# work. Fit virions must come after everything it depends on: polarity gives its
# density gate a sign, population gives it a radius, and Split gives it a volume
# whose merged virions have been separated. It sat directly after meshes, above
# all three.
_mb = [s["id"] for s in app.STAGES if s["group"] == "12. Membrane"]
for _dep in ("mb_polarity", "mb_population", "mb_split_clusters"):
    check(f"{_dep} is offered before Fit virions",
          _mb.index(_dep) < _mb.index("mb_fit_virions"))
check("and Fit still comes before what consumes its fits",
      _mb.index("mb_fit_virions") < _mb.index("mb_pick_surfaces"))
check("every membrane stage a job can be built from is still listed once",
      len(_mb) == len(set(_mb)))

# A job owns its output folder. A fixed default means the second run silently
# overwrites the first and two jobs claim the same directory. Only the A3
# measurement cards are held to this: the older pipeline stages default to
# membrane/<thing> by an established convention, with jobs already on disk
# pointing there, and churning those would be a change nobody asked for.
for _sid in ("mb_population", "mb_split_clusters", "mb_fit_virions"):
    _spec = next(s for s in app.STAGES if s["id"] == _sid)
    _out = {p["name"]: p.get("default") for p in _spec["params"]}
    for _key in (_spec.get("output_params") or []):
        check(f"{_sid} writes into its own job folder by default",
              "{jobid}" in str(_out.get(_key, "")))


# A RELION -> Warp re-extract and a crYOLO pick set are both STORED as
# ts_template_match jobs, deliberately, so the Export wiring and adoption work.
# The canvas titles them by what they are; the builder resolves by stage_id and
# so calls all three "Template matching". source_star is a BREADCRUMB that
# stage never declared — reporting it as a parameter the stage "no longer has"
# read as upgrade damage when nothing was ever lost, and buried the one value
# that answers "which job did these particles come from".
_reex = {"id": "J107", "stage_id": "ts_template_match", "tool": "reextract",
         "params": {"override_suffix": "_picks_class2_HE", "tomo_angpix": 6.28,
                    "source_star": "relion4/Select/job009/particles.star"}}
check("a re-extract is named for what it is, not the stage it rides",
      app.actual_job_kind(_reex) == "a RELION → Warp re-extract")
check("so is a crYOLO pick set",
      app.actual_job_kind({"stage_id": "ts_template_match", "tool": "cryolo"})
      == "a crYOLO pick set")
check("a real template match claims to be nothing else",
      app.actual_job_kind({"stage_id": "ts_template_match", "params": {}}) == "")

_spec = next(x for x in app.STAGES if x["id"] == "ts_template_match")
_kept, _dropped, _missing = app.params_for_builder(_spec, _reex["params"])
check("ts_template_match never declared source_star", "source_star" in _dropped)
_carried, _stale = app.carried_note(_dropped, _reex["params"])
check("and it is reported as a breadcrumb, not as a stale parameter",
      _carried.get("source_star") == "relion4/Select/job009/particles.star"
      and not _stale)
check("the breadcrumb says what it is for",
      "re-extracted from" in app.CARRIED_PARAMS["source_star"])
check("a genuinely removed parameter is still called out",
      app.carried_note(["some_old_flag"], {"some_old_flag": 1})[1]
      == ["some_old_flag"])


# A job's FOLDER must say what the job is. Three kinds ride ts_template_match so
# their downstream wiring works, and the folder name was taken from the stage
# alone: J47 adapted crYOLO picks and got jobs/J47_ts-template-match, a name
# that misleads anyone reading the directory later — including whoever made it.
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    (root / ".tomogration_jobs.json").write_text('{"seq":0,"jobs":{}}')
    cry = app.new_job(root, "ts_template_match", "crYOLO picks 'x'", {},
                      kind="cryolo")
    rex = app.new_job(root, "ts_template_match", "re-extract", {},
                      kind="reextract")
    sel = app.new_job(root, "relion4_result", "Subset selection", {},
                      kind="relion_selection")
    tm = app.new_job(root, "ts_template_match", "Template matching", {})

    check("a crYOLO conversion's folder says crYOLO",
          cry["output_dir"] == f"jobs/{cry['id']}_cryolo-picks")
    check("a re-extract's says re-extract",
          rex["output_dir"] == f"jobs/{rex['id']}_relion-reextract")
    check("a selection's says selection",
          sel["output_dir"] == f"jobs/{sel['id']}_relion-selection")
    check("and a REAL template match still says template match",
          tm["output_dir"] == f"jobs/{tm['id']}_ts-template-match")

    # The folder name and the card label are derived from two different fields
    # (kind at creation, tool afterwards). If those vocabularies drift, a card
    # says one thing and its directory another.
    check("every folder-name kind is a tool the canvas also recognises",
          all(app.actual_job_kind({"tool": k}) for k in app.KIND_DIR_SLUG))

    # {jobid} resolves to the folder's own basename, so paths built from it
    # follow the new names rather than reconstructing jobs/J47.
    check("{jobid} follows the named folder",
          app.job_dir_token(cry) == f"{cry['id']}_cryolo-picks")
    check("a record predating the naming still resolves to its bare id",
          app.job_dir_token({"id": "J10", "output_dir": "jobs/J10"}) == "J10")
    check("and an already-written folder name is never recomputed",
          app.load_jobs(root)["jobs"][cry["id"]]["output_dir"]
          == cry["output_dir"])

    # The outputs breadcrumb (jobs/<id>/outputs -> the real output dir) used to
    # reconstruct a BARE jobs/J47, creating a second nearly-empty folder next
    # to the real jobs/J47_cryolo-picks and pointing the trail at the one
    # nothing else used.
    st = app.load_jobs(root)
    (root / cry["output_dir"]).mkdir(parents=True)
    check("the real folder is found from the record",
          app.job_real_dir(root, cry["id"], "ts_template_match", st)
          == cry["output_dir"])
    check("a hand-renamed folder is still found by J-number",
          app.job_real_dir(root, "J404", "ts_template_match",
                           {"jobs": {}}) == "jobs/J404_ts-template-match")
    (root / "jobs" / f"{rex['id']}_renamed-by-hand").mkdir(parents=True)
    check("even when the record disagrees with the disk",
          app.job_real_dir(root, rex["id"], "ts_template_match",
                           {"jobs": {rex["id"]: {"output_dir": "jobs/gone"}}})
          == f"jobs/{rex['id']}_renamed-by-hand")


# The convert card's project folder is the field users get wrong, and the old
# help ("= ts_export_particles' output_processing", "the launch-root
# invariant") named the mechanism rather than the folder. The default is the
# relion4/ CONTAINER, which holds one folder per export and contains no star of
# its own, so it fails at run time with a message about a missing file.
_CONV = next(x for x in app.STAGES if x["id"] == "relion4_convert")
_pd = next(q for q in _CONV["params"] if q["name"] == "project_dir")
check("the help says where to FIND the folder, not just what it equals",
      "Extract particles" in _pd["help"] and "subtomo/" in _pd["help"])
check("it warns against the bare container",
      "NOT the bare relion4/" in _pd["help"])
check("and points at build-downstream as the reliable route",
      "Build downstream" in _pd["help"])
check("the bare container is refused before the run, not during",
      _CONV["validate"]({"project_dir": "relion4"}).startswith("\u26a0"))
check("a trailing slash does not sneak it past",
      _CONV["validate"]({"project_dir": "relion4/"}) != "")
check("a real export folder passes",
      _CONV["validate"]({"project_dir": "relion4/cryolo_isonetmodel_J48"}) == "")
check("the export stage still names the same field, so the two agree",
      any(q["name"] == "output_processing"
          for q in next(x for x in app.STAGES
                        if x["id"] == "ts_export_particles")["params"]))


# A failed run left a .NET stack trace and nothing else. The trace names the C#
# file that threw, which says nothing about what to change on the card. J48
# failed this way twice in a row on 2026-08-28.
_trace = ("Connecting to workers...\n"
          "Unhandled exception. \n"
          "System.TimeoutException: Worker process did not connect within the "
          "allotted time.\n"
          "   at Warp.WorkerWrapper.ListenForPort(String pipeName, Int32 "
          "timeoutMilliseconds)")
_h = app.run_failure_hint(_trace)
check("a worker timeout is explained, not just echoed", _h != "")
check("it names the setting to change",
      "perdevice" in _h.lower() or "workers per device" in _h.lower())
check("it says to check the GPUs are free first", "nvidia-smi" in _h)
# The first version asserted "4 devices x 2 = 8" as if it knew the run's
# settings. It printed that verbatim under a run using perdevice 1, and sent
# the reader after a cause (GPU contention) that nvidia-smi then ruled out.
check("the hint states no number it cannot know",
      not any(t in _h for t in ("4 devices", "= 8", "x 2")))
check("it covers the IDLE case, which is a different fault entirely",
      "idle" in _h.lower() and "launch" in _h.lower())
check("and gives a bisection rather than more guessing",
      "device list" in _h.lower() and "WarpWorker" in _h)
# The bisection came back "fails identically on one GPU", which rules out
# contention and concurrency both. What remains is the worker never starting:
# a missing binary, a stale worker holding the pipe, or the pipe itself being
# unable to exist. .NET named pipes are Unix sockets under TMPDIR.
check("it covers the pipe itself, not only the GPU and the binary",
      "/tmp" in _h and "TMPDIR" in _h)

# An input_pattern that matches nothing makes WarpTools build an empty table
# and die inside Star..ctor with IndexOutOfRangeException — a stack trace that
# names a C# file and says nothing about the glob. 2026-08-31.
_empty = app.run_failure_hint(
    "Found 0 files in warp_tiltseries/matching_reextract matching "
    "*reextract_2.star;\nUnhandled exception.\n"
    "System.IndexOutOfRangeException: Index was outside the bounds of the "
    "array.\n   at Warp.Star..ctor(Star[] tables) in ...Star.cs:line 19")
check("an empty pick-star match is explained, not left as a stack trace",
      _empty and "matched nothing" in _empty)
check("it points at the pattern rather than the C# frame",
      "suffix" in _empty and "Star..ctor" in _empty)
check("and says re-running after a fix is safe",
      "safe" in _empty)
check("the crash alone, with files found, is not claimed to be this",
      not (app.run_failure_hint("System.IndexOutOfRangeException at foo") or "")
      .startswith("The pick-star PATTERN"))
# 'which WarpWorker' came back empty on 2026-08-28, which LOOKS decisive but
# is not: Warp may resolve the worker beside its own assembly rather than off
# PATH. The hint has to say so, or an empty 'which' reads as a diagnosis.
# The worker binds a TCP port for a REST API and reports it back over a named
# pipe; the master waits on that. Since the master swallows the worker's
# stdout, starting one by hand is the only way to see the actual error — and
# it is what finally produced one on 2026-08-28.
check("it says how to see the error the master swallows",
      "WarpWorker --device 0 --port" in _h)
check("and warns that the env must be active, which cost a round trip",
      "activate the warp env" in _h.lower())
check("and that re-running is safe, since nothing was written",
      "re-running is safe" in _h.lower())
check("a clean line produces no hint", app.run_failure_hint("72 files found") == "")
check("no hint for an empty stream", app.run_failure_hint("") == ""
      and app.run_failure_hint(None) == "")
check("matching ignores case",
      app.run_failure_hint("WORKER PROCESS DID NOT CONNECT") != "")
check("an out-of-memory failure is explained too",
      "memory" in app.run_failure_hint("CUDA out of memory").lower())
check("every hint says what to DO, not just what happened",
      all(any(w in h.lower() for w in ("lower", "check", "re-run"))
          for _sig, h in app.RUN_FAILURE_HINTS))


# A pick-set card is a MARKER: crYOLO/re-extract/selection folders hold star
# files, not Warp's per-series .xml results. Exporting downstream of one passed
# --input_processing jobs/J47_ts-template-match, telling Warp to look for
# previous results in a folder that has none. threshold_picks was already
# exempted from exactly this trap; pick-set parents were not.
_exp = next(x for x in app.STAGES if x["id"] == "ts_export_particles")


def _io(parent):
    job = {"id": "J48", "stage_id": "ts_export_particles",
           "output_dir": "jobs/J48_ts-export-particles", "inputs": {"in": "J47"}}
    return app.io_flags_for_job(_exp, job, {"jobs": {"J47": parent}})


for _tool in ("cryolo", "reextract", "relion_selection"):
    _f = _io({"id": "J47", "stage_id": "ts_template_match", "tool": _tool,
              "output_dir": "jobs/J47_x"})
    check(f"a {_tool} parent contributes no --input_processing",
          "--input_processing" not in _f)
    check(f"but the export still gets its own --output_processing ({_tool})",
          "--output_processing jobs/J48_ts-export-particles" in _f)

_real = _io({"id": "J47", "stage_id": "ts_template_match",
             "output_dir": "jobs/J47_ts-template-match"})
check("a REAL template match still wires its processing dir through",
      "--input_processing jobs/J47_ts-template-match" in _real)
check("the exemption is keyed on the tool marker, not the folder name",
      "--input_processing" not in _io(
          {"id": "J47", "stage_id": "ts_template_match", "tool": "cryolo",
           "output_dir": "jobs/J47_ts-template-match"}))


# ---- the export card must report PARTICLES ---------------------------------
# J48 exported 18,431 particles from the 4 tomograms that had picks and skipped
# the other 68 that did not. The generic summariser counted the per-series XMLs
# Warp leaves beside the star, so the card read "1 star, 72 xml" — identical to
# an export that produced nothing — and the only way to learn it had worked was
# a terminal. 2026-08-28.
with tempfile.TemporaryDirectory() as _td:
    _d = Path(_td)
    _rows = "\n".join(
        f"  280.2 425.4 769.7 0 0 0 Position053.tomostar 10000.0 6.28 5.4 "
        f"subtomo/Position053/p{_i:07d}.mrc subtomo/Position053/p{_i:07d}_ctf.mrc "
        f"6.28 300.0 2.7" for _i in range(18431))
    (_d / "matching.star").write_text(
        "data_optics\nloop_\n_rlnOpticsGroup #1\n_rlnVoltage #2\n1 300.0\n\n"
        "data_particles\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n"
        "_rlnCoordinateZ #3\n_rlnAngleRot #4\n_rlnAngleTilt #5\n_rlnAnglePsi #6\n"
        "_rlnTomoName #7\n_rlnDefocusU #8\n_rlnDetectorPixelSize #9\n"
        "_rlnMagnification #10\n_rlnImageName #11\n_rlnCtfImage #12\n"
        "_rlnPixelSize #13\n_rlnVoltage #14\n_rlnSphericalAberration #15\n"
        + _rows + "\n")
    for _i in range(72):
        (_d / f"Position{_i:03d}.xml").write_text("<xml/>")
    for _pos in ("Position003", "Position045", "Position053", "Position102"):
        (_d / "subtomo" / _pos).mkdir(parents=True)

    _sum = app.summarize_job("ts_export_particles", _d)
    check("the card reports the particle count from the star",
          _sum.get("particles") == 18431)
    check("and how many tomograms they came from",
          _sum.get("tomograms") == 4)
    check("the 72 per-series XMLs are NOT what it reports",
          "xml" not in _sum and 72 not in _sum.values())
    # data_optics has one row; stopping at the first loop would say 1 particle.
    check("the optics block is not mistaken for the particles",
          _sum.get("particles") != 1)

    # An export that genuinely produced nothing must look DIFFERENT.
    _e = Path(_td) / "empty"
    (_e / "subtomo").mkdir(parents=True)
    (_e / "matching.star").write_text(
        "data_particles\nloop_\n_rlnCoordinateX #1\n")
    check("an empty export reports no particles, not a false count",
          not app.summarize_job("ts_export_particles", _e).get("particles"))
    check("and a missing folder is survivable",
          app.summarize_job("ts_export_particles", Path(_td) / "nope") == {})

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
