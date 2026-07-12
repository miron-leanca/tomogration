"""Unit tests for the Phase-1 job-model layer of tomogration_app.

Runs OFF the VM: stubs PySide6 (tests/stub), loads the app file by path (its name
is export-mangled, so we glob for it), and exercises the PURE job functions
against a temp project dir. No Qt, no cluster, no data.

    python3 tests/test_jobs.py
"""
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
    check("fork: two nodes on ts_ctf row", len(ctf_row) == 2)
    check("fork: distinct columns (side by side)",
          {n["col"] for n in ctf_row} == {0, 1}
          and ctf_row[0]["x"] != ctf_row[1]["x"])

    # summary_text formatting
    check("summary_text ctf", app.summary_text({"series": 15, "defocus_um": "5.25 ± 0.41"})
          == "15 series · 5.25 ± 0.41 µm")
    check("summary_text reconstruct",
          app.summary_text({"tomograms": 290, "angpix": "10.00"}) == "290 tomo · 10.00 Å")
    check("summary_text empty -> ''", app.summary_text({}) == "")

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

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
