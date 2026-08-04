"""Integration test for the Phase-1b job dispatch wiring.

The full Tomogration window won't construct off-Qt (a UI combo-index compare),
so we subclass it and SKIP __init__, wiring up only the state the job path
touches. That still exercises the REAL _run_job / _finalize_job / _on_finished
methods against a temp project — verifying the store lifecycle and that the
three-column stage path is left alone.

    python3 tests/test_dispatch.py
"""
import sys
import os
import tempfile
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE / "stub"))

app = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location(
        "tomapp", sorted(REPO.glob("*tomogration_app.py"))[0]))
sys.modules["tomapp"] = app
app.__spec__.loader.exec_module(app)

passed = failed = 0


def check(name, cond):
    global passed, failed
    ok = bool(cond)
    passed += ok
    failed += (not ok)
    print(f"{'  ok' if ok else 'FAIL'}  {name}")


class FakeRunner:
    def __init__(self):
        self.ran = []
        self._busy = False

    def busy(self):
        return self._busy

    def run(self, cmd, cwd):
        self.ran.append((cmd, cwd))
        self._busy = True          # a real run would block until _on_finished


class Win(app.Tomogration):
    """Real methods, no GUI. __init__ deliberately skips super().__init__."""
    def __init__(self, root):
        self.project_root = str(root)
        self.warp_launch = "module load warp && WarpTools"
        self._group_inputs = None
        self.runner = FakeRunner()
        self.queue = []
        self._active_job_id = None
        self._pending_parent = {}
        self._active_stage = "SENTINEL_STAGE"
        self._active_cmd = ""
        self._attempt = 1
        self._failed_file = None
        self.logs = []

    # GUI-only helpers stubbed to no-ops / capture
    def _log(self, msg, level="info"):
        self.logs.append((level, msg))

    def _refresh_status_dots(self):
        pass

    def _refresh_queue(self):
        pass

    def _dispatch(self, sid, cmd, fresh=True):
        self._dispatched = (sid, cmd, fresh)

    # canvas refresh + form are GUI; stub them for the build/fork tests
    current = None
    _param_store = None

    def _refresh_canvas(self):
        pass

    def _select_stage(self, spec):
        self._selected = spec["id"]

    def _persist_param_store(self):
        pass

    def _effective_params(self, spec):
        return app.stage_defaults(spec)


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    win = Win(root)

    # ---- run a reconstruct job wired to a parent CTF job -------------------
    parent = app.new_job(root, "ts_ctf", "CTF", {"window": "512"},
                         inputs={"processing": None})
    job = app.new_job(root, "ts_reconstruct", "Reconstruct", {"angpix": "10"},
                      inputs={"processing": parent["id"]})
    jid = job["id"]

    win._run_job(jid)

    st = app.load_jobs(root)["jobs"][jid]
    check("job marked running", st["status"] == "running")
    check("started stamped", st.get("started"))
    check("command persisted", st["command"] == win.runner.ran[0][0])
    check("command wired output", f"--output_processing jobs/{jid}" in st["command"])
    check("command wired input", f"--input_processing {parent['output_dir']}" in st["command"])
    check("processing dir created", (root / job["output_dir"]).is_dir())
    check("runner invoked with project cwd", win.runner.ran[0][1] == str(root))
    check("active_job_id set", win._active_job_id == jid)
    check("active_stage cleared (not a stage run)", win._active_stage is None)

    # busy guard: a second run while busy is refused, store untouched ---------
    before = app.load_jobs(root)["jobs"][jid]["status"]
    win._run_job(parent["id"])
    check("busy guard refuses 2nd run",
          app.load_jobs(root)["jobs"][parent["id"]]["status"] == "queued"
          and app.load_jobs(root)["jobs"][jid]["status"] == before)

    # ---- completion: fake a tomogram so the summarizer has something -------
    rec = root / job["output_dir"] / "reconstruction"
    rec.mkdir(parents=True)
    for i in range(2):
        (rec / f"Position{i:03d}_10.00Apx.mrc").write_bytes(b"\0")
    win.runner._busy = False
    win._on_finished(0)

    st = app.load_jobs(root)["jobs"][jid]
    check("job completed", st["status"] == "completed")
    check("exit_code recorded", st["exit_code"] == 0)
    check("finished stamped", st.get("finished"))
    check("summary attached", st["summary"].get("tomograms") == 2)
    check("summary angpix", st["summary"].get("angpix") == "10.00")
    # The finished job must be released. _on_finished then CHAINS into the next
    # queued job, which legitimately becomes the new active one — this used to read
    # as None only because a queued job with no stored command died on the spot.
    check("the finished job is no longer active", win._active_job_id != jid)

    # ---- failure path ------------------------------------------------------
    job2 = app.new_job(root, "ts_ctf", "CTF", {})
    win.runner._busy = False
    win._run_job(job2["id"])
    win.runner._busy = False
    win._on_finished(134)
    st2 = app.load_jobs(root)["jobs"][job2["id"]]
    check("failed job status", st2["status"] == "failed")
    check("failed exit_code", st2["exit_code"] == 134)

    # ---- _on_finished for a STAGE run must NOT touch jobs ------------------
    win._active_job_id = None
    win._active_stage = "ts_ctf"
    win._history_index = None
    win.node_buttons = {}          # stage path pokes dots; empty dict = no-op gets
    # give it the attrs the stage path reads
    import types
    win.project = types.SimpleNamespace(update_history=lambda *a, **k: None)
    try:
        win._on_finished(0)
        stage_path_ok = True
    except Exception as e:
        stage_path_ok = False
        print("   stage-path error:", type(e).__name__, e)
    check("stage-run _on_finished still works (no job side effects)", stage_path_ok)

    # ---- JobCanvas construction + refresh (Phase 2) ------------------------
    # The full window won't build off-Qt, but the canvas widget alone should
    # construct and render (stub Qt = no-ops) without throwing, for both an
    # empty store and one with jobs, and route a card click to the callback.
    picked, detailed = [], []
    try:
        canvas = app.JobCanvas(lambda: str(root), lambda sid: picked.append(sid),
                               on_details=lambda node: detailed.append(node["stage_id"]))
        canvas.refresh()                     # empty-ish store (has jobs from above)
        app.new_job(root, "aretomo", "AreTomo", {})
        canvas.refresh()                     # with an extra job
        canvas._pick({"stage_id": "ts_ctf"})
        canvas._details({"stage_id": "ts_reconstruct"})
        canvas_ok = True
    except Exception as e:
        import traceback
        traceback.print_exc()
        canvas_ok = False
    check("JobCanvas constructs + refreshes without throwing", canvas_ok)
    check("card click routes stage_id to callback", picked == ["ts_ctf"])
    check("Details chip routes node to on_details", detailed == ["ts_reconstruct"])

    # ---- _build_job / _fork_job (Phase 3 interactivity) -------------------
    bw = Win(root)
    # seed an upstream ctf job so the built reconstruct auto-wires to it
    up = app.new_job(root, "ts_ctf", "CTF", {})
    job = bw._build_job("ts_reconstruct", params={"angpix": "10"}, run=False)
    st = app.load_jobs(root)["jobs"][job["id"]]
    check("_build_job created a queued job", st["status"] == "queued")
    check("_build_job auto-wired input to upstream ctf",
          st["inputs"].get("processing") == up["id"])
    check("_build_job carried params", st["params"].get("angpix") == "10")

    bw2 = Win(root)
    bw2.runner._busy = False
    built = bw2._build_job("ts_ctf", params={"window": "1024"}, run=True)
    check("_build_job with run dispatched via runner", len(bw2.runner.ran) == 1)
    check("_build_job run marked the job running",
          app.load_jobs(root)["jobs"][built["id"]]["status"] == "running")

    bw3 = Win(root)
    fk = bw3._fork_job(job["id"])
    forks = [j for j in app.load_jobs(root)["jobs"].values()
             if j["label"].endswith("(fork)")]
    check("_fork_job created a fork", len(forks) == 1)
    check("_fork_job copied stage + params",
          forks[0]["stage_id"] == "ts_reconstruct"
          and forks[0]["params"].get("angpix") == "10")
    check("_fork_job opened it in the builder",
          getattr(bw3, "_selected", None) == "ts_reconstruct")

    # ---- _build_downstream: thread a SPECIFIC parent into the next stage --
    dw = Win(root)
    dw._param_store = {}
    dw._pending_parent = {}
    # two template-match jobs; downstream should wire to the CHOSEN one, not newest
    old_tm = app.new_job(root, "ts_template_match", "M old",
                         {"tomo_angpix": "12.56", "override_suffix": "_v1"})
    new_tm = app.new_job(root, "ts_template_match", "M new",
                         {"tomo_angpix": "12.56", "override_suffix": "_v3-optimized"})
    dw._build_downstream(old_tm["id"], "threshold_picks")   # deliberately the OLDER
    check("downstream seeds derived in_suffix into the form store",
          dw._param_store["threshold_picks"]["in_suffix"] == "12.56Apx_v1")
    check("downstream opened the child in the builder", dw._selected == "threshold_picks")
    # It must CREATE the card, not merely seed the builder: the menu says "Build
    # downstream from this", and when nothing appeared users dragged in a blank card
    # from the palette instead — which then ran with no parameters.
    thr = [j for j in app.load_jobs(root)["jobs"].values()
           if j["stage_id"] == "threshold_picks"]
    check("downstream actually creates the card", len(thr) == 1)
    st = thr[0]
    check("threshold wired to the chosen parent (not newest)",
          st["inputs"].get("processing") == old_tm["id"])
    check("the created card carries the derived params",
          st["params"].get("in_suffix") == "12.56Apx_v1")
    check("it is queued, not run", st["status"] == "queued")
    check("nothing is left stashed for a later unrelated build",
          "threshold_picks" not in dw._pending_parent)
    check("and it is placed below its parent",
          app.load_jobs(root)["positions"].get(st["id"]) is not None)
    dw.runner._busy = False
    check("pending parent consumed after build",
          "threshold_picks" not in dw._pending_parent)

    # ---- threshold job: parent's matching linked into the job's own dir ---
    pw = Win(root)
    pw._pending_parent = {}
    tm_job = app.new_job(root, "ts_template_match", "M",
                         {"tomo_angpix": "12.56", "override_suffix": "_v9"})
    pm = root / tm_job["output_dir"] / "matching"
    pm.mkdir(parents=True)
    for s in ("Position042", "Position046"):
        (pm / f"{s}_12.56Apx_v9.star").write_text("x")
        (pm / f"{s}_12.56Apx_emd_70905_corr.mrc").write_bytes(b"\0")
    thr2 = app.new_job(root, "threshold_picks", "T",
                       {"in_suffix": "12.56Apx_v9", "out_suffix": "clean", "minimum": 3},
                       inputs={"processing": tm_job["id"]})
    pw.runner._busy = False
    pw._run_job(thr2["id"])
    jm = root / thr2["output_dir"] / "matching"
    stars = sorted(jm.glob("*.star")) if jm.is_dir() else []
    check("threshold job's matching/ was populated from parent",
          len(stars) == 2 and all(s.is_symlink() for s in stars))
    check("threshold prep also linked corr maps",
          len(list(jm.glob("*_corr.mrc"))) == 2)
    check("threshold prep links resolve to real files",
          stars and Path(os.path.realpath(stars[0])).exists())

    # non-threshold job is not pre-populated
    rj = Win(root)
    rj._pending_parent = {}
    rec_job = app.new_job(root, "ts_reconstruct", "R", {"angpix": "10"})
    rj.runner._busy = False
    rj._run_job(rec_job["id"])
    check("non-threshold job has no matching/ prep",
          not (root / rec_job["output_dir"] / "matching").exists())

    # ---- _adopt_orphan: register an on-disk pick set as a job -------------
    aw = Win(root)
    md = root / "warp_tiltseries" / "matching"
    md.mkdir(parents=True, exist_ok=True)
    for series in ("Position042", "Position046"):
        (md / f"{series}_12.56Apx260712v9.star").write_text("x")
    orphs = app.discover_picksets(root, app.load_jobs(root))
    orph = next(o for o in orphs if o["suffix"] == "260712v9")
    before = set(app.load_jobs(root)["jobs"])
    aw._adopt_orphan(orph)
    after = app.load_jobs(root)["jobs"]
    new_ids = set(after) - before
    check("adopt created one job", len(new_ids) == 1)
    adopted = after[next(iter(new_ids))]
    check("adopted job is completed ts_template_match",
          adopted["stage_id"] == "ts_template_match" and adopted["status"] == "completed")
    check("adopted job records its suffix", adopted.get("orphan_suffix") == "260712v9")
    linkdir = root / adopted["output_dir"] / "matching"
    links = list(linkdir.glob("*.star")) if linkdir.is_dir() else []
    check("adopt symlinked the stars into the job dir", len(links) == 2)
    check("adopt used symlinks (non-destructive)",
          links and links[0].is_symlink() and (md / links[0].name).exists())
    check("adopted suffix no longer discovered as orphan",
          "260712v9" not in {o["suffix"] for o in
                             app.discover_picksets(root, app.load_jobs(root))})

    # ---- _active_info: the canvas RUNNING banner (job AND trunk runs) ------
    iw = Win(root)
    iw.runner._busy = False
    check("idle -> no running banner", iw._active_info() == {})
    # a trunk run (▶ Run) has no card of its own — it MUST still report as running
    iw.runner._busy = True
    iw._active_job_id = None
    iw._active_stage = "ts_template_match"
    iw._run_progress = "6/290, 01:21:50 remaining"
    info = iw._active_info()
    check("trunk run reports running", info.get("running") is True)
    check("trunk run carries its stage", info.get("stage_id") == "ts_template_match")
    check("trunk run label says it's not a job", "trunk run" in info.get("label", ""))
    check("live progress surfaced", info.get("progress").startswith("6/290"))
    # a job run reports its J-id so the card can be highlighted
    jrun = app.new_job(root, "ts_ctf", "CTF", {})
    iw._active_job_id = jrun["id"]
    info = iw._active_info()
    check("job run reports its job id", info.get("job_id") == jrun["id"])
    check("job run label names the job", jrun["id"] in info.get("label", ""))

    # ---- _apply_layout + _show_card_details (Phase 3 layout) ---------------
    # Exercise the real method bodies with controlled fakes (the stub can't run
    # a full window: `while layout.count()` never ends on a _Perm). Catches
    # attribute typos / bad references in the layout + details code.
    class FakeLayout:
        def __init__(self):
            self.widgets = []
        def count(self):
            return 0                     # always "empty" so _clear_box exits
        def takeAt(self, i):
            return None
        def addWidget(self, w):
            self.widgets.append(w)
    class FakeW:
        def __init__(self):
            self.visible = None
        def setParent(self, p):
            pass
        def setVisible(self, v):
            self.visible = v
        def setCurrentIndex(self, i):
            self.idx = i
        def setChecked(self, b):
            self.checked = b
        def width(self):
            return 1000
        def setSizes(self, s):
            self.sizes = s

    class LWin(app.Tomogration):
        def __init__(self):
            self._view_mode = "lists"
            self.job_stack = FakeW()
            self.details_card = FakeW()
            self.docs_card = FakeW(); self.align_list_card = FakeW()
            self.command_card = FakeW(); self.dir_card = FakeW()
            self.terminal_card = FakeW(); self.queue_card = FakeW()
            self._panels = [self.job_stack, self.details_card, self.docs_card,
                            self.align_list_card, self.command_card, self.dir_card,
                            self.terminal_card, self.queue_card]
            self._stash = FakeW()
            self._layout_host_v = FakeLayout()
            self._act_canvas = FakeW()
            self._canvas_split = None
            self.details_box = FakeLayout()
            self._cfg = {}
        def _load_config(self):
            return dict(self._cfg)
        def _save_config(self, cfg):
            self._cfg = dict(cfg)
        def _refresh_canvas(self):
            pass

    try:
        lw = LWin()
        lw._apply_layout("canvas")
        canvas_mode = (lw._view_mode == "canvas" and lw._cfg.get("view_mode") == "canvas")
        lw._canvas_split = FakeW()       # _apply_layout set a stub splitter; use a real width()
        lw._show_card_details({"id": "J01", "stage_id": "ts_reconstruct",
                               "label": "Reconstruct", "group": "7. Reconstruct",
                               "is_ghost": False, "status": "completed",
                               "summary": {"tomograms": 5, "angpix": "10.00"}})
        details_shown = (lw.details_card.visible is True)
        lw._show_card_details({"id": "ghost:aretomo", "stage_id": "aretomo",
                               "label": "AreTomo", "group": "5. Alignment",
                               "is_ghost": True, "status": "ghost", "summary": {}})
        lw._apply_layout("lists")
        back_to_lists = (lw._view_mode == "lists")
        layout_ok = True
    except Exception as e:
        import traceback
        traceback.print_exc()
        layout_ok = canvas_mode = details_shown = back_to_lists = False
    check("_apply_layout runs both modes without throwing", layout_ok)
    check("_apply_layout sets + persists canvas mode", canvas_mode)
    check("_show_card_details reveals the details pane", details_shown)
    check("_apply_layout toggles back to lists", back_to_lists)

# ---- dropping a card onto the canvas is not a run ---------------------------
# A card dragged in from the palette used to inherit _effective_params — the LAST
# RUN's values — so it arrived pointing at a real output folder from a previous
# round and immediately asked "output folder is not empty, continue?" about a
# command nobody had asked to execute.
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    (root / ".tomogration_jobs.json").write_text('{"seq":0,"jobs":{}}')
    w = Win(root)
    w._param_store = {"relion4_to_warp": {"out_dir": "picks_from_a_previous_round",
                                          "suffix": "old_v3"}}
    w._persist_param_store = lambda: None
    w._refresh_canvas = lambda: None
    w._select_stage = lambda spec: None
    asked = []
    w._confirm_overwrite = lambda spec, params: (asked.append("overwrite"), True)[1]
    w._confirm_validator = lambda spec, params: (asked.append("validate"), True)[1]

    w._add_stage_from_palette("relion4_to_warp", 500.0, 300.0)
    built = [j for j in app.load_jobs(root)["jobs"].values()
             if j["stage_id"] == "relion4_to_warp"]
    check("palette drop creates exactly one job", len(built) == 1)
    check("dropping a card asks nothing", asked == [])
    spec = next(s for s in app.STAGES if s["id"] == "relion4_to_warp")
    check("a dropped card uses template defaults, not the last run's values",
          built[0]["params"].get("out_dir")
          == app.stage_defaults(spec).get("out_dir"))
    check("the previous round's out_dir is NOT inherited",
          built[0]["params"].get("out_dir") != "picks_from_a_previous_round")
    check("the dropped card is queued, never run", built[0]["status"] == "queued")
    check("and it is pinned where it was dropped",
          app.load_jobs(root)["positions"][built[0]["id"]][0] < 500.0)

    # Building the same stage the NORMAL way must still ask.
    asked.clear()
    w._build_job("relion4_to_warp", params={"out_dir": "x"}, run=False)
    check("a normal build still runs the pre-flight checks",
          asked == ["validate", "overwrite"])

# ---- a queued card with no stored command is not a dead card ----------------
# Cards placed from the palette have params but were never queued through the
# builder, so they carry no resolved command. That is a normal state: the runner
# used to declare them broken ("delete it and re-queue") instead of building one.
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    (root / ".tomogration_jobs.json").write_text('{"seq":0,"jobs":{}}')
    w = Win(root)
    w._refresh_canvas = lambda: None
    w._refresh_status_dots = lambda: None
    w._set_status = lambda *a, **k: None
    w._prepare_job_inputs = lambda *a, **k: None
    j = app.new_job(root, "ts_reconstruct", "Tomogram reconstruction",
                    {"settings": "warp_tiltseries.settings", "angpix": "10",
                     "device_list": "0", "perdevice": 1})
    check("a palette-style job starts with no command",
          not app.load_jobs(root)["jobs"][j["id"]].get("command"))
    w._run_queued_job(j["id"])
    st = app.load_jobs(root)["jobs"][j["id"]]
    check("running it builds a command from its params", bool(st.get("command")))
    check("and the command is the right tool",
          "ts_reconstruct" in st.get("command", ""))
    check("the job is not marked failed", st["status"] != "failed")


print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
