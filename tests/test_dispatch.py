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
        # The real __init__ creates these. Without them the stubbed Qt base answers
        # getattr() with a permissive object rather than None, so "is None" checks
        # silently pass on garbage.
        self._builder_job_id = None
        self._builder_bindings = {}
        self._form_job_params = None
        self._showing_job = None
        self._exact_params_for = None

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
        # Mirrors the real one's binding handshake. STICKY and PER-STAGE: each
        # stage remembers its editable card, so a detour to another stage no
        # longer drops the binding; it dies only when the job stops being
        # editable. The clicked card's params arrive as a one-shot overlay
        # (captured in _form_shown), never by writing the param store.
        self._selected = spec["id"]
        bindings = getattr(self, "_builder_bindings", None)
        if bindings is None:
            bindings = self._builder_bindings = {}
        jobs = app.load_jobs(self.project_root).get("jobs") or {}
        want = getattr(self, "_builder_job_id", None)
        if want:
            j = jobs.get(want)
            if j:
                bindings[j.get("stage_id")] = want
        bound = None
        cand = bindings.get(spec["id"])
        if cand:
            j = jobs.get(cand)
            if (j and j.get("status") in ("building", "queued")
                    and j.get("stage_id") == spec["id"]):
                bound = cand
            else:
                bindings.pop(spec["id"], None)
        self._builder_job_id = bound
        ov = getattr(self, "_form_job_params", None)
        self._form_job_params = None
        self._form_shown = (ov["params"]
                            if ov and ov.get("stage") == spec["id"] else None)
        self.current = {"spec": spec, "job_id": bound, "manual": False}

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
          app.load_jobs(root)["jobs"][parent["id"]]["status"] == "building"
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
        # The pick callback receives the whole NODE, so a click can bind the
        # builder to that card. With only a stage id, ▶ Run launched a trunk run
        # and the card sat at Building while its own work ran untracked.
        canvas = app.JobCanvas(lambda: str(root),
                               lambda n: picked.append(n["stage_id"]),
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
    check("_build_job created a building job", st["status"] == "building")
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
    check("it is building, not run", st["status"] == "building")
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

    # ---- _apply_layout + the pop-out panels (canvas rework) -----------------
    # Exercise the real method bodies with controlled fakes (the stub can't run
    # a full window: `while layout.count()` never ends on a _Perm). Catches
    # attribute typos / bad references in the layout + popout code.
    class FakeLayout:
        def __init__(self):
            self.widgets = []
        def count(self):
            return 0                     # always "empty" so _clear_box exits
        def takeAt(self, i):
            return None
        def addWidget(self, w, *a):
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

    class _Pt:
        def x(self):
            return 100
        def y(self):
            return 100
    class _Geo:
        def topLeft(self):
            return _Pt()

    class FakePop:
        """Stands in for JobPopout: records what _open_job_popout does with it
        (the real one builds Qt widgets, which the stub cannot lay out)."""
        def __init__(self, app_, node):
            self.app = app_
            self.node = dict(node)
            self.node_id = node.get("id") or f"ghost:{node.get('stage_id')}"
            self.shown = False
            self.tab = None
            self.refreshed = 0
        def refresh(self, node=None, force=False):
            self.refreshed += 1
        def show_tab(self, name):
            self.tab = name
        def show(self):
            self.shown = True
        def raise_(self):
            pass
        def activateWindow(self):
            pass
        def move(self, *a):
            pass

    class _FakeCanvas:
        _node_index = {}
        view = None

    class LWin(app.Tomogration):
        def __init__(self):
            self._view_mode = "lists"
            self.job_stack = FakeW()
            self.outline_card = FakeW()
            self.docs_card = FakeW(); self.align_list_card = FakeW()
            self.command_card = FakeW(); self.dir_card = FakeW()
            self.terminal_card = FakeW(); self.queue_card = FakeW()
            self._panels = [self.job_stack, self.outline_card, self.docs_card,
                            self.align_list_card, self.command_card, self.dir_card,
                            self.terminal_card, self.queue_card]
            self._stash = FakeW()
            self._layout_host_v = FakeLayout()
            self._act_canvas = FakeW()
            self._popouts = {}
            self.canvas = _FakeCanvas()
            self.project_root = root
            self._docs = {}
            self._cfg = {}
        def _load_config(self):
            return dict(self._cfg)
        def _save_config(self, cfg):
            self._cfg = dict(cfg)
        def _refresh_canvas(self):
            pass
        def geometry(self):
            return _Geo()

    try:
        lw = LWin()
        lw._apply_layout("canvas")
        canvas_mode = (lw._view_mode == "canvas" and lw._cfg.get("view_mode") == "canvas")
        lw._apply_layout("lists")
        back_to_lists = (lw._view_mode == "lists")
        layout_ok = True
    except Exception:
        import traceback
        traceback.print_exc()
        layout_ok = canvas_mode = back_to_lists = False
    check("_apply_layout runs both modes without throwing", layout_ok)
    check("_apply_layout sets + persists canvas mode", canvas_mode)
    check("_apply_layout toggles back to lists", back_to_lists)

    # _open_job_popout: real body, fake JobPopout — registry, retarget, tabs.
    real_popout_cls = app.JobPopout
    try:
        app.JobPopout = FakePop
        lw = LWin()
        node = {"id": "J01", "stage_id": "ts_reconstruct",
                "label": "Reconstruct", "group": "7. Reconstruct",
                "is_ghost": False, "status": "completed",
                "summary": {"tomograms": 5}}
        lw._open_job_popout(node, tab="details")
        p1 = lw._popouts.get("J01")
        popout_made = (p1 is not None and p1.shown and p1.tab == "details")
        lw._open_job_popout(node)                     # second open = same panel
        popout_reused = (lw._popouts.get("J01") is p1 and p1.refreshed >= 1)
        lw._open_job_popout("aretomo")                # bare stage id = ghost panel
        popout_ghost = ("ghost:aretomo" in lw._popouts)
        # canvas-mode card clicks route to the popout, not the dead builder
        lw._view_mode = "canvas"
        lw._canvas_pick(node)
        popout_routed = (lw._popouts.get("J01") is p1)
    except Exception:
        import traceback
        traceback.print_exc()
        popout_made = popout_reused = popout_ghost = popout_routed = False
    finally:
        app.JobPopout = real_popout_cls
    check("_open_job_popout builds + shows a panel on the asked tab", popout_made)
    check("re-opening a node reuses its panel (no duplicates)", popout_reused)
    check("a bare stage id opens a ghost panel", popout_ghost)
    check("canvas-mode card click routes to the popout", popout_routed)

    # The popout tab populators: real bodies over fake boxes (typo net).
    try:
        lw = LWin()
        box = FakeLayout()
        lw._populate_details_box(box, {"id": "J01", "stage_id": "ts_reconstruct",
                                       "label": "Reconstruct",
                                       "group": "7. Reconstruct",
                                       "is_ghost": False, "status": "completed",
                                       "summary": {}})
        details_real = len(box.widgets) > 0
        box2 = FakeLayout()
        lw._populate_details_box(box2, {"id": "ghost:aretomo", "stage_id": "aretomo",
                                        "label": "AreTomo", "group": "5. Alignment",
                                        "is_ghost": True, "status": "ghost",
                                        "summary": {}})
        details_ghost = len(box2.widgets) > 0
        box3 = FakeLayout()
        lw._populate_outputs_box(box3, {"id": "J01", "stage_id": "ts_reconstruct",
                                        "is_ghost": False, "summary": {}})
        outputs_real = len(box3.widgets) > 0
    except Exception:
        import traceback
        traceback.print_exc()
        details_real = details_ghost = outputs_real = False
    check("_populate_details_box fills a real job's tab", details_real)
    check("_populate_details_box fills a ghost's tab", details_ghost)
    check("_populate_outputs_box fills the outputs tab", outputs_real)

    # _smart_path_value: the drop coercions (pure logic, worth real asserts).
    try:
        lw = LWin()
        p_dir = {"name": "tomo_dir", "kind": "text",
                 "title": "Tomogram folder", "_spec": {}}
        v1, n1 = lw._smart_path_value(p_dir, {"path": "recs/tomo1.mrc",
                                              "is_dir": False})
        coerce_dir = (v1 == "recs" and "folder" in (n1 or ""))
        p_abs = {"name": "exclusion_file", "kind": "text",
                 "_spec": {"abs_paths": ["exclusion_file"]}}
        v2, _n2 = lw._smart_path_value(p_abs, {"path": "mdocs/exclusion_list.txt",
                                               "is_dir": False})
        coerce_abs = v2 == str(Path(root) / "mdocs/exclusion_list.txt")
        p_rel = {"name": "input_star", "kind": "text", "_spec": {}}
        v3, _n3 = lw._smart_path_value(
            p_rel, {"path": str(Path(root) / "relion4/run_data.star"),
                    "is_dir": False})
        coerce_rel = v3 == "relion4/run_data.star"
    except Exception:
        import traceback
        traceback.print_exc()
        coerce_dir = coerce_abs = coerce_rel = False
    check("drop coercion: file into a folder param takes its folder", coerce_dir)
    check("drop coercion: abs_paths param gets an absolute path", coerce_abs)
    check("drop coercion: in-project absolute path goes relative", coerce_rel)

# ---- dropping a card onto the canvas is not a run ---------------------------
# A card dragged in from the palette used to inherit _effective_params — the LAST
# RUN's values — so it arrived pointing at a real output folder from a previous
# round and immediately asked "output folder is not empty, continue?" about a
# command nobody had asked to execute.
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    (root / ".tomogration_jobs.json").write_text('{"seq":0,"jobs":{}}')
    w = Win(root)
    w._param_store = {"relion4_convert": {"project_dir": "relion4/previous_round",
                                          "starfile": "old.star"}}
    w._persist_param_store = lambda: None
    w._refresh_canvas = lambda: None
    w._select_stage = lambda spec: None
    asked = []
    w._confirm_overwrite = lambda spec, params: (asked.append("overwrite"), True)[1]
    w._confirm_validator = lambda spec, params: (asked.append("validate"), True)[1]

    w._add_stage_from_palette("relion4_convert", 500.0, 300.0)
    built = [j for j in app.load_jobs(root)["jobs"].values()
             if j["stage_id"] == "relion4_convert"]
    check("palette drop creates exactly one job", len(built) == 1)
    check("dropping a card asks nothing", asked == [])
    spec = next(s for s in app.STAGES if s["id"] == "relion4_convert")
    check("a dropped card uses template defaults, not the last run's values",
          built[0]["params"].get("project_dir")
          == app.stage_defaults(spec).get("project_dir"))
    check("the previous round's project_dir is NOT inherited",
          built[0]["params"].get("project_dir") != "relion4/previous_round")
    check("the dropped card is building, never run",
          built[0]["status"] == "building")
    check("and it is pinned where it was dropped",
          app.load_jobs(root)["positions"][built[0]["id"]][0] < 500.0)

    # "＋ Create job (edit, run later)" is the menu twin of a palette drop: it
    # only places a card. Asking "Run anyway?" to create one read as if the run
    # had already started — the pre-flight checks belong at the run button.
    asked.clear()
    opened = []
    w._open_job_in_builder = lambda jid, sid: opened.append((jid, sid))
    n_before = len(app.load_jobs(root)["jobs"])
    w._create_job_card("relion4_convert")
    jobs = app.load_jobs(root)["jobs"]
    fresh = max(jobs.values(), key=lambda j: j["created"] + j["id"])
    check("create-job adds exactly one card", len(jobs) == n_before + 1)
    check("creating a card asks nothing", asked == [])
    check("the created card is building, never run", fresh["status"] == "building")
    check("the created card opens in the builder for editing",
          opened and opened[-1][1] == "relion4_convert")

    # Building the same stage the NORMAL way must still ask.
    asked.clear()
    w._build_job("relion4_convert", params={"project_dir": "x"}, run=False)
    check("a normal build still runs the pre-flight checks",
          asked == ["validate", "overwrite"])

# ---- the variant sweep: one card per COMBINATION -----------------------------
# The grid logic is pure (it reads value getters, not widgets), so it tests
# without a screen. The trap it guards: variants of a step whose output path is
# a parameter all write to the SAME folder and silently overwrite each other.
_SWEEP_SPEC = {
    "id": "mb_deconv", "label": "Deconvolve",
    "params": [{"name": "input_dir", "kind": "text", "title": "Tomograms"},
               {"name": "MB_STRENGTH", "kind": "env", "title": "Strength"},
               {"name": "MB_FALLOFF", "kind": "env", "title": "Falloff"},
               {"name": "output_dir", "kind": "text", "title": "Output"}],
    "output_params": ["output_dir"],
}


class _Box:                       # stands in for the suffix checkbox
    def __init__(self, on): self.on = on
    def isChecked(self): return self.on


def _sweep(rows, suffix=True):
    d = app.VariantsDialog.__new__(app.VariantsDialog)
    d.spec = _SWEEP_SPEC
    d.suffix_cb = _Box(suffix)
    d.rows = {k: [(None, (lambda v=v: v)) for v in vals] for k, vals in rows.items()}
    return d


d = _sweep({"input_dir": ["recon"], "MB_STRENGTH": ["0.5", "1.0", "1.5"],
            "MB_FALLOFF": ["1.0", "1.2"], "output_dir": ["membrane/deconv"]})
combos, varied = d._combinations()
check("sweep: 3 x 2 values = 6 cards", len(combos) == 6)
check("sweep: only the multi-valued params count as varied",
      varied == ["MB_STRENGTH", "MB_FALLOFF"])
check("sweep: fixed params ride along unchanged",
      all(c["input_dir"] == "recon" for c in combos))
check("sweep: every combination is distinct",
      len({(c["MB_STRENGTH"], c["MB_FALLOFF"]) for c in combos}) == 6)
sfx = d._suffixed(combos)
check("sweep: colliding output folders get _v1.._vN",
      [c["output_dir"] for c in sfx]
      == [f"membrane/deconv_v{i}" for i in range(1, 7)])
check("sweep: suffixing touches nothing else",
      all(c["MB_STRENGTH"] == o["MB_STRENGTH"] for c, o in zip(sfx, combos)))
check("sweep: the card name says which variant it is",
      "Strength=0.5" in d._label_for(sfx[0], varied, 1)
      and d._label_for(sfx[0], varied, 1).startswith("Deconvolve · "))
check("sweep: opting out leaves the output path alone",
      _sweep({"input_dir": ["recon"], "MB_STRENGTH": ["0.5", "1.0"],
              "MB_FALLOFF": ["1.0"], "output_dir": ["membrane/deconv"]},
             suffix=False)._suffixed(combos)[0]["output_dir"] == "membrane/deconv")

# A repeated value is a slip, not a request for the same job twice.
d2 = _sweep({"input_dir": ["recon"], "MB_STRENGTH": ["1.0", "1.0", "1.5"],
             "MB_FALLOFF": ["1.0"], "output_dir": ["membrane/deconv"]})
combos2, _ = d2._combinations()
check("sweep: duplicate values collapse", len(combos2) == 2)

# Output paths that already differ must NOT be renamed under the user.
d3 = _sweep({"input_dir": ["recon"], "MB_STRENGTH": ["0.5", "1.0"],
             "MB_FALLOFF": ["1.0"], "output_dir": ["membrane/deconv"]})
combos3, _ = d3._combinations()
distinct = [dict(combos3[0], output_dir="membrane/a"),
            dict(combos3[1], output_dir="membrane/b")]
check("sweep: already-distinct outputs are left as they are",
      [c["output_dir"] for c in d3._suffixed(distinct)] == ["membrane/a",
                                                            "membrane/b"])

# ---- the dialog must survive being CONSTRUCTED -----------------------------
# Every check above builds it with __new__ and hand-set attributes, so the sweep
# maths was covered while __init__ was not — and __init__ was broken for every
# stage: _param_block -> _add_row -> _refresh runs once per parameter, before the
# button row exists, so opening it raised
#     AttributeError: 'VariantsDialog' object has no attribute 'build_btn'
# The Qt stub cannot catch that on its own: its __getattr__ hands back a dummy
# for ANY missing attribute. This subclass restores real Python behaviour, so an
# attribute touched before it is assigned raises here exactly as it does in Qt.
# Only the dialog's OWN attributes are made strict; inherited Qt methods
# (setWindowTitle, resize, …) still come from the stub, as they would from Qt.
_OWN_ATTRS = {"build_btn", "queue_btn", "summary", "suffix_cb", "form", "rows",
              "spec", "variants", "labels", "queue"}


class _StrictVariants(app.VariantsDialog):
    def __getattr__(self, name):        # shadows the stub's permissive one
        if name in _OWN_ATTRS:
            raise AttributeError(
                f"VariantsDialog touched self.{name} before __init__ assigned it")
        inherited = getattr(super(), "__getattr__", None)
        if inherited is None:
            raise AttributeError(name)
        return inherited(name)


# Every dialog that connects a signal while building its widgets can hit this,
# so the guard is generic: any attribute a dialog assigns in __init__ must not
# be READ before that point. ViewerPickDialog reintroduced the exact bug this
# was written for, because the first version of this test named one class.
def _strict(cls, own):
    class Strict(cls):
        def __getattr__(self, name):
            if name in own:
                raise AttributeError(
                    f"{cls.__name__} touched self.{name} before __init__ set it")
            inherited = getattr(super(), "__getattr__", None)
            if inherited is None:
                raise AttributeError(name)
            return inherited(name)
    return Strict


_INV = {"Position003": ["/r/Position003.mrc", "/o/Position003_segmented.mrc"],
        "Position045": ["/r/Position045.mrc", "/o/Position045_segmented.mrc"]}
try:
    _S = _strict(app.ViewerPickDialog, {"tree", "count", "files"})
    _S(None, _INV)
    _ok, _why = True, ""
except Exception as e:                                          # noqa: BLE001
    _ok, _why = False, f"{type(e).__name__}: {e}"
check(f"ViewerPickDialog constructs without touching a widget too early {_why}",
      _ok)
try:
    _S2 = _strict(app.TomoPickDialog, {"list", "count", "stems", "groups"})
    _S2(None, "/nonexistent", "")
    _ok2, _why2 = True, ""
except Exception as e:                                          # noqa: BLE001
    _ok2, _why2 = False, f"{type(e).__name__}: {e}"
check(f"TomoPickDialog too {_why2}", _ok2)


for _sid in ("ts_reconstruct", "mb_deconv", "mb_isonet2_train", "ts_ctf"):
    _spec = next(s for s in app.STAGES if s["id"] == _sid)
    try:
        _StrictVariants(None, _spec, app.stage_defaults(_spec))
        _ok, _why = True, ""
    except Exception as e:                                  # noqa: BLE001
        _ok, _why = False, f"{type(e).__name__}: {e}"
    check(f"Queue variants opens for {_sid} {_why}", _ok)

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


# ---- the builder must not duplicate the card it just created ----------------
# _build_downstream creates the card AND opens the builder. The builder is
# stage-scoped, so its run button always built a NEW job — producing a second card
# for the same step that ran off on its own while the one being edited stayed
# queued forever.
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    (root / ".tomogration_jobs.json").write_text('{"seq":0,"jobs":{}}')
    w = Win(root)
    w._param_store = {}
    w._prepare_job_inputs = lambda *a, **k: None
    w._set_status = lambda *a, **k: None
    # A real RELION 4 star, so the export's pre-flight can read its pixel size.
    (root / "Select" / "job019").mkdir(parents=True)
    (root / "Select" / "job019" / "particles.star").write_text(
        "data_optics\nloop_\n_rlnOpticsGroup #1\n_rlnImagePixelSize #2\n1 6.28\n\n"
        "data_particles\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n"
        "_rlnCoordinateZ #3\n_rlnMicrographName #4\n_rlnOriginXAngst #5\n"
        "100.0 200.0 300.0 Position003.tomostar -4.7\n")
    sel = app.new_job(root, "relion4_result", "Subset selection",
                      {"source_star": "Select/job019/particles.star",
                       "job_dir": "Select/job019"})
    app.update_job(root, sel["id"], status="completed", tool="relion_selection")
    w._build_downstream(sel["id"], "ts_export_particles")
    made = [j for j in app.load_jobs(root)["jobs"].values()
            if j["stage_id"] == "ts_export_particles"]
    check("downstream created one export card", len(made) == 1)
    check("the builder is bound to that card", w.current["job_id"] == made[0]["id"])
    check("its star was derived from the selection",
          made[0]["params"].get("input_star") == "Select/job019/particles.star")
    check("and coords_angpix was read from the star",
          made[0]["params"].get("coords_angpix") == "6.28")
    check("with the pick-star route blanked",
          made[0]["params"].get("input_directory") == ""
          and made[0]["params"].get("normalized_coords") is False)

    # Pressing the builder's run button must run THAT job, not build another.
    w._values = lambda ctx=None: dict(made[0]["params"], diameter="160")
    w._confirm_validator = lambda spec, params: True
    w._confirm_overwrite = lambda spec, params: True
    w.runner._busy = False
    w._save_and_run_job(made[0]["id"])
    after = [j for j in app.load_jobs(root)["jobs"].values()
             if j["stage_id"] == "ts_export_particles"]
    check("running from the builder creates NO second card", len(after) == 1)
    check("it ran the bound card", after[0]["status"] == "running")
    check("the form's edits were saved into it",
          after[0]["params"].get("diameter") == "160")
    check("and its command was resolved", bool(after[0].get("command")))

    # The binding is STICKY but re-validated, so it lapses on its own rather than
    # being eagerly cleared. Once the bound job is running it is no longer editable,
    # and the next rebuild of that stage must drop it.
    spec_tw = next(x for x in app.STAGES if x["id"] == "ts_export_particles")
    w._select_stage(spec_tw)
    check("a running job's binding lapses on rebuild",
          w.current["job_id"] is None)
    # And it never leaks to an unrelated stage.
    w._select_stage(next(x for x in app.STAGES if x["id"] == "ts_ctf"))
    check("no binding leaks to another stage", w.current["job_id"] is None)


# ---- editing a QUEUED card must not clone it --------------------------------
# "+ Queue variant" always minted a new job. That is right for a second variant
# and wrong when the card being edited is already sitting in the queue: it made a
# duplicate card for the same step. Reopening a queued card also left the builder
# stage-scoped, so every button cloned.
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    (root / ".tomogration_jobs.json").write_text('{"seq":0,"jobs":{}}')
    w = Win(root)
    w._param_store = {}
    w._refresh_queue = lambda: None
    w._prepare_job_inputs = lambda *a, **k: None
    w._set_status = lambda *a, **k: None
    w._confirm_validator = lambda spec, params: True
    w._confirm_overwrite = lambda spec, params: True

    running = app.new_job(root, "ts_export_particles", "Export", {"box": "80"})
    app.update_job(root, running["id"], status="running")
    q = app.new_job(root, "relion4_convert", "convert STAR",
                    {"project_dir": "relion4/v5", "starfile": "matching.star"})
    app.update_job(root, q["id"], status="queued", command="echo old")
    check("the queued card is behind the running one",
          [j["id"] for j in app.queued_jobs(app.load_jobs(root))] == [q["id"]])

    # Reopening it must bind, so the buttons act on it.
    w._open_job_in_builder(q["id"], "relion4_convert")
    check("reopening a queued card binds the builder",
          w.current["job_id"] == q["id"])

    before = len(app.load_jobs(root)["jobs"])
    w._values = lambda ctx=None: {"project_dir": "relion4/v6", "starfile": "matching.star"}
    w.current["manual"] = False
    w._save_queued_job(q["id"])
    after = app.load_jobs(root)["jobs"]
    check("saving a queued card creates NO new card", len(after) == before)
    check("it stays queued", after[q["id"]]["status"] == "queued")
    check("the edits were saved",
          after[q["id"]]["params"]["project_dir"] == "relion4/v6")
    check("a rebuilt command is cleared so it re-resolves at run time",
          after[q["id"]]["command"] == "")
    check("still the only queued job",
          [j["id"] for j in app.queued_jobs(app.load_jobs(root))] == [q["id"]])

    # A manual command edit is kept verbatim instead of being rebuilt.
    w.current["manual"] = True
    w.current["cmd"] = type("C", (), {"toPlainText": lambda self: "  echo mine  "})()
    w._save_queued_job(q["id"])
    check("a hand-edited command is stored verbatim",
          app.load_jobs(root)["jobs"][q["id"]]["command"] == "echo mine")

    # A COMPLETED card must not bind — re-running it is the card's own action.
    done = app.new_job(root, "ts_ctf", "CTF", {})
    app.update_job(root, done["id"], status="completed")
    w._open_job_in_builder(done["id"], "ts_ctf")
    check("a finished card does not bind the builder",
          w.current["job_id"] is None)


# ---- the overwrite prompt must name what is actually at risk ----------------
# relion4_class3d's project_dir is a RELION PROJECT ROOT: matching_conv.star,
# subtomo/ and previous jobs live there and it touches none of them. It writes
# Class3D/job001/ — a HARDCODED name, so a second run really does overwrite the
# first. Warning about the container was a false alarm about the wrong files,
# and it hid the real risk.
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    (root / ".tomogration_jobs.json").write_text('{"seq":0,"jobs":{}}')
    w = Win(root)
    asked = []
    w._count_dir_entries = lambda rel, cap=3: (
        1 if (root / rel).is_dir() and any((root / rel).iterdir()) else 0)

    class Box:
        Yes, No = 1, 0
        @staticmethod
        def question(parent, title, text, *a, **k):
            asked.append(text)
            return 1
    app.QMessageBox = Box

    # A SYNTHETIC spec, not a real stage. The rule under test is about
    # output_subdirs — warn about the folder a stage actually writes, not the
    # project root that merely contains it — and it outlived the stage that
    # first exposed it (relion4_class3d, removed 2026-08-21 as something that
    # would never be run through tomogration). Pinning the test to a stage id
    # made a general guarantee look like a detail of one card.
    spec = {"id": "synthetic_relion_stage", "label": "Writes a job folder",
            "output_params": ["project_dir"],
            # Which param holds the CONTAINER the subdirs hang off. Without it
            # the base never resolves, the subdir rule never engages, and the
            # container gets named as at risk — the very false alarm this
            # block exists to prevent.
            "output_subdir_param": "project_dir",
            "output_subdirs": ["Class3D/job001"], "docs": {}}
    proj = root / "relion4/picks_v5"
    (proj / "subtomo").mkdir(parents=True)
    (proj / "matching_conv.star").write_text("x")
    params = {"project_dir": "relion4/picks_v5", "particles": "matching_conv.star"}

    asked.clear()
    ok = w._confirm_overwrite(spec, params)
    check("a populated RELION project root alone raises no alarm",
          ok is True and asked == [])

    (proj / "Class3D" / "job001").mkdir(parents=True)
    (proj / "Class3D" / "job001" / "run_it025_data.star").write_text("x")
    asked.clear()
    w._confirm_overwrite(spec, params)
    check("an existing Class3D/job001 DOES raise the alarm", len(asked) == 1)
    check("and the prompt names the job folder, not the project root",
          "Class3D/job001" in asked[0])
    check("the project root itself is not listed as at risk",
          "    relion4/picks_v5/\n" not in asked[0])
    check("a stage with no declared subdirs still warns about its output dir",
          w._confirm_overwrite({"id": "x", "output_params": ["project_dir"],
                                "docs": {}}, params) is not None)


# ---- a second child must not land on top of the first -----------------------
# Every child was pinned at exactly "parent + one row", so building a SECOND job
# downstream of the same parent dropped it precisely onto the first. The old card
# was untouched but completely hidden, which reads as the existing card having
# been reused or overwritten.
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    (root / ".tomogration_jobs.json").write_text('{"seq":0,"jobs":{}}')
    w = Win(root)
    w._param_store = {}
    sel = app.new_job(root, "relion4_result", "Subset selection",
                      {"source_star": "Select/job019/particles.star",
                       "job_dir": "Select/job019"})
    app.update_job(root, sel["id"], status="completed", tool="relion_selection")

    w._build_downstream(sel["id"], "ts_export_particles")
    w._build_downstream(sel["id"], "ts_export_particles")
    kids = sorted((j["id"] for j in app.load_jobs(root)["jobs"].values()
                   if j["stage_id"] == "ts_export_particles"), key=app._job_seq)
    check("building downstream twice makes TWO cards", len(kids) == 2)
    pos = app.load_jobs(root)["positions"]
    a, b = pos[kids[0]], pos[kids[1]]
    check("the second card does not sit on the first", a != b)
    check("they do not overlap at all",
          abs(a[0] - b[0]) >= app.CARD_W or abs(a[1] - b[1]) >= app.CARD_H)
    check("siblings share a row", a[1] == b[1])
    check("the second is to the RIGHT of the first", b[0] > a[0])
    px, py = w._node_position(sel["id"])
    check("both sit one row below their parent", a[1] == py + app.CARD_H + app.GAP_Y)
    # Not necessarily at the parent's own x: the row below already holds the
    # auto-laid-out cards for that stage, and stepping clear of those is the point.
    check("the first child starts at or right of its parent", a[0] >= px)
    others = [(n["x"], n["y"]) for n in
              app.canvas_layout(app.load_jobs(root))[0] if n["id"] not in kids]
    check("no child overlaps any other card",
          all(abs(c[0] - o[0]) >= app.CARD_W or abs(c[1] - o[1]) >= app.CARD_H
              for c in (a, b) for o in others))
    check("neither overwrote the other's record",
          len({j["id"] for j in app.load_jobs(root)["jobs"].values()}) == 3)

    # A third one keeps stepping right rather than stacking.
    w._build_downstream(sel["id"], "ts_export_particles")
    pos = app.load_jobs(root)["positions"]
    xs = sorted(pos[k][0] for k in
                sorted((j["id"] for j in app.load_jobs(root)["jobs"].values()
                        if j["stage_id"] == "ts_export_particles"), key=app._job_seq))
    check("three children occupy three distinct columns", len(set(xs)) == 3)


# ---- clear job: the results go, the card stays ------------------------------
# Between "delete the job, keep the files" and "delete both" there was no way to
# say "that attempt was wrong, try again": you deleted the card and lost its
# parameters and wiring with it. Re-running over a half-written output dir also
# mixes two attempts' files, which is how a failed export leaves a star that
# looks complete.
class _Box:
    Warning = 0
    DestructiveRole = 1
    RejectRole = 2
    refused = []

    def __init__(self, *a, **k):
        self._btns = []
        self._yes = None
        self._clicked = None

    def setIcon(self, *a): pass
    def setWindowTitle(self, *a): pass
    def setText(self, t): self.text = t
    def setInformativeText(self, t): self.info = t

    def addButton(self, label, role):
        b = object()
        self._btns.append(b)
        if role == _Box.DestructiveRole:
            self._yes = b
        return b

    def buttons(self): return self._btns
    def setDefaultButton(self, *a): pass
    def exec(self): self._clicked = self._yes          # always confirm
    def clickedButton(self): return self._clicked

    @staticmethod
    def warning(parent, title, text, *a, **k):
        _Box.refused.append(text)
        return None


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    (root / ".tomogration_jobs.json").write_text('{"seq":0,"jobs":{}}')
    w = Win(root)
    w._param_store = {}
    w._refresh_queue = lambda: None
    w._invalidate_orphans = lambda: None
    w._invalidate_status = lambda: None
    w._dir_size_human = lambda rel: "1 MB"
    _orig_box = app.QMessageBox
    app.QMessageBox = _Box
    try:
        j = app.new_job(root, "ts_reconstruct", "Reconstruct", {"angpix": "10"})
        out = root / j["output_dir"] / "reconstruction"
        out.mkdir(parents=True)
        (out / "Position001_10.00Apx.mrc").write_bytes(b"\0")
        app.update_job(root, j["id"], status="completed", exit_code=0,
                       command="WarpTools ts_reconstruct --angpix 10",
                       finished="2026-08-04 10:00:00", summary={"tomograms": 1})

        w._clear_job(j["id"])
        rec = app.load_jobs(root)["jobs"].get(j["id"])
        check("clear keeps the card", rec is not None)
        check("clear removes the results from disk",
              not (root / j["output_dir"]).exists())
        # Clear hands the card back for reconfiguring, so BUILDING — not queued,
        # which would let it run again before it had been looked at.
        check("clear returns it to building", rec["status"] == "building")
        check("clear keeps the parameters", rec["params"]["angpix"] == "10")
        check("clear resets the exit code", rec["exit_code"] is None)
        check("clear empties the summary", rec["summary"] == {})
        check("clear drops the finish time", rec["finished"] is None)
        # A stale command is how an edited card re-runs the OLD one.
        check("clear drops the stale command", rec["command"] == "")
        check("clear opens it for reconfiguring", w._selected == "ts_reconstruct")

        # Clearing a RUNNING job would delete files out from under a live process.
        _Box.refused.clear()
        j2 = app.new_job(root, "ts_ctf", "CTF", {})
        app.update_job(root, j2["id"], status="running")
        w._clear_job(j2["id"])
        check("clear refuses a running job",
              app.load_jobs(root)["jobs"][j2["id"]]["status"] == "running")
        check("and says why", any("running" in t for t in _Box.refused))
    finally:
        app.QMessageBox = _orig_box


# ---- every context-menu action must be able to run --------------------------
# "Clear job" and the job-bound "Open in job builder" were both inserted into the
# WRONG branch of _card_menu — the orphan and ghost branches, where `jid` is never
# assigned. Nothing complained: a lambda resolves its names when CLICKED, so the
# menu builds fine and the action raises NameError inside a Qt slot, which is the
# same silent failure mode as the lost symbol that once made Run do nothing.
# An unbound closure cell is detectable without invoking anything.
class _RecMenu:
    def __init__(self, *a, **k):
        self.items = []          # [(label, callback)]
        self.subs = []

    def addAction(self, label, cb=None):
        self.items.append((label, cb))
        return object()

    def addMenu(self, label):
        m = _RecMenu()
        self.subs.append((label, m))
        return m

    def addSeparator(self): pass
    def exec(self, *a, **k): pass

    def all_items(self):
        out = list(self.items)
        for _, m in self.subs:
            out.extend(m.all_items())
        return out


def _unbound(cb):
    """Names the callback closes over that are not actually bound."""
    bad = []
    for name, cell in zip(getattr(cb, "__code__", None).co_freevars if cb else (),
                          cb.__closure__ or ()):
        try:
            cell.cell_contents
        except ValueError:
            bad.append(name)
    return bad


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    (root / ".tomogration_jobs.json").write_text('{"seq":0,"jobs":{}}')
    w = Win(root)
    _orig_menu = app.QMenu
    app.QMenu = _RecMenu
    try:
        j = app.new_job(root, "ts_reconstruct", "Reconstruct", {"angpix": "10"})
        app.update_job(root, j["id"], status="completed")
        nodes, _ = app.canvas_layout(app.load_jobs(root))
        job_node = next(n for n in nodes if n["id"] == j["id"])
        ghost_node = next(n for n in nodes if n.get("is_template"))
        orphan_node = {"id": None, "stage_id": "ts_template_match", "is_orphan": True,
                       "is_ghost": False, "status": "orphan", "summary": {},
                       "orphan": {"kind": "picks", "dir": "x", "suffix": "s"}}

        menus = {}
        for name, node in (("job", job_node), ("template", ghost_node),
                           ("orphan", orphan_node)):
            m = _RecMenu()
            app.QMenu = lambda *a, _m=m, **k: _m
            w._card_menu(node, None)
            menus[name] = m
            app.QMenu = _RecMenu

        for name, m in menus.items():
            for label, cb in m.all_items():
                bad = _unbound(cb)
                check(f"{name} menu · '{label}' can run", not bad)

        labels = {k: [l for l, _ in m.all_items()] for k, m in menus.items()}
        check("Clear job is offered on a real job",
              any("Clear job" in l for l in labels["job"]))
        check("Clear job is NOT on a template",
              not any("Clear job" in l for l in labels["template"]))
        check("Clear job is NOT on an orphan",
              not any("Clear job" in l for l in labels["orphan"]))
        check("every menu offers 'Open in job builder'",
              all(any("Open in job builder" in l for l in labels[k])
                  for k in ("job", "template")))
    finally:
        app.QMenu = _orig_menu


# ---- clicking a card binds the builder to it --------------------------------
# Without the binding the form is stage-scoped, so ▶ Run launches a TRUNK run of
# the stage: the right command executes, nothing updates the card, and it sits at
# Building while its own work runs untracked beside it.
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    (root / ".tomogration_jobs.json").write_text('{"seq":0,"jobs":{}}')
    w = Win(root)
    w._param_store = {}

    b = app.new_job(root, "ts_export_particles", "Export", {"box": "112"})
    nodes, _ = app.canvas_layout(app.load_jobs(root))
    node = next(n for n in nodes if n["id"] == b["id"])
    w._canvas_pick(node)
    check("clicking a Building card binds the builder",
          w.current["job_id"] == b["id"])
    check("and shows that card's own parameters",
          (w._form_shown or {}).get("box") == "112")
    check("without touching the per-stage param store",
          "ts_export_particles" not in w._param_store)

    # THE DETOUR: visit another stage (to review something), then come back via
    # the stage itself (not the card). The binding must survive — this used to
    # drop it, the button reverted to plain "▶ Run", and the next click launched
    # an untracked trunk run instead of the card on screen.
    _spec = lambda sid: next(s for s in app.STAGES if s["id"] == sid)
    w._select_stage(_spec("ts_ctf"))
    check("another stage has no binding", w.current["job_id"] is None)
    w._select_stage(_spec("ts_export_particles"))
    check("the binding survives a detour to another stage",
          w.current["job_id"] == b["id"])

    app.update_job(root, b["id"], status="queued")
    w._canvas_pick(dict(node, status="queued"))
    check("clicking a Queued card binds too", w.current["job_id"] == b["id"])
    check("re-clicking the bound card keeps the working form (no reload)",
          w._form_shown is None)

    # A finished card must NOT bind — re-running it is its own card action, and
    # silently overwriting a completed job's parameters would be worse. But it MUST
    # still SHOW what that job ran: "what did J9 actually use?" is the commonest
    # question asked of a finished job, and the form used to answer it with this
    # week's values.
    app.update_job(root, b["id"], status="completed",
                   params={"box": "80", "output_angpix": "6.28"},
                   finished="2026-07-16 09:20:00")
    w._param_store["ts_export_particles"] = {"box": "112"}     # "today's" values
    w._canvas_pick(dict(node, status="completed"))
    check("clicking a Completed card does not bind", w.current["job_id"] is None)
    check("but it DOES show that job's recorded parameters",
          (w._form_shown or {}).get("box") == "80")
    check("shown values are the job's whole set, not merged with today's",
          (w._form_shown or {}).get("output_angpix") == "6.28")
    check("viewing history leaves today's saved values intact",
          w._param_store["ts_export_particles"] == {"box": "112"})

    # Templates and discovered cards have no job behind them at all.
    tmpl = next(n for n in nodes if n.get("is_template"))
    w._canvas_pick(tmpl)
    check("clicking a template does not bind", w.current["job_id"] is None)
    check("but it still opens that stage", w._selected == tmpl["stage_id"])

    # A bare stage id still works — the ghost menu passes one.
    w._canvas_pick("ts_ctf")
    check("a bare stage id still opens the stage", w._selected == "ts_ctf")
    check("and binds nothing", w.current["job_id"] is None)


# ---- reviewing a past run says it is a past run -----------------------------
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    (root / ".tomogration_jobs.json").write_text('{"seq":0,"jobs":{}}')
    w = Win(root)
    w._param_store = {}
    old_job = app.new_job(root, "ts_export_particles", "Export",
                          {"box": "80", "output_angpix": "6.28"})
    app.update_job(root, old_job["id"], status="completed",
                   finished="2026-07-16 09:20:00")
    w._open_job_in_builder(old_job["id"], "ts_export_particles")
    check("a finished job is flagged for the history banner",
          w._showing_job is not None and w._showing_job["id"] == old_job["id"])
    check("the banner carries its status", w._showing_job["status"] == "completed")
    check("and when it ran", w._showing_job["when"] == "2026-07-16 09:20:00")
    check("its params are shown (one-shot overlay, not the store)",
          (w._form_shown or {}).get("box") == "80")
    check("the per-stage store is untouched by the view",
          "ts_export_particles" not in w._param_store)
    check("and the buttons are NOT bound to it", w.current["job_id"] is None)

    # An editable job gets the binding and NO banner — it is not history.
    q = app.new_job(root, "ts_export_particles", "Export", {"box": "112"})
    w._open_job_in_builder(q["id"], "ts_export_particles")
    check("a building job binds instead", w.current["job_id"] == q["id"])
    check("and shows no history banner", w._showing_job is None)

    # A job with no recorded params (adopted from disk, or pre-job-model) must say
    # so rather than showing someone else's values as if they were its own.
    bare = app.new_job(root, "ts_export_particles", "old", {})
    app.update_job(root, bare["id"], status="completed")
    w._param_store["ts_export_particles"] = {"box": "999"}
    w._open_job_in_builder(bare["id"], "ts_export_particles")
    check("a job with no params is flagged as empty", w._showing_job["empty"] is True)
    check("and does not silently show the previous values as its own",
          w._showing_job["id"] == bare["id"])


# ---- the builder binding must survive a form rebuild ------------------------
# It used to be consumed by _select_stage, so ANY of the fifteen things that
# rebuild the form dropped it: the button quietly changed from "Run J2" back to
# "Run", and the next click built a second job instead of running the one on
# screen. That is exactly what happened to J2 in the EML46 project.
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    (root / ".tomogration_jobs.json").write_text('{"seq":0,"jobs":{}}')
    w = Win(root)
    w._param_store = {}
    spec = next(x for x in app.STAGES if x["id"] == "ts_export_particles")

    j = app.new_job(root, "ts_export_particles", "Export", {"box": "48"})
    w._open_job_in_builder(j["id"], "ts_export_particles")
    check("bound after opening", w.current["job_id"] == j["id"])

    # The stage sidebar, the details pane and a project-root refresh all do exactly
    # this — re-render the SAME stage.
    w._select_stage(spec)
    check("binding survives a rebuild of the same stage",
          w.current["job_id"] == j["id"])
    w._select_stage(spec)
    check("and a second one", w.current["job_id"] == j["id"])

    # A DIFFERENT stage's form is not editing this job — no binding there. But
    # coming BACK must restore it: a detour to review another card used to lose
    # the binding for good, the button reverted to plain "▶ Run", and the next
    # click ran an untracked trunk run instead of the card on screen.
    w._select_stage(next(x for x in app.STAGES if x["id"] == "ts_ctf"))
    check("switching stage shows no binding there", w.current["job_id"] is None)
    w._select_stage(spec)
    check("and coming back restores it", w.current["job_id"] == j["id"])

    # A job that starts running is no longer editable, so the binding must lapse
    # rather than let the form write into a live run.
    w._open_job_in_builder(j["id"], "ts_export_particles")
    check("re-bound", w.current["job_id"] == j["id"])
    app.update_job(root, j["id"], status="running")
    w._select_stage(spec)
    check("a job that starts running drops the binding",
          w.current["job_id"] is None)

    # And the click-time path refuses too, even if a stale id reached it.
    w._values = lambda ctx=None: {"box": "999"}
    w._confirm_validator = lambda sp, pa: True
    w._confirm_overwrite = lambda sp, pa: True
    w._save_and_run_job(j["id"])
    check("saving into a running job is refused",
          app.load_jobs(root)["jobs"][j["id"]]["params"]["box"] == "48")


# ---- Found-on-disk ▸ RELION Subset selection ▸ re-extract (canvas view) -----
# The drawer's re-extract action used to prefill a converter builder — a column
# that no longer exists in canvas view — and the converter itself is retired.
# Canvas mode now promotes the selection to a card and builds the EXPORT card
# downstream of it: star prefilled, coords_angpix read from the star, pop-out
# builder open. Lists view prefills the export form the same way.
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    star_rel = "relion4/Select/job009/particles.star"
    star_abs = root / star_rel
    star_abs.parent.mkdir(parents=True)
    star_abs.write_text(
        "data_optics\nloop_\n_rlnOpticsGroup #1\n_rlnImagePixelSize #2\n1 6.28\n\n"
        "data_particles\nloop_\n_rlnCoordinateX #1\n_rlnMicrographName #2\n"
        "1.0 Position003.tomostar\n2.0 Position003.tomostar\n"
        "3.0 Position004.tomostar\n")
    w = Win(root)
    w._param_store = {}
    w._view_mode = "canvas"
    w._orphan_cache = None
    w._last_orphan_keys = None
    pops = []
    w._open_job_popout = lambda target, tab=None: pops.append((target, tab))
    w._open_job_in_builder = lambda jid, sid: pops.append((jid, "builder"))
    orph = {"kind": "relion_job", "suffix": "Select/job009", "star": star_rel,
            "dir": "relion4/Select/job009"}
    w._open_relion_export(orph)
    jobs = app.load_jobs(root).get("jobs") or {}
    sels = [(jid, j) for jid, j in jobs.items()
            if j.get("tool") == "relion_selection"]
    exps = [(jid, j) for jid, j in jobs.items()
            if j.get("stage_id") == "ts_export_particles"]
    check("drawer re-extract (canvas): the selection becomes a card",
          len(sels) == 1)
    check("…as a completed relion4_result node",
          bool(sels) and sels[0][1].get("stage_id") == "relion4_result"
          and sels[0][1].get("status") == "completed")
    check("…with its particle count read once at creation",
          bool(sels) and str(sels[0][1].get("params", {}).get("n_particles")) == "3")
    check("drawer re-extract (canvas): an EXPORT card is created, no converter",
          len(exps) == 1 and not any(j.get("stage_id") == "relion4_to_warp"
                                     for j in jobs.values()))
    exp = exps[0][1] if exps else {}
    check("…wired to the selection card",
          bool(exps) and bool(sels)
          and exp.get("inputs", {}).get("processing") == sels[0][0])
    check("…reading the RELION star directly",
          exp.get("params", {}).get("input_star") == star_rel)
    check("…with coords_angpix taken from the star's optics block",
          exp.get("params", {}).get("coords_angpix") == "6.28")
    check("…and '0-1 fractions' off",
          exp.get("params", {}).get("normalized_coords") is False)
    check("…and its builder was opened",
          any(t == "builder" for _tg, t in pops))
    # Re-running the same drawer action must not mint a second selection card.
    w._open_relion_export(orph)
    jobs2 = app.load_jobs(root).get("jobs") or {}
    check("re-opening reuses the selection card",
          sum(1 for j in jobs2.values()
              if j.get("tool") == "relion_selection") == 1)

    # Lists view keeps the classic behaviour: prefill + select the stage form.
    w2 = Win(root)
    w2._param_store = {}
    w2._view_mode = "lists"
    w2._open_relion_export(orph)
    check("drawer re-extract (lists): opens the export form",
          getattr(w2, "_selected", "") == "ts_export_particles")
    check("…with the star and its pixel size prefilled",
          w2._param_store.get("ts_export_particles", {}).get("input_star") == star_rel
          and w2._param_store.get("ts_export_particles", {}).get("coords_angpix")
          == "6.28")

    # The pre-flight refuses a coords_angpix that is not the star's own.
    blocked = []
    w2._log = lambda msg, kind="info": blocked.append((kind, msg))
    ok = w2._check_direct_export({"input_star": star_rel, "coords_angpix": "12.56"},
                                 interactive=False)
    check("a wrong coords_angpix is refused before Warp runs", ok is False
          and any("coords_angpix is 12.56" in m for _k, m in blocked))
    check("the star's own pixel size passes",
          w2._check_direct_export({"input_star": star_rel, "coords_angpix": "6.28"},
                                  interactive=False) is True)
    check("a missing star is refused",
          w2._check_direct_export({"input_star": "nope/particles.star",
                                   "coords_angpix": "6.28"}, interactive=False)
          is False)
    check("the pick-star route is not touched by the check",
          w2._check_direct_export({"input_directory": "warp_tiltseries/matching",
                                   "coords_angpix": "12.56"}, interactive=False)
          is True)


print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
