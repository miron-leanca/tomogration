"""Integration test for the Phase-1b job dispatch wiring.

The full Tomogration window won't construct off-Qt (a UI combo-index compare),
so we subclass it and SKIP __init__, wiring up only the state the job path
touches. That still exercises the REAL _run_job / _finalize_job / _on_finished
methods against a temp project — verifying the store lifecycle and that the
three-column stage path is left alone.

    python3 tests/test_dispatch.py
"""
import sys
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
    check("active_job_id cleared after finish", win._active_job_id is None)

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

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
