"""Tests for the unified job/queue model.

THE POINT: the queue used to be an in-memory list of (label, command, stage_id)
tuples, separate from the job store. That is why queued work could not appear on
the canvas, could not be restarted, and vanished when the app closed. A queued item
is now an ordinary job with status='queued', so this suite asserts the properties
that unification is supposed to buy:

  * queued items are jobs -> they land in canvas_layout as cards
  * order is run order (J2 before J10 — numeric, not string)
  * queued work survives a reload of the store (persistence)
  * requeue flips a failed job back to waiting without cloning it
  * cancelling removes only waiting jobs, never the running one

    python3 tests/test_queue.py
"""
import sys
import tempfile
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE / "stub"))

matches = sorted(REPO.glob("*tomogration_app.py"))
if not matches:
    print("FAIL  could not find *tomogration_app.py next to tests/")
    sys.exit(2)
spec = importlib.util.spec_from_file_location("tomapp", matches[0])
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}")


def enqueue(root, stage_id, cmd):
    """What _enqueue_current does, minus the Qt form."""
    job = app.new_job(root, stage_id, f"{stage_id} (queued)", {}, inputs={})
    app.update_job(root, job["id"], status="queued", command=cmd, trunk=True)
    return job["id"]


def main():
    root = tempfile.mkdtemp()

    # ---- queued items are jobs, ordered by run order -----------------------
    a = enqueue(root, "ts_ctf", "echo A")
    b = enqueue(root, "ts_reconstruct", "echo B")
    pend = app.queued_jobs(app.load_jobs(root))
    check("both queued items are jobs", len(pend) == 2)
    check("queue order is creation order", [j["id"] for j in pend] == [a, b])
    check("queued job carries its command", pend[0]["command"] == "echo A")

    # numeric ordering: J2 must precede J10 (string sort would invert them)
    for _ in range(8):
        enqueue(root, "ts_ctf", "echo filler")
    j10 = enqueue(root, "ts_ctf", "echo ten")
    ids = [j["id"] for j in app.queued_jobs(app.load_jobs(root))]
    check("J-numbers sort numerically (J2 before J10)",
          ids.index(b) < ids.index(j10) and ids[-1] == j10)

    # ---- persistence: a fresh read of the store still sees the queue -------
    check("queue survives a store reload",
          len(app.queued_jobs(app.load_jobs(root))) == 11)

    # ---- queued jobs render as CARDS on the canvas -------------------------
    nodes, _ = app.canvas_layout(app.load_jobs(root))
    qnodes = [n for n in nodes if n.get("status") == "queued"]
    check("queued jobs become canvas cards", len(qnodes) == 11)
    check("queued cards are not ghosts", all(not n["is_ghost"] for n in qnodes))
    check("queued style exists (blue)", "queued" in app._CARD_STYLE)

    # ---- requeue: failed -> waiting, SAME record (no clone) ----------------
    app.update_job(root, a, status="failed", exit_code=1)
    before = len(app.load_jobs(root)["jobs"])
    app.update_job(root, a, status="queued", exit_code=None, finished=None)
    after = app.load_jobs(root)["jobs"]
    check("requeue does not create a second job", len(after) == before)
    check("requeued job is waiting again", after[a]["status"] == "queued")
    check("requeue clears the old exit code", after[a]["exit_code"] is None)

    # ---- cancel clears ONLY waiting jobs ----------------------------------
    app.update_job(root, b, status="running")
    for j in app.queued_jobs(app.load_jobs(root)):
        app.delete_job(root, j["id"])
    left = app.load_jobs(root)["jobs"]
    check("cancel removed every queued job",
          not app.queued_jobs(app.load_jobs(root)))
    check("cancel did NOT remove the running job", b in left)
    check("running job still marked running", left[b]["status"] == "running")

    # ---- a job with no command is not silently runnable --------------------
    empty = app.new_job(root, "ts_ctf", "no command", {})
    app.update_job(root, empty["id"], status="queued")
    check("queued job with no command is detectable",
          not (app.load_jobs(root)["jobs"][empty["id"]].get("command") or ""))

    # ---- stale 'running' jobs are reconciled at startup --------------------
    # A running job cannot outlive the app process; if the app is killed the
    # record stays 'running' forever, every sibling card looks live, and the
    # queue refuses to start anything because it thinks it is busy.
    root2 = tempfile.mkdtemp()
    r1 = app.new_job(root2, "ts_export_particles", "one", {})
    r2 = app.new_job(root2, "ts_export_particles", "two", {})
    ok = app.new_job(root2, "ts_ctf", "done", {})
    app.update_job(root2, r1["id"], status="running")
    app.update_job(root2, r2["id"], status="running")
    app.update_job(root2, ok["id"], status="completed", exit_code=0)
    cleared = app.reconcile_running(root2)
    js = app.load_jobs(root2)["jobs"]
    check("reconcile reports both stale jobs",
          sorted(cleared) == sorted([r1["id"], r2["id"]]))
    check("stale running -> failed", js[r1["id"]]["status"] == "failed")
    check("stale jobs flagged interrupted", js[r2["id"]].get("interrupted") is True)
    check("completed jobs untouched", js[ok["id"]]["status"] == "completed")
    check("reconcile is idempotent", app.reconcile_running(root2) == [])


    # ---- a FORK must be runnable ------------------------------------------
    # Forking used to copy params but not the command, so every forked card said
    # "J## has no command — delete it and re-queue". A fork of a script-style job
    # (whose command build_job_command cannot model) must inherit the literal one.
    root3 = tempfile.mkdtemp()
    src = app.new_job(root3, "m_core", "MCore round 1", {"population": "m/x.population"})
    app.update_job(root3, src["id"], status="completed", exit_code=0,
                   command="MCore --population m/x.population --refine_particles")
    s0 = app.load_jobs(root3)
    fork = app.new_job(root3, s0["jobs"][src["id"]]["stage_id"],
                       "MCore round 1 (fork)", s0["jobs"][src["id"]]["params"],
                       s0["jobs"][src["id"]].get("inputs", {}))
    # what _fork_job now does when build_job_command yields nothing
    app.update_job(root3, fork["id"], command=s0["jobs"][src["id"]]["command"])
    js = app.load_jobs(root3)["jobs"]
    check("fork is a separate job", fork["id"] != src["id"])
    check("fork inherits a runnable command",
          bool(js[fork["id"]].get("command")))
    check("fork command matches the source",
          js[fork["id"]]["command"] == js[src["id"]]["command"])
    check("forking does not disturb the source",
          js[src["id"]]["status"] == "completed")

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
