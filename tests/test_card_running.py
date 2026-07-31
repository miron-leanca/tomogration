"""Only ONE card may light up as running.

THE BUG: starting one "Extract particles" job turned every Extract card amber.
_add_card fell back to matching on stage_id, which is true for every job of that
stage. Trunk runs (the ▶ Run path) genuinely have no job record, which is why the
fallback existed — but it must resolve to the stage's ghost/template card, never
to sibling jobs.

    python3 tests/test_card_running.py
"""
import sys
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE / "stub"))
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("tomjobs", REPO / "tomogration_jobs.py")
jobs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jobs)
R = jobs.card_is_running

passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}")


def card(jid, stage="ts_export_particles", status="completed", ghost=False):
    return {"id": jid, "stage_id": stage, "status": status, "is_ghost": ghost}


def main():
    # the exact reported scenario: four Extract jobs, ONE running
    j19, j20, j24, j25 = (card("J19"), card("J20"), card("J24"),
                          card("J25", status="queued"))
    ghost = card("ghost:ts_export_particles", status="ghost", ghost=True)
    live = {"running": True, "job_id": "J24", "progress": "12/290"}

    check("the running job lights up", R(j24, live))
    check("sibling J19 does NOT light up", not R(j19, live))
    check("sibling J20 does NOT light up", not R(j20, live))
    check("queued sibling does NOT light up", not R(j25, live))
    check("the stage ghost does NOT light up", not R(ghost, live))
    check("exactly one card lights",
          sum(bool(R(c, live)) for c in (j19, j20, j24, j25, ghost)) == 1)

    # a TRUNK run (▶ Run, no job record) -> the ghost only
    trunk = {"running": True, "stage_id": "ts_export_particles"}
    check("trunk run lights the stage ghost", R(ghost, trunk))
    check("trunk run does NOT light real jobs",
          not any(R(c, trunk) for c in (j19, j20, j24, j25)))
    check("trunk run leaves other stages alone",
          not R(card("J3", stage="ts_ctf", status="ghost", ghost=True), trunk))

    # a job whose STORE record says running is always live (e.g. after a repaint)
    check("stored running status wins", R(card("J9", status="running"), {}))

    # nothing running -> nothing lights
    for a in ({}, None, {"running": False, "job_id": "J24"}):
        check(f"idle ({a}) lights nothing",
              not any(R(c, a) for c in (j19, j20, j24, ghost)))

    # a job run must not light a same-id card of a DIFFERENT stage (defensive)
    check("id match is exact",
          not R(card("J240"), {"running": True, "job_id": "J24"}))

    # ---- after a run FINISHES, nothing may still look live ------------------
    # A trunk run's "running" look lives only in the app's _active dict; if the
    # completion path forgets to clear it and repaint, the stage's ghost card keeps
    # the amber it had at the last repaint and looks stuck mid-run forever. That is
    # exactly what "job is done but still shows up in orange" was.
    finished = {}                      # what _active_info() returns once idle
    check("finished trunk run lights nothing",
          not any(R(c, finished) for c in (j19, j20, j24, j25, ghost)))
    stale = {"running": False, "stage_id": "ts_export_particles"}
    check("stale stage marker with runner idle lights nothing",
          not any(R(c, stale) for c in (j19, j24, ghost)))
    # and a card whose STORE status is terminal must not be amber even mid-run
    for st in ("completed", "failed", "queued", "ghost"):
        check(f"status {st!r} is not running while another job runs",
              not R(card("Jx", status=st), {"running": True, "job_id": "J24"}))

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
