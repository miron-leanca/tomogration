"""TERMINATE must reach the whole run, not just the shell it started.

QProcess.terminate() signals only the `bash -lc` it launched. A command that had
already spawned something — WarpTools spawning WarpWorker — left that child
running on the GPU with no way to stop it from the app. Reported 2026-08-28:
"I tried clicking Terminate but nothing happened."

The fix runs each job in its own process group and signals the GROUP. These
tests use start_new_session=True, which is what setsid does, so the group logic
is checked here even though setsid itself is util-linux only.

    python3 tests/test_terminate.py
"""
import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE / "stub"))
sys.path.insert(0, str(REPO))

passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}")


spec = importlib.util.spec_from_file_location("tomapp", REPO / "tomogration_app.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class FakeProc:
    def __init__(self, pid):
        self._pid = pid

    def processId(self):
        return self._pid


class Runner:
    """The real _signal_group, bound to a fake process."""
    _signal_group = A.ProcessRunner._signal_group
    _pgid = None

    def __init__(self, proc):
        self.proc = proc


def main():
    # ---- the guard ---------------------------------------------------------
    # If the run was NOT isolated it shares tomogration's process group, and
    # signalling that group would take the app down with the job.
    q = subprocess.Popen(["bash", "-lc", "sleep 5"])
    try:
        check("a plain spawn shares our group, so the guard must decline",
              os.getpgid(q.pid) == os.getpgid(0)
              and Runner(FakeProc(q.pid))._signal_group(signal.SIGTERM) is False)
        check("and the child is left alive for the caller's fallback",
              alive(q.pid))
    finally:
        q.kill()
        q.wait()

    check("a dead pid is declined, not signalled blindly",
          Runner(FakeProc(q.pid))._signal_group(signal.SIGTERM) is False)
    check("so is a bogus one", not any(
        Runner(FakeProc(v))._signal_group(signal.SIGTERM)
        for v in (0, -1, None, "x")))

    # ---- the bug, and the fix ----------------------------------------------
    p = subprocess.Popen(["bash", "-lc", "sleep 300 & echo $!; wait"],
                         stdout=subprocess.PIPE, text=True,
                         start_new_session=True)
    grandchild = int(p.stdout.readline().strip())
    time.sleep(0.3)
    pgid = os.getpgid(p.pid)
    check("an isolated run gets its own group, led by the tracked pid",
          pgid != os.getpgid(0) and pgid == p.pid)
    check("which the guard therefore accepts",
          Runner(FakeProc(p.pid))._signal_group(0) is True)

    # One runner across both signals, as the real one is: it learns the group
    # id now, while the pid still resolves.
    r = Runner(FakeProc(p.pid))
    r._signal_group(0)

    p.terminate()                       # what the old code did: bash only
    time.sleep(0.5)
    check("signalling only the shell ORPHANS what it spawned — the bug",
          alive(grandchild))

    # The escalation happens 3 s later, by which time the bash is a zombie and
    # getpgid cannot answer. Deriving the group fresh each time reached nothing
    # here, so a child ignoring SIGTERM survived exactly as before the fix.
    check("the group is remembered once the pid stops resolving",
          r._signal_group(signal.SIGKILL) is True)
    time.sleep(0.5)
    check("so the escalation still reaches the spawned child",
          not alive(grandchild))
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()

    # ---- and it must not break machines without setsid ---------------------
    src = (REPO / "tomogration_app.py").read_text()
    check("setsid is looked up, never assumed",
          'shutil.which("setsid")' in src)
    check("with a plain bash fallback, so a Mac still launches runs",
          'self.proc.start("bash", ["-lc", command])' in src)
    check("terminate falls back to the single process when not isolated",
          "if not self._signal_group(signal.SIGTERM):" in src
          and "self.proc.terminate()" in src)

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
