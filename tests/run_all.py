#!/usr/bin/env python3
"""Run every tomogration test suite and summarise.

    python3 tests/run_all.py

Each tests/test_*.py is a standalone script (they stub PySide6 from tests/stub and
load tomogration_app by path), so this just executes them in turn and tallies the
"N passed, M failed" line each one prints. No pytest, no PySide6, no cluster — it
runs anywhere Python does, which is the point: the pure logic is what breaks, and
it can be checked from the dev box before anything ships to the VM.

Exit code is the number of failing SUITES (0 = all green), so it drops straight
into a pre-commit hook or CI.
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TALLY = re.compile(r"(\d+)\s+passed,\s+(\d+)\s+failed")


def main():
    suites = sorted(HERE.glob("test_*.py"))
    if not suites:
        print("no test_*.py found next to run_all.py")
        return 1
    total_pass = total_fail = bad_suites = 0
    crashed = []
    width = max(len(s.name) for s in suites)
    print("=" * (width + 34))
    for s in suites:
        r = subprocess.run([sys.executable, str(s)], capture_output=True, text=True)
        out = (r.stdout or "") + (r.stderr or "")
        m = None
        for line in out.splitlines():
            hit = TALLY.search(line)
            if hit:
                m = hit
        if m is None:                       # suite crashed before its tally line
            bad_suites += 1
            crashed.append(s.name)
            print(f"{s.name:<{width}}  CRASHED (exit {r.returncode})")
            print("\n".join(out.strip().splitlines()[-12:]))
            continue
        npass, nfail = int(m.group(1)), int(m.group(2))
        total_pass += npass
        total_fail += nfail
        status = "ok  " if nfail == 0 else "FAIL"
        print(f"{s.name:<{width}}  {status}  {npass:4d} passed  {nfail:3d} failed")
        if nfail:
            bad_suites += 1
            for line in out.splitlines():
                if line.startswith("FAIL"):
                    print(f"      {line}")
    print("=" * (width + 34))
    # The crash count belongs on the SUMMARY line, not only in the exit code.
    # A suite that dies before printing its tally contributes 0 failures, so
    # this line read "1242 passed, 0 failed" while two suites were not running
    # at all — which is exactly the moment it is most likely to be believed.
    line = f"{len(suites)} suite(s): {total_pass} passed, {total_fail} failed"
    if crashed:
        line += (f", {len(crashed)} CRASHED before reporting "
                 f"({', '.join(crashed)})")
    print(line)
    return bad_suites


if __name__ == "__main__":
    sys.exit(main())
