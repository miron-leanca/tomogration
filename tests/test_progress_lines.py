"""Progress lines must collapse to ONE updating line, in both dialects.

WarpTools counts in integers ("239/5439, 08:06:22 remaining"); RELION counts in
DECIMAL minutes and redraws a little ASCII fish ("0.58/2.13 min ....~~(,_,\">").
The original pattern was integer-only, so every RELION tick appended a fresh line
and buried the log. Real log lines from both tools, plus lines that must NOT be
mistaken for progress.

    python3 tests/test_progress_lines.py
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE / "stub"))
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("tomapp", REPO / "tomogration_app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
from tomogration_core import progress_key
PAT = app._PROGRESS_RE


def is_progress(line):
    return progress_key(line) is not None

passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}")


PROGRESS = [
    "239/5439, 08:06:22 remaining",
    "5439/5439, previous metadata found for 278",
    "258/290, 02:51 remaining",
    "290/290, previous metadata found for 290",
    '0.58/2.13 min ................~~(,_,">',
    '0.95/2.15 min .........~~(,_,">',
    '1.27/1.27 min ...........................~~(,_,">',
    '000/??? sec ~~(,_,"> [oo]',
    '   .....~~(,_,">',
    "37/ 37 sec .........................~~(,_,\">",
    # counter LAST — MTools create_source prints 290 of these while hashing
    "Calculating data hashes... 285/290",
    "Calculating data hashes... 290/290",
    "Processing item 12/290",
]

NOT_PROGRESS = [
    "Connecting to workers...",
    "Connected to 8 workers",
    "Found 26641 particles in 289 tilt series",
    "relion/4.0.1 loaded",
    "+ Back-projecting all images ...",
    "Written /ceph/users/x/matching_conv.star",
    "Saying goodbye to all workers...",
    "Auto-refine: Iteration= 1",
    "ERROR: Cannot read file x.mrc It does not exist",
    "--- J24 completed (exit 0) ---",
    "",
    "Estimating initial noise spectra from 1000 particles",
    "Committing initial version...",
    "Population created: m/EML45-spike-closed.population",
]


def main():
    for line in PROGRESS:
        check(f"collapses: {line[:44]!r}", is_progress(line))
    for line in NOT_PROGRESS:
        check(f"kept as-is: {line[:44]!r}", not is_progress(line))

    # consecutive ticks of the SAME indicator share a key (so they replace each
    # other); a different indicator gets a different key and starts a new line.
    check("same indicator shares a key",
          progress_key("Calculating data hashes... 285/290")
          == progress_key("Calculating data hashes... 286/290"))
    check("different indicators differ",
          progress_key("Calculating data hashes... 285/290")
          != progress_key("239/5439, 08:06:22 remaining"))
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
