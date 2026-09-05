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
    # tqdm, from IsoNet 2. The description + bar runs past the 40-character
    # prose guard, so these were called prose and 35 lines a SECOND reached the
    # log during a 12-hour predict.
    "Predicting Tomogram 34:  77%|███████▋  | 721/940 "
    "[00:20<00:06, 36.41it/s]",
    "Predicting Tomogram 34: 100%|██████████| 940/940 "
    "[00:26<00:00, 35.22it/s]",
    "Predict:  47%|████      | 34/72 [2:07:57<2:20:59, 222.62s/tomogram]",
    "Averaging even and odd tomograms:  20%|██        | 1/5 "
    "[00:10<00:41, 10.34s/ tomograms]",
    "Preprocess tomograms:   0%|          | 0/5 [00:00<?, ?it/s]",
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
    # A percentage in prose is not a bar: the '%|' pair is what makes tqdm
    # unmistakable, and these must still reach the log.
    "Masked 50% of the volume",
    "MinQuality = 0.8, 95% of series pass",
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

    # A tqdm bar's identity is its DESCRIPTION: the percentage, the bar and the
    # counter all change every tick, so keying on them would make each tick a
    # new indicator.
    _t = lambda pct, n: (f"Predicting Tomogram 34: {pct:3d}%|"
                         f"{'█' * (pct // 10):<10}| {n}/940 [00:20<00:06, 36.4it/s]")
    check("tqdm: consecutive ticks are the same indicator",
          progress_key(_t(77, 721)) == progress_key(_t(78, 733)))
    check("tqdm: the key is the description, not the bar",
          progress_key(_t(77, 721)) == "Predicting Tomogram 34")
    check("tqdm: a different tomogram is a different indicator",
          progress_key(_t(77, 721))
          != progress_key("Predicting Tomogram 35:  77%|███ | 721/940 [00:20<00:06]"))
    check("tqdm: the outer bar keeps its own identity",
          progress_key("Predict:  47%|████ | 34/72 [2:07:57<2:20:59]") == "Predict")

    # Keras' Progbar (IsoNet refine) rewinds the cursor with backspaces before
    # each redraw. splitlines() breaks at the \r, so those arrived as lines of
    # pure \x08 and rendered as tofu blocks all down the log.
    keras = "\x08" * 40 + "160/200 [====>....] - ETA: 12s - loss: 0.1403"
    check("redraw codes are stripped from the visible line",
          app.clean_stream_line(keras)
          == "160/200 [====>....] - ETA: 12s - loss: 0.1403")
    check("and the cleaned line is still recognised as progress",
          is_progress(app.clean_stream_line(keras)))
    check("ANSI colour codes go too",
          app.clean_stream_line("\x1b[1;32mdone\x1b[0m") == "done")
    check("ordinary text and tabs are untouched",
          app.clean_stream_line("deconv:\tPosition003 | pixel: 12.56")
          == "deconv:\tPosition003 | pixel: 12.56")
    check("a chunk of nothing but redraw codes carries no message",
          not app.clean_stream_line("\x08" * 60).strip())
    # ---- per-item skips -------------------------------------------------------
    # Exporting picks made on 4 tomograms walks all 72 in the settings file and
    # announces every skip. On 2026-08-28 that put 68 identical lines between
    # "Found 18431 particles in 4 tilt series" and "Finished processing in
    # 00:02:02", and a run that had WORKED was read as a failure. They are state,
    # not history — one updating line in the status strip, none in the log.
    for _l in ("no particles found in Position004.tomostar, skipping...",
               "no particles found in Position116.tomostar, skipping...",
               "nothing to do for Position010, skipping"):
        check(f"skip collapses: {_l[:38]}", progress_key(_l) == "~skip~")
    check("every skip shares ONE key, so 68 of them are one line",
          progress_key("no particles found in Position004.tomostar, skipping...")
          == progress_key("no particles found in Position116.tomostar, skipping..."))

    # The lines that carry the RESULT must never be swallowed by that rule.
    for _l in ("Found 18431 particles in 4 tilt series",
               "Finished processing in 00:02:02",
               "Connected to 4 workers",
               "Done",
               "ERROR: skipping is not an option here, the file is missing"):
        check(f"kept in the log: {_l[:40]}", progress_key(_l) is None)

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
