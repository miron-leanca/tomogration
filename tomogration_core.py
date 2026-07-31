#!/usr/bin/env python3
# Part of tomogration2 — split out of the single-file app so the pieces can be
# edited (and tested) independently. Import graph is a strict DAG:
#     core  ->  stages  ->  jobs        (project depends only on core)
# tomogration_app.py imports all of them and holds the Qt window.

"""Shared primitives with no dependencies of their own."""

import re
from pathlib import Path

# Companion scripts ship next to this file. Resolve them to ABSOLUTE paths so
# stage commands work regardless of which project root the user selects at
# startup (commands run with cwd = the selected data dir, not the app dir).
_PKG_DIR = Path(__file__).resolve().parent


def _pkg_script(name):
    """Absolute path to a shipped companion script, bash-quoted if it has spaces."""
    p = str(_PKG_DIR / name)
    return f'"{p}"' if (" " in p or "\t" in p) else p


def expand_tilt_ranges(text):
    """Expand IMOD tilt-number ranges to explicit comma-separated values, since
    remake_mdocs only handles individual numbers:
        '1,4-12,48' -> '1,4,5,6,7,8,9,10,11,12,48'
    Unparseable tokens are dropped; order is preserved, duplicates removed."""
    out, seen = [], set()
    for tok in str(text).replace(" ", "").split(","):
        if not tok:
            continue
        if "-" in tok:
            try:
                a, b = (int(x) for x in tok.split("-", 1))
            except ValueError:
                continue
            rng = range(a, b + 1) if a <= b else range(a, b - 1, -1)
        else:
            try:
                rng = [int(tok)]
            except ValueError:
                continue
        for n in rng:
            if n not in seen:
                seen.add(n)
                out.append(n)
    return ",".join(str(n) for n in out)


# A counter inside a progress line. The counter is NOT always at the start:
#   WarpTools : "239/5439, 08:06:22 remaining"        -> counter first
#   MTools    : "Calculating data hashes... 285/290"  -> counter LAST
#   RELION    : "0.58/2.13 min ....~~(,_,\">"          -> decimal, counter first
# Anchoring at the start only collapsed the first kind, so M's hashing printed 290
# separate lines and flooded the log.
_COUNTER_RE = re.compile(
    r"^(?P<pre>.*?)(?P<num>\d+(?:\.\d+)?\s*/\s*(?:\d+(?:\.\d+)?|\?+))")
# RELION also redraws a bare ASCII progress bar with no counter at all.
_FISH_RE = re.compile(r'^[.\s]*~~\(,_,')


def progress_key(line):
    """A stable identity for a self-updating progress line, or None.

    Consecutive lines sharing a key are the SAME progress indicator ticking, so the
    terminal replaces the last one in place instead of appending. Keying on the text
    around the counter (rather than just 'starts with a number') is what makes this
    work for tools that put the counter at the end of a label.
    """
    if not line or not line.strip():
        return None
    if _FISH_RE.match(line):
        return "~fish~"
    m = _COUNTER_RE.match(line)
    if not m:
        return None
    pre = m.group("pre")
    # A prefix must look like a LABEL, not prose that happens to contain "n/m".
    if pre and not pre.rstrip().endswith((".", ":", "-", "\u2026")) and len(pre) > 40:
        return None
    return pre.strip() or "~lead~"
