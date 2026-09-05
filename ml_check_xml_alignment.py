#!/usr/bin/env python3
"""Which Warp XMLs actually carry an alignment — and therefore, where it landed.

A run with --output_processing writes its results into that job folder. For a
step that EDITS the shared per-series record rather than producing new data,
that matters enormously: miss-alignment, ts_ctf and ts_reconstruct all read
warp_tiltseries/, so an alignment imported into jobs/J18_.../ is invisible to
them and nothing says so.

VolumeDimensionsAngstrom is the tell: it is [0,0,0] until a tilt series has a
geometry, which is exactly what importing an alignment gives it. The
miss-alignment wrapper already relies on this to park un-alignable series.

    python3 ml_check_xml_alignment.py <dir> [<dir> ...]

Each dir is a folder of <series>.xml. Exit 0 if every dir has at least one
aligned series, 1 if any dir has none, 2 if nothing could be read.
"""
import os
import re
import sys

DIMS = re.compile(
    r"VolumeDimensionsAngstrom[^\d\-]*(-?[\d.]+)[,\s]+(-?[\d.]+)[,\s]+(-?[\d.]+)")


def aligned(path):
    """(has_alignment, dims) for one Warp tilt-series XML."""
    try:
        with open(path, errors="replace") as fh:
            text = fh.read()
    except OSError:
        return (False, None)
    m = DIMS.search(text)
    if not m:
        return (False, None)
    dims = tuple(float(v) for v in m.groups())
    return (any(d != 0 for d in dims), dims)


def scan(folder):
    """(n_aligned, n_total, [example dims]) for a folder of XMLs."""
    try:
        names = sorted(n for n in os.listdir(folder) if n.endswith(".xml"))
    except OSError:
        return (None, None, None)
    ok, example = 0, None
    for n in names:
        has, dims = aligned(os.path.join(folder, n))
        ok += bool(has)
        if has and example is None:
            example = (n, dims)
    return (ok, len(names), example)


def main(argv=None):
    dirs = list(argv or [])
    if not dirs:
        print(__doc__.strip().splitlines()[-4])
        return 2
    seen_any = False
    empty = 0
    for folder in dirs:
        n_ok, n_all, example = scan(folder)
        if n_ok is None:
            print(f"{folder}\n    cannot read this folder")
            continue
        seen_any = True
        if not n_all:
            print(f"{folder}\n    no .xml files here")
            empty += 1
            continue
        print(f"{folder}")
        print(f"    {n_ok} of {n_all} series carry an alignment")
        if example:
            name, dims = example
            print(f"    e.g. {name}: VolumeDimensionsAngstrom = "
                  f"{dims[0]:.0f} x {dims[1]:.0f} x {dims[2]:.0f}")
        else:
            print("    every one is [0,0,0] — nothing has been imported here")
            empty += 1
    if not seen_any:
        return 2
    print("\nThe folder that downstream steps read is warp_tiltseries/. "
          "If the alignments are only in a jobs/ folder, they are invisible "
          "to miss-alignment, ts_ctf and ts_reconstruct.")
    return 1 if empty else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
