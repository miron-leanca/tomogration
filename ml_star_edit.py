#!/usr/bin/env python3
# ml_star_edit.py
#
# Zero columns and/or take a random subset of a particle STAR, preserving the
# optics block and column order.
#
# WHY: to ask "are these particles centred on a common object?" you need an
# UNORIENTED average -- angles zeroed, so relion_reconstruct stacks them as
# extracted instead of rotating each one. And to ask it about the particles a
# selection kept, you want their ORIGINAL coordinates, with the refined
# origins zeroed so nothing is shifted. Both are one-line edits to a star, and
# doing them by hand on a 28,850-row file is how mistakes get made.
#
#   python3 ml_star_edit.py in.star --out out.star \
#           [--zero-angles] [--zero-origins] [--subset 1000] [--seed 0]
import argparse
import os
import random
import sys

ANGLES = ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")
ORIGINS = ("rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst",
           "rlnOriginX", "rlnOriginY", "rlnOriginZ")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("star")
    ap.add_argument("--out", required=True)
    ap.add_argument("--zero-angles", action="store_true")
    ap.add_argument("--zero-origins", action="store_true")
    ap.add_argument("--subset", type=int, default=0,
                    help="keep this many random particles (0 = all)")
    ap.add_argument("--tomo", default="",
                    help="keep only particles from tomograms whose name "
                         "contains this, e.g. Position003. Lets a reference be "
                         "built from ONE tomogram, which separates a "
                         "per-tomogram fault from one that only appears when "
                         "many are combined.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    if not os.path.isfile(a.star):
        sys.exit(f"ERROR: not found: {a.star}")

    # Streamed, block by block: only the LAST loop's rows are edited, which is
    # the particles block in every RELION 4 star. The optics block is copied
    # through untouched -- rewriting it is how pixel sizes get lost.
    out_lines, cur_cols, rows, block, in_loop = [], [], [], "", False
    blocks = []
    for raw in open(a.star):
        s = raw.rstrip("\n")
        t = s.strip()
        if t.startswith("data_"):
            blocks.append({"name": block, "head": out_lines, "cols": cur_cols,
                           "rows": rows})
            out_lines, cur_cols, rows = [s], [], []
            block, in_loop = t[5:], False
            continue
        if t.startswith("loop_"):
            in_loop = True
            out_lines.append(s)
            continue
        if t.startswith("_rln"):
            cur_cols.append(t.split()[0][1:])
            out_lines.append(s)
            continue
        if in_loop and t and not t.startswith("#"):
            rows.append(t.split())
            continue
        out_lines.append(s)
    blocks.append({"name": block, "head": out_lines, "cols": cur_cols,
                   "rows": rows})
    blocks = [b for b in blocks if b["head"] or b["rows"]]

    target = None
    for b in blocks:
        if b["rows"] and any(c.startswith("rlnCoordinate") or
                             c == "rlnImageName" for c in b["cols"]):
            target = b
    if target is None:
        sys.exit("ERROR: no particle block found")

    idx = {c: i for i, c in enumerate(target["cols"])}
    zeroed = []
    for group, want in ((ANGLES, a.zero_angles), (ORIGINS, a.zero_origins)):
        if not want:
            continue
        for c in group:
            if c in idx:
                zeroed.append(c)
    n_before = len(target["rows"])
    if a.tomo:
        # Match on the tomogram/micrograph name, falling back to the image
        # path: a Warp export names subtomo/<series>/<series>_NNNNN.mrc, so the
        # series is recoverable even when no tomo column is present.
        keys = [c for c in ("rlnTomoName", "rlnMicrographName", "rlnImageName")
                if c in idx]
        if not keys:
            sys.exit("ERROR: --tomo needs a tomogram, micrograph or image column")
        kept = [r for r in target["rows"]
                if any(a.tomo in r[idx[k]] for k in keys if idx[k] < len(r))]
        if not kept:
            sys.exit(f"ERROR: no particles matched --tomo {a.tomo}")
        target["rows"] = kept
    for r in target["rows"]:
        for c in zeroed:
            if idx[c] < len(r):
                r[idx[c]] = "0.000000"
    if a.subset and a.subset < n_before:
        random.Random(a.seed).shuffle(target["rows"])
        target["rows"] = target["rows"][:a.subset]

    with open(a.out, "w") as fh:
        for b in blocks:
            for l in b["head"]:
                fh.write(l + "\n")
            for r in b["rows"]:
                fh.write("  ".join(r) + "\n")
            fh.write("\n")

    print(f"{a.star}  ->  {a.out}")
    print(f"  particles: {n_before} -> {len(target['rows'])}"
          + (f"   (--tomo {a.tomo})" if a.tomo else ""))
    print(f"  zeroed   : {', '.join(zeroed) if zeroed else '(nothing)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
