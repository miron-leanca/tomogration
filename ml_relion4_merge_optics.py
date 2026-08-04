#!/usr/bin/env python3
"""ml_relion4_merge_optics.py — collapse a RELION star's optics groups into one.

WHY THIS EXISTS
    RELION estimates the initial noise spectrum PER OPTICS GROUP, sampling up to
    1000 particles from each. With one group that is a few seconds. With 267 it is
    hours before iteration 1 begins:

        2   optics groups  ->  "Estimating initial noise spectra from 1000 particles"
        267 optics groups  ->  "... from 266000 particles"   (5.9 hours)

    267 groups is not a mistake in the export. Optics groups are formed by grouping
    particles with IDENTICAL optics parameters, and after M refines spherical
    aberration per tilt series (MCore --ctf_cs), every series carries its own Cs —
    so every series becomes its own group. The information is real; it is just far
    more precision than 3D CLASSIFICATION can use, and it costs 266x the noise
    estimation.

    So: collapse for classification, keep the per-series groups for a final
    refinement where the CTF precision actually shows.

WHAT IT DOES
    Writes a NEW star (the input is never modified) with a single optics group.
    Values for that group are taken from the most common row, except any you
    override with --cs / --q0 / --angpix / --box. Every particle is reassigned to
    group 1. Anything else in the file is passed through untouched.

Usage:
    python3 ml_relion4_merge_optics.py <in.star> [-o <out.star>] [--cs 2.7]
    python3 ml_relion4_merge_optics.py <in.star> --report      # just count groups

Exit 0 on success, 2 on a malformed or unreadable star.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys


def parse_star(text):
    """Split a star into ordered blocks: [(name, header_lines, label_list, rows)].

    Rows are lists of fields. A block with no loop_ keeps its lines verbatim in
    `header_lines` and has an empty label list, so simple key/value blocks survive
    a round trip unchanged.
    """
    blocks, name, header, labels, rows, in_loop = [], None, [], [], [], False
    for raw in text.splitlines():
        s = raw.strip()
        if s.startswith("data_"):
            if name is not None:
                blocks.append([name, header, labels, rows])
            name, header, labels, rows, in_loop = s, [raw], [], [], False
            continue
        if name is None:                       # preamble before any data_ block
            blocks.append(["", [raw], [], []])
            continue
        if s.startswith("loop_"):
            in_loop = True
            header.append(raw)
            continue
        if s.startswith("_") and in_loop:
            labels.append(s.split()[0])
            header.append(raw)
            continue
        if in_loop and s and not s.startswith("#"):
            rows.append(s.split())
            continue
        header.append(raw)
    if name is not None:
        blocks.append([name, header, labels, rows])
    return blocks


def find_block(blocks, want):
    for b in blocks:
        if b[0].startswith(f"data_{want}"):
            return b
    return None


def merge_optics(text, overrides=None):
    """Return (new_text, n_before, n_after, chosen_row_dict).

    The kept row is the MODE — the parameter set shared by most optics groups —
    rather than the first, so one odd series cannot define the whole dataset.
    """
    overrides = overrides or {}
    blocks = parse_star(text)
    opt = find_block(blocks, "optics")
    par = find_block(blocks, "particles")
    if opt is None or not opt[2]:
        raise ValueError("no data_optics loop found")
    labels, rows = opt[2], opt[3]
    n_before = len(rows)
    if n_before == 0:
        raise ValueError("data_optics has no rows")

    idx = {lab: i for i, lab in enumerate(labels)}
    # Ignore the group number and name when deciding which row is typical: those
    # differ by construction and would make every row unique.
    ignore = {"_rlnOpticsGroup", "_rlnOpticsGroupName"}
    keyed = [tuple(v for lab, v in zip(labels, r) if lab not in ignore) for r in rows]
    common = collections.Counter(keyed).most_common(1)[0][0]
    keep = list(rows[keyed.index(common)])

    for lab, i in idx.items():
        if lab == "_rlnOpticsGroup":
            keep[i] = "1"
        elif lab == "_rlnOpticsGroupName":
            keep[i] = "opticsGroup1"
    for lab, val in overrides.items():
        if lab in idx and val is not None:
            keep[idx[lab]] = str(val)
    opt[3] = [keep]

    if par is not None and "_rlnOpticsGroup" in par[2]:
        gi = par[2].index("_rlnOpticsGroup")
        for r in par[3]:
            if gi < len(r):
                r[gi] = "1"

    kept = dict(zip(labels, keep))       # BEFORE the loop below rebinds `labels`
    out = []
    for _name, header, _labels, rows in blocks:
        # Trailing blanks in a block's header would land BETWEEN the loop_ labels
        # and the first data row. RELION's parser treats a blank line as the end of
        # the loop, so that would silently truncate the block to zero rows.
        while header and not header[-1].strip():
            header.pop()
        out.extend(header)
        for r in rows:
            out.append("  " + "  ".join(r))
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n", n_before, 1, kept


def count_groups(text):
    blocks = parse_star(text)
    opt = find_block(blocks, "optics")
    par = find_block(blocks, "particles")
    return (len(opt[3]) if opt else 0), (len(par[3]) if par else 0)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Collapse a RELION star's optics groups into a single group.")
    ap.add_argument("star", help="input star (never modified)")
    ap.add_argument("-o", "--out", default="",
                    help="output star (default: <in>_1optics.star)")
    ap.add_argument("--report", action="store_true",
                    help="just report the group/particle counts and exit")
    ap.add_argument("--cs", default=None,
                    help="override _rlnSphericalAberration (mm), e.g. 2.7")
    ap.add_argument("--q0", default=None, help="override _rlnAmplitudeContrast")
    ap.add_argument("--angpix", default=None, help="override _rlnImagePixelSize")
    ap.add_argument("--box", default=None, help="override _rlnImageSize")
    args = ap.parse_args(argv)

    try:
        text = open(args.star, errors="replace").read()
    except OSError as exc:
        print(f"ERROR: cannot read {args.star}: {exc}", file=sys.stderr)
        return 2

    try:
        n_opt, n_par = count_groups(text)
    except Exception as exc:
        print(f"ERROR: {args.star} is not a star file I can parse: {exc}",
              file=sys.stderr)
        return 2

    print("=" * 67)
    print(f"ml_relion4_merge_optics    {args.star}")
    print(f"optics groups: {n_opt}     particles: {n_par}")
    # RELION samples up to 1000 particles PER GROUP for the initial noise spectrum.
    print(f"RELION will estimate initial noise spectra from ~{max(0, n_opt - 1) * 1000} "
          f"particles")
    print("=" * 67)
    if args.report:
        if n_opt > 2:
            print(f"{n_opt} groups is a lot. Re-run without --report to collapse them.")
        return 0

    overrides = {"_rlnSphericalAberration": args.cs,
                 "_rlnAmplitudeContrast": args.q0,
                 "_rlnImagePixelSize": args.angpix,
                 "_rlnImageSize": args.box}
    try:
        new_text, before, after, kept = merge_optics(text, overrides)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    out = args.out or (os.path.splitext(args.star)[0] + "_1optics.star")
    try:
        open(out, "w").write(new_text)
    except OSError as exc:
        print(f"ERROR: cannot write {out}: {exc}", file=sys.stderr)
        return 2

    print(f"Collapsed {before} optics groups -> {after}.")
    print("Kept these values for the single group:")
    for lab, val in kept.items():
        print(f"    {lab:<28} {val}")
    print(f"\nWrote {out}")
    print(f"Noise estimation should now report ~0-1000 particles instead of "
          f"{max(0, before - 1) * 1000}.")
    print("\nNOTE the per-series optics groups were REAL — M refines spherical")
    print("aberration per tilt series (--ctf_cs), so each series had its own Cs.")
    print("That precision is worth keeping for a FINAL refinement; this collapsed")
    print("star is for classification, where it only costs time.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
