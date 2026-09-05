#!/usr/bin/env python3
"""Validate mdocs the way WarpTools ts_import reads them, and say which fail.

ts_import reports a per-series failure as one line among thousands
("Failed to parse Position017.mdoc: Value cannot be null. (Parameter 'key')")
and carries on, so a run ends "81/83, 2 failed" with no list of the two. This
finds them, and says WHY, without running Warp.

The checks are the shapes a dictionary-building parser dies on -- an entry
whose KEY is null or empty -- plus the structural damage that editing a mdoc by
hand (or by exclusion) can leave behind.

    python3 ml_check_mdocs.py [mdocs_dir] [--frames frames] [--verbose]

Exit 0 when every mdoc is clean, 1 when any is not.
"""
import argparse
import os
import sys


def parse(path):
    """(header_lines, [(zvalue_or_None, [lines]), ...]) -- no interpretation."""
    with open(path, errors="replace") as fh:
        lines = fh.read().splitlines()
    header, blocks, current, zval = [], [], None, None
    for line in lines:
        s = line.strip()
        if s.startswith("[ZValue"):
            if current is not None:
                blocks.append((zval, current))
            current, zval = [], _zvalue(s)
            continue
        (current if current is not None else header).append(line)
    if current is not None:
        blocks.append((zval, current))
    return header, blocks


def _zvalue(section):
    """The N in '[ZValue = N]', or None if it is not a plain integer."""
    body = section.strip().lstrip("[").rstrip("]")
    _, _, val = body.partition("=")
    try:
        return int(val.strip())
    except ValueError:
        return None


def _pairs(lines):
    """[(key, value)] for the 'key = value' lines, keys NOT stripped of empties
    -- an empty key is precisely what we are hunting."""
    out = []
    for line in lines:
        if "=" not in line or line.strip().startswith("["):
            continue
        key, _, val = line.partition("=")
        out.append((key.strip(), val.strip()))
    return out


def check(path, frames=None):
    """[problem strings] for one mdoc. Empty means it looks importable."""
    bad = []
    try:
        header, blocks = parse(path)
    except OSError as exc:
        return [f"unreadable: {exc}"]

    if not blocks:
        bad.append("no [ZValue] blocks at all")

    # A section header whose key is empty ('[ = x]', a bare '[]') is the direct
    # source of "Value cannot be null. (Parameter 'key')".
    for n, line in enumerate(header, 1):
        s = line.strip()
        if s.startswith("[") and not s.lstrip("[").partition("=")[0].strip():
            bad.append(f"line {n}: section header with no key: {s!r}")

    for key, _val in _pairs(header):
        if not key:
            bad.append("header has a line with an empty key (' = value')")
            break

    zvals = [z for z, _ in blocks]
    if any(z is None for z in zvals):
        bad.append("a [ZValue = ...] is not an integer")
    else:
        if len(set(zvals)) != len(zvals):
            bad.append(f"duplicate ZValue numbers: "
                       f"{sorted(z for z in set(zvals) if zvals.count(z) > 1)}")
        if zvals != list(range(len(zvals))):
            bad.append(f"ZValue numbering is not 0..{len(zvals) - 1} "
                       f"(got {zvals[:6]}{'...' if len(zvals) > 6 else ''})")

    for z, lines in blocks:
        pairs = dict(_pairs(lines))
        if any(not k for k, _ in _pairs(lines)):
            bad.append(f"ZValue {z}: a line with an empty key")
        # Warp keys its frame-series lookup on the SubFramePath BASENAME. A
        # block without one hands that lookup a null key.
        sub = pairs.get("SubFramePath", "")
        if not sub:
            bad.append(f"ZValue {z}: no SubFramePath")
        elif frames:
            name = sub.replace("\\", "/").rsplit("/", 1)[-1]
            if not os.path.isfile(os.path.join(frames, name)):
                bad.append(f"ZValue {z}: frame not in {frames}/: {name}")
        if not pairs.get("TiltAngle", ""):
            bad.append(f"ZValue {z}: no TiltAngle")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mdocs", nargs="?", default="mdocs", help="mdoc folder")
    ap.add_argument("--frames", default="", help="also check every SubFramePath "
                                                 "exists in this folder")
    ap.add_argument("--verbose", action="store_true", help="list clean files too")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.mdocs):
        print(f"not a folder: {args.mdocs}")
        return 2
    names = sorted(n for n in os.listdir(args.mdocs)
                   if n.lower().endswith(".mdoc")
                   and not n.lower().endswith("_override.mdoc"))
    if not names:
        print(f"no .mdoc files in {args.mdocs}")
        return 2

    n_bad = 0
    for name in names:
        problems = check(os.path.join(args.mdocs, name), args.frames or None)
        if problems:
            n_bad += 1
            print(f"\n{name}")
            for p in problems[:12]:
                print(f"    {p}")
            if len(problems) > 12:
                print(f"    ... and {len(problems) - 12} more")
        elif args.verbose:
            _h, blocks = parse(os.path.join(args.mdocs, name))
            print(f"{name}: ok ({len(blocks)} tilts)")

    print(f"\n{len(names)} mdoc(s) checked, {n_bad} with problems, "
          f"{len(names) - n_bad} clean")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
