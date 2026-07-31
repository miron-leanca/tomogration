#!/usr/bin/env python3
"""ml_m_check_ctf.py — find tilt series that were never CTF-estimated.

WHY THIS EXISTS
    A tilt series whose CTF estimation silently failed or was skipped still has an
    .xml in the processing folder, still imports into an M data source, and still
    reconstructs.  Everything looks normal.  Then MCore dies:

        System.IndexOutOfRangeException: Index was outside the bounds of the array.
          at Warp.TiltSeries.<>c__DisplayClass183_0
             .<PerformMultiParticleRefinement>b__19(Int32 t)
             ... TiltSeries.MPARefinement.cs:line 1250

    b__19(Int32 t) builds a PER-TILT array during multi-particle refinement.  With
    no CTF estimate there is nothing to index and it runs off the end.  The
    exception is raised inside a worker, so the tilt series name NEVER reaches
    stdout — with several workers the crash is not even deterministic.

    The trap is that `MCore --iter 0` succeeds anyway and reports a sensible
    resolution, because iter 0 performs no refinement and so never walks that
    array.  "iter 0 works, real refinement crashes" is the signature of this bug,
    and it is easy to misread as a problem with the mask, the population size, the
    particle star, or the refinement flags.  It is none of those.

    On EML45 this was 2 series out of 290.  Re-running ts_ctf on those two fixed
    it outright.

WHAT IT CHECKS
    1. CTFResolutionEstimate on the root <TiltSeries> element.  Missing, empty or
       0 means CTF estimation never produced a result.  This is the definitive
       test and the one that found the EML45 offenders.
    2. Whether the .xml contains any CTF-related elements at all.
    3. File size against the median.  A never-estimated .xml is dramatically
       shorter because the per-tilt CTF grids are absent.  A heuristic, reported
       as such — series legitimately differ in tilt count.

Usage:
    python3 ml_m_check_ctf.py <project_dir> [options]

      --processing DIR   tilt-series processing folder (default: warp_tiltseries)
      --tomostar DIR     .tomostar folder (default: tomostar)
      --write-list FILE  write the bad series' tomostar paths, one per line, for
                         `WarpTools ts_ctf --input_data`
      --quiet            only print the offenders and the summary

Exit code 0 = every series carries a CTF estimate.  1 = some do not (so it can
gate a script).  2 = could not run the check at all.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def series_ctf_state(xml_path: Path) -> dict:
    """Read one tilt-series .xml and report what it says about CTF.

    Never raises: an .xml that cannot be parsed is itself a finding, and the whole
    point of this tool is to survive a broken project and describe it.
    """
    info = {
        "name": xml_path.stem,
        "path": xml_path,
        "bytes": 0,
        "resolution": None,   # float, or None if the attribute is absent
        "ctf_elements": 0,
        "error": "",
    }
    try:
        info["bytes"] = xml_path.stat().st_size
    except OSError as exc:
        info["error"] = f"cannot stat: {exc}"
        return info

    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError) as exc:
        info["error"] = f"cannot parse: {exc}"
        return info

    raw = root.get("CTFResolutionEstimate")
    if raw is not None and raw.strip():
        try:
            info["resolution"] = float(raw)
        except ValueError:
            info["error"] = f"CTFResolutionEstimate is not a number: {raw!r}"

    # Schema-agnostic: count anything whose tag mentions CTF rather than assuming
    # Warp's element names, which differ between versions.
    info["ctf_elements"] = sum(
        1 for el in root.iter() if "ctf" in el.tag.lower()
    )
    return info


def count_tilts(tomostar_path: Path) -> int | None:
    """Number of data rows in a .tomostar, or None if it cannot be read."""
    try:
        lines = tomostar_path.read_text(errors="replace").splitlines()
    except OSError:
        return None
    in_loop = False
    n = 0
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("loop_"):
            in_loop = True
            continue
        if not in_loop:
            continue
        if s.startswith("_"):        # column header
            continue
        if s.startswith("data_"):    # a new block ends the loop
            in_loop = False
            continue
        n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Find tilt series that were never CTF-estimated.")
    ap.add_argument("project_dir", nargs="?", default=".",
                    help="Project root (holds warp_tiltseries/ and tomostar/).")
    ap.add_argument("--processing", default="warp_tiltseries",
                    help="Tilt-series processing folder, relative to project_dir.")
    ap.add_argument("--tomostar", default="tomostar",
                    help=".tomostar folder, relative to project_dir.")
    ap.add_argument("--write-list", default="",
                    help="Write the offenders' tomostar paths here, one per line.")
    ap.add_argument("--quiet", action="store_true",
                    help="Print only the offenders and the summary.")
    args = ap.parse_args(argv)

    root = Path(args.project_dir).expanduser().resolve()
    proc = root / args.processing
    tstar = root / args.tomostar

    if not proc.is_dir():
        print(f"ERROR: no processing folder at {proc}", file=sys.stderr)
        print("       pass --processing if yours is named differently.",
              file=sys.stderr)
        return 2

    xmls = sorted(proc.glob("*.xml"))
    if not xmls:
        print(f"ERROR: no tilt-series .xml files in {proc}", file=sys.stderr)
        return 2

    print("=" * 67)
    print("ml_m_check_ctf — CTF completeness")
    print(f"Project    : {root}")
    print(f"Processing : {proc.relative_to(root)}/  ({len(xmls)} tilt series)")
    print("=" * 67)

    rows = [series_ctf_state(p) for p in xmls]

    sizes = [r["bytes"] for r in rows if r["bytes"] > 0]
    median = statistics.median(sizes) if sizes else 0

    missing = [r for r in rows
               if r["error"] or r["resolution"] is None or r["resolution"] <= 0]
    ok = [r for r in rows if r not in missing]

    # "Much smaller than its peers" is a hint, not a verdict — a short series is
    # legitimately shorter.  Only report it for files that PASSED the real test,
    # so it adds information instead of repeating the offenders.
    suspicious = [r for r in ok
                  if median and r["bytes"] < median * 0.4]

    if not args.quiet and ok:
        res = [r["resolution"] for r in ok if r["resolution"]]
        if res:
            print(f"\n{len(ok)} series carry a CTF estimate  "
                  f"(resolution {min(res):.2f}–{max(res):.2f} Å, "
                  f"median {statistics.median(res):.2f} Å)")

    if missing:
        print(f"\n!! {len(missing)} SERIES WITHOUT A CTF ESTIMATE")
        print("   These will crash MCore with IndexOutOfRangeException the moment")
        print("   a real refinement runs (--iter 0 will keep succeeding).\n")
        for r in missing:
            n_tilts = count_tilts(tstar / f"{r['name']}.tomostar")
            bits = []
            if r["error"]:
                bits.append(r["error"])
            elif r["resolution"] is None:
                bits.append("no CTFResolutionEstimate attribute")
            else:
                bits.append(f'CTFResolutionEstimate="{r["resolution"]:g}"')
            bits.append(f"{r['ctf_elements']} CTF elements")
            bits.append(f"{r['bytes']} bytes")
            if median:
                bits.append(f"median {int(median)}")
            if n_tilts is not None:
                bits.append(f"{n_tilts} tilts")
            print(f"     {r['name']:<20} {', '.join(bits)}")

    if suspicious:
        print(f"\n   ({len(suspicious)} series have a CTF estimate but a much "
              f"smaller .xml than the rest —")
        print("    usually just a shorter tilt series, worth a look if M still "
              "misbehaves:")
        print("    " + ", ".join(r["name"] for r in suspicious) + ")")

    print("\n" + "-" * 67)
    if not missing:
        print(f"PASS — all {len(rows)} tilt series carry a CTF estimate.")
        print("If MCore still throws IndexOutOfRangeException, the cause is")
        print("elsewhere; reset M (M: reset / wipe) and rebuild the population.")
        return 0

    names = [r["name"] for r in missing]
    inputs = " ".join(f"{args.tomostar}/{n}.tomostar" for n in names)

    print(f"FAIL — {len(missing)} of {len(rows)} tilt series were never "
          f"CTF-estimated.")
    print("\nFix them, then rebuild the M population (the .source caches the")
    print("metadata, so an existing population will not pick up the new CTF):")
    print(f"\n    WarpTools ts_ctf --settings warp_tiltseries.settings \\")
    print(f"        --input_data {inputs} \\")
    print(f"        --defocus_max 8 --device_list 0")
    print("\nA series that failed at the default --defocus_max 5 usually just")
    print("sits outside the search range; widening it is the normal fix.")
    print("Re-run this check afterwards — it must report PASS.")

    if args.write_list:
        out = Path(args.write_list).expanduser()
        try:
            out.write_text("".join(
                f"{args.tomostar}/{n}.tomostar\n" for n in names))
            print(f"\nSeries list written to {out}")
        except OSError as exc:
            print(f"\nWARNING: could not write {out}: {exc}", file=sys.stderr)

    return 1


if __name__ == "__main__":
    sys.exit(main())
