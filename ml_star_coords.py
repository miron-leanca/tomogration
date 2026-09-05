#!/usr/bin/env python3
# ml_star_coords.py
#
# What pixel size are a star's coordinates in? Print the answer, and the evidence.
#
# WHY: every RELION->Warp re-extraction failure is silent. --coords_angpix too large
# and the particles land off the edge; too small and they cluster near the origin.
# Warp writes a well-formed star either way, RELION refines it happily, and the only
# symptom is a blob hours later. The units are DECIDABLE from the file: coordinates
# span the tomogram, so their maximum tells you the pixel size they are counted in.
#
#   python3 ml_star_coords.py <star> [more.star ...] [--dims 512x512x386@12.56]
#
# --dims is the reconstruction the picks were made on. Given it, this prints the
# implied --coords_angpix for each star instead of leaving you to divide.
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ml_relion4_select_picks import (read_star_blocks, particles_block,  # noqa: E402
                                     star_pixel_size)

COORDS = ("rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ")
ORIGINS = ("rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst")


def col_stats(pb, name):
    i = pb["cols"].get(name)
    if i is None:
        return None
    vals = []
    for r in pb["rows"]:
        try:
            vals.append(float(r[i]))
        except (ValueError, IndexError):
            pass
    return (min(vals), max(vals), len(vals)) if vals else None


def report(path, dims=None):
    print(f"\n=== {path}")
    if not os.path.isfile(path):
        print("    MISSING")
        return
    blocks = read_star_blocks(path)
    pb = particles_block(blocks, need=("rlnCoordinateX",))
    if pb is None:
        print("    no block with _rlnCoordinateX — not a particle star")
        return
    print(f"    blocks: {', '.join(b['name'] or '(unnamed)' for b in blocks)}")
    print(f"    particles: {len(pb['rows'])}")
    stated = star_pixel_size(blocks, pb)
    print(f"    stated pixel size: "
          + (f"{stated:g} A/px" if stated else "NOT STATED"))

    for n in COORDS:
        st = col_stats(pb, n)
        print(f"    {n:20s} " + (f"{st[0]:10.2f} .. {st[1]:10.2f}"
                                 if st else "     (absent)"))
    if all(col_stats(pb, n) for n in ORIGINS):
        for n in ORIGINS:
            lo, hi, _ = col_stats(pb, n)
            print(f"    {n:20s} {lo:10.2f} .. {hi:10.2f}   (refined shifts, A)")
        print("    -> refined origins ARE present: re-extraction should RECENTRE.")
    else:
        print("    -> no refined origins; nothing to recentre on.")

    # An OUTPUT star (one an export WROTE) names the subtomograms it cut. Its
    # coordinates are a RESULT, not something to feed back in, so export advice
    # here is nonsense — J58's star was told to "export with --coords_angpix
    # 0.785", a number that means nothing and could only mislead.
    is_output = "rlnImageName" in pb["cols"]
    xs = col_stats(pb, "rlnCoordinateX")

    if is_output and dims and xs and stated:
        nx, apx = dims
        width_px = (nx * apx) / float(stated)
        print(f"    this star is an export OUTPUT (it names subtomograms).")
        print(f"    at its stated {float(stated):g} A/px the tomogram is "
              f"{width_px:.0f} px wide; max X is {xs[1]:.1f}")
        if xs[1] > width_px * 1.02:
            over = xs[1] / width_px
            print(f"    !! COORDINATES RUN OFF THE TOMOGRAM by {over:.1f}x — "
                  f"particles past {width_px:.0f} px were cut from outside the "
                  f"volume. The export used a --coords_angpix {over:.0f}x too "
                  f"large. DO NOT refine this.")
        else:
            print(f"    => within the volume; check placement with "
                  f"ml_verify_reextract.py <source.star> <this star>")
        return

    if xs and max(xs[1], 1) <= 1.5:
        print("    => coordinates are 0-1 FRACTIONS: export with "
              "--normalized_coords and NO --coords_angpix.")
        return
    if dims and xs:
        nx, apx = dims
        # Coordinates span the tomogram, so max_x ~ width in whatever pixel the
        # star counts in. width_A / max_x is that pixel size.
        implied = (nx * apx) / xs[1]
        print(f"    => tomogram is {nx:g} px at {apx:g} A/px = {nx * apx:g} A "
              f"wide; max X is {xs[1]:.1f}")
        # The raw estimate always reads HIGH: picks stop short of the edge, so
        # the observed span is smaller than the tomogram. Snap it to the real
        # candidates instead of reporting a number nobody should type.
        cands = {apx * (2.0 ** k) for k in range(-4, 3)}
        if stated:
            cands.add(float(stated))
        best = min(cands, key=lambda c: abs(implied - c))
        print(f"    => raw estimate ~{implied:.2f} A/px (reads high: picks stop "
              f"short of the edge)")
        print(f"    => CONSISTENT WITH {best:g} A/px  -> export with "
              f"--coords_angpix {best:g} and NO --normalized_coords")
        if xs[1] > nx * apx / best * 1.02:
            print(f"    !! max X EXCEEDS the tomogram at {best:g} A/px — these "
                  f"coordinates are in a finer pixel than that, or not "
                  f"coordinates at all")
        if stated and abs(best - float(stated)) / float(stated) > 0.2:
            print(f"    !! the star's header SAYS {float(stated):g} A/px, which "
                  f"disagrees with its own numbers — trust the numbers")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stars", nargs="+")
    ap.add_argument("--dims", default="",
                    help="reconstruction the picks were made on, "
                         "e.g. 512x512x386@12.56")
    a = ap.parse_args(argv)
    dims = None
    if a.dims:
        try:
            box, apx = a.dims.split("@")
            dims = (float(box.lower().split("x")[0]), float(apx))
        except (ValueError, IndexError):
            sys.exit("--dims must look like 512x512x386@12.56")
    for s in a.stars:
        report(s, dims)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
