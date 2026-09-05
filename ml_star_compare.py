#!/usr/bin/env python3
# ml_star_compare.py
#
# Match particles BY IMAGE NAME across stars and show what moved.
#
# WHY: the question "did RELION already apply the refined shift to the
# coordinates, so that re-applying it double-counts?" cannot be answered from
# coordinate RANGES -- a subset changes the range for trivial reasons. It needs
# the same particle on both sides. rlnImageName is an exact key: a RELION
# selection still names the subtomogram file Warp wrote, so every particle can
# be paired without arithmetic.
#
#   python3 ml_star_compare.py in.star out.star [more.star ...]
#
# Consecutive pairs are compared. For each pair it reports the coordinate
# difference in ANGSTROMS, and tests it against the refined origins:
#
#   d == 0                  -> coordinates untouched; the shift lives only in
#                              the origin columns, and applying it once is right
#   d == -origin            -> ALREADY RECENTRED; applying it again double-counts
#   d == +origin            -> already recentred with the opposite sign
#
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ml_relion4_select_picks import (read_star_blocks, particles_block,  # noqa: E402
                                     star_pixel_size)

ORI = ("rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst")
ANG = ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")
CLS = "rlnClassNumber"


def load(path):
    blocks = read_star_blocks(path)
    pb = particles_block(blocks, need=("rlnCoordinateX",))
    if pb is None:
        sys.exit(f"ERROR: {path} has no particle block")
    apx = star_pixel_size(blocks, pb)
    if not apx:
        sys.exit(f"ERROR: {path} does not state its pixel size")
    img = pb["cols"].get("rlnImageName")
    if img is None:
        sys.exit(f"ERROR: {path} has no _rlnImageName to match on")
    out = {}
    for r in pb["rows"]:
        key = os.path.basename(r[img])
        xyz = tuple(float(r[pb["cols"][f"rlnCoordinate{a}"]]) * apx
                    for a in "XYZ")            # ANGSTROMS
        ori = None
        if all(c in pb["cols"] for c in ORI):
            ori = tuple(float(r[pb["cols"][c]]) for c in ORI)
        ang = None
        if all(c in pb["cols"] for c in ANG):
            ang = tuple(float(r[pb["cols"][c]]) for c in ANG)
        cls = None
        if CLS in pb["cols"]:
            try:
                cls = int(float(r[pb["cols"][CLS]]))
            except ValueError:
                cls = None
        out[key] = (xyz, ori, ang, cls)
    return out, float(apx), len(pb["rows"])


def compare(pa, pb_, n_show=5):
    a, apx_a, na = load(pa)
    b, apx_b, nb = load(pb_)
    shared = [k for k in b if k in a]
    print(f"\n{os.path.basename(pa)}  ({na} particles, {apx_a:g} A/px)")
    print(f"  vs {os.path.basename(pb_)}  ({nb} particles, {apx_b:g} A/px)")
    if not shared:
        print("  NO PARTICLES MATCHED BY IMAGE NAME -- these stars do not "
              "reference the same subtomograms, so nothing can be concluded.")
        return
    print(f"  matched {len(shared)} particles by image name")

    dmax = 0.0
    n_moved = 0
    agree_minus = agree_plus = 0
    checked = 0
    for k in shared:
        (xa, ya, za), oa, ga, _ca = a[k]
        (xb, yb, zb), ori, gb, _cb = b[k]
        d = (xb - xa, yb - ya, zb - za)
        m = max(abs(v) for v in d)
        dmax = max(dmax, m)
        if m > 0.5:
            n_moved += 1
        if ori is not None:
            checked += 1
            # Does the movement equal the refined origin, in either sense?
            if all(abs(d[i] + ori[i]) < 0.5 for i in range(3)):
                agree_minus += 1
            elif all(abs(d[i] - ori[i]) < 0.5 for i in range(3)):
                agree_plus += 1

    print(f"  coordinates moved in {n_moved}/{len(shared)} particles, "
          f"max |delta| = {dmax:.3f} A")

    # ORIGINS AND ANGLES, which the first version of this tool never compared.
    # RELION's Select has a 'Re-center the class averages' option; if it
    # compensates by adjusting each particle's origin, the origins in a
    # selection are measured against a MOVED reference and are a different
    # quantity from the ones the classification refined.
    def _delta(field):
        n_ch, mx, per_class = 0, 0.0, {}
        for k in shared:
            va = a[k][field]
            vb = b[k][field]
            if va is None or vb is None:
                return None
            dd = [vb[i] - va[i] for i in range(3)]
            mm = max(abs(v) for v in dd)
            if mm > 1e-4:
                n_ch += 1
            mx = max(mx, mm)
            cl = b[k][3]
            if cl is not None:
                per_class.setdefault(cl, []).append(dd)
        return n_ch, mx, per_class

    for field, name, unit in ((1, "origins", "A"), (2, "angles", "deg")):
        res = _delta(field)
        if res is None:
            print(f"  {name}: not present in both stars")
            continue
        n_ch, mx, per_class = res
        print(f"  {name} changed in {n_ch}/{len(shared)} particles, "
              f"max |delta| = {mx:.3f} {unit}")
        if field == 1 and n_ch:
            # Kept on ONE line each so the phrases stay greppable in a log.
            print("     -> the SELECTION rewrote the origins.")
            print("        If 'Re-center the class averages' was on, these are")
            print("        measured against a MOVED reference, not the one the")
            print("        particles were cut on.")
            for cl in sorted(per_class)[:6]:
                v = per_class[cl]
                mean = [sum(d[i] for d in v) / len(v) for i in range(3)]
                spread = max(max(abs(d[i] - mean[i]) for i in range(3)) for d in v)
                print(f"        class {cl:>3}: n={len(v):6d}  mean shift "
                      f"({mean[0]:+7.2f},{mean[1]:+7.2f},{mean[2]:+7.2f}) A"
                      f"   spread {spread:6.2f}")
            print("        A constant mean per class with LARGE spread means the")
            print("        shift was rotated per particle -- applying it along")
            print("        tomogram axes then scatters every particle differently.")
    if checked:
        print(f"  of {checked} with refined origins: "
              f"{agree_minus} match delta == -origin, "
              f"{agree_plus} match delta == +origin")
    print("  first few (Angstroms):")
    for k in shared[:n_show]:
        (xa, ya, za), _oa, _ga, _ca = a[k]
        (xb, yb, zb), ori, _gb, _cb = b[k]
        d = (xb - xa, yb - ya, zb - za)
        o = f"  origin ({ori[0]:+7.2f},{ori[1]:+7.2f},{ori[2]:+7.2f})" if ori else ""
        print(f"    {k[:38]:40s} delta ({d[0]:+7.2f},{d[1]:+7.2f},{d[2]:+7.2f}){o}")

    print("  ->", end=" ")
    if dmax < 0.5:
        print("COORDINATES UNCHANGED. RELION kept the pick position and put the")
        print("     alignment in the origin columns only. Applying the shift once,")
        print("     downstream, is therefore NOT a double-count.")
    elif agree_minus > 0.9 * checked and checked:
        print("ALREADY RECENTRED (delta == -origin). RELION moved the")
        print("     coordinates by the refined shift itself. Subtracting the origin")
        print("     again applies it TWICE -- this is the bug.")
    elif agree_plus > 0.9 * checked and checked:
        print("ALREADY RECENTRED with the opposite sign (delta == +origin).")
    else:
        print("coordinates moved, but NOT by the refined origins. Something")
        print("     else rewrote them -- compare the stars by hand before trusting")
        print("     any downstream arithmetic.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stars", nargs="+")
    ap.add_argument("-n", type=int, default=5, help="example rows to print")
    a = ap.parse_args(argv)
    if len(a.stars) < 2:
        sys.exit("give at least two stars")
    for i in range(len(a.stars) - 1):
        compare(a.stars[i], a.stars[i + 1], a.n)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
