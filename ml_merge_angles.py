#!/usr/bin/env python3
# ml_merge_angles.py
#
# Put the refined Euler angles from a RELION star onto a freshly EXPORTED star.
#
# WHY THIS EXISTS, AND WHY IT IS NOT OPTIONAL:
#
# Warp uses the angles in a PICK star to orient each subtomogram it
# reconstructs. Carry refined Eulers into the picks and the volumes come out
# already rotated into the reference frame; RELION then applies the same angles
# again as priors, and every particle ends up rotated twice by a different
# amount. Nothing errors. Coordinates verify to 0.000 px. The only symptom is
# that relion_reconstruct on a random subset returns featureless noise instead
# of the clear central density an unoriented average gives — which is what
# finally exposed it here, ten iterations into a classification.
#
# So the picks must be UNORIENTED, and the refined orientations belong on the
# star RELION reads, after extraction. That is this tool.
#
#   python3 ml_merge_angles.py <source.star> <exported.star> --out merged.star
#
# Particles are paired on (tomogram, coordinate), the same way
# ml_verify_reextract pairs them — safe because a correct re-extraction
# reproduces the coordinates exactly, up to the pixel-size ratio.
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ml_relion4_select_picks import (read_star_blocks, particles_block,  # noqa: E402
                                     star_pixel_size)

ANGLES = ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")


def key_of(row, cols, scale, mic_col):
    """(tomogram, x, y, z) rounded to a pixel — the pairing key."""
    mic = os.path.basename(row[cols[mic_col]])
    xyz = tuple(round(float(row[cols[f"rlnCoordinate{a}"]]) * scale)
                for a in "XYZ")
    return (mic,) + xyz


def mic_column(cols):
    for n in ("rlnTomoName", "rlnMicrographName"):
        if n in cols:
            return n
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source_star", help="the RELION star holding refined angles")
    ap.add_argument("new_star", help="the matching.star the export wrote")
    ap.add_argument("--out", default="", help="output star (default: <new>_ang.star)")
    ap.add_argument("--zero-origins", action="store_true",
                    help="also write _rlnOriginX/Y/ZAngst = 0. Correct when the "
                         "re-extraction was RECENTRED: the shifts are already "
                         "baked into the new coordinates, and leaving the old "
                         "ones would apply them a second time.")
    a = ap.parse_args(argv)

    sb = read_star_blocks(a.source_star)
    spb = particles_block(sb, need=("rlnCoordinateX",))
    nb = read_star_blocks(a.new_star)
    npb = particles_block(nb, need=("rlnCoordinateX",))
    if spb is None or npb is None:
        sys.exit("ERROR: one of these is not a particle star")

    s_apx = star_pixel_size(sb, spb)
    n_apx = star_pixel_size(nb, npb)
    if not s_apx or not n_apx:
        sys.exit("ERROR: a star does not state its pixel size; cannot pair")
    ratio = float(s_apx) / float(n_apx)

    have = [c for c in ANGLES if c in spb["cols"]]
    if len(have) != 3:
        sys.exit(f"ERROR: {a.source_star} has no refined angles ({have})")

    s_mic, n_mic = mic_column(spb["cols"]), mic_column(npb["cols"])
    if not s_mic or not n_mic:
        sys.exit("ERROR: no tomogram/micrograph column to pair on")

    print("===================================================================")
    print(f"source: {a.source_star}   {len(spb['rows'])} particles, {s_apx:g} A/px")
    print(f"new:    {a.new_star}   {len(npb['rows'])} particles, {n_apx:g} A/px")
    print(f"pairing on (tomogram, coordinate), scaling source by x{ratio:g}")
    print("===================================================================")

    lookup = {}
    for r in spb["rows"]:
        lookup[key_of(r, spb["cols"], ratio, s_mic)] = [
            r[spb["cols"][c]] for c in ANGLES]

    # Angle columns may already exist (Warp writes zeros); reuse or append.
    # read_star_blocks gives {name: index}; the order IS the index order.
    cols = dict(npb["cols"])
    width = max(cols.values()) + 1 if cols else 0
    for c in ANGLES:
        if c not in cols:
            cols[c] = width
            width += 1

    out_rows, hit, miss = [], 0, 0
    for r in npb["rows"]:
        row = list(r) + ["0.000000"] * (width - len(r))
        ang = lookup.get(key_of(r, npb["cols"], 1.0, n_mic))
        if ang is None:
            miss += 1
        else:
            hit += 1
            for c, v in zip(ANGLES, ang):
                row[cols[c]] = v
        if a.zero_origins:
            for c in ("rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"):
                if c in cols:
                    row[cols[c]] = "0.000000"
        out_rows.append(row)

    print(f"matched {hit} / {len(out_rows)} particles"
          + (f"   ({miss} UNMATCHED -- angles left as they were)" if miss else ""))
    if miss and hit == 0:
        sys.exit("ERROR: nothing paired. Are these two stars from the same "
                 "re-extraction? Run ml_verify_reextract.py first.")

    out = a.out or os.path.splitext(a.new_star)[0] + "_ang.star"
    order = sorted(cols, key=lambda c: cols[c])
    with open(out, "w") as fh:
        fh.write("\ndata_particles\n\nloop_\n")
        for i, c in enumerate(order, 1):
            fh.write(f"_{c} #{i}\n")
        for row in out_rows:
            fh.write(" ".join(str(v) for v in row) + "\n")
    print(f"wrote {out}")
    print("Point RELION at THIS star, not the raw export.")
    return 0 if hit else 1


if __name__ == "__main__":
    sys.exit(main())
