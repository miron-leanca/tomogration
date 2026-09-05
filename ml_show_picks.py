#!/usr/bin/env python3
# ml_show_picks.py
#
# Put pick coordinates ON the tomogram they claim to come from, in napari.
#
# WHY: every check in the RELION->Warp loop compares one star to another star.
# ml_verify_reextract proves the numbers scaled correctly; it cannot prove Warp
# CUT where those numbers point, because it never looks at a tomogram. Four
# hypotheses were chased on arithmetic alone (coords_angpix, carried Eulers,
# high-frequency noise) while the one question that settles all of them --
# "do the dots sit on particles?" -- had never been asked.
#
#   python3 ml_show_picks.py <tomogram.mrc> <star> [star2 ...] [--apx 12.56]
#
# Each star becomes its own coloured points layer, so two can be COMPARED --
# picks before and after recentring, say. If one set sits on density and the
# other sits beside it, the shift that separates them is the bug.
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ml_relion4_select_picks import (read_star_blocks, particles_block,  # noqa: E402
                                     star_pixel_size)

COLOURS = ("red", "cyan", "yellow", "magenta", "lime", "orange")


def tomo_angpix(path, override=None):
    """Å/px of the tomogram: the filename tag, else the MRC header."""
    if override:
        return float(override)
    m = re.search(r"_(\d+(?:[.p]\d+)?)Apx", os.path.basename(path))
    if m:
        return float(m.group(1).replace("p", "."))
    import struct                                             # noqa: PLC0415
    with open(path, "rb") as fh:
        h = fh.read(1024)
    mx = struct.unpack_from("<i", h, 28)[0]
    xlen = struct.unpack_from("<f", h, 40)[0]
    return float(xlen) / max(mx, 1)


def stem_of(path):
    base = os.path.basename(path)
    m = re.match(r"^(.+?)_\d+(?:[.p]\d+)?Apx", base)
    return m.group(1) if m else os.path.splitext(base)[0]


def points_for(star, stem, tomo_apx, coords_apx=None):
    """(N,3) array of (z,y,x) in TOMOGRAM voxels, for this tomogram only."""
    import numpy as np                                        # noqa: PLC0415
    blocks = read_star_blocks(star)
    pb = particles_block(blocks, need=("rlnCoordinateX",))
    if pb is None:
        print(f"  {star}: no particle block — skipped")
        return None, None
    c = pb["cols"]
    apx = coords_apx or star_pixel_size(blocks, pb)
    mic_col = next((n for n in ("rlnTomoName", "rlnMicrographName")
                    if n in c), None)
    xs = [float(r[c["rlnCoordinateX"]]) for r in pb["rows"]]
    frac = max(xs or [0]) <= 1.5
    if not frac and not apx:
        print(f"  {star}: pixels, but the star does not state its Å/px "
              f"— pass --coords-angpix")
        return None, None

    pts = []
    for r in pb["rows"]:
        if mic_col and stem not in os.path.basename(r[c[mic_col]]):
            continue
        x, y, z = (float(r[c[f"rlnCoordinate{a}"]]) for a in "XYZ")
        if frac:
            # 0-1 fractions of the volume; scaled to voxels by the caller.
            pts.append((z, y, x))
        else:
            s = float(apx) / tomo_apx
            pts.append((z * s, y * s, x * s))
    return (np.asarray(pts, float) if pts else None), ("fraction" if frac
                                                       else f"{apx:g} Å/px")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tomogram")
    ap.add_argument("stars", nargs="+")
    ap.add_argument("--apx", default="", help="tomogram Å/px (default: from name/header)")
    ap.add_argument("--coords-angpix", default="",
                    help="Å/px the STAR coordinates are in, if they do not say")
    ap.add_argument("--size", type=float, default=12.0, help="point size in voxels")
    a = ap.parse_args(argv)

    import mrcfile                                            # noqa: PLC0415
    import napari                                             # noqa: PLC0415
    import numpy as np                                        # noqa: PLC0415

    t_apx = tomo_angpix(a.tomogram, a.apx)
    stem = stem_of(a.tomogram)
    with mrcfile.open(a.tomogram, permissive=True) as m:
        vol = np.asarray(m.data)
    print(f"tomogram {a.tomogram}\n  {vol.shape}  {t_apx:g} Å/px  stem '{stem}'")

    lo, hi = np.percentile(vol[::4, ::4, ::4], [1, 99])
    v = napari.Viewer(title=f"{stem} — picks on the tomogram")
    v.add_image(vol, name=os.path.basename(a.tomogram), colormap="gray",
                contrast_limits=[float(lo), float(hi)])

    for i, star in enumerate(a.stars):
        pts, units = points_for(star, stem, t_apx,
                                a.coords_angpix or None)
        if pts is None or not len(pts):
            print(f"  {star}: no picks for {stem}")
            continue
        if units == "fraction":
            pts = pts * np.array(vol.shape, float)
        inside = ((pts >= 0) & (pts < np.array(vol.shape))).all(1)
        print(f"  {os.path.basename(star)}: {len(pts)} picks ({units}), "
              f"{int(inside.sum())} inside the volume"
              + ("" if inside.all() else
                 f"  <-- {int((~inside).sum())} OUTSIDE"))
        v.add_points(pts, name=os.path.basename(star), size=a.size,
                     face_color="transparent", border_color=COLOURS[i % len(COLOURS)],
                     border_width=0.15, out_of_slice_display=True)

    print("\nScroll in Z. Dots should sit ON density. If they sit beside it, "
          "the offset you see IS the bug.")
    napari.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
