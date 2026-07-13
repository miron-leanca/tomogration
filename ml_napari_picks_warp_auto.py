#!/usr/bin/env python
"""ml_napari_picks_warp_auto.py

Open a tomogram in napari with a WarpTools template-match / threshold pick STAR
overlaid as a points layer — so you can eyeball picking quality per tomogram.

    python ml_napari_picks_warp_auto.py <tomogram.mrc> <picks.star> [--angpix A]

  <tomogram.mrc>  a full tomogram, e.g.
                  warp_tiltseries/reconstruction/Position042_12.56Apx.mrc
  <picks.star>    the matching STAR for that SAME series + pixel size, e.g.
                  warp_tiltseries/matching/Position042_12.56Apx_emd_70905.star
                  (or your --override_suffix set: ..._<suffix>.star)
  --angpix        tomogram Å/px (== ts_reconstruct/ts_template_match tomo_angpix).
                  Defaults to the value in the MRC header.

Needs napari, mrcfile and starfile in the active env:
    pip install "napari[all]" mrcfile starfile      # (or use an env that has them)

Coordinate convention: WarpTools writes rlnCoordinateX/Y/Z in tomogram PIXELS at
tomo_angpix, which overlay directly on the reconstruction. This script also
handles normalized (0-1) or Angstrom coordinates defensively.
"""
import sys
import argparse

import numpy as np

try:
    import mrcfile
    import starfile
    import napari
except ImportError as e:                       # pragma: no cover - runtime env
    sys.exit(f"ERROR: missing package ({e}). This needs napari, mrcfile and "
             f"starfile in the active environment.")


def _coords_table(star):
    """Return the STAR table (DataFrame) that actually carries coordinates."""
    if isinstance(star, dict):
        for tbl in star.values():
            if hasattr(tbl, "columns") and "rlnCoordinateX" in tbl.columns:
                return tbl
        # fall back to the last table
        return list(star.values())[-1]
    return star


def _to_pixels(col, extent, angpix):
    """Map a coordinate column onto tomogram-pixel space along one axis of size
    `extent`. Handles pixel (default), normalized 0-1, and Angstrom coords."""
    arr = np.asarray(col, dtype=float)
    if arr.size == 0:
        return arr
    hi = np.nanmax(arr)
    if hi <= 1.5:                      # normalized fraction of the box
        return arr * extent
    if hi > extent * 2 and angpix:     # looks like Angstroms
        return arr / angpix
    return arr                         # already pixels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tomogram")
    ap.add_argument("star")
    ap.add_argument("--angpix", type=float, default=None)
    args = ap.parse_args()

    with mrcfile.open(args.tomogram, permissive=True) as m:
        vol = np.asarray(m.data)
        header_apx = float(m.voxel_size.x) if m.voxel_size.x else None
    if vol.ndim != 3:
        sys.exit(f"ERROR: {args.tomogram} is not a 3D volume (shape {vol.shape}).")
    nz, ny, nx = vol.shape
    angpix = args.angpix or header_apx or 1.0

    df = _coords_table(starfile.read(args.star))
    for c in ("rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"):
        if c not in getattr(df, "columns", []):
            sys.exit(f"ERROR: {args.star} has no {c} column.")
    px = _to_pixels(df["rlnCoordinateX"], nx, angpix)
    py = _to_pixels(df["rlnCoordinateY"], ny, angpix)
    pz = _to_pixels(df["rlnCoordinateZ"], nz, angpix)
    pts = np.column_stack([pz, py, px])        # napari axis order = (z, y, x)

    score = None
    for c in ("rlnAutopickFigureOfMerit", "rlnFigureOfMerit", "rlnLCCmax"):
        if c in getattr(df, "columns", []):
            score = np.asarray(df[c], dtype=float)
            break

    lo, hi = (float(np.percentile(vol, 2)), float(np.percentile(vol, 98)))
    viewer = napari.Viewer(title=f"{args.tomogram.split('/')[-1]}  ·  {len(pts)} picks")
    viewer.add_image(vol, name="tomogram", colormap="gray", contrast_limits=[lo, hi])
    size = max(6, int(150 / angpix))
    outline = "white" if score is not None else "red"
    face = score if (score is not None and score.size == len(pts)) else "red"
    fmap = "viridis" if not isinstance(face, str) else None
    # napari renamed edge_color -> border_color in 0.5; support both.
    for edge_kw in ("border_color", "edge_color"):
        try:
            kw = {"name": "picks", "size": size, "symbol": "ring", "opacity": 0.8,
                  "face_color": face, edge_kw: outline}
            if fmap:
                kw["face_colormap"] = fmap
            viewer.add_points(pts, **kw)
            break
        except TypeError:
            continue
    print(f"{len(pts)} picks over {args.tomogram} ({nx}x{ny}x{nz} px @ {angpix} A/px)")
    napari.run()


if __name__ == "__main__":
    main()
