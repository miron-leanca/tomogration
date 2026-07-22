#!/usr/bin/env python3
# ml_cryolo_to_warp_picks_auto.py
#
# Convert crYOLO 3D picks into per-tomogram Warp pick STARs, so crYOLO can stand
# in for ts_template_match + threshold_picks as the particle source. Its output
# feeds straight into WarpTools ts_export_particles (the tomogration "9. Export"
# stage), which extracts the subtomograms and writes matching.star. From there
# ml_relion4_convert_star_warp_auto.sh produces matching_conv.star as usual:
#
#   crYOLO COORDS/  ->  [THIS SCRIPT]  ->  warp_tiltseries/matching_cryolo/*.star
#                                            |
#                                            v
#                       ts_export_particles  (--input_directory <out> ,
#                                              --input_pattern *_<apx>Apx_<suffix>.star ,
#                                              --normalized_coords ON)  ->  matching.star
#                                            |
#                                            v
#                       ml_relion4_convert_star_warp_auto.sh  ->  matching_conv.star
#
# Two things it gets right that repeatedly bite a hand-rolled conversion:
#
#   1. NORMALISED COORDINATES. WarpTools' own pick STARs store rlnCoordinateX/Y/Z
#      as 0-1 fractions of the tomogram dimensions, NOT absolute voxels. crYOLO
#      COORDS are absolute voxels in the reconstruction grid, so we divide each
#      axis by the reconstruction's nx/ny/nz (read straight from the MRC header,
#      no dependencies). ts_export_particles must then run with --normalized_coords.
#   2. TOMOGRAM IDENTITY FROM THE FILENAME. ts_export_particles maps each pick STAR
#      to a tiltseries by the leading token before the suffix, and also needs
#      _rlnMicrographName = '<stem>.tomostar' inside. We mirror WarpTools' exact
#      8-column layout so the export can't tell the difference from a real match.
#
# The pick coordinates are left at the RECONSTRUCTION pixel scale (normalised) -
# do NOT pre-scale to the particle angpix. ts_export_particles --output_angpix
# does the reconstruction->particle rescale during extraction.
#
# Usage:
#   python3 ml_cryolo_to_warp_picks_auto.py <coords_dir> <recon_dir> [--execute]
#     <coords_dir>  crYOLO tomo-picking COORDS/ dir (PositionNNN_<apx>Apx.coords)
#     <recon_dir>   the reconstructions crYOLO picked on (PositionNNN_<apx>Apx.mrc)
#   Default is a DRY RUN (reports per-tomogram counts, writes nothing).
#   Add --execute to write the pick STARs.
#
# Knobs (flags, or the matching env var the tomogration GUI can set):
#   --out_dir / OUT_DIR    output dir       (default: <recon_dir>/../matching_cryolo)
#   --apx     / APX        angpix filename tag of the reconstructions (default: 12.56)
#   --suffix  / SUFFIX     pick-set tag; file = <stem>_<apx>Apx_<suffix>.star,
#                          export --input_pattern = *_<apx>Apx_<suffix>.star (default: cryolo)
#   --fom     / FOM        constant _rlnAutopickFigureOfMerit (default: 1.0; COORDS
#                          carries no score, and export does not threshold)
#   --flip_y  / FLIP_Y=1   mirror Y (y -> 1 - y). Use ONLY if the one-tomogram
#                          verification shows picks come out Y-flipped vs the tomogram.

import argparse
import glob
import os
import struct
import sys


def mrc_dims(path):
    """nx, ny, nz = the first three int32 of the MRC header (little-endian)."""
    with open(path, "rb") as f:
        return struct.unpack("<iii", f.read(12))


def env_default(name, fallback):
    v = os.environ.get(name)
    return v if v not in (None, "") else fallback


def main():
    ap = argparse.ArgumentParser(
        description="Convert crYOLO COORDS/ 3D picks into normalised per-tomogram "
                    "Warp pick STARs for ts_export_particles.")
    ap.add_argument("coords_dir", help="crYOLO COORDS/ directory")
    ap.add_argument("recon_dir", help="reconstruction dir crYOLO picked on")
    ap.add_argument("--out_dir", default=env_default("OUT_DIR", ""),
                    help="output dir (default: <recon_dir>/../matching_cryolo)")
    ap.add_argument("--apx", default=env_default("APX", "12.56"),
                    help="angpix filename tag of the reconstructions (default: 12.56)")
    ap.add_argument("--suffix", default=env_default("SUFFIX", "cryolo"),
                    help="pick-set tag in the output filename (default: cryolo)")
    ap.add_argument("--fom", type=float, default=float(env_default("FOM", "1.0")),
                    help="constant _rlnAutopickFigureOfMerit (default: 1.0)")
    ap.add_argument("--flip_y", action="store_true",
                    default=env_default("FLIP_Y", "0") == "1",
                    help="mirror Y (y -> 1 - y); use only if picks come out Y-flipped")
    ap.add_argument("--execute", action="store_true",
                    help="write the STARs (default is a dry run that writes nothing)")
    args = ap.parse_args()

    coords_dir = os.path.abspath(args.coords_dir)
    recon_dir = os.path.abspath(args.recon_dir)
    apx = str(args.apx)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir \
        else os.path.join(os.path.dirname(recon_dir), "matching_cryolo")
    file_suffix = f"_{apx}Apx_{args.suffix}"          # -> PositionNNN_12.56Apx_cryolo.star
    input_pattern = f"*{file_suffix}.star"            # ts_export_particles --input_pattern

    if not os.path.isdir(coords_dir):
        sys.exit(f"ERROR: coords_dir not found: {coords_dir}")
    if not os.path.isdir(recon_dir):
        sys.exit(f"ERROR: recon_dir not found: {recon_dir}")

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print("===================================================================")
    print(f"ml_cryolo_to_warp_picks_auto    [{mode}]")
    print(f"crYOLO COORDS: {coords_dir}")
    print(f"Reconstr.:     {recon_dir}")
    print(f"Output dir:    {out_dir}")
    print(f"Pick STARs:    *{file_suffix}.star   (normalised coords, angles = 0)")
    print(f"Flip Y:        {args.flip_y}")
    print("===================================================================")

    coords_files = sorted(glob.glob(os.path.join(coords_dir, f"*_{apx}Apx.coords")))
    if not coords_files:
        sys.exit(f"ERROR: no '*_{apx}Apx.coords' files in {coords_dir} "
                 f"(is --apx right? crYOLO writes COORDS/<stem>_{apx}Apx.coords)")

    hdr = ("\ndata_\n\nloop_\n"
           "_rlnCoordinateX #1\n_rlnCoordinateY #2\n_rlnCoordinateZ #3\n"
           "_rlnAngleRot #4\n_rlnAngleTilt #5\n_rlnAnglePsi #6\n"
           "_rlnMicrographName #7\n_rlnAutopickFigureOfMerit #8\n")

    if args.execute:
        os.makedirs(out_dir, exist_ok=True)

    total = 0
    written = 0
    for cf in coords_files:
        stem = os.path.basename(cf).split(f"_{apx}Apx")[0]        # Position046
        mrc = os.path.join(recon_dir, f"{stem}_{apx}Apx.mrc")
        if not os.path.exists(mrc):
            print(f"!! {stem}: no reconstruction ({os.path.basename(mrc)}), skipping")
            continue
        nx, ny, nz = mrc_dims(mrc)
        tomo = f"{stem}.tomostar"
        rows = []
        for line in open(cf):
            p = line.split()
            if len(p) >= 3:
                x = float(p[0]) / nx
                y = float(p[1]) / ny
                z = float(p[2]) / nz
                if args.flip_y:
                    y = 1.0 - y
                rows.append(f"{x:12.7f} {y:12.7f} {z:12.7f} "
                            f"{0.0:11.5f} {0.0:11.5f} {0.0:11.5f}  "
                            f"{tomo}  {args.fom:11.5f}")
        if not rows:
            print(f".. {stem}: 0 particles, skipping")
            continue
        total += len(rows)
        out = os.path.join(out_dir, f"{stem}{file_suffix}.star")
        if args.execute:
            with open(out, "w") as fh:
                fh.write(hdr + "\n".join(rows) + "\n")
            written += 1
        print(f"{stem}: {len(rows):5d} particles  (nx,ny,nz={nx},{ny},{nz})"
              f"{'  -> ' + out if args.execute else ''}")

    print("-------------------------------------------------------------------")
    print(f"{len(coords_files)} tomograms, {total} particles total.")
    if not args.execute:
        print("DRY-RUN - nothing written. Re-run with --execute to write the STARs.")
        return
    print(f"Wrote {written} pick STARs to {out_dir}")
    print("Next: ts_export_particles with")
    print(f"    --input_directory {out_dir}")
    print(f"    --input_pattern   {input_pattern}")
    print(f"    --normalized_coords            (REQUIRED: coords are 0-1 fractions)")
    print(f"    --output_angpix <particle apx>  --box <box>  --diameter <A>")
    print("Verify ONE tomogram's extracted positions land on particles before all 15;")
    print("if they are Y-mirrored, re-run this with --flip_y.")


if __name__ == "__main__":
    main()
