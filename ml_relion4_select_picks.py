#!/usr/bin/env python3
# ml_relion4_select_picks.py
#
# Select the "good" particles from a RELION 3D-classification result and turn them
# back into WarpTools pick STARs, so ts_export_particles can RE-EXTRACT just those
# particles at a finer pixel size (e.g. bin4 -> bin2) for a higher-resolution refine.
#
# WHY THIS EXISTS (the trap it avoids):
#   RELION's classification output (its run_itNNN_data.star, "File B") stores particle
#   coordinates in RELION's own convention + refined rlnOriginXYZAngst offsets. Feeding
#   those numbers back into Warp's extractor mangles them -> a garbage volume. The fix is
#   to NOT convert RELION coordinates at all. Warp already knows where every particle is,
#   from the pick STARs that made the subtomograms in the first place. RELION never moves
#   a particle during 3D classification -- it only LABELS it (_rlnClassNumber) -- and it
#   keeps the original subtomogram path in _rlnImageName. That path
#       .../subtomo/Position007/Position007_0000034_6.28A.mrc
#   is an exact, unambiguous key back to the pick that produced it. So we select on the
#   class label, map each kept particle to its pick by that key, and re-extract in Warp
#   using Warp's own untouched coordinates.
#
# TWO WAYS IN (pick whichever inputs you still have on disk):
#
#   MODE A  --picks-dir <matching_cryolo/>   (SIMPLEST, coordinates never touched)
#       The per-tomogram pick STARs you fed to Export still exist. ts_export_particles
#       numbers the subtomograms 0000000,0000001,... in the SAME order as the rows of
#       each pick STAR (no thresholding on the cryolo path), so the 7-digit index in the
#       RELION _rlnImageName is exactly the 0-based row index in that Position's pick STAR.
#       We simply drop the rows that aren't in a good class and rewrite each pick STAR.
#
#   MODE B  --from-matching <matching.star> --recon-dir <reconstructions/>   (only needs
#       files you definitely have). Rebuild the pick STARs directly from matching.star's
#       good rows. matching.star stores rlnCoordinateXYZ in RECONSTRUCTION pixels; we
#       re-normalise them (x/nx, y/ny, z/nz from the MRC header) -- the exact inverse of
#       ml_cryolo_to_warp_picks_auto.py -- so the result is byte-for-byte the same format
#       Warp already extracted from, and you re-run Export with --normalized_coords.
#
# After either mode, re-extract at the finer pixel size, e.g.:
#     WarpTools ts_export_particles \
#         --settings warp_tiltseries.settings \
#         --input_directory <OUT_DIR> \
#         --input_pattern '*_good.star' \
#         --normalized_coords \            # MODE B output is normalised; MODE A: keep the
#                                          #   flag only if your ORIGINAL export used it
#         --output_angpix 3.14 \           # bin2 (was 6.28 = bin4)
#         --box 160 --diameter <A> \       # bin2 box (was 80)
#         --3d --output_processing <relion_project_root>
#
# DRY-RUN by default: reports what it would keep per tomogram and writes nothing.
# Add --execute to write the STARs.
#
# NEVER writes outside the chosen --out-dir. Reads only the inputs you name.

import argparse
import glob
import os
import re
import struct
import sys

IMG_RE = re.compile(r'(?P<stem>.+?)_(?P<idx>\d{6,8})_')   # Position007_0000034_ -> stem, idx


def env_default(name, fallback):
    v = os.environ.get(name)
    return v if v not in (None, "") else fallback


# ---------------------------------------------------------------------------
# Minimal STAR reader. RELION 4 files carry several `data_<name>` blocks (an
# optics block, then the particles block); we return each block with its column
# map and its rows, so the caller can pick the block that actually has the data.
# ---------------------------------------------------------------------------
def read_star_blocks(path):
    blocks = []
    cur = None          # {"name", "cols": {name: idx}, "rows": [ [fields...] ]}
    in_loop = False     # inside a loop_ header (reading _col lines)
    reading_rows = False
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("data_"):
                cur = {"name": line[5:], "cols": {}, "rows": []}
                blocks.append(cur)
                in_loop = False
                reading_rows = False
                continue
            if cur is None:
                continue
            if line == "loop_":
                in_loop = True
                reading_rows = False
                continue
            if in_loop and line.startswith("_"):
                # e.g. "_rlnImageName #11"  or  "_rlnImageName"
                name = line.split()[0][1:]
                cur["cols"][name] = len(cur["cols"])
                continue
            if in_loop and line and not line.startswith("_"):
                in_loop = False
                reading_rows = True
            if reading_rows:
                if not line:
                    reading_rows = False
                    continue
                cur["rows"].append(line.split())
    return blocks


def particles_block(blocks, need=("rlnImageName",)):
    """Return the first block that carries all `need` columns."""
    for b in blocks:
        if all(c in b["cols"] for c in need):
            return b
    return None


def stem_idx(image_name):
    """('Position007', 34) from a full/relative _rlnImageName path, else (None, None)."""
    m = IMG_RE.match(os.path.basename(image_name))
    if not m:
        return None, None
    return m.group("stem"), int(m.group("idx"))


def mrc_dims(path):
    """nx, ny, nz = first three int32 of the MRC header (little-endian)."""
    with open(path, "rb") as f:
        return struct.unpack("<iii", f.read(12))


def star_pixel_size(blocks, pb):
    """The Å/px the star's _rlnCoordinateX/Y/Z are expressed in. RELION 4 keeps it in
    the optics block (_rlnImagePixelSize); Warp-written stars also carry a per-particle
    _rlnPixelSize. Either way this is the number to pass to ts_export_particles as
    --coords_angpix. Returns None if the star doesn't state it."""
    for b in blocks:
        if "rlnImagePixelSize" in b["cols"] and b["rows"]:
            try:
                return float(b["rows"][0][b["cols"]["rlnImagePixelSize"]])
            except (ValueError, IndexError):
                pass
    if pb is not None and "rlnPixelSize" in pb["cols"] and pb["rows"]:
        try:
            return float(pb["rows"][0][pb["cols"]["rlnPixelSize"]])
        except (ValueError, IndexError):
            pass
    return None


# ---------------------------------------------------------------------------
# Read the classification STAR, return {stem: set(good indices)} and a count map.
# ---------------------------------------------------------------------------
def good_indices_by_stem(class_star, keep_classes):
    blocks = read_star_blocks(class_star)
    pb = particles_block(blocks, need=("rlnImageName",))
    if pb is None:
        sys.exit(f"ERROR: no block with _rlnImageName in {class_star}")
    cols = pb["cols"]
    img_i = cols["rlnImageName"]
    cls_i = cols.get("rlnClassNumber")
    if cls_i is None and keep_classes is not None:
        sys.exit(f"ERROR: --classes given but {class_star} has no _rlnClassNumber column "
                 f"(is this a classification result? columns: {sorted(cols)})")

    good = {}          # stem -> set(idx)
    per_class = {}     # class -> count (for the report)
    n_total = 0
    n_kept = 0
    for r in pb["rows"]:
        n_total += 1
        cls = int(float(r[cls_i])) if cls_i is not None else 0
        per_class[cls] = per_class.get(cls, 0) + 1
        if keep_classes is not None and cls not in keep_classes:
            continue
        stem, idx = stem_idx(r[img_i])
        if stem is None:
            print(f"!! could not parse stem/index from _rlnImageName: {r[img_i]}")
            continue
        good.setdefault(stem, set()).add(idx)
        n_kept += 1
    return good, per_class, n_total, n_kept


# ---------------------------------------------------------------------------
# Read a single-loop pick STAR verbatim: preamble (everything up to the last
# column header) + the data rows as raw strings, so we can rewrite it unchanged
# apart from dropped rows.
# ---------------------------------------------------------------------------
def read_pick_star(path):
    preamble = []
    rows = []
    ncols = 0
    seen_col = False
    reading = False
    with open(path) as fh:
        for raw in fh:
            s = raw.rstrip("\n")
            t = s.strip()
            if reading:
                if not t:
                    continue
                rows.append(s)
                continue
            preamble.append(s)
            if t.startswith("_"):
                seen_col = True
                ncols += 1
            elif seen_col and t and not t.startswith("_"):
                # first data row landed in preamble; move it over
                preamble.pop()
                rows.append(s)
                reading = True
    return "\n".join(preamble), rows, ncols


def resolve_stem_for_file(base, known_stems):
    for s in known_stems:
        if base.startswith(s):
            return s
    return base.split("_")[0]


# ---------------------------------------------------------------------------
def mode_a(good, args, known_stems):
    """Filter existing per-tomogram pick STARs by row index."""
    picks = sorted(glob.glob(os.path.join(args.picks_dir, args.pattern)))
    if not picks:
        sys.exit(f"ERROR: no '{args.pattern}' files in {args.picks_dir}")
    if args.execute:
        os.makedirs(args.out_dir, exist_ok=True)

    tot_in = tot_out = files_written = 0
    for pf in picks:
        base = os.path.basename(pf)
        stem = resolve_stem_for_file(base, known_stems)
        preamble, rows, _ = read_pick_star(pf)
        keep_idx = good.get(stem, set())
        max_good = max(keep_idx) if keep_idx else -1
        if max_good >= len(rows):
            print(f"!! {stem}: good index {max_good} >= {len(rows)} pick rows -- ORDER "
                  f"MISMATCH, skipping (use MODE B for this one)")
            continue
        kept = [rows[i] for i in range(len(rows)) if i in keep_idx]
        tot_in += len(rows)
        tot_out += len(kept)
        out = os.path.join(args.out_dir, base.replace(".star", f"_{args.suffix}.star"))
        flag = ""
        if args.execute and kept:
            with open(out, "w") as fh:
                fh.write(preamble + "\n" + "\n".join(kept) + "\n")
            files_written += 1
            flag = f"  -> {os.path.basename(out)}"
        print(f"{stem}: {len(kept):5d} / {len(rows):5d} kept{flag}")
    return tot_in, tot_out, files_written


def mode_b(good, args, known_stems):
    """Rebuild normalised pick STARs from matching.star's good rows."""
    blocks = read_star_blocks(args.from_matching)
    mb = particles_block(blocks, need=("rlnImageName", "rlnCoordinateX"))
    if mb is None:
        sys.exit(f"ERROR: {args.from_matching} has no _rlnImageName/_rlnCoordinateX block")
    c = mb["cols"]
    xi, yi, zi = c["rlnCoordinateX"], c["rlnCoordinateY"], c["rlnCoordinateZ"]
    ii = c["rlnImageName"]
    mic_i = c.get("rlnMicrographName")

    # gather good rows per stem
    by_stem = {}       # stem -> list of (x_px, y_px, z_px, micrograph)
    for r in mb["rows"]:
        stem, idx = stem_idx(r[ii])
        if stem is None or idx not in good.get(stem, set()):
            continue
        mic = r[mic_i] if mic_i is not None else f"{stem}.tomostar"
        by_stem.setdefault(stem, []).append(
            (float(r[xi]), float(r[yi]), float(r[zi]), mic))

    if args.execute:
        os.makedirs(args.out_dir, exist_ok=True)

    hdr = ("\ndata_\n\nloop_\n"
           "_rlnCoordinateX #1\n_rlnCoordinateY #2\n_rlnCoordinateZ #3\n"
           "_rlnAngleRot #4\n_rlnAngleTilt #5\n_rlnAnglePsi #6\n"
           "_rlnMicrographName #7\n_rlnAutopickFigureOfMerit #8\n")

    tot_out = files_written = 0
    for stem in sorted(by_stem):
        mrc = os.path.join(args.recon_dir, f"{stem}_{args.apx}Apx.mrc")
        if not os.path.exists(mrc):
            print(f"!! {stem}: reconstruction not found ({os.path.basename(mrc)}); "
                  f"cannot normalise, skipping")
            continue
        nx, ny, nz = mrc_dims(mrc)
        out_rows = []
        for x, y, z, mic in by_stem[stem]:
            out_rows.append(f"{x/nx:12.7f} {y/ny:12.7f} {z/nz:12.7f} "
                            f"{0.0:11.5f} {0.0:11.5f} {0.0:11.5f}  "
                            f"{mic}  {args.fom:11.5f}")
        tot_out += len(out_rows)
        out = os.path.join(args.out_dir, f"{stem}_{args.apx}Apx_{args.suffix}.star")
        flag = ""
        if args.execute:
            with open(out, "w") as fh:
                fh.write(hdr + "\n".join(out_rows) + "\n")
            files_written += 1
            flag = f"  -> {os.path.basename(out)}"
        print(f"{stem}: {len(out_rows):5d} kept  (nx,ny,nz={nx},{ny},{nz}){flag}")
    return None, tot_out, files_written


def mode_c(args):
    """MODE C — coordinates straight from the RELION star (the robust path).

    The RELION particles.star already holds _rlnCoordinateX/Y/Z in a known pixel size
    (_rlnImagePixelSize in the optics block). So we don't need the Warp pick stars, the
    reconstruction dimensions, or any normalisation arithmetic: copy those coordinates
    out, optionally RECENTRE them on the refined particle centre using the refined
    _rlnOriginX/Y/ZAngst offsets, and optionally carry the refined Euler angles through
    as priors. Export then reads them with --coords_angpix <that pixel size> and NO
    --normalized_coords.

    Recentring is RELION's own convention, the same thing 'Re-extract refined particles'
    does:   centred_coord_px = coord_px - origin_angst / pixel_size
    Skipping it re-extracts on the ORIGINAL pick centre and throws away the shifts the
    refinement worked out (particles come out mis-centred by tens of Å)."""
    blocks = read_star_blocks(args.class_star)
    pb = particles_block(blocks, need=("rlnCoordinateX", "rlnMicrographName"))
    if pb is None:
        sys.exit(f"ERROR: {args.class_star} has no block with _rlnCoordinateX + "
                 f"_rlnMicrographName (is this a RELION particles star?)")
    c = pb["cols"]
    xi, yi, zi = c["rlnCoordinateX"], c["rlnCoordinateY"], c["rlnCoordinateZ"]
    mic_i = c["rlnMicrographName"]
    cls_i = c.get("rlnClassNumber")
    ox, oy, oz = (c.get("rlnOriginXAngst"), c.get("rlnOriginYAngst"),
                  c.get("rlnOriginZAngst"))
    rot, tilt, psi = (c.get("rlnAngleRot"), c.get("rlnAngleTilt"), c.get("rlnAnglePsi"))

    apx = args.coords_angpix or star_pixel_size(blocks, pb)
    if not apx:
        sys.exit("ERROR: the star does not state its pixel size (_rlnImagePixelSize / "
                 "_rlnPixelSize) — pass --coords-angpix <Å/px> explicitly.")
    apx = float(apx)

    have_origins = None not in (ox, oy, oz)
    have_angles = None not in (rot, tilt, psi)
    recenter = args.recenter and have_origins
    keep_ang = args.keep_angles and have_angles

    keep_classes = None
    if not args.keep_all:
        keep_classes = {int(x) for x in re.split(r"[,\s]+", args.classes.strip()) if x}

    by_stem, per_class, n_total, max_shift = {}, {}, 0, 0.0
    for r in pb["rows"]:
        n_total += 1
        if cls_i is not None:
            cl = int(float(r[cls_i]))
            per_class[cl] = per_class.get(cl, 0) + 1
            if keep_classes is not None and cl not in keep_classes:
                continue
        elif keep_classes is not None:
            sys.exit(f"ERROR: --classes given but {args.class_star} has no "
                     f"_rlnClassNumber column.")
        x, y, z = float(r[xi]), float(r[yi]), float(r[zi])
        if recenter:
            dx, dy, dz = float(r[ox]), float(r[oy]), float(r[oz])
            max_shift = max(max_shift, abs(dx), abs(dy), abs(dz))
            x -= dx / apx
            y -= dy / apx
            z -= dz / apx
        ang = ((float(r[rot]), float(r[tilt]), float(r[psi])) if keep_ang
               else (0.0, 0.0, 0.0))
        mic = os.path.basename(r[mic_i])
        stem = mic[:-9] if mic.endswith(".tomostar") else os.path.splitext(mic)[0]
        by_stem.setdefault(stem, []).append((x, y, z, ang, mic))

    n_kept = sum(len(v) for v in by_stem.values())
    apx_tag = f"{apx:g}"
    print("===================================================================")
    print(f"ml_relion4_select_picks    [{'EXECUTE' if args.execute else 'DRY-RUN'}]   MODE C")
    print(f"RELION star:    {args.class_star}")
    print(f"Coords pixel:   {apx_tag} Å/px   -> export with --coords_angpix {apx_tag} "
          f"(and NO --normalized_coords)")
    print(f"Recentre:       {'YES (refined origins applied)' if recenter else 'no'}"
          f"{'  [no origin columns in star]' if args.recenter and not have_origins else ''}")
    print(f"Carry angles:   {'YES (refined Eulers as priors)' if keep_ang else 'no'}"
          f"{'  [no angle columns in star]' if args.keep_angles and not have_angles else ''}")
    if recenter:
        print(f"Largest shift:  {max_shift:.1f} Å")
    print(f"Keep classes:   {'ALL' if keep_classes is None else sorted(keep_classes)}")
    if per_class:
        print("Class populations (class: count):")
        for cl in sorted(per_class):
            mark = "  <-- keep" if (keep_classes is None or cl in keep_classes) else ""
            print(f"    class {cl:>3}: {per_class[cl]:6d}{mark}")
    print(f"Selected {n_kept} / {n_total} particles across {len(by_stem)} tomograms.")
    print(f"Output dir:     {args.out_dir}")
    print("===================================================================")

    if args.execute:
        os.makedirs(args.out_dir, exist_ok=True)
    hdr = ("\ndata_\n\nloop_\n"
           "_rlnCoordinateX #1\n_rlnCoordinateY #2\n_rlnCoordinateZ #3\n"
           "_rlnAngleRot #4\n_rlnAngleTilt #5\n_rlnAnglePsi #6\n"
           "_rlnMicrographName #7\n_rlnAutopickFigureOfMerit #8\n")
    written = 0
    for stem in sorted(by_stem):
        rows = [f"{x:12.4f} {y:12.4f} {z:12.4f} "
                f"{a[0]:11.5f} {a[1]:11.5f} {a[2]:11.5f}  {mic}  {args.fom:11.5f}"
                for x, y, z, a, mic in by_stem[stem]]
        out = os.path.join(args.out_dir, f"{stem}_{apx_tag}Apx_{args.suffix}.star")
        if args.execute:
            with open(out, "w") as fh:
                fh.write(hdr + "\n".join(rows) + "\n")
            written += 1
        print(f"{stem}: {len(rows):5d} particles"
              f"{'  -> ' + os.path.basename(out) if args.execute else ''}")

    print("-------------------------------------------------------------------")
    print(f"Wrote {n_kept} picks.")
    if not args.execute:
        print("DRY-RUN - nothing written. Re-run with --execute to write the STARs.")
        return
    print(f"Wrote {written} pick STARs to {args.out_dir}")
    print("Next: re-extract with ts_export_particles:")
    print(f"    --input_directory {args.out_dir}")
    print(f"    --input_pattern   '*_{args.suffix}.star'")
    print(f"    --coords_angpix   {apx_tag}        <-- REQUIRED (coords are in {apx_tag} Å/px)")
    print(f"    NO --normalized_coords             <-- these coords are NOT 0-1 fractions")
    print(f"    --output_angpix <finer>  --box <box>  --diameter <A>  --3d")
    print("    Keep the box's PHYSICAL size >= the box that worked at the coarser bin")
    print("    (halving output_angpix means DOUBLING box to cover the same field).")


def main():
    ap = argparse.ArgumentParser(
        description="Select good-class particles from a RELION classification result "
                    "and write Warp pick STARs to re-extract them.")
    ap.add_argument("class_star", help="RELION classification result (run_itNNN_data.star)")
    ap.add_argument("--classes", default=env_default("KEEP_CLASSES", ""),
                    help="comma-separated good class numbers, e.g. '1,3' (required unless "
                         "--keep-all to just split every classified particle back to picks)")
    ap.add_argument("--keep-all", action="store_true",
                    help="ignore class label; keep every particle present in class_star")
    # Mode A
    ap.add_argument("--picks-dir", default=env_default("PICKS_DIR", ""),
                    help="MODE A: dir of the per-tomogram pick STARs fed to Export")
    ap.add_argument("--pattern", default=env_default("PICK_PATTERN", "*_cryolo.star"),
                    help="MODE A: pick-STAR glob within --picks-dir (default *_cryolo.star)")
    # Mode C (recommended): coordinates straight from the RELION star
    ap.add_argument("--relion-coords", action="store_true",
                    default=env_default("RELION_COORDS", "0") == "1",
                    help="MODE C: take coordinates from the RELION star itself (already "
                         "in a known Å/px), recentred on the refined origins and carrying "
                         "the refined angles. Needs no pick stars, no matching.star and no "
                         "reconstruction dims. Export with --coords_angpix, NOT normalised.")
    ap.add_argument("--coords-angpix", default=env_default("COORDS_ANGPIX", ""),
                    help="MODE C: override the coords' Å/px (default: read from the star's "
                         "_rlnImagePixelSize / _rlnPixelSize).")
    ap.add_argument("--no-recenter", dest="recenter", action="store_false",
                    default=env_default("RECENTER", "1") == "1",
                    help="MODE C: do NOT apply the refined _rlnOriginXYZAngst offsets. "
                         "Default is to apply them (re-extract on the refined centre).")
    ap.add_argument("--no-keep-angles", dest="keep_angles", action="store_false",
                    default=env_default("KEEP_ANGLES", "1") == "1",
                    help="MODE C: do NOT carry the refined Euler angles into the pick "
                         "stars. Default is to carry them as priors.")
    # Mode B
    ap.add_argument("--from-matching", default=env_default("FROM_MATCHING", ""),
                    help="MODE B: matching.star to rebuild pick STARs from")
    ap.add_argument("--recon-dir", default=env_default("RECON_DIR", ""),
                    help="MODE B: reconstruction dir (PositionNNN_<apx>Apx.mrc) to normalise by")
    ap.add_argument("--apx", default=env_default("APX", "12.56"),
                    help="MODE B: angpix filename tag of the reconstructions (default 12.56)")
    ap.add_argument("--fom", type=float, default=float(env_default("FOM", "1.0")),
                    help="MODE B: constant _rlnAutopickFigureOfMerit (default 1.0)")
    # common
    ap.add_argument("--out-dir", default=env_default("OUT_DIR", ""),
                    help="output dir for the filtered/rebuilt pick STARs (required)")
    ap.add_argument("--suffix", default=env_default("SUFFIX", "good"),
                    help="tag added to output filenames (default 'good')")
    ap.add_argument("--execute", action="store_true",
                    help="write the STARs (default is a dry run that writes nothing)")
    args = ap.parse_args()

    if not os.path.isfile(args.class_star):
        sys.exit(f"ERROR: class_star not found: {args.class_star}")
    keep_classes = None
    if not args.keep_all:
        if not args.classes.strip():
            sys.exit("ERROR: give --classes '1,3' (the good class numbers) or --keep-all")
        keep_classes = {int(x) for x in re.split(r"[,\s]+", args.classes.strip()) if x}

    use_c = bool(args.relion_coords)
    use_a = bool(args.picks_dir) and not use_c
    use_b = bool(args.from_matching) and not use_c
    if not use_c and use_a == use_b:
        sys.exit("ERROR: choose exactly one of MODE C (--relion-coords, recommended), "
                 "MODE A (--picks-dir) or MODE B (--from-matching + --recon-dir)")
    if use_b and not args.recon_dir:
        sys.exit("ERROR: MODE B needs --recon-dir (to normalise coordinates)")
    if not args.out_dir:
        sys.exit("ERROR: --out-dir is required")
    args.out_dir = os.path.abspath(args.out_dir)

    if use_c:                       # MODE C reads everything it needs from the star
        mode_c(args)
        return

    good, per_class, n_total, n_kept = good_indices_by_stem(args.class_star, keep_classes)
    known_stems = set(good.keys())

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print("===================================================================")
    print(f"ml_relion4_select_picks    [{mode}]   MODE {'A' if use_a else 'B'}")
    print(f"Classification: {args.class_star}")
    print(f"Keep classes:   {'ALL' if keep_classes is None else sorted(keep_classes)}")
    print(f"Output dir:     {args.out_dir}")
    print("Class populations (class: count):")
    for cl in sorted(per_class):
        mark = "  <-- keep" if (keep_classes is None or cl in keep_classes) else ""
        print(f"    class {cl:>3}: {per_class[cl]:6d}{mark}")
    print(f"Selected {n_kept} / {n_total} particles across {len(good)} tomograms.")
    print("===================================================================")

    if use_a:
        tot_in, tot_out, nfiles = mode_a(good, args, known_stems)
    else:
        tot_in, tot_out, nfiles = mode_b(good, args, known_stems)

    print("-------------------------------------------------------------------")
    if tot_in is not None:
        print(f"Kept {tot_out} / {tot_in} pick rows.")
    else:
        print(f"Wrote {tot_out} picks.")
    if not args.execute:
        print("DRY-RUN - nothing written. Re-run with --execute to write the STARs.")
        return
    print(f"Wrote {nfiles} pick STARs to {args.out_dir}")
    print("Next: re-extract at the finer pixel size with ts_export_particles:")
    print(f"    --input_directory {args.out_dir}")
    print(f"    --input_pattern   '*_{args.suffix}.star'")
    if use_b:
        print(f"    --normalized_coords        (MODE B coords are 0-1 fractions)")
    else:
        print(f"    --normalized_coords        (KEEP only if your ORIGINAL export used it)")
    print(f"    --output_angpix <finer>  --box <finer box>  --diameter <A>  --3d")
    print("Verify ONE tomogram's re-extracted positions land on particles before trusting all.")


if __name__ == "__main__":
    main()
