#!/usr/bin/env python3
# ============================================================================
# RETIRED 2026-09-02 — tomogration no longer uses this converter.
#
# Re-extraction is done by ts_export_particles reading the RELION star DIRECTLY:
#     WarpTools ts_export_particles --settings warp_tiltseries.settings \
#         --input_star Select/jobNNN/particles.star --coords_angpix <rlnImagePixelSize> \
#         --output_angpix <finer> --box <N> --diameter <A> --3d --relative_output_paths \
#         --output_star relion4/<dir>/matching.star --output_processing relion4/<dir>
# Warp subtracts the refined rlnOriginX/Y/ZAngst itself (÷ the star's pixel size)
# and copies the refined angles into the output star; no pick stars are needed.
# The three modes below produced three coordinate conventions and a silent
# failure for each. The star-reading helpers (read_star_blocks, particles_block,
# star_pixel_size) are still imported by the audit tools, which is why the file
# stays.
# ============================================================================
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
# COORDINATES COME FROM THE RELION STAR, and only from there. The star holds
# _rlnCoordinateX/Y/Z in a pixel size it states (_rlnImagePixelSize), so the
# coordinates are copied out, RECENTRED on the refined _rlnOriginXYZAngst
# offsets, and the refined Eulers carried through as priors. Nothing needs the
# Warp pick STARs, a matching.star, or the reconstruction dimensions.
#
#     centred_coord_px = coord_px - origin_angst / pixel_size
#
# Two older routes were removed on 2026-08-31: one rebuilt the pick STARs from
# the Warp pick files by row index, the other renormalised a matching.star
# against the reconstruction dimensions. Each produced a DIFFERENT coordinate
# convention (pixels vs 0-1 fractions), so the export flag that was right for
# one silently extracted noise under the other — a well-formed star full of
# subtomograms cut from the wrong places, refining happily for hours. One path
# with a stated pixel size removes the choice, and the failure with it.
#
# Re-extract at the finer pixel size:
#     WarpTools ts_export_particles \
#         --settings warp_tiltseries.settings \
#         --input_directory <OUT_DIR> \
#         --input_pattern '*_<suffix>.star' \
#         --coords_angpix <the A/px this tool reports>   # NOT --normalized_coords
#         --output_angpix 1.57 \           # bin1 (was 6.28 = bin4)
#         --box 200 --diameter <A> \       # halving output_angpix DOUBLES the box
#         --3d --output_processing <relion_project_root>
#
# DRY-RUN by default: reports what it would keep per tomogram and writes nothing.
# Add --execute to write the STARs.
#
# NEVER writes outside the chosen --out-dir. Reads only the inputs you name.

import argparse
import glob
import math
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
    """Longest stem whose match ends at a real boundary ('_' or '.').

    A bare prefix test over an unordered set is wrong twice with non-zero-padded
    names: 'Position1' prefix-matches 'Position10_...star', and set iteration
    order decides which stem wins — filtering one tomogram's picks with another
    tomogram's row indices."""
    best = ""
    for s in known_stems:
        if len(s) > len(best) and base.startswith(s) \
                and (len(base) == len(s) or base[len(s)] in "_."):
            best = s
    return best or base.split("_")[0]


# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# MODE A — RE-NORMALISE.  RELION coordinates -> 0-1 fractions.
# ---------------------------------------------------------------------------
def mode_a(args):
    """Write picks as 0-1 FRACTIONS of each tomogram, from the RELION star.

    WHY: the FIRST export of a crYOLO pick set runs with --normalized_coords,
    because fractions are what the crYOLO converter writes. If the re-extraction
    also writes fractions, the second export uses exactly the same coordinate
    flags as the first, and nothing has to agree about a pixel size — a fraction
    carries none.

    The reconstructions are read only for their DIMENSIONS: a fraction is the
    coordinate divided by the tomogram's width, height and depth.
    """
    blocks = read_star_blocks(args.class_star)
    pb = particles_block(blocks, need=("rlnCoordinateX", "rlnMicrographName"))
    if pb is None:
        sys.exit(f"ERROR: {args.class_star} has no block with _rlnCoordinateX + "
                 f"_rlnMicrographName")
    c = pb["cols"]
    apx = float(args.coords_angpix or star_pixel_size(blocks, pb) or 0)
    if not apx:
        sys.exit("ERROR: the star does not state its pixel size — pass "
                 "--coords-angpix")
    if not args.recon_dir or not os.path.isdir(args.recon_dir):
        sys.exit("ERROR: MODE A needs --recon-dir (the reconstructions, read "
                 "for their dimensions)")

    keep_classes = None
    if not args.keep_all:
        keep_classes = {int(x) for x in re.split(r"[,\s]+", args.classes.strip()) if x}
    cls_i = c.get("rlnClassNumber")

    # One header read per tomogram, cached: 28,850 particles across 71 series
    # would otherwise reopen the same MRC hundreds of times over a network
    # filesystem.
    cache = {}

    def recon_for(stem):
        if stem in cache:
            return cache[stem]
        hits = [h for h in sorted(glob.glob(os.path.join(args.recon_dir,
                                                         f"{stem}*.mrc")))
                if os.path.basename(h) == stem + ".mrc"
                or os.path.basename(h)[len(stem):len(stem) + 1] in "_."]
        if not hits:
            cache[stem] = None
            return None
        m = re.search(r"_(\d+(?:[.p]\d+)?)Apx", os.path.basename(hits[0]))
        rapx = float(m.group(1).replace("p", ".")) if m else apx
        cache[stem] = (mrc_dims(hits[0]), rapx)
        return cache[stem]

    by_stem, per_class, n_total, missing = {}, {}, 0, set()
    for r in pb["rows"]:
        n_total += 1
        if cls_i is not None:
            cl = int(float(r[cls_i]))
            per_class[cl] = per_class.get(cl, 0) + 1
            if keep_classes is not None and cl not in keep_classes:
                continue
        mic = os.path.basename(r[c["rlnMicrographName"]])
        stem = mic[:-9] if mic.endswith(".tomostar") else os.path.splitext(mic)[0]
        got = recon_for(stem)
        if got is None:
            missing.add(stem)
            continue
        (nx, ny, nz), rapx = got
        # Through ANGSTROMS, never by assuming the star's grid and the
        # reconstruction's grid are the same one.
        x, y, z = (float(r[c[f"rlnCoordinate{a}"]]) * apx for a in "XYZ")
        by_stem.setdefault(stem, []).append(
            (x / (nx * rapx), y / (ny * rapx), z / (nz * rapx), mic))

    _report(args, "A", "0-1 FRACTIONS of each tomogram", per_class,
            keep_classes, n_total, sum(len(v) for v in by_stem.values()))
    if missing:
        print(f"!! no reconstruction found for {len(missing)} tomogram(s): "
              f"{', '.join(sorted(missing)[:5])}"
              + (" ..." if len(missing) > 5 else ""))
    print("Export with:    --normalized_coords   and NO --coords_angpix")
    print("=" * 67)

    hdr = ("\ndata_\n\nloop_\n"
           "_rlnCoordinateX #1\n_rlnCoordinateY #2\n_rlnCoordinateZ #3\n"
           "_rlnAngleRot #4\n_rlnAngleTilt #5\n_rlnAnglePsi #6\n"
           "_rlnMicrographName #7\n_rlnAutopickFigureOfMerit #8\n")
    _write(args, by_stem, hdr, lambda v:
           f"{v[0]:12.6f} {v[1]:12.6f} {v[2]:12.6f} "
           f"{0.0:11.5f} {0.0:11.5f} {0.0:11.5f}  {v[3]}  {args.fom:11.5f}")
    return 0


# ---------------------------------------------------------------------------
# MODE B — FILTER THE ORIGINALS.  Drop rows, change nothing else.
# ---------------------------------------------------------------------------
def mode_b(args):
    """Filter the ORIGINAL Warp pick STARs down to the particles RELION kept.

    Nothing is recomputed. Each pick star is rewritten verbatim minus the rows
    that were not selected, so whatever convention it was in — crYOLO writes
    0-1 fractions — survives untouched, and the re-export uses the same flags
    as the export that produced it.

    This is the route with a working precedent: EML45, 2026-07-25, filtered
    picks exported with --normalized_coords.
    """
    if not args.picks_dir or not os.path.isdir(args.picks_dir):
        sys.exit("ERROR: MODE B needs --picks-dir (the ORIGINAL pick stars the "
                 "first export read)")
    keep_classes = None
    if not args.keep_all:
        keep_classes = {int(x) for x in re.split(r"[,\s]+", args.classes.strip()) if x}
    good, per_class, n_total, n_kept = good_indices_by_stem(args.class_star,
                                                            keep_classes)
    files = sorted(glob.glob(os.path.join(args.picks_dir, args.pattern or "*.star")))
    if not files:
        sys.exit(f"ERROR: no pick stars matching '{args.pattern or '*.star'}' in "
                 f"{args.picks_dir}")

    _report(args, "B", "UNCHANGED from the original pick stars", per_class,
            keep_classes, n_total, n_kept)
    print(f"Pick stars:     {len(files)} matching '{args.pattern or '*.star'}'")
    print("Export with:    the SAME coordinate flags the first export used")
    print("=" * 67)

    if args.execute:
        os.makedirs(args.out_dir, exist_ok=True)
    written = kept_total = 0
    for f in files:
        base = os.path.basename(f)
        stem = resolve_stem_for_file(base, good.keys())
        idxs = good.get(stem)
        if not idxs:
            continue
        preamble, rows, _n = read_pick_star(f)
        kept = [r for i, r in enumerate(rows) if i in idxs]
        if not kept:
            continue
        # Keep the ORIGINAL name and only add the suffix, so the Apx tag that
        # states the coordinate convention travels with the file.
        out = os.path.join(args.out_dir,
                           re.sub(r"\.star$", f"_{args.suffix}.star", base))
        print(f"{stem:24s} {len(kept):6d} / {len(rows):6d} picks  -> "
              f"{os.path.basename(out)}")
        if args.execute:
            with open(out, "w") as fh:
                fh.write(preamble + "\n" + "\n".join(kept) + "\n")
        written += 1
        kept_total += len(kept)
    print("-" * 67)
    print(f"{'Wrote' if args.execute else 'Would write'} {kept_total} picks in "
          f"{written} star(s) to {args.out_dir}")
    if written == 0:
        sys.exit("ERROR: nothing matched. Do --picks-dir and --pattern point at "
                 "the pick stars the FIRST export read?")
    return 0


def _report(args, mode, convention, per_class, keep_classes, n_total, n_kept):
    print("=" * 67)
    print(f"ml_relion4_select_picks    "
          f"[{'EXECUTE' if args.execute else 'DRY-RUN'}]   MODE {mode}")
    print(f"RELION star:    {args.class_star}")
    print(f"Coordinates:    {convention}")
    print(f"Keep classes:   {'ALL' if keep_classes is None else sorted(keep_classes)}")
    if per_class:
        print("Class populations (class: count):")
        for cl in sorted(per_class):
            mark = "  <-- keep" if (keep_classes is None or cl in keep_classes) else ""
            print(f"    class {cl:>3}: {per_class[cl]:6d}{mark}")
    print(f"Selected {n_kept} / {n_total} particles.")
    print(f"Output dir:     {args.out_dir}")


def _write(args, by_stem, hdr, fmt):
    if args.execute:
        os.makedirs(args.out_dir, exist_ok=True)
    written = 0
    for stem in sorted(by_stem):
        rows = [fmt(v) for v in by_stem[stem]]
        out = os.path.join(args.out_dir, f"{stem}_{args.suffix}.star")
        print(f"{stem:24s} {len(rows):6d} picks  -> {os.path.basename(out)}")
        if args.execute:
            with open(out, "w") as fh:
                fh.write(hdr + "\n".join(rows) + "\n")
        written += 1
    print("-" * 67)
    print(f"{'Wrote' if args.execute else 'Would write'} "
          f"{sum(len(v) for v in by_stem.values())} picks in {written} star(s) "
          f"to {args.out_dir}")


def rotate_zyz(x, y, z, rot, tilt, psi):
    """Rotate a vector by RELION's ZYZ Euler convention (rot, tilt, psi).

    Only used to TEST whether the refined origin lives in the particle frame
    rather than in tomogram axes. RELION composes R = Rz(rot) Ry(tilt) Rz(psi);
    this applies that matrix to the offset vector."""
    a, b, c = (math.radians(v) for v in (rot, tilt, psi))
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    cc, sc = math.cos(c), math.sin(c)
    m = (
        (ca * cb * cc - sa * sc, -ca * cb * sc - sa * cc, ca * sb),
        (sa * cb * cc + ca * sc, -sa * cb * sc + ca * cc, sa * sb),
        (-sb * cc,                sb * sc,                cb),
    )
    return (m[0][0] * x + m[0][1] * y + m[0][2] * z,
            m[1][0] * x + m[1][1] * y + m[1][2] * z,
            m[2][0] * x + m[2][1] * y + m[2][2] * z)


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
            if args.recenter_frame == "rotated" and have_angles:
                # HYPOTHESIS UNDER TEST: the refined origin is expressed in the
                # particle's own rotated frame, not in tomogram axes. If so it
                # must be rotated by that particle's Eulers before it can be
                # subtracted from a tomogram coordinate. Applying it unrotated
                # displaces every particle in a different wrong direction --
                # which is what a flattened average looks like.
                dx, dy, dz = rotate_zyz(dx, dy, dz, float(r[rot]),
                                        float(r[tilt]), float(r[psi]))
            sgn = args.recenter_sign
            x -= sgn * dx / apx
            y -= sgn * dy / apx
            z -= sgn * dz / apx
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
    if keep_ang:
        print("Carry angles:   YES -- WARNING: Warp ORIENTS each subtomogram "
              "by these, and RELION then rotates it again.")
        print("                Expect a featureless reference. Drop them and "
              "attach with ml_merge_angles.py after export.")
    else:
        print("Carry angles:   no (correct -- Warp needs UNORIENTED picks)")
        if have_angles:
            print("                the refined Eulers are in the source star; "
                  "attach them to the EXPORTED star with ml_merge_angles.py")
    if recenter:
        print(f"Largest shift:  {max_shift:.1f} Å")
        print(f"                sign {args.recenter_sign:+g}, frame "
              f"'{args.recenter_frame}'")
        print("                WARNING: recentring is OFF by default because "
              "it produced a featureless")
        print("                reference here. You have opted back in.")
    elif have_origins:
        print("                (the star's refined origins were IGNORED, which "
              "is what works)")
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
    # Accepted and ignored: this is now the only behaviour. Kept so the saved
    # jobs and cards that still pass it keep running instead of erroring on an
    # unrecognised flag.
    # THE MODE. Three ways to get coordinates out of a RELION selection, each
    # with a DIFFERENT output convention -- and choosing wrong is silent, so it
    # is one explicit choice rather than something inferred from which other
    # flags happen to be set.
    ap.add_argument("--mode", choices=("A", "B", "C"),
                    default=env_default("MODE", "B"),
                    help="A = re-normalise: write 0-1 FRACTIONS (export with "
                         "--normalized_coords). "
                         "B = filter the ORIGINAL pick stars verbatim, keeping "
                         "their convention (export with the same flags the "
                         "first export used) -- the route with a working "
                         "precedent. "
                         "C = coordinates straight from the RELION star as "
                         "PIXELS (export with --coords_angpix).")
    ap.add_argument("--picks-dir", default=env_default("PICKS_DIR", ""),
                    help="MODE B: the ORIGINAL pick stars the first export read.")
    ap.add_argument("--pattern", default=env_default("PATTERN", "*.star"),
                    help="MODE B: glob for those pick stars.")
    ap.add_argument("--recon-dir", default=env_default("RECON_DIR", ""),
                    help="MODE A: the reconstructions, read ONLY for their "
                         "dimensions (a fraction is coord / tomogram size).")
    ap.add_argument("--relion-coords", action="store_true", default=True,
                    help="(no longer optional — coordinates always come from the "
                         "RELION star; accepted for compatibility)")
    ap.add_argument("--coords-angpix", default=env_default("COORDS_ANGPIX", ""),
                    help="override the coords' Å/px (default: read from the star's "
                         "_rlnImagePixelSize / _rlnPixelSize).")
    ap.add_argument("--no-recenter", dest="recenter", action="store_false",
                    default=env_default("RECENTER", "0") == "1",
                    help="do NOT apply the refined _rlnOriginXYZAngst offsets. "
                         "This is now the DEFAULT — see --recenter.")
    # DEFAULT OFF, and it took a ruined classification to learn why.
    #
    # Warp uses the angles in a pick star to ORIENT each subtomogram it
    # reconstructs. Carry the refined Eulers here and the volumes come out
    # already rotated into the reference frame — then RELION applies the same
    # angles again as priors and every particle is rotated twice, by a
    # different amount each. Positions stay perfect (verified to 0.000 px), so
    # nothing errors and nothing looks wrong until a classification ten
    # iterations in is grinding on mush.
    #
    # The tell is relion_reconstruct on a random subset: with angles ZERO it
    # averages the subtomograms and shows clear central density; with angles
    # carried it is featureless noise.
    #
    # The refined orientations are still worth having — they just belong in
    # the EXPORTED star, after extraction, not in the pick star that Warp
    # reads. ml_merge_angles.py puts them there.
    # DEFAULT OFF, established by elimination on 2026-09-02 after a
    # re-extraction produced a featureless reference and a ruined
    # classification. Reference skew (max/|min| of a 40 A low-passed
    # 1000-particle reconstruction; ~1.8 means a centred object, ~0.9 means
    # noise):
    #
    #   bin4 crYOLO, normalized coords, no recentre     2.01  centred
    #   bin1 crYOLO, normalized coords, no recentre     1.77  centred
    #   bin1 crYOLO, coords_angpix,     no recentre     1.81  centred
    #   bin4 SELECTED particles, origins zeroed         1.79  centred
    #   bin1 re-extract, coords_angpix, RECENTRED       0.86  FLAT
    #
    # Every arrangement without recentring is centred, across two pixel sizes,
    # two coordinate conventions and both particle sets. The one with it is the
    # only failure. WHY it fails is not established: coord - origin/angpix is
    # RELION's documented convention and the shifts are small (~20 A typical,
    # 99.8 A max, about 3 px at bin4), so a plain sign inversion would blur the
    # average rather than flatten it. The origin is likely expressed in a frame
    # that is not the tomogram's axes. Until someone settles that, do not do it.
    #
    # Skipping it costs almost nothing: particles keep their original pick
    # centres -- exactly what produced the good map -- and the next refinement
    # finds the shifts again.
    # Two candidate explanations for why recentring destroys the average, kept
    # as switches so ONE export settles it instead of another argument. Both
    # displace a particle by a similar amount, which is why the resulting maps
    # look the same and reasoning cannot separate them.
    ap.add_argument("--recenter-sign", type=float, default=1.0,
                    help="1 subtracts the origin (RELION's documented "
                         "convention); -1 adds it. Use -1 to test whether the "
                         "sign is inverted here.")
    ap.add_argument("--recenter-frame", choices=("tomogram", "rotated"),
                    default="tomogram",
                    help="'tomogram' applies the origin along tomogram axes. "
                         "'rotated' first rotates it by the particle's Eulers, "
                         "testing whether the origin is expressed in the "
                         "particle's own frame.")
    ap.add_argument("--recenter", dest="recenter", action="store_true",
                    help="apply the refined _rlnOriginXYZAngst offsets. OFF by "
                         "default: doing this produced a featureless reference "
                         "and a classification on noise, while every run "
                         "without it reconstructed cleanly. Opt in only to "
                         "test the frame convention.")
    ap.add_argument("--keep-angles", dest="keep_angles", action="store_true",
                    default=env_default("KEEP_ANGLES", "0") == "1",
                    help="carry the refined Euler angles into the pick stars. "
                         "OFF by default: Warp ORIENTS each subtomogram by "
                         "them, so RELION then rotates an already-rotated "
                         "particle and the reconstruction is noise. Use "
                         "ml_merge_angles.py to attach them to the exported "
                         "star instead.")
    ap.add_argument("--no-keep-angles", dest="keep_angles", action="store_false",
                    help="explicit form of the default (kept so existing "
                         "commands and saved job cards still run).")
    # common
    ap.add_argument("--out-dir", default=env_default("OUT_DIR", ""),
                    help="output dir for the filtered/rebuilt pick STARs (required)")
    ap.add_argument("--suffix", default=env_default("SUFFIX", "good"),
                    help="tag added to output filenames (default 'good')")
    ap.add_argument("--fom", type=float, default=float(env_default("FOM", "1.0")),
                    help="constant _rlnAutopickFigureOfMerit written for every "
                         "pick (default 1.0). These coordinates come from a "
                         "RELION selection you already curated, so there is no "
                         "per-particle score left to carry — the column exists "
                         "because the format has it.")
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

    # --relion-coords used to BE the mode switch; honour it as MODE C so saved
    # job cards and old commands keep doing what they did.
    mode = (args.mode or "B").upper()
    # `flag`, not `a`: every other tool here binds `a = ap.parse_args()`, so a
    # loop variable of that name reads as the args namespace at a glance.
    if args.relion_coords and not any(f.startswith("--mode") for f in sys.argv):
        mode = "C"
    if not args.out_dir:
        sys.exit("ERROR: --out-dir is required")
    args.out_dir = os.path.abspath(args.out_dir)
    return {"A": mode_a, "B": mode_b, "C": mode_c}[mode](args)


if __name__ == "__main__":
    main()
