#!/usr/bin/env python3
# ml_star_check_particles.py
#
# Check every particle in a RELION/Warp particle STAR actually exists on disk, and
# optionally write a PRUNED star containing only the ones that do.
#
# WHY: RELION reads the star, not the disk. If the star lists a subtomogram whose .mrc
# was never written, the refinement runs happily for an iteration or two and THEN dies
# mid-expectation with
#     ERROR: Cannot read file .../PositionNNN_0000122_3.14A.mrc It does not exist
# having burned however long that took. That happens when an export was interrupted,
# ran out of space, skipped particles it could not extract (e.g. too close to the
# tomogram edge for the requested box), or when the star and the subtomo/ folder come
# from two different runs. This tool turns that late, expensive crash into a fast
# up-front check — and can repair the star so the refinement can just proceed.
#
# Usage:
#   python3 ml_star_check_particles.py <particles.star> [--root DIR] [--fix] [--out FILE]
#
#     <particles.star>  the star RELION will read (e.g. matching_conv.star)
#     --root DIR        resolve RELATIVE _rlnImageName paths against this dir
#                       (default: the star's own directory, then the cwd)
#     --fix             write a pruned star keeping only particles whose file exists
#     --out FILE        where to write it (default: <star>_present.star)
#
# Default is a REPORT ONLY — it writes nothing until you pass --fix. The pruned star
# preserves every other block (optics etc.) and the original formatting verbatim; only
# the missing particle rows are dropped.
#
# Reads each subtomo directory ONCE (listdir + set membership) instead of stat-ing
# every particle, so it stays fast on ceph with tens of thousands of particles.

import argparse
import os
import sys
from collections import Counter


def mrc_box_angpix(path):
    """(box, angpix) from an MRC header: nx at offset 0, grid mx at 28, cell dims (Å) at
    40. angpix = cella_x / mx. Returns (None, None) if unreadable."""
    import struct
    try:
        with open(path, "rb") as f:
            h = f.read(64)
        nx = struct.unpack("<i", h[0:4])[0]
        mx = struct.unpack("<i", h[28:32])[0] or nx
        cella_x = struct.unpack("<f", h[40:44])[0]
        return nx, (cella_x / mx if mx else None)
    except (OSError, struct.error):
        return None, None


def check_reference(ref, star_box, star_apx, roots):
    """RELION 4 does NOT rescale a reference to match the particles — a mismatch in box
    or pixel size is a hard error (or, worse, nonsense). Verify the map exists and that
    its box + angpix match the particle star. Returns True if it looks usable."""
    path = ref if os.path.isabs(ref) else None
    if path is None:
        for r in roots:
            if os.path.isfile(os.path.join(r, ref)):
                path = os.path.join(r, ref)
                break
        path = path or os.path.join(roots[0], ref)
    print("------------------------------ REFERENCE --------------------------")
    print(f"ref: {ref}")
    if not os.path.isfile(path):
        print(f"  MISSING — no such file: {path}")
        print("  RELION resolves RELATIVE paths from the directory it was LAUNCHED in.")
        print("  Give an absolute path, or a path relative to your launch dir.")
        near = os.path.dirname(path) or "."
        try:
            cand = [f for f in os.listdir(near) if f.lower().endswith((".mrc", ".map"))]
            if cand:
                print(f"  Maps found in {near}:")
                for c in sorted(cand)[:8]:
                    print(f"      {c}")
        except OSError:
            pass
        return False
    box, apx = mrc_box_angpix(path)
    print(f"  found: {path}")
    print(f"  reference box = {box}   angpix = {apx:.4f}" if apx else
          f"  reference box = {box}   angpix = ?")
    print(f"  particles  box = {star_box}   angpix = {star_apx}")
    ok = True
    # int(float(...)): RELION writes some integer fields as '64.000000'.
    if star_box and box and int(float(box)) != int(float(star_box)):
        print(f"  ✗ BOX MISMATCH ({box} vs {star_box}). RELION 4 will NOT rescale it. "
              f"Rescale first:")
        print(f"      relion_image_handler --i {path} --angpix {apx or '<ref apx>'} "
              f"--rescale_angpix {star_apx} --new_box {star_box} --o rescaled_ref.mrc")
        ok = False
    if star_apx and apx and abs(apx - float(star_apx)) > 0.01 * float(star_apx):
        print(f"  ✗ PIXEL-SIZE MISMATCH ({apx:.4f} vs {star_apx}).")
        ok = False
    if ok:
        print("  ✓ reference matches the particles (box + pixel size).")
    return ok


def resolve(img, roots):
    """Absolute path for an _rlnImageName. Handles RELION's 'N@stack.mrcs' prefix and
    relative paths (tried against each root in turn)."""
    p = img.split("@", 1)[1] if "@" in img else img
    if os.path.isabs(p):
        return p, None
    return None, p


def main():
    ap = argparse.ArgumentParser(
        description="Verify every particle in a STAR exists on disk; optionally prune.")
    ap.add_argument("star", help="particle STAR file to check")
    ap.add_argument("--root", default="",
                    help="dir to resolve relative _rlnImageName paths against "
                         "(default: the star's directory, then the cwd)")
    ap.add_argument("--fix", action="store_true",
                    help="write a pruned star with only the particles that exist")
    ap.add_argument("--out", default="",
                    help="output path for --fix (default: <star>_present.star)")
    ap.add_argument("--ref", default="",
                    help="also check a REFERENCE map: that it exists and that its box + "
                         "pixel size match the particles (RELION 4 does not rescale it).")
    args = ap.parse_args()

    if not os.path.isfile(args.star):
        sys.exit(f"ERROR: star not found: {args.star}")
    star_dir = os.path.dirname(os.path.abspath(args.star))
    roots = [r for r in (args.root, star_dir, os.getcwd()) if r]

    lines = open(args.star).read().splitlines()

    # ---- optics: the particles' box + pixel size, for the reference check ----
    star_box = star_apx = None
    o_in_loop, o_cols = False, {}
    for raw in lines:
        s = raw.strip()
        if s == "loop_":
            o_in_loop, o_cols = True, {}
            continue
        if o_in_loop and s.startswith("_"):
            o_cols[s.split()[0][1:]] = len(o_cols)
            continue
        if o_in_loop and s and not s.startswith(("_", "#", "data_")):
            if "rlnImageSize" in o_cols or "rlnImagePixelSize" in o_cols:
                f = s.split()
                if "rlnImageSize" in o_cols and len(f) > o_cols["rlnImageSize"]:
                    star_box = f[o_cols["rlnImageSize"]]
                if "rlnImagePixelSize" in o_cols and len(f) > o_cols["rlnImagePixelSize"]:
                    star_apx = f[o_cols["rlnImagePixelSize"]]
                break
            o_in_loop = False

    # ---- locate the particles block's loop header + its _rlnImageName column ----
    block, in_loop, cols = None, False, {}
    img_col, data_start = None, None
    for i, raw in enumerate(lines):
        s = raw.strip()
        if s.startswith("data_"):
            block, in_loop, cols = s[5:], False, {}
            continue
        if s == "loop_":
            in_loop, cols = True, {}
            continue
        if in_loop and s.startswith("_"):
            cols[s.split()[0][1:]] = len(cols)
            continue
        if in_loop and s and not s.startswith("_") and not s.startswith("#"):
            if "rlnImageName" in cols:            # first data row of the particles block
                img_col, data_start = cols["rlnImageName"], i
                break
            in_loop = False
    if img_col is None:
        sys.exit(f"ERROR: no loop block with _rlnImageName in {args.star}")

    # ---- walk the data rows; cache one listdir per subtomo directory ----
    dir_cache = {}

    def exists(path):
        d, name = os.path.split(path)
        names = dir_cache.get(d)
        if names is None:
            try:
                names = set(os.listdir(d))
            except OSError:
                names = set()
            dir_cache[d] = names
        return name in names

    keep_flags, n_rows, missing = [], 0, []
    per_tomo_missing, per_tomo_total = Counter(), Counter()
    in_particles = True
    for i in range(data_start, len(lines)):
        s = lines[i].strip()
        if s.startswith("data_"):
            # A later block's rows are NOT particle rows — never prune them.
            in_particles = False
        if not s or s.startswith("#") or s.startswith("data_") or s.startswith("loop_"):
            keep_flags.append((i, True))
            continue
        if not in_particles:
            keep_flags.append((i, True))
            continue
        f = s.split()
        if len(f) <= img_col:
            keep_flags.append((i, True))
            continue
        n_rows += 1
        img = f[img_col]
        absp, rel = resolve(img, roots)
        if absp is None:
            absp = next((os.path.join(r, rel) for r in roots
                         if os.path.isdir(os.path.dirname(os.path.join(r, rel)))),
                        os.path.join(roots[0], rel))
        tomo = os.path.basename(os.path.dirname(absp)) or "?"
        per_tomo_total[tomo] += 1
        ok = exists(absp)
        if not ok:
            missing.append(absp)
            per_tomo_missing[tomo] += 1
        keep_flags.append((i, ok))

    n_missing = len(missing)
    n_present = n_rows - n_missing
    print("===================================================================")
    print(f"ml_star_check_particles    [{'FIX' if args.fix else 'REPORT'}]")
    print(f"STAR:      {args.star}")
    print(f"particles: {n_rows}")
    print(f"present:   {n_present}")
    print(f"MISSING:   {n_missing}"
          + (f"   ({100.0 * n_missing / n_rows:.1f}%)" if n_rows else ""))
    print("===================================================================")
    if n_missing:
        print(f"Tomograms affected: {len(per_tomo_missing)} of {len(per_tomo_total)}")
        print("Worst tomograms (missing / total):")
        for tomo, k in per_tomo_missing.most_common(10):
            print(f"    {tomo}: {k} / {per_tomo_total[tomo]}")
        print("First few missing files:")
        for m in missing[:5]:
            print(f"    {m}")
        whole = sum(1 for t, k in per_tomo_missing.items() if k == per_tomo_total[t])
        print("-------------------------------------------------------------------")
        if whole:
            print(f"NOTE: {whole} tomogram(s) are missing EVERY particle — that looks "
                  f"like an export that never reached them (interrupted / failed), not "
                  f"edge-clipped particles. Consider re-running the export instead.")
        else:
            print("NOTE: missing particles are scattered within tomograms — typical of "
                  "particles the exporter could not extract (too close to the tomogram "
                  "edge for the requested box). Pruning them is the normal fix.")
    else:
        print("All particles present — this star is safe to refine.")
        if args.ref:
            check_reference(args.ref, star_box, star_apx, roots)
        return
    if args.ref:
        check_reference(args.ref, star_box, star_apx, roots)

    if not args.fix:
        print("REPORT ONLY - nothing written. Re-run with --fix to write a pruned star.")
        return

    out = args.out or (os.path.splitext(args.star)[0] + "_present.star")
    drop = {i for i, ok in keep_flags if not ok}
    with open(out, "w") as fh:
        for i, raw in enumerate(lines):
            if i not in drop:
                fh.write(raw + "\n")
    print(f"Wrote pruned star: {out}")
    print(f"    {n_present} particles kept, {n_missing} dropped.")
    print("Point RELION at this file instead (and re-run from the project root).")


if __name__ == "__main__":
    main()
