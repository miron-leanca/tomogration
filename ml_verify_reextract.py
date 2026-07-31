#!/usr/bin/env python3
# ml_verify_reextract.py
#
# Prove a re-extraction landed where the RELION star said it should — BEFORE spending
# hours on a refinement that turns out to be noise.
#
# WHY: every failure mode in the RELION->Warp re-extraction loop is SILENT. Wrong
# coordinate convention (normalised read as pixels, or vice versa), a wrong
# --coords_angpix, a missed recentring — none of them error. Warp writes a perfectly
# well-formed star full of subtomograms cut from the wrong places, RELION refines it
# happily, and you only find out hours later when the map is a blob. The one thing that
# distinguishes a good re-extraction from a bad one is arithmetic:
#
#     new_coord (at new angpix) == (source_coord - origin/source_angpix) * (source_angpix / new_angpix)
#
# So compare the two stars directly. Agreement to well under a pixel = the extraction
# used the right convention. A constant factor off (2x, 0.5x) = an angpix/scale mistake.
# Residuals the size of the tomogram = the normalised/pixel mix-up.
#
# Usage:
#   python3 ml_verify_reextract.py <source_relion.star> <new_matching.star> [--no-recenter]
#
#     <source_relion.star>  what you re-extracted FROM (e.g. Select/job009/particles.star)
#     <new_matching.star>   what the export WROTE     (e.g. relion4/<proj>/matching.star)
#
# Particles are paired per tomogram IN ORDER (the converter writes them in star order and
# ts_export_particles preserves that order), which needs no shared identifier between the
# two files. Read-only: reports, writes nothing.

import argparse
import os
import sys


def read_star_blocks(path):
    blocks, cur, in_loop, reading = [], None, False, False
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("data_"):
                cur = {"name": line[5:], "cols": {}, "rows": []}
                blocks.append(cur)
                in_loop = reading = False
                continue
            if cur is None or line.startswith("#"):
                continue
            if line == "loop_":
                in_loop, reading = True, False
                continue
            if in_loop and line.startswith("_"):
                cur["cols"][line.split()[0][1:]] = len(cur["cols"])
                continue
            if in_loop and line and not line.startswith("_"):
                in_loop, reading = False, True
            if reading:
                if not line:
                    reading = False
                    continue
                cur["rows"].append(line.split())
    return blocks


def particles_block(blocks, need):
    for b in blocks:
        if all(c in b["cols"] for c in need):
            return b
    return None


def pixel_size(blocks, pb):
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


def tomo_of(name):
    b = os.path.basename(name)
    return b[:-9] if b.endswith(".tomostar") else os.path.splitext(b)[0]


def load(path, want_origins):
    blocks = read_star_blocks(path)
    pb = particles_block(blocks, ("rlnCoordinateX", "rlnMicrographName"))
    if pb is None:
        sys.exit(f"ERROR: {path} has no _rlnCoordinateX + _rlnMicrographName block")
    c = pb["cols"]
    apx = pixel_size(blocks, pb)
    ox, oy, oz = (c.get("rlnOriginXAngst"), c.get("rlnOriginYAngst"),
                  c.get("rlnOriginZAngst"))
    has_org = want_origins and None not in (ox, oy, oz)
    out = {}
    for r in pb["rows"]:
        x, y, z = (float(r[c["rlnCoordinateX"]]), float(r[c["rlnCoordinateY"]]),
                   float(r[c["rlnCoordinateZ"]]))
        if has_org and apx:
            x -= float(r[ox]) / apx
            y -= float(r[oy]) / apx
            z -= float(r[oz]) / apx
        out.setdefault(tomo_of(r[c["rlnMicrographName"]]), []).append((x, y, z))
    return out, apx, has_org


def main():
    ap = argparse.ArgumentParser(
        description="Verify a re-extraction's coordinates against the RELION star it came from.")
    ap.add_argument("source_star", help="the RELION star you re-extracted FROM")
    ap.add_argument("new_star", help="the matching.star the export WROTE")
    ap.add_argument("--no-recenter", dest="recenter", action="store_false",
                    help="the converter did NOT apply the refined origins")
    args = ap.parse_args()

    for p in (args.source_star, args.new_star):
        if not os.path.isfile(p):
            sys.exit(f"ERROR: not found: {p}")

    src, src_apx, used_org = load(args.source_star, args.recenter)
    new, new_apx, _ = load(args.new_star, False)
    if not src_apx or not new_apx:
        sys.exit("ERROR: could not read the pixel size from both stars "
                 "(_rlnImagePixelSize / _rlnPixelSize).")
    scale = src_apx / new_apx

    n_src = sum(len(v) for v in src.values())
    n_new = sum(len(v) for v in new.values())
    print("===================================================================")
    print("ml_verify_reextract")
    print(f"source: {args.source_star}")
    print(f"        {n_src} particles, {len(src)} tomograms, {src_apx:g} Å/px"
          f"{'  (recentred by refined origins)' if used_org else ''}")
    print(f"new:    {args.new_star}")
    print(f"        {n_new} particles, {len(new)} tomograms, {new_apx:g} Å/px")
    print(f"expected coordinate scale: x{scale:g}")
    print("===================================================================")

    shared = sorted(set(src) & set(new))
    if not shared:
        sys.exit("ERROR: no tomograms in common — are these from the same project?")
    only_src, only_new = sorted(set(src) - set(new)), sorted(set(new) - set(src))
    if only_src:
        print(f"!! {len(only_src)} tomogram(s) in source but NOT in the new star "
              f"(e.g. {', '.join(only_src[:4])})")
    if only_new:
        print(f"!! {len(only_new)} tomogram(s) in the new star but not the source "
              f"(e.g. {', '.join(only_new[:4])})")

    worst, tot, cnt, mismatched = 0.0, 0.0, 0, []
    for t in shared:
        a, b = src[t], new[t]
        if len(a) != len(b):
            mismatched.append((t, len(a), len(b)))
            continue
        for (x1, y1, z1), (x2, y2, z2) in zip(a, b):
            d = max(abs(x1 * scale - x2), abs(y1 * scale - y2), abs(z1 * scale - z2))
            tot += d
            cnt += 1
            worst = max(worst, d)
    if mismatched:
        print(f"!! {len(mismatched)} tomogram(s) have different particle counts "
              f"(cannot pair in order):")
        for t, na, nb in mismatched[:5]:
            print(f"     {t}: source {na}  vs  new {nb}")
    if not cnt:
        sys.exit("ERROR: no tomogram could be paired — counts differ everywhere.")

    mean = tot / cnt
    print("-------------------------------------------------------------------")
    print(f"paired {cnt} particles across {len(shared) - len(mismatched)} tomograms")
    print(f"mean |error| = {mean:.3f} px      max |error| = {worst:.3f} px "
          f"(in new-star pixels of {new_apx:g} Å)")
    print("-------------------------------------------------------------------")
    if worst < 1.0:
        print("VERDICT: PASS — the re-extraction used the right coordinate convention.")
        print("         Particles were cut where the RELION star said. Safe to refine.")
    elif worst < 5.0:
        print("VERDICT: CLOSE — sub-5px disagreement. Usually rounding or a slightly "
              "different tomogram dimension; probably fine, but eyeball a few subtomos.")
    else:
        print("VERDICT: FAIL — the particles were NOT cut where the star says.")
        ratios = []
        for t in shared[:20]:
            a, b = src[t], new[t]
            if len(a) == len(b):
                for (x1, _, _), (x2, _, _) in zip(a[:20], b[:20]):
                    if abs(x1) > 1e-6:
                        ratios.append(x2 / x1)
        if ratios:
            r = sorted(ratios)[len(ratios) // 2]
            print(f"         median new/source X ratio = {r:.4f} "
                  f"(expected {scale:g}).")
            if abs(r) > 1e-9 and 0.4 < (r / scale) < 0.6:
                print("         ~HALF the expected scale — an angpix is doubled somewhere.")
            elif 1.8 < (r / scale) < 2.2:
                print("         ~TWICE the expected scale — an angpix is halved somewhere.")
            elif abs(r) < 1e-3:
                print("         Coordinates collapsed toward zero — classic sign of "
                      "NORMALISED coords exported WITHOUT --normalized_coords "
                      "(0-1 values read as pixels). Re-export with the right flags.")
        print("         Do NOT refine this. Fix the export flags and re-extract.")


if __name__ == "__main__":
    main()
