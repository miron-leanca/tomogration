#!/usr/bin/env python3
# ml_audit_chain.py
#
# Audit a whole pick -> export -> RELION -> re-extract chain in one pass, and
# say WHERE it diverges.
#
# WHY THIS EXISTS: every other check in this pipeline compares one star to
# another star. ml_verify_reextract proved Warp's output coordinates were
# exactly 4x its input -- and the extraction was still noise, because scaling
# numbers correctly says nothing about whether Warp CUT at those numbers. Four
# hypotheses (coords_angpix, carried Eulers, display filtering, recentring)
# were chased on arithmetic alone.
#
# So section 2 reads the SUBTOMOGRAMS. A stack of particles centred on real
# objects has a radial density profile that changes from centre to edge; a
# stack cut from arbitrary ice is flat. That measurement, run on a known-good
# export and a suspect one, answers in numbers what a reference volume only
# hints at.
#
#   python3 ml_audit_chain.py --good <good_export_dir> --bad <bad_export_dir> \
#                             [--star extra.star ...] [--tomo-dir DIR] [-n 40]
#
# Needs numpy + mrcfile for section 2. Without them sections 1 and 3 still run.
import argparse
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ml_relion4_select_picks import (read_star_blocks, particles_block,  # noqa: E402
                                     star_pixel_size)

ANG = ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")
ORI = ("rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst")
RULE = "=" * 72


def nums(pb, name):
    i = pb["cols"].get(name)
    if i is None:
        return None
    out = []
    for r in pb["rows"]:
        try:
            out.append(float(r[i]))
        except (ValueError, IndexError):
            pass
    return out or None


def describe(path):
    """Section 1: how does this star store coordinates and orientations?"""
    print(f"\n--- {path}")
    if not os.path.isfile(path):
        print("    MISSING")
        return None
    blocks = read_star_blocks(path)
    pb = particles_block(blocks, need=("rlnCoordinateX",))
    if pb is None:
        print("    no particle block")
        return None
    apx = star_pixel_size(blocks, pb)
    print(f"    {len(pb['rows'])} particles   stated pixel size: "
          + (f"{apx:g} A/px" if apx else "NOT STATED"))
    for a in "XYZ":
        v = nums(pb, f"rlnCoordinate{a}")
        if v:
            print(f"    Coordinate{a}   {min(v):10.2f} .. {max(v):10.2f}"
                  + (f"   ({min(v)*apx:.0f} .. {max(v)*apx:.0f} A)" if apx else ""))
    # Orientations: all-zero is a DIFFERENT state from absent, and the two
    # behave differently in relion_reconstruct.
    have_ang = [c for c in ANG if nums(pb, c) is not None]
    if not have_ang:
        print("    angles      : columns ABSENT")
    else:
        allv = [v for c in ANG for v in (nums(pb, c) or [])]
        nz = sum(1 for v in allv if abs(v) > 1e-9)
        print(f"    angles      : present, {nz}/{len(allv)} non-zero"
              + ("   <-- ALL ZERO (unoriented)" if nz == 0 else
                 f"   range {min(allv):.1f} .. {max(allv):.1f} deg"))
    have_ori = [c for c in ORI if nums(pb, c) is not None]
    if not have_ori:
        print("    origins     : columns ABSENT (nothing to recentre by)")
    else:
        allv = [abs(v) for c in ORI for v in (nums(pb, c) or [])]
        nz = sum(1 for v in allv if v > 1e-9)
        print(f"    origins     : present, {nz}/{len(allv)} non-zero, "
              f"max |shift| {max(allv):.1f} A"
              + (f" = {max(allv)/apx:.1f} px" if apx else ""))
    img = pb["cols"].get("rlnImageName")
    if img is not None and pb["rows"]:
        print(f"    first image : {pb['rows'][0][img]}")
    return pb


def radial_profile(box, n_shells=12):
    """Mean density in concentric shells, centre outward."""
    import numpy as np                                       # noqa: PLC0415
    c = (np.array(box.shape) - 1) / 2.0
    z, y, x = np.indices(box.shape)
    r = np.sqrt((z - c[0])**2 + (y - c[1])**2 + (x - c[2])**2)
    rmax = min(box.shape) / 2.0
    edges = np.linspace(0, rmax, n_shells + 1)
    return [float(box[(r >= a) & (r < b)].mean()) if ((r >= a) & (r < b)).any()
            else float("nan") for a, b in zip(edges[:-1], edges[1:])]


def centring_test(export_dir, n=40, label=""):
    """Section 2: do the SUBTOMOGRAMS contain a centred object?

    Averages n random subtomograms and reports the radial profile of that
    average. Particles centred on a real object give a profile that RISES or
    FALLS from centre to edge by much more than the scatter between shells.
    Particles cut from arbitrary positions give a flat profile -- which is the
    numeric form of "the reference is featureless noise"."""
    # BOTH imports guarded, not just mrcfile. Guarding one and not the other
    # made section 2 die with a traceback on a plain python3 instead of saying
    # which interpreter to use -- after section 1 had already printed useful
    # results, which were then lost with the exit code.
    try:
        import numpy as np
        import mrcfile
    except ImportError as e:
        print(f"    SKIPPED -- {e.name} is not available to this python.")
        print("    Section 2 reads voxels, so it needs numpy + mrcfile:")
        print("      module load miniconda/latest && conda activate membrainseg")
        print("    then re-run this same command.")
        return None
    files = sorted(glob.glob(os.path.join(export_dir, "subtomo", "*", "*.mrc")))
    files = [f for f in files if "_ctf" not in os.path.basename(f)]
    if not files:
        print(f"    no subtomograms under {export_dir}/subtomo/")
        return None
    rng = np.random.default_rng(0)
    pick = [files[i] for i in rng.choice(len(files), size=min(n, len(files)),
                                         replace=False)]
    acc, shape = None, None
    for f in pick:
        with mrcfile.open(f, permissive=True) as m:
            d = np.asarray(m.data, dtype=np.float64)
        d = (d - d.mean()) / (d.std() or 1.0)      # per-particle normalise
        if acc is None:
            acc, shape = d.copy(), d.shape
        elif d.shape == shape:
            acc += d
    n_used = len(pick)
    avg = acc / n_used
    prof = radial_profile(avg)
    swing = max(prof) - min(prof)
    # The scale that matters is ABSOLUTE, not relative. Each particle was
    # normalised to unit variance, so averaging n of them leaves noise of
    # about 1/sqrt(n) per voxel -- and a shell mean is quieter still. Compare
    # the swing against that. Dividing by the outer scatter instead (the first
    # version) called PURE NOISE "centred", because when there is no signal
    # both numbers are tiny and their ratio is meaningless.
    floor = 1.0 / (n_used ** 0.5)
    snr = swing / floor
    print(f"    {label}{n_used} subtomograms of {shape}, from "
          f"{len(files)} available")
    print("    radial profile, centre -> edge:")
    print("      " + "  ".join(f"{p:+.3f}" for p in prof))
    print(f"    centre-to-edge swing {swing:.3f}   noise floor "
          f"1/sqrt({n_used}) = {floor:.3f}   -> SNR {snr:.1f}")
    print(f"    profile runs {'DOWNHILL (dense centre, protein dark)' if prof[0] < prof[-1] else 'UPHILL (bright centre)'}")
    if snr > 3:
        verdict = "CENTRED on a common object"
    elif snr > 1:
        verdict = ("WEAK -- something is there but smeared. Consistent with "
                   "every particle displaced by a different vector.")
    else:
        verdict = "FLAT -- these boxes are not centred on anything in common"
    print(f"    VERDICT: {verdict}")
    return snr


def trace(star_a, star_b, n=3):
    """Section 3: follow individual particles across two stars, in ANGSTROMS.

    Units are where this pipeline goes wrong, so compare the one thing that is
    unit-free: the physical position."""
    ba, bb = read_star_blocks(star_a), read_star_blocks(star_b)
    pa = particles_block(ba, need=("rlnCoordinateX",))
    pb_ = particles_block(bb, need=("rlnCoordinateX",))
    if pa is None or pb_ is None:
        print("    cannot trace: a star has no particle block")
        return
    aa, ab = star_pixel_size(ba, pa), star_pixel_size(bb, pb_)
    if not aa or not ab:
        print("    cannot trace: a star does not state its pixel size")
        return
    micA = next((c for c in ("rlnTomoName", "rlnMicrographName")
                 if c in pa["cols"]), None)
    micB = next((c for c in ("rlnTomoName", "rlnMicrographName")
                 if c in pb_["cols"]), None)
    idx = {}
    for r in pb_["rows"]:
        k = (os.path.basename(r[pb_["cols"][micB]]),) + tuple(
            round(float(r[pb_["cols"][f"rlnCoordinate{a}"]]) * ab)
            for a in "XYZ")
        idx.setdefault(k, r)
    print(f"    {os.path.basename(star_a)} @ {aa:g} A/px   vs   "
          f"{os.path.basename(star_b)} @ {ab:g} A/px")
    shown = 0
    for r in pa["rows"]:
        mic = os.path.basename(r[pa["cols"][micA]])
        ang = tuple(float(r[pa["cols"][f"rlnCoordinate{a}"]]) * aa for a in "XYZ")
        k = (mic,) + tuple(round(v) for v in ang)
        hit = idx.get(k)
        o = [pa["cols"].get(c) for c in ORI]
        shift = ""
        if all(i is not None for i in o):
            dx, dy, dz = (float(r[i]) for i in o)
            shift = f"   refined shift ({dx:+.1f},{dy:+.1f},{dz:+.1f}) A"
        print(f"      {mic:24s} ({ang[0]:8.1f},{ang[1]:8.1f},{ang[2]:8.1f}) A"
              f"  -> {'FOUND at the same place' if hit else 'NO MATCH'}{shift}")
        shown += 1
        if shown >= n:
            break


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--good", default="", help="export dir that WORKED")
    ap.add_argument("--bad", default="", help="export dir that did not")
    ap.add_argument("--star", action="append", default=[],
                    help="any other star to describe (repeatable)")
    ap.add_argument("--source-star", default="",
                    help="the RELION star the re-extraction came from")
    ap.add_argument("-n", type=int, default=40,
                    help="subtomograms to average per export (default 40)")
    a = ap.parse_args(argv)

    print(RULE)
    print("SECTION 1 -- how each star stores coordinates and orientations")
    print(RULE)
    stars = list(a.star)
    for d in (a.good, a.bad):
        if d:
            stars.append(os.path.join(d, "matching.star"))
    if a.source_star:
        stars.append(a.source_star)
    for s in stars:
        describe(s)

    print("\n" + RULE)
    print("SECTION 2 -- do the SUBTOMOGRAMS contain a centred object?")
    print("(this is the test nothing else in the pipeline performs)")
    print(RULE)
    ratios = {}
    for tag, d in (("GOOD", a.good), ("BAD ", a.bad)):
        if not d:
            continue
        print(f"\n--- {tag}: {d}")
        ratios[tag.strip()] = centring_test(d, a.n)

    print("\n" + RULE)
    print("SECTION 3 -- the same particles, in ANGSTROMS, across stars")
    print(RULE)
    if a.source_star and a.bad:
        print("\n--- source vs BAD export")
        trace(a.source_star, os.path.join(a.bad, "matching.star"))
    if a.source_star and a.good:
        print("\n--- source vs GOOD export")
        trace(a.source_star, os.path.join(a.good, "matching.star"))

    print("\n" + RULE)
    print("READ IT LIKE THIS")
    print(RULE)
    print("  NOTE: section 2 does a PLAIN average with no CTF correction, so it")
    print("  is less sensitive than relion_reconstruct and can call a real but")
    print("  weak signal 'flat'. The measure that settled this pipeline was the")
    print("  SKEW of a CTF-corrected reference:")
    print("      relion_image_handler --i <export>/random_subset_ref.mrc \\")
    print("          --o /tmp/r.mrc --lowpass 40 --angpix <output_angpix>")
    print("      relion_image_handler --i /tmp/r.mrc --stats")
    print("  max/|min| about 1.8 = a centred object; about 0.9 = noise.")
    print("  Trust that over section 2 when they disagree.")
    g, b = ratios.get("GOOD"), ratios.get("BAD")
    if g is not None and b is not None:
        print(f"  centring SNR   GOOD {g:.1f}   BAD {b:.1f}"
              f"   (>3 centred, 1-3 smeared, <1 flat)")
        if g > 3 and b <= 3:
            print("  -> The good export's boxes hold a centred object and the")
            print("     bad one's do not. The coordinates moved between them:")
            print("     suspect the RECENTRING, which is the one step that")
            print("     changes position and which star-to-star checks cannot")
            print("     see. Re-run the pick conversion with --no-recenter.")
        elif g <= 3 and b <= 3:
            print("  -> NEITHER is centred. The problem predates the")
            print("     re-extraction: the picks themselves, or the pixel size")
            print("     the picks were made at, are wrong.")
        elif g > 3 and b > 3:
            print("  -> BOTH are centred. Extraction is fine; the fault is")
            print("     downstream, in the orientations or the reference.")
    else:
        print("  Section 2 did not run -- use a python with numpy + mrcfile.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
