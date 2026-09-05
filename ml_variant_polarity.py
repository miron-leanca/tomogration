#!/usr/bin/env python3
"""Detect the contrast polarity of a tomogram variant, and record it.

    python3 ml_variant_polarity.py <tomogram.mrc> <segmentation.mrc> [--root DIR]
                                   [--variant-path REL] [--quiet]

Membrane polarity decides the SIGN of the density-support gate. Getting it
wrong does not fail loudly: the gate simply selects the opposite of what it
should, so every real virion is rejected and phantoms in bulk ice are accepted
(cards spec draft 2, Trap 1).

So it is measured, not declared. The segmentation already says which voxels are
membrane; this samples the tomogram at exactly those voxels and compares their
median against the whole volume's. Membranes darker than their surroundings =
'dark' (what fitpop2's gate assumes); brighter = 'bright' (Warp's inverted
convention, and what IsoNet 2 inherits from it).

Needs numpy + mrcfile, so run it in an env that has them (membrainseg, or the
IsoNet 2 prefix env) — NOT the GUI's venv. The decision rule itself lives in
tomogration_variants.polarity_from_medians, so the GUI and this script cannot
drift apart.

Exit 0 with the polarity on stdout; exit 1 if the volumes are unusable.
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tomogration_variants import (        # noqa: E402  (path set above)
    BRIGHT, DARK, UNKNOWN, polarity_from_medians, record_polarity)

# A tomogram is several GB; a stride keeps this to seconds and cannot bias the
# comparison, since the SAME stride is applied to both volumes.
STRIDE = 2


def load(path):
    import mrcfile                        # noqa: PLC0415 (optional dependency)
    with mrcfile.mmap(str(path), permissive=True) as m:
        return m.data[::STRIDE, ::STRIDE, ::STRIDE]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tomogram")
    ap.add_argument("segmentation")
    ap.add_argument("--root", default="", help="project root, to record the result")
    ap.add_argument("--variant-path", default="",
                    help="the variant's registry path (project-relative)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="segmentation values above this count as membrane "
                         "(default: >0 for a label map, else the midpoint)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out", default="",
                    help="folder for polarity.json — the measurement's own "
                         "record, so the job is not an empty directory")
    a = ap.parse_args(argv)

    try:
        import numpy as np
    except ImportError:
        print("ERROR: numpy is not in this env. Run this from membrainseg or "
              "the IsoNet 2 env, not the GUI's venv.", file=sys.stderr)
        return 1

    for p in (a.tomogram, a.segmentation):
        if not Path(p).is_file():
            print(f"ERROR: not a file: {p}", file=sys.stderr)
            return 1

    tomo = np.asarray(load(a.tomogram), dtype="f4")
    seg = np.asarray(load(a.segmentation), dtype="f4")
    if tomo.shape != seg.shape:
        print(f"ERROR: tomogram {tomo.shape} and segmentation {seg.shape} are "
              f"different shapes — they must be the same volume at the same "
              f"binning.", file=sys.stderr)
        return 1

    if a.threshold is None:
        # A segmentation is either a 0/1 (or 0/255) mask or a score map. For a
        # mask anything above zero is membrane; for a score map, halfway.
        lo, hi = float(seg.min()), float(seg.max())
        thr = 0.0 if lo >= 0 and hi <= 1.0000001 else (lo + hi) / 2.0
    else:
        thr = a.threshold
    mask = seg > thr
    n = int(mask.sum())
    frac = n / float(mask.size or 1)
    if n < 1000:
        print(f"ERROR: only {n} voxel(s) above {thr:g} in {a.segmentation} — "
              f"too few to measure a polarity from. Segment first, or pass a "
              f"lower --threshold.", file=sys.stderr)
        return 1
    if frac > 0.5:
        print(f"WARNING: the segmentation covers {frac:.0%} of the volume; the "
              f"membrane median and the global median describe nearly the same "
              f"voxels, so this measurement is weak.", file=sys.stderr)

    memb = float(np.median(tomo[mask]))
    whole = float(np.median(tomo))
    # A tolerance in units of the volume's own spread: 'the same median' has to
    # mean something scale-free, or a variant with tiny dynamic range always
    # looks decisive.
    tol = 0.01 * float(tomo.std())
    polarity = polarity_from_medians(memb, whole, tol=tol)

    if not a.quiet:
        print(f"membrane voxels : {n} ({frac:.2%} of the volume, > {thr:g})")
        print(f"median at membrane: {memb:+.6g}")
        print(f"median overall    : {whole:+.6g}   (tolerance {tol:.3g})")
        print(f"polarity: {polarity}"
              + ("   — membranes are DARKER than their surroundings, which is "
                 "what the density-support gate assumes"
                 if polarity == DARK else
                 "   — membranes are BRIGHTER (Warp's inverted convention); "
                 "the density gate must flip its sign for this variant"
                 if polarity == BRIGHT else
                 "   — the two medians are within tolerance; do not gate on "
                 "density for this variant until this is resolved"))

    if a.root and not a.variant_path:
        # The GUI card's help promises 'Blank = derive it from the tomogram
        # path' — deliver it: the variant IS the folder the tomogram lives in,
        # relative to the project root. Without this, the measurement ran,
        # printed, and silently never reached the registry everything
        # downstream reads.
        try:
            a.variant_path = os.path.relpath(
                Path(a.tomogram).resolve().parent, Path(a.root).resolve())
        except ValueError:
            pass                          # different drive etc. — stay blank
        if a.variant_path and not a.variant_path.startswith(".."):
            print(f"variant path derived from the tomogram: {a.variant_path}")
        else:
            a.variant_path = ""
            print("WARNING: tomogram lies outside --root — polarity NOT "
                  "recorded to the registry (pass --variant-path).",
                  file=sys.stderr)
    if a.root and a.variant_path:
        warn = record_polarity(a.root, a.variant_path, polarity,
                               source=str(a.segmentation))
        if warn:
            print(f"WARNING: {warn}", file=sys.stderr)
    # A record in the job's own folder. The registry holds the LIVE value that
    # everything downstream reads, but a job whose output directory is empty
    # looks like one that did nothing — and the numbers behind a polarity call
    # are worth keeping when a later run disagrees with it.
    if a.out:
        import json                       # noqa: PLC0415
        out_dir = Path(a.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "polarity.json").write_text(json.dumps({
            "polarity": polarity,
            "tomogram": str(Path(a.tomogram).resolve()),
            "segmentation": str(Path(a.segmentation).resolve()),
            "variant_path": a.variant_path or "",
            "median_at_membrane": float(memb),
            "median_overall": float(whole),
            "tolerance": float(tol),
            "membrane_voxels": int(n),
            "membrane_fraction": float(frac),
            "recorded_in_registry": bool(a.variant_path),
        }, indent=1))
        print(f"wrote {out_dir / 'polarity.json'}")
    print(polarity)
    return 0 if polarity != UNKNOWN else 1


if __name__ == "__main__":
    sys.exit(main())
