#!/usr/bin/env python3
"""Label-volume utilities for the membrane branch (module spec §3.5).

Run inside the membrainseg env (needs mrcfile + numpy):

    python ml_membrain_label_tools.py histogram <labels.mrc> [--top N]
    python ml_membrain_label_tools.py extract   <labels.mrc> <label> <out.mrc>

histogram — voxel count per label, sorted largest first. This is how merged
virions are spotted: a chain of 3–4 virions is ~4× the voxels of a single
shell, so a histogram with one entry ~4× the mode is a merge, not a big
virion. Also the way to pick one representative single virion for testing.

extract — write `label == N` as a binary mask (uint8), for feeding exactly one
membrane to the mesh stage.
"""
import argparse
import sys


def _load(path):
    try:
        import mrcfile
        import numpy as np
    except ImportError as e:
        sys.exit(f"ERROR: {e}. Run inside the membrainseg env "
                 f"(conda activate membrainseg).")
    try:
        with mrcfile.open(path, permissive=True) as m:
            # Checked BEFORE np.asarray: asarray(None) is a 0-d object array,
            # not None, so a data-less MRC would crash later instead of here.
            if m.data is None:
                sys.exit(f"ERROR: {path} has no data block.")
            data = np.asarray(m.data)
            voxel = m.voxel_size.copy()
    except (OSError, ValueError) as e:
        sys.exit(f"ERROR: cannot read {path}: {e}")
    return data, voxel


def cmd_histogram(args):
    import numpy as np
    data, _ = _load(args.labels)
    lab = np.rint(data).astype(np.int64)
    ids, counts = np.unique(lab, return_counts=True)
    keep = ids != 0                       # 0 = background
    ids, counts = ids[keep], counts[keep]
    if ids.size == 0:
        print("no labels found (volume is all background)")
        return 0
    order = np.argsort(counts)[::-1]
    ids, counts = ids[order], counts[order]
    median = float(np.median(counts))
    print(f"{ids.size} label(s); median size {median:.0f} voxels")
    print(f"{'label':>7}  {'voxels':>10}  {'x median':>8}  note")
    shown = ids if args.top <= 0 else ids[:args.top]
    for i, label in enumerate(shown):
        c = int(counts[i])
        ratio = c / median if median else 0.0
        note = ""
        if ratio >= 3.0:
            note = "≥3× median — likely MERGED virions (or carbon/edge)"
        elif 0.7 <= ratio <= 1.5:
            note = "~median — single-virion candidate"
        print(f"{int(label):>7}  {c:>10}  {ratio:>7.1f}x  {note}")
    if args.top > 0 and ids.size > args.top:
        print(f"... {ids.size - args.top} smaller label(s) not shown "
              f"(--top {args.top})")
    return 0


def cmd_extract(args):
    import numpy as np
    import mrcfile
    data, voxel = _load(args.labels)
    lab = np.rint(data).astype(np.int64)
    mask = (lab == args.label)
    n = int(mask.sum())
    if n == 0:
        present = ", ".join(str(int(x)) for x in np.unique(lab)[:12])
        sys.exit(f"ERROR: label {args.label} not present "
                 f"(labels found: {present} ...)")
    out = mask.astype(np.uint8)
    with mrcfile.new(args.out, overwrite=True) as m:
        m.set_data(out)
        m.voxel_size = voxel
    print(f"wrote {args.out}: label {args.label}, {n} voxels, binary uint8")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("histogram", help="voxel count per label, sorted")
    h.add_argument("labels", help="label volume (.mrc) from membrain components")
    h.add_argument("--top", type=int, default=30,
                   help="show only the N largest labels (0 = all)")
    h.set_defaults(fn=cmd_histogram)
    e = sub.add_parser("extract", help="write label == N as a binary mask")
    e.add_argument("labels", help="label volume (.mrc)")
    e.add_argument("label", type=int, help="label number to extract")
    e.add_argument("out", help="output .mrc (binary uint8)")
    e.set_defaults(fn=cmd_extract)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
