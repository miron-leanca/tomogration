#!/usr/bin/env python
"""tomoview - tomogram + any number of segmentations/scoremaps in napari."""
import argparse, os, re, sys
from pathlib import Path
import numpy as np
import mrcfile
import napari

COLOURS = ["red", "cyan", "yellow", "magenta", "green", "orange", "blue"]


def read_mrc(path):
    with mrcfile.open(path, permissive=True) as mrc:
        data = np.asarray(mrc.data)
        try:
            voxel = float(mrc.voxel_size.x)
        except Exception:
            voxel = 0.0
    return data, voxel


def is_label_like(data):
    if data.dtype.kind in "iub":
        return True
    sample = data.ravel()[:: max(1, data.size // 200000)]
    uniq = np.unique(sample)
    return uniq.size <= 32 and np.allclose(uniq, np.round(uniq))




def display_name(path, all_paths=()):
    """A layer name that says what DISTINGUISHES this file from the others.

    Raw filenames are the wrong labels for comparison: '12.56Apx' is identical
    in every layer so it carries no information, the series stem repeats when
    every layer is the same tomogram, and the size cutoff is not in the name at
    all — it is the parent directory (cc1000/, cc10000/). So: drop what is
    shared, and lift what is not."""
    p = Path(path)
    base = re.sub(r"\.mrc$", "", p.name, flags=re.I)
    base = re.sub(r"_\d+(?:\.\d+)?Apx", "", base)      # same in every layer
    bits = []
    m = re.search(r"_threshold_(-?[\d.]+)", base)
    if m:
        bits.append(f"thr={m.group(1)}")
        base = base.replace(m.group(0), "")
    parent = p.parent.name
    mc = re.match(r"^cc(\d+)$", parent)
    if mc:
        bits.append(f"cut={mc.group(1)}")
    ms = re.match(r"^s([\d.]+)_f([\d.]+)$", parent)
    if ms:
        bits.append(f"deconv s{ms.group(1)}/f{ms.group(2)}")
    # A series stem shared by EVERY layer is noise; keep it when they differ.
    stems = {re.match(r"^([A-Za-z]+\d+)", re.sub(r"\.mrc$", "", Path(q).name,
                                                 flags=re.I))
             for q in (all_paths or [])}
    stems = {m2.group(1) for m2 in stems if m2}
    if len(stems) == 1:
        base = re.sub(r"^" + re.escape(next(iter(stems))) + r"_?", "", base)
    # The model checkpoint is in every segmentation's name and never differs.
    base = re.sub(r"_?MemBrain_seg_\S*?\.ckpt", "", base)
    base = re.sub(r"_scores_components$", "_components", base)
    base = re.sub(r"_+", " ", base).strip() or p.stem
    # Do not say "deconv" twice because it is in both the name and the folder.
    bits = [b for b in bits
            if not any(w and w in base.split() for w in b.split()[:1])]
    # A FIXED name written into a per-tomogram folder (fit_1.mrc, and every
    # other name a batch tool repeats per tomogram) says nothing about which
    # tomogram it came from — open two and napari lists two layers called
    # "fit 1". Take the series from the folder, but only when the layers
    # actually span more than one, so a single-tomogram view stays clean.
    mine = _folder_series(p)
    spanned = {s2 for s2 in (_folder_series(Path(q)) for q in (all_paths or []))
               if s2}
    if mine and len(spanned) > 1 and not re.search(r"_\d+(?:\.\d+)?Apx", p.name):
        base = f"{mine} {base}".strip()
    return f"{base}  {' '.join(bits)}".strip()


def _folder_series(path):
    """The series a file belongs to: from its own name, else from its folder."""
    for name in (path.name, path.parent.name):
        m = re.match(r"^(.+?)_\d+(?:[.p]\d+)?Apx", name)
        if m:
            return m.group(1)
    return ""


def looks_like_scoremap(name, data):
    """A continuous MemBrain score map, as opposed to a tomogram or labels."""
    return ("score" in name.lower() and not is_label_like(data))


# Everything a membrane tool COMPUTES from a tomogram carries one of these in
# its name. A file with none of them is a reconstruction.
_DERIVED_MARKS = ("_scores", "_segmented", "_threshold_", "_components",
                  "_split", "_mask", "_labels", "_seg.")


def is_base_tomogram(path):
    """A greyscale RECONSTRUCTION, rather than something computed from one.

    Only the FIRST file used to be treated as a tomogram; every later one fell
    through to the overlay branch. Open two tomograms to compare and the second
    was drawn as an additive coloured overlay at half opacity, clipped at the
    50th percentile — so half its histogram went to black and it appeared as
    speckle — and it was given a threshold slider, which means nothing for a
    reconstruction."""
    name = re.sub(r"\.mrc$", "", Path(path).name, flags=re.I).lower()
    return not (name.startswith("fit_")
                or any(m in name for m in _DERIVED_MARKS))


def threshold_dock(viewer, data, name, scale, coarse=2):
    """A live threshold on a score map: move the slider, see what survives.

    napari can only restyle a continuous ramp (contrast_limits), which shows
    the score distribution but NOT where a given cut would land — and the cut
    is the thing you are choosing. This paints the surviving voxels as a labels
    layer that follows the slider, so a threshold can be picked by eye and then
    typed into the 'Membrane: threshold sweep' card.

    The mask is computed on a `coarse`-fold decimated copy so the slider stays
    smooth (a full 386x512x512 pass per tick is ~0.3 s and feels broken);
    napari scales the layer back up, so the contour is in the right place.
    Tick 'full resolution' for the final look before committing a number.
    """
    try:
        from magicgui import magicgui
    except ImportError:
        print("  (magicgui not available — no threshold slider)")
        return
    sub = data[::coarse, ::coarse, ::coarse]
    lo, hi = (float(v) for v in np.percentile(sub, [0.5, 99.9]))
    start = float(np.percentile(sub, 99.0))
    cut_name = f"{name} [cut]"

    def paint(value, full):
        src = data if full else sub
        mask = (src >= value).astype(np.uint8)
        step = 1 if full else coarse
        sc = tuple(s * step for s in scale)
        if cut_name in viewer.layers:
            layer = viewer.layers[cut_name]
            layer.data = mask
            layer.scale = sc
        else:
            viewer.add_labels(mask, name=cut_name, scale=sc, opacity=0.5)
        kept = float(mask.mean()) * 100.0
        print(f"  threshold {value:+.3f} keeps {kept:5.2f}% of voxels"
              f"{'' if full else '  (decimated preview)'}")

    @magicgui(auto_call=True,
              value={"widget_type": "FloatSlider", "min": lo, "max": hi,
                     "step": (hi - lo) / 200.0 or 0.01,
                     "label": "threshold"},
              full={"label": "full resolution"})
    def controls(value: float = start, full: bool = False):
        paint(value, full)

    viewer.window.add_dock_widget(controls, area="right",
                                  name=f"threshold: {name}")
    paint(start, False)
    print(f"  threshold slider on {name}: {lo:+.2f} to {hi:+.2f}, "
          f"start {start:+.2f}")
    print("  the printed value is what goes in the threshold sweep card.")


def main():
    p = argparse.ArgumentParser(description="Open a tomogram + overlays in napari.")
    p.add_argument("files", nargs="+", help="tomogram first, then any overlays")
    p.add_argument("--opacity", type=float, default=0.55)
    p.add_argument("--no-threshold", action="store_true",
                   help="do not attach the live threshold slider to score maps")
    args = p.parse_args()

    for f in args.files:
        if not os.path.isfile(f):
            sys.exit(f"not found: {f}")

    viewer = napari.Viewer(title=os.path.basename(args.files[0]))

    tomo, voxel = read_mrc(args.files[0])
    base_shape = tomo.shape
    lo, hi = np.percentile(tomo[::4, ::4, ::4], [1, 99])
    viewer.add_image(tomo, name=os.path.basename(args.files[0]),
                     colormap="gray", contrast_limits=[float(lo), float(hi)])
    print(f"tomogram {base_shape}  {voxel:.2f} A/px")

    for i, path in enumerate(args.files[1:]):
        data, _ = read_mrc(path)
        name = display_name(path, args.files)
        scale = tuple(b / s for b, s in zip(base_shape, data.shape))
        if not np.allclose(scale, 1.0):
            print(f"  {name}: {data.shape} -> scaled by {tuple(round(s,3) for s in scale)}")
        if is_base_tomogram(path):
            # A second tomogram is a TOMOGRAM, not an overlay: grey, its own
            # 1-99 contrast, no colour map, no threshold slider. Stacked on the
            # first one — toggle the eye icons to compare.
            t_lo, t_hi = np.percentile(data[::4, ::4, ::4], [1, 99])
            viewer.add_image(data, name=name, colormap="gray", scale=scale,
                             contrast_limits=[float(t_lo), float(t_hi)])
            if not np.allclose(scale, 1.0):
                print(f"  WARNING: {name} is {data.shape}, not {base_shape} — "
                      f"it has been STRETCHED onto the first tomogram's grid. "
                      f"Open them separately to compare them undistorted.")
        elif is_label_like(data):
            viewer.add_labels(data.astype(np.int32), name=name,
                              scale=scale, opacity=args.opacity)
        elif looks_like_scoremap(name, data) and not args.no_threshold:
            v_lo, v_hi = np.percentile(data[::4, ::4, ::4], [50, 99.9])
            viewer.add_image(data, name=name, scale=scale,
                             colormap=COLOURS[i % len(COLOURS)],
                             blending="additive", opacity=args.opacity,
                             contrast_limits=[float(v_lo), float(v_hi)])
            threshold_dock(viewer, data, name, scale)
        else:
            # No "[threshold slider]" in the name: that suffix went on every
            # continuous overlay whether or not anything could slide, which is
            # a label that lies. A dock is attached below, and THEN the name
            # says so.
            v_lo, v_hi = np.percentile(data[::4, ::4, ::4], [50, 99.9])
            viewer.add_image(data, name=name, scale=scale,
                             colormap=COLOURS[i % len(COLOURS)], blending="additive",
                             opacity=args.opacity,
                             contrast_limits=[float(v_lo), float(v_hi)])
            if not args.no_threshold:
                threshold_dock(viewer, data, name, scale)
        print(f"  + {name}")

    napari.run()


if __name__ == "__main__":
    main()
