#!/usr/bin/env python3
"""Look inside a membrain-pick mesh container, and render it over its tomogram.

    ml_mesh_inspect.py <mesh.h5>                  # what is in it
    ml_mesh_inspect.py <mesh.h5> --tomogram t.mrc # ...and show it in napari

The mesh step is the one whose output nobody can eyeball: it writes .h5
containers meant for surforama, so "did that work, and what is it?" has no
answer short of opening another tool. This reads the container directly and
says what it holds — how many vertices, how big the surface is in nm, and what
density was sampled along the normals — then optionally draws it as a napari
Surface layer on top of the tomogram it came from.

Deliberately schema-agnostic: membrain-pick's layout has changed between
versions, so arrays are identified by SHAPE (an (N,3) float array is vertices;
an (M,3) integer array indexing them is faces) rather than by a hardcoded key
that would silently find nothing after an update.

Needs h5py + numpy; run it from membrainpick (which has both) — the same env
the mesh step itself uses.
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def walk(h5obj, prefix=""):
    """[(path, dataset)] for every dataset in the file, at any depth."""
    import h5py                            # noqa: PLC0415
    out = []
    for key, item in h5obj.items():
        path = f"{prefix}/{key}"
        if isinstance(item, h5py.Group):
            out += walk(item, path)
        else:
            out.append((path, item))
    return out


def classify(arrays):
    """Identify vertices / faces / per-vertex values by SHAPE, not by name.

    Returns {"vertices": path|None, "faces": path|None, "values": [paths]}.
    A hardcoded key would break silently the next time membrain-pick renames
    something; a shape cannot be renamed."""
    verts = faces = None
    values, n_vert = [], 0
    for path, shape, kind in arrays:
        if len(shape) == 2 and shape[1] == 3 and kind == "f" and shape[0] > n_vert:
            verts, n_vert = path, shape[0]
    for path, shape, kind in arrays:
        if len(shape) == 2 and shape[1] == 3 and kind in "iu":
            faces = path
    for path, shape, kind in arrays:
        if path in (verts, faces):
            continue
        if (len(shape) == 1 and shape[0] == n_vert) or \
           (len(shape) == 2 and shape[0] == n_vert and shape[1] != 3):
            values.append(path)
    return {"vertices": verts, "faces": faces, "values": values}


def summarise(verts, angpix):
    """Extent and scale of a surface, in voxels and nm."""
    lo, hi = verts.min(0), verts.max(0)
    span_px = hi - lo
    return {"n_vertices": int(len(verts)),
            "min_px": lo.tolist(), "max_px": hi.tolist(),
            "span_px": span_px.tolist(),
            "span_nm": (span_px * float(angpix) / 10.0).tolist()}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh")
    ap.add_argument("--tomogram", default="",
                    help="render the surface over this volume in napari")
    ap.add_argument("--angpix", type=float, default=12.56)
    a = ap.parse_args(argv)

    try:
        import h5py                        # noqa: PLC0415
    except ImportError:
        sys.exit("h5py is not in this env — run this from membrainpick.")
    p = Path(a.mesh)
    if not p.is_file():
        sys.exit(f"not a file: {p}")

    with h5py.File(str(p), "r") as f:
        datasets = walk(f)
        print(f"{p.name}\n{'-' * len(p.name)}")
        arrays = []
        for path, ds in datasets:
            print(f"  {path:<34} {str(ds.shape):<16} {ds.dtype}")
            arrays.append((path, tuple(ds.shape), ds.dtype.kind))
        found = classify(arrays)
        if not found["vertices"]:
            sys.exit("\nNo (N,3) float array — this does not look like a mesh.")
        verts = np.asarray(f[found["vertices"]][()], dtype="f4")
        faces = (np.asarray(f[found["faces"]][()], dtype="i8")
                 if found["faces"] else None)
        info = summarise(verts, a.angpix)
        print(f"\nvertices : {info['n_vertices']} (from {found['vertices']})")
        if faces is not None:
            print(f"faces    : {len(faces)} triangles (from {found['faces']})")
        print(f"extent   : {np.round(info['span_px'], 1).tolist()} voxels"
              f"  =  {np.round(info['span_nm'], 1).tolist()} nm")
        print(f"           a {a.angpix:g} A/px volume, so a virion ~80 nm across "
              f"spans ~{80 * 10 / a.angpix:.0f} voxels")
        for v in found["values"]:
            arr = np.asarray(f[v][()], dtype="f4").ravel()
            print(f"sampled  : {v:<28} min {arr.min():+.3g}  "
                  f"median {np.median(arr):+.3g}  max {arr.max():+.3g}")
        print("\nThe per-vertex values are the tomogram sampled ALONG EACH "
              "NORMAL — that is what the mesh is for: it turns 'where is the "
              "membrane' into 'what does the density look like either side of "
              "it', which is what picking on a surface needs.")

        if a.tomogram:
            import mrcfile                 # noqa: PLC0415
            import napari                  # noqa: PLC0415
            with mrcfile.open(a.tomogram, permissive=True) as m:
                tomo = np.asarray(m.data)
            v = napari.Viewer(title=p.name)
            lo, hi = np.percentile(tomo[::4, ::4, ::4], [1, 99])
            v.add_image(tomo, name=Path(a.tomogram).name, colormap="gray",
                        contrast_limits=[float(lo), float(hi)])
            if faces is not None:
                v.add_surface((verts, faces), name=f"mesh: {p.stem}",
                              opacity=0.7, colormap="magma")
            else:
                v.add_points(verts, name=f"vertices: {p.stem}", size=2)
            napari.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
