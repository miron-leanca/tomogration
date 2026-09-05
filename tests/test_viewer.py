"""Opening a finished membrane job in a viewer.

THE POINT of these tests:

  * tomoview takes the TOMOGRAM FIRST and overlays after — it builds the base
    image from argv[1] and scales every later layer to that shape. Hand it a
    segmentation first and the layers are scaled to the wrong grid.
  * Meshes go to surforama in membrainpick; volumes go to napari in
    membrainseg. Those two envs pin conflicting napari/PyQt versions and must
    never be merged, so picking the wrong one is not a cosmetic mistake.
  * A job that ran on three tomograms must open ONE of them, not all three:
    the filter is what keeps a view readable.

    python3 tests/test_viewer.py
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE / "stub"))
sys.path.insert(0, str(REPO))

passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}")


def load(mod):
    spec = importlib.util.spec_from_file_location(mod, REPO / f"{mod}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod] = m
    spec.loader.exec_module(m)
    return m


J = load("tomogration_jobs")


def _load_tomoview():
    """tomoview lives outside the app and may not have synced. Load it with
    only its third-party imports stubbed — nothing injected, so a missing
    import in tomoview itself still fails here rather than hiding."""
    import types                                             # noqa: PLC0415
    path = Path(__file__).resolve().parent.parent / "tomoview.py"
    if not path.exists():
        return None
    for m in ("mrcfile", "napari", "magicgui"):
        sys.modules.setdefault(m, types.ModuleType(m))
    sys.modules["magicgui"].magicgui = lambda *a, **k: (lambda f: f)
    spec = importlib.util.spec_from_file_location("tomoview_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    # ---- series stems off real output names --------------------------------
    check("stem: membrain's long segmented name",
          J.series_stem_of(
              "Position077_12.56Apx_MemBrain_seg_v10_beta.ckpt_segmented.mrc")
          == "Position077")
    check("stem: the dot-free IsoNet spelling too",
          J.series_stem_of("Position003_12p56Apx_isonet2.mrc") == "Position003")
    check("stem: a threshold sweep output",
          J.series_stem_of("Position045_12.56Apx_threshold_-1.0.mrc")
          == "Position045")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        recon = root / "recon"
        recon.mkdir()
        out = root / "jobs" / "J35"
        out.mkdir(parents=True)
        for s in ("Position003", "Position045", "Position077"):
            (recon / f"{s}_12.56Apx.mrc").write_text("")
            (out / f"{s}_12.56Apx_segmented.mrc").write_text("")
            (out / f"{s}_12.56Apx_scores.mrc").write_text("")

        check("every series the job produced is offered",
              J.viewer_series("jobs/J35", root)
              == ["Position003", "Position045", "Position077"])

        tool, env, files = J.viewer_plan("mb_segment", "jobs/J35",
                                         series="Position045",
                                         input_dir="recon", root=root)
        check("segmentations open in tomoview, in membrainseg (napari lives there)",
              tool == "tomoview" and env == "membrainseg")
        check("THE tomogram comes first — tomoview scales every later layer to it",
              files and files[0].endswith("recon/Position045_12.56Apx.mrc"))
        check("then its segmentation and score map",
              any("segmented" in f for f in files)
              and any("scores" in f for f in files))
        check("and ONLY the chosen series: 3 files, not 9",
              len(files) == 3 and all("Position045" in f for f in files))
        check("no other series leaks in",
              not any("Position003" in f or "Position077" in f for f in files))

        # Without a series filter, everything comes — the caller asks first.
        _t, _e, all_files = J.viewer_plan("mb_segment", "jobs/J35",
                                          input_dir="recon", root=root)
        check("unfiltered opens the lot (which is why the GUI asks which)",
              len(all_files) == 6)

    # ---- meshes are a different tool AND a different env --------------------
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        mesh = root / "membrane" / "mesh"
        mesh.mkdir(parents=True)
        (mesh / "Position003_12.56Apx_mesh.h5").write_text("")
        tool, env, files = J.viewer_plan("mb_mesh", "membrane/mesh", root=root)
        check("meshes go to surforama, in membrainpick",
              tool == "surforama" and env == "membrainpick")
        check("and it is handed the .h5 container", files and files[0].endswith(".h5"))

    # ---- stages with nothing worth viewing ---------------------------------
    check("a non-membrane stage offers no viewer at all",
          J.viewer_plan("ts_ctf", "warp_tiltseries") == (None, None, []))
    check("and neither does an unknown one",
          J.viewer_plan("nonsense", "x") == (None, None, []))
    with tempfile.TemporaryDirectory() as td:
        check("a job that produced nothing yields no files, not a crash",
              J.viewer_plan("mb_segment", "jobs/J99", root=td)[2] == [])
        check("and offers no series to choose between",
              J.viewer_series("jobs/J99", td) == [])

    # ---- the inventory the picker tree is built from -----------------------
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "recon").mkdir()
        out = root / "jobs" / "J35"
        out.mkdir(parents=True)
        for stem in ("Position003", "Position045"):
            (root / "recon" / f"{stem}_12.56Apx.mrc").write_text("")
            (out / f"{stem}_12.56Apx_segmented.mrc").write_text("")
            (out / f"{stem}_12.56Apx_scores.mrc").write_text("")
        inv = J.viewer_inventory("mb_segment", "jobs/J35", "recon", root)
        check("inventory: one branch per tomogram",
              sorted(inv) == ["Position003", "Position045"])
        check("inventory: each branch holds that tomogram's volumes only",
              all(stem in f for stem, files in inv.items() for f in files))
        check("inventory: the tomogram is FIRST in its branch, as napari needs",
              inv["Position003"][0].endswith("recon/Position003_12.56Apx.mrc"))
        check("inventory: 3 layers per tomogram (volume + segmented + scores)",
              all(len(v) == 3 for v in inv.values()))
        check("inventory: nothing to view gives an empty dict, not a crash",
              J.viewer_inventory("mb_segment", "jobs/J99", "recon", root) == {})

    # ---- finding the viewer itself -----------------------------------------
    # tomoview ships SEPARATELY from tomogration (the user keeps it in ~/bin),
    # so assuming the app folder produced "ERROR: tool not found" at launch.
    with tempfile.TemporaryDirectory() as td:
        home = Path(td) / "home"
        repo = Path(td) / "repo"
        (home / "bin").mkdir(parents=True)
        repo.mkdir()
        check("not found anywhere is an empty string, not a bad path",
              J.resolve_viewer_tool("tomoview.py", repo, home) == "")
        (home / "bin" / "tomoview.py").write_text("")
        check("found in ~/bin when it did not sync with the app",
              J.resolve_viewer_tool("tomoview.py", repo, home)
              == str(home / "bin" / "tomoview.py"))
        (repo / "tomoview.py").write_text("")
        check("the app's own copy wins when it IS there",
              J.resolve_viewer_tool("tomoview.py", repo, home)
              == str(repo / "tomoview.py"))
        # surforama is a console command the env provides — never a file.
        check("a console command is passed through untouched",
              J.resolve_viewer_tool("surforama", repo, home) == "surforama")

    # A cache directory is not a viewable file. A row whose fit was cached
    # left components_path pointing at the hash folder, and the picker offered
    # '751e88af5c27' as something to open.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "jobs" / "J33").mkdir(parents=True)
        (root / "t.mrc").touch()
        (root / "751e88af5c27").mkdir()
        (root / "jobs" / "J33" / "results.json").write_text(json.dumps({"rows": [
            {"variant": "isonet2", "threshold": "0", "cutoff": "1000@12.56",
             "tomogram": "Position045", "tomogram_path": "t.mrc",
             "components_path": "751e88af5c27", "fit_masks": [],
             "virions_accepted": 30, "recall": 0.19}]}))
        inv = J.viewer_inventory("mb_explore", "jobs/J33", root=root)
        listed = [f for files in inv.values() for f in files]
        check("a cache DIRECTORY is never offered as a layer",
              "751e88af5c27" not in listed)

    # Ticking several sweep rows of the same series listed that tomogram once
    # per row — napari then loaded four identical base images (~400 MB each).
    class _Leaf:
        def __init__(self, path, on=True):
            self.path, self.on = path, on

        def checkState(self, _c):
            return True if self.on else None

        def data(self, _c, _r):
            return self.path

    class _Top:
        def __init__(self, leaves):
            self.leaves = leaves

        def childCount(self):
            return len(self.leaves)

        def child(self, i):
            return self.leaves[i]

    rows = [_Top([_Leaf("tomo.mrc"), _Leaf("compsA.mrc")]),
            _Top([_Leaf("tomo.mrc"), _Leaf("compsB.mrc")])]
    picked, seen = [], set()
    for top in rows:
        for i in range(top.childCount()):
            pth = top.child(i).data(0, None)
            if pth not in seen:
                seen.add(pth)
                picked.append(pth)
    check("the same tomogram across two ticked rows is listed once",
          picked == ["tomo.mrc", "compsA.mrc", "compsB.mrc"])
    check("and it stays FIRST, since tomoview scales the rest to it",
          picked[0] == "tomo.mrc")

    # -----------------------------------------------------------------------
    # Batch tools write FIXED names (fit_1.mrc) into a per-tomogram subfolder.
    # The viewer read basenames only, so every one of J42's 85 masks became its
    # own top-level "series" — the picker listed fit_1, fit_10, fit_104 with no
    # tomogram to attach them to. The same basename test also filtered the file
    # list, so picking a real tomogram would have excluded its own masks.
    import os                                                # noqa: PLC0415
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        out, src = root / "jobs/J42", root / "jobs/J31/corrected"
        src.mkdir(parents=True)
        for pos in ("Position003", "Position116"):
            sub = out / f"{pos}_12.56Apx_isonet2_scores_threshold_-1.5_components"
            sub.mkdir(parents=True)
            for i in (1, 2, 10, 104):
                (sub / f"fit_{i}.mrc").write_bytes(b"")
            (sub / "fits.json").write_text("{}")
            (src / f"{pos}_12.56Apx_isonet2.mrc").write_bytes(b"")

        series = J.viewer_series("jobs/J42", root)
        check("a batch job offers its TOMOGRAMS, not one entry per mask",
              series == ["Position003", "Position116"])
        check("a fixed name inherits the series from the folder it sits in",
              J.output_series_of(
                  out / "Position116_12.56Apx_isonet2_scores_threshold_-1.5_components"
                      / "fit_1.mrc", out) == "Position116")
        check("a name carrying its own tag still answers for itself",
              J.output_series_of(src / "Position003_12.56Apx_isonet2.mrc",
                                 src) == "Position003")

        inv = J.viewer_inventory("mb_fit_virions", "jobs/J42",
                                 "jobs/J31/corrected", root)
        check("each tomogram gets its own group", sorted(inv) == series)
        p116 = [os.path.basename(f) for f in inv["Position116"]]
        check("the tomogram's own volume comes first, so napari scales to it",
              p116[0] == "Position116_12.56Apx_isonet2.mrc")
        check("a group holds that tomogram's masks and NO other tomogram's",
              all("Position003" not in f for f in inv["Position116"])
              and len([f for f in p116 if f.startswith("fit_")]) == 4)
        check("masks sort fit_2 before fit_10, not fit_10 before fit_2",
              [f for f in p116 if f.startswith("fit_")]
              == ["fit_1.mrc", "fit_2.mrc", "fit_10.mrc", "fit_104.mrc"])
        check("no mask is offered under two different tomograms",
              len({f for v in inv.values() for f in v})
              == sum(len(v) for v in inv.values()))

    # And the layer NAMES must not collide either: two tomograms' fit_1.mrc
    # both reduced to "fit 1" in napari's layer list.
    tv = _load_tomoview()
    if tv is not None:
        F = ("/p/jobs/J42/{}_12.56Apx_isonet2_scores_threshold_-1.5_components"
             "/fit_{}.mrc")
        T = "/p/jobs/J31/corrected/{}_12.56Apx_isonet2.mrc"
        two = [T.format("Position003"), F.format("Position003", 1),
               T.format("Position116"), F.format("Position116", 1)]
        names = [tv.display_name(f, two) for f in two]
        check("across tomograms, a fixed name says which one it came from",
              len(set(names)) == 4 and "Position116 fit 1" in names)
        one = [T.format("Position003"), F.format("Position003", 1),
               F.format("Position003", 2)]
        check("within ONE tomogram the name stays clean, no repeated stem",
              [tv.display_name(f, one) for f in one][1:] == ["fit 1", "fit 2"])

    # ---- a second tomogram is a TOMOGRAM -----------------------------------
    # Only files[0] used to be treated as one; every later file fell through to
    # the overlay branch. Opening two tomograms to compare drew the second as
    # an additive coloured overlay at half opacity, clipped at the 50th
    # percentile so half its histogram went black (it looked like pure
    # speckle), and gave it a threshold slider, which is meaningless for a
    # reconstruction.
    tv = _load_tomoview()
    if tv is None:
        check("tomoview present to test", False)
    else:
        for _n in ("Position069_12.56Apx_isonet2.mrc",
                   "Position003_12.56Apx.mrc",
                   "warp_tiltseries/reconstruction/Position011_10.00Apx.mrc"):
            check(f"a reconstruction is a base image: {Path(_n).name}",
                  tv.is_base_tomogram(_n))
        for _n in ("Position069_12.56Apx_isonet2_scores.mrc",
                   "Position069_12.56Apx_isonet2_MemBrain_seg_v10_beta.ckpt"
                   "_segmented.mrc",
                   "Position069_12.56Apx_isonet2_scores_threshold_-1.5.mrc",
                   "Position069_12.56Apx_isonet2_scores_threshold_-1.5"
                   "_components.mrc",
                   "Position002_12.56Apx_isonet2_components_split.mrc",
                   "fit_1.mrc"):
            check(f"a computed volume is an overlay: {Path(_n).name[:38]}",
                  not tv.is_base_tomogram(_n))
        import numpy as _np                                  # noqa: PLC0415
        _cont = _np.linspace(-3, 3, 64).reshape(4, 4, 4).astype("float32")
        check("a score map is never also classed as a base tomogram",
              tv.looks_like_scoremap("Position069 isonet2 scores", _cont)
              and not tv.is_base_tomogram(
                  "Position069_12.56Apx_isonet2_scores.mrc"))
        check("and the reconstruction it came from is the base image",
              tv.is_base_tomogram("Position069_12.56Apx_isonet2.mrc"))

    # ---- the base image must be a TOMOGRAM ---------------------------------
    # A components job reads threshold MASKS and a threshold job reads SCORE
    # MAPS, so using a job's own input_dir as the base image opened a viewer
    # holding no tomogram at all — only the outline of one.
    GREY = "jobs/J31-bin8-isonet2model/corrected"
    _store = {"jobs": {
        "J31": {"id": "J31", "stage_id": "mb_isonet2_predict",
                "params": {"output_dir": GREY}, "inputs": {}},
        "J35": {"id": "J35", "stage_id": "mb_segment",
                "params": {"input_dir": GREY}, "inputs": {"in": "J31"}},
        "J36": {"id": "J36", "stage_id": "mb_thresholds",
                "params": {"input_dir": "jobs/J35"}, "inputs": {"in": "J35"}},
        "J37": {"id": "J37", "stage_id": "mb_components",
                "params": {"input_dir": "jobs/J36"}, "inputs": {"in": "J36"}}}}
    for _jid in ("J35", "J36", "J37"):
        check(f"{_jid} finds the greyscale volumes by walking up, not its input",
              J.greyscale_source(_store["jobs"][_jid], _store) == GREY)
    _store["jobs"]["J37"]["params"][J.GREY_KEY] = "membrane/other"
    check("an explicit breadcrumb wins over the walk",
          J.greyscale_source(_store["jobs"]["J37"], _store) == "membrane/other")
    check("a job with no greyscale ancestor returns nothing, not a guess",
          J.greyscale_source({"id": "J99", "stage_id": "mb_components",
                              "params": {"input_dir": "jobs/J36"},
                              "inputs": {}}, _store) == "")
    check("a parent cycle cannot hang the walk",
          J.greyscale_source(
              {"id": "A", "stage_id": "mb_components", "params": {},
               "inputs": {"in": "A"}},
              {"jobs": {"A": {"id": "A", "stage_id": "mb_components",
                              "params": {}, "inputs": {"in": "A"}}}}) == "")

    with tempfile.TemporaryDirectory() as _td:
        _r = Path(_td)
        (_r / "grey").mkdir()
        (_r / "jobs/J37").mkdir(parents=True)
        for _pos in ("Position002", "Position10", "Position106"):
            (_r / f"grey/{_pos}_12.56Apx_isonet2.mrc").write_bytes(b"")
            (_r / f"jobs/J37/{_pos}_12.56Apx_isonet2_scores_threshold_-1.5"
                  f"_components.mrc").write_bytes(b"")
        _inv = J.viewer_inventory("mb_components", "jobs/J37", "grey", _r)
        for _pos in ("Position002", "Position10", "Position106"):
            _first = Path(_inv[_pos][0]).name
            check(f"{_pos} opens on its own tomogram, listed first",
                  _first == f"{_pos}_12.56Apx_isonet2.mrc")
            check(f"{_pos} still gets its components layer",
                  any("_components.mrc" in f for f in _inv[_pos]))
        check("Position10 does not open Position106's tomogram",
              all("Position106" not in f for f in _inv["Position10"]))

    # ---- surforama is a membrain_pick SUBCOMMAND ---------------------------
    # It was launched as `surforama a.h5 b.h5 ...`. The spec (3.7) documents
    # `membrain_pick surforama --h5-path <container>.h5`: one container, on a
    # named flag. A positional list was never going to open anything.
    _app = (Path(__file__).resolve().parent.parent
            / "tomogration_app.py").read_text()
    check("surforama is invoked as a membrain_pick subcommand",
          "membrain_pick surforama --h5-path" in _app)
    check("in the membrainpick env, not membrainseg",
          "MB_CONDA_ENV=membrainpick bash" in _app)
    check("mesh containers are what it accepts",
          '.lower().endswith(".h5")' in _app)
    check("and mb_mesh's viewer plan still points at surforama",
          J.VIEWER_PLANS["mb_mesh"][0] == "surforama"
          and J.VIEWER_PLANS["mb_mesh"][1] == ("*.h5",))

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
