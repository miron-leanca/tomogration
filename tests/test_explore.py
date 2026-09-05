"""The membrane parameter search (Part C) — the caching argument, tested.

THE POINT of these tests:

  * Segmentation must run once per (variant, tomogram), NOT once per cell.
    That is the whole reason the card exists in this shape: 72 cells over 12
    segmentations is ~50 minutes; 72 segmentations is five hours. A key that
    accidentally includes the threshold turns one into the other silently, and
    the only symptom is that it takes all afternoon.
  * Ranking uses virions accepted AND recall. Either alone rewards the wrong
    thing, so a cell that wins on one and loses badly on the other must not
    come top.

    python3 tests/test_explore.py
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


X = load("ml_explore_membrane")

VARIANTS = ["raw", "deconv", "isonet1", "isonet2"]
TOMOS = ["Position003", "Position045", "Position070"]
THRS = ["-2", "0", "2"]
CUTS = ["1000@12.56", "10000@12.56"]


def main():
    cells, work = X.plan_matrix(VARIANTS, TOMOS, THRS, CUTS)

    # ---- the shape of the matrix ------------------------------------------
    check("every combination becomes a cell (4x3x3x2 = 72)", len(cells) == 72)

    # ---- THE caching argument ---------------------------------------------
    check(f"segment runs once per (variant, tomogram): "
          f"{len(work['segment'])} not {len(cells)}",
          len(work["segment"]) == len(VARIANTS) * len(TOMOS) == 12)
    check("thresholds run once per (segment, threshold): 36",
          len(work["thresholds"]) == 12 * len(THRS) == 36)
    check("components run once per (threshold, cutoff): 72",
          len(work["components"]) == 36 * len(CUTS) == 72)
    check("fit runs once per components result: 72",
          len(work["fit"]) == 72)

    # The failure this guards against: a segment key that can see the
    # threshold or the cutoff. Change ONLY those and the segment keys must be
    # untouched, or every threshold costs 4 GPU-minutes a tomogram.
    _cells2, work2 = X.plan_matrix(VARIANTS, TOMOS, ["-3", "-1", "1", "5"],
                                   ["500@12.56"])
    check("changing thresholds and cutoffs does NOT change a single segment key",
          set(work["segment"]) == set(work2["segment"]))
    check("but it does change the threshold keys",
          set(work["thresholds"]) != set(work2["thresholds"]))

    # ...and the converse: a new variant or tomogram MUST re-segment.
    _c3, work3 = X.plan_matrix(VARIANTS + ["isonet2_bin8"], TOMOS, THRS, CUTS)
    check("a new variant adds exactly one segment run per tomogram",
          len(work3["segment"]) == len(work["segment"]) + len(TOMOS))
    _c4, work4 = X.plan_matrix(VARIANTS, TOMOS + ["Position077"], THRS, CUTS)
    check("a new tomogram adds one segment run per variant",
          len(work4["segment"]) == len(work["segment"]) + len(VARIANTS))

    # Segmentation parameters belong to the segment key, thresholds do not.
    _c5, work5 = X.plan_matrix(VARIANTS, TOMOS, THRS, CUTS,
                               seg_params={"ckpt": "other.ckpt"})
    check("a different model checkpoint re-segments everything",
          not (set(work["segment"]) & set(work5["segment"])))
    _c6, work6 = X.plan_matrix(VARIANTS, TOMOS, THRS, CUTS,
                               fit_params={"min_support": 0.7})
    check("a different gate setting re-fits but never re-segments",
          set(work6["segment"]) == set(work["segment"])
          and not (set(work6["fit"]) & set(work["fit"])))

    # Percentile vs absolute thresholds are different operating points.
    _c7, work7 = X.plan_matrix(VARIANTS, TOMOS, THRS, CUTS,
                               threshold_mode="percentile")
    check("percentile thresholds do not collide with absolute ones",
          not (set(work7["thresholds"]) & set(work["thresholds"]))
          and set(work7["segment"]) == set(work["segment"]))

    # Keys must be stable across runs (they are cache paths on disk).
    cells_again, _ = X.plan_matrix(VARIANTS, TOMOS, THRS, CUTS)
    check("keys are deterministic between runs",
          [c["keys"] for c in cells] == [c["keys"] for c in cells_again])
    check("and are filesystem-safe short hashes",
          all(k.isalnum() and len(k) == 12
              for c in cells for k in c["keys"].values()))

    # ---- the cache on disk -------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        k = cells[0]["keys"]["segment"]
        check("nothing is cached to begin with", not X.cached(td, "segment", k))
        X.mark_done(td, "segment", k, {"variant": "raw"})
        check("a completed stage is seen as cached", X.cached(td, "segment", k))
        check("and a different key is still not cached",
              not X.cached(td, "segment", cells[-1]["keys"]["segment"]))

    # ---- ranking -----------------------------------------------------------
    rows = [
        {"cell": "greedy", "virions_accepted": 3, "recall": 0.05},
        {"cell": "sloppy", "virions_accepted": 40, "recall": 0.99},
        {"cell": "balanced", "virions_accepted": 30, "recall": 0.80},
        {"cell": "empty", "virions_accepted": 0, "recall": 0.0},
    ]
    ranked = X.rank_rows(rows)
    check("a cell that accepts 3 virions with 5% recall does NOT win",
          ranked[0]["cell"] != "greedy")
    check("the cell strong on both wins", ranked[0]["cell"] == "sloppy")
    check("a cell strong on one axis only sinks below a balanced one",
          [r["cell"] for r in ranked].index("balanced")
          < [r["cell"] for r in ranked].index("greedy"))
    check("an empty cell scores zero",
          next(r for r in ranked if r["cell"] == "empty")["score"] == 0.0)
    check("scoring never crashes on a single row",
          X.rank_rows([{"virions_accepted": 5, "recall": 0.5}])[0]["score"] > 0)
    check("nor on missing columns",
          X.rank_rows([{"cell": "x"}])[0]["score"] == 0.0)

    # ---- the viewer hand-off ----------------------------------------------
    row = {"tomogram_path": "recon/Position003.mrc",
           "components_path": "cache/components/abc/comps.mrc",
           "fit_masks": ["fits/fit_3.mrc", "fits/fit_7.mrc"]}
    cmd = X.viewer_cmd(row, viewer="tomoview.py")
    check("viewer gets the tomogram FIRST (tomoview requires it)",
          cmd.split()[1] == "recon/Position003.mrc")
    check("then components, then every accepted fit",
          cmd.split()[2:] == ["cache/components/abc/comps.mrc",
                              "fits/fit_3.mrc", "fits/fit_7.mrc"])
    two = X.compare_cmd(row, dict(row, components_path="cache/components/def/comps.mrc",
                                  fit_masks=["fits2/fit_1.mrc"]),
                        viewer="tomoview.py")
    check("compare loads both fit sets against ONE tomogram",
          two.count("recon/Position003.mrc") == 1
          and "fits2/fit_1.mrc" in two and "fits/fit_3.mrc" in two)

    # ---- the table ---------------------------------------------------------
    table = X.format_table(ranked[:2])
    check("the table names both objectives", "virions" in table and "recall" in table)
    check("a missing value prints as a dash, not a crash",
          "—" in X.format_table([{"variant": "raw"}]))
    att = X.attrition_table([
        {"gate_attrition": {"support": 5, "extent": 2}},
        {"gate_attrition": {"support": 3, "in_volume": 1}}])
    check("gate attrition sums across cells, worst gate first",
          att[0] == ("support", 8))

    # ---- the executor: commands ------------------------------------------
    # membrain is not installed here, so the commands are what CAN be tested —
    # and they are the part that ruins a sweep silently by segmenting the wrong
    # folder or passing a cutoff in the wrong units.
    cell = dict(cells[0], variant_path="warp_tiltseries/reconstruction",
                tomogram_path="warp_tiltseries/reconstruction/Position003.mrc",
                cutoff_voxels=8000, polarity="dark",
                components_path="cache/components/xyz")
    opts = {"ckpt": "/models/MemBrain_seg_v10_beta.ckpt", "gpu": "0,1",
            "conda_env": "membrainseg"}
    seg = X.stage_command("segment", cell, "membrane/explore", opts)
    check("segment goes through the shipped wrapper, not membrain directly",
          "ml_membrain_segment_warp_auto.sh" in seg and "membrain segment" not in seg)
    check("segment reads the VARIANT folder and writes its own cache dir",
          "warp_tiltseries/reconstruction" in seg
          and "cache/segment/" in seg)
    check("segment is restricted to this cell's tomogram",
          "MB_TOMO_LIST=Position003" in seg)
    check("segment carries the checkpoint and GPUs",
          "MemBrain_seg_v10_beta.ckpt" in seg and "MB_GPU=0,1" in seg)
    # A path with a space must survive as ONE argument, or the wrapper silently
    # segments a folder that does not exist.
    spacey = X.stage_command("segment", dict(cell, variant_path="my recon/dir"),
                             "membrane/explore", opts)
    check("a path with a space stays one shell word",
          "'my recon/dir'" in spacey)

    thr = X.stage_command("thresholds", cell, "membrane/explore", opts)
    check("thresholds reads the SEGMENT cache, not the variant folder",
          "cache/segment/" in thr and "warp_tiltseries" not in thr)
    comp = X.stage_command("components", cell, "membrane/explore", opts)
    check("components reads the THRESHOLD cache",
          "cache/thresholds/" in comp and "cache/components/" in comp)
    check("components gets the cutoff in VOXELS, resolved for this variant",
          "MB_CC_THRES=8000" in comp)
    fit = X.stage_command("fit", cell, "membrane/explore", opts)
    check("fit is told the polarity explicitly",
          "--polarity dark" in fit)
    check("fit writes shell masks, so a row can be opened in the viewer",
          "--write-masks" in fit)
    check("every stage of a cell writes to a DIFFERENT cache dir",
          len({X.cache_dir("r", s, cell["keys"][s]) for s in X.STAGES}) == 4)

    # ---- resolving the volumes the stages actually wrote ------------------
    # Nothing can be built as f"{stem}.mrc": Warp tags the pixel size, membrain
    # appends its own suffixes, and the components wrapper nests each cutoff in
    # cc<voxels>/. Passing the cache DIRECTORY to the fitter is what made every
    # cell of the first real sweep die in mrcfile (2026-08-17).
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "k1" / "cc10000").mkdir(parents=True)
        (root / "k1" / "cc10000" /
         "Position003_12.56Apx_segmented_threshold_-2.0_components.mrc").touch()
        (root / "flat").mkdir()
        (root / "flat" / "Position003_12.56Apx.mrc").touch()
        (root / "amb" / "cc1000").mkdir(parents=True)
        (root / "amb" / "cc1000" / "Position_10_12.56Apx.mrc").touch()
        check("resolves a components volume nested in cc<voxels>/",
              Path(X.find_volume(root / "k1", "Position003")).name
              .startswith("Position003"))
        check("resolves a flat tomogram with its pixel-size tag",
              Path(X.find_volume(root / "flat", "Position003")).name
              == "Position003_12.56Apx.mrc")
        check("a stem that is not there resolves to NOTHING, never to whatever "
              "single file happens to be in the folder",
              X.find_volume(root / "k1", "Position999") == "")
        check("and Position_1 cannot claim Position_10's volume",
              X.find_volume(root / "amb", "Position_1") == "")

    # A DERIVED variant's volumes do not exist when the plan is built — the
    # derive step makes them. Resolving its tomogram path up front gave every
    # deconv cell an empty path and silently skipped all 18 (2026-08-17), so
    # the resolver must be called only once the files can be there.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        derived = root / "deconv" / "s1.0_f1.0"
        check("before the derive step, the variant folder resolves to nothing",
              X.find_volume(derived, "Position003") == "")
        derived.mkdir(parents=True)
        (derived / "Position003_12.56Apx_deconv.mrc").touch()
        check("and after it, the same call finds the volume",
              Path(X.find_volume(derived, "Position003")).name
              == "Position003_12.56Apx_deconv.mrc")

    # ---- the executor: metrics --------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "fits.json").write_text(json.dumps({
            "recall": 0.62,
            "gate_attrition": {"support": 4, "extent": 1},
            "virions": [
                {"support": 0.9, "diameter_nm": 100.0, "volume_nm3": 523598.0,
                 "residual_A": 30.0},
                {"support": 0.7, "diameter_nm": 120.0, "volume_nm3": 904778.0,
                 "residual_A": 50.0},
                {"support": 0.8, "diameter_nm": 110.0, "volume_nm3": 696910.0,
                 "residual_A": 40.0}]}))
        m = X.collect_metrics(d, "Found 91 components\nRelabeled to 37 components")
        check("metrics: virions accepted is counted from the fits",
              m["virions_accepted"] == 3)
        check("metrics: recall comes from the run that made the decisions",
              m["recall"] == 0.62)
        check("metrics: medians and means are taken over accepted virions",
              m["median_support"] == 0.8 and abs(m["mean_diameter_nm"] - 110.0) < 1e-9)
        check("metrics: gate attrition rides along",
              m["gate_attrition"]["support"] == 4)
        check("metrics: components found/kept parsed from the log",
              m.get("components_found") == 91 and m.get("components_kept") == 37)
    # A cell whose fit died must not take the sweep down with it.
    with tempfile.TemporaryDirectory() as td:
        m = X.collect_metrics(Path(td) / "never-ran")
        check("metrics: a missing fits.json is a row of None, not a crash",
              m["virions_accepted"] is None and m["gate_attrition"] == {})
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "fits.json").write_text("{ truncated")
        check("metrics: a half-written fits.json is survivable too",
              X.collect_metrics(Path(td))["recall"] is None)

    # ---- unique output names ----------------------------------------------
    # Two cells' components had IDENTICAL basenames in different folders, so
    # opening both in napari was guesswork. Names now carry what varies — and
    # ONLY what varies for that stage.
    cell = {"variant": "deconv", "tomogram": "Position003",
            "threshold": "-2", "cutoff": "1000@12.56"}
    other = dict(cell, cutoff="10000@12.56")
    check("a components name carries variant, threshold AND cutoff",
          X.canonical_name("Position003_12.56Apx_deconv_scores_threshold_-2.0"
                           "_components.mrc", cell, "components")
          == "Position003_deconv_thr-2_cut1000_components.mrc")
    check("so two cutoffs of the same cell no longer collide",
          X.canonical_name("x.mrc", cell, "components")
          != X.canonical_name("x.mrc", other, "components"))
    check("fit masks stop being fit_3.mrc in every folder",
          X.canonical_name("fit_3.mrc", cell, "fit")
          == "Position003_deconv_thr-2_cut1000_fit3.mrc")
    # A segmentation is shared by every threshold and cutoff of its variant.
    check("a segmentation is NOT tagged with a threshold it does not depend on",
          "thr" not in X.cell_tag(cell, "segment"))
    check("a threshold map is tagged with its threshold but not a cutoff",
          "thr" in X.cell_tag(cell, "thresholds")
          and "cut" not in X.cell_tag(cell, "thresholds"))
    check("renaming is idempotent — a canonical name is left alone",
          X.canonical_name("Position003_deconv_thr-2_cut1000_components.mrc",
                           cell, "components") == "")
    with tempfile.TemporaryDirectory() as td:
        seg = Path(td) / "seg"
        seg.mkdir()
        (seg / "Position003_12.56Apx_scores.mrc").touch()
        check("segment and thresholds outputs are NEVER renamed — the next "
              "tool globs for their suffixes",
              X.normalise_outputs("segment", cell, seg) == 0
              and (seg / "Position003_12.56Apx_scores.mrc").exists())
        comp = Path(td) / "comp" / "cc1000"
        comp.mkdir(parents=True)
        (comp / "Position003_12.56Apx_deconv_scores_threshold_-2.0"
                "_components.mrc").touch()
        X.normalise_outputs("components", cell, Path(td) / "comp")
        check("and the renamed components volume is still found by stem",
              Path(X.find_volume(Path(td) / "comp", "Position003")).name
              == "Position003_deconv_thr-2_cut1000_components.mrc")

    # ---- the analysis layer ------------------------------------------------
    A = [{"variant": "raw", "threshold": "-2", "cutoff": "1000@12.56",
          "tomogram": "P003", "virions_accepted": 10, "recall": 0.30,
          "mean_diameter_nm": 80.0},
         {"variant": "raw", "threshold": "0", "cutoff": "1000@12.56",
          "tomogram": "P003", "virions_accepted": 20, "recall": 0.40,
          "mean_diameter_nm": 82.0},
         {"variant": "isonet2", "threshold": "0", "cutoff": "1000@12.56",
          "tomogram": "P003", "virions_accepted": 50, "recall": 0.47,
          "mean_diameter_nm": 83.0},
         {"variant": "isonet2", "threshold": "0", "cutoff": "10000@12.56",
          "tomogram": "P003", "virions_accepted": 30, "recall": 0.45,
          "mean_diameter_nm": 83.0}]
    check("axis values come out in the order the sweep ran them",
          X.axis_values(A, "variant") == ["raw", "isonet2"])
    check("filtering to one variant keeps only its cells",
          len(X.filter_rows(A, {"variant": {"isonet2"}})) == 2)
    check("filtering on two axes at once intersects them",
          len(X.filter_rows(A, {"variant": {"isonet2"},
                                "cutoff": {"1000@12.56"}})) == 1)
    # Unticking everything on an axis must mean "no filter", not "no rows" —
    # a window that blanks itself when you clear a column is useless.
    check("an EMPTY tick-list means no filter, not zero rows",
          len(X.filter_rows(A, {"variant": set()})) == len(A))
    check("an axis not mentioned at all is unfiltered",
          len(X.filter_rows(A, {})) == len(A))
    stats = X.group_stats(A, "variant", "virions_accepted")
    check("grouping averages across the cells of each variant",
          dict((v, m) for v, m, _n in stats) == {"raw": 15.0, "isonet2": 40.0})
    check("and reports how many cells went into each mean",
          dict((v, n) for v, _m, n in stats) == {"raw": 2, "isonet2": 2})
    check("a metric no cell has averages to None, not zero",
          X.group_stats(A, "variant", "median_support")[0][1] is None)
    check("the trade-off scatter carries the row, so a point can be opened",
          X.tradeoff_points(A)[0][3]["variant"] == "raw")
    check("cells with no numbers are left out of the scatter",
          len(X.tradeoff_points(A + [{"variant": "x"}])) == len(A))
    ratios = X.sphere_vs_ellipsoid([{"variant": "raw", "tomogram": "P003",
        "virions": [{"label": 3, "volume_nm3": 100.0,
                     "ellipsoid_volume_nm3": 150.0}]}])
    check("sphere vs ellipsoid reports the inflation ratio per virion",
          abs(ratios[0][3] - 1.5) < 1e-9)
    check("a virion without an ellipsoid fit is skipped, not counted as 1.0",
          X.sphere_vs_ellipsoid([{"virions": [{"volume_nm3": 100.0}]}]) == [])

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
