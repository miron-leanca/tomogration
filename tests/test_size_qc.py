"""Virion size QC (mb_size_qc) — the outlier judgements a viewer would act on.

THE POINT, on synthetic label volumes where the truth is known:

  * Verdicts are POPULATION-RELATIVE (nm³ via angpix), so the same thresholds
    work at any binning; a component ~k× the virion volume reports k.
  * Nothing is deleted — every component appears in the report; outliers_only
    keeps the ORIGINAL label values (napari's picker must read the same
    numbers as the csv) and never writes int32 (the J44 MRC-mode lesson).
  * A missing population is an ERROR, not a guess — ordering is the design.

Runs the real tool end-to-end through the mrcfile stub.

    python3 tests/test_size_qc.py
"""
import importlib.util
import json
import subprocess
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


try:
    import numpy as np
except ImportError:                      # pragma: no cover
    print("numpy not installed — skipping size QC tests")
    print("0 passed, 0 failed")
    sys.exit(0)

import mrcfile                           # the stub


def load(mod):
    spec = importlib.util.spec_from_file_location(mod, REPO / f"{mod}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod] = m
    spec.loader.exec_module(m)
    return m


def run_tool(*args):
    env = dict(PYTHONPATH=str(HERE / "stub"), PATH="/usr/bin:/bin")
    return subprocess.run(
        [sys.executable, str(REPO / "ml_virion_qc.py"), *args],
        capture_output=True, text=True, env=env)


def main():
    vx = 12.56                                    # Å/px
    nm3 = (vx / 10.0) ** 3                        # nm³ per voxel
    # A "virion" of 1000 voxels; population volume set to exactly that.
    v_pop = 1000 * nm3

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        pop = root / "population.json"
        pop.write_text(json.dumps({"radius_A": 700.0,
                                   "whole_virion_nm3": v_pop,
                                   "n_virions": 12,
                                   "tomograms": ["Position003"]}))
        # labels: 1 = normal (1000 vox), 2 = fused pair (2100 vox),
        # 3 = debris (80 vox), 4 = big-ish but ok (1500 vox).
        d = np.zeros((40, 40, 40), dtype=np.float32)
        d.reshape(-1)[:1000] = 1
        d.reshape(-1)[1000:3100] = 2
        d.reshape(-1)[3100:3180] = 3
        d.reshape(-1)[3180:4680] = 4
        comp = root / "Position003_components.mrc"
        with mrcfile.new(str(comp)) as m:
            m.set_data(d)
            m.voxel_size = vx
        out = root / "qc"

        r = run_tool("--components", str(comp), "--population", str(pop),
                     "--angpix", str(vx), "--out-folder", str(out))
        check("tool exits 0", r.returncode == 0)
        rep = json.loads((out / "size_qc.json").read_text())
        by = {row["label"]: row for row in rep["rows"]}
        check("all four components reported (nothing dropped)", len(by) == 4)
        check("a one-virion component is ok", by[1]["verdict"] == "ok")
        check("a fused pair is too_big with k=2",
              by[2]["verdict"] == "too_big" and by[2]["k_estimate"] == 2)
        check("debris is too_small", by[3]["verdict"] == "too_small")
        check("1.5x the population volume stays ok at big_frac 1.7",
              by[4]["verdict"] == "ok")
        check("volumes are physical nm³",
              abs(by[1]["volume_nm3"] - v_pop) < 0.5)
        check("cluster headcount is summed",
              rep["extra_virions_in_clusters"] == 2)

        with mrcfile.open(str(out / "outliers_only.mrc")) as m:
            ov = np.asarray(m.data)
        check("outliers_only keeps ORIGINAL label values",
              set(np.unique(ov)) == {0, 2, 3})
        check("outliers_only is never int32 (MRC has no such mode)",
              ov.dtype != np.int32)
        check("inspection shortlist names the cluster first",
              (out / "inspect.txt").read_text().splitlines()[0]
              .startswith("2\ttoo_big"))
        csv_text = (out / "size_qc.csv").read_text()
        check("csv carries the verdict column", "too_big" in csv_text)

        # Missing population = hard error, not a guess.
        r2 = run_tool("--components", str(comp), "--population",
                      str(root / "nope.json"), "--angpix", str(vx),
                      "--out-folder", str(out))
        check("missing population refuses with guidance",
              r2.returncode != 0 and "population" in
              (r2.stderr + r2.stdout).lower())

        # Batch over a folder: per-volume subfolders + a batch summary.
        comp2 = root / "many"
        comp2.mkdir()
        for stem in ("Position003", "Position010"):
            with mrcfile.new(str(comp2 / f"{stem}_components.mrc")) as m:
                m.set_data(d)
                m.voxel_size = vx
        out2 = root / "qc_batch"
        r3 = run_tool("--components", str(comp2), "--population", str(pop),
                      "--angpix", str(vx), "--out-folder", str(out2))
        check("batch exits 0", r3.returncode == 0)
        batch = json.loads((out2 / "size_qc_batch.json").read_text())
        check("batch summarises every volume", len(batch["volumes"]) == 2)
        check("per-volume subfolders hold their own reports",
              (out2 / "Position003_components" / "size_qc.json").is_file()
              and (out2 / "Position010_components" /
                   "outliers_only.mrc").is_file())

    # ---- the stage node -----------------------------------------------------
    st = load("tomogration_stages")
    jb = load("tomogration_jobs")
    sp = next(s for s in st.STAGES if s["id"] == "mb_size_qc")
    vals = st.stage_defaults(sp)
    check("blank components blocks", "Components" in sp["validate"](vals))
    vals["components"] = "jobs/J40_components"
    check("blank population blocks", "REQUIRED" in sp["validate"](vals))
    vals["population"] = "jobs/J41_population/population.json"
    check("filled form validates clean", sp["validate"](vals) == "")
    cmd = st.build_command(sp, vals)
    check("command carries tool + both inputs",
          "ml_virion_qc.py" in cmd and "--components jobs/J40_components" in cmd
          and "--population jobs/J41_population/population.json" in cmd)
    check("registered everywhere",
          "mb_size_qc" in st.STAGE_IO and "mb_size_qc" in st.STAGE_OUTPUTS
          and "mb_size_qc" in jb.FRIENDLY_TITLES
          and "mb_size_qc" in jb.VIEWER_PLANS)

    # ---- Build-downstream wiring (and the dead-code fix around it) ----------
    d1 = jb.derive_child_params("mb_size_qc", "mb_components",
                                {"input_dir": "jobs/J39_thresholds",
                                 "_source_tomograms": "membrane/isonet2/corrected"},
                                "jobs/J40_components")
    # components nest one cc<voxels>/ folder per swept cutoff — the child gets
    # the cc dir that actually holds volumes, not the empty job root.
    check("components → size QC hands over the cc-nested components dir",
          d1.get("components") == "jobs/J40_components/cc50")
    check("and the tomogram breadcrumb for napari",
          d1.get("tomogram") == "membrane/isonet2/corrected")
    d2 = jb.derive_child_params("mb_size_qc", "mb_population",
                                {"components": "jobs/J40_components"},
                                "jobs/J41_population")
    check("population → size QC hands over population.json",
          d2.get("population") == "jobs/J41_population/population.json")
    check("and the components it measured",
          d2.get("components") == "jobs/J40_components")
    d3 = jb.derive_child_params("mb_split_clusters", "mb_size_qc",
                                {"components": "jobs/J40_components",
                                 "population": "p.json"}, "jobs/J42_qc")
    check("size QC → split clusters splits the ORIGINAL components",
          d3.get("components") == "jobs/J40_components")
    # --out is the star FILE pick_surfaces writes: a bare jobs/{jobid} named
    # the directory _run_job pre-creates, and write_text on a directory dies
    # with IsADirectoryError — the auto-wired hop failed on every run.
    d5 = jb.derive_child_params("mb_pick_surfaces", "mb_fit_virions",
                                {}, "jobs/J45_fit-virions")
    check("fit → pick surfaces derives a star FILE, not the job dir",
          d5.get("out", "").endswith("/oversample.star")
          and d5.get("fits") == "jobs/J45_fit-virions/fits.json")

    # Breadcrumbs survive the measurement hops: a fit card three hops down
    # must sample the variant the chain segmented, not the form's default.
    d6 = jb.derive_child_params("mb_fit_virions", "mb_split_clusters",
                                {"components": "jobs/J40_components/cc50",
                                 "population": "p.json",
                                 "_source_tomograms":
                                     "membrane/isonet2/corrected"},
                                "jobs/J44_split-merged-virions")
    check("split → fit carries the greyscale breadcrumb as tomogram",
          d6.get("tomogram") == "membrane/isonet2/corrected")
    check("and keeps the breadcrumb moving for the NEXT hop",
          d6.get("_source_tomograms") == "membrane/isonet2/corrected")

    # A palette job keeps its stage default (a shared membrane/ dir): the
    # wrapper wrote THERE, so the child must read there — the job record's
    # jobs/J## dir is an empty folder _run_job merely mkdir'd.
    d7 = jb.derive_child_params("mb_size_qc", "mb_components",
                                {"output_dir": "membrane/components",
                                 "MB_CC_THRES": "50"},
                                "jobs/J40_components")
    check("a literal output param beats the empty record dir",
          d7.get("components") == "membrane/components/cc50")
    d8 = jb.derive_child_params("mb_size_qc", "mb_components",
                                {"output_dir": "jobs/{jobid}",
                                 "MB_CC_THRES": "50"},
                                "jobs/J40_components")
    check("a {jobid}-shaped param still resolves via the record",
          d8.get("components") == "jobs/J40_components/cc50")

    # The dead-code fix: the SPECIFIC pairs must win over the generic rule.
    d4 = jb.derive_child_params("mb_population", "mb_components",
                                {"input_dir": "jobs/J39_thresholds"},
                                "jobs/J40_components")
    check("components → population fills 'components', not 'input_dir'",
          "components" in d4 and "input_dir" not in d4)

    # ---- the review-confirmed wiring fixes ----------------------------------
    grey = "membrane/isonet2/corrected"
    # Breadcrumbs survive the specific branches (the fit card must sample the
    # variant the chain segmented, not the form's default).
    d5 = jb.derive_child_params("mb_fit_virions", "mb_split_clusters",
                                {"components": "jobs/J40_components/cc50",
                                 "population": "p.json",
                                 jb.GREY_KEY: grey}, "jobs/J44_split")
    check("split → fit carries the greyscale tomogram",
          d5.get("tomogram") == grey and d5.get(jb.GREY_KEY) == grey)
    d6 = jb.derive_child_params("mb_fit_virions", "mb_population",
                                {"components": "jobs/J40_components/cc50",
                                 jb.GREY_KEY: grey}, "jobs/J41_population")
    check("population → fit carries the greyscale tomogram",
          d6.get("tomogram") == grey)
    # pick_surfaces --out is a FILE: the derived path must not be the job dir.
    d7 = jb.derive_child_params("mb_pick_surfaces", "mb_fit_virions",
                                {}, "jobs/J45_fit-virions")
    check("fit → pick surfaces derives a star FILE, not the job dir",
          d7.get("out", "").endswith(".star"))
    check("and the fits.json path",
          d7.get("fits") == "jobs/J45_fit-virions/fits.json")
    # Single-value thresholds parent → components inherits the exact pattern.
    d8 = jb.derive_child_params("mb_components", "mb_thresholds",
                                {"MB_THRESHOLDS": "-3",
                                 jb.GREY_KEY: grey}, "jobs/J39_thresholds")
    check("single-threshold parent pins MB_PATTERN (one-decimal form)",
          d8.get("MB_PATTERN") == "*_threshold_-3.0.mrc")
    d9 = jb.derive_child_params("mb_components", "mb_thresholds",
                                {"MB_THRESHOLDS": "-3 -2 -1"},
                                "jobs/J39_thresholds")
    check("multi-threshold parent leaves MB_PATTERN alone",
          "MB_PATTERN" not in d9)
    # Polarity is reachable via Build downstream, and feeds the fitter.
    check("segment → polarity edge exists",
          "mb_polarity" in jb.DOWNSTREAM.get("mb_segment", []))
    d10 = jb.derive_child_params("mb_polarity", "mb_segment",
                                 {"input_dir": grey,
                                  "output_dir": "jobs/{jobid}"},
                                 "jobs/J38_segment")
    check("segment → polarity pins the folders",
          d10.get("segmentation") == "jobs/J38_segment"
          and d10.get("tomogram") == grey)
    d11 = jb.derive_child_params("mb_fit_virions", "mb_polarity",
                                 {"tomogram": grey + "/Position003.mrc",
                                  jb.GREY_KEY: grey}, "jobs/J46_polarity")
    check("polarity → fit hands the tomogram FOLDER",
          d11.get("tomogram") == grey)
    # membrane/* is delete-protected as a prefix (shared tune-era results).
    with tempfile.TemporaryDirectory() as td:
        r = Path(td)
        (r / "membrane" / "segment").mkdir(parents=True)
        t, skipped = jb.job_delete_targets(
            r, "J9", "mb_segment", {"output_dir": "membrane/segment"},
            ["output_dir"])
        check("shared membrane dirs are never delete targets",
              "membrane/segment" not in t
              and any("membrane/segment" in s for s in skipped))

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
