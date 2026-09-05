"""Membrane branch (MemBrain + IsoNet module, spec draft 1) — Deconvolve node.

THE POINT of these tests, in spec order:

  * --df comes from the Warp per-series .xml and is converted µm → Å in exactly
    one place. A series without a readable defocus yields None (skip, never
    guess) — the spec calls this out as the critical detail.
  * PROVENANCE.json tags every membrane output as a processed variant, so
    deconvolution can refuse already-processed inputs and (later) extraction
    can refuse everything but the original reconstruction (§5 hard rule).
  * The stage assembles the exact tomo_preprocessing wrapper call, with the
    tune/batch and sweep knobs as env vars.

    python3 tests/test_membrane.py
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE / "stub"))
sys.path.insert(0, str(REPO))


def load(mod):
    spec = importlib.util.spec_from_file_location(mod, REPO / f"{mod}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod] = m
    spec.loader.exec_module(m)
    return m


st = load("tomogration_stages")
proj = load("tomogration_project")

passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}")


def main():
    # ---- defocus parsing: µm in the xml, Å out, never guessed ---------------
    xml = ('<TiltSeries CTFResolutionEstimate="7.6">\n'
           '  <CTF>\n'
           '    <Param Name="Defocus" Value="4.4482814473158649" />\n'
           '    <Param Name="DefocusDelta" Value="0.1" />\n'
           '  </CTF>\n'
           '</TiltSeries>')
    df = proj.defocus_angstroms(xml)
    check("defocus µm → Å (×10000)", df is not None and abs(df - 44482.814473) < 1e-3)
    check("missing Defocus param → None",
          proj.defocus_angstroms("<TiltSeries></TiltSeries>") is None)
    check("non-numeric Defocus → None (skip, never guess)",
          proj.defocus_angstroms('<Param Name="Defocus" Value="NaN-ish" />') is None)
    check("empty text → None", proj.defocus_angstroms("") is None)
    check("None text → None", proj.defocus_angstroms(None) is None)
    # DefocusDelta must not satisfy a Defocus lookup.
    check("DefocusDelta alone does not match",
          proj.defocus_angstroms('<Param Name="DefocusDelta" Value="0.1" />') is None)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "warp_tiltseries").mkdir()
        (root / "warp_tiltseries" / "Position003.xml").write_text(xml)
        ps = proj.ProjectState(root)
        check("series_defocus_A reads the series xml",
              abs(ps.series_defocus_A("Position003") - 44482.814473) < 1e-3)
        check("series_defocus_A → None for a missing series",
              ps.series_defocus_A("Position999") is None)

    # ---- provenance tags ----------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        d = root / "membrane" / "deconv" / "s1.0_f1.0"
        d.mkdir(parents=True)
        (d / "PROVENANCE.json").write_text(
            '{"variant": "deconvolved", "extraction_allowed": false}')
        check("provenance variant is read back",
              proj.provenance_variant(d) == "deconvolved")
        check("no PROVENANCE.json → ''",
              proj.provenance_variant(root) == "")
        (d / "PROVENANCE.json").write_text("{ not json")
        check("malformed PROVENANCE.json → '' (never crashes a status sweep)",
              proj.provenance_variant(d) == "")

        # status: count deconvolved volumes across variant folders
        (d / "Position003_12.56Apx_deconv.mrc").write_text("x")
        d2 = root / "membrane" / "deconv" / "s0.5_f1.0"
        d2.mkdir(parents=True)
        (d2 / "Position003_12.56Apx_deconv.mrc").write_text("x")
        ps = proj.ProjectState(root)
        n, msg = ps.status_mb_deconv()
        check("status counts volumes across sweep variants", n == 2)
        check("status message says what they are", "deconvolved" in (msg or ""))

    # ---- the stage node -----------------------------------------------------
    spec = next(s for s in st.STAGES if s["id"] == "mb_deconv")
    check("stage lives in the new membrane group", spec["group"] == "12. Membrane")
    check("the group has a canvas column",
          spec["group"] in st.COLUMN_OF_GROUP)
    check("stage declares its conda env (envs must never be merged)",
          spec.get("env_name") == "membrainseg")
    check("stage is tagged compute (fine on a headless GPU node)",
          spec.get("node_kind") == "compute")
    check("stage registers its IO dirs", "mb_deconv" in st.STAGE_IO)
    check("stage registers its output dir",
          st.STAGE_OUTPUTS.get("mb_deconv") == "membrane/deconv")

    vals = st.stage_defaults(spec)
    cmd = st.build_command(spec, vals)
    toks = cmd.split()
    check("command is env assignments, then bash, then the wrapper",
          "bash" in toks
          and all("=" in t for t in toks[:toks.index("bash")])
          and toks[toks.index("bash") + 1].strip("'\"")
              .endswith("ml_membrain_deconv_warp_auto.sh"))
    check("wrapper script is in the command",
          "ml_membrain_deconv_warp_auto.sh" in cmd)
    check("input dir is positional",
          "warp_tiltseries/reconstruction" in cmd)
    check("output base is positional", "membrane/deconv" in cmd)
    check("conda env rides as MB_CONDA_ENV", "MB_CONDA_ENV=membrainseg" in cmd)
    check("defaults carry the spec's constants",
          "MB_KV=300" in cmd and "MB_CS=2.7" in cmd and "MB_AMPCON=0.07" in cmd
          and "MB_HP=0.02" in cmd)
    check("strength default present", "MB_STRENGTH=1.0" in cmd)
    check("blank tune list emits nothing (batch mode)",
          "MB_TOMO_LIST" not in cmd)

    # Sweep + tune values flow through as env (quoted when they hold spaces).
    vals2 = dict(vals, MB_STRENGTH="0.5 1.0 1.5",
                 MB_TOMO_LIST="Position003 Position017")
    cmd2 = st.build_command(spec, vals2)
    check("sweep list is quoted into MB_STRENGTH",
          "MB_STRENGTH='0.5 1.0 1.5'" in cmd2)
    check("tune list is quoted into MB_TOMO_LIST",
          "MB_TOMO_LIST='Position003 Position017'" in cmd2)

    # ---- the --help-verified optional knobs --------------------------------
    vals3 = dict(vals, MB_PIXEL_SIZE="12.56", MB_SKIP_LOWPASS="1")
    cmd3 = st.build_command(spec, vals3)
    check("explicit pixel size rides as MB_PIXEL_SIZE",
          "MB_PIXEL_SIZE=12.56" in cmd3)
    check("skip-lowpass opt-in rides as MB_SKIP_LOWPASS",
          "MB_SKIP_LOWPASS=1" in cmd3)
    check("blank pixel size emits nothing (header fallback)",
          "MB_PIXEL_SIZE" not in cmd)

    # ---- human-readable form labels ----------------------------------------
    # Explicit titles win; prettified fallbacks stay honest elsewhere.
    check("explicit title wins",
          st.param_title({"name": "MB_STRENGTH", "title": "Strength"})
          == "Strength")
    check("env prefix is dropped and case fixed",
          st.param_title({"name": "MA_POOL_SIZE"}) == "Pool size")
    check("bare acronyms survive", st.param_title({"name": "CS"}) == "CS")
    check("prefixed acronyms survive", st.param_title({"name": "MB_KV"}) == "KV")
    check("snake_case reads as a sentence",
          st.param_title({"name": "output_angpix"}) == "Output angpix")
    check("wire name shows $VAR for env knobs",
          st.param_wire_name({"name": "MB_STRENGTH", "kind": "env",
                              "flag": "MB_STRENGTH"}) == "$MB_STRENGTH")
    check("wire name shows the flag for flagged params",
          st.param_wire_name({"name": "perdevice", "kind": "slider_int",
                              "flag": "--perdevice"}) == "--perdevice")
    check("wire name falls back to the raw name for positionals",
          st.param_wire_name({"name": "input_dir", "kind": "text",
                              "flag": None}) == "input_dir")
    check("every mb_deconv param has an explicit title",
          all(p.get("title") for p in spec["params"]))
    check("tool line names the wrapper script",
          st.stage_tool_line(spec) == "bash ml_membrain_deconv_warp_auto.sh")
    recon = next(s for s in st.STAGES if s["id"] == "ts_reconstruct")
    check("tool line for a WarpTools stage is the subcommand",
          st.stage_tool_line(recon) == "WarpTools ts_reconstruct")

    # ---- validator: the two footguns ---------------------------------------
    v = spec["validate"]
    check("clean defaults validate clean", v(vals) == "")
    check("processed-looking input dir warns",
          "PROCESSED" in v(dict(vals, input_dir="membrane/isonet_denoised")))
    check("sweep across ALL tomograms warns to narrow it first",
          "Narrow it first" in v(dict(vals, MB_STRENGTH="0.5 1.0")))
    check("sweep with a tune list is fine",
          v(dict(vals, MB_STRENGTH="0.5 1.0",
                 MB_TOMO_LIST="Position003")) == "")

    # ---- §3.3–3.6: the four nodes built from the real --help output --------
    ALL_MB = ("mb_deconv", "mb_segment", "mb_thresholds", "mb_components",
              "mb_mesh")
    for sid in ALL_MB:
        sp = next(s for s in st.STAGES if s["id"] == sid)
        check(f"{sid}: registered everywhere",
              sid in st.STAGE_IO and sid in st.STAGE_OUTPUTS
              and sp["group"] in st.COLUMN_OF_GROUP)
        check(f"{sid}: every param titled",
              all(p.get("title") for p in sp["params"]))
        check(f"{sid}: declares env + kind",
              sp.get("env_name") in ("membrainseg", "membrainpick")
              and sp.get("node_kind") == "compute")

    seg = next(s for s in st.STAGES if s["id"] == "mb_segment")
    sv = st.stage_defaults(seg)
    check("segment: blank checkpoint blocks",
          "REQUIRED" in seg["validate"](sv))
    sv["MB_CKPT"] = "/models/membrain_v10.ckpt"
    check("segment: checkpoint satisfies the validator",
          seg["validate"](sv) == "")
    check("segment: disabling score maps warns loudly",
          "GPU re-run" in seg["validate"](dict(sv, MB_NO_PROBS="1")))
    check("segment: uncertainty without TTA warns",
          "TTA" in seg["validate"](dict(sv, MB_UNCERTAINTY="1",
                                        MB_NO_TTA="1")))
    scmd = st.build_command(seg, sv)
    check("segment: wrapper + ckpt in command",
          "ml_membrain_segment_warp_auto.sh" in scmd
          and "MB_CKPT=/models/membrain_v10.ckpt" in scmd)
    check("segment: pixel size default rides along",
          "MB_PIXEL_SIZE=12.56" in scmd)

    thr = next(s for s in st.STAGES if s["id"] == "mb_thresholds")
    tv = st.stage_defaults(thr)
    check("thresholds: empty list blocks",
          "empty" in thr["validate"](dict(tv, MB_THRESHOLDS="  ")))
    tcmd = st.build_command(thr, dict(tv, MB_THRESHOLDS="-1.5 -0.5 0.0 0.5"))
    check("thresholds: sweep list quoted into the env",
          "MB_THRESHOLDS='-1.5 -0.5 0.0 0.5'" in tcmd)
    check("thresholds: explicit out dir is positional",
          "membrane/thresholds" in tcmd)

    comp = next(s for s in st.STAGES if s["id"] == "mb_components")
    cv = st.stage_defaults(comp)
    check("components: empty cutoff list blocks",
          "empty" in comp["validate"](dict(cv, MB_CC_THRES="")))
    ccmd = st.build_command(comp, dict(cv, MB_CC_THRES="20 50 100"))
    check("components: sweep list quoted into the env",
          "MB_CC_THRES='20 50 100'" in ccmd)

    mesh = next(s for s in st.STAGES if s["id"] == "mb_mesh")
    mv = st.stage_defaults(mesh)
    check("mesh: lives in the membrainpick env",
          mesh.get("env_name") == "membrainpick"
          and "MB_CONDA_ENV=membrainpick" in st.build_command(mesh, mv))
    check("mesh: only-largest defaults OFF and clean",
          mesh["validate"](mv) == "")
    check("mesh: only-largest ON warns about merged clusters",
          "MERGED" in mesh["validate"](dict(mv, MB_ONLY_LARGEST="1")))
    mcmd = st.build_command(mesh, mv)
    check("mesh: three positional dirs in order",
          mcmd.find("membrane/thresholds") < mcmd.find("membrane/deconv")
          < mcmd.find("membrane/mesh"))
    check("mesh: explicit pixel size rides along",
          "MB_PIXEL_SIZE=12.56" in mcmd)
    check("mesh: barycentric area pinned explicitly",
          "MB_BARY_AREA=400" in mcmd)

    # Status methods exist and start at zero on an empty project.
    with tempfile.TemporaryDirectory() as td:
        ps = proj.ProjectState(Path(td))
        for sid in ALL_MB:
            sp = next(s for s in st.STAGES if s["id"] == sid)
            n, _ = sp["status"](ps)
            check(f"{sid}: status runs clean on an empty project", n == 0)

    # ---- §3.2 IsoNet (both stages + the star defocus injector) -------------
    for sid in ("mb_isonet_train", "mb_isonet_predict"):
        sp = next(s for s in st.STAGES if s["id"] == sid)
        check(f"{sid}: registered everywhere",
              sid in st.STAGE_IO and sid in st.STAGE_OUTPUTS)
        check(f"{sid}: every param titled",
              all(p.get("title") for p in sp["params"]))
        check(f"{sid}: is a module stage, not conda",
              sp.get("env_name") == "isonet-module")
        with tempfile.TemporaryDirectory() as td:
            n, _ = sp["status"](proj.ProjectState(Path(td)))
            check(f"{sid}: status runs clean on an empty project", n == 0)

    tr = next(s for s in st.STAGES if s["id"] == "mb_isonet_train")
    trv = st.stage_defaults(tr)
    check("isonet train: blank tomo list nudges toward 1-5",
          "1–5" in tr["validate"](trv) or "1-5" in tr["validate"](trv))
    check("isonet train: a tomo list validates clean",
          tr["validate"](dict(trv, ISO_TOMO_LIST="Position003")) == "")
    check("isonet train: processed input warns",
          "ORIGINAL" in tr["validate"](dict(trv, ISO_TOMO_LIST="P3",
                                            input_dir="membrane/deconv/s1.0_f1.0")))
    tcmd2 = st.build_command(tr, dict(trv, ISO_TOMO_LIST="Position003"))
    check("isonet train: wrapper + module + work dir in command",
          "ml_isonet1_train_warp_auto.sh" in tcmd2
          and "ISO_MODULE=isonet/0.3" in tcmd2
          and "membrane/isonet" in tcmd2)

    pr = next(s for s in st.STAGES if s["id"] == "mb_isonet_predict")
    prv = st.stage_defaults(pr)
    check("isonet predict: missing model blocks",
          "REQUIRED" in pr["validate"](prv))
    prv["ISO_MODEL"] = "membrane/isonet/results/model_iter30.h5"
    check("isonet predict: model satisfies validator",
          pr["validate"](prv) == "")
    pcmd = st.build_command(pr, prv)
    check("isonet predict: model rides as ISO_MODEL",
          "ISO_MODEL=membrane/isonet/results/model_iter30.h5" in pcmd)

    # The star injector: header-driven, longest-stem, µm→Å, atomic.
    import importlib.util as _ilu
    _sp = _ilu.spec_from_file_location("iso_star", REPO / "ml_isonet_star_defocus.py")
    iso = _ilu.module_from_spec(_sp)
    _sp.loader.exec_module(iso)
    check("injector: longest stem wins at a boundary",
          iso.stem_for("Position_10_12.56Apx", ["Position_1", "Position_10"])
          == "Position_10")
    check("injector: Position_1 does not claim Position_10",
          iso.stem_for("Position_10_x", ["Position_1"]) == "")
    check("injector: exact stem matches",
          iso.stem_for("Position003", ["Position003"]) == "Position003")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "Position_1.xml").write_text(
            '<Param Name="Defocus" Value="3.2" />')
        (root / "Position_10.xml").write_text(
            '<Param Name="Defocus" Value="4.1" />')
        star = root / "tomograms.star"
        star.write_text(
            "data_\n\nloop_\n"
            "_rlnIndex #1\n_rlnMicrographName #2\n_rlnPixelSize #3\n"
            "_rlnDefocus #4\n_rlnNumberSubtomo #5\n"
            "1\tinput_tomos/Position_1_12.56Apx.mrc\t12.56\t0.0\t100\n"
            "2\tinput_tomos/Position_10_12.56Apx.mrc\t12.56\t0.0\t100\n")
        import subprocess as sp_
        r = sp_.run([sys.executable, str(REPO / "ml_isonet_star_defocus.py"),
                     str(star), str(root)], capture_output=True, text=True)
        check("injector: runs clean on a two-row star", r.returncode == 0)
        rows = [l.split() for l in star.read_text().splitlines()
                if "input_tomos/" in l]
        by_name = {r_[1]: r_[3] for r_ in rows}
        check("injector: Position_1 got ITS defocus (3.2 um -> 32000 A)",
              by_name.get("input_tomos/Position_1_12.56Apx.mrc") == "32000.0")
        check("injector: Position_10 got ITS defocus (4.1 um -> 41000 A)",
              by_name.get("input_tomos/Position_10_12.56Apx.mrc") == "41000.0")

    # ---- IsoNet 0.3's two traps, both hit on a real run (2026-08-15) -------
    # (a) it exits 0 after crashing, leaving the pre-training model_iter00.h5;
    #     a file count called that a trained model and the card went green.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        res = root / "membrane" / "isonet" / "results"
        res.mkdir(parents=True)
        ps = proj.ProjectState(root)
        (res / "model_iter00.h5").write_text("")
        n, msg = ps.status_mb_isonet_train()
        check("isonet status: iter00 alone is NOT a trained model",
              n == 0 and "crashed" in (msg or ""))
        (res / "model_iter30.h5").write_text("")
        n, msg = ps.status_mb_isonet_train()
        check("isonet status: reports the highest iteration reached",
              n == 2 and "iter 30" in (msg or ""))
    # (b) the wrapper must link the training set under dot-free names, or
    #     refine dies on 'results/Position003_12_iter00.mrc'.
    train_sh = (REPO / "ml_isonet1_train_warp_auto.sh").read_text()
    check("train wrapper: links tomograms with dots replaced",
          'safe="${stem//./p}"' in train_sh
          and 'ln -sf "$1" "$IN/$safe.mrc"' in train_sh)
    check("train wrapper: the refine SKIP needs the FINAL model, not any model",
          'if [ -f "$FINAL_MODEL" ] && [ "$FORCE" != "1" ]' in train_sh
          and "SKIP: results/ already holds trained models" not in train_sh)
    check("train wrapper: verifies the FINAL iteration's model",
          'FINAL_MODEL="results/model_iter$(printf' in train_sh
          and "exit 1" in train_sh.split("FINAL_MODEL=")[1])
    check("train wrapper: a failed refine writes no PROVENANCE",
          train_sh.index("FINAL_MODEL=") < train_sh.index("cat > PROVENANCE.json"))
    check("train wrapper: a stale star naming unlinked files is fatal",
          "references $GONE tomogram(s)" in train_sh)
    # 'the folder has files in it' skipped steps whose files belonged to a
    # DIFFERENT tomo list — every resume test is now per-tomogram.
    check("train wrapper: reused folders are stamped with their settings",
          all(f"stamp_check {d}" in train_sh and f"stamp_write {d}" in train_sh
              for d in ("deconv", "mask", "subtomo")))
    check("train wrapper: resume tests cover every selected tomogram",
          "have_all deconv" in train_sh and "have_all mask" in train_sh
          and "covers_all subtomo.star" in train_sh
          and "ls deconv/*.mrc >/dev/null" not in train_sh)
    pred_sh = (REPO / "ml_isonet1_predict_warp_auto.sh").read_text()
    check("predict wrapper: links dot-free the same way",
          'safe="${stem//./p}"' in pred_sh)

    # A knob the wrapper reads but the form never offers is invisible: the only
    # way to reach it is hand-editing the command box. ISO_SNRFALLOFF,
    # ISO_DECONVSTRENGTH, the two mask percentages, ISO_NCPU, ISO_EXTRA_REFINE
    # and predict's ISO_CUBE all sat unreachable that way until 2026-08-15.
    import re as _re
    for sid, sh_name in (("mb_isonet_train", "ml_isonet1_train_warp_auto.sh"),
                         ("mb_isonet_predict", "ml_isonet1_predict_warp_auto.sh")):
        sh = (REPO / sh_name).read_text()
        used = set(_re.findall(r"ISO_[A-Z_]+", sh)) - {"ISO_MOD"}   # ISO_MOD is local
        sp = next(s for s in st.STAGES if s["id"] == sid)
        offered = {p.get("flag") for p in sp["params"]}
        check(f"{sid}: the form offers every env its wrapper reads",
              not (used - offered))
    for sid, sh_name in (("mb_isonet2_train", "ml_isonet2_train_warp_auto.sh"),
                         ("mb_isonet2_predict", "ml_isonet2_predict_warp_auto.sh")):
        sh = (REPO / sh_name).read_text()
        # ISO2_KV/CS/AC/TILT_* are scope constants the wrapper defaults inline;
        # they are documented in its header rather than given a form row.
        used = (set(_re.findall(r"ISO2_[A-Z_]+", sh))
                - {"ISO2_KV", "ISO2_CS", "ISO2_AC", "ISO2_TILT_MIN",
                   "ISO2_TILT_MAX", "ISO2_HIGHPASS", "ISO2_PREFIX",
                   "ISO2_APPLY_MW", "ISO2_NCPUS", "ISO2_BATCH", "ISO2_LOSS",
                   "ISO2_SNRFALLOFF", "ISO2_DECONVSTRENGTH", "ISO2_DENSITY_PCT",
                   "ISO2_STD_PCT", "ISO2_EXTRA_REFINE",
                   # named only inside an error message, pointing at the TRAIN
                   # stage's knob — not read by the predict wrapper
                   "ISO2_METHOD"})
        sp = next(s for s in st.STAGES if s["id"] == sid)
        offered = {p.get("flag") for p in sp["params"]}
        missing = used - offered
        check(f"{sid}: the form offers every env its wrapper reads ({missing})",
              not missing)

    # ---- IsoNet 2: the second backend (§3.2), sibling stages ---------------
    # Its CLI is python-fire: -h is --highpassnyquist, and -p/-s/-d mean
    # different things per subcommand. A short flag anywhere in these wrappers
    # is a silent wrong-parameter bug, so there must not be one.
    import re as _re2
    for name in ("ml_isonet2_train_warp_auto.sh", "ml_isonet2_predict_warp_auto.sh"):
        body = (REPO / name).read_text()
        code = "\n".join(l for l in body.splitlines()
                         if not l.lstrip().startswith("#"))
        # Only the ARGUMENT part of a line counts: `[ -n "$X" ] && FLAGS+=(…)`
        # and `command -v isonet.py` carry shell operators, not tool flags.
        args = []
        for l in code.splitlines():
            if "FLAGS" in l and "(" in l:
                args.append(l.split("(", 1)[1])
            elif "isonet.py" in l and "command -v" not in l:
                args.append(l.split("isonet.py", 1)[1])
        bad = [s for a in args
               for s in _re2.findall(r"(?<![\w-])-[a-zA-Z](?=[\s\"'])", a)]
        check(f"{name}: no short flags on isonet.py calls ({sorted(set(bad))})",
              not bad)
        check(f"{name}: refuses to run without the IsoNet 2 env",
              "no IsoNet 2 env at" in body)
        check(f"{name}: activates a PREFIX, never -n",
              'conda activate "$ENV_PREFIX"' in code
              and "conda activate -n" not in code)

    # refine's 'auto' cannot choose once --create_average has put averaged full
    # volumes in the star next to the halves: it raises ValueError — AFTER
    # deconv and make_mask have run (30 min on five tomograms, 2026-08-15). The
    # wrapper must resolve auto itself, BEFORE the chain starts.
    train2_sh = (REPO / "ml_isonet2_train_warp_auto.sh").read_text()
    check("isonet2 train: auto is resolved for halves, not left to refine",
          'METHOD="isonet2-n2n"' in train2_sh)
    _resolve = train2_sh.split("mkdir -p \"$WORK_DIR\"")[1].split("echo \"===")[0]
    check("isonet2 train: and resolved BEFORE prepare_star/deconv run",
          'METHOD="isonet2-n2n"' in _resolve)
    check("isonet2 train: only 'auto' is overridden, an explicit method stands",
          '[ "$METHOD" = "auto" ]' in train2_sh)
    check("isonet2 train: the log says which method it chose and why",
          "$METHOD$METHOD_WHY" in train2_sh)

    # A complete 50-epoch run exited 1 at teardown: matplotlib in a Qt-carrying
    # env talks to X, and the dying connection kills the process AFTER the last
    # save (ICE … errno 32). Two defences: never open a display, and never let
    # an exit STATUS decide whether training happened — the checkpoints do.
    pred2_sh = (REPO / "ml_isonet2_predict_warp_auto.sh").read_text()
    for name, body in (("train", train2_sh), ("predict", pred2_sh)):
        check(f"isonet2 {name}: plots headless, so X cannot kill the run",
              "MPLBACKEND=Agg" in body and "QT_QPA_PLATFORM=offscreen" in body)
        check(f"isonet2 {name}: a non-zero exit does not abort before the check",
              "|| RC=$?" in body)
    check("isonet2 train: checkpoints decide success, not the exit status",
          "-name '*.pt' -newer" in train2_sh
          and 'if [ "$FRESH" = "0" ]' in train2_sh)
    check("isonet2 train: a failed refine lists what it DID write",
          '-newer "$STARTED" -type f' in train2_sh)

    # Every step reads the STAR, not input_tomos. A star left by a different
    # selection therefore runs the whole chain on THAT selection's volumes and
    # exits 0 — a 12.56 A single-map run resumed a bin4 n2n star and spent half
    # an hour deconvolving the old 6.28 A volumes (2026-08-16). Three guards.
    check("isonet2 train: resume checks the star COVERS the current selection",
          'not in the star' in train2_sh
          and 'do not appear in' in train2_sh)
    check("isonet2 train: and checks every folder a star can reference",
          "averaged_tomos" in train2_sh and "input_[a-z]*" in train2_sh)
    check("isonet2 train: pixel size and pairing are stamped on the star",
          'STAR_STAMP="pixel=$PIX pairing=' in train2_sh
          and 'stamp_check . "$STAR_STAMP"' in train2_sh
          and 'stamp_write . "$STAR_STAMP"' in train2_sh)
    check("isonet2 train: a single-map run clears stale halves",
          "clearing $half/" in train2_sh)
    # A single-map model trained on deconvolved volumes but PREDICTED from raw
    # ones is a quiet quality loss: predict builds its star from raw tomograms
    # and never deconvolves, so training must read the same column.
    check("isonet2 train: single-map refine trains on the column predict reads",
          '${ISO2_REFINE_INPUT_COL:-rlnTomoName}' in train2_sh)
    check("isonet2 train: and n2n is left alone, since it reads the halves",
          '[ "$METHOD" != "isonet2-n2n" ]' in train2_sh)

    # prepare_star fills columns it has no data for with the STRING 'None', so
    # predict's own default (--input_column rlnDeconvTomoName) made it open a
    # file named 'None' minutes in. The star this wrapper builds is raw
    # tomograms, so the column must be rlnTomoName — and be checked first.
    check("isonet2 predict: reads the raw-tomogram column, not the tool default",
          'INPUT_COL="${ISO2_INPUT_COL:-rlnTomoName}"' in pred2_sh
          and '--input_column "$INPUT_COL"' in pred2_sh)
    check("isonet2 predict: verifies the column holds paths, not 'None'",
          'predict.star' in pred2_sh and '"None", "none", ""' in pred2_sh)
    # IsoNet names outputs "<prefix>_<method>_<arch>_<the link we made>", and our
    # link spells the pixel size dot-free. That name is unusable downstream twice
    # over — the series stem is not at the front, so '<stem>_*.mrc' matches
    # nothing, and 12p56Apx does not parse as a pixel size. Outputs are renamed
    # back to <series>_<angpix>Apx_isonet2.mrc before anything else sees them.
    check("isonet2 predict: restores series-leading output names",
          '_isonet2.mrc' in pred2_sh and "orig_of_link" in pred2_sh)
    check("isonet2 predict: matched by the LONGEST link stem (Position1 vs Position10)",
          '[ "${#s}" -gt "${#best}" ]' in pred2_sh)
    check("isonet2 predict: never clobbers an existing output when renaming",
          'already exists — leaving' in pred2_sh)
    check("isonet2 predict: the resume test uses the FINAL name",
          '[ -e "$CORR/${orig}_isonet2.mrc" ]' in pred2_sh)
    # The renamed form must actually parse for the picker and the registry.
    check("a renamed corrected volume yields its series stem",
          proj.series_stem("Position002_12.56Apx_isonet2.mrc") == "Position002")

    _pr2 = next(s for s in st.STAGES if s["id"] == "mb_isonet2_predict")
    _col = next(p for p in _pr2["params"] if p["name"] == "ISO2_INPUT_COL")
    check("isonet2 predict: the form's default matches the wrapper's",
          _col["default"] == "rlnTomoName")

    # Their predict_row branches on the MODEL's method: 'regular'/'isonet2' read
    # input_column, everything else (n2n) reads Half1/Half2 and ignores it. So
    # an n2n model without half folders can only fail — catch it before the star
    # is even built, in the form and in the wrapper.
    _n2n = dict(st.stage_defaults(_pr2),
                ISO2_MODEL="membrane/isonet2/isonet_maps/"
                           "network_isonet2-n2n_unet-medium_96_epoch50_full.pt")
    check("isonet2 predict: an n2n model without halves is refused",
          "HALVES" in _pr2["validate"](_n2n))
    check("isonet2 predict: n2n + both halves validates clean",
          _pr2["validate"](dict(_n2n, ISO2_EVEN_DIR="jobs/J16/reconstruction/even",
                                ISO2_ODD_DIR="jobs/J16/reconstruction/odd")) == "")
    check("isonet2 predict: a single-map model needs no halves",
          _pr2["validate"](dict(st.stage_defaults(_pr2),
                                ISO2_MODEL="isonet_maps/network_isonet2_unet-medium_96.pt")) == "")
    check("isonet2 predict: the wrapper detects n2n from the checkpoint name",
          "*n2n*) IS_N2N=1" in pred2_sh)
    check("isonet2 predict: an n2n star is built from the halves",
          "--even input_even --odd input_odd" in pred2_sh)
    check("isonet2 predict: and the half columns are the ones verified",
          "rlnTomoReconstructedTomogramHalf1 rlnTomoReconstructedTomogramHalf2"
          in pred2_sh)
    # The injector must survive a halves-only star, where _rlnTomoName exists
    # but every row of it is the string 'None'.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "Position003.xml").write_text('<Param Name="Defocus" Value="4.99" />')
        star = root / "halves.star"
        star.write_text(
            "data_\n\nloop_\n_rlnIndex #1\n_rlnTomoName #2\n"
            "_rlnTomoReconstructedTomogramHalf1 #3\n_rlnDefocus #4\n"
            "1\tNone\tinput_even/Position003_6p28Apx.mrc\t10000\n")
        import subprocess as sp3_
        r3 = sp3_.run([sys.executable, str(REPO / "ml_isonet_star_defocus.py"),
                       str(star), str(root)], capture_output=True, text=True)
        check("injector: skips a name column full of 'None' for a usable one",
              r3.returncode == 0 and "Half1" in r3.stdout)
        check("injector: patched the halves-only row",
              "49900.0" in star.read_text())

    tr2 = next(s for s in st.STAGES if s["id"] == "mb_isonet2_train")
    tv2 = st.stage_defaults(tr2)
    check("isonet2 train: blank tomo list nudges toward 1-5",
          "1–5" in tr2["validate"](tv2) or "1-5" in tr2["validate"](tv2))
    check("isonet2 train: a tomo list validates clean",
          tr2["validate"](dict(tv2, ISO2_TOMO_LIST="Position003")) == "")
    # Half a pair is the dangerous case: prepare_star would accept it and
    # refine --method auto would quietly fall back to single-map training.
    check("isonet2 train: one half without the other is refused",
          "BOTH" in tr2["validate"](dict(tv2, ISO2_TOMO_LIST="P3",
                                         ISO2_EVEN_DIR="halves/even")))
    check("isonet2 train: both halves validate clean",
          tr2["validate"](dict(tv2, ISO2_TOMO_LIST="P3",
                               ISO2_EVEN_DIR="halves/even",
                               ISO2_ODD_DIR="halves/odd")) == "")
    check("isonet2 train: n2n without halves is refused",
          "n2n" in tr2["validate"](dict(tv2, ISO2_TOMO_LIST="P3",
                                        ISO2_METHOD="isonet2-n2n")))
    check("isonet2 train: processed input warns",
          "ORIGINAL" in tr2["validate"](dict(tv2, ISO2_TOMO_LIST="P3",
                                             input_dir="membrane/deconv/s1.0_f1.0")))
    tcmd = st.build_command(tr2, dict(tv2, ISO2_TOMO_LIST="Position003"))
    check("isonet2 train: wrapper + work dir in command",
          "ml_isonet2_train_warp_auto.sh" in tcmd and "membrane/isonet2" in tcmd)
    check("isonet2 train: does NOT carry the isonet/0.3 module",
          "ISO_MODULE" not in tcmd and "isonet/0.3" not in tcmd)

    pr2 = next(s for s in st.STAGES if s["id"] == "mb_isonet2_predict")
    pv2 = st.stage_defaults(pr2)
    check("isonet2 predict: missing model blocks", "REQUIRED" in pr2["validate"](pv2))
    # The two versions' models are not interchangeable, and pointing predict at
    # the wrong one is an easy mistake with two IsoNet folders side by side.
    check("isonet2 predict: an IsoNet 1 .h5 is rejected",
          "IsoNet 1" in pr2["validate"](
              dict(pv2, ISO2_MODEL="membrane/isonet/results/model_iter30.h5")))
    pv2["ISO2_MODEL"] = "membrane/isonet2/isonet_maps/model.pt"
    check("isonet2 predict: a .pt satisfies the validator",
          pr2["validate"](pv2) == "")
    check("isonet2 predict: model rides as ISO2_MODEL",
          "ISO2_MODEL=membrane/isonet2/isonet_maps/model.pt"
          in st.build_command(pr2, pv2))

    # Both versions must stay separately addressable: same work dir, or a
    # stage id colliding, and a v2 run would overwrite v1's models.
    outs = {s["id"]: st.STAGE_OUTPUTS.get(s["id"]) for s in st.STAGES
            if s["id"].startswith("mb_isonet")}
    check(f"isonet v1/v2 write to different folders ({outs})",
          len(set(outs.values())) == len(outs))
    with tempfile.TemporaryDirectory() as td:
        ps = proj.ProjectState(Path(td))
        for sid in ("mb_isonet2_train", "mb_isonet2_predict"):
            sp = next(s for s in st.STAGES if s["id"] == sid)
            n, _ = sp["status"](ps)
            check(f"{sid}: status runs clean on an empty project", n == 0)
        maps = Path(td) / "membrane" / "isonet2" / "isonet_maps"
        maps.mkdir(parents=True)
        (maps / "model_epoch50.pt").write_text("")
        n, msg = ps.status_mb_isonet2_train()
        check("isonet2 train status counts .pt checkpoints",
              n == 1 and "model_epoch50.pt" in msg)

    # The defocus injector serves BOTH versions: IsoNet 2's star is keyed on
    # _rlnTomoName, and hardcoding v1's column name would silently skip it.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "Position003.xml").write_text('<Param Name="Defocus" Value="4.99" />')
        star = root / "tomograms.star"
        star.write_text(
            "data_\n\nloop_\n"
            "_rlnIndex #1\n_rlnTomoName #2\n_rlnPixelSize #3\n_rlnDefocus #4\n"
            "1\tinput_tomos/Position003_12p56Apx.mrc\t12.56\t10000\n")
        import subprocess as sp2_
        r = sp2_.run([sys.executable, str(REPO / "ml_isonet_star_defocus.py"),
                      str(star), str(root)], capture_output=True, text=True)
        check("injector: accepts an IsoNet 2 star (_rlnTomoName)",
              r.returncode == 0 and "_rlnTomoName" in r.stdout)
        check("injector: patched the IsoNet 2 row (4.99 um -> 49900 A)",
              "49900.0" in star.read_text())
        # A star with neither spelling must fail LOUDLY and say what it saw —
        # a renamed column upstream is a one-line fix, but only if visible.
        odd = root / "odd.star"
        odd.write_text("data_\n\nloop_\n_rlnIndex #1\n_rlnWeirdName #2\n1\tx.mrc\n")
        r2 = sp2_.run([sys.executable, str(REPO / "ml_isonet_star_defocus.py"),
                       str(odd), str(root)], capture_output=True, text=True)
        check("injector: unknown columns fail loudly, naming what it found",
              r2.returncode != 0 and "_rlnWeirdName" in r2.stderr)

    # The parameter search must be able to RUN, not only plan — and it cannot
    # start without a checkpoint, so unticking Plan only while leaving that
    # blank has to be caught in the form rather than 40 minutes in.
    ex = next(s for s in st.STAGES if s["id"] == "mb_explore")
    exv = st.stage_defaults(ex)
    # THE bug this pins: the control used to be 'Plan only' wired to --plan,
    # so unticking it removed that flag and added nothing — and the tool, which
    # needs --run to execute, planned again. The checkbox must emit the flag
    # that DOES the thing, or "I unticked it and it still just planned".
    check("explore: planning is the default (costing is cheap, running is not)",
          exv.get("run") is False)
    plan_cmd = st.build_command(ex, exv)
    check("explore: the default command does NOT ask for a run",
          "--run" not in plan_cmd)
    run_cmd = st.build_command(ex, dict(exv, run=True,
                                        ckpt="/m/MemBrain_seg_v10_beta.ckpt"))
    check("explore: ticking 'Run the sweep' passes --run", "--run" in run_cmd)
    check("explore: and the checkpoint rides along", "--ckpt /m/" in run_cmd)
    check("explore: a run without a checkpoint is blocked in the form",
          "checkpoint" in ex["validate"](dict(exv, run=True, ckpt="")).lower())
    check("explore: planning without one is fine",
          ex["validate"](dict(exv, run=False, ckpt="")) == "")

    # Every tomogram-list param offers the picker, and its pick_from must name
    # a REAL sibling param — a typo there silently gives back a dead button.
    for sp in st.STAGES:
        names = {p["name"] for p in sp.get("params", [])}
        for p in sp.get("params", []):
            if p["name"].endswith("TOMO_LIST"):
                check(f"{sp['id']}/{p['name']}: has a picker",
                      p.get("pick_from") in names)
            elif p.get("pick_from"):
                check(f"{sp['id']}/{p['name']}: pick_from names a sibling",
                      p["pick_from"] in names)

    # ---- the tomogram picker's view of a folder ---------------------------
    # Every *_TOMO_LIST knob takes SERIES STEMS, and the wrappers match
    # '<stem>.mrc' or '<stem>_*.mrc' — so the picker must group a series'
    # derived files under the one stem the user would type.
    check("stem: pixel-size tag stripped",
          proj.series_stem("Position003_10.00Apx.mrc") == "Position003")
    check("stem: derived files collapse to the same stem",
          proj.series_stem("Position003_10.00Apx_scores.mrc") == "Position003"
          and proj.series_stem(
              "Position003_10.00Apx_segmented_threshold_-1.0.mrc") == "Position003")
    check("stem: no pixel-size tag = the basename itself",
          proj.series_stem("tomo_A.mrc") == "tomo_A")
    check("stem: an integer pixel size counts too",
          proj.series_stem("Position_7_10Apx.mrc") == "Position_7")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for n in ("Position2_12.56Apx.mrc", "Position10_12.56Apx.mrc",
                  "Position10_12.56Apx_scores.mrc", "notes.txt"):
            (d / n).write_text("")
        groups = proj.tomogram_stems(d)
        check("picker: one row per series, .txt ignored",
              [s for s, _ in groups] == ["Position2", "Position10"])
        check("picker: derived files ride under their series",
              len(dict(groups)["Position10"]) == 2)
        check("picker: missing folder is empty, not an error",
              proj.tomogram_stems(d / "nope") == [])

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
