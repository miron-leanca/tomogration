"""M (multi-particle refinement) stages: commands, launcher, and guards.

M ships as SEPARATE executables (MTools / MCore / EstimateWeights) inside the same
conda env as WarpTools. build_command only knew how to prefix the user's launcher
onto commands starting with "WarpTools", so without the WARP_SUITE handling every
M job would die with "MTools: command not found".

Also checks the guardrails that encode M's own guidance: exhaustive defocus search
is pointless without defocus refinement, a run with nothing enabled does nothing,
and piling on parameters at once invites overfitting.

    python3 tests/test_m_stages.py
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE / "stub"))
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("tomstages", REPO / "tomogration_stages.py")
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)

LAUNCH = "module load miniconda/latest && conda activate warp && WarpTools"
passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}{(' — ' + detail) if detail else ''}")


def stage(sid):
    return next(s for s in st.STAGES if s["id"] == sid)


def defaults(sp):
    return {p["name"]: p.get("default") for p in sp["params"]}


def main():
    ids = ["m_create_population", "m_create_source", "m_mask_create",
           "m_create_species", "m_core", "m_estimate_weights",
           "m_resample_trajectories"]
    have = {s["id"] for s in st.STAGES}
    for sid in ids:
        check(f"stage exists: {sid}", sid in have)

    # ---- the launcher must reach the non-WarpTools executables -------------
    for sid, exe in (("m_create_population", "MTools create_population"),
                     ("m_core", "MCore"),
                     ("m_estimate_weights", "EstimateWeights")):
        sp = stage(sid)
        cmd = st.build_command(sp, defaults(sp), LAUNCH)
        check(f"{sid}: conda env prefix applied",
              cmd.startswith("module load miniconda/latest && conda activate warp &&"),
              cmd[:70])
        check(f"{sid}: runs {exe.split()[0]}", exe in cmd, cmd[:90])
        check(f"{sid}: does not invoke WarpTools",
              " WarpTools " not in cmd and not cmd.rstrip().endswith("WarpTools"))

    # relion_mask_create is NOT a Warp tool — it must be left alone
    mk = stage("m_mask_create")
    cmd = st.build_command(mk, defaults(mk), LAUNCH)
    # It must not get the WARP launcher (it is a RELION binary) — but it DOES need
    # its own `module load relion/...` prefix, or it is "command not found".
    check("mask_create is not given the Warp launcher",
          "conda activate warp" not in cmd, cmd[:80])
    check("mask_create still loads RELION itself",
          "module load relion" in cmd and "relion_mask_create" in cmd, cmd[:80])

    # ---- MCore: the documented first-refinement command --------------------
    core = stage("m_core")
    v = defaults(core)
    v.update(population="m/EML45.population", iter="", refine_imagewarp="6x4",
             refine_particles=True, ctf_defocus=True, ctf_defocusexhaustive=True,
             perdevice_refine=4)
    cmd = st.build_command(core, v, LAUNCH)
    for flag in ("--population m/EML45.population", "--refine_imagewarp 6x4",
                 "--refine_particles", "--ctf_defocus", "--ctf_defocusexhaustive",
                 "--perdevice_refine 4"):
        check(f"MCore emits {flag}", flag in cmd, cmd[:120])
    check("blank iter is omitted", "--iter" not in cmd, cmd[:120])
    check("unticked flags are omitted", "--refine_mag" not in cmd)
    check("first-refinement command is clean", core["validate"](v) == "")

    # check run
    v0 = dict(defaults(core), population="m/x.population", iter="0")
    check("iter 0 emits --iter 0", "--iter 0" in st.build_command(core, v0, LAUNCH))
    check("iter 0 flagged as a check run", "CHECK RUN" in core["validate"](v0))

    # ---- guards ------------------------------------------------------------
    bad = dict(defaults(core), population="m/x.population",
               ctf_defocusexhaustive=True, ctf_defocus=False)
    check("exhaustive without defocus is caught",
          "only works together with ctf_defocus" in core["validate"](bad))
    check("nothing-enabled run is caught",
          "Nothing to refine" in core["validate"](dict(defaults(core),
                                                       population="m/x.population")))
    piled = dict(defaults(core), population="m/x.population", refine_imagewarp="6x4",
                 refine_particles=True, refine_stageangles=True, refine_mag=True,
                 ctf_defocus=True, ctf_cs=True, ctf_zernike3=True)
    check("too-many-parameters warns about overfitting",
          "ONE AT A TIME" in core["validate"](piled))

    # ---- species guards ----------------------------------------------------
    sp = stage("m_create_species")
    base = dict(defaults(sp), population="m/x.population", name="spike",
                diameter="150", particles_relion="r/run_data.star")
    check("species: filtered half map is flagged",
          "UNFILTERED" in sp["validate"](dict(base, half1="run_half1_class001.mrc",
                                              half2="h2.mrc")))
    check("species: unfiltered half maps pass",
          sp["validate"](dict(base, half1="run_half1_class001_unfil.mrc",
                              half2="run_half2_class001_unfil.mrc")) == "")
    check("species: missing half maps caught",
          "half1 + half2" in sp["validate"](dict(base, half1="", half2="")))

    # ---- weights + trajectories -------------------------------------------
    w = stage("m_estimate_weights")
    wv = dict(defaults(w), population="m/x.population", source="EML45")
    check("weights emits --resolve_items by default",
          "--resolve_items" in st.build_command(w, wv, LAUNCH))
    check("weights reminds you to run MCore after",
          "MCore" in w["validate"](wv))
    rt = stage("m_resample_trajectories")
    check("trajectories demands the hashed species path",
          "random hash" in rt["validate"](dict(defaults(rt),
                                               population="m/x.population", species="")))

    # ---- wiring ------------------------------------------------------------
    for sid in ids:
        check(f"{sid} has an output dir", sid in st.STAGE_OUTPUTS)
        check(f"{sid} has IO mapping", sid in st.STAGE_IO)
    check("M group is placed in a column",
          st.COLUMN_OF_GROUP.get("11. M refinement") is not None)

    # ---- {placeholder} substitution ----------------------------------------
    # A stage default may reference a SIBLING field ("m/{name}.population"). If it
    # is not substituted the tool receives the literal text: MTools was handed
    # "m/{name}.population" and created a population file called
    # "{name}.population", so the data source joined the wrong project.
    src = stage("m_create_source")
    sv = dict(defaults(src), name="EML45-spike-closed")
    cmd = st.build_command(src, sv, LAUNCH)
    check("{name} resolves from the sibling field",
          "m/EML45-spike-closed.population" in cmd, cmd)
    check("no literal placeholder survives", "{name}" not in cmd, cmd)
    check("no literal brace reaches the tool", "{" not in cmd, cmd)

    # an EMPTY sibling must leave the placeholder visible, not silently blank —
    # the GUI turns a surviving {placeholder} into a warning
    cmd_empty = st.build_command(src, dict(defaults(src), name=""), LAUNCH)
    check("empty sibling substitutes to empty (caught by the GUI guard)",
          "m/.population" in cmd_empty or "{name}" in cmd_empty, cmd_empty)

    # {jobid} is job-level and must NOT be eaten by sibling substitution
    exp = stage("ts_export_particles")
    check("{jobid} is left for the job layer",
          "{jobid}" in st.build_command(exp, defaults(exp)))

    # ---- M setup is NOT idempotent; the cards must say so -------------------
    # create_population on an existing population LOADS it (and every .source it
    # references), so a missing .source turns every later M command into a .NET
    # FileNotFoundException. Both setup cards must warn, and a reset must exist.
    cp = stage("m_create_population")
    msg = cp["validate"](dict(defaults(cp), name="EML45-spike-closed"))
    check("create_population warns it is run-once", "RUN THIS ONCE" in msg, msg)
    check("create_population points at the reset tool", "reset setup" in msg)
    cs = stage("m_create_source")
    msg2 = cs["validate"](dict(defaults(cs), name="EML45-spike-closed"))
    check("create_source warns it is run-once", "RUN THIS ONCE" in msg2, msg2)
    check("create_source says where the .source lands",
          "warp_tiltseries" in msg2, msg2)
    check("reset stage exists", "m_reset" in {x["id"] for x in st.STAGES})
    rs = stage("m_reset")
    rcmd = st.build_command(rs, defaults(rs), LAUNCH)
    check("reset defaults to report-only (no --execute)", "--execute" not in rcmd, rcmd)
    check("reset runs the shipped script",
          "ml_m_reset_warp_auto.sh" in rcmd, rcmd)
    check("reset with execute passes the flag",
          "--execute" in st.build_command(rs, dict(defaults(rs), execute=True), LAUNCH))

    # ---- cluster tools live behind lmod modules ----------------------------
    # relion_mask_create is a bare binary, not a Warp tool, so it is NOT on PATH:
    # running it unprefixed gives "relion_mask_create: command not found".
    mk2 = stage("m_mask_create")
    mv = dict(defaults(mk2), i="Refine3D/job017/run_class001.mrc",
              o="m/mask_closed_spike.mrc", ini_threshold="0.15")
    c = st.build_command(mk2, mv, LAUNCH)
    check("mask_create loads the RELION module first",
          c.startswith("module load relion/4.0.1 && relion_mask_create"), c)
    check("module value is not also passed as an argument",
          "--relion_module" not in c and " relion/4.0.1 --" not in c, c)
    check("blank module = assume it is on PATH",
          st.build_command(mk2, dict(mv, relion_module=""),
                           LAUNCH).startswith("relion_mask_create"))
    check("mask args survive the prefix",
          "--i Refine3D/job017/run_class001.mrc" in c and "--ini_threshold 0.15" in c, c)

    # no OTHER stage may invoke a bare binary without a module prefix
    suite = set(st.WARP_SUITE) | {"bash", "python3", "python", "module", ""}
    bare = []
    for x in st.STAGES:
        head = (x.get("base") or "").split(None, 1)
        if head and head[0] not in suite:
            if not any(pp["kind"] == "module" for pp in x.get("params", [])):
                bare.append(x["id"])
    check("no stage calls a bare binary without a module param", not bare, str(bare))

    # ---- MCore flags verified against `MCore --help` (2.0.0) ---------------
    core2 = stage("m_core")
    names = {pp["name"]: pp for pp in core2["params"]}
    check("GPU flag is --devicelist (NOT --device_list)",
          names.get("devicelist", {}).get("flag") == "--devicelist")
    check("no --device_list anywhere in m_core",
          all(pp.get("flag") != "--device_list" for pp in core2["params"]))
    for f in ("--min_particles", "--refine_volumewarp", "--refine_tiltmovies",
              "--ctf_zernike5", "--ctf_phase", "--cpu_memory", "--ctf_batch",
              "--first_iteration_fraction", "--perdevice_preprocess"):
        check(f"m_core exposes {f}",
              any(pp.get("flag") == f for pp in core2["params"]))
    check("min_particles defaults above MCore's dangerous 1",
          int(names["min_particles"]["default"]) >= 20)

    # the sparse-series crash guard
    v2 = {pp["name"]: pp.get("default") for pp in core2["params"]}
    v2.update(population="m/x.population", refine_imagewarp="6x4",
              refine_particles=True, ctf_defocus=True)
    check("warns when min_particles=1 with an image-warp grid",
          "IndexOutOfRange" in core2["validate"](dict(v2, min_particles="1")))
    check("warns when min_particles is blank",
          "IndexOutOfRange" in core2["validate"](dict(v2, min_particles="")))
    check("no warning at min_particles=20", core2["validate"](v2) == "")

    # the documented round-1 command must come out clean
    c1 = st.build_command(core2, dict(v2, ctf_defocusexhaustive=True,
                                      devicelist="1 2 3"), LAUNCH)
    check("round-1 command uses --devicelist", "--devicelist 1 2 3" in c1, c1)
    check("round-1 command sets --min_particles", "--min_particles 20" in c1, c1)
    check("round-1 omits late-stage flags",
          "--refine_mag" not in c1 and "--refine_stageangles" not in c1, c1)

    # adding stage angles + mag on round 1 is what M warns against
    check("4+ refinement params triggers the one-at-a-time warning",
          "ONE AT A TIME" in core2["validate"](
              dict(v2, refine_stageangles=True, refine_mag=True)))


    # ---- Class3D padding is a knob, not a hardcoded 2 -----------------------
    # --pad 2 pads the reconstruction volume to twice the box before the Fourier
    # transform: better interpolation, 8x the volume memory (2 cubed) per class
    # per MPI follower. It was hardcoded, and segfaulted inside libcuda during
    # Maximization on box 112 with 5 classes and 4 GPU followers.
    C3D = next(x for x in st.STAGES if x["id"] == "relion4_class3d")
    pad = next((q for q in C3D["params"] if q["name"] == "PAD"), None)
    check("Class3D exposes PAD", pad is not None)
    check("PAD defaults to 1 (classification only sorts classes)",
          pad["default"] == 1)
    check("PAD is bounded to 1 or 2", pad["min"] == 1 and pad["max"] == 2)
    check("PAD reaches the script as an env var", pad["flag"] == "PAD")
    check("its help explains the memory cost", "memory" in pad["help"].lower())

    cmd = st.build_command(C3D, {**st.stage_defaults(C3D), "PAD": 2})
    check("PAD is emitted as an environment assignment", "PAD=2" in cmd)
    check("the default is emitted too",
          "PAD=1" in st.build_command(C3D, st.stage_defaults(C3D)))

    _sh = (REPO / "ml_relion4_handoff_warp_auto.sh").read_text()
    check("the script no longer hardcodes --pad 2", "--pad 2 " not in _sh)
    check("the script takes PAD from the environment", '--pad "$PAD"' in _sh)
    check("with a default", 'PAD="${PAD:-1}"' in _sh)
    check("and rejects anything but 1 or 2", "PAD must be 1 or 2" in _sh)

    print(f"\n{passed} passed, {failed} failed")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
