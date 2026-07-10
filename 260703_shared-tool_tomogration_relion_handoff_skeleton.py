"""
tomogration :: Warp -> RELION handoff
Implementation skeleton for the fixes in tomogration_warp_to_relion_fixes.md.

Pseudocode-level: fill in the actual subprocess/GUI-scheduling calls to match
tomogration's existing structure. The control flow and the assertions are the point.
"""

from __future__ import annotations
import subprocess, shlex, re
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Config resolved once per run
# ---------------------------------------------------------------------------
@dataclass
class RelionEnv:
    version: int           # 4 or 5, DETECTED not assumed (see detect_relion_target)
    module: str            # e.g. "relion/4.0.1"  or "relion5/5.0.0"
    launch_cmd: list[str]  # ["relion"] for v4, ["relion", "--tomo"] for v5
    export_dim: str        # "--3d" for v4, "--2d" for v5
    ref_needs_prescale: bool  # True for v4 (no auto-resize)


def detect_relion_target(prefer_gpu: bool = True) -> RelionEnv:
    """
    Decide v4 vs v5 by what will actually RUN, not by preference.

    Key fact: on this VM class, relion5 is a container whose CUDA runtime is newer
    than the host driver -> GPU dies with error-code 35. relion4 is native -> GPU ok.
    So: probe GPU on v5; if it fails, fall back to v4. If the driver is later bumped,
    the probe passes and v5 is used automatically. NEVER hardcode "v5 = CPU only".
    """
    if prefer_gpu and gpu_probe_ok("relion5/5.0.0"):
        return RelionEnv(5, "relion5/5.0.0", ["relion", "--tomo"], "--2d", ref_needs_prescale=False)
    # v5 GPU unavailable -> use v4 (native GPU) as the working GPU path
    return RelionEnv(4, "relion/4.0.1", ["relion"], "--3d", ref_needs_prescale=True)


def gpu_probe_ok(module: str) -> bool:
    """
    Run a tiny GPU op inside the given module and look for CUDA error-code 35.
    Return False on error 35 (driver/runtime mismatch) OR any non-zero exit.
    Emit the IT-ticket message on 35 so the failure is actionable, not silent.
    """
    rc, out, err = run_in_module(module, "relion_refine --version")  # replace w/ real minimal GPU kernel
    blob = out + err
    if "error-code 35" in blob or "driver version is insufficient" in blob:
        log_it_ticket()
        return False
    return rc == 0


def log_it_ticket() -> None:
    print("[tomogration] RELION5 GPU blocked: container CUDA runtime > host driver "
          "(535.x / CUDA 12.2). File IT ticket for driver bump. Falling back to RELION4.")


# ---------------------------------------------------------------------------
# Export  (the launch-dir invariant lives here)
# ---------------------------------------------------------------------------
def export_particles(settings: Path, matching_dir: Path, pattern: str,
                     export_dir: Path, output_angpix: float, box: int,
                     diameter: float, env: RelionEnv) -> Path:
    """
    Always use --output_processing == export_dir and --relative_output_paths.
    RELION MUST later be launched from export_dir (see assert_launch_root).
    Returns the particles star path.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    star = export_dir / "matching.star"
    cmd = [
        "WarpTools", "ts_export_particles",
        "--settings", str(settings),
        "--input_directory", str(matching_dir),
        "--input_pattern", pattern,
        "--normalized_coords",
        "--output_star", str(star),
        "--output_processing", str(export_dir),
        "--output_angpix", str(output_angpix),
        "--box", str(box),
        "--diameter", str(diameter),
        "--relative_output_paths",
        env.export_dim,                # --3d (v4) or --2d (v5)
    ]
    run(cmd)                           # run from the Warp settings root, NOT export_dir
    assert_export_complete(star, export_dir, env)
    return star


def assert_export_complete(star: Path, export_dir: Path, env: RelionEnv) -> None:
    """Catch the silent-partial-export -> readStarList crash BEFORE launching RELION."""
    if not star.exists() or star.stat().st_size == 0:
        raise RuntimeError(f"export produced no/empty particles star: {star}")

    n = warp_reported_particle_count(star)     # or re-count rows; must be > 0
    if n == 0:
        raise RuntimeError("export matched 0 particles - check input_pattern / threshold")

    if env.version == 5:  # v5 optimisation-set pointer must be populated
        optset = next(export_dir.glob("*optimisation_set.star"), None)
        if optset is None or not _optset_particles_populated(optset):
            raise RuntimeError(f"blank/missing optimisation set: {optset}")


def _optset_particles_populated(optset: Path) -> bool:
    for line in optset.read_text().splitlines():
        if line.strip().startswith("_rlnTomoParticlesFile"):
            return len(line.split()) >= 2 and line.split()[1].strip() != ""
    return False


# ---------------------------------------------------------------------------
# THE invariant: launch RELION from the dir the star's image paths are relative to
# ---------------------------------------------------------------------------
def assert_launch_root(star: Path, project_root: Path) -> None:
    """
    star image paths are relative to export_dir. project_root MUST == export_dir,
    else every subtomo path is wrong by the offset (the recurring 'does not exist').
    """
    m = re.search(r"(\S+\.mrc)", star.read_text())
    if not m:
        raise RuntimeError(f"no image path found in {star}")
    first = m.group(1)
    if not (project_root / first).is_file():
        raise RuntimeError(
            f"PATH ROOT MISMATCH: '{first}' does not resolve from {project_root}. "
            f"Launch RELION from the --output_processing dir ({star.parent})."
        )


# ---------------------------------------------------------------------------
# Reference prep (v4 has no auto-resize)
# ---------------------------------------------------------------------------
def prepare_reference(ref_map: Path, src_angpix: float, output_angpix: float,
                      box: int, env: RelionEnv, out_dir: Path) -> Path:
    """v4: rescale/rebox the EMDB map to match particles. v5: return as-is (auto-resize)."""
    if not env.ref_needs_prescale:
        return ref_map
    out = out_dir / f"ref_{output_angpix:g}apx.mrc"
    run(["relion_image_handler", "--i", str(ref_map),
         "--angpix", str(src_angpix),
         "--rescale_angpix", str(output_angpix),
         "--new_box", str(box),
         "--o", str(out)])
    return out
    # Alternative if you don't pre-scale: pass "--trust_ref_size" in job Additional args
    # (only safe because the size delta is a known, deliberate downscale).


# ---------------------------------------------------------------------------
# Project hygiene: one clean project dir per RELION version
# ---------------------------------------------------------------------------
def clean_project_dir(root: Path) -> None:
    """
    Never launch one RELION version inside another's project (fn_img / in_optimisation
    crashes). Park cross-version pipeline state before launch.
    """
    stale = ["default_pipeline.star", ".Nodes", ".TMP_runfiles"]
    stale += [p.name for p in root.glob(".gui_*job.star")]
    junk = root / "_parked"
    for name in stale:
        p = root / name
        if p.exists():
            junk.mkdir(exist_ok=True)
            p.rename(junk / p.name)


# ---------------------------------------------------------------------------
# MPI sizing
# ---------------------------------------------------------------------------
def mpi_procs_for(n_gpu: int) -> int:
    """nGPU + 1: one non-GPU leader + one worker per GPU. Underfeeding = idle cards."""
    return n_gpu + 1


# ---------------------------------------------------------------------------
# Picking params (correctness guards, not tuning)
# ---------------------------------------------------------------------------
def validate_picking(diameter: float, peak_distance: float | None) -> None:
    # peak_distance defaults FROM diameter when unset -> forces you to think about NMS radius.
    if peak_distance is None:
        raise ValueError("set peak_distance explicitly (< inter-particle spacing); "
                         "do not let it inherit from diameter")
    if peak_distance >= diameter:
        # for dense clusters peak_distance must be BELOW centre-to-centre spacing
        log("warn: peak_distance >= diameter may suppress adjacent particles")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_classification(cfg) -> None:
    env = detect_relion_target(prefer_gpu=cfg.prefer_gpu)

    validate_picking(cfg.diameter, cfg.peak_distance)

    export_dir = cfg.project_parent / ("relion%d" % env.version) / "warp"
    star = export_particles(cfg.settings, cfg.matching_dir, cfg.pattern,
                            export_dir, cfg.output_angpix, cfg.box, cfg.diameter, env)

    project_root = export_dir              # <-- the invariant
    assert_launch_root(star, project_root)
    clean_project_dir(project_root)

    ref = prepare_reference(cfg.ref_map, cfg.ref_angpix, cfg.output_angpix,
                            cfg.box, env, out_dir=export_dir)

    n_gpu = cfg.n_gpu if env.version == 4 or env.gpu_ok else 0
    submit_class3d(
        project_root=project_root,
        launch=env.launch_cmd,
        particles=star.name,               # relative to project_root
        reference=ref,
        sym="C1",                          # clean in C1, symmetrise (C3) only at refine
        ini_lowpass=45,
        use_gpu=bool(n_gpu),
        gpus="0,1,2,3",
        mpi=mpi_procs_for(n_gpu) if n_gpu else cfg.cpu_mpi,
        threads=4,
    )


# ---- thin helpers (wire to tomogration's real runner) ---------------------
def run(cmd): ...
def run_in_module(module, cmd): ...
def submit_class3d(**kw): ...
def warp_reported_particle_count(star): ...
def log(*a): print(*a)
