#!/usr/bin/env python3
# Part of tomogration2 — split out of the single-file app so the pieces can be
# edited (and tested) independently. Import graph is a strict DAG:
#     core  ->  stages  ->  jobs        (project depends only on core)
# tomogration_app.py imports all of them and holds the Qt window.

"""The job model: the .tomogration_jobs.json store, the canvas DAG layout, and
discovery of work done outside the app. THE QUEUE IS JUST JOBS (status='queued'),
which is what makes queued work a card, restartable, and persistent."""

import datetime
import itertools
import json
import os
import re
from pathlib import Path

from tomogration_stages import STAGES, TRUNK_STAGES, build_command

JOBS_FILE = ".tomogration_jobs.json"

def jobs_path(root):
    return Path(root) / JOBS_FILE


def load_jobs(root):
    """The job store {'seq': int, 'jobs': {id: job}} — empty scaffold if missing
    or unreadable (same defensive contract as load_history)."""
    p = jobs_path(root)
    if not p.is_file():
        return {"seq": 0, "jobs": {}}
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        return {"seq": 0, "jobs": {}}
    if not isinstance(data, dict) or not isinstance(data.get("jobs"), dict):
        return {"seq": 0, "jobs": {}}
    data.setdefault("seq", len(data["jobs"]))
    return data


def save_jobs(root, store):
    try:
        jobs_path(root).write_text(json.dumps(store, indent=1))
    except OSError:
        pass


# The five states a job moves through. BUILDING is the one that was missing: a
# card created by "Build downstream" or dropped from the palette was born already
# 'queued', so a job nobody had finished configuring could start on its own the
# moment the queue drained. Building means "on the canvas, not going anywhere
# until you say so".
JOB_STATES = ("building", "queued", "running", "completed", "failed")


def queued_jobs(store):
    """Jobs waiting to run, in run order (creation order = J-number order).

    THE QUEUE IS JUST JOBS. It used to be a separate in-memory list of
    (label, command, stage_id) tuples, which is why queued work could never appear
    on the canvas, could not be restarted, and evaporated when the app closed. A
    queued item is now an ordinary job record with status='queued' and its command
    already resolved, so every job affordance (card, details, restart, clone,
    delete, persistence) applies to it for free."""
    jobs = store.get("jobs", {}) if isinstance(store, dict) else {}
    return [jobs[j] for j in sorted(jobs, key=_job_seq)
            if (jobs[j] or {}).get("status") == "queued"]


def reconcile_running(root):
    """Mark orphaned 'running' jobs as interrupted. Call once at startup.

    Job status is persisted, but a RUNNING job only exists while the app that
    launched it is alive. If the app is killed, freezes and is force-quit, or the
    machine reboots, those records stay 'running' forever — the canvas then shows
    several jobs as live at once and the queue refuses to start anything, because
    it believes something is already going. Nothing survives the process, so at
    startup any 'running' job is by definition finished-unknown."""
    store = load_jobs(root)
    jobs = store.get("jobs", {}) or {}
    hit = [j for j in jobs.values() if j.get("status") == "running"]
    for j in hit:
        j["status"] = "failed"
        j["exit_code"] = None
        j["interrupted"] = True
    if hit:
        save_jobs(root, store)
    return [j["id"] for j in hit]


def job_output_dir(job_id):
    """A job's processing dir, relative to the project root (cwd of every run)."""
    return f"jobs/{job_id}"


def new_job(root, stage_id, label, params, inputs=None):
    """Create + persist a fresh job; return the record. `inputs` maps an input
    slot name -> the parent job id feeding it (or None = read the project trunk /
    the .settings default). Ids are monotonic 'J<seq>' so they never collide even
    after deletions."""
    store = load_jobs(root)
    store["seq"] = int(store.get("seq", 0)) + 1
    jid = f"J{store['seq']}"
    job = {
        "id": jid,
        "stage_id": stage_id,
        "label": label,
        "params": dict(params or {}),
        "inputs": dict(inputs or {}),
        "output_dir": job_output_dir(jid),
        "command": "",
        "status": "building",
        "exit_code": None,
        "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "started": None,
        "finished": None,
        "summary": {},
    }
    store["jobs"][jid] = job
    save_jobs(root, store)
    return job


def update_job(root, job_id, **fields):
    """Merge `fields` into a stored job; return it (or None if unknown id)."""
    store = load_jobs(root)
    job = store.get("jobs", {}).get(job_id)
    if job is None:
        return None
    job.update(fields)
    save_jobs(root, store)
    return job


def is_warp_stage(spec):
    """True for stages whose command is a WarpTools subcommand — the ones that
    accept --input_processing / --output_processing for free (BaseCommand)."""
    return str(spec.get("base", "")).startswith("WarpTools")


def parent_job_id(job):
    """The single upstream job whose processing dir this job reads: the
    'processing' input slot, else the first slot carrying a job id."""
    inputs = job.get("inputs", {}) or {}
    if inputs.get("processing"):
        return inputs["processing"]
    for v in inputs.values():
        if v:
            return v
    return None


def io_flags_for_job(spec, job, store):
    """Extra tokens wiring a WarpTools job to its own processing dir (and its
    parent's, if any). Wrapper stages return '' — they resolve in/out dirs by
    their own means. A parent with no job id (trunk) yields no --input_processing,
    so the run reads the .settings default, exactly like the three-column view.

    threshold_picks is special: it reads AND writes its match stars in
    <output_processing>/matching in place, so its inputs are STAGED into its own
    dir (_prepare_job_inputs) and it only takes --output_processing (a stale
    --input_processing at the parent, which has no .xml, would look for previous
    results in the wrong place)."""
    if not is_warp_stage(spec):
        return ""
    if spec.get("id") in TRUNK_STAGES:
        return ""            # trunk-only: reads shared reconstructions, suffix-distinct output
    toks = []
    pid = parent_job_id(job)
    parent = store.get("jobs", {}).get(pid) if pid else None
    if parent and spec.get("id") != "threshold_picks":
        toks.append(f"--input_processing {parent['output_dir']}")
    toks.append(f"--output_processing {job['output_dir']}")
    return " ".join(toks)


def build_job_command(spec, job, store, warp_cmd=None, group_inputs=None):
    """A job's exact command = the pure stage command (build_command, unchanged)
    plus this job's processing-dir wiring. Manual --*_processing already typed
    into the params wins (we don't double it)."""
    cmd = build_command(spec, job.get("params", {}), warp_cmd, group_inputs)
    extra = io_flags_for_job(spec, job, store)
    if extra and "--output_processing" not in cmd and "--input_processing" not in cmd:
        cmd = f"{cmd} {extra}"
    # {jobid} in any param resolves to THIS job's id, so a stage whose output must
    # live outside jobs/ (the RELION handoff needs one project dir with the star +
    # subtomo/) still gets a per-job, collision-free home: relion4/{jobid} ->
    # relion4/J13. No shared default for a second run to overwrite.
    cmd = cmd.replace("{jobid}", job.get("id", ""))
    return cmd


def _jobnum(jid):
    """Numeric part of a 'J<seq>' id, for newest-first ordering."""
    try:
        return int(str(jid).lstrip("J"))
    except ValueError:
        return 0


def default_parent_for(stage_id, store):
    """When building a new job for `stage_id`, pick a sensible default input: the
    newest WarpTools job of the nearest preceding WarpTools stage (only warp
    stages have a processing dir worth reading via --input_processing). None means
    'read the project trunk / .settings default', like the three-column view."""
    order = [s["id"] for s in STAGES]
    if stage_id not in order:
        return None
    jobs = store.get("jobs", {})
    for up in reversed(order[:order.index(stage_id)]):
        spec = next((s for s in STAGES if s["id"] == up), None)
        if not spec or not is_warp_stage(spec):
            continue
        cands = [jid for jid, j in jobs.items() if j.get("stage_id") == up]
        if cands:
            return max(cands, key=_jobnum)
    return None


# Directories that must NEVER be deleted by a job wipe, even if a job names one as
# its output. These hold the raw data and the shared Warp/RELION state — losing one
# costs days of reprocessing, and a job's "output dir" can legitimately point at a
# shared location (ts_export_particles writes into relion4/, several jobs share it).
PROTECTED_DIRS = {
    "", ".", "frames", "mdocs", "gains", "tomostar", "Thumbnails", "logs",
    "warp_frameseries", "warp_tiltseries", "relion4", "m", "jobs", "selected",
    "aretomo_output", "Class3D", "Refine3D", "Select", "InitialModel",
    "MaskCreate", "PostProcess", "CtfRefine", "LocalRes", "Trash",
}


def declared_output_dirs(root, job):
    """The project-relative directories a job's parameters say it writes to.

    Resolves {jobid}, which build_job_command substitutes at run time — without
    that, a job-scoped path like relion4/{jobid} matches nothing and the job looks
    like it owns no output at all.
    """
    spec = next((s for s in STAGES if s["id"] == job.get("stage_id")), {})
    out = set()
    for key in spec.get("output_params") or []:
        rel = str((job.get("params") or {}).get(key, "") or "").strip().rstrip("/")
        if not rel:
            continue
        rel = rel.replace("{jobid}", str(job.get("id", "")))
        if os.path.isabs(rel):
            try:
                rel = os.path.relpath(rel, Path(root))
            except ValueError:
                continue
        out.add(rel)
    return out


def job_delete_targets(root, job_id, stage_id, params, output_params=None,
                       store=None):
    """What deleting or clearing this job would remove from disk.

    Returns (targets, skipped): both lists of project-relative paths. `targets` is
    what may be deleted; `skipped` is what was refused and why. Deliberately
    conservative — the raw-data directories must never be reachable from here at
    all. Pure function so the rules are testable rather than trusted.

    Pass `store` and it also refuses any directory ANOTHER job writes to. Several
    exports legitimately share one RELION project dir, so clearing a failed job
    could delete the successful job's particles sitting beside it — the outcome
    that makes "clear and re-run with different parameters" unsafe to offer at
    all. A job's own jobs/<id>/ is unique by construction and always deletable.
    """
    root = Path(root)
    targets, skipped = [], []

    jd = job_output_dir(job_id)                     # jobs/J###  — always this job's own
    if (root / jd).is_dir():
        targets.append(jd)

    # Directories some OTHER job also claims, mapped to who claims them.
    claimed = {}
    for other_id, other in ((store or {}).get("jobs", {}) or {}).items():
        if other_id == job_id:
            continue
        for rel in declared_output_dirs(root, other):
            claimed.setdefault(rel, []).append(other_id)

    for key in (output_params or []):
        rel = str((params or {}).get(key, "") or "").strip().rstrip("/")
        if not rel:
            continue
        rel = rel.replace("{jobid}", str(job_id))
        if os.path.isabs(rel):
            try:
                rel = os.path.relpath(rel, root)
            except ValueError:
                skipped.append(f"{rel} (outside the project)")
                continue
        if rel.startswith("..") or rel in PROTECTED_DIRS:
            skipped.append(f"{rel} (protected)")
            continue
        if rel in claimed:
            who = ", ".join(sorted(claimed[rel]))
            skipped.append(f"{rel} (also written by {who})")
            continue
        if not (root / rel).is_dir():
            continue
        if rel not in targets:
            targets.append(rel)
    return targets, skipped


def delete_job(root, job_id):
    """Remove a job record from the store (leaves any jobs/<id>/ dir on disk —
    outputs are never auto-deleted). Returns True if a record was removed."""
    store = load_jobs(root)
    if job_id in store.get("jobs", {}):
        del store["jobs"][job_id]
        save_jobs(root, store)
        return True
    return False


def fmt_angpix(a):
    """Warp names reconstructions/matches with a 2-decimal pixel size, e.g. '10' ->
    '10.00', '12.56' -> '12.56' (Position042_12.56Apx.mrc)."""
    try:
        return f"{float(a):.2f}"
    except (TypeError, ValueError):
        return str(a)


def template_corr_suffix(params):
    """Suffix on the CORRELATION VOLUME (_corr.mrc) + score maps: ALWAYS the
    template-derived name (_emd_<code> / _<stem>). The corr volume is the
    template×tomogram cross-correlation, so --override_suffix does NOT rename it —
    only the peak-list STAR. (Confirmed on disk: v2 stars alongside _emd_70905_corr.mrc.)"""
    emdb = str(params.get("template_emdb", "") or "").strip()
    if emdb:
        return f"_emd_{emdb}"
    tp = str(params.get("template_path", "") or "").strip()
    if tp:
        return "_" + os.path.splitext(os.path.basename(tp))[0]
    return ""


def template_match_suffix(params):
    """The STAR suffix a ts_template_match run writes: an explicit --override_suffix
    if set (used verbatim, leading underscore and all), else the template-derived
    name (same as the corr volume)."""
    ov = str(params.get("override_suffix", "") or "").strip()
    return ov if ov else template_corr_suffix(params)


def match_star_infix(params):
    """threshold_picks --in_suffix = the WHOLE middle of a template-match star name,
    '<tomo_angpix>Apx<suffix>' (e.g. 12.56Apx_v3-optimized) — because WarpTools looks
    for {item}_{in_suffix}.star. NOT just the override suffix (that's the recurring
    'No files found matching PositionNNN_<suffix>.star.star' trap)."""
    return f"{fmt_angpix(params.get('tomo_angpix', ''))}Apx{template_match_suffix(params)}"


# Valid next stages for "Build downstream" from a job card (the DAG's forward
# edges among the forkable back-half stages).
DOWNSTREAM = {
    "ts_ctf": ["ts_reconstruct"],
    "ts_reconstruct": ["ts_template_match"],
    "ts_template_match": ["threshold_picks", "ts_export_particles"],
    "threshold_picks": ["ts_export_particles"],
    "ts_export_particles": ["relion4_convert"],
    "relion4_convert": ["relion4_class3d"],
    "relion4_result": ["m_mask_create", "m_create_species",
                       "relion4_to_warp", "relion4_select_picks"],
    # The RELION→Warp converters produce a folder of per-series pick stars, and the
    # only thing that consumes those is an export. Without these edges the export
    # could not be built downstream of a selection at all: it fell back to "newest
    # threshold_picks", so a re-extraction silently inherited the wrong pick set,
    # the wrong pixel size and the previous round's output paths.
    "relion4_select_picks": ["ts_export_particles"],
    "relion4_to_warp": ["ts_export_particles"],
    "m_create_population": ["m_create_source"],
    "m_create_source": ["m_create_species"],
    "m_mask_create": ["m_create_species"],
    "m_create_species": ["m_core"],
    "m_core": ["m_core", "m_estimate_weights", "m_resample_trajectories"],
    "m_estimate_weights": ["m_core"],
    "m_resample_trajectories": ["m_core"],
}


def _picktag(in_suffix):
    """Short, filesystem-safe pick-set tag from a threshold in_suffix, stripping the
    '<angpix>Apx' prefix + leading underscore: '12.56Apx_v3-optimized' ->
    'v3-optimized'. Used to name each export's RELION dir so it maps to its pick set."""
    t = re.sub(r"^[\d.]+Apx", "", in_suffix or "").lstrip("_")
    return t or "picks"


def derive_child_params(child_stage, parent_stage, parent_params, parent_output_dir=""):
    """Params a downstream job should inherit from its chosen parent, so wiring
    'J5 -> threshold -> export -> convert -> Class3D' auto-fills the fiddly
    suffix/pattern/dir instead of the user reverse-engineering it. Two things this
    threads: ts_export_particles reads pick STARs from --input_directory (NOT
    --input_processing), so it points at the parent job's matching dir; and each
    export writes into a pick-set-named RELION dir (relion4/<tag>/) so multiple
    exports never collide and the RELION input path maps to its pick set."""
    # A RELION job adopted from disk feeds M. These are the three paths that are
    # tedious and error-prone to copy by hand (and where using a FILTERED half map
    # by mistake silently invalidates the FSC), so carry the ones adoption actually
    # found on disk rather than reconstructing filenames.
    if parent_stage == "relion4_result":
        jd = str(parent_params.get("job_dir", "") or "")
        if child_stage == "m_mask_create":
            src = parent_params.get("class_map") or ""
            return {"i": src, "o": "m/mask.mrc"} if src else {}
        if child_stage == "m_create_species":
            out = {"mask": "m/mask.mrc"}
            for k in ("half1", "half2"):
                if parent_params.get(k):
                    out[k] = parent_params[k]
            if parent_params.get("data_star"):
                out["particles_relion"] = parent_params["data_star"]
            return out
        if child_stage in ("relion4_to_warp", "relion4_select_picks"):
            star = parent_params.get("data_star") or ""
            key = "particles_star" if child_stage == "relion4_to_warp" else "class_star"
            return {key: star} if star else {}
        return {}

    # A promoted RELION SELECTION card. It rides the ts_template_match stage (so it
    # lands in the Pick row) but is really a RELION job, and its star is in
    # source_star. Without this, building a converter downstream of it derived
    # NOTHING — the new card had no particle star, and the script died on its one
    # required positional argument.
    if parent_params.get("source_star") and child_stage in ("relion4_to_warp",
                                                            "relion4_select_picks"):
        key = "particles_star" if child_stage == "relion4_to_warp" else "class_star"
        out = {key: parent_params["source_star"]}
        if child_stage == "relion4_to_warp":
            # These stars carry refined origins and Eulers, which is exactly what
            # MODE C exists to use.
            out["relion_coords"] = True
        return out

    if child_stage == "threshold_picks" and parent_stage == "ts_template_match":
        return {"in_suffix": match_star_infix(parent_params)}
    # An export fed by a RELION→Warp converter. Every one of these five values was
    # previously carried over by hand from the last round, and a re-extraction went
    # out with the previous round's directory, pattern, pixel size and output paths —
    # extracting 4x off-origin into the wrong project folder, exit 0, no error.
    # The converter already knows all of them.
    if child_stage == "ts_export_particles" and parent_stage in (
            "relion4_select_picks", "relion4_to_warp"):
        out_dir = str(parent_params.get("out_dir", "") or "").rstrip("/")
        suffix = str(parent_params.get("suffix", "") or "")
        capx = str(parent_params.get("coords_angpix", "") or "").strip()
        tag = _picktag(suffix) or "picks"
        # {jobid} resolves to the new job's own id, so two exports of the same pick
        # set never share a directory. Sharing one is what made "clear this job"
        # able to delete a DIFFERENT job's particles.
        outdir = f"relion4/{tag}_{{jobid}}"
        derived = {"input_directory": out_dir or "picks",
                   "input_pattern": f"*{suffix}.star" if suffix else "*.star",
                   "output_processing": outdir,
                   "output_star": f"{outdir}/matching.star",
                   "normalized_coords": False}
        if capx:
            # The converter states the pixel size its coordinates are in, and names
            # its files with it. Carrying it removes the single most damaging
            # hand-copied value in the pipeline.
            derived["coords_angpix"] = capx
        return derived
    if child_stage == "ts_export_particles":
        mdir = f"{parent_output_dir}/matching" if parent_output_dir else "warp_tiltseries/matching"
        if parent_stage == "threshold_picks":
            infix = parent_params.get("in_suffix", "")
            out = parent_params.get("out_suffix", "clean")
            pat = f"*{infix}_{out}.star"
        else:                                   # straight from template matching
            infix = match_star_infix(parent_params)
            pat = f"*{infix}.star"
        outdir = f"relion4/{_picktag(infix)}_{{jobid}}"   # e.g. relion4/v3_J14
        derived = {"input_directory": mdir, "input_pattern": pat,
                   "output_processing": outdir, "output_star": f"{outdir}/matching.star"}
        # A RELION-derived pick set (source_star present) holds ABSOLUTE pixel coords
        # at the pixel size in its filenames — unlike Warp's own and crYOLO's picks,
        # which are 0-1 fractions. Carry that, because the two conventions look
        # identical in the form and picking the wrong one extracts from empty space.
        if parent_params.get("source_star") and parent_params.get("tomo_angpix"):
            derived["coords_angpix"] = str(parent_params["tomo_angpix"])
            derived["normalized_coords"] = False
        return derived
    if child_stage == "relion4_convert" and parent_stage == "ts_export_particles":
        outdir = parent_params.get("output_processing", "relion4/warp")
        # resolve the export's {jobid} to its concrete id so convert runs in the
        # SAME dir the export wrote to (relion4/J13), not convert's own job id.
        outdir = outdir.replace("{jobid}", os.path.basename(parent_output_dir or ""))
        return {"project_dir": outdir,
                "starfile": os.path.basename(parent_params.get("output_star", "matching.star"))}
    if child_stage == "relion4_class3d" and parent_stage == "relion4_convert":
        outdir = parent_params.get("project_dir", "relion4/warp")
        base = os.path.splitext(os.path.basename(parent_params.get("starfile", "matching.star")))[0]
        return {"project_dir": outdir, "particles": f"{base}_conv.star"}
    return {}


# ---- per-stage result summaries (the one-line card readout) ----------------
# summarize_job(stage_id, job_abs_dir) -> {label: value}. A card must NEVER
# enumerate thousands of ceph files on the Qt thread (that crash is why
# ask_project_root / the History 40-cap exist), so counts are bounded and the
# specific parsers touch only a small fixed set of fields. All defensive: any
# error -> {}. Stage-specific parsers (ts_ctf defocus, tomogram counts, pick
# scores) are registered in STAGE_SUMMARIZERS as the real Warp XML/STAR layout is
# confirmed against a VM sample; until then every stage uses summarize_generic.
def _count_glob(dir_path, pattern, cap=5000):
    """(count, capped) for files matching pattern; stops at cap so a huge ceph
    dir never blocks the caller."""
    n = 0
    try:
        for _ in Path(dir_path).glob(pattern):
            n += 1
            if n >= cap:
                return n, True
    except OSError:
        pass
    return n, False


def summarize_generic(job_dir):
    """Cheap fallback: how many of each product type the job wrote."""
    d = Path(job_dir)
    if not d.is_dir():
        return {}
    out = {}
    for ext, key in ((".mrc", "mrc"), (".star", "star"), (".xml", "xml")):
        n, capped = _count_glob(d, f"*{ext}")
        if n:
            out[key] = f"{n}+" if capped else n
    return out


# One <CTF> block per series XML holds the fitted per-series AVERAGE defocus as
# `<Param Name="Defocus" Value="5.25432" />` (µm); DefocusDelta = astigmatism mag,
# DefocusAngle = its angle. The per-TILT values live in <GridCTF><Node Value=.../>
# (no Name= attr), so `Name="Defocus"` matches the scalar uniquely. The <CTF> block
# sits near the top, before the long GridCTF node lists, so a bounded head read
# gets it without parsing the whole (~430 KB) file. (VM sample 2026-07-10.)
_CTF_DEFOCUS_RE = re.compile(r'Name="Defocus"\s+Value="([-\d.eE]+)"')
_APX_RE = re.compile(r'_([\d.]+)Apx\.mrc$')


def _mean_std(vals):
    m = sum(vals) / len(vals)
    return m, (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5


def summarize_ts_ctf(job_dir):
    """series count + mean±std of the per-series average defocus (µm)."""
    d = Path(job_dir)
    if not d.is_dir():
        return {}
    vals = []
    for xml in itertools.islice(sorted(d.glob("*.xml")), 0, 5000):
        try:
            with xml.open("r", errors="ignore") as fh:
                head = fh.read(16384)               # <CTF> is well within this
        except OSError:
            continue
        m = _CTF_DEFOCUS_RE.search(head)
        if m:
            try:
                vals.append(float(m.group(1)))
            except ValueError:
                pass
    if not vals:
        return {}
    mean, std = _mean_std(vals)
    return {"series": len(vals), "defocus_um": f"{mean:.2f} ± {std:.2f}"}


def summarize_ts_reconstruct(job_dir):
    """tomogram count (+ pixel size parsed from the ..._<N>Apx.mrc name). A job's
    tomograms land in <job>/reconstruction/ (confirmed on the VM branch test)."""
    rec = Path(job_dir) / "reconstruction"
    if not rec.is_dir():
        return {}
    n, apx = 0, None
    for mrc in itertools.islice(rec.glob("*.mrc"), 0, 5000):
        n += 1
        if apx is None:
            m = _APX_RE.search(mrc.name)
            if m:
                apx = m.group(1)
    if not n:
        return {}
    out = {"tomograms": n}
    if apx:
        out["angpix"] = apx
    return out


STAGE_SUMMARIZERS = {
    "ts_ctf": summarize_ts_ctf,
    "ts_reconstruct": summarize_ts_reconstruct,
}


def summarize_job(stage_id, job_dir):
    fn = STAGE_SUMMARIZERS.get(stage_id, summarize_generic)
    try:
        return fn(job_dir) or {}
    except Exception:
        return {}


# ===========================================================================
# Card-canvas layout (Phase 2) — PURE (no Qt), so it's unit-testable. Maps the
# job store + STAGES onto positioned nodes + edges the QGraphicsView draws.
# ===========================================================================
CARD_W, CARD_H = 210, 92          # card box size (scene units)
GAP_X, GAP_Y = 44, 30             # spacing between forks (x) and stages (y)
# The DEFAULT PIPELINE lives in a fixed rail down the left edge — one template card
# per stage, always present, never movable. Real jobs start to the right of it and
# can never be placed inside it, so the template stays readable however tangled the
# actual project gets. Before this, a stage's template card and a user-placed job
# card competed for the same coordinates.
RAIL_GUTTER = 72
RAIL_W = CARD_W + RAIL_GUTTER     # x at which the working canvas begins


# MCore's only report of what a refinement achieved is one stdout line at the very
# end — "EML45-spike-closed_HWhelp: 6.71 Å" — written nowhere on disk. Without it a
# row of M cards is indistinguishable, and comparing runs means scrolling the log.
# Kept pure and here (not in the Qt file) so it can be unit-tested.
_M_RES_RE = re.compile(r"^\s*(\S.*?):\s*([0-9]+(?:\.[0-9]+)?)\s*(?:Å|A)\s*$")


def m_resolution(line):
    """The resolution in Å from an MCore result line, or None.

    Deliberately strict about the trailing unit: MCore prints plenty of other
    "label: number" lines during a run, and picking one of those up would label a
    card with a defocus or an iteration count.
    """
    m = _M_RES_RE.match(line or "")
    if not m:
        return None
    # "Global resolution is 10.000" and friends have no colon; a species name will
    # not be empty. Guard against a stray line that happens to fit the shape.
    name = m.group(1).strip()
    if not name or len(name) > 120:
        return None
    try:
        return float(m.group(2))
    except ValueError:
        return None


# ===========================================================================
# Where a job's results ACTUALLY landed
#
# jobs/<id>/ is a convention, not a fact. Only WarpTools stages that take
# --output_processing write there; every wrapper stage writes wherever its own
# parameters point, and M is the worst case — it writes a .population file, a
# .source next to the SETTINGS, and one randomly-named species version folder per
# refinement round, none of it under jobs/<id>. Showing "jobs/J64" for an MCore
# run is simply wrong, and the folder is usually empty.
# ===========================================================================
TS_FMT = "%Y-%m-%d %H:%M:%S"


def parse_ts(text):
    """A job-store timestamp -> datetime, or None."""
    try:
        return datetime.datetime.strptime(str(text), TS_FMT)
    except (TypeError, ValueError):
        return None


def species_version_dirs(root):
    """Every m*/species/*/versions/* folder in the project."""
    try:
        return sorted(d for d in Path(root).glob("m*/species/*/versions/*")
                      if d.is_dir())
    except OSError:
        return []


def folder_time(d):
    """When the contents of a folder were last written.

    The directory's own mtime moves whenever anything is added, so prefer the
    newest entry INSIDE it — that is the moment M committed the version.

    Deliberately one level deep (scandir, not rglob). This runs over every
    version folder in the project, including anything sitting in a multi-GB
    m_trash_*/, and walking those on ceph is how this app has frozen before.
    """
    newest = None
    try:
        with os.scandir(d) as it:
            for e in it:
                try:
                    t = e.stat().st_mtime
                except OSError:
                    continue
                if newest is None or t > newest:
                    newest = t
    except OSError:
        pass
    if newest is None:
        try:
            newest = os.stat(d).st_mtime
        except OSError:
            return None
    return datetime.datetime.fromtimestamp(newest)


def versions_for_job(root, job, slack_s=900):
    """The species version folders a given job wrote, project-relative.

    M leaves no link between a run and the folder it produced, so time is the
    only available join: a version folder belongs to the job that was running
    when it was written. `slack_s` allows for the version being committed as the
    run's last act, slightly after the store's recorded finish time.
    """
    started = parse_ts(job.get("started"))
    if started is None:
        return []
    finished = parse_ts(job.get("finished"))
    out = []
    for d in species_version_dirs(root):
        when = folder_time(d)
        if when is None or when < started:
            continue
        if finished is not None and (when - finished).total_seconds() > slack_s:
            continue
        try:
            out.append(str(d.relative_to(Path(root))))
        except ValueError:
            continue
    return sorted(out)


# Warp tools write into the PROCESSING FOLDER named inside the .settings file, not
# next to it and not into jobs/<id>. No parameter spells it out, so ts_reconstruct
# could only ever advertise an empty job folder while its tomograms went to
# warp_tiltseries/reconstruction/.
_PROC_PARAM_RE = re.compile(r'<Param\s+Name="([^"]+)"\s+Value="([^"]*)"', re.I)


def settings_processing_dir(root, settings_rel):
    """The processing folder a .settings file points at, project-relative, or "".

    Reads it out of the XML when the key is recognisable, and otherwise falls back
    to Warp's own naming convention (warp_tiltseries.settings -> warp_tiltseries/).
    Only ever returns a directory that exists — guessing a path for a button that
    opens folders is worse than returning nothing.
    """
    root = Path(root)
    raw = str(settings_rel or "").strip()
    if not raw:
        return ""
    p = Path(raw)
    if not p.is_absolute():
        p = root / p

    def ok(value):
        if not value:
            return ""
        d = Path(value)
        if not d.is_absolute():
            d = root / d
        try:
            return os.path.relpath(d, root) if d.is_dir() else ""
        except ValueError:
            return ""

    try:
        text = p.read_text(errors="replace")
    except OSError:
        text = ""
    for name, value in _PROC_PARAM_RE.findall(text):
        if "processing" in name.lower() and "folder" in name.lower():
            rel = ok(value)
            if rel:
                return rel
    # Convention: the settings' own stem names the processing folder.
    return ok(p.stem)


def job_real_outputs(root, job, spec):
    """Where this job's results are, as [(project_relative_dir, note)].

    Ordered most-specific first, and every entry is a directory that EXISTS —
    the details pane opens these, so a path that is merely plausible is worse
    than none. A parameter naming a FILE contributes its parent directory, with
    the file named in the note, because that is what there is to open.
    """
    root = Path(root)
    job = job or {}
    spec = spec or {}
    params = job.get("params", {}) or {}
    out, seen = [], set()

    def add(rel, note="", allow_root=False):
        rel = str(rel).rstrip("/") or "."
        # "." is the project root. For a param naming a file that happens to sit at
        # the root it is noise ("this job wrote to your whole project"); for a file
        # we went looking for and FOUND there it is the answer.
        if rel == ".." or rel in seen or (rel == "." and not allow_root):
            return
        if not (root / rel).is_dir():
            return
        seen.add(rel)
        out.append((rel, note))

    # Per-round M version folders: the one output that identifies THIS run.
    for rel in versions_for_job(root, job):
        add(rel, "this run's species version (M names it randomly)")

    for key in spec.get("output_params") or []:
        raw = str(params.get(key, "") or "").strip()
        if not raw:
            continue
        p = Path(raw)
        if not p.is_absolute():
            p = root / p
        try:
            rel = os.path.relpath(p, root)
        except ValueError:
            continue
        if p.is_dir():
            add(rel)
        elif p.is_file():
            add(os.path.dirname(rel), f"holds {p.name}")
        else:
            # Named but absent: a file M has not written yet, or a path typo.
            # Its parent is still the right place to look.
            add(os.path.dirname(rel), f"{p.name} is not there (yet)")

    # Warp tools write into a named SUBFOLDER of the processing folder
    # (reconstruction/, matching/, subtomo/ …). --output_processing overrides where
    # that is, so honour it before falling back to the .settings file.
    subdirs = spec.get("output_subdirs") or []
    if subdirs:
        proc = str(params.get("output_processing", "") or "").strip()
        proc = (os.path.relpath(root / proc, root) if proc
                else settings_processing_dir(
                    root, params.get(spec.get("settings_param", "settings"), "")))
        if proc:
            for sub in subdirs:
                add(f"{proc}/{sub}", "where the files actually land")
            add(proc)

    # Some outputs cannot be derived from a parameter at all. MTools create_source
    # writes <name>.source into the PROCESSING folder named inside the .settings
    # file — not beside the settings, and not into m/ — so the only honest way to
    # find it is to look. Bounded to two levels; the project root holds frames/.
    pattern = spec.get("output_find")
    if pattern:
        name = _subst_params(pattern, params)
        if name and "{" not in name:
            for cand in itertools.chain(root.glob(name), root.glob(f"*/{name}")):
                if cand.is_file():
                    add(os.path.relpath(cand.parent, root), f"holds {cand.name}",
                        allow_root=True)
    return out


def _subst_params(pattern, params):
    """Fill {param} placeholders in an output pattern from a job's params."""
    out = pattern
    for key, val in (params or {}).items():
        out = out.replace("{" + str(key) + "}", str(val))
    return out.strip()


# ===========================================================================
# Naming adopted RELION jobs
#
# A card reading "RELION selection / Select/job009 · used" says almost nothing:
# not what kind of job it was, not which one, not how big. RELION's own folder
# names carry the type and number, and the star carries the particle count — so
# read all three and put them on the card.
# ===========================================================================
RELION_JOB_TITLES = {
    "Select": "Subset selection",
    "Class3D": "3D classification",
    "Class2D": "2D classification",
    "Refine3D": "3D refinement",
    "InitialModel": "Initial model",
    "Extract": "Particle extraction",
    "MaskCreate": "Mask",
    "PostProcess": "Post-processing",
    "CtfRefine": "CTF refinement",
    "LocalRes": "Local resolution",
}


def relion_job_parts(job_dir):
    """Split a RELION job path into (type, number) — ('Select', 'job009')."""
    parts = [p for p in str(job_dir or "").replace("\\", "/").split("/") if p]
    jtype = jnum = ""
    for p in parts:
        if re.fullmatch(r"job\d+", p):
            jnum = p
        elif p in RELION_JOB_TITLES:
            jtype = p
    if not jtype:
        # e.g. "Select/job009" handed in without a project prefix
        for p in parts:
            if p in RELION_JOB_TITLES:
                jtype = p
                break
    return jtype, jnum


def star_particle_count(path, cap=2_000_000):
    """Rows in a RELION particle star's data block, or None if it can't be read.

    Streams and stops at `cap`; these files reach tens of MB and the count is only
    ever used as a card subtitle, so reading the whole thing into memory to label a
    rectangle would be a poor trade.
    """
    # Count PER BLOCK, not "the first loop that has rows". A RELION 4 star opens
    # with data_optics — one row per optics group — so stopping at the first loop
    # reports "1 particle" for every file in the project.
    counts, block, in_loop, seen_header, n = {}, "", False, False, 0
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if s.startswith("data_"):
                    if block:
                        counts[block] = n
                    block, in_loop, seen_header, n = s[5:].strip(), False, False, 0
                    continue
                if s.startswith("loop_"):
                    in_loop, seen_header = True, False
                    continue
                if s.startswith("_"):
                    if in_loop:
                        seen_header = True
                    continue
                if in_loop and seen_header:
                    n += 1
                    if n >= cap:
                        break
    except OSError:
        return None
    if block:
        counts[block] = n
    if not counts:
        return 0
    for name, c in counts.items():
        if "particles" in name.lower():
            return c
    return list(counts.values())[-1]


def relion_card_text(job):
    """(title, subtitle) for an adopted RELION job card.

    Prefers what was recorded at adoption time; falls back to the folder name so a
    job adopted by an older version still reads sensibly.
    """
    params = job.get("params", {}) or {}
    jdir = params.get("job_dir") or params.get("source_star") or job.get("label", "")
    # Parse the DIRECTORY: it carries both halves ("Select/job009"). job_type holds
    # only the type, so reading the number out of it always came back empty.
    jtype, jnum = relion_job_parts(jdir)
    if not jtype:
        jtype = relion_job_parts(params.get("job_type") or "")[0] \
            or str(params.get("job_type") or "")
    title = RELION_JOB_TITLES.get(jtype, "") or (jtype or "RELION job")
    bits = []
    if jnum:
        bits.append(jnum)
    n = params.get("n_particles")
    if n:
        try:
            bits.append(f"{int(n):,} particles")
        except (TypeError, ValueError):
            pass
    return title, " · ".join(bits) or (jdir or "")


# ===========================================================================
# Card positions — the canvas auto-packs stages into rows and forks into
# columns, which is right for a fresh project and wrong the moment a real
# project branches. A position stored here overrides the computed one for that
# node, so the user can arrange the graph to match how they actually think about
# it. Ghost ids ("ghost:<stage>") are stable, so ghosts can be placed too.
# ===========================================================================
def set_card_position(root, node_id, x, y):
    # Called from a Qt drag handler, so a bad coordinate must not raise into the
    # event loop — refuse the write and leave the stored layout untouched.
    try:
        xy = [round(float(x), 1), round(float(y), 1)]
    except (TypeError, ValueError):
        return None
    store = load_jobs(root)
    store.setdefault("positions", {})[str(node_id)] = xy
    save_jobs(root, store)
    return store


def clear_card_positions(root, node_id=None):
    """Forget one placement, or all of them (back to the computed layout)."""
    store = load_jobs(root)
    pos = store.get("positions") or {}
    if node_id is None:
        store["positions"] = {}
    else:
        pos.pop(str(node_id), None)
        store["positions"] = pos
    save_jobs(root, store)
    return store


def set_job_parent(root, job_id, parent_id, slot="processing"):
    """Re-wire a job's input to a different upstream job (or detach with None).

    The canvas draws edges from job["inputs"], and adoption records none — so an
    adopted RELION job sits unconnected no matter how obviously it feeds the next
    step. This makes the DAG editable instead of purely inferred.
    """
    store = load_jobs(root)
    job = (store.get("jobs") or {}).get(job_id)
    if job is None:
        return None
    inputs = dict(job.get("inputs") or {})
    if parent_id:
        if parent_id == job_id or parent_id not in (store.get("jobs") or {}):
            return None
        inputs[slot] = parent_id
    else:
        inputs.pop(slot, None)
    job["inputs"] = inputs
    save_jobs(root, store)
    return job


def params_for_builder(spec, recorded):
    """Split a past job's recorded params against what its stage declares TODAY.

    Returns (kept, dropped, missing):
      kept     values the builder can show, ready to drop into the param store
      dropped  names the stage no longer has — a stale value that is invisible in
               the form but would still reach build_command if it were kept
      missing  params the stage has gained since the run, which fall back to their
               defaults

    Stages gain and lose parameters as the pipeline evolves, so reloading an old
    job is exactly when the two disagree. Silently carrying a dropped key is the
    dangerous case; the caller reports both lists.
    """
    known = {p["name"] for p in (spec or {}).get("params", [])}
    recorded = recorded or {}
    kept = {k: v for k, v in recorded.items() if k in known}
    return kept, sorted(set(recorded) - known), sorted(known - set(recorded))


def summary_text(summary):
    """One-line human readout for a job card, from its summary dict. Known keys
    get friendly units; anything else falls back to 'value key' pairs."""
    if not summary:
        return ""
    parts = []
    if "resolution_A" in summary:
        parts.append(f"{summary['resolution_A']} Å")
    if "series" in summary:
        parts.append(f"{summary['series']} series")
    if "defocus_um" in summary:
        parts.append(f"{summary['defocus_um']} µm")
    if "tomograms" in summary:
        parts.append(f"{summary['tomograms']} tomo")
    if "angpix" in summary:
        parts.append(f"{summary['angpix']} Å")
    if not parts:
        parts = [f"{v} {k}" for k, v in list(summary.items())[:2]]
    return " · ".join(parts)


# Human-readable card titles (the STAGES 'label' is the raw command name, which
# reads like jargon on a card). Falls back to the label for anything unlisted.
FRIENDLY_TITLES = {
    "rename": "Rename EER + MDOC", "imod_warp_key": "IMOD → Warp key",
    "inspect_select": "Inspect tilt stacks", "remake_mdocs": "Remake MDOCs",
    "gain_convert": "Gain: convert", "gain_reciprocal": "Gain: reciprocal",
    "create_settings_fs": "Frameseries settings", "fs_motion_and_ctf": "Motion + CTF",
    "create_settings_ts": "Tilt-series settings", "ts_import": "Import tilt series",
    "ts_stack": "Build tilt stacks", "aretomo": "Align (AreTomo2)",
    "miss_align": "Refine alignment (train)", "miss_align_infer": "Refine alignment (infer)",
    "ts_import_alignments": "Import alignments", "sync_selection": "Sync selection",
    "ts_defocus_hand": "Defocus handedness", "ts_ctf": "CTF estimation",
    "ts_reconstruct": "Tomogram reconstruction", "ts_template_match": "Template matching",
    "threshold_picks": "Threshold picks", "ts_export_particles": "Extract particles (→ RELION)",
    "relion4_convert": "RELION 4: convert STAR", "relion4_class3d": "RELION 4: Class3D",
    "relion4_select_picks": "RELION 4: select good class",
    "relion4_to_warp": "RELION 4 → Warp: re-extract",
    "relion4_check_star": "RELION 4: check particles exist",
    "relion4_result": "RELION 4 result",
    "m_check_ctf": "M: CTF pre-flight",
    "m_index_versions": "M: label version folders",
    "m_reset": "M: check / reset setup",
    "m_kill_orphans": "M: kill stale processes",
    "m_create_population": "M: create population",
    "m_create_source": "M: create data source",
    "m_mask_create": "M: create mask",
    "m_create_species": "M: create species",
    "m_core": "M: refine (MCore)",
    "m_estimate_weights": "M: estimate weights",
    "m_resample_trajectories": "M: resample trajectories",
    "relion4_verify_reextract": "RELION 4: verify re-extraction",
}


def stage_title(stage_id, fallback=""):
    return FRIENDLY_TITLES.get(stage_id, fallback or stage_id)


def _job_seq(jid):
    """Numeric sequence of a job id 'J<seq>' for chronological ordering — so J2 sorts
    before J10 (a plain string sort puts J10 before J2 and scrambles the timeline)."""
    m = re.match(r"J(\d+)", str(jid))
    return int(m.group(1)) if m else 0


def is_cryolo_job(job):
    """A crYOLO pick set (created by Tools ▸ Convert crYOLO picks). It is stored as a
    ts_template_match job so downstream Export wiring/adoption works, but it is NOT a
    template match — so the canvas must NOT show it as 'Template matching'. Detect by
    the marker (new jobs) or the label prefix (retrofits existing crYOLO cards)."""
    return (job.get("tool") == "cryolo"
            or str(job.get("label", "")).lower().startswith("cryolo"))


def card_is_running(node, active):
    """Should this canvas card render as RUNNING?

    Pure decision logic, kept out of the Qt paint path so it can be tested. The
    bug this exists to prevent: the original rule fell back to matching on
    stage_id, which is true for EVERY job of that stage — so starting one
    "Extract particles" job lit up all four of them amber at once.

    The rules, in order:
      * the store says this job is running (a live record), OR
      * a JOB run is live and this is that exact job (id match, nothing else), OR
      * a TRUNK run (▶ Run — no job record of its own) is live for this stage,
        in which case it is drawn on the stage's ghost/template card only.
    """
    if (node or {}).get("status") == "running":
        return True
    act = active or {}
    if not act.get("running"):
        return False
    if act.get("job_id"):
        return act["job_id"] == node.get("id")
    return bool(act.get("stage_id")
                and act["stage_id"] == node.get("stage_id")
                and node.get("is_ghost"))


def canvas_layout(store, orphans=None, stage_status=None, hidden=None):
    """Positioned workflow graph for the canvas. Returns (nodes, edges).

    One ROW per stage, in canonical STAGES order. A stage with no jobs shows a
    single greyed GHOST node ('the default workflow, not yet run'); a stage with
    jobs shows one real node per job, spread across COLUMNS so forks sit side by
    side. Edges: real jobs link to their parent job (the true DAG); stages with
    no real parent are chained along the ghost trunk so the default pipeline
    reads as a connected flow. `orphans` (from discover_picksets) are on-disk
    outputs made outside the app — placed as extra 'orphan' cards in their stage's
    row, offering adoption. `hidden` (a set of node ids the user hid) is applied
    HERE, before columns are assigned, so surviving cards re-pack into contiguous
    columns instead of leaving a gap. Ghosts are never hidden (they're the template)."""
    jobs = store.get("jobs", {}) if isinstance(store, dict) else {}
    stage_status = stage_status or {}     # {stage_id: (ok_bool, label)} from disk
    hidden = hidden or set()
    # RELION stars already consumed by a re-extract/select job — their 'found on disk'
    # cards then render green ('used') instead of a purple untouched orphan.
    used_stars = {str((j.get("params") or {}).get("source_star"))
                  for j in jobs.values() if (j.get("params") or {}).get("source_star")}
    # RELION selections promoted to real job nodes (via a re-extract): their source
    # star is adopted, so the 'found on disk' orphan for it is dropped (no duplicate).
    adopted_sources = {str((j.get("params") or {}).get("source_star"))
                       for j in jobs.values() if j.get("tool") == "relion_selection"}
    by_stage = {}
    for jid, job in jobs.items():
        if jid in hidden:                 # user hid this job card — drop before layout
            continue
        by_stage.setdefault(job.get("stage_id"), []).append((jid, job))
    for lst in by_stage.values():
        lst.sort(key=lambda t: _job_seq(t[0]))   # chronological: J2 before J10, not string sort

    nodes, index, row_first, row_of, cols_used = [], {}, {}, {}, {}
    for row, spec in enumerate(STAGES):
        sid = spec["id"]
        row_of[sid] = row
        y = row * (CARD_H + GAP_Y)
        title = stage_title(sid, spec.get("label", sid))
        js = by_stage.get(sid, [])
        cols_used[sid] = max(1, len(js))

        # THE TEMPLATE RAIL. One card per stage, ALWAYS — this is the default
        # pipeline as a printed reference, not a placeholder for missing work. It
        # used to appear only for stages with no jobs, so the template dissolved
        # exactly as a project got complicated. Pinned to x=0 and never movable;
        # real jobs start at RAIL_W so the two can never overlap.
        nid = f"ghost:{sid}"
        # Its output may already exist on disk (run from the terminal, or a ▶ Run
        # trunk pass). If so, mark it done rather than 'not built'.
        st = stage_status.get(sid)
        done = bool(st and st[0])
        tmpl = {"id": nid, "stage_id": sid, "label": spec.get("label", sid),
                "title": title, "group": spec.get("group", ""), "row": row, "col": 0,
                "x": 0, "y": y, "w": CARD_W, "h": CARD_H,
                "is_ghost": True, "is_template": True, "on_disk": done,
                "n_jobs": len(js),
                "disk_label": (st[1] if done else "") or "",
                # ALWAYS 'ghost'. A template that reports 'completed' is a template
                # that every status-driven code path — colouring, the details pane,
                # the running check — treats as a finished job.
                "status": "ghost", "summary": {}}
        nodes.append(tmpl)
        index[nid] = tmpl
        if not js:
            row_first[sid] = nid
            # WORK THAT ALREADY EXISTS. Stages run from the terminal, or on the
            # trunk before the job model, leave real output and no job record.
            # Saying "output on disk" on the template was not enough: those steps
            # DID run, and a pipeline that shows nothing for them reads as a
            # pipeline that never started. Give them a card of their own in the
            # working canvas — dashed, because it is not a tracked job and cannot
            # be re-run, forked or deleted like one.
            if done:
                did = f"disk:{sid}"
                if did not in hidden:
                    disc = {
                        "id": did, "stage_id": sid,
                        "label": spec.get("label", sid), "title": title,
                        "subtitle": st[1] if st and st[1] else "found on disk",
                        "group": spec.get("group", ""), "row": row, "col": 0,
                        "x": RAIL_W, "y": y, "w": CARD_W, "h": CARD_H,
                        "is_ghost": True,        # menu: offer building a real job
                        "is_discovered": True, "on_disk": True,
                        "disk_label": (st[1] if done else "") or "",
                        "status": "completed", "summary": {}}
                    nodes.append(disc)
                    # In the index so a user placement applies to it and it can be
                    # found like any other card.
                    index[did] = disc
        else:
            for col, (jid, job) in enumerate(js):
                fork = "(fork)" in str(job.get("label", ""))
                # crYOLO pick sets and RELION→Warp re-extract pick sets ride the
                # ts_template_match stage but are NOT template matches — title/subtitle
                # them by what they actually are so the card never reads
                # 'Template matching' / 'ts_template_match'.
                if job.get("tool") == "relion_selection":
                    # "RELION selection · Select/job009 · used" told you nothing about
                    # WHICH selection or how big. Name the RELION job type, its number
                    # and its particle count instead.
                    node_title, node_sub = relion_card_text(job)
                elif is_cryolo_job(job):
                    node_title = "Convert crYOLO → Warp"
                    node_sub = job.get("label", "crYOLO picks")
                elif job.get("tool") == "reextract":
                    node_title = "RELION → Warp re-extract"
                    node_sub = job.get("label", "re-extract picks")
                elif sid == "relion4_result":
                    node_title, node_sub = relion_card_text(job)
                else:
                    node_title = title + (" (fork)" if fork else "")
                    node_sub = None
                n = {"id": jid, "stage_id": sid,
                     "label": job.get("label", spec.get("label", sid)),
                     "title": node_title,
                     "group": spec.get("group", ""), "row": row, "col": col,
                     "x": RAIL_W + col * (CARD_W + GAP_X), "y": y,
                     "w": CARD_W, "h": CARD_H, "is_ghost": False,
                     "status": job.get("status", "building"),
                     "summary": job.get("summary", {}) or {}}
                if node_sub is not None:
                    n["subtitle"] = node_sub
                nodes.append(n)
                index[jid] = n
            row_first[sid] = js[0][0]

    # USER PLACEMENTS WIN. Applied before edges are computed so the lines follow the
    # cards rather than pointing at where the auto-layout would have put them. A
    # position for a node that no longer exists is simply ignored (deleting a job
    # must not strand a coordinate that later gets reused by a new one).
    placed = store.get("positions", {}) if isinstance(store, dict) else {}
    for nid, xy in (placed or {}).items():
        n = index.get(nid)
        if not n or n.get("is_template") or not isinstance(xy, (list, tuple)) \
                or len(xy) != 2:
            continue          # the rail is fixed furniture; it is never placed
        try:
            # Clamped out of the rail: a position stored before the rail existed
            # would otherwise drop a real job straight onto the template.
            n["x"], n["y"] = max(float(xy[0]), float(RAIL_W)), float(xy[1])
            n["moved"] = True
        except (TypeError, ValueError):
            continue

    edges, has_real_parent = [], set()
    for jid, job in jobs.items():
        parent = next((pid for pid in (job.get("inputs") or {}).values()
                       if pid and pid in index), None)
        if parent:
            edges.append((parent, jid))
            has_real_parent.add(job.get("stage_id"))
    # The rail is its own chain, top to bottom: the default pipeline read as a
    # flow. It links template->template ONLY. Previously these trunk edges hopped
    # between whichever card happened to be first in each row, so lines shot from
    # the template across the canvas into unrelated jobs and back — which is most
    # of what made the graph unreadable once a project had real branches.
    order = [s["id"] for s in STAGES]
    for a, b in zip(order, order[1:]):
        edges.append((f"ghost:{a}", f"ghost:{b}"))

    # Orphan cards: on-disk pick sets made outside the app, placed after the real
    # jobs in their stage's row and flagged so the UI can offer 'Adopt as job'.
    for i, orph in enumerate(orphans or []):
        sid = orph.get("stage_id", "ts_template_match")
        if sid not in row_of:
            continue
        oid = f"orphan:{sid}:{orph.get('dir','')}:{orph.get('suffix','')}"
        if oid in hidden:                 # user hid this orphan — skip so cols re-pack
            continue
        if str(orph.get("star", "")) in adopted_sources:   # promoted to a real job node
            continue
        col = cols_used.get(sid, 1)
        cols_used[sid] = col + 1
        suffix = orph.get("suffix", "?")
        # A discovered RELION job: title with its own name (Class3D/job005…), not the
        # tomogration stage id; subtitle names the job type so the card reads clearly.
        status = "orphan"
        if orph.get("kind") == "relion_job":
            title = suffix
            star = str(orph.get("star", ""))
            used = star in used_stars or any(
                us.endswith("/" + star) or star.endswith("/" + us) for us in used_stars)
            subtitle = "RELION " + suffix.split("/")[0] + (" · used" if used else "")
            if used:
                status = "completed"          # green — this selection has been re-extracted
        else:
            title = stage_title(sid) + " · orphan"
            subtitle = sid
        nodes.append({
            "id": oid, "stage_id": sid, "is_orphan": True, "is_ghost": False,
            "label": suffix, "title": title, "subtitle": subtitle,
            "group": "found on disk", "status": status,
            "row": row_of[sid], "col": col,
            "x": col * (CARD_W + GAP_X), "y": row_of[sid] * (CARD_H + GAP_Y),
            "w": CARD_W, "h": CARD_H,
            "summary": {"series": orph.get("n_series", 0)},
            "orphan": orph,
        })
    return nodes, edges


_PICK_STAR_RE = re.compile(r'^(Position\d+)_([\d.]+)Apx(.+)\.star$')


def discover_picksets(root, store):
    """Scan warp_tiltseries/matching[.bak_*]/ for template-match pick sets made
    OUTSIDE the app (distinct by suffix) that aren't already a job. Returns orphan
    descriptors {stage_id, suffix, dir, angpix, n_series} for the canvas to surface
    as adoptable cards. Bounded + lazy (never enumerates whole data dirs)."""
    root = Path(root)
    known = set()
    for j in (store.get("jobs", {}) if isinstance(store, dict) else {}).values():
        if j.get("stage_id") == "ts_template_match":
            s = template_match_suffix(j.get("params", {}))
            if s:
                known.add(s)
        if j.get("orphan_suffix"):        # an already-adopted set
            known.add(j["orphan_suffix"])
    found = {}
    for mdir in sorted(root.glob("warp_tiltseries/matching*")):
        if not mdir.is_dir():
            continue
        rel = os.path.relpath(mdir, root)
        for star in itertools.islice(sorted(mdir.glob("Position*Apx*.star")), 0, 20000):
            m = _PICK_STAR_RE.match(star.name)
            if not m:
                continue
            series, angpix, suffix = m.groups()
            d = found.setdefault((suffix, rel), {"suffix": suffix, "dir": rel,
                                                 "angpix": angpix, "series": set()})
            d["series"].add(series)
    orphans = []
    for (suffix, rel), d in sorted(found.items()):
        if suffix in known:
            continue
        orphans.append({"stage_id": "ts_template_match", "suffix": suffix,
                        "dir": rel, "angpix": d["angpix"], "n_series": len(d["series"])})
    return orphans


# Directories that never contain a RELION project but DO contain enormous numbers
# of files. Discovery must never enumerate these — that is what froze the UI.
SKIP_SCAN_DIRS = {
    "frames", "mdocs", "gains", "Thumbnails", "tomostar", "logs", "jobs",
    "warp_frameseries", "warp_tiltseries", "subtomo", "selected",
    "cryolo-picked", "cryolo-manual-training", "_no_alignment",
}


def discover_relion_jobs(root, store=None, cap=64):
    """Find finished RELION Class2D/Class3D/Refine3D/Select jobs under the project.

    PERFORMANCE IS THE WHOLE DESIGN HERE. The obvious implementation —
    root.glob("*/*/Class3D/job*") — makes Python enumerate every directory two
    levels deep, which in a tomography project means listing frames/ (tens of
    thousands of .eer) on every call. Over ceph that is seconds per sweep, and this
    runs on the UI thread on a timer, so it froze the whole app.

    Instead: scandir the root ONCE, skip the known bulk-data directories outright,
    and only descend into plausible RELION project dirs (depth <= 2). Nothing ever
    walks a data directory, and the total syscall count is proportional to the
    number of PROJECT folders, not to the number of movies.
    """
    root = Path(root)
    jtypes = ("Class3D", "Class2D", "Refine3D", "Select")
    found, seen = [], set()

    def subdirs(d):
        """Immediate subdirectories, skipping bulk data and dotfiles. os.scandir
        avoids a stat() per entry (d.is_dir() uses the dirent type)."""
        out = []
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.name.startswith(".") or e.name in SKIP_SCAN_DIRS:
                        continue
                    try:
                        if e.is_dir(follow_symlinks=False):
                            out.append(Path(e.path))
                    except OSError:
                        pass
        except OSError:
            pass
        return out

    def harvest(project_dir):
        """Pull the jobs out of one RELION project dir (<proj>/<JobType>/jobNNN)."""
        for jt in jtypes:
            jdir = project_dir / jt
            if not jdir.is_dir():
                continue
            try:
                names = sorted(n for n in os.listdir(jdir) if n.startswith("job"))
            except OSError:
                continue
            for name in names:
                jobdir = jdir / name
                key = os.path.realpath(jobdir)
                if key in seen or not jobdir.is_dir():
                    continue
                seen.add(key)
                stars = (sorted(jobdir.glob("run_it*_data.star"))
                         or sorted(jobdir.glob("*_data.star"))
                         or sorted(jobdir.glob("particles.star")))
                if not stars:
                    continue
                star = stars[-1]
                m = re.search(r"run_it(\d+)_data\.star$", star.name)
                it = f" it{int(m.group(1))}" if m else ""
                found.append({"stage_id": "relion4_to_warp", "kind": "relion_job",
                              "suffix": f"{jt}/{name}{it}",
                              "dir": os.path.relpath(jobdir, root),
                              "star": os.path.relpath(star, root), "n_series": 0})
                if len(found) >= cap:
                    return True
        return False

    if harvest(root):                       # RELION project == the tomogration root
        return found
    for d in subdirs(root):                 # e.g. relion4/Class3D/...
        if harvest(d):
            return found
        for d2 in subdirs(d):               # e.g. relion4/warp/Class3D/...
            if harvest(d2):
                return found
    return found



# ===========================================================================
# QProcess wrapper: live stdout/stderr streaming + terminate
# ===========================================================================
