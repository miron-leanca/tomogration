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
import shutil
from pathlib import Path

from tomogration_stages import (STAGES, TRUNK_STAGES, STAGE_OUTPUTS,
                                build_command)

JOBS_FILE = ".tomogration_jobs.json"

def jobs_path(root):
    return Path(root) / JOBS_FILE


def load_jobs(root):
    """The job store {'seq': int, 'jobs': {id: job}} — empty scaffold if missing
    or unreadable (same defensive contract as load_history).

    A corrupt store is renamed aside, not left in place: returning the empty
    scaffold over a still-existing corrupt file means the very next save_jobs
    (even just dragging a card) would overwrite every job record ever made.
    Renaming preserves the evidence and makes the loss recoverable."""
    p = jobs_path(root)
    if not p.is_file():
        return {"seq": 0, "jobs": {}}
    try:
        data = json.loads(p.read_text())
    except OSError:
        # Transient read error (ceph hiccup): do NOT quarantine — the file may
        # be fine. The scaffold is a view, and save_jobs writes atomically, so
        # the worst case is one lost mutation, not a clobbered store.
        return {"seq": 0, "jobs": {}}
    except ValueError:
        data = None
    if not isinstance(data, dict) or not isinstance(data.get("jobs"), dict):
        _quarantine_corrupt(p)
        return {"seq": 0, "jobs": {}}
    # 'seq' must never fall below the highest issued J-number: len(jobs) does
    # after any deletion, and a reused id silently clobbers the old record.
    data.setdefault("seq", max((_jobnum(j) for j in data["jobs"]), default=0))
    _migrate_selection_rows(data)
    return data


def _migrate_selection_rows(data):
    """Move adopted RELION selections into the RELION 4 row.

    They used to be filed as ts_template_match jobs so they would sit in the
    green Pick section. The result was 'Subset selection' cards in TWO rows —
    the ones adopted this way under Pick, and the ones adopted as RELION jobs
    under RELION 4 — which reads as two different kinds of thing when it is
    one. Done on LOAD rather than by a one-off script so existing projects fix
    themselves; the record is only rewritten when something actually changed,
    so this is not a write on every read."""
    for job in (data.get("jobs") or {}).values():
        if (isinstance(job, dict) and job.get("tool") == "relion_selection"
                and job.get("stage_id") == "ts_template_match"):
            job["stage_id"] = "relion4_result"


def _quarantine_corrupt(p):
    """Rename an unparseable store aside (.corrupt_<ts>) so it can't be
    silently overwritten by the next save — and can be hand-recovered."""
    try:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        p.rename(p.with_name(p.name + f".corrupt_{stamp}"))
    except OSError:
        pass


def _atomic_write_text(path, text):
    """Write via a same-directory temp file + os.replace so a crash mid-write
    leaves either the old file or the new one — never a truncated store."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def save_jobs(root, store):
    try:
        _atomic_write_text(jobs_path(root), json.dumps(store, indent=1))
    except OSError:
        pass


# The five states a job moves through. BUILDING is the one that was missing: a
# card created by "Build downstream" or dropped from the palette was born already
# 'queued', so a job nobody had finished configuring could start on its own the
# moment the queue drained. Building means "on the canvas, not going anywhere
# until you say so".
JOB_STATES = ("building", "queued", "running", "completed", "failed")

# Statuses whose card the BUILDER may write back into and re-run IN PLACE.
# 'failed' belongs here: a failure is not precious history -- the whole point of
# one is to fix the parameter and run again, and the card should then go green.
# J4's gain path was wrong; the corrected re-run went in as an untracked trunk
# run because a failed job would not bind, so the card sat red until it was
# marked by hand -- and the hand-mark then buried the real success under "the
# run actually failed (exit 1)". A COMPLETED job stays off this list: re-running
# into it would erase the record of a success, so that still forks a new job.
RERUNNABLE = ("building", "queued", "failed")


def queued_jobs(store):
    """Jobs waiting to run, in run order (ENQUEUE order, oldest first).

    THE QUEUE IS JUST JOBS. It used to be a separate in-memory list of
    (label, command, stage_id) tuples, which is why queued work could never appear
    on the canvas, could not be restarted, and evaporated when the app closed. A
    queued item is now an ordinary job record with status='queued' and its command
    already resolved, so every job affordance (card, details, restart, clone,
    delete, persistence) applies to it for free.

    Order is `queued_at` (stamped whenever a job's status becomes queued), NOT
    J-number: creation order made re-queueing an old job silently jump ahead of
    everything queued since. Records without the stamp (pre-existing stores) fall
    back to their creation time, which is what the old rule effectively was;
    J-number breaks the remaining ties."""
    jobs = store.get("jobs", {}) if isinstance(store, dict) else {}
    def _order(jid):
        j = jobs[jid] or {}
        return (j.get("queued_at") or j.get("created") or "", _job_seq(jid))
    return [jobs[j] for j in sorted(jobs, key=_order)
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
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for j in hit:
        j["status"] = "failed"
        j["exit_code"] = None
        j["interrupted"] = True
        # An interrupted job must not keep finished=None: versions_for_job reads
        # that as "still running" and would attribute every M version folder
        # written after `started` — forever — to this dead job.
        if not j.get("finished"):
            j["finished"] = stamp
    if hit:
        save_jobs(root, store)
    return [j["id"] for j in hit]


def job_dir_slug(stage_id):
    """Short filesystem-safe tag naming what a job DID, from its stage label:
    'Variant polarity (measure)' -> 'variant-polarity'. Parentheticals drop,
    everything non-alphanumeric collapses to single dashes."""
    spec = next((s for s in STAGES if s.get("id") == stage_id), None)
    label = str((spec or {}).get("label") or stage_id or "")
    label = re.sub(r"\([^)]*\)", "", label)          # drop "(measure)" etc.
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return slug[:28].rstrip("-")


def _fuzzy_job_dir(root, jid):
    """The job's REAL folder under jobs/, matched by J-number prefix at a
    boundary (J2 must not claim J27's or J2_x's... it must claim J2_x but not
    J27_y): exact name, or the id followed by a non-digit. '' if nothing."""
    jdir = Path(root) / "jobs"
    if (jdir / jid).is_dir():
        return f"jobs/{jid}"
    try:
        cands = [d.name for d in jdir.iterdir() if d.is_dir()
                 and re.match(rf"{re.escape(jid)}[^0-9]", d.name)]
    except OSError:
        cands = []
    return f"jobs/{sorted(cands)[0]}" if cands else ""


def job_real_dir(root, job_id, stage_id=None, store=None):
    """A job's ACTUAL folder, project-relative.

    From the record when there is one — folder names carry a slug now, and the
    user may have renamed by hand — else reconstructed and fuzzily resolved by
    J-number. Reconstructing a bare jobs/J47 instead created a SECOND, nearly
    empty folder next to the real jobs/J47_cryolo-picks, so the outputs
    breadcrumb pointed at a directory nothing else used."""
    rec = ((store or {}).get("jobs", {}) or {}).get(job_id) or {}
    jd = rec.get("output_dir") or job_output_dir(job_id, stage_id)
    if not (Path(root) / jd).is_dir():
        found = _fuzzy_job_dir(root, job_id)
        if found:
            jd = found
    return jd


def job_dir_token(job):
    """What {jobid} resolves to for THIS job: the basename of its own dir
    (J34_variant-polarity), so every {jobid}-derived path shares one name.
    Records that predate the naming (output_dir jobs/J10) resolve to the bare
    id, exactly as before."""
    od = str((job or {}).get("output_dir") or "").rstrip("/")
    base = os.path.basename(od)
    jid = str((job or {}).get("id") or "")
    return base if jid and base.startswith(jid) else (jid or base)


# Three kinds of job ride ts_template_match so their downstream wiring works
# (see actual_job_kind). Their folder must say what they ARE: J47 adapted
# crYOLO picks and got jobs/J47_ts-template-match, which is a name that will
# mislead anyone reading the directory a year from now — including the person
# who made it.
KIND_DIR_SLUG = {
    "cryolo": "cryolo-picks",
    "reextract": "relion-reextract",
    "relion_selection": "relion-selection",
}


def job_output_dir(job_id, stage_id=None, slug=None):
    """A job's processing dir, relative to the project root (cwd of every run).

    Named, not bare: jobs/J34_variant-polarity instead of jobs/J34, so a
    directory listing reads as a run log instead of a numbers column. (Users
    were already renaming the bare dirs by hand — jobs/J27-bin8-isonet2model
    in the EML46 project — which is how this convention was chosen.) Bare
    jobs/J34 remains valid for records that predate the naming; everything
    resolves through the stored output_dir / resolve_job_dir, never by
    reconstructing the path."""
    slug = slug or (job_dir_slug(stage_id) if stage_id else "")
    return f"jobs/{job_id}_{slug}" if slug else f"jobs/{job_id}"


def new_job(root, stage_id, label, params, inputs=None, kind=None):
    """Create + persist a fresh job; return the record. `inputs` maps an input
    slot name -> the parent job id feeding it (or None = read the project trunk /
    the .settings default). Ids are monotonic 'J<seq>' so they never collide even
    after deletions.

    `kind` names what the job really is when that differs from its stage — the
    crYOLO/re-extract/selection cards that ride ts_template_match. It only
    affects the FOLDER NAME, and it has to be given here rather than read off
    `tool` later: the tool marker is set by a follow-up update_job, by which
    time the directory name is already baked into the record."""
    store = load_jobs(root)
    store["seq"] = int(store.get("seq", 0)) + 1
    jid = f"J{store['seq']}"
    job = {
        "id": jid,
        "stage_id": stage_id,
        "label": label,
        "params": dict(params or {}),
        "inputs": dict(inputs or {}),
        "output_dir": job_output_dir(jid, stage_id,
                                     KIND_DIR_SLUG.get(kind)),
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
    # A fresh run supersedes a hand-set verdict: the mark described the PREVIOUS
    # result, and leaving it attached would label the new run with the old
    # judgement.
    if fields.get("status") in ("running", "queued"):
        job.pop("manual_status", None)
    job.update(fields)
    save_jobs(root, store)
    return job


# A job's exit code answers "did the process return 0", which is not the same
# question as "did this work". Both failure modes are real and both happened on
# 2026-08-16: an IsoNet run exited 0 after silently reusing another run's data,
# and a training run that finished all 50 epochs exited 1 on an X11 teardown.
# So the verdict can be set by hand — but never by overwriting the machine's
# record, which stays underneath and can be restored.
MANUAL_STATUSES = ("completed", "failed")
# A BUILDING card can be judged too. It is a job that was created and never
# run, and there are two ordinary reasons to settle it by hand: the work was
# actually done outside tomogration (by a terminal run, or by another job that
# wrote the same outputs), or the card is a dead end that should stop showing
# as outstanding work. Running and queued stay off this list — a running job's
# status is owned by its process, and a queued one has produced nothing to
# judge.
_JUDGEABLE = ("completed", "failed", "building")


def set_manual_status(root, job_id, status, reason=""):
    """Mark a FINISHED job completed or failed by hand. -> (ok, message).

    Refused while a job is running or queued: a running job's status is owned by
    the process (kill it first), and a queued one has produced no result to
    judge."""
    if status not in MANUAL_STATUSES:
        return (False, f"{status!r} is not a status that can be set by hand "
                       f"({', '.join(MANUAL_STATUSES)}).")
    store = load_jobs(root)
    job = (store.get("jobs") or {}).get(job_id)
    if job is None:
        return (False, f"{job_id} no longer exists.")
    current = job.get("status", "")
    if current not in _JUDGEABLE:
        return (False, f"{job_id} is {current} — it cannot be marked by hand. "
                       + ("Kill it first." if current == "running" else
                          "Dequeue it first."))
    prior = job.get("manual_status") or {}
    job["manual_status"] = {
        # 'was' is the MACHINE's verdict, kept across repeated marks so one
        # restore always returns to what actually happened.
        "was": prior.get("was", current),
        "exit_code": prior.get("exit_code", job.get("exit_code")),
        "at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reason": str(reason or ""),
    }
    job["status"] = status
    save_jobs(root, store)
    was = job["manual_status"]["was"]
    code = job["manual_status"]["exit_code"]
    return (True, f"{job_id} marked {status} by hand "
                  f"(it actually {was}"
                  + (f", exit {code}" if code is not None else "") + ").")


def clear_manual_status(root, job_id):
    """Put back the status the machine recorded. -> (ok, message)."""
    store = load_jobs(root)
    job = (store.get("jobs") or {}).get(job_id)
    if job is None:
        return (False, f"{job_id} no longer exists.")
    manual = job.pop("manual_status", None)
    if not manual:
        return (False, f"{job_id} has no hand-set status to clear.")
    job["status"] = manual.get("was", job.get("status", "failed"))
    save_jobs(root, store)
    return (True, f"{job_id} restored to {job['status']} (as recorded).")


# ===========================================================================
# CANVAS ANNOTATIONS — notes and branch frames
#
# The graph shows what ran and what fed what. It cannot show why a branch
# exists, which of three variants is the one being kept, or what to do next —
# and that context currently lives in the user's head while five IsoNet folders
# accumulate. Notes and frames are that layer: they carry NO data meaning, sit
# beside the jobs in the same store, and are never consulted by anything that
# runs. A frame is a labelled region you drop behind a set of cards; a note is
# a sticky.
# ===========================================================================
NOTE_KINDS = ("note", "frame")
NOTE_COLOURS = ("amber", "blue", "green", "violet", "red", "grey")
_DEFAULT_SIZE = {"note": (220, 96), "frame": (520, 340)}


def wrap_lines(text, max_width, measure, max_lines=0):
    """Greedy word-wrap `text` to `max_width`, as a list of lines.

    `measure(str) -> width` keeps this pure: the app passes
    QFontMetrics.horizontalAdvance, the tests pass a fixed width per character,
    and neither needs the other.

    Explicit newlines are paragraph breaks and are kept. A single word too long
    for the width is broken across lines rather than allowed to overflow -- a
    pasted path is one "word" and would otherwise run off the note. With
    max_lines, the last kept line ends in an ellipsis so a clipped note looks
    clipped instead of merely short.
    """
    if not str(text or "").strip():
        return []
    out = []
    for para in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        words, line = para.split(), ""
        if not words:
            out.append("")
            continue
        for word in words:
            while measure(word) > max_width and len(word) > 1:
                cut = len(word)
                while cut > 1 and measure(word[:cut]) > max_width:
                    cut -= 1
                if line:
                    out.append(line)
                    line = ""
                out.append(word[:cut])
                word = word[cut:]
            trial = f"{line} {word}" if line else word
            if line and measure(trial) > max_width:
                out.append(line)
                line = word
            else:
                line = trial
        out.append(line)
    if max_lines and len(out) > max_lines:
        out = out[:max_lines]
        last = out[-1]
        while last and measure(last + "…") > max_width:
            last = last[:-1]
        out[-1] = last + "…"
    return out


def load_notes(root):
    """Every annotation on this project's canvas, oldest first."""
    notes = load_jobs(root).get("notes")
    return [n for n in notes if isinstance(n, dict)] if isinstance(notes, list) else []


def add_note(root, kind, x, y, text="", colour="amber", w=None, h=None):
    """Place a note or frame. Ids are 'N<seq>' from their own counter, so they
    can never collide with a job id however many are deleted."""
    if kind not in NOTE_KINDS:
        raise ValueError(f"{kind!r} is not one of {NOTE_KINDS}")
    store = load_jobs(root)
    notes = store.setdefault("notes", [])
    seq = int(store.get("note_seq", 0)) + 1
    store["note_seq"] = seq
    dw, dh = _DEFAULT_SIZE[kind]
    note = {"id": f"N{seq}", "kind": kind,
            "x": float(x), "y": float(y),
            "w": float(w or dw), "h": float(h or dh),
            "text": str(text),
            "colour": colour if colour in NOTE_COLOURS else "amber",
            "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    notes.append(note)
    save_jobs(root, store)
    return note


def update_note(root, note_id, **fields):
    """Move, resize, retitle or recolour one. Unknown keys are ignored rather
    than stored, so a typo cannot quietly become part of the record."""
    allowed = {"x", "y", "w", "h", "text", "colour", "kind"}
    store = load_jobs(root)
    for n in store.get("notes", []) or []:
        if n.get("id") != note_id:
            continue
        for k, v in fields.items():
            if k in allowed:
                n[k] = v
        save_jobs(root, store)
        return n
    return None


def delete_note(root, note_id):
    store = load_jobs(root)
    notes = store.get("notes", []) or []
    keep = [n for n in notes if n.get("id") != note_id]
    if len(keep) == len(notes):
        return False
    store["notes"] = keep
    save_jobs(root, store)
    return True


def notes_for_canvas(store):
    """Annotations split for painting: frames go BEHIND the cards, notes in
    front. Anything with an unknown kind is dropped rather than drawn in the
    wrong layer."""
    notes = store.get("notes") if isinstance(store, dict) else None
    notes = [n for n in notes if isinstance(n, dict)] if isinstance(notes, list) else []
    return ([n for n in notes if n.get("kind") == "frame"],
            [n for n in notes if n.get("kind") == "note"])


def cards_inside(frame, nodes):
    """Job ids whose card centre falls inside a frame — what the frame is
    'about', computed from geometry rather than stored, so moving a card in or
    out of a branch needs no bookkeeping."""
    x, y = float(frame.get("x", 0)), float(frame.get("y", 0))
    w, h = float(frame.get("w", 0)), float(frame.get("h", 0))
    out = []
    for n in nodes:
        if n.get("is_ghost") or n.get("is_template") or not n.get("id"):
            continue
        cx = float(n.get("x", 0)) + float(n.get("w", 0)) / 2
        cy = float(n.get("y", 0)) + float(n.get("h", 0)) / 2
        if x <= cx <= x + w and y <= cy <= y + h:
            out.append(n["id"])
    return out


def manual_status_note(job):
    """One line for the details pane, or '' — what was overridden, and when."""
    m = (job or {}).get("manual_status") or {}
    if not m:
        return ""
    code = m.get("exit_code")
    return (f"status set by hand to '{job.get('status')}' on {m.get('at', '?')}"
            f" — the run actually {m.get('was', '?')}"
            + (f" (exit {code})" if code is not None else "")
            + (f": {m['reason']}" if m.get("reason") else ""))


def is_warp_stage(spec):
    """True for stages whose command is a WarpTools subcommand — the ones that
    accept --input_processing / --output_processing for free (BaseCommand)."""
    return str(spec.get("base", "")).startswith("WarpTools")


def takes_processing_flags(spec):
    """True for stages that actually accept --input_processing/--output_processing.

    NOT every WarpTools subcommand does. The flags live in the base class shared
    by commands that operate on an existing .settings file; the commands that
    CREATE that context have no processing dir to redirect and reject them:

        create_settings   builds the .settings file itself
        ts_import         reads mdocs + a frameseries folder, writes tomostar/

    and WarpTools 2.0.0 answers an unknown option by printing its help and
    exiting ZERO. So J8 and J9 were recorded green having written nothing, and
    J12 was the first job to notice ("Could not find warp_tiltseries.settings").

    Passing a --settings file is exactly the property that separates the two
    groups, and it is checked here rather than kept as a hand-maintained list so
    a stage added later cannot silently rejoin the broken set.
    """
    if not is_warp_stage(spec):
        return False
    return any(p.get("flag") == "--settings" or p.get("name") == "settings"
               for p in (spec.get("params") or []))


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
    results in the wrong place). A pick-set card is special for the same
    reason: crYOLO/re-extract/selection folders hold stars, not .xml."""
    if not takes_processing_flags(spec):
        return ""
    if spec.get("id") in TRUNK_STAGES:
        return ""            # trunk-only: reads shared reconstructions, suffix-distinct output
    toks = []
    pid = parent_job_id(job)
    parent = store.get("jobs", {}).get(pid) if pid else None
    # A pick-set card (crYOLO / re-extract / promoted selection) is a MARKER:
    # its folder holds star files, not Warp's per-series .xml results. Pointing
    # --input_processing there tells Warp to look for previous results where
    # there are none — the same trap threshold_picks is exempted from above.
    # Omitting it lets the run read the .settings default, which is what the
    # equivalent hand-typed command does.
    if parent and actual_job_kind(parent):
        parent = None
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
    # Each flag is suppressed independently: a manual --input_processing must not
    # also drop the automatic --output_processing (the job would then write into
    # the trunk .settings dir, silently breaking per-job isolation).
    for tok in re.findall(r"--\w+_processing \S+", extra):
        if tok.split()[0] not in cmd:
            cmd = f"{cmd} {tok}"
    # {jobid} in any param resolves to THIS job's id, so a stage whose output must
    # live outside jobs/ (the RELION handoff needs one project dir with the star +
    # subtomo/) still gets a per-job, collision-free home: relion4/{jobid} ->
    # relion4/J13. No shared default for a second run to overwrite.
    cmd = cmd.replace("{jobid}", job_dir_token(job))
    return cmd


def _jobnum(jid):
    """Numeric part of a 'J<seq>' id, for newest-first ordering."""
    try:
        return int(str(jid).lstrip("J"))
    except ValueError:
        return 0


def latest_stage_output_dir(store, stage_id, statuses=("completed",)):
    """Where the newest finished job of `stage_id` actually wrote, or "".

    Some stages cannot be job-scoped with --input_processing (ts_import takes
    no such flag), so an explicit path param is the ONLY thing connecting them
    to an upstream job's results. The trunk default -- warp_frameseries/ --
    is empty whenever that upstream stage ran as a job, and Warp then says
    "No frame series metadata found" pointing at the empty trunk.
    """
    return str(latest_stage_job(store, stage_id, statuses).get("output_dir", ""))


def latest_stage_job(store, stage_id, statuses=("completed",)):
    """The newest finished job record of `stage_id`, or {}."""
    jobs = (store or {}).get("jobs", {}) or {}
    cands = [j for j in jobs.values()
             if j.get("stage_id") == stage_id
             and (not statuses or j.get("status") in statuses)
             and str(j.get("output_dir") or "").strip()]
    if not cands:
        return {}
    return max(cands, key=lambda j: _jobnum(j.get("id", "")))


def refined_since_last_import(store):
    """Job id of a miss-alignment run NEWER than the newest alignment import,
    or "".

    Order matters and the damage is silent. ts_import_alignments is AreTomo's
    hand-off: it reads .xf/.tlt and writes them into the Warp XMLs.
    miss-alignment REFINES what is already in those XMLs and writes its result
    straight back -- it produces no .xf at all. So importing after refining
    (a) overwrites the refined geometry with the coarse alignment it started
    from, and (b) fails on every series, and a failed import marks each one
    UNSELECTED, which Warp persists -- ts_ctf and ts_reconstruct then process
    nothing at all and report success.
    """
    jobs = (store or {}).get("jobs", {}) or {}
    def newest(*stage_ids):
        got = [j for j in jobs.values()
               if j.get("stage_id") in stage_ids and j.get("status") == "completed"]
        return max(got, key=lambda j: _jobnum(j.get("id", "")), default=None)
    refined = newest("miss_align", "miss_align_infer")
    if refined is None:
        return ""
    imported = newest("ts_import_alignments")
    if imported is not None and _jobnum(imported.get("id", "")) > _jobnum(
            refined.get("id", "")):
        return ""
    return str(refined.get("id", ""))


def alignment_pixel_size(store, mdoc_angpix=""):
    """Å/px that AreTomo's .xf shifts are expressed in, or "" if unknown.

    The AreTomo JOB's own angpix first: ts_stack can build BINNED stacks, and
    the shifts are then in those pixels, which the mdoc's unbinned value would
    contradict. The mdoc is the fallback for an alignment folder produced
    outside tomogration.

    Returns "" rather than a guess when neither is known. This number was
    hardcoded 1.57 -- EML46's -- so EML50 at 1.98 would have imported every
    shift 21% short, with no error anywhere: the run succeeds and the tomograms
    come out quietly misaligned. A silently wrong default is the whole bug; an
    empty one at least stops.
    """
    job_angpix = str(latest_stage_job(store, "aretomo",
                                      ("completed", "failed")
                                      ).get("params", {}).get("angpix", "") or "").strip()
    for cand in (job_angpix, str(mdoc_angpix or "").strip()):
        try:
            if float(cand) > 0:
                return cand
        except ValueError:
            continue
    return ""


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
        # A create_settings/ts_import job owns no processing dir, so wiring a
        # downstream job to READ it points --input_processing at a folder with
        # no results in it.
        if not spec or not takes_processing_flags(spec):
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
    # The membrane root holds every tune-era shared dir (segment, components,
    # population…) — multiple runs' results live side by side in there.
    "membrane",
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
        rel = rel.replace("{jobid}", job_dir_token(job))
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

    # This job's own dir — from the RECORD when we have it (named dirs, and
    # the user may have renamed it by hand), else reconstructed + fuzzily
    # resolved. Never assume the bare jobs/J### spelling.
    rec = (store or {}).get("jobs", {}).get(job_id) or {}
    jd = (rec.get("output_dir")
          or job_output_dir(job_id, stage_id))
    if not (root / jd).is_dir():
        found = _fuzzy_job_dir(root, job_id)
        jd = found or jd
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
        rel = rel.replace(
            "{jobid}",
            job_dir_token((store or {}).get("jobs", {}).get(job_id)
                          or {"id": job_id,
                              "output_dir": job_output_dir(job_id, stage_id)}))
        if os.path.isabs(rel):
            try:
                rel = os.path.relpath(rel, root)
            except ValueError:
                skipped.append(f"{rel} (outside the project)")
                continue
        # membrane/* is protected as a PREFIX, not just the root: the shared
        # tune-era dirs (membrane/segment, membrane/components/cc50…) hold
        # several runs' results side by side, and a palette-created job that
        # wrote there must not offer them for deletion. Named-dir jobs write
        # under jobs/, which stays deletable per job.
        if (rel.startswith("..") or rel in PROTECTED_DIRS
                or rel.split("/", 1)[0] == "membrane"):
            skipped.append(f"{rel} (protected)")
            continue
        # Refuse when another job's dir IS the target or sits anywhere inside it:
        # deleting `myout` must not take job B's `myout/sub` with it.
        nested = sorted({jid for other, jids in claimed.items()
                         for jid in jids
                         if other == rel or other.startswith(rel + "/")})
        if nested:
            skipped.append(f"{rel} (also written by {', '.join(nested)})")
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
    "ts_template_match": ["threshold_picks", "ts_export_particles"],
    "threshold_picks": ["ts_export_particles"],
    "ts_export_particles": ["relion4_convert", "relion4_verify_reextract"],
    "relion4_convert": ["relion4_merge_optics"],
    # A RELION result — an adopted job or a promoted Subset selection — is
    # re-extracted by an EXPORT that reads its star directly: Warp applies the
    # refined shifts itself. The pick-star converters that used to sit between
    # the two are retired (relion4_to_warp in the stages, kept for old cards).
    "relion4_result": ["ts_export_particles", "m_mask_create", "m_create_species"],
    "m_create_population": ["m_create_source"],
    "m_create_source": ["m_create_species"],
    "m_mask_create": ["m_create_species"],
    "m_create_species": ["m_core"],
    "m_core": ["m_core", "m_estimate_weights", "m_resample_trajectories"],
    "m_estimate_weights": ["m_core"],
    "m_resample_trajectories": ["m_core"],
    # ---- membrane branch -------------------------------------------------
    # Without these edges every membrane step had to be wired by hand, and the
    # question "which folder did the last one write to?" came up at each one —
    # the answer being a jobs/J## path nothing on screen showed.
    "ts_reconstruct": ["ts_template_match", "mb_deconv", "mb_segment",
                       "mb_explore"],
    "mb_deconv": ["mb_segment", "mb_explore"],
    "mb_isonet_predict": ["mb_segment", "mb_explore"],
    "mb_isonet2_predict": ["mb_segment", "mb_explore"],
    "mb_segment": ["mb_thresholds", "mb_polarity"],
    "mb_thresholds": ["mb_components", "mb_polarity"],
    "mb_polarity": ["mb_fit_virions"],
    "mb_components": ["mb_population", "mb_size_qc", "mb_fit_virions",
                      "mb_mesh"],
    "mb_population": ["mb_size_qc", "mb_split_clusters", "mb_fit_virions"],
    "mb_size_qc": ["mb_split_clusters"],
    "mb_split_clusters": ["mb_size_qc", "mb_fit_virions"],
    "mb_fit_virions": ["mb_pick_surfaces"],
}


def _picktag(in_suffix):
    """Short, filesystem-safe pick-set tag from a threshold in_suffix, stripping the
    '<angpix>Apx' prefix + leading underscore: '12.56Apx_v3-optimized' ->
    'v3-optimized'. Used to name each export's RELION dir so it maps to its pick set."""
    t = re.sub(r"^[\d.]+Apx", "", in_suffix or "").lstrip("_")
    return t or "picks"


def relion_star_tag(star, job_dir=""):
    """Filesystem-safe name for what a RELION star is: 'Select-job029' from
    'Select/job029' or '.../Select/job029/particles.star'. It names the export
    dir, so the extraction maps back to the selection it came from."""
    for cand in (job_dir, star):
        m = re.search(r"([A-Za-z0-9]+)/(job\d+)", str(cand or "").replace("\\", "/"))
        if m:
            return f"{m.group(1)}-{m.group(2)}"
    stem = os.path.splitext(os.path.basename(str(star or "")))[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", stem) or "relion"


def direct_export_params(star, job_dir=""):
    """ts_export_particles parameters for re-extracting a RELION star DIRECTLY.

    The one re-extraction route: Warp reads the star, subtracts the refined
    rlnOrigin*Angst shifts itself and scales the coordinates by coords_angpix,
    which must be the star's own pixel size. That value lives in the file, so
    it is left blank here (this function is pure) and filled in by the app
    from the star's optics block; the export validator refuses a blank and the
    pre-run check refuses a mismatch. The pick-star fields are blanked so they
    are neither sent nor reported as ignored. {jobid} keeps two exports of the
    same selection in separate folders."""
    tag = relion_star_tag(star, job_dir)
    outdir = f"relion4/{tag}_{{jobid}}"
    return {"input_star": str(star), "input_directory": "", "input_pattern": "",
            "normalized_coords": False, "coords_angpix": "",
            "output_processing": outdir, "output_star": f"{outdir}/matching.star"}


# Membrane steps all read a FOLDER their parent wrote and write one of their
# own. The folder is the parent job's own output dir (jobs/J35), which is the
# one thing the form cannot guess and the log only mentions in passing — the
# card's default ("membrane/segment") is where a THREE-COLUMN run would have
# put it, and pointing a job-mode child there fails with "not found".
MEMBRANE_CHAIN = {
    ("mb_deconv", "mb_segment"), ("mb_isonet_predict", "mb_segment"),
    ("mb_isonet2_predict", "mb_segment"), ("ts_reconstruct", "mb_segment"),
    ("ts_reconstruct", "mb_deconv"),
    ("mb_segment", "mb_thresholds"), ("mb_thresholds", "mb_components"),
    ("mb_components", "mb_mesh"), ("mb_components", "mb_population"),
    ("mb_components", "mb_split_clusters"),
    ("mb_split_clusters", "mb_fit_virions"),
}
# What each parent actually leaves for its child to read. Most write straight
# into their job dir; the ones that nest say so here rather than in five
# scattered f-strings.
MEMBRANE_SUBDIR = {
    "mb_deconv": "s{strength}_f{falloff}",      # one folder per swept value
    "mb_isonet_predict": "corrected",
    "mb_isonet2_predict": "corrected",
    "ts_reconstruct": "reconstruction",
}


def membrane_parent_dir(parent_stage, parent_params, parent_output_dir):
    """The folder a membrane parent leaves behind, ready to be read.

    The parent's own OUTPUT PARAM wins over its job record: a palette- or
    builder-created job keeps its stage default (a shared membrane/ dir), so
    its wrapper wrote THERE while the record claims jobs/J##_<slug> — an empty
    folder _run_job merely mkdir'd. A {jobid}-shaped param and the record
    agree by construction, so only a literal path overrides.

    Deconvolution writes one subfolder per swept strength/falloff, and a sweep
    has several — the FIRST value is the only defensible default, and naming it
    in the child's field is what makes the choice visible instead of implied."""
    base = str(parent_output_dir or "").rstrip("/")
    spec = next((s for s in STAGES if s.get("id") == parent_stage), {})
    for key in spec.get("output_params") or []:
        v = str((parent_params or {}).get(key, "") or "").strip().rstrip("/")
        if v and "{jobid}" not in v:
            base = v
            break
    if not base:
        return ""
    sub = MEMBRANE_SUBDIR.get(parent_stage, "")
    first = lambda k, d: (str(parent_params.get(k, "") or d).split() or [d])[0]
    if parent_stage == "mb_deconv":
        sub = f"s{first('MB_STRENGTH', '1.0')}_f{first('MB_FALLOFF', '1.0')}"
    elif parent_stage == "mb_components":
        # components writes one cc<voxels>/ folder per swept size cutoff, so the
        # job root holds no volumes at all.
        sub = f"cc{first('MB_CC_THRES', '50')}"
    return f"{base}/{sub}" if sub else base


# The greyscale tomograms a membrane chain started from, carried down as a
# breadcrumb. Every step after segmentation reads MASKS, so by the time you
# reach mesh or fit-virions the original volumes are two or three hops back and
# the immediate parent's input_dir is binary — which is what made the mesh card
# default to a thresholds folder and the fitter's density gate read a mask.
# build_command only ever looks at a stage's declared params, so an extra key
# rides along in the job record without reaching any command line.
GREY_KEY = "_source_tomograms"


def _norm_thr(t):
    """The one-decimal spelling membrain thresholds bakes into filenames
    (-3 -> '-3.0', -1.5 -> '-1.5') — mirrors the wrapper's norm_thr awk."""
    try:
        v = float(t)
    except (TypeError, ValueError):
        return str(t)
    s = f"{v:g}"
    return s + ".0" if v == int(v) else s


def _mb_carry(parent_params, out):
    """Breadcrumbs every specific membrane branch must keep moving: the
    greyscale-tomogram source and the tune-list restriction. Dropping them in
    a specific branch is how a fit card three hops down ends up sampling
    warp_tiltseries/reconstruction while the chain actually segmented an
    IsoNet-corrected variant — same grid, wrong densities, exit 0."""
    grey = str(parent_params.get(GREY_KEY, "") or "")
    if grey:
        out.setdefault(GREY_KEY, grey)
        # A branch that wants the greyscale dir but found no better source
        # takes the carried one (source_tomograms returns "" for parents
        # whose params hold neither GREY_KEY nor input_dir).
        if "tomogram" in out and not out["tomogram"]:
            out["tomogram"] = grey
    if parent_params.get("MB_TOMO_LIST"):
        out.setdefault("MB_TOMO_LIST", parent_params["MB_TOMO_LIST"])
    return out


def source_tomograms(parent_stage, parent_params):
    """Where the greyscale volumes are, for a child that needs densities."""
    carried = str(parent_params.get(GREY_KEY, "") or "")
    if carried:
        return carried                      # already breadcrumbed down the chain
    if parent_stage in ("mb_segment", "mb_deconv"):
        return str(parent_params.get("input_dir", "") or "")
    if parent_stage in ("mb_isonet_predict", "mb_isonet2_predict"):
        return str(parent_params.get("output_dir", "") or "")
    return str(parent_params.get("input_dir", "") or "")


# Stages whose output IS a folder of greyscale reconstructions, and which of
# their params holds it. Everything else in the membrane chain produces score
# maps, masks or labels — none of which is a tomogram.
_GREY_SOURCE = {
    "mb_isonet_predict": "output_dir", "mb_isonet2_predict": "output_dir",
    "mb_deconv": "output_dir", "ts_reconstruct": "output_dir",
}


def greyscale_source(job, store, depth=8):
    """The folder of greyscale volumes a job's overlays should be drawn ON.

    A job's own input_dir is NOT it: components reads threshold MASKS and
    thresholds read SCORE MAPS, so opening a components job used the mask as
    the base image and the viewer held no tomogram at all — only the outlines
    of one.

    The breadcrumb carried down by build-downstream answers when it is there.
    When it is not — a job built by hand, or adopted — walk up the parents
    until a stage that really does hold reconstructions is reached, rather
    than falling back to a folder that merely happens to contain .mrc files."""
    jobs = (store or {}).get("jobs") or {}
    seen = set()
    for _ in range(depth):
        if not isinstance(job, dict) or job.get("id") in seen:
            break
        seen.add(job.get("id"))
        params = job.get("params") or {}
        carried = str(params.get(GREY_KEY, "") or "").strip()
        if carried:
            return carried
        key = _GREY_SOURCE.get(job.get("stage_id", ""))
        if key:
            got = str(params.get(key, "") or "").strip()
            if got:
                return got
        if job.get("stage_id") == "mb_segment":
            got = str(params.get("input_dir", "") or "").strip()
            if got:
                return got
        parent = next((pid for pid in (job.get("inputs") or {}).values() if pid),
                      None)
        if not parent:
            break
        job = jobs.get(parent)
    return ""


def derive_child_params(child_stage, parent_stage, parent_params, parent_output_dir=""):
    """Params a downstream job should inherit from its chosen parent, so wiring
    'J5 -> threshold -> export -> convert -> Class3D' auto-fills the fiddly
    suffix/pattern/dir instead of the user reverse-engineering it. Two things this
    threads: ts_export_particles reads pick STARs from --input_directory (NOT
    --input_processing), so it points at the parent job's matching dir; and each
    export writes into a pick-set-named RELION dir (relion4/<tag>/) so multiple
    exports never collide and the RELION input path maps to its pick set."""
    # ---- membrane branch -------------------------------------------------
    # SPECIFIC pairs first, the generic folder-in/folder-out rule after: the
    # generic rule fills "input_dir", and the measurement stages' param is
    # "components" — with the generic rule first, their dedicated wiring below
    # was unreachable and Build-downstream silently prefilled a key the child
    # stage does not have (leaving "components" at its tune-era default).
    if child_stage == "mb_explore" and parent_stage in (
            "ts_reconstruct", "mb_deconv", "mb_isonet_predict",
            "mb_isonet2_predict"):
        # The sweep card's folder param is tomo_dir, not input_dir — and the
        # tomograms it sweeps ARE the greyscale source, so the breadcrumb
        # starts here for everything built downstream of the sweep.
        src = membrane_parent_dir(parent_stage, parent_params,
                                  parent_output_dir)
        return _mb_carry(parent_params, {
            "tomo_dir": src, "root": ".", "out": "jobs/{jobid}",
            GREY_KEY: src})
    if parent_stage in ("mb_segment", "mb_thresholds") \
            and child_stage == "mb_polarity":
        # tomogram/segmentation are FILES; the derive pins their FOLDERS so
        # the Browse buttons open in the right place, and the validator's
        # pairing check catches an unedited folder-for-file run loudly.
        return _mb_carry(parent_params, {
            "segmentation": membrane_parent_dir(parent_stage, parent_params,
                                                parent_output_dir),
            "tomogram": source_tomograms(parent_stage, parent_params),
            "root": "."})
    if parent_stage == "mb_polarity" and child_stage == "mb_fit_virions":
        # The measured verdict lives in the variant registry (looked up at
        # wiring time by the app); here only the breadcrumbs move.
        out = {"out_folder": "jobs/{jobid}"}
        tomo = str(parent_params.get("tomogram", "") or "").rstrip("/")
        if tomo:
            # The fitter wants the FOLDER of greyscale volumes; the polarity
            # card may hold either a file or (fresh from its own derive) a dir.
            out["tomogram"] = (os.path.dirname(tomo)
                               if tomo.endswith(".mrc") else tomo)
        return _mb_carry(parent_params, out)
    if parent_stage in ("mb_components", "mb_split_clusters") \
            and child_stage == "mb_size_qc":
        return _mb_carry(parent_params, {
            "components": membrane_parent_dir(parent_stage, parent_params,
                                              parent_output_dir),
            "tomogram": source_tomograms(parent_stage, parent_params),
            "out_folder": "jobs/{jobid}"})
    if parent_stage == "mb_population" and child_stage == "mb_size_qc":
        out = {"population": f"{str(parent_output_dir or '').rstrip('/')}"
                             "/population.json",
               "tomogram": source_tomograms(parent_stage, parent_params),
               "out_folder": "jobs/{jobid}"}
        if parent_params.get("components"):
            out["components"] = parent_params["components"]
        return _mb_carry(parent_params, out)
    if parent_stage == "mb_size_qc" and child_stage == "mb_split_clusters":
        # Split the ORIGINAL components the QC judged — same population.
        return _mb_carry(parent_params, {
            "components": parent_params.get("components", ""),
            "population": parent_params.get("population", ""),
            "out_folder": "jobs/{jobid}"})
    if parent_stage == "mb_components" and child_stage == "mb_population":
        # A FOLDER here, deliberately: the whole point of this step is to
        # measure across every tomogram at once, so it must not be prefilled
        # with a single volume the way the per-tomogram steps are.
        return _mb_carry(parent_params, {
            "components": membrane_parent_dir(parent_stage, parent_params,
                                              parent_output_dir),
            "out_folder": "jobs/{jobid}"})
    if parent_stage == "mb_components" and child_stage == "mb_split_clusters":
        return _mb_carry(parent_params, {
            "components": membrane_parent_dir(parent_stage, parent_params,
                                              parent_output_dir),
            "out_folder": "jobs/{jobid}"})
    if parent_stage == "mb_population" and child_stage in ("mb_split_clusters",
                                                           "mb_fit_virions"):
        # Both children read the measured population; the splitter cannot run
        # without it, and the fitter is far steadier with it.
        out = {"population": f"{str(parent_output_dir or '').rstrip('/')}"
                             "/population.json",
               "out_folder": "jobs/{jobid}"}
        if parent_params.get("components"):
            out["components"] = parent_params["components"]
        if child_stage == "mb_fit_virions":
            # The density-support gate reads actual densities — it needs the
            # greyscale variant the chain segmented, not whatever the form's
            # default points at.
            out["tomogram"] = source_tomograms(parent_stage, parent_params)
        return _mb_carry(parent_params, out)
    if parent_stage == "mb_split_clusters" and child_stage == "mb_fit_virions":
        # Fit the SPLIT volume, not the merged one it came from.
        return _mb_carry(parent_params, {
            "components": str(parent_output_dir or "").rstrip("/"),
            "population": parent_params.get("population", ""),
            "tomogram": source_tomograms(parent_stage, parent_params),
            "out_folder": "jobs/{jobid}"})
    if parent_stage == "mb_components" and child_stage == "mb_fit_virions":
        # The fitter takes the components FOLDER and fits every tomogram in
        # it, pairing each with its own tomogram. It used to take one file,
        # and this prefill handed it a folder — which opened as a directory
        # and died on the first volume.
        return _mb_carry(parent_params, {
            "components": membrane_parent_dir(parent_stage, parent_params,
                                              parent_output_dir),
            # NOT the components job's input — that is the THRESHOLDS
            # folder, i.e. binary masks. The density-support gate reads
            # actual densities, so it needs the greyscale volumes
            # segmentation ran on, carried down the chain.
            "tomogram": source_tomograms(parent_stage, parent_params),
            "out_folder": "jobs/{jobid}"})
    if parent_stage == "mb_fit_virions" and child_stage == "mb_pick_surfaces":
        # --out is the output star FILE. "jobs/{jobid}" alone named the job
        # DIRECTORY _run_job pre-creates, and write_text on a directory dies
        # with IsADirectoryError — the auto-wired hop failed every time.
        return {"fits": f"{str(parent_output_dir or '').rstrip('/')}/fits.json",
                "out": "jobs/{jobid}/oversample.star"}
    # Generic membrane rule LAST: folder in, folder out, breadcrumbs carried.
    if (parent_stage, child_stage) in MEMBRANE_CHAIN:
        src = membrane_parent_dir(parent_stage, parent_params, parent_output_dir)
        out = {"input_dir": src, "output_dir": "jobs/{jobid}"}
        if parent_stage == "mb_thresholds" and child_stage == "mb_components":
            # A single-value threshold parent knows EXACTLY which files the
            # child should label; without the pattern, MB_PATTERN='' means
            # *.mrc and a folder holding several swept values labels them
            # all as if they were one segmentation.
            toks = str(parent_params.get("MB_THRESHOLDS", "")
                       ).replace(",", " ").split()
            if len(toks) == 1:
                out["MB_PATTERN"] = f"*_threshold_{_norm_thr(toks[0])}.mrc"
        # Carry the tomogram restriction: a parent that segmented ONE series
        # must not hand its child a job that runs on all 72.
        if parent_params.get("MB_TOMO_LIST"):
            out["MB_TOMO_LIST"] = parent_params["MB_TOMO_LIST"]
        out[GREY_KEY] = source_tomograms(parent_stage, parent_params)
        if child_stage == "mb_mesh":
            out["tomo_dir"] = out[GREY_KEY]
        return out

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
        if child_stage == "ts_export_particles":
            # Two kinds of card live in this row and they name their star
            # differently: a RELION job adopted from disk has data_star, a
            # promoted Subset selection has source_star. Either way the export
            # reads that star DIRECTLY — no pick stars, no hand conversion.
            star = str(parent_params.get("data_star")
                       or parent_params.get("source_star") or "")
            if not star:
                return {}
            return direct_export_params(star, str(parent_params.get("job_dir") or ""))
        return {}

    # The same, for a promoted selection from before it moved into this row: a
    # ts_template_match-shaped card whose content is its source_star. (A
    # re-extract PICK SET also carries source_star, but with override_suffix and
    # tomo_angpix — that one is a folder of pick stars and keeps the pick-star
    # derivation below.)
    if (child_stage == "ts_export_particles" and parent_params.get("source_star")
            and not str(parent_params.get("override_suffix", "") or "").strip()):
        return direct_export_params(str(parent_params["source_star"]),
                                    str(parent_params.get("job_dir") or ""))

    # ts_import takes no --input_processing, so nothing points it at the parent
    # automatically -- its --frameseries has to be aimed by hand or it reads the
    # empty trunk. J7 wrote its averages into jobs/J7_fs-motion-and-ctf and J9
    # was left looking in warp_frameseries/.
    if child_stage == "ts_import" and parent_stage == "fs_motion_and_ctf":
        return {"frameseries": parent_output_dir or "warp_frameseries"}

    # The alignment chain is wrapper scripts taking explicit in/out dirs, so
    # NOTHING wires it automatically the way --input_processing does for the
    # WarpTools stages. Every hop was left pointing at the trunk: J13 read
    # jobs/J12_ts-stack/tiltstack (hand-fixed) but still wrote to the shared
    # aretomo_output/, and J14 imported from there rather than from J13.
    if child_stage == "aretomo" and parent_stage == "ts_stack":
        src = str(parent_output_dir or "warp_tiltseries").rstrip("/")
        return {"input_dir": f"{src}/tiltstack", "output_dir": "jobs/{jobid}"}
    if child_stage == "ts_import_alignments" and parent_stage == "aretomo":
        # AreTomo writes <output>/Imod/<series>_Imod/<series>.xf.
        out = str(parent_output_dir
                  or parent_params.get("output_dir", "") or "").rstrip("/")
        return {"alignments": f"{out}/Imod/"} if out else {}

    if child_stage == "threshold_picks" and parent_stage == "ts_template_match":
        return {"in_suffix": match_star_infix(parent_params)}
    # Verifying an export made from a RELION star: pair the source star with the
    # star the export wrote. Warp applied the refined shifts itself, so the check
    # runs WITH recentring (no_recenter off).
    if child_stage == "relion4_verify_reextract" and parent_stage == "ts_export_particles":
        out_star = str(parent_params.get("output_star", "") or "")
        out_star = out_star.replace("{jobid}", os.path.basename(parent_output_dir or ""))
        derived = {"new_star": out_star, "no_recenter": False}
        src = str(parent_params.get("input_star", "") or "").strip()
        if src:
            derived["source_star"] = src
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


def summarize_ts_export(job_dir):
    """Particles exported, and how many tomograms they came from.

    The generic summariser counted the 72 per-series XMLs Warp leaves beside
    the star and said nothing about particles — so an export that legitimately
    skipped 68 unpicked tomograms was indistinguishable on the card from one
    that produced nothing, and the only way to tell was a terminal. The number
    that matters is the row count in the star.

    Tomograms come from counting subtomo/ subdirectories (one readdir), NOT
    from walking the tree: a real export is thousands of files and gigabytes,
    and this runs on the UI thread the moment a job finishes.
    """
    d = Path(job_dir)
    out = {}
    star = next(iter(sorted(d.glob("*.star"))), None)
    if star is not None:
        n = star_particle_count(star)
        if n:
            out["particles"] = n
    sub = d / "subtomo"
    if sub.is_dir():
        try:
            n = sum(1 for f in sub.iterdir() if f.is_dir())
            if n:
                out["tomograms"] = n
        except OSError:
            pass
    return out


STAGE_SUMMARIZERS = {
    "ts_ctf": summarize_ts_ctf,
    "ts_reconstruct": summarize_ts_reconstruct,
    "ts_export_particles": summarize_ts_export,
}


_POLARITY_RE = re.compile(r"^\s*polarity:\s*(dark|bright|unknown)\b", re.I)


def polarity_result(line):
    """The polarity from ml_variant_polarity's own report line, or None.

    Its result is one WORD, written to the variant registry rather than to a
    file — so the job's folder is legitimately empty and the card had nothing
    to show. Caught from the log, the measurement lands on the card where it
    was made."""
    m = _POLARITY_RE.match(line or "")
    return m.group(1).lower() if m else None


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
                # Ignore anything WE wrote. ml_m_index_versions drops a label file
                # into every version folder, which then becomes the newest entry —
                # so running it once reset every folder's apparent write time to
                # "just now" and unmatched every round from its job. The tool
                # destroyed the evidence it exists to preserve.
                if e.name.startswith("_tomogration"):
                    continue
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
        # build_job_command substitutes {jobid} at run time, and
        # declared_output_dirs resolves it here -- this did not, so a job-scoped
        # output (mb_polarity's jobs/{jobid}) resolved to a literal '{jobid}'
        # that exists nowhere and silently contributed nothing.
        raw = raw.replace("{jobid}", job_dir_token(job))
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

    # Last resort: the stage's CONVENTIONAL output dir. Thirty of the wrapper
    # stages declare no output param at all, and the card then advertised
    # jobs/<id>/ -- a folder only WarpTools --output_processing stages ever
    # create. J4 wrote gains/original_gain.mrc and its card pointed at a
    # jobs/J4 that did not exist. STAGE_OUTPUTS already knew the answer; it
    # simply was not being asked. Skipped when a parameter resolved something
    # more specific, and "." is never an answer.
    # ...but ONLY for a stage with no output machinery at all. When a stage
    # DOES declare output_params/_subdirs/_find and none of them resolved, the
    # job genuinely wrote somewhere this cannot name (an M population saved
    # outside the project, say) and the convention would be a guess pointing
    # away from the truth -- "merely plausible is worse than none" is the rule
    # every other row here obeys.
    knows = (spec.get("output_params") or spec.get("output_subdirs")
             or spec.get("output_find"))
    if not out and not knows:
        conv = str(STAGE_OUTPUTS.get(spec.get("id"), "") or "").strip()
        if conv and conv != ".":
            add(conv, "where this step writes (by convention)")
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


def star_header_info(path):
    """What an export needs to know about a RELION/Warp particle star, from its
    header alone: {"pixel_size": float|None, "columns": [...], "has_optics": bool,
    "origins": bool}. None if the file cannot be read.

    Reads the optics block (one row per group) and the particle block's column
    list plus its FIRST data row, then stops — a 30 MB star costs a few KB.
    pixel_size is the optics rlnImagePixelSize (RELION 3.1+/4), else the first
    row's rlnPixelSize (raw Warp export), else rlnDetectorPixelSize. That is the
    pixel size the coordinates are counted in: the export's coords_angpix."""
    blocks, block, in_loop, cols, first = {}, "", False, [], None
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if s.startswith("data_"):
                    if block:
                        blocks[block] = (cols, first)
                    block, in_loop, cols, first = s[5:].strip() or "_", False, [], None
                    continue
                if s.startswith("loop_"):
                    in_loop = True
                    continue
                if s.startswith("_"):
                    name = s.split()[0][1:]
                    if in_loop:
                        cols.append(name)
                    else:                        # key-value line outside a loop
                        parts = s.split(None, 1)
                        cols.append(name)
                        first = (first or []) + [parts[1] if len(parts) > 1 else ""]
                    continue
                if in_loop and cols and first is None:
                    first = s.split()
                    if "rlnCoordinateX" in cols:  # the particle block: done
                        break
    except OSError:
        return None
    if block:
        blocks[block] = (cols, first)
    if not blocks:
        return None
    pb = next((b for b in blocks.values() if "rlnCoordinateX" in b[0]), None)
    if pb is None:
        pb = list(blocks.values())[-1]
    optics = next((b for n, b in blocks.items() if "optics" in n.lower()), None)

    def _val(colsrow, name):
        c, r = colsrow
        if r is None or name not in c:
            return None
        try:
            return float(r[c.index(name)])
        except (ValueError, IndexError):
            return None

    apx = _val(optics, "rlnImagePixelSize") if optics is not None else None
    if not apx:
        apx = (_val(pb, "rlnImagePixelSize") or _val(pb, "rlnPixelSize")
               or _val(pb, "rlnDetectorPixelSize"))
    return {"pixel_size": apx or None, "columns": list(pb[0]),
            "has_optics": optics is not None,
            "origins": any(c in pb[0] for c in ("rlnOriginXAngst", "rlnOriginX"))}


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


def would_cycle(store, job_id, parent_id):
    """True if feeding `job_id` from `parent_id` closes a loop.

    set_job_parent already refuses to make a job its own parent, but a longer
    loop (A feeds B, then B is wired to feed A) is just as impossible and the
    canvas would draw it forever. Walking up from the proposed parent answers
    it exactly; the `seen` set means an ALREADY-cyclic store terminates instead
    of hanging the UI that is trying to describe it."""
    jobs = (store or {}).get("jobs") or {}
    seen, cur = set(), parent_id
    while cur and cur not in seen:
        if cur == job_id:
            return True
        seen.add(cur)
        cur = parent_job_id(jobs.get(cur) or {})
    return False


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
        if (parent_id == job_id or parent_id not in (store.get("jobs") or {})
                or would_cycle(store, job_id, parent_id)):
            return None
        inputs[slot] = parent_id
    else:
        inputs.pop(slot, None)
    job["inputs"] = inputs
    save_jobs(root, store)
    return job


# Params a job legitimately carries that its stage never declared. They are
# BREADCRUMBS, not stale settings: a RELION -> Warp re-extract and a crYOLO
# pick set are both stored as ts_template_match jobs so the Export wiring and
# adoption work, and source_star records which RELION star the particles came
# from — which is exactly the provenance you go looking for months later.
# Reporting them as parameters "this stage no longer has" reads as damage from
# an upgrade, when nothing was ever lost.
CARRIED_PARAMS = {
    "source_star": "the RELION star these picks were re-extracted from",
    "job_dir": "the RELION job directory this came from",
    "orphan_suffix": "the pick-set suffix this run wrote",
}


def carried_note(dropped, params):
    """(carried, stale) split of the names params_for_builder dropped."""
    carried = {k: params.get(k) for k in dropped if k in CARRIED_PARAMS}
    stale = [k for k in dropped if k not in CARRIED_PARAMS]
    return carried, stale


def actual_job_kind(job):
    """What a job IS, when that differs from the stage it is filed under.

    Three kinds ride ts_template_match so their downstream wiring works, and
    the canvas already titles them by what they are. The builder resolves by
    stage_id alone, so it says 'Template matching' for all of them — which is
    why an old re-extract looks like a template match when you reopen it."""
    if (job or {}).get("tool") == "reextract":
        return "a RELION → Warp re-extract"
    if is_cryolo_job(job or {}):
        return "a crYOLO pick set"
    if (job or {}).get("tool") == "relion_selection":
        return "a RELION selection"
    return ""


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
# ===========================================================================
# Viewer hand-off: which volumes a finished membrane job is worth LOOKING at
# ===========================================================================
# napari via tomoview for volumes; surforama for mesh containers. The env
# matters as much as the tool: napari lives in membrainseg (PyQt6) and
# surforama in membrainpick (PyQt5), and the two must never be merged.
VIEWER_PLANS = {
    "mb_segment":        ("tomoview", ("*_segmented.mrc", "*_scores.mrc")),
    "mb_thresholds":     ("tomoview", ("*_threshold_*.mrc",)),
    "mb_components":     ("tomoview", ("*.mrc",)),
    "mb_deconv":         ("tomoview", ()),      # the output IS the tomogram
    "mb_isonet_predict":  ("tomoview", ()),
    "mb_isonet2_predict": ("tomoview", ()),
    "mb_fit_virions":    ("tomoview", ("fit_*.mrc",)),
    "mb_size_qc":        ("tomoview", ("outliers_only.mrc",)),
    "mb_split_clusters": ("tomoview", ("*_split.mrc",)),
    "mb_mesh":           ("surforama", ("*.h5",)),
    # The sweep's unit is a matrix CELL, not a series: one row is a variant +
    # threshold + cutoff + tomogram, and comparing two rows is the whole point
    # of the card. Its inventory is built from results.json instead of a glob.
    "mb_explore":        ("tomoview", ()),
}


def viewer_plan(stage_id, out_dir, series=None, input_dir=None, root="."):
    """(tool, env, [paths]) for opening a finished job, or (None, None, []).

    tomoview wants the TOMOGRAM first and overlays after, so the source volume
    is found from the job's input folder and the overlays from its output. A
    job that segmented 3 tomograms has 3 sets; `series` picks one, because
    opening all of them at once is a wall of layers nobody reads."""
    plan = VIEWER_PLANS.get(stage_id)
    if not plan:
        return (None, None, [])
    tool, patterns = plan
    env = "membrainpick" if tool == "surforama" else "membrainseg"
    root = Path(root)
    out = root / out_dir if not os.path.isabs(str(out_dir)) else Path(out_dir)
    files = []

    # The tomogram itself, when there is a separate input folder to find it in.
    if input_dir:
        src = root / input_dir if not os.path.isabs(str(input_dir)) else Path(input_dir)
        if src.is_dir() and series:
            # The series must end at a boundary. A bare '{series}*' glob lets
            # Position10 match Position106's tomogram, and the viewer would
            # then draw one tomogram's components over another's densities —
            # wrong, and impossible to notice by eye.
            hits = [f for f in sorted(src.glob(f"{series}*.mrc"),
                                      key=lambda f: natural_key(f.name))
                    if f.stem == series or f.name[len(series)] in "_."]
            if hits:
                files.append(str(hits[0]))
        elif src.is_file():
            files.append(str(src))

    # Match on the series a file BELONGS to, not on its basename: a batch tool
    # writes fixed names (fit_1.mrc) into a per-tomogram folder, and those
    # names contain no series at all.
    pats = patterns or ("*.mrc",)         # no patterns = the volumes ARE the subject
    for pat in pats:
        for f in sorted(out.rglob(pat), key=lambda p: natural_key(p.relative_to(out))):
            if series and output_series_of(f, out) != series:
                continue
            if str(f) not in files:
                files.append(str(f))
    return (tool, env, files)


def resolve_viewer_tool(name, repo_dir, home=None):
    """Where a viewer actually LIVES on this machine, or "" if nowhere.

    tomoview is not part of tomogration — it is a separate script the user
    keeps in ~/bin, and it may simply not have synced with the app. Look in
    the app folder, then on PATH, then the usual ~/bin, and return an absolute
    path so the wrapper's file check passes. A bare command name that the conda
    env provides (surforama) is returned unchanged: only the env can resolve
    it, and it does that at launch."""
    if not str(name).endswith(".py"):
        return name                        # console command; env resolves it
    cand = Path(repo_dir) / name
    if cand.is_file():
        return str(cand)
    found = shutil.which(name) or shutil.which(Path(name).stem)
    if found:
        return found
    hm = Path(home or Path.home())
    for rel in (f"bin/{name}", f"bin/{Path(name).stem}"):
        if (hm / rel).is_file():
            return str(hm / rel)
    return ""


def explore_inventory(out_dir, root="."):
    """{row label: [tomogram, components, fits…]} for a parameter sweep.

    Groups by matrix CELL rather than by series, best-scoring first, so the
    picker offers "raw thr=-2 cut=1000@12.56 Position003" and two of them can
    be ticked to compare. Rows whose stages failed have no files and are left
    out — offering a row that opens nothing is worse than not listing it."""
    base = Path(root) / out_dir if not os.path.isabs(str(out_dir)) else Path(out_dir)
    try:
        data = json.loads((base / "results.json").read_text())
    except (OSError, ValueError):
        return {}
    out = {}
    for r in data.get("rows", []):
        # is_file(), not exists(): a directory passes exists() and is what a
        # half-resolved row leaves behind — offering one to napari opens
        # nothing and looks like a naming bug.
        files = [f for f in [r.get("tomogram_path"), r.get("components_path")]
                 if f and Path(root, f).is_file()]
        files += [f for f in (r.get("fit_masks") or [])
                  if Path(root, f).is_file()]
        if len(files) < 2:               # a tomogram alone is not worth a row
            continue
        label = (f"{r.get('variant')} thr={r.get('threshold')} "
                 f"cut={r.get('cutoff')} {r.get('tomogram')}")
        n = r.get("virions_accepted")
        if n is not None:
            label += f"  ({n} virions, recall {r.get('recall', 0):.2f})"
        out[label] = files
    return out


def viewer_inventory(stage_id, out_dir, input_dir=None, root="."):
    """{series: [paths]} — everything a finished job has worth looking at.

    The tomogram comes FIRST within each series, because tomoview builds its
    base image from the first path and scales every later layer to that shape.
    Ordering here rather than in the dialog keeps that rule in one place."""
    if stage_id == "mb_explore":
        return explore_inventory(out_dir, root)
    plan = VIEWER_PLANS.get(stage_id)
    if not plan:
        return {}
    _tool, patterns = plan
    root = Path(root)
    out = root / out_dir if not os.path.isabs(str(out_dir)) else Path(out_dir)
    if not out.is_dir():
        return {}

    # ONE walk per pattern, not one per series. This used to call viewer_plan
    # in a loop, so a 72-tomogram job walked the output tree 73 times and
    # globbed the tomogram folder 72 times — on the UI thread. On a local disk
    # that is a stutter; on /ceph, where every stat is a network round trip, it
    # froze the app long enough for the desktop to offer to kill it.
    # ONE index of the tomogram folder, keyed by series, built FIRST so its
    # stems can name outputs whose own filenames no longer carry a series. An
    # exact key also removes the prefix hazard by construction: Position10
    # cannot pick up Position106's volume however the names are sorted.
    tomo_of, single = {}, None
    if input_dir:
        src = (root / input_dir if not os.path.isabs(str(input_dir))
               else Path(input_dir))
        if src.is_dir():
            for f in sorted(src.glob("*.mrc"), key=lambda q: natural_key(q.name)):
                tomo_of.setdefault(_tagged_stem(f.name) or f.stem, str(f))
        elif src.is_file():
            single = str(src)

    grouped = {}
    for pat in (patterns or ("*.mrc",)):
        for f in out.rglob(pat):
            grouped.setdefault(output_series_of(f, out, tomo_of),
                               set()).add(str(f))

    inv = {}
    for stem in sorted(grouped, key=natural_key):
        files = []
        # Only tomoview takes a base image. surforama is handed ONE .h5
        # container whose densities were already projected from the tomogram
        # when the mesh was built, so offering the .mrc there would just be an
        # entry that cannot be opened.
        head = (single or tomo_of.get(stem)) if _tool == "tomoview" else None
        if head:
            files.append(head)                 # the base image goes first
        files += [f for f in sorted(grouped[stem], key=natural_key)
                  if f != head]
        if files:
            inv[stem] = files
    return inv


def viewer_series(out_dir, root="."):
    """Series stems a finished job produced, so one can be chosen to view."""
    root = Path(root)
    out = root / out_dir if not os.path.isabs(str(out_dir)) else Path(out_dir)
    if not out.is_dir():
        return []
    seen = {}
    for f in out.rglob("*.mrc"):
        stem = output_series_of(f, out)
        seen.setdefault(stem, 0)
        seen[stem] += 1
    return sorted(seen, key=natural_key)


def series_stem_of(name):
    """Series stem of an output filename: everything before the pixel-size tag,
    which every membrane tool appends its own suffixes after."""
    return _tagged_stem(name) or name.rsplit(".", 1)[0]


def _tagged_stem(name):
    """The series stem when the NAME ITSELF carries a pixel-size tag, else None.

    Separated from series_stem_of because the difference matters: a name with
    no tag has not told us which series it belongs to, and treating its own
    basename as the answer is a guess."""
    m = re.match(r"^(.+?)_\d+(?:[.p]\d+)?Apx", name)
    return m.group(1) if m else None


def natural_key(name):
    """Sort key that puts fit_2 before fit_10."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", str(name))]


def _prefix_series(name, known):
    """The longest KNOWN series that prefixes `name` at a boundary.

    The last resort, and the one that rescues third-party names. membrain_pick
    builds its mesh filenames by truncating BOTH inputs at the first dot and
    concatenating them, so 'Position002_12.56Apx_isonet2.mrc' plus its
    segmentation becomes 'Position002_12_Position002_12_48.h5'. The pixel-size
    tag every other test keys on is gone, so each of ~70 meshes per tomogram
    became its own top-level 'series' and no tomogram was offered with any of
    them. Matching against the series we already know exist puts them back
    under the right tomogram.

    Longest first, and the match must end at a separator, so Position10 cannot
    claim Position106's meshes."""
    for k in sorted(known or (), key=len, reverse=True):
        if name.startswith(k) and (len(name) == len(k)
                                   or name[len(k)] in "_.-"):
            return k
    return ""


def output_series_of(path, out, known=()):
    """Which series an output file belongs to.

    A filename carrying its own pixel-size tag answers for itself. One that
    does NOT — fit_1.mrc, fits.json, and every other fixed name a batch tool
    writes into a per-tomogram SUBFOLDER — inherits from the folder it sits in.

    Reading the basename alone made every one of 85 fit masks its own series,
    so the viewer offered 85 top-level entries called fit_1, fit_10, fit_104
    with no tomogram to attach them to — and because the same basename test
    also filtered the file list, picking a real tomogram would have excluded
    that tomogram's own masks."""
    p = Path(path)
    tagged = _tagged_stem(p.name)
    if tagged:
        return tagged
    try:
        rel = p.relative_to(out)
    except ValueError:
        rel = None
    if rel is not None:
        for part in reversed(rel.parts[:-1]):   # innermost folder wins
            tagged = _tagged_stem(part)
            if tagged:
                return tagged
    return _prefix_series(p.name, known) or p.stem


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
    "relion4_convert": "RELION 4: convert STAR",
    "relion4_to_warp": "RELION 4 → Warp: pick-star converter (retired)",
    "relion4_merge_optics": "RELION 4: merge optics groups",
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
    "mb_deconv": "Membrane: deconvolve",
    "mb_isonet_train": "IsoNet: train",
    "mb_isonet_predict": "IsoNet: predict",
    "mb_isonet2_train": "IsoNet 2: train",
    "mb_isonet2_predict": "IsoNet 2: predict",
    "mb_segment": "Membrane: segment",
    "mb_thresholds": "Membrane: threshold sweep",
    "mb_components": "Membrane: components",
    "mb_mesh": "Membrane: meshes",
    "mb_polarity": "Variant polarity",
    "mb_fit_virions": "Fit virions",
    "mb_population": "Population radius + volume",
    "mb_size_qc": "Virion size QC (outliers)",
    "mb_split_clusters": "Split merged virions",
    "mb_pick_surfaces": "Pick surfaces",
    "mb_explore": "Parameter search",
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


# A failed run leaves a .NET stack trace in the terminal and nothing else. The
# trace names the C# file that threw, which tells you nothing about what to
# change on the card. Each entry: (signature, what it means and what to do).
RUN_FAILURE_HINTS = (
    ("Found 0 files in",
     "The pick-star PATTERN matched nothing, so WarpTools built an empty "
     "table and died on it (System.IndexOutOfRangeException in Star..ctor — "
     "a crash, not a message).\n"
     "  * The line just above says which folder and which glob it tried. "
     "List that folder and compare: the suffix on the files is set by "
     "whatever wrote them, and a pick set written with --suffix reextract is "
     "'*reextract.star', not '*reextract_2.star'.\n"
     "  * If you meant a NEW pick set, the step that writes it has to run "
     "FIRST — an export cannot invent stars that were never written.\n"
     "  * Nothing was read and nothing written, so fixing the pattern and "
     "re-running is safe."),
    ("Worker process did not connect",
     "WarpTools could not start its GPU workers. It spawns one worker per "
     "device per 'workers per device' and gives up if they do not call back "
     "in time. Every particle was already parsed by this point, so your picks "
     "are fine and nothing was written — re-running is safe.\n"
     "  * If nvidia-smi shows the GPUs BUSY, that is the cause: a worker "
     "cannot start on a card another job is filling.\n"
     "  * If they are IDLE, the workers are failing to LAUNCH rather than "
     "failing to fit, and lowering the count will not help. Bisect: device "
     "list '0', workers per device 1. Working means contention; failing the "
     "same way means the worker process itself never starts.\n"
     "  * Look for stale workers left by a killed run — 'pgrep -af "
     "WarpWorker'. They hold the pipes the new ones connect on.\n"
     "  * Start a worker BY HAND — the fastest way to a real error, since "
     "the master swallows the worker's output: 'WarpWorker --device 0 --port "
     "32456'. It binds a TCP port for its REST API and reports that port back "
     "through a named pipe, so a healthy one initialises the GPU and waits. "
     "(Activate the warp env first, or you are testing the wrong PATH.)\n"
     "  * Check /tmp. .NET named pipes are Unix sockets under TMPDIR, so a "
     "full or unwritable /tmp — or a TMPDIR pointed at network storage — "
     "stops the handshake with no error of its own: 'df -h /tmp; echo "
     "$TMPDIR'. This is the one that fits 'it used to work and the GPUs are "
     "idle'."),
    ("out of memory",
     "The GPU ran out of memory. Lower 'Workers per device' first (each worker "
     "holds its own copy), then the box size. Re-running is safe."),
    ("CUDA error",
     "A CUDA call failed. Usually the GPU is held by another job or in a bad "
     "state — check nvidia-smi before re-running."),
)


# A run can exit 0 and still have thrown work away. WarpTools deselects a movie
# whose CTF fit diverges and CARRIES ON -- one line among thousands, an hour of
# log ago -- so the card goes green and the loss is invisible until a tilt turns
# up missing from a series. These signatures are COUNTED, not matched once: the
# number is the whole finding, and it belongs on the card next to the successes.
RUN_WARNING_SIGNATURES = (
    {"sig": "marked as unselected",
     "key": "deselected",
     "label": "item(s) deselected by WarpTools",
     "why": "WarpTools could not process them, so it dropped them and carried "
            "on. Deselection PERSISTS in the Warp metadata: ts_ctf, "
            "ts_reconstruct, ts_export_particles and M all skip a deselected "
            "series silently, and report success having done nothing. Check "
            "the count against what you expected -- if it is every item, the "
            "step itself was misconfigured rather than the data being bad. "
            "'WarpTools change_selection --select' puts them back."},
)


# An exit code of 0 is not proof that anything happened. WarpTools 2.0.0 answers
# an unrecognised option by printing its full help and exiting ZERO, so J8
# (create_settings) and J9 (ts_import) were both recorded "completed (exit 0)"
# having written no settings file and no tomostar/ -- and the pipeline marched on
# until J12 tripped over the absence. A job that provably did nothing must be
# FAILED, whatever the shell says.
_FALSE_SUCCESS = (
    (re.compile(r"option '[^']+' is unknown", re.I),
     "WarpTools rejected an option, printed its help and exited 0 — nothing "
     "ran and nothing was written. The unknown option is named in the log just "
     "above. Not every WarpTools subcommand takes --input_processing/"
     "--output_processing: the ones that CREATE the processing context "
     "(create_settings, ts_import) do not."),
)


# A batch step that processed 76 of 81 series has not failed -- it has a result
# and five casualties. The wrapper exits 2 to say exactly that, and treating it
# as a failure marked a card red over 76 usable alignments and dequeued
# everything behind it. Only stages that declare `partial_exit` get this
# reading: `exit 2` means "edit the config and run again" in other wrappers,
# which really is a hard failure.
_BATCH_TALLY = re.compile(
    r"^\s*(Processed successfully|Skipped|Failed)\s*:\s*(\d+)\s*$", re.I)
_TALLY_KEY = {"processed successfully": "ok", "skipped": "skipped",
              "failed": "failed"}


def batch_tally_line(line):
    """('ok'|'skipped'|'failed', n) for a wrapper summary line, else None."""
    m = _BATCH_TALLY.match(str(line or ""))
    if not m:
        return None
    return (_TALLY_KEY[m.group(1).lower()], int(m.group(2)))


def partial_batch_result(spec, code, tally):
    """(status, note) for a run that ended on a declared partial exit code.

    Returns (None, "") when this is not a partial outcome, so the caller keeps
    whatever the exit code said. A batch where NOTHING succeeded is a failure
    however it exits -- there is no result to carry forward.
    """
    want = (spec or {}).get("partial_exit")
    if want is None or code != want:
        return (None, "")
    ok = int((tally or {}).get("ok", 0))
    bad = int((tally or {}).get("failed", 0))
    if ok <= 0:
        return ("failed", f"every item failed ({bad} of {ok + bad})")
    total = ok + bad + int((tally or {}).get("skipped", 0))
    return ("completed",
            f"{ok} of {total} succeeded, {bad} failed — the card is green "
            f"because the run produced usable output for the rest. The failures "
            f"are named above, each with its own log.")


def false_success_reason(line):
    """Why a line proves an 'exit 0' run actually did nothing, or ''.

    Pure and per-LINE, because the app keeps no log tail: by the time the
    process exits the evidence has scrolled past.
    """
    text = str(line or "")
    for pat, why in _FALSE_SUCCESS:
        if pat.search(text):
            return why
    return ""


def run_warning_key(line):
    """The warning signature a single log line matches, or None.

    Matched per LINE and accumulated by the caller, because the app keeps no
    log tail -- by the time the process exits the lines are gone.
    """
    low = str(line or "").lower()
    for w in RUN_WARNING_SIGNATURES:
        if w["sig"] in low:
            return w["key"]
    return None


def run_warning_report(counts):
    """[(key, count, one-line summary, explanation), ...] for a finished run.

    Pure so the wording is testable. Zero and negative counts are dropped --
    "0 items deselected" is noise on a card that succeeded cleanly.
    """
    out = []
    for w in RUN_WARNING_SIGNATURES:
        n = int((counts or {}).get(w["key"], 0) or 0)
        if n > 0:
            out.append((w["key"], n, f"{n} {w['label']}", w["why"]))
    return out


def run_failure_hint(text):
    """An actionable explanation for a known failure signature, or ''.

    Pure and case-insensitive so it can be tested without a terminal. Matched
    on the FIRST signature that appears, since a stack trace often mentions
    several layers of the same failure."""
    low = str(text or "").lower()
    for sig, hint in RUN_FAILURE_HINTS:
        if sig.lower() in low:
            return hint
    return ""


def newly_added(prev_index, index):
    """The card a repaint introduced and should scroll to, or None.

    Pure decision logic, kept out of the Qt paint path so it can be tested —
    the same reason card_is_running lives here.

    Three things it deliberately refuses:
      * the FIRST paint (prev_index is None), where every card is new and
        snapping anywhere would be arbitrary;
      * ghosts and templates, which appear and vanish as stages gain jobs;
      * ORPHANS. Those are found on disk by a timer, so snapping to one would
        yank the view while the user is working somewhere else. Adopting an
        orphan creates a real job with its own id, and THAT is what scrolls.

    With several new at once, the newest by job number wins.
    """
    if prev_index is None:
        return None
    fresh = [i for i, n in (index or {}).items()
             if i not in prev_index
             and not n.get("is_ghost") and not n.get("is_template")
             and not n.get("is_orphan")]
    return max(fresh, key=_job_seq) if fresh else None


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
    # is_template, not just is_ghost: discovered 'disk:<stage>' cards are ghosts
    # too, and must not light up alongside the rail template during a trunk run.
    return bool(act.get("stage_id")
                and act["stage_id"] == node.get("stage_id")
                and node.get("is_ghost") and node.get("is_template"))


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
    # A RETIRED stage keeps a row only while the project still holds its
    # cards; otherwise it is left out entirely — no template card, no rail edge.
    visible = [sp for sp in STAGES if not sp.get("legacy") or by_stage.get(sp["id"])]
    for row, spec in enumerate(visible):
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
                     # A hand-set verdict must LOOK different from a measured
                     # one, or the canvas quietly presents an opinion as a fact.
                     "manual": bool(job.get("manual_status")),
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
    order = [sp["id"] for sp in visible]
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
            "x": RAIL_W + col * (CARD_W + GAP_X),
            "y": row_of[sid] * (CARD_H + GAP_Y),
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
                found.append({"stage_id": "relion4_result", "kind": "relion_job",
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
