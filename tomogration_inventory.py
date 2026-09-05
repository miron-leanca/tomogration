#!/usr/bin/env python3
# Part of tomogration2 — split out of the single-file app so the pieces can be
# edited (and tested) independently. Import graph is a strict DAG:
#     core  ->  stages  ->  jobs  ->  inventory     (project depends only on core)
# tomogration_app.py imports all of them and holds the Qt window.

"""The project inventory: what ran, and where its files are.

Feeds the input-field browser ("pick an input from what already exists"). Two
levels, both PURE and bounded so a ceph project can't freeze the GUI:

  inventory_sources()  — every unit of past work as one row: jobs from the
                         store (newest first), plus the work that exists only
                         as directories (trunk-era pipeline dirs, AreTomo
                         versions, RELION jobs, relion4 exports, membrane
                         runs, crYOLO pick sets).
  list_dir()           — one directory level: subdirs first, then files, with
                         LARGE homogeneous families collapsed to one row per
                         name template ("Position###_12.56Apx.mrc · 72 files").
                         The EML46 inventory has 17k-file subtomo dirs; a
                         browser that lists them raw is a browser that hangs.

Lessons baked in from the real EML46 project dump (docs/membrain_module_spec
era, 2026-08): job dirs get RENAMED by hand (jobs/J27-bin8-isonet2model), so a
job's folder is resolved by J-number prefix, never by trusting output_dir
verbatim; noise dirs (results~, filtered_tmp, __pycache__) are hidden."""

import datetime
import os
import re
from pathlib import Path

from tomogration_jobs import (       # noqa: F401  (re-exported for the app)
    load_jobs, stage_title, _job_seq, job_output_dir, _fuzzy_job_dir,
)

# Never shown while browsing: backup/temp/noise. (Hidden, not forbidden — the
# user can still type such a path by hand.)
_HIDDEN_RE = re.compile(r"^(\.|__pycache__$|filtered_tmp$)|~$")

# Digit runs become '#' so Position002/Position115 share one template. The
# pixel-size tag is kept literal — different binnings must NOT collapse into
# one row, they are different volumes. Both spellings occur in real projects:
# 12.56Apx and the dot-free 12p56Apx some tools write.
_DIGITS_RE = re.compile(r"\d+")
_APX_RE = re.compile(r"(\d+[.p]\d+)Apx")


def name_template(name):
    """Collapse a filename to its family template: digit runs -> '#', but
    pixel-size tags stay literal (12.56Apx != 6.28Apx)."""
    keep = {}
    def _hold(m):
        # Digit-free placeholder — a digit in it would itself be collapsed.
        key = "\x00" + chr(ord("a") + len(keep)) + "\x00"
        keep[key] = m.group(0)
        return key
    s = _APX_RE.sub(_hold, name)
    s = _DIGITS_RE.sub("#", s)
    for key, lit in keep.items():
        s = s.replace(key, lit)
    return s


def collapse_files(names, min_family=4):
    """Group filenames into display rows. A template with >= min_family
    members becomes ONE row {'template', 'count', 'files'}; smaller groups
    stay individual rows. Rows keep first-seen order."""
    order, groups = [], {}
    for n in names:
        t = name_template(n)
        if t not in groups:
            groups[t] = []
            order.append(t)
        groups[t].append(n)
    rows = []
    for t in order:
        fs = groups[t]
        if len(fs) >= min_family:
            rows.append({"template": t, "count": len(fs), "files": fs})
        else:
            rows.extend({"name": n} for n in fs)
    return rows


def list_dir(root, rel, cap=600, min_family=4):
    """One level of a project directory, browser-shaped and bounded.

    Returns (entries, truncated). Each entry:
      {"name", "rel", "is_dir": True,  "count": n_children}          a subdir
      {"name", "rel", "is_dir": False, "size": bytes}                a file
      {"name": template, "rel": rel-of-dir, "is_dir": False,
       "family": [names], "count": n}                                collapsed
    Reads at most `cap` directory entries (ceph discipline) and says so."""
    base = Path(root) / rel if rel not in ("", ".") else Path(root)
    dirs, files, truncated = [], [], False
    try:
        with os.scandir(base) as it:
            for i, e in enumerate(it):
                if i >= cap:
                    truncated = True
                    break
                if _HIDDEN_RE.search(e.name):
                    continue
                try:
                    if e.is_dir(follow_symlinks=False):
                        dirs.append(e.name)
                    else:
                        files.append((e.name, e.stat(follow_symlinks=False).st_size))
                except OSError:
                    continue
    except OSError:
        return [], False
    rrel = "" if rel in ("", ".") else str(rel).rstrip("/")
    def _rel(name):
        return f"{rrel}/{name}" if rrel else name
    entries = []
    for d in sorted(dirs):
        # A cheap child count caps at 100 — enough to say "lots".
        n = 0
        try:
            with os.scandir(base / d) as it:
                for n, _ in enumerate(it, 1):
                    if n >= 100:
                        break
        except OSError:
            pass
        entries.append({"name": d, "rel": _rel(d), "is_dir": True, "count": n})
    sizes = dict(files)
    for row in collapse_files(sorted(sizes), min_family=min_family):
        if "template" in row:
            entries.append({"name": row["template"], "rel": rrel,
                            "is_dir": False, "family": row["files"],
                            "count": row["count"]})
        else:
            n = row["name"]
            entries.append({"name": n, "rel": _rel(n), "is_dir": False,
                            "size": sizes[n]})
    return entries, truncated


def job_locations(root, job):
    """Every on-disk place this job's artifacts live, deduped, jobs/ first.

    output_dir always says jobs/<id>, but the REAL products often live where a
    PARAM pointed them (the EML46 store proves it: exports in
    relion4/<suffix>_J10, IsoNet work dirs in membrane/isonet2, predictions
    written into ANOTHER job's renamed folder). So: the resolved jobs/ dir
    plus every param value that names an existing project directory."""
    root = Path(root)
    locs = []
    jid = job.get("id", "")
    own = resolve_job_dir(root, jid) if jid else ""
    if own:
        locs.append(own)
    for v in (job.get("params") or {}).values():
        s = str(v or "").strip().rstrip("/")
        if not s or "/" not in s or s.startswith("/") or "*" in s:
            continue
        if (root / s).is_dir() and s not in locs:
            locs.append(s)
    return locs


def resolve_job_dir(root, jid):
    """The job's REAL folder: jobs/J27 may be named jobs/J27_isonet-predict
    (the app's own convention) or hand-renamed to jobs/J27-bin8-isonet2model —
    matched by J-number prefix at a boundary (J2 never claims J27's folder).
    Shared with the delete-target rules, so what the browser shows and what
    Clear removes can never disagree."""
    return _fuzzy_job_dir(root, jid)


# Directory families that hold work not represented by any job record. Each
# matcher yields (title, subtitle) rows for what it finds on disk.
_PIPELINE_DIRS = [
    ("warp_tiltseries/reconstruction", "Tomograms (shared reconstruction)"),
    ("warp_tiltseries/matching", "Template-match / pick stars"),
    ("warp_tiltseries", "Warp tilt-series metadata"),
    ("tomostar", "Tilt series (.tomostar)"),
    ("mdocs", "Mdocs"),
    ("membrane", "Membrane branch"),
    ("relion4", "RELION exports"),
]
_VERSIONED_RE = re.compile(r"^aretomo_output(-v\d+)?$")
_RELION_JOB_DIRS = ("Class3D", "Refine3D", "InitialModel", "Select", "Extract")


def inventory_sources(root, store=None, cap_jobs=200):
    """Every unit of past work, one row each, jobs first (newest first).

    Row: {"kind": job|dir|relion, "id", "title", "subtitle", "rel"}.
    Bounded: no recursive walks — only the store plus single-level peeks at
    the known families."""
    root = Path(root)
    if store is None:
        store = load_jobs(root)
    rows = []
    jobs = store.get("jobs", {}) or {}
    for jid in sorted(jobs, key=_job_seq, reverse=True)[:cap_jobs]:
        j = jobs[jid] or {}
        locs = job_locations(root, j) or [j.get("output_dir", "")]
        title = f"{jid} · {stage_title(j.get('stage_id', ''), j.get('stage_id', ''))}"
        when = (j.get("finished") or j.get("started") or j.get("created") or "")
        sub = " · ".join(x for x in (j.get("status", ""), when.split(" ")[0],
                                     j.get("label", "")) if x)
        rows.append({"kind": "job", "id": jid, "title": title,
                     "subtitle": sub, "rel": locs[0], "locs": locs})
    # RELION pipeline job dirs at the root (Class3D/job003 …).
    for fam in _RELION_JOB_DIRS:
        d = root / fam
        if not d.is_dir():
            continue
        try:
            subs = sorted(x.name for x in d.iterdir() if x.is_dir())
        except OSError:
            subs = []
        for s in subs:
            rows.append({"kind": "relion", "id": f"{fam}/{s}",
                         "title": f"RELION {fam}/{s}",
                         "subtitle": "pipeline job on disk",
                         "rel": f"{fam}/{s}"})
    # AreTomo versioned folders.
    try:
        for d in sorted(root.iterdir()):
            if d.is_dir() and _VERSIONED_RE.match(d.name):
                rows.append({"kind": "dir", "id": d.name,
                             "title": f"AreTomo · {d.name}",
                             "subtitle": "versioned alignment run",
                             "rel": d.name})
    except OSError:
        pass
    # Canonical pipeline dirs + browse-from-root.
    for rel, label in _PIPELINE_DIRS:
        if (root / rel).is_dir():
            rows.append({"kind": "dir", "id": rel, "title": label,
                         "subtitle": rel, "rel": rel})
    rows.append({"kind": "dir", "id": ".", "title": "Project root",
                 "subtitle": "browse everything", "rel": ""})
    return rows
