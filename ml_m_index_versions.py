#!/usr/bin/env python3
"""ml_m_index_versions.py — label M's randomly-named version folders.

WHY THIS EXISTS
    Every MCore run commits a new version of each species into

        m/species/<species>_<hash>/versions/<random>/

    where <random> is an opaque token like "Hhhxx2to" or "-2-ij4-g". M keeps no
    index, the names carry no order, and nothing inside says which run produced
    it.  After a handful of refinement rounds you have a row of folders and no
    way to tell which one was the 5.96 A run.

    Renaming them is NOT an option: the .species file and the population refer to
    version names internally, so renaming silently breaks the species.  Instead
    this reads each folder's modification time, matches it against tomogration's
    job store (.tomogration_jobs.json), and — with --write — drops a plain-text
    label INSIDE each folder.  Nothing M reads is touched.

    A version folder is matched to the job whose run window contains its
    timestamp.  Runs started from a terminal rather than tomogration have no job
    to match and are reported as such, with their timestamp, which is still
    enough to line them up against your notes.

Usage:
    python3 ml_m_index_versions.py <project_dir> [--write] [--species NAME]

      --write        write _tomogration_version.txt into each version folder
                     (safe: M never reads it). Without this the tool only prints.
      --species NAME only index species whose folder name contains NAME.

Exit code 0 = indexed, 2 = nothing found / could not run.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

INFO_NAME = "_tomogration_version.txt"
TS_FMT = "%Y-%m-%d %H:%M:%S"

# Every version folder holds a <name>.species file, and it records what the round
# ACHIEVED and what it came FROM:
#     <Param Name="GlobalResolution" Value="4.295464" />
#     <Param Name="PreviousVersion"  Value="kKTyRA7g" />
# That is authoritative and survives regardless of how the run was launched, so it
# beats both the job store (which only knows runs made through tomogration) and the
# write-time match (which is an inference). Read it first, fall back to the job.
_SPECIES_PARAM = re.compile(r'Name="([^"]+)"\s+Value="([^"]*)"')


def species_info(d):
    """{param: value} from the .species file in a version folder, or {}."""
    try:
        hits = sorted(Path(d).glob("*.species"))
    except OSError:
        return {}
    if not hits:
        return {}
    try:
        text = hits[0].read_text(errors="replace")
    except OSError:
        return {}
    return dict(_SPECIES_PARAM.findall(text))


def species_resolution(d):
    """The round's global resolution in A, or None."""
    try:
        return float(species_info(d).get("GlobalResolution", ""))
    except (TypeError, ValueError):
        return None

# The refinement switches that actually distinguish one MCore round from the next.
# Everything else (paths, device list, port) is noise when you are comparing runs.
REFINE_FLAGS = re.compile(
    r"--(?:iter|refine_\w+|ctf_\w+|min_particles|angpix_resample|"
    r"perdevice_refine|temporal_samples)\b(?:\s+(?!--)\S+)*")


# Share the folder-timing rules with the app rather than reimplementing them: the
# details pane answers "where did THIS job write?" using the same join, and two
# copies of a timestamp heuristic would drift apart silently. Falls back to a local
# copy if the script has been lifted out of the package on its own.
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tomogration_jobs import folder_time, parse_ts       # noqa: E402
except Exception:                                            # pragma: no cover
    def parse_ts(text):
        """A job-store timestamp -> datetime, or None."""
        try:
            return datetime.datetime.strptime(str(text), TS_FMT)
        except (TypeError, ValueError):
            return None

    def folder_time(d):
        """When M finished writing this version folder — its newest entry."""
        newest = None
        try:
            for e in Path(d).iterdir():
                if e.name.startswith("_tomogration"):
                    continue          # our own label; see tomogration_jobs.folder_time
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
                newest = Path(d).stat().st_mtime
            except OSError:
                return None
        return datetime.datetime.fromtimestamp(newest)


def folder_size(d: Path):
    total = 0
    try:
        for f in d.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except OSError:
        pass
    return total


def human(n):
    """Byte count as a short string: 812B, 4.1M, 2.3G."""
    size = float(n)
    for unit in ("B", "K", "M", "G"):
        if size < 1024:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}T"


def load_jobs(root: Path):
    """M jobs from tomogration's store, newest first, with parsed times."""
    p = root / ".tomogration_jobs.json"
    if not p.is_file():
        return []
    try:
        store = json.loads(p.read_text())
    except (OSError, ValueError):
        return []
    out = []
    for job in (store.get("jobs", {}) or {}).values():
        if not str(job.get("stage_id", "")).startswith("m_"):
            continue
        started = parse_ts(job.get("started")) or parse_ts(job.get("created"))
        finished = parse_ts(job.get("finished"))
        if started is None:
            continue
        out.append({
            "id": job.get("id", "?"),
            "stage_id": job.get("stage_id", ""),
            "label": job.get("label", ""),
            "command": job.get("command", "") or "",
            "status": job.get("status", ""),
            "summary": job.get("summary", {}) or {},
            "started": started,
            "finished": finished,
        })
    out.sort(key=lambda j: j["started"])
    return out


def match_job(when, jobs, slack_s=900):
    """The job that was running when this version was written.

    Exact containment first. Failing that, the most recent job that STARTED
    before it — M commits a version as its last act, so a version written shortly
    after a job's recorded finish still belongs to that job. `slack_s` bounds how
    far after a finish we are willing to claim it.
    """
    if when is None:
        return None
    for j in jobs:
        end = j["finished"]
        if j["started"] <= when and (end is None or when <= end):
            return j
    best = None
    for j in jobs:
        if j["started"] > when:
            continue
        end = j["finished"] or j["started"]
        if (when - end).total_seconds() <= slack_s:
            if best is None or j["started"] > best["started"]:
                best = j
    return best


def refine_flags(command):
    """Just the switches that make one MCore round different from another."""
    return " ".join(m.group(0) for m in REFINE_FLAGS.finditer(command or ""))


def describe(job, res=None):
    """Job id + resolution. The resolution comes from the .species file when it is
    there, so a round run outside tomogration still reports one."""
    bits = []
    if res is not None:
        bits.append(f"{res:.2f} Å")
    if not job:
        bits.append("(not run from tomogration)")
        return "  ".join(bits)
    bits.insert(0, job["id"])
    if res is None:
        jr = job["summary"].get("resolution_A")
        if jr:
            bits.append(f"{jr} Å")
    if job["status"] and job["status"] != "completed":
        bits.append(job["status"])
    return "  ".join(bits)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Label M's randomly-named species version folders.")
    ap.add_argument("project_dir", nargs="?", default=".",
                    help="Project root (holds m*/ and .tomogration_jobs.json).")
    ap.add_argument("--write", action="store_true",
                    help=f"Write {INFO_NAME} inside each version folder.")
    ap.add_argument("--species", default="",
                    help="Only index species whose folder name contains this.")
    args = ap.parse_args(argv)

    root = Path(args.project_dir).expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 2

    version_dirs = sorted(
        d for d in root.glob("m*/species/*/versions/*")
        if d.is_dir() and (not args.species or args.species in d.parents[1].name))
    if not version_dirs:
        print(f"No species version folders under {root}/m*/species/*/versions/.",
              file=sys.stderr)
        print("Nothing to index — M writes these only once a refinement commits.",
              file=sys.stderr)
        return 2

    jobs = load_jobs(root)

    print("=" * 78)
    print("ml_m_index_versions")
    print(f"Project  : {root}")
    print(f"Versions : {len(version_dirs)}   M jobs in store: {len(jobs)}")
    print("=" * 78)

    rows = []
    for d in version_dirs:
        when = folder_time(d)
        info = species_info(d)
        rows.append({
            "dir": d,
            "when": when,
            "size": folder_size(d),
            "species": d.parents[1].name,
            "job": match_job(when, jobs),
            "res": species_resolution(d),
            "prev": info.get("PreviousVersion", ""),
            "angpix": info.get("PixelSize", ""),
        })
    # ORDER. Prefer M's own chain: each .species records the PreviousVersion it was
    # built from, which is exact and survives any amount of file touching. Fall back
    # to write time only where the chain is broken (a version deleted, or a species
    # rebuilt from scratch).
    by_name = {r["dir"].name: r for r in rows}
    prevs = {r["dir"].name: r.get("prev") or "" for r in rows}
    ordered, seen = [], set()
    roots = [n for n, pv in prevs.items() if pv not in by_name]
    roots.sort(key=lambda n: by_name[n]["when"] or datetime.datetime.min)
    nxt = {}
    for n, pv in prevs.items():
        if pv in by_name:
            nxt.setdefault(pv, []).append(n)
    for r0 in roots:
        cur = r0
        while cur and cur not in seen:
            seen.add(cur)
            ordered.append(by_name[cur])
            kids = sorted(nxt.get(cur, []),
                          key=lambda n: by_name[n]["when"] or datetime.datetime.min)
            cur = kids[0] if kids else None
    for r in sorted(rows, key=lambda r: (r["when"] or datetime.datetime.min)):
        if r["dir"].name not in seen:
            ordered.append(r)
    rows = ordered

    species_seen = None
    for r in rows:
        if r["species"] != species_seen:
            species_seen = r["species"]
            print(f"\n{species_seen}")
            print(f"  {'when':<17} {'folder':<14} {'size':>7}  job")
        when = r["when"].strftime("%Y-%m-%d %H:%M") if r["when"] else "unknown"
        print(f"  {when:<17} {r['dir'].name:<14} "
              f"{human(r['size']):>7}  {describe(r['job'], r.get('res'))}")
        job = r["job"]
        if job:
            flags = refine_flags(job["command"])
            if flags:
                print(f"  {'':<17} {'':<14} {'':>7}  {flags}")

    matched = sum(1 for r in rows if r["job"])
    print("\n" + "-" * 78)
    print(f"{matched} of {len(rows)} version folders matched to a tomogration job.")
    if matched < len(rows):
        print("Unmatched folders were written by an MCore run started outside "
              "tomogration;")
        print("their timestamp is still recorded below and in the written label.")

    if not args.write:
        print(f"\nNothing written. Re-run with --write to drop {INFO_NAME} into "
              f"each folder.")
        return 0

    written = failed = 0
    for r in rows:
        job = r["job"]
        when = r["when"].strftime(TS_FMT) if r["when"] else "unknown"
        lines = [
            "Written by tomogration (ml_m_index_versions.py).",
            "M never labels its version folders; this file is that label.",
            "It is inert — M does not read it — but do NOT rename the folder "
            "itself:",
            "the .species file refers to version names internally.",
            "",
            f"version folder : {r['dir'].name}",
            f"species        : {r['species']}",
            f"written        : {when}",
            f"size           : {human(r['size'])}",
        ]
        if r.get("res") is not None:
            lines.append(f"resolution     : {r['res']:.2f} A   (from the .species file)")
        if r.get("prev"):
            lines.append(f"previous round : {r['prev']}")
        if r.get("angpix"):
            lines.append(f"pixel size     : {r['angpix']} A/px")
        if job:
            lines += [
                f"tomogration job: {job['id']}  ({job['status']})",
                f"stage          : {job['stage_id']}",
                f"label          : {job['label']}",
            ]
            if job["summary"].get("resolution_A"):
                lines.append(f"resolution     : {job['summary']['resolution_A']} Å")
            lines += ["", "command:", f"  {job['command']}"]
        else:
            lines += [
                "tomogration job: none — this run was started outside tomogration,",
                "                 so only the timestamp above identifies it.",
            ]
        try:
            (r["dir"] / INFO_NAME).write_text("\n".join(lines) + "\n")
            written += 1
        except OSError as exc:
            print(f"  WARNING: could not write into {r['dir']}: {exc}",
                  file=sys.stderr)
            failed += 1

    print(f"\nWrote {INFO_NAME} into {written} folder(s)"
          + (f", {failed} failed." if failed else "."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
