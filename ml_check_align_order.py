#!/usr/bin/env python3
"""Did miss-alignment have a coarse alignment to refine? Answer from the record.

miss-alignment REFINES the geometry already in the Warp XMLs; run on XMLs that
never received AreTomo's alignment it refines nothing and the tomograms come out
featureless. tomogration prepends `ts_import_alignments` inside the
miss-alignment job's own command, so there is no separate card to look for --
the evidence is the stored command line.

    python3 ml_check_align_order.py [project_dir]

Reads .tomogration_jobs.json only. Exit 0 if every refinement was preceded by an
import, 1 if any was not, 2 if it cannot tell.
"""
import json
import os
import sys

REFINERS = ("miss_align", "miss_align_infer")


def jobnum(jid):
    try:
        return int(str(jid).lstrip("J"))
    except ValueError:
        return 0


def imported_before(job, earlier):
    """(bool, how) -- did this refinement have AreTomo's alignments in hand?"""
    cmd = str(job.get("command") or "")
    if "ts_import_alignments" in cmd:
        return (True, "prepended inside its own command")
    for other in earlier:
        if other.get("stage_id") == "ts_import_alignments" \
                and other.get("status") == "completed":
            return (True, f"separate job {other.get('id')}")
        if "ts_import_alignments" in str(other.get("command") or "") \
                and other.get("status") == "completed":
            return (True, f"prepended in {other.get('id')}")
    return (False, "")


def main(argv=None):
    root = (argv or [None])[0] or "."
    path = os.path.join(root, ".tomogration_jobs.json")
    try:
        with open(path) as fh:
            store = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"cannot read {path}: {exc}")
        return 2

    jobs = sorted((store.get("jobs") or {}).values(),
                  key=lambda j: jobnum(j.get("id", "")))
    refiners = [j for j in jobs if j.get("stage_id") in REFINERS]
    if not refiners:
        print(f"{path}\nNo miss-alignment jobs recorded here.")
        return 0

    print(f"{path}\n")
    bad = 0
    for job in refiners:
        earlier = [j for j in jobs if jobnum(j.get("id", "")) < jobnum(job.get("id", ""))]
        ok, how = imported_before(job, earlier)
        mark = "OK  " if ok else "BAD "
        if not ok:
            bad += 1
        print(f"{mark}{job.get('id'):>5}  {job.get('stage_id'):18s} "
              f"{job.get('status', '?'):10s} "
              + (f"import: {how}" if ok else "NO ts_import_alignments before it"))

    print()
    if bad:
        print(f"{bad} refinement(s) ran with no AreTomo alignment imported first.")
        print("Those refined nothing — re-import, re-select, and re-run them.")
    else:
        print("Every refinement had AreTomo's alignments imported first.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
