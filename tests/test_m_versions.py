"""test_m_versions.py — labelling M's random version folders, and reading MCore's
resolution off stdout.

M commits each refinement round into m/species/<name>_<hash>/versions/<random>/ and
records nothing about which run made it. Two pieces close that gap:

  * m_resolution()          picks MCore's one result line out of the log so the job
                            card can carry "6.71 Å"
  * ml_m_index_versions.py  matches each version folder's write time to the job that
                            was running, and labels it

    python3 tests/test_m_versions.py
"""
import datetime
import importlib.util
import io
import json
import os
import sys
import tempfile
import time
from contextlib import redirect_stdout
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


jobs = load("tomogration_jobs")
idx = load("ml_m_index_versions")
passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}")


# ---- m_resolution: MCore's result line -------------------------------------
check("plain result line",
      jobs.m_resolution("EML45-spike-closed_HWhelp: 6.71 Å") == 6.71)
check("leading whitespace tolerated",
      jobs.m_resolution("   spike: 5.96 Å") == 5.96)
check("integer resolution",
      jobs.m_resolution("spike: 34 Å") == 34.0)
check("ASCII 'A' accepted (log encodings mangle Å)",
      jobs.m_resolution("spike: 6.73 A") == 6.73)
check("species names with colons and spaces",
      jobs.m_resolution("EML45-spike-closed_bin2_experiment: 33.65 Å") == 33.65)

# The lines that must NOT be mistaken for a result — every one of these really
# appears in an MCore/MTools run, and any of them on a card would be a lie.
for bad in ("Global resolution is 10.000",
            "Loading population...",
            "Adding 290 items.",
            "found 290 files",
            "defocus_max = 8",
            "2/2, 00:00 remaining",
            "device_list = { 1 }",
            "Species created: 'EML45' (fd2d54d3)",
            "",
            "   "):
    check(f"not a result: {bad!r}", jobs.m_resolution(bad) is None)
check("None input is safe", jobs.m_resolution(None) is None)

# summary_text must surface it, since that is the whole point of capturing it
check("summary_text shows resolution",
      jobs.summary_text({"resolution_A": "6.71"}) == "6.71 Å")
check("summary_text still handles other keys",
      "15 series" in jobs.summary_text({"series": 15}))


# ---- the indexer -----------------------------------------------------------
TS = "%Y-%m-%d %H:%M:%S"


def build(tmp, versions, jobs_list):
    """A project with species version folders and a job store.

    `versions` is [(species, folder, age_seconds_ago)]; `jobs_list` is dicts merged
    over a template.
    """
    root = Path(tmp)
    now = time.time()
    for species, folder, ago in versions:
        d = root / "m" / "species" / species / "versions" / folder
        d.mkdir(parents=True)
        f = d / "map.mrc"
        f.write_bytes(b"x" * 1024)
        os.utime(f, (now - ago, now - ago))
    store = {"seq": len(jobs_list), "jobs": {}}
    for j in jobs_list:
        rec = {"id": j["id"], "stage_id": j.get("stage_id", "m_core"),
               "label": j.get("label", "M: refine (MCore)"),
               "params": {}, "inputs": {}, "output_dir": f"jobs/{j['id']}",
               "command": j.get("command", ""), "status": j.get("status", "completed"),
               "exit_code": 0, "created": j["started"], "started": j["started"],
               "finished": j.get("finished"), "summary": j.get("summary", {})}
        store["jobs"][j["id"]] = rec
    (root / ".tomogration_jobs.json").write_text(json.dumps(store))
    return root


def ago_ts(seconds):
    return (datetime.datetime.now()
            - datetime.timedelta(seconds=seconds)).strftime(TS)


def run(root, *extra):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = idx.main([str(root), *extra])
    return code, buf.getvalue()


with tempfile.TemporaryDirectory() as tmp:
    root = build(
        tmp,
        versions=[("spike_abc123", "Hhhxx2to", 300),   # inside J60's window
                  ("spike_abc123", "-2-ij4-g", 60)],   # inside J61's window
        jobs_list=[
            {"id": "J60", "started": ago_ts(400), "finished": ago_ts(290),
             "command": "MCore --population m/p.population --iter 3 "
                        "--refine_particles --port 14350 --devicelist 0 1 2 3",
             "summary": {"resolution_A": "6.71"}},
            {"id": "J61", "started": ago_ts(200), "finished": ago_ts(50),
             "command": "MCore --population m/p.population --iter 3 "
                        "--refine_imagewarp 4x4 --refine_particles --port 14350",
             "summary": {"resolution_A": "5.96"}},
        ])
    code, out = run(root)
    check("indexer exits 0 when it finds versions", code == 0)
    check("both version folders listed",
          "Hhhxx2to" in out and "-2-ij4-g" in out)
    check("each folder matched to its job", "J60" in out and "J61" in out)
    check("resolution shown", "6.71 Å" in out and "5.96 Å" in out)
    check("distinguishing flags shown", "--refine_imagewarp 4x4" in out)
    check("noise flags suppressed", "--devicelist" not in out and "--port" not in out)
    check("all matched", "2 of 2 version folders matched" in out)
    check("dry run writes nothing",
          not list(Path(tmp).rglob(idx.INFO_NAME)))

    # chronological: the older folder must be printed first
    check("printed oldest-first",
          out.index("Hhhxx2to") < out.index("-2-ij4-g"))

    code, out = run(root, "--write")
    labels = sorted(Path(tmp).rglob(idx.INFO_NAME))
    check("--write drops a label in every folder", len(labels) == 2)
    body = (Path(tmp) / "m/species/spike_abc123/versions/Hhhxx2to"
            / idx.INFO_NAME).read_text()
    check("label names its job", "J60" in body)
    check("label carries the resolution", "6.71" in body)
    check("label carries the full command", "--refine_particles" in body)
    check("label warns against renaming", "do NOT rename" in body.lower()
          or "NOT rename" in body)

with tempfile.TemporaryDirectory() as tmp:
    # A run started from a terminal: no job covers it. It must still be listed,
    # not dropped, because its timestamp is the only handle the user has.
    root = build(tmp, versions=[("spike_abc123", "ZnCqDJNu", 100)], jobs_list=[])
    code, out = run(root)
    check("unmatched version still listed", "ZnCqDJNu" in out)
    check("unmatched is named as such", "not run from tomogration" in out)
    check("unmatched counted", "0 of 1 version folders matched" in out)
    code, out = run(root, "--write")
    body = next(Path(tmp).rglob(idx.INFO_NAME)).read_text()
    check("unmatched label still records a timestamp",
          "written        :" in body and "unknown" not in body)

with tempfile.TemporaryDirectory() as tmp:
    root = build(tmp, versions=[("spikeA_1", "aaa", 100), ("spikeB_2", "bbb", 50)],
                 jobs_list=[])
    code, out = run(root, "--species", "spikeA")
    check("--species filters", "aaa" in out and "bbb" not in out)

with tempfile.TemporaryDirectory() as tmp:
    code, out = run(Path(tmp))
    check("empty project exits 2", code == 2)

# match_job: a version committed just AFTER a job's recorded finish belongs to it
# (M commits as its last act, and the store's timestamp is when the process exited)
_now = datetime.datetime.now()
_jobs = [{"id": "J1", "stage_id": "m_core", "label": "", "command": "",
          "status": "completed", "summary": {},
          "started": _now - datetime.timedelta(seconds=600),
          "finished": _now - datetime.timedelta(seconds=300)}]
check("version inside the window matches",
      idx.match_job(_now - datetime.timedelta(seconds=400), _jobs)["id"] == "J1")
check("version just after the finish still matches",
      idx.match_job(_now - datetime.timedelta(seconds=290), _jobs)["id"] == "J1")
check("version long after the finish does not match",
      idx.match_job(_now + datetime.timedelta(seconds=5000), _jobs) is None)
check("version before the job started does not match",
      idx.match_job(_now - datetime.timedelta(seconds=900), _jobs) is None)


# ---- reloading a past job's params into the builder ------------------------
# The stage a job ran under gains and loses parameters over time, and reloading an
# OLD job is exactly when that bites. A dropped key is the dangerous one: the form
# cannot show it, so the user cannot see or remove it, but build_command still gets
# it.
_spec = {"id": "m_core", "params": [{"name": "population"}, {"name": "iter"},
                                    {"name": "port"}]}
_kept, _dropped, _missing = jobs.params_for_builder(
    _spec, {"population": "m/p.population", "iter": "3", "devicelist": "0 1"})
check("builder keeps the params the stage still has",
      _kept == {"population": "m/p.population", "iter": "3"})
check("builder drops a param the stage no longer has", _dropped == ["devicelist"])
check("builder reports a param the stage has gained", _missing == ["port"])

_kept, _dropped, _missing = jobs.params_for_builder(_spec, {})
check("empty recorded params is safe", _kept == {} and _dropped == [])
check("all params reported missing when nothing was recorded",
      _missing == ["iter", "population", "port"])

_kept, _dropped, _missing = jobs.params_for_builder(_spec, None)
check("None recorded params is safe", _kept == {} and _dropped == [])

# Real stages, real reload: every stage must round-trip its own defaults with
# nothing dropped — otherwise reloading any job of that stage loses values.
st = load("tomogration_stages")
_bad = []
for _s in st.STAGES:
    _defaults = {p["name"]: p.get("default", "") for p in _s.get("params", [])}
    _k, _d, _m = jobs.params_for_builder(_s, _defaults)
    if _d or _m or set(_k) != set(_defaults):
        _bad.append(_s["id"])
check(f"every stage round-trips its own params{(' — ' + ', '.join(_bad)) if _bad else ''}",
      not _bad)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
