"""Project switching, and the coarse tilt stacks getting their own folder.

Two things that were broken by omission rather than by a bug:

  * sorting a Tomo5 dump had no bucket for the per-series Position*.mrc tilt
    stacks, so they stayed in the project root — thousands of MRCs beside the
    dozen folders that belong there — and the tilt inspector went looking for
    its stacks in that pile;
  * re-rooting the app meant typing a full ceph path every time.

The sort/sweep half is pure stdlib and runs against real temp dirs, so the moves
are checked by looking at the filesystem afterwards, not by trusting a return
value. The menu half drives the real Tomogration methods against a fake config
holder — the Qt stub cannot show a menu, but the target list is plain data.

    python3 tests/test_project_switch.py
"""
import importlib.util
import os
import shutil
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


import tomogration_project as tp

spec = importlib.util.spec_from_file_location("tomapp", REPO / "tomogration_app.py")
tomapp = importlib.util.module_from_spec(spec)
sys.modules["tomapp"] = tomapp
spec.loader.exec_module(tomapp)

COARSE = tp.COARSE_DIR


def touch(path, text="x"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# ===================================================================
# 1. the sort bucket
# ===================================================================
print("\n-- _sort_bucket --")
B = tp.ProjectState._sort_bucket

check("Position .mrc goes to the coarse folder",
      B("Position003.mrc") == COARSE)
check("lowercase position .mrc too",
      B("position_12.mrc") == COARSE)
check("the coarse folder is not the project root",
      COARSE not in ("", ".") and "/" not in COARSE)
check(".eer still goes to frames", B("Position003_0001.eer") == "frames")
check(".mdoc still goes to mdocs", B("Position003.mdoc") == "mdocs")
check("a gain .mrc goes to gains, NOT to the coarse folder",
      B("CountGain_x1.mrc") == "gains")
check("a Position-named gain still reads as a gain, not a stack",
      B("Position_gain.mrc") == "gains")
check("an override mdoc is not bucketed here",
      B("Position003_override.mdoc") is None)
check("an unrelated .mrc is left alone", B("template_emd_70905.mrc") is None)
check("a reconstruction is left alone",
      B("rec_Position003_12.56Apx.mrc") is None)
check("a Tomo5-named stack IS claimed when its mdoc is in the dump",
      B("Grid1_1.mrc", {"Grid1_1"}) == COARSE)
check("a Tomo5-named .mrc with no mdoc sibling is left alone",
      B("Grid1_1.mrc", {"Grid1_2"}) is None)

# ===================================================================
# 2. plan_sort / sort_files over a real dump
# ===================================================================
print("\n-- plan_sort / sort_files --")
tmp = Path(tempfile.mkdtemp())
root = tmp / "proj"
dump = tmp / "dump"
root.mkdir()
for n in ("Position003.mrc", "Position004.mrc", "Position003.mdoc",
          "Position003_0001.eer", "CountGain_x1.mrc", "Session.dm",
          "Grid7_9.mrc", "Grid7_9.mdoc", "Stray.mrc",
          "Position005_override.mdoc"):
    touch(dump / n)

st = tp.ProjectState(str(root))
plan = st.plan_sort(str(dump))
check("plan has a coarse bucket", COARSE in plan)
check("both Position stacks planned into it",
      {"Position003.mrc", "Position004.mrc"} <= set(plan[COARSE]))
check("the mdoc-backed Tomo5 stack is planned into it too",
      "Grid7_9.mrc" in plan[COARSE])
check("a stack with no mdoc is skipped, not swept up",
      "Stray.mrc" in plan["skipped"] and "Stray.mrc" not in plan[COARSE])
check("the gain still goes to gains",
      plan["gains"] == ["CountGain_x1.mrc"])
check("Session.dm is still left in place", "Session.dm" in plan["skipped"])
check("the override is still reported separately",
      plan["overrides"] == ["Position005_override.mdoc"])

moved = st.sort_files(str(dump))
check("sort_files reports the coarse count", moved[COARSE] == 3)
check("the stacks are ON DISK in the coarse folder",
      (root / COARSE / "Position003.mrc").is_file()
      and (root / COARSE / "Position004.mrc").is_file()
      and (root / COARSE / "Grid7_9.mrc").is_file())
check("and gone from the dump",
      not (dump / "Position003.mrc").exists())
check("the project root itself holds no loose Position*.mrc",
      not list(root.glob("Position*.mrc")))
check("initialize_structure made the folder",
      COARSE in tp.EXPECTED_DIRS and (root / COARSE).is_dir())

# never overwrite: a second dump with the same name must not clobber
touch(dump / "Position003.mrc", "NEW")
moved2 = st.sort_files(str(dump))
check("a same-named stack is not moved over an existing one",
      moved2[COARSE] == 0
      and (root / COARSE / "Position003.mrc").read_text() == "x"
      and (dump / "Position003.mrc").read_text() == "NEW")

# ===================================================================
# 3. collect_coarse_stacks — projects sorted by the older build
# ===================================================================
print("\n-- collect_coarse_stacks --")
old = Path(tempfile.mkdtemp()) / "old"
old.mkdir(parents=True)
st_old = tp.ProjectState(str(old))
for n in ("Position001.mrc", "Position002.mrc", "position003.mrc"):
    touch(old / n)
touch(old / "Position_gain.mrc")             # a gain, must stay
touch(old / "template.mrc")                  # not a stack, must stay
touch(old / "warp_tiltseries" / "Position001.mrc")   # subfolder, out of scope
touch(old / "Position099.mdoc")              # not a .mrc

n = st_old.collect_coarse_stacks()
check("sweeps exactly the Position stacks from the root", n == 3)
check("they are in the coarse folder now",
      sorted(p.name for p in (old / COARSE).glob("*.mrc"))
      == ["Position001.mrc", "Position002.mrc", "position003.mrc"])
check("the gain is left in the root", (old / "Position_gain.mrc").is_file())
check("an unrelated .mrc is left in the root", (old / "template.mrc").is_file())
check("a stack in a subfolder is untouched",
      (old / "warp_tiltseries" / "Position001.mrc").is_file())
check("running it again moves nothing", st_old.collect_coarse_stacks() == 0)
check("no folder is made when there is nothing to file",
      tp.ProjectState(str(Path(tempfile.mkdtemp())
                          )).collect_coarse_stacks() == 0)
check("a missing root is survivable",
      tp.ProjectState(str(old / "nope" / "nope")).collect_coarse_stacks() == 0)

# a name collision must not lose the root copy
touch(old / "Position001.mrc", "ROOT")
check("a collision leaves the root copy alone",
      st_old.collect_coarse_stacks() == 0
      and (old / "Position001.mrc").read_text() == "ROOT")

# ===================================================================
# 4. finding a stack to open in 3dmod
# ===================================================================
print("\n-- find_series_stack / coarse_stack_stems --")
f = Path(tempfile.mkdtemp()) / "f"
(f / COARSE).mkdir(parents=True)
st_f = tp.ProjectState(str(f))
touch(f / COARSE / "Position003.mrc")
check("found under the renamed name in the coarse folder",
      st_f.find_series_stack("Grid1_3", "Position003")
      == f / COARSE / "Position003.mrc")
touch(f / COARSE / "Grid1_4.mrc")
check("found under the Tomo5 name in the coarse folder",
      st_f.find_series_stack("Grid1_4", "Position004")
      == f / COARSE / "Grid1_4.mrc")
check("the renamed name wins over the Tomo5 one",
      st_f.find_series_stack("Grid1_4", "Position003")
      == f / COARSE / "Position003.mrc")
touch(f / "Position009.mrc")
check("the root is still searched, for older projects",
      st_f.find_series_stack("Grid1_9", "Position009") == f / "Position009.mrc")
check("the coarse folder is preferred over the root",
      (touch(f / "Position003.mrc") is not None)
      and st_f.find_series_stack("Grid1_3", "Position003")
      == f / COARSE / "Position003.mrc")
check("nothing found stays None",
      st_f.find_series_stack("Grid9_9", "Position999") is None)
check("coarse_stack_stems lists the series",
      st_f.coarse_stack_stems() == ["Grid1_4", "Position003"])
check("coarse_stack_stems is empty with no folder",
      tp.ProjectState(str(f / "nope")).coarse_stack_stems() == [])

# ===================================================================
# 5. what counts as a project root
# ===================================================================
print("\n-- project_marker / find_projects --")
lib = Path(tempfile.mkdtemp()) / "EMDatasets"
(lib / "EML46" / "OC43-3mM-disacch" / "mdocs").mkdir(parents=True)
touch(lib / "EML46" / "OC43-3mM-disacch" / "warp_tiltseries.settings")
(lib / "EML46" / "OC43-control" / "frames").mkdir(parents=True)
(lib / "EML46" / "notes").mkdir(parents=True)          # not a project
touch(lib / "EML46" / "notes" / "readme.txt")
(lib / "EML46" / ".hidden-proj" / "mdocs").mkdir(parents=True)

check("a settings file marks a project root",
      tp.project_marker(lib / "EML46" / "OC43-3mM-disacch")
      == "warp_tiltseries.settings")
check("frames/ alone marks a project root",
      tp.project_marker(lib / "EML46" / "OC43-control") == "frames")
check("a folder with none of the markers is not a project",
      tp.project_marker(lib / "EML46" / "notes") == "")
check("a missing folder is not a project",
      tp.project_marker(lib / "nope") == "")
check("every marker name is a plain name, not a path",
      all("/" not in m for m in tp.PROJECT_MARKERS))

found = tp.find_projects(lib / "EML46")
check("finds the projects under a container, sorted",
      [n for n, _m in found] == ["OC43-3mM-disacch", "OC43-control"])
check("reports which marker matched",
      dict(found)["OC43-control"] == "frames")
check("skips non-project subfolders", "notes" not in dict(found))
check("skips dotfolders", ".hidden-proj" not in dict(found))
check("a missing parent gives an empty list", tp.find_projects(lib / "nope") == [])
check("a container is not itself reported as a child",
      lib.name not in dict(found))
check("limit is honoured", tp.find_projects(lib / "EML46", limit=0) == [])

# ===================================================================
# 6. the Project menu's target list (real methods, fake config)
# ===================================================================
print("\n-- project_dirs / project_menu_targets --")
T = tomapp.Tomogration

check("the menu caps how many projects it probes per bookmark",
      isinstance(T.PROJECT_MENU_LIMIT, int) and 0 < T.PROJECT_MENU_LIMIT <= 200)

S = T._short_path
check("a long path is shown by its tail",
      S("/ceph/users/haq21239/EMDatasets/EML46") == "…/EMDatasets/EML46")
check("a short path is shown whole", S("/ceph/data") == "/ceph/data")
check("a bare root is shown whole", S("/") == "/")
check("the tail keeps the folder being named",
      S("/a/b/c/d/OC43-3mM-disacch").endswith("OC43-3mM-disacch"))


class FakeWin:
    """Just enough for the config-backed bookmark methods to run for real."""
    PROJECT_DIRS_KEY = T.PROJECT_DIRS_KEY
    PROJECT_MENU_LIMIT = T.PROJECT_MENU_LIMIT
    project_dirs = T.project_dirs
    project_menu_targets = T.project_menu_targets
    _save_project_dirs = T._save_project_dirs

    def __init__(self, root, cfg=None, shared=None):
        self.project_root = str(root)
        self.cfg = dict(cfg or {})
        self.shared = dict(shared or {})
        self.logs = []

    def _load_config(self):
        return dict(self.cfg)

    def _save_config(self, cfg):
        self.cfg = dict(cfg)

    # The shared store lives beside the code on ceph; here it is just a dict.
    def _load_shared_config(self):
        return dict(self.shared)

    def _save_shared_config(self, cfg):
        self.shared = dict(cfg)
        return (True, "shared.json")

    def _shared_config_path(self):
        return "shared.json"

    def _log(self, msg, kind="info"):
        self.logs.append((kind, msg))


proj = lib / "EML46" / "OC43-3mM-disacch"
w = FakeWin(proj)
check("with no config, the folder above the project is seeded",
      w.project_dirs() == [str(lib / "EML46")])

w = FakeWin(proj, {"project_dirs": ["/a/", "/a", "  ", "/b/c/"]})
check("bookmarks are absolute, de-duplicated and order-preserving",
      w.project_dirs() == ["/a", "/b/c"])
check("blank lines are dropped", "" not in w.project_dirs())

w = FakeWin(proj, {"project_dirs": ["~"]})
check("a ~ path is expanded",
      w.project_dirs() == [os.path.abspath(os.path.expanduser("~"))])

w = FakeWin(proj, {"project_dirs": [str(lib / "EML46"), str(proj),
                                    str(lib / "gone")]})
rows = w.project_menu_targets()
check("one row per bookmark, in order",
      [b for b, _t in rows] == [str(lib / "EML46"), str(proj), str(lib / "gone")])
check("a container lists the projects inside it",
      [lbl for lbl, _p in rows[0][1]] == ["OC43-3mM-disacch", "OC43-control"])
check("their paths are the real ones",
      dict(rows[0][1])["OC43-control"] == str(lib / "EML46" / "OC43-control"))
check("a bookmark that IS a project offers itself",
      rows[1][1] == [("OC43-3mM-disacch", str(proj))])
check("a missing bookmark is flagged as missing, not as empty",
      rows[2][1] is None)

w = FakeWin(proj, {"project_dirs": [str(lib / "EML46" / "notes")]})
check("a bookmark with neither is reported empty, not missing",
      w.project_menu_targets()[0][1] == [])

# a bookmark that is a project AND holds projects: both appear, itself first
nested = lib / "EML45"
(nested / "mdocs").mkdir(parents=True)
(nested / "combined" / "frames").mkdir(parents=True)
w = FakeWin(proj, {"project_dirs": [str(nested)]})
labels = [lbl for lbl, _p in w.project_menu_targets()[0][1]]
check("a bookmark that is both lists itself first, then its children",
      labels == ["EML45", "combined"])

w = FakeWin(proj, {})
w._save_project_dirs(["/x", "/y"])
check("saving bookmarks round-trips through the SHARED store",
      w.shared["project_dirs"] == ["/x", "/y"])
check("and not through the per-machine one", "project_dirs" not in w.cfg)
check("saving does not disturb other config keys",
      FakeWin(proj, {"last_root": "/keep"}).cfg.get("last_root") == "/keep")
w = FakeWin(proj, {"last_root": "/keep"})
w._save_project_dirs(["/x"])
check("last_root survives a bookmark save", w.cfg["last_root"] == "/keep")
w = FakeWin(proj, {}, {"theme": "dark"})
w._save_project_dirs(["/x"])
check("other SHARED keys survive a bookmark save", w.shared["theme"] == "dark")

for d in (tmp, old.parent, f.parent, lib.parent):
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Bookmarks must survive a workstation switch. Home is local to each VM on this
# cluster, so ~/.tomogration.json lost them every time. They live beside the
# code -- which is on ceph and is the same install every workstation launches.
# ---------------------------------------------------------------------------
import json as _json                                          # noqa: E402
import tempfile as _tf                                        # noqa: E402
from pathlib import Path as _P                                # noqa: E402


class _Bookmarks(tomapp.Tomogration):
    """Only the config plumbing -- no Qt, no __init__ chain."""
    def __init__(self, home, shared, root):
        self._home, self._shared, self.project_root = home, shared, root
        self.logs = []

    def _config_path(self):
        return _P(self._home)

    def _shared_config_path(self):
        return _P(self._shared)

    def _log(self, msg, kind="info"):
        self.logs.append((kind, msg))


def _bm():
    d = _tf.mkdtemp()
    root = os.path.join(d, "EML50", "manually-selected")
    os.makedirs(root)
    return (_Bookmarks(os.path.join(d, "home.json"),
                       os.path.join(d, "shared.json"), root), d)


_b, _d = _bm()
check("seeded with the folder holding this project's siblings",
      _b.project_dirs() == [os.path.join(_d, "EML50")])

# A bookmark made before the move must not be lost.
open(_b._home, "w").write(_json.dumps({"project_dirs": ["/ceph/a", "/ceph/b"]}))
check("bookmarks from ~/.tomogration.json still load",
      _b.project_dirs() == ["/ceph/a", "/ceph/b"])

_b.logs.clear()
_b._save_project_dirs(["/ceph/a", "/ceph/b", "/ceph/c"])
check("saving writes the SHARED file",
      _json.load(open(_b._shared))["project_dirs"] == ["/ceph/a", "/ceph/b", "/ceph/c"])
check("and leaves the home file alone",
      _json.load(open(_b._home))["project_dirs"] == ["/ceph/a", "/ceph/b"])
check("shared wins over home once written",
      _b.project_dirs() == ["/ceph/a", "/ceph/b", "/ceph/c"])
check("the user is told where they landed",
      "shared by every workstation" in _b.logs[-1][1])

# A group-owned install can be read-only. Losing the edit silently is the very
# bug being fixed, so it falls back to home and says so.
_ro = _Bookmarks(_b._home, "/nonexistent-dir/shared.json", _b.project_root)
_ro._save_project_dirs(["/ceph/z"])
check("a read-only install falls back to home",
      _json.load(open(_b._home))["project_dirs"] == ["/ceph/z"])
check("and warns rather than failing silently", _ro.logs[-1][0] == "warning")

open(_b._shared, "w").write("{ not json")
check("a corrupt shared file falls back instead of losing everything",
      _b.project_dirs() == ["/ceph/z"])

open(_b._shared, "w").write(_json.dumps({"project_dirs": "not-a-list"}))
check("a non-list shared value falls back too",
      _b.project_dirs() == ["/ceph/z"])

_b2, _d2 = _bm()
_b2._save_project_dirs(["/ceph/x/", "/ceph/x", "~/rel"])
check("paths are absolute and de-duplicated",
      _b2.project_dirs() == ["/ceph/x", os.path.abspath(os.path.expanduser("~/rel"))])

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
