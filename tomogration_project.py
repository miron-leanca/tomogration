#!/usr/bin/env python3
# Part of tomogration2 — split out of the single-file app so the pieces can be
# edited (and tested) independently. Import graph is a strict DAG:
#     core  ->  stages  ->  jobs        (project depends only on core)
# tomogration_app.py imports all of them and holds the Qt window.

"""ProjectState — inspects what exists under a project root and reports per-stage
status. Pure stdlib, no GUI: ported verbatim from the original warp_auto.py."""

import datetime
import filecmp
import itertools
import json
import os
import re
import shutil
from pathlib import Path

from tomogration_core import expand_tilt_ranges

# Where the coarse (as-acquired) per-series tilt stacks live. A Tomo5 dump
# carries a Position*.mrc stack beside every .eer/.mdoc, and sorting used to have
# no bucket for them: they stayed in the project root, so a root that should hold
# a dozen folders held thousands of MRCs instead, and the tilt inspector went
# looking for its stacks in that pile.
COARSE_DIR = "mrcs-tiltseries-coarse"

EXPECTED_DIRS = [
    "frames", "frames/bad", "mdocs", "mdocs/bad", "gains",
    COARSE_DIR,
    "aretomo_output", "aretomo_output/Imod",
    "warp_frameseries", "warp_tiltseries", "tomostar",
]

# Names that mark a directory as a tomogration project root. Checked one stat at
# a time rather than by listing the directory: a project root holds thousands of
# entries, this runs for every candidate the Project menu offers, and it all sits
# on ceph.
PROJECT_MARKERS = (
    ".tomogration_jobs.json", "warp_tiltseries.settings",
    "warp_frameseries.settings", "warp_tiltseries", "tomostar",
    "mdocs", "frames",
)


def project_marker(path):
    """The first PROJECT_MARKERS name present under path, or "" for none —
    so "" means 'this is not a project root'."""
    p = Path(path)
    for m in PROJECT_MARKERS:
        try:
            if (p / m).exists():
                return m
        except OSError:
            return ""
    return ""


def find_projects(parent, limit=40):
    """Immediate subdirectories of parent that look like project roots, as
    [(name, marker)] sorted by name.

    One readdir plus a handful of stats per child, never a recursive walk: the
    parent may itself be a raw dump of 10k files, and the projects underneath
    each hold 10k more."""
    out = []
    try:
        kids = sorted((e for e in os.scandir(parent)
                       if not e.name.startswith(".") and e.is_dir()),
                      key=lambda e: e.name)
    except OSError:
        return out
    for e in kids[:limit]:
        m = project_marker(e.path)
        if m:
            out.append((e.name, m))
    return out

# WarpTools per-item failure line; used by auto-recovery to find the culprit.
# (The progress-line regex lives in tomogration_app.py, which owns the logger.)
_FAILED_FILE_RE = re.compile(
    r"Failed to process\s+(\S+\.(?:eer|mrc|tiff?|st|tomostar))", re.IGNORECASE)

# Warp stores the per-series defocus in MICROMETRES on a Param element.
_DEFOCUS_RE = re.compile(r'<Param\s+Name="Defocus"\s+Value="([^"]+)"')


def defocus_angstroms(xml_text):
    """Per-series defocus from a warp_tiltseries/<series>.xml, in ÅNGSTRÖMS.

    Warp writes <Param Name="Defocus" Value="..."/> in MICROMETRES; the
    membrane tools want --df in Å, so ×10000 here — in exactly one place,
    never hardcoded per caller. Returns None when the attribute is absent or
    not a number (callers must skip the series, not guess)."""
    m = _DEFOCUS_RE.search(xml_text or "")
    if not m:
        return None
    try:
        return float(m.group(1)) * 10000.0
    except ValueError:
        return None


def provenance_variant(dirpath):
    """The 'variant' recorded in a folder's PROVENANCE.json ('' if none).

    Membrane-branch outputs (deconvolved, denoised, wedge-restored) carry this
    tag so processed volumes can be refused where they would do harm: as input
    to another deconvolution, or as the source of particle EXTRACTION (the §5
    hard rule — coordinates may come from processed tomograms, extraction must
    reference the original reconstruction)."""
    p = Path(dirpath) / "PROVENANCE.json"
    try:
        if not p.is_file():
            return ""
        data = json.loads(p.read_text())
        return str(data.get("variant", "")) if isinstance(data, dict) else ""
    except (OSError, ValueError):
        return ""


# Warp writes the reconstruction pixel size into the filename
# (Position003_10.00Apx.mrc), and the membrane tools append their own suffixes
# after it (…Apx_scores.mrc, …Apx_segmented_threshold_-1.0.mrc). Everything the
# *_TOMO_LIST wrappers match is '<stem>.mrc' or '<stem>_*.mrc', so the stem is
# whatever comes before the pixel-size tag.
_STEM_APX_RE = re.compile(r"^(.+?)_\d+(?:\.\d+)?Apx(?:[_.].*)?$")


def series_stem(filename):
    """The series stem a *_TOMO_LIST entry should name for this file.

    'Position003_10.00Apx.mrc' -> 'Position003', and so does every derived file
    of that series ('..._scores.mrc', '..._segmented_threshold_-1.0.mrc'), which
    is exactly the grouping the wrappers' '${stem}_*.mrc' glob produces. Names
    without a pixel-size tag fall back to the basename minus its extension —
    still an exact '<stem>.mrc' match."""
    base = Path(str(filename)).name
    if base.lower().endswith(".mrc"):
        base = base[:-4]
    m = _STEM_APX_RE.match(base)
    return m.group(1) if m else base


def _natural_key(s):
    """Sort key that orders Position2 before Position10."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", str(s))]


def tomogram_stems(dirpath):
    """[(stem, [filenames…]), …] for the .mrc in a folder, grouped by series
    stem and naturally sorted. Empty list when the folder doesn't exist — the
    caller distinguishes 'no folder' from 'no tomograms' itself."""
    d = Path(dirpath)
    if not d.is_dir():
        return []
    groups = {}
    try:
        names = [p.name for p in d.glob("*.mrc")]
    except OSError:
        return []
    for n in names:
        groups.setdefault(series_stem(n), []).append(n)
    return [(stem, sorted(groups[stem], key=_natural_key))
            for stem in sorted(groups, key=_natural_key)]


def _atomic_write_text(path, text):
    """Write via a same-directory temp file + os.replace so a crash mid-write
    leaves either the old file or the new one — never a truncated file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


# ===========================================================================
# ProjectState  —  framework-agnostic backend (ported from warp_auto.py)
# ===========================================================================
class ProjectState:
    """Inspects what exists under a project root and reports per-stage status.
    Pure stdlib (pathlib/os/json/re/datetime) — no GUI dependencies."""

    AUTO_EXCLUSION_HEADER = (
        "# --- Auto-excluded during processing (do not edit above this line) ---")

    def __init__(self, root):
        self.root = Path(root)

    # ---- generic helpers -------------------------------------------------
    def exists(self, rel_path):
        return (self.root / rel_path).exists()

    def count_files(self, rel_path, pattern):
        p = self.root / rel_path
        return len(list(p.glob(pattern))) if p.is_dir() else 0

    def count_rglob(self, rel_path, pattern):
        p = self.root / rel_path
        return len(list(p.rglob(pattern))) if p.is_dir() else 0

    def list_files(self, rel_path, pattern):
        p = self.root / rel_path
        return sorted(f.name for f in p.glob(pattern)) if p.is_dir() else []

    # ---- processing history (per-dataset job log for the History window) -----
    HISTORY_FILE = ".tomogration_history.json"

    def load_history(self):
        p = self.root / self.HISTORY_FILE
        if not p.is_file():
            return []
        try:
            data = json.loads(p.read_text())
        except OSError:
            # Transient read error: the file may be fine — return an empty view
            # but leave it in place (saves are atomic, so nothing clobbers it).
            return []
        except ValueError:
            data = None
        if isinstance(data, list):
            return data
        # Corrupt file: rename it aside so the next append_history can't
        # persist a 1-element list over the entire run history.
        try:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            p.rename(p.with_name(p.name + f".corrupt_{stamp}"))
        except OSError:
            pass
        return []

    def _save_history(self, hist):
        try:
            _atomic_write_text(self.root / self.HISTORY_FILE,
                               json.dumps(hist, indent=1))
        except OSError:
            pass

    def append_history(self, record):
        hist = self.load_history()
        hist.append(record)
        self._save_history(hist)
        return len(hist) - 1

    def update_history(self, index, **fields):
        hist = self.load_history()
        if isinstance(index, int) and 0 <= index < len(hist):
            hist[index].update(fields)
            self._save_history(hist)

    def archive_output_dir(self, rel):
        """Rename a non-empty output dir aside (timestamped) so a re-run keeps the
        old result instead of overwriting it. Returns the archive rel-path or ''."""
        d = self.root / rel
        try:
            if not d.is_dir() or not any(d.iterdir()):
                return ""
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = d.with_name(d.name + f".bak_{stamp}")
            d.rename(dest)
            return os.path.relpath(dest, self.root)
        except OSError:
            return ""

    # ---- per-stage status (each returns (count_or_bool, label_or_None)) ---
    def status_renamed_frames(self):
        n = self.count_files("frames", "Position*.eer")
        return (n, f"{n} .eer files" if n else None)

    def status_listfile(self):
        """'Rename ran' marker: the listfile rename writes (in the project root,
        wherever rename was run). Independent of whether files were sorted yet."""
        files = sorted(self.root.glob("listfile_*.txt")) if self.root.is_dir() else []
        if (self.root / "listfile.txt").is_file():
            files.append(self.root / "listfile.txt")
        return (bool(files), files[0].name if files else None)

    def status_mdocs(self):
        n = self.count_files("mdocs", "Position*.mdoc")
        return (n, f"{n} .mdoc files" if n else None)

    def status_exclusion_list(self):
        ok = self.exists("exclusion_list.txt")
        return (ok, "exclusion_list.txt" if ok else None)

    def status_exclusions(self):
        """Green when exclusion_list.txt actually holds exclusion entries (a
        non-comment, non-blank line) — i.e. the inspect/select step has been done."""
        p = self.root / "exclusion_list.txt"
        if not p.is_file():
            return (0, None)
        n = sum(1 for ln in p.read_text(errors="replace").splitlines()
                if ln.strip() and not ln.strip().startswith("#"))
        return (n, f"{n} exclusion entries" if n else None)

    def status_conv_key(self):
        ok = self.exists("new_imod_conv_key.txt")
        return (ok, "new_imod_conv_key.txt" if ok else None)

    def status_gain_original(self):
        p = self.root / "gains"
        if not p.is_dir():
            return (False, None)
        # Match _classify_dump's gain rule EXACTLY. Sorting files any *gain*.mrc
        # into gains/ while looking here for *_gain*.mrc meant a 'gainref.mrc'
        # was filed correctly and then never found: gain_source() returned "",
        # the card kept its placeholder default, and the run died on a path that
        # had never existed.
        names = [e.name for e in p.iterdir() if e.is_file()]
        matches = [n for n in names
                   if (n.lower().endswith(".gain")
                       or ("gain" in n.lower() and n.lower().endswith(".mrc")))
                   and "reciprocal" not in n.lower()]
        # .gain first: gain_convert's whole job is .gain -> .mrc.
        matches.sort(key=lambda n: (not n.lower().endswith(".gain"), n))
        return (True, matches[0]) if matches else (False, None)

    def status_gain_reciprocal(self):
        p = self.root / "gains"
        if not p.is_dir():
            return (False, None)
        matches = sorted(p.glob("*reciprocal*.mrc"))
        return (True, matches[0].name) if matches else (False, None)

    def gain_source(self):
        """Relative path to the detected non-reciprocal gain (gains/<name>), or ''."""
        ok, name = self.status_gain_original()
        return f"gains/{name}" if ok and name else ""

    def status_fs_settings(self):
        ok = self.exists("warp_frameseries.settings")
        return (ok, "warp_frameseries.settings" if ok else None)

    def status_ts_settings(self):
        ok = self.exists("warp_tiltseries.settings")
        return (ok, "warp_tiltseries.settings" if ok else None)

    def status_fs_motion_ctf(self):
        n = self.count_files("warp_frameseries", "*.xml")
        return (n, f"{n} processed frames" if n else None)

    def status_tomostar(self):
        n = self.count_files("tomostar", "*.tomostar")
        return (n, f"{n} .tomostar files" if n else None)

    def status_ts_stacks(self):
        p = self.root / "warp_tiltseries" / "tiltstack"
        if not p.is_dir():
            return (0, None)
        n = sum(1 for d in p.iterdir() if d.is_dir() and (d / f"{d.name}.st").exists())
        return (n, f"{n} stacks" if n else None)

    def status_aretomo_xf(self):
        """Count AreTomo .xf alignments in the newest version folder."""
        versions = self.list_aretomo_versions()
        if not versions:
            return (0, None)
        active = versions[-1]
        n = 0
        for layout in ("Imod", "imod"):
            ld = active / layout
            if ld.is_dir():
                n += sum(1 for sub in ld.iterdir()
                         if sub.is_dir() and list(sub.glob("*.xf")))
        label = f"{n} .xf ({active.name})"
        if len(versions) > 1:
            label += f" [{len(versions)} versions]"
        return (n, label if n else None)

    def status_alignments_imported(self):
        """imod subdirs with both .xf and .tlt, ready for ts_import_alignments."""
        versions = self.list_aretomo_versions()
        if not versions:
            return (0, None)
        active = versions[-1]
        n = 0
        for layout in ("Imod", "imod"):
            ld = active / layout
            if ld.is_dir():
                n += sum(1 for sub in ld.iterdir() if sub.is_dir()
                         and list(sub.glob("*.xf")) and list(sub.glob("*.tlt")))
        return (n, f"{n} alignments ready ({active.name})" if n else None)

    def status_selection_sync(self):
        if self.status_tomostar()[0] == 0:
            return (False, None)
        missing = self.tomostars_without_alignments()
        if missing:
            return (False, f"{len(missing)} tilt series missing alignment")
        return (True, "all tomostars have alignments")

    def status_ts_ctf(self):
        """Green only when the series actually carry a CTF FIT, not merely an .xml.

        The per-series .xml already exists after ts_import/ts_stack, so counting
        files lit this dot green before ts_ctf ever ran — and a skipped ts_ctf
        surfaces days later as MCore's IndexOutOfRangeException. The test is the
        same one ml_m_check_ctf uses (a non-empty CTFResolutionEstimate on the
        root tag), read from the file head and cached by (mtime, size), so a
        sweep costs one stat per file after the first pass."""
        d = self.root / "warp_tiltseries"
        if not d.is_dir():
            return (0, None)
        cache = getattr(self, "_ctf_dot_cache", None)
        if cache is None:
            cache = self._ctf_dot_cache = {}
        total = fitted = 0
        for p in d.glob("*.xml"):
            total += 1
            try:
                st = p.stat()
                key = (st.st_mtime_ns, st.st_size)
            except OSError:
                continue
            hit = cache.get(p.name)
            if hit is not None and hit[0] == key:
                has = hit[1]
            else:
                has = self._xml_has_ctf_fit(p)
                cache[p.name] = (key, has)
            if has:
                fitted += 1
        if not total:
            return (0, None)
        if fitted < total:
            return (0, f"{fitted}/{total} series have a CTF fit — run ts_ctf")
        return (fitted, f"{fitted} series CTF-fitted")

    @staticmethod
    def _xml_has_ctf_fit(path):
        """True if the tilt-series .xml root tag carries a non-empty
        CTFResolutionEstimate. The attribute lives in the opening tag, so the
        file head is enough — no full XML parse per status sweep."""
        try:
            with open(path, "rb") as f:
                head = f.read(16384)
        except OSError:
            return False
        m = re.search(rb'CTFResolutionEstimate="([^"]*)"', head)
        return bool(m and m.group(1).strip())

    def status_warp_tomograms(self):
        p = self.root / "warp_tiltseries" / "reconstruction"
        if not p.is_dir():
            return (0, None)
        n = len(list(p.glob("*.mrc")))
        return (n, f"{n} tomograms" if n else None)

    def status_template_matches(self):
        n = self.count_rglob("warp_tiltseries", "*.star")
        return (n, f"{n} match .star" if n else None)

    def status_thresholded(self):
        n = self.count_rglob("warp_tiltseries", "*clean.star")
        return (n, f"{n} clean .star" if n else None)

    def status_exported(self):
        """Count export stars WITHOUT walking the subtomograms.

        This used to be count_rglob("relion4", "*.star") — a recursive walk of
        relion4/, which holds one subtomo/PositionNNN/ tree per export with tens of
        thousands of .mrc in it. It ran on every canvas repaint and froze the UI for
        seconds at a time. The stars we care about sit at the TOP of each project
        dir, so look exactly there and never descend into subtomo/."""
        base = self.root / "relion4"
        if not base.is_dir():
            return (0, None)
        n = 0
        try:
            for proj in itertools.islice(base.iterdir(), 0, 200):
                if not proj.is_dir():
                    continue
                n += sum(1 for _ in itertools.islice(proj.glob("*.star"), 0, 500))
        except OSError:
            pass
        return (n, f"{n} .star in relion4/" if n else None)


    # ---- M (multi-particle refinement) status ----------------------------
    # All bounded, single-level globs: m/ holds a handful of files and the .source
    # sits beside the settings. Nothing here may walk a data directory — these run
    # on every canvas repaint (see the status cache).
    def status_m_population(self):
        n = self.count_files("m", "*.population")
        return (n, f"{n} population" if n else None)

    def status_m_source(self):
        """MTools writes <name>.source next to the PROCESSING SETTINGS, not into m/."""
        n = self.count_files("warp_tiltseries", "*.source") + self.count_files(".", "*.source")
        return (n, f"{n} data source" if n else None)

    def status_m_mask(self):
        n = self.count_files("m", "*.mrc")
        return (n, f"{n} mask" if n else None)

    def status_m_species(self):
        """m/species/<name>_<hash>/<name>.species — one level of hashed dirs."""
        p = self.root / "m" / "species"
        if not p.is_dir():
            return (0, None)
        try:
            n = sum(1 for d in itertools.islice(p.iterdir(), 0, 200)
                    if d.is_dir() and any(d.glob("*.species")))
        except OSError:
            return (0, None)
        return (n, f"{n} species" if n else None)

    def status_m_refined(self):
        """MCore writes updated maps into each species folder as it refines."""
        p = self.root / "m" / "species"
        if not p.is_dir():
            return (0, None)
        try:
            n = 0
            for d in itertools.islice(p.iterdir(), 0, 200):
                if d.is_dir():
                    n += sum(1 for _ in itertools.islice(d.glob("*.mrc"), 0, 50))
        except OSError:
            return (0, None)
        return (n, f"{n} map(s) in species" if n else None)

    # ---- AreTomo versioned-folder pattern --------------------------------
    def list_aretomo_versions(self):
        """aretomo_output folders sorted oldest->newest. 'aretomo_output' is
        version 1; 'aretomo_output-v<N>' follow. Only folders with a standard
        subdir (mrc/Imod/imod/aln/proj) or a PARAMETERS.txt are considered."""
        candidates = []
        if not self.root.is_dir():
            return []
        for d in sorted(self.root.iterdir()):
            if not d.is_dir():
                continue
            name = d.name
            if name == "aretomo_output":
                version = 1
            elif name.startswith("aretomo_output-v"):
                try:
                    version = int(name[len("aretomo_output-v"):])
                except ValueError:
                    continue
            else:
                continue
            has_content = any((d / sub).is_dir()
                              for sub in ("mrc", "Imod", "imod", "aln", "proj"))
            if has_content or (d / "PARAMETERS.txt").is_file():
                candidates.append((version, d))
        candidates.sort(key=lambda x: x[0])
        return [c[1] for c in candidates]

    def latest_aretomo_imod(self):
        """Root-relative '<aretomo_output[-vN]>/Imod/' for the NEWEST AreTomo folder
        that actually contains alignment files (a .xf directly or one level down, as
        AreTomo writes <series>_Imod/<series>.xf), or '' if none. This only sets the
        ts_import_alignments DEFAULT — the 'alignments' field is editable, so type in a
        specific version (e.g. aretomo_output-v2/Imod/) to import from an older or more
        complete run. AreTomo auto-versions; a stale base 'aretomo_output/Imod/' fails
        with 'Could not find <series>.xf'."""
        for d in self._aretomo_candidates():
            imod = d / "Imod"
            if not imod.is_dir():
                continue
            if next(imod.glob("*.xf"), None) or next(imod.glob("*/*.xf"), None):
                return os.path.relpath(imod, self.root) + "/"
        return ""

    def _aretomo_candidates(self):
        """Folders that may hold an AreTomo run, newest-looking LAST-searched
        first: this job's own jobs/<id>/ dirs before the shared, auto-versioned
        aretomo_output[-vN]/.

        A job-scoped AreTomo writes into jobs/J13_align-with-aretomo2/Imod/, and
        a scan that only knew the shared layout reported "no alignment found"
        for a run that had just succeeded.
        """
        out = []
        jobs = self.root / "jobs"
        if jobs.is_dir():
            try:
                out = sorted((d for d in jobs.iterdir()
                              if d.is_dir() and (d / "Imod").is_dir()),
                             key=lambda d: _natural_key(d.name), reverse=True)
            except OSError:
                out = []
        return out + list(reversed(self.list_aretomo_versions()))

    def next_aretomo_version_path(self):
        """Path to use for the next AreTomo run. If no versions exist, returns
        aretomo_output; otherwise the next aretomo_output-v<N+1>."""
        versions = self.list_aretomo_versions()
        if not versions:
            return self.root / "aretomo_output"
        highest = 1
        for d in versions:
            if d.name == "aretomo_output":
                highest = max(highest, 1)
            elif d.name.startswith("aretomo_output-v"):
                try:
                    highest = max(highest, int(d.name[len("aretomo_output-v"):]))
                except ValueError:
                    pass
        return self.root / f"aretomo_output-v{highest + 1}"

    def write_aretomo_parameters(self, output_dir, values, command):
        """Drop a PARAMETERS.txt audit record into a versioned AreTomo folder,
        so a run can always be matched back to the parameters that produced it.
        Generalises the bash-script pattern (brief §4.6 / §5 iteration queue)."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = ["# AreTomo2 run parameters",
                 f"# Written by tomogration at {ts}", ""]
        for k, v in values.items():
            s = str(v).strip()
            if s:
                lines.append(f"{k:<16}= {s}")
        lines += ["", "# full command", command, ""]
        pf = out / "PARAMETERS.txt"
        pf.write_text("\n".join(lines))
        return pf

    # ---- selection sync helper ------------------------------------------
    # [0,0,0] until a tilt series has a geometry -- which is exactly what
    # importing an alignment gives it. The miss-alignment wrapper already parks
    # series on this test, so the app and the wrapper agree by construction.
    _VOL_DIMS = re.compile(
        r"VolumeDimensionsAngstrom[^\d\-]*(-?[\d.]+)[,\s]+(-?[\d.]+)[,\s]+(-?[\d.]+)")

    def series_without_imported_alignment(self):
        """tomostar stems whose warp_tiltseries/<stem>.xml carries no alignment.

        The question miss-alignment actually cares about. Asking instead "does a
        .xf exist somewhere?" is wrong in both directions:

          * a .xf that was never IMPORTED leaves the XML untouched -- J17
            refined nothing on precisely that, and the .xf check would have
            waved it through;
          * an alignment that arrived another way (a hand-run import, etomo
            patches) has no .xf to find, so a correct project gets warned at.

        Unreadable or absent XMLs count as unaligned: the honest answer when the
        record cannot be read is "not proven", not "fine".
        """
        tomostar_dir = self.root / "tomostar"
        xml_dir = self.root / "warp_tiltseries"
        if not tomostar_dir.is_dir():
            return []
        out = []
        for f in sorted(tomostar_dir.glob("*.tomostar")):
            xml = xml_dir / f"{f.stem}.xml"
            try:
                # Bounded read: these are ~45 KB each and there can be hundreds,
                # over ceph. The field sits in the header block.
                text = xml.read_text(errors="replace")[:262144]
            except OSError:
                out.append(f.stem)
                continue
            m = self._VOL_DIMS.search(text)
            if not m or not any(float(v) != 0 for v in m.groups()):
                out.append(f.stem)
        return out

    def tomostars_without_alignments(self):
        """tomostar basenames present in tomostar/ but with no .xf in the newest
        AreTomo version folder. These crash ts_reconstruct unless deselected."""
        tomostar_dir = self.root / "tomostar"
        if not tomostar_dir.is_dir():
            return []
        all_tomostars = {f.stem for f in tomostar_dir.glob("*.tomostar")}
        if not all_tomostars:
            return []
        # _aretomo_candidates, NOT list_aretomo_versions: the latter scans only
        # the project root for aretomo_output[-vN]. A job-scoped run writes into
        # jobs/J13_aretomo_output/, so on this project every series looked
        # unaligned -- and the "deselect all unaligned" helper would have built
        # a command deselecting the entire dataset.
        aligned = set()
        for active in self._aretomo_candidates():
            for layout in ("Imod", "imod"):
                ld = active / layout
                if not ld.is_dir():
                    continue
                try:
                    subs = list(ld.iterdir())
                except OSError:
                    continue
                for sub in subs:
                    if sub.is_dir() and next(sub.glob("*.xf"), None):
                        name = sub.name
                        if name.endswith("_Imod"):
                            name = name[:-len("_Imod")]
                        aligned.add(name)
                    elif sub.is_file() and sub.suffix == ".xf":
                        aligned.add(sub.stem)
            if aligned:
                break            # newest folder that actually aligned anything
        return sorted(all_tomostars - aligned)

    # ---- position-level inspection: mdoc contents vs frames/ -------------
    def parse_mdoc_subframes(self, mdoc_path):
        """[(acq_index, tilt_angle, eer_basename), ...] in acquisition order."""
        entries = []
        current_angle = None
        current_subframe = None
        acq_index = 0

        def flush():
            nonlocal current_angle, current_subframe, acq_index
            if current_subframe is not None:
                entries.append((acq_index, current_angle, current_subframe))
            current_angle = None
            current_subframe = None

        if not os.path.isfile(mdoc_path):
            return entries
        with open(mdoc_path, errors="replace") as f:
            for line in f:
                s = line.strip()
                if s.startswith("[ZValue"):
                    flush()
                    acq_index += 1
                elif s.startswith("TiltAngle") and "=" in s:
                    try:
                        current_angle = float(s.split("=", 1)[1].strip())
                    except ValueError:
                        pass
                elif s.startswith("SubFramePath") and "=" in s:
                    raw = s.split("=", 1)[1].strip().replace("\\", "/")
                    current_subframe = raw.rsplit("/", 1)[-1]
        flush()
        return entries

    def mdoc_acquisition(self):
        """{'angpix','exposure','image_size'} read from a real mdoc, or {}.

        The stage defaults are ONE dataset's numbers hard-coded (1.57 A/px,
        3.5 e/A^2). A project collected at a different magnification inherits
        them in silence: nothing errors, no path is missing, every downstream
        box is simply wrong. The mdoc states what was actually collected.

        PixelSpacing/ExposureDose repeat per [ZValue] block; the first is the
        series' own value. Values that don't parse as numbers are dropped
        rather than passed on -- a bad default is worse than the known one.
        """
        md = self.root / "mdocs"
        if not md.is_dir():
            return {}
        try:
            mdocs = sorted(e for e in md.glob("*.mdoc")
                           if not e.name.lower().endswith("_override.mdoc"))
        except OSError:
            return {}
        if not mdocs:
            return {}
        out = {}
        try:
            with open(mdocs[0], errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key, val = key.strip(), val.strip()
                    if key == "PixelSpacing" and "angpix" not in out:
                        out["angpix"] = val
                    elif key == "ExposureDose" and "exposure" not in out:
                        out["exposure"] = val
                    elif key == "ImageSize" and "image_size" not in out:
                        out["image_size"] = val.split()
                    if len(out) == 3:
                        break
        except OSError:
            return {}
        for k in ("angpix", "exposure"):
            try:
                if not float(out.get(k, "")) > 0:
                    out.pop(k, None)
            except ValueError:
                out.pop(k, None)
        dims = out.get("image_size") or []
        if not (len(dims) >= 2 and all(d.isdigit() and int(d) > 0 for d in dims[:2])):
            out.pop("image_size", None)
        return out

    def inspect_positions(self):
        """For every Position with a .mdoc, compare mdoc subframes against
        frames/. Returns a list of per-position dicts."""
        results = []
        mdocs_dir = self.root / "mdocs"
        frames_dir = self.root / "frames"
        if not mdocs_dir.is_dir():
            return results
        frames_by_pos = {}
        if frames_dir.is_dir():
            for f in frames_dir.glob("Position*.eer"):
                frames_by_pos.setdefault(f.name.split("_")[0], []).append(f.name)
        for mdoc in sorted(mdocs_dir.glob("Position*.mdoc")):
            pos = mdoc.stem
            entries = self.parse_mdoc_subframes(str(mdoc))
            referenced = {e[2] for e in entries if e[2]}
            on_disk = set(frames_by_pos.get(pos, []))
            results.append({
                "name": pos,
                "mdoc_path": str(mdoc),
                "mdoc_entries": entries,
                "frames_on_disk": sorted(on_disk),
                "matched": sorted(referenced & on_disk),
                "extra_on_disk": sorted(on_disk - referenced),
                "missing_from_disk": sorted(referenced - on_disk),
            })
        return results

    def move_frames_to_bad(self, names):
        """Move named .eer files from frames/ to frames/bad/. Returns count."""
        frames = self.root / "frames"
        bad = frames / "bad"
        bad.mkdir(parents=True, exist_ok=True)
        moved = 0
        for n in names:
            src = frames / n
            if src.is_file():
                try:
                    src.rename(bad / n)
                    moved += 1
                except OSError:
                    pass
        return moved

    # ---- raw-dump sorter -------------------------------------------------
    @staticmethod
    def _sort_bucket(name, stack_stems=()):
        """Which standard folder a file belongs in (or None to leave alone).
        Tomo5 *_override.mdoc are handled separately, not here.

        stack_stems: stems known to be tilt series because a standard .mdoc of
        the same stem sits in the same dump. A dump sorted BEFORE rename has
        Tomo5-named stacks, so a Position* test alone would leave those behind —
        which is exactly how the MRCs came to pile up in the root. The gain test
        runs first, so a *gain*.mrc never reaches this."""
        low = name.lower()
        if low.endswith(".eer"):
            return "frames"
        if low.endswith(".gain") or ("gain" in low and low.endswith(".mrc")):
            return "gains"
        if low.endswith(".mdoc") and not low.endswith("_override.mdoc"):
            return "mdocs"
        if low.endswith(".mrc") and (low.startswith("position")
                                     or name[:-len(".mrc")] in stack_stems):
            return COARSE_DIR
        return None

    @staticmethod
    def _override_standard_name(override_name):
        """Standard mdoc name for a Tomo5 override, e.g.
        'Position_1_override.mdoc' -> 'Position_1.mdoc'."""
        return override_name[:-len("_override.mdoc")] + ".mdoc"

    def plan_sort(self, src):
        """Preview of sort_files: {frames, mdocs, gains, mrcs-tiltseries-coarse,
        overrides, skipped} lists of filenames."""
        plan = {"frames": [], "mdocs": [], "gains": [], COARSE_DIR: [],
                "overrides": [], "skipped": []}
        src = Path(src)
        if not src.is_dir():
            return plan
        files = [f for f in sorted(src.iterdir()) if f.is_file()]
        # Every stem with a standard .mdoc is a tilt series, whatever it is
        # named — that is what makes an unrenamed dump sortable too.
        stack_stems = {f.name[:-len(".mdoc")] for f in files
                       if f.name.lower().endswith(".mdoc")
                       and not f.name.lower().endswith("_override.mdoc")}
        for f in files:
            if f.name.lower().endswith("_override.mdoc"):
                plan["overrides"].append(f.name)
                continue
            bucket = self._sort_bucket(f.name, stack_stems)
            plan[bucket if bucket else "skipped"].append(f.name)
        return plan

    def clean_override_mdocs(self, folder):
        """Quarantine Tomo5 *_override.mdoc that are byte-identical to their
        standard mdoc into mdocs/bad/ — they otherwise make the rename script
        consume bogus position numbers. Overrides that DIFFER (or have no
        standard) are left in place. Returns (removed, kept)."""
        folder = Path(folder)
        bad = self.root / "mdocs" / "bad"
        bad.mkdir(parents=True, exist_ok=True)
        removed = kept = 0
        for ov in sorted(folder.glob("*_override.mdoc")):
            std = folder / self._override_standard_name(ov.name)
            identical = std.is_file() and filecmp.cmp(str(ov), str(std), shallow=False)
            if identical:
                dest = bad / ov.name
                if dest.exists():
                    continue
                try:
                    shutil.move(str(ov), str(dest))
                    removed += 1
                except OSError:
                    pass
            else:
                kept += 1
        return removed, kept

    def sort_files(self, src):
        """Make the standard dirs and move files from src into frames/ (.eer),
        mdocs/ (.mdoc), gains/ (gain), mrcs-tiltseries-coarse/ (the per-series
        tilt stacks). Dedups Tomo5 *_override.mdoc first. Never overwrites an
        existing target. Returns a counts dict.

        NOTE: this SPLITS .eer and .mdoc apart, so run it AFTER rename (the
        rename script needs them in one folder)."""
        self.initialize_structure()
        src = Path(src)
        removed, kept = self.clean_override_mdocs(src)
        plan = self.plan_sort(src)                       # re-plan after cleanup
        moved = {"frames": 0, "mdocs": 0, "gains": 0, COARSE_DIR: 0,
                 "overrides_removed": removed, "overrides_kept": kept}
        for bucket in ("frames", "mdocs", "gains", COARSE_DIR):
            dest_dir = self.root / bucket
            for name in plan[bucket]:
                s = src / name
                d = dest_dir / name
                if not s.is_file():
                    continue
                try:
                    if d.exists() or s.resolve() == d.resolve():
                        continue       # don't overwrite / no-op if already there
                    shutil.move(str(s), str(d))
                    moved[bucket] += 1
                except OSError:
                    pass
        return moved

    def collect_coarse_stacks(self):
        """Move loose top-level Position*.mrc into mrcs-tiltseries-coarse/,
        making the folder if needed. Returns how many moved.

        For projects sorted by an older build, whose coarse stacks were left in
        the root because nothing claimed them. Scoped tightly on purpose: the
        root's own top level only, Position-prefixed .mrc only, no gains, never
        overwriting. Reconstructions, templates and corr volumes live in
        subfolders and are not Position-prefixed at the root, so none of them is
        in scope. scandir, not glob, because the pile being cleaned up is
        thousands of files on ceph and its entries already carry their type."""
        if not self.root.is_dir():
            return 0
        try:
            names = sorted(e.name for e in os.scandir(self.root)
                           if e.is_file()
                           and e.name.lower().startswith("position")
                           and e.name.lower().endswith(".mrc")
                           and "gain" not in e.name.lower())
        except OSError:
            return 0
        if not names:
            return 0
        dest = self.root / COARSE_DIR
        dest.mkdir(parents=True, exist_ok=True)
        moved = 0
        for name in names:
            target = dest / name
            if target.exists():
                continue
            try:
                shutil.move(str(self.root / name), str(target))
                moved += 1
            except OSError:
                pass
        return moved

    def coarse_stack_stems(self):
        """Series names taken from mrcs-tiltseries-coarse/*.mrc, sorted.

        Once sorted, this folder is the most direct statement of which tilt
        series the project actually has — one stack per series, nothing else."""
        d = self.root / COARSE_DIR
        if not d.is_dir():
            return []
        try:
            return sorted(p.stem for p in d.glob("*.mrc"))
        except OSError:
            return []

    # ---- visual tilt inspection support ---------------------------------
    def find_thumbnails_dir(self):
        """Locate the Tomo5 Thumbnails/ folder (per-series montage .mrc).
        Checks the root and one level down. Returns a Path or None."""
        direct = self.root / "Thumbnails"
        if direct.is_dir():
            return direct
        for d in sorted(self.root.iterdir()) if self.root.is_dir() else []:
            if d.is_dir():
                cand = d / "Thumbnails"
                if cand.is_dir():
                    return cand
        return None

    def load_listfile(self):
        """Parse listfile_*.txt (old_name -> new_name) into {tomo5: renamed}."""
        mapping = {}
        if not self.root.is_dir():
            return mapping
        candidates = list(self.root.glob("listfile_*.txt")) + [self.root / "listfile.txt"]
        for path in candidates:
            if not path.is_file():
                continue
            for line in path.read_text(errors="replace").splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split()
                if len(parts) >= 2:
                    mapping[parts[0]] = parts[1]
            if mapping:
                break
        return mapping

    def load_listfile_reverse(self):
        """{renamed -> original_tomo5}, read directly from the file so a merged
        dataset (where several grids may share a Tomo5 name) still maps each unique
        renamed Position### back to the right original. Used for the inspector's
        'was <Tomo5>' conversion note."""
        rev = {}
        if not self.root.is_dir():
            return rev
        candidates = list(self.root.glob("listfile_*.txt")) + [self.root / "listfile.txt"]
        for path in candidates:
            if not path.is_file():
                continue
            for line in path.read_text(errors="replace").splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split()
                if len(parts) >= 2:
                    rev[parts[1]] = parts[0]      # renamed -> tomo5
            if rev:
                break
        return rev

    def find_series_stack(self, tomo5_name, renamed):
        """Full-res per-series .mrc to open in 3dmod. mrcs-tiltseries-coarse/
        first (where sorting now puts the coarse stacks), then the root itself
        (projects sorted by an older build), each under the renamed name before
        the Tomo5 one, then the thumbnail montage. Returns a Path or None."""
        coarse = self.root / COARSE_DIR
        for cand in (coarse / f"{renamed}.mrc",
                     coarse / f"{tomo5_name}.mrc",
                     self.root / f"{renamed}.mrc",
                     self.root / f"{tomo5_name}.mrc"):
            if cand.is_file():
                return cand
        thumbs = self.find_thumbnails_dir()
        if thumbs:
            t = thumbs / f"{tomo5_name}.mrc"
            if t.is_file():
                return t
        return None

    def write_manual_exclusions(self, mapping):
        """Merge {position: 'tilt,tilt'} into the MANUAL section of
        exclusion_list.txt (above AUTO_EXCLUSION_HEADER), preserving the auto
        section. position is the RENAMED name; tilts are IMOD-order numbers."""
        path = self.root / "exclusion_list.txt"
        existing = path.read_text() if path.exists() else ""
        if self.AUTO_EXCLUSION_HEADER in existing:
            manual, auto = existing.split(self.AUTO_EXCLUSION_HEADER, 1)
            auto = self.AUTO_EXCLUSION_HEADER + auto
        else:
            manual, auto = existing, ""
        current = {}
        for line in manual.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split(None, 1)
            current[parts[0]] = parts[1].strip().rstrip(",") if len(parts) > 1 else ""
        for pos, tilts in mapping.items():
            current[pos] = str(tilts).strip().rstrip(",")
        # Expand ranges (1,4-12,48 -> explicit list) — remake_mdocs can't parse
        # ranges. Normalises any pre-existing ranged entries too.
        lines = []
        for pos in sorted(current):
            expanded = expand_tilt_ranges(current[pos])
            if expanded:
                lines.append(f"{pos}\t{expanded}")
        new_manual = "\n".join(lines) + ("\n" if lines else "")
        # Atomic: this file is hand-curated tilt knowledge — a crash mid-write
        # must not leave it truncated.
        _atomic_write_text(path, new_manual + (auto if auto.strip() else ""))
        return len(lines)

    def series_defocus_A(self, stem):
        """Defocus (Å) for one tilt series, or None (missing xml / attribute)."""
        p = self.root / "warp_tiltseries" / f"{stem}.xml"
        try:
            return defocus_angstroms(p.read_text()) if p.is_file() else None
        except OSError:
            return None

    def status_mb_deconv(self):
        """Deconvolved volumes across all membrane/deconv variants."""
        n = self.count_rglob("membrane/deconv", "*_deconv.mrc")
        return (n, f"{n} deconvolved volume(s)" if n else None)

    def status_mb_segment(self):
        """Score maps are the completion marker (the artifact downstream
        thresholding needs); plain segmentations without scores are counted
        second so a no-probs run still shows."""
        n = self.count_rglob("membrane/segment", "*_scores.mrc")
        if n:
            return (n, f"{n} score map(s)")
        n = self.count_rglob("membrane/segment", "*.mrc")
        return (n, f"{n} segmentation(s), NO score maps" if n else None)

    def status_mb_isonet_train(self):
        """Trained models, labelled by the HIGHEST iteration reached.

        A plain file count lied: IsoNet writes results/model_iter00.h5 BEFORE
        the first iteration trains, so a refine that crashed inside iteration 1
        left one .h5 behind and the card went green with '1 trained model(s)'
        (seen 2026-08-15). iter00 alone is not a model, so it counts as zero
        and says why."""
        d = self.root / "membrane" / "isonet" / "results"
        if not d.is_dir():
            return (0, None)
        iters = []
        for p in d.glob("model_iter*.h5"):
            m = re.search(r"model_iter(\d+)\.h5$", p.name)
            if m:
                iters.append(int(m.group(1)))
        if not iters:
            return (0, None)
        top = max(iters)
        if top == 0:
            return (0, "only model_iter00 — refine crashed in iteration 1")
        return (len(iters), f"{len(iters)} model(s), trained to iter {top}")

    def status_mb_isonet_predict(self):
        n = self.count_rglob("membrane/isonet_corrected/corrected", "*.mrc")
        return (n, f"{n} wedge-restored tomogram(s)" if n else None)

    def status_mb_isonet2_train(self):
        """IsoNet 2 checkpoints. Its refine writes .pt (not .h5) into
        isonet_maps/, and save_interval writes them DURING training, so a
        count is 'how far did it get', not 'is it finished' — the wrapper's
        own did-it-train check is the authority on that. Reported by newest
        checkpoint name so a card shows which one predict should point at."""
        d = self.root / "membrane" / "isonet2" / "isonet_maps"
        if not d.is_dir():
            return (0, None)
        try:
            pts = sorted(d.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        except OSError:
            return (0, None)
        if not pts:
            return (0, None)
        return (len(pts), f"{len(pts)} checkpoint(s), newest {pts[-1].name}")

    def status_mb_isonet2_predict(self):
        n = self.count_rglob("membrane/isonet2_corrected/corrected", "*.mrc")
        return (n, f"{n} corrected tomogram(s)" if n else None)

    def status_mb_thresholds(self):
        n = self.count_rglob("membrane/thresholds", "*_threshold_*.mrc")
        return (n, f"{n} thresholded volume(s)" if n else None)

    def status_mb_components(self):
        n = self.count_rglob("membrane/components", "*.mrc")
        return (n, f"{n} labelled volume(s)" if n else None)

    def status_mb_polarity(self):
        """How many variants have a MEASURED polarity. Zero is the state the
        fit gate silently tolerates, so it is worth showing on the card."""
        try:
            data = json.loads(
                (self.root / ".tomogration_variants.json").read_text())
            variants = data.get("variants") or []
        except (OSError, ValueError):
            return (0, None)
        known = [v for v in variants
                 if str(v.get("polarity", "unknown")) in ("dark", "bright")]
        if not variants:
            return (0, None)
        return (len(known),
                f"{len(known)}/{len(variants)} variant(s) measured")

    def status_mb_size_qc(self):
        try:
            found = self._find_outputs("size_qc.json")
        except OSError:
            return (0, None)
        n = len(dict.fromkeys(found))
        return (n, f"{n} QC report(s)" if n else None)

    def status_mb_population(self):
        """The pooled measurement, read from population.json.

        A folder full of per-tomogram scratch would count as progress even when
        the pooling never happened, so only the pooled file counts."""
        import json                                          # noqa: PLC0415
        best = None
        try:
            found = self._find_outputs("population.json")
        except OSError:
            return (0, None)
        for f in dict.fromkeys(found):
            try:
                d = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            n = int(d.get("n_virions") or 0)
            if n and (best is None or n > best[0]):
                best = (n, float(d.get("radius_A") or 0.0))
        if not best:
            return (0, None)
        return (best[0], f"{best[0]} virions, radius {best[1]:.0f} Å")

    def status_mb_split(self):
        """Re-labelled component volumes written by the splitter."""
        try:
            n = len(self._find_outputs("*_split.mrc"))
        except OSError:
            return (0, None)
        return (n, f"{n} volume(s) re-labelled" if n else None)

    def _find_outputs(self, pattern):
        """Files matching `pattern` wherever a job may have put them.

        A job's output folder defaults to jobs/{jobid} so runs cannot overwrite
        each other, but the folder is a free-text parameter and a membrane/…
        path is equally valid. Looking in only one of the two places meant the
        card read 'nothing yet' next to a job that had plainly just written
        the file."""
        found = []
        for base in ("jobs", "membrane"):
            d = self.root / base
            if d.is_dir():
                found += list(d.rglob(pattern))
        return list(dict.fromkeys(found))

    def status_mb_fits(self):
        """Accepted virions, read from the fits.json the fitter writes.

        Counting fit_*.mrc masks would lie twice over: they are optional, and a
        run that accepted nothing still leaves the previous run's masks."""
        n_files = n_virions = 0
        try:
            # Both homes: jobs/J##_fit-virions (the default since named dirs)
            # and the tune-era membrane/ folders.
            found = self._find_outputs("fits.json")
        except OSError:
            return (0, None)
        for p in set(found):
            try:
                d = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            n_files += 1
            n_virions += int(d.get("accepted") or 0)
        if not n_files:
            return (0, None)
        return (n_virions, f"{n_virions} virion(s) in {n_files} fit run(s)")

    def status_mb_picks(self):
        n = self.count_files("membrane/picks", "*.star")
        return (n, f"{n} pick star(s)" if n else None)

    def status_mb_explore(self):
        """Sweep cells completed, plus how much of the expensive axis is cached
        — the number that says whether a re-run is minutes or hours."""
        rows = seg = 0
        try:
            results = self._find_outputs("results.json")
        except OSError:
            results = []
        for p in results:
            try:
                data = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            rows += len(data.get("rows") or [])
            d = p.parent / "cache" / "segment"
            if d.is_dir():
                try:
                    seg += sum(1 for x in d.iterdir()
                               if (x / "DONE").is_file())
                except OSError:
                    pass
        if not rows and not seg:
            return (0, None)
        bits = []
        if rows:
            bits.append(f"{rows} cell(s)")
        if seg:
            bits.append(f"{seg} segmentation(s) cached")
        return (rows or seg, ", ".join(bits))

    def status_mb_mesh(self):
        """Mesh containers; membrain-pick writes .h5 containers (surforama
        opens them), with any other non-provenance file counted as fallback."""
        n = self.count_rglob("membrane/mesh", "*.h5")
        if n:
            return (n, f"{n} mesh container(s)")
        n = self.count_rglob("membrane/mesh", "*.mrc")
        return (n, f"{n} mesh output(s)" if n else None)

    def latest_relion4_export(self):
        """Newest relion4/<export>/ dir holding a .star — the live default for
        the RELION stages' project_dir. Exports write per-job dirs (relion4/J##,
        relion4/trunk_*); the fixed relion4/warp predates them."""
        base = self.root / "relion4"
        if not base.is_dir():
            return ""
        best, best_m = "", -1.0
        try:
            entries = list(base.iterdir())
        except OSError:
            return ""
        for d in entries:
            try:
                if not d.is_dir() or not any(d.glob("*.star")):
                    continue
                m = d.stat().st_mtime
            except OSError:
                continue
            if m > best_m:
                best, best_m = f"relion4/{d.name}", m
        return best

    def is_series_excluded(self, renamed):
        """True if this series was quarantined (its mdoc is now in mdocs/bad/)."""
        return (self.root / "mdocs" / "bad" / f"{renamed}.mdoc").is_file()

    def output_versions(self, output_spec):
        """[(label, Path), …] of existing output dirs for a stage, newest last.
        'aretomo' -> the versioned AreTomo folders; otherwise the single dir."""
        if output_spec == "aretomo":
            return [(p.name, p) for p in self.list_aretomo_versions()]
        base = self.root / output_spec
        if base.is_dir():
            return [("(root)" if output_spec == "." else output_spec, base)]
        return []

    def quarantine_series(self, renamed):
        """Exclude a whole tilt series: move its mdoc -> mdocs/bad/ and its .eer
        -> frames/bad/ (wherever they currently live). Returns (mdocs, eers)."""
        (self.root / "mdocs" / "bad").mkdir(parents=True, exist_ok=True)
        (self.root / "frames" / "bad").mkdir(parents=True, exist_ok=True)
        n_mdoc = n_eer = 0
        for mdoc in (self.root / "mdocs" / f"{renamed}.mdoc",
                     self.root / f"{renamed}.mdoc"):
            if mdoc.is_file():
                try:
                    mdoc.rename(self.root / "mdocs" / "bad" / mdoc.name)
                    n_mdoc += 1
                except OSError:
                    pass
        for folder in (self.root / "frames", self.root):
            if folder.is_dir():
                for eer in folder.glob(f"{renamed}_*.eer"):
                    try:
                        eer.rename(self.root / "frames" / "bad" / eer.name)
                        n_eer += 1
                    except OSError:
                        pass
        return n_mdoc, n_eer

    def quarantine_listed_whole_series(self):
        """Quarantine every MANUAL exclusion-list line that names a series with NO
        tilt numbers — i.e. a bare 'PositionNNN' line means 'drop the whole series'.
        Returns the list of names quarantined. Called just before _normalize_exclusions
        (which would otherwise silently drop the tilt-less lines, since the remake
        script only trims individual tilts). Idempotent: already-excluded series are
        skipped. Lines WITH tilt numbers are left for Remake mdocs to apply."""
        path = self.root / "exclusion_list.txt"
        if not path.is_file():
            return []
        text = path.read_text()
        if self.AUTO_EXCLUSION_HEADER in text:
            text = text.split(self.AUTO_EXCLUSION_HEADER, 1)[0]
        done = []
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split(None, 1)
            tilts = parts[1].strip().rstrip(",") if len(parts) > 1 else ""
            if tilts:                       # has tilt numbers → Remake handles it
                continue
            name = parts[0]
            if self.is_series_excluded(name):
                continue
            nm, ne = self.quarantine_series(name)
            if nm or ne:
                done.append(name)
        return done

    # ---- mdoc surgical editing (brief gotcha §4.3) -----------------------
    @staticmethod
    def remove_mdoc_zvalue_block(mdoc_path, eer_basename):
        """Remove the ZValue block referencing eer_basename and renumber the
        remaining [ZValue = N] blocks contiguously from 0. Returns True/False."""
        if not os.path.isfile(mdoc_path):
            return False
        with open(mdoc_path, errors="replace") as f:
            lines = f.readlines()
        header_end = None
        for i, line in enumerate(lines):
            if line.lstrip().startswith("[ZValue"):
                header_end = i
                break
        if header_end is None:
            return False
        header, rest = lines[:header_end], lines[header_end:]
        blocks, current = [], []
        for line in rest:
            if line.lstrip().startswith("[ZValue") and current:
                blocks.append(current)
                current = [line]
            else:
                current.append(line)
        if current:
            blocks.append(current)
        target_idx = None
        for i, blk in enumerate(blocks):
            if eer_basename in "".join(blk):
                target_idx = i
                break
        if target_idx is None:
            return False
        kept = blocks[:target_idx] + blocks[target_idx + 1:]
        renumbered = []
        for new_idx, blk in enumerate(kept):
            replaced = False
            for line in blk:
                stripped = line.lstrip()
                if not replaced and stripped.startswith("[ZValue"):
                    leading = line[:len(line) - len(stripped)]
                    renumbered.append(f"{leading}[ZValue = {new_idx}]\n")
                    replaced = True
                else:
                    renumbered.append(line)
        with open(mdoc_path, "w") as f:
            f.writelines(header)
            f.writelines(renumbered)
        return True

    def repair_all_mdocs(self):
        """Strip ZValue blocks referencing .eer files no longer in frames/.
        Returns [(position, eer_basename), ...] for everything removed."""
        removed = []
        mdocs_dir = self.root / "mdocs"
        frames_dir = self.root / "frames"
        if not mdocs_dir.is_dir() or not frames_dir.is_dir():
            return removed
        present = {f.name for f in frames_dir.glob("*.eer")}
        for mdoc in sorted(mdocs_dir.glob("Position*.mdoc")):
            while True:
                changed = False
                for acq, angle, subframe in self.parse_mdoc_subframes(str(mdoc)):
                    if subframe and subframe not in present:
                        if self.remove_mdoc_zvalue_block(str(mdoc), subframe):
                            removed.append((mdoc.stem, subframe))
                            changed = True
                            break
                if not changed:
                    break
        return removed

    def dry_run_repair_mdocs(self):
        """Preview of repair_all_mdocs: list of (mdoc_name, subframe) stale entries."""
        to_remove = []
        mdocs_dir = self.root / "mdocs"
        frames_dir = self.root / "frames"
        if not mdocs_dir.is_dir() or not frames_dir.is_dir():
            return to_remove
        present = {f.name for f in frames_dir.glob("*.eer")}
        for mdoc in sorted(mdocs_dir.glob("Position*.mdoc")):
            for acq, angle, subframe in self.parse_mdoc_subframes(str(mdoc)):
                if subframe and subframe not in present:
                    to_remove.append((mdoc.name, subframe))
        return to_remove

    # ---- auto-exclusion logging (brief gotcha §4.5) ----------------------
    def append_auto_exclusion(self, eer_basename, reason, position=None,
                              tilt_angle=None):
        """Append an entry below AUTO_EXCLUSION_HEADER in exclusion_list.txt.
        Idempotent on filename. Manual exclusions above the header are left
        untouched (remake_mdocs must stop parsing at the header)."""
        path = self.root / "exclusion_list.txt"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        existing = path.read_text() if path.exists() else ""
        if self.AUTO_EXCLUSION_HEADER in existing:
            manual_part, auto_part = existing.split(self.AUTO_EXCLUSION_HEADER, 1)
        else:
            manual_part, auto_part = existing, ""
        if eer_basename in auto_part:
            return False
        bits = [f"# {timestamp}"]
        if position:
            bits.append(f"position={position}")
        if tilt_angle is not None:
            bits.append(f"tilt={tilt_angle}")
        bits.append(f"reason={reason}")
        new_entry = f"{' '.join(bits)}\n{eer_basename}\n"
        manual_part = manual_part.rstrip() + "\n" if manual_part.strip() else ""
        new_auto = (auto_part.rstrip() + "\n" + new_entry
                    if auto_part.strip() else "\n" + new_entry)
        _atomic_write_text(path,
                           manual_part + self.AUTO_EXCLUSION_HEADER + new_auto)
        return True

    # ---- project structure -----------------------------------------------
    def initialize_structure(self):
        for d in EXPECTED_DIRS:
            (self.root / d).mkdir(parents=True, exist_ok=True)

    # ---- tilt-series groups (process named subsets for optimization) ------
    GROUPS_FILE = ".tomogration_groups.json"
    ALL_GROUP = "All tilt series"

    def load_groups(self):
        """{'active': name, 'groups': {name: [series, ...]}}. 'All tilt series'
        is implicit (no restriction)."""
        p = self.root / self.GROUPS_FILE
        if p.is_file():
            try:
                d = json.loads(p.read_text())
                if isinstance(d, dict) and isinstance(d.get("groups"), dict):
                    d.setdefault("active", self.ALL_GROUP)
                    return d
            except (OSError, ValueError):
                pass
        return {"active": self.ALL_GROUP, "groups": {}}

    def save_groups(self, data):
        _atomic_write_text(self.root / self.GROUPS_FILE,
                           json.dumps(data, indent=2))

    def available_tiltseries(self):
        """Sorted unique tilt-series base names — from tomostar/ if present,
        else mdocs/, else frames/ (so groups work at any stage)."""
        names = set()
        td = self.root / "tomostar"
        if td.is_dir():
            names |= {f.stem for f in td.glob("*.tomostar")}
        md = self.root / "mdocs"
        if md.is_dir():
            names |= {f.stem for f in md.glob("Position*.mdoc")}
        if not names:
            fd = self.root / "frames"
            if fd.is_dir():
                names |= {f.name.split("_")[0] for f in fd.glob("Position*.eer")}
        return sorted(names)

    def write_group_inputs(self, series):
        """Materialise a group into WarpTools --input_data lists. Returns
        {'ts': path|None, 'fs': path|None}. WarpTools resolves --input_data
        entries RELATIVE TO THE PROJECT ROOT, so paths include the data folder:
        ts = 'tomostar/<name>.tomostar' per line; fs = 'frames/<eer>' per line
        (the .eer of each series, read from its mdoc)."""
        out = {"ts": None, "fs": None}
        series = [s for s in series if s]
        if not series:
            return out
        ts_path = self.root / ".group_tiltseries.txt"
        ts_path.write_text("\n".join(f"tomostar/{n}.tomostar" for n in series) + "\n")
        out["ts"] = ts_path
        eers = []
        md = self.root / "mdocs"
        for n in series:
            mdoc = md / f"{n}.mdoc"
            if mdoc.is_file():
                eers += [f"frames/{sub}" for _, _, sub
                         in self.parse_mdoc_subframes(str(mdoc)) if sub]
        if eers:
            fs_path = self.root / ".group_frameseries.txt"
            fs_path.write_text("\n".join(eers) + "\n")
            out["fs"] = fs_path
        return out

    def group_dir(self, name):
        """Path where materialize_group would build this group's subset folder."""
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "group"
        return self.root / "groups" / safe

    def materialize_group(self, name, series):
        """Build a self-contained working folder for a group at groups/<name>/,
        SYMLINKING the group's frames (.eer) + mdocs (.mdoc) + gains into it, so
        the whole pipeline can run on just that subset (set it as the project
        root). Returns (path, n_eer, n_mdoc). Symlinks, not copies — no extra disk."""
        base = self.group_dir(name)
        for sub in ("frames", "mdocs", "gains"):
            (base / sub).mkdir(parents=True, exist_ok=True)
        frames, mdocs, gains = (self.root / "frames", self.root / "mdocs",
                                self.root / "gains")
        n_eer = n_mdoc = 0
        for s in series:
            md = mdocs / f"{s}.mdoc"
            if md.is_file() and self._symlink(md, base / "mdocs" / md.name):
                n_mdoc += 1
            if frames.is_dir():
                for eer in frames.glob(f"{s}_*.eer"):
                    if self._symlink(eer, base / "frames" / eer.name):
                        n_eer += 1
        if gains.is_dir():
            for g in gains.iterdir():
                if g.is_file():
                    self._symlink(g, base / "gains" / g.name)
        return base, n_eer, n_mdoc

    @staticmethod
    def _symlink(src, dst):
        try:
            if dst.exists() or dst.is_symlink():
                return True
            dst.symlink_to(Path(src).resolve())
            return True
        except OSError:
            return False


