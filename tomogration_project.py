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
import struct
from pathlib import Path

from tomogration_core import expand_tilt_ranges

EXPECTED_DIRS = [
    "frames", "frames/bad", "mdocs", "mdocs/bad", "gains",
    "aretomo_output", "aretomo_output/Imod",
    "warp_frameseries", "warp_tiltseries", "tomostar",
]

# WarpTools per-item failure line; used by auto-recovery to find the culprit.
_FAILED_FILE_RE = re.compile(
    r"Failed to process\s+(\S+\.(?:eer|mrc|tiff?|st|tomostar))", re.IGNORECASE)
# WarpTools per-item progress line, e.g. "239/5439, 08:06:22 remaining" or
# "5439/5439, previous metadata found for 278" — collapsed to one updating line.
_PROGRESS_RE = re.compile(r"^\s*\d+\s*/\s*\d+\b")


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
            return data if isinstance(data, list) else []
        except (OSError, ValueError):
            return []

    def _save_history(self, hist):
        try:
            (self.root / self.HISTORY_FILE).write_text(json.dumps(hist, indent=1))
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
        matches = sorted(p.glob("*.gain")) + sorted(p.glob("*_gain*.mrc"))
        matches = [m for m in matches if "reciprocal" not in m.name.lower()]
        return (True, matches[0].name) if matches else (False, None)

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
        n = self.count_files("warp_tiltseries", "*.xml")
        return (n, f"{n} processed tilt series" if n else None)

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
        for d in reversed(self.list_aretomo_versions()):
            imod = d / "Imod"
            if not imod.is_dir():
                continue
            if next(imod.glob("*.xf"), None) or next(imod.glob("*/*.xf"), None):
                return os.path.relpath(imod, self.root) + "/"
        return ""

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
    def tomostars_without_alignments(self):
        """tomostar basenames present in tomostar/ but with no .xf in the newest
        AreTomo version folder. These crash ts_reconstruct unless deselected."""
        tomostar_dir = self.root / "tomostar"
        if not tomostar_dir.is_dir():
            return []
        all_tomostars = {f.stem for f in tomostar_dir.glob("*.tomostar")}
        if not all_tomostars:
            return []
        versions = self.list_aretomo_versions()
        aligned = set()
        if versions:
            active = versions[-1]
            for layout in ("Imod", "imod"):
                ld = active / layout
                if not ld.is_dir():
                    continue
                for sub in ld.iterdir():
                    if sub.is_dir() and list(sub.glob("*.xf")):
                        name = sub.name
                        if name.endswith("_Imod"):
                            name = name[:-5]
                        aligned.add(name)
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
    def _sort_bucket(name):
        """Which standard folder a file belongs in (or None to leave alone).
        Tomo5 *_override.mdoc are handled separately, not here."""
        low = name.lower()
        if low.endswith(".eer"):
            return "frames"
        if low.endswith(".gain") or ("gain" in low and low.endswith(".mrc")):
            return "gains"
        if low.endswith(".mdoc") and not low.endswith("_override.mdoc"):
            return "mdocs"
        return None

    @staticmethod
    def _override_standard_name(override_name):
        """Standard mdoc name for a Tomo5 override, e.g.
        'Position_1_override.mdoc' -> 'Position_1.mdoc'."""
        return override_name[:-len("_override.mdoc")] + ".mdoc"

    def plan_sort(self, src):
        """Preview of sort_files:
        {frames, mdocs, gains, overrides, skipped} lists of filenames."""
        plan = {"frames": [], "mdocs": [], "gains": [], "overrides": [], "skipped": []}
        src = Path(src)
        if not src.is_dir():
            return plan
        for f in sorted(src.iterdir()):
            if not f.is_file():
                continue
            if f.name.lower().endswith("_override.mdoc"):
                plan["overrides"].append(f.name)
                continue
            bucket = self._sort_bucket(f.name)
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
        mdocs/ (.mdoc), gains/ (gain). Dedups Tomo5 *_override.mdoc first.
        Never overwrites an existing target. Returns a counts dict.

        NOTE: this SPLITS .eer and .mdoc apart, so run it AFTER rename (the
        rename script needs them in one folder)."""
        self.initialize_structure()
        src = Path(src)
        removed, kept = self.clean_override_mdocs(src)
        plan = self.plan_sort(src)                       # re-plan after cleanup
        moved = {"frames": 0, "mdocs": 0, "gains": 0,
                 "overrides_removed": removed, "overrides_kept": kept}
        for bucket in ("frames", "mdocs", "gains"):
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
        """Full-res per-series .mrc to open in 3dmod (renamed first, then Tomo5,
        then the thumbnail montage). Returns a Path or None."""
        for cand in (self.root / f"{renamed}.mrc",
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
        path.write_text(new_manual + (auto if auto.strip() else ""))
        return len(lines)

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
        path.write_text(manual_part + self.AUTO_EXCLUSION_HEADER + new_auto)
        return True

    # ---- project save/load ----------------------------------------------
    def initialize_structure(self):
        for d in EXPECTED_DIRS:
            (self.root / d).mkdir(parents=True, exist_ok=True)

    def save_project(self, settings_dict):
        state_file = self.root / "warp_auto.project.json"
        state_file.write_text(json.dumps({
            "saved_at": datetime.datetime.now().isoformat(),
            "settings": settings_dict,
        }, indent=2))
        return state_file

    def load_project(self):
        state_file = self.root / "warp_auto.project.json"
        return json.loads(state_file.read_text()) if state_file.exists() else None

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
        (self.root / self.GROUPS_FILE).write_text(json.dumps(data, indent=2))

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


