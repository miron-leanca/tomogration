#!/usr/bin/env python3
"""
tomogration — three-panel cryo-ET pipeline controller (PySide6).

Scope: raw data prep -> particle export -> a RELION 4 Class3D handoff. High-
resolution refinement / RELION 5 --tomo / M live in those tools' own pipeliners.

Layout:  [ LEFT docs/theory ] [ CENTER pipeline axis + param form ] [ RIGHT terminal ]

The pipeline is data-driven: add a stage = add a dict to STAGES (see the schema
note above the list). The framework-agnostic backend (ProjectState, mdoc repair,
Positions Inspector, AreTomo versioned-folder + PARAMETERS.txt audit, auto-
recovery) is ported verbatim from the old tkinter warp_auto.py; only the view
layer is Qt. See tomogration_brief.md for the parameter defaults and gotchas.

TARGET: a Linux virtual machine (ceph cluster compute node). The file-manager /
editor affordances assume Linux (xdg-open, gedit). Do NOT expect this to run on
macOS — it is developed there but deployed on Linux.

GPU GUARD (brief §4.1): the perdevice>1 + --deconv SIGABRT was diagnosed on
V100. The guard is kept conservative. If the deploy node (EML45) is NOT on V100,
re-test that combination rather than trusting the guard blindly.

Run (on the Linux VM):   pip install PySide6   &&   python tomogration_app.py
"""

import os
import re
import sys
import json
import shutil
import filecmp
import datetime
import itertools
import subprocess
from pathlib import Path

import struct
import array as _array

from PySide6.QtCore import Qt, QObject, QEvent, Signal, QProcess, QTimer
from PySide6.QtGui import (
    QBrush, QColor, QImage, QPixmap, QTextCursor, QFont, QPen, QPainter,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QPlainTextEdit, QScrollArea, QSlider, QCheckBox,
    QLineEdit, QFileDialog, QDialog, QListWidget, QTreeWidget, QTreeWidgetItem,
    QMessageBox, QHeaderView, QGridLayout, QFrame, QStackedWidget, QComboBox,
    QInputDialog, QListWidgetItem, QTextBrowser,
    QAbstractScrollArea, QAbstractSpinBox, QSpinBox,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsSimpleTextItem,
    QMenu,
)


class WheelGuard(QObject):
    """App-wide event filter: the mouse wheel must NOT change combo boxes, sliders
    or spin boxes. Hovering one while scrolling the centre panel would otherwise
    "steal" the wheel and silently change a value (and the panel scroll gets stuck).
    We eat the wheel on those widgets and resend it to the enclosing scroll area so
    the panel scrolls as expected. An OPEN combo popup is an item view, not a
    QComboBox, so its own list still scrolls normally."""

    # Guard the VALUE widgets only. NOT QAbstractSlider — that also covers QScrollBar
    # (and QDial), and hijacking a scroll bar's wheel breaks normal panel/terminal
    # scrolling (e.g. the terminal's line-based scrollbar jumped dozens of lines).
    _GUARDED = (QComboBox, QSlider, QAbstractSpinBox)

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.Wheel and isinstance(obj, self._GUARDED):
            # Scroll the enclosing scroll area by a fixed, predictable amount rather
            # than re-sending the raw wheel event (re-sending scrolled to the extreme
            # on this box). One wheel notch (angleDelta 120) -> ~50 px.
            w = obj.parentWidget()
            while w is not None and not isinstance(w, QAbstractScrollArea):
                w = w.parentWidget()
            if w is not None:
                delta = ev.angleDelta().y() or ev.angleDelta().x()
                if delta:
                    bar = w.verticalScrollBar()
                    bar.setValue(bar.value() - round(delta / 120.0 * 50))
            return True  # never let the widget itself act on the wheel
        return False


def mrc_to_qimage(path, max_dim=180):
    """Best-effort grayscale render of an .mrc montage/stack to a QImage.
    Stdlib only (no numpy/mrcfile). Handles modes 0/1/2/6, little-endian; takes
    the middle slice of a stack and decimates large images. Returns QImage|None."""
    try:
        with open(path, "rb") as f:
            hdr = f.read(1024)
            if len(hdr) < 1024:
                return None
            nx, ny, nz, mode = struct.unpack("<4i", hdr[:16])
            nsymbt = struct.unpack("<i", hdr[92:96])[0]
            if not (0 < nx <= 20000 and 0 < ny <= 20000):
                return None
            spec = {0: ("b", 1), 1: ("h", 2), 2: ("f", 4), 6: ("H", 2)}.get(mode)
            if spec is None:                       # e.g. float16 (12) — use 3dmod
                return None
            typecode, isize = spec
            nz = max(nz, 1)
            f.seek(1024 + max(nsymbt, 0) + (nz // 2) * nx * ny * isize)
            raw = f.read(nx * ny * isize)
        if len(raw) < nx * ny * isize:
            return None
        arr = _array.array(typecode)
        arr.frombytes(raw)
        lo, hi = min(arr), max(arr)
        scale = 255.0 / ((hi - lo) or 1)
        step = max(1, int(max(nx, ny) / max_dim))   # decimate to ~max_dim
        ow = len(range(0, nx, step))
        oh = len(range(0, ny, step))
        buf = bytearray(ow * oh)
        idx = 0
        for yy in range(0, ny, step):
            base = yy * nx
            for xx in range(0, nx, step):
                v = int((arr[base + xx] - lo) * scale)
                buf[idx] = 0 if v < 0 else 255 if v > 255 else v
                idx += 1
        return QImage(bytes(buf), ow, oh, ow, QImage.Format_Grayscale8).copy()
    except (OSError, struct.error, ValueError, OverflowError):
        return None


# Companion scripts ship next to this file. Resolve them to ABSOLUTE paths so
# stage commands work regardless of which project root the user selects at
# startup (commands run with cwd = the selected data dir, not the app dir).
_PKG_DIR = Path(__file__).resolve().parent


def _pkg_script(name):
    """Absolute path to a shipped companion script, bash-quoted if it has spaces."""
    p = str(_PKG_DIR / name)
    return f'"{p}"' if (" " in p or "\t" in p) else p


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
        n = self.count_rglob("relion4", "*.star")
        return (n, f"{n} .star in relion4/" if n else None)

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


# ===========================================================================
# Stage definitions  (declarative — extend the pipeline by adding a dict)
#
# Schema (copy ts_reconstruct as the template):
#   group/id/label : center-axis grouping + node identity
#   base           : command head. Empty for env-only/positional commands.
#   params[]:
#     name         : control label + key
#     kind         : "text" | "slider_int" | "check" | "env" | "env_int"
#     flag         : CLI flag (e.g. "--perdevice"). None = positional value.
#                    For env/env_int kinds, flag is the VAR name (VAR=value).
#     default/min/max/step  : widget config
#     help         : inline description + range + effect (shown under control)
#   validate(values)->str   : optional live red warning under the form
#   docs           : what / range / effect / pitfall  (LEFT panel)
#   status(ps)->(truthy,label) : drives the node status dot via file existence
#   aretomo / auto_recover / sync_helper : special-behaviour flags
#
# Param-widget honesty (brief §5): sliders only for bounded numerics (B-factor,
# threshold, binning, perdevice); checkboxes for boolean flags; everything else
# (paths, patterns, grids like 2x2x1, env 0/1 toggles) stays free-text. The
# editable command box at the bottom of each form is the single source of truth.
# ===========================================================================
STAGES = [
    # ---------------- 1. Data prep ----------------
    {
        "group": "1. Data prep", "id": "rename", "label": "Rename .eer/.mdoc",
        "base": f'bash {_pkg_script("ml_batch_rename_eer_mdoc_mrc_warp_auto.sh")}',
        "clean_overrides": True,
        "params": [
            {"name": "source_dir", "kind": "text", "flag": None,
             "default": ".",
             "help": "Folder with the raw Tomo5 .eer + .mdoc (+ .mrc) TOGETHER, "
                     "relative to the project root. '.' = the project root itself "
                     "(use this when the root IS your acquisition folder). Run this "
                     "BEFORE Sort files — rename needs .eer and .mdoc in one folder."},
            {"name": "rootname", "kind": "text", "flag": None,
             "default": "Position", "help": "Prefix for renamed files (e.g. Position, TS)."},
            {"name": "start_number", "kind": "text", "flag": None,
             "default": "0", "help": "Starting counter (0 -> first output is 001)."},
        ],
        "docs": {
            "what": "Renames Tomo5 beam-shift names (Position_1_2) to Warp form "
                    "(Position001). Writes listfile_<root>.txt mapping old->new "
                    "and fixes mdoc dates to yy-mmm-dd (required by Warp).",
            "range": "n/a",
            "effect": "listfile is required later to translate IMOD-order exclusion tables.",
            "pitfall": "ORDER: rename FIRST (on the raw folder, .eer+.mdoc together), "
                       "THEN Sort files to split into frames/ + mdocs/. Renames IN "
                       "PLACE — work on a copy. Identical Tomo5 *_override.mdoc are "
                       "auto-moved to mdocs/bad/ before renaming (they would consume "
                       "position numbers). No 'selected/' folder is needed.",
        },
        "status": lambda ps: ps.status_listfile(),
    },
    {
        "group": "1. Data prep", "id": "imod_warp_key", "label": "IMOD→Warp key",
        "base": f'python {_pkg_script("ml_imodtowarpkey_generator_warp_auto.py")}',
        "params": [
            {"name": "input_mdoc", "kind": "text", "flag": None,
             "default": "mdocs/Position001.mdoc",
             "help": "A COMPLETE reference mdoc (all tilts present)."},
            {"name": "output_key", "kind": "text", "flag": None,
             "default": "new_imod_conv_key.txt",
             "help": "Where to write the IMOD->acquisition-order key."},
        ],
        "docs": {
            "what": "Builds the IMOD->Warp tilt-number key. mdoc ZValue blocks "
                    "are in dose-symmetric acquisition order; 3dmod shows tilts "
                    "in angle order. The key translates between them.",
            "range": "n/a",
            "effect": "Used by remake_mdocs to map IMOD-order exclusions to acq order.",
            "pitfall": "Use a mdoc with EVERY tilt present, or the mapping is wrong.",
        },
        "status": lambda ps: ps.status_conv_key(),
    },
    {
        "group": "1. Data prep", "id": "inspect_select",
        "label": "Inspect tilt stacks",
        "tool": "inspector",
        "docs": {
            "what": "Visually review each tilt series (thumbnail grid + 3dmod) and "
                    "mark bad tilts (IMOD #, ranges OK e.g. 1,4-12,48) or whole "
                    "series for exclusion before remaking the mdocs.",
            "range": "n/a",
            "effect": "Tilt exclusions feed exclusion_list.txt (Remake mdocs applies "
                      "them); excluded whole series move to mdocs/bad/ + frames/bad/. "
                      "You can also hand-edit exclusion_list.txt: 'PositionNNN<tab>1,4-12' "
                      "trims those tilts; a bare 'PositionNNN' line drops the whole series.",
            "pitfall": "Do this BEFORE Remake mdocs. Already-excluded series are "
                       "flagged ✗EXCLUDED. Tip: 'Open ALL in 3dmod + list' opens "
                       "every series in one 3dmod next to a compact type-in list.",
        },
        "status": lambda ps: ps.status_exclusions(),
    },
    {
        "group": "1. Data prep", "id": "remake_mdocs", "label": "Remake mdocs",
        "base": f'bash {_pkg_script("ml_batch_remake_mdocs_warp_auto.sh")}',
        # The script cd's into mdocs/ then reads these by name, so pass them
        # absolute or they're looked up inside mdocs/ and not found.
        "abs_paths": ["mdocs_dir", "exclusion_list", "conv_key"],
        # Expand tilt ranges (1-3) in exclusion_list.txt first — the script
        # can't parse ranges (sed chokes on "1-3p").
        "normalize_exclusions": True,
        "params": [
            {"name": "mdocs_dir", "kind": "text", "flag": None,
             "default": "mdocs", "help": "Dir with <root>NNN.mdoc files."},
            {"name": "exclusion_list", "kind": "text", "flag": None,
             "default": "exclusion_list.txt",
             "help": "Manual tilt exclusions (IMOD order), above the auto header."},
            {"name": "conv_key", "kind": "text", "flag": None,
             "default": "new_imod_conv_key.txt", "help": "IMOD->acq key file."},
            {"name": "rootname", "kind": "text", "flag": None,
             "default": "Position", "help": "File prefix (optional)."},
        ],
        "docs": {
            "what": "Removes excluded tilts from mdocs and renumbers remaining "
                    "ZValue blocks contiguously, then fixes mdoc date format.",
            "range": "n/a",
            "effect": "Quarantining a tilt without fixing the mdoc breaks ts_import.",
            "pitfall": "exclusion_list.txt: manual tilt numbers go ABOVE the "
                       "auto-excluded header; the script stops parsing at it. A bare "
                       "'PositionNNN' line (no tilts) is applied on enqueue by "
                       "quarantining that whole series (mdoc/.eer → bad/). "
                       "For files already moved to frames/bad/, use 'Repair mdocs'.",
        },
        "status": lambda ps: ps.status_mdocs(),
    },

    # ---------------- 2. Gain ----------------
    {
        "group": "2. Gain", "id": "gain_convert", "label": "gain .gain→.mrc",
        "base": "module load eman && e2proc3d.py",
        "params": [
            {"name": "in_gain", "kind": "text", "flag": None,
             "default": "gains/original.gain", "help": "Input .gain reference."},
            {"name": "out_mrc", "kind": "text", "flag": None,
             "default": "gains/original_gain.mrc", "help": "Output .mrc gain."},
        ],
        "docs": {
            "what": "Converts a camera .gain reference to .mrc via EMAN2.",
            "range": "n/a",
            "effect": "Produces the .mrc that the reciprocal step inverts.",
            "pitfall": "Needs `module load eman`. This is the NON-reciprocal gain.",
        },
        "status": lambda ps: ps.status_gain_original(),
    },
    {
        "group": "2. Gain", "id": "gain_reciprocal", "label": "reciprocal gain",
        "base": "module load eman && e2proc2d.py",
        "params": [
            {"name": "in_mrc", "kind": "text", "flag": None,
             "default": "gains/original_gain.mrc", "help": "Input .mrc gain."},
            {"name": "out_reciprocal", "kind": "text", "flag": None,
             "default": "gains/gain_reciprocal.mrc", "help": "Output reciprocal gain."},
            {"name": "reciprocal", "kind": "check", "flag": "--process math.reciprocal",
             "default": True, "help": "Take the per-pixel reciprocal."},
        ],
        "docs": {
            "what": "Computes the reciprocal gain that Linux WarpTools expects.",
            "range": "n/a",
            "effect": "create_settings --gain_path must point at THIS file.",
            "pitfall": "Linux WarpTools wants the reciprocal; using the plain gain "
                       "double-applies the correction.",
        },
        "status": lambda ps: ps.status_gain_reciprocal(),
    },

    # ---------------- 3. Frameseries ----------------
    {
        "group": "3. Frameseries", "id": "create_settings_fs",
        "label": "create_settings (fs)",
        "base": "WarpTools create_settings",
        "params": [
            {"name": "folder_data", "kind": "text", "flag": "--folder_data",
             "default": "frames", "help": "Raw .eer folder."},
            {"name": "folder_processing", "kind": "text", "flag": "--folder_processing",
             "default": "warp_frameseries", "help": "Processing output folder."},
            {"name": "output", "kind": "text", "flag": "--output",
             "default": "warp_frameseries.settings", "help": "Settings file to write."},
            {"name": "extension", "kind": "text", "flag": "--extension",
             "default": "*.eer", "help": "Input file glob."},
            {"name": "angpix", "kind": "text", "flag": "--angpix",
             "default": "1.57", "help": "Pixel size (Å/px). This dataset: 1.57."},
            {"name": "gain_path", "kind": "text", "flag": "--gain_path",
             "default": "gains/gain_reciprocal.mrc",
             "help": "RECIPROCAL gain (Linux WarpTools)."},
            {"name": "exposure", "kind": "text", "flag": "--exposure",
             "default": "3.5", "help": "Dose per TILT (e/Å², not per frame)."},
            {"name": "eer_ngroups", "kind": "text", "flag": "--eer_ngroups",
             "default": "10", "help": "EER frame groups."},
        ],
        "docs": {
            "what": "Writes the Warp frame-series .settings file.",
            "range": "apix 1.57; dose 3.5 e/Å²/tilt; eer_ngroups 10.",
            "effect": "Every downstream fs_* step reads these settings.",
            "pitfall": "Point gain_path at the RECIPROCAL gain; dose is per tilt.",
        },
        "status": lambda ps: ps.status_fs_settings(),
    },
    {
        "group": "3. Frameseries", "id": "fs_motion_and_ctf",
        "label": "fs_motion_and_ctf",
        "base": "WarpTools fs_motion_and_ctf",
        "auto_recover": True,
        "group_scope": "fs",
        "params": [
            {"name": "settings", "kind": "text", "flag": "--settings",
             "default": "warp_frameseries.settings", "help": "fs .settings file."},
            {"name": "m_grid", "kind": "text", "flag": "--m_grid",
             "default": "1x1x3", "help": "Motion grid XxYxT; temporal ≈ frame count."},
            {"name": "c_grid", "kind": "text", "flag": "--c_grid",
             "default": "2x2x1", "help": "CTF grid XxYxT (not slider-able)."},
            {"name": "m_range_min", "kind": "text", "flag": "--m_range_min",
             "default": "500", "help": "Motion fit low-res bound (Å)."},
            {"name": "m_range_max", "kind": "text", "flag": "--m_range_max",
             "default": "10", "help": "Motion fit high-res bound (Å)."},
            {"name": "m_bfac", "kind": "slider_int", "flag": "--m_bfac",
             "default": -500, "min": -1000, "max": 0, "step": 50,
             "help": "Motion B-factor; more negative = stronger low-pass."},
            {"name": "c_range_max", "kind": "text", "flag": "--c_range_max",
             "default": "7", "help": "CTF fit max resolution (Å)."},
            {"name": "c_defocus_max", "kind": "text", "flag": "--c_defocus_max",
             "default": "8", "help": "Max defocus to search (µm)."},
            {"name": "out_averages", "kind": "check", "flag": "--out_averages",
             "default": True, "help": "Write aligned averages. REQUIRED — ts_import "
             "needs them ('no aligned average result' error if off)."},
            {"name": "out_average_halves", "kind": "check", "flag": "--out_average_halves",
             "default": True, "help": "Write odd/even half-averages (for Noise2Noise "
             "denoising)."},
            {"name": "device_list", "kind": "text", "flag": "--device_list", "gpu_sep": " ",
             "default": "0", "help": "GPU id(s), e.g. 0 or '0 1'."},
            {"name": "perdevice", "kind": "slider_int", "flag": "--perdevice",
             "default": 2, "min": 1, "max": 4, "step": 1,
             "help": "Workers per GPU (no deconv here, so 2 is fine)."},
        ],
        "docs": {
            "what": "Per-frame-series motion correction + CTF estimation.",
            "range": "motion grid 1x1x3; CTF grid 2x2x1; m_range 500→10 Å; m_bfac −500.",
            "effect": "Finer grids model more local motion/CTF, at higher cost.",
            "pitfall": "Auto-recovery is ON: a cuFFT crash on a bad .eer is "
                       "quarantined to frames/bad/, its mdoc ZValue block removed, "
                       "logged to exclusion_list.txt, and the run retried. Only "
                       "fs_* recovers — ts_* crashes are GPU/resource, not bad files.",
        },
        "status": lambda ps: ps.status_fs_motion_ctf(),
    },

    # ---------------- 4. Tilt series ----------------
    {
        "group": "4. Tilt series", "id": "create_settings_ts",
        "label": "create_settings (ts)",
        "base": "WarpTools create_settings",
        "params": [
            {"name": "folder_data", "kind": "text", "flag": "--folder_data",
             "default": "tomostar", "help": "tomostar folder."},
            {"name": "folder_processing", "kind": "text", "flag": "--folder_processing",
             "default": "warp_tiltseries", "help": "Processing output folder."},
            {"name": "output", "kind": "text", "flag": "--output",
             "default": "warp_tiltseries.settings", "help": "Settings file to write."},
            {"name": "extension", "kind": "text", "flag": "--extension",
             "default": "*.tomostar", "help": "Input glob."},
            {"name": "angpix", "kind": "text", "flag": "--angpix",
             "default": "1.57", "help": "Pixel size (Å/px)."},
            {"name": "gain_path", "kind": "text", "flag": "--gain_path",
             "default": "gains/gain_reciprocal.mrc", "help": "Reciprocal gain."},
            {"name": "exposure", "kind": "text", "flag": "--exposure",
             "default": "3.5", "help": "Dose per tilt (e/Å²)."},
            {"name": "tomo_dimensions", "kind": "text", "flag": "--tomo_dimensions",
             "default": "4096x4096x3088", "help": "Tomogram dims XxYxZ (unbinned)."},
        ],
        "docs": {
            "what": "Writes the Warp tilt-series .settings file.",
            "range": "tomo_dimensions XxYxZ; apix 1.57.",
            "effect": "Every ts_* step reads these settings.",
            "pitfall": "Z dimension must exceed the lamella thickness.",
        },
        "status": lambda ps: ps.status_ts_settings(),
    },
    {
        "group": "4. Tilt series", "id": "ts_import", "label": "ts_import",
        "base": "WarpTools ts_import",
        "params": [
            {"name": "mdocs", "kind": "text", "flag": "--mdocs",
             "default": "mdocs", "help": "Mdocs folder."},
            {"name": "frameseries", "kind": "text", "flag": "--frameseries",
             "default": "warp_frameseries", "help": "Frameseries processing folder."},
            {"name": "tilt_exposure", "kind": "text", "flag": "--tilt_exposure",
             "default": "3.5", "help": "Dose per tilt (e/Å²)."},
            {"name": "min_intensity", "kind": "text", "flag": "--min_intensity",
             "default": "0", "help": "Min intensity filter."},
            {"name": "dont_invert", "kind": "check", "flag": "--dont_invert",
             "default": True, "help": "Keep tilt polarity as-is (dataset-specific)."},
            {"name": "output", "kind": "text", "flag": "--output",
             "default": "tomostar", "help": "tomostar output folder."},
            {"name": "override_axis", "kind": "text", "flag": "--override_axis",
             "default": "", "help": "Tilt-AXIS rotation (deg) — the IN-PLANE angle of the "
             "tilt axis, NOT the stage tilt range. Blank = use the mdoc value (fine: it's "
             "only the STARTING guess; AreTomo then searches & refines it). Set a number "
             "only if you know the correct axis (e.g. from AreTomo's solved .aln)."},
        ],
        "docs": {
            "what": "Builds .tomostar files by pairing mdocs with frameseries.",
            "range": "n/a",
            "effect": "tomostar is the unit AreTomo and ts_* operate on.",
            "pitfall": "Fails with 'failed to parse specific tilts' when a "
                       "quarantined .eer left a stale mdoc ZValue block — run "
                       "'Repair mdocs' first (brief gotcha §4.3). The 'tilt axis angle … "
                       "Tomo5 mdoc files are known to provide incorrect values' message is "
                       "ADVISORY — printed for every Tomo5 mdoc, not a detected error; the "
                       "axis is the in-plane rotation (e.g. ~-174°), and AreTomo refines it, "
                       "so blank is usually right. Override only with a known-good value.",
        },
        "status": lambda ps: ps.status_tomostar(),
    },
    {
        "group": "4. Tilt series", "id": "ts_stack", "label": "ts_stack",
        "base": "WarpTools ts_stack",
        "group_scope": "ts",
        "params": [
            {"name": "settings", "kind": "text", "flag": "--settings",
             "default": "warp_tiltseries.settings", "help": "ts .settings file."},
            {"name": "angpix", "kind": "text", "flag": "--angpix",
             "default": "", "help": "Output pixel size (Å/px). Blank = native."},
            {"name": "device_list", "kind": "text", "flag": "--device_list", "gpu_sep": " ",
             "default": "0", "help": "GPU id(s)."},
            {"name": "perdevice", "kind": "slider_int", "flag": "--perdevice",
             "default": 2, "min": 1, "max": 4, "step": 1, "help": "Workers per GPU."},
        ],
        "docs": {
            "what": "Builds aligned tilt stacks (.st) per tilt series for AreTomo.",
            "range": "n/a",
            "effect": "Produces warp_tiltseries/tiltstack/<Position>/<Position>.st.",
            "pitfall": "Run ts_import first; a missing mdoc entry stalls a stack.",
        },
        "status": lambda ps: ps.status_ts_stacks(),
    },

    # ---------------- 5. Alignment ----------------
    {
        "group": "5. Alignment", "id": "aretomo", "label": "Align with AreTomo2",
        "base": "bash",
        "aretomo": True,
        "params": [
            {"name": "script", "kind": "text", "flag": None,
             "default": _pkg_script("ml_aretomo2_warp_auto.sh"),
             "help": "AreTomo2 wrapper script (shipped with the app)."},
            {"name": "input_dir", "kind": "text", "flag": None,
             "default": "warp_tiltseries/tiltstack", "help": "Folder of <Pos>/<Pos>.st."},
            {"name": "output_dir", "kind": "text", "flag": None,
             "default": "aretomo_output",
             "help": "Versioned output folder (auto-bumped to -vN; PARAMETERS.txt written here)."},
            {"name": "gpu", "kind": "text", "flag": None,
             "default": "0", "help": "Fallback SINGLE GPU id, used only if the 'GPUs' "
             "list below is left blank. A single id here = sequential on one GPU. To "
             "use several GPUs, fill the GPUs field instead (do NOT put '0 1 2 3' here — "
             "extra tokens here shift the positional args and corrupt angpix)."},
            {"name": "angpix", "kind": "text", "flag": None,
             "default": "1.57", "help": "Input pixel size (Å/px). 1.57 for this data."},
            {"name": "ARETOMO_GPUS", "kind": "env", "flag": "ARETOMO_GPUS", "gpu_sep": " ",
             "default": "0 1 2 3",
             "help": "GPUs to spread tilt series across (space- or comma-separated). Each "
             "series runs on ONE GPU; with N GPUs, N series align at once (~N× faster). "
             "Blank = use the single 'gpu' field above (sequential)."},
            {"name": "ARETOMO_JOBS_PER_GPU", "kind": "env_int", "flag": "ARETOMO_JOBS_PER_GPU",
             "default": 1, "min": 1, "max": 4, "step": 1,
             "help": "Concurrent AreTomo jobs PER GPU. 1 is safe; a 32 GB V100 can usually "
             "fit 2 at bin 8. Total concurrency = (#GPUs) × this."},
            {"name": "ARETOMO_ALIGNZ", "kind": "env", "flag": "ARETOMO_ALIGNZ",
             "default": "670", "help": "Alignment Z (unbinned px) ≈ lamella thickness."},
            {"name": "ARETOMO_VOLZ", "kind": "env", "flag": "ARETOMO_VOLZ",
             "default": "3088", "help": "Output Z height; must exceed lamella thickness. "
             "Set 0 for ALIGNMENT-ONLY (skips the slow tomogram, still writes the .xf you "
             "import) — the fast way to align a whole dataset for ts_import / miss-alignment."},
            {"name": "ARETOMO_OUTBIN", "kind": "env_int", "flag": "ARETOMO_OUTBIN",
             "default": 8, "min": 1, "max": 16, "step": 1,
             "help": "Output binning. 8 → 12.56 Å/px at 1.57 input."},
            {"name": "ARETOMO_DARKTOL", "kind": "env", "flag": "ARETOMO_DARKTOL",
             "default": "0.000001", "help": "Dark-frame tol; ~0 disables (pre-curated tilts)."},
            {"name": "ARETOMO_TILTCOR", "kind": "env", "flag": "ARETOMO_TILTCOR",
             "default": "0", "help": "Tilt-offset correction 0/1. Usually 0 for lamellae."},
            {"name": "ARETOMO_FLIPVOLZ", "kind": "env", "flag": "ARETOMO_FLIPVOLZ",
             "default": "1", "help": "Flip handedness for Warp 0/1. Usually 1."},
            {"name": "ARETOMO_WBP", "kind": "env", "flag": "ARETOMO_WBP",
             "default": "1", "help": "Weighted back projection 0/1."},
            {"name": "ARETOMO_TILTAXIS", "kind": "env", "flag": "ARETOMO_TILTAXIS",
             "default": "", "help": "Tilt-axis (deg). Blank = AreTomo searches."},
            {"name": "ARETOMO_PATCH", "kind": "env", "flag": "ARETOMO_PATCH",
             "default": "", "help": "Patch align e.g. '4 4'. Blank = skip patch tracking."},
            {"name": "ARETOMO_ALIGN", "kind": "env", "flag": "ARETOMO_ALIGN",
             "default": "1", "help": "1 = align+recon, 0 = reconstruct only."},
            {"name": "ARETOMO_BIN", "kind": "env", "flag": "ARETOMO_BIN",
             "default": "/ceph/groups/structbio/Programs/AreTomo2/AreTomo2",
             "help": "AreTomo2 executable path."},
            {"name": "ARETOMO_CUDA_LIB", "kind": "env", "flag": "ARETOMO_CUDA_LIB",
             "default": "",
             "help": "Optional. Dir holding libcufft.so.11 + the CUDA-12 runtime. Normally "
             "BLANK — the wrapper uses the cuda module's libs (via a stub-free symlink "
             "farm so the real driver is found). Set it only to override with your own "
             "CUDA-12 runtime, e.g. /ceph/users/<you>/.conda/envs/cuda12rt/lib."},
        ],
        "docs": {
            "what": "Marker-free tilt-series alignment (+optional recon). Writes "
                    ".xf/.tlt per series under <output>/Imod/.",
            "range": "ALIGNZ ≈ lamella thickness; OUTBIN 8 → 12.56 Å/px; patch 4×4–6×6.",
            "effect": "Patch tracking improves local alignment but is slower. GPUs runs "
                      "tilt series in parallel (one per GPU); '0 1 2 3' is ~4× faster than one.",
            "pitfall": "Each run writes a NEW versioned folder (aretomo_output, "
                       "-v2, …) with a PARAMETERS.txt audit. IMOD can't read Warp "
                       "float16 MRC — export WARP_FORCE_MRC_FLOAT32=1 before 3dmod "
                       "(brief gotcha §4.2).",
        },
        "status": lambda ps: ps.status_aretomo_xf(),
    },
    {
        # miss-alignment TRAIN: train a model on THIS set (which must already hold a
        # coarse AreTomo alignment) and refine it. Enforced order: AreTomo ->
        # ts_import_alignments -> select -> THIS -> ts_ctf. On a fresh run the GUI
        # auto-prepends the AreTomo import + select; on raw stacks it's blocked outright.
        "group": "5. Alignment", "id": "miss_align",
        "label": "miss-alignment (train)",
        "base": "bash",
        "requires_coarse_alignment": True,
        "fixed_env": {"MA_MODE": "train"},
        "params": [
            {"name": "script", "kind": "text", "flag": None,
             "default": _pkg_script("ml_missalignment_warp_auto.sh"),
             "help": "miss-alignment wrapper script (shipped with the app)."},
            {"name": "config", "kind": "text", "flag": None,
             "default": "missalignment_config.yaml",
             "help": "TRAIN YAML config (relative to root). If missing, the wrapper seeds "
             "a training template and stops so you can review it, then re-run."},
            {"name": "input_dir", "kind": "text", "flag": None,
             "default": "warp_tiltseries",
             "help": "Warp tilt-series dir with <series>.xml + tiltstack/<series>/"
             "<series>.st (run ts_import + ts_stack first, same prereqs as AreTomo)."},
            {"name": "MA_TRAINING_DEVICES", "kind": "env", "flag": "MA_TRAINING_DEVICES",
             "default": "0", "help": "--training-devices. KEEP THIS A SINGLE GPU (e.g. "
             "'0'). >1 makes torch spawn one trainer per GPU and they race to wipe the "
             "shared pool dir → FileNotFoundError on a partition_*.pickle. Scale speed "
             "with RECON devices + dataloaders instead."},
            {"name": "MA_RECON_DEVICES", "kind": "env", "flag": "MA_RECON_DEVICES", "gpu_sep": ",",
             "default": "0,0,0", "help": "--reconstruction-devices: this is where you add "
             "GPUs for speed (recon feeds the pool and is the bottleneck). e.g. '0,1,2,3' "
             "or repeat an id to stack workers on it ('0,0,0')."},
            {"name": "MA_DATALOADERS", "kind": "env_int", "flag": "MA_DATALOADERS",
             "default": 5, "min": 1, "max": 16, "step": 1,
             "help": "--dataloaders-per-trainer. The recon pool is split into "
             "(training_devices × this) partitions, each needing ≥ 2×batch_size; "
             "if it errors, raise Pool size or lower this."},
            {"name": "MA_POOL_SIZE", "kind": "env_int", "flag": "MA_POOL_SIZE",
             "default": 2000, "min": 500, "max": 8000, "step": 100,
             "help": "--pool-size: subtomogram reconstructions cached in the temp "
             "pool. Must be ≥ 2×batch_size×training_devices×dataloaders (2000 keeps "
             "4 GPU × 5 loaders × batch 32 valid). Type the exact number."},
            {"name": "MA_START_ITER", "kind": "env_int", "flag": "MA_START_ITER",
             "default": 0, "min": 0, "max": 20, "step": 1,
             "help": "--start-at-iteration (resume from the HIGHEST existing iterN)."},
            {"name": "MA_PREPARE_STACKS", "kind": "env", "flag": "MA_PREPARE_STACKS",
             "default": "10.0", "help": "--prepare-stacks pixel size (Å/px) for the "
             "reconstruction patches. Blank = skip stack preparation."},
            {"name": "MA_CONDA_ENV", "kind": "env", "flag": "MA_CONDA_ENV",
             "default": "miss-alignment",
             "help": "conda env that has miss-alignment installed (its own CUDA 12.9 / "
             "torch stack — NOT the warp env)."},
        ],
        "docs": {
            "what": "Deep-learning REFINEMENT of an EXISTING alignment by TRAINING a model "
                    "on this set (warpem/miss-alignment). Does NOT align raw stacks — the "
                    "docs state 'miss-alignment starts from an initially coarse aligned "
                    "dataset', so coarse-align FIRST (AreTomo → ts_import_alignments).",
            "range": "training/recon devices, prepare-stacks Å/px, start-iteration.",
            "effect": "Trains a 3D CNN to score reconstruction quality, then gradient-"
                      "optimises the shifts against it, writing the refined alignment back "
                      "into the .xml. The trained iterN/model.ckpt can then be REUSED via "
                      "the 'miss-alignment (infer)' step on a larger set.",
            "pitfall": "MUST have a coarse prior alignment first — on RAW stacks it yields "
                       "a featureless tomogram (the GUI blocks this). Single TRAINING GPU "
                       "only. First run seeds a config and STOPS for review. Re-train if the "
                       "prior alignment changed.",
        },
        "status": None,
    },
    {
        # miss-alignment INFER: REUSE a finished model to align a new/larger set WITHOUT
        # training. No coarse-align auto-chain (you coarse-align the big set yourself and
        # deselect the unaligned); needs MA_MODEL_RUN_DIR = the training run's iterN dir.
        "group": "5. Alignment", "id": "miss_align_infer",
        "label": "miss-alignment (infer — reuse model)",
        "base": "bash",
        "fixed_env": {"MA_MODE": "infer"},
        "params": [
            {"name": "script", "kind": "text", "flag": None,
             "default": _pkg_script("ml_missalignment_warp_auto.sh"),
             "help": "miss-alignment wrapper script (shipped with the app)."},
            {"name": "config", "kind": "text", "flag": None,
             "default": "missalignment_infer_config.yaml",
             "help": "INFER YAML config (relative to root). If missing, the wrapper seeds "
             "an inference template (data_directory + model_run_directory) and stops for "
             "review. iteration_settings MUST match the training run."},
            {"name": "input_dir", "kind": "text", "flag": None,
             "default": "warp_tiltseries",
             "help": "This dataset's warp_tiltseries — already coarse-aligned + imported, "
             "with the unaligned series deselected."},
            {"name": "MA_MODEL_RUN_DIR", "kind": "env", "flag": "MA_MODEL_RUN_DIR",
             "default": "", "help": "REQUIRED: the finished TRAINING run dir holding "
             "iter1/model.ckpt … iterN/model.ckpt (e.g. <selected>/warp_tiltseries)."},
            {"name": "MA_INFER_DEVICES", "kind": "env", "flag": "MA_INFER_DEVICES", "gpu_sep": ",",
             "default": "0,1,2,3", "help": "GPUs for alignment (CUDA_VISIBLE_DEVICES). "
             "Inference has no training race, so use all the idle cards (check util%)."},
            {"name": "MA_START_ITER", "kind": "env_int", "flag": "MA_START_ITER",
             "default": 0, "min": 0, "max": 20, "step": 1,
             "help": "--start-at-iteration (resume inference from iteration N)."},
            {"name": "MA_PREPARE_STACKS", "kind": "env", "flag": "MA_PREPARE_STACKS",
             "default": "10.0", "help": "--prepare-stacks pixel size (Å/px). MUST equal the "
             "resolution you TRAINED at (e.g. 12.56) — the model only works at its scale."},
            {"name": "MA_CONDA_ENV", "kind": "env", "flag": "MA_CONDA_ENV",
             "default": "miss-alignment",
             "help": "conda env with miss-alignment installed."},
        ],
        "docs": {
            "what": "Reuse a model trained by 'miss-alignment (train)' to align THIS "
                    "(usually larger) dataset with NO retraining — runs 'miss-alignment "
                    "infer', loading iterN/model.ckpt for each iteration.",
            "range": "model_run_directory, infer devices, prepare-stacks Å/px.",
            "effect": "Applies the trained models to refine the coarse alignment already "
                      "in this set's .xml. Much faster than training. Alignment uses all "
                      "visible GPUs (no training-worker race).",
            "pitfall": "This set must ALREADY be coarse-aligned (AreTomo → import → "
                       "deselect unaligned) — infer refines, it does not align from scratch. "
                       "prepare-stacks and the config's iteration_settings MUST match the "
                       "training run, and length ≤ number of iterN/model.ckpt.",
        },
        "status": None,
    },
    {
        "group": "5. Alignment", "id": "ts_import_alignments",
        "label": "ts_import_alignments",
        "base": "WarpTools ts_import_alignments",
        "params": [
            {"name": "settings", "kind": "text", "flag": "--settings",
             "default": "warp_tiltseries.settings", "help": "ts .settings file."},
            {"name": "alignments", "kind": "text", "flag": "--alignments",
             "default": "aretomo_output/Imod/", "help": "AreTomo Imod/ folder."},
            {"name": "alignment_angpix", "kind": "text", "flag": "--alignment_angpix",
             "default": "1.57", "help": "Pixel size AreTomo aligned at (1.57)."},
        ],
        "docs": {
            "what": "Imports AreTomo .xf/.tlt alignments back into Warp.",
            "range": "n/a",
            "effect": "ts_ctf / ts_reconstruct use these alignments.",
            "pitfall": "alignment_angpix must match the AreTomo INPUT pixel size, "
                       "not the binned output (1.57 here).",
        },
        "status": lambda ps: ps.status_alignments_imported(),
    },
    {
        "group": "5. Alignment", "id": "sync_selection",
        "label": "sync selection ↔ alignments",
        "base": "WarpTools change_selection",
        "sync_helper": True,
        "params": [
            {"name": "settings", "kind": "text", "flag": "--settings",
             "default": "warp_tiltseries.settings", "help": "ts .settings file."},
            {"name": "mode", "kind": "choice", "flag": None,
             "choices": [
                 ("Deselect", "--deselect"),
                 ("Select (re-enable)", "--select"),
                 ("Null (reset to unset)", "--null"),
                 ("Invert", "--invert"),
             ],
             "default": "--deselect",
             "help": "WarpTools accepts EXACTLY ONE. Deselect = drop the listed series; "
                     "Select = re-enable them. Without --input_data it applies to ALL."},
            {"name": "input_data", "kind": "text", "flag": "--input_data",
             "default": "", "help": "One tomostar to (de)select. Use the button to "
             "fill a chained command for ALL unaligned tomostars. Blank = all series."},
        ],
        "docs": {
            "what": "Deselects tilt series with no AreTomo alignment so "
                    "ts_reconstruct won't crash trying to reconstruct them.",
            "range": "n/a",
            "effect": "Reconstruct only operates on selected, aligned series.",
            "pitfall": "Reversible — re-run with mode 'Select' to re-enable. WarpTools "
                       "errors ('Choose exactly 1 of the options') if no mode is given. "
                       "The dot is green only when every tomostar is aligned.",
        },
        "status": lambda ps: ps.status_selection_sync(),
    },

    # ---------------- 6. CTF ----------------
    {
        "group": "6. CTF", "id": "ts_defocus_hand", "label": "ts_defocus_hand",
        "base": "WarpTools ts_defocus_hand",
        "params": [
            {"name": "settings", "kind": "text", "flag": "--settings",
             "default": "warp_tiltseries.settings", "help": "ts .settings file."},
            {"name": "mode", "kind": "choice", "flag": None,
             "choices": [
                 ("Check handedness only (no change)", "--check"),
                 ("Set: flip", "--set_flip"),
                 ("Set: no-flip", "--set_noflip"),
                 ("Set: auto (apply the checked result)", "--set_auto"),
                 ("Set: switch (toggle current)", "--set_switch"),
             ],
             "default": "--check",
             "help": "WarpTools accepts EXACTLY ONE mode. Run 'Check' first; if it "
                     "reports a negative average correlation, switch to 'Set: flip' "
                     "(or 'Set: auto') and Run again."},
        ],
        "validate": lambda v: (
            "⚠ Run 'Check handedness only' first; pick a Set option only after it "
            "reports a negative correlation."
            if v.get("mode") not in (None, "--check") else ""),
        "docs": {
            "what": "Checks (and optionally flips) defocus handedness. Exactly one "
                    "mode runs per invocation.",
            "range": "n/a",
            "effect": "Wrong handedness inverts the CTF and ruins refinement.",
            "pitfall": "Check first; only set flip/auto on a confirmed negative "
                       "correlation. Passing --check together with a --set_ option "
                       "errors ('Choose exactly 1 of the options').",
        },
        "status": None,  # no distinct file output
    },
    {
        "group": "6. CTF", "id": "ts_ctf", "label": "ts_ctf",
        "base": "WarpTools ts_ctf",
        "group_scope": "ts",
        "params": [
            {"name": "settings", "kind": "text", "flag": "--settings",
             "default": "warp_tiltseries.settings", "help": "ts .settings file."},
            {"name": "range_high", "kind": "text", "flag": "--range_high",
             "default": "7", "help": "CTF fit max resolution (Å)."},
            {"name": "defocus_max", "kind": "text", "flag": "--defocus_max",
             "default": "8", "help": "Max defocus to search (µm)."},
            {"name": "device_list", "kind": "text", "flag": "--device_list", "gpu_sep": " ",
             "default": "0", "help": "GPU id(s)."},
            {"name": "perdevice", "kind": "slider_int", "flag": "--perdevice",
             "default": 2, "min": 1, "max": 4, "step": 1, "help": "Workers per GPU."},
        ],
        "docs": {
            "what": "Per-tilt CTF refinement across each series.",
            "range": "range_high ~7 Å; defocus_max ~8 µm.",
            "effect": "Better per-tilt CTF improves reconstruction + averaging.",
            "pitfall": "Run ts_defocus_hand first to fix handedness.",
        },
        "status": lambda ps: ps.status_ts_ctf(),
    },

    # ---------------- 7. Reconstruct ----------------
    {
        # ---------- REFERENCE STAGE / GPU GUARD ----------
        "group": "7. Reconstruct", "id": "ts_reconstruct", "label": "ts_reconstruct",
        "base": "WarpTools ts_reconstruct",
        "group_scope": "ts",
        # Warp writes float16 MRC by default; force float32 so IMOD/3dmod/Dynamo can
        # read the tomograms (and ts_reconstruct itself needs it set in the env).
        "env_export": {"WARP_FORCE_MRC_FLOAT32": "1"},
        "params": [
            {"name": "settings", "kind": "text", "flag": "--settings",
             "default": "warp_tiltseries.settings", "help": "ts .settings file."},
            {"name": "angpix", "kind": "text", "flag": "--angpix",
             "default": "10", "help": "OUTPUT tomogram pixel size (Å/px). 10 = a normal "
             "viewable/pickable tomogram. DO NOT use native (1.57) for full tomograms: "
             "the volume scales as (10/1.57)³ ≈ 260×, so each is tens of GB and ~40 min. "
             "Particles get reconstructed at fine res later by ts_export_particles."},
            {"name": "device_list", "kind": "text", "flag": "--device_list", "gpu_sep": " ",
             "default": "0", "help": "GPU id(s), e.g. 0 or '0 1'. Pick GPUs whose "
             "nvidia-smi GPU-Util is ~0% — low memory-used alone does NOT mean free."},
            {"name": "perdevice", "kind": "slider_int", "flag": "--perdevice",
             "default": 1, "min": 1, "max": 4, "step": 1,
             "help": "Workers per GPU. KEEP AT 1 when --deconv is on (V100 cuFFT crash)."},
            {"name": "deconv", "kind": "check", "flag": "--deconv",
             "default": False, "help": "Deconvolve for visual contrast (not for STA)."},
            {"name": "dont_invert", "kind": "check", "flag": "--dont_invert",
             "default": True, "help": "Skip contrast inversion (dataset-specific)."},
        ],
        "validate": lambda v: (
            "⚠ perdevice > 1 with --deconv crashes on V100 (SIGABRT exit 134). "
            "Set perdevice 1 — or, if EML45 is NOT V100, re-test before overriding."
            if v.get("perdevice", 1) > 1 and v.get("deconv") else ""),
        "docs": {
            "what": "Back-projects aligned, CTF-corrected tilts into 3D tomograms.",
            "range": "angpix ~10 for viewable tomograms; perdevice 1-2; deconv off for averaging.",
            "effect": "deconv boosts low-freq contrast; dont_invert flips densities.",
            "pitfall": "angpix native (1.57 / blank) makes tens-of-GB tomograms (~260× a "
                       "10 Å one) — it looks 'stuck at 0/15' but is just grinding; use ~10. "
                       "perdevice 2 + deconv = SIGABRT on V100 (cuFFT collision). Float16 MRC "
                       "output needs WARP_FORCE_MRC_FLOAT32=1 to open in IMOD/3dmod.",
        },
        "status": lambda ps: ps.status_warp_tomograms(),
    },

    # ---------------- 8. Pick ----------------
    {
        "group": "8. Pick", "id": "ts_template_match", "label": "ts_template_match",
        "base": "WarpTools ts_template_match",
        "group_scope": "ts",
        "params": [
            {"name": "settings", "kind": "text", "flag": "--settings",
             "default": "warp_tiltseries.settings", "help": "ts .settings file."},
            {"name": "tomo_angpix", "kind": "text", "flag": "--tomo_angpix",
             "default": "10", "help": "Matching pixel size (Å). MUST equal a ts_reconstruct "
             "--angpix you have ALREADY run — matching reuses that full tomogram "
             "(warp_tiltseries/reconstruction/<pos>_<angpix>Apx.mrc). Mismatch → "
             "'A reconstruction at the desired resolution was not found'. 8-12 typical."},
            {"name": "template_emdb", "kind": "text", "flag": "--template_emdb",
             "default": "", "help": "EMDB code to fetch + use as the template, e.g. 70905. "
             "Set EITHER this OR template_path (not both)."},
            {"name": "template_path", "kind": "text", "flag": "--template_path",
             "default": "", "help": "Path to a local template .mrc. Set EITHER this OR "
             "template_emdb (not both)."},
            {"name": "override_suffix", "kind": "text", "flag": "--override_suffix",
             "default": "", "help": "Overrides the STAR suffix (normally derived from the "
             "template name) so this pick set gets its OWN name: files become "
             "warp_tiltseries/matching/<pos>_<tomo_angpix>Apx<suffix>.star. INCLUDE A LEADING "
             "UNDERSCORE if you want one (e.g. '_run2'; without it the suffix abuts 'Apx'). "
             "Use a different suffix per run to keep parallel pick sets side by side — "
             "threshold_picks / export then pick a set via --in_suffix (this is how you fork "
             "picking). Blank = default template-derived name."},
            {"name": "subdivisions", "kind": "slider_int", "flag": "--subdivisions",
             "default": 3, "min": 1, "max": 6, "step": 1,
             "help": "Angular subdivisions of the search (finer = more orientations, slower)."},
            {"name": "template_diameter", "kind": "text", "flag": "--template_diameter",
             "default": "", "help": "Particle diameter (Å)."},
            {"name": "symmetry", "kind": "text", "flag": "--symmetry",
             "default": "C1", "help": "Point group, e.g. O, D2, C1."},
            {"name": "whiten", "kind": "check", "flag": "--whiten",
             "default": True, "help": "Spectral whitening; helps with good alignments."},
            {"name": "optimize_poses", "kind": "check", "flag": "--optimize_poses",
             "default": False, "help": "Locally refine each hit's orientation/position after "
             "the coarse search (better picks, a bit slower). ON in the reference workflow."},
            {"name": "check_hand", "kind": "slider_int", "flag": "--check_hand",
             "default": 2, "min": 0, "max": 2, "step": 1,
             "help": "2 = verify geometry/handedness during matching."},
            {"name": "npeaks", "kind": "slider_int", "flag": "--npeaks",
             "default": 2000, "min": 100, "max": 50000, "step": 500,
             "help": "Max peaks SAVED per tilt series. This is a HARD CAP — if a series "
             "actually has more particles you'll silently keep only the top-scoring 2000. "
             "For crowded samples raise it (you can tell you're capped when every series "
             "returns exactly this many). Costs disk, not match time."},
            {"name": "peak_distance", "kind": "text", "flag": "--peak_distance",
             "default": "", "help": "Minimum spacing between peaks in Å. Blank = the template "
             "diameter. Lower it (e.g. 30) for tightly-packed particles so neighbours aren't "
             "suppressed; raise it to avoid double-picking one particle."},
            {"name": "max_missing_tilts", "kind": "slider_int", "flag": "--max_missing_tilts",
             "default": 2, "min": -1, "max": 20, "step": 1,
             "help": "Drop positions not covered by at least this many tilts. -1 disables "
             "culling (keep everything, e.g. thin/edge regions); default 2."},
            {"name": "subvolume_size", "kind": "slider_int", "flag": "--subvolume_size",
             "default": 192, "min": 48, "max": 512, "step": 16,
             "help": "Local matching TILE size, in TOMOGRAM pixels (at tomo_angpix, NOT raw "
             "pixels). Just needs to comfortably exceed the template — 192 does so hugely. "
             "It is NOT the particle box (that's export --box). Reduce only if you hit GPU "
             "OOM or want speed; keep it even (FFT-friendly)."},
            {"name": "device_list", "kind": "text", "flag": "--device_list", "gpu_sep": " ",
             "default": "", "help": "GPU id(s), space-separated e.g. '2 3'. BLANK = ALL "
             "GPUs (Warp's default — it WILL grab 0/1). Set this to the idle cards (check "
             "nvidia-smi util%) to leave others' jobs alone. Or prefix CUDA_VISIBLE_DEVICES=2,3."},
            {"name": "perdevice", "kind": "slider_int", "flag": "--perdevice",
             "default": 1, "min": 1, "max": 4, "step": 1,
             "help": "Worker processes per GPU (raise only on big-memory cards)."},
        ],
        "validate": lambda v: (
            "⚠ Set EXACTLY ONE of template_emdb / template_path — matching needs a template."
            if bool(str(v.get("template_emdb", "")).strip())
            == bool(str(v.get("template_path", "")).strip())
            else "⚠ check_hand does NOT work with override_suffix: Warp reads the handedness "
            "test back under the DEFAULT template name and dies ('Could not find "
            "…_emd_XXXXX.star', all items fail). Set check_hand 0 for suffixed/forked runs — "
            "determine handedness ONCE without a suffix, then reuse check_hand 0."
            if str(v.get("override_suffix", "")).strip() and int(v.get("check_hand") or 0) > 0
            else ""),
        "docs": {
            "what": "CTF-aware template matching to locate particles "
                    "(apoferritin example values — adapt per target).",
            "range": "tomo_angpix 8-12; subdivisions 3-4; check_hand 2 (0 with a suffix).",
            "effect": "Lower tomo_angpix + higher subdivisions = finer, MUCH slower. With "
                      "--optimize_poses, coarser subdivisions (3-4) suffice — local refinement "
                      "recovers the precision.",
            "pitfall": "tomo_angpix MUST match a ts_reconstruct --angpix you already ran "
                       "(matching reuses that full tomogram) — else 'A reconstruction at the "
                       "desired resolution was not found' and every series fails. check_hand>0 "
                       "is INCOMPATIBLE with override_suffix (handedness readback uses the "
                       "default template name → 'Could not find …_emd_XXXXX.star'): set "
                       "check_hand 0 for suffixed runs. Defaults to ALL GPUs — set --device_list "
                       "(e.g. '2 3') to avoid disturbing others on 0/1. Scores are "
                       "background-normalised, so a threshold is comparable across tomograms.",
        },
        "status": lambda ps: ps.status_template_matches(),
    },
    {
        "group": "8. Pick", "id": "threshold_picks", "label": "threshold_picks",
        "base": "WarpTools threshold_picks",
        "params": [
            {"name": "settings", "kind": "text", "flag": "--settings",
             "default": "warp_tiltseries.settings", "help": "ts .settings file."},
            {"name": "in_suffix", "kind": "text", "flag": "--in_suffix",
             "default": "", "help": "Suffix of the template-match star files to threshold."},
            {"name": "out_suffix", "kind": "text", "flag": "--out_suffix",
             "default": "clean", "help": "Suffix for thresholded output star files."},
            {"name": "minimum", "kind": "slider_int", "flag": "--minimum",
             "default": 3, "min": 0, "max": 10, "step": 1,
             "help": "Min normalised score (≈ σ above background). 3 is a good start."},
        ],
        "docs": {
            "what": "Keeps picks above a normalised score threshold; writes "
                    "*<out_suffix>.star.",
            "range": "minimum ~3 (σ above background).",
            "effect": "Higher minimum = fewer, cleaner picks.",
            "pitfall": "Scores compare across tomograms thanks to bg normalisation, "
                       "so one minimum works project-wide.",
        },
        "status": lambda ps: ps.status_thresholded(),
    },

    # ---------------- 9. Export ----------------
    {
        "group": "9. Export", "id": "ts_export_particles",
        "label": "ts_export_particles",
        "base": "WarpTools ts_export_particles",
        "group_scope": "ts",
        "params": [
            {"name": "settings", "kind": "text", "flag": "--settings",
             "default": "warp_tiltseries.settings", "help": "ts .settings file."},
            {"name": "input_directory", "kind": "text", "flag": "--input_directory",
             "default": "warp_tiltseries/matching",
             "help": "Where the thresholded pick stars live."},
            {"name": "input_pattern", "kind": "text", "flag": "--input_pattern",
             "default": "*clean.star", "help": "Glob for thresholded pick star files."},
            {"name": "output_star", "kind": "text", "flag": "--output_star",
             "default": "relion4/warp/matching.star",
             "help": "Output star path. Put it INSIDE the RELION project dir "
             "(output_processing) — RELION must later be launched from that dir."},
            {"name": "output_processing", "kind": "text", "flag": "--output_processing",
             "default": "relion4/warp",
             "help": "RELION project/export dir. The subtomo image paths in the star are "
             "written RELATIVE to this, so you MUST launch RELION from here (the recurring "
             "'file does not exist' bug is launching from the wrong dir)."},
            {"name": "output_angpix", "kind": "text", "flag": "--output_angpix",
             "default": "4", "help": "Export pixel size (Å). Choose so Nyquist sits just "
             "below feature resolution."},
            {"name": "box", "kind": "slider_int", "flag": "--box",
             "default": 64, "min": 32, "max": 256, "step": 8, "help": "Box size (px)."},
            {"name": "diameter", "kind": "text", "flag": "--diameter",
             "default": "", "help": "Particle diameter (Å)."},
            {"name": "relion_format", "kind": "choice", "flag": None,
             "choices": [
                 ("3D subtomograms (RELION 4)", "--3d"),
                 ("2D image series (RELION 5)", "--2d"),
             ],
             "default": "--3d",
             "help": "RELION 4 uses 3D subtomos (--3d). RELION 5 --tomo uses the 2D "
             "image series (--2d). WarpTools needs exactly one of these — pick to match "
             "the RELION you'll hand off to."},
            {"name": "normalized_coords", "kind": "check", "flag": "--normalized_coords",
             "default": True, "help": "Coords normalised to tomogram dimensions."},
            {"name": "relative_output_paths", "kind": "check",
             "flag": "--relative_output_paths", "default": True,
             "help": "Write relative paths into the star (portable projects). Keep ON — "
             "the RELION-launch-from-output_processing rule depends on it."},
        ],
        "validate": lambda v: (
            "⚠ output_star should live INSIDE output_processing so RELION resolves the "
            "subtomo paths (launch RELION from output_processing)."
            if v.get("output_processing") and not str(v.get("output_star", "")).startswith(
                str(v.get("output_processing", "")).rstrip("/") + "/") else ""),
        "docs": {
            "what": "Extracts CTF-corrected particles into a RELION project dir — a "
                    "particles star (+ optimisation_set.star for RELION 5).",
            "range": "box 64-128; output_angpix 3-5 for most targets.",
            "effect": "3D subtomos (--3d) = RELION 4; --2d = RELION 5 --tomo. Paths are "
                      "relative to output_processing — that dir IS the RELION project root.",
            "pitfall": "LAUNCH RELION FROM output_processing, or every subtomo path is "
                       "wrong ('file does not exist'). RELION 4 does NOT auto-resize the "
                       "reference — pre-scale it (see 'RELION 4: Class3D'). Then continue "
                       "in the 'RELION 4' steps below, or hand off to RELION's own GUI.",
        },
        "status": lambda ps: ps.status_exported(),
    },

    # ---------------- 10. RELION 4 handoff (subtomo averaging) ----------------
    {
        # Bridge between ts_export_particles and the Class3D handoff: convert the Warp
        # export star to RELION 4 format (relion_convert_star), rewriting the relative
        # 'subtomo/' particle paths to absolute so RELION finds every subtomogram, and
        # optionally build a de-novo initial reference from a random particle subset
        # (relion_reconstruct). Driven by ml_relion4_convert_star_warp_auto.sh.
        "group": "10. RELION 4", "id": "relion4_convert",
        "label": "RELION 4: convert STAR + init ref",
        "base": "bash",
        "params": [
            {"name": "script", "kind": "text", "flag": None,
             "default": _pkg_script("ml_relion4_convert_star_warp_auto.sh"),
             "help": "STAR conversion + initial-reference wrapper (shipped with the app)."},
            {"name": "project_dir", "kind": "text", "flag": None,
             "default": "relion4/warp",
             "help": "The RELION project dir = ts_export_particles' output_processing. "
             "The script runs relion FROM here (the launch-root invariant)."},
            {"name": "starfile", "kind": "text", "flag": None,
             "default": "matching.star",
             "help": "Warp export star, RELATIVE to project_dir (e.g. matching.star)."},
            {"name": "RELION_MODULE", "kind": "env", "flag": "RELION_MODULE",
             "default": "relion/4.0.1", "help": "module load name for RELION 4 on your cluster."},
            {"name": "PARTICLEDIR", "kind": "env", "flag": "PARTICLEDIR",
             "default": "", "help": "Absolute path to the exported subtomo/ dir (the "
             "'subtomo/' prefix in the star is rewritten to this). Blank = "
             "<project_dir>/subtomo/. A trailing slash is enforced."},
            {"name": "PATH_MATCH", "kind": "env", "flag": "PATH_MATCH",
             "default": "subtomo/", "help": "Path prefix in the Warp star to replace with "
             "PARTICLEDIR. Change only if export wrote a different prefix."},
            {"name": "CS", "kind": "env", "flag": "CS",
             "default": "2.7", "help": "Spherical aberration (mm) for relion_convert_star."},
            {"name": "Q0", "kind": "env", "flag": "Q0",
             "default": "0.07", "help": "Amplitude contrast for relion_convert_star."},
            {"name": "NREF", "kind": "env_int", "flag": "NREF",
             "default": 1000, "min": 100, "max": 5000, "step": 100,
             "help": "Random particles used to reconstruct the initial reference."},
            {"name": "MAKE_REF", "kind": "env", "flag": "MAKE_REF",
             "default": "1", "help": "1 = also build random_subset_ref.mrc (relion_reconstruct); "
             "0 = only convert the star."},
            {"name": "execute", "kind": "check", "flag": "--execute",
             "default": False, "help": "OFF = dry run (prints the plan + relion commands, "
             "runs nothing). Turn ON to actually convert + reconstruct."},
        ],
        "docs": {
            "what": "Converts the Warp ts_export_particles star to RELION 4 format and builds "
                    "a de-novo initial reference. Rewrites the relative 'subtomo/' particle "
                    "paths to absolute (so RELION finds every subtomogram), runs "
                    "relion_convert_star, then samples NREF random particles and "
                    "relion_reconstructs random_subset_ref.mrc.",
            "range": "NREF 500-2000; Cs 2.7 mm, Q0 0.07 (300 kV cryo defaults).",
            "effect": "Writes <star>_conv.star (the RELION 4 particles) and, unless MAKE_REF=0, "
                      "random_subset_ref.mrc — a ready-to-use reference for Class3D (no external "
                      "EMDB map needed). Defaults to a DRY RUN — tick EXECUTE to run.",
            "pitfall": "Runs FROM project_dir (paths are relative to it). PATH_MATCH must match "
                       "how export wrote the paths ('subtomo/' by default) or the rewrite is a "
                       "no-op and RELION can't find the particles. The header split is "
                       "auto-detected from the star's '_rln' labels (replaces the old hardcoded "
                       "head -n 33 / tail -n +35).",
        },
        "status": None,
    },
    {
        # Extends the pipeline past export into a RELION 4 Class3D handoff, driven by
        # ml_relion4_handoff_warp_auto.sh. Encodes the invariants that repeatedly bite:
        # launch-from-export-dir, v4 reference pre-scale, clean project dir, nGPU+1 MPI.
        "group": "10. RELION 4", "id": "relion4_class3d",
        "label": "RELION 4: Class3D handoff",
        "base": "bash",
        "params": [
            {"name": "script", "kind": "text", "flag": None,
             "default": _pkg_script("ml_relion4_handoff_warp_auto.sh"),
             "help": "RELION 4 handoff wrapper (shipped with the app)."},
            {"name": "project_dir", "kind": "text", "flag": None,
             "default": "relion4/warp",
             "help": "The RELION project dir = ts_export_particles' output_processing. "
             "The script runs relion FROM here (the launch-root invariant)."},
            {"name": "particles", "kind": "text", "flag": None,
             "default": "matching.star",
             "help": "Particles star, RELATIVE to project_dir (e.g. matching.star)."},
            {"name": "RELION_MODULE", "kind": "env", "flag": "RELION_MODULE",
             "default": "relion/4.0.1", "help": "module load name for RELION 4 on your cluster."},
            {"name": "REF_MAP", "kind": "env", "flag": "REF_MAP",
             "default": "", "help": "Reference map (.mrc) — e.g. the EMDB map you template-"
             "matched with. RELION 4 does NOT auto-resize; the script rescales it to match."},
            {"name": "REF_ANGPIX", "kind": "env", "flag": "REF_ANGPIX",
             "default": "", "help": "Pixel size (Å) of REF_MAP (from its header/EMDB page)."},
            {"name": "OUTPUT_ANGPIX", "kind": "env", "flag": "OUTPUT_ANGPIX",
             "default": "4", "help": "Must equal the export output_angpix (particles' Å/px)."},
            {"name": "BOX", "kind": "env_int", "flag": "BOX",
             "default": 64, "min": 32, "max": 256, "step": 8,
             "help": "Must equal the export box size (px)."},
            {"name": "DIAMETER", "kind": "env", "flag": "DIAMETER",
             "default": "", "help": "Particle diameter (Å) for the mask."},
            {"name": "SYMMETRY", "kind": "env", "flag": "SYMMETRY",
             "default": "C1", "help": "Classify in C1; symmetrise only at Refine3D."},
            {"name": "NCLASSES", "kind": "env_int", "flag": "NCLASSES",
             "default": 4, "min": 1, "max": 12, "step": 1, "help": "Number of 3D classes (K)."},
            {"name": "GPUS", "kind": "env", "flag": "GPUS", "gpu_sep": ",",
             "default": "0,1,2,3", "help": "GPU ids for RELION, comma- or space-separated "
             "(idle ones — check util%). MPI is set to (#GPUs + 1) automatically."},
            {"name": "execute", "kind": "check", "flag": "--execute",
             "default": False, "help": "OFF = dry run (prints the plan + relion command, "
             "runs nothing). Turn ON to actually submit Class3D."},
        ],
        "docs": {
            "what": "Hands the exported subtomograms to RELION 4 for 3D classification — "
                    "asserts the export is complete, checks the launch-root path invariant, "
                    "pre-scales the reference, cleans the project dir, and submits Class3D.",
            "range": "NCLASSES 3-6; ini-lowpass 45 Å; MPI = #GPUs + 1.",
            "effect": "Runs relion_refine_mpi from project_dir. Defaults to a DRY RUN — tick "
                      "EXECUTE to launch. RELION 4 is the GPU-native path on this VM class "
                      "(RELION 5's container CUDA can outrun the host driver → GPU error 35).",
            "pitfall": "REF must be pre-scaled to OUTPUT_ANGPIX + BOX (the script does it via "
                       "relion_image_handler). OUTPUT_ANGPIX/BOX MUST match the export. Never "
                       "launch inside another RELION version's project (the script parks stale "
                       "pipeline files first).",
        },
        "status": None,
    },
]

def expand_tilt_ranges(text):
    """Expand IMOD tilt-number ranges to explicit comma-separated values, since
    remake_mdocs only handles individual numbers:
        '1,4-12,48' -> '1,4,5,6,7,8,9,10,11,12,48'
    Unparseable tokens are dropped; order is preserved, duplicates removed."""
    out, seen = [], set()
    for tok in str(text).replace(" ", "").split(","):
        if not tok:
            continue
        if "-" in tok:
            try:
                a, b = (int(x) for x in tok.split("-", 1))
            except ValueError:
                continue
            rng = range(a, b + 1) if a <= b else range(a, b - 1, -1)
        else:
            try:
                rng = [int(tok)]
            except ValueError:
                continue
        for n in rng:
            if n not in seen:
                seen.add(n)
                out.append(n)
    return ",".join(str(n) for n in out)


def _h(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_docs_html(doc):
    """Rich left-panel HTML for a stage doc dict (from tomogration_docs.json):
    title → what → why → parameter table → pitfalls (red) → good-output QC →
    refs as clickable links."""
    h = [f"<h2 style='color:#eaeaea;margin:0 0 8px 0;'>{_h(doc.get('title', ''))}</h2>"]
    for key, head, col in (("what", "WHAT", "#9ec5ff"), ("why", "WHY", "#9ec5ff")):
        if doc.get(key):
            h.append(f"<p style='margin:6px 0;'><b style='color:{col};'>{head}</b><br>"
                     f"{_h(doc[key])}</p>")
    params = doc.get("params") or []
    if params:
        h.append("<p style='margin:8px 0 2px;'><b style='color:#9ec5ff;'>PARAMETERS</b></p>")
        h.append("<table cellspacing='0' cellpadding='3' width='100%' "
                 "style='border-collapse:collapse;font-size:11px;'>")
        h.append("<tr style='color:#9a9a9a;'><th align='left'>name</th>"
                 "<th align='left'>flag</th><th align='left'>default</th>"
                 "<th align='left'>range</th><th align='left'>effect</th></tr>")
        for i, p in enumerate(params):
            bg = "#202020" if i % 2 else "#262626"
            h.append(
                f"<tr style='background:{bg};'>"
                f"<td valign='top'><code style='color:#cfe;'>{_h(p.get('name', ''))}</code></td>"
                f"<td valign='top'><code style='color:#cc9;'>{_h(p.get('flag', ''))}</code></td>"
                f"<td valign='top'>{_h(p.get('default', ''))}</td>"
                f"<td valign='top' style='color:#9a9a9a;'>{_h(p.get('range', ''))}</td>"
                f"<td valign='top'>{_h(p.get('effect', ''))}</td></tr>")
        h.append("</table>")
    if doc.get("pitfalls"):
        h.append(f"<p style='margin:8px 0;'><b style='color:#e24b4a;'>PITFALLS</b><br>"
                 f"<span style='color:#f0a0a0;'>{_h(doc['pitfalls'])}</span></p>")
    if doc.get("qc"):
        h.append(f"<p style='margin:8px 0;'><b style='color:#27ae60;'>GOOD OUTPUT "
                 f"LOOKS LIKE</b><br><span style='color:#b8e0c0;'>{_h(doc['qc'])}"
                 f"</span></p>")
    refs = doc.get("refs") or []
    if refs:
        h.append("<p style='margin:8px 0 2px;'><b style='color:#9ec5ff;'>REFS</b></p>"
                 "<ul style='margin:2px 0 2px 16px;padding:0;'>")
        for r in refs:
            h.append(f"<li><a href='{_h(r)}' style='color:#7fb4ff;'>{_h(r)}</a></li>")
        h.append("</ul>")
    return "".join(h)


def render_inline_docs_html(spec):
    """Fallback left-panel HTML for stages not covered by tomogration_docs.json:
    render the stage's own inline docs dict (what/range/effect/pitfall)."""
    d = spec.get("docs", {})
    h = [f"<h2 style='color:#eaeaea;margin:0 0 8px 0;'>{_h(spec['label'])}</h2>"]
    for key, head, col in (("what", "WHAT", "#9ec5ff"), ("range", "RANGE", "#9a9a9a"),
                           ("effect", "EFFECT", "#9a9a9a")):
        if d.get(key) and d.get(key) != "n/a":
            h.append(f"<p style='margin:6px 0;'><b style='color:{col};'>{head}</b><br>"
                     f"{_h(d[key])}</p>")
    if d.get("pitfall"):
        h.append(f"<p style='margin:8px 0;'><b style='color:#e24b4a;'>PITFALL</b><br>"
                 f"<span style='color:#f0a0a0;'>{_h(d['pitfall'])}</span></p>")
    return "".join(h)


def stage_defaults(spec):
    """{param_name: default} for a stage — the values the widgets start at."""
    return {p["name"]: p.get("default") for p in spec.get("params", [])}


def _norm_gpu(s, sep):
    """Normalise a GPU-id list to the separator the target tool wants, so the user
    can type either '0 1 2 3' or '0,1,2,3' anywhere and it comes out correct
    (WarpTools/AreTomo need spaces; RELION/miss-alignment need commas). No-op when
    the param has no gpu_sep hint."""
    if not sep or not s:
        return s
    return sep.join(t for t in re.split(r"[ ,]+", s.strip()) if t)


def build_command(spec, values, warp_cmd=None, group_inputs=None):
    """Pure command assembler (no Qt) — the single source of truth the editable
    command box is seeded from. env/env_int params become a VAR=value prefix;
    checks emit their flag when truthy; flagged params emit 'flag value';
    flag=None params emit their value positionally, in declaration order.

    base = (env prefix) + spec.base + (flags/positionals). If warp_cmd is given,
    a leading 'WarpTools' in the base is replaced by it (so the user's module
    load / conda activate / path runs before every WarpTools subcommand)."""
    env_parts, body_parts = [], []
    for p in spec.get("params", []):
        v = values.get(p["name"])
        kind = p["kind"]
        if kind in ("env", "env_int"):
            s = _norm_gpu(str(v).strip(), p.get("gpu_sep"))
            if s != "":
                env_parts.append(f"{p['flag']}='{s}'" if any(c in s for c in " \t")
                                 else f"{p['flag']}={s}")
        elif kind == "check":
            if v:
                body_parts.append(p["flag"])
        else:
            s = _norm_gpu(str(v).strip(), p.get("gpu_sep"))
            if s != "":
                body_parts.append(f"{p['flag']} {s}" if p.get("flag") else s)
    # Active tilt-series group: restrict this step to the subset via --input_data
    # (unless the user already typed an --input_data into the params).
    scope = spec.get("group_scope")
    if (group_inputs and scope and group_inputs.get(scope)
            and not any(part.startswith("--input_data") for part in body_parts)):
        body_parts.append(f"--input_data {group_inputs[scope]}")
    # Always-on VAR=value the stage bakes in (e.g. MA_MODE=infer) — kept out of the
    # form so there's no confusing editable field for a value that must not change.
    for k, v in (spec.get("fixed_env") or {}).items():
        env_parts.insert(0, f"{k}={v}")
    base = spec.get("base", "")
    if warp_cmd and base.startswith("WarpTools"):
        base = warp_cmd + base[len("WarpTools"):]
    segs = []
    if env_parts:
        segs.append(" ".join(env_parts))
    if base:
        segs.append(base)
    segs.extend(body_parts)
    cmd = " ".join(segs)
    # Vars that must be EXPORTED into the shell before the tool runs. A bare
    # "VAR=val cmd" prefix only applies to the first word, which for WarpTools is
    # `module` (base = "module load … && conda activate … && WarpTools …"), so the
    # var would never reach WarpTools. `export VAR=val && …` puts it in the
    # environment for the whole chain. (ts_reconstruct needs WARP_FORCE_MRC_FLOAT32=1
    # so its tomograms are float32 and IMOD/3dmod can read them.)
    exports = spec.get("env_export")
    if exports:
        ex = " ".join(f"{k}={v}" for k, v in exports.items())
        cmd = f"export {ex} && {cmd}"
    return cmd


# Primary output directory per stage (relative to the project root) for the
# per-step "open output" button + iteration dropdown. "aretomo" = the versioned
# AreTomo folders. Stages absent here (tool/action steps) get no output controls.
STAGE_OUTPUTS = {
    "rename": ".", "imod_warp_key": ".", "remake_mdocs": "mdocs",
    "gain_convert": "gains", "gain_reciprocal": "gains",
    "create_settings_fs": "warp_frameseries", "fs_motion_and_ctf": "warp_frameseries",
    "create_settings_ts": "warp_tiltseries", "ts_import": "tomostar",
    "ts_stack": "warp_tiltseries/tiltstack", "aretomo": "aretomo",
    "miss_align": "warp_tiltseries", "miss_align_infer": "warp_tiltseries",
    "ts_import_alignments": "warp_tiltseries", "ts_ctf": "warp_tiltseries",
    "ts_reconstruct": "warp_tiltseries/reconstruction",
    "ts_template_match": "warp_tiltseries/matching",
    "threshold_picks": "warp_tiltseries/matching", "ts_export_particles": "relion4/warp",
    "relion4_convert": "relion4/warp", "relion4_class3d": "relion4/warp",
}

# Which mockup column each stage group belongs to (the three job-list panels).
COLUMN_OF_GROUP = {
    "1. Data prep": "curation", "2. Gain": "curation",
    "3. Frameseries": "stackprep", "4. Tilt series": "stackprep",
    "5. Alignment": "alignrecon", "6. CTF": "alignrecon",
    "7. Reconstruct": "alignrecon", "8. Pick": "alignrecon",
    "9. Export": "alignrecon", "10. RELION 4": "alignrecon",
}
COLUMN_TITLES = {
    "curation": "Tilt curation", "stackprep": "Stack preparation",
    "alignrecon": "Alignment & Reconstruction",
}

# Directory-overview schematic: the key project dirs to draw, in pipeline order.
KEY_DIRS = [
    ("frames", "frames/"), ("mdocs", "mdocs/"), ("gains", "gains/"),
    ("Thumbnails", "Thumbnails/"),
    ("warp_frameseries", "warp_frameseries/"),
    ("tomostar", "tomostar/"),
    ("warp_tiltseries", "warp_tiltseries/"),
    ("warp_tiltseries/tiltstack", "…/tiltstack/"),
    ("aretomo_output", "aretomo_output*/"),
    ("warp_tiltseries/reconstruction", "…/reconstruction/"),
    ("warp_tiltseries/matching", "…/matching/"),
    ("relion4/warp", "relion4/ (RELION project)"),
]
# Per-stage (inputs, outputs) as dir paths relative to the project root — drives
# the directory-overview highlighting (blue=input, green=output, grey=other).
STAGE_IO = {
    "rename":               (["."], ["mdocs", "frames"]),
    "imod_warp_key":        (["mdocs"], ["."]),
    "inspect_select":       (["Thumbnails", "mdocs"], ["mdocs"]),
    "remake_mdocs":         (["mdocs"], ["mdocs"]),
    "gain_convert":         (["gains"], ["gains"]),
    "gain_reciprocal":      (["gains"], ["gains"]),
    "create_settings_fs":   (["frames", "gains"], ["warp_frameseries"]),
    "fs_motion_and_ctf":    (["frames", "warp_frameseries"], ["warp_frameseries"]),
    "create_settings_ts":   (["mdocs"], ["warp_tiltseries"]),
    "ts_import":            (["mdocs", "warp_frameseries"], ["tomostar", "warp_tiltseries"]),
    "ts_stack":             (["warp_tiltseries", "tomostar"], ["warp_tiltseries/tiltstack"]),
    "aretomo":              (["warp_tiltseries/tiltstack"], ["aretomo_output"]),
    "miss_align":           (["warp_tiltseries", "warp_tiltseries/tiltstack"], ["warp_tiltseries"]),
    "miss_align_infer":     (["warp_tiltseries", "warp_tiltseries/tiltstack"], ["warp_tiltseries"]),
    "ts_import_alignments": (["aretomo_output", "warp_tiltseries"], ["warp_tiltseries"]),
    "sync_selection":       (["warp_tiltseries"], ["warp_tiltseries"]),
    "ts_defocus_hand":      (["warp_tiltseries"], ["warp_tiltseries"]),
    "ts_ctf":               (["warp_tiltseries", "warp_tiltseries/tiltstack"], ["warp_tiltseries"]),
    "ts_reconstruct":       (["warp_tiltseries"], ["warp_tiltseries/reconstruction"]),
    "ts_template_match":    (["warp_tiltseries/reconstruction"], ["warp_tiltseries/matching"]),
    "threshold_picks":      (["warp_tiltseries/matching"], ["warp_tiltseries/matching"]),
    "ts_export_particles":  (["warp_tiltseries", "warp_tiltseries/matching"], ["relion4/warp"]),
    "relion4_convert":      (["relion4/warp"], ["relion4/warp"]),
    "relion4_class3d":      (["relion4/warp"], ["relion4/warp"]),
}

# Generic filename patterns each key directory is searched for — shown in the
# Job details INPUTS/OUTPUTS lists so the user knows WHICH files a step consumes
# or produces in each folder (not just the folder name). Keyed by the same rel
# dirs used in STAGE_IO.
DIR_FILE_HINTS = {
    "frames": "*.eer  (raw movies)",
    "mdocs": "*.mdoc  (per-series)",
    "gains": "*.gain / gain reference",
    "Thumbnails": "*.mrc  (per-series montage)",
    "warp_frameseries": "*.xml  (per-movie metadata)",
    "tomostar": "*.tomostar  (per-series)",
    "warp_tiltseries": "*.xml  (per-series metadata)",
    "warp_tiltseries/tiltstack": "*.st + *.rawtlt  (aligned stacks)",
    "aretomo_output": "Imod/*.xf  (alignments)",
    "warp_tiltseries/reconstruction": "*_<angpix>Apx.mrc  (tomograms)",
    "warp_tiltseries/matching": "*_<suffix>.star  (pick lists)",
    "relion4/warp": "*.star + subtomo/*.mrc",
    ".": "(project root)",
}

# Stages whose output dir should be ARCHIVED (renamed aside, timestamped) instead
# of overwritten when re-run, so historic results stay reviewable. Only stages that
# write a self-contained product into a dedicated dir (downstream reads the fresh
# one) — NOT incremental/idempotent steps (fs_motion_and_ctf, ts_stack, settings).
# AreTomo already self-versions (aretomo_output-vN), so it's not listed here.
ARCHIVE_ON_RERUN = {"ts_reconstruct", "ts_template_match"}

# File-open routing for the Processing-History detail view.
THREEDMOD_EXTS = {".mrc", ".mrcs", ".st", ".ali", ".rec", ".preali", ".mod", ".map"}
TEXT_EXTS = {".txt", ".star", ".xml", ".mdoc", ".settings", ".yaml", ".yml",
             ".log", ".csv", ".com", ".json", ".tlt", ".rawtlt", ".xf", ".aln"}

MONO = "Menlo, Consolas, monospace"
DOT_GREY = "color:#bbb;font-size:14px;"
DOT_GREEN = "color:#27ae60;font-size:14px;"
DOT_ORANGE = "color:#e0a850;font-size:14px;"
DOT_RED = "color:#c0392b;font-size:14px;"

# Force Qt's own file dialog. The native/portal chooser (xdg-desktop-portal) is
# broken on this XFCE box (the qt.qpa.theme portal.Settings errors) and comes up
# EMPTY over ceph; Qt's own dialog enumerates the filesystem directly.
NONATIVE = QFileDialog.Option.DontUseNativeDialog
# For DIRECTORY pickers also show dirs only — the data folders hold 10k+ files and
# listing them all over ceph hangs/crashes the dialog. cephfs reports entry types,
# so dirs-only skips stat-ing the thousands of .eer/.mrc/.mdoc.
NONATIVE_DIR = QFileDialog.Option.DontUseNativeDialog | QFileDialog.Option.ShowDirsOnly


# ===========================================================================
# JOB MODEL (Phase 1) — a CryoSPARC-style DAG of job INSTANCES.
#
# The three-column view treats each STAGE as a singleton whose result lives at
# one conventional path (STAGE_OUTPUTS). The card view instead models each RUN
# as a job instance with its OWN processing directory, wired to upstream jobs by
# named input slots. This is possible because every WarpTools command accepts
# --input_processing / --output_processing (they live in BaseCommand): a job
# READS its parent's processing dir and WRITES its own, sharing no mutable state.
# Verified on the VM 2026-07-10 — a branched `ts_ctf --output_processing
# warp_tiltseries_b` left the trunk XML byte-identical (md5 OK), kept all upstream
# alignment metadata (size 430,737 -> 431,020, not a stripped rewrite), and a
# `ts_reconstruct --input_processing warp_tiltseries_b` read it back and
# reconstructed. Non-WarpTools wrappers (aretomo, miss_align, relion4_*) take
# explicit in/out dirs instead, so they get no processing-dir flags here.
#
# Store: .tomogration_jobs.json in the project root:
#     {"seq": <int>, "jobs": {"J1": {..job..}, "J2": {...}}}
# This whole layer is PURE (stdlib only, no Qt) so it unit-tests off the VM; the
# GUI wiring (dispatch, the canvas) sits on top of it in the Tomogration class.
# ===========================================================================
JOBS_FILE = ".tomogration_jobs.json"
JOB_STATUSES = ("queued", "running", "completed", "failed")


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
        "status": "queued",
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
    so the run reads the .settings default, exactly like the three-column view."""
    if not is_warp_stage(spec):
        return ""
    toks = []
    pid = parent_job_id(job)
    parent = store.get("jobs", {}).get(pid) if pid else None
    if parent:
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


def summary_text(summary):
    """One-line human readout for a job card, from its summary dict. Known keys
    get friendly units; anything else falls back to 'value key' pairs."""
    if not summary:
        return ""
    parts = []
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
    "threshold_picks": "Threshold picks", "ts_export_particles": "Export particles",
    "relion4_convert": "RELION 4: convert STAR", "relion4_class3d": "RELION 4: Class3D",
}


def stage_title(stage_id, fallback=""):
    return FRIENDLY_TITLES.get(stage_id, fallback or stage_id)


def canvas_layout(store):
    """Positioned workflow graph for the canvas. Returns (nodes, edges).

    One ROW per stage, in canonical STAGES order. A stage with no jobs shows a
    single greyed GHOST node ('the default workflow, not yet run'); a stage with
    jobs shows one real node per job, spread across COLUMNS so forks sit side by
    side. Edges: real jobs link to their parent job (the true DAG); stages with
    no real parent are chained along the ghost trunk so the default pipeline
    reads as a connected flow."""
    jobs = store.get("jobs", {}) if isinstance(store, dict) else {}
    by_stage = {}
    for jid, job in jobs.items():
        by_stage.setdefault(job.get("stage_id"), []).append((jid, job))
    for lst in by_stage.values():
        lst.sort(key=lambda t: t[0])

    nodes, index, row_first = [], {}, {}
    for row, spec in enumerate(STAGES):
        sid = spec["id"]
        y = row * (CARD_H + GAP_Y)
        title = stage_title(sid, spec.get("label", sid))
        js = by_stage.get(sid, [])
        if not js:
            nid = f"ghost:{sid}"
            n = {"id": nid, "stage_id": sid, "label": spec.get("label", sid),
                 "title": title, "group": spec.get("group", ""), "row": row, "col": 0,
                 "x": 0, "y": y, "w": CARD_W, "h": CARD_H,
                 "is_ghost": True, "status": "ghost", "summary": {}}
            nodes.append(n)
            index[nid] = n
            row_first[sid] = nid
        else:
            for col, (jid, job) in enumerate(js):
                fork = "(fork)" in str(job.get("label", ""))
                n = {"id": jid, "stage_id": sid,
                     "label": job.get("label", spec.get("label", sid)),
                     "title": title + (" (fork)" if fork else ""),
                     "group": spec.get("group", ""), "row": row, "col": col,
                     "x": col * (CARD_W + GAP_X), "y": y,
                     "w": CARD_W, "h": CARD_H, "is_ghost": False,
                     "status": job.get("status", "queued"),
                     "summary": job.get("summary", {}) or {}}
                nodes.append(n)
                index[jid] = n
            row_first[sid] = js[0][0]

    edges, has_real_parent = [], set()
    for jid, job in jobs.items():
        parent = next((pid for pid in (job.get("inputs") or {}).values()
                       if pid and pid in index), None)
        if parent:
            edges.append((parent, jid))
            has_real_parent.add(job.get("stage_id"))
    order = [s["id"] for s in STAGES]
    for a, b in zip(order, order[1:]):
        if b in has_real_parent:          # already linked via a real parent edge
            continue
        src, dst = row_first.get(a), row_first.get(b)
        if src and dst:
            edges.append((src, dst))
    return nodes, edges


# ===========================================================================
# QProcess wrapper: live stdout/stderr streaming + terminate
# ===========================================================================
class ProcessRunner(QObject):
    line = Signal(str, str)        # (text, level: out|err|info|ok|fail)
    finished = Signal(int)         # exit code

    def __init__(self):
        super().__init__()
        self.proc = None

    def run(self, command, cwd):
        self.proc = QProcess()
        self.proc.setWorkingDirectory(cwd)
        self.proc.setProcessChannelMode(QProcess.SeparateChannels)
        self.proc.readyReadStandardOutput.connect(self._stdout)
        self.proc.readyReadStandardError.connect(self._stderr)
        self.proc.finished.connect(self._done)
        self.line.emit(f"$ {command}", "info")
        # bash -lc so `module load`, VAR=value env prefixes and && chains work.
        self.proc.start("bash", ["-lc", command])

    def terminate(self):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            self.proc.terminate()      # SIGTERM
            self.line.emit("[terminate requested]", "fail")
            # Escalate to SIGKILL if it's still alive after 3s (brief: Ctrl-C
            # equivalent that kills the job, not the GUI).
            QTimer.singleShot(3000, self._kill_if_alive)

    def _kill_if_alive(self):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            self.proc.kill()
            self.line.emit("[killed]", "fail")

    def busy(self):
        return self.proc is not None and self.proc.state() != QProcess.NotRunning

    def _stdout(self):
        for ln in bytes(self.proc.readAllStandardOutput()).decode(errors="replace").splitlines():
            self.line.emit(ln, "out")

    def _stderr(self):
        for ln in bytes(self.proc.readAllStandardError()).decode(errors="replace").splitlines():
            self.line.emit(ln, "err")

    def _done(self, code, _status):
        self.line.emit(f"[exit {code}]", "ok" if code == 0 else "fail")
        self.finished.emit(code)


# ===========================================================================
# Positions Inspector  (Qt port of warp_auto.py PositionInspectorWindow)
# ===========================================================================
class PositionsInspector(QDialog):
    """Per-position inspection: mdoc contents vs frames/ on disk. Lets you
    quarantine bad/orphan tilts to frames/bad/, fixing the mdoc as it goes."""

    def __init__(self, parent, project, log_fn):
        super().__init__(parent)
        self.project = project
        self.log_fn = log_fn
        self.results = []
        self.current = None
        self.setWindowTitle("Positions Inspector — mdoc vs frames/")
        self.resize(1000, 700)

        v = QVBoxLayout(self)

        top = QHBoxLayout()
        self.summary = QLabel("Scanning…")
        self.summary.setStyleSheet("font-weight:600;")
        top.addWidget(self.summary)
        top.addStretch(1)
        del_all = QPushButton("Delete ALL orphan .eer")
        del_all.clicked.connect(self._delete_all_orphans)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        top.addWidget(del_all)
        top.addWidget(refresh)
        v.addLayout(top)

        split = QSplitter(Qt.Horizontal)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(QLabel("Positions (click to inspect)"))
        self.pos_list = QListWidget()
        self.pos_list.setStyleSheet(f"font-family:{MONO};")
        self.pos_list.currentRowChanged.connect(self._on_select_position)
        lv.addWidget(self.pos_list)
        split.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        self.detail_header = QLabel("Select a position on the left")
        self.detail_header.setStyleSheet("font-weight:600;")
        rv.addWidget(self.detail_header)

        btn_row = QHBoxLayout()
        self.del_sel = QPushButton("Delete SELECTED tilt")
        self.del_sel.setEnabled(False)
        self.del_sel.clicked.connect(self._delete_selected_tilt)
        self.del_extras = QPushButton("Delete this position's orphans")
        self.del_extras.setEnabled(False)
        self.del_extras.clicked.connect(self._delete_extras)
        self.open_frames = QPushButton("Open frames/")
        self.open_frames.setEnabled(False)
        self.open_frames.clicked.connect(self._open_frames)
        for b in (self.del_sel, self.del_extras, self.open_frames):
            btn_row.addWidget(b)
        btn_row.addStretch(1)
        rv.addLayout(btn_row)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(5)
        self.tree.setHeaderLabels(["Acq #", "Tilt (°)", "Mdoc subframe",
                                   "On disk?", "Status"])
        self.tree.header().setSectionResizeMode(2, QHeaderView.Stretch)
        self.tree.itemSelectionChanged.connect(self._on_tree_select)
        rv.addWidget(self.tree, 1)

        self.orphan_label = QLabel("")
        self.orphan_label.setStyleSheet("color:#b76f00;font-style:italic;")
        rv.addWidget(self.orphan_label)
        split.addWidget(right)
        split.setSizes([280, 720])
        v.addWidget(split, 1)

        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        v.addWidget(close)

        self.refresh()

    def refresh(self):
        self.results = self.project.inspect_positions()
        self.pos_list.blockSignals(True)
        self.pos_list.clear()
        total_missing = total_extra = 0
        for r in self.results:
            n_missing = len(r["missing_from_disk"])
            n_extra = len(r["extra_on_disk"])
            total_missing += n_missing
            total_extra += n_extra
            flag = (" !MISSING" if n_missing else "") + (" +extra" if n_extra else "")
            self.pos_list.addItem(
                f"{r['name']:<14} mdoc={len(r['mdoc_entries']):>3} "
                f"disk={len(r['frames_on_disk']):>3} "
                f"matched={len(r['matched']):>3}{flag}")
        self.pos_list.blockSignals(False)
        self.summary.setText(
            f"{len(self.results)} positions — {total_missing} missing, "
            f"{total_extra} orphan .eer")
        self.current = None
        self.tree.clear()
        self.detail_header.setText("Select a position on the left")
        for b in (self.del_sel, self.del_extras, self.open_frames):
            b.setEnabled(False)
        self.orphan_label.setText("")

    def _on_select_position(self, row):
        if row < 0 or row >= len(self.results):
            return
        r = self.results[row]
        self.current = r
        self.del_sel.setEnabled(False)
        self.detail_header.setText(
            f"{r['name']} — mdoc: {len(r['mdoc_entries'])} tilts, on disk: "
            f"{len(r['frames_on_disk'])} .eer, matched: {len(r['matched'])}")
        self.tree.clear()
        disk_set = set(r["frames_on_disk"])
        green, red, orange = QColor("#1a8a3a"), QColor("#c0392b"), QColor("#b76f00")
        for acq, angle, subframe in r["mdoc_entries"]:
            on_disk = subframe in disk_set if subframe else False
            angle_text = f"{angle:+.2f}" if angle is not None else "?"
            sub_text = subframe if subframe else "(no SubFramePath)"
            status = "ok" if on_disk else "missing .eer"
            item = QTreeWidgetItem([str(acq), angle_text, sub_text,
                                    "yes" if on_disk else "NO", status])
            item.setData(0, Qt.UserRole, ("ok" if on_disk else "missing", subframe))
            for c in range(5):
                item.setForeground(c, QBrush(green if on_disk else red))
            self.tree.addTopLevelItem(item)
        for extra in r["extra_on_disk"]:
            item = QTreeWidgetItem(["?", "?", extra, "orphan", "not in mdoc"])
            item.setData(0, Qt.UserRole, ("orphan", extra))
            for c in range(5):
                item.setForeground(c, QBrush(orange))
            self.tree.addTopLevelItem(item)
        if r["extra_on_disk"]:
            self.orphan_label.setText(
                f"{len(r['extra_on_disk'])} orphan .eer for {r['name']} "
                f"(on disk, not referenced by mdoc)")
            self.del_extras.setEnabled(True)
        else:
            self.orphan_label.setText("No orphan .eer for this position.")
            self.del_extras.setEnabled(False)
        self.open_frames.setEnabled(True)

    def _selected_row_data(self):
        items = self.tree.selectedItems()
        if not items:
            return None
        return items[0], items[0].data(0, Qt.UserRole)

    def _on_tree_select(self):
        sel = self._selected_row_data()
        self.del_sel.setEnabled(bool(sel) and sel[1] and sel[1][0] in ("ok", "orphan"))

    def _delete_selected_tilt(self):
        if not self.current:
            return
        sel = self._selected_row_data()
        if not sel:
            return
        item, (tag, eer_basename) = sel
        if not eer_basename:
            return
        acq_raw = item.text(0)
        angle_raw = item.text(1)
        detail = ("  • REMOVE its ZValue block from the mdoc (renumbering the rest)"
                  if tag == "ok" else
                  "  • (orphan — the mdoc is not touched)")
        if QMessageBox.question(
                self, "Delete this tilt?",
                f"Position: {self.current['name']}\nAcq #: {acq_raw}\n"
                f"Tilt: {angle_raw}\nFile: {eer_basename}\n\nThis will:\n"
                f"  • MOVE {eer_basename} from frames/ to frames/bad/\n{detail}\n\nProceed?"
        ) != QMessageBox.Yes:
            return
        moved = self.project.move_frames_to_bad([eer_basename])
        if tag == "ok" and moved:
            if ProjectState.remove_mdoc_zvalue_block(
                    self.current["mdoc_path"], eer_basename):
                self.log_fn(f"Removed ZValue block for {eer_basename}", "ok")
            else:
                self.log_fn(f"Failed to update mdoc for {eer_basename}", "fail")
        if moved:
            tilt_angle = None
            try:
                tilt_angle = float(str(angle_raw).replace("+", "").strip())
            except ValueError:
                pass
            reason = ("orphan (not in mdoc) deleted via Positions Inspector"
                      if tag == "orphan" else "manual deletion via Positions Inspector")
            try:
                self.project.append_auto_exclusion(
                    eer_basename, reason, position=self.current["name"],
                    tilt_angle=tilt_angle)
            except OSError as e:
                self.log_fn(f"Could not update exclusion_list.txt: {e}", "fail")
        name = self.current["name"]
        self.refresh()
        for i, r in enumerate(self.results):
            if r["name"] == name:
                self.pos_list.setCurrentRow(i)
                break

    def _delete_extras(self):
        if not self.current or not self.current["extra_on_disk"]:
            return
        extras = self.current["extra_on_disk"]
        preview = "\n".join(extras[:5]) + ("\n…" if len(extras) > 5 else "")
        if QMessageBox.question(
                self, "Delete orphan .eer?",
                f"Move {len(extras)} .eer not referenced by "
                f"{self.current['name']}.mdoc to frames/bad/?\n\n{preview}"
        ) != QMessageBox.Yes:
            return
        moved = self.project.move_frames_to_bad(extras)
        if moved:
            for eer in extras:
                try:
                    self.project.append_auto_exclusion(
                        eer, "orphan (not in mdoc) bulk-deleted via Positions Inspector",
                        position=self.current["name"])
                except OSError:
                    pass
        self.refresh()

    def _delete_all_orphans(self):
        by_pos = [(r["name"], e) for r in self.results for e in r["extra_on_disk"]]
        if not by_pos:
            QMessageBox.information(self, "Nothing to do", "No orphan .eer found.")
            return
        if QMessageBox.question(
                self, "Delete all orphans?",
                f"Move {len(by_pos)} orphan .eer (not in any Position*.mdoc) "
                f"to frames/bad/?"
        ) != QMessageBox.Yes:
            return
        moved = self.project.move_frames_to_bad([e for _, e in by_pos])
        if moved:
            for pos, eer in by_pos:
                try:
                    self.project.append_auto_exclusion(
                        eer, "orphan (not in mdoc) bulk-deleted via Positions Inspector",
                        position=pos)
                except OSError:
                    pass
        self.refresh()

    def _open_frames(self):
        try:
            subprocess.Popen(["xdg-open", str(self.project.root / "frames")])
        except OSError as e:
            self.log_fn(f"Could not open folder: {e}", "fail")


# ===========================================================================
# AreTomo runs viewer  (Qt port of show_aretomo_versions)
# ===========================================================================
class AreTomoVersions(QDialog):
    def __init__(self, parent, project):
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("AreTomo2 runs")
        self.resize(900, 600)
        versions = project.list_aretomo_versions()

        v = QVBoxLayout(self)
        v.addWidget(QLabel(f"{len(versions)} AreTomo2 run folder(s)"))
        split = QSplitter(Qt.Horizontal)

        self.versions = versions
        self.listw = QListWidget()
        for d in versions:
            n_mrc = len(list((d / "mrc").glob("*.mrc"))) if (d / "mrc").is_dir() else 0
            has_p = (d / "PARAMETERS.txt").is_file()
            self.listw.addItem(
                f"{d.name}  ({n_mrc} mrc, "
                f"{'params recorded' if has_p else 'no PARAMETERS.txt'})")
        self.listw.currentRowChanged.connect(self._show)
        split.addWidget(self.listw)

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setStyleSheet(f"font-family:{MONO};font-size:12px;")
        split.addWidget(self.text)
        split.setSizes([300, 600])
        v.addWidget(split, 1)

        row = QHBoxLayout()
        open_btn = QPushButton("Open folder")
        open_btn.clicked.connect(self._open_folder)
        row.addWidget(open_btn)
        row.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        v.addLayout(row)

        if versions:
            self.listw.setCurrentRow(len(versions) - 1)

    def _show(self, row):
        if row < 0 or row >= len(self.versions):
            return
        pf = self.versions[row] / "PARAMETERS.txt"
        if pf.is_file():
            try:
                self.text.setPlainText(pf.read_text())
            except OSError as e:
                self.text.setPlainText(f"(could not read PARAMETERS.txt: {e})")
        else:
            self.text.setPlainText("(no PARAMETERS.txt — older run)")

    def _open_folder(self):
        row = self.listw.currentRow()
        if 0 <= row < len(self.versions):
            try:
                subprocess.Popen(["xdg-open", str(self.versions[row])])
            except OSError:
                pass


# ===========================================================================
# Visual tilt inspector (thumbnail grid + 3dmod) — exclude tilts/positions
# ===========================================================================
class TiltInspector(QDialog):
    """Thumbnail-FREE tilt-series inspection list. Rendering a montage per series
    (Thumbnails/*.mrc) was costly over ceph, so this is now a plain list — one row
    per series: name (tomo5 → renamed), an 'exclude' checkbox, a 'bad tilts (IMOD #)'
    field, a per-series '3dmod' button, and a checkbox per user group. 'Open ALL in
    3dmod' opens every non-excluded series in ONE 3dmod so you inspect visually while
    typing exclusions here. Save writes the manual section of exclusion_list.txt
    (keyed by RENAMED name) and quarantines excluded series, feeding Remake mdocs."""

    def __init__(self, parent, project, log_fn):
        super().__init__(parent)
        self.project = project
        self.log_fn = log_fn
        self.setWindowTitle("Tilt series — inspect & exclude")
        self.resize(680, 800)
        self.rows = []          # [(tomo5, renamed, exclude_cb, tilts_edit)]
        self.groups = project.load_groups()   # {active, groups:{name:[series]}}

        v = QVBoxLayout(self)
        self.header = QLabel("…")
        self.header.setWordWrap(True)
        v.addWidget(self.header)

        bar = QHBoxLayout()
        openall = QPushButton("Open ALL in 3dmod")
        openall.setToolTip("Open every non-excluded tilt series in ONE 3dmod window "
                           "(WARP_FORCE_MRC_FLOAT32=1) — inspect there, type bad tilts here.")
        openall.clicked.connect(self._open_all_3dmod)
        newg = QPushButton("＋ New group")
        newg.setToolTip("Create a tilt-series group, then tick series below to add "
                        "them to it. Set the active group from the main window.")
        newg.clicked.connect(self._new_group)
        save = QPushButton("Save exclusions")
        save.clicked.connect(self._save)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._populate)
        for b in (openall, newg, save, refresh):
            bar.addWidget(b)
        bar.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        bar.addWidget(close)
        v.addLayout(bar)

        self.list_box = QVBoxLayout()
        self.list_box.setAlignment(Qt.AlignTop)
        inner = QWidget()
        inner.setLayout(self.list_box)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        v.addWidget(scroll, 1)

        self._populate()

    def _series_names(self):
        """Series to list: prefer Thumbnails/* (Tomo5 names), else mdocs/*.mdoc."""
        thumbs = self.project.find_thumbnails_dir()
        if thumbs is not None:
            return sorted(p.stem for p in thumbs.glob("*.mrc"))
        md = self.project.root / "mdocs"
        return sorted(f.stem for f in md.glob("*.mdoc")) if md.is_dir() else []

    def _populate(self):
        while self.list_box.count():
            w = self.list_box.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.rows = []
        self.listmap = self.project.load_listfile()         # tomo5 -> renamed
        self.revmap = self.project.load_listfile_reverse()  # renamed -> tomo5
        series = self._series_names()
        if not series:
            self.header.setText(
                "No tilt series found. Set the project root to a folder with "
                "Thumbnails/ (acquisition) or mdocs/ (after Sort files).")
            return
        mapped = sum(1 for s in series if s in self.listmap or s in self.revmap)
        self.header.setText(
            f"{len(series)} tilt series.  {mapped} mapped to original Tomo5 names"
            + ("" if (self.listmap or self.revmap) else " — NO listfile, so names show "
               "as-is; run Rename (or the thumbnails helper for a merged set).")
            + "   Tick 'exclude' to drop a whole series; type IMOD tilt #s (1,4-12,48) "
              "to drop individual tilts.")

        gnames = sorted(self.groups["groups"])
        for stem in series:
            # A listed name may be a Tomo5 name (raw dataset) or a renamed Position
            # (merged/sorted dataset). Show the conversion note in whichever direction
            # we have. 'renamed' is the exclusion / group key (always the Position).
            if stem in self.listmap and self.listmap[stem] != stem:
                renamed = self.listmap[stem]
                note = f"{stem}  →  {renamed}"          # Tomo5 → Position
            elif stem in self.revmap:
                renamed = stem
                note = f"{stem}   ⟵ was {self.revmap[stem]}"   # Position ⟵ Tomo5
            else:
                renamed = stem
                note = stem
            excluded = self.project.is_series_excluded(renamed)
            row = QHBoxLayout()
            if excluded:
                note += "   ✗EXCL"
            lab = QLabel(note)
            lab.setFixedWidth(300)
            lab.setStyleSheet("color:#e24b4a;font-size:11px;font-weight:600;" if excluded
                              else "font-size:11px;")
            lab.setToolTip(note)
            excl = QCheckBox("exclude")
            excl.setChecked(excluded)
            excl.setEnabled(not excluded)   # already-quarantined: shown, locked
            tilts = QLineEdit()
            tilts.setPlaceholderText("bad tilts, IMOD #  e.g. 1,4-12,48")
            tilts.setEnabled(not excluded)
            d3 = QPushButton("3dmod")
            d3.setFixedWidth(58)
            d3.setToolTip("Open just this series in 3dmod")
            d3.clicked.connect(lambda _=False, t=stem, r=renamed: self._open_3dmod(t, r))
            row.addWidget(lab)
            row.addWidget(excl)
            row.addWidget(tilts, 1)
            row.addWidget(d3)
            # One checkbox per user group: tick to include this series in it.
            for gname in gnames:
                gcb = QCheckBox(gname)
                gcb.setChecked(renamed in self.groups["groups"][gname])
                gcb.setEnabled(not excluded)
                gcb.stateChanged.connect(
                    lambda st, g=gname, r=renamed: self._toggle_group(g, r, st))
                row.addWidget(gcb)
            rw = QWidget()
            rw.setLayout(row)
            self.list_box.addWidget(rw)
            self.rows.append((stem, renamed, excl, tilts))

    def _open_3dmod(self, tomo5, renamed):
        stack = self.project.find_series_stack(tomo5, renamed)
        if not stack:
            self.log_fn(f"No .mrc stack found for {tomo5} ({renamed}).", "fail")
            return
        # Login shell so `module` is available. Load the 3dmod/imod module, view
        # with WARP_FORCE_MRC_FLOAT32 (IMOD can't read Warp float16), then unload
        # once 3dmod is closed. Popen keeps the GUI responsive meanwhile.
        cmd = (f'module load 3dmod 2>/dev/null || module load imod 2>/dev/null; '
               f'WARP_FORCE_MRC_FLOAT32=1 3dmod "{stack}"; '
               f'module unload 3dmod 2>/dev/null || module unload imod 2>/dev/null')
        try:
            subprocess.Popen(["bash", "-lc", cmd])
            self.log_fn(f"3dmod {stack}  (module load 3dmod → unload on close)", "info")
        except OSError as e:
            self.log_fn(f"Could not launch 3dmod: {e}", "fail")

    def _open_all_3dmod(self):
        """Open every non-excluded tilt series in ONE 3dmod window."""
        stacks = []
        for tomo5, renamed, _excl, _tilts in self.rows:
            if self.project.is_series_excluded(renamed):
                continue
            s = self.project.find_series_stack(tomo5, renamed)
            if s:
                stacks.append(str(s))
        if not stacks:
            self.log_fn("No series stacks found to open in 3dmod.", "fail")
            return
        quoted = " ".join(f'"{s}"' for s in stacks)
        cmd = (f'module load 3dmod 2>/dev/null || module load imod 2>/dev/null; '
               f'WARP_FORCE_MRC_FLOAT32=1 3dmod {quoted}; '
               f'module unload 3dmod 2>/dev/null || module unload imod 2>/dev/null')
        try:
            subprocess.Popen(["bash", "-lc", cmd])
            self.log_fn(f"3dmod: opened {len(stacks)} tilt series in one window.", "info")
        except OSError as e:
            self.log_fn(f"Could not launch 3dmod: {e}", "fail")

    def _save(self):
        mapping = {}
        quarantine = []
        bad_tilt_lines = []
        for tomo5, renamed, excl, tilts in self.rows:
            if excl.isChecked():
                quarantine.append(renamed)
                continue
            t = tilts.text().strip().rstrip(",")
            if t:
                mapping[renamed] = t
                bad_tilt_lines.append(f"{renamed}: {t}")
        if not mapping and not quarantine:
            QMessageBox.information(self, "Nothing to save",
                                   "No series excluded and no tilts entered.")
            return
        msg = (f"Write {len(mapping)} tilt-exclusion line(s) to exclusion_list.txt"
               + (f" and quarantine {len(quarantine)} whole series "
                  f"(mdoc→mdocs/bad/, .eer→frames/bad/)" if quarantine else "")
               + "?")
        if QMessageBox.question(self, "Save exclusions?", msg) != QMessageBox.Yes:
            return
        if mapping:
            self.project.write_manual_exclusions(mapping)
            self.log_fn(f"exclusion_list.txt updated: {len(mapping)} series with "
                        f"tilt exclusions ({'; '.join(bad_tilt_lines[:6])}"
                        + (" …" if len(bad_tilt_lines) > 6 else "") + ").", "ok")
        for renamed in quarantine:
            nm, ne = self.project.quarantine_series(renamed)
            self.log_fn(f"Excluded series {renamed}: {nm} mdoc, {ne} .eer → bad/.",
                        "warning")
        self._populate()

    # ---- tilt-series groups (membership edited visually, here) ----
    def _new_group(self):
        name, ok = QInputDialog.getText(self, "New tilt-series group", "Group name:")
        name = name.strip()
        if not ok or not name:
            return
        if name == ProjectState.ALL_GROUP or name in self.groups["groups"]:
            QMessageBox.information(self, "Exists",
                                    "A group with that name already exists.")
            return
        self.groups["groups"][name] = []
        self.project.save_groups(self.groups)
        self._populate()        # re-render so every cell gets the new group's checkbox
        self.log_fn(f"Created tilt-series group '{name}'. Tick series to add them; "
                    f"set it active from the main window.", "ok")

    def _toggle_group(self, gname, renamed, state):
        members = self.groups["groups"].setdefault(gname, [])
        if state:                       # Qt.Checked
            if renamed not in members:
                members.append(renamed)
        elif renamed in members:
            members.remove(renamed)
        self.project.save_groups(self.groups)


# ===========================================================================
# Tilt-series groups manager — define named subsets to process
# ===========================================================================
class GroupsDialog(QDialog):
    """Define named groups of tilt series and pick which one the pipeline runs
    on. 'All tilt series' = no restriction. Active group injects --input_data
    into the WarpTools processing steps. On accept, result_groups holds the new
    {'active':…, 'groups':{…}}."""

    ALL = ProjectState.ALL_GROUP

    def __init__(self, parent, project, groups):
        super().__init__(parent)
        self.project = project
        # working copy (cancel discards)
        self.groups = {"active": groups.get("active", self.ALL),
                       "groups": {k: list(v) for k, v in groups.get("groups", {}).items()}}
        self.result_groups = None
        self.available = project.available_tiltseries()
        self._loading = False
        self.setWindowTitle("Tilt-series groups")
        self.resize(720, 560)

        v = QVBoxLayout(self)
        v.addWidget(QLabel(
            "Define subsets of tilt series and set one ACTIVE. The active group's "
            "series are processed by the WarpTools steps (via --input_data); "
            "'All tilt series' processes everything. Great for optimizing on a few "
            "series, then switching to All."))
        if not self.available:
            v.addWidget(QLabel("(No tilt series found yet — run rename/sort/ts_import "
                               "first so there are mdocs or tomostar to group.)"))

        split = QSplitter(Qt.Horizontal)
        # left: group names
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(QLabel("Groups (★ = active)"))
        self.group_list = QListWidget()
        self.group_list.currentRowChanged.connect(self._on_select_group)
        lv.addWidget(self.group_list)
        row = QHBoxLayout()
        for label, slot in (("New", self._new), ("Rename", self._rename),
                            ("Delete", self._delete), ("Set active", self._set_active)):
            b = QPushButton(label)
            b.clicked.connect(slot)
            row.addWidget(b)
        lv.addLayout(row)
        split.addWidget(left)

        # right: tilt series membership
        right = QWidget()
        rv = QVBoxLayout(right)
        self.member_header = QLabel("Tilt series in group")
        rv.addWidget(self.member_header)
        self.series_list = QListWidget()
        self.series_list.itemChanged.connect(self._on_member_toggled)
        rv.addWidget(self.series_list)
        split.addWidget(right)
        split.setSizes([240, 480])
        v.addWidget(split, 1)

        bottom = QHBoxLayout()
        bottom.addStretch(1)
        save = QPushButton("Save & Close")
        save.clicked.connect(self._save)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        bottom.addWidget(save)
        bottom.addWidget(cancel)
        v.addLayout(bottom)

        self._refresh_group_list()

    def _names(self):
        return [self.ALL] + sorted(self.groups["groups"])

    def _refresh_group_list(self, select=None):
        active = self.groups["active"]
        self.group_list.blockSignals(True)
        self.group_list.clear()
        for name in self._names():
            self.group_list.addItem(("★ " if name == active else "    ") + name)
        self.group_list.blockSignals(False)
        names = self._names()
        target = select if select in names else (
            self.group_list.currentRow() if self.group_list.currentRow() >= 0 else 0)
        idx = names.index(select) if select in names else (
            target if isinstance(target, int) else 0)
        self.group_list.setCurrentRow(min(idx, len(names) - 1))

    def _current_name(self):
        row = self.group_list.currentRow()
        names = self._names()
        return names[row] if 0 <= row < len(names) else None

    def _on_select_group(self, _row):
        name = self._current_name()
        if name is None:
            return
        is_all = name == self.ALL
        members = set(self.available if is_all else self.groups["groups"].get(name, []))
        self.member_header.setText(
            f"Tilt series in '{name}'  ({'all' if is_all else len(members)} selected)"
            + ("  — read-only" if is_all else ""))
        self._loading = True
        self.series_list.clear()
        for s in self.available:
            it = QListWidgetItem(s)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if s in members else Qt.Unchecked)
            if is_all:
                it.setFlags(it.flags() & ~Qt.ItemIsEnabled)
            self.series_list.addItem(it)
        self._loading = False

    def _on_member_toggled(self, _item):
        if self._loading:
            return
        name = self._current_name()
        if name is None or name == self.ALL:
            return
        chosen = [self.series_list.item(i).text()
                  for i in range(self.series_list.count())
                  if self.series_list.item(i).checkState() == Qt.Checked]
        self.groups["groups"][name] = chosen
        self.member_header.setText(f"Tilt series in '{name}'  ({len(chosen)} selected)")

    def _new(self):
        name, ok = QInputDialog.getText(self, "New group", "Group name:")
        name = name.strip()
        if not ok or not name:
            return
        if name == self.ALL or name in self.groups["groups"]:
            QMessageBox.information(self, "Exists", "A group with that name already exists.")
            return
        self.groups["groups"][name] = []
        self._refresh_group_list(select=name)

    def _rename(self):
        name = self._current_name()
        if not name or name == self.ALL:
            return
        new, ok = QInputDialog.getText(self, "Rename group", "New name:", text=name)
        new = new.strip()
        if not ok or not new or new == name:
            return
        if new == self.ALL or new in self.groups["groups"]:
            QMessageBox.information(self, "Exists", "That name is taken.")
            return
        self.groups["groups"][new] = self.groups["groups"].pop(name)
        if self.groups["active"] == name:
            self.groups["active"] = new
        self._refresh_group_list(select=new)

    def _delete(self):
        name = self._current_name()
        if not name or name == self.ALL:
            return
        if QMessageBox.question(self, "Delete group?",
                                f"Delete group '{name}'?") != QMessageBox.Yes:
            return
        self.groups["groups"].pop(name, None)
        if self.groups["active"] == name:
            self.groups["active"] = self.ALL
        self._refresh_group_list(select=self.ALL)

    def _set_active(self):
        name = self._current_name()
        if not name:
            return
        if name != self.ALL and not self.groups["groups"].get(name):
            QMessageBox.information(self, "Empty group",
                                    "This group has no tilt series selected.")
            return
        self.groups["active"] = name
        self._refresh_group_list(select=name)

    def _save(self):
        self.result_groups = self.groups
        self.accept()


# ===========================================================================
# Processing History — per-dataset job flowchart with clickable inputs/outputs
# ===========================================================================
class ProcessingHistory(QDialog):
    """Per-dataset job history as a flowchart (oldest → newest). Each job shows its
    name + the non-template parameters used (bullets). Click a job for its full
    command and its input/output files — every file opens with one click (3dmod for
    stacks, gedit for text). 'Load these settings' restores that run's command into
    the stage form as a manual override (it does NOT change Template defaults)."""

    def __init__(self, parent, project, open_file_fn, load_settings_fn):
        super().__init__(parent)
        self.project = project
        self.open_file = open_file_fn
        self.load_settings = load_settings_fn
        self.setWindowTitle("Processing History")
        self.resize(1060, 780)
        self.history = project.load_history()

        v = QVBoxLayout(self)
        head = QHBoxLayout()
        self.header = QLabel("")
        head.addWidget(self.header)
        head.addStretch(1)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._reload)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        head.addWidget(refresh)
        head.addWidget(close)
        v.addLayout(head)

        split = QSplitter(Qt.Horizontal)
        self.flow_box = QVBoxLayout()
        self.flow_box.setAlignment(Qt.AlignTop)
        flow_inner = QWidget()
        flow_inner.setLayout(self.flow_box)
        flow_scroll = QScrollArea()
        flow_scroll.setWidgetResizable(True)
        flow_scroll.setWidget(flow_inner)
        split.addWidget(flow_scroll)

        self.detail_box = QVBoxLayout()
        self.detail_box.setAlignment(Qt.AlignTop)
        det_inner = QWidget()
        det_inner.setLayout(self.detail_box)
        det_scroll = QScrollArea()
        det_scroll.setWidgetResizable(True)
        det_scroll.setWidget(det_inner)
        split.addWidget(det_scroll)
        split.setSizes([470, 590])
        v.addWidget(split, 1)

        self._render_flow()
        self.detail_box.addWidget(QLabel("← click a job to see its inputs / outputs."))

    def _reload(self):
        self.history = self.project.load_history()
        self._clear(self.detail_box)
        self._render_flow()

    @staticmethod
    def _clear(box):
        while box.count():
            w = box.takeAt(0).widget()
            if w:
                w.deleteLater()

    @staticmethod
    def _status_colour(code):
        if code == 0:
            return "#27ae60", "OK"
        if code in (None, "", "None"):
            return "#e0a850", "running…"
        return "#e24b4a", f"exit {code}"

    def _render_flow(self):
        self._clear(self.flow_box)
        self.header.setText(f"{len(self.history)} job(s) run in this dataset "
                            f"(oldest → newest). Click one for details.")
        if not self.history:
            self.flow_box.addWidget(QLabel("No jobs recorded yet — pipeline runs are "
                                           "logged here as you execute them."))
            return
        for i, rec in enumerate(self.history):
            colour, status = self._status_colour(rec.get("exit_code"))
            card = QFrame()
            card.setFrameShape(QFrame.StyledPanel)
            card.setStyleSheet(f"QFrame{{border:1px solid #333;border-left:4px solid "
                               f"{colour};border-radius:4px;}}")
            cv = QVBoxLayout(card)
            cv.setContentsMargins(8, 6, 8, 6)
            hdr = QPushButton(f"{rec.get('label', rec.get('stage_id',''))}   "
                              f"[{status}]   {rec.get('ts','')}")
            hdr.setStyleSheet("text-align:left;border:none;font-weight:600;")
            hdr.clicked.connect(lambda _=False, r=rec: self._show_detail(r))
            cv.addWidget(hdr)
            for p in rec.get("params", [])[:12]:
                lab = QLabel("    • " + p)
                lab.setStyleSheet("color:#bbb;font-size:11px;")
                cv.addWidget(lab)
            self.flow_box.addWidget(card)
            if i < len(self.history) - 1:
                arr = QLabel("↓")
                arr.setStyleSheet("color:#666;margin-left:12px;")
                self.flow_box.addWidget(arr)

    def _file_buttons_for_dir(self, rel):
        out = []
        base = self.project.root / rel
        folder = QPushButton(f"📂  {rel}/")
        folder.setStyleSheet("text-align:left;")
        folder.clicked.connect(lambda _=False, p=str(base): self.open_file(p))
        out.append(folder)
        if base.is_dir():
            try:
                entries = sorted(
                    itertools.islice((e for e in base.iterdir() if e.is_file()), 41),
                    key=lambda e: e.name)
            except OSError:
                entries = []
            for e in entries[:40]:
                fb = QPushButton("      " + e.name)
                fb.setStyleSheet("text-align:left;border:none;color:#9bc0ff;")
                fb.clicked.connect(lambda _=False, p=str(e): self.open_file(p))
                out.append(fb)
            if len(entries) > 40:
                more = QLabel("      … (first 40 shown — use the folder button for all)")
                more.setStyleSheet("color:#888;font-size:10px;")
                out.append(more)
        else:
            na = QLabel("      (folder not present)")
            na.setStyleSheet("color:#888;font-size:10px;")
            out.append(na)
        return out

    @staticmethod
    def _section(text):
        lab = QLabel(text)
        lab.setStyleSheet("font-size:12px;font-weight:700;color:#cfcfcf;margin-top:8px;")
        return lab

    def _show_detail(self, rec):
        self._clear(self.detail_box)
        title = QLabel(rec.get("label", rec.get("stage_id", "")))
        title.setStyleSheet("font-size:14px;font-weight:700;")
        self.detail_box.addWidget(title)
        _, status = self._status_colour(rec.get("exit_code"))
        meta = QLabel(f"{rec.get('ts','')}   ·   {status}")
        meta.setStyleSheet("color:#888;font-size:11px;")
        self.detail_box.addWidget(meta)

        self.detail_box.addWidget(self._section("Command"))
        cmd = QPlainTextEdit(rec.get("command", ""))
        cmd.setReadOnly(True)
        cmd.setFixedHeight(96)
        cmd.setStyleSheet(f"font-family:{MONO};font-size:11px;")
        self.detail_box.addWidget(cmd)

        self.detail_box.addWidget(self._section("Inputs"))
        ins = rec.get("inputs") or []
        if not ins:
            self.detail_box.addWidget(QLabel("    (none recorded)"))
        for rel in ins:
            for w in self._file_buttons_for_dir(rel):
                self.detail_box.addWidget(w)

        self.detail_box.addWidget(self._section("Outputs"))
        outs = rec.get("outputs") or []
        if not outs:
            self.detail_box.addWidget(QLabel("    (none recorded)"))
        for rel in outs:
            for w in self._file_buttons_for_dir(rel):
                self.detail_box.addWidget(w)

        load = QPushButton("Load these settings into the form (manual override)")
        load.setToolTip("Restores this run's exact command into its stage; does NOT "
                        "change the Template defaults.")
        load.clicked.connect(lambda _=False, r=rec:
                             self.load_settings(r.get("stage_id"), r.get("command", "")))
        self.detail_box.addWidget(load)


# ===========================================================================
# Card canvas (Phase 2) — a QGraphicsView rendering canvas_layout(). Read-only:
# real job cards + greyed ghost cards for the default (un-run) pipeline. Clicking
# a card selects its stage in the shared form (like clicking a stage row). Run /
# fork controls are Phase 3.
# ===========================================================================
# (fill, border) per status. Ghost = dim + dashed; others tint by outcome.
_CARD_STYLE = {
    "ghost":     ("#242424", "#555555"),
    "queued":    ("#26313a", "#3a6ea5"),
    "running":   ("#3a3320", "#e0a850"),
    "completed": ("#1d3326", "#27ae60"),
    "failed":    ("#3a2320", "#c0392b"),
}


class _CardItem(QGraphicsRectItem):
    """A single card. Holds its node dict and routes clicks to the canvas."""
    def __init__(self, node, canvas):
        super().__init__(0, 0, node["w"], node["h"])
        self._node = node
        self._canvas = canvas
        self.setPos(node["x"], node["y"])
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.RightButton:
            self._canvas._menu(self._node, ev.screenPos())
            ev.accept()
            return
        self._canvas._pick(self._node)
        super().mousePressEvent(ev)


class _DetailsChip(QGraphicsRectItem):
    """Small 'Details ▸' button on a card; opens the Details side pane. Its own
    mousePressEvent handles the click so it doesn't also select the card."""
    def __init__(self, node, canvas, x, y, w=64, h=18):
        super().__init__(0, 0, w, h)
        self._node = node
        self._canvas = canvas
        self.setPos(x, y)
        self.setBrush(QBrush(QColor("#2a3340")))
        self.setPen(QPen(QColor("#3a6ea5")))
        self.setCursor(Qt.PointingHandCursor)
        t = QGraphicsSimpleTextItem("Details ▸", self)
        t.setBrush(QColor("#9ec5ff"))
        f = QFont()
        f.setPointSize(8)
        t.setFont(f)
        t.setPos(7, 2)

    def mousePressEvent(self, ev):
        self._canvas._details(self._node)
        ev.accept()


class JobCanvas(QWidget):
    def __init__(self, root_getter, on_pick, on_details=None, on_menu=None,
                 parent=None):
        super().__init__(parent)
        self._root_getter = root_getter      # callable -> project_root str
        self._on_pick = on_pick              # callable(stage_id)
        self._on_details = on_details        # callable(node) | None
        self._on_menu = on_menu              # callable(node, global_qpoint) | None
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setBackgroundBrush(QColor("#0e0e0e"))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view)

    def refresh(self):
        self.scene.clear()
        try:
            store = load_jobs(self._root_getter())
        except Exception:
            store = {"jobs": {}}
        nodes, edges = canvas_layout(store)
        index = {n["id"]: n for n in nodes}

        edge_pen = QPen(QColor("#4a4a4a"))
        edge_pen.setWidth(2)
        for src, dst in edges:
            a, b = index.get(src), index.get(dst)
            if not a or not b:
                continue
            self.scene.addLine(a["x"] + a["w"] / 2, a["y"] + a["h"],
                               b["x"] + b["w"] / 2, b["y"], edge_pen)

        for n in nodes:
            self._add_card(n)

        rect = self.scene.itemsBoundingRect()
        self.scene.setSceneRect(rect.adjusted(-40, -40, 40, 40))

    def _add_card(self, n):
        fill, border = _CARD_STYLE.get(n["status"], _CARD_STYLE["ghost"])
        ghost = n["is_ghost"]
        item = _CardItem(n, self)
        item.setBrush(QBrush(QColor(fill)))
        pen = QPen(QColor(border))
        pen.setWidth(2)
        if ghost:
            pen.setStyle(Qt.PenStyle.DashLine)
        item.setPen(pen)
        self.scene.addItem(item)

        def text(s, x, y, pt, colour, bold=False):
            t = QGraphicsSimpleTextItem(s, item)
            t.setBrush(QColor(colour))
            f = QFont()
            f.setPointSize(pt)
            f.setBold(bold)
            t.setFont(f)
            t.setPos(x, y)
            return t

        # group tag (tiny) · friendly title (bold) · raw command · status/summary
        text(n.get("group", ""), 11, 6, 8, "#6f6f6f")
        text(n.get("title", n["label"]), 11, 20, 11,
             "#8a8a8a" if ghost else "#ececec", bold=True)
        text(n["stage_id"], 11, 40, 8, "#6f6f6f")     # raw command, for power users
        if ghost:
            sub = "not built"
        else:
            st = summary_text(n["summary"])
            sub = n["status"] + (f" · {st}" if st else "")
        text(sub[:36], 11, 56, 9, "#7d7d7d")
        if not ghost:
            text(n["id"], n["w"] - 42, 6, 8, "#9ec5ff")

        # Details chip (only if the canvas has a details handler)
        if self._on_details is not None:
            self.scene.addItem(_DetailsChip(n, self, n["x"] + n["w"] - 72,
                                            n["y"] + n["h"] - 24))

    def _pick(self, node):
        try:
            self._on_pick(node["stage_id"])
        except Exception:
            pass

    def _details(self, node):
        if self._on_details is not None:
            try:
                self._on_details(node)
            except Exception:
                pass

    def _menu(self, node, global_pos):
        if self._on_menu is not None:
            try:
                self._on_menu(node, global_pos)
            except Exception:
                pass


# ===========================================================================
# Main window
# ===========================================================================
class Tomogration(QMainWindow):
    def __init__(self, project_root):
        super().__init__()
        self.project_root = project_root
        self.project = ProjectState(project_root)
        self.setWindowTitle(f"tomogration — {project_root}")
        self.resize(1500, 900)          # 50% larger so panels/scrollbars aren't clipped
        self._child_windows = []        # keep non-modal inspectors alive

        self.runner = ProcessRunner()
        self.runner.line.connect(self._log)
        self.runner.finished.connect(self._on_finished)

        self.queue = []          # list of (label, command, stage_id)
        self.current = None      # active stage form state
        # Persisted per-stage parameter edits: {stage_id: {param_name: value}}. A user's
        # edits stick (across stage switches AND restarts) until they change them again,
        # hit "Reset defaults", or a dynamic default (e.g. a newer AreTomo version) wins.
        # Per-stage param edits live in the PROJECT dir (on ceph, shared) not
        # ~/.tomogration.json ($HOME is local per VM), so edits follow the dataset
        # across machines. Falls back to the old global config once for migration.
        self._param_store = self._load_param_store()
        # Migration: the export 3D/2D choice used to emit "" for 3D (so --3d was
        # missing and had to be typed by hand). Any persisted "" is rewritten to
        # the real flag so the fix takes even on machines with an old stored value.
        _ep = self._param_store.get("ts_export_particles")
        if isinstance(_ep, dict) and _ep.get("relion_format") == "":
            _ep["relion_format"] = "--3d"
        # Debounce disk writes: a single-shot timer flushes _param_store to config
        # ~0.6s after the last edit (so typing doesn't hammer the JSON).
        self._persist_timer = QTimer(self)
        self._persist_timer.setSingleShot(True)
        self._persist_timer.setInterval(600)
        self._persist_timer.timeout.connect(self._persist_param_store)
        self._active_stage = None
        self._active_cmd = ""
        self._active_job_id = None    # set while a card-view job (not a stage) runs
        self._failed_file = None
        self._attempt = 1
        self._max_retries = 20
        # How WarpTools is invoked (it's behind a module on this cluster). The
        # leading 'WarpTools' in every Warp step's command is replaced by this.
        # Default launcher for WarpTools on this cluster: a proper conda env
        # (the `warp` env) — a real conda install wires its own CUDA/.NET libs so
        # the GPU worker resolves them (the from-source shared build does not).
        self.warp_launch = self._load_config().get(
            "warp_launch", "module load miniconda/latest && conda activate warp && WarpTools")
        # How to launch warp-tm-vis (github.com/warpem/warp-tm-vis), the pick
        # viewer: it overlays template-match picks + correlation volumes on the
        # tomograms. It lives in its OWN conda env (NOT the warp env; not the same
        # as `uvx`, which is only on some VMs). Conda envs are on ceph (shared), so
        # `conda activate tmvis && warp-tm-vis` works from any VM. Configurable via
        # Tools if the env is named differently or you prefer `uvx warp-tm-vis`.
        self.tm_vis_launch = self._load_config().get(
            "tm_vis_launch", "module load miniconda/latest && conda activate tmvis && warp-tm-vis")

        self._last_was_progress = False     # terminal progress-line collapsing

        # Tilt-series groups: process only a named subset (for parameter
        # optimization on a few series, then apply to all). Active group ->
        # --input_data lists injected into the WarpTools steps that accept it.
        self._groups = self.project.load_groups()
        self._group_inputs = None           # {"ts": path, "fs": path} or None for All
        self._refresh_group_inputs()

        # Rich left-panel docs, keyed by stage id (tomogration_docs.json next to
        # this file). Falls back to each stage's inline docs dict where absent.
        self._docs = self._load_docs_json()

        # Cards: each region is a bordered, tinted QFrame so they read as distinct.
        self.setStyleSheet("""
            QFrame[card="true"] { background:#191919; border:1px solid #333; border-radius:8px; }
            QFrame#docsCard  { background:#1d1d1d; }
            QFrame#dirCard, QFrame#rightCard, QFrame#queueCard { background:#0e0e0e; }
            QFrame[card="true"] QScrollArea { border:none; background:transparent; }
            QSplitter::handle { background:#3a3a3a; border-radius:2px; }
            QSplitter::handle:hover { background:#4f4f4f; }
        """)

        # Shared registries the per-column stage lists populate.
        self.node_buttons = {}      # stage_id -> status dot
        self.stage_buttons = {}     # stage_id -> stage button
        self.output_combos = {}     # stage_id -> output-iteration combo
        self.dir_buttons = {}       # rel_dir -> directory-overview button

        self._build_menus()

        # ---- persistent leaf panels (built once; the two view layouts just
        # arrange these same widgets differently, so no state is duplicated) ----
        self.docs_card = self._panel("docsCard", "Per-job information",
                                     self._build_docs_panel())
        # Stage picker: page 0 = classic 3-column lists, page 1 = card canvas.
        # Both drive the SAME job builder, so the toggle only changes HOW you
        # pick a node, nothing downstream.
        self.job_stack = QStackedWidget()
        self.job_stack.addWidget(
            self._panel("listsCard", "Pipeline jobs", self._build_job_lists()))
        self.canvas = JobCanvas(lambda: self.project_root, self._canvas_pick,
                                on_details=self._show_card_details,
                                on_menu=self._card_menu)
        self.job_stack.addWidget(
            self._panel("canvasCard", "Workflow graph", self.canvas))
        self._build_align_and_command()   # sets self.align_list_card + self.command_card
        self.details_card = self._panel("detailsCard", "Job details",
                                        self._build_details_panel())
        self.dir_card = self._panel("dirCard", "Directory overview",
                                    self._build_directory_overview())
        self.terminal_card = self._panel("rightCard", "Terminal",
                                          self._build_terminal_panel())
        self.queue_card = self._panel("queueCard", "Jobs queue",
                                      self._build_queue_panel())
        # Stash keeps panels parented (and hidden) while they're not in the live
        # layout — a parentless shown QWidget would pop up as its own window.
        self._stash = QWidget()
        self._stash.hide()
        self._panels = [self.docs_card, self.job_stack, self.align_list_card,
                        self.command_card, self.details_card, self.dir_card,
                        self.terminal_card, self.queue_card]

        central = QWidget()
        cv = QVBoxLayout(central)
        cv.setContentsMargins(8, 6, 8, 8)
        cv.setSpacing(6)
        cv.addWidget(self._build_root_bar())
        self._layout_host = QWidget()          # holds the current mode's main splitter
        self._layout_host_v = QVBoxLayout(self._layout_host)
        self._layout_host_v.setContentsMargins(0, 0, 0, 0)
        cv.addWidget(self._layout_host, 1)
        self.setCentralWidget(central)

        start_mode = "canvas" if self._load_config().get("view_mode") == "canvas" else "lists"
        self._apply_layout(start_mode)

        self._refresh_status_dots()
        self._select_stage(self._stage_by_id("ts_reconstruct") or STAGES[0])
        self._save_config({**self._load_config(), "last_root": self.project_root})

    # ---- menu bar (affordances live here to keep the center panel narrow) ----
    def _build_menus(self):
        mb = self.menuBar()
        proj = mb.addMenu("Project")
        proj.addAction("Init dirs", self._init_dirs)
        proj.addAction("Sort files…", self._sort_files)
        proj.addAction("Browse root…", self._browse_root)
        tools = mb.addMenu("Tools")
        tools.addAction("Processing History", self._open_history)
        tools.addAction("Repair mdocs", self._repair_mdocs)
        tools.addAction("Tilt Inspector", self._open_tilt_inspector)
        tools.addAction("Positions Inspector", self._open_inspector)
        tools.addAction("AreTomo runs", self._open_aretomo_versions)
        tools.addAction("Open file…", self._open_file_in_editor)
        tools.addAction("ChimeraX…", self._launch_chimerax)
        tools.addSeparator()
        tools.addAction("Set WarpTools launch command…", self._set_warp_launch)
        tools.addAction("Set warp-tm-vis launch command…", self._set_tm_vis_launch)
        view = mb.addMenu("View")
        self._act_canvas = view.addAction("Card (graph) view", self._toggle_view)
        self._act_canvas.setCheckable(True)
        view.addSeparator()
        view.addAction("Refresh status", self._refresh_status_dots)
        view.addAction("Tilt-series groups…", self._open_groups)

    # ---- card canvas (Phase 2) ----
    def _canvas_pick(self, stage_id):
        """Clicking a card selects its stage in the shared form (read-only)."""
        spec = self._stage_by_id(stage_id)
        if spec:
            self._select_stage(spec)

    def _refresh_canvas(self):
        """Rebuild the canvas from the job store. Safe to call before the canvas
        exists (early in construction) and is the hook _finalize_job calls."""
        if getattr(self, "canvas", None) is not None:
            self.canvas.refresh()

    def _toggle_view(self):
        self._apply_layout("lists" if self._view_mode == "canvas" else "canvas")

    @staticmethod
    def _clear_box(layout):
        while layout.count():
            it = layout.takeAt(0)
            w = it.widget()
            if w is not None:
                w.setParent(None)

    def _apply_layout(self, mode):
        """Arrange the persistent leaf panels for the chosen view. LIST mode is the
        classic 3-column layout; CARD mode gives the canvas ~2/3 on the left (with a
        collapsible Details pane) and stacks the job builder over the terminal on the
        right. Same widgets, rebuilt containers — so no state is duplicated."""
        self._view_mode = mode
        canvas = (mode == "canvas")
        # Park every leaf in the stash first so none becomes an orphan top-level
        # window while we swap containers, then drop the previous main splitter.
        for w in self._panels:
            w.setParent(self._stash)
        self._clear_box(self._layout_host_v)
        self.job_stack.setCurrentIndex(1 if canvas else 0)

        if canvas:
            self._canvas_split = QSplitter(Qt.Horizontal)
            self._canvas_split.setHandleWidth(8)
            self._canvas_split.addWidget(self.job_stack)
            self._canvas_split.addWidget(self.details_card)
            self.details_card.setVisible(False)          # revealed by "Details"
            rightcol = QSplitter(Qt.Vertical)
            rightcol.setHandleWidth(8)
            rightcol.addWidget(self.command_card)
            rightcol.addWidget(self.terminal_card)
            rightcol.setSizes([460, 380])
            main = QSplitter(Qt.Horizontal)
            main.setHandleWidth(8)
            main.addWidget(self._canvas_split)
            main.addWidget(rightcol)
            main.setSizes([1080, 520])                   # ~2/3 canvas, 1/3 right
        else:
            self._canvas_split = None
            form_area = QSplitter(Qt.Vertical)
            form_area.setHandleWidth(8)
            form_area.addWidget(self.align_list_card)
            form_area.addWidget(self.command_card)
            form_area.setSizes([240, 430])
            work = QSplitter(Qt.Horizontal)
            work.setHandleWidth(8)
            work.addWidget(self.job_stack)
            work.addWidget(form_area)
            work.setSizes([330, 470])
            left = QSplitter(Qt.Vertical)
            left.setHandleWidth(8)
            left.addWidget(self.docs_card)
            left.addWidget(work)
            left.setSizes([320, 580])
            rightcol = QSplitter(Qt.Vertical)
            rightcol.setHandleWidth(8)
            rightcol.addWidget(self.dir_card)
            rightcol.addWidget(self.terminal_card)
            rightcol.addWidget(self.queue_card)
            rightcol.setSizes([240, 470, 120])
            main = QSplitter(Qt.Horizontal)
            main.setHandleWidth(8)
            main.addWidget(left)
            main.addWidget(rightcol)
            main.setSizes([1000, 500])

        self._layout_host_v.addWidget(main)
        if hasattr(self, "_act_canvas"):
            self._act_canvas.setChecked(canvas)
        if canvas:
            self._refresh_canvas()
        self._save_config({**self._load_config(), "view_mode": mode})

    # ---- card "Details" side pane (card view) ----
    def _build_details_panel(self):
        self.details_box = QVBoxLayout()
        self.details_box.setAlignment(Qt.AlignTop)
        self.details_box.setContentsMargins(12, 8, 18, 8)
        self.details_box.setSpacing(6)
        ph = QLabel("Click “Details” on a card to inspect its inputs, outputs "
                    "and parameters.")
        ph.setStyleSheet("color:#888;font-size:12px;")
        ph.setWordWrap(True)
        self.details_box.addWidget(ph)
        inner = QWidget()
        inner.setLayout(self.details_box)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(inner)
        return scroll

    @staticmethod
    def _details_heading(text):
        lab = QLabel(text)
        lab.setStyleSheet("color:#9ec5ff;font-size:11px;font-weight:700;margin-top:6px;")
        return lab

    def _open_dir_button(self, label, rel):
        b = QPushButton(label)
        b.setStyleSheet("text-align:left;padding:3px 8px;")
        b.setToolTip(f"Open {rel} in the file manager")
        b.clicked.connect(lambda _=False, r=rel: self._open_dir(r))
        return b

    def _add_pattern_hint(self, verb, pattern):
        """Dim 'searches: *.tomostar' line under a dir button (skips if unknown)."""
        if not pattern:
            return
        lab = QLabel(f"      {verb}:  {pattern}")
        lab.setStyleSheet("color:#7c7c7c;font-size:10px;")
        lab.setWordWrap(True)
        self.details_box.addWidget(lab)

    def _show_card_details(self, node):
        """Populate + reveal the Details pane beside the canvas for a clicked card.
        Directory access is via lazy 'Open dir' buttons (no enumeration on show —
        ceph scandir is what has crashed this app before)."""
        if getattr(self, "_view_mode", "lists") != "canvas":
            return
        self._clear_box(self.details_box)
        stage_id = node.get("stage_id")
        spec = self._stage_by_id(stage_id)

        title = QLabel(node.get("title", node.get("label", stage_id)))
        title.setStyleSheet("font-size:15px;font-weight:700;color:#ececec;")
        title.setWordWrap(True)
        self.details_box.addWidget(title)

        if node.get("is_ghost"):
            meta = f"{node.get('group', '')} · {stage_id} · not built yet"
        else:
            meta = f"{node.get('group', '')} · {stage_id} · {node.get('status', '')}  ({node.get('id')})"
        ml = QLabel(meta)
        ml.setStyleSheet("color:#9a9a9a;font-size:11px;")
        ml.setWordWrap(True)
        self.details_box.addWidget(ml)

        st = summary_text(node.get("summary", {}))
        if st:
            s = QLabel(st)
            s.setStyleSheet("color:#cfcfcf;font-size:12px;")
            s.setWordWrap(True)
            self.details_box.addWidget(s)

        ins, outs = STAGE_IO.get(stage_id, ([], []))
        if ins:
            self.details_box.addWidget(self._details_heading("INPUTS"))
            for rel in ins:
                self.details_box.addWidget(self._open_dir_button(f"📂  {rel}", rel))
                self._add_pattern_hint("searches", DIR_FILE_HINTS.get(rel))
        self.details_box.addWidget(self._details_heading("OUTPUTS"))
        if not node.get("is_ghost") and node.get("id"):
            outrel = f"jobs/{node['id']}"      # a real job writes into its own jobs/<id>
            self.details_box.addWidget(self._open_dir_button(f"📂  {outrel}", outrel))
            self._add_pattern_hint("writes", DIR_FILE_HINTS.get(outs[0]) if outs else None)
        else:
            for rel in outs:
                self.details_box.addWidget(self._open_dir_button(f"📂  {rel}", rel))
                self._add_pattern_hint("writes", DIR_FILE_HINTS.get(rel))

        if spec:
            self.details_box.addWidget(self._details_heading("ACTIONS"))
            if node.get("is_ghost"):
                build = QPushButton("▶ Build & run job")
                build.setToolTip("Create a job instance for this stage (input auto-wired "
                                 "to the newest upstream job) and run it.")
                build.clicked.connect(lambda _=False, sid=stage_id: self._build_job(sid))
                self.details_box.addWidget(build)
            else:
                jid = node.get("id")
                run = QPushButton("▶ Run / re-run")
                run.clicked.connect(lambda _=False, j=jid: self._run_job(j))
                self.details_box.addWidget(run)
                fork = QPushButton("⑂ Duplicate (fork)")
                fork.clicked.connect(lambda _=False, j=jid: self._fork_job(j))
                self.details_box.addWidget(fork)
            edit = QPushButton("Open in job builder →")
            edit.setToolTip("Load this stage's parameters into the job builder on the right.")
            edit.clicked.connect(lambda _=False, s=spec: self._select_stage(s))
            self.details_box.addWidget(edit)
            if stage_id in ("ts_template_match", "threshold_picks"):
                nap = QPushButton("🔍 View picks (warp-tm-vis)")
                nap.setToolTip("Open this pick set in warp-tm-vis — overlays the picks + "
                               "correlation volumes on the tomograms (edit the command if "
                               "the suffix/paths differ).")
                nap.clicked.connect(lambda _=False, n=node: self._view_picks_tm_vis(n))
                self.details_box.addWidget(nap)

        self.details_card.setVisible(True)
        if getattr(self, "_canvas_split", None) is not None:
            w = max(self._canvas_split.width(), 900)
            self._canvas_split.setSizes([int(w * 0.55), int(w * 0.45)])

    # ---- card actions (Phase 3): build / fork / run / delete ----
    def _effective_params(self, spec):
        """Template defaults overlaid with the user's persisted per-stage edits —
        the values a fresh job of this stage should start from."""
        vals = stage_defaults(spec)
        vals.update(self._param_store.get(spec["id"], {}))
        return vals

    def _card_menu(self, node, global_pos):
        """Right-click menu on a canvas card."""
        menu = QMenu(self)
        sid = node.get("stage_id")
        if node.get("is_ghost"):
            menu.addAction("Build & run job", lambda: self._build_job(sid, run=True))
            menu.addAction("Build (don't run)", lambda: self._build_job(sid, run=False))
            menu.addAction("Open in job builder", lambda: self._canvas_pick(sid))
        else:
            jid = node.get("id")
            menu.addAction("Run / re-run", lambda: self._run_job(jid))
            menu.addAction("Duplicate (fork)", lambda: self._fork_job(jid))
            menu.addAction("Details", lambda: self._show_card_details(node))
            menu.addAction("Open in job builder", lambda: self._canvas_pick(sid))
            menu.addSeparator()
            menu.addAction("Delete job", lambda: self._delete_job(jid))
        menu.exec(global_pos)

    def _build_job(self, stage_id, params=None, run=True, parent=None):
        """Create a job instance for a stage (auto-wiring its input to the newest
        upstream WarpTools job) and optionally run it."""
        spec = self._stage_by_id(stage_id)
        if not spec:
            return None
        store = load_jobs(self.project_root)
        if params is None:
            # prefer the live form if this stage is the one on screen
            if self.current and self.current.get("spec", {}).get("id") == stage_id:
                params = self._values()
            else:
                params = self._effective_params(spec)
        if parent is None:
            parent = default_parent_for(stage_id, store)
        inputs = {"processing": parent} if parent else {}
        job = new_job(self.project_root, stage_id, spec.get("label", stage_id),
                      params, inputs)
        self._log(f"Built {job['id']} · {job['label']}"
                  + (f"  (input ← {parent})" if parent else "  (input ← trunk)"), "ok")
        self._refresh_canvas()
        if run:
            self._run_job(job["id"])
        return job

    def _fork_job(self, job_id):
        """Duplicate a job (same stage, params and input wiring) as a new queued
        job, and open it in the builder so its params can be tweaked before Run."""
        store = load_jobs(self.project_root)
        src = store.get("jobs", {}).get(job_id)
        if not src:
            return
        job = new_job(self.project_root, src["stage_id"],
                      src.get("label", src["stage_id"]) + " (fork)",
                      src.get("params", {}), src.get("inputs", {}))
        self._log(f"Forked {job_id} → {job['id']}. Edit params in the job builder, "
                  f"then right-click → Run.", "ok")
        self._refresh_canvas()
        spec = self._stage_by_id(src["stage_id"])
        if spec:
            self._select_stage(spec)

    def _delete_job(self, job_id):
        if QMessageBox.question(
                self, "Delete job?",
                f"Remove job {job_id} from the workflow?\n\n"
                f"Its output folder (jobs/{job_id}/) is left on disk — delete that "
                f"by hand if you want the space back.") != QMessageBox.Yes:
            return
        if delete_job(self.project_root, job_id):
            self._log(f"Deleted job {job_id}.", "info")
            self._refresh_canvas()

    # ---- helpers ----
    @staticmethod
    def _stage_by_id(stage_id):
        return next((s for s in STAGES if s["id"] == stage_id), None)

    @staticmethod
    def _load_docs_json():
        p = _PKG_DIR / "tomogration_docs.json"
        if p.is_file():
            try:
                d = json.loads(p.read_text())
                d.pop("_meta", None)
                return d
            except (OSError, ValueError):
                pass
        return {}

    def _panel(self, object_name, title, content):
        """Wrap a panel's content in a bordered card with a header bar."""
        frame = QFrame()
        frame.setObjectName(object_name)
        frame.setProperty("card", "true")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(14, 12, 14, 14)
        lay.setSpacing(8)
        header = QLabel(title)
        header.setStyleSheet("font-size:13px;font-weight:700;color:#e6e6e6;"
                             "padding-bottom:6px;border-bottom:1px solid #3a3a3a;")
        lay.addWidget(header)
        lay.addWidget(content, 1)
        return frame

    # ---- LEFT: per-job docs ("Per-job information") ----
    def _build_docs_panel(self):
        self.docs_view = QTextBrowser()
        self.docs_view.setOpenExternalLinks(True)   # ref links open in a browser
        self.docs_view.setStyleSheet("background:#1d1d1d;border:none;color:#dddddd;")
        return self.docs_view

    def _render_docs(self, spec):
        doc = self._docs.get(spec["id"])
        html = render_docs_html(doc) if doc else render_inline_docs_html(spec)
        self.docs_view.setHtml(html)
        self.docs_view.verticalScrollBar().setValue(0)

    # ---- top: always-visible project-root bar ----
    def _build_root_bar(self):
        root_row = QHBoxLayout()
        root_row.setContentsMargins(4, 0, 4, 0)
        root_row.addWidget(QLabel("Project root:"))
        self.root_edit = QLineEdit(self.project_root)
        self.root_edit.setToolTip("Working directory for all stage commands. "
                                  "Type/paste a path and press Enter, or Browse.")
        self.root_edit.returnPressed.connect(
            lambda: self._set_root(self.root_edit.text().strip()))
        root_row.addWidget(self.root_edit, 1)
        set_btn = QPushButton("Set")
        set_btn.clicked.connect(lambda: self._set_root(self.root_edit.text().strip()))
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_root)
        root_row.addWidget(set_btn)
        root_row.addWidget(browse_btn)
        w = QWidget()
        w.setLayout(root_row)
        return w

    # ---- left work column: Tilt curation + Stack preparation job lists ----
    def _build_job_lists(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        bar = QHBoxLayout()
        for label, slot in (("Init dirs", self._init_dirs),
                            ("Sort files", self._sort_files),
                            ("Repair mdocs", self._repair_mdocs)):
            b = QPushButton(label)
            b.clicked.connect(slot)
            bar.addWidget(b)
        grp = QPushButton("⊟ Groups…")
        grp.setToolTip("Define named subsets of tilt series and choose the active one "
                       "(optimize on a few, then run all).")
        grp.clicked.connect(self._open_groups)
        bar.addWidget(grp)
        bar.addStretch(1)
        bw = QWidget()
        bw.setLayout(bar)
        v.addWidget(bw)
        self.group_label = QLabel()
        self.group_label.setWordWrap(True)
        v.addWidget(self.group_label)
        self._update_group_label()
        inner = QWidget()
        iv = QVBoxLayout(inner)
        iv.setAlignment(Qt.AlignTop)
        iv.setContentsMargins(0, 0, 6, 0)
        iv.addWidget(self._build_stage_list("curation"))
        iv.addWidget(self._build_stage_list("stackprep"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        v.addWidget(scroll, 1)
        return w

    # ---- Alignment & Reconstruction list + the command form (built as two
    # separate cards so the layout can place them independently: together in
    # list view, command-only beside the canvas in card view) ----
    def _build_align_and_command(self):
        a_inner = QWidget()
        av = QVBoxLayout(a_inner)
        av.setAlignment(Qt.AlignTop)
        av.setContentsMargins(0, 0, 6, 0)
        av.addWidget(self._build_stage_list("alignrecon"))
        a_scroll = QScrollArea()
        a_scroll.setWidgetResizable(True)
        a_scroll.setWidget(a_inner)
        self.align_list_card = self._panel("alignCard", "Alignment & Reconstruction",
                                           a_scroll)

        self.form_box = QVBoxLayout()
        self.form_box.setAlignment(Qt.AlignTop)
        # Right margin clears the vertical scrollbar so it never overlaps text;
        # tight spacing keeps rows dense.
        self.form_box.setContentsMargins(12, 8, 20, 8)
        self.form_box.setSpacing(3)
        form_inner = QWidget()
        form_inner.setLayout(self.form_box)
        form_scroll = QScrollArea()
        form_scroll.setWidgetResizable(True)
        # Never scroll horizontally — force content to the viewport width so the
        # command box + help labels WRAP instead of extending off to the right.
        form_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        form_scroll.setWidget(form_inner)
        self.command_card = self._panel("cmdCard", "Job builder", form_scroll)

    # ---- one mockup column of stage rows; fills the shared registries ----
    def _build_stage_list(self, column_key):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setAlignment(Qt.AlignTop)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        if column_key != "alignrecon":     # alignrecon already has a card header
            t = QLabel(COLUMN_TITLES[column_key])
            t.setStyleSheet("font-size:12px;font-weight:700;color:#d0d0d0;margin-top:2px;")
            v.addWidget(t)
        last_group = None
        for spec in STAGES:
            if COLUMN_OF_GROUP.get(spec["group"]) != column_key:
                continue
            if spec["group"] != last_group:
                g = QLabel(spec["group"])
                gf = g.font()
                gf.setCapitalization(QFont.AllUppercase)
                gf.setLetterSpacing(QFont.AbsoluteSpacing, 1.5)
                gf.setBold(True)
                g.setFont(gf)
                g.setStyleSheet("color:#8a8a8a;font-size:10px;margin-top:6px;")
                v.addWidget(g)
                last_group = spec["group"]
            row = QHBoxLayout()
            dot = QLabel("●")
            dot.setStyleSheet(DOT_GREY)
            btn = QPushButton(spec["label"])
            btn.setStyleSheet("text-align:left;")
            btn.clicked.connect(lambda _=False, s=spec: self._select_stage(s))
            row.addWidget(dot)
            row.addWidget(btn, 1)
            self.node_buttons[spec["id"]] = dot
            self.stage_buttons[spec["id"]] = btn
            wrap = QWidget()
            wrap.setLayout(row)
            v.addWidget(wrap)
            if spec["id"] in STAGE_OUTPUTS:
                out_row = QHBoxLayout()
                out_row.setContentsMargins(22, 0, 0, 0)
                open_btn = QPushButton("📂")
                open_btn.setFixedWidth(32)
                open_btn.setToolTip("Open this step's output folder (file manager)")
                open_btn.clicked.connect(
                    lambda _=False, sid=spec["id"]: self._open_stage_output(sid))
                mod_btn = QPushButton("3dmod")
                mod_btn.setFixedWidth(52)
                mod_btn.setToolTip("Open this folder's .mrc volumes in 3dmod "
                                   "(module load 3dmod; WARP_FORCE_MRC_FLOAT32=1 3dmod *.mrc)")
                mod_btn.clicked.connect(
                    lambda _=False, sid=spec["id"]: self._open_stage_output_3dmod(sid))
                combo = QComboBox()
                combo.setToolTip("Output iteration to open (versioned runs)")
                out_row.addWidget(open_btn)
                out_row.addWidget(mod_btn)
                out_row.addWidget(combo, 1)
                self.output_combos[spec["id"]] = combo
                outw = QWidget()
                outw.setLayout(out_row)
                v.addWidget(outw)
        return w

    # ---- right: directory overview (schematic, click-to-open) ----
    def _build_directory_overview(self):
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setAlignment(Qt.AlignTop)
        v.setContentsMargins(2, 2, 2, 2)
        legend = QLabel("blue = input · green = output · grey = other.  "
                        "Click a folder to open it.")
        legend.setStyleSheet("color:#888;font-size:10px;")
        legend.setWordWrap(True)
        v.addWidget(legend)
        for rel, label in KEY_DIRS:
            b = QPushButton(label)
            b.setStyleSheet("text-align:left;border:none;color:#666;")
            b.setToolTip(rel)
            b.clicked.connect(lambda _=False, r=rel: self._open_dir(r))
            v.addWidget(b)
            self.dir_buttons[rel] = b
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        return scroll

    def _refresh_dir_overview(self, spec):
        if not getattr(self, "dir_buttons", None):
            return
        ins, outs = STAGE_IO.get(spec["id"], ([], []))
        ins, outs = set(ins), set(outs)
        for rel, btn in self.dir_buttons.items():
            if rel == "aretomo_output":
                exists = bool(self.project.list_aretomo_versions())
            else:
                exists = (self.project.root / rel).is_dir()
            if rel in outs:
                color, weight = "#3fb567", "700"       # green output
            elif rel in ins:
                color, weight = "#5b9bd5", "700"        # blue input
            else:
                color, weight = ("#888" if exists else "#444"), "400"  # grey other
            btn.setStyleSheet(f"text-align:left;border:none;color:{color};"
                              f"font-weight:{weight};")

    def _open_dir(self, rel):
        if rel == "aretomo_output":
            versions = self.project.list_aretomo_versions()
            target = versions[-1] if versions else self.project.root / "aretomo_output"
        else:
            target = self.project.root / rel
        if not target.is_dir():
            self._log(f"Directory does not exist yet: {target}", "info")
            return
        try:
            subprocess.Popen(["xdg-open", str(target)])
            self._log(f"xdg-open {target}", "info")
        except OSError as e:
            self._log(f"Could not open {target}: {e}", "fail")

    # ---- open any file the right way (3dmod for stacks, gedit for text, else xdg) ----
    def _open_path_smart(self, path):
        p = Path(path)
        if not p.exists():
            self._log(f"Not found: {path}", "info")
            return
        if p.is_dir():
            try:
                subprocess.Popen(["xdg-open", str(p)])
            except OSError as e:
                self._log(f"Could not open {p}: {e}", "fail")
            return
        ext = p.suffix.lower()
        try:
            if ext in THREEDMOD_EXTS:
                cmd = ('module load 3dmod 2>/dev/null || module load imod 2>/dev/null; '
                       f'WARP_FORCE_MRC_FLOAT32=1 3dmod "{p}"; '
                       'module unload 3dmod 2>/dev/null || module unload imod 2>/dev/null')
                subprocess.Popen(["bash", "-lc", cmd])
                self._log(f"3dmod {p.name}", "info")
            elif ext in TEXT_EXTS:
                subprocess.Popen(["gedit", str(p)])
                self._log(f"gedit {p.name}", "info")
            else:
                subprocess.Popen(["xdg-open", str(p)])
                self._log(f"xdg-open {p.name}", "info")
        except OSError as e:
            self._log(f"Could not open {p}: {e}", "fail")

    def _open_history(self):
        win = ProcessingHistory(self, self.project, self._open_path_smart,
                                self._load_history_settings)
        self._child_windows.append(win)
        win.show()

    def _load_history_settings(self, stage_id, command):
        """Restore a past run's exact command into its stage form (a manual override,
        which is authoritative). Does NOT touch the stage's template defaults."""
        spec = self._stage_by_id(stage_id)
        if not spec:
            self._log(f"Unknown stage in history: {stage_id}", "fail")
            return
        self._select_stage(spec)
        if self.current and self.current.get("cmd") is not None:
            self.current["cmd"].setPlainText(command)   # marks manual / authoritative
            self._log(f"Loaded history settings into '{spec['label']}' (manual "
                      f"override — template defaults unchanged).", "ok")

    # ---- right: jobs queue (own card, with cancel) ----
    def _build_queue_panel(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        self.queue_view = QPlainTextEdit()
        self.queue_view.setReadOnly(True)
        self.queue_view.setStyleSheet(f"font-family:{MONO};font-size:11px;")
        v.addWidget(self.queue_view, 1)
        row = QHBoxLayout()
        run_q = QPushButton("Run queue")
        run_q.clicked.connect(self._run_queue)
        cancel_q = QPushButton("Cancel queued")
        cancel_q.setToolTip("Clear all queued (not-yet-started) jobs.")
        cancel_q.clicked.connect(self._cancel_queue)
        row.addWidget(run_q)
        row.addWidget(cancel_q)
        rw = QWidget()
        rw.setLayout(row)
        v.addWidget(rw)
        self._refresh_queue()
        return w

    def _cancel_queue(self):
        if not self.queue:
            return
        n = len(self.queue)
        self.queue = []
        self._refresh_queue()
        self._log(f"Cleared {n} queued job(s).", "info")

    def _select_stage(self, spec):
        # Highlight the active stage so selection is visible.
        for sid, b in getattr(self, "stage_buttons", {}).items():
            b.setStyleSheet("text-align:left;")
        active = getattr(self, "stage_buttons", {}).get(spec["id"])
        if active:
            active.setStyleSheet(
                "text-align:left;background:#2d4a6b;color:#ffffff;font-weight:600;")
        self._render_docs(spec)
        self._refresh_dir_overview(spec)              # recolour the dir schematic
        while self.form_box.count():
            w = self.form_box.takeAt(0).widget()
            if w:
                w.deleteLater()

        controls = {}        # name -> getter()
        title = QLabel(spec["label"])
        title.setStyleSheet("font-size:15px;font-weight:600;")
        self.form_box.addWidget(title)

        # Interactive tool stage: launch a window instead of running a command.
        if spec.get("tool"):
            desc = QLabel(spec["docs"].get("what", ""))
            desc.setWordWrap(True)
            self.form_box.addWidget(desc)
            launchers = {
                "inspector": [("Open all tilt series (3dmod + exclusion list)",
                               self._open_all_tilts_and_list),
                              ("Open exclusion list only (no 3dmod)",
                               self._open_tilt_inspector),
                              ("Open Positions Inspector (mdoc vs frames)",
                               self._open_inspector)],
            }
            for label, slot in launchers.get(spec["tool"], []):
                btn = QPushButton(label)
                btn.clicked.connect(slot)
                self.form_box.addWidget(btn)
            if spec["tool"] == "inspector":
                excl_btn = QPushButton("View exclusion_list.txt (gedit)")
                excl_btn.clicked.connect(self._open_exclusion_list)
                self.form_box.addWidget(excl_btn)
            self.current = {"spec": spec, "controls": {}, "cmd": None,
                            "warn": None, "manual": False, "guard": False}
            return

        # Dynamic default: AreTomo output_dir = next versioned folder.
        overrides = {}
        if spec.get("aretomo"):
            try:
                nv = self.project.next_aretomo_version_path()
                overrides["output_dir"] = os.path.relpath(nv, self.project.root)
            except (OSError, ValueError):
                pass
        # Dynamic default: point ts_import_alignments at the NEWEST AreTomo folder
        # that actually has alignments (AreTomo auto-versions; a stale base folder
        # fails with "Could not find <series>.xf").
        if spec.get("id") == "ts_import_alignments":
            try:
                rel = self.project.latest_aretomo_imod()
                if rel:
                    overrides["alignments"] = rel
            except (OSError, ValueError):
                pass

        # Auto-detect the real gain file (its name varies per dataset) so the
        # gain steps don't default to a placeholder that doesn't exist.
        if spec["id"] == "gain_convert":
            g = self.project.gain_source()
            if g:
                overrides["in_gain"] = g
        elif spec["id"] == "gain_reciprocal":
            conv = self.project.root / "gains" / "original_gain.mrc"
            if conv.is_file():
                overrides["in_mrc"] = "gains/original_gain.mrc"
            else:
                g = self.project.gain_source()
                if g.endswith(".mrc"):
                    overrides["in_mrc"] = g

        # Make listed path params ABSOLUTE — some companion scripts `cd` into a
        # working dir and then read the other paths, so relative paths break
        # (e.g. remake_mdocs cd's into mdocs/ then reads exclusion_list.txt).
        for pname in spec.get("abs_paths", []):
            p = next((pp for pp in spec.get("params", []) if pp["name"] == pname), None)
            d = str(p.get("default", "")) if p else ""
            if d and not os.path.isabs(d):
                overrides[pname] = str(self.project.root / d)

        # Effective value per param: a fresh DYNAMIC default (new-version-available)
        # wins; else the user's persisted edit; else the static template default.
        stored = self._param_store.get(spec["id"], {})
        for p in spec.get("params", []):
            name = p["name"]
            eff = overrides[name] if name in overrides else stored.get(name)
            self.form_box.addWidget(self._param_row(p, controls, eff))

        warn = QLabel("")
        warn.setStyleSheet("color:#c0392b;")
        warn.setWordWrap(True)
        self.form_box.addWidget(warn)

        cmd = QPlainTextEdit()
        cmd.setFixedHeight(96)
        cmd.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)   # wrap, don't scroll
        cmd.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        cmd.setStyleSheet(f"font-family:{MONO};font-size:12px;")
        cmd_lbl = QLabel("Command (editable — this is what runs):")
        cmd_lbl.setWordWrap(True)
        self.form_box.addWidget(cmd_lbl)
        self.form_box.addWidget(cmd)

        btns = QHBoxLayout()
        run = QPushButton("▶ Run")
        build_job = QPushButton("▶ Build & run as job")
        build_job.setToolTip("Card view: create a job instance on the canvas from these "
                             "parameters (input auto-wired to the newest upstream job) "
                             "and run it. Fork a finished job to try variants.")
        rebuild = QPushButton("↻ Rebuild from controls")
        enqueue = QPushButton("+ Queue variant")
        reset = QPushButton("⟲ Reset defaults")
        reset.setToolTip("Discard your saved edits for THIS step and restore the "
                         "template defaults (and current dynamic defaults).")
        for b in (run, build_job, rebuild, enqueue, reset):
            btns.addWidget(b)
        if spec.get("sync_helper"):
            fill = QPushButton("Fill: deselect all unaligned")
            fill.clicked.connect(self._fill_sync_command)
            btns.addWidget(fill)
        bw = QWidget()
        bw.setLayout(btns)
        self.form_box.addWidget(bw)

        self.current = {"spec": spec, "controls": controls, "cmd": cmd,
                        "warn": warn, "manual": False, "guard": False}

        cmd.textChanged.connect(self._on_cmd_edited)
        run.clicked.connect(self._run_current)
        build_job.clicked.connect(lambda: self._build_job(spec["id"], run=True))
        rebuild.clicked.connect(lambda: self._set_manual(False))
        enqueue.clicked.connect(self._enqueue_current)
        reset.clicked.connect(lambda: self._reset_stage_defaults(spec["id"]))
        self._rebuild_cmd()

    def _reset_stage_defaults(self, stage_id):
        """Forget this step's persisted edits and rebuild it from the template +
        dynamic defaults."""
        if self._param_store.pop(stage_id, None) is not None:
            self._persist_param_store()
            self._log(f"Reset '{stage_id}' parameters to defaults.", "info")
        spec = self._stage_by_id(stage_id)
        if spec:
            self._select_stage(spec)

    def _params_path(self):
        return Path(self.project_root) / ".tomogration_params.json"

    def _load_param_store(self):
        """Per-stage param edits, from the project dir (shared across VMs via ceph).
        Migrates once from the old per-VM ~/.tomogration.json 'param_store' key."""
        p = self._params_path()
        if p.is_file():
            try:
                d = json.loads(p.read_text())
                if isinstance(d, dict):
                    return d
            except (OSError, ValueError):
                pass
        legacy = self._load_config().get("param_store")
        return dict(legacy) if isinstance(legacy, dict) else {}

    def _persist_param_store(self):
        try:
            self._params_path().write_text(json.dumps(self._param_store, indent=1))
        except OSError as e:
            self._log(f"Could not save parameters: {e}", "fail")

    def _param_row(self, p, controls, default_override=None):
        # Compact two-line row: [name | control] on top, dim smaller help beneath.
        # Tight margins so rows don't waste vertical space (the form scrolls).
        default = default_override if default_override is not None else p.get("default")
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        name = QLabel(p["name"])
        name.setStyleSheet("color:#d6d6d6;font-size:12px;")
        name.setMinimumWidth(96)
        row.addWidget(name)
        kind = p["kind"]
        if kind == "check":
            cb = QCheckBox()
            cb.setChecked(bool(default))
            cb.stateChanged.connect(self._on_control_changed)
            controls[p["name"]] = lambda c=cb: c.isChecked()
            row.addWidget(cb)
            row.addStretch(1)
        elif kind in ("slider_int", "env_int"):
            # A spin box (type OR step), not a slider — sliders can't hit an exact
            # large value (e.g. MA_POOL_SIZE 2400 jumps 1988→2004). Type it directly.
            s = QSpinBox()
            s.setMinimum(int(p["min"]))
            s.setMaximum(int(p["max"]))
            s.setSingleStep(int(p.get("step", 1)))
            s.setValue(int(default))
            s.setKeyboardTracking(False)   # emit once on commit, not per digit
            s.valueChanged.connect(lambda _val: self._on_control_changed())
            controls[p["name"]] = lambda c=s: c.value()
            row.addWidget(s, 1)
        elif kind == "choice":
            # Mutually-exclusive options where each choice IS a flag string (or "").
            cb = QComboBox()
            for label, val in p.get("choices", []):
                cb.addItem(label, val)
            idx = cb.findData(default)
            if idx >= 0:
                cb.setCurrentIndex(idx)
            cb.currentIndexChanged.connect(self._on_control_changed)
            controls[p["name"]] = lambda c=cb: c.currentData()
            row.addWidget(cb, 1)
        else:  # "text" or "env"
            e = QLineEdit(str(default if default is not None else ""))
            e.textChanged.connect(self._on_control_changed)
            controls[p["name"]] = lambda c=e: c.text()
            row.addWidget(e, 1)
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 3, 0, 3)
        outer.setSpacing(1)
        rw = QWidget()
        rw.setLayout(row)
        outer.addWidget(rw)
        help_text = p.get("help", "")
        if help_text:
            help_lab = QLabel(help_text)
            help_lab.setStyleSheet("color:#7c7c7c;font-size:10px;")
            help_lab.setWordWrap(True)
            outer.addWidget(help_lab)
        box = QWidget()
        box.setLayout(outer)
        return box

    # ---- command-box-as-source-of-truth ----
    def _values(self):
        return {n: g() for n, g in self.current["controls"].items()}

    def _build_command(self):
        return build_command(self.current["spec"], self._values(),
                             getattr(self, "warp_launch", None),
                             getattr(self, "_group_inputs", None))

    # ---- tilt-series groups ----
    def _refresh_group_inputs(self):
        """Recompute the --input_data lists for the active group (None = All)."""
        active = self._groups.get("active", ProjectState.ALL_GROUP)
        if active == ProjectState.ALL_GROUP:
            self._group_inputs = None
            return
        series = self._groups.get("groups", {}).get(active, [])
        self._group_inputs = self.project.write_group_inputs(series) if series else None

    def _open_groups(self):
        self._groups = self.project.load_groups()    # pick up any inspector edits
        dlg = GroupsDialog(self, self.project, self._groups)
        if dlg.exec():
            self._groups = dlg.result_groups
            self.project.save_groups(self._groups)
            self._refresh_group_inputs()
            self._update_group_label()
            if self.current and self.current.get("cmd") is not None \
                    and not self.current["manual"]:
                self._rebuild_cmd()
            active = self._groups.get("active", ProjectState.ALL_GROUP)
            n = len(self._groups.get("groups", {}).get(active, [])) \
                if active != ProjectState.ALL_GROUP else 0
            self._log(f"Active tilt-series group: {active}"
                      + (f" ({n} series)" if n else ""), "ok")
            # Offer a self-contained subset working folder when a non-All group is
            # active and its subset hasn't been built yet (so it doesn't nag once
            # built, but does fire even if the group was already active before).
            if (active != ProjectState.ALL_GROUP and n
                    and not self.project.group_dir(active).exists()):
                self._offer_materialize(active, self._groups["groups"][active])

    def _offer_materialize(self, name, series):
        """Prompt to build a self-contained subset working folder for the group
        (symlinks its frames/mdocs/gains) and switch the project root to it, so
        every step — including ts_import & AreTomo — runs only on the subset."""
        if QMessageBox.question(
                self, "Build subset working folder?",
                f"Group '{name}' has {len(series)} tilt series.\n\n"
                f"Build a self-contained working folder under groups/ (symlinking "
                f"its frames + mdocs + gains) and switch the project root to it? "
                f"Then EVERY step — create_settings, fs_motion_and_ctf, ts_import, "
                f"AreTomo, … — runs on just this subset, and you re-make its "
                f"settings there.\n\n"
                f"Choose No to stay in the current root (the WarpTools processing "
                f"steps are still restricted to the group via --input_data, but "
                f"ts_import/AreTomo see all series)."
        ) != QMessageBox.Yes:
            return
        try:
            base, n_eer, n_mdoc = self.project.materialize_group(name, series)
        except OSError as e:
            self._log(f"Could not build subset folder: {e}", "fail")
            return
        self._log(f"Built subset working folder {base} "
                  f"({n_mdoc} mdocs + {n_eer} .eer symlinked).", "ok")
        self._set_root(str(base))
        self._log("Project root switched to the subset. Run from create_settings "
                  "(fs) onward — fs/ts now process only these series.", "info")

    def _update_group_label(self):
        if getattr(self, "group_label", None) is None:
            return
        active = self._groups.get("active", ProjectState.ALL_GROUP)
        if active == ProjectState.ALL_GROUP:
            self.group_label.setText("Group: All tilt series")
            self.group_label.setStyleSheet("color:#888;font-size:11px;")
        else:
            n = len(self._groups.get("groups", {}).get(active, []))
            self.group_label.setText(f"Group: {active} ({n} series)")
            self.group_label.setStyleSheet("color:#27ae60;font-size:11px;font-weight:600;")

    # ---- WarpTools launch command (it's behind a module on this cluster) ----
    def _config_path(self):
        return Path.home() / ".tomogration.json"

    def _load_config(self):
        try:
            return json.loads(self._config_path().read_text())
        except (OSError, ValueError):
            return {}

    def _save_config(self, cfg):
        try:
            self._config_path().write_text(json.dumps(cfg, indent=2))
        except OSError as e:
            self._log(f"Could not save config: {e}", "fail")

    def _set_warp_launch(self):
        text, ok = QInputDialog.getText(
            self, "WarpTools launch command",
            "Command that runs WarpTools — the leading 'WarpTools' in every Warp "
            "step is replaced by this (e.g. 'module load warp && WarpTools', "
            "'conda activate warp && WarpTools', or a full path):",
            text=self.warp_launch)
        if ok and text.strip():
            self.warp_launch = text.strip()
            cfg = self._load_config()
            cfg["warp_launch"] = self.warp_launch
            self._save_config(cfg)
            self._log(f"WarpTools launch set to: {self.warp_launch}", "ok")
            if self.current and self.current.get("cmd") is not None \
                    and not self.current["manual"]:
                self._rebuild_cmd()

    def _set_tm_vis_launch(self):
        text, ok = QInputDialog.getText(
            self, "warp-tm-vis launch command",
            "How to launch warp-tm-vis (its -rdir/-mdir/-mp/-cvp args are appended) "
            "— e.g. 'uvx warp-tm-vis', a bare 'warp-tm-vis', or a full path:",
            text=self.tm_vis_launch)
        if ok and text.strip():
            self.tm_vis_launch = text.strip()
            cfg = self._load_config()
            cfg["tm_vis_launch"] = self.tm_vis_launch
            self._save_config(cfg)
            self._log(f"warp-tm-vis launch set to: {self.tm_vis_launch}", "ok")

    def _view_picks_tm_vis(self, node):
        """Launch warp-tm-vis (github.com/warpem/warp-tm-vis) on this pick set: it
        overlays the template-match picks + correlation volumes on the tomograms.
        Patterns are scoped to this run's suffix; shown editable before launch so
        the paths/suffix (or a jobs/<id>/ dir) can be adjusted."""
        stage_id = node.get("stage_id")
        if not node.get("is_ghost") and node.get("id"):
            params = ((load_jobs(self.project_root).get("jobs", {})
                       .get(node["id"], {}) or {}).get("params", {}))
        elif self.current and self.current.get("spec", {}).get("id") == stage_id:
            params = self._values()
        else:
            spec = self._stage_by_id(stage_id) or {}
            params = self._effective_params(spec) if spec else {}
        apx = fmt_angpix(params.get("tomo_angpix", "12.56"))
        # STAR uses the run's suffix (override or template); the CORR volume always
        # uses the template suffix (--override_suffix doesn't rename it).
        star_pat = f"*{apx}Apx{template_match_suffix(params)}.star"
        corr_pat = f"*{apx}Apx{template_corr_suffix(params)}_corr.mrc"
        cmd = (f'{self.tm_vis_launch} '
               f'-rdir warp_tiltseries/reconstruction '
               f'-mdir warp_tiltseries/matching '
               f'-mp "{star_pat}" -cvp "{corr_pat}"')
        cmd, ok = QInputDialog.getText(
            self, "Launch warp-tm-vis",
            "Command (edit the suffix / paths if needed; add --no-load-volumes if "
            "this run didn't save _corr.mrc volumes):", text=cmd)
        if not ok or not cmd.strip():
            return
        try:
            subprocess.Popen(["bash", "-lc", cmd.strip()], cwd=str(self.project_root))
            self._log(f"warp-tm-vis: {cmd.strip()}", "info")
        except OSError as e:
            self._log(f"Could not launch warp-tm-vis: {e}", "fail")

    def _rebuild_cmd(self):
        self.current["guard"] = True
        self.current["cmd"].setPlainText(self._build_command())
        self.current["guard"] = False
        self._update_warning()

    def _on_control_changed(self):
        # Persist the user's edits for this step (in-memory now, disk shortly).
        if self.current and self.current.get("spec"):
            self._param_store[self.current["spec"]["id"]] = self._values()
            self._persist_timer.start()
        if self.current and not self.current["manual"]:
            self._rebuild_cmd()
        self._update_warning()

    def _on_cmd_edited(self):
        if self.current and not self.current["guard"]:
            self._set_manual(True)

    def _set_manual(self, manual):
        self.current["manual"] = manual
        if not manual:
            self._rebuild_cmd()

    def _update_warning(self):
        spec = self.current["spec"]
        msg = spec["validate"](self._values()) if spec.get("validate") else ""
        self.current["warn"].setText(msg)

    def _fill_sync_command(self):
        missing = self.project.tomostars_without_alignments()
        if not missing:
            self._log("All tomostars have alignments — nothing to deselect.", "ok")
            return
        vals = self._values()
        settings = vals.get("settings", "warp_tiltseries.settings")
        # First command carries the WarpTools launcher (module load + conda activate);
        # the rest run in the same shell so plain 'WarpTools' is on PATH by then.
        cmds = [f"WarpTools change_selection --settings {settings} --deselect "
                f"--input_data tomostar/{n}.tomostar" for n in missing]
        cmds[0] = cmds[0].replace("WarpTools", self.warp_launch, 1)
        full = " && \\\n  ".join(cmds)
        self.current["cmd"].setPlainText(full)   # marks manual (authoritative)
        self._log(f"Filled deselect command for {len(missing)} unaligned tilt series.",
                  "info")

    # ---- run / queue ----
    def _run_current(self):
        cmd = self.current["cmd"].toPlainText().strip()
        if not cmd:
            return
        spec = self.current["spec"]
        if spec.get("requires_coarse_alignment"):
            if not self._coarse_alignment_gate():
                return
            cmd = self._missalign_chain_cmd(cmd)
        if spec.get("aretomo"):
            self._prepare_aretomo_run(cmd)
        if spec.get("clean_overrides"):
            self._prepare_rename_run()
        if spec.get("normalize_exclusions"):
            self._normalize_exclusions()
        self._dispatch(spec["id"], cmd, fresh=True)

    def _coarse_alignment_gate(self):
        """ENFORCE the order AreTomo -> ts_import_alignments -> select -> miss-alignment.
        miss-alignment REFINES a coarse alignment; on raw stacks it makes a featureless
        tomogram. Return True to proceed, False to block. Keyed off the AreTomo .xf the
        import reads (latest_aretomo_imod), which also carries AreTomo's computed tilt
        axis — exactly what we want miss-alignment to start from."""
        if not self.project.latest_aretomo_imod():
            QMessageBox.warning(
                self, "Coarse alignment required first",
                "miss-alignment REFINES an existing alignment — it cannot align raw "
                "stacks (that is what produced the featureless tomogram).\n\n"
                "Run the coarse alignment first, in this order:\n"
                "   1. Align with AreTomo2   (computes the tilt axis)\n"
                "   2. ts_import_alignments  (imports AreTomo's axis + shifts into Warp)\n"
                "   3. sync selection -> Select (re-enable)\n"
                "then run miss-alignment.\n\n"
                "No AreTomo alignment (.xf) was found under this project.")
            self._log("miss-alignment blocked: no AreTomo alignment found. It refines, "
                      "it does not align — run AreTomo -> ts_import_alignments -> select "
                      "first.", "fail")
            return False
        missing = self.project.tomostars_without_alignments()
        if missing:
            ok = QMessageBox.question(
                self, "Some series have no AreTomo alignment",
                f"{len(missing)} tilt series still have no AreTomo .xf "
                f"(e.g. {', '.join(missing[:5])}{' …' if len(missing) > 5 else ''}).\n\n"
                "miss-alignment can only refine the aligned ones; the rest would come "
                "out garbage. Continue anyway?") == QMessageBox.Yes
            return ok
        return True

    def _missalign_chain_cmd(self, cmd):
        """For a FRESH miss-alignment run, prepend the AreTomo import + 'select all' so
        AreTomo's computed tilt axis + shifts are written into the Warp .xml right
        before miss-alignment refines them — making the enforced order a single step.
        Skipped on resume (MA_START_ITER > 0) so it never clobbers refinement in
        progress. alignment_angpix 1.57 = this dataset's native (the ts_import_alignments
        stage default)."""
        vals = self._values()
        start = str(vals.get("MA_START_ITER", "0")).strip() or "0"
        imod = self.project.latest_aretomo_imod()
        if start != "0" or not imod:
            return cmd
        pre = (f"module load miniconda/latest && conda activate warp && "
               f"WarpTools ts_import_alignments --settings warp_tiltseries.settings "
               f"--alignments {imod} --alignment_angpix 1.57 && "
               f"WarpTools change_selection --settings warp_tiltseries.settings --select "
               f"&& conda deactivate")
        self._log("Fresh miss-alignment: prepending AreTomo import + select (enforced "
                  f"order; AreTomo axis from {imod}).", "info")
        return f"{pre} && {cmd}"

    def _normalize_exclusions(self):
        """Expand any tilt ranges (e.g. 1-3) in exclusion_list.txt before remake —
        the remake script can't parse ranges. Safe on hand-edited files too."""
        path = self.project.root / "exclusion_list.txt"
        if not path.is_file():
            return
        try:
            # A bare 'PositionNNN' line (no tilt numbers) = drop the WHOLE series:
            # quarantine it now, BEFORE the normalizer strips tilt-less lines.
            q = self.project.quarantine_listed_whole_series()
            if q:
                self._log(f"Excluded {len(q)} whole series listed without tilt numbers "
                          f"(mdoc→mdocs/bad/, .eer→frames/bad/): "
                          f"{', '.join(q[:8])}" + (" …" if len(q) > 8 else ""),
                          "warning")
            n = self.project.write_manual_exclusions({})   # re-normalizes in place
            self._log(f"Normalized exclusion_list.txt: {n} tilt-exclusion line(s), "
                      f"tilt ranges expanded.", "info")
        except OSError as e:
            self._log(f"Could not normalize exclusion_list.txt: {e}", "fail")

    def _prepare_rename_run(self):
        """Quarantine identical Tomo5 *_override.mdoc in the rename source dir
        before the rename script runs (it would otherwise consume their numbers)."""
        vals = self._values()
        src_rel = str(vals.get("source_dir", ".")).strip() or "."
        src = self.project.root / src_rel
        if not src.is_dir():
            return
        removed, kept = self.project.clean_override_mdocs(src)
        if removed or kept:
            self._log(f"Override mdocs in {src_rel}: {removed} identical → mdocs/bad/"
                      + (f", {kept} differing left in place (check them)" if kept else ""),
                      "warning" if kept else "ok")

    def _prepare_aretomo_run(self, cmd):
        """Create the versioned output folder and drop its PARAMETERS.txt audit
        before AreTomo launches (brief §4.6 / §5)."""
        vals = self._values()
        out_rel = str(vals.get("output_dir", "")).strip()
        if not out_rel:
            return
        out_dir = self.project.root / out_rel
        try:
            pf = self.project.write_aretomo_parameters(out_dir, vals, cmd)
            self._log(f"AreTomo output -> {out_rel} (PARAMETERS.txt written: {pf.name})",
                      "info")
        except OSError as e:
            self._log(f"Could not write PARAMETERS.txt: {e}", "fail")

    def _enqueue_current(self):
        cmd = self.current["cmd"].toPlainText().strip()
        if not cmd:
            return
        spec = self.current["spec"]
        if spec.get("requires_coarse_alignment"):
            if not self._coarse_alignment_gate():
                return
            cmd = self._missalign_chain_cmd(cmd)
        if spec.get("aretomo"):
            self._prepare_aretomo_run(cmd)
        if spec.get("clean_overrides"):
            self._prepare_rename_run()
        if spec.get("normalize_exclusions"):
            self._normalize_exclusions()
        label = f'{spec["id"]} v{len(self.queue) + 1}'
        self.queue.append((label, cmd, spec["id"]))
        self._log(f"queued: {label}", "info")
        self._refresh_queue()

    def _dispatch(self, stage_id, cmd, fresh=True):
        if self.runner.busy():
            self._log("a job is already running — queue it instead.", "fail")
            return
        if fresh:
            self._attempt = 1
            self._failed_file = None
            self._record_history_start(stage_id, cmd)   # logs + may archive output
        self._active_stage = stage_id
        self._active_cmd = cmd
        self._active_job_id = None      # this is a three-column stage run, not a job
        self.node_buttons[stage_id].setStyleSheet(DOT_ORANGE)
        self.runner.run(cmd, self.project_root)

    # ---- card-view job runs (Phase 1b) ----
    # A parallel dispatch path to _dispatch: instead of running a stage at its
    # conventional STAGE_OUTPUTS path, run a JOB INSTANCE in its own processing
    # dir (jobs/J###), wired to its parent via --input_processing. The canvas
    # (Phase 2) calls _run_job; the three-column view keeps using _dispatch. Both
    # share the single runner, so the busy-check keeps them mutually exclusive.
    # NOTE: jobs deliberately do NOT auto-recover (that fs_motion_and_ctf retry
    # logic is bound to the three-column _active_stage/_active_cmd path).
    def _run_job(self, job_id):
        if self.runner.busy():
            self._log("a job is already running — wait for it to finish.", "fail")
            return
        store = load_jobs(self.project_root)
        job = store.get("jobs", {}).get(job_id)
        if job is None:
            self._log(f"job {job_id} not found.", "fail")
            return
        spec = self._stage_by_id(job["stage_id"])
        if spec is None:
            self._log(f"job {job_id}: unknown stage '{job['stage_id']}'.", "fail")
            return
        # The processing dir must exist before the run (cwd = project root).
        try:
            (Path(self.project_root) / job["output_dir"]).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self._log(f"job {job_id}: cannot create {job['output_dir']}: {e}", "fail")
            return
        cmd = build_job_command(spec, job, store, self.warp_launch, self._group_inputs)
        update_job(self.project_root, job_id, command=cmd, status="running",
                   exit_code=None, finished=None,
                   started=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self._active_job_id = job_id
        self._active_stage = None
        self._active_cmd = cmd
        self._attempt = 1
        self._failed_file = None
        self._log(f"--- running {job_id} · {spec['label']} ---", "info")
        self._refresh_canvas()          # flip the card to 'running' immediately
        self.runner.run(cmd, self.project_root)

    def _finalize_job(self, code):
        """Record a finished job: status/exit/finish-time + a per-stage result
        summary for the card readout. Called from _on_finished."""
        job_id = self._active_job_id
        self._active_job_id = None
        store = load_jobs(self.project_root)
        job = store.get("jobs", {}).get(job_id)
        if job is None:
            return
        status = "completed" if code == 0 else "failed"
        summary = summarize_job(job["stage_id"],
                                Path(self.project_root) / job["output_dir"])
        update_job(self.project_root, job_id, status=status, exit_code=code,
                   finished=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   summary=summary)
        self._log(f"--- {job_id} {status} (exit {code}) ---",
                  "ok" if code == 0 else "fail")
        if hasattr(self, "_refresh_canvas"):     # Phase 2 hook; harmless until then
            self._refresh_canvas()

    # ---- processing history ----
    def _non_default_tokens(self, spec, cmd):
        """The bits of cmd that differ from the stage's template command — shown as
        bullet points under the job in the History window."""
        if not spec:
            return []
        try:
            default_cmd = build_command(spec, stage_defaults(spec),
                                        getattr(self, "warp_launch", None),
                                        getattr(self, "_group_inputs", None))
        except Exception:
            default_cmd = ""
        def_toks = set(default_cmd.split())
        toks, extras, i = cmd.split(), [], 0
        while i < len(toks):
            t = toks[i]
            if t in def_toks or t == "bash" or t.endswith(".sh"):
                i += 1
                continue
            if t.startswith("-") and i + 1 < len(toks) and not toks[i + 1].startswith("-"):
                extras.append(f"{t} {toks[i + 1]}")
                i += 2
            else:
                extras.append(t)
                i += 1
        return extras

    def _record_history_start(self, stage_id, cmd):
        spec = self._stage_by_id(stage_id)
        archived = ""
        if stage_id in ARCHIVE_ON_RERUN:
            for out in STAGE_IO.get(stage_id, ([], []))[1]:
                a = self.project.archive_output_dir(out)
                if a:
                    archived = a
                    self._log(f"Kept previous '{out}' as '{a}' (not overwritten).", "info")
        ins, outs = STAGE_IO.get(stage_id, ([], []))
        rec = {
            "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stage_id": stage_id,
            "label": spec["label"] if spec else stage_id,
            "command": cmd,
            "params": self._non_default_tokens(spec, cmd),
            "inputs": list(ins),
            "outputs": ([archived] if archived else []) + list(outs),
            "exit_code": None,
        }
        self._history_index = self.project.append_history(rec)

    def _on_finished(self, code):
        # Card-view job runs finalize on their own path (their own store record,
        # no stage dots / history / auto-recovery). A queued STAGE run may still
        # start afterwards, so fall through to the chaining block below.
        if getattr(self, "_active_job_id", None):
            self._finalize_job(code)
            self._refresh_status_dots()
            if self.queue and not self.runner.busy():
                label, cmd, sid = self.queue.pop(0)
                self._log(f"--- running {label} ---", "info")
                self._refresh_queue()
                self._dispatch(sid, cmd, fresh=True)
            return

        stage_id = self._active_stage
        spec = self._stage_by_id(stage_id)

        # Auto-recovery — fs_motion_and_ctf only (brief §5). A cuFFT crash on a
        # bad .eer is quarantined and the run retried; ts_* crashes are GPU/
        # resource issues, not bad files, so they do NOT recover.
        if (code != 0 and spec and spec.get("auto_recover")
                and self._failed_file and self._attempt < self._max_retries):
            if self._recover_failed_file(self._failed_file, spec):
                self._attempt += 1
                self._failed_file = None
                self._log(f"--- auto-recover retry "
                          f"{self._attempt}/{self._max_retries} ---", "info")
                self._dispatch(stage_id, self._active_cmd, fresh=False)
                return

        dot = self.node_buttons.get(stage_id)
        if dot and code != 0:
            dot.setStyleSheet(DOT_RED)
            dot.setToolTip("last run failed")
        # finalize this job's history record (the queue dispatch below starts a new one)
        idx = getattr(self, "_history_index", None)
        if idx is not None:
            self.project.update_history(idx, exit_code=code)
            self._history_index = None
        self._refresh_status_dots()

        if self.queue and not self.runner.busy():
            label, cmd, sid = self.queue.pop(0)
            self._log(f"--- running {label} ---", "info")
            self._refresh_queue()
            self._dispatch(sid, cmd, fresh=True)

    def _recover_failed_file(self, path, spec):
        basename = os.path.basename(path)
        resolved = self.project.root / "frames" / basename
        if not resolved.is_file():
            alt = self.project.root / path
            if alt.is_file():
                basename = alt.name
            else:
                self._log(f"auto-recover: can't find {path} on disk — giving up.",
                          "fail")
                return False
        if not self.project.move_frames_to_bad([basename]):
            return False
        m = re.match(r"(Position\d+)", basename)
        position = m.group(1) if m else None
        try:
            self.project.append_auto_exclusion(
                basename, f"auto-quarantined during {spec['id']} (crash)",
                position=position)
        except OSError:
            pass
        mdoc_edited = False
        if position:
            mdoc_path = self.project.root / "mdocs" / f"{position}.mdoc"
            if mdoc_path.is_file():
                mdoc_edited = ProjectState.remove_mdoc_zvalue_block(
                    str(mdoc_path), basename)
        self._log(f"quarantined {basename} -> frames/bad/ "
                  f"({'mdoc updated' if mdoc_edited else 'mdoc NOT updated'})",
                  "warning")
        return True

    # ---- status dots (real file-existence checks) ----
    def _refresh_status_dots(self):
        for spec in STAGES:
            dot = self.node_buttons.get(spec["id"])
            if dot is None:
                continue
            if self._active_stage == spec["id"] and self.runner.busy():
                continue
            fn = spec.get("status")
            if not fn:
                dot.setStyleSheet(DOT_GREY)
                dot.setToolTip("")
                continue
            try:
                ok, label = fn(self.project)
            except Exception:   # status checks must never crash the GUI
                ok, label = False, None
            dot.setStyleSheet(DOT_GREEN if ok else DOT_GREY)
            dot.setToolTip(label or "")
        self._refresh_output_combos()
        # re-colour the directory overview for the active stage (dirs come and go)
        if self.current and self.current.get("spec"):
            self._refresh_dir_overview(self.current["spec"])

    def _refresh_output_combos(self):
        """Repopulate each step's output-iteration dropdown from disk."""
        for sid, combo in getattr(self, "output_combos", {}).items():
            keep = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            versions = self.project.output_versions(STAGE_OUTPUTS[sid])
            if versions:
                for label, path in versions:
                    combo.addItem(label, str(path))
                i = combo.findText(keep)
                combo.setCurrentIndex(i if i >= 0 else combo.count() - 1)
            else:
                combo.addItem("(no output yet)", "")
            combo.blockSignals(False)

    def _open_stage_output(self, stage_id):
        combo = self.output_combos.get(stage_id)
        path = combo.currentData() if combo else None
        if not path:
            self._log(f"No output folder yet for {stage_id}.", "info")
            return
        try:
            subprocess.Popen(["xdg-open", path])
            self._log(f"opened {path}", "info")
        except OSError as e:
            self._log(f"Could not open {path}: {e}", "fail")

    def _open_stage_output_3dmod(self, stage_id):
        """Open every .mrc in the selected output folder in one 3dmod window — the
        module-load + WARP_FORCE_MRC_FLOAT32 dance done for you."""
        combo = self.output_combos.get(stage_id)
        path = combo.currentData() if combo else None
        if not path:
            self._log(f"No output folder yet for {stage_id}.", "info")
            return
        folder = Path(path)
        mrcs = sorted(folder.glob("*.mrc")) or sorted(folder.glob("*/*.mrc"))
        if not mrcs:
            self._log(f"No .mrc files in {folder} (nor one level down).", "info")
            return
        if len(mrcs) > 40 and QMessageBox.question(
                self, "Open many volumes?",
                f"{len(mrcs)} .mrc files in {folder.name}. Opening all in one 3dmod "
                f"can use a lot of memory. Open them anyway?") != QMessageBox.Yes:
            return
        quoted = " ".join(f'"{m}"' for m in mrcs)
        cmd = ('module load 3dmod 2>/dev/null || module load imod 2>/dev/null; '
               f'WARP_FORCE_MRC_FLOAT32=1 3dmod {quoted}; '
               'module unload 3dmod 2>/dev/null || module unload imod 2>/dev/null')
        try:
            subprocess.Popen(["bash", "-lc", cmd])
            self._log(f"3dmod: {len(mrcs)} volume(s) from {folder.name}", "info")
        except OSError as e:
            self._log(f"Could not launch 3dmod: {e}", "fail")

    # ---- project root / working directory ----
    def _browse_root(self):
        # Paste-first (no enumeration); browsing INTO a raw folder of thousands of
        # files crashes the Qt picker, so default to typing/pasting the path.
        new = ask_project_root(self, self.project_root)
        if new:
            self._set_root(new)

    def _set_root(self, new_root):
        """Re-root the app on a different working directory at any time."""
        if not new_root or not os.path.isdir(new_root):
            self._log(f"Not a directory: {new_root}", "fail")
            self.root_edit.setText(self.project_root)
            return
        new_root = os.path.abspath(new_root)
        if new_root == self.project_root:
            return
        if self.runner.busy():
            self._log("Note: a job is still running in the previous directory; "
                      "it keeps that cwd. New runs use the new root.", "warning")
        self.project_root = new_root
        self.project = ProjectState(new_root)
        self.root_edit.setText(new_root)
        self.setWindowTitle(f"tomogration — {new_root}")
        self._save_config({**self._load_config(), "last_root": new_root})  # resume here next launch
        # Groups are per-root — reload them (a subset root has its own / none).
        self._groups = self.project.load_groups()
        self._refresh_group_inputs()
        self._update_group_label()
        # Param edits are per-project (shared across VMs) — reload for the new root.
        self._param_store = self._load_param_store()
        self._refresh_status_dots()
        self._refresh_canvas()
        if self.current:                      # refresh dynamic defaults (e.g. AreTomo vN)
            self._select_stage(self.current["spec"])
        self._log(f"Project root -> {new_root}", "ok")

    # ---- raw-file affordances ----
    def _init_dirs(self):
        """Create the standard project subdirs under the chosen root, on demand.
        No 'selected/' folder is needed — the root you picked IS the project."""
        self.project.initialize_structure()
        self._log(f"Created standard project dirs ({', '.join(EXPECTED_DIRS)}) "
                  f"under {self.project_root}", "ok")
        self._refresh_status_dots()

    def _sort_files(self):
        """Make the standard dirs and move raw files from a chosen folder into
        frames/ (.eer), mdocs/ (.mdoc), gains/ (gain). Confirms counts first."""
        src = QFileDialog.getExistingDirectory(
            self, "Folder of raw files to sort into the project",
            str(Path(self.project_root).parent), options=NONATIVE_DIR)
        if not src:
            return
        plan = self.project.plan_sort(src)
        n_total = sum(len(plan[k]) for k in ("frames", "mdocs", "gains", "overrides"))
        if n_total == 0:
            QMessageBox.information(
                self, "Nothing to sort",
                f"No .eer / .mdoc / gain files found directly in:\n{src}\n\n"
                f"({len(plan['skipped'])} other files would be left in place.)")
            return
        if QMessageBox.question(
                self, "Sort files into project?",
                f"From:\n{src}\n\nMove into {self.project_root}:\n"
                f"  • .eer  → frames/ : {len(plan['frames'])}\n"
                f"  • .mdoc → mdocs/  : {len(plan['mdocs'])}\n"
                f"  • gain  → gains/  : {len(plan['gains'])}\n\n"
                f"Tomo5 *_override.mdoc found: {len(plan['overrides'])} — those "
                f"identical to their standard mdoc go to mdocs/bad/; any that "
                f"differ are left in place and flagged.\n\n"
                f"Leave in place: {len(plan['skipped'])} other files "
                f"(.mrc, Thumbnails, Session.dm…).\n\n"
                f"Existing files are never overwritten. Proceed?"
        ) != QMessageBox.Yes:
            return
        moved = self.project.sort_files(src)
        self._log(f"Sorted raw files from {src}: "
                  f"{moved['frames']} → frames/, {moved['mdocs']} → mdocs/, "
                  f"{moved['gains']} → gains/; "
                  f"{moved['overrides_removed']} identical *_override.mdoc → mdocs/bad/.",
                  "ok")
        if moved["overrides_kept"]:
            self._log(f"⚠ {moved['overrides_kept']} *_override.mdoc DIFFER from "
                      f"their standard mdoc (or have none) — left in place; "
                      f"inspect them before running rename.", "warning")
        self._refresh_status_dots()

    def _repair_mdocs(self):
        to_remove = self.project.dry_run_repair_mdocs()
        if not to_remove:
            QMessageBox.information(
                self, "Nothing to repair",
                "All mdoc entries reference files present in frames/.")
            return
        by_mdoc = {}
        for mdoc_name, sub in to_remove:
            by_mdoc.setdefault(mdoc_name, []).append(sub)
        summary = "\n".join(f"  {m}: {len(s)} stale"
                            for m, s in sorted(by_mdoc.items())[:12])
        if len(by_mdoc) > 12:
            summary += f"\n  … and {len(by_mdoc) - 12} more"
        if QMessageBox.question(
                self, "Repair mdocs?",
                f"Found {len(to_remove)} stale mdoc entries across {len(by_mdoc)} "
                f"mdocs (referencing .eer no longer in frames/, e.g. moved to "
                f"frames/bad/).\n\n{summary}\n\nRemove them and renumber ZValue "
                f"blocks?") != QMessageBox.Yes:
            return
        removed = self.project.repair_all_mdocs()
        self._log(f"Repaired mdocs: removed {len(removed)} stale ZValue blocks.", "ok")
        for pos, sub in removed:
            self._log(f"  {pos}: stripped {sub}", "info")
        self._refresh_status_dots()

    def _open_inspector(self):
        PositionsInspector(self, self.project, self._log).exec()
        self._refresh_status_dots()

    def _open_tilt_inspector(self):
        # Non-modal so it can sit beside 3dmod. Kept on a list so it isn't
        # garbage-collected while open. Returns the window so callers can chain.
        win = TiltInspector(self, self.project, self._log)
        self._child_windows.append(win)
        # The inspector may edit group membership — reload + refresh on close.
        win.finished.connect(lambda _=0: self._reload_groups())
        win.show()
        return win

    def _open_all_tilts_and_list(self):
        """The main inspect action: pop the (thumbnail-free) exclusion list AND
        open every tilt series in one 3dmod, so you eyeball them there and type
        bad tilts in the list."""
        win = self._open_tilt_inspector()
        win._open_all_3dmod()

    def _reload_groups(self):
        """Re-read groups from disk (the Tilt Inspector edits them) and refresh
        the active-group inputs, label, command and status dots."""
        self._groups = self.project.load_groups()
        self._refresh_group_inputs()
        self._update_group_label()
        if self.current and self.current.get("cmd") is not None \
                and not self.current["manual"]:
            self._rebuild_cmd()
        self._refresh_status_dots()

    def _open_aretomo_versions(self):
        if not self.project.list_aretomo_versions():
            QMessageBox.information(
                self, "No AreTomo runs",
                "No aretomo_output* folders yet. Run the AreTomo2 stage first.")
            return
        AreTomoVersions(self, self.project).exec()

    def _open_exclusion_list(self):
        path = self.project.root / "exclusion_list.txt"
        if not path.is_file():
            self._log("No exclusion_list.txt yet — none written by the inspectors.",
                      "info")
            return
        try:
            subprocess.Popen(["gedit", str(path)])
            self._log(f"gedit {path}", "info")
        except OSError as e:
            self._log(f"Could not launch gedit: {e}", "fail")

    def _open_file_in_editor(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open file in editor", str(self.project.root), options=NONATIVE)
        if path:
            try:
                subprocess.Popen(["gedit", path])
            except OSError as e:
                self._log(f"Could not launch gedit: {e}", "fail")

    def _launch_chimerax(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open volume in ChimeraX",
            str(self.project.root), "MRC volumes (*.mrc);;All files (*)",
            options=NONATIVE)
        if not path:
            return
        try:
            subprocess.Popen(["chimerax", path])
            self._log(f"chimerax {path}", "info")
        except OSError as e:
            self._log(f"Could not launch ChimeraX: {e}", "fail")

    # ---- RIGHT: terminal + queue ----
    def _build_terminal_panel(self):
        v = QVBoxLayout()
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Log"))
        term_btn = QPushButton("TERMINATE")
        term_btn.setStyleSheet("color:#c0392b;")
        term_btn.clicked.connect(self.runner.terminate)
        bar.addStretch(1)
        bar.addWidget(term_btn)
        barw = QWidget()
        barw.setLayout(bar)
        v.addWidget(barw)

        self.term = QPlainTextEdit()
        self.term.setReadOnly(True)
        self.term.setMaximumBlockCount(5000)
        self.term.setStyleSheet(
            f"background:#111;color:#ddd;font-family:{MONO};font-size:12px;")
        v.addWidget(self.term, 1)

        w = QWidget()
        w.setLayout(v)
        return w

    def _run_queue(self):
        if self.queue and not self.runner.busy():
            label, cmd, sid = self.queue.pop(0)
            self._log(f"--- running {label} ---", "info")
            self._refresh_queue()
            self._dispatch(sid, cmd, fresh=True)

    def _refresh_queue(self):
        self.queue_view.setPlainText(
            "\n".join(f"• {l}" for l, _, _ in self.queue) or "(empty)")

    def _log(self, text, level):
        # Tap the stream for auto-recovery's failed-file detection.
        m = _FAILED_FILE_RE.search(text)
        if m:
            self._failed_file = m.group(1)
        colours = {"out": "#dddddd", "err": "#e0a850", "info": "#7fb4ff",
                   "ok": "#27ae60", "success": "#27ae60", "warning": "#e0a850",
                   "error": "#e24b4a", "fail": "#e24b4a"}
        is_progress = level in ("out", "err") and bool(_PROGRESS_RE.match(text))
        # WarpTools prints a blank spacer line between progress updates — swallow
        # it so it doesn't break the in-place collapse below.
        if not text.strip() and self._last_was_progress:
            return
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html = f'<span style="color:{colours.get(level, "#ddd")}">{safe}</span>'
        # Follow the tail only if already at the bottom; if the user scrolled up to
        # read, leave their position alone (don't yank the view around).
        sb = self.term.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 4
        prev = sb.value()
        # Collapse WarpTools "N/M …" progress spam: keep ONE line that updates in
        # place (via a local cursor — never touch the widget cursor/scroll).
        if is_progress and self._last_was_progress:
            cur = self.term.textCursor()
            cur.movePosition(QTextCursor.End)
            cur.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
            cur.removeSelectedText()          # empty the last block
            cur.insertHtml(html)              # replace it with the new progress line
        else:
            self.term.appendHtml(html)
        self._last_was_progress = is_progress
        sb.setValue(sb.maximum() if at_bottom else prev)


def ask_project_root(parent, start):
    """Get a project-root path WITHOUT enumerating it. Qt's directory browser
    scandir's every entry to render the list, which hangs/crashes on raw folders
    holding thousands of .eer/.mdoc over ceph (the symptom: paste the path into
    the picker -> it navigates in -> crash). So ask for a pasted path first (the
    text box never touches the filesystem); only browse if the user asks. Returns
    an absolute dir path, or None if cancelled."""
    while True:
        text, ok = QInputDialog.getText(
            parent, "Project root",
            "Paste the FULL path to your working directory and click OK.\n\n"
            "Recommended for raw folders with thousands of files — the file\n"
            "browser can hang/crash while listing them.\n\n"
            "Leave the box blank and click OK to browse instead.",
            text=(start or ""))
        if not ok:
            return None
        text = text.strip().rstrip("/")
        if not text:
            chosen = QFileDialog.getExistingDirectory(
                parent, "Select project root (pick the folder, don't open it)",
                str(Path(start).parent) if start else str(Path.home()),
                options=NONATIVE_DIR)
            return chosen or None
        if os.path.isdir(text):
            return os.path.abspath(text)
        QMessageBox.warning(parent, "Not a directory",
                            f"Not a directory:\n{text}\n\nCheck the path and try again.")


def main():
    app = QApplication(sys.argv)
    # Stop the mouse wheel from changing combo boxes/sliders when scrolling a panel.
    app._wheel_guard = WheelGuard(app)
    app.installEventFilter(app._wheel_guard)
    # Resume the last project root if it still exists (so a relaunch doesn't lose
    # e.g. a subset working folder). Otherwise ask.
    last = None
    try:
        last = json.loads((Path.home() / ".tomogration.json").read_text()).get("last_root")
    except (OSError, ValueError):
        pass
    if last and os.path.isdir(last):
        root = last
    else:
        root = ask_project_root(None, str(Path.home()))
    if not root:
        print("No project selected.")
        return
    win = Tomogration(root)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
