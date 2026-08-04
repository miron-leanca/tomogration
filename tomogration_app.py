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
import time
import array as _array

from PySide6.QtCore import (
    Qt, QObject, QEvent, Signal, QProcess, QTimer, QSize, QRect, QPoint, QMimeData,
)
from PySide6.QtGui import (
    QBrush, QColor, QImage, QPixmap, QTextCursor, QFont, QPen, QPainter,
    QPalette, QFontMetrics, QDrag,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QPlainTextEdit, QScrollArea, QSlider, QCheckBox,
    QLineEdit, QFileDialog, QDialog, QListWidget, QTreeWidget, QTreeWidgetItem,
    QMessageBox, QHeaderView, QGridLayout, QFrame, QStackedWidget, QComboBox,
    QInputDialog, QListWidgetItem, QTextBrowser,
    QAbstractScrollArea, QAbstractSpinBox, QSpinBox,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsSimpleTextItem,
    QGraphicsItem, QMenu, QLayout, QSizePolicy, QProgressBar, QToolButton,
    QAbstractItemView,
)


class FlowLayout(QLayout):
    """A layout that lays widgets left-to-right and WRAPS to the next line when it
    runs out of width — like text. Used for the Job-builder button row so a narrow
    right panel wraps the buttons onto extra rows instead of forcing the whole form
    wider than the panel (which clipped the wrapped help text). Its minimumSize is
    just the widest single child, so the form can shrink freely."""

    def __init__(self, parent=None, margin=0, spacing=6):
        super().__init__(parent)
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)
        self._items = []

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _do_layout(self, rect, test_only):
        x, y, line_h = rect.x(), rect.y(), 0
        sp = self.spacing()
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + sp
            if next_x - sp > rect.right() and line_h > 0:
                x = rect.x()
                y = y + line_h + sp
                next_x = x + hint.width() + sp
                line_h = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_h = max(line_h, hint.height())
        return y + line_h - rect.y()


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


# ---------------------------------------------------------------------------
# The app was one 7,900-line file; the pieces below now live in siblings so they
# can be edited and unit-tested on their own (and so two people editing the app
# stop colliding on the same file). Import graph: core -> stages -> jobs, with
# project depending only on core. Everything is re-exported into this module's
# namespace, so existing references — and the tests, which load this file by
# path — keep working unchanged.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))   # siblings importable
                                                           # when loaded by path
# The app is split across sibling modules, so ALL of tomogration_*.py must be copied
# to the VM together. Copying only tomogration_app.py leaves a stale sibling and the
# failure is a raw ImportError deep in a traceback — turn it into a plain instruction.
try:
    from tomogration_core import (_PKG_DIR, _pkg_script, expand_tilt_ranges,
                                  progress_key)
except ImportError as _e:
    _here = Path(__file__).resolve().parent
    _need = ["tomogration_core.py", "tomogration_stages.py",
             "tomogration_project.py", "tomogration_jobs.py"]
    _missing = [f for f in _need if not (_here / f).is_file()]
    raise SystemExit(
        "\n".join([
            "",
            "tomogration: this install is out of date or incomplete.",
            f"  {_e}",
            "",
            "tomogration is split across several files and they must be copied",
            "TOGETHER — updating tomogration_app.py alone leaves a stale sibling.",
            f"  folder: {_here}",
            ("  MISSING: " + ", ".join(_missing)) if _missing
            else "  All files are present, so one of them is an OLD version.",
            "",
            "Copy every tomogration_*.py from your dev folder and relaunch.",
            "",
        ]))
from tomogration_stages import *            # noqa: F401,F403
from tomogration_stages import (            # explicit: the names used below
    # underscore names are NOT re-exported by `import *` — list them
    _h, _norm_gpu, _validate_export,
    STAGES, STAGE_OUTPUTS, STAGE_IO, DIR_FILE_HINTS, COLUMN_OF_GROUP,
    COLUMN_TITLES, KEY_DIRS, JOB_STAGES, TRUNK_STAGES, ARCHIVE_ON_RERUN,
    THREEDMOD_EXTS, TEXT_EXTS, build_command, stage_defaults, _norm_gpu,
    render_docs_html, render_inline_docs_html, _validate_export,
)
from tomogration_project import ProjectState, EXPECTED_DIRS, _FAILED_FILE_RE
from tomogration_jobs import *              # noqa: F401,F403
from tomogration_jobs import (
    # underscore names are NOT re-exported by `import *` — list them
    _APX_RE, _CTF_DEFOCUS_RE, _PICK_STAR_RE, _count_glob, _job_seq, _jobnum, _mean_std, _picktag,
    JOBS_FILE, jobs_path, load_jobs, save_jobs, queued_jobs, job_output_dir,
    reconcile_running, job_delete_targets, PROTECTED_DIRS,
    new_job, update_job, delete_job, is_warp_stage, parent_job_id,
    io_flags_for_job, build_job_command, default_parent_for, fmt_angpix,
    template_corr_suffix, template_match_suffix, match_star_infix, DOWNSTREAM,
    derive_child_params, summarize_job, summary_text, FRIENDLY_TITLES,
    stage_title, _job_seq, is_cryolo_job, canvas_layout, card_is_running,
    discover_picksets, m_resolution, params_for_builder, job_real_outputs,
    star_particle_count, relion_card_text, set_card_position, clear_card_positions,
    set_job_parent, settings_processing_dir,
    discover_relion_jobs, _PICK_STAR_RE,
)

# A progress line to be collapsed into ONE updating line. Covers both dialects:
#   WarpTools : "239/5439, 08:06:22 remaining"   "5439/5439, previous metadata..."
#   RELION    : "0.58/2.13 min ......~~(,_,\">"   "000/??? sec ~~(,_,\"> [oo]"
# RELION's are DECIMAL and its ETA bar redraws constantly — the old integer-only
# pattern missed them, so every tick appended a new line and flooded the log.
_PROGRESS_RE = re.compile(
    r"^\s*(?:"
    r"\d+(?:\.\d+)?\s*/\s*(?:\d+(?:\.\d+)?\b|\?+)"   # N/M, 0.58/2.13, 000/???
    r"|[.\s]*~~\(,_,"                                    # RELION's progress fish
    r")")


# ===========================================================================
# ProjectState  —  framework-agnostic backend (ported from warp_auto.py)
# ===========================================================================
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
# The template rail reads as a PRINTED REFERENCE, not as work: cool slate on a
# darker ground, so a real job of any status is unmistakably brighter than it.
_TEMPLATE_STYLE = ("#151b22", "#33414f")
_RAIL_BG = "#0b0f13"
_RAIL_HATCH = "#16202a"
_RAIL_EDGE = "#22303c"

_CARD_STYLE = {
    "ghost":     ("#242424", "#555555"),
    # BUILDING — violet. Yours to configure; it will not run until you queue it.
    "building":  ("#2c2440", "#8b6ed6"),
    "queued":    ("#26313a", "#3a6ea5"),   # blue: waiting its turn
    "running":   ("#4a3a12", "#f0a92a"),   # BRIGHT amber — the unmistakable one
    "completed": ("#1d3326", "#27ae60"),
    "failed":    ("#3a2320", "#c0392b"),
    # Found-on-disk, not yet a job. VIOLET on purpose: the old amber-brown was too
    # close to 'running' and read as "this job is live" when it wasn't.
    "orphan":    ("#2b2436", "#8a6ec0"),
}


class _QueueChip(QWidget):
    """One queued job in the terminal's queue strip: its id in blue, with a red ✕
    that appears on hover to take it back out of the queue.

    The cross is hidden until hover on purpose — the strip is read far more often
    than it is acted on, and a row of permanent crosses reads as a list of things
    to dismiss rather than a list of what runs next.
    """
    def __init__(self, job_id, label, on_cancel, parent=None):
        super().__init__(parent)
        self._job_id = job_id
        self._on_cancel = on_cancel
        row = QHBoxLayout(self)
        row.setContentsMargins(6, 1, 4, 1)
        row.setSpacing(3)
        self.name = QLabel(job_id)
        self.name.setStyleSheet(f"color:#6ea8f0;font-family:{MONO};font-size:11px;")
        row.addWidget(self.name)
        self.x = QLabel("✕")
        self.x.setStyleSheet("color:#c0392b;font-size:11px;font-weight:700;")
        self.x.setVisible(False)
        self.x.setCursor(Qt.PointingHandCursor)
        row.addWidget(self.x)
        self.setToolTip(f"{job_id} · {label}\nQueued — click ✕ to take it out of the "
                        f"queue and edit it again.")
        self.setStyleSheet("background:#151c24;border:1px solid #24354a;"
                           "border-radius:3px;")

    def enterEvent(self, ev):
        self.x.setVisible(True)
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        self.x.setVisible(False)
        super().leaveEvent(ev)

    def mousePressEvent(self, ev):
        # Only the cross cancels: clicking the id itself must not silently unqueue
        # a job you were only pointing at.
        if self.x.isVisible() and self.x.geometry().contains(ev.pos()):
            self._on_cancel(self._job_id)
            ev.accept()
            return
        super().mousePressEvent(ev)


class _CardItem(QGraphicsRectItem):
    """A single card. Holds its node dict and routes clicks to the canvas.

    Draggable when the canvas is unlocked. The auto-layout (one row per stage,
    forks across columns) is right for a fresh project and wrong as soon as a real
    one branches, so a dragged card's position is stored and wins from then on.
    Locking exists because the same drag gesture also pans the canvas — without a
    lock you cannot help nudging cards while navigating.
    """
    def __init__(self, node, canvas):
        super().__init__(0, 0, node["w"], node["h"])
        self._node = node
        self._canvas = canvas
        self._press_pos = None
        self.setPos(node["x"], node["y"])
        # The template rail is fixed furniture: it is the reference the working
        # canvas is read against, so it never moves and never gets dragged out of
        # order by accident.
        movable = not canvas.locked and not node.get("is_template")
        self.setCursor(Qt.OpenHandCursor if movable else Qt.PointingHandCursor)
        if movable:
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.RightButton:
            self._canvas._menu(self._node, ev.screenPos())
            ev.accept()
            return
        self._press_pos = self.pos()
        self._canvas._pick(self._node)
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        if self._canvas.locked or self._press_pos is None:
            return
        now = self.pos()
        # A click is a drag of zero distance; only a real move is worth storing (and
        # worth a repaint, which would otherwise fire on every card selection).
        if (abs(now.x() - self._press_pos.x()) < 2
                and abs(now.y() - self._press_pos.y()) < 2):
            return
        self._press_pos = None
        self._canvas._card_moved(self._node, now.x(), now.y())


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


class _CanvasView(QGraphicsView):
    """QGraphicsView for the workflow graph (wide, with forks). Adds horizontal
    navigation that survives remote desktops: trackpad horizontal swipes and
    Shift+wheel scroll left/right; click-dragging empty canvas pans (clicks on
    cards still select, since a drag only starts where there's no item)."""
    # Grab-hand ("pan") mode. Remote desktops routinely drop horizontal scroll
    # events entirely (a MacBook two-finger swipe never reaches the Linux VM as
    # angleDelta().x()), so panning must not depend on them. With hand mode ON,
    # dragging ANYWHERE moves the canvas; with it off, dragging empty space pans
    # and clicks still select cards.
    hand_mode = False
    _drop_handler = None

    def set_drop_handler(self, fn):
        """Accept stage rows dragged out of the job palette. Drops are enabled only
        once a handler exists, so the view never advertises a drop it can't act on."""
        self._drop_handler = fn
        self.setAcceptDrops(fn is not None)

    def dragEnterEvent(self, ev):
        if self._drop_handler is not None and ev.mimeData().hasFormat(MIME_STAGE):
            ev.acceptProposedAction()
        else:
            super().dragEnterEvent(ev)

    def dragMoveEvent(self, ev):
        if self._drop_handler is not None and ev.mimeData().hasFormat(MIME_STAGE):
            ev.acceptProposedAction()
        else:
            super().dragMoveEvent(ev)

    def dropEvent(self, ev):
        if self._drop_handler is None or not ev.mimeData().hasFormat(MIME_STAGE):
            super().dropEvent(ev)
            return
        sid = bytes(ev.mimeData().data(MIME_STAGE)).decode("utf-8", "replace")
        # Drop where the cursor is, in SCENE coordinates — the view is scrolled and
        # zoomed, so viewport pixels are not scene units.
        p = self.mapToScene(ev.position().toPoint())
        ev.acceptProposedAction()
        if sid:
            self._drop_handler(sid, p.x(), p.y())

    def set_hand_mode(self, on):
        self.hand_mode = bool(on)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag if self.hand_mode
                         else QGraphicsView.DragMode.NoDrag)
        self.viewport().setCursor(Qt.OpenHandCursor if self.hand_mode
                                  else Qt.ArrowCursor)

    def wheelEvent(self, ev):
        d = ev.angleDelta()
        horiz = d.x()
        if horiz == 0 and (ev.modifiers() & Qt.ShiftModifier):
            horiz = d.y()
        # No horizontal delta and no Shift: if the graph is far wider than tall
        # (it always is — one row per stage, forks spreading right), plain vertical
        # wheel is much more useful as horizontal pan than as a 2-row nudge.
        if horiz == 0 and d.y():
            hbar, vbar = self.horizontalScrollBar(), self.verticalScrollBar()
            if hbar.maximum() > 0 and vbar.maximum() <= 0:
                horiz = d.y()
        if horiz:
            bar = self.horizontalScrollBar()
            bar.setValue(bar.value() - horiz)
            ev.accept()
        else:
            super().wheelEvent(ev)

    def keyPressEvent(self, ev):
        """Arrow keys pan — the fallback that works on every remote protocol."""
        step = 200 if ev.modifiers() & Qt.ShiftModifier else 60
        if ev.key() in (Qt.Key_Left, Qt.Key_Right):
            bar = self.horizontalScrollBar()
            bar.setValue(bar.value() + (step if ev.key() == Qt.Key_Right else -step))
            ev.accept()
            return
        if ev.key() in (Qt.Key_Up, Qt.Key_Down):
            bar = self.verticalScrollBar()
            bar.setValue(bar.value() + (step if ev.key() == Qt.Key_Down else -step))
            ev.accept()
            return
        super().keyPressEvent(ev)

    def mousePressEvent(self, ev):
        if (ev.button() == Qt.LeftButton and not self.hand_mode
                and self.itemAt(ev.pos()) is None):
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        if not self.hand_mode:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)




class GpuPanel(QWidget):
    """A small always-on GPU monitor pinned to the canvas.

    Replaces squinting at nvidia-smi. One row per GPU: a load bar, memory, temp.
    Deliberately minimal — this is glanceable status, not a dashboard.

    Two things it gets right that a naive version wouldn't:
      * it polls with a QProcess, never a blocking subprocess call, so a slow or
        hung nvidia-smi can't freeze the UI (the thing that just bit this app);
      * it colours by UTILISATION, not memory. nvidia-smi hides other users' PIDs
        on this cluster, so a GPU can look free by process list while another user
        pins it at 100%. Load is the honest signal for 'can I run here?'.
    """
    QUERY = ("index,utilization.gpu,memory.used,memory.total,temperature.gpu")

    def __init__(self, parent=None, interval_ms=5000):
        super().__init__(parent)
        self._rows = []                 # [(idx, util%, used_MiB, total_MiB, tempC)]
        self._error = None
        self._proc = None
        self.setAttribute(Qt.WA_TransparentForMouseEvents)   # never eats canvas clicks
        self.setFixedWidth(196)
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self.poll)
        self._timer.start()
        self.poll()

    def poll(self):
        if self._proc is not None and self._proc.state() != QProcess.NotRunning:
            return                       # previous poll still going; skip this tick
        self._proc = QProcess(self)
        self._proc.finished.connect(self._read)
        self._proc.errorOccurred.connect(lambda _e: self._fail("nvidia-smi not found"))
        self._proc.start("nvidia-smi",
                         [f"--query-gpu={self.QUERY}", "--format=csv,noheader,nounits"])

    def _fail(self, msg):
        self._error, self._rows = msg, []
        self._resize_to_rows()
        self.update()

    def _read(self):
        try:
            txt = bytes(self._proc.readAllStandardOutput()).decode("utf-8", "replace")
        except Exception:
            return self._fail("nvidia-smi unreadable")
        rows = []
        for line in txt.strip().splitlines():
            parts = [c.strip() for c in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                rows.append((int(parts[0]), int(parts[1]), int(parts[2]),
                             int(parts[3]), int(parts[4])))
            except ValueError:
                continue
        if not rows:
            return self._fail("no GPUs reported")
        self._error, self._rows = None, rows
        self._resize_to_rows()
        self.update()

    def _resize_to_rows(self):
        """One compact cell per GPU, laid out horizontally to sit in the toolbar."""
        self.setFixedWidth(max(120, 22 + 74 * max(1, len(self._rows))))
        self.setFixedHeight(26)

    def paintEvent(self, _ev):
        pt = QPainter(self)
        pt.setRenderHint(QPainter.RenderHint.Antialiasing)
        f = QFont()
        f.setPointSize(8)
        pt.setFont(f)
        pt.setPen(QColor("#6f6f6f"))
        pt.drawText(2, 17, "GPU")
        if self._error:
            pt.setPen(QColor("#e0a850"))
            pt.drawText(26, 17, self._error)
            return
        x = 24
        for idx, util, used, total, temp in self._rows:
            # colour by LOAD, not memory: nvidia-smi hides other users' PIDs on this
            # cluster, so a GPU can look free by process list while pinned at 100%.
            col = QColor("#27ae60") if util < 25 else (
                QColor("#e0a850") if util < 80 else QColor("#e24b4a"))
            pt.setPen(Qt.NoPen)
            pt.setBrush(QColor("#242424"))
            pt.drawRoundedRect(x, 3, 66, 20, 4, 4)          # cell
            pt.setBrush(QColor("#333333"))
            pt.drawRoundedRect(x + 4, 15, 58, 5, 2, 2)      # bar track
            if util > 0:
                pt.setBrush(col)
                pt.drawRoundedRect(x + 4, 15, max(3, int(58 * util / 100)), 5, 2, 2)
            pt.setPen(QColor("#9a9a9a"))
            pt.drawText(x + 5, 12, str(idx))
            pt.setPen(col)
            pt.drawText(x + 16, 12, f"{util}%")
            pt.setPen(QColor("#e24b4a") if temp >= 80 else QColor("#6f6f6f"))
            pt.drawText(x + 40, 12, f"{temp}\u00b0")
            x += 74


MIME_STAGE = "application/x-tomogration-stage"


class _JobPalette(QWidget):
    """Drawer listing every stage that can be added, grouped as in the sidebar.

    Exists because the canvas only ever showed the stages the auto-layout decided
    to draw: to add anything else you had to leave the graph, find the stage in the
    builder's list, and build from there. Rows are draggable so a job can be placed
    where it belongs in the branch, and double-clickable for anyone who would
    rather not drag.
    """
    def __init__(self, on_activate, parent=None):
        super().__init__(parent)
        self._on_activate = on_activate
        self.setFixedWidth(232)
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(5)

        self.search = QLineEdit()
        self.search.setPlaceholderText("filter jobs…")
        self.search.textChanged.connect(self._repopulate)
        v.addWidget(self.search)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(True)
        self.tree.setDragEnabled(True)
        self.tree.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.tree.itemDoubleClicked.connect(self._activate)
        self.tree.startDrag = self._start_drag          # bound below
        v.addWidget(self.tree, 1)

        hint = QLabel("drag onto the canvas, or double-click")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#6f6f6f;font-size:10px;")
        v.addWidget(hint)
        self.setStyleSheet("background:#131a22;")
        self._repopulate("")

    def _repopulate(self, text):
        q = (text or "").strip().lower()
        self.tree.clear()
        groups = {}
        for spec in STAGES:
            title = stage_title(spec["id"], spec.get("label", spec["id"]))
            hay = f"{title} {spec['id']} {spec.get('label','')}".lower()
            if q and q not in hay:
                continue
            g = spec.get("group", "other")
            if g not in groups:
                gi = QTreeWidgetItem([g])
                gi.setFlags(Qt.ItemFlag.ItemIsEnabled)      # a header, not draggable
                self.tree.addTopLevelItem(gi)
                gi.setExpanded(True)
                groups[g] = gi
            it = QTreeWidgetItem([title])
            it.setData(0, Qt.UserRole, spec["id"])
            it.setToolTip(0, f"{spec['id']} — {spec.get('docs', {}).get('what', '')}")
            groups[g].addChild(it)

    def _stage_of(self, item):
        return item.data(0, Qt.UserRole) if item is not None else None

    def _activate(self, item, _col=0):
        sid = self._stage_of(item)
        if sid:
            self._on_activate(sid)

    def _start_drag(self, _actions):
        sid = self._stage_of(self.tree.currentItem())
        if not sid:
            return
        md = QMimeData()
        md.setData(MIME_STAGE, sid.encode("utf-8"))
        drag = QDrag(self.tree)
        drag.setMimeData(md)
        drag.exec(Qt.DropAction.CopyAction)


class JobCanvas(QWidget):
    # Seconds a per-stage on-disk status stays fresh (see refresh()).
    STATUS_TTL = 45

    def __init__(self, root_getter, on_pick, on_details=None, on_menu=None,
                 on_orphans=None, on_active=None, on_add_stage=None, parent=None):
        super().__init__(parent)
        self._root_getter = root_getter      # callable -> project_root str
        self._on_pick = on_pick              # callable(stage_id)
        self._on_details = on_details        # callable(node) | None
        self._on_menu = on_menu              # callable(node, global_qpoint) | None
        self._on_add_stage = on_add_stage    # callable(stage_id, x, y) | None
        # Cards are LOCKED by default: dragging empty canvas pans, and an unlocked
        # card under the cursor would move instead — surprising for anyone who has
        # not asked to rearrange anything.
        self.locked = True
        self._status_cache = None    # (ts, root, {stage: (ok,label)})
        self._on_orphans = on_orphans        # callable() -> [orphan descriptor] | None
        self._on_active = on_active          # callable() -> {running, label, progress,
                                             #                job_id|stage_id} | None
        self._active = {}
        self.scene = QGraphicsScene(self)
        self.view = _CanvasView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setBackgroundBrush(QColor("#0e0e0e"))
        # Scrollbars ALWAYS visible: on a remote desktop they may be the only
        # horizontal navigation that survives the protocol.
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.view.setFocusPolicy(Qt.StrongFocus)          # arrow keys pan
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        tools = QHBoxLayout()
        tools.setContentsMargins(4, 2, 4, 0)
        tools.setSpacing(6)
        self.hand_btn = QPushButton("✋ Pan")
        self.hand_btn.setCheckable(True)
        self.hand_btn.setToolTip("Grab-hand: drag anywhere to move the canvas.\n"
                                 "Use this when your trackpad's horizontal scroll "
                                 "doesn't reach the VM.")
        self.hand_btn.toggled.connect(self.view.set_hand_mode)
        tools.addWidget(self.hand_btn)
        fit = QPushButton("Fit")
        fit.setToolTip("Zoom to fit the whole workflow")
        fit.clicked.connect(self.fit_all)
        tools.addWidget(fit)
        reset = QPushButton("1:1")
        reset.setToolTip("Reset zoom")
        reset.clicked.connect(lambda: self.view.resetTransform())
        tools.addWidget(reset)

        self.lock_btn = QPushButton("🔒 Locked")
        self.lock_btn.setCheckable(True)
        self.lock_btn.setChecked(True)
        self.lock_btn.setToolTip(
            "Locked: cards stay where the layout puts them and dragging pans the "
            "canvas.\nUnlocked: drag cards to arrange the workflow into branches "
            "that make sense.\nPositions are saved with the project.")
        self.lock_btn.toggled.connect(self._set_locked)
        tools.addWidget(self.lock_btn)

        self.tidy_btn = QPushButton("Auto-arrange")
        self.tidy_btn.setToolTip("Discard your card positions and go back to the "
                                 "computed layout (one row per stage).")
        self.tidy_btn.clicked.connect(self._reset_positions)
        tools.addWidget(self.tidy_btn)

        hint = QLabel("drag or ✋ to pan · ← → arrows")
        hint.setStyleSheet("color:#6f6f6f;font-size:10px;")
        tools.addWidget(hint)
        tools.addStretch(1)
        # GPU strip, right-aligned in the canvas toolbar = the top-right of the card
        # view, positioned by the layout so it can never drift or scroll away.
        # Guarded: decorative status must never stop the canvas from building.
        self.gpu_panel = None
        try:
            self.gpu_panel = GpuPanel()
            tools.addWidget(self.gpu_panel)
        except Exception:
            self.gpu_panel = None
        tw = QWidget()
        tw.setLayout(tools)
        lay.addWidget(tw)

        # ---- job palette + canvas -------------------------------------------
        # The palette is a drawer, not a permanent column: the canvas is the thing
        # being read, and a stage list is only wanted while you are adding one.
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self.palette_btn = QToolButton()
        self.palette_btn.setText("＋\nA\nD\nD\n \nJ\nO\nB")
        self.palette_btn.setCheckable(True)
        self.palette_btn.setToolTip("Show the list of jobs you can add.\n"
                                    "Drag one onto the canvas to place it there, "
                                    "or double-click to drop it in the middle.")
        self.palette_btn.setStyleSheet(
            "QToolButton{background:#1b2430;color:#9ec5ff;border:1px solid #2f3d4d;"
            "border-left:none;font-size:9px;padding:8px 2px;}"
            "QToolButton:checked{background:#24405e;color:#dceaff;}")
        self.palette_btn.toggled.connect(self._toggle_palette)
        body.addWidget(self.palette_btn)

        self.palette = _JobPalette(self._add_stage_at_centre)
        self.palette.setVisible(False)
        body.addWidget(self.palette)
        body.addWidget(self.view, 1)
        bw = QWidget()
        bw.setLayout(body)
        lay.addWidget(bw, 1)
        self.view.set_drop_handler(self._drop_stage)

    # ---- card placement --------------------------------------------------
    def _set_locked(self, locked):
        self.locked = bool(locked)
        self.lock_btn.setText("🔒 Locked" if self.locked else "🔓 Unlocked")
        # Grab-hand and card dragging both claim a plain left-drag, so turning one
        # on turns the other off rather than letting them fight over the gesture.
        if not self.locked and self.hand_btn.isChecked():
            self.hand_btn.setChecked(False)
        self.refresh()

    def _card_moved(self, node, x, y):
        # Keep real work out of the rail — that overlap is what made the default
        # pipeline unreadable in the first place.
        x = max(float(x), float(RAIL_W))
        try:
            set_card_position(self._root_getter(), node["id"], x, y)
        except Exception:
            return
        self.refresh()          # redraw the edges to follow the card

    def _reset_positions(self):
        try:
            clear_card_positions(self._root_getter())
        except Exception:
            return
        self.refresh()

    def _toggle_palette(self, on):
        self.palette.setVisible(bool(on))

    def _add_stage_at_centre(self, stage_id):
        c = self.view.mapToScene(self.view.viewport().rect().center())
        self._drop_stage(stage_id, c.x(), c.y())

    def _drop_stage(self, stage_id, x, y):
        if self._on_add_stage is None:
            return
        try:
            self._on_add_stage(stage_id, x, y)
        except Exception:
            pass

    def _place_gpu_panel(self):
        """No-op: the GPU strip is laid out by the toolbar, not free-floating.

        It used to be a child widget of the graphics viewport, moved by hand to the
        top-right on every resize/repaint. Manual overlay geometry is fiddly and
        impossible to verify without running the GUI — it drifted into the middle of
        the canvas and panned with the content. A right-aligned widget in the canvas
        toolbar IS the top-right of the card view, and the layout keeps it there."""
        return

    def fit_all(self):
        r = self.scene.itemsBoundingRect()
        if r.isValid():
            self.view.fitInView(r.adjusted(-20, -20, 20, 20), Qt.KeepAspectRatio)

    def refresh(self):
        # Keep the user's scroll position — this repaints every few seconds while a
        # run is live, and yanking the view around would be maddening.
        hb = self.view.horizontalScrollBar().value()
        vb = self.view.verticalScrollBar().value()
        self.scene.clear()
        try:
            store = load_jobs(self._root_getter())
        except Exception:
            store = {"jobs": {}}
        # Orphans (found-on-disk work) are NOT canvas nodes any more. With ~50
        # RELION jobs they formed a mile-wide row that made the graph unreadable and
        # forced a disk scan on every repaint. They live in the Found-on-disk drawer
        # instead, and only appear here once adopted into the project.
        orphans = []
        self._active = {}
        if self._on_active is not None:
            try:
                self._active = self._on_active() or {}
            except Exception:
                self._active = {}
        # Per-stage on-disk status, so stages run OUTSIDE the app (terminal / trunk
        # runs, which create no job) still show as completed rather than 'not built'.
        stage_status = {}
        try:
            # CACHED. Every stage's status lambda hits the filesystem (globs, counts),
            # and this runs on every repaint — including every 5 s while a job is
            # live. Rescanning ceph that often is what makes the UI stutter when a
            # job starts. Recompute at most once per STATUS_TTL seconds; View ▸
            # Refresh status forces it, as does a job finishing.
            now = time.monotonic()
            root = self._root_getter()
            c = self._status_cache
            if c is not None and c[1] == root and now - c[0] < self.STATUS_TTL:
                stage_status = dict(c[2])
            else:
                ps = ProjectState(root)
                for spec in STAGES:
                    fn = spec.get("status")
                    if not fn:
                        continue
                    try:
                        ok, label = fn(ps)
                        if ok:
                            stage_status[spec["id"]] = (ok, label)
                    except Exception:
                        pass
                self._status_cache = (now, root, dict(stage_status))
        except Exception:
            pass
        # Cards the user hid (right-click ▸ Hide) are applied INSIDE canvas_layout so
        # the remaining cards re-pack into contiguous columns (no gap). Not deleted —
        # View ▸ Show hidden cards clears the set. Ghosts are never hidden (template).
        hidden = set(store.get("hidden", []))
        nodes, edges = canvas_layout(store, orphans, stage_status, hidden)
        index = {n["id"]: n for n in nodes}

        # The rail's own ground, drawn first so every card sits on top of it. Gives
        # the default pipeline a place of its own instead of leaving it as loose
        # cards the working canvas can drift over.
        tmpl = [n for n in nodes if n.get("is_template")]
        if tmpl:
            top = min(n["y"] for n in tmpl) - 34
            bot = max(n["y"] + n["h"] for n in tmpl) + 20
            bg = self.scene.addRect(-26, top, CARD_W + 44, bot - top,
                                    QPen(Qt.PenStyle.NoPen), QBrush(QColor(_RAIL_BG)))
            bg.setZValue(-20)
            # A fine diagonal hatch over the rail's ground: it reads as a plinth the
            # template stands on, which separates it from the working canvas without
            # a hard border fighting the cards for attention.
            hatch = QBrush(QColor(_RAIL_HATCH))
            hatch.setStyle(Qt.BrushStyle.BDiagPattern)
            plinth = self.scene.addRect(-26, top, CARD_W + 44, bot - top,
                                        QPen(Qt.PenStyle.NoPen), hatch)
            plinth.setZValue(-19)
            sep = self.scene.addLine(CARD_W + 22, top, CARD_W + 22, bot,
                                     QPen(QColor(_RAIL_EDGE), 2))
            sep.setZValue(-19)
            cap = QGraphicsSimpleTextItem("DEFAULT PIPELINE")
            cap.setBrush(QColor("#5c7186"))
            cf = QFont()
            cf.setPointSize(8)
            cf.setBold(True)
            cap.setFont(cf)
            cap.setPos(-18, top + 8)
            cap.setZValue(-18)
            self.scene.addItem(cap)

        edge_pen = QPen(QColor("#4a4a4a"))
        edge_pen.setWidth(2)
        rail_pen = QPen(QColor(_RAIL_EDGE))
        rail_pen.setWidth(2)
        for src, dst in edges:
            a, b = index.get(src), index.get(dst)
            if not a or not b:
                continue
            ln = self.scene.addLine(
                a["x"] + a["w"] / 2, a["y"] + a["h"],
                b["x"] + b["w"] / 2, b["y"],
                rail_pen if (a.get("is_template") and b.get("is_template"))
                else edge_pen)
            ln.setZValue(-10)

        for n in nodes:
            self._add_card(n)
        self._place_gpu_panel()      # stays pinned top-right across repaints

        # RUNNING banner — a trunk run (▶ Run) has no card of its own, so without
        # this there'd be NO on-canvas sign that anything is live.
        if self._active.get("running"):
            prog = self._active.get("progress", "")
            txt = "▶  RUNNING:  " + self._active.get("label", "")
            if prog:
                txt += f"      {prog}"
            banner = QGraphicsSimpleTextItem(txt)
            banner.setBrush(QColor("#f0a92a"))
            bf = QFont()
            bf.setPointSize(13)
            bf.setBold(True)
            banner.setFont(bf)
            banner.setPos(4, -48)
            self.scene.addItem(banner)

        rect = self.scene.itemsBoundingRect()
        self.scene.setSceneRect(rect.adjusted(-40, -40, 40, 40))
        self.view.horizontalScrollBar().setValue(hb)
        self.view.verticalScrollBar().setValue(vb)

    def _add_card(self, n):
        ghost = n["is_ghost"]
        orphan = n.get("is_orphan", False)
        # Running if this node's own job is live, OR a trunk run (▶ Run — it has no
        # card of its own) is executing this stage, so the stage's card lights up too.
        # Light up ONLY the card that is actually running. The stage_id fallback
        # exists for TRUNK runs (▶ Run), which have no job record — but it must hit
        # the stage's ghost/template card, never every sibling job of that stage
        # (that lit all four Extract cards amber at once).
        act = self._active or {}
        running = card_is_running(n, act)
        if n.get("is_template"):
            # NEVER a job colour. The rail is the pipeline reference; the moment a
            # template card goes green it reads as completed work, and the eye stops
            # being able to tell the template from the jobs beside it. Everything
            # that has actually run is a card to the RIGHT of the separator.
            fill, border = _TEMPLATE_STYLE
        else:
            fill, border = _CARD_STYLE.get("running" if running else n["status"],
                                           _CARD_STYLE["ghost"])
        item = _CardItem(n, self)
        item.setBrush(QBrush(QColor(fill)))
        pen = QPen(QColor(border))
        pen.setWidth(1 if n.get("is_template")
                     else (3 if running else 2))   # running gets a heavier outline
        if ghost or orphan:                 # un-built / found-on-disk look dashed
            pen.setStyle(Qt.PenStyle.DashLine)
        item.setPen(pen)
        self.scene.addItem(item)

        def text(s, x, y, pt, colour, bold=False, maxw=None):
            f = QFont()
            f.setPointSize(pt)
            f.setBold(bold)
            if maxw is not None:                       # keep text inside the card
                s = QFontMetrics(f).elidedText(s, Qt.TextElideMode.ElideRight, int(maxw))
            t = QGraphicsSimpleTextItem(s, item)
            t.setBrush(QColor(colour))
            t.setFont(f)
            t.setPos(x, y)
            return t

        inner_w = n["w"] - 22                           # text column width (11px margins)
        # group tag (tiny) · friendly title (bold) · raw command · status/summary
        is_tmpl = n.get("is_template", False)
        text(n.get("group", ""), 11, 6, 8, "#4c5a68" if is_tmpl else "#6f6f6f",
             maxw=inner_w)
        text(n.get("title", n["label"]), 11, 20, 11,
             "#7b8b9a" if is_tmpl else ("#8a8a8a" if ghost else "#ececec"),
             bold=True, maxw=inner_w)
        text(n.get("subtitle", n["stage_id"]), 11, 40, 8,
             "#4c5a68" if is_tmpl else "#6f6f6f", maxw=inner_w)
        template = n.get("is_template", False)
        if template:
            # A template has no state of its own — it is one step of the default
            # pipeline. What it CAN say is whether this project has any work for
            # that step, which is navigation, not status, so it stays slate.
            k = n.get("n_jobs", 0)
            sub = (f"{k} job{'s' if k != 1 else ''} →" if k
                   else ("output on disk" if n.get("on_disk") else "no jobs yet"))
            sub_colour = "#5c7186" if (k or n.get("on_disk")) else "#3f4d5a"
        elif running:
            prog = act.get("progress", "")
            sub = "▶ running" + (f" · {prog}" if prog else "")
            sub_colour = "#f0a92a"
        elif n.get("on_disk"):                 # completed outside the app (on disk)
            sub = "✓ done (on disk)" + (
                f" · {n['disk_label']}" if n.get("disk_label") else "")
            sub_colour = "#27ae60"
        elif ghost:
            sub = "not built"
            sub_colour = "#7d7d7d"
        else:
            st = summary_text(n["summary"])
            sub = n["status"] + (f" · {st}" if st else "")
            sub_colour = "#7d7d7d"
        text(sub, 11, 56, 9, sub_colour, maxw=inner_w)
        if not ghost and not orphan:
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
    # Seconds a disk-discovery result stays fresh. The canvas repaints every few
    # seconds while a job runs; without this it would rescan ceph each time.
    DISCOVERY_TTL = 60

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

        # (the queue is no longer an in-memory list — it lives in the job store
        #  as jobs with status='queued'; see queued_jobs())
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
        # One-shot: set to a stage id by _load_job_params so the next _select_stage
        # shows a past run's values VERBATIM, with the dynamic defaults suppressed.
        self._exact_params_for = None
        # One-shot: the QUEUED job the builder is about to edit. Without it the
        # builder is stage-scoped, so pressing its run button after "Build
        # downstream" created a SECOND job instead of running the one just made.
        self._builder_job_id = None
        # Debounce disk writes: a single-shot timer flushes _param_store to config
        # ~0.6s after the last edit (so typing doesn't hammer the JSON).
        self._persist_timer = QTimer(self)
        self._persist_timer.setSingleShot(True)
        self._persist_timer.setInterval(600)
        self._persist_timer.timeout.connect(self._persist_param_store)
        # Periodically re-scan for on-disk pick sets made outside the app (direct
        # terminal use); only redraw the canvas when the set actually changes.
        self._last_orphan_keys = None
        self._orphan_cache = None        # (monotonic_ts, root, [orphans])
        # A 'running' job cannot survive the process that launched it — clear any
        # left over from a crash/force-quit, or the queue thinks it is still busy.
        try:
            stale = reconcile_running(root)
            if stale:
                self._stale_jobs = stale
        except Exception:
            pass
        self._discovery_timer = QTimer(self)
        self._discovery_timer.setInterval(45000)
        self._discovery_timer.timeout.connect(self._maybe_rediscover)
        self._discovery_timer.start()
        # Live run indicator: latest "N/M" progress line + a 5s canvas repaint while
        # anything is running, so the RUNNING banner/card actually tracks the run.
        self._run_progress = ""
        self._live_timer = QTimer(self)
        self._live_timer.setInterval(5000)
        self._live_timer.timeout.connect(self._live_tick)
        self._live_timer.start()
        self._active_stage = None
        self._active_cmd = ""
        self._active_job_id = None    # set while a card-view job (not a stage) runs
        self._m_resolution = None     # MCore's final 'name: N Å', caught in _log
        self._pending_parent = {}     # stage_id -> chosen parent job for the next build
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
                                on_menu=self._card_menu,
                                on_orphans=self._discover_orphans,
                                on_active=self._active_info,
                                on_add_stage=self._add_stage_from_palette)
        self.job_stack.addWidget(
            self._panel("canvasCard", "Workflow graph", self.canvas))
        self._build_align_and_command()   # sets self.align_list_card + self.command_card
        self.details_card = self._panel("detailsCard", "Job details",
                                        self._build_details_panel(),
                                        on_close=self._hide_details)
        self.dir_card = self._panel("dirCard", "Directory overview",
                                    self._build_directory_overview())
        self.terminal_card = self._panel("rightCard", "Terminal",
                                          self._build_terminal_panel())
        self.queue_card = self._panel("queueCard", "Jobs queue",
                                      self._build_queue_panel())
        # Floor width for the right-hand column so the horizontal splitter can never
        # squeeze the command box / terminal narrow enough to truncate the command
        # text (looked broken; wrapped text needs room). Users can still widen it.
        for _c in (self.command_card, self.terminal_card):
            _c.setMinimumWidth(400)
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
        tools.addAction("Convert crYOLO picks…", self._convert_cryolo_picks)
        tools.addSeparator()
        tools.addAction("Set WarpTools launch command…", self._set_warp_launch)
        tools.addAction("Set warp-tm-vis launch command…", self._set_tm_vis_launch)
        view = mb.addMenu("View")
        self._act_canvas = view.addAction("Card (graph) view", self._toggle_view)
        self._act_canvas.setCheckable(True)
        view.addSeparator()
        view.addAction("Refresh status", self._refresh_status_dots)
        view.addAction("Show hidden cards", self._show_hidden_cards)
        view.addAction("Found on disk…", self._open_orphan_drawer)
        view.addSeparator()
        self._act_lock = view.addAction("Lock card positions", self._toggle_card_lock)
        self._act_lock.setCheckable(True)
        self._act_lock.setChecked(True)
        view.addAction("Auto-arrange cards (discard positions)",
                       self._reset_all_card_positions)
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

    def _hide_details(self):
        """Close the Job details pane (the ✕ in its header) and give the space back to
        the canvas — it is revealed by 'Details' and otherwise had no way to dismiss it."""
        self.details_card.setVisible(False)
        if getattr(self, "_canvas_split", None) is not None:
            self._canvas_split.setSizes([max(self._canvas_split.width(), 900), 0])

    def _bridge_job_outputs(self, job_id, stage_id):
        """Wrapper stages write to a path from their params, not into jobs/<id>/ — which
        left the job folder empty and the trail cold. Drop a symlink
        jobs/<id>/outputs -> <real dir> (plus a one-line WHERE_ARE_THE_OUTPUTS.txt for
        anyone browsing over ceph/SMB where symlinks may not resolve). Never overwrites
        real files; failures are logged, not raised."""
        pairs = self._job_output_dirs(job_id, stage_id, with_notes=True)
        real = [rel for rel, _ in pairs]
        if not real:
            return
        jobdir = Path(self.project_root) / job_output_dir(job_id)
        try:
            jobdir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return
        for i, rel in enumerate(real):
            link = jobdir / ("outputs" if i == 0 else f"outputs_{i + 1}")
            target = (Path(self.project_root) / rel).resolve()
            try:
                if link.is_symlink():
                    link.unlink()          # refresh a stale link from a previous run
                elif link.exists():
                    continue               # a real file/dir is there — leave it alone
                link.symlink_to(target)
            except OSError as e:
                self._log(f"{job_id}: could not link outputs -> {rel} ({e})", "warning")
        try:
            (jobdir / "WHERE_ARE_THE_OUTPUTS.txt").write_text(
                f"{job_id} wrote its results to:\n"
                + "".join(f"    {r}/{('   — ' + w) if w else ''}\n" for r, w in pairs)
                + f"(relative to the project root {self.project_root})\n"
                  f"The 'outputs' symlink here points at the first of these.\n")
        except OSError:
            pass
        self._log(f"{job_id}: outputs are in {', '.join(real)} "
                  f"(linked as jobs/{job_id}/outputs).", "info")

    def _count_dir_entries(self, rel, cap=3):
        """How many entries a project-relative dir holds (capped — ceph scandir is slow
        and we only need 'empty or not'). 0 for a missing dir, or before a project root
        is set (the details pane can be built before one exists)."""
        root = getattr(self, "project_root", None)
        if not isinstance(root, (str, os.PathLike)) or not rel:
            return 0
        try:
            p = Path(root) / rel
            if not p.is_dir():
                return 0
            return sum(1 for _ in itertools.islice(p.iterdir(), cap))
        except (OSError, TypeError, ValueError):
            return 0

    def _job_output_dirs(self, job_id, stage_id, with_notes=False):
        """Where a job's files REALLY land, project-relative.

        jobs/<id>/ is a convention, not a fact: only WarpTools stages that take
        --output_processing write there. Wrapper stages write wherever their own
        params point, and M writes a .population, a .source next to the SETTINGS,
        and a randomly-named species version folder per round — none of it under
        jobs/<id>, which is why an MCore card used to advertise an empty folder.
        The resolution itself is pure and lives in tomogration_jobs.

        Returns bare paths by default (callers that just want somewhere to look),
        or [(path, note)] with `with_notes`.
        """
        root = getattr(self, "project_root", None)
        if not isinstance(root, (str, os.PathLike)):
            return []
        try:
            job = load_jobs(root).get("jobs", {}).get(job_id) or {}
            pairs = job_real_outputs(root, job, self._stage_by_id(stage_id) or {})
        except Exception:
            return []
        return pairs if with_notes else [rel for rel, _ in pairs]

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

        # Orphan = a pick set found on disk (made outside the app). Show where it
        # is + Adopt/View, then stop (it has no job record to describe).
        if node.get("is_orphan"):
            orph = node.get("orphan", {})
            info = QLabel(f"Found on disk — not yet a job.\n\nsuffix:  {orph.get('suffix','?')}\n"
                          f"dir:  {orph.get('dir','?')}\nseries:  {orph.get('n_series','?')}"
                          f"    pixel size:  {orph.get('angpix','?')} Å")
            info.setStyleSheet("color:#c8b78a;font-size:12px;")
            info.setWordWrap(True)
            self.details_box.addWidget(info)
            self.details_box.addWidget(self._details_heading("ACTIONS"))
            adopt = QPushButton("✦ Adopt as job")
            adopt.setToolTip("Register this pick set as a completed job (symlinks its "
                             "files into jobs/J###/matching — originals stay put) so it "
                             "wires into the workflow like any other job.")
            adopt.clicked.connect(lambda _=False, o=orph: self._adopt_orphan(o))
            self.details_box.addWidget(adopt)
            view = QPushButton("🔍 View picks (warp-tm-vis)")
            view.clicked.connect(lambda _=False, n=node: self._view_picks_tm_vis(n))
            self.details_box.addWidget(view)
            opendir = self._open_dir_button(f"📂  {orph.get('dir','')}", orph.get("dir", ""))
            self.details_box.addWidget(opendir)
            self.details_card.setVisible(True)
            if getattr(self, "_canvas_split", None) is not None:
                w = max(self._canvas_split.width(), 900)
                self._canvas_split.setSizes([int(w * 0.55), int(w * 0.45)])
            return

        # crYOLO / re-extract cards borrow the ts_template_match stage but must not
        # display that id — show their own subtitle instead.
        disp = node.get("subtitle") or stage_id
        if node.get("is_ghost"):
            meta = f"{node.get('group', '')} · {stage_id} · not built yet"
        else:
            meta = f"{node.get('group', '')} · {disp} · {node.get('status', '')}  ({node.get('id')})"
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
            # WHERE THE FILES ACTUALLY ARE. Only WarpTools job stages write into
            # jobs/<id>/; wrapper stages (relion4_*, aretomo, miss-alignment) write to a
            # path given in their own params, which used to leave the details pane
            # pointing at an empty jobs/<id> folder. List the real destinations first.
            real = self._job_output_dirs(node["id"], stage_id, with_notes=True)
            for rel, why in real:
                self.details_box.addWidget(self._open_dir_button(f"📂  {rel}", rel))
                # A note earned from THIS job (which version folder it wrote, which
                # file lives there) beats the stage's generic pattern hint.
                self._add_pattern_hint("writes", why or DIR_FILE_HINTS.get(rel))
            outrel = f"jobs/{node['id']}"
            n_here = self._count_dir_entries(outrel)
            if n_here or not real:
                self.details_box.addWidget(self._open_dir_button(f"📂  {outrel}", outrel))
                if not real:
                    self._add_pattern_hint("writes",
                                           DIR_FILE_HINTS.get(outs[0]) if outs else None)
            else:
                note = QLabel(f"    (jobs/{node['id']}/ is empty — this step writes to "
                              f"the path(s) above, set in its parameters)")
                note.setStyleSheet("color:#7c7c7c;font-size:10px;")
                note.setWordWrap(True)
                self.details_box.addWidget(note)
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
                for ch in DOWNSTREAM.get(stage_id, []):
                    b = QPushButton(f"→ Build {stage_title(ch, ch)} from this")
                    b.setToolTip("Create the next job wired to THIS job, with the suffix / "
                                 "pattern auto-derived — then set your threshold and run.")
                    b.clicked.connect(lambda _=False, j=jid, c=ch: self._build_downstream(j, c))
                    self.details_box.addWidget(b)
                fork = QPushButton("⑂ Duplicate (fork)")
                fork.clicked.connect(lambda _=False, j=jid: self._fork_job(j))
                self.details_box.addWidget(fork)
                # The stage-level "Open in job builder" below shows whatever you
                # last edited, which after a few variants is nobody's run. This
                # recovers THIS job's values.
                load = QPushButton("⤓ Load this run's parameters into builder")
                load.setToolTip("Fill the job builder with the parameters this job "
                                "actually ran with, so you can inspect them or change "
                                "one and re-run. Creates nothing on its own.")
                load.clicked.connect(lambda _=False, j=jid: self._load_job_params(j))
                self.details_box.addWidget(load)
            edit = QPushButton("Open in job builder →")
            edit.setToolTip("Open this STAGE in the job builder with your last-used "
                            "values — not necessarily any particular run's.")
            if node.get("is_ghost") or not node.get("id"):
                edit.clicked.connect(lambda _=False, s=spec: self._select_stage(s))
            else:
                edit.clicked.connect(
                    lambda _=False, j=node.get("id"), sid2=stage_id:
                    self._open_job_in_builder(j, sid2))
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


    # ---- Found-on-disk drawer -------------------------------------------------
    def _open_orphan_drawer(self):
        """Everything discovered on disk but not yet part of the project, in ONE
        searchable list instead of a mile-wide row of cards on the canvas. With ~50
        RELION jobs the canvas row was unreadable and forced a disk scan on every
        repaint; this scans once when opened, and on demand."""
        if not self.project_root:
            QMessageBox.warning(self, "No project", "Open a project root first.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Found on disk")
        dlg.resize(680, 520)
        v = QVBoxLayout(dlg)

        head = QLabel("Work found under the project root that is not yet a tracked "
                      "job. Pick one, then choose what to do with it.")
        head.setWordWrap(True)
        head.setStyleSheet("color:#9a9a9a;font-size:11px;")
        v.addWidget(head)

        bar = QHBoxLayout()
        search = QLineEdit()
        search.setPlaceholderText("filter… (e.g. Select, Class3D, job009)")
        bar.addWidget(search, 1)
        rescan = QPushButton("Rescan")
        bar.addWidget(rescan)
        bw = QWidget()
        bw.setLayout(bar)
        v.addWidget(bw)

        tree = QTreeWidget()
        tree.setColumnCount(3)
        tree.setHeaderLabels(["Item", "Kind", "Location"])
        tree.setRootIsDecorated(True)
        v.addWidget(tree, 1)

        info = QLabel("")
        info.setWordWrap(True)
        info.setStyleSheet("color:#c8b78a;font-size:11px;")
        v.addWidget(info)

        row = QHBoxLayout()
        b_conv = QPushButton("→ RELION → Warp converter")
        b_cls = QPushButton("→ Select good class")
        b_adopt = QPushButton("✦ Adopt as job")
        b_open = QPushButton("📂 Open folder")
        for b in (b_conv, b_cls, b_adopt, b_open):
            row.addWidget(b)
        rw = QWidget()
        rw.setLayout(row)
        v.addWidget(rw)

        state = {"items": []}

        def selected():
            it = tree.currentItem()
            return it.data(0, Qt.UserRole) if it is not None else None

        def repopulate(force=False):
            tree.clear()
            orphs = self._discover_orphans(force=force)
            state["items"] = orphs
            groups = {}
            for o in orphs:
                kind = ("RELION " + o.get("suffix", "").split("/")[0]
                        if o.get("kind") == "relion_job" else "Pick set")
                groups.setdefault(kind, []).append(o)
            q = search.text().strip().lower()
            shown = 0
            for kind in sorted(groups):
                parent = QTreeWidgetItem([kind, "", ""])
                kids = 0
                for o in groups[kind]:
                    label = o.get("suffix", "?")
                    loc = o.get("dir", "")
                    if q and q not in f"{label} {loc} {kind}".lower():
                        continue
                    child = QTreeWidgetItem([label, kind, loc])
                    child.setData(0, Qt.UserRole, o)
                    parent.addChild(child)
                    kids += 1
                if kids:
                    tree.addTopLevelItem(parent)
                    parent.setExpanded(True)
                    shown += kids
            info.setText(f"{shown} shown · {len(orphs)} found on disk"
                         + ("  (filtered)" if q else ""))
            for i in range(3):
                tree.resizeColumnToContents(i)

        def act(fn, close=True):
            o = selected()
            if not o:
                info.setText("Select an item first.")
                return
            fn(o)
            if close:
                dlg.accept()

        search.textChanged.connect(lambda _t: repopulate(False))
        rescan.clicked.connect(lambda: repopulate(True))
        tree.itemDoubleClicked.connect(
            lambda *_: act(self._open_relion_converter))
        b_conv.clicked.connect(lambda: act(self._open_relion_converter))
        b_cls.clicked.connect(lambda: act(self._open_relion_job))
        b_adopt.clicked.connect(lambda: act(self._adopt_orphan))
        b_open.clicked.connect(
            lambda: act(lambda o: self._open_dir(o.get("dir", "")), close=False))
        repopulate(True)
        self._child_windows = getattr(self, "_child_windows", [])
        self._child_windows.append(dlg)
        dlg.show()

    # ---- orphan discovery + adoption ----
    def _discover_orphans(self, force=False):
        """On-disk work made outside the app: pick sets + finished RELION jobs.

        CACHED. This touches the filesystem, and the canvas repaints every few
        seconds while a job runs — rescanning ceph on every repaint is what made the
        UI freeze for long stretches. The scan now happens at most once per
        DISCOVERY_TTL seconds (or when something explicitly invalidates it); every
        other caller gets the cached list, which costs nothing."""
        now = time.monotonic()
        cache = getattr(self, "_orphan_cache", None)
        if (not force and cache is not None
                and now - cache[0] < self.DISCOVERY_TTL
                and cache[1] == self.project_root):
            return cache[2]
        out = []
        try:
            out += discover_picksets(self.project_root, load_jobs(self.project_root))
        except Exception:
            pass
        try:
            out += discover_relion_jobs(self.project_root)
        except Exception:
            pass
        self._orphan_cache = (now, self.project_root, out)
        return out

    def _invalidate_status(self):
        """Force the next canvas repaint to re-read per-stage on-disk status. A run
        just changed the filesystem, so the cached sweep is stale."""
        c = getattr(self, "canvas", None)
        if c is not None:
            c._status_cache = None

    def _invalidate_orphans(self):
        """Force the next discovery to hit disk (after adopting / deleting / a run)."""
        self._orphan_cache = None

    def _maybe_rediscover(self):
        """Timer tick: refresh the canvas only if the found-on-disk set changed (and
        only in card view, to avoid churn while the user is in the list view)."""
        if getattr(self, "_view_mode", "lists") != "canvas":
            return
        keys = {(o.get("dir"), o.get("suffix")) for o in self._discover_orphans(force=True)}
        if keys != self._last_orphan_keys:
            self._last_orphan_keys = keys
            self._refresh_canvas()

    def _active_info(self):
        """What's running right now, for the canvas RUNNING banner. Covers BOTH a
        card-view job AND a three-column trunk run (▶ Run), which has no card of its
        own — without this the canvas gives no sign that anything is live."""
        if not self.runner.busy():
            return {}
        prog = getattr(self, "_run_progress", "")
        jid = getattr(self, "_active_job_id", None)
        if jid:
            job = load_jobs(self.project_root).get("jobs", {}).get(jid, {}) or {}
            return {"running": True, "job_id": jid, "progress": prog,
                    "label": f"{jid} · {stage_title(job.get('stage_id', ''))}"}
        sid = getattr(self, "_active_stage", None)
        if sid:
            return {"running": True, "stage_id": sid, "progress": prog,
                    "label": f"{stage_title(sid)}  (trunk run — not a job)"}
        return {"running": True, "progress": prog, "label": "job"}

    def _live_tick(self):
        """While something is running, repaint the canvas so the RUNNING banner and
        its N/M progress stay current. Idle -> no work."""
        if getattr(self, "_view_mode", "lists") == "canvas" and self.runner.busy():
            self._refresh_canvas()

    def _adopt_relion_job(self, orph):
        """Adopt a finished RELION job (Refine3D / Class3D / Select) as a DAG node.

        Adoption used to assume every orphan was a template-match pick set, so a
        Refine3D result came back labelled "Template matching" with 0 series and no
        way to build M from it. A RELION job is a different animal: nothing needs
        linking (the files stay where RELION put them), and what downstream steps
        actually need are its half maps and particle star. RELION's filenames differ
        between a converged run (run_half1_class001_unfil.mrc) and a mid-iteration
        one (run_itNNN_half1_class001_unfil.mrc), so LOOK rather than guess."""
        rel = orph.get("dir", "")
        d = Path(self.project_root) / rel
        if not d.is_dir():
            self._log(f"Adopt: RELION job folder gone: {rel}", "fail")
            return

        def newest(*patterns):
            for pat in patterns:
                hits = sorted(d.glob(pat))
                if hits:
                    return os.path.relpath(hits[-1], self.project_root)
            return ""

        jtype = str(orph.get("suffix", "")).split("/")[0] or "RELION"
        # Count the particles ONCE, here, and store it. The canvas repaints every few
        # seconds while a job runs; re-reading a 6 MB star each time to label a card
        # would put a filesystem hit on every frame.
        star = orph.get("star", "")
        n_particles = star_particle_count(Path(self.project_root) / star) if star else None
        params = {
            "job_dir": rel,
            "job_type": jtype,
            "n_particles": n_particles or "",
            "data_star": orph.get("star", ""),
            "half1": newest("run_half1_class001_unfil.mrc",
                            "run_it*_half1_class001_unfil.mrc", "*half1*unfil.mrc"),
            "half2": newest("run_half2_class001_unfil.mrc",
                            "run_it*_half2_class001_unfil.mrc", "*half2*unfil.mrc"),
            "class_map": newest("run_class001.mrc", "run_it*_class001.mrc",
                                "*_class001.mrc"),
        }
        job = new_job(self.project_root, "relion4_result",
                      orph.get("suffix", "RELION job"), params, inputs={})
        update_job(self.project_root, job["id"], status="completed", exit_code=0,
                   tool="adopted", summary={},
                   finished=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        found = [k for k in ("half1", "half2", "class_map", "data_star") if params[k]]
        self._log(f"Adopted {orph.get('suffix', rel)} as {job['id']} "
                  f"(found: {', '.join(found) or 'star only'}).", "ok")
        if jtype == "Refine3D" and params["half1"] and params["half2"]:
            self._log(f"  → right-click {job['id']} ▸ Build downstream ▸ "
                      f"'M: create mask' then 'M: create species' — the half maps and "
                      f"particle star are filled in for you.", "info")
        elif not (params["half1"] and params["half2"]):
            self._log(f"  note: no unfiltered half maps in {rel} — M's create-species "
                      f"needs them, so this job can feed re-extraction but not M yet.",
                      "warning")
        self._invalidate_orphans()
        self._refresh_canvas()

    def _adopt_orphan(self, orph):
        """Turn a found-on-disk pick set into a real (completed) job: symlink its
        STAR + corr/score files into jobs/J###/matching (non-destructive — the
        originals stay put), and record the job with its inferred params so it
        wires into the DAG like any other. Symlinks (not copies) keep it cheap.

        RELION jobs take a different path — they need no linking, and what matters is
        their half maps / particle star (see _adopt_relion_job)."""
        if orph.get("kind") == "relion_job":
            return self._adopt_relion_job(orph)
        suffix = orph.get("suffix", "")
        src_rel = orph.get("dir", "")
        src = Path(self.project_root) / src_rel
        if not src.is_dir():
            self._log(f"Adopt: source dir gone: {src_rel}", "fail")
            return
        params = {"override_suffix": suffix, "tomo_angpix": orph.get("angpix", "")}
        job = new_job(self.project_root, "ts_template_match",
                      f"matching {suffix} (adopted)", params, inputs={})
        dst = Path(self.project_root) / job["output_dir"] / "matching"
        try:
            dst.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self._log(f"Adopt: cannot make {dst}: {e}", "fail")
            delete_job(self.project_root, job["id"])
            return
        # link this suffix's stars + the shared template corr/score maps
        corr_suffix = template_corr_suffix(params)
        n = 0
        for f in itertools.islice(sorted(src.glob(f"*Apx{suffix}.star")), 0, 20000):
            n += self._symlink_into(f, dst)
        if corr_suffix:
            for pat in (f"*Apx{corr_suffix}_corr.mrc", f"*Apx{corr_suffix}_angleid.mrc"):
                for f in itertools.islice(sorted(src.glob(pat)), 0, 20000):
                    self._symlink_into(f, dst)
        update_job(self.project_root, job["id"], status="completed", exit_code=0,
                   orphan_suffix=suffix,
                   summary={"series": orph.get("n_series", n)},
                   finished=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self._log(f"Adopted pick set '{suffix}' from {src_rel} as {job['id']} "
                  f"({n} series linked into {job['output_dir']}/matching).", "ok")
        self._invalidate_orphans()
        self._refresh_canvas()

    def _open_relion_job(self, orph):
        """A found-on-disk RELION job: load its star into the 'select good class'
        builder so the user only types the good class number and picks MODE A/B. Use
        this for a raw Class3D result (has class numbers). Reuses the same param-store
        prefill as 'Build downstream' — no new job is created here."""
        star = orph.get("star", "")
        spec = self._stage_by_id("relion4_select_picks")
        if not spec:
            return
        self._param_store.setdefault("relion4_select_picks", {})["class_star"] = star
        self._persist_param_store()
        self._select_stage(spec)
        self._log(f"Loaded {star} into 'RELION 4: select good class'. Type the good "
                  f"class number(s), choose MODE A or B, and dry-run first (EXECUTE off) "
                  f"to see the class populations.", "ok")

    def _open_relion_converter(self, orph):
        """A found-on-disk RELION job: load its star into the 'RELION 4 → Warp
        re-extract' builder to re-extract ALL its particles at a finer bin. Use this
        for a Subset-selection output (good class, duplicates removed) — the star you
        fed to Refine3D. Prefills particles_star; the user just picks MODE A/B."""
        star = orph.get("star", "")
        spec = self._stage_by_id("relion4_to_warp")
        if not spec:
            return
        self._param_store.setdefault("relion4_to_warp", {})["particles_star"] = star
        self._persist_param_store()
        self._select_stage(spec)
        self._log(f"Loaded {star} into 'RELION 4 → Warp: re-extract'. Choose MODE A or B "
                  f"and dry-run first (EXECUTE off) to see how many particles matched.", "ok")

    def _register_reextract_pickset(self, values):
        """After a successful RELION→Warp re-extract (relion4_to_warp /
        relion4_select_picks with EXECUTE on), register the pick stars it wrote as a
        COMPLETED pick-set card — shaped like an adopted template-match set (crYOLO-
        style) so right-click ▸ Build downstream ▸ ts_export_particles wires straight
        to it. Stamps the source RELION star so its 'found on disk' card turns green
        (used). No-op on a dry run or if no pick stars were written."""
        out_rel = str(values.get("out_dir", "")).strip()
        source = str(values.get("particles_star") or values.get("class_star") or "").strip()
        if not out_rel or not self.project_root:
            return
        root = Path(self.project_root)
        out_dir = root / out_rel
        if not out_dir.is_dir():
            return
        stars = sorted(itertools.islice(out_dir.glob("Position*Apx*.star"), 0, 20000))
        angpix = suffix = None
        for s in stars:
            m = _PICK_STAR_RE.match(s.name)
            if m:
                _, angpix, suffix = m.groups()
                break
        if not stars or suffix is None:
            return                      # dry run / unexpected naming — nothing to register
        src_tag = "/".join(source.split("/")[-3:-1]) if "/" in source else source  # Select/job009
        # Promote the source RELION selection into a real (green) job node FIRST, so it
        # leaves the found-on-disk orphan chain and the re-extract card can wire to it.
        sel_id = self._ensure_selection_job(source, src_tag)
        params = {"override_suffix": suffix, "tomo_angpix": angpix, "source_star": source}
        label = f"re-extract picks ({src_tag})" if src_tag else "re-extract picks"
        inputs = {"selection": sel_id} if sel_id else {}
        job = new_job(self.project_root, "ts_template_match", label, params, inputs)
        dst = root / job["output_dir"] / "matching"
        try:
            dst.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self._log(f"Re-extract card: cannot make {dst}: {e}", "fail")
            delete_job(self.project_root, job["id"])
            return
        n = 0
        for s in stars:
            n += self._symlink_into(s, dst)
        update_job(self.project_root, job["id"], status="completed", exit_code=0,
                   tool="reextract", orphan_suffix=suffix, summary={"series": n},
                   finished=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        # These coords are ABSOLUTE PIXELS at the pixel size in the filename — the
        # opposite of crYOLO's normalised 0-1 picks. This message used to say "keep
        # --normalized_coords ON", copied from the crYOLO path, which is precisely
        # the mistake that extracts every particle in one corner. State the flags the
        # converter itself printed, using the pixel size parsed off these very files.
        self._log(f"Re-extract pick set ({src_tag or 'RELION'}) → {job['id']} "
                  f"({n} tomograms linked). Right-click it ▸ Build downstream ▸ "
                  f"ts_export_particles. These coords are pixels at {angpix} Å/px: set "
                  f"--coords_angpix {angpix} and leave --normalized_coords OFF, then "
                  f"choose output_angpix (finer bin) and DOUBLE the box each time you "
                  f"halve it.", "ok")
        self._refresh_canvas()

    def _ensure_selection_job(self, source, src_tag):
        """Get (or create) the tracked job node that represents a consumed RELION
        selection, so its 'found on disk' card is promoted into the green job tree.
        Reused across re-runs (keyed on source_star) so a star never spawns duplicates.
        Shaped as a ts_template_match job (Pick row = the green section) with
        tool='relion_selection' so the canvas titles it 'RELION selection' and its menu
        offers no bogus downstream. Returns the job id (or None if it can't be made)."""
        if not source:
            return None
        store = load_jobs(self.project_root)
        for jid, j in (store.get("jobs", {}) or {}).items():
            if j.get("tool") == "relion_selection" and \
                    str((j.get("params") or {}).get("source_star")) == source:
                return jid
        try:
            # Same reasoning as _adopt_relion_job: read the star once, at creation,
            # so the card can say WHICH selection and how big without touching disk
            # on every repaint.
            n_particles = star_particle_count(Path(self.project_root) / source)
            job = new_job(self.project_root, "ts_template_match",
                          src_tag or "RELION selection",
                          {"source_star": source, "job_dir": src_tag or source,
                           "job_type": src_tag or source,
                           "n_particles": n_particles or ""}, inputs={})
            update_job(self.project_root, job["id"], tool="relion_selection",
                       status="completed", exit_code=0, summary={},
                       finished=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            return job["id"]
        except Exception:
            return None

    def _maybe_register_reextract(self, values, code):
        """Fire _register_reextract_pickset only on a successful EXECUTE run. Guarded so
        a dry run or a failure never spawns a card. Never raises into the run loop."""
        if code != 0 or not values or not values.get("execute"):
            return
        try:
            self._register_reextract_pickset(values)
        except Exception as e:
            self._log(f"Re-extract card registration failed: {e}", "fail")

    @staticmethod
    def _symlink_into(src_file, dst_dir):
        link = dst_dir / src_file.name
        try:
            if link.exists() or link.is_symlink():
                return 0
            link.symlink_to(os.path.abspath(src_file))
            return 1
        except OSError:
            return 0

    def _convert_cryolo_picks(self):
        """Tools ▶ Convert crYOLO picks: turn a crYOLO tomo-picking COORDS/ folder
        into normalised per-tomogram Warp pick STARs (via
        ml_cryolo_to_warp_picks_auto.py) and register them as a COMPLETED pick-set
        job card — shaped exactly like an adopted template-match set, so
        'Build downstream ▶ ts_export_particles' extracts straight off the crYOLO
        picks. crYOLO stands in for ts_template_match + threshold_picks; everything
        downstream is unchanged."""
        if not self.project_root:
            QMessageBox.warning(self, "No project", "Open a project root first.")
            return
        root = Path(self.project_root)
        # 1. gather inputs with standard dialogs (no new widgets to get wrong).
        coords = QFileDialog.getExistingDirectory(
            self, "crYOLO COORDS/ folder", str(root), NONATIVE_DIR)
        if not coords:
            return
        recon_default = root / "warp_tiltseries" / "reconstruction"
        recon = QFileDialog.getExistingDirectory(
            self, "Reconstruction folder crYOLO picked on",
            str(recon_default if recon_default.is_dir() else root), NONATIVE_DIR)
        if not recon:
            return
        apx, ok = QInputDialog.getText(
            self, "Pixel-size tag", "Reconstruction angpix tag in the filenames "
            "(PositionNNN_<apx>Apx.mrc):", text="12.56")
        if not ok or not apx.strip():
            return
        tag, ok = QInputDialog.getText(
            self, "Pick-set name", "Short tag for this pick set "
            "(files become *_<apx>Apx_<tag>.star):", text="cryolo")
        if not ok or not tag.strip():
            return
        apx, tag = apx.strip(), tag.strip()
        flip_y = QMessageBox.question(
            self, "Mirror Y?",
            "Mirror the Y axis (y → 1 − y)?\n\nChoose No unless a prior one-tomogram "
            "check showed crYOLO picks come out Y-flipped versus the tomogram.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes

        # 2. create the pick-set card FIRST, so the converter writes into its dir.
        #    Params mirror an adopted template-match set: match_star_infix(params) =
        #    '<apx>Apx_<tag>', which is both the file infix and the export pattern.
        params = {"override_suffix": f"_{tag}", "tomo_angpix": apx}
        job = new_job(self.project_root, "ts_template_match",
                      f"crYOLO picks '{tag}'", params, inputs={})
        dst = root / job["output_dir"] / "matching"
        try:
            dst.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            delete_job(self.project_root, job["id"])
            QMessageBox.critical(self, "Convert crYOLO picks", f"Cannot make {dst}: {e}")
            return

        # 3. run the converter into the card's matching dir.
        script = Path(__file__).resolve().parent / "ml_cryolo_to_warp_picks_auto.py"
        cmd = [sys.executable, str(script), coords, recon,
               "--out_dir", str(dst), "--apx", apx, "--suffix", tag, "--execute"]
        if flip_y:
            cmd.append("--flip_y")
        self._log("Convert crYOLO picks: " + " ".join(cmd), "info")
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.SubprocessError) as e:
            delete_job(self.project_root, job["id"])
            QMessageBox.critical(self, "Convert crYOLO picks", f"Converter failed: {e}")
            return
        out = (res.stdout or "") + (res.stderr or "")
        n = len(list(dst.glob(f"*_{apx}Apx_{tag}.star")))
        if res.returncode != 0 or n == 0:
            delete_job(self.project_root, job["id"])
            QMessageBox.critical(
                self, "Convert crYOLO picks",
                f"No pick STARs written (exit {res.returncode}).\n\n{out[-3000:]}")
            return

        # 4. mark completed — now it wires into the DAG like any adopted pick set.
        #    tool="cryolo" so the canvas titles it as crYOLO, not 'Template matching'.
        update_job(self.project_root, job["id"], status="completed", exit_code=0,
                   tool="cryolo", orphan_suffix=f"_{tag}", summary={"series": n},
                   finished=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self._log(f"crYOLO picks '{tag}' → {job['id']} ({n} tomograms) in "
                  f"{job['output_dir']}/matching.", "ok")
        self._refresh_canvas()
        QMessageBox.information(
            self, "Convert crYOLO picks",
            f"Registered {n} tomograms as pick-set job {job['id']}.\n\n"
            f"Right-click the card ▶ Build downstream from this ▶ ts_export_particles "
            f"to extract. Turn --normalized_coords ON in the export step.\n\n{out[-1500:]}")

    def _card_menu(self, node, global_pos):
        """Right-click menu on a canvas card."""
        menu = QMenu(self)
        sid = node.get("stage_id")
        if node.get("is_orphan"):
            orph = node.get("orphan", {})
            if orph.get("kind") == "relion_job":
                menu.addAction("Open in RELION → Warp converter (re-extract all)",
                               lambda: self._open_relion_converter(orph))
                menu.addAction("Open in select good class (filter by class)",
                               lambda: self._open_relion_job(orph))
            else:
                menu.addAction("Adopt as job", lambda: self._adopt_orphan(orph))
                menu.addAction("View picks (warp-tm-vis)",
                               lambda: self._view_picks_tm_vis(node))
            menu.addAction("Details", lambda: self._show_card_details(node))
            menu.addSeparator()
            menu.addAction("Hide (remove from view)", lambda: self._hide_card(node))
            menu.addAction("Delete folder from disk…",
                           lambda: self._delete_orphan_dir(orph))
            menu.exec(global_pos)
            return
        if node.get("is_ghost"):
            menu.addAction("Build & run job", lambda: self._build_job(sid, run=True))
            menu.addAction("＋ Queue this job", lambda: self._queue_stage(sid))
            menu.addAction("Build (don't run)", lambda: self._build_job(sid, run=False))
            # A template has no job to bind to — open the STAGE.
            menu.addAction("Open in job builder", lambda: self._canvas_pick(sid))
        else:
            jid = node.get("id")
            status = node.get("status", "")
            if status == "running":
                menu.addAction("■ Kill (stop this job)", lambda: self._kill_job(jid))
                menu.addSeparator()
            elif status == "queued":
                menu.addAction("▶ Run now (jump the queue)",
                               lambda: self._run_queued_job(jid))
                menu.addAction("✕ Remove from queue", lambda: self._delete_job(jid))
                menu.addSeparator()
            else:
                # Failed or finished: requeue is the CryoSPARC-style 'try it again'.
                menu.addAction("↻ Restart (queue again)", lambda: self._requeue_job(jid))
            menu.addAction("Run / re-run", lambda: self._run_job(jid))
            # A promoted RELION selection rides ts_template_match but has no picks of
            # its own — don't offer threshold/export downstream from it.
            children = [] if node.get("title") == "RELION selection" else DOWNSTREAM.get(sid, [])
            if children:
                sub = menu.addMenu("Build downstream from this")
                for ch in children:
                    sub.addAction(stage_title(ch, ch),
                                  lambda _=False, c=ch: self._build_downstream(jid, c))
            menu.addAction("Duplicate (fork)", lambda: self._fork_job(jid))
            menu.addAction("Details", lambda: self._show_card_details(node))
            menu.addAction("⤓ Load this run's parameters into builder",
                           lambda: self._load_job_params(jid))
            menu.addAction("⇄ Set input (which job feeds this)…",
                           lambda: self._set_job_parent(jid))
            menu.addAction("⌖ Reset this card's position",
                           lambda: self._reset_card_position(jid))
            menu.addAction("Open in job builder",
                           lambda: self._open_job_in_builder(jid, sid))
            menu.addSeparator()
            menu.addAction("⟲ Clear job — delete its results, back to Building",
                           lambda: self._clear_job(jid))
            menu.addAction("Hide (remove from view)", lambda: self._hide_card(node))
            menu.addAction("Delete job (keep files)", lambda: self._delete_job(jid))
            menu.addAction("⚠ Delete job AND its files…",
                           lambda: self._delete_job_permanent(jid))
        menu.exec(global_pos)

    def _build_job(self, stage_id, params=None, run=True, parent=None, confirm=True):
        """Create a job instance for a stage (auto-wiring its input to the newest
        upstream WarpTools job) and optionally run it.

        `confirm=False` skips the pre-flight dialogs. They ask about things a RUN
        would do — overwriting an output folder, ignoring a validator warning — so
        asking them while merely placing a card on the canvas is a question about a
        command that is not going to be executed.
        """
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
            # a parent explicitly chosen via "Build downstream" wins over the
            # newest-upstream default
            parent = self._pending_parent.pop(stage_id, None) or default_parent_for(stage_id, store)
        inputs = {"processing": parent} if parent else {}
        if confirm and not self._confirm_validator(spec, params):
            return None
        if confirm and not self._confirm_overwrite(spec, params):
            return None
        job = new_job(self.project_root, stage_id, spec.get("label", stage_id),
                      params, inputs)
        self._log(f"Built {job['id']} · {job['label']}"
                  + (f"  (input ← {parent})" if parent else "  (input ← trunk)"), "ok")
        self._refresh_canvas()
        if run:
            self._run_job(job["id"])
        return job

    def _confirm_validator(self, spec, params):
        """Make a stage's ⚠ warnings BLOCK, not merely decorate.

        The warnings were a red label under the form, and nothing stopped a job being
        built, queued or run with one active. An export went out with the coordinate
        scale 4x wrong and its star aimed at the previous round's folder — both
        already detected, both silently ignored, and the mistake only surfaced hours
        later in the extracted data. Returns True to proceed.
        """
        fn = spec.get("validate")
        if not fn:
            return True
        try:
            msg = fn(params) or ""
        except Exception:
            return True
        if not msg.strip():
            return True
        return QMessageBox.warning(
            self, f"{spec.get('label', spec['id'])} — check these first",
            f"{msg}\n\nThese are silent failures: the command will run and exit 0, "
            f"and the damage only shows up in the results.\n\nRun anyway?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes

    def _confirm_overwrite(self, spec, params):
        """If this step's declared output dir already holds files, say so and ask.

        Wrapper stages write to a path from their own params (output_processing,
        out_dir, project_dir), so a re-run silently writes over the last result.
        Warp/RELION won't warn — the first sign is mixed-up output. Returns True to
        proceed. Only ever LOOKS; nothing is deleted here."""
        keys = spec.get("output_params") or []
        root = getattr(self, "project_root", None)
        subdirs = spec.get("output_subdirs") or []
        if not (keys or subdirs) or not isinstance(root, (str, os.PathLike)):
            return True
        # A stage that writes into a fixed SUBFOLDER only endangers that subfolder.
        # relion4_class3d's project_dir is a RELION project root holding
        # matching_conv.star, subtomo/ and previous jobs — none of which it touches
        # (it writes Class3D/job001/ and parks pipeline state aside). Warning about
        # the container was a false alarm about the wrong files, and it drowned out
        # the real risk. So when subdirs are declared, warn about THOSE, and drop the
        # container they hang off.
        cands, base = [], ""
        if subdirs:
            key = spec.get("output_subdir_param") or "output_processing"
            base = str((params or {}).get(key, "") or "").strip().rstrip("/")
            if not base:
                try:
                    base = settings_processing_dir(
                        root, (params or {}).get(spec.get("settings_param", "settings"), ""))
                except Exception:
                    base = ""
            if base:
                cands += [f"{base}/{s}" for s in subdirs]
        for k in keys:
            rel = str((params or {}).get(k, "") or "").strip().rstrip("/")
            if rel and rel != base:
                cands.append(rel)
        hits = []
        for rel in cands:
            if not rel:
                continue
            n = self._count_dir_entries(rel, cap=2)
            if n and rel not in hits:
                hits.append(rel)
        if not hits:
            return True
        where = "\n".join(f"    {h}/" for h in hits)
        return QMessageBox.question(
            self, "Output folder is not empty",
            f"{spec.get('label', spec['id'])} writes into:\n\n{where}\n\n"
            f"There are already files there. Re-running will write over results "
            f"with the same names (anything with a different name is left alone).\n\n"
            f"Continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) == QMessageBox.Yes

    def _build_downstream(self, parent_id, child_stage_id):
        """CREATE the next job, wired to a SPECIFIC parent, and place it below it.

        This used to create nothing at all: it derived the fiddly params, seeded the
        job builder and stashed the parent for whenever the user next pressed
        'Build & run as job'. The menu says "Build downstream from this", so a card
        was expected on the canvas — and when none appeared the natural next move was
        to drag one in from the palette, which arrives BLANK and then inherited the
        stashed parent. That is how a re-extract ran with no particle star at all.

        It now makes the job immediately: queued, wired, parameters derived, pinned
        under its parent, and opened in the builder for editing.
        """
        store = load_jobs(self.project_root)
        parent = store.get("jobs", {}).get(parent_id)
        spec = self._stage_by_id(child_stage_id)
        if not parent or not spec:
            return
        derived = derive_child_params(child_stage_id, parent.get("stage_id"),
                                      parent.get("params", {}),
                                      parent.get("output_dir", ""))
        params = self._effective_params(spec)
        params.update(derived)
        self._param_store.setdefault(child_stage_id, {}).update(derived)
        self._persist_param_store()
        # Explicit parent, so nothing is left stashed for a later, unrelated build.
        self._pending_parent.pop(child_stage_id, None)
        # Pick the slot BEFORE creating the job, so the new card is not compared
        # against itself.
        try:
            slot = self._free_slot_below(parent_id)
        except Exception:
            slot = None
        job = self._build_job(child_stage_id, params=params, run=False,
                              parent=parent_id, confirm=False)
        jid = (job or {}).get("id")
        if not jid:
            return
        if slot is not None:
            try:
                set_card_position(self.project_root, jid, slot[0], slot[1])
            except Exception:
                pass
        bits = [f"{k}={v}" for k, v in derived.items() if v not in ("", None, False)]
        self._log(f"Built {jid} · {stage_title(child_stage_id, child_stage_id)} "
                  f"from {parent_id}"
                  + (f"  ({', '.join(bits[:3])})" if bits else "")
                  + f".  Adjust its parameters, then ▶ Save & run {jid}.", "ok")
        self._refresh_canvas()
        # Bind the builder to THIS card, so its run button runs it instead of
        # creating a second job for the same step.
        self._builder_job_id = jid
        self._select_stage(spec)

    def _open_job_in_builder(self, job_id, stage_id):
        """Open a card in the builder, bound to it when it is still QUEUED.

        Binding only happened right after a card was created, so reopening a queued
        card later left the builder stage-scoped — and its buttons then made a
        duplicate card instead of editing the one on screen. A queued job is exactly
        the case where the form should be editing THAT job.
        """
        spec = self._stage_by_id(stage_id)
        if not spec:
            return
        job = (load_jobs(self.project_root).get("jobs") or {}).get(job_id) or {}
        if job.get("status") in ("building", "queued"):
            self._param_store[stage_id] = dict(job.get("params") or {})
            self._persist_param_store()
            self._exact_params_for = stage_id     # show its values, not the defaults
            self._builder_job_id = job_id
        self._select_stage(spec)

    def _save_queued_job(self, job_id):
        """Write the form's values into an already-queued job and LEAVE it queued.

        "+ Queue variant" always minted a new job, which is right when you want a
        second variant to run alongside and wrong when you are editing a card that is
        already sitting in the queue — it produced a duplicate card for the same step.
        Queued jobs run in card order when the one before them finishes, so there is
        nothing else to do here but save.
        """
        store = load_jobs(self.project_root)
        job = (store.get("jobs") or {}).get(job_id)
        spec = (self.current or {}).get("spec") or {}
        if not job:
            self._log(f"{job_id} no longer exists — nothing saved.", "warn")
            return
        values = self._values()
        if not self._confirm_validator(spec, values):
            return
        cmd = (self.current["cmd"].toPlainText().strip()
               if self.current.get("manual") else "")
        update_job(self.project_root, job_id, params=values, status="queued",
                   command=cmd)
        ahead = [j["id"] for j in queued_jobs(load_jobs(self.project_root))
                 if _job_seq(j["id"]) < _job_seq(job_id)]
        when = (f"after {', '.join(ahead)}" if ahead else "next")
        self._log(f"Saved {job_id}; it stays queued and runs {when}"
                  + (" (running your edited command verbatim)." if cmd else "."), "ok")
        self._refresh_canvas()
        self._refresh_queue()

    def _save_and_run_job(self, job_id):
        """Write the form's values back into an existing QUEUED job and run it.

        The builder is stage-scoped, so its run button always built a NEW job. After
        "Build downstream" — which now creates the card first — that produced a
        second card for the same step, running off on its own while the one you were
        editing sat queued forever.
        """
        store = load_jobs(self.project_root)
        job = (store.get("jobs") or {}).get(job_id)
        spec = (self.current or {}).get("spec") or {}
        if not job:
            # The card was deleted while the builder was open. Building a fresh job
            # is the sane fallback, but say so rather than silently doing it.
            self._log(f"{job_id} no longer exists — building a new job instead.",
                      "warn")
            if spec:
                self._build_job(spec["id"], run=True)
            return
        values = self._values()
        if not self._confirm_validator(spec, values):
            return
        if not self._confirm_overwrite(spec, values):
            return
        update_job(self.project_root, job_id, params=values)
        if self.current.get("manual"):
            # A hand-edited command is the user's explicit intent — store it and run
            # it verbatim rather than rebuilding it from the controls.
            update_job(self.project_root, job_id,
                       command=self.current["cmd"].toPlainText().strip())
            self._log(f"Saved your edited command into {job_id}.", "info")
            self._run_queued_job(job_id)
        else:
            update_job(self.project_root, job_id, command="")   # rebuilt by _run_job
            self._run_job(job_id)

    def _free_slot_below(self, parent_id):
        """A free position below `parent_id` for a new child card.

        Every child used to be pinned at exactly "parent + one row", so building a
        SECOND job downstream of the same parent dropped it precisely on top of the
        first. The old card was still there and untouched, but it was completely
        hidden — which reads as the existing card having been reused or overwritten.
        Siblings step to the right instead, matching how forks are already laid out.
        """
        px, py = self._node_position(parent_id)
        x, y = px, py + CARD_H + GAP_Y
        store = load_jobs(self.project_root)
        nodes, _ = canvas_layout(store, [], {}, set(store.get("hidden", [])))
        taken = [(n["x"], n["y"]) for n in nodes if n.get("id") != parent_id]
        # Bounded: a canvas with a card in every column would otherwise spin.
        for _ in range(len(taken) + 2):
            if not any(abs(x - tx) < CARD_W and abs(y - ty) < CARD_H
                       for tx, ty in taken):
                break
            x += CARD_W + GAP_X
        return x, y

    def _node_position(self, node_id):
        """Where a card currently sits on the canvas, honouring any user placement."""
        store = load_jobs(self.project_root)
        pos = (store.get("positions") or {}).get(str(node_id))
        if isinstance(pos, (list, tuple)) and len(pos) == 2:
            return float(pos[0]), float(pos[1])
        for n in canvas_layout(store, [], {}, set(store.get("hidden", [])))[0]:
            if n["id"] == node_id:
                return float(n["x"]), float(n["y"])
        return 0.0, 0.0

    def _toggle_card_lock(self, checked):
        """View-menu mirror of the canvas toolbar's lock button (one state, two
        places to reach it)."""
        c = getattr(self, "canvas", None)
        if c is not None:
            c.lock_btn.setChecked(bool(checked))

    def _reset_card_position(self, node_id):
        try:
            clear_card_positions(self.project_root, node_id)
        except Exception as e:
            self._log(f"Could not reset position: {e}", "warn")
            return
        self._refresh_canvas()

    def _reset_all_card_positions(self):
        try:
            clear_card_positions(self.project_root)
        except Exception as e:
            self._log(f"Could not reset positions: {e}", "warn")
            return
        self._log("Card positions discarded — back to the computed layout.", "info")
        self._refresh_canvas()

    def _add_stage_from_palette(self, stage_id, x, y):
        """Create a job for a stage dragged out of the palette, pinned where it was
        dropped. QUEUED, never run: dropping a card is a layout gesture, and running
        a GPU job because someone let go of the mouse in the wrong place would be
        indefensible. Open it in the builder so its parameters are the next thing
        you see."""
        spec = self._stage_by_id(stage_id)
        if not spec or not self.project_root:
            return
        # TEMPLATE defaults, not _effective_params: the persisted store holds the
        # last run's values, so a freshly dropped card arrived pre-loaded with a real
        # output folder from a previous round and immediately asked about overwriting
        # it. A card you just dragged onto the canvas knows nothing yet.
        job = self._build_job(stage_id, params=stage_defaults(spec), run=False,
                              confirm=False)
        jid = (job or {}).get("id")
        if not jid:
            self._log(f"Could not add {stage_title(stage_id, stage_id)} to the canvas.",
                      "warn")
            return
        try:
            set_card_position(self.project_root, jid, x - CARD_W / 2, y - CARD_H / 2)
        except Exception:
            pass
        self._log(f"Added {stage_title(stage_id, stage_id)} as {jid} (queued — set "
                  f"its parameters, then ▶ Save & run {jid}).", "ok")
        self._refresh_canvas()
        self._builder_job_id = jid
        self._select_stage(spec)

    def _set_job_parent(self, job_id):
        """Re-wire which job feeds this one. Adoption records no inputs, so an
        adopted RELION job draws no edge however obviously it feeds the next step —
        and the graph then lies about the lineage. This makes it editable."""
        store = load_jobs(self.project_root)
        jobs_map = store.get("jobs", {}) or {}
        job = jobs_map.get(job_id)
        if not job:
            return
        choices, labels = [], []
        for jid, j in sorted(jobs_map.items(), key=lambda t: _job_seq(t[0])):
            if jid == job_id:
                continue
            choices.append(jid)
            labels.append(f"{jid} · {stage_title(j.get('stage_id',''), j.get('stage_id',''))}"
                          f" · {j.get('label','')[:40]}")
        if not choices:
            self._log("No other jobs to connect to yet.", "info")
            return
        cur = next((p for p in (job.get("inputs") or {}).values() if p), None)
        labels.insert(0, "(no input — detach)")
        choices.insert(0, "")
        idx = choices.index(cur) if cur in choices else 0
        pick, ok = QInputDialog.getItem(
            self, "Set input", f"Which job feeds {job_id}?", labels, idx, False)
        if not ok:
            return
        parent = choices[labels.index(pick)]
        if set_job_parent(self.project_root, job_id, parent or None) is None and parent:
            self._log(f"Could not connect {job_id} to {parent}.", "warn")
            return
        self._log(f"{job_id} now reads from {parent}." if parent
                  else f"{job_id} detached from its input.", "ok")
        self._refresh_canvas()

    def _load_job_params(self, job_id):
        """Populate the job builder with the parameters a past job actually ran with.

        Every job record keeps the params it was built from, but until now the only
        way back to them was to read the card's details and retype. That matters
        most for M, where you run MCore a dozen times with one flag different and
        the useful question is always "what exactly did the good one use?".

        This does NOT create a job — it fills the builder so you can inspect the
        values, tweak one, and then Run / Build & run / Queue. Fork if you want a
        new card straight away.
        """
        store = load_jobs(self.project_root)
        job = store.get("jobs", {}).get(job_id)
        if not job:
            self._log(f"{job_id}: no such job.", "warn")
            return
        stage_id = job.get("stage_id", "")
        spec = self._stage_by_id(stage_id)
        if not spec:
            self._log(f"{job_id}: its stage '{stage_id}' no longer exists in this "
                      f"version of tomogration, so its parameters cannot be loaded "
                      f"into the builder. The command it ran is on the card.", "warn")
            return

        # Keep only params the stage still declares. A stage that gained or lost a
        # parameter since the run would otherwise inject a key the form cannot show
        # (invisible, but still passed to build_command) — so drop those and say so.
        kept, dropped, missing = params_for_builder(spec, job.get("params", {}))

        self._param_store[spec["id"]] = kept
        self._persist_param_store()
        self._exact_params_for = spec["id"]      # suppress dynamic defaults, once
        self._select_stage(spec)

        note = f"Loaded {job_id}'s parameters into the job builder ({len(kept)} values)."
        if dropped:
            note += (f"  Ignored {len(dropped)} parameter(s) this stage no longer has: "
                     f"{', '.join(dropped)}.")
        if missing:
            note += (f"  {len(missing)} newer parameter(s) fell back to defaults: "
                     f"{', '.join(missing)}.")
        self._log(note, "ok")

        # The builder rebuilds the command from the controls. If that does not
        # reproduce what the job actually ran, the difference is the interesting
        # part — surface it rather than letting a silently different command run.
        try:
            rebuilt = (self.current or {}).get("cmd")
            rebuilt = rebuilt.toPlainText().strip() if rebuilt is not None else ""
        except (AttributeError, RuntimeError):
            rebuilt = ""
        ran = (job.get("command", "") or "").strip()
        if ran and rebuilt and " ".join(ran.split()) != " ".join(rebuilt.split()):
            self._log(f"Note: the rebuilt command differs from what {job_id} ran. "
                      f"It ran:\n    {ran}", "info")

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
        # CARRY THE COMMAND ACROSS. A fork used to copy only the params, so the new
        # card had no command and every attempt to run or queue it reported "J## has
        # no command — delete it and re-queue". Prefer rebuilding from the params
        # (picks up the new job's own id/paths); fall back to the source's literal
        # command for trunk-style script stages that build_job_command can't model.
        cmd = ""
        spec = self._stage_by_id(src["stage_id"])
        if spec:
            try:
                store2 = load_jobs(self.project_root)
                cmd = build_job_command(spec, store2["jobs"][job["id"]], store2,
                                        self.warp_launch, self._group_inputs)
            except Exception:
                cmd = ""
        if not cmd:
            cmd = src.get("command", "") or ""
        if cmd:
            update_job(self.project_root, job["id"], command=cmd)
        self._log(f"Forked {job_id} → {job['id']}"
                  + ("" if cmd else "  (no command could be derived — open it in the "
                                    "job builder and press ↻ Rebuild from controls)")
                  + (". Edit params in the job builder, then right-click → Run."
                     if cmd else ""), "ok")
        self._refresh_canvas()
        if spec:
            self._select_stage(spec)

    def _delete_job(self, job_id, confirm=True):
        if confirm and QMessageBox.question(
                self, "Delete job?",
                f"Remove job {job_id} from the workflow?\n\n"
                f"Its output folder (jobs/{job_id}/) is left on disk — delete that "
                f"by hand if you want the space back.") != QMessageBox.Yes:
            return
        if delete_job(self.project_root, job_id):
            self._log(f"Deleted job {job_id}.", "info")
            self._refresh_canvas()

    def _queue_stage(self, stage_id):
        """Queue a stage straight from its card, without running it now.

        The ghost-card menu only offered "Build & run" and "Build (don't run)", so
        the only way to line work up was the job builder's + Queue variant button —
        there was no way to queue the NEXT step while something was already running,
        which is exactly when you want to. Builds the job, resolves its command now
        (so what you queued is what runs) and parks it as status='queued'."""
        spec = self._stage_by_id(stage_id)
        if not spec:
            return
        job = self._build_job(stage_id, run=False)
        if not job:
            return                      # cancelled at the overwrite prompt
        store = load_jobs(self.project_root)
        try:
            cmd = build_job_command(spec, store["jobs"][job["id"]], store,
                                    self.warp_launch, self._group_inputs)
        except Exception as e:
            self._log(f"Could not build a command for {job['id']}: {e}", "fail")
            delete_job(self.project_root, job["id"])
            return
        update_job(self.project_root, job["id"], status="queued", command=cmd)
        n = len(queued_jobs(load_jobs(self.project_root)))
        self._log(f"{job['id']} · {stage_title(stage_id, stage_id)} queued "
                  f"(position {n}). It starts when the queue reaches it — "
                  f"right-click ▸ Run now to jump ahead.", "ok")
        self._refresh_queue()
        self._refresh_canvas()

    def _requeue_job(self, job_id):
        """Put a finished/failed job back in the waiting list, unchanged. This is the
        'it died, run it again' path — the record (and its card) is reused rather than
        cloned, so the canvas doesn't accumulate a copy per attempt. Its resolved
        command is kept; edit it in the job builder first if you want a variant."""
        store = load_jobs(self.project_root)
        job = (store.get("jobs", {}) or {}).get(job_id)
        if not job:
            return
        if job.get("status") == "running":
            self._log(f"{job_id} is running — kill it first.", "fail")
            return
        update_job(self.project_root, job_id, status="queued", exit_code=None,
                   finished=None, started=None)
        self._log(f"{job_id} re-queued. It runs when the queue reaches it "
                  f"(right-click ▸ Run now to jump ahead).", "ok")
        self._refresh_queue()
        self._refresh_canvas()

    def _kill_job(self, job_id):
        """Stop the running job. The runner is shared, so this is the same terminate
        the TERMINATE button does — routed through the card for discoverability."""
        if getattr(self, "_active_job_id", None) != job_id or not self.runner.busy():
            self._log(f"{job_id} is not the running job.", "info")
            return
        if QMessageBox.question(
                self, "Kill job?",
                f"Stop {job_id}? Partial output stays on disk.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self._log(f"killing {job_id}…", "warning")
        self.runner.terminate()

    def _clear_job(self, job_id):
        """Delete a job's RESULTS but keep the card, so it can be reconfigured and
        re-run in place.

        Between "delete the job, keep the files" and "delete both" there was no way
        to say "this attempt was wrong, try again" — you deleted the card, lost the
        parameters and the wiring, and rebuilt it from scratch. Worse, re-running
        over a half-written output directory mixes two attempts' files, which is how
        a failed export leaves a star that looks complete.

        Uses the same resolver as the permanent delete, so it can never touch raw
        data or a shared directory.
        """
        store = load_jobs(self.project_root)
        job = (store.get("jobs", {}) or {}).get(job_id)
        if not job:
            return
        if job.get("status") == "running":
            QMessageBox.warning(self, "Job is running",
                                f"{job_id} is still running — kill it first.")
            return
        spec = self._stage_by_id(job.get("stage_id")) or {}
        targets, skipped = job_delete_targets(
            self.project_root, job_id, job.get("stage_id"),
            job.get("params", {}), spec.get("output_params"))

        detail = ("\n".join(f"    {r}/   ({self._dir_size_human(r)})" for r in targets)
                  if targets else "    (nothing on disk yet)")
        note = ("\n\nNOT touched (protected or shared):\n    "
                + "\n    ".join(skipped)) if skipped else ""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Clear this job's results?")
        box.setText(f"Clear {job_id} · {job.get('label', '')}?")
        box.setInformativeText(
            f"The CARD stays, with its parameters and wiring intact, and goes back to "
            f"'queued' so you can change it and run it again.\n\n"
            f"These results are REMOVED FROM DISK — this cannot be undone:\n\n"
            f"{detail}{note}")
        yes = box.addButton("Clear results", QMessageBox.DestructiveRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(box.buttons()[-1])
        box.exec()
        if box.clickedButton() is not yes:
            return

        removed = []
        for rel in targets:
            try:
                shutil.rmtree(Path(self.project_root) / rel)
                removed.append(rel)
            except OSError as e:
                self._log(f"could not clear {rel}: {e}", "fail")
        # Back to a fresh queued job. The command is dropped so it is rebuilt from
        # whatever the parameters say NEXT time — keeping a stale command is how an
        # edited card re-runs the old one.
        update_job(self.project_root, job_id, status="building", exit_code=None,
                   started=None, finished=None, summary={}, command="",
                   interrupted=False)
        self._log(f"Cleared {job_id}"
                  + (f" — removed {', '.join(removed)}" if removed
                     else " (nothing was on disk)")
                  + ". It is queued again; adjust its parameters and run it.",
                  "warning")
        self._invalidate_orphans()
        self._invalidate_status()
        self._refresh_queue()
        self._refresh_canvas()
        if spec:
            self._builder_job_id = job_id      # edit THIS card, don't clone it
            self._select_stage(spec)

    def _delete_job_permanent(self, job_id):
        """Delete the job record AND its output files. Irreversible.

        The plain delete keeps everything on disk, which is right most of the time —
        but it leaves failed attempts cluttering the project. This removes the files
        too. It is deliberately paranoid: it resolves the targets with
        job_delete_targets (which refuses raw-data and shared directories), SHOWS the
        exact list with sizes, defaults to No, and never touches anything it did not
        name."""
        store = load_jobs(self.project_root)
        job = (store.get("jobs", {}) or {}).get(job_id)
        if not job:
            return
        if job.get("status") == "running":
            QMessageBox.warning(self, "Job is running",
                                f"{job_id} is still running — kill it first.")
            return
        spec = self._stage_by_id(job.get("stage_id")) or {}
        targets, skipped = job_delete_targets(
            self.project_root, job_id, job.get("stage_id"),
            job.get("params", {}), spec.get("output_params"))

        if not targets:
            if QMessageBox.question(
                    self, "Nothing on disk",
                    f"{job_id} has no deletable files (its outputs are shared or "
                    f"protected).\n\nRemove the job from the workflow anyway?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
                self._delete_job(job_id, confirm=False)
            return

        lines = []
        for rel in targets:
            lines.append(f"    {rel}/   ({self._dir_size_human(rel)})")
        detail = "\n".join(lines)
        note = ("\n\nNOT touched (protected or shared):\n    "
                + "\n    ".join(skipped)) if skipped else ""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Delete job and its files?")
        box.setText(f"PERMANENTLY delete {job_id} · {job.get('label', '')}?")
        box.setInformativeText(
            f"These folders will be REMOVED FROM DISK — this cannot be undone:\n\n"
            f"{detail}{note}")
        yes = box.addButton("Delete permanently", QMessageBox.DestructiveRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(box.buttons()[-1])
        box.exec()
        if box.clickedButton() is not yes:
            return

        removed = []
        for rel in targets:
            try:
                shutil.rmtree(Path(self.project_root) / rel)
                removed.append(rel)
            except OSError as e:
                self._log(f"could not delete {rel}: {e}", "fail")
        delete_job(self.project_root, job_id)
        self._log(f"Permanently deleted {job_id} and {len(removed)} folder(s): "
                  f"{', '.join(removed)}", "warning")
        self._invalidate_orphans()
        self._invalidate_status()
        self._refresh_queue()
        self._refresh_canvas()

    def _dir_size_human(self, rel):
        """Rough size of a project-relative dir, capped so ceph is never walked deep."""
        total = n = 0
        try:
            for dp, _dn, fn in os.walk(Path(self.project_root) / rel):
                for f in fn:
                    try:
                        total += os.path.getsize(os.path.join(dp, f))
                    except OSError:
                        pass
                    n += 1
                    if n > 20000:
                        return f"{total / 1e9:.1f}+ GB"
        except OSError:
            return "?"
        for unit, div in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
            if total >= div:
                return f"{total / div:.1f} {unit}"
        return f"{total} B"

    def _hide_card(self, node):
        """Remove a card from the canvas without deleting anything on disk. The hidden
        set lives in the job store (per-project); View ▸ Show hidden cards restores
        them all. Handy for clearing failed/finished jobs and stale RELION orphans."""
        nid = node.get("id")
        if not nid:
            return
        try:
            store = load_jobs(self.project_root)
            hidden = set(store.get("hidden", []))
            hidden.add(nid)
            store["hidden"] = sorted(hidden)
            save_jobs(self.project_root, store)
        except Exception as e:
            self._log(f"Hide failed: {e}", "fail")
            return
        self._log(f"Hid '{node.get('title', nid)}'. View ▸ Show hidden cards to "
                  f"bring it back.", "info")
        self._refresh_canvas()

    def _show_hidden_cards(self):
        """View ▸ Show hidden cards: clear the hidden set so everything reappears."""
        try:
            store = load_jobs(self.project_root)
            n = len(store.get("hidden", []))
            if not n:
                self._log("No hidden cards.", "info")
                return
            store["hidden"] = []
            save_jobs(self.project_root, store)
        except Exception as e:
            self._log(f"Show hidden failed: {e}", "fail")
            return
        self._log(f"Restored {n} hidden card(s).", "ok")
        self._refresh_canvas()

    def _delete_orphan_dir(self, orph):
        """HARD-DELETE a found-on-disk RELION job folder (right-click ▸ Delete folder).
        Confirms first, naming the exact path — this recursively removes the directory
        and cannot be undone. Only ever targets the discovered dir under the project."""
        rel = orph.get("dir", "")
        if not rel:
            return
        d = Path(self.project_root) / rel
        if not d.is_dir():
            self._log(f"Delete: folder already gone: {rel}", "info")
            self._refresh_canvas()
            return
        if QMessageBox.question(
                self, "Delete folder from disk?",
                f"HARD-DELETE this folder and everything inside it?\n\n{d}\n\n"
                f"This removes the RELION job from disk and cannot be undone. "
                f"(To just clear it from the view instead, use Hide.)",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            shutil.rmtree(d)
        except OSError as e:
            self._log(f"Delete failed: {e}", "fail")
            QMessageBox.critical(self, "Delete failed", str(e))
            return
        self._log(f"Hard-deleted {rel}.", "info")
        self._invalidate_orphans()
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

    def _panel(self, object_name, title, content, on_close=None):
        """Wrap a panel's content in a bordered card with a header bar. `on_close` adds
        an ✕ button on the right of the header (for panels that are revealed on demand
        and would otherwise have no way to dismiss them)."""
        frame = QFrame()
        frame.setObjectName(object_name)
        frame.setProperty("card", "true")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(14, 12, 14, 14)
        lay.setSpacing(8)
        header = QLabel(title)
        header.setStyleSheet("font-size:13px;font-weight:700;color:#e6e6e6;"
                             "padding-bottom:6px;border-bottom:1px solid #3a3a3a;")
        if on_close is None:
            lay.addWidget(header)
        else:
            bar = QHBoxLayout()
            bar.setContentsMargins(0, 0, 0, 0)
            bar.setSpacing(6)
            bar.addWidget(header, 1)
            x = QPushButton("✕")
            x.setFixedSize(22, 22)
            x.setToolTip("Close this panel")
            x.setStyleSheet("border:none;color:#9a9a9a;font-size:13px;font-weight:700;")
            x.clicked.connect(on_close)
            bar.addWidget(x, 0, Qt.AlignTop)
            bw = QWidget()
            bw.setLayout(bar)
            lay.addWidget(bw)
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
        """Delete every waiting job. Running jobs are untouched (use TERMINATE)."""
        pend = queued_jobs(load_jobs(self.project_root))
        if not pend:
            return
        for j in pend:
            delete_job(self.project_root, j["id"])
        self._refresh_queue()
        self._refresh_canvas()
        self._log(f"Cleared {len(pend)} queued job(s).", "info")

    def _select_stage(self, spec):
        # Which queued job (if any) this form is editing. Consumed here so any other
        # route into the builder is plain stage-scoped editing, as before. Only a
        # QUEUED job binds: a finished one is re-run from its own card.
        bound_job = None
        want = getattr(self, "_builder_job_id", None)
        self._builder_job_id = None
        if want:
            try:
                j = (load_jobs(self.project_root).get("jobs") or {}).get(want)
                if (j and j.get("status") in ("building", "queued")
                        and j.get("stage_id") == spec["id"]):
                    bound_job = want
            except Exception:
                bound_job = None

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

        # "Load this run's parameters" must reproduce a past job EXACTLY. Every
        # override above is a convenience for a NEW run (next AreTomo version,
        # newest alignments folder, detected gain) and would silently replace the
        # value the job actually used — the one thing you are looking at it for.
        # One-shot, cleared here so the next visit behaves normally.
        if self._exact_params_for == spec["id"]:
            overrides = {}
        self._exact_params_for = None

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

        btns = FlowLayout(spacing=6)   # wraps to extra rows when the panel is narrow
        run = QPushButton(f"▶ Run {bound_job}" if bound_job else "▶ Run")
        build_job = QPushButton(f"▶ Save & run {bound_job}" if bound_job
                                else "▶ Build & run as job")
        build_job.setToolTip(
            f"Save these parameters into {bound_job} — the card already on the "
            f"canvas — and run it. (Without this, pressing run here would create a "
            f"SECOND job for the same step.)" if bound_job else
            "Card view: create a job instance on the canvas from these parameters "
            "(input auto-wired to the newest upstream job) and run it. Fork a "
            "finished job to try variants.")
        rebuild = QPushButton("↻ Rebuild from controls")
        enqueue = QPushButton(f"+ Save {bound_job} (stays queued)" if bound_job
                              else "+ Queue variant")
        reset = QPushButton("⟲ Reset defaults")
        reset.setToolTip("Discard your saved edits for THIS step and restore the "
                         "template defaults (and current dynamic defaults).")
        if bound_job:
            enqueue.setToolTip(
                f"Save these parameters into {bound_job} and leave it queued. It runs "
                f"automatically when the job before it finishes — queued jobs run in "
                f"card order. This does NOT create another card.")
            run.setToolTip(f"Save these parameters into {bound_job} and run it NOW, "
                           f"ahead of the queue.")
        buttons = [run, build_job, rebuild, enqueue, reset]
        if spec.get("sync_helper"):
            fill = QPushButton("Fill: deselect all unaligned")
            fill.clicked.connect(self._fill_sync_command)
            buttons.append(fill)
        for b in buttons:
            b.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            btns.addWidget(b)
        bw = QWidget()
        bw.setLayout(btns)
        self.form_box.addWidget(bw)

        self.current = {"spec": spec, "controls": controls, "cmd": cmd,
                        "warn": warn, "manual": False, "guard": False,
                        "job_id": bound_job}

        cmd.textChanged.connect(self._on_cmd_edited)
        run.clicked.connect(
            (lambda: self._save_and_run_job(bound_job)) if bound_job
            else self._run_current)
        build_job.clicked.connect(
            (lambda: self._save_and_run_job(bound_job)) if bound_job
            else (lambda: self._build_job(spec["id"], run=True)))
        rebuild.clicked.connect(lambda: self._set_manual(False))
        enqueue.clicked.connect(
            (lambda: self._save_queued_job(bound_job)) if bound_job
            else self._enqueue_current)
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
        orph = node.get("orphan")
        if orph:
            # found-on-disk set: point at its (possibly .bak) dir.
            apx = fmt_angpix(orph.get("angpix", "12.56"))
            mdir = orph.get("dir", "warp_tiltseries/matching")
            star_pat = f"*Apx{orph.get('suffix', '')}.star"
            corr_suffix = ""                 # template unknown from a star name alone
        else:
            job = None
            if not node.get("is_ghost") and node.get("id"):
                job = load_jobs(self.project_root).get("jobs", {}).get(node["id"])
            if job:
                params = job.get("params", {})
                # a job's picks live in ITS OWN dir, not the (often archived) trunk
                mdir = f"{job.get('output_dir', 'warp_tiltseries')}/matching"
            elif self.current and self.current.get("spec", {}).get("id") == stage_id:
                params = self._values()
                mdir = "warp_tiltseries/matching"
            else:
                spec = self._stage_by_id(stage_id) or {}
                params = self._effective_params(spec) if spec else {}
                mdir = "warp_tiltseries/matching"
            apx = fmt_angpix(params.get("tomo_angpix", "12.56"))
            # STAR uses the run's suffix (override or template); the CORR volume
            # always uses the template suffix (--override_suffix doesn't rename it).
            star_pat = f"*{apx}Apx{template_match_suffix(params)}.star"
            corr_suffix = template_corr_suffix(params)
        # If the template suffix is known, target it exactly; otherwise a wildcard
        # middle catches the template name (e.g. *12.56Apx*_corr.mrc matches
        # *_emd_70905_corr.mrc) — one corr per tomogram when there's one template.
        corr_pat = (f"*{apx}Apx{corr_suffix}_corr.mrc" if corr_suffix
                    else f"*{apx}Apx*_corr.mrc")
        # If this dir has no corr volumes (e.g. a whitened run, or an adopted set),
        # view picks-only — warp-tm-vis errors otherwise. Bounded: one dir, first hit.
        no_vol = ""
        try:
            mp = Path(self.project_root) / mdir
            if mp.is_dir() and next(mp.glob("*_corr.mrc"), None) is None:
                no_vol = " --no-load-volumes"
        except OSError:
            pass
        cmd = (f'{self.tm_vis_launch} '
               f'-rdir warp_tiltseries/reconstruction '
               f'-mdir {mdir} '
               f'-mp "{star_pat}" -cvp "{corr_pat}"{no_vol}')
        cmd, ok = QInputDialog.getText(
            self, "Launch warp-tm-vis",
            "Command (edit the suffix / paths if needed; add --no-load-volumes if "
            "this run didn't save _corr.mrc volumes):", text=cmd)
        if not ok or not cmd.strip():
            return
        # Detached (it's a blocking GUI) but tee stdout+stderr to a log so a silent
        # failure (e.g. a pattern that matched nothing) is inspectable.
        logf = Path(self.project_root) / ".tomogration_tmvis.log"
        try:
            with open(logf, "wb") as fh:
                subprocess.Popen(["bash", "-lc", cmd.strip()],
                                 cwd=str(self.project_root), stdout=fh, stderr=fh)
            self._log(f"warp-tm-vis launching (GUI opens in its own window). "
                      f"If nothing appears, check {logf} or run the command in a "
                      f"terminal:", "info")
            self._log(cmd.strip(), "info")
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
        # An unresolved {placeholder} means a field points at another field that is
        # still empty. If it reaches the tool it is used LITERALLY — that is how
        # MTools ended up creating a population file named "{name}.population".
        # Checked for EVERY stage, so no future default can reintroduce this.
        try:
            cmd = self.current["cmd"].toPlainText()
        except Exception:
            cmd = ""
        left = sorted(set(re.findall(r"\{(\w+)\}", cmd)) - {"jobid"})
        if left:
            names = ", ".join("{" + k + "}" for k in left)
            warn = (f"⚠ Unresolved placeholder{'s' if len(left) > 1 else ''} {names} "
                    f"— fill in the field(s) they refer to. Run as-is and the tool "
                    f"will treat it as a literal name and create the wrong file.")
            msg = warn + ("\n" + msg if msg else "")
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
        # Same gate as _build_job: a live ⚠ must be acknowledged, not decorative.
        if not self._confirm_validator(spec, self._values()):
            return
        # Per-job nudge: for back-half stages, prefer a job (own dir, forkable) over
        # overwriting the shared trunk. Manual command edits can't carry into a job
        # (it rebuilds from params + wiring), so only offer this on an unedited cmd.
        if spec["id"] in JOB_STAGES and not self.current.get("manual"):
            box = QMessageBox(self)
            box.setWindowTitle("Run as a job?")
            box.setText(f"Run '{stage_title(spec['id'], spec['id'])}' as a job?")
            box.setInformativeText(
                "A job writes to its own jobs/J### folder — it never overwrites other "
                "runs and can be forked and wired into downstream jobs. 'Overwrite "
                "shared' runs the old way, replacing warp_tiltseries/… in place.")
            as_job = box.addButton("Run as job", QMessageBox.AcceptRole)
            overwrite = box.addButton("Overwrite shared", QMessageBox.DestructiveRole)
            box.addButton("Cancel", QMessageBox.RejectRole)
            box.setDefaultButton(as_job)
            box.exec()
            clicked = box.clickedButton()
            if clicked is as_job:
                self._build_job(spec["id"], run=True)
                return
            if clicked is not overwrite:
                return
            # else: fall through to the classic overwrite-in-place path
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
        # A trunk (▶ Run) re-extract has no job record — stash its values so
        # _on_finished can register the output pick-set card on success.
        self._pending_reextract = (
            dict(self._values())
            if spec["id"] in ("relion4_to_warp", "relion4_select_picks") else None)
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
        # Queueing is the easiest way to ignore a warning: you set it up, walk away,
        # and it runs overnight. Gate it too.
        if not self._confirm_validator(spec, self._values()):
            return
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
        # Queueing creates a REAL job (status='queued') carrying its resolved command,
        # so it shows on the canvas as a card and survives a restart.
        n = len(queued_jobs(load_jobs(self.project_root))) + 1
        job = new_job(self.project_root, spec["id"],
                      f'{stage_title(spec["id"], spec["id"])} (queued {n})',
                      self._values(), inputs={})
        update_job(self.project_root, job["id"], status="queued", command=cmd,
                   trunk=True)
        self._log(f"queued: {job['id']} · {spec['label']}", "info")
        self._refresh_queue()
        self._refresh_canvas()

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
        self._set_status(f"\u25b6 {stage_title(stage_id, stage_id)} \u2014 starting\u2026")
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
    @staticmethod
    def _link_into(src_dir, dst_dir, keep):
        """Symlink files in src_dir accepted by keep(name) into dst_dir, pointing at
        the REAL files (no symlink chains). Non-destructive: never clobbers. -> count."""
        n = 0
        for f in itertools.islice(sorted(src_dir.iterdir()), 0, 40000):
            if not keep(f.name):
                continue
            link = dst_dir / f.name
            if link.exists() or link.is_symlink():
                continue
            try:
                link.symlink_to(os.path.realpath(f))
                n += 1
            except OSError:
                pass
        return n

    def _prepare_job_inputs(self, job, store):
        """threshold_picks reads AND writes its match stars in <output_processing>/
        matching/ IN PLACE (and reads the per-series .xml for selection/metadata),
        so a fresh jobs/J### has nothing to read. Stage its inputs into its own dir:
        the PARENT's matching stars (+corr) into jobs/J###/matching, and the trunk
        warp_tiltseries/*.xml (selection state) into jobs/J###/. All symlinks to the
        real files — non-destructive."""
        if job.get("stage_id") != "threshold_picks":
            return
        parent = store.get("jobs", {}).get(parent_job_id(job) or "")
        root = Path(self.project_root)
        jobdir = root / job["output_dir"]
        if not parent:
            return
        msrc = root / parent["output_dir"] / "matching"
        if not msrc.is_dir():
            self._log(f"{job['id']}: parent {parent['id']} has no matching/ to read.", "fail")
            return
        mdst = jobdir / "matching"
        mdst.mkdir(parents=True, exist_ok=True)
        nm = self._link_into(msrc, mdst,
                             lambda nm: nm.endswith(".star") or nm.endswith("_corr.mrc"))
        # per-series metadata (.xml) — carries the selection; take it from the trunk
        xsrc = root / "warp_tiltseries"
        nx = self._link_into(xsrc, jobdir, lambda nm: nm.endswith(".xml")) \
            if xsrc.is_dir() else 0
        self._log(f"{job['id']}: staged {nm} match file(s) from {parent['id']} + {nx} "
                  f".xml into {job['output_dir']} for thresholding.", "info")

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
        self._prepare_job_inputs(job, store)
        cmd = build_job_command(spec, job, store, self.warp_launch, self._group_inputs)
        update_job(self.project_root, job_id, command=cmd, status="running",
                   exit_code=None, finished=None,
                   started=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self._active_job_id = job_id
        self._m_resolution = None
        self._active_stage = None
        self._active_cmd = cmd
        self._attempt = 1
        self._failed_file = None
        self._log(f"--- running {job_id} · {spec['label']} ---", "info")
        self._set_status(f"▶ {job_id} · {spec['label']} — starting…")
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
        # M writes no summary file, so its result only exists as the line _log
        # caught. Consume it here (and clear it, so the next job cannot inherit
        # the previous run's number).
        res = getattr(self, "_m_resolution", None)
        self._m_resolution = None
        if res is not None and code == 0:
            summary["resolution_A"] = f"{res:g}"
        update_job(self.project_root, job_id, status=status, exit_code=code,
                   finished=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   summary=summary)
        self._log(f"--- {job_id} {status} (exit {code}) ---",
                  "ok" if code == 0 else "fail")
        self._set_status(
            f"{'\u2713' if code == 0 else '\u2717'} {job_id} {status} (exit {code})")
        if code == 0:
            self._bridge_job_outputs(job_id, job.get("stage_id"))
        # A re-extract run as a job: register its output pick-set card too.
        if job.get("stage_id") in ("relion4_to_warp", "relion4_select_picks"):
            self._maybe_register_reextract(job.get("params", {}), code)
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
            self._finalize_job(code)          # records status + repaints the canvas
            # _run_queued_job sets BOTH markers, so clear both or the stage's ghost
            # card stays lit after the job finishes.
            self._active_stage = None
            self._active_cmd = ""
            self._run_progress = ""
            self._refresh_status_dots()
            self._run_queue()          # chain the next queued job, if any
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

        self._set_status(
            f"{'\u2713' if code == 0 else '\u2717'} "
            f"{stage_title(stage_id, stage_id or 'run')} finished (exit {code})")
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

        # Trunk re-extract just finished — register its output pick-set card (crYOLO-
        # style) so it wires into ts_export_particles. Cleared regardless.
        pr = getattr(self, "_pending_reextract", None)
        self._pending_reextract = None
        if pr is not None:
            self._maybe_register_reextract(pr, code)

        # A trunk run has no job record, so its "running" look lives entirely in
        # self._active_stage. Clear it and REPAINT — otherwise the stage's ghost card
        # keeps the amber it had at the last repaint and looks stuck mid-run forever
        # (the job path repaints via _finalize_job; this path never did).
        self._active_stage = None
        self._active_cmd = ""
        self._run_progress = ""
        if hasattr(self, "_invalidate_status"):
            self._invalidate_status()   # the run just changed what's on disk
        self._refresh_canvas()

        self._run_queue()              # chain the next queued job, if any

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
        v.setSpacing(4)

        # ---- STATUS STRIP: progress lives HERE, not in the log ----------------
        # WarpTools emits "N/M …" continuously. Collapsing those in the log stream
        # only worked while nothing else printed: any interleaved line (a queue
        # message) reset the collapse and left a trail of 1/52, 2/52, 3/52 through
        # the scrollback. Progress is STATE, not history — so it gets its own strip
        # that overwrites in place, and the log keeps only real output (which makes
        # it worth selecting and copying from).
        self.status_strip = QLabel("idle")
        self.status_strip.setStyleSheet(
            "background:#181818;border:1px solid #333;border-radius:4px;"
            "padding:5px 8px;color:#9a9a9a;font-size:11px;")
        self.status_strip.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self.status_strip)
        self.status_bar_w = QProgressBar()
        self.status_bar_w.setTextVisible(False)
        self.status_bar_w.setFixedHeight(4)
        self.status_bar_w.setRange(0, 100)
        self.status_bar_w.setValue(0)
        self.status_bar_w.setStyleSheet(
            "QProgressBar{background:#181818;border:none;border-radius:2px;}"
            "QProgressBar::chunk{background:#f0a92a;border-radius:2px;}")
        self.status_bar_w.setVisible(False)
        v.addWidget(self.status_bar_w)

        bar = QHBoxLayout()
        bar.setSpacing(6)
        bar.addWidget(QLabel("Log"))
        bar.addStretch(1)
        find = QLineEdit()
        find.setPlaceholderText("find…")
        find.setFixedWidth(120)
        find.returnPressed.connect(lambda: self._term_find(find.text()))
        bar.addWidget(find)
        copy_btn = QPushButton("Copy")
        copy_btn.setToolTip("Copy the selection, or the whole log if nothing is selected")
        copy_btn.clicked.connect(self._term_copy)
        bar.addWidget(copy_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(lambda: self.term.clear())
        bar.addWidget(clear_btn)
        term_btn = QPushButton("TERMINATE")
        term_btn.setStyleSheet("color:#c0392b;")
        term_btn.clicked.connect(self.runner.terminate)
        bar.addWidget(term_btn)
        barw = QWidget()
        barw.setLayout(bar)
        v.addWidget(barw)

        self.term = QPlainTextEdit()
        self.term.setReadOnly(True)
        self.term.setMaximumBlockCount(20000)
        self.term.setTextInteractionFlags(Qt.TextSelectableByMouse
                                          | Qt.TextSelectableByKeyboard)
        self.term.setStyleSheet(
            f"background:#111;color:#ddd;font-family:{MONO};font-size:12px;"
            f"selection-background-color:#2d4a6b;")
        v.addWidget(self.term, 1)

        # ---- COMMAND INPUT: quick checks without leaving the app --------------
        # Runs in the project root on its OWN process, so it never collides with a
        # running job. Meant for `<tool> --help`, ls, head, the check scripts.
        row = QHBoxLayout()
        row.setSpacing(6)
        prompt = QLabel("❯")
        prompt.setStyleSheet("color:#7fb4ff;font-weight:700;")
        row.addWidget(prompt)
        self.cmd_input = QLineEdit()
        self.cmd_input.setPlaceholderText(
            "run a command in the project root (↑/↓ history) — e.g. nvidia-smi, "
            "WarpTools ts_ctf --help")
        self.cmd_input.setStyleSheet(f"font-family:{MONO};font-size:12px;")
        self.cmd_input.returnPressed.connect(self._run_console_cmd)
        self.cmd_input.installEventFilter(self)          # ↑/↓ history
        row.addWidget(self.cmd_input, 1)
        roww = QWidget()
        roww.setLayout(row)
        v.addWidget(roww)
        self._console_hist = []
        self._console_pos = 0

        # ---- QUEUE STRIP: what runs next, where you are already looking --------
        # The queue was only legible by reading the canvas for blue cards, which is
        # a poor way to answer "what happens when this finishes?". One right-aligned
        # row of ids, in run order.
        qrow = QHBoxLayout()
        qrow.setContentsMargins(0, 0, 0, 0)
        qrow.setSpacing(6)
        self.queue_label = QLabel("queue")
        self.queue_label.setStyleSheet("color:#5c6b7a;font-size:10px;")
        qrow.addWidget(self.queue_label)
        qrow.addStretch(1)
        self.queue_chips = FlowLayout(spacing=4)     # wraps when the panel is narrow
        chipw = QWidget()
        chipw.setLayout(self.queue_chips)
        qrow.addWidget(chipw)
        self.queue_strip = QWidget()
        self.queue_strip.setLayout(qrow)
        v.addWidget(self.queue_strip)

        w = QWidget()
        w.setLayout(v)
        return w

    def _refresh_queue_chips(self):
        """The queue as a row of ids under the terminal, in run order."""
        lay = getattr(self, "queue_chips", None)
        if lay is None:
            return
        while lay.count():                       # FlowLayout owns the old chips
            it = lay.takeAt(0)
            wdg = it.widget() if it is not None else None
            if wdg is not None:
                wdg.setParent(None)
        root = getattr(self, "project_root", None)
        pend = []
        if isinstance(root, (str, os.PathLike)):
            try:
                pend = queued_jobs(load_jobs(root))
            except Exception:
                pend = []
        self.queue_label.setText("queue" if pend else "queue empty")
        for job in pend:
            lay.addWidget(_QueueChip(job["id"], job.get("label", ""),
                                     self._unqueue_job))

    def _unqueue_job(self, job_id):
        """Take a job out of the queue and hand it back for editing.

        Deliberately BUILDING, not deleted: cancelling a queued job means "not this
        one, not yet", and throwing away its parameters and wiring to express that
        would be absurd. Re-queue it from the builder when it is right.
        """
        store = load_jobs(self.project_root)
        job = (store.get("jobs") or {}).get(job_id)
        if not job or job.get("status") != "queued":
            return
        update_job(self.project_root, job_id, status="building")
        self._log(f"{job_id} taken out of the queue — it is yours to edit again.",
                  "info")
        self._refresh_queue()
        self._refresh_canvas()

    # ---- status strip -------------------------------------------------------
    def _set_status(self, text, progress=None):
        """Update the strip above the log. `progress` = (done, total) or None."""
        if not getattr(self, "status_strip", None):
            return
        self.status_strip.setText(text)
        bar = getattr(self, "status_bar_w", None)
        if bar is None:
            return
        if progress and progress[1]:
            bar.setValue(max(0, min(100, int(100 * progress[0] / progress[1]))))
            bar.setVisible(True)
        else:
            bar.setVisible(False)

    def _term_copy(self):
        cur = self.term.textCursor()
        QApplication.clipboard().setText(
            cur.selectedText().replace(" ", "\n") if cur.hasSelection()
            else self.term.toPlainText())
        self._log("Copied log to clipboard.", "info")

    def _term_find(self, text):
        if not text:
            return
        if not self.term.find(text):                 # wrap to the top and retry
            cur = self.term.textCursor()
            cur.movePosition(QTextCursor.Start)
            self.term.setTextCursor(cur)
            if not self.term.find(text):
                self._log(f"'{text}' not found in the log.", "info")

    # ---- console command input ---------------------------------------------
    def _run_console_cmd(self):
        cmd = self.cmd_input.text().strip()
        if not cmd:
            return
        root = getattr(self, "project_root", None)
        if not isinstance(root, (str, os.PathLike)):
            self._log("Set a project root first.", "fail")
            return
        self._console_hist.append(cmd)
        self._console_pos = len(self._console_hist)
        self.cmd_input.clear()
        self._log(f"❯ {cmd}", "info")
        proc = QProcess(self)
        proc.setWorkingDirectory(str(root))
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(
            lambda p=proc: self._console_out(p))
        proc.finished.connect(lambda code, _s, p=proc: self._console_done(p, code))
        self._console_procs = getattr(self, "_console_procs", [])
        self._console_procs.append(proc)             # keep a ref (GC would kill it)
        proc.start("bash", ["-lc", cmd])

    def _console_out(self, proc):
        data = bytes(proc.readAllStandardOutput()).decode("utf-8", "replace")
        for line in data.splitlines():
            self._log(line, "out")

    def _console_done(self, proc, code):
        self._log(f"❯ exit {code}", "ok" if code == 0 else "fail")
        try:
            self._console_procs.remove(proc)
        except (ValueError, AttributeError):
            pass

    def eventFilter(self, obj, ev):
        """↑/↓ recall previous console commands."""
        if (obj is getattr(self, "cmd_input", None)
                and ev.type() == QEvent.KeyPress and self._console_hist):
            key = ev.key()
            if key in (Qt.Key_Up, Qt.Key_Down):
                self._console_pos += -1 if key == Qt.Key_Up else 1
                self._console_pos = max(0, min(len(self._console_hist),
                                               self._console_pos))
                self.cmd_input.setText(
                    self._console_hist[self._console_pos]
                    if self._console_pos < len(self._console_hist) else "")
                return True
        return super().eventFilter(obj, ev)

    def _run_queue(self):
        """Start the next queued job (if nothing is running)."""
        if self.runner.busy():
            return
        nxt = queued_jobs(load_jobs(self.project_root))
        if nxt:
            self._run_queued_job(nxt[0]["id"])

    def _run_queued_job(self, job_id):
        """Run a QUEUED job: its command was resolved when it was queued, so it runs
        verbatim (no rebuild/re-wiring — what you queued is what runs). Tracked as a
        job so _finalize_job records the outcome and the card turns green/red."""
        store = load_jobs(self.project_root)
        job = (store.get("jobs", {}) or {}).get(job_id)
        if not job:
            self._log(f"queued job {job_id} not found.", "fail")
            return
        cmd = job.get("command") or ""
        if not cmd:
            # A card placed from the palette (or built with run=False) has params but
            # no resolved command yet — it was never queued through the builder. That
            # is a normal state, not a broken job: build the command from its current
            # params now rather than telling the user to delete and start over.
            spec = self._stage_by_id(job.get("stage_id", ""))
            if spec is not None:
                try:
                    cmd = build_job_command(spec, job, store, self.warp_launch,
                                            self._group_inputs)
                except Exception:
                    cmd = ""
            if cmd:
                update_job(self.project_root, job_id, command=cmd)
                self._log(f"{job_id} had no stored command — built one from its "
                          f"parameters.", "info")
        if not cmd:
            self._log(f"{job_id} has no command, and one could not be built from its "
                      f"parameters. Open it in the job builder, set them, then "
                      f"▶ Build & run as job.", "fail")
            update_job(self.project_root, job_id, status="failed", exit_code=-1)
            self._refresh_canvas()
            return
        update_job(self.project_root, job_id, status="running", exit_code=None,
                   finished=None,
                   started=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self._active_job_id = job_id
        self._m_resolution = None
        self._active_stage = job.get("stage_id")   # keep the stage dot live too
        self._active_cmd = cmd
        self._attempt = 1
        self._failed_file = None
        self._log(f"--- running {job_id} · "
                  f"{stage_title(job.get('stage_id', ''), job.get('stage_id', ''))} ---",
                  "info")
        self._refresh_queue()
        self._refresh_canvas()
        self.runner.run(cmd, self.project_root)

    def _refresh_queue(self):
        """Render the waiting list straight from the job store (the single source of
        truth). Safe before a project root exists."""
        self._refresh_queue_chips()
        if not getattr(self, "queue_view", None):
            return
        root = getattr(self, "project_root", None)
        if not isinstance(root, (str, os.PathLike)):
            self.queue_view.setPlainText("(empty)")
            return
        try:
            pend = queued_jobs(load_jobs(root))
        except Exception:
            pend = []
        self.queue_view.setPlainText(
            "\n".join(f"{i + 1}. {j['id']}  {stage_title(j.get('stage_id', ''), '')}"
                      for i, j in enumerate(pend)) or "(empty)")

    def _log(self, text, level):
        # Tap the stream for auto-recovery's failed-file detection.
        m = _FAILED_FILE_RE.search(text)
        if m:
            self._failed_file = m.group(1)
        # MCore reports what a refinement achieved on ONE stdout line at the end and
        # writes it nowhere. Catch it here so _finalize_job can put it on the card —
        # otherwise a column of M jobs is unreadable and the only way to compare
        # rounds is to scroll back through the log.
        if level == "out" and getattr(self, "_active_job_id", None):
            res = m_resolution(text)
            if res is not None:
                self._m_resolution = res
        colours = {"out": "#dddddd", "err": "#e0a850", "info": "#7fb4ff",
                   "ok": "#27ae60", "success": "#27ae60", "warning": "#e0a850",
                   "error": "#e24b4a", "fail": "#e24b4a"}
        # Detect a self-updating progress line. The counter is NOT always first:
        # MTools prints "Calculating data hashes... 285/290", which a start-
        # anchored pattern misses — so all 290 ticks landed in the log as
        # separate lines instead of the status strip.
        is_progress = level in ("out", "err") and progress_key(text) is not None
        if is_progress:
            # Progress is STATE, not history: it goes to the status strip and NEVER
            # into the log. That kills the 1/52 · 2/52 · 3/52 trail that appeared
            # whenever another line interleaved, and keeps the log copy-worthy.
            prog = text.strip()
            self._run_progress = prog[:40]
            m = _PROGRESS_RE.match(text)
            done = total = None
            nums = re.search(r"(\d+)\s*/\s*(\d+)", text)
            if nums:
                done, total = int(nums.group(1)), int(nums.group(2))
            what = getattr(self, "_active_stage", None) or ""
            label = stage_title(what, what) if what else "running"
            self._set_status(f"▶ {label} — {prog}"[:160],
                             (done, total) if total else None)
            self._last_was_progress = True
            return
        # WarpTools prints a blank spacer line between progress updates — swallow
        # it so the log doesn't collect blank gaps where progress used to be.
        if not text.strip() and self._last_was_progress:
            return
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html = f'<span style="color:{colours.get(level, "#ddd")}">{safe}</span>'
        # Follow the tail only if already at the bottom; if the user scrolled up to
        # read, leave their position alone (don't yank the view around).
        sb = self.term.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 4
        prev = sb.value()
        self.term.appendHtml(html)
        self._last_was_progress = False
        # Follow the tail only if already at the bottom; if the user scrolled up to
        # read or select, leave their position alone.
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


def apply_dark_theme(app):
    """Pin the look to Fusion + an explicit dark palette.

    Every colour this app hardcodes in a stylesheet (#888 labels, #ececec
    titles, #0e0e0e cards) assumes a dark background. Widgets we do NOT style —
    buttons, line edits, combos, scrollbars, menus, dialogs — otherwise take
    their colours from the desktop theme, so on a machine whose theme resolves
    light (or fails to resolve at all, e.g. no xdg-desktop-portal on a VM) you
    get pale grey text on white. Setting both here makes the app look the same
    everywhere and removes the dependency on the host theme entirely.
    """
    app.setStyle("Fusion")
    bg, base, text = QColor("#232323"), QColor("#191919"), QColor("#e6e6e6")
    accent, disabled = QColor("#3d6fa5"), QColor("#6f6f6f")
    p = QPalette()
    for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive,
                  QPalette.ColorGroup.Disabled):
        dim = group == QPalette.ColorGroup.Disabled
        role = QPalette.ColorRole
        p.setColor(group, role.Window, bg)
        p.setColor(group, role.Base, base)
        p.setColor(group, role.AlternateBase, bg)
        p.setColor(group, role.Button, bg)
        p.setColor(group, role.ToolTipBase, base)
        p.setColor(group, role.Highlight, disabled if dim else accent)
        p.setColor(group, role.Link, QColor("#9bc0ff"))
        for r in (role.WindowText, role.Text, role.ButtonText,
                  role.ToolTipText, role.HighlightedText):
            p.setColor(group, r, disabled if dim else text)
        p.setColor(group, role.PlaceholderText, disabled)
    app.setPalette(p)



def install_exception_surface(window):
    """Route uncaught exceptions into the app's own log (and a dialog) instead of
    stderr. Qt calls slots from C++, so a Python exception inside one is printed to
    the terminal the app was launched from and then SWALLOWED — the UI just stops
    responding to that button with no visible clue. Every 'I click Run and nothing
    happens' bug in this app has had that shape."""
    import traceback

    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        try:
            window._log("UNCAUGHT ERROR — please report this traceback:", "fail")
            for line in text.rstrip().splitlines():
                window._log("  " + line, "fail")
            window._set_status(f"\u2717 error: {exc_type.__name__}: {exc}"[:160])
        except Exception:
            pass
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = hook


def main():
    app = QApplication(sys.argv)
    apply_dark_theme(app)
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
    install_exception_surface(win)   # errors land in the log, not stderr
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
