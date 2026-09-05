#!/usr/bin/env python3
"""
tomogration — three-panel cryo-ET pipeline controller (PySide6).

Scope: raw data prep -> particle export -> a RELION 4 Class3D handoff. High-
resolution refinement / RELION 5 --tomo / M live in those tools' own pipeliners.

Layout (card view, default):
    [ pinned pipeline outline | workflow canvas | full-height tabbed terminal ]
Each card pops out its own floating panel (Builder · Details · Outputs); the
Outputs rows drag into any builder's input fields. The classic list view keeps
[ LEFT docs/theory ] [ CENTER pipeline lists + param form ] [ RIGHT terminal ].

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
import html as _html
import json
import shlex
import signal
import shutil
import datetime
import itertools
import subprocess
from pathlib import Path

import time

from PySide6.QtCore import (
    Qt, QObject, QEvent, Signal, QProcess, QTimer, QSize, QRect, QRectF,
    QPoint, QMimeData,
)
from PySide6.QtGui import (
    QBrush, QColor, QTextCursor, QFont, QPen, QPainter,
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
    QGraphicsItem, QGraphicsTextItem, QMenu, QLayout, QSizePolicy, QProgressBar, QToolButton,
    QAbstractItemView, QTabWidget,
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
    from tomogration_core import _PKG_DIR, _pkg_script, progress_key
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
from tomogration_inventory import inventory_sources, list_dir
from tomogration_stages import *            # noqa: F401,F403
from tomogration_stages import (            # explicit: the names used below
    # underscore names are NOT re-exported by `import *` — list them
    _h, _norm_gpu, _validate_export,
    STAGES, STAGE_OUTPUTS, STAGE_IO, DIR_FILE_HINTS, COLUMN_OF_GROUP,
    COLUMN_TITLES, KEY_DIRS, JOB_STAGES, TRUNK_STAGES, ARCHIVE_ON_RERUN,
    THREEDMOD_EXTS, TEXT_EXTS, build_command, stage_defaults, _norm_gpu,
    render_docs_html, render_inline_docs_html, _validate_export,
    param_title, param_wire_name, param_is_pathish, stage_tool_line,
)
HERE = Path(__file__).resolve().parent

# The sweep's analysis maths lives with the sweep, not in the GUI: imported by
# path because ml_*.py are scripts, not an installed package.
import importlib.util as _ilu
_EX_SPEC = _ilu.spec_from_file_location(
    "ml_explore_membrane", str(Path(__file__).resolve().parent / "ml_explore_membrane.py"))
EX = _ilu.module_from_spec(_EX_SPEC)
try:
    _EX_SPEC.loader.exec_module(EX)
except Exception:                      # the analysis window degrades, app runs
    EX = None

from tomogration_project import (ProjectState, EXPECTED_DIRS, COARSE_DIR,
                                 PROJECT_MARKERS, project_marker, find_projects,
                                 _FAILED_FILE_RE,
                                 _atomic_write_text, series_stem, tomogram_stems)
def _now():
    """Timestamp in the store's format (created/started/finished/queued_at)."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


from tomogration_jobs import *              # noqa: F401,F403
from tomogration_jobs import (
    # underscore names are NOT re-exported by `import *` — list them
    _APX_RE, _CTF_DEFOCUS_RE, _PICK_STAR_RE, _count_glob, _job_seq, _jobnum, _mean_std, _picktag,
    JOBS_FILE, jobs_path, load_jobs, save_jobs, queued_jobs, job_output_dir,
    reconcile_running, job_delete_targets, PROTECTED_DIRS,
    new_job, update_job, delete_job, is_warp_stage, parent_job_id,
    io_flags_for_job, build_job_command, default_parent_for, fmt_angpix,
    actual_job_kind, carried_note, CARRIED_PARAMS,
    VIEWER_PLANS, viewer_plan, viewer_series, viewer_inventory,
    greyscale_source,
    resolve_viewer_tool,
    template_corr_suffix, template_match_suffix, match_star_infix, DOWNSTREAM,
    derive_child_params, summarize_job, summary_text, FRIENDLY_TITLES,
    stage_title, _job_seq, is_cryolo_job, canvas_layout, card_is_running,
    discover_picksets, m_resolution, params_for_builder, job_real_outputs,
    star_particle_count, relion_card_text, set_card_position, clear_card_positions,
    star_header_info, direct_export_params,
    job_real_dir,
    set_job_parent, settings_processing_dir,
    discover_relion_jobs, _PICK_STAR_RE,
    set_manual_status, clear_manual_status, manual_status_note,
    RERUNNABLE,
    load_notes, add_note, update_note, delete_note, notes_for_canvas,
    wrap_lines,
    newly_added, run_failure_hint, run_warning_key, run_warning_report,
    false_success_reason, latest_stage_output_dir,
    batch_tally_line, partial_batch_result, alignment_pixel_size,
    refined_since_last_import,
    cards_inside, NOTE_COLOURS, would_cycle,
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

# In-place redraw control codes. A tool that animates a progress bar rewinds the
# cursor first: Keras' Progbar (IsoNet's refine) writes a run of \x08 backspaces
# and then \r before each update. splitlines() breaks at the \r, so every redraw
# arrived as its own "line" of pure backspaces and the log filled with tofu
# blocks (▯▯▯▯…) while the real counter went to the status strip. Stripped at the
# one place raw process bytes become text.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_REDRAW_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_stream_line(text):
    """A process line with terminal redraw codes (ANSI escapes, backspaces)
    removed. Tabs and the text itself are left alone."""
    return _REDRAW_RE.sub("", _ANSI_RE.sub("", text))


# setsid isolates a run in its own process group so TERMINATE can reach the
# whole tree. util-linux only — absent on macOS, where we degrade to a
# single-process terminate rather than failing to launch anything at all.
_SETSID = shutil.which("setsid") or ""


class ProcessRunner(QObject):
    # Process group of the current run, cached while its pid still resolves.
    _pgid = None
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
        # Wrapped in setsid WHERE IT EXISTS, which puts that bash in its own
        # process group — the thing that makes TERMINATE actually work.
        # QProcess.terminate() signals only the bash, so a WarpTools that had
        # already spawned WarpWorker left the worker running on the GPU with no
        # way to stop it from the app. Signalling the group reaches every
        # descendant. setsid execs in place (our child is not a group leader),
        # so the pid QProcess tracks is still the bash.
        #
        # Conditional because setsid is util-linux and absent on macOS: naming
        # it unconditionally would fail to start EVERY run there. Without it
        # the group guard declines and terminate falls back to its old
        # single-process behaviour, which is no worse than before.
        self._pgid = None                  # re-learned on the first signal
        if _SETSID:
            self.proc.start(_SETSID, ["bash", "-lc", command])
        else:
            self.proc.start("bash", ["-lc", command])

    def _signal_group(self, sig):
        """Signal the whole process group, or return False to fall back.

        REFUSES to signal our own group: if setsid did not take effect the
        child shares tomogration's group, and killing that would take the app
        down with the job.

        The group id is REMEMBERED while the pid is still resolvable, because
        the SIGKILL escalation runs 3 s after the SIGTERM — by which time the
        bash is usually a zombie and getpgid can no longer answer. Deriving it
        fresh each time meant the escalation silently reached nothing, so a
        child that ignored SIGTERM survived exactly as before."""
        pgid = self._pgid
        try:
            pid = int(self.proc.processId())
            if pid > 0:
                pgid = os.getpgid(pid)
                self._pgid = pgid
        except (OSError, ValueError, TypeError, AttributeError):
            pass                               # dead pid: use what we cached
        try:
            if not pgid or pgid == os.getpgid(0):
                return False                   # not isolated — never signal it
            os.killpg(pgid, sig)
            return True
        except (OSError, ValueError, TypeError):
            return False

    def terminate(self):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            # The group first, so children (WarpWorker, and anything else the
            # command spawned) go too; the bare terminate is the fallback.
            if not self._signal_group(signal.SIGTERM):
                self.proc.terminate()      # SIGTERM, this process only
            self.line.emit("[terminate requested]", "fail")
            # Escalate to SIGKILL if it's still alive after 3s (brief: Ctrl-C
            # equivalent that kills the job, not the GUI). The timer must hold
            # the process it terminated, NOT read self.proc when it fires: if
            # SIGTERM works quickly and the queue chains the next job inside
            # those 3 s, self.proc is already the new run — which must not be
            # the one that gets killed.
            QTimer.singleShot(3000, lambda p=self.proc: self._kill_if_alive(p))

    def _kill_if_alive(self, proc):
        if proc and proc.state() != QProcess.NotRunning:
            if not (proc is self.proc and self._signal_group(signal.SIGKILL)):
                proc.kill()
            self.line.emit("[killed]", "fail")

    def busy(self):
        return self.proc is not None and self.proc.state() != QProcess.NotRunning

    def _emit(self, raw, level):
        text = clean_stream_line(raw)
        # A chunk that was nothing BUT redraw codes carried no message — dropping
        # it keeps the log copy-worthy instead of trading tofu for blank lines.
        if raw.strip() and not text.strip():
            return
        self.line.emit(text, level)

    def _stdout(self):
        for ln in bytes(self.proc.readAllStandardOutput()).decode(errors="replace").splitlines():
            self._emit(ln, "out")

    def _stderr(self):
        for ln in bytes(self.proc.readAllStandardError()).decode(errors="replace").splitlines():
            self._emit(ln, "err")

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
        """Series to list: prefer Thumbnails/* (Tomo5 names), then mdocs/*.mdoc,
        then the coarse stacks themselves.

        The stacks are last because a repaired/quarantined project has the
        truest series list in mdocs/ — but they are here so a project that was
        sorted without its mdocs still lists something to inspect."""
        thumbs = self.project.find_thumbnails_dir()
        if thumbs is not None:
            return sorted(p.stem for p in thumbs.glob("*.mrc"))
        md = self.project.root / "mdocs"
        names = sorted(f.stem for f in md.glob("*.mdoc")) if md.is_dir() else []
        return names or self.project.coarse_stack_stems()

    def _populate(self):
        while self.list_box.count():
            w = self.list_box.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.rows = []
        # Inspection is where the coarse stacks are first actually needed, so
        # this is where they get filed. Sorting a Tomo5 dump leaves a
        # Position*.mrc per series loose in the root; they go to
        # mrcs-tiltseries-coarse/ and are opened from there.
        n_filed = self.project.collect_coarse_stacks()
        if n_filed:
            self.log_fn(f"Filed {n_filed} coarse tilt stack(s) from the project "
                        f"root into {COARSE_DIR}/ — 3dmod opens them from there.",
                        "ok")
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
# Viewer picker — which volumes of a finished job to open
# ===========================================================================
class ViewerPickDialog(QDialog):
    """Tree of what a job produced: one branch per tomogram, its volumes as
    checkable leaves. On accept, .files holds the chosen paths IN ORDER.

    Order is load-bearing: tomoview takes its base image from the first path
    and scales every later layer to that shape, so each series' tomogram is
    listed (and returned) before its overlays."""

    def __init__(self, parent, inventory, title="Open in napari", note=""):
        super().__init__(parent)
        self.files = None
        self.setWindowTitle(title)
        self.resize(640, 560)
        v = QVBoxLayout(self)
        lab = QLabel(note or (
            "Tick what to open. A tomogram's own volume is listed first — "
            "napari scales every other layer to it — and layers can be turned "
            "on and off inside napari afterwards."))
        lab.setWordWrap(True)
        v.addWidget(lab)

        # Created BEFORE the tree is populated: setCheckState fires itemChanged
        # for every row, and the handler reports the count — so the label has to
        # exist first. (VariantsDialog had exactly this bug; fixing it there and
        # not generalising the test is why it came back here.)
        self.count = QLabel("")
        self.count.setStyleSheet("color:#9a9a9a;font-size:11px;")

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["tomogram / volume"])
        self.tree.setColumnCount(1)
        self.tree.itemChanged.connect(self._cascade)
        for i, (series, files) in enumerate(sorted(inventory.items())):
            top = QTreeWidgetItem(self.tree, [series or "(all volumes)"])
            top.setFlags(top.flags() | Qt.ItemIsUserCheckable
                         | Qt.ItemIsAutoTristate)
            top.setCheckState(0, Qt.Unchecked)
            for f in files:
                leaf = QTreeWidgetItem(top, [os.path.basename(f)])
                leaf.setFlags(leaf.flags() | Qt.ItemIsUserCheckable)
                leaf.setCheckState(0, Qt.Unchecked)
                leaf.setData(0, Qt.UserRole, f)
                leaf.setToolTip(0, f)
            # The first tomogram is on by default: a readable view, with
            # everything else one click away. Ticking all 72 by default would
            # be a wall of layers.
            if i == 0:
                top.setCheckState(0, Qt.Checked)
            top.setExpanded(i == 0)
        v.addWidget(self.tree, 1)
        v.addWidget(self.count)

        row = QHBoxLayout()
        for label, on in (("Select all", True), ("Select none", False)):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, o=on: self._set_all(o))
            row.addWidget(b)
        row.addStretch(1)
        ok = QPushButton("Open")
        ok.setDefault(True)
        ok.clicked.connect(self._accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        row.addWidget(ok)
        row.addWidget(cancel)
        v.addLayout(row)
        self._update_count()

    def _tops(self):
        return [self.tree.topLevelItem(i)
                for i in range(self.tree.topLevelItemCount())]

    def _cascade(self, item, _col):
        # A parent's tick means "all of this tomogram"; Qt's auto-tristate
        # handles child -> parent, this handles parent -> children.
        if item.childCount():
            state = item.checkState(0)
            if state != Qt.PartiallyChecked:
                self.tree.blockSignals(True)
                for i in range(item.childCount()):
                    item.child(i).setCheckState(0, state)
                self.tree.blockSignals(False)
        self._update_count()

    def _set_all(self, on):
        self.tree.blockSignals(True)
        for top in self._tops():
            top.setCheckState(0, Qt.Checked if on else Qt.Unchecked)
            for i in range(top.childCount()):
                top.child(i).setCheckState(0, Qt.Checked if on else Qt.Unchecked)
        self.tree.blockSignals(False)
        self._update_count()

    def _chosen(self):
        """Ticked layers, in order, WITHOUT duplicates.

        Every sweep row lists its own tomogram, so ticking four rows of the
        same series sent the identical volume to napari four times — four full
        copies in memory and four indistinguishable layers to scroll past. The
        first occurrence wins, which keeps the tomogram first (tomoview scales
        every later layer to it)."""
        out, seen = [], set()
        for top in self._tops():
            for i in range(top.childCount()):
                leaf = top.child(i)
                if leaf.checkState(0) != Qt.Checked:
                    continue
                path = leaf.data(0, Qt.UserRole)
                if path in seen:
                    continue
                seen.add(path)
                out.append(path)
        return out

    def _update_count(self, *_):
        n = len(self._chosen())
        self.count.setText("nothing ticked" if not n else
                           f"{n} layer(s) — the first is the base image")

    def _accept(self):
        self.files = self._chosen()
        self.accept()


# ===========================================================================
# Tomogram picker — fills a *_TOMO_LIST field from what is on disk
# ===========================================================================
class TomoPickDialog(QDialog):
    """Check-list of the tomograms found in a step's input folder.

    The *_TOMO_LIST knobs (IsoNet train/predict, every MemBrain step) take
    SERIES STEMS, not counts and not 'the first N files' — so the honest way to
    fill one is to look at the folder. Files are grouped by stem (a series' own
    derived files — _scores.mrc, _segmented_threshold_*.mrc — collapse into one
    row, matching the wrappers' '${stem}_*.mrc' glob). On accept, .stems holds
    the chosen stems in list order."""

    def __init__(self, parent, folder, current="", title="Choose tomograms",
                 hint=""):
        super().__init__(parent)
        self.stems = None
        self.groups = tomogram_stems(folder)
        self.setWindowTitle(title)
        self.resize(520, 520)

        v = QVBoxLayout(self)
        head = QLabel(f"<b>{len(self.groups)}</b> tomogram series in "
                      f"<code>{_html.escape(str(folder))}</code>")
        head.setTextFormat(Qt.RichText)
        head.setWordWrap(True)
        v.addWidget(head)
        if hint:
            h = QLabel(hint)
            h.setWordWrap(True)
            h.setStyleSheet("color:#9a9a9a;font-size:11px;")
            v.addWidget(h)

        # Same rule as ViewerPickDialog: the label the handler writes to must
        # exist before the handler can be reached.
        self.count = QLabel("")
        self.count.setStyleSheet("color:#9a9a9a;font-size:11px;")
        self.list = QListWidget()
        self.list.itemChanged.connect(self._update_count)
        already = set(str(current).replace(",", " ").split())
        for stem, files in self.groups:
            n = len(files)
            it = QListWidgetItem(stem if n == 1 else f"{stem}   ({n} files)")
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if stem in already else Qt.Unchecked)
            it.setData(Qt.UserRole, stem)
            it.setToolTip("\n".join(files))
            self.list.addItem(it)
        v.addWidget(self.list, 1)

        # Names that are in the field but NOT on disk: the wrappers warn about
        # these at run time; say so here, while it is still cheap to fix.
        missing = sorted(already - {s for s, _ in self.groups})
        if missing:
            m = QLabel("⚠ in the field but not in this folder: " + " ".join(missing))
            m.setStyleSheet("color:#c0392b;font-size:11px;")
            m.setWordWrap(True)
            v.addWidget(m)

        v.addWidget(self.count)

        row = QHBoxLayout()
        for label, slot in (("Select all", lambda: self._set_all(True)),
                            ("Select none", lambda: self._set_all(False))):
            b = QPushButton(label)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        ok = QPushButton("Use these")
        ok.setDefault(True)
        ok.clicked.connect(self._accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        row.addWidget(ok)
        row.addWidget(cancel)
        v.addLayout(row)
        self._update_count()

    def _items(self):
        return [self.list.item(i) for i in range(self.list.count())]

    def _set_all(self, on):
        self.list.blockSignals(True)
        for it in self._items():
            it.setCheckState(Qt.Checked if on else Qt.Unchecked)
        self.list.blockSignals(False)
        self._update_count()

    def _checked(self):
        return [it.data(Qt.UserRole) for it in self._items()
                if it.checkState() == Qt.Checked]

    def _update_count(self, *_):
        n = len(self._checked())
        self.count.setText("none selected — the field will be cleared (= ALL "
                           "tomograms)" if not n else f"{n} selected")

    def _accept(self):
        self.stems = self._checked()
        self.accept()


# ===========================================================================
# Inventory browser — fill a path field from what already exists
# ===========================================================================
class InventoryDialog(QDialog):
    """Pick a file or folder from the project's past work.

    Three views, one list widget: SOURCES (every job newest-first, plus the
    work that only exists as directories — RELION jobs, AreTomo versions,
    canonical pipeline dirs), then DIRECTORY levels with ../ at the top, then
    an expanded FAMILY (a collapsed '72 files' row opened up). The model is
    tomogration_inventory — pure, bounded, ceph-safe; this class only draws.

    On accept, .chosen holds the project-relative path for the field."""

    PAGE_ROWS = 10                     # the visible window; the list scrolls

    def __init__(self, parent, root, title="Choose from this project",
                 hint=""):
        super().__init__(parent)
        self.root = Path(root)
        self.chosen = None
        self._mode = "sources"         # sources | dir | family
        self._rel = ""                 # current dir (dir/family modes)
        self._family = None            # (dir_rel, [names]) in family mode
        self.setWindowTitle(title)
        self.resize(560, 480)

        v = QVBoxLayout(self)
        self.head = QLabel("")
        self.head.setTextFormat(Qt.RichText)
        self.head.setWordWrap(True)
        v.addWidget(self.head)
        if hint:
            h = QLabel(hint)
            h.setWordWrap(True)
            h.setStyleSheet("color:#9a9a9a;font-size:11px;")
            v.addWidget(h)
        self.list = QListWidget()
        self.list.itemActivated.connect(self._enter)   # double-click / Enter
        v.addWidget(self.list, 1)
        self.note = QLabel("")
        self.note.setStyleSheet("color:#9a9a9a;font-size:11px;")
        self.note.setWordWrap(True)
        v.addWidget(self.note)
        row = QHBoxLayout()
        self.use_dir = QPushButton("Use this folder")
        self.use_dir.clicked.connect(self._accept_dir)
        row.addWidget(self.use_dir)
        row.addStretch(1)
        use = QPushButton("Use selection")
        use.setDefault(True)
        use.clicked.connect(self._accept_selection)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        row.addWidget(use)
        row.addWidget(cancel)
        v.addLayout(row)
        self._show_sources()

    # ---- views ----
    def _add(self, text, payload, tip=""):
        it = QListWidgetItem(text)
        it.setData(Qt.UserRole, payload)
        if tip:
            it.setToolTip(tip)
        self.list.addItem(it)

    def _show_sources(self):
        self._mode, self._rel, self._family = "sources", "", None
        self.list.clear()
        try:
            rows = inventory_sources(self.root)
        except Exception:
            rows = []
        for r in rows:
            mark = {"job": "🗂", "relion": "🧭", "dir": "📁"}.get(r["kind"], "📁")
            self._add(f"{mark} {r['title']}     {r.get('subtitle', '')}",
                      ("src", r), tip=r.get("rel", ""))
        self.head.setText(f"<b>{len(rows)}</b> places with past work — "
                          f"open one (double-click / Enter)")
        self.note.setText("")
        self.use_dir.setEnabled(False)

    def _show_dir(self, rel):
        self._mode, self._rel, self._family = "dir", rel, None
        self.list.clear()
        entries, truncated = list_dir(self.root, rel)
        self._add("⬆  ../", ("up", None))
        for e in entries:
            if e["is_dir"]:
                self._add(f"📁 {e['name']}/     ({e['count']}"
                          f"{'+' if e['count'] >= 100 else ''} items)",
                          ("dir", e))
            elif e.get("family"):
                self._add(f"≡ {e['name']}     × {e['count']} files",
                          ("family", e),
                          tip="a family of similarly-named files — open to "
                              "pick one")
            else:
                self._add(f"   {e['name']}     ({_human_size(e.get('size', 0))})",
                          ("file", e))
        shown = rel or "(project root)"
        self.head.setText(f"<code>{_html.escape(shown)}</code>")
        self.note.setText("listing capped — this folder holds more than shown"
                          if truncated else "")
        self.use_dir.setEnabled(True)

    def _show_locations(self, title, locs):
        """A job's several on-disk locations, each one hop from its files."""
        self._mode, self._rel, self._family = "sources", "", None
        self.list.clear()
        self._add("⬆  ../", ("up", None))
        for rel in locs:
            self._add(f"📁 {rel}/", ("dir", {"rel": rel}))
        self.head.setText(f"{_html.escape(title)} — its files live in "
                          f"<b>{len(locs)}</b> places")
        self.note.setText("")
        self.use_dir.setEnabled(False)

    def _show_family(self, dir_rel, fam):
        self._mode, self._family = "family", (dir_rel, fam)
        self.list.clear()
        self._add("⬆  ../", ("up", None))
        cap = 500
        for n in fam[:cap]:
            self._add(f"   {n}", ("file", {"name": n, "is_dir": False,
                                           "rel": f"{dir_rel}/{n}" if dir_rel
                                           else n}))
        self.head.setText(f"<code>{_html.escape(dir_rel or '(project root)')}"
                          f"</code> — {len(fam)} files in this family")
        self.note.setText(f"showing the first {cap} of {len(fam)}"
                          if len(fam) > cap else "")
        self.use_dir.setEnabled(True)

    # ---- navigation ----
    def _enter(self, item):
        kind, payload = item.data(Qt.UserRole)
        if kind == "up":
            if self._mode == "family":
                self._show_dir(self._rel)
            elif self._rel:
                parent = str(Path(self._rel).parent)
                self._show_dir("" if parent == "." else parent)
            else:
                self._show_sources()
        elif kind == "src":
            locs = payload.get("locs") or []
            if len(locs) > 1:
                # A job whose artifacts live in SEVERAL places (its jobs/ dir
                # plus wherever params pointed the outputs) — list them first.
                self._show_locations(payload["title"], locs)
            else:
                self._show_dir(payload.get("rel", ""))   # '' = project root
        elif kind == "dir":
            self._show_dir(payload["rel"])
        elif kind == "family":
            self._show_family(payload["rel"], payload["family"])
        elif kind == "file":
            self.chosen = payload["rel"]
            self.accept()

    def _accept_selection(self):
        it = self.list.currentItem()
        if it is None:
            return
        kind, payload = it.data(Qt.UserRole)
        if kind == "file":
            self.chosen = payload["rel"]
            self.accept()
        elif kind == "dir":
            self.chosen = payload["rel"]
            self.accept()
        elif kind == "src" and payload.get("rel"):
            self.chosen = payload["rel"]
            self.accept()
        elif kind == "family":
            self._show_family(payload["rel"], payload["family"])
        elif kind == "up":
            self._enter(it)

    def _accept_dir(self):
        if self._mode in ("dir", "family"):
            self.chosen = self._rel
            self.accept()


def _human_size(n):
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{n} B"


# ===========================================================================
# Variant sweep — one card per combination of the values you list
# ===========================================================================
class VariantsDialog(QDialog):
    """Build a sweep: every parameter can hold SEVERAL values, and one job card
    is built per combination.

    Sweeping used to mean editing the form, pressing '+ Queue variant', editing
    it again, pressing it again — with nothing on screen saying what you had
    already queued. Here the whole sweep is visible at once before anything is
    created. On accept, .variants holds one params dict per combination, .labels
    the matching card names, and .queue says whether to queue them immediately.

    Values are typed exactly as the form types them (spin box -> int, choice ->
    its flag string, checkbox -> bool), because these dicts go straight into
    build_command."""

    MAX_VARIANTS = 64          # a runaway grid is a mistake, not a plan

    def __init__(self, parent, spec, values):
        super().__init__(parent)
        self.spec = spec
        self.variants, self.labels, self.queue = [], [], False
        self.rows = {}          # param name -> [(row widget, getter), …]
        self.setWindowTitle(f"Variants — {spec.get('label', spec['id'])}")
        self.resize(760, 640)

        # EVERY widget _refresh() touches is created FIRST, because building the
        # form calls _refresh once per parameter row (via _add_row) — long before
        # the button row further down would have existed. Creating them here and
        # adding them to the layout below keeps the on-screen order unchanged.
        # Without this the dialog raised AttributeError: 'VariantsDialog' object
        # has no attribute 'build_btn' on open, for every stage.
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        # Variants of a step that writes to a fixed path all write to the SAME
        # path: without this they would overwrite each other one after another,
        # and only the last would survive to be compared.
        self.suffix_cb = QCheckBox("Give each variant its own output folder "
                                   "(append _v1, _v2, …)")
        self.suffix_cb.setChecked(True)
        self.build_btn = QPushButton("Build cards")
        self.queue_btn = QPushButton("Build & queue")

        v = QVBoxLayout(self)
        v.addWidget(QLabel(
            "Add values with ＋ — one card is built per COMBINATION. Parameters "
            "left with a single value are shared by every variant."))

        inner = QWidget()
        self.form = QVBoxLayout(inner)
        self.form.setAlignment(Qt.AlignTop)
        self.form.setSpacing(2)
        for p in spec.get("params", []):
            self.form.addWidget(self._param_block(p, values.get(p["name"])))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(inner)
        v.addWidget(scroll, 1)

        v.addWidget(self.summary)
        self.suffix_cb.stateChanged.connect(self._refresh)
        v.addWidget(self.suffix_cb)

        row = QHBoxLayout()
        row.addStretch(1)
        self.build_btn.clicked.connect(lambda: self._accept(queue=False))
        self.queue_btn.clicked.connect(lambda: self._accept(queue=True))
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        for b in (self.build_btn, self.queue_btn, cancel):
            row.addWidget(b)
        v.addLayout(row)
        self._refresh()

    # ---- one parameter: its title, its stack of values, its ＋ ----
    def _param_block(self, p, value):
        box = QWidget()
        outer = QVBoxLayout(box)
        outer.setContentsMargins(0, 4, 0, 4)
        outer.setSpacing(2)
        head = QLabel(f"<b>{_html.escape(param_title(p))}</b> "
                      f"<code style='color:#8fb4d8;font-size:10px;'>"
                      f"{_html.escape(param_wire_name(p))}</code>")
        head.setTextFormat(Qt.RichText)
        outer.addWidget(head)
        stack = QVBoxLayout()
        stack.setContentsMargins(12, 0, 0, 0)
        stack.setSpacing(2)
        outer.addLayout(stack)
        self.rows[p["name"]] = []
        self._add_row(p, stack, value)
        add = QPushButton("＋")
        add.setFixedWidth(34)
        add.setToolTip(f"Add another value for {param_title(p)} — every value "
                       f"multiplies the number of cards built.")
        add.clicked.connect(lambda _=False, p=p, s=stack: self._add_row(p, s))
        arow = QHBoxLayout()
        arow.setContentsMargins(12, 0, 0, 0)
        arow.addWidget(add)
        arow.addStretch(1)
        outer.addLayout(arow)
        return box

    def _add_row(self, p, stack, value=None):
        rows = self.rows[p["name"]]
        if value is None and rows:
            value = rows[-1][1]()          # seed from the last value: edit, don't retype
        w, getter = self._value_widget(p, value)
        rw = QWidget()
        h = QHBoxLayout(rw)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(w, 1)
        rm = QPushButton("✕")
        rm.setFixedWidth(28)
        rm.setToolTip("Remove this value")
        rm.clicked.connect(lambda _=False, name=p["name"], rw=rw: self._drop_row(name, rw))
        h.addWidget(rm)
        stack.addWidget(rw)
        rows.append((rw, getter))
        self._refresh()

    def _drop_row(self, name, rw):
        rows = self.rows[name]
        if len(rows) <= 1:                 # every parameter keeps one value
            return
        self.rows[name] = [r for r in rows if r[0] is not rw]
        rw.setParent(None)
        rw.deleteLater()
        self._refresh()

    def _value_widget(self, p, value):
        """The same widget the job builder would use, so the value has the same
        TYPE — build_command sees ints from spin boxes and bools from checks."""
        kind = p["kind"]
        if kind == "check":
            w = QCheckBox()
            w.setChecked(bool(value))
            w.stateChanged.connect(self._refresh)
            return w, (lambda w=w: w.isChecked())
        if kind in ("slider_int", "env_int"):
            w = QSpinBox()
            w.setMinimum(int(p["min"]))
            w.setMaximum(int(p["max"]))
            w.setSingleStep(int(p.get("step", 1)))
            try:
                w.setValue(int(value))
            except (TypeError, ValueError):
                w.setValue(int(p.get("default") or p["min"]))
            w.setKeyboardTracking(False)
            w.valueChanged.connect(lambda _v: self._refresh())
            return w, (lambda w=w: w.value())
        if kind == "choice":
            w = QComboBox()
            for label, val in p.get("choices", []):
                w.addItem(label, val)
            i = w.findData(value)
            w.setCurrentIndex(i if i >= 0 else 0)
            w.currentIndexChanged.connect(self._refresh)
            return w, (lambda w=w: w.currentData())
        w = QLineEdit(str(value if value is not None else ""))
        w.textChanged.connect(self._refresh)
        return w, (lambda w=w: w.text())

    # ---- the grid ----
    def _combinations(self):
        """One params dict per combination, deduplicated (two identical boxes
        are a slip, not a request for the same job twice)."""
        names = [p["name"] for p in self.spec.get("params", [])]
        lists = []
        for n in names:
            seen, vals = set(), []
            for _, get in self.rows.get(n, []):
                val = get()
                key = repr(val)
                if key not in seen:
                    seen.add(key)
                    vals.append(val)
            lists.append(vals or [""])
        out = [{}]
        for n, vals in zip(names, lists):
            out = [dict(base, **{n: v}) for base in out for v in vals]
            if len(out) > self.MAX_VARIANTS * 4:      # bail early on a runaway grid
                break
        varied = [n for n, vals in zip(names, lists) if len(vals) > 1]
        return out, varied

    def _suffixed(self, variants):
        """Append _v<i> to the step's output paths so variants cannot overwrite
        one another. Only touches variants that would otherwise collide."""
        keys = [k for k in (self.spec.get("output_params") or [])]
        if not (keys and self.suffix_cb.isChecked() and len(variants) > 1):
            return variants
        seen = {}
        for i, vals in enumerate(variants, 1):
            sig = tuple(str(vals.get(k, "")) for k in keys)
            seen.setdefault(sig, []).append(i)
        if all(len(idx) == 1 for idx in seen.values()):
            return variants                      # already distinct: leave them alone
        out = []
        for i, vals in enumerate(variants, 1):
            v = dict(vals)
            for k in keys:
                base = str(v.get(k, "") or "").rstrip("/")
                if base:
                    v[k] = f"{base}_v{i}"
            out.append(v)
        return out

    def _label_for(self, vals, varied, i):
        bits = []
        for p in self.spec.get("params", []):
            if p["name"] in varied:
                bits.append(f"{param_title(p)}={vals.get(p['name'])}")
        tail = ", ".join(bits) if bits else f"variant {i}"
        return f"{self.spec.get('label', self.spec['id'])} · {tail}"[:120]

    def _refresh(self, *_):
        variants, varied = self._combinations()
        n = len(variants)
        too_many = n > self.MAX_VARIANTS
        self.build_btn.setEnabled(not too_many and n > 0)
        self.queue_btn.setEnabled(not too_many and n > 0)
        if too_many:
            self.summary.setStyleSheet("color:#c0392b;")
            self.summary.setText(
                f"{n} combinations — more than {self.MAX_VARIANTS}. Trim the "
                f"values: every extra value MULTIPLIES the grid.")
            return
        self.summary.setStyleSheet("color:#9a9a9a;")
        variants = self._suffixed(variants)
        preview = "; ".join(
            self._label_for(v, varied, i).split(" · ", 1)[-1]
            for i, v in enumerate(variants[:6], 1))
        more = f" … (+{n - 6} more)" if n > 6 else ""
        self.summary.setText(
            f"{n} card{'s' if n != 1 else ''} will be built"
            + (f", varying {', '.join(varied)}:  {preview}{more}"
               if varied else " (nothing varies yet — press ＋ on a parameter)"))

    def _accept(self, queue):
        variants, varied = self._combinations()
        if not variants or len(variants) > self.MAX_VARIANTS:
            return
        variants = self._suffixed(variants)
        self.variants = variants
        self.labels = [self._label_for(v, varied, i)
                       for i, v in enumerate(variants, 1)]
        self.queue = queue
        self.accept()


class _Chart(QWidget):
    """A small bar or scatter chart, painted directly.

    QtCharts is not guaranteed present and matplotlib lives in the conda envs,
    not the GUI's venv — so these are drawn with QPainter. Bars for "one number
    per setting", scatter for the recall/virions trade-off, where the SHAPE is
    the message and a table cannot show it."""

    def __init__(self, title, kind="bar", note=""):
        super().__init__()
        self.title, self.kind, self.note = title, kind, note
        self.points = []            # bar: [(label, value, n)]  scatter: [(l,x,y,row)]
        self.setMinimumHeight(190)

    def set_points(self, points):
        self.points = list(points or [])
        self.update()

    def paintEvent(self, _ev):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        qp.fillRect(0, 0, w, h, QColor("#1b1b1b"))
        qp.setPen(QColor("#d6d6d6"))
        qp.drawText(8, 16, self.title)
        if self.note:
            qp.setPen(QColor("#9a9a9a"))
            qp.drawText(8, 32, self.note)
        left, right, top, bottom = 46, w - 10, 40, h - 26
        if not self.points:
            qp.setPen(QColor("#9a9a9a"))
            qp.drawText(left, (top + bottom) // 2, "nothing selected")
            return
        qp.setPen(QColor("#3a3a3a"))
        qp.drawLine(left, bottom, right, bottom)
        qp.drawLine(left, top, left, bottom)

        if self.kind == "bar":
            vals = [v for _l, v, _n in self.points if v is not None]
            top_v = max(vals) if vals else 1.0
            top_v = top_v or 1.0
            n = len(self.points)
            slot = max(14, (right - left) // max(n, 1))
            for i, (label, v, cnt) in enumerate(self.points):
                x = left + i * slot + 4
                if v is None:
                    qp.setPen(QColor("#666"))
                    qp.drawText(x, bottom - 4, "—")
                else:
                    bh = int((bottom - top) * (v / top_v))
                    qp.fillRect(x, bottom - bh, slot - 8, bh, QColor("#4a90d9"))
                    qp.setPen(QColor("#d6d6d6"))
                    qp.drawText(x, bottom - bh - 4, f"{v:.3g}")
                qp.setPen(QColor("#9a9a9a"))
                qp.drawText(x, bottom + 14, str(label)[:12])
        else:                                    # scatter: recall vs virions
            xs = [p[1] for p in self.points] or [0]
            ys = [p[2] for p in self.points] or [0]
            xmax, ymax = max(xs) or 1.0, max(ys) or 1.0
            for _label, x, y, _row in self.points:
                px = left + int((right - left) * (x / xmax) * 0.95)
                py = bottom - int((bottom - top) * (y / ymax) * 0.95)
                qp.setBrush(QColor("#4a90d9"))
                qp.setPen(QColor("#8fb4d8"))
                qp.drawEllipse(px - 4, py - 4, 8, 8)
            qp.setPen(QColor("#9a9a9a"))
            qp.drawText(left, bottom + 16, "recall →")
            qp.save()
            qp.translate(14, bottom)
            qp.rotate(-90)
            qp.drawText(0, 0, "virions accepted →")
            qp.restore()


class AnalysisDialog(QDialog):
    """Charts over a finished parameter sweep, with the picture the table cannot
    show: which variant, which threshold, which cutoff — and the recall/virions
    trade-off that neither column settles alone."""

    def __init__(self, parent, rows, root, on_compare=None, on_pick=None):
        super().__init__(parent)
        self.rows = list(rows or [])
        self.root = root
        self.on_compare = on_compare
        self.on_pick = on_pick
        self.boxes = {}                  # axis -> {value: QCheckBox}
        self.charts = []
        self.setWindowTitle("Parameter sweep — analysis")
        self.resize(1000, 760)

        v = QVBoxLayout(self)
        head = QLabel(
            "<b>What these numbers are.</b> Each component is fitted with a "
            "SPHERE — centre and radius — because a free ellipsoid follows the "
            "missing wedge's smear along z and its volume runs ~50% high. "
            "<b>Virions</b> is how many components passed all eight gates. "
            "<b>Recall</b> is the fraction of segmented voxels that ended up "
            "inside an accepted virion: it is what stops a setting scoring well "
            "by keeping three easy virions and discarding everything else, so "
            "read the two together.")
        head.setWordWrap(True)
        head.setStyleSheet("color:#c8c8c8;font-size:11px;")
        v.addWidget(head)

        # Everything _refresh() touches is built BEFORE the first checkbox is
        # ticked: setChecked(True) fires stateChanged, and the handler reads all
        # of these. (The dialog-init audit caught this here before it ever ran —
        # the same shape as VariantsDialog and ViewerPickDialog.)
        self.charts = [
            ("variant", "virions_accepted", _Chart("Virions by input tomogram")),
            ("variant", "recall", _Chart("Recall by input tomogram")),
            ("threshold", "virions_accepted", _Chart("Virions by threshold")),
            ("threshold", "recall", _Chart("Recall by threshold")),
            ("cutoff", "virions_accepted", _Chart("Virions by size cutoff")),
            ("cutoff", "mean_diameter_nm", _Chart("Mean diameter (nm) by cutoff")),
        ]
        self.scatter = _Chart("Recall vs virions — every cell", kind="scatter",
                              note="up and to the right is better; a point high "
                                   "on one axis only is not a good setting")
        self.shape = _Chart("Sphere vs free ellipsoid volume (ratio per virion)",
                            note="1.0 = the wedge is not distorting this virion; "
                                 "~1.5 = the ellipsoid is inflating it")
        self.count = QLabel("")
        self.count.setStyleSheet("color:#9a9a9a;font-size:11px;")

        show = QLabel("<b>Show:</b>")
        v.addWidget(show)
        filt = QHBoxLayout()
        for axis, title in (("variant", "Input tomogram"),
                            ("threshold", "Threshold"),
                            ("cutoff", "Size cutoff")):
            col = QVBoxLayout()
            col.addWidget(QLabel(title))
            self.boxes[axis] = {}
            for val in EX.axis_values(self.rows, axis):
                cb = QCheckBox(str(val))
                cb.setChecked(True)
                cb.stateChanged.connect(self._refresh)
                self.boxes[axis][str(val)] = cb
                col.addWidget(cb)
            col.addStretch(1)
            box = QWidget()
            box.setLayout(col)
            filt.addWidget(box)
        filt.addStretch(1)
        fw = QWidget()
        fw.setLayout(filt)
        v.addWidget(fw)

        grid = QGridLayout()
        for i, (_ax, _m, ch) in enumerate(self.charts):
            grid.addWidget(ch, i // 2, i % 2)
        grid.addWidget(self.scatter, 3, 0, 1, 2)
        grid.addWidget(self.shape, 4, 0, 1, 2)
        gw = QWidget()
        gw.setLayout(grid)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(gw)
        v.addWidget(scroll, 1)

        v.addWidget(self.count)

        row = QHBoxLayout()
        row.addStretch(1)
        pick = QPushButton("🔍 View in napari…")
        pick.setToolTip("Choose exactly which cells and layers to open — the "
                        "same picker as the job's own View button.")
        pick.clicked.connect(self._pick)
        row.addWidget(pick)
        best = QPushButton("🔍 Compare the best two in napari")
        best.setToolTip("Open the two highest-scoring SELECTED cells, each in "
                        "its own napari window, with tomogram + components + "
                        "accepted fits.")
        best.clicked.connect(self._compare_best)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(best)
        row.addWidget(close)
        v.addLayout(row)
        self._refresh()

    def _selection(self):
        return {ax: {v for v, cb in boxes.items() if cb.isChecked()}
                for ax, boxes in self.boxes.items()}

    def _visible(self):
        return EX.filter_rows(self.rows, self._selection())

    def _refresh(self, *_):
        rows = self._visible()
        for axis, metric, chart in self.charts:
            chart.set_points(EX.group_stats(rows, axis, metric))
        self.scatter.set_points(EX.tradeoff_points(rows))
        ratios = EX.sphere_vs_ellipsoid(rows)
        # One bar per virion is unreadable; bucket the ratios instead.
        buckets = [("<1.1", 0), ("1.1-1.3", 0), ("1.3-1.5", 0), (">1.5", 0)]
        counts = dict(buckets)
        for _l, _s, _e, ratio in ratios:
            key = ("<1.1" if ratio < 1.1 else "1.1-1.3" if ratio < 1.3
                   else "1.3-1.5" if ratio < 1.5 else ">1.5")
            counts[key] += 1
        self.shape.set_points([(k, counts[k], counts[k]) for k, _ in buckets])
        self.count.setText(
            f"{len(rows)} of {len(self.rows)} cells shown"
            + (f"   ·   {len(ratios)} virions with both fits" if ratios else
               "   ·   no ellipsoid comparison yet (re-run the fits to record it)"))

    def _pick(self):
        """Hand off to the full picker, so 'the best two' is never the only
        way to look at a sweep."""
        if self.on_pick:
            self.on_pick()

    def _compare_best(self):
        rows = EX.rank_rows(self._visible())
        pick = [r for r in rows if r.get("fit_masks")][:2]
        if not pick:
            QMessageBox.information(
                self, "Nothing to compare",
                "No selected cell has accepted fits to show.")
            return
        if self.on_compare:
            self.on_compare(pick)



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
# Card canvas (Phase 2) — a QGraphicsView rendering canvas_layout(). Real job
# cards + dashed discovered/orphan cards; the default (un-run) pipeline is NOT
# drawn here — it lives in the pinned outline widget (_PipelineOutline), which
# shares its slate palette so the two read as one system.
# ===========================================================================
# (fill, border) per status. Ghost = dim + dashed; others tint by outcome.
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
            # Deferred: menu.exec runs a nested event loop INSIDE this handler,
            # and menu actions (delete/hide/clear) rebuild the canvas —
            # scene.clear() would destroy this very item mid-event delivery
            # (intermittent segfault). Let the press finish first.
            node, pos = self._node, ev.screenPos()
            QTimer.singleShot(0, lambda c=self._canvas: c._menu(node, pos))
            ev.accept()
            return
        self._press_pos = self.pos()
        # Only a locked (navigation-mode) click opens the card: unlocked, the same
        # press starts a drag, and popping a panel on every arrange-gesture is
        # exactly the kind of noise that makes people relock the canvas.
        if self._canvas.locked:
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


# Annotation palette. Deliberately muted: a note that shouts is a note that
# competes with the cards, and the cards are the thing being read.
_NOTE_COLOURS = {
    "amber":  ("#3a3018", "#c9a227"),
    "blue":   ("#1b2b3a", "#4a86b8"),
    "green":  ("#1c2f22", "#4f9d69"),
    "violet": ("#2a2440", "#8b6ed6"),
    "red":    ("#33201e", "#b05a52"),
    "grey":   ("#262b30", "#6b7780"),
}


class _NoteItem(QGraphicsRectItem):
    """A canvas annotation: a sticky note, or a labelled frame behind a branch.

    Carries no data meaning and nothing that runs consults it — the dashed
    border says so at a glance, the same way ✋ says a status was set by hand.
    """
    def __init__(self, note, canvas):
        super().__init__(0, 0, float(note.get("w", 220)), float(note.get("h", 96)))
        self._note = note
        self._canvas = canvas
        self._press_pos = None
        frame = note.get("kind") == "frame"
        fill, edge = _NOTE_COLOURS.get(note.get("colour", "amber"),
                                       _NOTE_COLOURS["amber"])
        bg = QColor(fill)
        bg.setAlpha(90 if frame else 235)     # a frame must not dim its cards
        self.setBrush(QBrush(bg))
        pen = QPen(QColor(edge), 2 if frame else 1)
        pen.setStyle(Qt.PenStyle.DashLine)    # dashed = hand-made, carries nothing
        self.setPen(pen)
        self.setPos(float(note.get("x", 0)), float(note.get("y", 0)))
        self.setZValue(-15 if frame else 12)

        # QGraphicsSimpleTextItem, wrapped by hand -- NOT QGraphicsTextItem with
        # setTextWidth, which stored and reopened its text perfectly while
        # painting an empty box on the canvas. Every card label on this scene is
        # a simple item and every one of them renders, so the note is drawn the
        # same way. Simple items do not wrap, hence wrap_lines.
        f = QFont()
        f.setPointSize(10 if frame else 9)
        f.setBold(frame)
        fm = QFontMetrics(f)
        avail_w = max(40.0, float(note.get("w", 220)) - 14)
        # Clip to the note's own height: text spilling past the dashed border
        # reads as a rendering fault rather than as a note that needs resizing.
        rows = max(1, int(max(14.0, float(note.get("h", 96)) - 8)
                          // max(1, fm.lineSpacing())))
        body = "\n".join(wrap_lines(str(note.get("text", "")), avail_w,
                                    fm.horizontalAdvance, max_lines=rows))
        t = QGraphicsSimpleTextItem(body, self)
        t.setBrush(QColor(edge if frame else "#e8e2d0"))
        t.setFont(f)
        t.setPos(7, 4)
        if not canvas.locked:
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
            self.setCursor(Qt.OpenHandCursor)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.RightButton:
            note, pos = self._note, ev.screenPos()
            QTimer.singleShot(0, lambda c=self._canvas: c._note_menu(note, pos))
            ev.accept()
            return
        self._press_pos = self.pos()
        super().mousePressEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        note = self._note
        QTimer.singleShot(0, lambda c=self._canvas: c._edit_note(note))
        ev.accept()

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        if self._canvas.locked or self._press_pos is None:
            return
        now = self.pos()
        if (abs(now.x() - self._press_pos.x()) < 2
                and abs(now.y() - self._press_pos.y()) < 2):
            return
        self._press_pos = None
        self._canvas._note_moved(self._note, now.x(), now.y())


class _ConnectorHandle(QGraphicsRectItem):
    """The output port on a card. Drag it onto another card to make that card
    read this one's output.

    A separate item because the card body is draggable: a gesture starting on
    the card has to mean "move the card", so wiring needs its own grab point.
    The wire it creates is the SAME one "⇄ Set input" makes — this is an
    affordance over existing behaviour, not a second way to store lineage.
    """
    SIZE = 13

    def __init__(self, node, canvas, x, y):
        super().__init__(0, 0, self.SIZE, self.SIZE)
        self._node = node
        self._canvas = canvas
        self._line = None
        self.setPos(x, y)
        self.setBrush(QBrush(QColor("#2b3a2c")))
        self.setPen(QPen(QColor("#4f9d69"), 2))
        self.setZValue(6)
        self.setCursor(Qt.CrossCursor)
        self.setToolTip("Drag onto another card to feed it this job's output")

    def mousePressEvent(self, ev):
        if ev.button() != Qt.LeftButton or self._canvas.locked:
            ev.ignore()
            return
        pen = QPen(QColor("#8b6ed6"), 2)      # purple: a wire being proposed
        pen.setStyle(Qt.PenStyle.DashLine)
        p = self.scenePos()
        self._line = self.scene().addLine(p.x() + self.SIZE / 2,
                                          p.y() + self.SIZE / 2,
                                          p.x() + self.SIZE / 2,
                                          p.y() + self.SIZE / 2, pen)
        self._line.setZValue(20)
        ev.accept()

    def mouseMoveEvent(self, ev):
        if self._line is None:
            return
        p = self.scenePos()
        q = ev.scenePos()
        self._line.setLine(p.x() + self.SIZE / 2, p.y() + self.SIZE / 2,
                           q.x(), q.y())
        ev.accept()

    def mouseReleaseEvent(self, ev):
        if self._line is None:
            return
        scene = self.scene()
        scene.removeItem(self._line)
        self._line = None
        target = None
        for it in scene.items(ev.scenePos()):
            if isinstance(it, _CardItem):
                target = it._node
                break
        ev.accept()
        if target is None or target is self._node:
            return
        # Deferred for the same reason as the card menu: connecting repaints the
        # canvas, and scene.clear() would destroy this item mid-event.
        src, dst = self._node, target
        QTimer.singleShot(0, lambda c=self._canvas: c._connect_cards(src, dst))


class _PopoutChip(QGraphicsRectItem):
    """The little ❯ arrow on a card; pops out the card's own panel (builder ·
    details · outputs). Its own mousePressEvent handles the click so it doesn't
    also select or drag the card."""
    def __init__(self, node, canvas, x, y, w=22, h=18):
        super().__init__(0, 0, w, h)
        self._node = node
        self._canvas = canvas
        self.setPos(x, y)
        self.setBrush(QBrush(QColor("#2a3340")))
        self.setPen(QPen(QColor("#3a6ea5")))
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Pop out this card's panel — builder, details, outputs")
        t = QGraphicsSimpleTextItem("❯", self)
        t.setBrush(QColor("#9ec5ff"))
        f = QFont()
        f.setPointSize(9)
        f.setBold(True)
        t.setFont(f)
        t.setPos(7, 1)

    def mousePressEvent(self, ev):
        # Deferred like the card menu: the popout can trigger a canvas refresh,
        # and scene.clear() must not destroy this item mid-event delivery.
        node = self._node
        QTimer.singleShot(0, lambda c=self._canvas: c._details(node))
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
    _menu_handler = None
    _zoom_handler = None

    # A 72-job graph is far wider than any window, so zooming OUT matters more
    # than zooming in — hence the asymmetric range. Below ~15% the cards stop
    # being distinguishable and it is no longer a view of anything.
    MIN_ZOOM = 0.15
    MAX_ZOOM = 4.0

    def zoom_level(self):
        """Current scale, 1.0 = 1:1. Read off the transform rather than tracked
        separately, so Fit (which calls fitInView) reports honestly too."""
        return float(self.transform().m11()) or 1.0

    def set_zoom_handler(self, fn):
        """Called with the new level whenever it changes, for the readout."""
        self._zoom_handler = fn

    def zoom_to(self, level, anchor_mouse=False):
        """Scale to an ABSOLUTE level, clamped. Returns the level actually set."""
        level = max(self.MIN_ZOOM, min(self.MAX_ZOOM, float(level)))
        # zoom_level() already reads a degenerate transform as 1:1, so there is
        # nothing here to divide by zero.
        cur = self.zoom_level()
        prev = self.transformationAnchor()
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse if anchor_mouse
            else QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.scale(level / cur, level / cur)
        self.setTransformationAnchor(prev)
        if self._zoom_handler:
            self._zoom_handler(level)
        return level

    def zoom_by(self, factor, anchor_mouse=False):
        return self.zoom_to(self.zoom_level() * factor, anchor_mouse)

    def set_menu_handler(self, fn):
        """Right-click on EMPTY canvas. Cards and notes handle their own press,
        so this only fires where there is nothing — which is exactly where a new
        annotation should be placed."""
        self._menu_handler = fn

    def contextMenuEvent(self, ev):
        if self._menu_handler is None or self.itemAt(ev.pos()) is not None:
            super().contextMenuEvent(ev)
            return
        p = self.mapToScene(ev.pos())
        self._menu_handler(p.x(), p.y(), ev.globalPos())
        ev.accept()

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
        # Ctrl (or ⌘) + wheel zooms about the POINTER, checked before the pan
        # rules below because those consume a plain vertical wheel entirely.
        if ev.modifiers() & (Qt.ControlModifier | Qt.MetaModifier):
            step = d.y() or d.x()
            if step:
                self.zoom_by(1.0015 ** step, anchor_mouse=True)
                ev.accept()
                return
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
        if ev.modifiers() & (Qt.ControlModifier | Qt.MetaModifier):
            # Both spellings of each key: Ctrl+= is what you actually press for
            # "zoom in" on most layouts, and the numpad sends its own codes.
            if ev.key() in (Qt.Key_Plus, Qt.Key_Equal):
                self.zoom_by(1.25)
                ev.accept()
                return
            if ev.key() in (Qt.Key_Minus, Qt.Key_Underscore):
                self.zoom_by(1 / 1.25)
                ev.accept()
                return
            if ev.key() == Qt.Key_0:
                self.zoom_to(1.0)
                ev.accept()
                return
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
        if self._proc is not None:
            self._proc.deleteLater()     # else every tick leaks one QProcess
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
            if spec.get("legacy"):          # retired: readable on old cards only
                continue
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


class _OutlineRow(QFrame):
    """One stage in the pinned pipeline outline.

    Compact on purpose: the outline is the default pipeline as a printed
    reference down the screen edge, not a second set of cards. Click shows the
    stage's work on the canvas, double-click pops out its panel, drag drops a
    new job where it belongs, right-click offers the stage actions.
    """
    def __init__(self, stage_id, title, outline):
        super().__init__()
        self._stage_id = stage_id
        self._outline = outline
        self._press_at = None
        self.setObjectName("outlineRow")
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Click: show this step on the canvas.\n"
                        "Double-click: open its panel.\n"
                        "Drag onto the canvas to add a job there.")
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 3, 6, 3)
        v.setSpacing(0)
        self.title_lbl = QLabel(title)
        self.title_lbl.setStyleSheet(
            "color:#7b8b9a;font-size:11px;font-weight:600;background:transparent;")
        v.addWidget(self.title_lbl)
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet(
            "color:#3f4d5a;font-size:9px;background:transparent;")
        v.addWidget(self.status_lbl)
        self._set_row_style(False)

    def _set_row_style(self, active):
        # Slate, like the old rail cards — NEVER a job colour (see _add_card).
        # Active (a run is live on this stage) gets the amber edge instead.
        edge = "#f0a92a" if active else "#232f3b"
        self.setStyleSheet(
            f"QFrame#outlineRow {{background:#151b22;border:1px solid {edge};"
            f"border-radius:4px;}}")

    def set_status(self, n_jobs, on_disk, disk_label, running):
        if running:
            sub, colour = "▶ running", "#f0a92a"
        elif n_jobs:
            sub = f"{n_jobs} job{'s' if n_jobs != 1 else ''} →"
            colour = "#5c7186"
        elif on_disk:
            sub = "ran, no job record →" + (f" {disk_label}" if disk_label else "")
            colour = "#5c7186"
        else:
            sub, colour = "no jobs yet", "#3f4d5a"
        self.status_lbl.setText(sub)
        self.status_lbl.setStyleSheet(
            f"color:{colour};font-size:9px;background:transparent;")
        self._set_row_style(running)

    # Click / double-click / drag / menu all start from the same press, so the
    # row does its own gesture split: a real move becomes a stage drag (same
    # MIME as the palette — the canvas drop handler already understands it).
    def mousePressEvent(self, ev):
        if ev.button() == Qt.RightButton:
            self._outline._menu(self._stage_id, ev.globalPosition().toPoint()
                                if hasattr(ev, "globalPosition") else ev.globalPos())
            return
        self._press_at = ev.position().toPoint() if hasattr(ev, "position") \
            else ev.pos()

    def mouseMoveEvent(self, ev):
        if self._press_at is None:
            return
        here = ev.position().toPoint() if hasattr(ev, "position") else ev.pos()
        if (here - self._press_at).manhattanLength() < 8:
            return
        self._press_at = None
        md = QMimeData()
        md.setData(MIME_STAGE, self._stage_id.encode("utf-8"))
        drag = QDrag(self)
        drag.setMimeData(md)
        drag.exec(Qt.DropAction.CopyAction)

    def mouseReleaseEvent(self, ev):
        if self._press_at is not None:
            self._press_at = None
            self._outline._activate(self._stage_id)

    def mouseDoubleClickEvent(self, ev):
        self._press_at = None
        self._outline._open(self._stage_id)


class _OutlineGroupHeader(QFrame):
    """A category header in the pinned outline: '▾ 7. RECONSTRUCT · 3 jobs'.

    The categories are the stage groups the pipeline already declares; here
    they become real containers — click to collapse/expand, with a rolled-up
    job count so a folded category still says whether work lives inside it.
    """
    def __init__(self, group, outline):
        super().__init__()
        self._group = group
        self._outline = outline
        self.setObjectName("outlineGroup")
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("A category of job types — click to collapse/expand.")
        self.setStyleSheet(
            "QFrame#outlineGroup {background:#10161c;border:none;"
            "border-left:2px solid #22303c;border-radius:2px;}")
        h = QHBoxLayout(self)
        h.setContentsMargins(5, 2, 6, 2)
        h.setSpacing(4)
        self.arrow = QLabel("▾")
        self.arrow.setStyleSheet("color:#5c7186;font-size:8px;background:transparent;")
        self.arrow.setFixedWidth(10)
        h.addWidget(self.arrow)
        name = QLabel(group)
        gf = name.font()
        gf.setCapitalization(QFont.AllUppercase)
        gf.setLetterSpacing(QFont.AbsoluteSpacing, 1.0)
        gf.setBold(True)
        name.setFont(gf)
        name.setStyleSheet("color:#5c7186;font-size:8px;background:transparent;")
        h.addWidget(name, 1)
        self.count_lbl = QLabel("")
        self.count_lbl.setStyleSheet(
            "color:#3f4d5a;font-size:8px;background:transparent;")
        h.addWidget(self.count_lbl)

    def set_rollup(self, n_jobs, running):
        # The folded category's one line of truth: how much work is inside it,
        # and amber when any of it is live right now.
        self.count_lbl.setText("▶ running" if running
                               else (f"{n_jobs} job{'s' if n_jobs != 1 else ''}"
                                     if n_jobs else ""))
        self.count_lbl.setStyleSheet(
            "color:#f0a92a;font-size:8px;background:transparent;" if running
            else "color:#5c7186;font-size:8px;background:transparent;")

    def set_collapsed(self, collapsed):
        self.arrow.setText("▸" if collapsed else "▾")

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._outline._toggle_group(self._group)


class _PipelineOutline(QWidget):
    """The default pipeline, pinned down the left edge of the window.

    This replaces the in-scene template rail: painted in the scene it scrolled
    and zoomed with the work, so on any real project the reference you read
    the canvas against was usually off-screen. Pinned as a widget it is always
    there, slightly smaller than the cards, and the canvas keeps the full
    scene for actual work.

    Stages are filed under their CATEGORY (the pipeline's stage groups):
    each category is a collapsible section with a rolled-up job count, and the
    per-stage rows — the importable job types — live inside it unchanged.
    """
    def __init__(self, on_activate, on_open, on_menu,
                 collapsed=None, on_collapse=None, parent=None):
        super().__init__(parent)
        self._on_activate = on_activate    # callable(stage_id) — show on canvas
        self._on_open = on_open            # callable(stage_id) — pop out panel
        self._on_menu = on_menu            # callable(stage_id, global_pos)
        self._on_collapse = on_collapse    # callable(group, collapsed) — persist
        self._rows = {}                    # stage_id -> _OutlineRow
        self._group_of = {}                # stage_id -> group name
        self._headers = {}                 # group -> _OutlineGroupHeader
        self._bodies = {}                  # group -> body QWidget (collapsible)
        self._collapsed = set(collapsed or ())
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        cap = QLabel("DEFAULT PIPELINE")
        cf = cap.font()
        cf.setPointSize(8)
        cf.setBold(True)
        cf.setLetterSpacing(QFont.AbsoluteSpacing, 1.2)
        cap.setFont(cf)
        cap.setStyleSheet("color:#5c7186;padding:2px 2px 0 4px;")
        v.addWidget(cap)

        inner = QWidget()
        iv = QVBoxLayout(inner)
        iv.setAlignment(Qt.AlignTop)
        iv.setContentsMargins(4, 2, 6, 6)
        iv.setSpacing(3)
        body_box = None
        last_group = None
        for spec in STAGES:
            if spec.get("legacy"):          # retired: readable on old cards only
                continue
            g = spec.get("group", "")
            if g != last_group:
                header = _OutlineGroupHeader(g, self)
                iv.addWidget(header)
                self._headers[g] = header
                body = QWidget()
                body_box = QVBoxLayout(body)
                body_box.setContentsMargins(4, 0, 0, 2)
                body_box.setSpacing(3)
                iv.addWidget(body)
                self._bodies[g] = body
                last_group = g
            row = _OutlineRow(spec["id"],
                              stage_title(spec["id"], spec.get("label", spec["id"])),
                              self)
            body_box.addWidget(row)
            self._rows[spec["id"]] = row
            self._group_of[spec["id"]] = g
        for g in self._collapsed & set(self._bodies):
            self._bodies[g].setVisible(False)
            self._headers[g].set_collapsed(True)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        scroll.setWidget(inner)
        v.addWidget(scroll, 1)
        hint = QLabel("click = show · 2×click = open · drag → canvas")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#3f4d5a;font-size:9px;padding:0 4px 2px 4px;")
        v.addWidget(hint)

    def _toggle_group(self, group):
        body = self._bodies.get(group)
        if body is None:
            return
        collapsed = body.isVisible()          # visible -> about to collapse
        body.setVisible(not collapsed)
        header = self._headers.get(group)
        if header is not None:
            header.set_collapsed(collapsed)
        if collapsed:
            self._collapsed.add(group)
        else:
            self._collapsed.discard(group)
        if self._on_collapse is not None:
            try:
                self._on_collapse(group, collapsed)
            except Exception:
                pass

    def update_stages(self, tmpl_nodes, active=None):
        """Refresh the per-stage status lines + per-category roll-ups from the
        canvas repaint's template nodes. Updates in place — the outline repaints
        every few seconds while a job runs, and rebuilding rows would flicker
        and drop scroll position."""
        act = active or {}
        totals = {}                       # group -> [n_jobs, running]
        for n in tmpl_nodes or []:
            row = self._rows.get(n.get("stage_id"))
            if row is None:
                continue
            running = card_is_running(n, act)
            row.set_status(n.get("n_jobs", 0), n.get("on_disk"),
                           n.get("disk_label", ""), running)
            g = self._group_of.get(n.get("stage_id"), "")
            t = totals.setdefault(g, [0, False])
            t[0] += n.get("n_jobs", 0) or 0
            t[1] = t[1] or running
        for g, (k, running) in totals.items():
            header = self._headers.get(g)
            if header is not None:
                header.set_rollup(k, running)

    def _activate(self, stage_id):
        try:
            self._on_activate(stage_id)
        except Exception:
            pass

    def _open(self, stage_id):
        try:
            self._on_open(stage_id)
        except Exception:
            pass

    def _menu(self, stage_id, global_pos):
        if self._on_menu is None:
            return
        try:
            self._on_menu(stage_id, global_pos)
        except Exception:
            pass


class JobCanvas(QWidget):
    # The previous repaint's nodes, so the next one can tell which cards are
    # NEW and scroll to them. None means "nothing painted yet" — on the first
    # paint every card is new, and snapping then would be arbitrary.
    _node_index = None

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
        # Set by the main window: callable(template_nodes, active) that feeds the
        # pinned pipeline outline each repaint (the rail's replacement).
        self.on_outline = None
        # Set by the main window: callable() fired whenever what the viewport
        # shows changes (pan, zoom, repaint) — the pop-out panels re-tether to
        # their cards through it.
        self.on_view_changed = None
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
        zoom_out = QPushButton("−")
        zoom_out.setToolTip("Zoom out  (Ctrl+−, or Ctrl+wheel)")
        zoom_out.setFixedWidth(28)
        zoom_out.clicked.connect(lambda: self.view.zoom_by(1 / 1.25))
        tools.addWidget(zoom_out)

        # The readout doubles as the affordance: without a number on screen,
        # nobody discovers that Ctrl+wheel zooms.
        self.zoom_lbl = QLabel("100%")
        self.zoom_lbl.setFixedWidth(44)
        self.zoom_lbl.setAlignment(Qt.AlignCenter)
        self.zoom_lbl.setToolTip("Ctrl+wheel to zoom, Ctrl+0 for 1:1")
        self.zoom_lbl.setStyleSheet("color:#9a9a9a;font-size:11px;")
        tools.addWidget(self.zoom_lbl)

        zoom_in = QPushButton("+")
        zoom_in.setToolTip("Zoom in  (Ctrl++, or Ctrl+wheel)")
        zoom_in.setFixedWidth(28)
        zoom_in.clicked.connect(lambda: self.view.zoom_by(1.25))
        tools.addWidget(zoom_in)

        fit = QPushButton("Fit")
        fit.setToolTip("Zoom to fit the whole workflow")
        fit.clicked.connect(self.fit_all)
        tools.addWidget(fit)
        reset = QPushButton("1:1")
        reset.setToolTip("Reset zoom to 100%  (Ctrl+0)")
        reset.clicked.connect(lambda: self.view.zoom_to(1.0))
        tools.addWidget(reset)
        self.view.set_zoom_handler(self._show_zoom)

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

        hint = QLabel("drag or ✋ to pan · ← → arrows · Ctrl+wheel zooms")
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
        self.view.set_menu_handler(self._canvas_menu)
        # Any scroll re-tethers the pop-out panels (zoom reports through
        # _show_zoom, and zooming moves the scrollbars anyway).
        for sb in (self.view.horizontalScrollBar(),
                   self.view.verticalScrollBar()):
            sb.valueChanged.connect(self._view_moved)

    def _view_moved(self, *_):
        cb = self.on_view_changed
        if cb is None:
            return
        try:
            cb()
        except Exception:
            pass

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
        # Deferred: this is called from the card's own mouseReleaseEvent, and
        # refresh() does scene.clear() — destroying the item that is still the
        # scene's mouse grabber, mid-delivery of its own event (intermittent
        # segfault on drag-release).
        QTimer.singleShot(0, self.refresh)

    def _connect_cards(self, src, dst):
        """Wire dst to read src's output — the gesture form of '⇄ Set input'.

        Refusals are explained rather than silent: a drag that does nothing and
        says nothing reads as a broken canvas."""
        root = self._root_getter()
        store = load_jobs(root)
        sid, did = src.get("id"), dst.get("id")
        jobs = store.get("jobs") or {}
        if not sid or not did or sid not in jobs or did not in jobs:
            return
        if would_cycle(store, did, sid):
            self._log(f"{sid} → {did} would make a loop: {did} already feeds "
                      f"{sid}, directly or through the chain.", "warn")
            return
        if dst.get("status") == "running":
            self._log(f"{did} is running — its inputs are in use. Kill it first, "
                      f"or wire a copy.", "warn")
            return
        if set_job_parent(root, did, sid) is None:
            self._log(f"Could not wire {sid} → {did}.", "warn")
            return
        pending = dst.get("status") in ("building", "queued")
        self._log(f"{did} now reads {sid}"
                  + (" — it picks up those outputs when it runs."
                     if pending else
                     " (already ran; the edge records the lineage, it does not "
                     "re-run it)."), "ok")
        self.refresh()

    # ---- canvas annotations ----
    def _note_moved(self, note, x, y):
        try:
            update_note(self._root_getter(), note["id"], x=float(x), y=float(y))
        except Exception:
            return
        # Same deferral as _card_moved: refresh() clears the scene, and this runs
        # inside the moved item's own event.
        QTimer.singleShot(0, self.refresh)

    def _add_note(self, kind, x, y):
        try:
            add_note(self._root_getter(), kind, x, y,
                     text="Frame — double-click to name it" if kind == "frame"
                          else "Note — double-click to edit")
        except Exception:
            return
        self.refresh()

    def _edit_note(self, note):
        text, ok = QInputDialog.getMultiLineText(
            self, "Edit annotation",
            "Text (this is a note — nothing here affects what runs):",
            str(note.get("text", "")))
        if not ok:
            return
        update_note(self._root_getter(), note["id"], text=text)
        self.refresh()

    def _note_menu(self, note, global_pos):
        menu = QMenu(self)
        menu.addAction("Edit text…", lambda: self._edit_note(note))
        colours = menu.addMenu("Colour")
        for c in NOTE_COLOURS:
            colours.addAction(c, lambda _=False, c=c: (
                update_note(self._root_getter(), note["id"], colour=c),
                self.refresh()))
        if note.get("kind") == "frame":
            sizes = menu.addMenu("Resize")
            for label, (w, h) in (("Small", (380, 240)), ("Medium", (520, 340)),
                                  ("Large", (760, 520)), ("Wide", (1100, 380))):
                sizes.addAction(label, lambda _=False, w=w, h=h: (
                    update_note(self._root_getter(), note["id"], w=w, h=h),
                    self.refresh()))
        menu.addSeparator()
        menu.addAction("Delete", lambda: (
            delete_note(self._root_getter(), note["id"]), self.refresh()))
        menu.exec(global_pos)

    def _canvas_menu(self, scene_x, scene_y, global_pos):
        """Right-click on empty canvas: the only place annotations are made, so
        they land where the pointer is rather than at a fixed corner."""
        menu = QMenu(self)
        menu.addAction("＋ Note here",
                       lambda: self._add_note("note", scene_x, scene_y))
        menu.addAction("＋ Branch frame here",
                       lambda: self._add_note("frame", scene_x, scene_y))
        menu.addSeparator()
        menu.addAction("Auto-arrange cards", self._reset_positions)
        menu.exec(global_pos)

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
            # fitInView sets the transform directly, so the readout has to be
            # told — otherwise it still claims whatever it said before Fit.
            self._show_zoom(self.view.zoom_level())

    def _show_zoom(self, level):
        self.zoom_lbl.setText(f"{level * 100:.0f}%")
        self._view_moved()          # zoom changes where cards sit on screen

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
        # Kept so a card can be scrolled to AFTER the repaint that creates it.
        # A newly adopted or built job lands rightmost in its stage's row, which
        # on a wide graph is routinely off-screen — the card appeared, but not
        # anywhere the user was looking.
        # Cards this repaint introduced. Asked HERE rather than at the eight
        # places that create jobs: adoption, build-downstream, fork, queue and
        # drag-drop all land here, and a rule the canvas owns cannot be missed
        # by a creation path added later. It also keeps the app's job-creation
        # calls free of a canvas method that a test double would have to grow.
        fresh = newly_added(self._node_index, index)
        self._node_index = index
        if fresh:
            # After the scene is built, not during: centring on a card the
            # paint has not placed yet scrolls to where it used to be.
            QTimer.singleShot(0, lambda i=fresh: self.reveal(i))

        # Annotations. Frames are painted before anything else so they read as
        # ground the branch stands on; notes go on last, above the cards, since
        # a sticky hidden behind a card is a sticky nobody reads.
        try:
            frames, stickies = notes_for_canvas(store)
        except Exception:
            frames, stickies = [], []
        for fr in frames:
            self.scene.addItem(_NoteItem(fr, self))

        # The default pipeline is NOT painted in the scene any more: it lives in
        # the pinned outline widget down the window's left edge (_PipelineOutline),
        # which is always on screen however far the work scrolls or zooms. The
        # layout still computes the template nodes — they carry the per-stage
        # status the outline shows — but the scene draws only real work.
        tmpl = [n for n in nodes if n.get("is_template")]
        if self.on_outline is not None:
            try:
                self.on_outline(tmpl, self._active)
            except Exception:
                pass

        # CATEGORY BANDS. The canvas is one row per stage, and stages come in
        # groups — so the groups are horizontal stripes: Warp processing here,
        # RELION there, M below it, the membrane/IsoNet branch at the bottom.
        # Quiet dividing lines + a slate caption make those stripes readable;
        # the auto-layout already drops every job into its stage's row, so a
        # new card pops into its category's band by construction.
        real = [n for n in nodes if not n.get("is_template")]
        if tmpl:
            x0 = RAIL_W - 30
            x1 = max([n["x"] + n["w"] for n in real]
                     or [RAIL_W + 4 * (CARD_W + GAP_X)]) + 90
            bands = {}
            for t in tmpl:
                g = t.get("group", "")
                y0, y1 = bands.get(g, (t["y"], t["y"]))
                bands[g] = (min(y0, t["y"]), max(y1, t["y"] + t["h"]))
            sep_pen = QPen(QColor("#1f2830"))
            sep_pen.setWidth(1)
            first_top = min(b[0] for b in bands.values())
            for g, (top, _bot) in sorted(bands.items(), key=lambda kv: kv[1][0]):
                ly = top - GAP_Y + 8          # in the gutter above the band's rows
                if top > first_top:           # no divider above the very first band
                    ln = self.scene.addLine(x0, ly, x1, ly, sep_pen)
                    ln.setZValue(-17)
                cap = QGraphicsSimpleTextItem(str(g).upper())
                cap.setBrush(QColor("#3f4d5a"))
                cf = QFont()
                cf.setPointSize(8)
                cf.setBold(True)
                cap.setFont(cf)
                cap.setPos(x0 + 2, ly + 3)
                cap.setZValue(-17)
                self.scene.addItem(cap)

        edge_pen = QPen(QColor("#4a4a4a"))
        edge_pen.setWidth(2)
        # Edge colour says what the line MEANS, which is the whole point of
        # having only one kind of solid line on this canvas:
        #   grey   — data moved: the downstream job has run against it
        #   purple — real wiring into a job that has NOT run yet
        # (dashed, in any colour, is a hand-drawn annotation and carries nothing)
        pending_pen = QPen(QColor("#8b6ed6"))
        pending_pen.setWidth(2)
        for src, dst in edges:
            a, b = index.get(src), index.get(dst)
            if not a or not b:
                continue
            if a.get("is_template") or b.get("is_template"):
                continue          # the rail chain is the outline's, not the scene's
            if b.get("status") in ("building", "queued"):
                pen = pending_pen
            else:
                pen = edge_pen
            ln = self.scene.addLine(
                a["x"] + a["w"] / 2, a["y"] + a["h"],
                b["x"] + b["w"] / 2, b["y"], pen)
            ln.setZValue(-10)

        for n in nodes:
            if n.get("is_template"):
                continue          # drawn as the pinned outline instead
            self._add_card(n)
        for st_note in stickies:
            self.scene.addItem(_NoteItem(st_note, self))
        # A brand-new project draws nothing (the default pipeline is the outline
        # widget now) — say where to start instead of presenting a void.
        if not any(not n.get("is_template") for n in nodes):
            tip = QGraphicsSimpleTextItem(
                "Nothing here yet — drag a step from the pipeline outline "
                "(left) onto the canvas, or double-click one to open its "
                "builder.")
            tip.setBrush(QColor("#5c6b7a"))
            tf = QFont()
            tf.setPointSize(11)
            tip.setFont(tf)
            tip.setPos(RAIL_W, 20)
            self.scene.addItem(tip)
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
        self._view_moved()          # a repaint can move cards (drag, re-layout)

    def _add_card(self, n):
        if n.get("is_template"):
            return          # the default pipeline lives in the pinned outline
        ghost = n["is_ghost"]
        orphan = n.get("is_orphan", False)
        # Running if this node's own job is live, OR a trunk run (▶ Run — it has no
        # card of its own) is executing this stage, so the stage's card lights up too.
        # Light up ONLY the card that is actually running. The stage_id fallback
        # exists for TRUNK runs (▶ Run), which have no job record — but it must hit
        # the stage's ghost card, never every sibling job of that stage
        # (that lit all four Extract cards amber at once).
        act = self._active or {}
        running = card_is_running(n, act)
        if n.get("is_discovered"):
            # Green, because it really did run — dashed, because there is no job
            # record behind it.
            fill, border = _CARD_STYLE["completed"]
        else:
            fill, border = _CARD_STYLE.get("running" if running else n["status"],
                                           _CARD_STYLE["ghost"])
        item = _CardItem(n, self)
        item.setBrush(QBrush(QColor(fill)))
        pen = QPen(QColor(border))
        pen.setWidth(3 if running else 2)   # running gets a heavier outline
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
        text(n.get("group", ""), 11, 6, 8, "#6f6f6f", maxw=inner_w)
        text(n.get("title", n["label"]), 11, 20, 11,
             "#8a8a8a" if ghost else "#ececec", bold=True, maxw=inner_w)
        text(n.get("subtitle", n["stage_id"]), 11, 40, 8, "#6f6f6f", maxw=inner_w)
        if running:
            prog = act.get("progress", "")
            sub = "▶ running" + (f" · {prog}" if prog else "")
            sub_colour = "#f0a92a"
        elif n.get("on_disk"):                 # ran outside the app — no job record
            sub = "✓ ran (no job record)" + (
                f" · {n['disk_label']}" if n.get("disk_label") else "")
            sub_colour = "#27ae60"
        elif ghost:
            sub = "not built"
            sub_colour = "#7d7d7d"
        else:
            st = summary_text(n["summary"])
            sub = n["status"] + (f" · {st}" if st else "")
            sub_colour = "#7d7d7d"
            # ✋ = a verdict someone set, not one the run reported. The card's
            # colour follows the mark (that is the point of marking it), so the
            # only honest thing is to say the colour is an opinion.
            if n.get("manual"):
                sub = "✋ " + sub + " (by hand)"
                sub_colour = "#c9a227"
        text(sub, 11, 56, 9, sub_colour, maxw=inner_w)
        if not ghost and not orphan:
            text(n["id"], n["w"] - 42, 6, 8, "#9ec5ff")

        # Pop-out arrow (only if the canvas has a popout handler)
        if self._on_details is not None:
            self.scene.addItem(_PopoutChip(n, self, n["x"] + n["w"] - 28,
                                           n["y"] + n["h"] - 24))

        # Output port. Only real jobs have one: a ghost is a template with no
        # output to feed anything, and the rail is fixed furniture.
        if (not ghost and not orphan and not n.get("is_template")
                and n.get("id") and not self.locked):
            self.scene.addItem(_ConnectorHandle(
                n, self, n["x"] + 10, n["y"] + n["h"] - 20))

    def reveal(self, node_id, force=False):
        """Scroll the card into view, if it is not already there.

        Only scrolls when it has to: this runs after every job creation, and a
        canvas that jumps when the card was already visible is worse than one
        that never moves. Returns True if it scrolled."""
        n = (self._node_index or {}).get(node_id)
        if not n:
            return False
        rect = QRectF(n["x"] - 40, n["y"] - 40, n["w"] + 80, n["h"] + 80)
        if not force:
            seen = self.view.mapToScene(
                self.view.viewport().rect()).boundingRect()
            if seen.contains(rect):
                return False
        self.view.centerOn(n["x"] + n["w"] / 2, n["y"] + n["h"] / 2)
        return True

    def _pick(self, node):
        # Pass the whole NODE, not just its stage. Clicking a card and pressing Run
        # used to launch a TRUNK run of that stage — the card stayed Building while
        # its own command ran untracked beside it.
        try:
            self._on_pick(node)
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


class _ConsoleTab(QWidget):
    """One extra shell tab in the terminal dock.

    Each tab is its own scratch console: commands run in the project root on
    their own QProcess (never the shared job runner, so they cannot collide
    with a live job), and the output stays in the tab. For `<tool> --help`,
    ls, head, the check scripts — several lines of enquiry side by side.
    """

    def __init__(self, root_getter, parent=None):
        super().__init__(parent)
        self._root_getter = root_getter
        self._hist = []
        self._pos = 0
        self._procs = []                 # keep refs (GC would kill a QProcess)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 4, 0, 0)
        v.setSpacing(4)
        self.out = QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setMaximumBlockCount(8000)
        self.out.setTextInteractionFlags(Qt.TextSelectableByMouse
                                         | Qt.TextSelectableByKeyboard)
        self.out.setStyleSheet(f"background:#0c0c0c;color:#cfcfcf;border:none;"
                               f"font-family:{MONO};font-size:11px;"
                               f"selection-background-color:#2d4a6b;")
        v.addWidget(self.out, 1)
        row = QHBoxLayout()
        row.setSpacing(6)
        prompt = QLabel("❯")
        prompt.setStyleSheet("color:#7fb4ff;font-weight:700;")
        row.addWidget(prompt)
        self.input = QLineEdit()
        self.input.setPlaceholderText("run in the project root (↑/↓ history)")
        self.input.setStyleSheet(f"font-family:{MONO};font-size:11px;")
        self.input.returnPressed.connect(self._run)
        self.input.installEventFilter(self)
        row.addWidget(self.input, 1)
        rw = QWidget()
        rw.setLayout(row)
        v.addWidget(rw)

    def eventFilter(self, obj, ev):
        """↑/↓ recall previous commands (same behaviour as the log tab's input)."""
        if obj is self.input and ev.type() == QEvent.KeyPress and self._hist:
            key = ev.key()
            if key in (Qt.Key_Up, Qt.Key_Down):
                self._pos += -1 if key == Qt.Key_Up else 1
                self._pos = max(0, min(len(self._hist), self._pos))
                self.input.setText(self._hist[self._pos]
                                   if self._pos < len(self._hist) else "")
                return True
        return super().eventFilter(obj, ev)

    def _run(self):
        cmd = self.input.text().strip()
        if not cmd:
            return
        root = self._root_getter()
        if not isinstance(root, (str, os.PathLike)):
            self.out.appendPlainText("(set a project root first)")
            return
        self._hist.append(cmd)
        self._pos = len(self._hist)
        self.input.clear()
        self.out.appendPlainText(f"❯ {cmd}")
        proc = QProcess(self)
        proc.setWorkingDirectory(str(root))
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(lambda p=proc: self._out(p))
        proc.finished.connect(lambda code, _s, p=proc: self._done(p, code))
        self._procs.append(proc)
        proc.start("bash", ["-lc", cmd])

    def _out(self, proc):
        data = bytes(proc.readAllStandardOutput()).decode("utf-8", "replace")
        for line in data.splitlines():
            self.out.appendPlainText(clean_stream_line(line))

    def _done(self, proc, code):
        self.out.appendPlainText(f"❯ exit {code}")
        try:
            self._procs.remove(proc)
        except ValueError:
            pass


# ===========================================================================
# Per-card pop-out panel (builder · details · outputs)
# ===========================================================================
MIME_OUTPUT = "application/x-tomogration-output"


class _DragPathRow(QWidget):
    """One path in a popout's Outputs tab: drag it into any path field of any
    open builder to use it there (the drop side coerces file/folder and
    relative/absolute to what that parameter wants), or open it directly."""

    def __init__(self, app, rel, note="", job_id="", is_dir=True, small=False):
        super().__init__(None)
        self._app = app
        self._rel = rel
        self._job_id = job_id
        self._is_dir = is_dir
        self._press_at = None
        self.setCursor(Qt.OpenHandCursor)
        tip = (f"{rel}\nDrag into an input field of another job's builder "
               f"to use this path there.")
        if note:
            tip += f"\n({note})"
        self.setToolTip(tip)
        row = QHBoxLayout(self)
        row.setContentsMargins(14 if small else 0, 0, 0, 0)
        row.setSpacing(4)
        grip = QLabel("⠿")
        grip.setStyleSheet("color:#5c6b7a;font-size:11px;")
        row.addWidget(grip)
        icon = "📂" if is_dir else "📄"
        lab = QLabel(f"{icon}  {rel}")
        lab.setStyleSheet("color:#b9c6d2;font-size:10px;" if small
                          else "color:#d6d6d6;font-size:11px;")
        lab.setWordWrap(True)
        row.addWidget(lab, 1)
        if is_dir:
            b = QPushButton("open")
            b.setFixedWidth(44)
            b.setStyleSheet("font-size:10px;padding:1px 4px;")
            b.setToolTip(f"Open {rel} in the file manager")
            b.clicked.connect(lambda _=False, r=rel: app._open_dir(r))
            row.addWidget(b)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._press_at = ev.position().toPoint() if hasattr(ev, "position") \
                else ev.pos()

    def mouseMoveEvent(self, ev):
        if self._press_at is None:
            return
        here = ev.position().toPoint() if hasattr(ev, "position") else ev.pos()
        if (here - self._press_at).manhattanLength() < 8:
            return
        self._press_at = None
        md = QMimeData()
        payload = json.dumps({"job_id": self._job_id, "path": self._rel,
                              "is_dir": self._is_dir})
        md.setData(MIME_OUTPUT, payload.encode("utf-8"))
        md.setText(self._rel)          # plain text so any editor accepts it too
        drag = QDrag(self)
        drag.setMimeData(md)
        drag.exec(Qt.DropAction.CopyAction)

    def mouseReleaseEvent(self, _ev):
        self._press_at = None


class _PathDropFilter(QObject):
    """Makes a builder's path field a smart drop target for output rows.

    QLineEdit would happily accept the plain-text drop on its own — inserted at
    the cursor, mid-text, verbatim. What a parameter wants is the WHOLE value,
    coerced: a folder param handed a file takes the containing folder, an
    abs_paths param gets an absolute path, everything else goes project-relative
    so commands stay portable. The coercion lives in app._smart_path_value.
    """

    def __init__(self, edit, p, app):
        super().__init__(edit)
        self._edit = edit
        self._p = p
        self._app = app
        edit.setAcceptDrops(True)
        edit.installEventFilter(self)

    def _payload(self, md):
        if md.hasFormat(MIME_OUTPUT):
            try:
                return json.loads(bytes(md.data(MIME_OUTPUT)).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return None
        if md.hasText():
            t = md.text().strip()
            # Only a single path-looking line — a paragraph of dropped prose
            # should keep QLineEdit's normal insert behaviour.
            if t and "\n" not in t and (("/" in t) or ("." in t)):
                return {"path": t, "is_dir": None, "job_id": ""}
        return None

    def eventFilter(self, obj, ev):
        t = ev.type()
        if t in (QEvent.DragEnter, QEvent.DragMove):
            if self._payload(ev.mimeData()) is not None:
                ev.acceptProposedAction()
                return True
            return False
        if t == QEvent.Drop:
            pay = self._payload(ev.mimeData())
            if pay is None:
                return False
            value, note = self._app._smart_path_value(self._p, pay)
            if value is None:
                ev.acceptProposedAction()
                return True
            self._edit.setText(value)   # fires textChanged -> command rebuild
            src = pay.get("job_id") or "dropped path"
            self._app._log(f"{src} → {param_title(self._p)}: {value}"
                           + (f"  ({note})" if note else ""), "info")
            ev.acceptProposedAction()
            return True
        return False


class JobPopout(QDialog):
    """A card's own floating panel: builder · details · outputs.

    Non-modal and persistent — it stays open (and live) while you work the
    canvas, and several can be open at once, which is what makes dragging one
    job's output into another job's input field possible at all. Qt.Tool keeps
    it above the main window without claiming a taskbar entry of its own.
    All content is populated by the main window (it owns the project state);
    this class is the shell: header, tabs, refresh plumbing.
    """

    TABS = ("builder", "details", "outputs")

    def __init__(self, app, node):
        super().__init__(app)
        self.app = app
        self.node = dict(node)
        self.node_id = node.get("id") or f"ghost:{node.get('stage_id')}"
        self.ctx = {}          # this panel's builder form state (its own current)
        self._last_status = None
        self._rebuild_queued = False
        # TETHERED to its card: the panel follows the card across pans, zooms
        # and drags. None = hug the card's right edge; once the user moves the
        # panel, the offset they chose is kept relative to the card instead.
        self.tether_offset = None
        self._prog_move = False
        self.setWindowFlag(Qt.Tool, True)
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)   # closed = gone, not parked
        self.setSizeGripEnabled(True)
        self.resize(440, 580)
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(6)
        self.head = QLabel("")
        self.head.setWordWrap(True)
        self.head.setStyleSheet("font-size:13px;font-weight:700;color:#ececec;")
        v.addWidget(self.head)
        self.meta = QLabel("")
        self.meta.setWordWrap(True)
        self.meta.setStyleSheet("color:#9a9a9a;font-size:10px;")
        v.addWidget(self.meta)
        self.tabs = QTabWidget()
        self.boxes = {}
        if not node.get("is_orphan"):
            self.boxes["builder"] = self._make_tab("Builder")
        self.boxes["details"] = self._make_tab("Details")
        if not node.get("is_orphan"):
            self.boxes["outputs"] = self._make_tab("Outputs")
        v.addWidget(self.tabs, 1)
        self.refresh(force=True)

    def _make_tab(self, title):
        box = QVBoxLayout()
        box.setAlignment(Qt.AlignTop)
        box.setContentsMargins(8, 6, 12, 8)
        box.setSpacing(4)
        inner = QWidget()
        inner.setLayout(box)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(inner)
        self.tabs.addTab(scroll, title)
        return box

    def show_tab(self, name):
        order = [k for k in self.TABS if k in self.boxes]
        if name in order:
            self.tabs.setCurrentIndex(order.index(name))

    def refresh(self, node=None, force=False):
        """Bring the panel up to date with the store. Cheap when nothing changed:
        the header always updates, but the tabs — including a builder form the
        user may be mid-edit in — are rebuilt only when the JOB's status actually
        moved (queued -> running -> completed …), or on demand."""
        if node is not None:
            self.node = dict(node)
        jid = self.node.get("id")
        job = None
        if jid and not self.node.get("is_ghost") and not self.node.get("is_orphan"):
            job = (load_jobs(self.app.project_root).get("jobs") or {}).get(jid)
            if job is None:                     # deleted while the panel was open
                self.close()
                return
            self.node["status"] = job.get("status", self.node.get("status"))
            self.node["summary"] = job.get("summary", self.node.get("summary")) or {}
            self.node["manual"] = bool(job.get("manual_status"))
        self.app._popout_header(self, job)
        status = self.node.get("status")
        if not force and status == self._last_status:
            return
        self._last_status = status
        # DEFERRED. This runs from buttons that live in the very tabs being
        # rebuilt (Save & run → store change → refresh): clearing the box then
        # would destroy the clicked button mid-delivery of its own signal — the
        # same intermittent segfault the canvas guards against everywhere.
        if not self._rebuild_queued:
            self._rebuild_queued = True
            QTimer.singleShot(0, self._rebuild_tabs)

    def _rebuild_tabs(self):
        self._rebuild_queued = False
        for box in self.boxes.values():
            self._clear(box)
        if "builder" in self.boxes:
            self.app._populate_popout_builder(self.boxes["builder"], self.node,
                                              self.ctx)
        self.app._populate_details_box(self.boxes["details"], self.node, self)
        if "outputs" in self.boxes:
            self.app._populate_outputs_box(self.boxes["outputs"], self.node)

    @staticmethod
    def _clear(layout):
        # deleteLater, not just re-parent: these boxes are rebuilt for the life
        # of the panel, and merely orphaned widgets would pile up.
        while layout.count():
            it = layout.takeAt(0)
            w = it.widget() if it is not None else None
            if w is not None:
                w.deleteLater()

    def moveEvent(self, ev):
        """A move the USER made records their chosen offset from the card, so
        re-tethering preserves it ('my panel to the left of the card' stays to
        the left of the card). Programmatic moves are guarded out — they ARE
        the tether and must not rewrite it."""
        super().moveEvent(ev)
        if self._prog_move:
            return
        try:
            base = self.app._popout_anchor(self)
        except Exception:
            base = None
        if base is not None:
            self.tether_offset = self.pos() - base

    def closeEvent(self, ev):
        self.app._popouts.pop(self.node_id, None)
        super().closeEvent(ev)


# ===========================================================================
# Main window
# ===========================================================================
def _wants_directory(p):
    """True when a path param means a FOLDER. Inferred, because the params
    predate any explicit path_kind field: a '_dir' name or a title saying
    'folder' is what every such param in the app already looks like."""
    name = str(p.get("name", "")).lower()
    title = str(p.get("title", "")).lower()
    return (name.endswith("_dir") or name in ("out", "root", "work_dir")
            or "folder" in title)


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
        # Per-card pop-out panels (builder · details · outputs), keyed by node
        # id. Made BEFORE any widget that can call _open_job_popout exists.
        # Non-modal and plural: dragging an output from one job into another
        # job's input field needs both panels on screen at once.
        self._popouts = {}

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
        self._job_finalize_hook = None   # one-shot post-run hook (crYOLO convert)
        self._builder_bindings = {}   # stage_id -> its editable card (sticky)
        # One-shot: a clicked card's recorded params, overlaid on the next form
        # build WITHOUT touching the per-stage param store.
        self._form_job_params = None
        # One-shot: describes a FINISHED job whose parameters the form is showing,
        # so the banner can say "this is history" rather than letting it read as
        # something you are about to run.
        self._showing_job = None
        # Debounce disk writes: a single-shot timer flushes _param_store to config
        # ~0.6s after the last edit (so typing doesn't hammer the JSON).
        self._persist_timer = QTimer(self)
        self._persist_timer.setSingleShot(True)
        self._persist_timer.setInterval(600)
        self._persist_timer.timeout.connect(self._persist_param_store)
        # Same debounce for the canvas splitter: splitterMoved fires per pixel
        # of drag, and the point is only to remember where the user LEFT it.
        self._split_save_timer = QTimer(self)
        self._split_save_timer.setSingleShot(True)
        self._split_save_timer.setInterval(600)
        self._split_save_timer.timeout.connect(self._flush_split_sizes)
        # Periodically re-scan for on-disk pick sets made outside the app (direct
        # terminal use); only redraw the canvas when the set actually changes.
        self._last_orphan_keys = None
        self._orphan_cache = None        # (monotonic_ts, root, [orphans])
        # A 'running' job cannot survive the process that launched it — clear any
        # left over from a crash/force-quit, or the queue thinks it is still busy.
        # (This used to call reconcile_running(root) — a NameError the bare
        # except swallowed, so crash recovery never actually ran.)
        try:
            self._stale_jobs = reconcile_running(project_root)
        except OSError:
            self._stale_jobs = []
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
        self._polarity_result = None
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
        # Two deliberate exceptions: the pipeline outline (left) and the terminal
        # (right) are NOT cards — they are pinned to the window edges and styled
        # flat, so they read as part of the window itself rather than panels
        # floating over it.
        self.setStyleSheet("""
            QFrame[card="true"] { background:#191919; border:1px solid #333; border-radius:8px; }
            QFrame#docsCard  { background:#1d1d1d; }
            QFrame#dirCard, QFrame#queueCard { background:#0e0e0e; }
            QFrame[card="true"] QScrollArea { border:none; background:transparent; }
            QFrame#outlineDock { background:#0b0f13; border:none;
                                 border-right:1px solid #22303c; border-radius:0; }
            QFrame#termDock { background:#101010; border:none;
                              border-left:1px solid #242424; border-radius:0; }
            QFrame#termDock QTabWidget::pane { border:none; background:#101010; }
            QFrame#termDock QTabBar::tab {
                background:#101010; color:#6f6f6f; padding:3px 10px;
                border:none; border-bottom:2px solid transparent; font-size:10px; }
            QFrame#termDock QTabBar::tab:selected {
                color:#c9c9c9; border-bottom:2px solid #3d6fa5; }
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
                                on_details=self._open_job_popout,
                                on_menu=self._card_menu,
                                on_orphans=self._discover_orphans,
                                on_active=self._active_info,
                                on_add_stage=self._add_stage_from_palette)
        self.job_stack.addWidget(
            self._panel("canvasCard", "Workflow graph", self.canvas))
        # Two-way lock sync: the View-menu item drives the toolbar button, so the
        # toolbar must drive the menu check back or unlocking via the toolbar
        # leaves the menu asserting the opposite state. (setChecked does not
        # re-emit triggered, so this cannot loop.)
        self.canvas.lock_btn.toggled.connect(self._act_lock.setChecked)
        self._build_align_and_command()   # sets self.align_list_card + self.command_card
        # The default pipeline, pinned down the left window edge (replaces both
        # the in-scene template rail and the right-column job builder as the way
        # into a stage). Category collapse state is per-user, not per-project:
        # which sections you keep folded is a reading habit, so it lives in the
        # same config as the view mode.
        self.outline = _PipelineOutline(
            self._outline_show, self._outline_open, self._outline_menu,
            collapsed=self._load_config().get("outline_collapsed", []),
            on_collapse=self._save_outline_collapse)
        self.canvas.on_outline = self.outline.update_stages
        # Panels follow their cards: any pan/zoom/repaint re-tethers them.
        self.canvas.on_view_changed = self._retether_popouts
        self.outline_card = self._dock("outlineDock", self.outline)
        self.outline_card.setMinimumWidth(168)
        self.outline_card.setMaximumWidth(300)
        self.dir_card = self._panel("dirCard", "Directory overview",
                                    self._build_directory_overview())
        self.terminal_card = self._dock("termDock", self._build_terminal_panel())
        self.queue_card = self._panel("queueCard", "Jobs queue",
                                      self._build_queue_panel())
        # Floor widths: the command card so the splitter can never squeeze the
        # command text into truncation (lists view), the terminal narrower — it is
        # a full-height side rail now, not a reading pane.
        self.command_card.setMinimumWidth(400)
        self.terminal_card.setMinimumWidth(280)
        # Stash keeps panels parented (and hidden) while they're not in the live
        # layout — a parentless shown QWidget would pop up as its own window.
        self._stash = QWidget()
        self._stash.hide()
        self._panels = [self.docs_card, self.job_stack, self.align_list_card,
                        self.command_card, self.outline_card, self.dir_card,
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

        # Canvas is the primary view now (outline | canvas | terminal, with the
        # builder in per-card pop-outs); lists stays available and sticky.
        start_mode = "lists" if self._load_config().get("view_mode") == "lists" else "canvas"
        self._apply_layout(start_mode)

        self._refresh_status_dots()
        self._select_stage(self._stage_by_id("ts_reconstruct") or STAGES[0])
        self._save_config({**self._load_config(), "last_root": self.project_root})
        if self._stale_jobs:
            self._log(f"Recovered from an interrupted session: "
                      f"{', '.join(self._stale_jobs)} marked failed (was "
                      f"'running' when the app last closed).", "warning")

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
    def _canvas_pick(self, target):
        """Open a card (or a bare stage id) that was clicked on the canvas.

        In canvas view the job builder has no column of its own any more — it
        lives in each card's pop-out panel, so a click opens that panel (which
        binds its Builder tab to the job). In lists view the singleton builder
        is on screen, so a bare stage id still lands there.
        """
        if getattr(self, "_view_mode", "lists") == "canvas":
            self._open_job_popout(target)
            return
        if isinstance(target, dict):
            node = target
            stage_id = node.get("stage_id")
            jid = node.get("id")
            # Ghost/template/discovered cards have no job behind them.
            if jid and not node.get("is_ghost") and not node.get("is_orphan"):
                self._open_job_in_builder(jid, stage_id)
                return
        else:
            stage_id = target
        spec = self._stage_by_id(stage_id)
        if spec:
            self._select_stage(spec)

    def _refresh_canvas(self):
        """Rebuild the canvas from the job store. Safe to call before the canvas
        exists (early in construction) and is the hook _finalize_job calls.
        Open pop-out panels ride along: they are views of the same store."""
        if getattr(self, "canvas", None) is not None:
            self.canvas.refresh()
        if getattr(self, "_popouts", None):
            self._refresh_popouts()

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
        classic 3-column layout. CARD mode is [pinned pipeline outline | canvas |
        terminal]: the canvas gets most of the width, the outline is fixed to the
        left window edge, the terminal runs the full height on the right, and the
        job builder lives in each card's pop-out panel rather than in a column.
        Same widgets, rebuilt containers — so no state is duplicated."""
        self._view_mode = mode
        canvas = (mode == "canvas")
        # Park every leaf in the stash first so none becomes an orphan top-level
        # window while we swap containers, then drop the previous main splitter.
        for w in self._panels:
            w.setParent(self._stash)
        self._clear_box(self._layout_host_v)
        self.job_stack.setCurrentIndex(1 if canvas else 0)

        if canvas:
            main = QSplitter(Qt.Horizontal)
            main.setHandleWidth(6)
            main.addWidget(self.outline_card)
            main.addWidget(self.job_stack)
            main.addWidget(self.terminal_card)
            # Only the canvas grows with the window; the side rails hold their
            # width, and the canvas can never be collapsed to nothing.
            main.setStretchFactor(0, 0)
            main.setStretchFactor(1, 1)
            main.setStretchFactor(2, 0)
            main.setCollapsible(1, False)
            # The widths the user last dragged the splitters to ARE the right
            # default — restore them; first launch gets a reading-width terminal.
            saved = self._load_config().get("canvas_split_sizes")
            if (isinstance(saved, list) and len(saved) == 3
                    and all(isinstance(v, (int, float)) and v >= 0 for v in saved)
                    and sum(saved) > 0):
                main.setSizes([int(v) for v in saved])
            else:
                main.setSizes([200, 950, 460])
            self._canvas_main_split = main
            main.splitterMoved.connect(self._save_canvas_split)
        else:
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

    # ---- per-card pop-out panels (canvas view) ----
    @staticmethod
    def _details_heading(text):
        lab = QLabel(text)
        lab.setStyleSheet("color:#9ec5ff;font-size:11px;font-weight:700;margin-top:6px;")
        return lab

    def _open_job_popout(self, target, tab=None):
        """Open (or focus) a card's floating panel — builder · details · outputs.

        `target` is a canvas node dict, or a bare stage id (outline, palette and
        menu paths). One panel per node id: re-opening focuses and retargets the
        existing panel instead of stacking duplicates. Several DIFFERENT nodes'
        panels can be open at once — that is the point (drag an output from one
        into an input field of another)."""
        if isinstance(target, str):
            spec = self._stage_by_id(target)
            if not spec:
                return
            target = {"id": f"ghost:{target}", "stage_id": target,
                      "label": spec.get("label", target),
                      "title": stage_title(target, spec.get("label", target)),
                      "group": spec.get("group", ""), "is_ghost": True,
                      "status": "ghost", "summary": {}}
        node_id = target.get("id") or f"ghost:{target.get('stage_id')}"
        pop = self._popouts.get(node_id)
        if pop is None:
            pop = JobPopout(self, target)
            self._popouts[node_id] = pop
            self._place_popout(pop, target)
        else:
            pop.refresh(node=target, force=True)
        if tab:
            pop.show_tab(tab)
        pop.show()
        pop.raise_()
        pop.activateWindow()

    def _place_popout(self, pop, node):
        """First show: tethered beside the card it belongs to, falling back to a
        cascade for panels with no card (stage/ghost panels) so several opened
        blind don't stack exactly on top of each other."""
        base = self._popout_anchor(pop)
        if base is not None:
            pop._prog_move = True
            try:
                pop.move(base)
            finally:
                pop._prog_move = False
            return
        origin = self.geometry().topLeft()
        k = len(self._popouts)
        pop._prog_move = True
        try:
            pop.move(origin.x() + 120 + 32 * (k % 6),
                     origin.y() + 90 + 32 * (k % 6))
        finally:
            pop._prog_move = False

    def _popout_anchor(self, pop):
        """The screen point a panel tethers to: just right of its card's top
        edge. None when the panel has no drawn card to follow (stage/ghost
        panels, hidden cards, lists view) — those float free."""
        try:
            if getattr(self, "_view_mode", "lists") != "canvas":
                return None
            n = (self.canvas._node_index or {}).get(pop.node.get("id"))
            if n is None or n.get("is_template"):
                return None
            v = self.canvas.view
            return v.mapToGlobal(v.mapFromScene(n["x"] + n["w"] + 12, n["y"]))
        except Exception:
            return None

    def _retether_popouts(self):
        """Glue every panel to its card as the canvas pans, zooms or repaints —
        without this the panels stand still while the cards move under them,
        and the card↔panel association dissolves the first time you pan.
        Clamped into the window so a big pan can't fling a panel off-screen."""
        pops = getattr(self, "_popouts", None)
        if not pops:
            return
        try:
            geo = self.geometry()
        except Exception:
            return
        for pop in list(pops.values()):
            base = self._popout_anchor(pop)
            if base is None:
                continue
            off = pop.tether_offset
            tx = base.x() + (off.x() if off is not None else 0)
            ty = base.y() + (off.y() if off is not None else 0)
            tx = max(geo.left() - 40, min(tx, geo.right() - 140))
            ty = max(geo.top(), min(ty, geo.bottom() - 90))
            pop._prog_move = True
            try:
                pop.move(tx, ty)
            except Exception:
                pass
            finally:
                pop._prog_move = False

    def _refresh_popouts(self):
        """Keep every open panel current — called wherever the canvas repaints,
        so a panel left open tracks its job through queued → running → done."""
        for pop in list(self._popouts.values()):
            try:
                pop.refresh()
            except Exception:
                pass
        self._retether_popouts()

    def _popout_header(self, pop, job):
        """The panel's always-current header: title line + status/meta line."""
        node = pop.node
        stage_id = node.get("stage_id", "")
        pop.head.setText(node.get("title", node.get("label", stage_id)))
        disp = node.get("subtitle") or stage_id
        if node.get("is_orphan"):
            meta = f"{node.get('group', '')} · found on disk — not yet a job"
        elif node.get("is_discovered"):
            meta = (f"{node.get('group', '')} · {stage_id} · ran outside the app "
                    f"— output on disk, no job record")
        elif node.get("is_ghost"):
            meta = f"{node.get('group', '')} · {stage_id} · not built yet"
        else:
            meta = (f"{node.get('group', '')} · {disp} · {node.get('status', '')}"
                    f"  ({node.get('id')})")
            st = summary_text(node.get("summary", {}) or {})
            if st:
                meta += f"\n{st}"
            if node.get("manual") and job is not None:
                note = manual_status_note(job)
                if note:
                    meta += f"\n✋ {note}"
        pop.meta.setText(meta)
        pop.setWindowTitle(f"{node.get('id', stage_id)} — "
                           f"{node.get('title', stage_id)}")

    def _save_outline_collapse(self, group, collapsed):
        cfg = self._load_config()
        folded = set(cfg.get("outline_collapsed", []))
        (folded.add if collapsed else folded.discard)(group)
        self._save_config({**cfg, "outline_collapsed": sorted(folded)})

    # ---- canvas splitter widths: where you drag them is the new default ----
    def _save_canvas_split(self, *_):
        sp = getattr(self, "_canvas_main_split", None)
        if sp is None:
            return
        try:
            self._pending_split_sizes = [int(v) for v in sp.sizes()]
        except Exception:
            return
        self._split_save_timer.start()

    def _flush_split_sizes(self):
        sizes = getattr(self, "_pending_split_sizes", None)
        if isinstance(sizes, list) and len(sizes) == 3 and sum(sizes) > 0:
            self._save_config({**self._load_config(),
                               "canvas_split_sizes": sizes})

    # ---- pinned outline: the three gestures ----
    def _outline_show(self, stage_id):
        """Click on an outline row: bring that stage's work into view.

        Only scrolls when there is something to show. Snapping to the empty
        spot where a stage's cards WOULD go read as a card having been made
        and lost — say there is nothing yet instead of pointing at a void."""
        idx = self.canvas._node_index or {}
        real = [n for n in idx.values()
                if n.get("stage_id") == stage_id and not n.get("is_template")]
        if real:
            leftmost = min(real, key=lambda n: (n["x"], n["y"]))
            self.canvas.reveal(leftmost["id"], force=True)
            return
        self._log(f"No cards for '{stage_title(stage_id, stage_id)}' yet — "
                  f"double-click it in the outline to open its builder, or drag "
                  f"it onto the canvas to place a job.", "info")

    def _outline_open(self, stage_id):
        self._open_job_popout(stage_id)

    def _outline_menu(self, stage_id, global_pos):
        # The ghost-card menu already says everything a stage can do; reuse it.
        node = {"id": f"ghost:{stage_id}", "stage_id": stage_id,
                "is_ghost": True, "status": "ghost", "summary": {}}
        self._card_menu(node, global_pos)

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
        # The job's REAL folder. Reconstructing jobs/<id> here ignored the
        # slug in the name, so the breadcrumb was dropped into a second,
        # otherwise-empty jobs/J47 beside the real jobs/J47_cryolo-picks.
        jobdir = Path(self.project_root) / job_real_dir(
            self.project_root, job_id, stage_id,
            load_jobs(self.project_root))
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
        # Name the REAL job folder. "linked as jobs/J18/outputs" pointed at a
        # path that does not exist -- the link lives in jobs/J18_<slug>/ -- so
        # the breadcrumb this exists to provide looked like it had not been made.
        here = os.path.relpath(jobdir, self.project_root)
        self._log(f"{job_id}: outputs are in {', '.join(real)} "
                  f"(linked as {here}/outputs).", "info")

    def _dir_state(self, rel):
        """'missing' | 'empty' | 'full' for a project-relative dir.

        _count_dir_entries returns 0 for BOTH missing and empty, which is the
        distinction the outputs pane needs: an empty folder is worth offering
        (the step may yet fill it), a folder that was never created is not.
        """
        root = getattr(self, "project_root", None)
        if not isinstance(root, (str, os.PathLike)) or not rel:
            return "missing"
        try:
            p = Path(root) / rel
            if not p.is_dir():
                return "missing"
            return "full" if any(True for _ in itertools.islice(p.iterdir(), 1)) \
                else "empty"
        except (OSError, TypeError, ValueError):
            return "missing"

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

    def _add_pattern_hint(self, box, verb, pattern):
        """Dim 'searches: *.tomostar' line under a dir button (skips if unknown)."""
        if not pattern:
            return
        lab = QLabel(f"      {verb}:  {pattern}")
        lab.setStyleSheet("color:#7c7c7c;font-size:10px;")
        lab.setWordWrap(True)
        box.addWidget(lab)

    @staticmethod
    def _params_reference_html(spec):
        """What every parameter of this stage MEANS, as one block of HTML for the
        Details tab — title, wire name (flag/env var), and the help text that
        otherwise only shows once the form is open."""
        if not spec or not spec.get("params"):
            return ""
        rows = []
        for p in spec.get("params", []):
            wire = _html.escape(param_wire_name(p))
            t = _html.escape(param_title(p))
            h = _html.escape(p.get("help", ""))
            rows.append(
                f"<p style='margin:3px 0;'><b>{t}</b>&nbsp; "
                f"<code style='color:#8fb4d8;font-size:10px;'>{wire}</code><br>"
                f"<span style='color:#9a9a9a;font-size:11px;'>{h}</span></p>")
        return ("<h3 style='color:#9ec5ff;margin-bottom:2px;'>Parameters</h3>"
                + "".join(rows))

    def _populate_details_box(self, box, node, pop=None):
        """A pop-out's Details tab: what the job does (the stage docs), what its
        parameters mean, where it reads from, and the card's actions. Directory
        access is via lazy 'open' buttons (no enumeration on show — ceph scandir
        is what has crashed this app before)."""
        stage_id = node.get("stage_id")
        spec = self._stage_by_id(stage_id)

        # Orphan = a pick set found on disk (made outside the app). Show where it
        # is + Adopt/View, then stop (it has no job record to describe).
        if node.get("is_orphan"):
            orph = node.get("orphan", {})
            info = QLabel(f"Found on disk — not yet a job.\n\nsuffix:  {orph.get('suffix','?')}\n"
                          f"dir:  {orph.get('dir','?')}\nseries:  {orph.get('n_series','?')}"
                          f"    pixel size:  {orph.get('angpix','?')} Å")
            info.setStyleSheet("color:#c8b78a;font-size:12px;")
            info.setWordWrap(True)
            box.addWidget(info)
            box.addWidget(self._details_heading("ACTIONS"))
            adopt = QPushButton("✦ Adopt as job")
            adopt.setToolTip("Register this pick set as a completed job (symlinks its "
                             "files into jobs/J###/matching — originals stay put) so it "
                             "wires into the workflow like any other job.")
            adopt.clicked.connect(lambda _=False, o=orph: self._adopt_orphan(o))
            box.addWidget(adopt)
            view = QPushButton("🔍 View picks (warp-tm-vis)")
            view.clicked.connect(lambda _=False, n=node: self._view_picks_tm_vis(n))
            box.addWidget(view)
            box.addWidget(self._open_dir_button(f"📂  {orph.get('dir','')}",
                                                orph.get("dir", "")))
            return

        # WHAT THIS JOB DOES + WHAT THE PARAMETERS MEAN, in one browser (rich
        # docs where tomogration_docs.json has them, the stage's inline docs
        # otherwise; the browser scrolls itself for the long ones).
        doc = self._docs.get(stage_id) if spec else None
        html = render_docs_html(doc) if doc else (
            render_inline_docs_html(spec) if spec else "")
        html += self._params_reference_html(spec)
        if html:
            browser = QTextBrowser()
            browser.setOpenExternalLinks(True)
            browser.setStyleSheet("background:#161b20;border:1px solid #2a2f36;"
                                  "border-radius:4px;color:#dddddd;")
            browser.setHtml(html)
            browser.setMinimumHeight(260)
            box.addWidget(browser)

        ins, _outs = STAGE_IO.get(stage_id, ([], []))
        if ins:
            box.addWidget(self._details_heading("READS FROM"))
            for rel in ins:
                box.addWidget(self._open_dir_button(f"📂  {rel}", rel))
                self._add_pattern_hint(box, "searches", DIR_FILE_HINTS.get(rel))

        if spec:
            box.addWidget(self._details_heading("ACTIONS"))
            if node.get("is_ghost"):
                build = QPushButton("▶ Build & run job")
                build.setToolTip("Create a job instance for this stage (input auto-wired "
                                 "to the newest upstream job) and run it.")
                build.clicked.connect(lambda _=False, sid=stage_id: self._build_job(sid))
                box.addWidget(build)
            else:
                jid = node.get("id")
                run = QPushButton("▶ Run / re-run")
                run.clicked.connect(lambda _=False, j=jid: self._run_job(j))
                box.addWidget(run)
                if VIEWER_PLANS.get(stage_id):
                    view = QPushButton("🔍 View in napari")
                    view.setToolTip("Open this job's result on top of its "
                                    "tomogram — tomoview for volumes, "
                                    "surforama for meshes.")
                    view.clicked.connect(lambda _=False, j=jid: self._view_job(j))
                    box.addWidget(view)
                for ch in DOWNSTREAM.get(stage_id, []):
                    b = QPushButton(f"→ Build {stage_title(ch, ch)} from this")
                    b.setToolTip("Create the next job wired to THIS job, with the suffix / "
                                 "pattern auto-derived — then set your threshold and run.")
                    b.clicked.connect(lambda _=False, j=jid, c=ch: self._build_downstream(j, c))
                    box.addWidget(b)
                if stage_id == "mb_explore":
                    ana = QPushButton("📊 Analyse results and view")
                    ana.setToolTip("Charts over this sweep: which variant, "
                                   "which threshold, which cutoff — plus the "
                                   "recall/virions trade-off and the sphere vs "
                                   "ellipsoid comparison. Filter with Show:, "
                                   "then open the best two in napari.")
                    ana.clicked.connect(lambda _=False, j=jid: self._analyse_sweep(j))
                    box.addWidget(ana)
                fork = QPushButton("⑂ Duplicate (fork)")
                fork.clicked.connect(lambda _=False, j=jid: self._fork_job(j))
                box.addWidget(fork)
            if pop is not None:
                edit = QPushButton("⚙ Edit in the Builder tab →")
                edit.setToolTip("The Builder tab of this panel — the job's own "
                                "parameter form.")
                edit.clicked.connect(lambda _=False, p=pop: p.show_tab("builder"))
                box.addWidget(edit)
            if stage_id in ("ts_template_match", "threshold_picks"):
                nap = QPushButton("🔍 View picks (warp-tm-vis)")
                nap.setToolTip("Open this pick set in warp-tm-vis — overlays the picks + "
                               "correlation volumes on the tomograms (edit the command if "
                               "the suffix/paths differ).")
                nap.clicked.connect(lambda _=False, n=node: self._view_picks_tm_vis(n))
                box.addWidget(nap)

    def _populate_outputs_box(self, box, node):
        """A pop-out's Outputs tab: where this job's files actually land, as
        DRAGGABLE rows — drop one into an input field of any open builder to
        hand-wire data between jobs. Only WarpTools job stages write into
        jobs/<id>/; wrapper stages write wherever their params point, so the
        REAL destinations are listed first (see _job_output_dirs). Nothing is
        enumerated until asked (the ⠇ button lists a folder's files on demand)."""
        stage_id = node.get("stage_id")
        _ins, outs = STAGE_IO.get(stage_id, ([], []))
        hint = QLabel("Drag a ⠿ row into an input field of another job's builder "
                      "to use that path there. File vs folder and relative vs "
                      "absolute are converted to what the field wants.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#7c7c7c;font-size:10px;")
        box.addWidget(hint)
        box.addWidget(self._details_heading("OUTPUTS"))
        jid = node.get("id") if not node.get("is_ghost") else None
        if jid:
            real = self._job_output_dirs(jid, stage_id, with_notes=True)
            for rel, why in real:
                box.addWidget(self._output_dir_block(
                    rel, why or DIR_FILE_HINTS.get(rel), jid))
            # jobs/<id>/ is offered ONLY when it exists. Advertising it
            # unconditionally is what put a fictitious jobs/J4 on a card whose
            # gain went to gains/ — "open" then reported a directory that had
            # never been created, on a step that had in fact succeeded.
            try:
                outrel = job_real_dir(self.project_root, jid, stage_id,
                                      load_jobs(self.project_root))
            except (OSError, ValueError):
                outrel = f"jobs/{jid}"
            here = self._dir_state(outrel)
            if here != "missing":
                box.addWidget(self._output_dir_block(
                    outrel,
                    (DIR_FILE_HINTS.get(outs[0]) if (outs and not real) else ""),
                    jid))
            elif real:
                note = QLabel(f"    (no {outrel}/ — this step writes to the "
                              f"path(s) above)")
                note.setStyleSheet("color:#7c7c7c;font-size:10px;")
                note.setWordWrap(True)
                box.addWidget(note)
            else:
                note = QLabel("    (nothing recorded yet — this step has not "
                              "written anywhere tomogration can name)")
                note.setStyleSheet("color:#7c7c7c;font-size:10px;")
                note.setWordWrap(True)
                box.addWidget(note)
        else:
            for rel in outs:
                box.addWidget(self._output_dir_block(rel, DIR_FILE_HINTS.get(rel), ""))

    def _output_dir_block(self, rel, note, jid):
        """One output folder: a draggable row + an on-demand file listing, so a
        single .star (etc.) can be dragged too, not only whole folders."""
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(1)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(4)
        head.addWidget(_DragPathRow(self, rel, note or "", jid, is_dir=True), 1)
        lister = QPushButton("⠇")
        lister.setFixedWidth(24)
        lister.setToolTip("List this folder's files (top level, first 40) so "
                          "single files can be dragged too.")
        head.addWidget(lister)
        hw = QWidget()
        hw.setLayout(head)
        v.addWidget(hw)
        if note:
            self._add_pattern_hint(v, "writes", note)
        files_box = QVBoxLayout()
        files_box.setContentsMargins(0, 0, 0, 0)
        files_box.setSpacing(1)
        fw = QWidget()
        fw.setLayout(files_box)
        fw.setVisible(False)
        v.addWidget(fw)

        def toggle(_=False):
            if fw.isVisible():
                fw.setVisible(False)
                return
            if files_box.count() == 0:
                entries = self._list_dir_entries(rel, cap=40)
                for name, is_dir in entries:
                    files_box.addWidget(_DragPathRow(
                        self, f"{rel.rstrip('/')}/{name}", "", jid,
                        is_dir=is_dir, small=True))
                if not entries:
                    empty = QLabel("      (empty, or not created yet)")
                    empty.setStyleSheet("color:#5c6b7a;font-size:10px;")
                    files_box.addWidget(empty)
            fw.setVisible(True)

        lister.clicked.connect(toggle)
        return w

    def _list_dir_entries(self, rel, cap=40):
        """Top-level entries of a project-relative dir, capped and sorted (dirs
        first). Bounded on purpose: this feeds the on-demand file rows, and an
        unbounded scandir over ceph is what has frozen this app before."""
        root = getattr(self, "project_root", None)
        if not isinstance(root, (str, os.PathLike)) or not rel:
            return []
        try:
            p = Path(root) / rel
            if not p.is_dir():
                return []
            out = []
            with os.scandir(p) as it:
                for e in itertools.islice(it, cap):
                    try:
                        out.append((e.name, e.is_dir(follow_symlinks=False)))
                    except OSError:
                        continue
            out.sort(key=lambda t: (not t[1], t[0].lower()))
            return out
        except (OSError, TypeError, ValueError):
            return []

    def _smart_path_value(self, p, payload):
        """Coerce a dropped path to what parameter `p` wants — the 'intelligent'
        half of output→input drag. Returns (value, note); value None refuses.

        Rules, in order: a folder param handed a FILE takes the containing
        folder (same convention as the inventory browser); a param the stage
        lists in abs_paths gets an ABSOLUTE path (its wrapper cd's elsewhere);
        everything else goes project-relative when inside the project, so the
        built command stays portable across VMs."""
        raw = str(payload.get("path", "")).strip().strip('"').strip("'")
        if not raw:
            return None, None
        root = Path(self.project_root) \
            if isinstance(getattr(self, "project_root", None),
                          (str, os.PathLike)) else None
        ap = Path(os.path.expanduser(raw))
        if not ap.is_absolute() and root is not None:
            ap = root / raw
        notes = []
        is_dir = payload.get("is_dir")
        if is_dir is None:
            try:
                is_dir = ap.is_dir()
            except OSError:
                is_dir = not ap.suffix
        if _wants_directory(p) and not is_dir:
            ap = ap.parent
            notes.append("field wants a folder — used the containing folder")
        spec = p.get("_spec") or {}
        if p.get("name") in (spec.get("abs_paths") or []):
            value = str(ap)
            notes.append("absolute (this step resolves paths from elsewhere)")
        elif root is not None:
            try:
                value = os.path.relpath(ap, root)
            except ValueError:
                value = str(ap)
            if value.startswith(".."):
                value = str(ap)          # outside the project — keep it absolute
            elif value != raw:
                notes.append("made relative to the project root")
        else:
            value = str(ap)
        return value, "; ".join(notes)

    def _populate_popout_builder(self, box, node, ctx):
        """A pop-out's Builder tab: the same parameter form the stage builder
        makes, bound to THIS card. Editable (building/queued) jobs get the
        save/run-this-card buttons; finished ones show their recorded values
        behind the history banner; ghosts and discovered rows get a fresh
        stage form."""
        stage_id = node.get("stage_id")
        spec = self._stage_by_id(stage_id)
        if not spec:
            lab = QLabel(f"No stage spec for '{stage_id}'.")
            lab.setWordWrap(True)
            box.addWidget(lab)
            return
        jid = node.get("id")
        job = None
        if jid and not node.get("is_ghost") and not node.get("is_orphan"):
            job = (load_jobs(self.project_root).get("jobs") or {}).get(jid)
        if job is None:
            self._populate_builder(box, spec, ctx)
        else:
            kept, dropped, _missing = params_for_builder(spec, job.get("params"))
            editable = job.get("status") in RERUNNABLE
            showing = None if editable else {
                "id": jid, "status": job.get("status", ""),
                "when": (job.get("finished") or job.get("started")
                         or job.get("created", "")),
                "empty": not kept, "dropped": dropped, "job": job}
            self._populate_builder(box, spec, ctx,
                                   bound_job=jid if editable else None,
                                   form_params=kept if kept else None,
                                   showing=showing, exact=bool(kept))
        pop = self._popouts.get(node.get("id") or f"ghost:{stage_id}")
        if pop is not None:
            ctx["rebuild"] = lambda p=pop: p.refresh(force=True)

    # ---- card actions (Phase 3): build / fork / run / delete ----
    def _effective_params(self, spec):
        """Template defaults overlaid with the user's persisted per-stage edits —
        the values a fresh job of this stage should start from."""
        vals = stage_defaults(spec)
        vals.update(self._param_store.get(spec["id"], {}))
        # ...and the values only the PROJECT knows. Without this a job built by
        # the auto-chain ran on the template's numbers -- EML46's pixel size, a
        # gain path that did not exist, the trunk warp_frameseries -- because
        # every dynamic default lived in the builder form and a form was never
        # opened. "Click ↻ Rebuild from controls" was a workaround for this line
        # being absent.
        try:
            vals.update(self._dynamic_overrides(spec))
        except Exception:
            pass
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
        b_conv = QPushButton("→ Re-extract with Warp")
        b_adopt = QPushButton("✦ Adopt as job")
        b_open = QPushButton("📂 Open folder")
        for b in (b_conv, b_adopt, b_open):
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
            lambda *_: act(self._open_relion_export))
        b_conv.clicked.connect(lambda: act(self._open_relion_export))
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

    def _open_relion_export(self, orph):
        """A found-on-disk RELION job (typically a Subset selection): re-extract
        its particles with Warp at a new pixel size.

        ONE route, the one Warp is built for: ts_export_particles reads the
        RELION star directly (--input_star), subtracts the refined
        rlnOrigin*Angst shifts itself and scales the coordinates by the star's
        own pixel size (--coords_angpix). No pick stars are written and no
        coordinate is converted by hand — the pick-star converter with its
        three modes produced three conventions and a silent failure for each.

        Canvas view promotes the selection to a green card and builds the export
        card downstream of it: wired, star prefilled, coords_angpix read from
        the star, output folder named after the selection, builder open. Lists
        view prefills the export form the same way."""
        star = str(orph.get("star", "")).strip()
        spec = self._stage_by_id("ts_export_particles")
        if not spec or not star:
            return
        src_tag = "/".join(star.split("/")[-3:-1]) if "/" in star else star
        if getattr(self, "_view_mode", "lists") == "canvas":
            sel_id = self._ensure_selection_job(star, src_tag)
            if sel_id:
                self._invalidate_orphans()      # its found-on-disk entry is consumed
                self._log(f"{src_tag or star} → {sel_id} on the canvas; building the "
                          f"Warp export downstream of it.", "ok")
                self._build_downstream(sel_id, "ts_export_particles")
                return
        derived = direct_export_params(star, src_tag)
        derived.update(self._direct_export_from_star(derived))
        self._param_store.setdefault("ts_export_particles", {}).update(derived)
        self._persist_param_store()
        if getattr(self, "_view_mode", "lists") == "canvas":
            self._open_job_popout("ts_export_particles", tab="builder")
        else:
            self._select_stage(spec)
        self._log(f"Loaded {star} into ts_export_particles (RELION-star route). Set "
                  f"output_angpix, box and diameter, then run.", "ok")

    def _direct_export_from_star(self, derived):
        """The one export value only the star can supply: coords_angpix, which
        must be the star's own pixel size. Returns the extra params (possibly
        empty) and logs what the star says about itself."""
        star = str((derived or {}).get("input_star", "") or "").strip()
        if not star or not self.project_root:
            return {}
        path = Path(star) if os.path.isabs(star) else Path(self.project_root) / star
        info = star_header_info(path)
        if not info:
            self._log(f"Could not read {star}: set coords_angpix by hand to the star's "
                      f"optics rlnImagePixelSize.", "warn")
            return {}
        out = {}
        apx = info.get("pixel_size")
        if apx:
            out["coords_angpix"] = f"{float(apx):g}"
            self._log(f"coords_angpix = {out['coords_angpix']} Å/px, read from {star} "
                      f"({'optics rlnImagePixelSize' if info.get('has_optics') else 'rlnPixelSize'}).",
                      "info")
        else:
            self._log(f"{star} does not state its pixel size — set coords_angpix by "
                      f"hand (the pixel size its coordinates are counted in).", "warn")
        cols = set(info.get("columns") or [])
        if not ({"rlnMicrographName", "rlnTomoName"} & cols):
            self._log(f"{star} has neither rlnMicrographName nor rlnTomoName — Warp "
                      f"cannot map its particles to tilt series.", "warn")
        if not info.get("origins"):
            self._log("No refined shifts (rlnOrigin*Angst) in the star: particles will "
                      "be re-extracted on their original pick centres.", "info")
        return out

    def _check_direct_export(self, values, interactive=True):
        """RELION-star route pre-flight: the star must exist and coords_angpix
        must be the star's own pixel size. Warp scales the coordinates by
        coords_angpix, so any other value cuts every particle from the wrong
        place — and exits 0. Returns True to proceed."""
        star = str((values or {}).get("input_star", "") or "").strip()
        if not star:
            return True
        root = getattr(self, "project_root", None)
        path = Path(star) if os.path.isabs(star) else (
            Path(root) / star if root else Path(star))
        msg = ""
        if not path.is_file():
            msg = (f"RELION star not found: {star}\n(paths are relative to the "
                   f"project root)")
        else:
            info = star_header_info(path) or {}
            apx = info.get("pixel_size")
            capx = str((values or {}).get("coords_angpix", "") or "").strip()
            if apx and capx:
                try:
                    got = float(capx)
                except ValueError:
                    got = None
                if got is None or abs(got - float(apx)) > 1e-3 * float(apx):
                    msg = (f"coords_angpix is {capx}, but {star} states "
                           f"{float(apx):g} Å/px (its optics rlnImagePixelSize).\n\n"
                           f"Warp multiplies the star's coordinates by coords_angpix, "
                           f"so every particle would be cut "
                           f"{(got / float(apx)) if got else 0:.3g}× off its "
                           f"position, with exit 0.\n\nSet coords_angpix to "
                           f"{float(apx):g}.")
        if not msg:
            return True
        self._log("export blocked: " + msg.replace("\n", " "), "fail")
        if interactive:
            QMessageBox.warning(self, "ts_export_particles — RELION star", msg)
        return False

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
        job = new_job(self.project_root, "ts_template_match", label, params,
                      inputs, kind="reextract")
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
        Filed under relion4_result with tool='relion_selection': it IS a RELION
        job, and shaping it as a ts_template_match (to put it in the green Pick
        row) meant 'Subset selection' cards appeared in two different rows
        depending on which way they were adopted. Returns the job id (or None
        if it can't be made)."""
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
            job = new_job(self.project_root, "relion4_result",
                          src_tag or "RELION selection",
                          {"source_star": source, "job_dir": src_tag or source,
                           "job_type": src_tag or source,
                           "n_particles": n_particles or ""}, inputs={},
                          kind="relion_selection")
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
        if self.runner.busy():
            self._log("A job is already running — convert crYOLO picks after it "
                      "finishes (the converter runs through the same runner).",
                      "fail")
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
                      f"crYOLO picks '{tag}'", params, inputs={},
                      kind="cryolo")
        dst = root / job["output_dir"] / "matching"
        try:
            dst.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            delete_job(self.project_root, job["id"])
            QMessageBox.critical(self, "Convert crYOLO picks", f"Cannot make {dst}: {e}")
            return

        # 3. run the converter into the card's matching dir — through the shared
        #    ProcessRunner, NOT a blocking subprocess.run: a big COORDS folder
        #    over ceph froze the whole app (no repaint, no TERMINATE) for up to
        #    ten minutes. Now the output streams to the terminal like any run.
        script = Path(__file__).resolve().parent / "ml_cryolo_to_warp_picks_auto.py"
        argv = [sys.executable, str(script), coords, recon,
                "--out_dir", str(dst), "--apx", apx, "--suffix", tag, "--execute"]
        if flip_y:
            argv.append("--flip_y")
        cmd = " ".join(shlex.quote(a) for a in argv)
        update_job(self.project_root, job["id"], status="running", command=cmd,
                   tool="cryolo", started=_now())
        self._active_job_id = job["id"]
        self._active_stage = None
        self._active_cmd = cmd
        self._m_resolution = None

        self._polarity_result = None
        self._attempt = 1
        self._failed_file = None
        # _finalize_job records the exit; this hook then does the crYOLO-specific
        # bookkeeping (count the stars, wire the card — or delete it on failure).
        self._job_finalize_hook = (
            lambda jid, code, d=dst, a=apx, t=tag:
            self._finish_cryolo_convert(jid, code, d, a, t))
        self._log(f"--- running {job['id']} · convert crYOLO picks ---", "info")
        self._refresh_canvas()
        self.runner.run(cmd, self.project_root)

    def _finish_cryolo_convert(self, job_id, code, dst, apx, tag):
        """Post-run bookkeeping for Tools ▶ Convert crYOLO picks."""
        n = len(list(dst.glob(f"*_{apx}Apx_{tag}.star")))
        if code != 0 or n == 0:
            delete_job(self.project_root, job_id)
            self._refresh_canvas()
            QMessageBox.critical(
                self, "Convert crYOLO picks",
                f"No pick STARs written (exit {code}) — see the terminal log. "
                f"The card was removed.")
            return
        # Completed — now it wires into the DAG like any adopted pick set.
        # tool="cryolo" so the canvas titles it as crYOLO, not 'Template matching'.
        update_job(self.project_root, job_id, status="completed", exit_code=0,
                   orphan_suffix=f"_{tag}", summary={"series": n})
        self._log(f"crYOLO picks '{tag}' → {job_id} ({n} tomograms) registered.",
                  "ok")
        self._refresh_canvas()
        QMessageBox.information(
            self, "Convert crYOLO picks",
            f"Registered {n} tomograms as pick-set job {job_id}.\n\n"
            f"Right-click the card ▶ Build downstream from this ▶ "
            f"ts_export_particles to extract. Turn --normalized_coords ON in "
            f"the export step.")

    def _card_menu(self, node, global_pos):
        """Right-click menu on a canvas card."""
        menu = QMenu(self)
        sid = node.get("stage_id")
        if node.get("is_orphan"):
            orph = node.get("orphan", {})
            if orph.get("kind") == "relion_job":
                menu.addAction("Re-extract with Warp (export from this star)",
                               lambda: self._open_relion_export(orph))
                menu.addAction("Adopt as job (feeds M)",
                               lambda: self._adopt_orphan(orph))
            else:
                menu.addAction("Adopt as job", lambda: self._adopt_orphan(orph))
                menu.addAction("View picks (warp-tm-vis)",
                               lambda: self._view_picks_tm_vis(node))
            menu.addAction("Details", lambda: self._open_job_popout(node, tab="details"))
            menu.addSeparator()
            menu.addAction("Hide (remove from view)", lambda: self._hide_card(node))
            menu.addAction("Delete folder from disk…",
                           lambda: self._delete_orphan_dir(orph))
            menu.exec(global_pos)
            return
        if node.get("is_ghost"):
            menu.addAction("Build & run job", lambda: self._build_job(sid, run=True))
            menu.addAction("＋ Queue this job", lambda: self._queue_stage(sid))
            menu.addAction("＋ Create job (edit, run later)",
                           lambda: self._create_job_card(sid))
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
                               lambda: self._run_queued_job(jid,
                                                            interactive=True))
                menu.addAction("✕ Remove from queue", lambda: self._delete_job(jid))
                menu.addSeparator()
            else:
                # Failed or finished: requeue is the CryoSPARC-style 'try it again'.
                menu.addAction("↻ Restart (queue again)", lambda: self._requeue_job(jid))
            menu.addAction("Run / re-run", lambda: self._run_job(jid))
            if VIEWER_PLANS.get(sid):
                menu.addAction("🔍 View in napari", lambda: self._view_job(jid))
            # A promoted RELION selection rides ts_template_match but has no picks of
            # its own — don't offer threshold/export downstream from it.
            children = [] if node.get("title") == "RELION selection" else DOWNSTREAM.get(sid, [])
            if children:
                sub = menu.addMenu("Build downstream from this")
                for ch in children:
                    sub.addAction(stage_title(ch, ch),
                                  lambda _=False, c=ch: self._build_downstream(jid, c))
            menu.addAction("Duplicate (fork)", lambda: self._fork_job(jid))
            # The exit code answers "did the process return 0", which is not
            # the same question as "did this work". Both failure modes are
            # real: a run that exits 0 having quietly used the wrong data, and
            # a run that finishes the work and dies on an X11 teardown.
            # Building cards are judgeable too: the work may have been done
            # outside tomogration, or the card may be a dead end that should
            # stop reading as outstanding.
            if status in ("completed", "failed", "building"):
                mark = menu.addMenu("✋ Set status by hand")
                if status != "completed":
                    mark.addAction("Mark completed",
                                   lambda: self._mark_job(jid, "completed"))
                if status != "failed":
                    mark.addAction("Mark failed",
                                   lambda: self._mark_job(jid, "failed"))
                if node.get("manual"):
                    mark.addSeparator()
                    mark.addAction("↺ Restore the recorded status",
                                   lambda: self._unmark_job(jid))
            menu.addAction("Details", lambda: self._open_job_popout(node, tab="details"))
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

    def _mark_job(self, job_id, status):
        """Set a finished job's verdict by hand. The machine's record is kept
        underneath, so this is an annotation, not a rewrite."""
        ok, msg = set_manual_status(self.project_root, job_id, status)
        self._log(msg, "ok" if ok else "warn")
        if ok:
            self._refresh_canvas()

    def _unmark_job(self, job_id):
        ok, msg = clear_manual_status(self.project_root, job_id)
        self._log(msg, "ok" if ok else "warn")
        if ok:
            self._refresh_canvas()

    def _create_job_card(self, stage_id):
        """Put a Building card on the canvas and open it for editing. Nothing
        runs, so nothing is asked: the ⚠ validator and the overwrite check are
        questions about a command that is not going to execute, and answering
        "Run anyway?" to place a card reads as if the run already started. They
        fire later, at the button that actually runs it."""
        job = self._build_job(stage_id, run=False, confirm=False)
        if job:
            self._open_job_in_builder(job["id"], stage_id)

    def _build_job(self, stage_id, params=None, run=True, parent=None, confirm=True,
                   label=None, ctx=None):
        """Create a job instance for a stage (auto-wiring its input to the newest
        upstream WarpTools job) and optionally run it.

        `confirm=False` skips the pre-flight dialogs. They ask about things a RUN
        would do — overwriting an output folder, ignoring a validator warning — so
        asking them while merely placing a card on the canvas is a question about a
        command that is not going to be executed.

        `label` overrides the stage's name on the card — a sweep needs each card to
        say which variant it is, not six cards reading "IsoNet: train".
        """
        spec = self._stage_by_id(stage_id)
        if not spec:
            return None
        store = load_jobs(self.project_root)
        auto_params = params is None
        from_form = False
        if auto_params:
            # prefer the live form this was pressed in (a pop-out passes its ctx),
            # else the singleton form if it shows this stage, else stored defaults
            cur = ctx if ctx is not None else self.current
            if cur and cur.get("spec", {}).get("id") == stage_id:
                params = self._values(cur)
                from_form = True
            else:
                params = self._effective_params(spec)
        if parent is None:
            # a parent explicitly chosen via "Build downstream" wins over the
            # newest-upstream default
            parent = self._pending_parent.pop(stage_id, None) or default_parent_for(stage_id, store)
        # Wire to the parent's real outputs. _build_downstream has always done
        # this; the auto-chain never did, so J4 imported from warp_frameseries
        # while its parent J2 had written to jobs/J2_fs-motion-and-ctf. Only for
        # params we chose ourselves -- an explicit `params` (already derived, or
        # typed into a form) is the caller's, not ours to overwrite.
        if auto_params and not from_form and parent:
            pj = store.get("jobs", {}).get(parent) or {}
            if pj:
                try:
                    wired = derive_child_params(stage_id, pj.get("stage_id"),
                                                pj.get("params", {}),
                                                pj.get("output_dir", ""))
                except Exception:
                    wired = {}
                keep = {k: v for k, v in wired.items()
                        if k in {pp["name"] for pp in spec.get("params", [])}}
                if keep:
                    params = {**params, **keep}
                    self._log(f"{stage_id} wired to {parent}: "
                              + ", ".join(f"{k}={v}" for k, v in keep.items()),
                              "info")
        inputs = {"processing": parent} if parent else {}
        if confirm and not self._confirm_validator(spec, params):
            return None
        if confirm and not self._confirm_overwrite(spec, params):
            return None
        job = new_job(self.project_root, stage_id,
                      label or spec.get("label", stage_id), params, inputs)
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
        # A stage whose project_dir is a RELION project root holding
        # matching_conv.star, subtomo/ and previous jobs touches none of them —
        # it writes its own job folder and parks pipeline state aside. Warning
        # about the container was a false alarm about the wrong files, and it
        # drowned out the real risk. So when subdirs are declared, warn about
        # THOSE, and drop the
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
        # The polarity the docs promise "downstream reads from the registry":
        # deliver it at wiring time. derive_child_params is pure and cannot
        # read the registry, so the lookup lives here, where the root is known.
        if (child_stage_id == "mb_fit_virions"
                and str(derived.get("tomogram", "")).strip()
                and not str(derived.get("polarity", "")).strip()):
            try:
                import tomogration_variants as _tv     # noqa: PLC0415
                tomo = str(derived["tomogram"]).rstrip("/")
                for e in _tv.load(self.project_root).get("variants", []):
                    if (str(e.get("path", "")).rstrip("/") == tomo
                            and e.get("polarity") in ("dark", "bright")):
                        derived["polarity"] = e["polarity"]
                        self._log(f"polarity '{e['polarity']}' prefilled from "
                                  f"the variant registry ({tomo}).", "info")
                        break
            except Exception:
                pass
        # RELION-star export: the one value only the file can supply.
        if (child_stage_id == "ts_export_particles"
                and str(derived.get("input_star", "") or "").strip()):
            derived.update(self._direct_export_from_star(derived))
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
        # Open THIS card's builder (bound, so its run button runs it instead of
        # creating a second job for the same step) — in canvas view that is the
        # card's pop-out panel.
        self._open_job_in_builder(jid, child_stage_id)

    def _open_job_in_builder(self, job_id, stage_id):
        """Open a card in the builder, showing THAT job's parameters.

        Two separate things, which used to be conflated:

          SHOWING a past run's parameters is the whole point of clicking a card —
          "what did J9 actually run?" is the commonest question anyone asks of a
          finished job. Clicking a completed card used to leave the form on the
          last-used values, so an export from July displayed this week's v6 paths.

          BINDING the run buttons to that job is only right while it can still
          change. A completed job is history; running from here creates a NEW job,
          and the buttons keep saying so.

        The job's values reach the form as a ONE-SHOT override, never by writing
        the per-stage param store: merely LOOKING at a card must not destroy the
        edits you have saved for that stage. (They commit only if you then edit —
        at which point you have chosen these values as your starting point.)
        Exception: re-clicking the card the builder is ALREADY bound to keeps the
        form's working edits — that is your live editing session for this card,
        not a request to reload the card's last-saved values.

        In canvas view "the builder" IS the card's pop-out panel — same
        semantics, delivered by _populate_popout_builder instead of the
        singleton form.
        """
        if getattr(self, "_view_mode", "lists") == "canvas":
            node = (self.canvas._node_index or {}).get(job_id) \
                or {"id": job_id, "stage_id": stage_id, "summary": {}}
            self._open_job_popout(node, tab="builder")
            return
        spec = self._stage_by_id(stage_id)
        if not spec:
            return
        job = (load_jobs(self.project_root).get("jobs") or {}).get(job_id) or {}
        kept, dropped, _missing = params_for_builder(spec, job.get("params"))
        editable = job.get("status") in RERUNNABLE
        rebound = (editable and
                   getattr(self, "_builder_bindings", {}).get(stage_id) == job_id)
        if kept and not rebound:
            self._form_job_params = {"stage": stage_id, "params": kept}
            self._exact_params_for = stage_id     # its values, not today's defaults
        if editable:
            self._builder_job_id = job_id
        # Rendered as a banner over the form, so nobody edits a finished run
        # believing they are changing it.
        self._showing_job = None if editable else {
            "id": job_id, "status": job.get("status", ""),
            "when": job.get("finished") or job.get("started") or job.get("created", ""),
            "empty": not kept, "dropped": dropped,
            # The banner needs the job itself to say what it actually IS: a
            # re-extract and a crYOLO pick set are both filed as template
            # matches, so stage_id alone cannot tell the reader which they
            # are looking at.
            "job": job}
        self._select_stage(spec)

    def _save_queued_job(self, job_id, ctx=None):
        """Write the form's values into an already-queued job and LEAVE it queued.

        "+ Queue variant" always minted a new job, which is right when you want a
        second variant to run alongside and wrong when you are editing a card that is
        already sitting in the queue — it produced a duplicate card for the same step.
        Queued jobs run in card order when the one before them finishes, so there is
        nothing else to do here but save.
        """
        cur = ctx if ctx is not None else self.current
        store = load_jobs(self.project_root)
        job = (store.get("jobs") or {}).get(job_id)
        spec = (cur or {}).get("spec") or {}
        if not job:
            self._log(f"{job_id} no longer exists — nothing saved.", "warn")
            return
        values = self._values(cur)
        if not self._confirm_validator(spec, values):
            return
        cmd = (cur["cmd"].toPlainText().strip()
               if cur.get("manual") else "")
        update_job(self.project_root, job_id, params=values, status="queued",
                   command=cmd,
                   # Saving an ALREADY-queued job keeps its place in line.
                   queued_at=job.get("queued_at") or _now())
        pend = [j["id"] for j in queued_jobs(load_jobs(self.project_root))]
        ahead = pend[:pend.index(job_id)] if job_id in pend else []
        when = (f"after {', '.join(ahead)}" if ahead else "next")
        self._log(f"Saved {job_id}; it stays queued and runs {when}"
                  + (" (running your edited command verbatim)." if cmd else "."), "ok")
        self._refresh_canvas()
        self._refresh_queue()

    def _save_and_run_job(self, job_id, ctx=None):
        """Write the form's values back into an existing QUEUED job and run it.

        The builder is stage-scoped, so its run button always built a NEW job. After
        "Build downstream" — which now creates the card first — that produced a
        second card for the same step, running off on its own while the one you were
        editing sat queued forever.
        """
        cur = ctx if ctx is not None else self.current
        store = load_jobs(self.project_root)
        job = (store.get("jobs") or {}).get(job_id)
        spec = (cur or {}).get("spec") or {}
        if not job:
            # The card was deleted while the builder was open. Building a fresh job
            # is the sane fallback, but say so rather than silently doing it.
            self._log(f"{job_id} no longer exists — building a new job instead.",
                      "warn")
            if spec:
                self._build_job(spec["id"], run=True, ctx=cur)
            return
        # A job can start running, or be cleared, while the form sits open. Writing
        # the form into a job that is no longer yours to edit would be worse than
        # refusing.
        if job.get("status") not in RERUNNABLE:
            self._log(f"{job_id} is {job.get('status')} — not editable from the "
                      f"builder. Use its card to re-run it.", "warn")
            return
        values = self._values(cur)
        if not self._confirm_validator(spec, values):
            return
        if not self._confirm_overwrite(spec, values):
            return
        update_job(self.project_root, job_id, params=values)
        if cur.get("manual"):
            # A hand-edited command is the user's explicit intent — store it and run
            # it verbatim rather than rebuilding it from the controls.
            update_job(self.project_root, job_id,
                       command=cur["cmd"].toPlainText().strip())
            self._log(f"Saved your edited command into {job_id}.", "info")
            self._run_queued_job(job_id, interactive=True, confirmed=True)
        else:
            update_job(self.project_root, job_id, command="")   # rebuilt by _run_job
            self._run_job(job_id, confirmed=True)

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
        # Canvas view: the stage builder lives in pop-outs now. The values are
        # committed to the store above, so a fresh stage form shows them —
        # open one so "load into builder" still ends with a builder on screen.
        if getattr(self, "_view_mode", "lists") == "canvas":
            self._open_job_popout(stage_id, tab="builder")

        note = f"Loaded {job_id}'s parameters into the job builder ({len(kept)} values)."
        kind = actual_job_kind(job)
        if kind:
            note += (f"  {job_id} is {kind}, which is filed under "
                     f"'{spec.get('label', stage_id)}' so its downstream wiring "
                     f"works — that is why the builder says that.")
        carried, stale = carried_note(dropped, job.get("params", {}))
        for k, v in carried.items():
            note += f"  {k} = {v} ({CARRIED_PARAMS[k]})."
        if stale:
            note += (f"  Ignored {len(stale)} parameter(s) this stage does not "
                     f"declare: {', '.join(stale)}.")
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
            # The fork's own builder, bound to it (its pop-out in canvas view) —
            # not the bare stage form, which would edit nobody's card.
            self._open_job_in_builder(job["id"], src["stage_id"])

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

    def _mark_queued(self, job_id, spec):
        """Resolve a built job's command NOW and park it as status='queued', so
        what you queued is what runs. A job whose command cannot be built is
        deleted rather than left as a card that fails the moment it starts.
        Returns True when it is queued."""
        store = load_jobs(self.project_root)
        try:
            cmd = build_job_command(spec, store["jobs"][job_id], store,
                                    self.warp_launch, self._group_inputs)
        except Exception as e:
            self._log(f"Could not build a command for {job_id}: {e}", "fail")
            delete_job(self.project_root, job_id)
            return False
        update_job(self.project_root, job_id, status="queued", command=cmd,
                   queued_at=_now())
        return True

    def _queue_stage(self, stage_id):
        """Queue a stage straight from its card, without running it now.

        The ghost-card menu only offered "Build & run" and a build-only action, so
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
        if not self._mark_queued(job["id"], spec):
            return
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
        # Fresh queued_at: a re-queued job joins the BACK of the queue, exactly
        # as the message promises (its J-number would have jumped it ahead).
        update_job(self.project_root, job_id, status="queued", exit_code=None,
                   finished=None, started=None, queued_at=_now())
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
            job.get("params", {}), spec.get("output_params"), store=store)

        detail = ("\n".join(f"    {r}/   ({self._dir_size_human(r)})" for r in targets)
                  if targets else "    (nothing on disk yet)")
        # Spell out what is being SPARED, not just what goes. A directory shared
        # with another job is the case that matters: it means the results you can
        # see on this card will still be there afterwards, written by someone else.
        note = ("\n\nLEFT ALONE (protected, or written by another job):\n    "
                + "\n    ".join(skipped)) if skipped else ""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Clear this job's results?")
        box.setText(f"Clear {job_id} · {job.get('label', '')}?")
        box.setInformativeText(
            f"The CARD stays, with its parameters and wiring intact, and goes back "
            f"to 'Building' — change what you like, then queue or run it again.\n\n"
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
        # Back to Building (NOT queued — a cleared job must not start on its own
        # the moment the queue drains). The command is dropped so it is rebuilt
        # from whatever the parameters say NEXT time — keeping a stale command is
        # how an edited card re-runs the old one.
        update_job(self.project_root, job_id, status="building", exit_code=None,
                   started=None, finished=None, summary={}, command="",
                   interrupted=False)
        self._log(f"Cleared {job_id}"
                  + (f" — removed {', '.join(removed)}" if removed
                     else " (nothing was on disk)")
                  + ". It is back to Building; adjust its parameters, then queue "
                    "or run it.",
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
            job.get("params", {}), spec.get("output_params"), store=store)

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

    def _dir_size_human(self, rel, budget_s=1.5):
        """Rough size of a project-relative dir, capped by file count AND wall
        clock: this runs on the GUI thread inside the Clear/Delete confirmation,
        and a big jobs/J## over a slow ceph froze the app for the whole walk.
        A partial answer with a '+' is worth more than a frozen window."""
        total = n = 0
        deadline = time.monotonic() + budget_s
        try:
            for dp, _dn, fn in os.walk(Path(self.project_root) / rel):
                for f in fn:
                    try:
                        total += os.path.getsize(os.path.join(dp, f))
                    except OSError:
                        pass
                    n += 1
                    if n > 20000 or (n % 256 == 0
                                     and time.monotonic() > deadline):
                        return f"{max(total, 1e8) / 1e9:.1f}+ GB" \
                            if total >= 1e8 else "big (walk cut short)"
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

    @staticmethod
    def _dock(object_name, content):
        """Wrap content as a flat edge dock: no card border, no header, minimal
        padding. For the pipeline outline and the terminal, which are pinned to
        the window edges and should read as part of the window, not as panels
        floating over it (the frame's look lives in the app stylesheet)."""
        frame = QFrame()
        frame.setObjectName(object_name)
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
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
        proj_btn = QPushButton("Project")
        proj_btn.setToolTip(
            "Switch project without typing a path. Each bookmarked folder is "
            "listed with the projects inside it; the list is yours to edit.")
        self._project_menu = QMenu(self)
        # Rebuilt on every open, not once at startup: projects appear and vanish
        # between clicks, and a menu that re-roots the app onto a folder that is
        # no longer there is worse than no menu.
        self._project_menu.aboutToShow.connect(self._fill_project_menu)
        proj_btn.setMenu(self._project_menu)
        root_row.addWidget(set_btn)
        root_row.addWidget(browse_btn)
        root_row.addWidget(proj_btn)
        w = QWidget()
        w.setLayout(root_row)
        return w

    # ---- Project : bookmarked folders, and the projects inside them ----
    PROJECT_DIRS_KEY = "project_dirs"
    PROJECT_MENU_LIMIT = 40      # per bookmark; each probe is a few ceph stats

    def project_dirs(self):
        """Bookmarked project folders.

        Read from the SHARED file beside the code first, then from
        ~/.tomogration.json — which both migrates bookmarks made before the move
        and keeps them working if the install folder is read-only. Seeded on
        first use with the folder ABOVE the current root, because that is the
        one holding this project's siblings — the switch the button exists for.
        Absolute, de-duplicated, order preserved."""
        raw = self._load_shared_config().get(self.PROJECT_DIRS_KEY)
        if not isinstance(raw, list) or not raw:
            raw = self._load_config().get(self.PROJECT_DIRS_KEY)
        if not isinstance(raw, list) or not raw:
            raw = [str(Path(self.project_root).parent)]
        out = []
        for d in raw:
            if not isinstance(d, str) or not d.strip():
                continue
            a = os.path.abspath(os.path.expanduser(d.strip().rstrip("/")))
            if a not in out:
                out.append(a)
        return out

    def _save_project_dirs(self, dirs):
        """Write to the shared file; fall back to home if it is not writable, and
        say which happened — a bookmark that silently failed to save is exactly
        the bug this is fixing."""
        ok, where = self._save_shared_config(
            {**self._load_shared_config(), self.PROJECT_DIRS_KEY: dirs})
        if ok:
            self._log(f"Bookmarks saved to {where} — shared by every workstation "
                      f"that runs this install.", "info")
            return
        self._save_config({**self._load_config(), self.PROJECT_DIRS_KEY: dirs})
        self._log(f"{self._shared_config_path()} is not writable — bookmarks "
                  f"saved to {self._config_path()} instead, so they stay on this "
                  f"machine only.", "warning")

    def project_menu_targets(self):
        """What the Project menu offers, as [(bookmark, [(label, path), ...])].

        A bookmark counts twice over: if it is itself a project root it is a
        target under its own basename, and any of its immediate children that
        look like project roots are targets too. That covers a bookmark used as
        'my project' and one used as 'the folder my projects live in' without
        the user having to say which they meant."""
        rows = []
        for d in self.project_dirs():
            if not os.path.isdir(d):
                rows.append((d, None))          # None: missing, not merely empty
                continue
            targets = []
            if project_marker(d):
                targets.append((os.path.basename(d) or d, d))
            for name, _marker in find_projects(d, self.PROJECT_MENU_LIMIT):
                path = os.path.join(d, name)
                if path != d:
                    targets.append((name, path))
            rows.append((d, targets))
        return rows

    def _fill_project_menu(self):
        m = self._project_menu
        m.clear()
        cur = os.path.abspath(self.project_root)
        for bookmark, targets in self.project_menu_targets():
            # Section header trimmed to the tail: a full ceph path would stretch
            # the menu wider than the window. The entries carry the whole path
            # in their tooltips.
            m.addSection(self._short_path(bookmark))
            if targets is None:
                m.addAction("(folder not found)").setEnabled(False)
                continue
            if not targets:
                m.addAction("(no projects in here)").setEnabled(False)
                continue
            for label, path in targets:
                a = m.addAction(label)
                a.setCheckable(True)
                a.setChecked(os.path.abspath(path) == cur)
                a.setToolTip(path)
                a.triggered.connect(lambda _=False, pth=path: self._set_root(pth))
            if len(targets) >= self.PROJECT_MENU_LIMIT:
                # Say so rather than quietly show a partial list — a missing
                # project reads as "it's gone", which is a worse lie than a
                # long menu.
                m.addAction(f"(first {self.PROJECT_MENU_LIMIT} only — use "
                            f"Browse… for the rest)").setEnabled(False)
        m.addSeparator()
        up = str(Path(self.project_root).parent)
        if up not in self.project_dirs():
            # Only offered when it would actually do something — an action that
            # just logs "already bookmarked" is menu clutter.
            a = m.addAction(f"★  Bookmark {self._short_path(up)}",
                            lambda: self._add_project_dir(up))
            a.setToolTip(up)
        m.addAction("＋  Bookmark another folder…", self._add_project_dir)
        m.addAction("✎  Edit bookmarked folders…", self._edit_project_dirs)

    @staticmethod
    def _short_path(path):
        """Tail of a path for a menu label: a full ceph path would stretch the
        menu wider than the window. The full path goes in the tooltip."""
        parts = Path(path).parts
        return ("…/" + "/".join(parts[-2:])) if len(parts) > 3 else str(path)

    def _add_project_dir(self, path=None):
        """Bookmark a folder. Paste-first, like every other path prompt here —
        Qt's browser chokes on raw folders of thousands of files over ceph."""
        if not path:
            path = ask_project_root(self, str(Path(self.project_root).parent))
        if not path:
            return
        path = os.path.abspath(path.rstrip("/"))
        dirs = self.project_dirs()
        if path in dirs:
            self._log(f"Already bookmarked: {path}", "info")
            return
        self._save_project_dirs(dirs + [path])
        self._log(f"Bookmarked project folder: {path}", "ok")

    def _edit_project_dirs(self):
        """One path per line — add, remove and reorder in a single edit, which
        beats a list widget with three buttons for a handful of paths."""
        text, ok = QInputDialog.getMultiLineText(
            self, "Bookmarked project folders",
            "One folder per line. Each may be a project root or a folder that "
            "holds several; both are listed in the Project menu.\n"
            "Folders that no longer exist are shown greyed out, not dropped.\n"
            f"Saved to {self._shared_config_path()} — shared by every "
            f"workstation running this install.",
            "\n".join(self.project_dirs()))
        if not ok:
            return
        dirs = []
        for line in text.splitlines():
            a = line.strip().rstrip("/")
            if not a:
                continue
            a = os.path.abspath(os.path.expanduser(a))
            if a not in dirs:
                dirs.append(a)
        self._save_project_dirs(dirs)
        missing = [d for d in dirs if not os.path.isdir(d)]
        self._log(f"Project bookmarks: {len(dirs)} folder(s)."
                  + (f" {len(missing)} do not exist yet: {', '.join(missing)}"
                     if missing else ""),
                  "warning" if missing else "ok")

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
            if spec.get("legacy"):          # retired: readable on old cards only
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
            # Human titles in the sidebar too (same names as the cards); the
            # tool identity lives in the form's subtitle and the command box.
            btn = QPushButton(stage_title(spec["id"], spec["label"]))
            btn.setStyleSheet("text-align:left;")
            btn.setToolTip(stage_tool_line(spec))
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
        # Which editable job (if any) this form is acting on.
        #
        # STICKY, not one-shot — and remembered PER STAGE. It used to be a single
        # id that any form rebuild re-validated against the current stage, so
        # visiting another stage (to review a different card) dropped it for
        # good: the button quietly changed from "Run J2" back to "Run", and the
        # next click launched an untracked trunk run — or built a second job —
        # instead of running the card on screen. Now each stage keeps its own
        # binding; it survives detours and is dropped only when the job itself
        # stops being editable (runs, completes, or is deleted).
        bindings = getattr(self, "_builder_bindings", None)
        if bindings is None:
            bindings = self._builder_bindings = {}
        try:
            jobs = load_jobs(self.project_root).get("jobs") or {}
        except Exception:
            jobs = {}
        want = getattr(self, "_builder_job_id", None)
        if want:
            j = jobs.get(want)
            if j:                       # explicit bind (card click / new card)
                bindings[j.get("stage_id")] = want
        bound_job = None
        cand = bindings.get(spec["id"])
        if cand:
            j = jobs.get(cand)
            if (j and j.get("status") in RERUNNABLE
                    and j.get("stage_id") == spec["id"]):
                bound_job = cand
            else:
                bindings.pop(spec["id"], None)
        self._builder_job_id = bound_job

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

        # One-shots (a clicked card's exact values / the history banner) are
        # consumed HERE — they are singleton-form affordances; the pop-out
        # builders compute their own equivalents directly from their job.
        shown = getattr(self, "_showing_job", None)
        self._showing_job = None                      # one-shot, like bound_job
        ov = getattr(self, "_form_job_params", None)
        self._form_job_params = None
        form_params = ov["params"] if (ov and ov.get("stage") == spec["id"]) else None
        exact = (self._exact_params_for == spec["id"])
        self._exact_params_for = None
        ctx = {}
        self.current = ctx
        self._populate_builder(self.form_box, spec, ctx, bound_job=bound_job,
                               form_params=form_params, showing=shown, exact=exact)

    def _dynamic_overrides(self, spec):
        """Parameter values resolved from the PROJECT rather than the
        template: the detected gain, the mdoc's pixel size and dose, the
        newest upstream job's output dir, the next AreTomo version folder.

        These lived inside _populate_builder, so they only ever applied when a
        human OPENED a form. Every job the auto-chain built got
        _effective_params -- template defaults plus stored edits -- and none of
        this. That is why J4/J9 imported from an empty warp_frameseries, J13
        wrote to the trunk aretomo_output, and why the fix was always "click
        Rebuild from controls": Rebuild opens the form. Now both paths call it.
        """
        overrides = {}
        # Dynamic default: AreTomo output_dir = next versioned folder.
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

        # Dynamic default: RELION stages point at the NEWEST relion4/<export>
        # dir that holds a star (exports write per-job dirs now; the static
        # relion4/warp default predates them).
        if spec.get("id") == "relion4_convert":
            try:
                rel = self.project.latest_relion4_export()
                if rel:
                    overrides["project_dir"] = rel
            except (OSError, ValueError):
                pass

        # The alignment chain resolves nothing on its own -- wrapper scripts take
        # explicit dirs -- so the same answers the derive computes at job
        # CREATION have to be available as defaults, or ↻ Rebuild on an existing
        # card silently restores the trunk path.
        if spec["id"] == "aretomo":
            try:
                st = latest_stage_output_dir(load_jobs(self.project_root), "ts_stack")
            except (OSError, ValueError):
                st = ""
            if st:
                overrides["input_dir"] = f"{st.rstrip('/')}/tiltstack"
        if spec["id"] == "ts_import_alignments":
            try:
                imod = self.project.latest_aretomo_imod()
            except OSError:
                imod = ""
            if imod:
                overrides["alignments"] = imod

        # ts_import takes no --input_processing, so --frameseries is the ONLY
        # thing aiming it at the motion+CTF results. Its trunk default is empty
        # whenever that stage ran as a job (J7 wrote into its own dir and J9
        # was left reading warp_frameseries/). A dynamic default, so ↻ Rebuild
        # picks it up on a job that already exists.
        if spec["id"] == "ts_import":
            try:
                fs = latest_stage_output_dir(load_jobs(self.project_root),
                                             "fs_motion_and_ctf")
            except (OSError, ValueError):
                fs = ""
            if fs:
                overrides["frameseries"] = fs

        # Pixel size and dose are per-DATASET facts, but the stage defaults are
        # one dataset's numbers. A new project collected at another
        # magnification inherits them silently -- nothing errors, no file is
        # missing, every box downstream is just wrong. The mdoc knows.
        if spec["id"] in ("create_settings_fs", "create_settings_ts", "aretomo",
                          "ts_import_alignments"):
            try:
                acq = self.project.mdoc_acquisition()
            except OSError:
                acq = {}
            if spec["id"] == "ts_import_alignments":
                # The same quantity under another name -- the pixel size of the
                # STACKS whose shifts it imports. ts_stack can bin them, so the
                # AreTomo job's own angpix beats the mdoc's unbinned value.
                try:
                    ap = alignment_pixel_size(load_jobs(self.project_root),
                                              acq.get("angpix", ""))
                except (OSError, ValueError):
                    ap = ""
                if ap:
                    overrides["alignment_angpix"] = ap
            elif acq.get("angpix"):
                overrides["angpix"] = acq["angpix"]
            if acq.get("exposure") and spec["id"] != "aretomo":
                overrides["exposure"] = acq["exposure"]
            dims = acq.get("image_size")
            if dims and spec["id"] == "create_settings_ts":
                cur = str(next((pp.get("default") for pp in spec["params"]
                                if pp["name"] == "tomo_dimensions"), "")).lower()
                # Keep Z: tomogram thickness is the user's choice, not the
                # detector's. Only X/Y are dictated by the readout area.
                if cur.count("x") == 2:
                    overrides["tomo_dimensions"] = (
                        f"{dims[0]}x{dims[1]}x{cur.split('x')[2]}")

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
        return overrides

    def _populate_builder(self, box, spec, ctx, *, bound_job=None,
                          form_params=None, showing=None, exact=False):
        """Build the parameter form for `spec` into `box`, wiring every control to
        `ctx` — the form's OWN state dict. The singleton stage form (lists view)
        passes the dict behind self.current; each pop-out panel passes its own,
        which is what lets several builders be alive at once. `showing` renders
        the this-is-history banner, `form_params` overlays a specific job's
        recorded values, and `exact` suppresses the dynamic defaults so those
        values appear verbatim."""
        ctx.clear()
        ctx.update({"spec": spec, "controls": {}, "cmd": None, "warn": None,
                    "manual": False, "guard": False, "job_id": bound_job,
                    # How THIS form re-renders itself (reset defaults etc.); the
                    # pop-outs overwrite it with their own refresh.
                    "rebuild": lambda s=spec: self._select_stage(s)})
        controls = ctx["controls"]        # name -> getter()
        # Human title first, the actual invocation right under it in mono —
        # 'Tomogram reconstruction' reads, 'WarpTools ts_reconstruct' informs.
        title = QLabel(stage_title(spec["id"], spec["label"]))
        title.setStyleSheet("font-size:15px;font-weight:600;")
        box.addWidget(title)
        tool_line = stage_tool_line(spec)
        if tool_line:
            sub = QLabel(tool_line)
            sub.setStyleSheet(f"font-family:{MONO};font-size:10px;"
                              f"color:#8fb4d8;margin-bottom:2px;")
            box.addWidget(sub)

        # Showing a FINISHED job's recorded parameters. Say so loudly: the form is
        # otherwise indistinguishable from one you are about to run, and the values
        # in it belong to a run that already happened.
        if showing and showing.get("id"):
            if showing.get("empty"):
                msg = (f"⌛ {showing['id']} recorded no parameters — it predates the job "
                       f"model, or was adopted from disk. Its command is on the card.")
            else:
                msg = (f"⌛ Showing {showing['id']}'s recorded parameters "
                       f"({showing.get('status', '')}"
                       + (f", {showing['when']}" if showing.get("when") else "") + "). "
                       f"This is history — running from here creates a NEW job.")
            _kind = actual_job_kind(showing.get("job") or {})
            if _kind:
                msg += (f" {showing['id']} is {_kind} — it rides this stage so "
                        f"its downstream wiring works, which is why this form "
                        f"is the matching one.")
            _carried, _stale = carried_note(showing.get("dropped") or [],
                                            (showing.get("job") or {}).get("params")
                                            or {})
            for _k, _v in _carried.items():
                msg += f" {_k} = {_v} ({CARRIED_PARAMS[_k]})."
            if _stale:
                msg += (f"  Ignored {len(_stale)} parameter(s) this stage does "
                        f"not declare: {', '.join(_stale)}.")
            banner = QLabel(msg)
            banner.setWordWrap(True)
            banner.setStyleSheet(
                "background:#2c2440;border:1px solid #8b6ed6;border-radius:3px;"
                "padding:5px 8px;color:#c9b8f0;font-size:11px;")
            box.addWidget(banner)

        # Interactive tool stage: launch a window instead of running a command.
        if spec.get("tool"):
            desc = QLabel(spec["docs"].get("what", ""))
            desc.setWordWrap(True)
            box.addWidget(desc)
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
                box.addWidget(btn)
            if spec["tool"] == "inspector":
                excl_btn = QPushButton("View exclusion_list.txt (gedit)")
                excl_btn.clicked.connect(self._open_exclusion_list)
                box.addWidget(excl_btn)
            return

        overrides = {} if exact else self._dynamic_overrides(spec)

        # Effective value per param: a fresh DYNAMIC default (new-version-available)
        # wins; else the user's persisted edit; else the static template default.
        # A clicked card's recorded values arrive as `form_params` instead of
        # `stored` — shown, not committed (the per-stage store survives the view).
        stored = self._param_store.get(spec["id"], {})
        if form_params is not None:
            stored = form_params
        for p in spec.get("params", []):
            name = p["name"]
            eff = overrides[name] if name in overrides else stored.get(name)
            box.addWidget(self._param_row(p, controls, eff, ctx=ctx, spec=spec))

        warn = QLabel("")
        warn.setStyleSheet("color:#c0392b;")
        warn.setWordWrap(True)
        box.addWidget(warn)
        QTimer.singleShot(0, lambda c=ctx: self._apply_skip_if(c))

        cmd = QPlainTextEdit()
        cmd.setFixedHeight(96)
        cmd.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)   # wrap, don't scroll
        cmd.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        cmd.setStyleSheet(f"font-family:{MONO};font-size:12px;")
        cmd_lbl = QLabel("Command (editable — this is what runs):")
        cmd_lbl.setWordWrap(True)
        box.addWidget(cmd_lbl)
        box.addWidget(cmd)

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
        variants = QPushButton("⧉ Queue variants…")
        variants.setToolTip(
            "Sweep: give any parameter several values and get one card per "
            "combination, built from these settings. Nothing runs until you "
            "queue or run the cards.")
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
        buttons = [run, build_job, rebuild, enqueue, variants, reset]
        if spec.get("sync_helper"):
            fill = QPushButton("Fill: deselect all unaligned")
            fill.clicked.connect(lambda _=False, c=ctx: self._fill_sync_command(c))
            buttons.append(fill)
        for b in buttons:
            b.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            btns.addWidget(b)
        bw = QWidget()
        bw.setLayout(btns)
        box.addWidget(bw)

        ctx["cmd"] = cmd
        ctx["warn"] = warn

        # Every handler names ITS ctx explicitly (never a positional signal arg —
        # stateChanged(int) etc. would land in the ctx slot), so two forms being
        # edited in two panels can never act on each other's state.
        cmd.textChanged.connect(lambda c=ctx: self._on_cmd_edited(c))
        run.clicked.connect(
            (lambda _=False, c=ctx: self._save_and_run_job(bound_job, c))
            if bound_job else (lambda _=False, c=ctx: self._run_current(c)))
        build_job.clicked.connect(
            (lambda _=False, c=ctx: self._save_and_run_job(bound_job, c))
            if bound_job
            else (lambda _=False, c=ctx: self._build_job(spec["id"], run=True,
                                                         ctx=c)))
        rebuild.clicked.connect(lambda _=False, c=ctx: self._set_manual(False, c))
        enqueue.clicked.connect(
            (lambda _=False, c=ctx: self._save_queued_job(bound_job, c))
            if bound_job else (lambda _=False, c=ctx: self._enqueue_current(c)))
        variants.clicked.connect(lambda _=False, c=ctx: self._open_variants(c))
        reset.clicked.connect(
            lambda _=False, c=ctx: self._reset_stage_defaults(spec["id"], c))
        self._rebuild_cmd(ctx)

    def _open_variants(self, ctx=None):
        """Sweep a parameter (or several) into one card per combination."""
        cur = ctx if ctx is not None else self.current
        if not cur:
            return
        spec = cur["spec"]
        dlg = VariantsDialog(self, spec, self._values(cur))
        if not dlg.exec() or not dlg.variants:
            return
        # The ⚠ validator runs ONCE for the whole sweep. Per-variant dialogs
        # would mean answering the same question six times, and that is how a
        # real warning gets clicked through.
        flagged = []
        fn = spec.get("validate")
        if fn:
            for label, vals in zip(dlg.labels, dlg.variants):
                try:
                    msg = (fn(vals) or "").strip()
                except Exception:
                    msg = ""
                if msg:
                    flagged.append((label, msg))
        if flagged:
            shown = "\n\n".join(f"• {lab.split(' · ', 1)[-1]}\n  {msg}"
                                for lab, msg in flagged[:5])
            more = (f"\n\n(+{len(flagged) - 5} more)" if len(flagged) > 5 else "")
            if QMessageBox.warning(
                    self, f"{len(flagged)} of {len(dlg.variants)} variants have "
                    f"warnings",
                    f"{shown}{more}\n\nBuild them anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No) != QMessageBox.Yes:
                return
        built = []
        for label, vals in zip(dlg.labels, dlg.variants):
            job = self._build_job(spec["id"], params=vals, run=False,
                                  confirm=False, label=label)
            if not job:
                continue
            if dlg.queue and not self._mark_queued(job["id"], spec):
                continue
            built.append(job["id"])
        if not built:
            self._log("No variants were built.", "warn")
            return
        self._log(f"Built {len(built)} variant card(s): {', '.join(built)}"
                  + (" — all queued, they run in card order."
                     if dlg.queue else
                     " — edit any of them, then Queue variant / Run when ready."),
                  "ok")
        self._refresh_queue()
        self._refresh_canvas()


    def _view_job(self, job_id):
        """Open a finished membrane job in napari (or surforama for meshes).

        A job that segmented three tomograms has three sets of volumes, so ask
        which — twelve layers at once is not a view of anything."""
        store = load_jobs(self.project_root)
        job = (store.get("jobs") or {}).get(job_id)
        if not job:
            return
        sid = job.get("stage_id", "")
        out_dir = job.get("output_dir", "")
        spec = self._stage_by_id(sid) or {}
        params = job.get("params") or {}
        # NOT params["input_dir"]: a components job reads threshold MASKS and a
        # threshold job reads SCORE MAPS, so using a job's own input as the
        # base image opened a viewer with no tomogram in it — just the outline
        # of one. The greyscale volumes are found by breadcrumb, or by walking
        # up to the stage that actually produced reconstructions.
        input_dir = (greyscale_source(job, store)
                     or params.get("input_dir") or params.get("tomogram") or "")

        inventory = viewer_inventory(sid, out_dir, input_dir, self.project_root)
        if not inventory:
            QMessageBox.information(
                self, "Nothing to view",
                f"{job_id} has no volumes to open in {out_dir}.\n\n"
                f"Has it run? A job that failed part-way can leave the folder "
                f"empty.")
            return
        tool, env, _ = viewer_plan(sid, out_dir, None, input_dir,
                                   self.project_root)
        note = ""
        if tool == "surforama":
            note = ("Tick ONE mesh container. surforama opens a single .h5 at "
                    "a time, and the container already holds the tomogram's "
                    "densities — the mesh step projected them onto the surface "
                    "when it ran, which is what the 'Tomogram folder' setting "
                    "on that card was for. There is no separate tomogram to "
                    "open alongside it.")
        dlg = ViewerPickDialog(self, inventory, note=note,
                               title=f"{job_id} · {stage_title(sid, sid)} — "
                                     f"open in {tool}")
        if not dlg.exec() or not dlg.files:
            return
        files = dlg.files
        if tool == "surforama":
            return self._launch_surforama(job_id, files)
        script = "tomoview.py" if tool == "tomoview" else "surforama"
        # tomoview is NOT part of tomogration — it is a separate script that
        # may live in ~/bin and may not have synced with the app, so resolve it
        # rather than assuming the app folder.
        resolved = resolve_viewer_tool(
            script, str(Path(_pkg_script("tomoview.py")).parent))
        if not resolved:
            QMessageBox.warning(
                self, "Viewer not found",
                f"Could not find {script} on this machine.\n\n"
                f"Looked in the app folder, on PATH, and in ~/bin.\n\n"
                f"It ships separately from tomogration — copy it next to the "
                f"app, or put it on your PATH.")
            return
        # Through the same env wrapper every membrane tool uses: napari lives
        # in membrainseg (PyQt6), surforama in membrainpick (PyQt5).
        parts = [str(_pkg_script("ml_membrane_tool_warp_auto.sh")), resolved]
        cmd = (f"MB_CONDA_ENV={shlex.quote(env)} bash "
               + " ".join(shlex.quote(p) for p in parts) + " "
               + " ".join(shlex.quote(f) for f in files))
        self._log(f"{tool} · {job_id} ({len(files)} layer(s))", "info")
        self._log(f"$ {cmd}", "info")
        try:
            subprocess.Popen(cmd, shell=True, cwd=self.project_root)
        except OSError as e:
            self._log(f"Could not launch {tool}: {e}", "fail")

    def _launch_surforama(self, job_id, files):
        """One `membrain_pick surforama --h5-path <container>` per container.

        NOT `surforama a.h5 b.h5 ...`, which is what this used to run: the tool
        is a membrain_pick subcommand taking a single container on a named
        flag, so a positional list was never going to open anything. Each
        container is its own napari window, so opening a handful at once is
        asked about first — a mesh job writes dozens per tomogram."""
        if len(files) > 3 and QMessageBox.question(
                self, "Open several windows?",
                f"{len(files)} containers are ticked, and surforama opens ONE "
                f"napari window each.\n\nOpen all {len(files)}?") \
                != QMessageBox.StandardButton.Yes:
            return
        wrapper = str(_pkg_script("ml_membrane_tool_warp_auto.sh"))
        files = [f for f in files if str(f).lower().endswith(".h5")]
        if not files:
            QMessageBox.information(
                self, "Nothing to open",
                "surforama opens mesh containers (.h5). Nothing ticked was "
                "one.")
            return
        for f in files:
            cmd = (f"MB_CONDA_ENV=membrainpick bash {shlex.quote(wrapper)} "
                   f"membrain_pick surforama --h5-path {shlex.quote(f)}")
            self._log(f"$ {cmd}", "info")
            try:
                subprocess.Popen(cmd, shell=True, cwd=self.project_root)
            except OSError as e:
                self._log(f"Could not launch surforama: {e}", "fail")
                return
        self._log(f"surforama · {job_id} ({len(files)} container(s))", "info")

    def _analyse_sweep(self, job_id):
        """Open the analysis window for a finished parameter sweep."""
        if EX is None:
            QMessageBox.warning(self, "Analysis unavailable",
                                "ml_explore_membrane.py could not be loaded.")
            return
        job = (load_jobs(self.project_root).get("jobs") or {}).get(job_id) or {}
        out = job.get("params", {}).get("out") or job.get("output_dir", "")
        path = Path(self.project_root) / out / "results.json"
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            QMessageBox.information(
                self, "No results yet",
                f"{path} is not there.\n\nRun the sweep with 'Plan only' "
                f"unticked first — planning writes no results.")
            return
        rows = data.get("rows") or []
        # The per-virion fits live in each cell's fits.json, not in results.json,
        # and the sphere-vs-ellipsoid chart needs them.
        for r in rows:
            fit_dir = Path(self.project_root) / out / "cache" / "fit" / \
                (r.get("keys", {}).get("fit") or "")
            try:
                r["virions"] = json.loads(
                    (fit_dir / "fits.json").read_text()).get("virions", [])
            except (OSError, ValueError):
                r["virions"] = []
        dlg = AnalysisDialog(self, rows, self.project_root,
                             on_compare=self._compare_rows_in_napari,
                             on_pick=lambda j=job_id: self._view_job(j))
        dlg.exec()

    def _compare_rows_in_napari(self, rows):
        """One napari window per row — separate processes, so they sit side by
        side and neither can take the other down."""
        for r in rows:
            files = [f for f in [r.get("tomogram_path"), r.get("components_path")]
                     if f]
            files += list(r.get("fit_masks") or [])
            tool = resolve_viewer_tool("tomoview.py", str(HERE))
            if not tool:
                QMessageBox.warning(self, "Viewer not found",
                                    "tomoview.py is not on this machine.")
                return
            cmd = " ".join(shlex.quote(str(x)) for x in
                           [str(_pkg_script("ml_membrane_tool_warp_auto.sh")),
                            tool] + files)
            self._log(f"napari: {r.get('variant')} thr={r.get('threshold')} "
                      f"cut={r.get('cutoff')} {r.get('tomogram')}", "info")
            subprocess.Popen(f"MB_CONDA_ENV=membrainseg bash {cmd}", shell=True,
                             cwd=self.project_root)


    def _reset_stage_defaults(self, stage_id, ctx=None):
        """Forget this step's persisted edits and rebuild it from the template +
        dynamic defaults."""
        if self._param_store.pop(stage_id, None) is not None:
            self._persist_param_store()
            self._log(f"Reset '{stage_id}' parameters to defaults.", "info")
        # Re-render the FORM the button lives in: the singleton rebuilds via
        # _select_stage, a pop-out via the refresh its ctx carries.
        rebuild = (ctx or {}).get("rebuild")
        if rebuild is not None and ctx is not self.current:
            rebuild()
            return
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
            _atomic_write_text(self._params_path(),
                               json.dumps(self._param_store, indent=1))
        except OSError as e:
            self._log(f"Could not save parameters: {e}", "fail")

    def _param_row(self, p, controls, default_override=None, ctx=None, spec=None):
        # Compact two-line row: [human name | control] on top, help beneath.
        # The row label is a READABLE title (param_title); the exact flag/env
        # var opens the help line as a mono chip, so the wire name is one
        # glance away but no longer masquerades as the label.
        # `ctx` is the owning form's state dict — handlers must name it (see
        # _populate_builder); `spec` lets path fields coerce drops correctly.
        default = default_override if default_override is not None else p.get("default")
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        name = QLabel(param_title(p))
        name.setStyleSheet("color:#d6d6d6;font-size:12px;")
        name.setMinimumWidth(96)
        name.setToolTip(param_wire_name(p))
        row.addWidget(name)
        kind = p["kind"]
        changed = lambda *_a, c=ctx: self._on_control_changed(c)   # noqa: E731
        if kind == "check":
            cb = QCheckBox()
            cb.setChecked(bool(default))
            cb.stateChanged.connect(changed)
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
            s.valueChanged.connect(changed)
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
            cb.currentIndexChanged.connect(changed)
            controls[p["name"]] = lambda c=cb: c.currentData()
            row.addWidget(cb, 1)
        else:  # "text" or "env"
            e = QLineEdit(str(default if default is not None else ""))
            e.textChanged.connect(changed)
            controls[p["name"]] = lambda c=e: c.text()
            row.addWidget(e, 1)
            # A param that names tomograms gets a picker reading the sibling
            # folder param — nobody should have to `ls` in a terminal to learn
            # what the stems in this dataset are called.
            if p.get("pick_from"):
                pick = QPushButton("📂 Pick…")
                pick.setToolTip(f"List the .mrc in the '{p['pick_from']}' folder "
                                f"and tick the ones to name here.")
                pick.clicked.connect(
                    lambda _=False, e=e, p=p, c=controls: self._pick_tomos(e, p, c))
                row.addWidget(pick)
            elif param_is_pathish(p):
                br = QPushButton("🗂 Browse…")
                br.setToolTip("Fill this from the project's past work: every "
                              "job (newest first), RELION jobs, AreTomo "
                              "versions, the pipeline folders — then drill "
                              "into folders/files, ../ to go back.")
                br.clicked.connect(
                    lambda _=False, e=e, p=p: self._browse_inventory(e, p))
                row.addWidget(br)
            if param_is_pathish(p) or _wants_directory(p):
                # A path field is also a drop target: drag a row from any
                # popout's Outputs tab straight in — the filter replaces the
                # whole value, coerced to what this parameter wants (the
                # filter keeps a Python ref via its QObject parent, the edit).
                _PathDropFilter(e, dict(p, _spec=spec or {}), self)
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 3, 0, 3)
        outer.setSpacing(1)
        rw = QWidget()
        rw.setLayout(row)
        outer.addWidget(rw)
        help_text = p.get("help", "")
        # A GPU list is normalised to whatever the target tool wants (commas for
        # membrain/RELION, spaces for WarpTools/AreTomo), so either form is
        # correct here. Said once, in one place, for every GPU field there is
        # or ever will be — rather than in a dozen help strings that drift.
        if p.get("gpu_sep") and help_text:
            which = "commas" if p["gpu_sep"] == "," else "spaces"
            help_text += (f"  Type '0 1 2 3' or '0,1,2,3' — either is fine, "
                          f"it is rewritten with {which} for this tool.")
        if help_text:
            # Rich text: mono chip with the wire name, then the description.
            # The help itself is HTML-escaped — many strings contain <series>,
            # <Param …> etc., which Qt would otherwise swallow as markup.
            wire = _html.escape(param_wire_name(p))
            body = _html.escape(help_text)
            help_lab = QLabel(
                f"<code style='color:#8fb4d8;font-size:10px;'>{wire}</code>"
                f"&nbsp; <span style='color:#9a9a9a;font-size:11px;'>{body}"
                f"</span>")
            help_lab.setTextFormat(Qt.RichText)
            help_lab.setStyleSheet("padding-left:2px;")
            help_lab.setWordWrap(True)
            outer.addWidget(help_lab)
        box = QWidget()
        box.setLayout(outer)
        # Registered by NAME so a mode switch can grey the rows that do
        # not apply. skip_if already drops them from the command; without
        # this the form still invited you to fill in fields that would be
        # silently discarded.
        if isinstance(controls, dict):
            controls.setdefault("__rows__", {})[p["name"]] = box
        return box

    def _browse_inventory(self, edit, p):
        """Fill a path field from the inventory browser."""
        if not getattr(self, "project_root", None):
            QMessageBox.information(self, "No project",
                                    "Open a project root first.")
            return
        dlg = InventoryDialog(self, self.project_root,
                              title=f"{param_title(p)} — choose from this "
                                    f"project",
                              hint=p.get("help", ""))
        if not (dlg.exec() and dlg.chosen is not None):
            return
        chosen = dlg.chosen
        # The browser can drill into FILES, but a folder param handed a file
        # fails in the wrapper ("input dir not found") several steps later.
        # Offer the containing folder instead of letting it through.
        if _wants_directory(p):
            abs_path = Path(chosen)
            if not abs_path.is_absolute():
                abs_path = Path(self.project_root) / chosen
            if abs_path.is_file():
                parent = str(Path(chosen).parent)
                if QMessageBox.question(
                        self, "That is a file, not a folder",
                        f"'{param_title(p)}' takes a FOLDER of tomograms, but "
                        f"you picked a file:\n\n    {chosen}\n\n"
                        f"Use its folder instead?\n\n    {parent}\n\n"
                        f"(To process just this one tomogram, set the folder "
                        f"here and name the series in the tomogram list.)"
                ) == QMessageBox.Yes:
                    chosen = parent
        edit.setText(chosen)

    def _pick_tomos(self, edit, p, controls):
        """Fill a *_TOMO_LIST field by ticking what is actually on disk."""
        src = p["pick_from"]
        getter = controls.get(src)
        folder = str(getter()).strip() if getter else ""
        if not folder:
            QMessageBox.information(
                self, "No folder yet",
                f"Fill in the '{src}' field above first — that's the folder "
                f"this picker lists.")
            return
        path = Path(folder)
        if not path.is_absolute():
            path = Path(self.project_root) / folder
        if not path.is_dir():
            QMessageBox.warning(
                self, "Folder not found",
                f"{path}\n\nNothing to list yet — run the step that produces "
                f"this folder first, or point the field somewhere else.")
            return
        dlg = TomoPickDialog(self, path, edit.text(),
                             title=f"{param_title(p)} — {path.name}",
                             hint=p.get("help", ""))
        if not dlg.groups:
            QMessageBox.information(self, "No tomograms",
                                    f"No .mrc files in {path}.")
            return
        if dlg.exec() and dlg.stems is not None:
            edit.setText(" ".join(dlg.stems))    # blank = ALL, as the wrappers read it

    # ---- command-box-as-source-of-truth ----
    # Every method here takes an optional `ctx` — the form state dict to act on —
    # defaulting to self.current (the singleton stage form). Pop-out builders
    # pass their own, so several live forms cannot trample each other.
    def _values(self, ctx=None):
        cur = ctx if ctx is not None else self.current
        # __rows__ is the widget registry the skip_if greying uses, not a
        # control getter — calling it would raise on every keystroke.
        return {n: g() for n, g in cur["controls"].items()
                if n != "__rows__"}

    def _build_command(self, ctx=None):
        cur = ctx if ctx is not None else self.current
        return build_command(cur["spec"], self._values(cur),
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

    # Bookmarks are the one setting that must NOT be per-machine. Home is local
    # to each workstation here, so every VM switch lost them. They live beside
    # the code instead -- the code is on ceph and is the same install every
    # workstation launches -- while everything else (warp_launch, tm_vis_launch)
    # stays in home, where a machine-specific override belongs.
    SHARED_CONFIG_NAME = ".tomogration-shared.json"

    def _shared_config_path(self):
        return Path(__file__).resolve().parent / self.SHARED_CONFIG_NAME

    def _load_shared_config(self):
        try:
            cfg = json.loads(self._shared_config_path().read_text())
            return cfg if isinstance(cfg, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_shared_config(self, cfg):
        """-> (ok, where). A group-owned install can be read-only, so a failure
        here falls back to home rather than losing the edit."""
        try:
            _atomic_write_text(self._shared_config_path(),
                               json.dumps(cfg, indent=2))
            return (True, self._shared_config_path())
        except OSError:
            return (False, None)

    def _load_config(self):
        try:
            return json.loads(self._config_path().read_text())
        except (OSError, ValueError):
            return {}

    def _save_config(self, cfg):
        try:
            _atomic_write_text(self._config_path(), json.dumps(cfg, indent=2))
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

    def _rebuild_cmd(self, ctx=None):
        cur = ctx if ctx is not None else self.current
        cur["guard"] = True
        cur["cmd"].setPlainText(self._build_command(cur))
        cur["guard"] = False
        self._update_warning(cur)

    def _apply_skip_if(self, ctx=None):
        """Grey the rows the current settings make irrelevant.

        build_command already DROPS them, so a field left visible and editable
        was inviting you to set something that would be silently discarded —
        which is how a re-extraction went out carrying both input routes at
        once. Hidden rather than disabled: a disabled row looks identical
        in this theme and merely refuses input, which reads as a broken form."""
        cur = ctx if ctx is not None else self.current
        if not cur or not cur.get("spec"):
            return
        rows = (cur.get("controls") or {}).get("__rows__") or {}
        if not rows:
            return
        vals = self._values(cur)
        for prm in cur["spec"].get("params", []):
            w = rows.get(prm["name"])
            skip = prm.get("skip_if")
            if w is None or skip is None:
                continue
            try:
                off = bool(skip(vals))
            except Exception:
                off = False
            # HIDDEN, not disabled. A disabled row is not visibly different
            # in this theme — it just stops accepting input — so MODE A and
            # C fields stayed on screen during a MODE B run and the only
            # symptom was boxes that refused text.
            w.setVisible(not off)

    def _on_control_changed(self, ctx=None):
        cur = ctx if ctx is not None else self.current
        self._apply_skip_if(cur)
        # Persist the user's edits for this step (in-memory now, disk shortly).
        if cur and cur.get("spec"):
            self._param_store[cur["spec"]["id"]] = self._values(cur)
            self._persist_timer.start()
        if cur and not cur["manual"]:
            self._rebuild_cmd(cur)
        elif cur:
            self._update_warning(cur)

    def _on_cmd_edited(self, ctx=None):
        cur = ctx if ctx is not None else self.current
        if cur and not cur["guard"]:
            self._set_manual(True, cur)

    def _set_manual(self, manual, ctx=None):
        cur = ctx if ctx is not None else self.current
        cur["manual"] = manual
        if not manual:
            self._rebuild_cmd(cur)

    def _update_warning(self, ctx=None):
        cur = ctx if ctx is not None else self.current
        spec = cur["spec"]
        msg = spec["validate"](self._values(cur)) if spec.get("validate") else ""
        # An unresolved {placeholder} means a field points at another field that is
        # still empty. If it reaches the tool it is used LITERALLY — that is how
        # MTools ended up creating a population file named "{name}.population".
        # Checked for EVERY stage, so no future default can reintroduce this.
        try:
            cmd = cur["cmd"].toPlainText()
        except Exception:
            cmd = ""
        left = sorted(set(re.findall(r"\{(\w+)\}", cmd)) - {"jobid"})
        if left:
            names = ", ".join("{" + k + "}" for k in left)
            warn = (f"⚠ Unresolved placeholder{'s' if len(left) > 1 else ''} {names} "
                    f"— fill in the field(s) they refer to. Run as-is and the tool "
                    f"will treat it as a literal name and create the wrong file.")
            msg = warn + ("\n" + msg if msg else "")
        cur["warn"].setText(msg)

    def _fill_sync_command(self, ctx=None):
        cur = ctx if ctx is not None else self.current
        missing = self.project.tomostars_without_alignments()
        if not missing:
            self._log("All tomostars have alignments — nothing to deselect.", "ok")
            return
        vals = self._values(cur)
        settings = vals.get("settings", "warp_tiltseries.settings")
        # First command carries the WarpTools launcher (module load + conda activate);
        # the rest run in the same shell so plain 'WarpTools' is on PATH by then.
        cmds = [f"WarpTools change_selection --settings {settings} --deselect "
                f"--input_data tomostar/{n}.tomostar" for n in missing]
        cmds[0] = cmds[0].replace("WarpTools", self.warp_launch, 1)
        full = " && \\\n  ".join(cmds)
        cur["cmd"].setPlainText(full)   # marks manual (authoritative)
        self._log(f"Filled deselect command for {len(missing)} unaligned tilt series.",
                  "info")

    # ---- run / queue ----
    def _run_current(self, ctx=None):
        cur = ctx if ctx is not None else self.current
        cmd = cur["cmd"].toPlainText().strip()
        if not cmd:
            return
        spec = cur["spec"]
        # Same gate as _build_job: a live ⚠ must be acknowledged, not decorative.
        if not self._confirm_validator(spec, self._values(cur)):
            return
        # Per-job nudge: for back-half stages, prefer a job (own dir, forkable) over
        # overwriting the shared trunk. Manual command edits can't carry into a job
        # (it rebuilds from params + wiring), so only offer this on an unedited cmd.
        if spec["id"] in JOB_STAGES and not cur.get("manual"):
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
                self._build_job(spec["id"], run=True, ctx=cur)
                return
            if clicked is not overwrite:
                return
            # else: fall through to the classic overwrite-in-place path
        cmd = self._stage_prerun(spec, cmd, self._values(cur), confirmed=True)
        if cmd is None:
            return
        # A trunk (▶ Run) re-extract has no job record — stash its values so
        # _on_finished can register the output pick-set card on success.
        self._pending_reextract = (
            dict(self._values(cur))
            if spec["id"] in ("relion4_to_warp", "relion4_select_picks") else None)
        # A trunk run has no job to resolve {jobid} (only build_job_command does
        # that), and bash passes the braces through literally — Warp would then
        # export into a directory named 'relion4/{jobid}/'. Give the run a
        # collision-free trunk tag instead.
        if "{jobid}" in cmd:
            cmd = cmd.replace(
                "{jobid}",
                "trunk_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        self._dispatch(spec["id"], cmd, fresh=True)

    def _stage_prerun(self, spec, cmd, values, interactive=True, confirmed=False):
        """Every safety hook a stage declares, applied in ONE place for EVERY
        dispatch path — trunk ▶ Run, queueing, job cards, and Run now. The job
        paths used to skip all of them, so a miss-alignment card dropped from
        the palette would happily refine raw stacks (the exact featureless-
        tomogram failure the gate exists to block) and an AreTomo card ran with
        no PARAMETERS.txt audit. Returns the (possibly rewritten) command, or
        None to block the run.

        interactive=False is the queue-drain path: no dialogs (an overnight
        queue must not stall on a prompt), but hard blocks still hold."""
        if interactive and not confirmed and not self._confirm_validator(spec,
                                                                         values):
            return None
        if spec.get("id") == "ts_export_particles" and \
                not self._check_direct_export(values, interactive):
            return None
        # AreTomo's hand-off, run AFTER a refinement that has already written
        # its geometry into the same XMLs. It would overwrite the refined
        # alignment with the coarse one -- and, finding no .xf, DESELECT every
        # series, which Warp persists: ts_ctf and ts_reconstruct then process
        # nothing and report success. Silent, and expensive to discover.
        if spec.get("id") == "ts_import_alignments":
            try:
                refined = refined_since_last_import(load_jobs(self.project_root))
            except (OSError, ValueError):
                refined = ""
            if refined:
                msg = (f"{refined} already refined these alignments with "
                       f"miss-alignment, which writes straight into the Warp "
                       f"XMLs — there are no .xf files left to import.\n\n"
                       f"Importing now replaces the refined geometry with "
                       f"AreTomo's coarse one, and every series it cannot find "
                       f"a .xf for is marked UNSELECTED — a state Warp keeps, "
                       f"so ts_ctf and ts_reconstruct would then quietly "
                       f"process nothing.\n\nRun this BEFORE miss-alignment, "
                       f"not after.")
                if interactive:
                    if QMessageBox.question(
                            self, "Import alignments after refining?",
                            msg + "\n\nImport anyway?",
                            QMessageBox.Yes | QMessageBox.No,
                            QMessageBox.No) != QMessageBox.Yes:
                        return None
                else:
                    self._log("ts_import_alignments blocked: " + msg.replace("\n", " "),
                              "fail")
                    return None
        if spec.get("requires_coarse_alignment"):
            if interactive:
                if not self._coarse_alignment_gate():
                    return None
            elif not self.project.latest_aretomo_imod():
                self._log("miss-alignment blocked: no AreTomo alignment found. "
                          "It refines a coarse alignment — run AreTomo → "
                          "ts_import_alignments → select first.", "fail")
                return None
            cmd = self._missalign_chain_cmd(cmd, values)
        if spec.get("aretomo"):
            self._prepare_aretomo_run(cmd, values)
        if spec.get("clean_overrides"):
            self._prepare_rename_run(values)
        if spec.get("normalize_exclusions"):
            self._normalize_exclusions()
        return cmd

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
        # Ask the question that matters: is the alignment IN THE XML, where
        # miss-alignment will look for it? A .xf on disk proves only that
        # AreTomo ran -- J17 refined nothing with a full aretomo_output folder
        # sitting right there, because nothing had imported it. And an
        # alignment that arrived another way has no .xf to find at all.
        missing = self.project.series_without_imported_alignment()
        if not missing:
            return True
        total = len(list((self.project.root / "tomostar").glob("*.tomostar"))) \
            if (self.project.root / "tomostar").is_dir() else 0
        every = total and len(missing) >= total
        head = ", ".join(missing[:5]) + (" …" if len(missing) > 5 else "")
        return QMessageBox.question(
            self, "Some series carry no alignment to refine",
            f"{len(missing)}"
            + (f" of {total}" if total else "")
            + f" tilt series have no alignment in their warp_tiltseries/*.xml "
              f"(e.g. {head}).\n\n"
            + ("Every series is unaligned — that usually means the import step "
               "has not run yet, or wrote somewhere else. miss-alignment would "
               "refine nothing and the tomograms would come out featureless.\n\n"
               if every else
               "miss-alignment refines what is already in the XML; these would "
               "come out garbage. The rest are fine.\n\n")
            + "Continue anyway?") == QMessageBox.Yes

    def _missalign_chain_cmd(self, cmd, vals):
        """For a FRESH miss-alignment run, prepend the AreTomo import + 'select all' so
        AreTomo's computed tilt axis + shifts are written into the Warp .xml right
        before miss-alignment refines them — making the enforced order a single step.
        Skipped on resume (MA_START_ITER > 0) so it never clobbers refinement in
        progress, and when the chain is ALREADY in the command (a stored queued
        command must not get it twice).

        alignment_angpix is the pixel size of the STACKS AreTomo aligned, since
        that is the unit its .xf shifts are in. It was hardcoded 1.57 -- EML46's
        number -- so EML50 (1.98) would import every shift 26% short, with no
        error anywhere: the run succeeds and the tomograms are quietly
        misaligned. Read it from the mdoc."""
        start = str(vals.get("MA_START_ITER", "0")).strip() or "0"
        imod = self.project.latest_aretomo_imod()
        if start != "0" or not imod or "ts_import_alignments" in cmd:
            return cmd
        try:
            angpix = alignment_pixel_size(
                load_jobs(self.project_root),
                self.project.mdoc_acquisition().get("angpix", ""))
        except (OSError, ValueError):
            angpix = ""
        if not angpix:
            # Refuse rather than guess. The old fallback was a literal "1.57";
            # importing shifts at the wrong pixel size raises no error anywhere
            # and leaves the tomograms quietly misaligned, which is far worse
            # than a step that says it cannot proceed.
            self._log("Cannot determine the pixel size AreTomo aligned at (no "
                      "AreTomo job on record and no PixelSpacing in the mdocs). "
                      "Run ts_import_alignments yourself with an explicit "
                      "--alignment_angpix, then re-run this step.", "fail")
            return cmd
        pre = (f"module load miniconda/latest && conda activate warp && "
               f"WarpTools ts_import_alignments --settings warp_tiltseries.settings "
               f"--alignments {imod} --alignment_angpix {angpix} && "
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

    def _prepare_rename_run(self, vals):
        """Quarantine identical Tomo5 *_override.mdoc in the rename source dir
        before the rename script runs (it would otherwise consume their numbers)."""
        src_rel = str(vals.get("source_dir", ".")).strip() or "."
        src = self.project.root / src_rel
        if not src.is_dir():
            return
        removed, kept = self.project.clean_override_mdocs(src)
        if removed or kept:
            self._log(f"Override mdocs in {src_rel}: {removed} identical → mdocs/bad/"
                      + (f", {kept} differing left in place (check them)" if kept else ""),
                      "warning" if kept else "ok")

    def _prepare_aretomo_run(self, cmd, vals):
        """Create the versioned output folder and drop its PARAMETERS.txt audit
        before AreTomo launches (brief §4.6 / §5)."""
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

    def _enqueue_current(self, ctx=None):
        cur = ctx if ctx is not None else self.current
        cmd = cur["cmd"].toPlainText().strip()
        if not cmd:
            return
        spec = cur["spec"]
        # Queueing is the easiest way to ignore a warning: you set it up, walk away,
        # and it runs overnight — so the full hook set applies here too.
        cmd = self._stage_prerun(spec, cmd, self._values(cur))
        if cmd is None:
            return
        # Queueing creates a REAL job (status='queued') carrying its resolved command,
        # so it shows on the canvas as a card and survives a restart.
        n = len(queued_jobs(load_jobs(self.project_root))) + 1
        job = new_job(self.project_root, spec["id"],
                      f'{stage_title(spec["id"], spec["id"])} (queued {n})',
                      self._values(cur), inputs={})
        update_job(self.project_root, job["id"], status="queued", command=cmd,
                   trunk=True, queued_at=_now())
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

    def _run_job(self, job_id, confirmed=False):
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
        # The same safety hooks as a trunk run: the job path used to skip them
        # all, letting a palette-dropped miss-alignment card run on raw stacks.
        cmd = self._stage_prerun(spec, cmd, job.get("params", {}),
                                 confirmed=confirmed)
        if cmd is None:
            return
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
        # One-shot post-run hook (e.g. the crYOLO converter's bookkeeping) —
        # consumed unconditionally so a vanished job can't leave it armed for
        # the next, unrelated run.
        hook = getattr(self, "_job_finalize_hook", None)
        self._job_finalize_hook = None
        store = load_jobs(self.project_root)
        job = store.get("jobs", {}).get(job_id)
        if job is None:
            return
        status = "completed" if code == 0 else "failed"
        # An exit code is a claim, not evidence. WarpTools prints its help and
        # exits 0 on an unknown option, so a job that wrote nothing was being
        # recorded green and the pipeline built on top of the absence.
        # isinstance, not a truth test: the Qt stub answers any unknown
        # attribute with a dummy object, and a dummy is truthy — it would fail
        # every job that finished before this attribute was first set.
        false_ok = getattr(self, "_false_success", None)
        self._false_success = None
        if not isinstance(false_ok, str):
            false_ok = None
        if code == 0 and false_ok:
            status = "failed"
        # A DECLARED partial exit: some items failed, the rest produced output.
        # Only stages carrying `partial_exit` get this reading -- `exit 2` means
        # "edit the config and run again" in other wrappers, a hard failure.
        tally = getattr(self, "_batch_tally", None)
        self._batch_tally = {}
        if not isinstance(tally, dict):
            tally = {}
        part_status, part_note = partial_batch_result(
            self._stage_by_id(job["stage_id"]) or {}, code, tally)
        if part_status:
            status = part_status
        summary = summarize_job(job["stage_id"],
                                Path(self.project_root) / job["output_dir"])
        # M writes no summary file, so its result only exists as the line _log
        # caught. Consume it here (and clear it, so the next job cannot inherit
        # the previous run's number).
        res = getattr(self, "_m_resolution", None)
        self._m_resolution = None
        if res is not None and code == 0:
            summary["resolution_A"] = f"{res:g}"
        # Polarity's whole result is one word in the registry, so without this
        # its card reads "no outputs" after a successful measurement.
        # isinstance, not "is not None": the Qt stub answers any unknown
        # attribute with a dummy object, so a plain None-check writes that
        # dummy into the job summary and json.dump then refuses it.
        pol = getattr(self, "_polarity_result", None)
        self._polarity_result = None
        if isinstance(pol, str) and code == 0:
            summary["polarity"] = pol
        # Work a successful run threw away. On the CARD, not only in the log:
        # J7 exited 0 after dropping a movie whose CTF fit diverged, and the
        # only trace was one line an hour back in an 4145-item run.
        warn_counts = getattr(self, "_run_warn_counts", None)
        self._run_warn_counts = {}
        if not isinstance(warn_counts, dict):
            warn_counts = {}
        warnings = run_warning_report(warn_counts)
        for key, n, line, why in warnings:
            summary[key] = n
        # On the CARD, not only in the log: what this batch lost, and kept.
        if part_status and tally.get("failed"):
            summary["items failed"] = tally["failed"]
        if part_status and tally.get("ok"):
            summary["items done"] = tally["ok"]
        update_job(self.project_root, job_id, status=status, exit_code=code,
                   finished=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   summary=summary)
        self._log(f"--- {job_id} {status} (exit {code}) ---",
                  "ok" if status == "completed" else "fail")
        if part_note:
            self._log(f"{job_id}: {part_note}",
                      "warning" if part_status == "completed" else "fail")
        if false_ok and code == 0:
            self._log(f"{job_id} exited 0 but did nothing — recorded as FAILED. "
                      f"{false_ok}", "fail")
        for _key, _n, line, why in warnings:
            self._log(f"{job_id}: {line} — {why}", "warn")
        hint = getattr(self, "_failure_hint", None)
        self._failure_hint = None          # never inherited by the next run
        if hint and code != 0:
            self._log(hint, "warn")
        self._set_status(
            f"{'\u2713' if code == 0 else '\u2717'} {job_id} {status} (exit {code})")
        if code == 0:
            self._bridge_job_outputs(job_id, job.get("stage_id"))
        # A re-extract run as a job: register its output pick-set card too.
        if job.get("stage_id") in ("relion4_to_warp", "relion4_select_picks"):
            self._maybe_register_reextract(job.get("params", {}), code)
        if hasattr(self, "_refresh_canvas"):     # Phase 2 hook; harmless until then
            self._refresh_canvas()
        if hook:
            hook(job_id, code)

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
            # A job finishing changes what's on disk — drop the cached per-stage
            # sweep or the rail badges stay stale for up to STATUS_TTL.
            self._invalidate_status()
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
            # Refuse, don't warn: when the run finishes, _on_finished would
            # write its exit code into the NEW root's history (whatever record
            # sits at that index) and fail to find its job record — the old
            # root's job would be stranded at 'running' forever.
            self._log("A job is still running — switching the project root now "
                      "would record its result in the wrong project. TERMINATE "
                      "it or let it finish first.", "fail")
            self.root_edit.setText(self.project_root)
            return
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
        # Builder bindings name jobs of the OLD root — J-numbers are per-project.
        self._builder_bindings = {}
        self._builder_job_id = None
        self._form_job_params = None
        # Pop-out panels show the OLD root's jobs; a J-number means something
        # else in the new one, so close them rather than let them mislead.
        for pop in list(getattr(self, "_popouts", {}).values()):
            try:
                pop.close()
            except Exception:
                pass
        self._popouts = {}
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
        n_filed = self.project.collect_coarse_stacks()
        if n_filed:
            self._log(f"Moved {n_filed} loose Position*.mrc from the project root "
                      f"into {COARSE_DIR}/.", "ok")
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
        n_total = sum(len(plan[k]) for k in ("frames", "mdocs", "gains",
                                             COARSE_DIR, "overrides"))
        if n_total == 0:
            QMessageBox.information(
                self, "Nothing to sort",
                f"No .eer / .mdoc / gain / tilt-stack files found directly in:"
                f"\n{src}\n\n"
                f"({len(plan['skipped'])} other files would be left in place.)")
            return
        if QMessageBox.question(
                self, "Sort files into project?",
                f"From:\n{src}\n\nMove into {self.project_root}:\n"
                f"  • .eer  → frames/ : {len(plan['frames'])}\n"
                f"  • .mdoc → mdocs/  : {len(plan['mdocs'])}\n"
                f"  • gain  → gains/  : {len(plan['gains'])}\n"
                f"  • tilt stacks → {COARSE_DIR}/ : {len(plan[COARSE_DIR])}\n\n"
                f"Tomo5 *_override.mdoc found: {len(plan['overrides'])} — those "
                f"identical to their standard mdoc go to mdocs/bad/; any that "
                f"differ are left in place and flagged.\n\n"
                f"Leave in place: {len(plan['skipped'])} other files "
                f"(Thumbnails, Session.dm…).\n\n"
                f"Existing files are never overwritten. Proceed?"
        ) != QMessageBox.Yes:
            return
        moved = self.project.sort_files(src)
        moved[COARSE_DIR] += self.project.collect_coarse_stacks()
        self._log(f"Sorted raw files from {src}: "
                  f"{moved['frames']} → frames/, {moved['mdocs']} → mdocs/, "
                  f"{moved['gains']} → gains/, "
                  f"{moved[COARSE_DIR]} → {COARSE_DIR}/; "
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

        # ---- TABS: the app log + any number of scratch shell tabs ------------
        # The terminal is a full-height side rail now, so it can afford several
        # lines of enquiry at once: tab 0 is the app's own run log (always
        # there), ＋ adds an independent shell tab in the project root.
        self.term_tabs = QTabWidget()
        self.term_tabs.setDocumentMode(True)
        self.term_tabs.setTabsClosable(True)
        self.term_tabs.tabCloseRequested.connect(self._close_console_tab)
        add_tab = QToolButton()
        add_tab.setText("＋")
        add_tab.setToolTip("New shell tab (its commands run in the project root, "
                           "on their own process — never the job runner)")
        add_tab.clicked.connect(self._add_console_tab)
        self.term_tabs.setCornerWidget(add_tab, Qt.TopRightCorner)

        log_page = QWidget()
        lv = QVBoxLayout(log_page)
        lv.setContentsMargins(0, 4, 0, 0)
        lv.setSpacing(4)
        bar = QHBoxLayout()
        bar.setSpacing(6)
        bar.addStretch(1)
        find = QLineEdit()
        find.setPlaceholderText("find…")
        find.setFixedWidth(110)
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
        lv.addWidget(barw)

        self.term = QPlainTextEdit()
        self.term.setReadOnly(True)
        self.term.setMaximumBlockCount(20000)
        self.term.setTextInteractionFlags(Qt.TextSelectableByMouse
                                          | Qt.TextSelectableByKeyboard)
        # Flat and near-black on purpose: the dock has no card chrome, so the
        # log reads as the window's own coding background, not a panel on it.
        self.term.setStyleSheet(
            f"background:#0c0c0c;color:#ddd;border:none;"
            f"font-family:{MONO};font-size:12px;"
            f"selection-background-color:#2d4a6b;")
        lv.addWidget(self.term, 1)

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
        lv.addWidget(roww)
        self._console_hist = []
        self._console_pos = 0

        self.term_tabs.addTab(log_page, "log")
        try:      # the app log is not closable — drop its ✕ button
            from PySide6.QtWidgets import QTabBar
            self.term_tabs.tabBar().setTabButton(0, QTabBar.RightSide, None)
            self.term_tabs.tabBar().setTabButton(0, QTabBar.LeftSide, None)
        except Exception:
            pass
        v.addWidget(self.term_tabs, 1)

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

    def _add_console_tab(self):
        tab = _ConsoleTab(lambda: getattr(self, "project_root", None))
        idx = self.term_tabs.addTab(tab, f"sh{self.term_tabs.count()}")
        self.term_tabs.setCurrentIndex(idx)
        tab.input.setFocus()

    def _close_console_tab(self, idx):
        if idx == 0:
            return                      # the app log is not closable
        w = self.term_tabs.widget(idx)
        self.term_tabs.removeTab(idx)
        if w is not None:
            w.deleteLater()

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

    def moveEvent(self, ev):
        """Dragging the main window moves every card's screen position — the
        tethered panels follow (guarded no-op before the canvas exists)."""
        super().moveEvent(ev)
        self._retether_popouts()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._retether_popouts()

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

    def _run_queued_job(self, job_id, interactive=False, confirmed=False):
        """Run a QUEUED job: its command was resolved when it was queued, so it runs
        verbatim (no rebuild/re-wiring — what you queued is what runs). Tracked as a
        job so _finalize_job records the outcome and the card turns green/red.

        interactive=True marks a user-initiated launch (▶ Run now, the builder's
        run button): safety hooks may prompt. The queue-drain path never prompts —
        a hook that blocks there fails the job visibly instead of stalling the
        queue on a dialog nobody is at the keyboard to answer."""
        if self.runner.busy():
            # Same guard as _run_job/_dispatch: starting a second QProcess would
            # rebind self.proc, destroy the live process uncleanly, and strand
            # its record at 'running'. Reachable mid-run via ▶ Run now.
            self._log(f"A job is already running — {job_id} stays queued and "
                      f"runs when the current one finishes (or TERMINATE it "
                      f"first).", "warning")
            return
        store = load_jobs(self.project_root)
        job = (store.get("jobs", {}) or {}).get(job_id)
        if not job:
            self._log(f"queued job {job_id} not found.", "fail")
            return
        cmd = job.get("command") or ""
        rebuilt = False
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
                rebuilt = True
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
        # Safety hooks for non-trunk cards. Trunk-queued STORED commands are
        # exempt — their hooks ran at enqueue time and the command already
        # carries any prep — but a trunk command REBUILT here never got them
        # (that is how a builder-saved miss-alignment job lost its import
        # chain), so a rebuild always goes through the hooks.
        if rebuilt or not job.get("trunk"):
            spec = self._stage_by_id(job.get("stage_id", ""))
            if spec is not None:
                cmd2 = self._stage_prerun(spec, cmd, job.get("params", {}),
                                          interactive=interactive,
                                          confirmed=confirmed)
                if cmd2 is None:
                    if not interactive:
                        # Fail it visibly and let the queue move on.
                        update_job(self.project_root, job_id, status="failed",
                                   exit_code=-1)
                        self._refresh_queue()
                        self._refresh_canvas()
                        self._run_queue()
                    return
                if cmd2 != cmd:
                    cmd = cmd2
                    update_job(self.project_root, job_id, command=cmd)
        update_job(self.project_root, job_id, status="running", exit_code=None,
                   finished=None,
                   started=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self._active_job_id = job_id
        self._m_resolution = None
        self._run_warn_counts = {}
        self._batch_tally = {}
        self._false_success = None
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
        # ...and for a failure we can explain. Stashed here rather than parsed
        # in _finalize_job because the app keeps no log tail: by the time the
        # process exits, the line that said WHY is gone.
        hint = run_failure_hint(text)
        if hint:
            self._failure_hint = hint
        # ...and for work a SUCCESSFUL run quietly discarded. Counted, not
        # stashed: one deselected movie and four hundred are different findings.
        # A batch wrapper's summary tally, for stages that can end part-done.
        tally = batch_tally_line(text)
        if tally:
            if not isinstance(getattr(self, "_batch_tally", None), dict):
                self._batch_tally = {}
            self._batch_tally[tally[0]] = tally[1]
        # ...and for a run that will exit 0 having done nothing at all.
        why = false_success_reason(text)
        if why:
            self._false_success = why
        wkey = run_warning_key(text)
        if wkey:
            counts = getattr(self, "_run_warn_counts", None)
            if not isinstance(counts, dict):
                counts = {}
            counts[wkey] = counts.get(wkey, 0) + 1
            self._run_warn_counts = counts
        # MCore reports what a refinement achieved on ONE stdout line at the end and
        # writes it nowhere. Catch it here so _finalize_job can put it on the card —
        # otherwise a column of M jobs is unreadable and the only way to compare
        # rounds is to scroll back through the log.
        if level == "out" and getattr(self, "_active_job_id", None):
            res = m_resolution(text)
            if res is not None:
                self._m_resolution = res
            pol = polarity_result(text)
            if pol is not None:
                self._polarity_result = pol
        colours = {"out": "#dddddd", "err": "#e0a850", "info": "#7fb4ff",
                   "ok": "#27ae60", "success": "#27ae60", "warning": "#e0a850",
                   "warn": "#e0a850",   # both spellings occur at call sites
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
