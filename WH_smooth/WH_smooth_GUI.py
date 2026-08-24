"""
wh_range_smoother.py v1.0.2 (08/21/2026)
────────────────────
Standalone GUI — Whittaker-Henderson smoothing over selected q ranges.

Supports:  I(q)  |  S(q)  |  F(q)   (.chi / .iq / .sq / .fq / .txt / .dat)

Workflow:
  1. Load a data file
  2. Select data type (I(q) / S(q) / F(q))
  3. Add one or more q ranges for smoothing
  4. Set WH parameters (lambda, order)
  5. Apply → inspect Original vs Smoothed + Difference plots
  6. Save result

Plot navigation (both the main and difference plots):
  * Left-drag       : draw a rectangle to zoom into that region (box-zoom).
  * Double-click    : reset the view to full auto-range (zoom out).
  * Right-drag      : pyqtgraph's built-in continuous zoom.
  * Z key           : toggle the left-drag behaviour between Box-Zoom and Pan.
  * R key           : auto-range (reset zoom), same as double-click.
  Note: while a "Manual Spline" range is in point-editing mode, the main plot
  temporarily switches to Pan mode so left-clicks add anchor points instead of
  drawing a zoom box; box-zoom is restored automatically when editing ends.

Requirements: PySide6, numpy, scipy, pyqtgraph
Usage       : python wh_range_smoother.py

Author      : Gihan Kwon
Affiliation : National Synchrotron Light Source II (NSLS-II),
              Brookhaven National Laboratory, Upton, NY, USA
Contact     : gkwon@bnl.gov
Version     : 1.0.2
License     : (add your license here)
"""

__author__ = "Gihan Kwon"
__version__ = "1.0.2"

import sys
import os
import time
import json
import copy
import warnings
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.interpolate import CubicSpline
import scipy.sparse as sp

# Suppress pyqtgraph internal log-axis overflow warning
# (occurs harmlessly when log mode is toggled or before data is plotted)
warnings.filterwarnings(
    "ignore",
    message="overflow encountered in power",
    category=RuntimeWarning,
    module="pyqtgraph.*"
)
np.seterr(over='ignore')

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog,
    QScrollArea, QFrame, QMessageBox, QSplitter,
    QGroupBox, QSpinBox, QButtonGroup, QRadioButton,
    QCheckBox, QComboBox, QListWidget, QListWidgetItem,
    QAbstractItemView, QSizePolicy
)
from PySide6.QtGui import QKeySequence, QFont, QAction, QPixmap, QIcon
from PySide6.QtCore import Qt, QSettings, QTimer
import pyqtgraph as pg


def _resource_path(relative_path):
    """Return the absolute path to a bundled resource.

    Works both when running from source and from a PyInstaller bundle.
    PyInstaller sets sys._MEIPASS to the folder holding bundled files; when
    running from source we fall back to this file's directory.
    """
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


# ══════════════════════════════════════════════════════════════
#  Whittaker-Henderson core
# ══════════════════════════════════════════════════════════════

def _make_dtd(n: int, d: int):
    D = sp.eye(n, format='csc')
    for _ in range(d):
        D = D[1:] - D[:-1]
    return D.T @ D


def smooth_whittaker(y: np.ndarray,
                     lambda_: float = 1000.0,
                     order: int = 2) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < order + 2:
        raise ValueError(f"Segment too short (n={n}) for order={order}.")
    DTD = _make_dtd(n, order)
    A   = np.eye(n) + lambda_ * DTD.toarray()
    c, low = cho_factor(A)
    return cho_solve((c, low), y)


def _cosine_blend(orig_seg, new_seg, blend_pts):
    """Cosine-taper blend between two arrays of the same length."""
    n = len(orig_seg)
    if blend_pts <= 0 or blend_pts * 2 >= n:
        return new_seg
    b = int(blend_pts)
    w = np.ones(n)
    w[:b]   = 0.5 * (1 - np.cos(np.pi * np.arange(b) / b))   # 0 → 1
    w[n-b:] = 0.5 * (1 + np.cos(np.pi * np.arange(b) / b))   # 1 → 0
    return w * new_seg + (1 - w) * orig_seg


def _linear_interp_seg(q_seg, q_full, y_full, mask):
    """Linear interpolation between boundary points just outside the range.
    Performed in log10 space when all anchor values are positive,
    so the result follows the true log-linear baseline trend."""
    idxs = np.where(mask)[0]
    i0, i1 = idxs[0], idxs[-1]
    left  = i0 - 1 if i0 > 0 else i0
    right = i1 + 1 if i1 < len(q_full) - 1 else i1
    q_anchor = [q_full[left], q_full[right]]
    y_anchor = [y_full[left], y_full[right]]

    if y_anchor[0] > 0 and y_anchor[1] > 0:
        # Log-space interpolation: fit line in log10(y) vs q
        log_interp = np.interp(q_seg, q_anchor,
                               [np.log10(y_anchor[0]), np.log10(y_anchor[1])])
        return 10.0 ** log_interp
    else:
        return np.interp(q_seg, q_anchor, y_anchor)


def _spline_interp_seg(q_seg, q_full, y_full, mask, anchor_pts=10):
    """Cubic spline through 2N anchor points (N each side of the range).
    Performed in log10 space when all anchor values are positive,
    so the spline follows the smooth log-scale envelope of the data."""
    idxs = np.where(mask)[0]
    i0, i1 = idxs[0], idxs[-1]
    n = int(max(2, anchor_pts))
    left_lo  = max(0, i0 - n)
    left_hi  = i0
    right_lo = i1 + 1
    right_hi = min(len(q_full), i1 + 1 + n)

    q_anc = np.concatenate([q_full[left_lo:left_hi], q_full[right_lo:right_hi]])
    y_anc = np.concatenate([y_full[left_lo:left_hi], y_full[right_lo:right_hi]])

    if len(q_anc) < 4:
        return _linear_interp_seg(q_seg, q_full, y_full, mask)

    if np.all(y_anc > 0):
        cs = CubicSpline(q_anc, np.log10(y_anc), bc_type='natural',
                         extrapolate=True)
        return 10.0 ** cs(q_seg)
    else:
        cs = CubicSpline(q_anc, y_anc, bc_type='natural', extrapolate=True)
        return cs(q_seg)


def _manual_spline_seg(q_seg, manual_pts):
    """Fit a cubic spline through user-defined (q, y) anchor points.
    Fitting is performed in log10 space when all y values are positive.
    Duplicate or too-close q values are merged automatically."""
    pts = sorted(manual_pts, key=lambda p: p[0])
    q_pts = np.array([p[0] for p in pts])
    y_pts = np.array([p[1] for p in pts])

    # Remove duplicate / too-close q values (keep last y for each unique q)
    # Threshold: points within 1e-6 of each other are merged
    q_unique, idx_unique = np.unique(np.round(q_pts, 6), return_index=True)
    q_pts = q_pts[idx_unique]
    y_pts = y_pts[idx_unique]

    if len(q_pts) < 2:
        return np.full(len(q_seg), y_pts[0] if len(y_pts) else 1.0)

    # Ensure strictly increasing (extra safety)
    diffs = np.diff(q_pts)
    if np.any(diffs <= 0):
        keep = np.concatenate([[True], diffs > 0])
        q_pts = q_pts[keep]
        y_pts = y_pts[keep]

    if len(q_pts) < 2:
        return np.full(len(q_seg), y_pts[0])

    if len(q_pts) == 2:
        if np.all(y_pts > 0):
            log_interp = np.interp(q_seg, q_pts, np.log10(y_pts))
            return 10.0 ** log_interp
        else:
            return np.interp(q_seg, q_pts, y_pts)

    # 3+ points: cubic spline with 'natural' boundary condition
    # (zero second derivative at endpoints → smooth, no overshoot at boundaries)
    if np.all(y_pts > 0):
        cs = CubicSpline(q_pts, np.log10(y_pts), bc_type='natural',
                         extrapolate=True)
        return 10.0 ** cs(q_seg)
    else:
        cs = CubicSpline(q_pts, y_pts, bc_type='natural', extrapolate=True)
        return cs(q_seg)


def range_smooth(q: np.ndarray,
                 y: np.ndarray,
                 ranges: list) -> np.ndarray:
    """
    Apply per-range processing with cosine-taper blending at the boundaries.

    Each entry in `ranges` is a dict:
      {
        'q0', 'q1'         : range bounds (Å⁻¹)
        'mode'             : 'wh' | 'linear' | 'spline'
        'blend'            : cosine-taper width (points, both sides)
        'lambda', 'order'  : WH parameters (mode='wh')
        'anchor_pts'       : anchor count per side (mode='spline')
      }
    """
    result = y.copy()
    for r in ranges:
        q0, q1 = r['q0'], r['q1']
        mode   = r.get('mode', 'wh')
        blend  = r.get('blend', 0)

        mask = (q >= q0) & (q <= q1)
        n = mask.sum()
        if n < 4:
            continue

        if mode == 'wh':
            lam   = r.get('lambda', 1000.0)
            order = r.get('order', 2)
            if n < order + 2:
                continue
            new_seg = smooth_whittaker(y[mask], lam, order)
        elif mode == 'linear':
            new_seg = _linear_interp_seg(q[mask], q, y, mask)
        elif mode == 'spline':
            anc = r.get('anchor_pts', 10)
            new_seg = _spline_interp_seg(q[mask], q, y, mask, anchor_pts=anc)
        elif mode == 'manual':
            pts = r.get('manual_pts', [])
            if len(pts) < 2:
                continue
            new_seg = _manual_spline_seg(q[mask], pts)
        else:
            continue

        result[mask] = _cosine_blend(y[mask], new_seg, blend)

    return result


# ══════════════════════════════════════════════════════════════
#  File loader  — auto-detects header rows
# ══════════════════════════════════════════════════════════════

def load_file(path: str):
    """Load a two-column data file, automatically skipping any header.

    Handles the file variety produced by common reduction tools:
      * plain two-column data with no header (.dat / .xy / .txt)
      * pyFAI / Fit2d style headers with '#' comment lines (.xy)
      * headers prefixed with '#', '!', ';', '%', '*', '//' or 'x'/'q' titles
      * Windows (CRLF) or Unix line endings
      * non-UTF-8 bytes in the header (e.g. the 'µ' in 'µm'), which would
        otherwise crash a strict UTF-8 read
      * whitespace-, tab-, comma- or semicolon-separated columns (.csv/.tsv)
      * stray NaN / inf rows, which are skipped

    Only the first two numeric columns are used; any extra columns
    (uncertainties, etc.) are ignored. Every line is tried as data, so header
    rows are skipped automatically no matter how many there are.

    Returns
    -------
    (x, y) : tuple of numpy.ndarray
        The first two numeric columns of the file.
    """
    # Read as latin-1 so any byte decodes without error; we only need the
    # numbers, and latin-1 maps every byte 1:1.
    with open(path, 'r', encoding='latin-1') as f:
        lines = f.readlines()

    x_vals = []
    y_vals = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        # Skip obvious comment / header lines (common comment markers).
        if s[0] in ('#', '!', ';', '%', '*', '@') or s.startswith('//'):
            continue
        # Normalise separators: treat commas, semicolons and tabs like spaces
        # so whitespace-, comma-, semicolon- and tab-separated files all work.
        parts = s.replace(',', ' ').replace(';', ' ').replace('\t', ' ').split()
        if len(parts) < 2:
            continue
        # Quick reject: if the first token doesn't start like a number, this is
        # a header/title line — skip without the cost of a full float() attempt.
        first = parts[0]
        if not (first[0].isdigit() or first[0] in '+-.'):
            continue
        try:
            a = float(parts[0])
            b = float(parts[1])
        except ValueError:
            # Any remaining header text (e.g. column titles like "q  I") is
            # skipped automatically because it fails to parse as a number.
            continue
        # Drop non-finite rows so they cannot corrupt later math / plotting.
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        x_vals.append(a)
        y_vals.append(b)

    if len(x_vals) < 2:
        raise ValueError(
            f"No two-column numeric data found in:\n{os.path.basename(path)}\n\n"
            "The file needs at least two numeric columns (q and intensity). "
            "Header and comment lines are skipped automatically. If this is a "
            "session (.whses) or other non-data file, open it via the correct "
            "menu instead."
        )

    return np.asarray(x_vals, dtype=float), np.asarray(y_vals, dtype=float)


# ══════════════════════════════════════════════════════════════
#  Q-range row widget
# ══════════════════════════════════════════════════════════════

def _urls_to_paths(mime):
    """Extract local file paths from a drag-and-drop mime payload."""
    paths = []
    if mime.hasUrls():
        for url in mime.urls():
            p = url.toLocalFile()
            if p:
                paths.append(p)
    return paths


class DropListWidget(QListWidget):
    """QListWidget that accepts files dropped from the OS file manager.

    On drop it calls `on_files_dropped(list_of_paths)` if that callback is set.

    Hover vs click (when `hover_select` is True):
      * moving the mouse over a sample row calls `on_hover_preview(row)`.
        In the sample list this activates that file and its smoothing workspace.
      * clicking a row also confirms the current selection.
      * leaving the list keeps the active sample workspace intact.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_files_dropped = None
        self.on_hover_preview = None
        self.on_click_select = None
        self.on_hover_leave = None
        self.hover_select = False
        self._last_hover_row = -1
        self.setAcceptDrops(True)
        # QListWidget is a QAbstractScrollArea: mouse events are delivered through
        # its viewport.  Enable tracking on BOTH the view and viewport so hover
        # works immediately after startup even when no plot window exists yet.
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        # itemEntered is more reliable than relying on mouseMoveEvent alone on
        # Windows/Qt when entering an already-selected row.  mouseMoveEvent stays
        # as a fallback; _last_hover_row prevents duplicate callbacks.
        self.itemEntered.connect(self._on_item_entered)

    def _dispatch_hover_item(self, it):
        if not self.hover_select or it is None:
            return
        row = self.row(it)
        if row != self._last_hover_row:
            self._last_hover_row = row
            if callable(self.on_hover_preview):
                self.on_hover_preview(row)

    def _on_item_entered(self, it):
        self._dispatch_hover_item(it)

    def mouseMoveEvent(self, event):
        if self.hover_select:
            try:
                pt = event.position().toPoint()
            except AttributeError:
                pt = event.pos()
            self._dispatch_hover_item(self.itemAt(pt))
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        # Mouse left the list: forget the hover preview and let the controller
        # revert to the confirmed (clicked) selection.
        self._last_hover_row = -1
        if self.hover_select and callable(self.on_hover_leave):
            self.on_hover_leave()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)   # let Qt update the current row
        it = self.itemAt(event.position().toPoint()
                         if hasattr(event, "position") else event.pos())
        if it is not None and callable(self.on_click_select):
            self.on_click_select(self.row(it))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        paths = _urls_to_paths(event.mimeData())
        if paths and callable(self.on_files_dropped):
            self.on_files_dropped(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class DropLineEdit(QLineEdit):
    """QLineEdit that accepts a single dropped file (e.g. the reference file).

    On drop it calls `on_file_dropped(path)` with the first dropped file.

    Note: a *read-only* QLineEdit silently rejects drops on some platforms, so
    this widget stays editable but blocks keyboard editing (see keyPressEvent)
    to behave like a read-only field while still accepting dropped files.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_file_dropped = None
        self.on_clear_requested = None
        self._block_typing = False
        self.setAcceptDrops(True)

    def set_display_only(self, on=True):
        """Prevent keyboard editing while still accepting drops."""
        self._block_typing = on

    def keyPressEvent(self, event):
        # Delete / Backspace clears the field (e.g. remove the reference),
        # even in display-only mode, so the user can clear it from the keyboard.
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if callable(self.on_clear_requested):
                self.on_clear_requested()
                event.accept()
                return
        if self._block_typing:
            event.ignore()
            return
        super().keyPressEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        paths = _urls_to_paths(event.mimeData())
        if paths and callable(self.on_file_dropped):
            self.on_file_dropped(paths[0])
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


# ══════════════════════════════════════════════════════════════

class QRangeRow(QWidget):
    """A single q-range row with a mode selector and dynamic parameters."""
    MODES = ["WH Smooth", "Linear Interp", "Spline Interp", "Manual Spline"]

    def __init__(self, default_lambda=1000, default_order=2, parent=None):
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 2, 0, 4)
        outer.setSpacing(2)

        # ── Row 1: enable + q range + mode + delete ──
        row1 = QHBoxLayout()
        row1.setSpacing(4)

        self.chk = QCheckBox()
        self.chk.setChecked(True)
        self.chk.setToolTip("Enable / disable this range")

        self.q0 = QLineEdit()
        self.q0.setPlaceholderText("q start")
        self.q0.setFixedWidth(72)

        self.q1 = QLineEdit()
        self.q1.setPlaceholderText("q end")
        self.q1.setFixedWidth(72)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(self.MODES)
        self.mode_combo.setFixedWidth(125)
        self.mode_combo.setToolTip(
            "WH Smooth   : Whittaker-Henderson smoothing (noise removal)\n"
            "Linear Interp: replace range with straight line between boundaries\n"
            "Spline Interp: replace range with cubic spline through anchor points")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        self.btn_del = QPushButton("✕")
        self.btn_del.setFixedWidth(24)
        self.btn_del.setStyleSheet("color:#c0392b; font-weight:bold;")
        self.btn_del.setToolTip("Remove this range")

        for w in (self.chk, QLabel("q:"), self.q0, QLabel("~"), self.q1,
                  QLabel("Å⁻¹"), self.mode_combo, self.btn_del):
            row1.addWidget(w)
        row1.addStretch()

        # ── Row 2: parameters (mode-dependent) ──
        row2 = QHBoxLayout()
        row2.setSpacing(4)
        row2.addSpacing(22)

        # WH-only widgets
        self.lambda_label = QLabel("λ:")
        self.lambda_edit  = QLineEdit(str(default_lambda))
        self.lambda_edit.setFixedWidth(80)

        self.order_label = QLabel("ord:")
        self.order_spin  = QSpinBox()
        self.order_spin.setRange(1, 2147483647)
        self.order_spin.setValue(default_order)
        self.order_spin.setFixedWidth(58)

        # Spline-only widget
        self.anchor_label = QLabel("anchor pts:")
        self.anchor_spin  = QSpinBox()
        self.anchor_spin.setRange(2, 2147483647)
        self.anchor_spin.setValue(10)
        self.anchor_spin.setFixedWidth(65)
        self.anchor_spin.setToolTip(
            "Number of anchor points on EACH side of the range\n"
            "used to fit the cubic spline through the gap")

        # Blend (all modes)
        self.blend_label = QLabel("blend:")
        self.blend_spin  = QSpinBox()
        self.blend_spin.setRange(0, 2147483647)
        self.blend_spin.setValue(20)
        self.blend_spin.setFixedWidth(65)
        self.blend_spin.setToolTip(
            "Cosine-taper blend width at each boundary (in points)\n"
            "0 = hard cut  |  20-50 = smooth transition")

        # Manual Spline widgets
        self.manual_btn = QPushButton("📍 Edit Points")
        self.manual_btn.setFixedWidth(105)
        self.manual_btn.setCheckable(True)
        self.manual_btn.setStyleSheet(
            "QPushButton:checked { background:#E84855; color:white; font-weight:bold; }")
        self.manual_pts_label = QLabel("0 pts")
        self.manual_pts_label.setFixedWidth(40)
        self.manual_undo_btn = QPushButton("Undo")
        self.manual_undo_btn.setFixedWidth(48)
        self.manual_undo_btn.setToolTip("Remove the most recently added point")
        self.manual_undo_btn.clicked.connect(self._undo_manual_pt)
        self.manual_clear_btn = QPushButton("Clear")
        self.manual_clear_btn.setFixedWidth(48)
        self.manual_clear_btn.clicked.connect(self._clear_manual_pts)
        self._manual_pts = []          # list of (q, y) tuples (sorted by q)
        self._manual_add_order = []    # (q, y) in the order they were added

        for w in (self.lambda_label, self.lambda_edit,
                  self.order_label, self.order_spin,
                  self.anchor_label, self.anchor_spin,
                  self.blend_label, self.blend_spin,
                  self.manual_btn, self.manual_pts_label,
                  self.manual_undo_btn,
                  self.manual_clear_btn):
            row2.addWidget(w)
        row2.addStretch()

        # ── Separator ──
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #333;")

        outer.addLayout(row1)
        outer.addLayout(row2)
        outer.addWidget(sep)

        # Set initial visibility for default mode (WH)
        self._on_mode_changed()

    def _refresh_pts_label(self):
        self.manual_pts_label.setText(f"{len(self._manual_pts)} pts")

    def _clear_manual_pts(self):
        self._manual_pts.clear()
        self._manual_add_order.clear()
        self._refresh_pts_label()
        # Note: does NOT toggle off edit mode, so you can keep placing points.

    def _undo_manual_pt(self):
        """Remove the most recently added point (button + convenience)."""
        self.remove_last_point()

    def remove_last_point(self):
        """Drop the last-added point. Returns the removed (q, y) or None."""
        if not self._manual_add_order:
            return None
        pt = self._manual_add_order.pop()
        # Remove the matching entry from the sorted list.
        try:
            self._manual_pts.remove(pt)
        except ValueError:
            # Fall back: nearest match by value if exact tuple isn't found.
            if self._manual_pts:
                self._manual_pts.pop()
        self._refresh_pts_label()
        return pt

    def remove_point_near(self, q, y, tol_frac=0.02):
        """Remove the point closest to (q, y) in normalised plot space.

        Distances are scaled by the current point spread so the pick works
        regardless of the very different q- and y-magnitudes. Returns the
        removed (q, y) tuple, or None if nothing was close enough.
        """
        if not self._manual_pts:
            return None
        qs = [p[0] for p in self._manual_pts]
        ys = [p[1] for p in self._manual_pts]
        q_span = (max(qs) - min(qs)) or 1.0
        y_span = (max(ys) - min(ys)) or 1.0

        best_i, best_d = -1, None
        for i, (pq, py) in enumerate(self._manual_pts):
            dq = (pq - q) / q_span
            dy = (py - y) / y_span
            d = dq * dq + dy * dy
            if best_d is None or d < best_d:
                best_d, best_i = d, i

        # Require the click to be reasonably close (within tol_frac of the
        # normalised span) so an empty-space click doesn't delete a far point.
        if best_i < 0 or best_d > (tol_frac * 5) ** 2:
            return None
        pt = self._manual_pts.pop(best_i)
        try:
            self._manual_add_order.remove(pt)
        except ValueError:
            pass
        self._refresh_pts_label()
        return pt

    def add_manual_point(self, q, y):
        self._manual_pts.append((q, y))
        self._manual_add_order.append((q, y))
        self._manual_pts.sort(key=lambda p: p[0])
        self._refresh_pts_label()

    def _on_mode_changed(self):
        mode = self.mode_combo.currentText()
        is_wh     = (mode == "WH Smooth")
        is_spline = (mode == "Spline Interp")
        is_manual = (mode == "Manual Spline")
        self.lambda_label.setVisible(is_wh)
        self.lambda_edit.setVisible(is_wh)
        self.order_label.setVisible(is_wh)
        self.order_spin.setVisible(is_wh)
        self.anchor_label.setVisible(is_spline)
        self.anchor_spin.setVisible(is_spline)
        self.manual_btn.setVisible(is_manual)
        self.manual_pts_label.setVisible(is_manual)
        self.manual_undo_btn.setVisible(is_manual)
        self.manual_clear_btn.setVisible(is_manual)
        # Blend visible for all modes

    def get_range(self, for_display=False):
        """Return a dict describing this range, or None if disabled/invalid.

        With `for_display=True` the range is returned even when a Manual Spline
        row has fewer than 2 points, so its highlighted region still shows on
        the plot (it just can't be *processed* yet). For actual smoothing
        (`for_display=False`) an incomplete Manual row returns None.
        """
        if not self.chk.isChecked():
            return None
        try:
            q0 = float(self.q0.text().replace(",", "."))
            q1 = float(self.q1.text().replace(",", "."))
            if not (q0 < q1):
                return None
            mode_label = self.mode_combo.currentText()
            mode = {"WH Smooth": "wh",
                    "Linear Interp": "linear",
                    "Spline Interp": "spline",
                    "Manual Spline": "manual"}[mode_label]
            d = {"q0": q0, "q1": q1, "mode": mode,
                 "blend": self.blend_spin.value()}
            if mode == "wh":
                d["lambda"] = float(self.lambda_edit.text().replace(",", "."))
                d["order"]  = self.order_spin.value()
            elif mode == "spline":
                d["anchor_pts"] = self.anchor_spin.value()
            elif mode == "manual":
                if len(self._manual_pts) < 2 and not for_display:
                    return None
                d["manual_pts"] = list(self._manual_pts)
            return d
        except ValueError:
            return None

    def serialize(self):
        """Return a plain-dict snapshot of every field, for session saving.

        Unlike get_range(), this captures the row even when it is disabled or
        incomplete, so the exact UI state is restored on the next launch.
        """
        return {
            "enabled":    self.chk.isChecked(),
            "q0":         self.q0.text(),
            "q1":         self.q1.text(),
            "mode_label": self.mode_combo.currentText(),
            "lambda":     self.lambda_edit.text(),
            "order":      self.order_spin.value(),
            "anchor_pts": self.anchor_spin.value(),
            "blend":      self.blend_spin.value(),
            "manual_pts": list(self._manual_pts),
        }

    def apply_state(self, s: dict):
        """Restore this row from a serialize() snapshot."""
        try:
            self.chk.setChecked(bool(s.get("enabled", True)))
            self.q0.setText(str(s.get("q0", "")))
            self.q1.setText(str(s.get("q1", "")))
            label = s.get("mode_label", "WH Smooth")
            if label in self.MODES:
                self.mode_combo.setCurrentText(label)
            self.lambda_edit.setText(str(s.get("lambda", "1000")))
            self.order_spin.setValue(int(s.get("order", 2)))
            self.anchor_spin.setValue(int(s.get("anchor_pts", 10)))
            self.blend_spin.setValue(int(s.get("blend", 20)))
            self._manual_pts = [tuple(p) for p in s.get("manual_pts", [])]
            # Restored points keep their saved order as the undo order.
            self._manual_add_order = list(self._manual_pts)
            self.manual_pts_label.setText(f"{len(self._manual_pts)} pts")
            self._on_mode_changed()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════
#  Standalone plot window (one per sample selection)
# ══════════════════════════════════════════════════════════════

class PlotWindow(QMainWindow):
    """A detached window that shows Original-vs-Smoothed and Difference plots
    for a single snapshot of data.

    Each time a sample is selected in the main window, a fresh PlotWindow is
    created and shown. It is self-contained (its own copy of the arrays), so
    earlier windows keep displaying their own data for side-by-side comparison.

    Interaction is view-only: left-drag box-zoom, double-click reset,
    right-drag zoom, plus a live crosshair with a coordinate read-out.
    """

    _counter = 0   # running index used in the window title

    def __init__(self, snapshot: dict, parent=None):
        super().__init__(parent)
        PlotWindow._counter += 1
        self._idx = PlotWindow._counter
        self._is_result = bool(snapshot.get("is_result", False))
        self.resize(900, 720)

        _icon_path = _resource_path("WH.ico")
        if os.path.exists(_icon_path):
            self.setWindowIcon(QIcon(_icon_path))

        # Manual-editing state (set later via set_editor()).
        self._editor_row = None
        self._controller = None
        self._last_click_time = 0.0

        self._load_snapshot(snapshot)
        self._build()
        self._draw()

    def _load_snapshot(self, s: dict, set_title=True):
        """Copy a snapshot dict into instance fields.

        When `set_title` is True the window title is updated to reflect the
        file. On plain refreshes (Apply, parameter edits) we keep the title so a
        window always shows the file it was opened for.
        """
        if set_title:
            title = s.get("title", "data")
            self.setWindowTitle(f"WH_smooth Plot [{self._idx}] — {title}")
        self._q          = s.get("q")
        self._y_raw      = s.get("y_raw")
        self._y          = s.get("y")
        self._y_bkg      = s.get("y_bkg_scaled")
        self._y_smoothed = s.get("y_smoothed")
        self._alpha      = s.get("alpha", 1.0)
        self._data_type  = s.get("data_type", "I(q)")
        self._q_ref      = s.get("q_ref")
        self._y_ref      = s.get("y_ref")
        self._ref_scale  = s.get("ref_scale", 1.0)
        self._use_log    = s.get("use_log", False)
        self._ranges     = s.get("ranges", [])
        self._manual     = s.get("manual", [])   # list of (color, [(q,y),...])

    def update_snapshot(self, s: dict, reset_view=False):
        """Replace the displayed data with a new snapshot and redraw.

        By default the current zoom / pan of BOTH plots is preserved ('Apply
        Smoothing' and live edits refresh in place). Pass reset_view=True when
        the underlying data changes entirely (a different file is loaded) so the
        axes refit to the new data instead of leaving the curve off-screen.
        """
        prev_log = self._use_log
        had_smoothed = self._y_smoothed is not None

        # Remember the current view ranges so they can be restored after redraw.
        main_range = self.plot_main.getViewBox().viewRange()
        diff_range = self.plot_diff.getViewBox().viewRange()

        self._load_snapshot(s, set_title=reset_view)
        # Only touch the log mode if it actually changed, so the view is not
        # reset on every apply.
        log_changed = (self._use_log != prev_log)
        if log_changed:
            want_log = (self._use_log)
            self.plot_main.setLogMode(x=False, y=want_log)
        self.plot_main.clear()
        self.plot_diff.clear()
        # Re-add crosshair overlay items removed by clear().
        self.plot_main.addItem(self._vline, ignoreBounds=True)
        self.plot_main.addItem(self._hline, ignoreBounds=True)
        self.plot_main.addItem(self._coord, ignoreBounds=True)
        self.plot_diff.addItem(self._vline_d, ignoreBounds=True)
        self.plot_diff.addItem(self._hline_d, ignoreBounds=True)
        self.plot_diff.addItem(self._coord_d, ignoreBounds=True)

        if reset_view:
            # New data: fit both axes to it (curve is guaranteed on-screen).
            self._draw(reset_view=True)
            return
        # Keep the user's current zoom — do not rescale the axes.
        self._draw(reset_view=False)

        if log_changed:
            # Log↔linear changes the y-scale entirely, so the old y-range is
            # meaningless — refit y to the data while keeping the x zoom.
            self._reset_view()
            self.plot_main.getViewBox().setXRange(
                main_range[0][0], main_range[0][1], padding=0)
        else:
            # Restore the exact view ranges (x AND y) so nothing rescales.
            self.plot_main.getViewBox().setRange(
                xRange=main_range[0], yRange=main_range[1], padding=0)
        now_smoothed = self._y_smoothed is not None
        if had_smoothed and now_smoothed and not log_changed:
            # Difference plot already had data: keep the user's view.
            self.plot_diff.getViewBox().setRange(
                xRange=diff_range[0], yRange=diff_range[1], padding=0)
        elif now_smoothed:
            # First smoothed result (or log toggle): anchor diff at q = 0.
            if self._q is not None and len(self._q) > 1:
                self.plot_diff.setXRange(0, float(self._q[-1]), padding=0.02)

    def set_editor(self, row, controller):
        """Enable/disable manual-point editing in this window.

        `row` is the QRangeRow whose points are being edited (or None to
        disable). `controller` is the main window, which actually mutates the
        row and refreshes state.
        """
        self._editor_row = row
        self._controller = controller
        if row is not None:
            self.statusBar().showMessage(
                "Editing manual points here — LEFT-CLICK add, RIGHT-CLICK delete")

    def clear_plot(self):
        """Empty both plots but keep the window open (source file removed)."""
        self._editor_row = None
        self._manual = []
        self._manual_scatter = []
        self._q = self._y = self._y_raw = self._y_bkg = self._y_smoothed = None
        self.plot_main.clear()
        self.plot_diff.clear()
        # Re-add the crosshair overlay items removed by clear().
        for it in (self._vline, self._hline, self._coord):
            self.plot_main.addItem(it, ignoreBounds=True)
        for it in (self._vline_d, self._hline_d, self._coord_d):
            self.plot_diff.addItem(it, ignoreBounds=True)
        self.statusBar().showMessage("Data removed — plot cleared.")

    # ── UI ──
    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        lay = QVBoxLayout(central)
        lay.setContentsMargins(4, 4, 4, 4)

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        self.plot_main = pg.PlotWidget(title="Original vs Smoothed")
        self.plot_main.setLabel('bottom', "q", units="Å⁻¹")
        self.plot_main.setLabel('left', self._data_type)
        self.plot_main.showGrid(x=True, y=True, alpha=0.3)
        self.plot_main.getViewBox().setLimits(xMin=0)
        if self._use_log:
            self.plot_main.setLogMode(x=False, y=True)

        self.plot_diff = pg.PlotWidget(title="Difference  (Original − Smoothed)")
        self.plot_diff.setLabel('bottom', "q", units="Å⁻¹")
        self.plot_diff.setLabel('left', "Δ")
        self.plot_diff.showGrid(x=True, y=True, alpha=0.3)
        self.plot_diff.getViewBox().setLimits(xMin=0)
        self.plot_diff.setMaximumHeight(240)
        self.plot_diff.setMinimumHeight(120)

        lay.addWidget(self.plot_main, 4)
        lay.addWidget(self.plot_diff, 1)

        # Crosshair + coordinate read-out on the main plot.
        self._coord = pg.TextItem("", anchor=(1, 1), color='#0055AA')
        self._coord.setFont(pg.Qt.QtGui.QFont("Consolas", 10,
                                               pg.Qt.QtGui.QFont.Weight.Bold))
        self._coord.setZValue(100)
        self._vline = pg.InfiniteLine(angle=90, movable=False,
            pen=pg.mkPen('#AAAAAA', width=1, style=Qt.PenStyle.DashLine))
        self._hline = pg.InfiniteLine(angle=0, movable=False,
            pen=pg.mkPen('#AAAAAA', width=1, style=Qt.PenStyle.DashLine))
        self.plot_main.addItem(self._vline, ignoreBounds=True)
        self.plot_main.addItem(self._hline, ignoreBounds=True)
        self.plot_main.addItem(self._coord, ignoreBounds=True)
        self.plot_main.scene().sigMouseMoved.connect(self._on_move)
        self.plot_main.scene().sigMouseClicked.connect(self._on_click)

        # Crosshair + coordinate read-out on the difference plot.
        self._coord_d = pg.TextItem("", anchor=(1, 1), color='#0055AA')
        self._coord_d.setFont(pg.Qt.QtGui.QFont("Consolas", 10,
                                                 pg.Qt.QtGui.QFont.Weight.Bold))
        self._coord_d.setZValue(100)
        self._vline_d = pg.InfiniteLine(angle=90, movable=False,
            pen=pg.mkPen('#AAAAAA', width=1, style=Qt.PenStyle.DashLine))
        self._hline_d = pg.InfiniteLine(angle=0, movable=False,
            pen=pg.mkPen('#AAAAAA', width=1, style=Qt.PenStyle.DashLine))
        self.plot_diff.addItem(self._vline_d, ignoreBounds=True)
        self.plot_diff.addItem(self._hline_d, ignoreBounds=True)
        self.plot_diff.addItem(self._coord_d, ignoreBounds=True)
        self.plot_diff.scene().sigMouseMoved.connect(self._on_move_diff)

        # Box-zoom by default. Double-click is intentionally NOT bound to
        # reset, because in Manual-Spline editing a double-click would also add
        # a point. Use the R key to reset the view instead (see keyPressEvent).
        for pw in (self.plot_main, self.plot_diff):
            vb = pw.getViewBox()
            vb.setMouseMode(pg.ViewBox.RectMode)
            pw.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.statusBar().showMessage(
            "Drag: Box-Zoom   R: Reset view   Right-drag: Zoom")

    def keyPressEvent(self, event):
        """R resets both plots to the DATA range.

        The reset is computed from the actual curves (not the manual anchor
        points), so stray/edge anchor dots can't stretch the view.
        """
        if event.key() == Qt.Key.Key_R:
            self._reset_view()
            self.statusBar().showMessage("View reset (R)")
            event.accept()
            return
        super().keyPressEvent(event)

    def _reset_view(self):
        """Fit both plots to the data curves, anchored at q = 0."""
        if self._q is None or len(self._q) < 2:
            self.plot_main.getViewBox().autoRange()
            self.plot_diff.getViewBox().autoRange()
            return

        qmax = float(self._q[-1])

        # ── Main plot y-range from the visible curves only ──
        ys = []
        if self._y_bkg is not None:
            # When a background was used, all three same-grid curves are visible:
            # raw sample, scaled background, and processed/subtracted data.
            for arr in (self._y_raw, self._y_bkg, self._y):
                if arr is not None:
                    ys.append(np.asarray(arr, dtype=float))
        else:
            arr = self._y_raw if self._y_raw is not None else self._y
            if arr is not None:
                ys.append(np.asarray(arr, dtype=float))
        if self._y_smoothed is not None:
            ys.append(np.asarray(self._y_smoothed, dtype=float))
        if self._q_ref is not None and self._y_ref is not None:
            ys.append(np.asarray(self._y_ref, dtype=float) * self._ref_scale)

        if ys:
            allv = np.concatenate(ys)
            allv = allv[np.isfinite(allv)]
            if self._use_log:
                allv = allv[allv > 0]
            if allv.size:
                lo, hi = float(np.min(allv)), float(np.max(allv))
                if self._use_log:
                    lo, hi = np.log10(lo), np.log10(hi)
                pad = 0.05 * (hi - lo) if hi > lo else 1.0
                self.plot_main.getViewBox().setRange(
                    xRange=(0, qmax), yRange=(lo - pad, hi + pad), padding=0)
            else:
                self.plot_main.setXRange(0, qmax, padding=0.02)
        else:
            self.plot_main.setXRange(0, qmax, padding=0.02)

        # ── Difference plot ──
        if self._y_smoothed is not None and self._y is not None:
            diff = np.asarray(self._y, dtype=float) - np.asarray(self._y_smoothed, dtype=float)
            diff = diff[np.isfinite(diff)]
            if diff.size:
                lo, hi = float(np.min(diff)), float(np.max(diff))
                pad = 0.05 * (hi - lo) if hi > lo else 1.0
                self.plot_diff.getViewBox().setRange(
                    xRange=(0, qmax), yRange=(lo - pad, hi + pad), padding=0)
            else:
                self.plot_diff.setXRange(0, qmax, padding=0.02)
        else:
            self.plot_diff.setXRange(0, qmax, padding=0.02)

    def _on_click(self, event):
        """Left-click adds / right-click deletes a manual point (edit mode)."""
        if self._editor_row is None or self._controller is None:
            return
        now = time.time()
        if now - self._last_click_time < 0.3:
            return
        self._last_click_time = now
        from pyqtgraph.Qt import QtCore as _QC
        btn = event.button()
        vb = self.plot_main.getViewBox()
        pt = vb.mapSceneToView(event.scenePos())
        q_c, y_c = pt.x(), pt.y()
        if self._use_log:
            if -300 < y_c < 300:
                y_c = 10.0 ** y_c
            else:
                return
        if btn == _QC.Qt.MouseButton.RightButton:
            self._controller._manual_point_deleted(self._editor_row, q_c, y_c)
        elif btn == _QC.Qt.MouseButton.LeftButton:
            self._controller._manual_point_added(self._editor_row, q_c, y_c)
        else:
            return
        # Redraw this window's manual scatter to reflect the change.
        self._manual = self._controller._current_manual_snapshot()
        self._redraw_manual()
        event.accept()

    def _redraw_manual(self):
        """Remove and re-add the manual scatter overlay without changing view.

        Adding scatter items can trigger pyqtgraph's auto-range and make the
        plot jump/zoom out while placing points, so the current view range is
        saved and restored around the update.
        """
        vb = self.plot_main.getViewBox()
        cur_range = vb.viewRange()
        if not hasattr(self, "_manual_scatter"):
            self._manual_scatter = []
        for sc in self._manual_scatter:
            try:
                self.plot_main.removeItem(sc)
            except Exception:
                pass
        self._manual_scatter = []
        for col, pts in self._manual:
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            if self._use_log:
                ys = [np.log10(v) if v > 0 else 0 for v in ys]
            sc = pg.ScatterPlotItem(x=xs, y=ys, symbol='o', size=12,
                pen=pg.mkPen(col, width=2), brush=pg.mkBrush(col + 'AA'))
            sc.setZValue(100)
            self.plot_main.addItem(sc, ignoreBounds=True)
            self._manual_scatter.append(sc)
        # Restore the exact view so placing a point never rescales the plot.
        vb.setRange(xRange=cur_range[0], yRange=cur_range[1], padding=0)

    def _on_move(self, pos):
        vb = self.plot_main.getViewBox()
        if not self.plot_main.sceneBoundingRect().contains(pos):
            self._vline.setVisible(False)
            self._hline.setVisible(False)
            self._coord.setText("")
            return
        pt = vb.mapSceneToView(pos)
        x, y = pt.x(), pt.y()
        self._vline.setVisible(True)
        self._hline.setVisible(True)
        self._vline.setPos(x)
        self._hline.setPos(y)
        if self._use_log:
            if -300 < y < 300:
                txt = f" q={x:.4f} Å⁻¹   y={10.0 ** y:.6g} "
            else:
                txt = f" q={x:.4f} Å⁻¹   y=(out of range) "
        else:
            txt = f" q={x:.4f} Å⁻¹   y={y:.6g} "
        vr = vb.viewRange()
        self._coord.setPos(vr[0][1], vr[1][0])
        self._coord.setText(txt)

    def _on_move_diff(self, pos):
        """Crosshair + coordinate read-out for the difference plot."""
        vb = self.plot_diff.getViewBox()
        if not self.plot_diff.sceneBoundingRect().contains(pos):
            self._vline_d.setVisible(False)
            self._hline_d.setVisible(False)
            self._coord_d.setText("")
            return
        pt = vb.mapSceneToView(pos)
        x, y = pt.x(), pt.y()
        self._vline_d.setVisible(True)
        self._hline_d.setVisible(True)
        self._vline_d.setPos(x)
        self._hline_d.setPos(y)
        # Difference plot is always linear.
        txt = f" q={x:.4f} Å⁻¹   Δ={y:.6g} "
        vr = vb.viewRange()
        self._coord_d.setPos(vr[0][1], vr[1][0])
        self._coord_d.setText(txt)

    # ── Drawing ──
    def _draw(self, reset_view=True):
        leg = self.plot_main.addLegend()
        leg.setLabelTextSize('12pt')
        leg.setLabelTextColor('k')
        leg.setBrush(pg.mkBrush(0, 0, 0, 0))
        leg.setPen(pg.mkPen(None))
        # Pin the legend to the TOP-RIGHT corner (inset slightly from the axes)
        # so it doesn't cover the data on the left.
        try:
            leg.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-10, 10))
        except Exception:
            pass

        if self._q is None or self._y is None:
            return

        if self._y_bkg is not None:
            # A saved/reloaded result is self-contained: show the exact curves
            # that were present at save time, regardless of data type.
            raw_to_plot = self._y_raw if self._y_raw is not None else self._y
            if raw_to_plot is not None:
                self.plot_main.plot(self._q, raw_to_plot,
                    pen=pg.mkPen('#000000', width=1.8),
                    name=f"Raw {self._data_type}  [sample]")
            self.plot_main.plot(self._q, self._y_bkg,
                pen=pg.mkPen('#1565C0', width=1.8),
                name=f"α×Background  (α={self._alpha:g})")
            self.plot_main.plot(self._q, self._y,
                pen=pg.mkPen('#E84855', width=1.8),
                name=f"Processed {self._data_type}")
        else:
            raw_to_plot = self._y_raw if self._y_raw is not None else self._y
            self.plot_main.plot(self._q, raw_to_plot,
                pen=pg.mkPen('#000000', width=1.8),
                name=f"Raw / Original {self._data_type}")

        if self._y_smoothed is not None:
            self.plot_main.plot(self._q, self._y_smoothed,
                pen=pg.mkPen('#7B2D8B', width=1.8), name="Smoothed")

        if self._q_ref is not None and self._y_ref is not None:
            self.plot_main.plot(self._q_ref, self._y_ref * self._ref_scale,
                pen=pg.mkPen('#2E7D32', width=1.8, style=Qt.PenStyle.DashLine),
                name=f"Reference × {self._ref_scale:g}")

        # Highlight the q ranges.
        for r in self._ranges:
            mode = r.get("mode", "wh")
            brush = {"wh":     pg.mkBrush(255, 180,   0, 50),
                     "linear": pg.mkBrush( 50, 180, 255, 50),
                     "manual": pg.mkBrush(255, 100,  50, 50),
                     "spline": pg.mkBrush(180,  80, 255, 50)}.get(
                         mode, pg.mkBrush(180, 80, 255, 50))
            lr = pg.LinearRegionItem([r["q0"], r["q1"]], movable=False,
                                     brush=brush)
            lr.setZValue(-10)
            self.plot_main.addItem(lr)

        # Manual anchor points.
        self._manual_scatter = []
        for col, pts in self._manual:
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            if self._use_log:
                ys = [np.log10(v) if v > 0 else 0 for v in ys]
            sc = pg.ScatterPlotItem(x=xs, y=ys, symbol='o', size=12,
                pen=pg.mkPen(col, width=2), brush=pg.mkBrush(col + 'AA'))
            sc.setZValue(100)
            self.plot_main.addItem(sc, ignoreBounds=True)
            self._manual_scatter.append(sc)

        if reset_view and len(self._q) > 1:
            self.plot_main.setXRange(0, float(self._q[-1]), padding=0.02)

        # Difference plot.
        if self._y_smoothed is not None:
            diff = self._y - self._y_smoothed
            self.plot_diff.setLabel('left', f"Δ {self._data_type}")
            self.plot_diff.plot(self._q, diff,
                pen=pg.mkPen('#1A8A4A', width=1.8))
            self.plot_diff.addLine(y=0,
                pen=pg.mkPen('#888888', style=Qt.PenStyle.DashLine))

        if reset_view:
            # Fit BOTH axes (x and y) of both plots to the new data. This is
            # what makes switching to a different file auto-reset the zoom.
            self._reset_view()


# ══════════════════════════════════════════════════════════════
#  Main window
# ══════════════════════════════════════════════════════════════

class WHRangeSmoother(QMainWindow):

    # Colour palette
    COL_ORIG  = '#1A6FA8'   # blue  – original data
    COL_SMOOTH= '#E84855'   # red   – smoothed
    COL_DIFF  = '#1A8A4A'   # green – difference

    def __init__(self):
        super().__init__()
        self.setWindowTitle("WH Range Smoother")

        # Window icon (title bar and taskbar).
        _icon_path = _resource_path("WH.ico")
        if os.path.exists(_icon_path):
            self.setWindowIcon(QIcon(_icon_path))

        # Size the window to fit the available screen area (which excludes the
        # taskbar) and centre it, so the bottom buttons are never cut off on
        # smaller or scaled displays.
        self._fit_and_center(1200, 1000)

        self._settings = QSettings("EZPDF", "WHRangeSmoother")
        self._last_dir = self._settings.value("last_dir", "")

        # Live session persistence is kept INTERNAL (QSettings) so editing /
        # smoothing never creates .whses files beside the sample.  A short
        # debounce timer coalesces rapid edits into one QSettings update.
        # A physical .whses file is created only when the user explicitly chooses
        # File -> Save Session....
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(600)   # ms
        self._autosave_timer.timeout.connect(self._write_autosave)

        self.q = self.y = None          # loaded data
        self.y_smoothed     = None      # result after smoothing
        self._data_type     = "I(q)"   # current type label
        self._y_raw         = None      # original I(q) before subtraction
        self._y_bkg_scaled  = None      # alpha * Bkg(q)
        self._alpha_used    = 1.0
        self._q_ref         = None      # reference q array
        self._y_ref         = None      # reference y array (raw)
        self._ref_scale     = 1.0       # multiplicative scale for reference
        self._active_row     = None     # QRangeRow in "Edit Points" mode
        self._manual_scatter = None     # scatter plot item for manual points
        self._last_click_time = 0.0     # debounce timestamp for manual clicks

        self._build_ui()
        self._init_help_menu()
        # Restore the previous workspace, but startup itself must NOT open a plot.
        # A plot is created only when the user hovers/clicks a sample afterwards.
        self._startup_restoring = True
        try:
            self._restore_session()
        finally:
            self._startup_restoring = False

    # ── Session persistence (auto save on close / restore on start) ──

    def _collect_file_session(self) -> dict:
        """Return the metadata embedded in one self-contained saved result.

        Version 4 stores the primary same-q-grid curves in the NUMERIC BODY
        instead of duplicating thousands of values inside one huge JSON header
        line.  The JSON therefore carries settings/state plus reference data
        (whose q grid may differ).  On reload, _read_saved_result() combines the
        metadata and numeric columns back into one complete session dict.
        """
        sample = self._current_sample_path()
        bkg = self._current_bkg_path()

        # Reference data can live on a different q grid, so keep it in metadata.
        data = {}
        if self._q_ref is not None and self._y_ref is not None:
            data["q_ref"] = [float(v) for v in self._q_ref]
            data["y_ref"] = [float(v) for v in self._y_ref]

        return {
            "version":       4,
            "kind":          "single_file",
            "data_layout":   "q_smoothed_processed_raw_bkg_v4",
            "n_points":      int(len(self.q)) if self.q is not None else 0,
            "has_background": self._y_bkg_scaled is not None,
            "source_path":   sample,
            "background_path": bkg,
            "data_type":     self._data_type,
            "def_lambda":    self.lambda_edit.text(),
            "def_order":     self.order_spin.value(),
            "alpha":         self.alpha_edit.text(),
            "alpha_used":    self._alpha_used,
            "log_y":         self.log_chk.isChecked(),
            "ref_scale":     self.ref_scale_edit.text(),
            "ref_path":      self.ref_edit.text(),
            "sample_files":  [sample] if sample else [],
            "sample_row":    0 if sample else -1,
            "bkg_files":     [bkg] if bkg else [],
            "bkg_row":       0 if bkg else -1,
            "ranges":        [row.serialize() for row in self._get_rows()],
            "data":          data,
        }

    def _collect_session(self) -> dict:
        """Gather every user-facing setting into a JSON-serialisable dict."""
        def _list_paths(list_widget):
            out = []
            for i in range(list_widget.count()):
                it = list_widget.item(i)
                out.append(it.data(Qt.ItemDataRole.UserRole) or it.text())
            return out

        return {
            "version":       2,
            "data_type":     self._data_type,
            "def_lambda":    self.lambda_edit.text(),
            "def_order":     self.order_spin.value(),
            "alpha":         self.alpha_edit.text(),
            "log_y":         self.log_chk.isChecked(),
            "ref_scale":     self.ref_scale_edit.text(),
            "ref_path":      self.ref_edit.text(),
            "sample_files":  _list_paths(self.file_list),
            "sample_row":    self.file_list.currentRow(),
            "bkg_files":     _list_paths(self.bkg_list),
            "bkg_row":       self.bkg_list.currentRow(),
            "ranges":        [row.serialize() for row in self._get_rows()],
        }

    def _schedule_autosave(self):
        """Queue a debounced INTERNAL session save (no .whses file is written)."""
        if getattr(self, "_suppress_autosave", False):
            return
        try:
            self._autosave_timer.start()
        except Exception:
            pass

    def _write_autosave(self):
        """Persist the workspace to QSettings only; never create a sidecar file."""
        if getattr(self, "_suppress_autosave", False):
            return
        self._save_session()

    def _save_session(self):
        """Persist the whole session to QSettings as a JSON blob."""
        try:
            blob = json.dumps(self._collect_session())
            self._settings.setValue("session_json", blob)
            self._settings.setValue("last_dir", self._last_dir)
            self._settings.sync()
        except Exception:
            pass   # never let a save error crash the app on exit

    def _restore_session(self):
        """Restore the auto-saved session from QSettings, if present."""
        blob = self._settings.value("session_json", "")
        if not blob:
            return
        try:
            s = json.loads(blob)
        except Exception:
            return
        self._apply_session(s)
        self.statusBar().showMessage("Previous session restored.")

    def _apply_session(self, s: dict, clear_first: bool = False):
        """Apply a session dict to the UI.

        Shared by the automatic QSettings restore and by explicit "Load
        Session…". When `clear_first` is True (file load) the sample and
        background lists are emptied first so the loaded setup fully replaces
        the current one instead of appending to it.
        """
        # Guard so that loading the sample inside here doesn't re-trigger the
        # embedded-session restore in _load_sample (infinite recursion).
        self._restoring_session = True
        try:
            self._apply_session_impl(s, clear_first)
        finally:
            self._restoring_session = False

    def _apply_session_impl(self, s: dict, clear_first: bool = False):
        embedded = s.get("data") or {}

        if clear_first:
            self.file_list.blockSignals(True)
            self.bkg_list.blockSignals(True)
            self.file_list.clear()
            self.bkg_list.clear()
            self.file_list.blockSignals(False)
            self.bkg_list.blockSignals(False)
            self.q = self.y = self._y_raw = self.y_smoothed = None
            self._y_bkg_scaled = None
            self._q_ref = self._y_ref = None
            self.ref_edit.blockSignals(True)
            self.ref_edit.clear()
            self.ref_edit.blockSignals(False)

        # ---- Data type first -------------------------------------------------
        # _on_type() changes the default log mode, so restore the saved log flag
        # AFTER the type is selected.
        dtype = s.get("data_type", "I(q)")
        for btn in self._type_bg.buttons():
            btn.setChecked(btn.text() == dtype)
        self._data_type = dtype

        # ---- Scalar settings ------------------------------------------------
        self.lambda_edit.setText(str(s.get("def_lambda", "1000")))
        try:
            self.order_spin.setValue(int(s.get("def_order", 2)))
        except Exception:
            pass

        self.alpha_edit.blockSignals(True)
        self.alpha_edit.setText(str(s.get("alpha", "1.0")))
        self.alpha_edit.blockSignals(False)
        try:
            self._alpha_used = float(s.get("alpha_used", s.get("alpha", 1.0)))
        except Exception:
            self._alpha_used = 1.0

        self.log_chk.blockSignals(True)
        self.log_chk.setChecked(bool(s.get("log_y", True)))
        self.log_chk.blockSignals(False)

        self.ref_scale_edit.blockSignals(True)
        self.ref_scale_edit.setText(str(s.get("ref_scale", "1.0")))
        self.ref_scale_edit.blockSignals(False)
        try:
            self._ref_scale = float(s.get("ref_scale", 1.0))
        except Exception:
            self._ref_scale = 1.0

        # ---- q-range rows / all per-range settings -------------------------
        for row in self._get_rows():
            self._range_layout.removeWidget(row)
            row.deleteLater()
        saved_ranges = s.get("ranges", [])
        if saved_ranges:
            for rs in saved_ranges:
                self._add_range_row()
                self._get_rows()[-1].apply_state(rs)
        else:
            self._add_range_row()

        # ---- Background path metadata ---------------------------------------
        # For an embedded result, keep the saved background entry even if the
        # original file was moved/deleted: the plotted scaled background curve
        # itself is embedded below.
        has_embedded_bkg = embedded.get("y_bkg_scaled") is not None
        self.bkg_list.blockSignals(True)
        for p in s.get("bkg_files", []):
            if os.path.exists(p) or has_embedded_bkg:
                self._append_list_item(self.bkg_list, p)
        bkg_row = s.get("bkg_row", -1)
        if 0 <= bkg_row < self.bkg_list.count():
            self.bkg_list.setCurrentRow(bkg_row)
            self._confirmed_bkg_row = bkg_row
        self.bkg_list.blockSignals(False)
        self._update_ext_label(self.bkg_ext_label, self._current_bkg_path())

        # ---- Reference path metadata ----------------------------------------
        ref_path = s.get("ref_path", "")
        self.ref_edit.blockSignals(True)
        self.ref_edit.setText(ref_path)
        self.ref_edit.blockSignals(False)

        # Only fall back to the external reference file for old sessions that
        # do not embed the reference arrays.  New saved results always prefer
        # the exact embedded curve.
        if not (embedded.get("q_ref") is not None and embedded.get("y_ref") is not None):
            if ref_path and os.path.exists(ref_path):
                try:
                    q_r, y_r = load_file(ref_path)
                    self._q_ref, self._y_ref = q_r, y_r
                except Exception:
                    pass

        # ---- Sample list ----------------------------------------------------
        self.file_list.blockSignals(True)
        for p in s.get("sample_files", []):
            if os.path.exists(p):
                self._append_list_item(self.file_list, p)
        sample_row = s.get("sample_row", -1)
        if 0 <= sample_row < self.file_list.count():
            self.file_list.setCurrentRow(sample_row)
            self._confirmed_sample_row = sample_row
        self.file_list.blockSignals(False)
        self._update_ext_label(self.file_ext_label, self._current_sample_path())

        used_embedded = False
        if embedded.get("q") is not None:
            # New single-file results restore every plotted curve from their own
            # header.  No source/background/reference file is required.
            import numpy as _np

            self.q = _np.asarray(embedded["q"], dtype=float)

            proc = embedded.get("y_processed")
            if proc is None:
                proc = embedded.get("y_original")  # v2 compatibility
            if proc is None:
                proc = embedded.get("y_smoothed")
            self.y = None if proc is None else _np.asarray(proc, dtype=float)

            raw = embedded.get("y_raw")
            self._y_raw = (
                _np.asarray(raw, dtype=float) if raw is not None
                else (None if self.y is None else self.y.copy())
            )

            bkg_scaled = embedded.get("y_bkg_scaled")
            self._y_bkg_scaled = (
                None if bkg_scaled is None
                else _np.asarray(bkg_scaled, dtype=float)
            )

            sm = embedded.get("y_smoothed")
            self.y_smoothed = None if sm is None else _np.asarray(sm, dtype=float)

            q_ref = embedded.get("q_ref")
            y_ref = embedded.get("y_ref")
            if q_ref is not None and y_ref is not None:
                self._q_ref = _np.asarray(q_ref, dtype=float)
                self._y_ref = _np.asarray(y_ref, dtype=float)

            sf = s.get("sample_files") or []
            if sf:
                self._current_loaded_name = os.path.basename(sf[0])
                self._current_loaded_path = sf[0]
            else:
                self._current_loaded_name = "embedded data"
                self._current_loaded_path = None

            used_embedded = (self.y is not None)

        elif 0 <= sample_row < self.file_list.count():
            # Old non-embedded session: load the selected external sample.
            self._on_sample_selected(sample_row)
        else:
            self._update_main_plot()
            self._update_diff_plot()

        # Re-assert the saved type/log values after all file-list work; loading an
        # external file can auto-detect a type, but the saved result is authoritative.
        if dtype in ("I(q)", "S(q)", "F(q)", "G(r)"):
            for btn in self._type_bg.buttons():
                btn.blockSignals(True)
                btn.setChecked(btn.text() == dtype)
                btn.blockSignals(False)
            self._data_type = dtype
        self.log_chk.blockSignals(True)
        self.log_chk.setChecked(bool(s.get("log_y", True)))
        self.log_chk.blockSignals(False)

        if used_embedded:
            self._new_plot_window()
        elif self.y is not None:
            self._auto_reapply()

    def _save_session_to_file(self):
        """Explicitly save the whole setup to a named .whses file.

        The default file name matches the current sample file (with a .whses
        extension), so a saved setup sits next to the data it belongs to.
        """
        # Base the default name on the loaded sample, if any.
        sample = self._current_sample_path()
        if sample:
            stem = os.path.splitext(os.path.basename(sample))[0]
            base_dir = os.path.dirname(sample) or (self._last_dir or "")
            default = os.path.join(base_dir, stem + ".whses")
        else:
            default = os.path.join(self._last_dir or "", "session.whses")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", default,
            "WH Session (*.whses);;JSON (*.json);;All Files (*)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._collect_session(), f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))
            return
        self._last_dir = os.path.dirname(path)
        self._settings.setValue("last_dir", self._last_dir)
        self._settings.sync()
        self.statusBar().showMessage(f"Session saved → {path}")

    def _load_session_from_file(self):
        """Explicitly load a setup from a named .whses file (replaces current)."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", self._last_dir or "",
            "WH Session (*.whses *.json);;All Files (*)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                s = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Load Error",
                                 f"Could not read session file:\n{e}")
            return
        if not isinstance(s, dict):
            QMessageBox.critical(self, "Load Error",
                                 "This file is not a valid WH session.")
            return
        self._apply_session(s, clear_first=True)
        self._last_dir = os.path.dirname(path)
        self._settings.setValue("last_dir", self._last_dir)
        self._settings.sync()
        self.statusBar().showMessage(f"Session loaded ← {path}")

    def closeEvent(self, event):
        """Auto-save the session when the window closes."""
        self._save_session()
        super().closeEvent(event)

    # ── Window placement ──────────────────────────────────────

    def _fit_and_center(self, width: int, height: int):
        """Resize to at most the available screen area, then centre the window.

        `availableGeometry()` excludes the taskbar, so the window never extends
        underneath it and the buttons at the bottom stay reachable. On smaller
        or display-scaled screens the window shrinks to fit instead of being
        clipped.
        """
        try:
            screen = self.screen() or QApplication.primaryScreen()
            avail = screen.availableGeometry()

            # Leave a margin so the window does not touch the edges. The
            # vertical margin is larger because window title bars and some
            # display-scaling setups eat extra height, which otherwise pushes
            # the bottom buttons under the taskbar.
            max_w = max(800, avail.width() - 40)
            max_h = max(600, avail.height() - 80)

            w = min(width, max_w)
            h = min(height, max_h)
            self.resize(w, h)

            # Centre within the available area.
            x = avail.x() + (avail.width() - w) // 2
            y = avail.y() + (avail.height() - h) // 2
            self.move(x, y)
        except Exception:
            # If anything about the screen query fails, fall back to a plain
            # resize rather than leaving the window unsized.
            self.resize(width, height)

    # ── File menu / Help menu / About ─────────────────────────

    def _init_help_menu(self):
        menu_bar = self.menuBar()

        # File menu: explicit named-session save / load. This is separate from
        # the automatic save-on-close; it lets you keep several named setups.
        file_menu = menu_bar.addMenu("File")
        # While the File menu is open the user often moves the mouse across the
        # file list on the way to "Save Session…". Freeze hover-select during
        # that time so the moved-over file isn't silently selected (which would
        # otherwise save the wrong file's session).
        file_menu.aboutToShow.connect(self._suspend_hover_select)
        file_menu.aboutToHide.connect(self._resume_hover_select)

        save_action = QAction("Save Session…", self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.setToolTip("Save the whole setup (files, ranges, parameters) to a file")
        save_action.triggered.connect(self._save_session_to_file)
        file_menu.addAction(save_action)

        load_action = QAction("Load Session…", self)
        load_action.setShortcut(QKeySequence("Ctrl+O"))
        load_action.setToolTip("Load a previously saved setup (replaces the current one)")
        load_action.triggered.connect(self._load_session_from_file)
        file_menu.addAction(load_action)

        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        help_menu = menu_bar.addMenu("Help")
        about_action = QAction("About WH Smooth", self)
        about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_action)

    def _show_about_dialog(self):
        about_text = """
        <b>WH Smooth (Whittaker&ndash;Henderson Range Smoother) Version 1.0.2</b>
        <p>: A standalone tool for Whittaker&ndash;Henderson smoothing over
        selected q ranges of I(q), S(q), F(q), or G(r) data, developed by the
        NSLS-II team.</p>

        <p><b>Release date:</b><br>
        08/21/2026</p>

        <p><b>Author:</b><br>
        <b>National Synchrotron Light Source II:</b><br>
        Gihan Kwon</p>

        <p><b>Contact:</b><br>
        gkwon@bnl.gov</p>

        <p><b>Website:</b><br>
        https://github.com/ezpit/ezpit</p>
        """
        box = QMessageBox(self)
        box.setWindowTitle("About WH Smooth")
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(about_text)

        # Show the logo on the left. Prefer the high-resolution PNG so the
        # logo stays sharp; fall back to the .ico if the PNG is not bundled.
        logo_path = _resource_path("WHsmooth.png")
        if not os.path.exists(logo_path):
            logo_path = _resource_path("WH.ico")
        if os.path.exists(logo_path):
            pix = QPixmap(logo_path)
            if not pix.isNull():
                box.setIconPixmap(
                    pix.scaledToWidth(220, Qt.TransformationMode.SmoothTransformation))
        box.exec()

    # ── UI ────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        # Let the user drag column boundaries freely (columns may be made very
        # narrow or even collapsed).
        splitter.setChildrenCollapsible(True)
        root.addWidget(splitter)

        # Helper: build a scrollable column and return (container_layout, scroll).
        # `width` is only the INITIAL preferred width (via setSizes later); a
        # tiny minimum lets the user shrink the column as far as they like.
        # If `with_footer` is True, a non-scrolling footer layout is added below
        # the scroll area (for buttons that must always stay visible) and is
        # returned as the third value.
        def _make_column(width, with_footer=False):
            container = QWidget()
            cbox = QVBoxLayout(container)
            cbox.setContentsMargins(0, 0, 0, 0)
            cbox.setSpacing(6)

            panel = QWidget()
            lay = QVBoxLayout(panel)
            lay.setSpacing(10)
            lay.setContentsMargins(4, 4, 4, 4)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(panel)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            # Horizontal scrollbar ON so long file names stay reachable even
            # when the column is dragged narrow.
            scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll.setMinimumWidth(0)   # no floor: user can shrink to minimum
            cbox.addWidget(scroll, 1)

            footer = None
            if with_footer:
                footer = QVBoxLayout()
                footer.setContentsMargins(4, 0, 4, 4)
                footer.setSpacing(6)
                cbox.addLayout(footer)

            container.setMinimumWidth(0)
            splitter.addWidget(container)
            if with_footer:
                return lay, scroll, footer
            return lay, scroll

        # ── Column 1: Sample files ──  Column 2: Background files ──
        # ── Column 3: Data type + all settings ──
        col1_lay, _c1 = _make_column(250)                       # Sample list
        col2_lay, _c2 = _make_column(270)                       # Background list
        col3_lay, _c3, col3_footer = _make_column(360, with_footer=True)

        # Backwards-compatible alias: most groups below were written to add
        # themselves to `llay`. Point it at column 3 (settings) by default and
        # override per-group where a different column is wanted.
        llay = col3_lay

        # 1. Sample files — a list you can add several files to. The currently
        #    selected row is the "active" sample that gets plotted / processed.
        file_grp = QGroupBox("1.  Sample Files")
        fg = QGridLayout(file_grp)
        fg.setSpacing(6)

        self.file_list = DropListWidget()
        self.file_list.on_files_dropped = self._on_sample_files_dropped
        self.file_list.hover_select = True   # hover previews; click confirms
        self.file_list.on_hover_preview = self._preview_sample
        self.file_list.on_hover_leave = self._revert_sample_preview
        self.file_list.on_click_select = self._confirm_sample_click
        self.file_list.setMinimumHeight(140)
        self.file_list.setMinimumWidth(0)    # allow the column to be dragged very narrow
        self.file_list.setSizePolicy(
            self.file_list.sizePolicy().horizontalPolicy(),
            QSizePolicy.Policy.Expanding)
        self.file_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.file_list.setToolTip(
            "Add only registers files; it does not open a plot.\n"
            "Hover a file to make it active, plot it, and restore saved settings.\n"
            "Double-click to open it in a NEW plot window for comparison.\n"
            "Ctrl/Shift-click to select several, then Remove to delete them.")
        self.file_list.currentRowChanged.connect(self._on_sample_selected)
        self.file_list.itemDoubleClicked.connect(self._on_sample_double_clicked)
        fg.addWidget(self.file_list, 0, 0, 1, 3)
        fg.setRowStretch(0, 1)   # the list row grows to fill the group

        btn_add_file = QPushButton("＋ Add…")
        btn_add_file.clicked.connect(self._add_sample_files)
        btn_del_file = QPushButton("－ Remove")
        btn_del_file.clicked.connect(self._remove_sample_file)
        btn_clr_file = QPushButton("Clear")
        btn_clr_file.clicked.connect(self._clear_sample_files)
        for _b in (btn_add_file, btn_del_file, btn_clr_file):
            _b.setMinimumWidth(0)   # don't force a wide column
        fg.addWidget(btn_add_file, 1, 0)
        fg.addWidget(btn_del_file, 1, 1)
        fg.addWidget(btn_clr_file, 1, 2)
        # Shows the selected sample's file extension (handy when long names hide
        # the extension in the narrow list).
        self.file_ext_label = QLabel("ext: —")
        self.file_ext_label.setStyleSheet("QLabel { color:#555; }")
        fg.addWidget(self.file_ext_label, 2, 0, 1, 3)
        col1_lay.addWidget(file_grp, 1)   # group fills the whole column

        # 2. Data type
        type_grp = QGroupBox("3.  Data Type")
        type_lay = QVBoxLayout(type_grp)
        tg = QHBoxLayout()
        self._type_bg = QButtonGroup(self)
        for label in ("I(q)", "S(q)", "F(q)", "G(r)"):
            rb = QRadioButton(label)
            rb.setChecked(label == "I(q)")
            rb.toggled.connect(lambda checked, l=label: self._on_type(l, checked))
            self._type_bg.addButton(rb)
            tg.addWidget(rb)
        type_lay.addLayout(tg)

        # Log Y-axis toggle lives here (not in the Background box) so it's
        # available for S(q) / F(q) too, where there is no background section.
        log_row = QHBoxLayout()
        self.log_chk = QCheckBox("Log Y-axis (main plot)")
        self.log_chk.setChecked(True)
        self.log_chk.stateChanged.connect(lambda _: self._apply_log_scale())
        log_row.addWidget(self.log_chk)
        log_row.addStretch()
        type_lay.addLayout(log_row)

        llay.addWidget(type_grp)

        # 2b. Background subtraction — visible only when I(q) is selected.
        #     Like the sample list, several backgrounds can be loaded and the
        #     selected one is subtracted from the active sample.
        self.bkg_grp = QGroupBox("2.  Background  ( α × I_Bkg(q) )")
        bg_lay = QGridLayout(self.bkg_grp)
        bg_lay.setSpacing(6)

        self.bkg_list = DropListWidget()
        self.bkg_list.on_files_dropped = self._on_bkg_files_dropped
        self.bkg_list.hover_select = True   # hover previews; click confirms
        self.bkg_list.on_hover_preview = self._preview_bkg
        self.bkg_list.on_hover_leave = self._revert_bkg_preview
        self.bkg_list.on_click_select = self._confirm_bkg_click
        self.bkg_list.setMinimumHeight(140)
        self.bkg_list.setMinimumWidth(0)    # allow the column to be dragged very narrow
        self.bkg_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.bkg_list.setSizePolicy(
            self.bkg_list.sizePolicy().horizontalPolicy(),
            QSizePolicy.Policy.Expanding)
        self.bkg_list.setToolTip(
            "Loaded background files. Click one to subtract it from the sample.\n"
            "Ctrl/Shift-click to select several, then Remove to delete them.")
        self.bkg_list.currentRowChanged.connect(self._on_bkg_selected)
        bg_lay.addWidget(QLabel("Bkg files:"), 0, 0, 1, 3)
        bg_lay.addWidget(self.bkg_list, 1, 0, 1, 3)
        bg_lay.setRowStretch(1, 1)   # the list row grows to fill the group

        btn_add_bkg = QPushButton("＋ Add…")
        btn_add_bkg.clicked.connect(self._add_bkg_files)
        btn_del_bkg = QPushButton("－ Remove")
        btn_del_bkg.clicked.connect(self._remove_bkg_file)
        btn_clr_bkg = QPushButton("Clear")
        btn_clr_bkg.clicked.connect(self._clear_bkg_files)
        for _b in (btn_add_bkg, btn_del_bkg, btn_clr_bkg):
            _b.setMinimumWidth(0)   # don't force a wide column
        bg_lay.addWidget(btn_add_bkg, 2, 0)
        bg_lay.addWidget(btn_del_bkg, 2, 1)
        bg_lay.addWidget(btn_clr_bkg, 2, 2)

        # Shows the selected background's file extension.
        self.bkg_ext_label = QLabel("ext: —")
        self.bkg_ext_label.setStyleSheet("QLabel { color:#555; }")
        bg_lay.addWidget(self.bkg_ext_label, 3, 0, 1, 3)

        bg_lay.addWidget(QLabel("α ="), 4, 0)
        self.alpha_edit = QLineEdit("1.0")
        self.alpha_edit.setFixedWidth(80)
        self.alpha_edit.setToolTip("Background scale factor")
        # Auto-update on every keystroke (textChanged) and on Enter
        self.alpha_edit.textChanged.connect(self._subtract_bkg)
        self.alpha_edit.editingFinished.connect(self._on_alpha_committed)
        self.alpha_edit.returnPressed.connect(self._on_alpha_committed)
        bg_lay.addWidget(self.alpha_edit, 4, 1)

        self.bkg_grp.setVisible(True)    # visible by default (I(q) mode)
        col2_lay.addWidget(self.bkg_grp, 1)   # group fills the whole column

        # 3. Default WH params (used when adding new rows)
        wh_grp = QGroupBox("4.  Default WH Parameters (for new ranges)")
        wg = QGridLayout(wh_grp)
        wg.setSpacing(6)
        wg.addWidget(QLabel("lambda:"), 0, 0)
        self.lambda_edit = QLineEdit("1000")
        self.lambda_edit.setFixedWidth(90)
        self.lambda_edit.setToolTip(
            "Default λ for new ranges. Also used to smooth the WHOLE q range "
            "live when no range is defined.")
        self.lambda_edit.textChanged.connect(self._auto_reapply)
        self.lambda_edit.editingFinished.connect(self._auto_reapply)
        wg.addWidget(self.lambda_edit, 0, 1)
        wg.addWidget(QLabel("Order:"), 1, 0)
        self.order_spin = QSpinBox()
        self.order_spin.setRange(1, 2147483647)
        self.order_spin.setValue(2)
        self.order_spin.setFixedWidth(90)
        self.order_spin.setToolTip(
            "Default order for new ranges. Also used for the live full-range "
            "smoothing when no range is defined.")
        self.order_spin.valueChanged.connect(lambda _v: self._auto_reapply())
        wg.addWidget(self.order_spin, 1, 1)
        llay.addWidget(wh_grp)

        # 4. Q ranges
        range_grp = QGroupBox("5.  q Ranges for Smoothing")
        rg = QVBoxLayout(range_grp)
        rg.setSpacing(4)
        # Don't let this group stretch vertically and leave a big gap above the
        # range rows — keep it just tall enough for its contents.
        range_grp.setSizePolicy(QSizePolicy.Policy.Preferred,
                                QSizePolicy.Policy.Maximum)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        # Flexible height so the Apply / Save buttons below stay visible when
        # the app first opens. The range list scrolls internally if many ranges
        # are added.
        # Give the range list a taller default so the "Add Range" button sits
        # well below the q-Ranges header instead of hugging it. It still grows
        # up to the max (then scrolls) as more ranges are added.
        scroll.setMinimumHeight(340)
        scroll.setMaximumHeight(460)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._range_container = QWidget()
        self._range_layout = QVBoxLayout(self._range_container)
        self._range_layout.setContentsMargins(0, 0, 0, 0)
        self._range_layout.setSpacing(0)
        # Trailing stretch: rows are inserted before it (see _add_range_row) so
        # they stack from the top with no blank space above the first row.
        self._range_layout.addStretch()
        scroll.setWidget(self._range_container)
        rg.addWidget(scroll)

        btn_add = QPushButton("＋  Add Range")
        btn_add.clicked.connect(self._add_range_row)
        rg.addWidget(btn_add)
        llay.addWidget(range_grp)

        # Spacer that pushes the Reference section well below the q-Ranges box
        # (it was sitting too close to section 5). The larger stretch factor
        # here vs. the trailing one places Reference lower in the column.
        llay.addStretch(3)

        # 6. Reference (optional)
        ref_grp = QGroupBox("6.  Reference Data (optional)")
        rgrid = QGridLayout(ref_grp)
        rgrid.setSpacing(6)

        self.ref_edit = DropLineEdit()
        # A normal, editable field: the user can type/edit and delete character
        # by character. Being editable (not read-only) also means dropped files
        # are accepted reliably across platforms.
        self.ref_edit.setAcceptDrops(True)
        self.ref_edit.setPlaceholderText("Drop a reference file here (or use Browse…)")
        self.ref_edit.setToolTip(
            "Drop a reference file here, type/paste a path, or use Browse….\n"
            "Delete the text (character by character or all at once) to clear it.")
        self.ref_edit.on_file_dropped = self._on_reference_dropped
        # When the field is emptied by editing, clear the loaded reference.
        self.ref_edit.textChanged.connect(self._on_ref_text_changed)
        # Make the drop target taller and mark it visually as a drop zone so
        # it's easy to hit when dragging a file onto it.
        self.ref_edit.setMinimumHeight(48)
        self.ref_edit.setStyleSheet(
            "DropLineEdit { border: 1px dashed #888; border-radius: 4px;"
            " padding: 4px; background: #fafafa; }")
        self.ref_edit.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        btn_ref = QPushButton("Browse…")
        btn_ref.clicked.connect(self._pick_reference)

        btn_ref_clear = QPushButton("Clear")
        btn_ref_clear.clicked.connect(self._clear_reference)

        rgrid.addWidget(QLabel("Ref file:"), 0, 0)
        rgrid.addWidget(self.ref_edit, 1, 0, 1, 2)
        rgrid.addWidget(btn_ref, 1, 2)
        rgrid.addWidget(btn_ref_clear, 2, 2)

        rgrid.addWidget(QLabel("scale:"), 2, 0)
        self.ref_scale_edit = QLineEdit("1.0")
        self.ref_scale_edit.setFixedWidth(80)
        self.ref_scale_edit.setToolTip("Multiplicative scale applied to reference y values")
        self.ref_scale_edit.textChanged.connect(self._on_ref_scale_changed)
        rgrid.addWidget(self.ref_scale_edit, 2, 1)

        llay.addWidget(ref_grp)
        # Push all settings sections to the top of the column; any spare space
        # collects at the bottom instead of between sections.
        llay.addStretch(1)

        # Action buttons — pinned to the bottom of the settings column so they
        # stay visible without scrolling, even when the window is short.
        # (No "Apply Smoothing" button — smoothing is fully live: editing any
        #  range or the Section 4 default parameters updates the plot at once.)
        btn_save = QPushButton("💾  Save Smoothed Data")
        btn_save.clicked.connect(self._save_result)
        col3_footer.addWidget(btn_save)

        # ── Right pane: info placeholder ──────────────────────
        # Plots are shown in separate windows now (one new window per sample
        # selection), so the main window hosts only the three control columns:
        # [Sample files] [Background files] [Data type + settings].
        # Let the settings column take up any remaining width but keep it
        # reasonably narrow so it doesn't stretch too wide.
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([300, 320, 360])

        pg.setConfigOptions(antialias=True)
        pg.setConfigOption('background', 'w')   # white background
        pg.setConfigOption('foreground', 'k')   # black axes/text

        # List of open plot windows (kept referenced so they aren't garbage
        # collected while visible).
        self._plot_windows = []
        self._current_loaded_name = None   # basename of the loaded sample
        self._current_loaded_path = None   # full path of the loaded sample
        self._suppress_autosave = False    # set during explicit Save Smoothed
        self._confirmed_sample_row = -1    # active sample (hover or click)
        self._confirmed_bkg_row = -1       # click-confirmed background
        self._active_plot_window = None   # the window that live-updates
        self._preserve_hover_view = False # hover previews keep current zoom/range
        self._confirmed_plot_snapshot = None  # exact non-hover plot state
        self._confirmed_plot_path = None      # file that snapshot belongs to
        # Sample hover now ACTIVATES the hovered sample so its plot/settings
        # remain available when the pointer moves to the controls. Background
        # hover remains view-only.
        self._sample_hover_preview_active = False
        self._bkg_hover_preview_active = False

        self.statusBar().showMessage("Ready — load a data file.")
        self._add_range_row()

    # ── Plot windows ──────────────────────────────────────────

    def _apply_log_scale(self):
        """Apply the Log Y-axis toggle to the active plot window.

        The main window no longer hosts a live plot, so toggling log/linear
        refreshes the active plot window (which reads the log flag from the
        snapshot).
        """
        self._refresh_active_window()


    def _on_type(self, label: str, checked: bool):
        if checked:
            self._data_type = label
            is_iq = (label == "I(q)")
            # Keep the Background section visible for every data type so the
            # background files and α scale stay available (e.g. subtracting a
            # measured background from S(q)/F(q)/G(r) too).
            self.bkg_grp.setVisible(True)
            # Default log scale ON for I(q), OFF for S(q)/F(q)/G(r) (the checkbox
            # can still be toggled manually afterwards). blockSignals avoids an
            # extra refresh; _apply_log_scale below does the update.
            self.log_chk.blockSignals(True)
            self.log_chk.setChecked(is_iq)
            self.log_chk.blockSignals(False)
            self._apply_log_scale()
            self._update_main_plot()

    # ── Background file list ──────────────────────────────────

    def _on_sample_files_dropped(self, paths):
        """Files dropped onto the sample list are added (not auto-plotted).

        We only append them; the graph is created when the user clicks a file,
        so dropping a batch doesn't spawn plot windows.
        """
        added = 0
        for p in paths:
            if os.path.isfile(p):
                self._append_list_item(self.file_list, p)
                added += 1
        if added:
            self._last_dir = os.path.dirname(paths[-1])
            self._settings.setValue("last_dir", self._last_dir)
            self._settings.sync()
            self.statusBar().showMessage(
                f"Added {added} sample file(s) by drop — click one to plot it.")

    def _on_bkg_files_dropped(self, paths):
        """Files dropped onto the background list are added (not auto-applied)."""
        added = 0
        for p in paths:
            if os.path.isfile(p):
                self._append_list_item(self.bkg_list, p)
                added += 1
        if added:
            self._last_dir = os.path.dirname(paths[-1])
            self._settings.setValue("last_dir", self._last_dir)
            self._settings.sync()
            self.statusBar().showMessage(
                f"Added {added} background file(s) by drop — click one to use it.")

    def _on_reference_dropped(self, path):
        """A file dropped onto the reference field is loaded as the reference."""
        if os.path.isfile(path):
            self._load_reference_path(path)

    def _add_bkg_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add Background File(s)", self._last_dir,
            "Data Files (*.chi *.iq *.sq *.fq *.xy *.gr *.txt *.dat *.csv *.tsv);;"
            "All Files (*)")
        if not paths:
            return
        for p in paths:
            self._append_list_item(self.bkg_list, p)
        self._last_dir = os.path.dirname(paths[-1])
        self._settings.setValue("last_dir", self._last_dir)
        self._settings.sync()
        # Select the first added file if nothing is selected yet.
        if self.bkg_list.currentRow() < 0:
            self.bkg_list.setCurrentRow(self.bkg_list.count() - len(paths))

    def _remove_bkg_file(self):
        # Remove every selected row (multi-select). Delete from the bottom up
        # so row indices stay valid while removing. Block currentRowChanged so
        # removal doesn't auto-select/plot a different background.
        rows = sorted((self.bkg_list.row(it)
                       for it in self.bkg_list.selectedItems()), reverse=True)
        if not rows:
            r = self.bkg_list.currentRow()
            if r < 0:
                return
            rows = [r]
        self.bkg_list.blockSignals(True)
        for r in rows:
            self.bkg_list.takeItem(r)
        self.bkg_list.setCurrentRow(-1)
        self.bkg_list.blockSignals(False)
        self._update_ext_label(self.bkg_ext_label, "")
        # Re-subtract (now with no background selected → raw sample), updating an
        # open plot but not spawning a new one.
        if self.q is not None and self._y_raw is not None:
            self._y_bkg_scaled = None
            self.y = self._y_raw.copy()
            self.y_smoothed = None
            self._refresh_active_window_if_open()

    def _clear_bkg_files(self):
        self.bkg_list.clear()
        self._y_bkg_scaled = None
        # Dropping the background invalidates any previous smoothing result,
        # so clear it and refresh BOTH plots (the diff plot too) — otherwise
        # the old Smoothed / Difference curves linger on screen.
        self.y_smoothed = None
        if self.y is not None and self._y_raw is not None:
            self.y = self._y_raw.copy()
        self._apply_log_scale()
        self._update_main_plot()
        self._update_diff_plot()

    def _current_bkg_path(self):
        it = self.bkg_list.currentItem()
        if it is None:
            return ""
        return it.data(Qt.ItemDataRole.UserRole) or it.text()

    def _on_bkg_selected(self, _row):
        """Apply a background only on a real selection, then re-smooth.

        Background HOVER is handled separately as a view-only snapshot.  A real
        click may legitimately change Processed data, so if a smoothed curve was
        already present we immediately recompute it with the current ranges.
        """
        self._update_ext_label(self.bkg_ext_label, self._current_bkg_path())
        self._confirmed_bkg_row = self.bkg_list.currentRow()
        had_smoothed = self.y_smoothed is not None
        self._subtract_bkg()
        if had_smoothed and self.y is not None and self._y_bkg_scaled is not None:
            self._auto_reapply()

    def _background_preview_snapshot(self, path):
        """Build a background-hover preview WITHOUT mutating controller data.

        This is the key difference from the old implementation: hover used to
        call ``_subtract_bkg()``, which set ``self.y_smoothed = None``.  Merely
        moving the mouse across the Background list could therefore erase the
        selected saved result's Smoothed curve.
        """
        if self.q is None or self._y_raw is None or not path or not os.path.exists(path):
            return None
        try:
            q_b, y_b = load_file(path)
            alpha = float(self.alpha_edit.text().replace(',', '.'))
        except Exception:
            return None

        q = np.asarray(self.q, dtype=float).copy()
        raw = np.asarray(self._y_raw, dtype=float).copy()
        bkg_scaled = alpha * np.interp(q, q_b, y_b)
        proc = raw - bkg_scaled

        # Recompute a temporary smoothed curve using the CURRENT range settings.
        # Nothing below is assigned back to self.*.
        ranges = self._get_ranges()
        work_ranges = ranges
        if not work_ranges:
            try:
                work_ranges = [{
                    'q0': float(q[0]), 'q1': float(q[-1]),
                    'mode': 'wh',
                    'lambda': float(self.lambda_edit.text().replace(',', '.')),
                    'order': self.order_spin.value(), 'blend': 0,
                }]
            except Exception:
                work_ranges = []
        y_sm = None
        if work_ranges:
            try:
                y_sm = range_smooth(q, proc, work_ranges)
            except Exception:
                y_sm = None

        snap = self._build_snapshot()
        snap['q'] = q
        snap['y_raw'] = raw
        snap['y'] = proc
        snap['y_bkg_scaled'] = bkg_scaled
        snap['y_smoothed'] = y_sm
        snap['alpha'] = alpha
        return snap

    def _preview_bkg(self, row):
        """View-only background hover preview; never alter live arrays."""
        if row < 0 or row >= self.bkg_list.count():
            return
        it = self.bkg_list.item(row)
        path = it.data(Qt.ItemDataRole.UserRole) or it.text()
        if not path:
            return
        self._update_ext_label(self.bkg_ext_label, path)

        # Hovering the already-selected background is a strict no-op.  If we had
        # been previewing another row just before this, restore the live plot.
        selected = self._current_bkg_path()
        if self._same_path(path, selected):
            if self._bkg_hover_preview_active:
                self._revert_bkg_preview()
            return

        snap = self._background_preview_snapshot(path)
        win = self._active_plot_window
        if snap is not None and win is not None and win.isVisible():
            self._bkg_hover_preview_active = True
            win.update_snapshot(self._copy_plot_snapshot(snap), reset_view=False)
            win.show()
            win.raise_()

    def _revert_bkg_preview(self):
        """Restore live selected data only if a different Bkg was previewed."""
        self._update_ext_label(self.bkg_ext_label, self._current_bkg_path())
        if not getattr(self, '_bkg_hover_preview_active', False):
            return
        self._bkg_hover_preview_active = False

        # Hover preview never mutated self.*, so the live controller snapshot is
        # authoritative.  If a saved-result Smoothed curve is cached, protect it
        # from any transient None left by unrelated Qt signals.
        snap = self._build_snapshot() if self.q is not None and self.y is not None else None
        remembered = self._confirmed_plot_snapshot
        current = getattr(self, '_current_loaded_path', None)
        if (snap is not None and snap.get('y_smoothed') is None and
                remembered is not None and remembered.get('y_smoothed') is not None and
                self._same_path(getattr(self, '_confirmed_plot_path', None), current)):
            snap['y_smoothed'] = np.asarray(remembered['y_smoothed'], dtype=float).copy()
            self.y_smoothed = snap['y_smoothed'].copy()

        win = self._active_plot_window
        if snap is not None and win is not None and win.isVisible():
            self._remember_confirmed_plot(snap, path=current)
            win.update_snapshot(self._copy_plot_snapshot(snap), reset_view=False)
            win.show()
            win.raise_()

    def _confirm_bkg_click(self, row):
        """A real click on a background row records it as confirmed.

        The load happens via currentRowChanged → _on_bkg_selected.
        """
        if 0 <= row < self.bkg_list.count():
            self._confirmed_bkg_row = row

    def _update_ext_label(self, label, path):
        """Show the file extension of the selected item in the given label."""
        if not path:
            label.setText("ext: —")
            return
        ext = os.path.splitext(path)[1]
        label.setText(f"ext: {ext if ext else '(none)'}")

    def _subtract_bkg(self, preview_path=None):
        """Compute I(q) - alpha * Bkg(q) using the selected background file.

        Called automatically when alpha changes or the selected background
        changes. Silent (no popups) for the common "not ready yet" cases so
        typing alpha does not spam warnings. When `preview_path` is given it is
        used instead of the confirmed selection (for hover preview).
        """
        if self.q is None or self._y_raw is None:
            return   # no sample loaded yet — silent
        bkg_path = (preview_path or self._current_bkg_path()).strip()
        if not bkg_path or not os.path.exists(bkg_path):
            # No background selected → show raw sample as-is and drop any
            # stale smoothing result. Rescale so the raw curve is on-screen.
            self._y_bkg_scaled = None
            self.y = self._y_raw.copy()
            self.y_smoothed = None
            self._refresh_active_window(reset_view=not getattr(self, "_preserve_hover_view", False))
            return
        try:
            alpha = float(self.alpha_edit.text())
        except ValueError:
            return   # invalid alpha while typing — silent
        try:
            q_b, y_b = load_file(bkg_path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return

        # Interpolate background onto sample q grid
        y_b_interp = np.interp(self.q, q_b, y_b)
        self._y_bkg_scaled = alpha * y_b_interp   # alpha * Bkg(q), for plotting
        self._alpha_used    = alpha
        self.y              = self._y_raw - self._y_bkg_scaled
        self.y_smoothed     = None

        # Subtraction changes the y-scale (the subtracted curve can be far
        # smaller than the raw data), so rescale the view to keep it visible.
        self._refresh_active_window(reset_view=not getattr(self, "_preserve_hover_view", False))
        self.statusBar().showMessage(
            f"Background subtracted  |  alpha = {alpha}  |  "
            f"Bkg: {os.path.basename(bkg_path)}")

    def _pick_reference(self):
        """Load an optional reference curve for visual comparison."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Reference File", self._last_dir,
            "Data Files (*.chi *.iq *.sq *.fq *.xy *.gr *.txt *.dat *.csv *.tsv);;"
            "All Files (*)")
        if not path:
            return
        self._load_reference_path(path)

    def _load_reference_path(self, path):
        """Load `path` as the reference curve (used by Browse and drag-drop)."""
        try:
            q_r, y_r = load_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return
        self._q_ref = q_r
        self._y_ref = y_r
        self.ref_edit.setText(path)
        self._last_dir = os.path.dirname(path)
        self._settings.setValue("last_dir", self._last_dir)
        self._settings.sync()
        # Reference is auxiliary: update an open plot if there is one, but don't
        # spawn a new window just for a reference curve.
        self._refresh_active_window_if_open()
        self.statusBar().showMessage(
            f"Reference loaded: {os.path.basename(path)}  "
            f"({len(q_r)} pts, q = {q_r[0]:.3f}–{q_r[-1]:.3f})")

    def _clear_reference(self):
        self._q_ref = None
        self._y_ref = None
        # Avoid re-entering via textChanged while we blank the field.
        self.ref_edit.blockSignals(True)
        self.ref_edit.clear()
        self.ref_edit.blockSignals(False)
        self._refresh_active_window_if_open()
        self.statusBar().showMessage("Reference cleared.")

    def _on_ref_text_changed(self, text):
        """If the user empties the reference field, drop the loaded reference.

        Loading a file (which populates the field) does not need handling here
        because it sets the reference explicitly; only the emptied case matters.
        """
        if text.strip() == "" and self._y_ref is not None:
            self._q_ref = None
            self._y_ref = None
            self._refresh_active_window_if_open()
            self.statusBar().showMessage("Reference cleared.")

    def _on_ref_scale_changed(self, *_):
        try:
            self._ref_scale = float(self.ref_scale_edit.text())
        except ValueError:
            return
        if self._y_ref is not None:
            self._refresh_active_window_if_open()

    # ── Sample file list ──────────────────────────────────────

    def _append_list_item(self, list_widget, path):
        """Add a file to a QListWidget, showing the basename but storing the
        full path in the item's UserRole (and as a tooltip).

        Session/non-data files (.whses, .json) are rejected here so they never
        end up in a sample/background list where they'd fail to load as data.
        Returns the created item, or None if the file was skipped.
        """
        if os.path.splitext(path)[1].lower() in (".whses", ".json"):
            self.statusBar().showMessage(
                f"Skipped {os.path.basename(path)} — that's a session file, "
                "not data. Use File ▸ Load Session to open it.")
            return None
        item = QListWidgetItem(os.path.basename(path))
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setToolTip(path)
        list_widget.addItem(item)
        return item

    def _add_sample_files(self):
        """Add sample files to the list without selecting or plotting them.

        Add is intentionally a registration-only action.  Plotting begins only
        when the user hovers a row (preview) or clicks a row (confirmed load).
        Any sample that was already active before Add remains active.
        """
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add Sample File(s)", self._last_dir,
            "Data Files (*.chi *.iq *.sq *.fq *.xy *.gr *.txt *.dat *.csv *.tsv);;"
            "All Files (*)")
        if not paths:
            return

        # Prevent QListWidget insertion/current-index side effects from invoking
        # currentRowChanged while a batch is being appended.  Restore the same
        # current row afterwards; if there was none, keep currentRow == -1.
        current_row_before = self.file_list.currentRow()
        self.file_list.blockSignals(True)
        added = 0
        try:
            for p in paths:
                if os.path.isfile(p):
                    if self._append_list_item(self.file_list, p) is not None:
                        added += 1
            if 0 <= current_row_before < self.file_list.count():
                self.file_list.setCurrentRow(current_row_before)
            else:
                self.file_list.setCurrentRow(-1)
        finally:
            self.file_list.blockSignals(False)

        self._last_dir = os.path.dirname(paths[-1])
        self._settings.setValue("last_dir", self._last_dir)
        self._settings.sync()

        if added:
            self.statusBar().showMessage(
                f"Added {added} sample file(s) — hover a file to activate and plot it.")
        self._schedule_autosave()

    def _blank_active_plot_window(self):
        """Clear the curves in the active plot window without closing it.

        Kept for internal compatibility; Remove/Clear now close plot windows.
        """
        win = self._active_plot_window
        if win is not None and win.isVisible():
            try:
                win.clear_plot()
            except Exception:
                pass

    def _close_active_plot_window(self):
        """Close and forget the active plot window.

        Removing the currently loaded sample should make its graph disappear,
        not leave an empty PlotWindow behind.
        """
        win = self._active_plot_window
        if win is None:
            return
        try:
            win.close()
        except Exception:
            pass
        self._plot_windows = [w for w in self._plot_windows
                              if w is not win and w.isVisible()]
        self._active_plot_window = None

    def _close_all_plot_windows(self):
        """Close all plot windows, used when the whole sample list is cleared."""
        for win in list(self._plot_windows):
            try:
                win.close()
            except Exception:
                pass
        self._plot_windows = []
        self._active_plot_window = None

    def _clear_loaded_sample_state(self, reset_ranges=True):
        """Forget all state that belongs to the currently loaded sample.

        This is intentionally stronger than just blanking the PlotWindow.  A
        saved *_wh_smoothed.chi can populate q-ranges/manual points and cache a
        confirmed plot snapshot.  If those objects survive after Remove/Clear,
        a later leaveEvent or delayed QTimer can make the deleted file appear to
        come back.  Clear every sample-owned cache together.
        """
        self.q = self.y = self._y_raw = self.y_smoothed = None
        self._y_bkg_scaled = None
        self._current_loaded_name = None
        self._current_loaded_path = None
        self._confirmed_sample_row = -1
        self._sample_hover_preview_active = False
        self._confirmed_plot_snapshot = None
        self._confirmed_plot_path = None
        self._previewing = False

        # q-ranges and Manual-Spline points are sample-specific.  Removing the
        # active sample must not leave the removed file's processing setup in UI.
        if reset_ranges:
            self._reset_ranges()

        # The active sample no longer exists, so its PlotWindow should also
        # disappear instead of remaining as an empty/stale graph.
        self._close_active_plot_window()

    def _remove_sample_file(self):
        # Remove every selected row (multi-select), bottom-up.  If the active
        # sample itself is removed, clear its data, plot cache, and q-range setup.
        # Removing some OTHER file must not disturb the active plot/settings.
        rows = sorted((self.file_list.row(it)
                       for it in self.file_list.selectedItems()), reverse=True)
        if not rows:
            r = self.file_list.currentRow()
            if r < 0:
                return
            rows = [r]

        # Capture paths before takeItem() destroys the QListWidgetItems.
        removed_paths = []
        for r in rows:
            it = self.file_list.item(r)
            if it is not None:
                removed_paths.append(it.data(Qt.ItemDataRole.UserRole) or it.text())

        # _current_loaded_path is authoritative once a file has been plotted.
        # During startup restore there may be a selected row but no plotted file,
        # so fall back to the current list item in that case.
        active_path = getattr(self, "_current_loaded_path", None) or self._current_sample_path()
        removed_active = any(self._same_path(p, active_path) for p in removed_paths)

        self.file_list.blockSignals(True)
        for r in rows:
            self.file_list.takeItem(r)

        if removed_active or self.file_list.count() == 0:
            self.file_list.setCurrentRow(-1)
        else:
            # Preserve the actually loaded sample if it was not among the rows
            # removed.  Find its new row index after the deletions.
            keep_row = -1
            for i in range(self.file_list.count()):
                it = self.file_list.item(i)
                pth = it.data(Qt.ItemDataRole.UserRole) or it.text()
                if self._same_path(pth, active_path):
                    keep_row = i
                    break
            self.file_list.setCurrentRow(keep_row)
            self._confirmed_sample_row = keep_row
        self.file_list.blockSignals(False)

        if removed_active or self.file_list.count() == 0:
            self._update_ext_label(self.file_ext_label, "")
            self._clear_loaded_sample_state(reset_ranges=True)
            self.statusBar().showMessage(
                "Removed active sample — plot closed and q-range settings cleared.")
        else:
            # The active sample remains loaded.  If a different file happened to
            # be in hover-preview when Remove was pressed, return to the real one.
            self._sample_hover_preview_active = False
            self._update_ext_label(self.file_ext_label, active_path)
            win = self._active_plot_window
            snap = self._confirmed_plot_snapshot
            if win is not None and win.isVisible() and snap is not None:
                win.update_snapshot(self._copy_plot_snapshot(snap), reset_view=False)
            self.statusBar().showMessage("Removed selected sample file(s).")

        self._schedule_autosave()

    def _clear_sample_files(self):
        self.file_list.blockSignals(True)
        self.file_list.clear()
        self.file_list.setCurrentRow(-1)
        self.file_list.blockSignals(False)
        self._update_ext_label(self.file_ext_label, "")
        self._clear_loaded_sample_state(reset_ranges=True)
        # Clear means no sample remains, so comparison/static plot windows should
        # not remain on screen either.
        self._close_all_plot_windows()
        self.statusBar().showMessage(
            "Sample list cleared — all plots closed and q-range settings cleared.")
        self._schedule_autosave()

    def _suspend_hover_select(self):
        """Temporarily disable hover-to-select on both file lists."""
        for lst in (getattr(self, "file_list", None), getattr(self, "bkg_list", None)):
            if lst is not None:
                lst.hover_select = False

    def _resume_hover_select(self):
        """Re-enable hover-to-select on both file lists."""
        for lst in (getattr(self, "file_list", None), getattr(self, "bkg_list", None)):
            if lst is not None:
                lst.hover_select = True

    def _current_sample_path(self):
        it = self.file_list.currentItem()
        if it is None:
            return ""
        return it.data(Qt.ItemDataRole.UserRole) or it.text()

    def _on_sample_selected(self, row):
        """Single click / arrow key: load the sample and confirm the selection."""
        if row < 0:
            self._update_ext_label(self.file_ext_label, "")
            return
        path = self._current_sample_path()
        self._update_ext_label(self.file_ext_label, path)
        if path and os.path.exists(path):
            # This path is reached by a real selection change (click or arrow),
            # so it CONFIRMS the sample.
            self._confirmed_sample_row = row
            self._load_sample(path, new_window=False)

    def _serialized_ranges_for_preview(self, saved_ranges):
        """Convert serialized q-range rows into PlotWindow snapshot data.

        This is deliberately independent of the live UI widgets so merely
        hovering over a saved result can preview *its* ranges/manual points
        without changing the currently confirmed setup.
        """
        mode_map = {
            "WH Smooth": "wh",
            "Linear Interp": "linear",
            "Spline Interp": "spline",
            "Manual Spline": "manual",
        }
        row_colors = ['#E84855', '#FF6B35', '#9B59B6', '#00A8CC', '#27AE60']
        ranges = []
        manual = []
        for i, rs in enumerate(saved_ranges or []):
            if not bool(rs.get("enabled", True)):
                continue
            try:
                q0 = float(str(rs.get("q0", "")).replace(",", "."))
                q1 = float(str(rs.get("q1", "")).replace(",", "."))
            except Exception:
                continue
            if not q0 < q1:
                continue
            mode = mode_map.get(rs.get("mode_label", "WH Smooth"), "wh")
            r = {"q0": q0, "q1": q1, "mode": mode,
                 "blend": int(rs.get("blend", 0))}
            if mode == "wh":
                try:
                    r["lambda"] = float(str(rs.get("lambda", "1000")).replace(",", "."))
                except Exception:
                    r["lambda"] = 1000.0
                r["order"] = int(rs.get("order", 2))
            elif mode == "spline":
                r["anchor_pts"] = int(rs.get("anchor_pts", 10))
            elif mode == "manual":
                pts = []
                for pt in rs.get("manual_pts", []) or []:
                    try:
                        pts.append((float(pt[0]), float(pt[1])))
                    except Exception:
                        pass
                r["manual_pts"] = pts
                if pts:
                    manual.append((row_colors[i % len(row_colors)], pts))
            ranges.append(r)
        return ranges, manual

    def _hover_snapshot(self, path):
        """Build a view-only snapshot for a hovered sample file.

        No controller arrays, q-range widgets, selections, or settings are
        changed.  In particular, a self-contained WH result is decoded here so
        hover shows Raw + Background + Processed + Smoothed immediately.
        """
        # First try our self-contained WH result format.
        sess = self._read_saved_result(path)
        if isinstance(sess, dict):
            data = sess.get("data") or {}
            q = data.get("q")
            sm = data.get("y_smoothed")
            if q is not None and sm is not None:
                q = np.asarray(q, dtype=float)
                sm = np.asarray(sm, dtype=float)
                proc = data.get("y_processed")
                if proc is None:
                    proc = data.get("y_original")
                if proc is None:
                    proc = sm
                proc = np.asarray(proc, dtype=float)
                raw = data.get("y_raw")
                raw = proc if raw is None else np.asarray(raw, dtype=float)
                bkg = data.get("y_bkg_scaled")
                bkg = None if bkg is None else np.asarray(bkg, dtype=float)
                q_ref = data.get("q_ref")
                y_ref = data.get("y_ref")
                q_ref = None if q_ref is None else np.asarray(q_ref, dtype=float)
                y_ref = None if y_ref is None else np.asarray(y_ref, dtype=float)
                ranges, manual = self._serialized_ranges_for_preview(
                    sess.get("ranges", []))
                try:
                    alpha = float(sess.get("alpha_used", sess.get("alpha", 1.0)))
                except Exception:
                    alpha = 1.0
                try:
                    ref_scale = float(sess.get("ref_scale", 1.0))
                except Exception:
                    ref_scale = 1.0
                return {
                    "title": os.path.basename(path),
                    "q": q, "y_raw": raw, "y": proc,
                    "y_bkg_scaled": bkg, "y_smoothed": sm,
                    "alpha": alpha,
                    "data_type": sess.get("data_type", self._data_type),
                    "q_ref": q_ref, "y_ref": y_ref,
                    "ref_scale": ref_scale,
                    "use_log": bool(sess.get("log_y", self.log_chk.isChecked())),
                    "ranges": ranges, "manual": manual,
                }

        # Ordinary data file: preview it with the currently confirmed processing
        # settings, again without mutating the main GUI state.
        try:
            q, raw = load_file(path)
        except Exception:
            return None
        q = np.asarray(q, dtype=float)
        raw = np.asarray(raw, dtype=float)
        proc = raw.copy()
        bkg_scaled = None
        try:
            bkg_path = self._current_bkg_path().strip()
            if bkg_path and os.path.exists(bkg_path):
                q_b, y_b = load_file(bkg_path)
                alpha = float(self.alpha_edit.text().replace(",", "."))
                bkg_scaled = alpha * np.interp(q, q_b, y_b)
                proc = raw - bkg_scaled
            else:
                alpha = float(self.alpha_edit.text().replace(",", "."))
        except Exception:
            alpha = getattr(self, "_alpha_used", 1.0)
            bkg_scaled = None
            proc = raw.copy()

        # Detect type locally without changing radio buttons.
        base = os.path.basename(path).lower()
        stem, ext = os.path.splitext(base)
        type_map = {'.sq': 'S(q)', '.fq': 'F(q)', '.iq': 'I(q)',
                    '.chi': 'I(q)', '.gr': 'G(r)'}
        dtype = type_map.get(ext)
        if dtype is None:
            if 'sq' in stem or 's(q)' in stem:
                dtype = 'S(q)'
            elif 'fq' in stem or 'f(q)' in stem:
                dtype = 'F(q)'
            elif 'gr' in stem or 'g(r)' in stem:
                dtype = 'G(r)'
            else:
                dtype = self._data_type

        ranges = self._get_ranges()
        y_sm = None
        can_smooth = not (dtype == "I(q)" and bkg_scaled is None)
        if can_smooth:
            work_ranges = ranges
            if not work_ranges:
                try:
                    work_ranges = [{
                        "q0": float(q[0]), "q1": float(q[-1]),
                        "mode": "wh",
                        "lambda": float(self.lambda_edit.text().replace(",", ".")),
                        "order": self.order_spin.value(), "blend": 0,
                    }]
                except Exception:
                    work_ranges = []
            if work_ranges:
                try:
                    y_sm = range_smooth(q, proc, work_ranges)
                except Exception:
                    y_sm = None

        return {
            "title": os.path.basename(path),
            "q": q, "y_raw": raw, "y": proc,
            "y_bkg_scaled": bkg_scaled, "y_smoothed": y_sm,
            "alpha": alpha, "data_type": dtype,
            "q_ref": None if self._q_ref is None else self._q_ref.copy(),
            "y_ref": None if self._y_ref is None else self._y_ref.copy(),
            "ref_scale": self._ref_scale,
            "use_log": self.log_chk.isChecked(),
            "ranges": self._get_ranges(for_display=True),
            "manual": self._current_manual_snapshot(),
        }

    def _copy_plot_snapshot(self, snap):
        """Return a detached copy of a PlotWindow snapshot.

        Hover previews must never modify the remembered state of the actually
        selected file.  NumPy arrays are copied explicitly and the remaining
        nested lists/dicts (ranges/manual points) are deep-copied.
        """
        if snap is None:
            return None
        out = {}
        for key, value in snap.items():
            if isinstance(value, np.ndarray):
                out[key] = value.copy()
            else:
                out[key] = copy.deepcopy(value)
        return out

    def _remember_confirmed_plot(self, snap=None, path=None):
        """Remember the exact non-hover plot state for the selected file."""
        if snap is None:
            if self.q is None or self.y is None:
                self._confirmed_plot_snapshot = None
                self._confirmed_plot_path = None
                return
            snap = self._build_snapshot()
        self._confirmed_plot_snapshot = self._copy_plot_snapshot(snap)
        self._confirmed_plot_path = path or getattr(self, "_current_loaded_path", None)

    @staticmethod
    def _same_path(a, b):
        if not a or not b:
            return False
        try:
            return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))
        except Exception:
            return a == b

    def _replace_q_range_ui(self, saved_ranges=None):
        """Replace Section 5 q-range controls with one file's own workspace.

        Hover changes the active sample, so the q-range panel must change at the
        same time as the plot.  This helper removes every old row synchronously
        from the layout, creates the new rows, applies their serialized state,
        and explicitly invalidates/repaints the layout.  It is used for both
        saved WH results and ordinary files (which start with one fresh row).
        """
        # Stop manual editing tied to a row that is about to disappear.
        self._active_row = None
        for w in list(getattr(self, "_plot_windows", [])):
            if getattr(w, "_editor_row", None) is not None:
                try:
                    w.set_editor(None, self)
                except Exception:
                    pass

        for rr in list(self._get_rows()):
            self._range_layout.removeWidget(rr)
            rr.setParent(None)
            rr.deleteLater()

        saved_ranges = list(saved_ranges or [])
        if saved_ranges:
            for rs in saved_ranges:
                rr = self._add_range_row()
                rr.apply_state(rs)
        else:
            self._add_range_row()

        # Force the controls to visually update immediately while the pointer is
        # still moving between rows.  Without this, Qt can defer geometry/paint
        # work and the plot changes before Section 5 appears to change.
        try:
            self._range_layout.invalidate()
            parent = self._range_layout.parentWidget()
            if parent is not None:
                parent.updateGeometry()
                parent.update()
        except Exception:
            pass

    def _apply_saved_result_workspace_from_hover(self, row, path, sess, snap):
        """Make a self-contained WH result the live workspace during hover.

        ``Add...`` deliberately does not select or plot a file.  Therefore the
        first hover is the natural load action.  A saved ``*_wh_smoothed.chi``
        must restore not only its curves in the PlotWindow but also every saved
        processing control (especially q Ranges / Manual Spline points) so the
        user can move the pointer away from the file list and continue editing.

        This helper updates the live controller/UI without clearing the Sample
        list and without opening/rescaling a plot.  The caller displays the
        resulting snapshot with ``reset_view=False`` when a plot already exists.
        """
        if not isinstance(sess, dict) or snap is None:
            return snap

        self._restoring_hover_workspace = True
        try:
            # Hover is the activation action after Add: keep the row selected so
            # Save/Remove/current-path logic refers to the file the user just saw.
            self.file_list.blockSignals(True)
            try:
                self.file_list.setCurrentRow(row)
                self._confirmed_sample_row = row
            finally:
                self.file_list.blockSignals(False)

            # ---- data type ---------------------------------------------------
            dtype = sess.get("data_type", snap.get("data_type", self._data_type))
            if dtype in ("I(q)", "S(q)", "F(q)", "G(r)"):
                for btn in self._type_bg.buttons():
                    btn.blockSignals(True)
                    btn.setChecked(btn.text() == dtype)
                    btn.blockSignals(False)
                self._data_type = dtype

            # ---- scalar controls --------------------------------------------
            self.lambda_edit.blockSignals(True)
            self.lambda_edit.setText(str(sess.get("def_lambda", self.lambda_edit.text())))
            self.lambda_edit.blockSignals(False)

            self.order_spin.blockSignals(True)
            try:
                self.order_spin.setValue(int(sess.get("def_order", self.order_spin.value())))
            except Exception:
                pass
            self.order_spin.blockSignals(False)

            self.alpha_edit.blockSignals(True)
            self.alpha_edit.setText(str(sess.get("alpha", sess.get("alpha_used", self.alpha_edit.text()))))
            self.alpha_edit.blockSignals(False)
            try:
                self._alpha_used = float(sess.get("alpha_used", sess.get("alpha", snap.get("alpha", 1.0))))
            except Exception:
                self._alpha_used = float(snap.get("alpha", 1.0))

            self.log_chk.blockSignals(True)
            self.log_chk.setChecked(bool(sess.get("log_y", snap.get("use_log", self.log_chk.isChecked()))))
            self.log_chk.blockSignals(False)

            self.ref_scale_edit.blockSignals(True)
            self.ref_scale_edit.setText(str(sess.get("ref_scale", snap.get("ref_scale", self.ref_scale_edit.text()))))
            self.ref_scale_edit.blockSignals(False)
            try:
                self._ref_scale = float(sess.get("ref_scale", snap.get("ref_scale", 1.0)))
            except Exception:
                self._ref_scale = float(snap.get("ref_scale", 1.0))

            self.ref_edit.blockSignals(True)
            self.ref_edit.setText(str(sess.get("ref_path", "")))
            self.ref_edit.blockSignals(False)

            # ---- exact saved q-range UI -------------------------------------
            # Every hovered saved file owns its own Section-5 setup.  Replace the
            # previous file's rows immediately so moving A -> B updates both the
            # graph AND q Ranges for Smoothing in the same hover event.
            self._replace_q_range_ui(sess.get("ranges", []) or [])

            # ---- saved background metadata ---------------------------------
            # Keep existing entries, append the saved background if necessary,
            # and select it without recomputing subtraction.  The exact scaled
            # background curve is already embedded in the saved result.
            saved_bkgs = sess.get("bkg_files", []) or []
            if saved_bkgs:
                self.bkg_list.blockSignals(True)
                try:
                    existing = []
                    for i in range(self.bkg_list.count()):
                        it = self.bkg_list.item(i)
                        existing.append(it.data(Qt.ItemDataRole.UserRole) or it.text())
                    for pth in saved_bkgs:
                        if not any(self._same_path(pth, ex) for ex in existing):
                            if os.path.exists(pth) or snap.get("y_bkg_scaled") is not None:
                                if self._append_list_item(self.bkg_list, pth) is not None:
                                    existing.append(pth)
                    target = saved_bkgs[0]
                    target_row = -1
                    for i in range(self.bkg_list.count()):
                        it = self.bkg_list.item(i)
                        pth = it.data(Qt.ItemDataRole.UserRole) or it.text()
                        if self._same_path(pth, target):
                            target_row = i
                            break
                    self.bkg_list.setCurrentRow(target_row)
                    self._confirmed_bkg_row = target_row
                finally:
                    self.bkg_list.blockSignals(False)
                self._update_ext_label(self.bkg_ext_label, self._current_bkg_path())

            # ---- exact embedded curves become the live controller state -----
            self.q = np.asarray(snap["q"], dtype=float).copy()
            self._y_raw = (None if snap.get("y_raw") is None
                           else np.asarray(snap["y_raw"], dtype=float).copy())
            self.y = np.asarray(snap["y"], dtype=float).copy()
            self._y_bkg_scaled = (None if snap.get("y_bkg_scaled") is None
                                  else np.asarray(snap["y_bkg_scaled"], dtype=float).copy())
            self.y_smoothed = (None if snap.get("y_smoothed") is None
                               else np.asarray(snap["y_smoothed"], dtype=float).copy())
            self._q_ref = (None if snap.get("q_ref") is None
                           else np.asarray(snap["q_ref"], dtype=float).copy())
            self._y_ref = (None if snap.get("y_ref") is None
                           else np.asarray(snap["y_ref"], dtype=float).copy())
            self._current_loaded_name = os.path.basename(path)
            self._current_loaded_path = path

            # Rebuild from the live UI so the PlotWindow ranges/manual points and
            # the controls are guaranteed to describe the same saved workspace.
            live_snap = self._build_snapshot()
            self._remember_confirmed_plot(live_snap, path=path)
            self._sample_hover_preview_active = False
            self._update_ext_label(self.file_ext_label, path)
            return live_snap
        finally:
            self._restoring_hover_workspace = False

    def _preview_sample(self, row):
        """Hover a sample: activate it, update Section 5, and update the plot.

        A saved ``*_wh_smoothed.chi`` restores its embedded q-range/settings
        workspace.  An ordinary data file has no embedded smoothing workspace, so
        when the pointer moves to a *different* ordinary file Section 5 is reset
        to one fresh range row instead of leaving the previous file's q ranges on
        screen.  Thus the controls always describe the file currently plotted.
        """
        if row < 0 or row >= self.file_list.count():
            return
        it = self.file_list.item(row)
        path = it.data(Qt.ItemDataRole.UserRole) or it.text()
        if not (path and os.path.exists(path)):
            return
        self._update_ext_label(self.file_ext_label, path)

        previous_path = getattr(self, "_current_loaded_path", None)
        switching_file = not self._same_path(path, previous_path)

        # Saved WH result: its embedded ranges/settings are authoritative.
        sess = self._read_saved_result(path)
        if isinstance(sess, dict) and (sess.get("data") or {}).get("q") is not None:
            snap = self._hover_snapshot(path)
            if snap is None:
                self.statusBar().showMessage(path)
                return
            snap = self._apply_saved_result_workspace_from_hover(row, path, sess, snap)
        else:
            # Ordinary file: there is no saved Section-5 workspace.  When moving
            # from another sample, clear stale q ranges/manual points FIRST, then
            # build the preview so its range overlay matches the visible controls.
            if switching_file:
                self._restoring_hover_workspace = True
                try:
                    self._replace_q_range_ui([])
                finally:
                    self._restoring_hover_workspace = False

            snap = self._hover_snapshot(path)
            if snap is None:
                self.statusBar().showMessage(path)
                return

            self.file_list.blockSignals(True)
            try:
                self.file_list.setCurrentRow(row)
                self._confirmed_sample_row = row
            finally:
                self.file_list.blockSignals(False)

            self.q = np.asarray(snap["q"], dtype=float).copy()
            self._y_raw = (None if snap.get("y_raw") is None
                           else np.asarray(snap["y_raw"], dtype=float).copy())
            self.y = np.asarray(snap["y"], dtype=float).copy()
            self._y_bkg_scaled = (None if snap.get("y_bkg_scaled") is None
                                  else np.asarray(snap["y_bkg_scaled"], dtype=float).copy())
            self.y_smoothed = (None if snap.get("y_smoothed") is None
                               else np.asarray(snap["y_smoothed"], dtype=float).copy())
            self._current_loaded_name = os.path.basename(path)
            self._current_loaded_path = path
            self._sample_hover_preview_active = False

            # Auto-select the data type from the hovered file's extension /
            # name, matching the behaviour of a real load, so hovering to a
            # different file type updates the I(q)/S(q)/F(q)/G(r) selection.
            _base = os.path.basename(path).lower()
            _stem, _ext = os.path.splitext(_base)
            _type_map = {'.sq': 'S(q)', '.fq': 'F(q)',
                         '.iq': 'I(q)', '.chi': 'I(q)', '.gr': 'G(r)'}
            _detected = None
            if _ext in _type_map:
                _detected = _type_map[_ext]
            elif 'sq' in _stem or 's(q)' in _stem:
                _detected = 'S(q)'
            elif 'fq' in _stem or 'f(q)' in _stem:
                _detected = 'F(q)'
            elif 'gr' in _stem or 'g(r)' in _stem:
                _detected = 'G(r)'
            if _detected is not None:
                for _btn in self._type_bg.buttons():
                    _btn.setChecked(_btn.text() == _detected)

            snap = self._build_snapshot()
            self._remember_confirmed_plot(snap, path=path)

        self._plot_windows = [w for w in self._plot_windows if w.isVisible()]
        win = self._active_plot_window
        if win is None or not win.isVisible():
            win = PlotWindow(self._copy_plot_snapshot(snap), parent=self)
            self._position_plot_window(win)
            self._plot_windows.append(win)
            self._active_plot_window = win
            win.show()
            win.raise_()
        else:
            # Hovering another file must not disturb the user's zoom/pan.
            win.update_snapshot(self._copy_plot_snapshot(snap), reset_view=False)
            win.show()
            win.raise_()

        self._sample_hover_preview_active = False
        if isinstance(sess, dict) and (sess.get("data") or {}).get("q") is not None:
            msg = f"Active: {os.path.basename(path)} — saved plot and q-range settings restored."
        else:
            msg = f"Active: {os.path.basename(path)} — ordinary data; q-range panel reset for this file."
        self.statusBar().showMessage(msg)

    def _revert_sample_preview(self):
        """Compatibility fallback for legacy temporary previews.

        Normal sample hover now activates the hovered file and sets
        _sample_hover_preview_active=False, so moving the pointer to the plot
        or q-range controls leaves the active file and its saved settings intact.
        """
        self._update_ext_label(self.file_ext_label, self._current_sample_path())
        if not getattr(self, '_sample_hover_preview_active', False):
            return
        self._sample_hover_preview_active = False

        win = self._active_plot_window
        if win is None or not win.isVisible():
            return

        # The controller was never changed by hover, so use its live state.
        snap = self._build_snapshot() if self.q is not None and self.y is not None else None
        current = getattr(self, '_current_loaded_path', None)
        remembered = self._confirmed_plot_snapshot

        # Defensive saved-result guard: never let a transient None erase a
        # previously confirmed Smoothed curve when returning from hover.
        if (snap is not None and snap.get('y_smoothed') is None and
                remembered is not None and remembered.get('y_smoothed') is not None and
                self._same_path(getattr(self, '_confirmed_plot_path', None), current)):
            snap['y_smoothed'] = np.asarray(remembered['y_smoothed'], dtype=float).copy()
            self.y_smoothed = snap['y_smoothed'].copy()

        # If necessary, recover directly from the selected self-contained file.
        selected_path = self._current_sample_path()
        if (snap is None or snap.get('y_smoothed') is None) and selected_path:
            recovered = self._restore_saved_result_curves(selected_path, update_window=False)
            if recovered is not None:
                snap = recovered

        if snap is not None:
            self._remember_confirmed_plot(snap, path=current or selected_path)
            win.update_snapshot(self._copy_plot_snapshot(snap), reset_view=False)
            win.show()
            win.raise_()

    def _confirm_sample_click(self, row):
        """A real click on a row: record it as the confirmed sample.

        The actual load happens via currentRowChanged → _on_sample_selected
        (Qt updates the current row on click), so we only record confirmation
        here to avoid loading twice.
        """
        if 0 <= row < self.file_list.count():
            self._confirmed_sample_row = row

    def _on_sample_double_clicked(self, item):
        """Double click: open the sample in a NEW plot window for comparison."""
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole) or item.text()
        if path and os.path.exists(path):
            self._load_sample(path, new_window=True)

    def _reset_ranges(self):
        """Remove all q-range rows and add a single fresh one.

        Called when a different file is loaded so leftover ranges / manual
        points from the previous file don't carry over to the new data.
        """
        # Stop any manual editing in progress.
        self._active_row = None
        for w in list(self._plot_windows):
            if getattr(w, "_editor_row", None) is not None:
                try:
                    w.set_editor(None, self)
                except Exception:
                    pass
        # Remove every existing range row.
        for row in self._get_rows():
            self._range_layout.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        # Start the new file with one empty range row.
        self._add_range_row()

    def _read_embedded_session(self, path):
        """Return the embedded metadata dict from a WH-smoothed file, or None."""
        try:
            with open(path, "r", encoding="latin-1") as f:
                for line in f:
                    s = line.strip().lstrip("#").strip()
                    if s.startswith("WHSES-JSON:"):
                        return json.loads(s[len("WHSES-JSON:"):].strip())
                    # Stop once numeric data begins (JSON line is in the header).
                    if s and (s[0].isdigit() or s[0] in "+-."):
                        break
        except Exception:
            pass
        return None

    def _read_saved_result(self, path):
        """Read a self-contained WH result and return a complete session dict.

        IMPORTANT: saved WH files are parsed BEFORE the generic two-column
        loader.  Version 4 uses five numeric columns:

            q, y_smoothed, y_processed, y_raw, alpha_times_background

        This makes q + smoothed the first two columns (useful to ordinary .chi
        readers) while still preserving every curve needed by this GUI.  Older
        v2/v3 files are supported through their embedded data arrays, and a
        three-column v3 body can be used as a fallback.
        """
        sess = self._read_embedded_session(path)
        if not isinstance(sess, dict):
            return None

        data = dict(sess.get("data") or {})
        layout = sess.get("data_layout", "")
        try:
            version = int(sess.get("version", 0))
        except Exception:
            version = 0

        # New v4: the numeric body is authoritative for the main curves.
        if version >= 4 or layout == "q_smoothed_processed_raw_bkg_v4":
            try:
                arr = np.loadtxt(path, comments="#", ndmin=2)
                if arr.ndim != 2 or arr.shape[1] < 5 or arr.shape[0] < 2:
                    raise ValueError("WH v4 file needs at least five numeric columns")

                q = np.asarray(arr[:, 0], dtype=float)
                sm = np.asarray(arr[:, 1], dtype=float)
                proc = np.asarray(arr[:, 2], dtype=float)
                raw = np.asarray(arr[:, 3], dtype=float)
                bkg = np.asarray(arr[:, 4], dtype=float)

                if not (np.all(np.isfinite(q)) and np.all(np.isfinite(sm))
                        and np.all(np.isfinite(proc)) and np.all(np.isfinite(raw))):
                    raise ValueError("Non-finite q/main data in WH v4 file")

                data["q"] = q
                data["y_smoothed"] = sm
                data["y_processed"] = proc
                data["y_original"] = proc       # compatibility alias
                data["y_raw"] = raw

                # Missing background is represented by an all-NaN fifth column.
                finite_bkg = np.isfinite(bkg)
                if finite_bkg.any():
                    if not finite_bkg.all():
                        raise ValueError("Partially invalid background column")
                    data["y_bkg_scaled"] = bkg
                else:
                    data["y_bkg_scaled"] = None

                sess["data"] = data
                return sess
            except Exception as e:
                self.statusBar().showMessage(
                    f"Cannot read embedded WH data from {os.path.basename(path)}: {e}")
                return None

        # v2/v3: normally all curves were duplicated inside WHSES-JSON.
        if data.get("q") is not None and data.get("y_smoothed") is not None:
            sess["data"] = data
            return sess

        # Fallback for a v3-style numeric body: q, processed, smoothed.
        try:
            arr = np.loadtxt(path, comments="#", ndmin=2)
            if arr.ndim == 2 and arr.shape[1] >= 3 and arr.shape[0] >= 2:
                data["q"] = np.asarray(arr[:, 0], dtype=float)
                data["y_processed"] = np.asarray(arr[:, 1], dtype=float)
                data["y_original"] = data["y_processed"]
                data["y_smoothed"] = np.asarray(arr[:, 2], dtype=float)
                if data.get("y_raw") is None:
                    data["y_raw"] = data["y_processed"].copy()
                sess["data"] = data
                return sess
        except Exception:
            pass

        return sess

    def _restore_saved_result_curves(self, path, update_window=True):
        """Re-assert all embedded curves from a self-contained saved WH file.

        The saved file is authoritative for q, Raw, alpha*Background, Processed,
        and Smoothed arrays.  This helper is intentionally called once again
        after the load event returns to the Qt event loop because some widget
        signals generated while restoring settings can arrive late and clear the
        live Smoothed array.  Re-asserting the arrays makes the selected file's
        full plot persistent, independent of hover state.

        Returns the rebuilt live snapshot, or None when ``path`` is not a valid
        self-contained WH result.
        """
        if not path or not os.path.exists(path):
            return None
        sess = self._read_saved_result(path)
        if not isinstance(sess, dict):
            return None
        data = sess.get("data") or {}
        q = data.get("q")
        sm = data.get("y_smoothed")
        if q is None or sm is None:
            return None

        proc = data.get("y_processed")
        if proc is None:
            proc = data.get("y_original")
        if proc is None:
            proc = sm
        raw = data.get("y_raw")
        if raw is None:
            raw = proc
        bkg = data.get("y_bkg_scaled")

        self.q = np.asarray(q, dtype=float).copy()
        self.y = np.asarray(proc, dtype=float).copy()
        self._y_raw = np.asarray(raw, dtype=float).copy()
        self._y_bkg_scaled = (None if bkg is None
                              else np.asarray(bkg, dtype=float).copy())
        self.y_smoothed = np.asarray(sm, dtype=float).copy()

        q_ref = data.get("q_ref")
        y_ref = data.get("y_ref")
        if q_ref is not None and y_ref is not None:
            self._q_ref = np.asarray(q_ref, dtype=float).copy()
            self._y_ref = np.asarray(y_ref, dtype=float).copy()

        try:
            self._alpha_used = float(sess.get("alpha_used", sess.get("alpha", self._alpha_used)))
        except Exception:
            pass
        try:
            self._ref_scale = float(sess.get("ref_scale", self._ref_scale))
        except Exception:
            pass

        self._current_loaded_name = os.path.basename(path)
        self._current_loaded_path = path

        # Use the CURRENT UI ranges/manual points so subsequent edits continue
        # naturally, while the curve arrays themselves come from the saved file.
        snap = self._build_snapshot()
        self._remember_confirmed_plot(snap, path=path)

        if update_window:
            win = self._active_plot_window
            if win is not None and win.isVisible():
                win.update_snapshot(self._copy_plot_snapshot(snap), reset_view=False)
                win.show()
                win.raise_()
        return snap

    def _finalize_saved_result_after_load(self, path):
        """Post-event-loop guard that keeps the saved Smoothed curve persistent.

        A guard is essential here because this function is also scheduled by
        QTimer.singleShot().  If the user removes the file before that timer runs,
        an empty selection must NOT allow the removed file to be restored again.
        """
        selected = self._current_sample_path()
        current = getattr(self, "_current_loaded_path", None)
        if not selected or not current:
            return
        if not self._same_path(selected, path) or not self._same_path(current, path):
            return
        self._restore_saved_result_curves(path, update_window=True)

    def _load_sample(self, path, new_window=False, silent=False):
        """Load the given sample file as the active dataset and refresh plots.

        `silent=True` (used for hover preview) reports problems in the status
        bar instead of a pop-up, so moving the mouse over a bad file doesn't
        spawn a dialog on every hover.
        """
        # Session / non-data files (.whses, .json) are not sample data — skip
        # them quietly rather than failing to parse them as two-column data.
        ext = os.path.splitext(path)[1].lower()
        if ext in (".whses", ".json"):
            self.statusBar().showMessage(
                f"{os.path.basename(path)} is a session file, not sample data "
                "— use File ▸ Load Session to open it.")
            return
        # Saved WH result?  Parse it BEFORE the generic loader so a multi-column
        # self-contained file is never mistaken for ordinary two-column data.
        if not silent and not getattr(self, "_restoring_session", False):
            sess = self._read_saved_result(path)
            if sess:
                # IMPORTANT: opening a self-contained WH result must restore the
                # file's settings/data WITHOUT destroying files that are already
                # present in the Sample/Background lists.  This matters especially
                # for Add..., which appends a batch and then selects the first new
                # item; the old clear_first=True behaviour erased the entire batch
                # (and every previously loaded file) as soon as that first saved
                # result was selected.
                sample_paths_before = []
                for i in range(self.file_list.count()):
                    it = self.file_list.item(i)
                    sample_paths_before.append(
                        it.data(Qt.ItemDataRole.UserRole) or it.text())

                bkg_paths_before = []
                for i in range(self.bkg_list.count()):
                    it = self.bkg_list.item(i)
                    bkg_paths_before.append(
                        it.data(Qt.ItemDataRole.UserRole) or it.text())
                bkg_current_before = self._current_bkg_path()

                # Point the restored UI/data at THIS self-contained result file.
                # clear_first=True is still useful internally because it guarantees
                # a clean settings/data restore; the list contents are rebuilt just
                # below from the snapshots so Add... remains non-destructive.
                sess["saved_file_path"] = path
                sess["sample_files"] = [path]
                sess["sample_row"] = 0
                self._apply_session(sess, clear_first=True)

                # Rebuild the Sample list exactly as it looked before loading the
                # embedded result.  The just-opened `path` is already in this
                # snapshot because selection came from the list (including Add...).
                # If _load_sample was called programmatically, make sure it is added.
                if path not in sample_paths_before:
                    sample_paths_before.append(path)
                self.file_list.blockSignals(True)
                self.file_list.clear()
                active_row = -1
                for pth in sample_paths_before:
                    item = self._append_list_item(self.file_list, pth)
                    if item is not None and os.path.normcase(os.path.abspath(pth)) == \
                            os.path.normcase(os.path.abspath(path)):
                        active_row = self.file_list.row(item)
                if active_row >= 0:
                    self.file_list.setCurrentRow(active_row)
                    self._confirmed_sample_row = active_row
                self.file_list.blockSignals(False)
                self._update_ext_label(self.file_ext_label, path)

                # Preserve the Background list too.  Merge in any background path
                # recorded by the saved result (without duplicates) so its metadata
                # remains available while previously loaded backgrounds stay listed.
                merged_bkg = list(bkg_paths_before)
                for pth in sess.get("bkg_files", []) or []:
                    if pth not in merged_bkg:
                        merged_bkg.append(pth)
                saved_bkg_paths = sess.get("bkg_files", []) or []
                target_bkg = saved_bkg_paths[0] if saved_bkg_paths else bkg_current_before

                self.bkg_list.blockSignals(True)
                self.bkg_list.clear()
                active_bkg_row = -1
                for pth in merged_bkg:
                    item = self._append_list_item(self.bkg_list, pth)
                    if (item is not None and target_bkg and
                            os.path.normcase(os.path.abspath(pth)) ==
                            os.path.normcase(os.path.abspath(target_bkg))):
                        active_bkg_row = self.bkg_list.row(item)
                if active_bkg_row >= 0:
                    self.bkg_list.setCurrentRow(active_bkg_row)
                    self._confirmed_bkg_row = active_bkg_row
                else:
                    self.bkg_list.setCurrentRow(-1)
                    self._confirmed_bkg_row = -1
                self.bkg_list.blockSignals(False)
                self._update_ext_label(self.bkg_ext_label, self._current_bkg_path())

                # Re-decode the saved result as an exact view snapshot, then make
                # those arrays authoritative in the controller as well.  This is
                # important because the hover path always had the complete raw
                # curve, while older controller restore paths could later lose it
                # when the mouse left the Sample list.
                loaded_snap = self._hover_snapshot(path)
                if loaded_snap is None:
                    loaded_snap = self._build_snapshot()
                else:
                    self.q = np.asarray(loaded_snap["q"], dtype=float).copy()
                    self._y_raw = (None if loaded_snap.get("y_raw") is None
                                   else np.asarray(loaded_snap["y_raw"], dtype=float).copy())
                    self.y = np.asarray(loaded_snap["y"], dtype=float).copy()
                    self._y_bkg_scaled = (None if loaded_snap.get("y_bkg_scaled") is None
                                          else np.asarray(loaded_snap["y_bkg_scaled"], dtype=float).copy())
                    self.y_smoothed = (None if loaded_snap.get("y_smoothed") is None
                                       else np.asarray(loaded_snap["y_smoothed"], dtype=float).copy())
                    if loaded_snap.get("q_ref") is not None and loaded_snap.get("y_ref") is not None:
                        self._q_ref = np.asarray(loaded_snap["q_ref"], dtype=float).copy()
                        self._y_ref = np.asarray(loaded_snap["y_ref"], dtype=float).copy()
                    self._current_loaded_name = os.path.basename(path)
                    self._current_loaded_path = path
                    # Rebuild once from controller state so subsequent range edits
                    # use the same Raw/Background/Processed arrays as this display.
                    loaded_snap = self._build_snapshot()

                self._remember_confirmed_plot(loaded_snap, path=path)
                win = self._active_plot_window
                if win is not None and win.isVisible():
                    win.update_snapshot(self._copy_plot_snapshot(loaded_snap),
                                        reset_view=True)
                    win.show()
                    win.raise_()

                # Some Qt widget signals created while restoring the embedded
                # settings can be delivered after this function returns.  A late
                # background/range refresh used to clear y_smoothed, which is why
                # the Smoothed curve appeared only while hovering the filename.
                # Re-assert the saved arrays after the event queue settles.
                QTimer.singleShot(0, lambda p=path: self._finalize_saved_result_after_load(p))
                QTimer.singleShot(75, lambda p=path: self._finalize_saved_result_after_load(p))

                self.statusBar().showMessage(
                    f"Loaded {os.path.basename(path)} — settings + raw/background/"
                    "processed/smoothed data restored; existing file lists preserved.")
                return

        try:
            q, y = load_file(path)
        except Exception as e:
            # Bad data file (wrong content, binary, empty, one column, …).
            # Report it without a modal pop-up so repeated hovers/clicks on a
            # stray file don't spam dialogs and block the user. The message is
            # shown in the status bar instead.
            self.statusBar().showMessage(
                f"Cannot load {os.path.basename(path)}: "
                "not a valid two-column data file.")
            return

        # If this is a different file from the one currently loaded, clear the
        # previous q-range / manual-point setup so it doesn't apply to new data.
        new_name = os.path.basename(path)
        if getattr(self, "_current_loaded_name", None) not in (None, new_name):
            self._reset_ranges()

        self.q = q
        self.y = y
        self._y_raw = y.copy()   # keep original for re-subtracting bkg
        self.y_smoothed = None
        self._y_bkg_scaled = None
        # Remember the file that was actually loaded so the plot window title
        # and session-save target reflect it (currentItem may differ, e.g. when
        # the mouse hovers over another row).
        self._current_loaded_name = new_name
        self._current_loaded_path = path

        # Auto-detect the data type, first from the extension and then, for
        # generic extensions such as .dat/.xy/.txt, from the file name itself
        # (e.g. "sample_sq.dat" is treated as S(q)). Anything still unknown is
        # assumed to be raw intensity, I(q).
        # Detect the data type from the extension first, then from the file
        # name. If neither gives a clear answer, DON'T guess — leave the
        # currently selected type unchanged so the user can pick it themselves.
        base = os.path.basename(path).lower()
        stem, ext = os.path.splitext(base)

        type_map = {'.sq': 'S(q)', '.fq': 'F(q)',
                    '.iq': 'I(q)', '.chi': 'I(q)',
                    '.gr': 'G(r)'}

        detected = None
        if ext in type_map:
            detected = type_map[ext]
        elif 'sq' in stem or 's(q)' in stem:
            detected = 'S(q)'
        elif 'fq' in stem or 'f(q)' in stem:
            detected = 'F(q)'
        elif 'gr' in stem or 'g(r)' in stem:
            detected = 'G(r)'

        if detected is not None:
            for btn in self._type_bg.buttons():
                btn.setChecked(btn.text() == detected)
        else:
            # Unknown type: keep the current selection and let the user know.
            self.statusBar().showMessage(
                f"Loaded {os.path.basename(path)} — data type not recognised; "
                f"please select I(q) / S(q) / F(q) / G(r) manually.")

        # If a background is selected, re-subtract it against the newly loaded
        # sample (works for any data type, not just I(q)).
        if self._current_bkg_path():
            self._subtract_bkg()

        # Update the plot. A single click refreshes the active window in place;
        # a double-click (see _on_sample_double_clicked) opens a new window so
        # samples can be compared side by side. Either way the data is new, so
        # the axes refit to it (reset_view) instead of keeping an off-screen
        # zoom from the previous file.
        if new_window:
            self._new_plot_window()
        else:
            self._refresh_active_window(reset_view=True)

        if not silent:
            self.statusBar().showMessage(
                f"Loaded: {os.path.basename(path)}  |  "
                f"{len(q)} points,  q = {q[0]:.3f} ~ {q[-1]:.3f} Å⁻¹")
        # If this file was produced by this tool (has a WH-smoothed header),
        # surface the processing info so the user sees what was done to it.
        # Skipped during hover so the full-path message stays visible.
        if not silent:
            self._report_smoothed_header(path)
        # Live smoothing: with no range defined this smooths the whole q range
        # using the Section 4 defaults, so a result shows without any button.
        # During a hover preview we set a flag so _auto_reapply does NOT write
        # an autosave file — hovering to glance at files shouldn't save anything.
        self._previewing = silent
        try:
            self._auto_reapply()
        finally:
            self._previewing = False

    def _apply_smoothing(self):
        if self.y is None:
            QMessageBox.warning(self, "No Data", "Load a file first.")
            return
        # I(q) mode: must subtract background first
        if self._data_type == "I(q)" and self._y_bkg_scaled is None:
            QMessageBox.warning(self, "Background Required",
                "I(q) mode: add and select a background file first (Section 2).")
            return
        ranges = self._get_ranges()

        # No ranges defined → use Section 4 defaults to WH-smooth the full q range
        if not ranges:
            try:
                def_lam = float(self.lambda_edit.text())
            except ValueError:
                QMessageBox.warning(self, "Bad Parameter",
                    "Section 4 default lambda is invalid.")
                return
            def_ord = self.order_spin.value()
            ranges = [{
                "q0": float(self.q[0]), "q1": float(self.q[-1]),
                "mode": "wh", "lambda": def_lam, "order": def_ord,
                "blend": 0,
            }]
            full_range_mode = True
        else:
            full_range_mode = False

        try:
            self.y_smoothed = range_smooth(self.q, self.y, ranges)
        except Exception as e:
            QMessageBox.critical(self, "Smoothing Error", str(e))
            return

        # Update the active plot window in place with the smoothed result and
        # the difference plot — no new window is opened.
        self._refresh_active_window()

        n_pts = sum(((self.q >= r["q0"]) & (self.q <= r["q1"])).sum()
                    for r in ranges)

        def _r_summary(r):
            m = r["mode"]
            if m == "wh":
                return (f"q[{r['q0']}–{r['q1']}] WH λ={r['lambda']:g} "
                        f"ord={r['order']} blend={r['blend']}")
            elif m == "linear":
                return f"q[{r['q0']}–{r['q1']}] Linear blend={r['blend']}"
            elif m == "spline":
                return (f"q[{r['q0']}–{r['q1']}] Spline "
                        f"anc={r.get('anchor_pts', '?')} blend={r['blend']}")
            elif m == "manual":
                n = len(r.get('manual_pts', []))
                return f"q[{r['q0']}–{r['q1']}] Manual({n} pts) blend={r['blend']}"
            else:
                return f"q[{r['q0']}–{r['q1']}] {m} blend={r['blend']}"

        if full_range_mode:
            r = ranges[0]
            self.statusBar().showMessage(
                f"Smoothed full range [{r['q0']:.3f}–{r['q1']:.3f}] using "
                f"Section 4 defaults: λ={r['lambda']:g} ord={r['order']}  ({n_pts} pts)")
        else:
            self.statusBar().showMessage(
                f"Processed {n_pts} pts across {len(ranges)} range(s):  "
                + "  |  ".join(_r_summary(r) for r in ranges))

    def _on_alpha_committed(self):
        """Alpha committed with Enter/focus-out: re-subtract, re-smooth, rescale.

        Subtraction clears the smoothed curve, so if a result existed we re-run
        smoothing with the new alpha. Either way the view is rescaled so the
        (much smaller) background-subtracted curve is visible.
        """
        had_result = self.y_smoothed is not None
        # Recompute I - alpha*bkg. This clears y_smoothed and already rescales.
        self._subtract_bkg()
        if had_result and self.y is not None:
            ranges = self._get_ranges()
            if ranges:
                try:
                    self.y_smoothed = range_smooth(self.q, self.y, ranges)
                except Exception:
                    pass
        # Rescale to the new (subtracted) data so it's on-screen.
        self._refresh_active_window(reset_view=True)

    def _auto_reapply(self):
        """Live update on parameter edits — smooths automatically.

        Whenever a valid q range exists, the smoothing is (re)computed and the
        plot updates as you type. If NO range is defined, the whole q range is
        smoothed using the Section 4 "Default WH Parameters" (λ, order), so
        editing those fields updates the plot too. There is no separate "Apply"
        step — everything is live.
        """
        # Programmatic restoration of a saved hover workspace changes many row
        # widgets in sequence.  Do not recompute smoothing from half-restored
        # controls; the exact saved Smoothed array is restored after the UI.
        if getattr(self, "_restoring_hover_workspace", False):
            return
        if self.y is None:
            return
        # I(q) needs a background subtracted first; until then just show the
        # region overlay.
        if self._data_type == "I(q)" and self._y_bkg_scaled is None:
            self._refresh_active_window()
            return
        ranges = self._get_ranges()
        full_range = False
        if not ranges:
            # No explicit range → smooth the ENTIRE q range with the Section 4
            # default WH parameters. Editing those fields re-smooths live.
            try:
                def_lam = float(self.lambda_edit.text().replace(",", "."))
            except ValueError:
                # Default lambda not a valid number yet — just refresh overlay.
                if self.y_smoothed is not None:
                    self.y_smoothed = None
                self._refresh_active_window()
                return
            ranges = [{
                "q0": float(self.q[0]), "q1": float(self.q[-1]),
                "mode": "wh", "lambda": def_lam,
                "order": self.order_spin.value(), "blend": 0,
            }]
            full_range = True
        try:
            self.y_smoothed = range_smooth(self.q, self.y, ranges)
        except Exception:
            # Bad/in-progress parameter (e.g. blank field) — leave as-is.
            return
        self._refresh_active_window()
        if full_range and not getattr(self, "_previewing", False):
            self.statusBar().showMessage(
                f"Smoothed full range with defaults: "
                f"λ={ranges[0]['lambda']:g}  order={ranges[0]['order']}")
        # A real smoothing was computed, so persist the workspace internally in
        # QSettings (no .whses file). Hover previews do not persist anything.
        if not getattr(self, "_previewing", False):
            self._schedule_autosave()

    def _save_result(self):
        if self.y_smoothed is None:
            QMessageBox.warning(self, "No Result",
                                "Adjust parameters to smooth first.")
            return
        # "Save Smoothed Data" produces ONE self-contained data file.  All
        # settings and all plotted curves are embedded in its WHSES-JSON header;
        # no automatic .whses sidecar is created.
        src = self._current_sample_path()
        base = os.path.splitext(os.path.basename(src))[0]
        # Default extension follows the data type (I(q)→.chi, S(q)→.sq,
        # F(q)→.fq, G(r)→.gr) so the saved file name reflects its content.
        ext_by_type = {"I(q)": ".chi", "S(q)": ".sq",
                       "F(q)": ".fq", "G(r)": ".gr"}
        def_ext = ext_by_type.get(self._data_type, ".chi")
        default = os.path.join(self._last_dir, f"{base}_wh_smoothed{def_ext}")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Smoothed Data", default,
            "Chi Files (*.chi);;I(q) Files (*.iq);;S(q) Files (*.sq);;"
            "F(q) Files (*.fq);;G(r) Files (*.gr);;XY Files (*.xy);;"
            "Text Files (*.txt *.dat);;All Files (*)")
        if not path:
            return

        ranges = self._get_ranges()
        def _r_str(r):
            m = r["mode"]
            if m == "wh":
                return (f"q[{r['q0']}-{r['q1']}] WH lambda={r['lambda']:g} "
                        f"order={r['order']} blend={r['blend']}")
            elif m == "linear":
                return f"q[{r['q0']}-{r['q1']}] Linear blend={r['blend']}"
            elif m == "spline":
                return (f"q[{r['q0']}-{r['q1']}] Spline "
                        f"anchor_pts={r.get('anchor_pts', '?')} blend={r['blend']}")
            elif m == "manual":
                pts = r.get('manual_pts', [])
                pts_str = "; ".join(f"({q:g},{y:g})" for q, y in pts)
                return (f"q[{r['q0']}-{r['q1']}] Manual({len(pts)} pts) "
                        f"blend={r['blend']}  points=[{pts_str}]")
            else:
                return f"q[{r['q0']}-{r['q1']}] {m} blend={r['blend']}"

        # Full record of everything done to this file.
        header_lines = [
            f"WH Range-Smoothed  [{self._data_type}]",
            f"Source       : {src}",
            f"Data type    : {self._data_type}",
        ]
        # Background info (if used).
        bkg = self._current_bkg_path()
        if bkg:
            header_lines.append(
                f"Background   : {os.path.basename(bkg)}  (alpha={self._alpha_used:g})")
        # Reference info (if used).
        if self._y_ref is not None:
            header_lines.append(f"Reference    : scale={self._ref_scale:g}")
        # Log axis + each range.
        header_lines.append(
            f"Log Y-axis   : {'on' if self.log_chk.isChecked() else 'off'}")
        if ranges:
            header_lines.append(f"Ranges ({len(ranges)}):")
            for i, r in enumerate(ranges, 1):
                header_lines.append(f"  [{i}] {_r_str(r)}")
        else:
            header_lines.append(
                f"Ranges       : (none — full-range WH, "
                f"lambda={self.lambda_edit.text()} order={self.order_spin.value()})")
        header_lines.append("")
        header_lines.append(
            "WH data format v4: first two columns are q + final smoothed data; "
            "remaining columns preserve the full plotted state")
        # Metadata contains every setting plus reference data.  Main same-grid
        # curves are stored once in the numeric body (not duplicated in JSON).
        try:
            sess_json = json.dumps(self._collect_file_session(), separators=(",", ":"))
            header_lines.append("")
            header_lines.append("WHSES-JSON: " + sess_json)
        except Exception:
            pass

        # Keep the human-readable column definition as the LAST header line so
        # it sits immediately above the first numeric data row in the saved file.
        # This makes the five-column layout obvious when the .chi file is opened
        # directly in a text editor or another analysis program.
        header_lines.append("")
        header_lines.append(
            "Columns: q(A-1)   y_smoothed   y_processed   y_raw   alpha_x_background")

        # One self-contained file, five numeric columns.  The first two are q and
        # y_smoothed so ordinary two-column readers still see the FINAL result.
        # Missing background is represented by NaN only in the fifth column.
        try:
            raw_col = (self._y_raw if self._y_raw is not None else self.y)
            if self._y_bkg_scaled is None:
                bkg_col = np.full_like(self.q, np.nan, dtype=float)
            else:
                bkg_col = self._y_bkg_scaled
            body = np.column_stack([
                self.q, self.y_smoothed, self.y, raw_col, bkg_col
            ])
            np.savetxt(path,
                       body,
                       header="\n".join(header_lines),
                       fmt="%.8f")
        except Exception as e:
            QMessageBox.critical(
                self, "Save Failed",
                f"Could not save the smoothed data:\n{e}")
            return

        self._last_dir = os.path.dirname(path)
        self._settings.setValue("last_dir", self._last_dir)
        self._settings.sync()
        self.statusBar().showMessage(f"Saved → {os.path.basename(path)}")
        # Keep the internal QSettings snapshot in sync with the just-saved state.
        self._save_session()

    # ── Range rows ────────────────────────────────────────────

    def _report_smoothed_header(self, path):
        """If `path` is a WH-smoothed file, note it quietly in the status bar.

        Files saved by "Save Smoothed Data" carry a header describing how they
        were processed. We just flag that in the status bar — no pop-up — so
        browsing files doesn't interrupt the user.
        """
        try:
            with open(path, "r", encoding="latin-1") as f:
                head = f.read(400)
            if "WH Range-Smoothed" in head:
                self.statusBar().showMessage(
                    f"{os.path.basename(path)} is a WH-smoothed file "
                    "(processing info is in its header).")
        except Exception:
            pass   # never let header reading break a normal load

    def _add_range_row(self):
        try:
            def_lam = float(self.lambda_edit.text())
        except ValueError:
            def_lam = 1000
        def_ord = self.order_spin.value()
        row = QRangeRow(default_lambda=def_lam, default_order=def_ord)
        row.btn_del.clicked.connect(lambda: self._del_row(row))
        # Any change to a range recomputes the smoothing (if a result exists)
        # and refreshes the active plot window, so edits update the curve on
        # Enter. Before any smoothing, only the region overlay updates.
        row.chk.stateChanged.connect(lambda _: self._auto_reapply())
        row.q0.editingFinished.connect(self._auto_reapply)
        row.q1.editingFinished.connect(self._auto_reapply)
        row.q0.returnPressed.connect(self._auto_reapply)
        row.q1.returnPressed.connect(self._auto_reapply)
        row.mode_combo.currentIndexChanged.connect(
            lambda _=0: self._auto_reapply())
        row.blend_spin.valueChanged.connect(
            lambda _=0: self._auto_reapply())
        # WH parameters (lambda / order) and spline anchor count also update.
        row.lambda_edit.editingFinished.connect(self._auto_reapply)
        row.lambda_edit.returnPressed.connect(self._auto_reapply)
        row.order_spin.valueChanged.connect(
            lambda _=0: self._auto_reapply())
        row.anchor_spin.valueChanged.connect(
            lambda _=0: self._auto_reapply())
        row.manual_btn.toggled.connect(
            lambda checked, r=row: self._on_manual_btn_toggled(r, checked))
        # Refresh the active window after Clear / Undo so the dots disappear
        # from the plot immediately.
        row.manual_clear_btn.clicked.connect(self._auto_reapply)
        row.manual_undo_btn.clicked.connect(self._auto_reapply)
        idx = self._range_layout.count() - 1
        self._range_layout.insertWidget(idx, row)
        return row

    def _on_manual_btn_toggled(self, row, checked):
        if checked:
            if self._active_row and self._active_row is not row:
                self._active_row.manual_btn.setChecked(False)
            self._active_row = row
            # Manual points are placed by clicking inside a plot window. Make
            # sure one is open and tell it which row is being edited.
            if not any(w.isVisible() for w in self._plot_windows):
                self._open_plot_window()
            self._set_active_editor_window(row)
            self.statusBar().showMessage(
                "Manual Spline: in the newest plot window, "
                "LEFT-CLICK = add point, RIGHT-CLICK = delete nearest "
                "(uncheck 'Edit Points' to finish)")
        else:
            if self._active_row is row:
                self._active_row = None
            self._set_active_editor_window(None)
            self.statusBar().showMessage(
                f"Manual editing stopped  ({len(row._manual_pts)} pts stored)")

    def _refresh_editor_manual(self):
        """Redraw manual anchor points on the current editor window.

        Called after Clear / Undo so the dots disappear from the open plot
        window immediately, instead of leaving the previous points behind.
        """
        for w in self._plot_windows:
            if getattr(w, "_editor_row", None) is not None and w.isVisible():
                w._manual = self._current_manual_snapshot()
                w._redraw_manual()

    def _set_active_editor_window(self, row):
        """Enable manual-point editing on the newest visible plot window."""
        target = None
        for w in reversed(self._plot_windows):
            if w.isVisible():
                target = w
                break
        for w in self._plot_windows:
            # Only the newest window edits; others are static snapshots.
            w.set_editor(row if (w is target) else None, self)

    def _manual_point_added(self, row, q_c, y_c):
        """Add a manual point, but only if the click is inside the row's q range.

        Returns True if the point was added, False if the click was outside
        [q0, q1] (so it's ignored — no stray points outside the region).
        """
        try:
            q0 = float(row.q0.text().replace(",", "."))
            q1 = float(row.q1.text().replace(",", "."))
        except ValueError:
            q0, q1 = None, None
        if q0 is not None and q1 is not None and not (q0 <= q_c <= q1):
            self.statusBar().showMessage(
                f"Ignored: q={q_c:.4f} is outside the range [{q0:g}, {q1:g}]")
            return False
        row.add_manual_point(q_c, y_c)
        n = len(row._manual_pts)
        self.statusBar().showMessage(
            f"Added: q={q_c:.4f}, y={y_c:.4g}  ({n} pts)")
        # Live update: once there are ≥2 points the manual spline is computed
        # and the curve appears/updates without pressing Apply.
        self._auto_reapply()
        return True

    def _manual_point_deleted(self, row, q_c, y_c):
        """Called by a PlotWindow when the user right-clicks to delete a point."""
        removed = row.remove_point_near(q_c, y_c)
        if removed is not None:
            n = len(row._manual_pts)
            self.statusBar().showMessage(
                f"Deleted point q={removed[0]:.4f}  ({n} pts)")
        else:
            self.statusBar().showMessage("No point close enough to delete")
        # Live update after removing a point too.
        self._auto_reapply()
        return removed

    def _del_row(self, row: QRangeRow):
        # If this row is currently in manual-edit mode, stop editing first.
        if getattr(self, "_active_row", None) is row:
            self._active_row = None
        # If any plot window is editing this row's manual points, disable its
        # editor so it stops capturing clicks and clears its scatter.
        for w in list(self._plot_windows):
            if getattr(w, "_editor_row", None) is row:
                try:
                    w.set_editor(None, self)
                except Exception:
                    w._editor_row = None
        self._range_layout.removeWidget(row)
        row.deleteLater()
        # deleteLater() is asynchronous, so the row would still be counted by
        # _get_ranges if we recomputed right now. Drop it from the layout
        # immediately so the recompute below excludes it.
        row.setParent(None)
        # Recompute smoothing without the deleted range and refresh the plot so
        # its coloured region, smoothed curve, and manual dots disappear.
        self._auto_reapply()

    def _get_ranges(self, for_display=False) -> list:
        out = []
        for i in range(self._range_layout.count()):
            w = self._range_layout.itemAt(i).widget()
            if isinstance(w, QRangeRow):
                r = w.get_range(for_display=for_display)
                if r:
                    out.append(r)
        return out

    def _get_rows(self) -> list:
        """Return all QRangeRow widgets (including disabled ones)."""
        rows = []
        for i in range(self._range_layout.count()):
            w = self._range_layout.itemAt(i).widget()
            if isinstance(w, QRangeRow):
                rows.append(w)
        return rows

    # ── Plots ─────────────────────────────────────────────────

    def _ylabel(self) -> str:
        return self._data_type

    def _current_manual_snapshot(self):
        """Return the current manual points as [(color, [(q,y),...]), ...]."""
        row_colors = ['#E84855', '#FF6B35', '#9B59B6', '#00A8CC', '#27AE60']
        out = []
        for i, row in enumerate(self._get_rows()):
            if row.mode_combo.currentText() == "Manual Spline" and row._manual_pts:
                out.append((row_colors[i % len(row_colors)],
                            list(row._manual_pts)))
        return out

    def _build_snapshot(self) -> dict:
        """Capture the current data + settings into a dict for a PlotWindow."""
        # Manual anchor points, coloured per range row.
        row_colors = ['#E84855', '#FF6B35', '#9B59B6', '#00A8CC', '#27AE60']
        manual = []
        for i, row in enumerate(self._get_rows()):
            if row.mode_combo.currentText() == "Manual Spline" and row._manual_pts:
                manual.append((row_colors[i % len(row_colors)],
                               list(row._manual_pts)))

        # Log Y-axis works for any data type; the user unchecks it when a
        # linear scale is preferred.
        use_log = (hasattr(self, 'log_chk') and self.log_chk.isChecked())

        title = getattr(self, "_current_loaded_name", None) or "data"

        return {
            "title":         title,
            "q":             None if self.q is None else self.q.copy(),
            "y_raw":         None if self._y_raw is None else self._y_raw.copy(),
            "y":             None if self.y is None else self.y.copy(),
            "y_bkg_scaled":  None if self._y_bkg_scaled is None else self._y_bkg_scaled.copy(),
            "y_smoothed":    None if self.y_smoothed is None else self.y_smoothed.copy(),
            "alpha":         self._alpha_used,
            "data_type":     self._data_type,
            "q_ref":         None if self._q_ref is None else self._q_ref.copy(),
            "y_ref":         None if self._y_ref is None else self._y_ref.copy(),
            "ref_scale":     self._ref_scale,
            "use_log":       use_log,
            "ranges":        self._get_ranges(for_display=True),
            "manual":        manual,
        }

    def _new_plot_window(self):
        """Open a brand-new plot window for the current data and make it the
        active (live-updating) window.

        Called when a sample is selected, so earlier windows stay open as
        static snapshots for comparison. The window is placed to the RIGHT of
        the main GUI so it doesn't overlap the controls.
        """
        if getattr(self, "_startup_restoring", False):
            return
        if self.q is None or self.y is None:
            return
        self._plot_windows = [w for w in self._plot_windows if w.isVisible()]
        snap = self._build_snapshot()
        self._remember_confirmed_plot(snap, path=getattr(self, "_current_loaded_path", None))
        win = PlotWindow(self._copy_plot_snapshot(snap), parent=self)
        self._position_plot_window(win)
        self._plot_windows.append(win)
        self._active_plot_window = win
        win.show()
        win.raise_()

    def _position_plot_window(self, win):
        """Place `win` to the right of the main window, within the screen.

        Extra windows are offset diagonally a little so they don't stack
        exactly on top of each other.
        """
        try:
            screen = self.screen() or QApplication.primaryScreen()
            avail = screen.availableGeometry()
            main_geo = self.frameGeometry()
            gap = 8
            n = len([w for w in self._plot_windows if w.isVisible()])
            offset = 30 * n   # cascade extra windows slightly

            x = main_geo.right() + gap + offset
            y = main_geo.top() + offset

            w = win.width() or 900
            h = win.height() or 720

            # If it would run off the right edge, tuck it against the right
            # edge (still not overlapping the main window if possible).
            if x + w > avail.right():
                x = max(avail.left(), avail.right() - w)
            if y + h > avail.bottom():
                y = max(avail.top(), avail.bottom() - h)
            win.move(x, y)
        except Exception:
            pass

    def _refresh_active_window_if_open(self):
        """Refresh the active window ONLY if one is already open.

        Used for auxiliary updates (reference curve / scale) that should not
        create a plot window on their own — a reference alone shouldn't spawn a
        window when no sample has been plotted yet.
        """
        if getattr(self, "_startup_restoring", False):
            return
        win = self._active_plot_window
        if win is not None and win.isVisible():
            snap = self._build_snapshot()
            if not getattr(self, "_preserve_hover_view", False):
                self._remember_confirmed_plot(snap, path=getattr(self, "_current_loaded_path", None))
            win.update_snapshot(self._copy_plot_snapshot(snap))
            win.show()
            win.raise_()

    def _refresh_active_window(self, reset_view=False):
        """Update the active plot window in place with the current snapshot.

        Used by Apply Smoothing, q-range edits, and Clear so the open plot
        reflects the latest state instead of leaving stale curves behind or
        spawning a new window. If the active window was closed, a new one is
        opened. Pass reset_view=True when the data changed entirely (new file)
        so the axes refit instead of keeping an off-screen view.
        """
        if self.q is None or self.y is None:
            return
        # Startup session restoration may recompute data/settings, but it must not
        # create or raise a PlotWindow.  Hover/click after startup will do that.
        if getattr(self, "_startup_restoring", False):
            return
        self._plot_windows = [w for w in self._plot_windows if w.isVisible()]
        win = self._active_plot_window
        if win is None or not win.isVisible():
            self._new_plot_window()
            return
        snap = self._build_snapshot()
        if not getattr(self, "_preserve_hover_view", False):
            self._remember_confirmed_plot(snap, path=getattr(self, "_current_loaded_path", None))
        win.update_snapshot(self._copy_plot_snapshot(snap), reset_view=reset_view)
        win.show()
        win.raise_()

    # Backwards-compatible shim: a few callers still use the old name.
    def _open_plot_window(self, is_result=False):
        if is_result:
            self._refresh_active_window()
        else:
            self._new_plot_window()

    # The old embedded-plot updates are now no-ops: the main window has no live
    # plot. Callers throughout the code still invoke these (e.g. after typing a
    # q value), but a new window should only open on an explicit action
    # (selecting a sample, or Apply Smoothing), not on every keystroke.
    def _update_main_plot(self):
        return

    def _update_diff_plot(self):
        return



# ══════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = WHRangeSmoother()
    # Re-apply the fit/centre now that the application and screen are fully
    # initialised; at __init__ time the screen may not be known yet.
    win._fit_and_center(1200, 1000)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()