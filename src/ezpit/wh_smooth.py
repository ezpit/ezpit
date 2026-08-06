"""
wh_range_smoother.py
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

Requirements: PySide6, numpy, scipy, pyqtgraph
Usage       : python wh_range_smoother.py
"""

import sys
import os
import time
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
    QCheckBox, QComboBox
)
from PySide6.QtGui import QKeySequence, QFont
from PySide6.QtCore import Qt, QSettings, QTimer
from PySide6.QtCore import Qt, QSettings
import pyqtgraph as pg


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
    """Load a two-column data file, skipping header lines."""
    skip = 0
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    float(parts[0]); float(parts[1])
                    skip = i
                    break
                except ValueError:
                    continue
    data = np.loadtxt(path, skiprows=skip)
    if data.ndim == 1 or data.shape[1] < 2:
        raise ValueError("File must have at least two numeric columns.")
    return data[:, 0], data[:, 1]


# ══════════════════════════════════════════════════════════════
#  Q-range row widget
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
        self.manual_clear_btn = QPushButton("Clear")
        self.manual_clear_btn.setFixedWidth(48)
        self.manual_clear_btn.clicked.connect(self._clear_manual_pts)
        self._manual_pts = []   # list of (q, y) tuples

        for w in (self.lambda_label, self.lambda_edit,
                  self.order_label, self.order_spin,
                  self.anchor_label, self.anchor_spin,
                  self.blend_label, self.blend_spin,
                  self.manual_btn, self.manual_pts_label,
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

    def _clear_manual_pts(self):
        self._manual_pts.clear()
        self.manual_pts_label.setText("0 pts")
        self.manual_btn.setChecked(False)

    def add_manual_point(self, q, y):
        self._manual_pts.append((q, y))
        self._manual_pts.sort(key=lambda p: p[0])
        self.manual_pts_label.setText(f"{len(self._manual_pts)} pts")

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
        self.manual_clear_btn.setVisible(is_manual)
        # Blend visible for all modes

    def get_range(self):
        """Return a dict describing this range, or None if disabled/invalid."""
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
                if len(self._manual_pts) < 2:
                    return None
                d["manual_pts"] = list(self._manual_pts)
            return d
        except ValueError:
            return None


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
        self.resize(1200, 760)

        self._settings = QSettings("EZPDF", "WHRangeSmoother")
        self._last_dir = self._settings.value("last_dir", "")

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

    # ── UI ────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left control panel ────────────────────────────────
        left = QWidget()
        left.setFixedWidth(420)
        llay = QVBoxLayout(left)
        llay.setSpacing(10)
        splitter.addWidget(left)

        # 1. File
        file_grp = QGroupBox("1.  Load File")
        fg = QGridLayout(file_grp)
        fg.setSpacing(6)
        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)
        self.file_edit.setPlaceholderText("No file loaded")
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._pick_file)
        fg.addWidget(self.file_edit, 0, 0)
        fg.addWidget(btn_browse,     0, 1)
        llay.addWidget(file_grp)

        # 2. Data type
        type_grp = QGroupBox("2.  Data Type")
        tg = QHBoxLayout(type_grp)
        self._type_bg = QButtonGroup(self)
        for label in ("I(q)", "S(q)", "F(q)"):
            rb = QRadioButton(label)
            rb.setChecked(label == "I(q)")
            rb.toggled.connect(lambda checked, l=label: self._on_type(l, checked))
            self._type_bg.addButton(rb)
            tg.addWidget(rb)
        llay.addWidget(type_grp)

        # 2b. Background subtraction — visible only when I(q) is selected
        self.bkg_grp = QGroupBox("3.  Background  ( α × I_Bkg(q) )")
        bg_lay = QGridLayout(self.bkg_grp)
        bg_lay.setSpacing(6)

        self.bkg_edit = QLineEdit()
        self.bkg_edit.setReadOnly(True)
        self.bkg_edit.setPlaceholderText("No background file")
        btn_bkg = QPushButton("Browse…")
        btn_bkg.clicked.connect(self._pick_bkg)
        bg_lay.addWidget(QLabel("Bkg file:"), 0, 0)
        bg_lay.addWidget(self.bkg_edit,       1, 0)
        bg_lay.addWidget(btn_bkg,             1, 1)

        bg_lay.addWidget(QLabel("α ="), 2, 0)
        self.alpha_edit = QLineEdit("1.0")
        self.alpha_edit.setFixedWidth(80)
        self.alpha_edit.setToolTip("Background scale factor")
        # Auto-update on every keystroke (textChanged) and on Enter
        self.alpha_edit.textChanged.connect(self._subtract_bkg)
        self.alpha_edit.editingFinished.connect(self._subtract_bkg)
        bg_lay.addWidget(self.alpha_edit, 3, 0)

        # Log-scale toggle for I(q) main plot
        log_row = QHBoxLayout()
        self.log_chk = QCheckBox("Log Y-axis (main plot)")
        self.log_chk.setChecked(True)
        self.log_chk.stateChanged.connect(lambda _: self._apply_log_scale())
        log_row.addWidget(self.log_chk)
        log_row.addStretch()
        bg_lay.addLayout(log_row, 4, 0, 1, 2)

        self.bkg_grp.setVisible(True)    # visible by default (I(q) mode)
        llay.addWidget(self.bkg_grp)

        # 3. Default WH params (used when adding new rows)
        wh_grp = QGroupBox("4.  Default WH Parameters (for new ranges)")
        wg = QGridLayout(wh_grp)
        wg.setSpacing(6)
        wg.addWidget(QLabel("lambda:"), 0, 0)
        self.lambda_edit = QLineEdit("1000")
        self.lambda_edit.setFixedWidth(90)
        self.lambda_edit.setToolTip("Default lambda applied to newly added ranges")
        wg.addWidget(self.lambda_edit, 0, 1)
        wg.addWidget(QLabel("Order:"), 1, 0)
        self.order_spin = QSpinBox()
        self.order_spin.setRange(1, 2147483647)
        self.order_spin.setValue(2)
        self.order_spin.setFixedWidth(90)
        wg.addWidget(self.order_spin, 1, 1)
        llay.addWidget(wh_grp)

        # 4. Q ranges
        range_grp = QGroupBox("5.  q Ranges for Smoothing")
        rg = QVBoxLayout(range_grp)
        rg.setSpacing(4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setFixedHeight(380)
        self._range_container = QWidget()
        self._range_layout = QVBoxLayout(self._range_container)
        self._range_layout.setContentsMargins(0, 0, 0, 0)
        self._range_layout.setSpacing(0)
        self._range_layout.addStretch()
        scroll.setWidget(self._range_container)
        rg.addWidget(scroll)

        btn_add = QPushButton("＋  Add Range")
        btn_add.clicked.connect(self._add_range_row)
        rg.addWidget(btn_add)
        llay.addWidget(range_grp)

        # 5. Reference (optional)
        ref_grp = QGroupBox("6.  Reference Data (optional)")
        rgrid = QGridLayout(ref_grp)
        rgrid.setSpacing(6)

        self.ref_edit = QLineEdit()
        self.ref_edit.setReadOnly(True)
        self.ref_edit.setPlaceholderText("No reference loaded")

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

        # Action buttons
        btn_apply = QPushButton("▶  Apply Smoothing")
        btn_apply.setStyleSheet("font-weight:bold; padding:6px;")
        btn_apply.clicked.connect(self._apply_smoothing)
        llay.addWidget(btn_apply)

        btn_save = QPushButton("💾  Save Smoothed Data")
        btn_save.clicked.connect(self._save_result)
        llay.addWidget(btn_save)

        llay.addStretch()

        # ── Right plot panel ──────────────────────────────────
        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(right)
        splitter.setSizes([420, 780])

        pg.setConfigOptions(antialias=True)
        pg.setConfigOption('background', 'w')   # white background
        pg.setConfigOption('foreground', 'k')   # black axes/text

        # ── Main plot ──
        self.plot_main = pg.PlotWidget(title="Original vs Smoothed")
        self.plot_main.setLabel('bottom', "q", units="Å⁻¹")
        self.plot_main.setLabel('left',   "Intensity")
        # Legend created/re-created in _update_main_plot after each clear()
        self._legend = None
        self.plot_main.showGrid(x=True, y=True, alpha=0.3)

        # In-plot coordinate label (top-right, always in view)
        self._coord_main = pg.TextItem("", anchor=(1, 1), color='#0055AA')
        self._coord_main.setFont(pg.Qt.QtGui.QFont("Consolas", 10, pg.Qt.QtGui.QFont.Weight.Bold))
        self._coord_main.setZValue(100)
        # (added/re-added in _update_main_plot after each clear)

        # Crosshair lines
        self._vline_main = pg.InfiniteLine(angle=90, movable=False,
            pen=pg.mkPen('#AAAAAA', width=1,
                         style=Qt.PenStyle.DashLine))
        self._hline_main = pg.InfiniteLine(angle=0,  movable=False,
            pen=pg.mkPen('#AAAAAA', width=1,
                         style=Qt.PenStyle.DashLine))
        self.plot_main.addItem(self._vline_main)
        self.plot_main.addItem(self._hline_main)

        self.plot_main.scene().sigMouseMoved.connect(
            lambda pos: self._on_mouse_move(pos, 'main'))
        self.plot_main.scene().sigMouseClicked.connect(
            self._on_plot_clicked)
        rlay.addWidget(self.plot_main, 3)


        # ── Diff plot ──
        self.plot_diff = pg.PlotWidget(title="Difference  (Original − Smoothed)")
        self.plot_diff.setLabel('bottom', "q", units="Å⁻¹")
        self.plot_diff.setLabel('left',   "Δ")
        self.plot_diff.showGrid(x=True, y=True, alpha=0.3)

        self._coord_diff = pg.TextItem("", anchor=(1, 1), color='#0055AA')
        self._coord_diff.setFont(pg.Qt.QtGui.QFont("Consolas", 10, pg.Qt.QtGui.QFont.Weight.Bold))
        self._coord_diff.setZValue(100)
        # (added/re-added in _update_diff_plot after each clear)

        self._vline_diff = pg.InfiniteLine(angle=90, movable=False,
            pen=pg.mkPen('#AAAAAA', width=1,
                         style=Qt.PenStyle.DashLine))
        self._hline_diff = pg.InfiniteLine(angle=0,  movable=False,
            pen=pg.mkPen('#AAAAAA', width=1,
                         style=Qt.PenStyle.DashLine))
        self.plot_diff.addItem(self._vline_diff)
        self.plot_diff.addItem(self._hline_diff)

        self.plot_diff.scene().sigMouseMoved.connect(
            lambda pos: self._on_mouse_move(pos, 'diff'))
        rlay.addWidget(self.plot_diff, 1)

        # Install Z / R key shortcuts on both plot widgets
        for pw, name in ((self.plot_main, "main"),
                         (self.plot_diff, "diff")):
            pw.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            pw.keyPressEvent = lambda e, p=pw, n=name: self._plot_key(e, p, n)

        self.statusBar().showMessage("Ready — load a data file.    |    Z: Box-Zoom    R: Auto-Range")
        self._add_range_row()

    # ── Mouse cursor tracking ──────────────────────────────────

    def _on_mouse_move(self, pos, which: str):
        """Update crosshair and coordinate label for the given plot."""
        if which == 'main':
            plot  = self.plot_main
            vline = self._vline_main
            hline = self._hline_main
            coord = self._coord_main
        else:
            plot  = self.plot_diff
            vline = self._vline_diff
            hline = self._hline_diff
            coord = self._coord_diff

        vb = plot.getViewBox()
        if not plot.sceneBoundingRect().contains(pos):
            vline.setVisible(False)
            hline.setVisible(False)
            coord.setText("")
            return

        mouse_pt = vb.mapSceneToView(pos)
        x, y = mouse_pt.x(), mouse_pt.y()

        vline.setVisible(True)
        hline.setVisible(True)
        vline.setPos(x)
        hline.setPos(y)

        # In log mode the viewBox y-value is log10(y); convert back
        if which == 'main' and self._data_type == "I(q)" and self.log_chk.isChecked():
            # In log mode the view-coord y is log10(y).
            # Clamp to avoid OverflowError when cursor is far above plot.
            if -300 < y < 300:
                y_disp = 10.0 ** y
                txt = f" q={x:.4f} Å⁻¹   y={y_disp:.6g} "
            else:
                txt = f" q={x:.4f} Å⁻¹   y=(out of range) "
        else:
            txt = f" q={x:.4f} Å⁻¹   y={y:.6g} "

        # Place TextItem at top-right corner of the current view.
        # viewRange() returns the *displayed* axis range (already log10
        # for log mode), so using it directly keeps the label in view.
        vr = vb.viewRange()   # [[xmin, xmax], [ymin, ymax]] in view coords
        coord.setPos(vr[0][1], vr[1][0])
        coord.setText(txt)
        coord.setVisible(True)

    # ── Callbacks ─────────────────────────────────────────────

    def _plot_key(self, event, plot_widget, which: str):
        """
        Z  — toggle Box-Zoom mode (left mouse drag = zoom box)
        R  — auto-range (reset zoom)
        Other keys fall through to pyqtgraph default handler.
        """
        key = event.key()
        vb  = plot_widget.getViewBox()

        if key == Qt.Key.Key_Z:
            # Toggle between RectMode (box-zoom) and PanMode
            if vb.state["mouseMode"] == pg.ViewBox.RectMode:
                vb.setMouseMode(pg.ViewBox.PanMode)
                self.statusBar().showMessage("Mouse mode: Pan  (press Z to switch)")
            else:
                vb.setMouseMode(pg.ViewBox.RectMode)
                self.statusBar().showMessage("Mouse mode: Box-Zoom  (press Z to switch)")

        elif key == Qt.Key.Key_R:
            vb.autoRange()
            self.statusBar().showMessage("Auto-range applied")

        else:
            # Let pyqtgraph handle all other keys normally
            pg.PlotWidget.keyPressEvent(plot_widget, event)

    def _apply_log_scale(self):
        """Toggle log Y-axis on main plot for I(q) mode."""
        is_iq  = (self._data_type == "I(q)")
        use_log = is_iq and self.log_chk.isChecked()
        self.plot_main.setLogMode(x=False, y=use_log)

    def _on_type(self, label: str, checked: bool):
        if checked:
            self._data_type = label
            is_iq = (label == "I(q)")
            self.bkg_grp.setVisible(is_iq)
            self._apply_log_scale()
            self._update_main_plot()

    def _pick_bkg(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Background File", self._last_dir,
            "Data Files (*.chi *.iq *.txt *.dat);;All Files (*)")
        if not path:
            return
        self.bkg_edit.setText(path)
        self._last_dir = os.path.dirname(path)
        self._settings.setValue("last_dir", self._last_dir)
        self._settings.sync()

    def _subtract_bkg(self):
        """Compute I(q) - alpha * Bkg(q); called automatically when alpha changes."""
        if self.q is None or self._y_raw is None:
            return   # no sample loaded yet — silent
        bkg_path = self.bkg_edit.text().strip()
        if not bkg_path or not os.path.exists(bkg_path):
            return   # no background yet — silent
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

        self._apply_log_scale()
        self._update_main_plot()
        self.statusBar().showMessage(
            f"Background subtracted  |  alpha = {alpha}  |  "
            f"Bkg: {os.path.basename(bkg_path)}")

    def _pick_reference(self):
        """Load an optional reference curve for visual comparison."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Reference File", self._last_dir,
            "Data Files (*.chi *.iq *.sq *.fq *.txt *.dat);;All Files (*)")
        if not path:
            return
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
        self._update_main_plot()
        self.statusBar().showMessage(
            f"Reference loaded: {os.path.basename(path)}  "
            f"({len(q_r)} pts, q = {q_r[0]:.3f}–{q_r[-1]:.3f})")

    def _clear_reference(self):
        self._q_ref = None
        self._y_ref = None
        self.ref_edit.clear()
        self._update_main_plot()
        self.statusBar().showMessage("Reference cleared.")

    def _on_ref_scale_changed(self, *_):
        try:
            self._ref_scale = float(self.ref_scale_edit.text())
        except ValueError:
            return
        if self._y_ref is not None:
            self._update_main_plot()

    def _pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Data File", self._last_dir,
            "Data Files (*.chi *.iq *.sq *.fq *.txt *.dat);;"
            "All Files (*)")
        if not path:
            return
        try:
            q, y = load_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return

        self.q = q
        self.y = y
        self._y_raw = y.copy()   # keep original for re-subtracting bkg
        self.y_smoothed = None
        self.file_edit.setText(path)
        self._last_dir = os.path.dirname(path)
        self._settings.setValue("last_dir", self._last_dir)
        self._settings.sync()

        # Auto-detect type from extension
        ext = os.path.splitext(path)[1].lower()
        type_map = {'.sq': 'S(q)', '.fq': 'F(q)', '.iq': 'I(q)', '.chi': 'I(q)'}
        detected = type_map.get(ext, 'F(q)')
        for btn in self._type_bg.buttons():
            btn.setChecked(btn.text() == detected)

        self._update_main_plot()
        self.statusBar().showMessage(
            f"Loaded: {os.path.basename(path)}  |  "
            f"{len(q)} points,  q = {q[0]:.3f} ~ {q[-1]:.3f} Å⁻¹")

    def _apply_smoothing(self):
        if self.y is None:
            QMessageBox.warning(self, "No Data", "Load a file first.")
            return
        # I(q) mode: must subtract background first
        if self._data_type == "I(q)" and self._y_bkg_scaled is None:
            QMessageBox.warning(self, "Background Required",
                "I(q) mode: subtract background first (Subtract Bkg button).")
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

        self._update_main_plot()
        self._update_diff_plot()

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

    def _save_result(self):
        if self.y_smoothed is None:
            QMessageBox.warning(self, "No Result",
                                "Apply smoothing first.")
            return

        src = self.file_edit.text()
        base = os.path.splitext(os.path.basename(src))[0]
        default = os.path.join(self._last_dir,
                               f"{base}_wh_smoothed.chi")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Smoothed Data", default,
            "Chi Files (*.chi);;Text Files (*.txt);;All Files (*)")
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
                n = len(r.get('manual_pts', []))
                return f"q[{r['q0']}-{r['q1']}] Manual({n} pts) blend={r['blend']}"
            else:
                return f"q[{r['q0']}-{r['q1']}] {m} blend={r['blend']}"
        range_str = "  |  ".join(_r_str(r) for r in ranges)
        header_lines = [
            f"WH Range-Smoothed  [{self._data_type}]",
            f"Source  : {src}",
            f"Ranges  : {range_str}",
            f"q(A-1)   y_smoothed",
        ]
        np.savetxt(path,
                   np.column_stack([self.q, self.y_smoothed]),
                   header="\n".join(header_lines),
                   fmt="%.8f")
        self._last_dir = os.path.dirname(path)
        self._settings.setValue("last_dir", self._last_dir)
        self._settings.sync()
        self.statusBar().showMessage(f"Saved → {path}")

    # ── Range rows ────────────────────────────────────────────

    def _add_range_row(self):
        try:
            def_lam = float(self.lambda_edit.text())
        except ValueError:
            def_lam = 1000
        def_ord = self.order_spin.value()
        row = QRangeRow(default_lambda=def_lam, default_order=def_ord)
        row.btn_del.clicked.connect(lambda: self._del_row(row))
        row.chk.stateChanged.connect(lambda _: self._update_main_plot())
        row.q0.editingFinished.connect(self._update_main_plot)
        row.q1.editingFinished.connect(self._update_main_plot)
        row.manual_btn.toggled.connect(
            lambda checked, r=row: self._on_manual_btn_toggled(r, checked))
        # Redraw after Clear so scatter dots disappear immediately
        row.manual_clear_btn.clicked.connect(self._update_main_plot)
        idx = self._range_layout.count() - 1
        self._range_layout.insertWidget(idx, row)

    def _on_manual_btn_toggled(self, row, checked):
        if checked:
            if self._active_row and self._active_row is not row:
                self._active_row.manual_btn.setChecked(False)
            self._active_row = row
            self.statusBar().showMessage(
                "Manual Spline: LEFT-CLICK on graph to add points  |  RIGHT-CLICK to finish")
        else:
            if self._active_row is row:
                self._active_row = None
            self.statusBar().showMessage(
                f"Manual editing stopped  ({len(row._manual_pts)} pts stored)")

    def _on_plot_clicked(self, event):
        """Capture left-click on plot_main to add manual spline points."""
        if self._active_row is None:
            return
        # Debounce: sigMouseClicked fires multiple times per click
        # (once per overlapping GraphicsItem). Ignore within 300 ms.
        now = time.time()
        if now - self._last_click_time < 0.3:
            return
        self._last_click_time = now
        from pyqtgraph.Qt import QtCore as _QC
        btn = event.button()
        if btn == _QC.Qt.MouseButton.RightButton:
            self._active_row.manual_btn.setChecked(False)
            self._active_row = None
            self._update_main_plot()
            self.statusBar().showMessage("Manual point editing finished.")
            return
        if btn != _QC.Qt.MouseButton.LeftButton:
            return
        vb  = self.plot_main.getViewBox()
        pt  = vb.mapSceneToView(event.scenePos())
        q_c, y_c = pt.x(), pt.y()
        if self._data_type == "I(q)" and self.log_chk.isChecked():
            if -300 < y_c < 300:
                y_c = 10.0 ** y_c
            else:
                return
        self._active_row.add_manual_point(q_c, y_c)
        n = len(self._active_row._manual_pts)
        self._update_main_plot()
        self.statusBar().showMessage(
            f"Added: q={q_c:.4f}, y={y_c:.4g}  ({n} pts)  |  Right-click to finish")
        event.accept()

    def _del_row(self, row: QRangeRow):
        self._range_layout.removeWidget(row)
        row.deleteLater()
        self._update_main_plot()

    def _get_ranges(self) -> list:
        out = []
        for i in range(self._range_layout.count()):
            w = self._range_layout.itemAt(i).widget()
            if isinstance(w, QRangeRow):
                r = w.get_range()
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

    def _update_main_plot(self):
        self.plot_main.clear()
        self.plot_main.setLabel('left', self._ylabel())
        # Re-create legend after clear() (clear() removes it)
        self._legend = self.plot_main.addLegend(offset=(10, 10))
        self._legend.setLabelTextSize('13pt')
        self._legend.setLabelTextColor('k')
        self._legend.setBrush(pg.mkBrush(0, 0, 0, 0))   # transparent background
        self._legend.setPen(pg.mkPen(None))              # no frame
        # Re-add crosshairs + coord after clear()
        self.plot_main.addItem(self._vline_main, ignoreBounds=True)
        self.plot_main.addItem(self._hline_main, ignoreBounds=True)
        self.plot_main.addItem(self._coord_main, ignoreBounds=True)

        if self.y is None:
            return

        if self._data_type == "I(q)" and self._y_bkg_scaled is not None:
            # ── I(q) mode: show three curves ──────────────────
            # 1. Raw I(q)
            self.plot_main.plot(
                self.q, self._y_raw,
                pen=pg.mkPen('#000000', width=1.8),
                name="I(q)  [sample]")
            # 2. alpha * Bkg(q)
            self.plot_main.plot(
                self.q, self._y_bkg_scaled,
                pen=pg.mkPen('#1565C0', width=1.8),
                name=f"α×I_Bkg(q)  (α={self._alpha_used})")
            # 3. Subtracted I(q) = I - alpha*Bkg
            self.plot_main.plot(
                self.q, self.y,
                pen=pg.mkPen('#E84855', width=1.8),
                name="I(q) − α×I_Bkg(q)")
        elif self._data_type == "I(q)" and self._y_bkg_scaled is None:
            # I(q) loaded but background not yet subtracted
            self.plot_main.plot(
                self.q, self._y_raw,
                pen=pg.mkPen('#000000', width=1.8),
                name="I(q) [sample — no bkg yet]")
        else:
            # ── S(q) / F(q) mode: single original curve ───────
            self.plot_main.plot(
                self.q, self.y,
                pen=pg.mkPen('#000000', width=1.8),
                name=f"Original {self._data_type}")

        # Smoothed result (if available)
        if self.y_smoothed is not None:
            self.plot_main.plot(
                self.q, self.y_smoothed,
                pen=pg.mkPen('#7B2D8B', width=1.8),
                name="Smoothed")

        # Reference curve (if loaded) — dashed dark-green line for comparison
        if self._q_ref is not None and self._y_ref is not None:
            y_ref_scaled = self._y_ref * self._ref_scale
            self.plot_main.plot(
                self._q_ref, y_ref_scaled,
                pen=pg.mkPen('#2E7D32', width=1.8, style=Qt.PenStyle.DashLine),
                name=f"Reference × {self._ref_scale:g}")

        # Highlight active q ranges (color varies by mode)
        for r in self._get_ranges():
            mode = r.get("mode", "wh")
            if mode == "wh":
                brush = pg.mkBrush(255, 180,   0, 50)   # yellow
            elif mode == "linear":
                brush = pg.mkBrush( 50, 180, 255, 50)   # cyan
            elif mode == "manual":
                brush = pg.mkBrush(255, 100,  50, 50)   # orange-red
            else:  # spline
                brush = pg.mkBrush(180,  80, 255, 50)   # purple
            lr = pg.LinearRegionItem(
                [r["q0"], r["q1"]], movable=False, brush=brush)
            lr.setZValue(-10)
            self.plot_main.addItem(lr)

        # Draw manual anchor points as scatter dots on the graph
        if not hasattr(self, '_manual_scatter') or self._manual_scatter is None:
            self._manual_scatter = []
        for sc in self._manual_scatter:
            try:
                self.plot_main.removeItem(sc)
            except Exception:
                pass
        self._manual_scatter = []
        row_colors = ['#E84855', '#FF6B35', '#9B59B6', '#00A8CC', '#27AE60']
        use_log = (self._data_type == "I(q)" and
                   hasattr(self, 'log_chk') and self.log_chk.isChecked())
        for i, row in enumerate(self._get_rows()):
            if row.mode_combo.currentText() != "Manual Spline":
                continue
            pts = row._manual_pts
            if not pts:
                continue
            col = row_colors[i % len(row_colors)]
            x_pts = [p[0] for p in pts]
            y_pts = [p[1] for p in pts]
            # ScatterPlotItem does NOT auto-apply ViewBox log transform.
            # In log mode the ViewBox expects log10(y) values directly.
            if use_log:
                y_pts = [np.log10(y) if y > 0 else 0 for y in y_pts]
            sc = pg.ScatterPlotItem(
                x=x_pts, y=y_pts,
                symbol='o', size=12,
                pen=pg.mkPen(col, width=2),
                brush=pg.mkBrush(col + 'AA'))
            sc.setZValue(100)
            self.plot_main.addItem(sc, ignoreBounds=True)
            self._manual_scatter.append(sc)

    def _update_diff_plot(self):
        self.plot_diff.clear()
        # Re-add crosshairs + coord after clear()
        self.plot_diff.addItem(self._vline_diff, ignoreBounds=True)
        self.plot_diff.addItem(self._hline_diff, ignoreBounds=True)
        self.plot_diff.addItem(self._coord_diff, ignoreBounds=True)
        if self.y is None or self.y_smoothed is None:
            return
        diff = self.y - self.y_smoothed
        self.plot_diff.setLabel('left', f"Δ {self._data_type}")
        self.plot_diff.plot(
            self.q, diff,
            pen=pg.mkPen('#1A8A4A', width=1.8))
        self.plot_diff.addLine(
            y=0,
            pen=pg.mkPen('#888888', style=Qt.PenStyle.DashLine))


# ══════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = WHRangeSmoother()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()