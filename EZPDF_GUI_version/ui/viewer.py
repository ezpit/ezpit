# ui/viewer.py
import os

# [Important] Force pyqtgraph to use PySide6
os.environ["QT_API"] = "pyside6"

import numpy as np
import pyqtgraph as pg

# PySide6 Imports
from PySide6.QtCore import Qt, QPoint, QSize, QSizeF, QMarginsF, QSettings
from PySide6.QtGui import (
    QAction, QFont, QImage, QPainter, QColor, QPageSize, QCursor, QPdfWriter
)
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QCheckBox, QMenuBar, QWidgetAction,
    QSizePolicy, QPushButton, QSlider, QLineEdit,
    QDialog, QFileDialog, QMessageBox, QApplication
)

# SVG Generator (Check availability)
try:
    from PySide6.QtSvg import QSvgGenerator
except ImportError:
    QSvgGenerator = None

from .plotter import make_pen, setup_plot, plot_curve, update_visibility
from model.math_functions import lorch_window
from .save_menu import SaveMenu

LABEL_PT = 9
HUD_PT = 9

# [Viewer Style - Light]
TEAL_STYLE = """
QWidget {
    font-size: 9pt;
    color: #000;
    background-color: #F0F0F0;
    font-family: 'Segoe UI', 'Malgun Gothic', sans-serif;
}
QLabel {
    font-weight: normal; 
    color: #222;
}
QLineEdit {
    background-color: #FFFFFF;
    color: #000;
    border: 1px solid #A0A0A0;
    border-radius: 2px;
    padding: 2px;
}
QLineEdit:focus {
    border: 1px solid #009688;
}

/* SLIDER STYLE */
QSlider::groove:horizontal {
    border: 1px solid #D0D0D0;
    background: transparent;
    height: 4px;
    margin: 0px;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #009688;
    height: 4px;
    border-radius: 2px;
}
QSlider::add-page:horizontal {
    background: #E0E0E0;
    height: 4px;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #FFFFFF;
    border: 3px solid #009688;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: #F0F0F0;
    border-color: #00796B;
}

/* Checkbox Style */
QCheckBox {
    font-weight: normal; 
    color: #000;
}
QCheckBox::indicator:checked {
    background-color: #009688;
    border: 1px solid #009688;
}
"""

# [Viewer Style - Dark]
DARK_STYLE = """
QWidget {
    font-size: 9pt;
    color: #E0E0E0;
    background-color: #2b2b2b;
    font-family: 'Segoe UI', 'Malgun Gothic', sans-serif;
}
QLabel {
    font-weight: normal; 
    color: #E0E0E0;
}
QLineEdit {
    background-color: #3b3b3b;
    color: #E0E0E0;
    border: 1px solid #555555;
    border-radius: 2px;
    padding: 2px;
}
QLineEdit:focus {
    border: 1px solid #009688;
}

/* SLIDER STYLE */
QSlider::groove:horizontal {
    border: 1px solid #555555;
    background: transparent;
    height: 4px;
    margin: 0px;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #009688;
    height: 4px;
    border-radius: 2px;
}
QSlider::add-page:horizontal {
    background: #444444;
    height: 4px;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #b0b0b0;
    border: 3px solid #009688;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: #ffffff;
    border-color: #00796B;
}

/* Checkbox Style */
QCheckBox {
    font-weight: normal; 
    color: #E0E0E0;
}
QCheckBox::indicator {
    border: 1px solid #777;
    background: #444;
}
QCheckBox::indicator:checked {
    background-color: #009688;
    border: 1px solid #009688;
}
QMenuBar {
    background-color: #1e1e1e;
    color: #E0E0E0;
}
QMenuBar::item:selected {
    background-color: #3a3a3a;
}
QMenu {
    background-color: #2b2b2b;
    color: #E0E0E0;
    border: 1px solid #555;
}
QMenu::item:selected {
    background-color: #3a3a3a;
}
"""


class PlotWindow(QMainWindow):
    """
    Visualize I(q), S(q), F(q), G(r) with PyQtGraph (PySide6 backend).
    Supports single file analysis and multiple file 'waterfall' plots.
    """

    def __init__(self):
        super().__init__()
        self.left_button_pan_enabled = True
        pg.setConfigOption('leftButtonPan', True)
        # Smoother, sharper curves on screen (and higher-quality raster export).
        pg.setConfigOptions(antialias=True)

        # Default to Light Mode
        self.setStyleSheet(TEAL_STYLE)

        self.setWindowTitle("EZPDF Plot")
        self.resize(900, 700)

        # Stores the QTreeWidgetItem(s) associated with this window
        self.associated_items = None

        # True when the window currently shows a computed Compton curve
        # (set by calculate_compton). Reset whenever a file is plotted.
        self.is_compton = False

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # ---------- TOP ROW ----------
        FONT_PX = 11
        INDICATOR_PX = 10
        self.top_row = QHBoxLayout()
        self.top_row.setContentsMargins(6, 6, 6, 2)
        self.top_row.setSpacing(8)

        self.top_hint = QLabel(
            "Z: Toggle left-click Pan <-> Box-Zoom (left mouse),  R: Auto-range (reset zoom)"
        )
        self.top_hint.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.top_hint.setWordWrap(False)
        self.top_hint.setStyleSheet(f"color:#444; font-size:{FONT_PX}px; font-weight:normal; margin-right:8px;")
        self.top_row.addWidget(self.top_hint, 1)

        self.CB_STYLE_LIGHT = f"""
        QCheckBox {{
            font-size: {FONT_PX}px;
            color: #444;
            font-weight: bold;
        }}
        QCheckBox::indicator {{
            width: {INDICATOR_PX}px;
            height: {INDICATOR_PX}px;
            border-radius: {INDICATOR_PX // 2}px;
            border: 1px solid #777;
            background: #fff;
        }}
        QCheckBox::indicator:checked {{ background: #009688; border: 1px solid #009688; }}
        """

        self.CB_STYLE_DARK = f"""
        QCheckBox {{
            font-size: {FONT_PX}px;
            color: #E0E0E0;
            font-weight: bold;
        }}
        QCheckBox::indicator {{
            width: {INDICATOR_PX}px;
            height: {INDICATOR_PX}px;
            border-radius: {INDICATOR_PX // 2}px;
            border: 1px solid #777;
            background: #444;
        }}
        QCheckBox::indicator:checked {{ background: #009688; border: 1px solid #009688; }}
        """

        self.plot_checkboxes = []
        for name in ['I(q)', 'S(q)', 'F(q)', 'G(r)']:
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.setStyleSheet(self.CB_STYLE_LIGHT + "QCheckBox { margin-left:6px; }")
            cb.setFixedHeight(INDICATOR_PX + 6)
            cb.stateChanged.connect(self.update_visibility)
            self.plot_checkboxes.append(cb)

        self.layout.addLayout(self.top_row)

        self.label = QLabel("No plot loaded.")
        self.label.setStyleSheet(f"color:#444; font-size:{FONT_PX}px; font-weight:normal; margin-left:6px;")
        self.label.setWordWrap(False)
        self.layout.addWidget(self.label)

        self.slider_layout = None
        self.x_offset_slider = None
        self.y_offset_slider = None
        self.x_offset_input = None
        self.y_offset_input = None
        self.show_offset_sliders()

        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_layout.setSpacing(0)

        self.plots = []
        # Per-panel flag: True once the user manually zooms/pans that panel.
        # Parameter changes then keep that panel's view; unzoomed panels
        # auto-scale. Reset on fresh load and on R (reset zoom).
        self._user_zoomed = [False, False, False, False]

        def _fixed_tick_strings(values, scale, spacing):
            # 2 decimals, right-aligned to a constant width so every panel's
            # y labels occupy the same space and the plot areas line up.
            out = []
            for v in values:
                out.append(("%.2f" % (v * scale)).rjust(8))
            return out

        def _make_iq_ticks(axis):
            # I(q) can span a huge range. In linear mode use scientific notation
            # for large/small values (bounded width, never gets cut off) and
            # plain numbers otherwise. In log mode defer to pyqtgraph's default
            # (nice 10^n formatting).
            def _ticks(values, scale, spacing):
                if getattr(axis, 'logMode', False):
                    return pg.AxisItem.tickStrings(axis, values, scale, spacing)
                out = []
                for v in values:
                    val = v * scale
                    av = abs(val)
                    if av != 0 and (av >= 1e4 or av < 1e-2):
                        s = "%.1e" % val
                    else:
                        s = "%.2f" % val
                    out.append(s.rjust(8))
                return out
            return _ticks

        for _ in range(4):
            pw = pg.PlotWidget()
            pw.setBackground('w')
            pw.showGrid(x=False, y=False, alpha=0.3)

            vb = pw.getViewBox()
            vb.setLimits(xMin=0)
            vb.setDefaultPadding(0.0)

            idx = len(self.plots)

            # Left (y) axis: constant width + fixed-width labels so all panels'
            # plot areas start at the same x (same q position lines up). No ×10ⁿ
            # SI prefix; I(q) uses scientific notation for large linear values.
            try:
                left_axis = pw.getAxis('left')
                left_axis.setWidth(72)
                left_axis.enableAutoSIPrefix(False)
                if idx == 0:
                    left_axis.tickStrings = _make_iq_ticks(left_axis)
                else:
                    left_axis.tickStrings = _fixed_tick_strings
            except Exception:
                pass

            try:
                vb.sigRangeChangedManually.connect(
                    lambda _mask=None, i=idx: self._mark_user_zoomed(i))
            except Exception:
                pass

            self.plot_layout.addWidget(pw, 1)
            self.plots.append(pw)

        self.layout.addWidget(self.plot_container, 1)

        self._install_xy_readouts()
        self.init_bottom_menu()

        # Data placeholders
        self.file_name = None
        self.bkg_curve = None
        self.raw_curve = None
        self.sq_curve_original = None
        self.sq_curve_polynomial = None
        self.mean_sq_fi_curve = None
        self.sq_mean_fi_curve = None
        self.fq_smoothed_curve = None
        self.gr_smoothed_curve = None
        self.gr_lorch_curve = None
        self.gr_smoothed_lorch_curve = None
        self.fq_lorch_curve = None
        self.fq_smoothed_lorch_curve = None
        self.fq_main_curve = None
        self.gr_main_curve = None

        self.xs = None
        self.ys = None
        self.raw_iq = None
        self.background_data = None
        self.sq_original_data = None
        self.sq_polynomial_data = None
        self.fq_smoothed_data = None
        self.gr_smoothed_data = None
        self.gr_lorch_data = None
        self.gr_smoothed_lorch_data = None
        self.mean_sq_fi = None
        self.sq_mean_fi = None

        # Multi-plot cache
        self.cached_xs = None
        self.cached_ys = None
        self.cached_titles = None

        # Check Global Theme
        app_instance = QApplication.instance()
        if hasattr(app_instance, 'is_dark_mode') and app_instance.is_dark_mode:
            self.apply_theme(True)

    def apply_theme(self, is_dark):
        """
        Updates the style of the PlotWindow for Light/Dark mode.
        """
        FONT_PX = 11

        if is_dark:
            self.setStyleSheet(DARK_STYLE)
            self.top_hint.setStyleSheet(f"color:#BBB; font-size:{FONT_PX}px; font-weight:normal; margin-right:8px;")
            self.label.setStyleSheet(f"color:#BBB; font-size:{FONT_PX}px; font-weight:normal; margin-left:6px;")
            self.bottom_menu.setStyleSheet("margin-top: 6px; background-color:#1e1e1e; color:#E0E0E0;")

            for cb in self.plot_checkboxes:
                cb.setStyleSheet(self.CB_STYLE_DARK + "QCheckBox { margin-left:6px; }")

            for pw in self.plots:
                pw.setBackground('#1e1e1e')  # Dark background
                plot_item = pw.getPlotItem()
                for axis in ['left', 'bottom']:
                    ax = plot_item.getAxis(axis)
                    ax.setPen('#E0E0E0')
                    ax.setTextPen('#E0E0E0')

            if self.slider_bar_widget:
                self.slider_bar_widget.setStyleSheet("margin:0; padding:0; background-color: #2b2b2b;")

            # [추가] 다크 모드 시 곡선 및 토글 색상 개선 (흰색으로 통일)
            pen_white = make_pen((255, 255, 255))
            if self.raw_curve: self.raw_curve.setPen(pen_white)
            if self.sq_curve_original: self.sq_curve_original.setPen(pen_white)
            if self.fq_smoothed_curve: self.fq_smoothed_curve.setPen(pen_white)
            if self.gr_smoothed_curve: self.gr_smoothed_curve.setPen(pen_white)

            base_style = "QCheckBox::indicator { width: 12px; height: 12px; border-radius: 6px; border: 1px solid #777; background: #fff; } "
            white_checked = "QCheckBox::indicator:checked { background: #FFFFFF; border:1px solid #FFFFFF; }"

            if hasattr(self, 'original_checkbox'):
                self.original_checkbox.setStyleSheet(base_style + white_checked)
            if hasattr(self, 'original_sq_checkbox'):
                self.original_sq_checkbox.setStyleSheet(base_style + white_checked)
            if hasattr(self, 'smoothed_fq_checkbox'):
                self.smoothed_fq_checkbox.setStyleSheet(base_style + white_checked)

        else:
            self.setStyleSheet(TEAL_STYLE)
            self.top_hint.setStyleSheet(f"color:#444; font-size:{FONT_PX}px; font-weight:normal; margin-right:8px;")
            self.label.setStyleSheet(f"color:#444; font-size:{FONT_PX}px; font-weight:normal; margin-left:6px;")
            self.bottom_menu.setStyleSheet("margin-top: 6px; background-color:#F0F0F0; color:#000;")

            for cb in self.plot_checkboxes:
                cb.setStyleSheet(self.CB_STYLE_LIGHT + "QCheckBox { margin-left:6px; }")

            for pw in self.plots:
                pw.setBackground('w')
                plot_item = pw.getPlotItem()
                for axis in ['left', 'bottom']:
                    ax = plot_item.getAxis(axis)
                    ax.setPen('k')
                    ax.setTextPen('k')

            if self.slider_bar_widget:
                self.slider_bar_widget.setStyleSheet("margin:0; padding:0; background-color: #F0F0F0;")

            # [추가] 라이트 모드 시 곡선 및 토글 색상 복구 (검은색으로 통일)
            pen_black = make_pen((0, 0, 0))
            if self.raw_curve: self.raw_curve.setPen(pen_black)
            if self.sq_curve_original: self.sq_curve_original.setPen(pen_black)
            if self.fq_smoothed_curve: self.fq_smoothed_curve.setPen(pen_black)
            if self.gr_smoothed_curve: self.gr_smoothed_curve.setPen(pen_black)

            base_style = "QCheckBox::indicator { width: 12px; height: 12px; border-radius: 6px; border: 1px solid #777; background: #fff; } "
            black_checked = "QCheckBox::indicator:checked { background: #000000; border:1px solid #000000; }"

            if hasattr(self, 'original_checkbox'):
                self.original_checkbox.setStyleSheet(base_style + black_checked)
            if hasattr(self, 'original_sq_checkbox'):
                self.original_sq_checkbox.setStyleSheet(base_style + black_checked)
            if hasattr(self, 'smoothed_fq_checkbox'):
                self.smoothed_fq_checkbox.setStyleSheet(base_style + black_checked)

    def init_bottom_menu(self):
        self.bottom_menu = QMenuBar()
        self.bottom_menu.setStyleSheet("margin-top: 6px; background-color:#F0F0F0;")

        save_data_action = QAction("Save data", self)
        save_data_action.triggered.connect(self._save_data_dialog)
        self.bottom_menu.addAction(save_data_action)

        save_figure_action = QAction("Save figure", self)
        save_figure_action.triggered.connect(self.save_combined_graph_image)
        self.bottom_menu.addAction(save_figure_action)

        self.bottom_menu.addMenu("Option:")

        # I(q) Menu
        iq_menu = self.bottom_menu.addMenu("I(q)")
        iq_widget = QWidget()
        iq_layout = QVBoxLayout()
        iq_layout.setContentsMargins(10, 10, 10, 10)
        iq_widget.setLayout(iq_layout)

        self.log_checkbox = QCheckBox("Log Y Scale")
        self.log_checkbox.setChecked(False)
        self.log_checkbox.stateChanged.connect(self.toggle_log_scale)

        self.background_checkbox = QCheckBox("Background")
        self.background_checkbox.setChecked(False)
        self.background_checkbox.stateChanged.connect(self.show_background_data)

        self.original_checkbox = QCheckBox("Original I(q) Curve")
        self.original_checkbox.setChecked(False)
        self.original_checkbox.stateChanged.connect(self.show_original_data)

        for checkbox in [self.log_checkbox, self.background_checkbox, self.original_checkbox]:
            iq_layout.addWidget(checkbox)

        iq_widget_action = QWidgetAction(self)
        iq_widget_action.setDefaultWidget(iq_widget)
        iq_menu.addAction(iq_widget_action)

        # S(q) Menu
        sq_menu = self.bottom_menu.addMenu("S(q)")
        sq_widget = QWidget()
        sq_layout = QVBoxLayout()
        sq_layout.setContentsMargins(10, 10, 10, 10)
        sq_widget.setLayout(sq_layout)

        self.original_sq_checkbox = QCheckBox("Original S(q)")
        self.original_sq_checkbox.setChecked(False)
        self.original_sq_checkbox.stateChanged.connect(self.toggle_original_sq)
        sq_layout.addWidget(self.original_sq_checkbox)

        self.polynomial_sq_checkbox = QCheckBox("Polynomial for S(q)")
        self.polynomial_sq_checkbox.setChecked(False)
        self.polynomial_sq_checkbox.stateChanged.connect(self.toggle_polynomial_sq)
        sq_layout.addWidget(self.polynomial_sq_checkbox)

        self.mean_sq_fi_checkbox = QCheckBox("<f^2>")
        self.mean_sq_fi_checkbox.setChecked(False)
        self.mean_sq_fi_checkbox.stateChanged.connect(self.toggle_mean_sq_fi)
        sq_layout.addWidget(self.mean_sq_fi_checkbox)

        self.sq_mean_fi_checkbox = QCheckBox("<f>^2")
        self.sq_mean_fi_checkbox.setChecked(False)
        self.sq_mean_fi_checkbox.stateChanged.connect(self.toggle_sq_mean_fi)
        sq_layout.addWidget(self.sq_mean_fi_checkbox)

        sq_widget_action = QWidgetAction(self)
        sq_widget_action.setDefaultWidget(sq_widget)
        sq_menu.addAction(sq_widget_action)

        # F(q) Menu
        fq_menu = self.bottom_menu.addMenu("F(q)")
        fq_widget = QWidget()
        fq_layout = QVBoxLayout()
        fq_layout.setContentsMargins(10, 10, 10, 10)
        fq_widget.setLayout(fq_layout)

        self.smoothed_fq_checkbox = QCheckBox("Smoothed F(q)")
        self.smoothed_fq_checkbox.setChecked(False)
        self.smoothed_fq_checkbox.stateChanged.connect(self.toggle_smoothed_fq)
        fq_layout.addWidget(self.smoothed_fq_checkbox)

        fq_widget_action = QWidgetAction(self)
        fq_widget_action.setDefaultWidget(fq_widget)
        fq_menu.addAction(fq_widget_action)

        # G(r) Menu -- Lorch-modified variants (experimental data only).
        # These overlay curves are precomputed; toggling only shows/hides.
        gr_menu = self.bottom_menu.addMenu("G(r)")
        gr_widget = QWidget()
        gr_layout = QVBoxLayout()
        gr_layout.setContentsMargins(10, 10, 10, 10)
        gr_widget.setLayout(gr_layout)

        # Lorch controls apply to BOTH F(q) and G(r) panels. The "Smoothed"
        # axis reuses the existing "Smoothed F(q)" checkbox in the F(q) menu.
        _lorch_cb_style = (
            "QCheckBox::indicator { width: 12px; height: 12px; border-radius: 6px; "
            "border: 1px solid #777; background: #fff; } "
            "QCheckBox::indicator:checked { background: #FF8C00; border: 1px solid #FF8C00; }")
        self.apply_lorch_checkbox = QCheckBox("Apply Lorch")
        self.apply_lorch_checkbox.setStyleSheet(_lorch_cb_style)
        self.apply_lorch_checkbox.setChecked(False)
        self.apply_lorch_checkbox.stateChanged.connect(lambda: self._on_lorch_toggle(self.apply_lorch_checkbox))
        gr_layout.addWidget(self.apply_lorch_checkbox)

        self.compare_lorch_checkbox = QCheckBox("Compare (Lorch on/off)")
        self.compare_lorch_checkbox.setStyleSheet(_lorch_cb_style)
        self.compare_lorch_checkbox.setChecked(False)
        self.compare_lorch_checkbox.stateChanged.connect(lambda: self._on_lorch_toggle(self.compare_lorch_checkbox))
        gr_layout.addWidget(self.compare_lorch_checkbox)

        self.compare_all_checkbox = QCheckBox("Compare all")
        self.compare_all_checkbox.setStyleSheet(_lorch_cb_style)
        self.compare_all_checkbox.setChecked(False)
        self.compare_all_checkbox.stateChanged.connect(lambda: self._on_lorch_toggle(self.compare_all_checkbox))
        gr_layout.addWidget(self.compare_all_checkbox)

        gr_widget_action = QWidgetAction(self)
        gr_widget_action.setDefaultWidget(gr_widget)
        gr_menu.addAction(gr_widget_action)

        # Indicator Colors
        base_indicator_style = """
            QCheckBox::indicator {
                width: 12px; height: 12px; border-radius: 6px;
                border: 1px solid #777; background: #fff;
            }
        """
        self.background_checkbox.setStyleSheet(
            base_indicator_style + "QCheckBox::indicator:checked { background: #FF0000; border:1px solid #FF0000; }")
        self.original_checkbox.setStyleSheet(
            base_indicator_style + "QCheckBox::indicator:checked { background: #000000; border:1px solid #000000; }")
        self.original_sq_checkbox.setStyleSheet(
            base_indicator_style + "QCheckBox::indicator:checked { background: #000000; border:1px solid #000000; }")
        self.polynomial_sq_checkbox.setStyleSheet(
            base_indicator_style + "QCheckBox::indicator:checked { background: #FF0000; border:1px solid #FF0000; }")
        self.mean_sq_fi_checkbox.setStyleSheet(
            base_indicator_style + "QCheckBox::indicator:checked { background: #009600; border:1px solid #009600; }")
        self.sq_mean_fi_checkbox.setStyleSheet(
            base_indicator_style + "QCheckBox::indicator:checked { background: #800080; border:1px solid #800080; }")

        default_checked_style = base_indicator_style + "QCheckBox::indicator:checked { background: #FFA500; border:1px solid #FFA500; }"
        self.log_checkbox.setStyleSheet(default_checked_style)
        self.smoothed_fq_checkbox.setStyleSheet(
            base_indicator_style + "QCheckBox::indicator:checked { background: #000000; border:1px solid #000000; }")

        self.bottom_container_widget = QWidget()
        bottom_container_layout = QHBoxLayout(self.bottom_container_widget)
        bottom_container_layout.setContentsMargins(4, 0, 4, 0)
        bottom_container_layout.setSpacing(6)

        bottom_container_layout.addWidget(self.bottom_menu)
        bottom_container_layout.addStretch(1)

        self.lock_graph_checkbox = QCheckBox("Lock graph")
        self.lock_graph_checkbox.setStyleSheet("""
            QCheckBox { font-weight: bold; color: #B22222; }
            QCheckBox::indicator { width: 12px; height: 12px; border-radius: 3px; border: 1px solid #B22222; }
            QCheckBox::indicator:checked { background: #B22222; }
        """)
        self.lock_graph_checkbox.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        bottom_container_layout.addWidget(self.lock_graph_checkbox)

        if hasattr(self, 'plot_checkboxes'):
            for cb in self.plot_checkboxes:
                bottom_container_layout.addWidget(cb)

        self.layout.addWidget(self.bottom_container_widget)

    def is_locked(self):
        return self.lock_graph_checkbox.isChecked()

    def _reset_intermediates(self):
        self.bkg_curve = None
        self.raw_curve = None
        self.sq_curve_original = None
        self.sq_curve_polynomial = None
        self.mean_sq_fi_curve = None
        self.sq_mean_fi_curve = None
        self.fq_smoothed_curve = None
        self.gr_smoothed_curve = None
        self.gr_lorch_curve = None
        self.gr_smoothed_lorch_curve = None
        self.fq_lorch_curve = None
        self.fq_smoothed_lorch_curve = None
        self.fq_main_curve = None
        self.gr_main_curve = None

        self.raw_iq = None
        self.background_data = None
        self.sq_original_data = None
        self.sq_polynomial_data = None
        self.fq_smoothed_data = None
        self.mean_sq_fi = None
        self.sq_mean_fi = None
        self.gr_smoothed_data = None
        self.gr_lorch_data = None
        self.gr_smoothed_lorch_data = None
        self.fq_lorch_data = None
        self.fq_smoothed_lorch_data = None

    def _align_pair(self, x, y):
        try:
            if x is None or y is None:
                return x, y
            n = min(len(x), len(y))
            return x[:n], y[:n]
        except Exception:
            return x, y

    def _align_with_x(self, x, y):
        try:
            if x is None or y is None:
                return y
            n = min(len(x), len(y))
            return y[:n]
        except Exception:
            return y

    def plot_data(self, xs, ys, bkg_x, bkg_y, raw_x, raw_y,
                  list_Sq=None, Fq_smoothed=None, mean_sq_fi=None, sq_mean_fi=None,
                  r_smoothed=None, G_smoothed=None, title="", bring_front=True,
                  gr_lorch=None, gr_smoothed_lorch=None, **_kwargs):
        self._reset_intermediates()
        self.is_compton = False

        self.cached_xs = self.cached_ys = self.cached_titles = None

        raw_x, raw_y = self._align_pair(raw_x, raw_y)
        bkg_x, bkg_y = self._align_pair(bkg_x, bkg_y)
        if xs is not None and ys is not None:
            xs = list(xs)
            ys = list(ys)
            for i in range(min(len(xs), len(ys))):
                xs[i], ys[i] = self._align_pair(xs[i], ys[i])
        if xs is not None and len(xs) > 1:
            x_sq = xs[1]
            if list_Sq is not None:  list_Sq = self._align_with_x(x_sq, list_Sq)
            if mean_sq_fi is not None:  mean_sq_fi = self._align_with_x(x_sq, mean_sq_fi)
            if sq_mean_fi is not None:  sq_mean_fi = self._align_with_x(x_sq, sq_mean_fi)
        if xs is not None and len(xs) > 2 and Fq_smoothed is not None:
            Fq_smoothed = self._align_with_x(xs[2], Fq_smoothed)
        r_smoothed, G_smoothed = self._align_pair(r_smoothed, G_smoothed)
        _r_main = xs[3] if (xs is not None and len(xs) > 3) else None
        _r_lorch, gr_lorch = self._align_pair(_r_main, gr_lorch)
        _r_sm_lorch, gr_smoothed_lorch = self._align_pair(r_smoothed, gr_smoothed_lorch)

        # Only bring the window to the front when a new file is opened here.
        # During a parameter change every open plot window is replotted; raising
        # them all would make the windows jump back and forth, so replots pass
        # bring_front=False and stay wherever the user put them.
        if bring_front:
            self.bring_to_front()
        self.label.setText(f"Displaying: {title}")
        self.clear_slider_layout()

        self.file_name = title
        self.xs = xs
        self.ys = ys
        self.raw_iq = [raw_x, raw_y]
        self.background_data = [bkg_x, bkg_y]
        self.sq_original_data = list_Sq
        self.sq_polynomial_data = (
            (list_Sq - ys[1])
            if (list_Sq is not None and ys is not None and len(ys) > 1 and len(list_Sq) == len(ys[1]))
            else None
        )
        self.fq_smoothed_data = Fq_smoothed
        self.mean_sq_fi = mean_sq_fi
        self.sq_mean_fi = sq_mean_fi
        self.gr_smoothed_data = [r_smoothed, G_smoothed]
        self.gr_lorch_data = [_r_lorch, gr_lorch]
        self.gr_smoothed_lorch_data = [_r_sm_lorch, gr_smoothed_lorch]

        x_labels = ['q(1/Å)', 'q(1/Å)', 'q(1/Å)', 'r(Å)']
        y_labels = ['I(q)', 'S(q)', 'F(q)', 'G(Å⁻²)']

        pen_main = make_pen((0, 0, 255))
        pen_bkg = make_pen((255, 0, 0))
        pen_raw = make_pen((0, 0, 0))
        pen_f2 = make_pen((0, 150, 0))
        pen_f_avg2 = make_pen((128, 0, 128))
        pen_lorch = make_pen((255, 140, 0))      # raw + Lorch (orange)
        pen_lorch2 = make_pen((0, 150, 136))     # smoothed + Lorch (teal)

        is_dark = False
        app_instance = QApplication.instance()
        if hasattr(app_instance, 'is_dark_mode') and app_instance.is_dark_mode:
            is_dark = True
            pen_raw = make_pen((255, 255, 255))  # White for raw data in dark mode

        for i in range(4):
            setup_plot(self.plots[i], x_labels[i], y_labels[i])
            if hasattr(self, '_xy_labels') and len(self._xy_labels) > i and self._xy_labels[i] is not None:
                self.plots[i].addItem(self._xy_labels[i])
                self._reposition_xy_label(i)

            pi = self.plots[i].getPlotItem()
            pi.setLabel('bottom', f'<span style="font-size:{LABEL_PT}pt;">{x_labels[i]}</span>')
            pi.setLabel('left', f'<span style="font-size:{LABEL_PT}pt;">{y_labels[i]}</span>')

            if is_dark:
                for axis in ['left', 'bottom']:
                    pi.getAxis(axis).setPen('#E0E0E0')
                    pi.getAxis(axis).setTextPen('#E0E0E0')

            _main_curve = self.plots[i].plot(xs[i], ys[i], pen=pen_main)
            if i == 2:
                self.fq_main_curve = _main_curve
            elif i == 3:
                self.gr_main_curve = _main_curve

            if i == 0:
                self.plots[i].setLogMode(y=self.log_checkbox.isChecked())
                self.bkg_curve = plot_curve(self.plots[i], bkg_x, bkg_y, pen_bkg,
                                            visible=self.background_checkbox.isChecked())
                self.raw_curve = plot_curve(self.plots[i], raw_x, raw_y, pen_raw,
                                            visible=self.original_checkbox.isChecked())
            if i == 1:
                self.sq_curve_original = plot_curve(self.plots[i], xs[1], list_Sq, pen_raw,
                                                    visible=self.original_sq_checkbox.isChecked())
                self.sq_curve_polynomial = plot_curve(self.plots[i], xs[1], self.sq_polynomial_data, pen_bkg,
                                                      visible=self.polynomial_sq_checkbox.isChecked())
                self.mean_sq_fi_curve = plot_curve(self.plots[i], xs[1], self.mean_sq_fi, pen_f2,
                                                   visible=self.mean_sq_fi_checkbox.isChecked())
                self.sq_mean_fi_curve = plot_curve(self.plots[i], xs[1], self.sq_mean_fi, pen_f_avg2,
                                                   visible=self.sq_mean_fi_checkbox.isChecked())
            if i == 2:
                self.fq_smoothed_curve = plot_curve(self.plots[i], xs[2], Fq_smoothed, pen_raw,
                                                    visible=False)
                # F(q).Lorch = F(q) * M(q); derived here (no FFT needed).
                _qf = np.asarray(xs[2], dtype=float) if (xs[2] is not None and len(xs[2]) > 0) else None
                _Mf = lorch_window(_qf, float(_qf[-1])) if _qf is not None else None
                _fq_l = (np.asarray(ys[2], dtype=float) * _Mf) if (_Mf is not None and ys[2] is not None and len(ys[2]) == len(_Mf)) else None
                _fq_sm_l = (np.asarray(Fq_smoothed, dtype=float) * _Mf) if (_Mf is not None and Fq_smoothed is not None and len(Fq_smoothed) == len(_Mf)) else None
                self.fq_lorch_data = [xs[2], _fq_l]
                self.fq_smoothed_lorch_data = [xs[2], _fq_sm_l]
                self.fq_lorch_curve = plot_curve(self.plots[i], xs[2], _fq_l, pen_lorch, visible=False)
                self.fq_smoothed_lorch_curve = plot_curve(self.plots[i], xs[2], _fq_sm_l, pen_lorch2, visible=False)
            if i == 3:
                self.gr_smoothed_curve = plot_curve(self.plots[i], r_smoothed, G_smoothed, pen_raw, visible=False)
                self.gr_lorch_curve = plot_curve(self.plots[i], _r_lorch, gr_lorch, pen_lorch, visible=False)
                self.gr_smoothed_lorch_curve = plot_curve(self.plots[i], _r_sm_lorch, gr_smoothed_lorch, pen_lorch2, visible=False)

        self.update_visibility()
        self._apply_lorch_view()
        for i in range(4):
            self._autorange_and_fix_x0(i)

    def plot_multiple(self, list_of_xs, list_of_ys, titles=None, bring_front=True):
        self._reset_intermediates()
        self.is_compton = False

        if bring_front:
            self.bring_to_front()
        self.cached_xs = list_of_xs
        self.cached_ys = list_of_ys
        self.cached_titles = titles

        self.xs = None
        self.ys = None

        if self.slider_layout is None:
            self.show_offset_sliders()

        num_plots = 4
        offset_x = self.get_x_offset()
        offset_y = self.get_y_offset()

        x_labels = ['q(1/Å)', 'q(1/Å)', 'q(1/Å)', 'r(Å)']
        y_labels = ['I(q)', 'S(q)', 'F(1/Å)', 'G(Å⁻²)']
        colors = [(0, 0, 255), (255, 0, 0), (0, 150, 0), (255, 165, 0),
                  (128, 0, 128), (0, 255, 255), (255, 105, 180), (128, 128, 0)]

        is_dark = False
        app_instance = QApplication.instance()
        if hasattr(app_instance, 'is_dark_mode') and app_instance.is_dark_mode:
            is_dark = True

        def shorten_name(name, max_len=80):
            if not name: return ""
            if len(name) <= max_len: return name
            return f"{name[:30]}...{name[-20:]}"

        for i in range(num_plots):
            plot_item = self.plots[i].getPlotItem()

            # 1. Legend Handling: Keep existing legend object to preserve manual position
            if plot_item.legend is None:
                legend = self.plots[i].addLegend(offset=(-5, 5))
            else:
                legend = plot_item.legend
                legend.clear()  # Only clear the labels/items inside, not the position

            # 2. Clear Data Curves ONLY (do not clear the whole plot to save legend/readout)
            for item in self.plots[i].listDataItems():
                self.plots[i].removeItem(item)

            # 3. Setup Plot Appearance (Grid and Labels)
            self.plots[i].showGrid(x=False, y=False, alpha=0.3)

            # Legend Style
            legend.setBrush(pg.mkBrush(255, 255, 255, 0))
            if is_dark:
                legend.setLabelTextColor(pg.mkColor(255, 255, 255))
            else:
                legend.setLabelTextColor(pg.mkColor(0, 0, 0))

            legend.setFont(QFont("Segoe UI", 6))
            try:
                legend.layout.setColumnFixedWidth(1, 600)
                legend.layout.setColumnSpacing(0, 20)
                legend.layout.setVerticalSpacing(0)
                legend.layout.setContentsMargins(0, 0, 0, 0)
            except Exception:
                pass

            # 4. Ensure XY Readout is present
            if hasattr(self, '_xy_labels') and len(self._xy_labels) > i and self._xy_labels[i] is not None:
                if self._xy_labels[i] not in self.plots[i].items():
                    self.plots[i].addItem(self._xy_labels[i])
                self._reposition_xy_label(i)

            pi = self.plots[i].getPlotItem()
            pi.setLabel('bottom', f'<span style="font-size:{LABEL_PT}pt;">{x_labels[i]}</span>')
            pi.setLabel('left', f'<span style="font-size:{LABEL_PT}pt;">{y_labels[i]}</span>')

            if is_dark:
                for axis in ['left', 'bottom']:
                    pi.getAxis(axis).setPen('#E0E0E0')
                    pi.getAxis(axis).setTextPen('#E0E0E0')

        valid = 0
        for file_idx, (xs, ys) in enumerate(zip(list_of_xs, list_of_ys)):
            try:
                # ---------------- [버그 수정 구간 유지] ----------------
                if not xs or not ys or len(xs) != len(ys):
                    continue
                # -----------------------------------------------------
                color = colors[file_idx % len(colors)]
                pen = make_pen(color)

                raw_name = titles[file_idx] if titles and file_idx < len(titles) else None
                legend_name = shorten_name(raw_name)

                for i in range(min(len(xs), len(ys), num_plots)):
                    if len(xs[i]) == 0 or len(ys[i]) == 0:
                        continue
                    x = xs[i] + valid * offset_x
                    y = ys[i] + valid * offset_y
                    self.plots[i].plot(x, y, pen=pen, name=legend_name)
                valid += 1
            except Exception:
                continue

        self.label.setText(f"Displaying: {valid} file(s).")
        self.update_visibility()
        for i in range(4):
            self._autorange_and_fix_x0(i)

    def show_offset_sliders(self):
        if self.slider_layout is None:
            self.slider_bar_widget = QWidget(self)
            self.slider_bar_widget.setContentsMargins(0, 0, 0, 0)
            self.slider_bar_widget.setStyleSheet("margin:0; padding:0; background-color: #F0F0F0;")
            row = QHBoxLayout(self.slider_bar_widget)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(10)
            row.setAlignment(Qt.AlignmentFlag.AlignLeft)
            xlab = QLabel("X Offset")
            xlab.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            self.x_offset_slider = QSlider(Qt.Orientation.Horizontal)
            self.x_offset_slider.setRange(0, int(10.0 * 100))
            self.x_offset_slider.setSingleStep(1)
            self.x_offset_slider.setValue(0)
            self.x_offset_slider.setMaximumHeight(16)
            self.x_offset_slider.setMinimumWidth(120)
            self.x_offset_slider.setMaximumWidth(200)
            self.x_offset_input = QLineEdit("0.00")
            self.x_offset_input.setFixedWidth(50)
            self.x_offset_input.setAlignment(Qt.AlignmentFlag.AlignRight)
            xgrp = QHBoxLayout()
            xgrp.setContentsMargins(0, 0, 0, 0)
            xgrp.setSpacing(4)
            xgrp.addWidget(xlab)
            xgrp.addWidget(self.x_offset_slider)
            xgrp.addWidget(self.x_offset_input)
            xwrap = QWidget(self.slider_bar_widget)
            xwrap.setContentsMargins(0, 0, 0, 0)
            xwrap.setLayout(xgrp)
            xwrap.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            ylab = QLabel("Y Offset")
            ylab.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            self.y_offset_slider = QSlider(Qt.Orientation.Horizontal)
            self.y_offset_slider.setRange(0, int(10.0 * 100))
            self.y_offset_slider.setSingleStep(1)
            self.y_offset_slider.setValue(int(10.0 * 100))
            self.y_offset_slider.setMaximumHeight(16)
            self.y_offset_slider.setMinimumWidth(120)
            self.y_offset_slider.setMaximumWidth(200)
            self.y_offset_input = QLineEdit("10.00")
            self.y_offset_input.setFixedWidth(50)
            self.y_offset_input.setAlignment(Qt.AlignmentFlag.AlignRight)
            minus_btn = QPushButton("−")
            plus_btn = QPushButton("+")
            for b in (minus_btn, plus_btn):
                b.setFixedWidth(22)
                b.setStyleSheet("margin:0; padding:0;")

            def update_y_max(multiplier):
                new_max = max(100, int(self.y_offset_slider.maximum() * multiplier))
                self.y_offset_slider.setMaximum(new_max)
                if self.y_offset_slider.value() > new_max:
                    self.y_offset_slider.setValue(new_max)

            minus_btn.clicked.connect(lambda: update_y_max(0.5))
            plus_btn.clicked.connect(lambda: update_y_max(2.0))
            ygrp = QHBoxLayout()
            ygrp.setContentsMargins(0, 0, 0, 0)
            ygrp.setSpacing(4)
            ygrp.addWidget(ylab)
            ygrp.addWidget(self.y_offset_slider)
            ygrp.addWidget(self.y_offset_input)
            ygrp.addWidget(minus_btn)
            ygrp.addWidget(plus_btn)
            ywrap = QWidget(self.slider_bar_widget)
            ywrap.setContentsMargins(0, 0, 0, 0)
            ywrap.setLayout(ygrp)
            ywrap.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            row.addWidget(xwrap)
            row.addWidget(ywrap)
            row.addStretch(1)

            def _s2txt(slider, edit):
                edit.setText(f"{slider.value() / 100.0:.2f}")

            def _txt2s(edit, slider):
                try:
                    val = float(edit.text())
                    slider.setValue(int(round(val * 100)))
                except Exception:
                    pass

            self.x_offset_slider.valueChanged.connect(
                lambda _: (_s2txt(self.x_offset_slider, self.x_offset_input),
                           self.update_offsets_and_replot()))
            self.y_offset_slider.valueChanged.connect(
                lambda _: (_s2txt(self.y_offset_slider, self.y_offset_input),
                           self.update_offsets_and_replot()))
            self.x_offset_input.editingFinished.connect(
                lambda: (_txt2s(self.x_offset_input, self.x_offset_slider),
                         self.update_offsets_and_replot()))
            self.y_offset_input.editingFinished.connect(
                lambda: (_txt2s(self.y_offset_input, self.y_offset_slider),
                         self.update_offsets_and_replot()))
            self.slider_layout = row
            self.layout.insertWidget(2, self.slider_bar_widget)
            self._resize_offset_controls()
            app_instance = QApplication.instance()
            if hasattr(app_instance, 'is_dark_mode') and app_instance.is_dark_mode:
                self.slider_bar_widget.setStyleSheet("margin:0; padding:0; background-color: #2b2b2b;")

    def _resize_offset_controls(self):
        maxw = max(150, self.width() // 5)
        if self.x_offset_slider:
            self.x_offset_slider.setMaximumWidth(min(self.x_offset_slider.maximumWidth(), maxw))
        if self.y_offset_slider:
            self.y_offset_slider.setMaximumWidth(min(self.y_offset_slider.maximumWidth(), maxw))

    def clear_slider_layout(self):
        if self.slider_layout:
            for sig in (getattr(self, "x_offset_slider", None),
                        getattr(self, "y_offset_slider", None)):
                if sig:
                    try:
                        sig.valueChanged.disconnect()
                    except Exception:
                        pass
            for edit in (getattr(self, "x_offset_input", None),
                         getattr(self, "y_offset_input", None)):
                if edit:
                    try:
                        edit.editingFinished.disconnect()
                    except Exception:
                        pass
            parent_widget = self.slider_layout.parentWidget()
            while self.slider_layout.count():
                item = self.slider_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.setParent(None)
                    widget.deleteLater()
            if parent_widget:
                parent_widget.setParent(None)
                parent_widget.deleteLater()
            self.slider_layout = None
            self.slider_bar_widget = None
            self.x_offset_slider = self.y_offset_slider = None
            self.x_offset_input = self.y_offset_input = None

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._resize_offset_controls()

    def update_visibility(self):
        update_visibility(self.plot_layout, self.plot_checkboxes, self.plots, self.plot_container)

    def toggle_log_scale(self):
        # Preserve the current view across the linear/log switch so zooming in
        # and toggling the scale does not reset to the full auto-range.
        keep_view = self._user_zoomed[0] or self._is_zoomed(0)
        vb = self.plots[0].getViewBox()
        try:
            (x0, x1), (y0, y1) = vb.viewRange()
        except Exception:
            keep_view = False
        self.plots[0].setLogMode(y=self.log_checkbox.isChecked())
        self.enable_graphs(enable_iq=True)
        if keep_view:
            # y is in log units when log mode is on; keep x, refit y to the
            # visible data so the zoomed q-window stays put.
            try:
                vb.enableAutoRange(x=False, y=True)
                self.plots[0].setXRange(float(x0), float(x1), padding=0)
            except Exception:
                pass
        else:
            self._autorange_and_fix_x0(0)

    def show_background_data(self):
        if self.bkg_curve:
            self.bkg_curve.setVisible(self.background_checkbox.isChecked())
            self.enable_graphs(enable_iq=True)
            if not (self._user_zoomed[0] or self._is_zoomed(0)):
                try:
                    self.plots[0].getViewBox().enableAutoRange(x=True, y=True)
                except Exception:
                    pass
                self._autorange_and_fix_x0(0)

    def show_original_data(self):
        if self.raw_curve:
            self.raw_curve.setVisible(self.original_checkbox.isChecked())
            self.enable_graphs(enable_iq=True)
            if not (self._user_zoomed[0] or self._is_zoomed(0)):
                try:
                    self.plots[0].getViewBox().enableAutoRange(x=True, y=True)
                except Exception:
                    pass
                self._autorange_and_fix_x0(0)

    def toggle_original_sq(self):
        if self.sq_curve_original:
            self.sq_curve_original.setVisible(self.original_sq_checkbox.isChecked())
            self.enable_graphs(enable_sq=True)
            if not self._user_zoomed[1]:
                self._autorange_and_fix_x0(1)

    def toggle_polynomial_sq(self):
        if self.sq_curve_polynomial:
            self.sq_curve_polynomial.setVisible(self.polynomial_sq_checkbox.isChecked())
            self.enable_graphs(enable_sq=True)
            if not self._user_zoomed[1]:
                self._autorange_and_fix_x0(1)

    def toggle_mean_sq_fi(self):
        if self.mean_sq_fi_curve:
            self.mean_sq_fi_curve.setVisible(self.mean_sq_fi_checkbox.isChecked())
            self.enable_graphs(enable_sq=True)
            if not self._user_zoomed[1]:
                self._autorange_and_fix_x0(1)

    def toggle_sq_mean_fi(self):
        if self.sq_mean_fi_curve:
            self.sq_mean_fi_curve.setVisible(self.sq_mean_fi_checkbox.isChecked())
            self.enable_graphs(enable_sq=True)
            if not self._user_zoomed[1]:
                self._autorange_and_fix_x0(1)

    def toggle_smoothed_fq(self):
        # "Smoothed" is now a state axis feeding _apply_lorch_view (it selects
        # the smoothed curves as the base instead of overlaying on raw).
        self._apply_lorch_view()

    def _on_lorch_toggle(self, sender):
        # The three Lorch modes are mutually exclusive: turning one on turns
        # the others off; turning all off returns to the raw view.
        if sender.isChecked():
            for cb in (self.apply_lorch_checkbox, self.compare_lorch_checkbox,
                       self.compare_all_checkbox):
                if cb is not sender and cb.isChecked():
                    cb.blockSignals(True)
                    cb.setChecked(False)
                    cb.blockSignals(False)
        self._apply_lorch_view()

    def _apply_lorch_view(self):
        """Show/hide the F(q) and G(r) curves per the Lorch toggles.

        Axes: Smoothed (smoothed_fq_checkbox), Apply Lorch, Compare, Compare all.
        The same rule drives both the F(q) and G(r) panels.
        """
        # A Compton plot has only an I(q)-type curve; the Lorch/Smoothed toggles
        # don't apply, so do nothing here. Otherwise this would force-enable the
        # (empty) F(q) and G(r) panels.
        if getattr(self, "is_compton", False):
            return
        # "Smoothed" only has an effect when smoothed curves actually exist
        # (they don't for the xyz/CAL path). Otherwise a checked "Smoothed F(q)"
        # would route Apply Lorch / Compare to non-existent smoothed variants,
        # making those toggles look broken for xyz.
        sm_available = (getattr(self, "fq_smoothed_curve", None) is not None
                        or getattr(self, "gr_smoothed_curve", None) is not None)
        sm = bool(self.smoothed_fq_checkbox.isChecked()) and sm_available
        lorch = bool(getattr(self, "apply_lorch_checkbox", None) and self.apply_lorch_checkbox.isChecked())
        comp = bool(getattr(self, "compare_lorch_checkbox", None) and self.compare_lorch_checkbox.isChecked())
        compall = bool(getattr(self, "compare_all_checkbox", None) and self.compare_all_checkbox.isChecked())

        show = {"raw": False, "raw_l": False, "sm": False, "sm_l": False}
        if compall:
            show = {"raw": True, "raw_l": True, "sm": True, "sm_l": True}
        elif comp:
            if sm:
                show["sm"] = True; show["sm_l"] = True
            else:
                show["raw"] = True; show["raw_l"] = True
        else:
            if sm and lorch:   show["sm_l"] = True
            elif sm:
                # Smoothed on: show the unsmoothed curve together with the
                # smoothed one so they can be compared (e.g. while tuning the
                # WH-smoothing lambda / order).
                show["raw"] = True; show["sm"] = True
            elif lorch:        show["raw_l"] = True
            else:              show["raw"] = True

        def _set_panel(curves):
            # curves: list of (attr_name, show_key); curves[0] must be the raw base.
            any_shown = False
            for attr, key in curves:
                c = getattr(self, attr, None)
                if c is not None:
                    vis = bool(show[key])
                    c.setVisible(vis)
                    any_shown = any_shown or vis
            # If the requested variant doesn't exist (e.g. an xyz panel without a
            # smoothed/Lorch curve), fall back to the raw base so the panel is
            # never left blank.
            if not any_shown:
                base = getattr(self, curves[0][0], None)
                if base is not None:
                    base.setVisible(True)

        _set_panel([("fq_main_curve", "raw"), ("fq_lorch_curve", "raw_l"),
                    ("fq_smoothed_curve", "sm"), ("fq_smoothed_lorch_curve", "sm_l")])
        _set_panel([("gr_main_curve", "raw"), ("gr_lorch_curve", "raw_l"),
                    ("gr_smoothed_curve", "sm"), ("gr_smoothed_lorch_curve", "sm_l")])
        # Only set curve visibility here; do NOT force the F(q)/G(r) panels on.
        # Panel visibility is the user's choice (legend checkboxes), and this
        # method also runs on every re-plot (e.g. moving the background-scale
        # slider) -- forcing panels on there would resurrect panels the user
        # hid. Autorange only visible panels that the user has NOT zoomed (a
        # zoomed panel keeps its view; the toggled curve stays in range anyway).
        zoomed = getattr(self, "_user_zoomed", [False, False, False, False])
        if len(self.plots) > 2 and self.plots[2].isVisible() and not zoomed[2]:
            self._autorange_and_fix_x0(2)
        if len(self.plots) > 3 and self.plots[3].isVisible() and not zoomed[3]:
            self._autorange_and_fix_x0(3)

    def plot_under_cursor(self, global_pos):
        for plot in self.plots:
            if not plot.isVisible():
                continue
            local_pos = plot.mapFromGlobal(global_pos)
            if plot.rect().contains(local_pos):
                return plot
        return None

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        if event.key() == Qt.Key.Key_R:
            self.reset_zoom_on_hovered_plot()
        elif event.key() == Qt.Key.Key_Z:
            self.left_button_pan_enabled = not self.left_button_pan_enabled
            for plot in self.plots:
                vb = plot.getViewBox()
                vb.setMouseMode(pg.ViewBox.PanMode if self.left_button_pan_enabled else pg.ViewBox.RectMode)

    def _mark_user_zoomed(self, idx):
        try:
            if 0 <= idx < len(self._user_zoomed):
                self._user_zoomed[idx] = True
        except Exception:
            pass

    def _is_zoomed(self, idx):
        """True if the panel's current view spans notably less than its data
        extent (i.e. the user has zoomed in). Deterministic — does not rely on
        the manual-range signal, which can miss some zoom interactions."""
        try:
            vb = self.plots[idx].getViewBox()
            (vx0, vx1), (vy0, vy1) = vb.viewRange()
            b = vb.childrenBounds()
            if not b:
                return bool(self._user_zoomed[idx]) if idx < len(self._user_zoomed) else False
            zoomed = False
            if b[0] is not None:
                dsx = float(b[0][1]) - float(b[0][0])
                if dsx > 0 and (float(vx1) - float(vx0)) < 0.98 * dsx:
                    zoomed = True
            if b[1] is not None:
                dsy = float(b[1][1]) - float(b[1][0])
                if dsy > 0 and (float(vy1) - float(vy0)) < 0.98 * dsy:
                    zoomed = True
            return zoomed
        except Exception:
            try:
                return bool(self._user_zoomed[idx])
            except Exception:
                return False

    def reset_zoom_on_hovered_plot(self):
        global_pos = QCursor.pos()
        plot = self.plot_under_cursor(global_pos)
        if not plot:
            return
        try:
            idx = self.plots.index(plot)
            # R returns this panel to auto-scaling: clear its user-zoom flag so
            # later parameter changes auto-fit it again.
            if 0 <= idx < len(self._user_zoomed):
                self._user_zoomed[idx] = False
            self._autorange_and_fix_x0(idx)
        except Exception:
            try:
                plot.getViewBox().autoRange()
                self._force_x_from_zero(plot)
            except Exception:
                pass

    def export_graphs(self):
        dialog = SaveMenu(self)
        if dialog.exec():
            dialog.export_images()

    def export_graph_data(self):
        dialog = SaveMenu(self)
        if dialog.exec():
            dialog.export_files()

    def _save_data_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Select data to save")
        v = QVBoxLayout(dlg)
        CB_ROUND_STYLE = """
        QCheckBox::indicator { width:12px; height:12px; border-radius:6px;
            border:1px solid #777; background:#fff; }
        QCheckBox::indicator:checked { background:#444; }
        """
        is_compton = getattr(self, "is_compton", False)
        is_multi_plot = hasattr(self, 'cached_xs') and self.cached_xs is not None
        is_single_exp = False
        if not is_multi_plot:
            is_single_exp = (self.raw_iq is not None and self.raw_iq[0] is not None) or \
                            (self.background_data is not None and self.background_data[0] is not None) or \
                            (self.mean_sq_fi is not None)
        is_experimental = (is_multi_plot or is_single_exp) and not is_compton
        # A Compton curve is neither experimental data nor calculated (.xyz)
        # output; it has its own save option below.
        is_calculated = not is_experimental and not is_compton
        cols = QHBoxLayout()
        exp_col = QVBoxLayout()
        cal_col = QVBoxLayout()
        compton_col = QVBoxLayout()

        # --- Experimental Data column ---
        exp_col.addWidget(QLabel("--- Experimental Data ---", dlg))
        checks_core = []
        core_labels = [("I(q)", 0, ".iq"), ("S(q)", 1, ".sq"), ("F(q)", 2, ".fq"), ("G(r)", 3, ".gr")]
        for text_label, idx, _ext in core_labels:
            cb = QCheckBox(text_label, dlg)
            cb.setStyleSheet(CB_ROUND_STYLE)
            cb.setChecked(False)
            has_data = False
            try:
                xs_src = self.cached_xs[0] if (is_multi_plot and self.cached_xs) else self.xs
                has_data = xs_src is not None and len(xs_src) > idx and xs_src[idx] is not None and len(xs_src[idx]) > 0
            except Exception:
                pass
            if not (has_data and is_experimental):
                cb.setEnabled(False)
            exp_col.addWidget(cb)
            checks_core.append(cb)
        # F(q) variants are listed before G(r) variants.
        extras_def = [
            ("Original S(q)", 1, lambda: self.sq_original_data, "_Sq_original", ".sq"),
            ("Polynomial for S(q)", 1, lambda: self.sq_polynomial_data, "_Sq_polynomial", ".sq"),
            ("<f^2>", 1, lambda: self.mean_sq_fi, "_f2", ".dat"),
            ("<f>^2", 1, lambda: self.sq_mean_fi, "_favg2", ".dat"),
            ("Smoothed F(q)", 2, lambda: self.fq_smoothed_data, "_Fq_smoothed", ".fq"),
            ("Lorch F(q)", 2, lambda: self.fq_lorch_data[1] if getattr(self, "fq_lorch_data", None) else None,
             "_Fq_lorch", ".fq"),
            ("Smoothed Lorch F(q)", 2, lambda: self.fq_smoothed_lorch_data[1] if getattr(self, "fq_smoothed_lorch_data", None) else None,
             "_Fq_smoothed_lorch", ".fq"),
            ("Smoothed G(r)", 3, lambda: self.gr_smoothed_data[1] if self.gr_smoothed_data else None,
             "_Gr_smoothed", ".gr"),
            ("Lorch G(r)", 3, lambda: self.gr_lorch_data[1] if getattr(self, "gr_lorch_data", None) else None,
             "_Gr_lorch", ".gr"),
            ("Smoothed Lorch G(r)", 3, lambda: self.gr_smoothed_lorch_data[1] if getattr(self, "gr_smoothed_lorch_data", None) else None,
             "_Gr_smoothed_lorch", ".gr"),
        ]
        checks_extras = []
        if not is_multi_plot:
            for label, _xidx, y_fn, _suf, _ext in extras_def:
                cb = QCheckBox(label, dlg)
                cb.setStyleSheet(CB_ROUND_STYLE)
                cb.setChecked(False)
                y_ok = False
                if is_experimental:
                    try:
                        y = y_fn()
                        y_ok = y is not None and (isinstance(y, (np.ndarray, list)) and len(y) > 0)
                    except Exception:
                        y_ok = False
                if not y_ok:
                    cb.setEnabled(False)
                exp_col.addWidget(cb)
                checks_extras.append(cb)
        exp_col.addStretch()

        # --- Calculated (.xyz) column ---
        cal_col.addWidget(QLabel("--- Calculated (.xyz) ---", dlg))
        checks_cal = []
        # Grouped so each Lorch variant follows its base: I(q), S(q),
        # F(q), F(q) Lorch, G(r), G(r) Lorch. The save mapping in
        # _save_selected_data uses this exact order.
        cal_defs = [
            ("calI(q)", "core", 0),
            ("calS(q)", "core", 1),
            ("calF(q)", "core", 2),
            ("calF(q) Lorch", "lorch", "fq_lorch_data"),
            ("calG(r)", "core", 3),
            ("calG(r) Lorch", "lorch", "gr_lorch_data"),
        ]
        for text_label, kind, ref in cal_defs:
            cb = QCheckBox(text_label, dlg)
            cb.setStyleSheet(CB_ROUND_STYLE)
            cb.setChecked(False)
            has_data = False
            try:
                if kind == "core":
                    src = self.cached_xs[0] if is_multi_plot and self.cached_xs else self.xs
                    has_data = (src is not None and len(src) > ref
                                and src[ref] is not None and len(src[ref]) > 0)
                else:
                    d = getattr(self, ref, None)
                    has_data = d is not None and d[1] is not None and len(d[1]) > 0
            except Exception:
                pass
            if not (has_data and is_calculated):
                cb.setEnabled(False)
            cal_col.addWidget(cb)
            checks_cal.append(cb)
        cal_col.addStretch()

        # --- Compton column ---
        compton_col.addWidget(QLabel("--- Compton ---", dlg))
        cb_compton = QCheckBox("Compton", dlg)
        cb_compton.setStyleSheet(CB_ROUND_STYLE)
        cb_compton.setChecked(False)
        compton_has_data = False
        try:
            compton_has_data = (self.xs is not None and len(self.xs) > 0
                                and self.xs[0] is not None and len(self.xs[0]) > 0)
        except Exception:
            compton_has_data = False
        if not (is_compton and compton_has_data):
            cb_compton.setEnabled(False)
        compton_col.addWidget(cb_compton)
        compton_col.addStretch()

        cols.addLayout(exp_col)
        cols.addLayout(cal_col)
        cols.addLayout(compton_col)
        v.addLayout(cols)
        btns = QHBoxLayout()
        ok = QPushButton("Save", dlg)
        cancel = QPushButton("Cancel", dlg)
        btns.addWidget(ok);
        btns.addWidget(cancel)
        v.addLayout(btns)

        def do_save():
            selected_core = [i for i, cb in enumerate(checks_core) if cb.isChecked()]
            selected_extras = [i for i, cb in enumerate(checks_extras) if cb.isChecked()]
            selected_cal = [i for i, cb in enumerate(checks_cal) if cb.isChecked()]
            selected_compton = cb_compton.isChecked()
            if not selected_core and not selected_extras and not selected_cal and not selected_compton:
                QMessageBox.information(self, "Nothing selected", "Please select at least one dataset to save.")
                return

            # Remember the last folder used for saving, shared with the other
            # save dialogs (same QSettings key), so every save reopens where
            # the previous one left off.
            settings = QSettings("EZPDF", "EZPDF")
            last_dir = settings.value("last_dir", "") or ""

            # Compton is a single file, so let the user type the file name
            # directly (choosing the folder at the same time). On a Compton
            # plot the other datasets are disabled, so nothing else can be
            # selected here.
            if selected_compton:
                default_name = self.file_name or "compton"
                if default_name.lower().endswith(".compton"):
                    default_name = default_name[:-len(".compton")]
                start_path = os.path.join(last_dir, default_name + ".compton") if last_dir \
                    else default_name + ".compton"
                path, _ = QFileDialog.getSaveFileName(
                    self, "Save Compton File", start_path,
                    "Compton Files (*.compton)")
                if not path:
                    return
                if not path.lower().endswith(".compton"):
                    path += ".compton"
                written = self._write_compton_file(path)
                if written:
                    settings.setValue("last_dir", os.path.dirname(path))
                    settings.sync()
                    QMessageBox.information(self, "Saved", f"Compton file saved to:\n{path}")
                else:
                    QMessageBox.warning(self, "Save failed", "No Compton data was available to save.")
                dlg.accept()
                return

            folder = QFileDialog.getExistingDirectory(self, "Select folder to save", last_dir)
            if not folder: return
            settings.setValue("last_dir", folder)
            settings.sync()
            written = self._save_selected_data(folder, selected_core, selected_extras, selected_cal)
            QMessageBox.information(self, "Saved", f"{written} file(s) saved to:\n{folder}")
            dlg.accept()

        ok.clicked.connect(do_save)
        cancel.clicked.connect(dlg.reject)
        dlg.exec()

    def _write_compton_file(self, path):
        """Write the Compton I(q) curve (panel 0) to a .compton file.

        Returns 1 on success, 0 if there is no data or the write fails.
        """
        try:
            x = self.xs[0] if (self.xs is not None and len(self.xs) > 0) else None
            y = self.ys[0] if (self.ys is not None and len(self.ys) > 0) else None
            if x is None or y is None or len(x) == 0 or len(y) == 0:
                return 0
            n = min(len(x), len(y))
            arr = np.column_stack([x[:n], y[:n]])
            np.savetxt(path, arr, fmt="%.10g", header="x y", comments="")
            return 1
        except Exception:
            return 0

    def _save_selected_data(self, folder, selected_core, selected_extras, selected_cal=None):
        def _write_xy(path, x, y):
            try:
                if x is None or y is None or len(x) == 0 or len(y) == 0: return 0
                n = min(len(x), len(y))
                arr = np.column_stack([x[:n], y[:n]])
                np.savetxt(path, arr, fmt="%.10g", header="x y", comments="")
                return 1
            except Exception:
                return 0

        written = 0
        is_multi = hasattr(self, "cached_xs") and self.cached_xs is not None
        if is_multi:
            count = len(self.cached_xs)
            core_map = {0: ("_I", 0, ".iq"), 1: ("_S", 1, ".sq"), 2: ("_F", 2, ".fq"), 3: ("_G", 3, ".gr")}
            cal_map = {0: (".caliq", 0), 1: (".calsq", 1), 2: (".calfq", 2), 3: (".calgr", 3)}
            for k in range(count):
                if hasattr(self, "cached_titles") and self.cached_titles and k < len(self.cached_titles):
                    base_name = self.cached_titles[k]
                else:
                    base_name = f"data_{k}"
                base = os.path.splitext(base_name)[0]
                xs_src = self.cached_xs[k]
                ys_src = self.cached_ys[k]
                for idx in selected_core:
                    suf, ax, ext = core_map[idx]
                    try:
                        if xs_src and ys_src and len(xs_src) > ax and len(ys_src) > ax:
                            written += _write_xy(os.path.join(folder, f"{base}{suf}{ext}"), xs_src[ax], ys_src[ax])
                    except Exception:
                        pass
                if selected_cal:
                    for idx in selected_cal:
                        ext, ax = cal_map[idx]
                        try:
                            if xs_src and ys_src and len(xs_src) > ax and len(ys_src) > ax:
                                written += _write_xy(os.path.join(folder, f"{base}{ext}"), xs_src[ax], ys_src[ax])
                        except Exception:
                            pass
        else:
            xs_src = self.xs;
            ys_src = self.ys;
            base_name = self.file_name or "data";
            base = os.path.splitext(base_name)[0]
            core_map = {0: ("_I", 0, ".iq"), 1: ("_S", 1, ".sq"), 2: ("_F", 2, ".fq"), 3: ("_G", 3, ".gr")}
            for idx in selected_core:
                suf, ax, ext = core_map[idx]
                try:
                    if xs_src and ys_src and len(xs_src) > ax and len(ys_src) > ax:
                        written += _write_xy(os.path.join(folder, f"{base}{suf}{ext}"), xs_src[ax], ys_src[ax])
                except Exception:
                    pass

            def _xy_or_none(ax, y):
                try:
                    if xs_src is None or len(xs_src) <= ax or y is None: return None, None
                    return xs_src[ax], y
                except Exception:
                    return None, None

            extra_sources = [
                ("_Sq_original", 1, lambda: _xy_or_none(1, self.sq_original_data), ".sq"),
                ("_Sq_polynomial", 1, lambda: _xy_or_none(1, self.sq_polynomial_data), ".sq"),
                ("_f2", 1, lambda: _xy_or_none(1, self.mean_sq_fi), ".dat"),
                ("_favg2", 1, lambda: _xy_or_none(1, self.sq_mean_fi), ".dat"),
                ("_Fq_smoothed", 2, lambda: _xy_or_none(2, self.fq_smoothed_data), ".fq"),
                ("_Fq_lorch", 2,
                 lambda: (self.fq_lorch_data[0], self.fq_lorch_data[1]) if getattr(self, "fq_lorch_data", None) else (None, None),
                 ".fq"),
                ("_Fq_smoothed_lorch", 2,
                 lambda: (self.fq_smoothed_lorch_data[0], self.fq_smoothed_lorch_data[1]) if getattr(self, "fq_smoothed_lorch_data", None) else (None, None),
                 ".fq"),
                ("_Gr_smoothed", 3,
                 lambda: (self.gr_smoothed_data[0], self.gr_smoothed_data[1]) if self.gr_smoothed_data else (None, None),
                 ".gr"),
                ("_Gr_lorch", 3,
                 lambda: (self.gr_lorch_data[0], self.gr_lorch_data[1]) if getattr(self, "gr_lorch_data", None) else (None, None),
                 ".gr"),
                ("_Gr_smoothed_lorch", 3,
                 lambda: (self.gr_smoothed_lorch_data[0], self.gr_smoothed_lorch_data[1]) if getattr(self, "gr_smoothed_lorch_data", None) else (None, None),
                 ".gr"),
            ]
            for i in selected_extras:
                suf, _ax, getter, ext = extra_sources[i]
                try:
                    x, y = getter()
                    if x is not None and y is not None:
                        written += _write_xy(os.path.join(folder, f"{base}{suf}{ext}"), x, y)
                except Exception:
                    pass
            if selected_cal:
                # Same order as the dialog's cal_defs:
                # I(q), S(q), F(q), F(q) Lorch, G(r), G(r) Lorch.
                cal_defs = [
                    (".caliq", "core", 0),
                    (".calsq", "core", 1),
                    (".calfq", "core", 2),
                    ("_lorch.calfq", "lorch", "fq_lorch_data"),
                    (".calgr", "core", 3),
                    ("_lorch.calgr", "lorch", "gr_lorch_data"),
                ]
                for idx in selected_cal:
                    try:
                        ext, kind, ref = cal_defs[idx]
                        if kind == "core":
                            if xs_src and ys_src and len(xs_src) > ref and len(ys_src) > ref:
                                written += _write_xy(os.path.join(folder, f"{base}{ext}"), xs_src[ref], ys_src[ref])
                        else:
                            d = getattr(self, ref, None)
                            if d and d[0] is not None and d[1] is not None:
                                written += _write_xy(os.path.join(folder, f"{base}{ext}"), d[0], d[1])
                    except Exception:
                        pass
        return written

    def save_combined_graph_image(self):
        try:
            from PySide6.QtSvg import QSvgGenerator
        except Exception:
            QSvgGenerator = None
        filters = "PNG (*.png);;JPEG (*.jpg *.jpeg);;SVG (*.svg);;PDF (*.pdf);;All Supported (*.png *.jpg *.jpeg *.svg *.pdf);;All Files (*)"
        default_name = (self.file_name or "plot") + "_figure.png"
        # Reuse the folder remembered by the other save dialogs.
        settings = QSettings("EZPDF", "EZPDF")
        last_dir = settings.value("last_dir", "") or ""
        start_path = os.path.join(last_dir, default_name) if last_dir else default_name
        file_path, selected_filter = QFileDialog.getSaveFileName(self, "Save figure", start_path, filters)
        if not file_path: return
        ext = os.path.splitext(file_path)[1].lower()
        if not ext:
            if "SVG" in selected_filter.upper():
                ext = ".svg";
                file_path += ext
            elif "PDF" in selected_filter.upper():
                ext = ".pdf";
                file_path += ext
            elif "JPEG" in selected_filter.upper():
                ext = ".jpg";
                file_path += ext
            else:
                ext = ".png";
                file_path += ext
        selected = [pw for cb, pw in zip(self.plot_checkboxes, self.plots) if cb.isChecked()]
        if not selected:
            QMessageBox.information(self, "No plots selected", "Please enable at least one plot to save.")
            return
        # Remember this folder for the next save.
        settings.setValue("last_dir", os.path.dirname(file_path))
        settings.sync()
        union_rect = self.main_widget.rect()
        if ext == ".svg" and QSvgGenerator is not None:
            gen = QSvgGenerator();
            gen.setFileName(file_path);
            gen.setSize(QSize(union_rect.width(), union_rect.height()));
            gen.setViewBox(union_rect)
            p = QPainter(gen);
            self.main_widget.render(p, QPoint());
            p.end();
            return
        if ext == ".pdf":
            writer = QPdfWriter(file_path);
            writer.setResolution(300);
            writer.setPageMargins(QMarginsF(0, 0, 0, 0));
            dpi = writer.resolution();
            pdf_rect = union_rect;
            w_pt = pdf_rect.width() * 72.0 / dpi;
            h_pt = pdf_rect.height() * 72.0 / dpi;
            writer.setPageSize(QPageSize(QSizeF(w_pt, h_pt), QPageSize.Unit.Point))
            p = QPainter(writer);
            self.main_widget.render(p, QPoint());
            p.end();
            return
        # Raster (PNG/JPG): render each selected plot with pyqtgraph's own
        # ImageExporter at a high target width so lines and text stay vector-
        # crisp, then stack the panels vertically. (QWidget.render() + painter
        # scaling just upscales pyqtgraph's cached bitmap, which looks blurry.)
        target_width = 3600  # px; raises effective DPI of the saved figure
        panel_imgs = None
        try:
            from pyqtgraph.exporters import ImageExporter
            panel_imgs = []
            for pw in selected:
                exp = ImageExporter(pw.plotItem)
                try:
                    exp.parameters()['width'] = int(target_width)
                    exp.parameters()['antialias'] = True
                except Exception:
                    pass
                qi = exp.export(toBytes=True)
                if qi is not None and not qi.isNull():
                    panel_imgs.append(qi)
            if not panel_imgs:
                panel_imgs = None
        except Exception:
            panel_imgs = None

        if panel_imgs:
            total_h = sum(im.height() for im in panel_imgs)
            max_w = max(im.width() for im in panel_imgs)
            img = QImage(max_w, total_h, QImage.Format.Format_ARGB32)
            img.fill(QColor(255, 255, 255))
            p = QPainter(img)
            y = 0
            for im in panel_imgs:
                p.drawImage(0, y, im)
                y += im.height()
            p.end()
        else:
            # Fallback: high-scale widget render (still better than 1x).
            scale = 6.0
            w = max(1, int(union_rect.width() * scale))
            h = max(1, int(union_rect.height() * scale))
            img = QImage(w, h, QImage.Format.Format_ARGB32)
            img.fill(QColor(255, 255, 255))
            p = QPainter(img)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            try:
                p.setRenderHint(QPainter.RenderHint.HighQualityAntialiasing, True)
            except Exception:
                pass
            p.scale(scale, scale)
            self.main_widget.render(p, QPoint())
            p.end()

        if ext in (".jpg", ".jpeg"):
            img.save(file_path, "JPG", quality=100)
        else:
            img.save(file_path, "PNG")

    def get_x_offset(self):
        try:
            return float(self.x_offset_input.text())
        except (ValueError, AttributeError):
            return 0.0

    def get_y_offset(self):
        try:
            return float(self.y_offset_input.text())
        except (ValueError, AttributeError):
            return 10.0

    def update_offsets_and_replot(self):
        if hasattr(self, "cached_xs") and hasattr(self, "cached_ys"): self.plot_multiple(self.cached_xs, self.cached_ys,
                                                                                         self.cached_titles)

    def bring_to_front(self):
        self.show();
        self.raise_()

    def enable_graphs(self, enable_iq=None, enable_sq=None, enable_fq=None, enable_gr=None):
        def set_graph_state(index, enable):
            if enable is not None:
                self.plot_checkboxes[index].blockSignals(True);
                self.plot_checkboxes[index].setChecked(enable);
                self.plot_checkboxes[index].blockSignals(False)
                if enable:
                    self.plots[index].show()
                else:
                    self.plots[index].hide()

        set_graph_state(0, enable_iq);
        set_graph_state(1, enable_sq);
        set_graph_state(2, enable_fq);
        set_graph_state(3, enable_gr);
        self.update_visibility()

    def _install_xy_readouts(self):
        self._xy_labels = [];
        self._mouse_proxies = [];
        self._active_idx = None
        for idx, pw in enumerate(self.plots):
            vb = pw.getPlotItem().getViewBox();
            lbl = pg.TextItem("", anchor=(1, 1), color=(40, 40, 40))
            try:
                lbl.setFill(pg.mkBrush(255, 255, 255, 180))
            except Exception:
                pass
            lbl.setFont(QFont("Arial", HUD_PT));
            lbl.setZValue(1e6);
            lbl.setVisible(False);
            pw.addItem(lbl);
            self._xy_labels.append(lbl)
            vb.sigRangeChanged.connect(lambda _vb, _vr, i=idx: self._reposition_xy_label(i))
            proxy = pg.SignalProxy(pw.scene().sigMouseMoved, rateLimit=60,
                                   slot=lambda ev, i=idx, viewbox=vb: self._on_mouse_moved(i, viewbox, ev))
            self._mouse_proxies.append(proxy);
            self._reposition_xy_label(idx)

    def _ensure_xy_label(self, idx: int):
        if not hasattr(self, "_xy_labels"): self._xy_labels = [None] * len(self.plots)
        pw = self.plots[idx];
        lbl = None if idx >= len(self._xy_labels) else self._xy_labels[idx]
        if lbl is None or lbl.scene() is None:
            lbl = pg.TextItem("", anchor=(1, 1), color=(40, 40, 40))
            try:
                lbl.setFill(pg.mkBrush(255, 255, 255, 180))
            except Exception:
                pass
            lbl.setFont(QFont("Arial", HUD_PT));
            lbl.setZValue(1e6);
            lbl.setVisible(False);
            pw.addItem(lbl)
            if idx >= len(self._xy_labels): self._xy_labels.extend([None] * (idx + 1 - len(self._xy_labels)))
            self._xy_labels[idx] = lbl
        self._reposition_xy_label(idx)

    def _reposition_xy_label(self, idx: int):
        try:
            pw = self.plots[idx];
            vb = pw.getViewBox();
            (x0, x1), (y0, y1) = vb.viewRange();
            dx = 0.05 * (x1 - x0);
            dy = 0.08 * (y1 - y0);
            self._xy_labels[idx].setPos(x1 - dx, y0 + dy)
        except Exception:
            pass

    def _on_mouse_moved(self, idx: int, vb, ev):
        try:
            pos = ev[0] if isinstance(ev, (tuple, list)) else ev;
            inside = vb.sceneBoundingRect().contains(pos)
            if inside:
                for j, lbl in enumerate(self._xy_labels):
                    if lbl is None: continue
                    lbl.setVisible(j == idx)
                p = vb.mapSceneToView(pos);
                val_x = p.x();
                val_y = p.y()
                if vb.state['logMode'][1]: val_y = 10 ** val_y
                self._xy_labels[idx].setText(f"x = {val_x:.3f},  y = {val_y:.3f}");
                self._reposition_xy_label(idx);
                self._active_idx = idx
            else:
                if self._active_idx == idx and self._xy_labels[idx] is not None: self._xy_labels[idx].setVisible(
                    False); self._active_idx = None
        except Exception:
            pass

    def _force_x_from_zero(self, plot_widget):
        try:
            vb = plot_widget.getViewBox();
            (x0, x1), (_y0, _y1) = vb.viewRange()
            if x1 <= 0:
                plot_widget.setXRange(0, 1, padding=0)
            else:
                plot_widget.setXRange(0, x1, padding=0)
        except Exception:
            pass

    def _autorange_and_fix_x0(self, idx: int):
        try:
            pw = self.plots[idx]
            vb = pw.getViewBox()
            xmax = None
            ymin = None
            ymax = None

            # Only include items that are truly visible on screen.
            # item.isVisible() alone can return True even for items whose
            # opacity is 0 or whose parent is hidden — use item.opts to
            # check the pen alpha as a reliable visibility indicator.
            def _is_truly_visible(item):
                try:
                    if not item.isVisible():
                        return False
                    # Check pen alpha: items hidden via setPen(transparent)
                    # still report isVisible()=True
                    pen = item.opts.get('pen', None)
                    if pen is not None:
                        try:
                            from pyqtgraph import mkPen
                            p = mkPen(pen)
                            if p.color().alpha() == 0:
                                return False
                        except Exception:
                            pass
                    return True
                except Exception:
                    return item.isVisible()

            visible_items = [it for it in pw.listDataItems() if _is_truly_visible(it)]
            if not visible_items:
                vb.autoRange()
                self._force_x_from_zero(pw)
                return

            for it in visible_items:
                try:
                    x, y = it.getData()
                    if x is None or len(x) == 0 or y is None or len(y) == 0:
                        continue
                    finite_mask = np.isfinite(x) & np.isfinite(y)
                    if not np.any(finite_mask):
                        continue
                    x_finite = x[finite_mask]
                    y_finite = y[finite_mask]
                    if len(x_finite) == 0 or len(y_finite) == 0:
                        continue
                    cur_xmax = float(x_finite.max())
                    cur_ymin = float(y_finite.min())
                    cur_ymax = float(y_finite.max())
                    if xmax is None or cur_xmax > xmax: xmax = cur_xmax
                    if ymin is None or cur_ymin < ymin: ymin = cur_ymin
                    if ymax is None or cur_ymax > ymax: ymax = cur_ymax
                except Exception:
                    continue

            if xmax is not None and ymin is not None and ymax is not None:
                padding = (ymax - ymin) * 0.05
                if padding == 0:
                    padding = 1.0
                pw.setYRange(ymin - padding, ymax + padding, padding=0.0)
                pw.setXRange(0.0, xmax, padding=0.0)
            else:
                vb.autoRange()
                self._force_x_from_zero(pw)
        except Exception as e:
            try:
                pw.getViewBox().autoRange()
                self._force_x_from_zero(pw)
            except Exception:
                pass


# # ui/viewer.py
# import os
#
# # [Important] Force pyqtgraph to use PySide6
# os.environ["QT_API"] = "pyside6"
#
# import numpy as np
# import pyqtgraph as pg
#
# # PySide6 Imports
# from PySide6.QtCore import Qt, QPoint, QSize, QSizeF, QMarginsF, QSettings
# from PySide6.QtGui import (
#     QAction, QFont, QImage, QPainter, QColor, QPageSize, QCursor, QPdfWriter
# )
# from PySide6.QtWidgets import (
#     QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
#     QCheckBox, QMenuBar, QWidgetAction,
#     QSizePolicy, QPushButton, QSlider, QLineEdit,
#     QDialog, QFileDialog, QMessageBox, QApplication
# )
#
# # SVG Generator (Check availability)
# try:
#     from PySide6.QtSvg import QSvgGenerator
# except ImportError:
#     QSvgGenerator = None
#
# from .plotter import make_pen, setup_plot, plot_curve, update_visibility
# from model.math_functions import lorch_window
# from .save_menu import SaveMenu
#
# LABEL_PT = 9
# HUD_PT = 9
#
# # [Viewer Style - Light]
# TEAL_STYLE = """
# QWidget {
#     font-size: 9pt;
#     color: #000;
#     background-color: #F0F0F0;
#     font-family: 'Segoe UI', 'Malgun Gothic', sans-serif;
# }
# QLabel {
#     font-weight: normal;
#     color: #222;
# }
# QLineEdit {
#     background-color: #FFFFFF;
#     color: #000;
#     border: 1px solid #A0A0A0;
#     border-radius: 2px;
#     padding: 2px;
# }
# QLineEdit:focus {
#     border: 1px solid #009688;
# }
#
# /* SLIDER STYLE */
# QSlider::groove:horizontal {
#     border: 1px solid #D0D0D0;
#     background: transparent;
#     height: 4px;
#     margin: 0px;
#     border-radius: 2px;
# }
# QSlider::sub-page:horizontal {
#     background: #009688;
#     height: 4px;
#     border-radius: 2px;
# }
# QSlider::add-page:horizontal {
#     background: #E0E0E0;
#     height: 4px;
#     border-radius: 2px;
# }
# QSlider::handle:horizontal {
#     background: #FFFFFF;
#     border: 3px solid #009688;
#     width: 14px;
#     height: 14px;
#     margin: -5px 0;
#     border-radius: 7px;
# }
# QSlider::handle:horizontal:hover {
#     background: #F0F0F0;
#     border-color: #00796B;
# }
#
# /* Checkbox Style */
# QCheckBox {
#     font-weight: normal;
#     color: #000;
# }
# QCheckBox::indicator:checked {
#     background-color: #009688;
#     border: 1px solid #009688;
# }
# """
#
# # [Viewer Style - Dark]
# DARK_STYLE = """
# QWidget {
#     font-size: 9pt;
#     color: #E0E0E0;
#     background-color: #2b2b2b;
#     font-family: 'Segoe UI', 'Malgun Gothic', sans-serif;
# }
# QLabel {
#     font-weight: normal;
#     color: #E0E0E0;
# }
# QLineEdit {
#     background-color: #3b3b3b;
#     color: #E0E0E0;
#     border: 1px solid #555555;
#     border-radius: 2px;
#     padding: 2px;
# }
# QLineEdit:focus {
#     border: 1px solid #009688;
# }
#
# /* SLIDER STYLE */
# QSlider::groove:horizontal {
#     border: 1px solid #555555;
#     background: transparent;
#     height: 4px;
#     margin: 0px;
#     border-radius: 2px;
# }
# QSlider::sub-page:horizontal {
#     background: #009688;
#     height: 4px;
#     border-radius: 2px;
# }
# QSlider::add-page:horizontal {
#     background: #444444;
#     height: 4px;
#     border-radius: 2px;
# }
# QSlider::handle:horizontal {
#     background: #b0b0b0;
#     border: 3px solid #009688;
#     width: 14px;
#     height: 14px;
#     margin: -5px 0;
#     border-radius: 7px;
# }
# QSlider::handle:horizontal:hover {
#     background: #ffffff;
#     border-color: #00796B;
# }
#
# /* Checkbox Style */
# QCheckBox {
#     font-weight: normal;
#     color: #E0E0E0;
# }
# QCheckBox::indicator {
#     border: 1px solid #777;
#     background: #444;
# }
# QCheckBox::indicator:checked {
#     background-color: #009688;
#     border: 1px solid #009688;
# }
# QMenuBar {
#     background-color: #1e1e1e;
#     color: #E0E0E0;
# }
# QMenuBar::item:selected {
#     background-color: #3a3a3a;
# }
# QMenu {
#     background-color: #2b2b2b;
#     color: #E0E0E0;
#     border: 1px solid #555;
# }
# QMenu::item:selected {
#     background-color: #3a3a3a;
# }
# """
#
#
# class PlotWindow(QMainWindow):
#     """
#     Visualize I(q), S(q), F(q), G(r) with PyQtGraph (PySide6 backend).
#     Supports single file analysis and multiple file 'waterfall' plots.
#     """
#
#     def __init__(self):
#         super().__init__()
#         self.left_button_pan_enabled = True
#         pg.setConfigOption('leftButtonPan', True)
#         # Smoother, sharper curves on screen (and higher-quality raster export).
#         pg.setConfigOptions(antialias=True)
#
#         # Default to Light Mode
#         self.setStyleSheet(TEAL_STYLE)
#
#         self.setWindowTitle("EZPDF Plot")
#         self.resize(900, 700)
#
#         # Stores the QTreeWidgetItem(s) associated with this window
#         self.associated_items = None
#
#         # True when the window currently shows a computed Compton curve
#         # (set by calculate_compton). Reset whenever a file is plotted.
#         self.is_compton = False
#
#         self.main_widget = QWidget()
#         self.setCentralWidget(self.main_widget)
#         self.layout = QVBoxLayout(self.main_widget)
#
#         # ---------- TOP ROW ----------
#         FONT_PX = 11
#         INDICATOR_PX = 10
#         self.top_row = QHBoxLayout()
#         self.top_row.setContentsMargins(6, 6, 6, 2)
#         self.top_row.setSpacing(8)
#
#         self.top_hint = QLabel(
#             "Z: Toggle left-click Pan <-> Box-Zoom (left mouse),  R: Auto-range (reset zoom)"
#         )
#         self.top_hint.setAlignment(Qt.AlignmentFlag.AlignLeft)
#         self.top_hint.setWordWrap(False)
#         self.top_hint.setStyleSheet(f"color:#444; font-size:{FONT_PX}px; font-weight:normal; margin-right:8px;")
#         self.top_row.addWidget(self.top_hint, 1)
#
#         self.CB_STYLE_LIGHT = f"""
#         QCheckBox {{
#             font-size: {FONT_PX}px;
#             color: #444;
#             font-weight: bold;
#         }}
#         QCheckBox::indicator {{
#             width: {INDICATOR_PX}px;
#             height: {INDICATOR_PX}px;
#             border-radius: {INDICATOR_PX // 2}px;
#             border: 1px solid #777;
#             background: #fff;
#         }}
#         QCheckBox::indicator:checked {{ background: #009688; border: 1px solid #009688; }}
#         """
#
#         self.CB_STYLE_DARK = f"""
#         QCheckBox {{
#             font-size: {FONT_PX}px;
#             color: #E0E0E0;
#             font-weight: bold;
#         }}
#         QCheckBox::indicator {{
#             width: {INDICATOR_PX}px;
#             height: {INDICATOR_PX}px;
#             border-radius: {INDICATOR_PX // 2}px;
#             border: 1px solid #777;
#             background: #444;
#         }}
#         QCheckBox::indicator:checked {{ background: #009688; border: 1px solid #009688; }}
#         """
#
#         self.plot_checkboxes = []
#         for name in ['I(q)', 'S(q)', 'F(q)', 'G(r)']:
#             cb = QCheckBox(name)
#             cb.setChecked(True)
#             cb.setStyleSheet(self.CB_STYLE_LIGHT + "QCheckBox { margin-left:6px; }")
#             cb.setFixedHeight(INDICATOR_PX + 6)
#             cb.stateChanged.connect(self.update_visibility)
#             self.plot_checkboxes.append(cb)
#
#         self.layout.addLayout(self.top_row)
#
#         self.label = QLabel("No plot loaded.")
#         self.label.setStyleSheet(f"color:#444; font-size:{FONT_PX}px; font-weight:normal; margin-left:6px;")
#         self.label.setWordWrap(False)
#         self.layout.addWidget(self.label)
#
#         self.slider_layout = None
#         self.x_offset_slider = None
#         self.y_offset_slider = None
#         self.x_offset_input = None
#         self.y_offset_input = None
#         self.show_offset_sliders()
#
#         self.plot_container = QWidget()
#         self.plot_layout = QVBoxLayout(self.plot_container)
#         self.plot_layout.setContentsMargins(0, 0, 0, 0)
#         self.plot_layout.setSpacing(0)
#
#         self.plots = []
#         # Per-panel flag: True once the user manually zooms/pans that panel.
#         # Parameter changes then keep that panel's view; unzoomed panels
#         # auto-scale. Reset on fresh load and on R (reset zoom).
#         self._user_zoomed = [False, False, False, False]
#
#         def _fixed_tick_strings(values, scale, spacing):
#             # 2 decimals, right-aligned to a constant width so every panel's
#             # y labels occupy the same space and the plot areas line up.
#             out = []
#             for v in values:
#                 out.append(("%.2f" % (v * scale)).rjust(8))
#             return out
#
#         def _make_iq_ticks(axis):
#             # I(q) can span a huge range. In linear mode use scientific notation
#             # for large/small values (bounded width, never gets cut off) and
#             # plain numbers otherwise. In log mode defer to pyqtgraph's default
#             # (nice 10^n formatting).
#             def _ticks(values, scale, spacing):
#                 if getattr(axis, 'logMode', False):
#                     return pg.AxisItem.tickStrings(axis, values, scale, spacing)
#                 out = []
#                 for v in values:
#                     val = v * scale
#                     av = abs(val)
#                     if av != 0 and (av >= 1e4 or av < 1e-2):
#                         s = "%.1e" % val
#                     else:
#                         s = "%.2f" % val
#                     out.append(s.rjust(8))
#                 return out
#             return _ticks
#
#         for _ in range(4):
#             pw = pg.PlotWidget()
#             pw.setBackground('w')
#             pw.showGrid(x=False, y=False, alpha=0.3)
#
#             vb = pw.getViewBox()
#             vb.setLimits(xMin=0)
#             vb.setDefaultPadding(0.0)
#
#             idx = len(self.plots)
#
#             # Left (y) axis: constant width + fixed-width labels so all panels'
#             # plot areas start at the same x (same q position lines up). No ×10ⁿ
#             # SI prefix; I(q) uses scientific notation for large linear values.
#             try:
#                 left_axis = pw.getAxis('left')
#                 left_axis.setWidth(72)
#                 left_axis.enableAutoSIPrefix(False)
#                 if idx == 0:
#                     left_axis.tickStrings = _make_iq_ticks(left_axis)
#                 else:
#                     left_axis.tickStrings = _fixed_tick_strings
#             except Exception:
#                 pass
#
#             try:
#                 vb.sigRangeChangedManually.connect(
#                     lambda _mask=None, i=idx: self._mark_user_zoomed(i))
#             except Exception:
#                 pass
#
#             self.plot_layout.addWidget(pw, 1)
#             self.plots.append(pw)
#
#         self.layout.addWidget(self.plot_container, 1)
#
#         self._install_xy_readouts()
#         self.init_bottom_menu()
#
#         # Data placeholders
#         self.file_name = None
#         self.bkg_curve = None
#         self.raw_curve = None
#         self.sq_curve_original = None
#         self.sq_curve_polynomial = None
#         self.mean_sq_fi_curve = None
#         self.sq_mean_fi_curve = None
#         self.fq_smoothed_curve = None
#         self.gr_smoothed_curve = None
#         self.gr_lorch_curve = None
#         self.gr_smoothed_lorch_curve = None
#         self.fq_lorch_curve = None
#         self.fq_smoothed_lorch_curve = None
#         self.fq_main_curve = None
#         self.gr_main_curve = None
#
#         self.xs = None
#         self.ys = None
#         self.raw_iq = None
#         self.background_data = None
#         self.sq_original_data = None
#         self.sq_polynomial_data = None
#         self.fq_smoothed_data = None
#         self.gr_smoothed_data = None
#         self.gr_lorch_data = None
#         self.gr_smoothed_lorch_data = None
#         self.mean_sq_fi = None
#         self.sq_mean_fi = None
#
#         # Multi-plot cache
#         self.cached_xs = None
#         self.cached_ys = None
#         self.cached_titles = None
#
#         # Check Global Theme
#         app_instance = QApplication.instance()
#         if hasattr(app_instance, 'is_dark_mode') and app_instance.is_dark_mode:
#             self.apply_theme(True)
#
#     def apply_theme(self, is_dark):
#         """
#         Updates the style of the PlotWindow for Light/Dark mode.
#         """
#         FONT_PX = 11
#
#         if is_dark:
#             self.setStyleSheet(DARK_STYLE)
#             self.top_hint.setStyleSheet(f"color:#BBB; font-size:{FONT_PX}px; font-weight:normal; margin-right:8px;")
#             self.label.setStyleSheet(f"color:#BBB; font-size:{FONT_PX}px; font-weight:normal; margin-left:6px;")
#             self.bottom_menu.setStyleSheet("margin-top: 6px; background-color:#1e1e1e; color:#E0E0E0;")
#
#             for cb in self.plot_checkboxes:
#                 cb.setStyleSheet(self.CB_STYLE_DARK + "QCheckBox { margin-left:6px; }")
#
#             for pw in self.plots:
#                 pw.setBackground('#1e1e1e')  # Dark background
#                 plot_item = pw.getPlotItem()
#                 for axis in ['left', 'bottom']:
#                     ax = plot_item.getAxis(axis)
#                     ax.setPen('#E0E0E0')
#                     ax.setTextPen('#E0E0E0')
#
#             if self.slider_bar_widget:
#                 self.slider_bar_widget.setStyleSheet("margin:0; padding:0; background-color: #2b2b2b;")
#
#             # [추가] 다크 모드 시 곡선 및 토글 색상 개선 (흰색으로 통일)
#             pen_white = make_pen((255, 255, 255))
#             if self.raw_curve: self.raw_curve.setPen(pen_white)
#             if self.sq_curve_original: self.sq_curve_original.setPen(pen_white)
#             if self.fq_smoothed_curve: self.fq_smoothed_curve.setPen(pen_white)
#             if self.gr_smoothed_curve: self.gr_smoothed_curve.setPen(pen_white)
#
#             base_style = "QCheckBox::indicator { width: 12px; height: 12px; border-radius: 6px; border: 1px solid #777; background: #fff; } "
#             white_checked = "QCheckBox::indicator:checked { background: #FFFFFF; border:1px solid #FFFFFF; }"
#
#             if hasattr(self, 'original_checkbox'):
#                 self.original_checkbox.setStyleSheet(base_style + white_checked)
#             if hasattr(self, 'original_sq_checkbox'):
#                 self.original_sq_checkbox.setStyleSheet(base_style + white_checked)
#             if hasattr(self, 'smoothed_fq_checkbox'):
#                 self.smoothed_fq_checkbox.setStyleSheet(base_style + white_checked)
#
#         else:
#             self.setStyleSheet(TEAL_STYLE)
#             self.top_hint.setStyleSheet(f"color:#444; font-size:{FONT_PX}px; font-weight:normal; margin-right:8px;")
#             self.label.setStyleSheet(f"color:#444; font-size:{FONT_PX}px; font-weight:normal; margin-left:6px;")
#             self.bottom_menu.setStyleSheet("margin-top: 6px; background-color:#F0F0F0; color:#000;")
#
#             for cb in self.plot_checkboxes:
#                 cb.setStyleSheet(self.CB_STYLE_LIGHT + "QCheckBox { margin-left:6px; }")
#
#             for pw in self.plots:
#                 pw.setBackground('w')
#                 plot_item = pw.getPlotItem()
#                 for axis in ['left', 'bottom']:
#                     ax = plot_item.getAxis(axis)
#                     ax.setPen('k')
#                     ax.setTextPen('k')
#
#             if self.slider_bar_widget:
#                 self.slider_bar_widget.setStyleSheet("margin:0; padding:0; background-color: #F0F0F0;")
#
#             # [추가] 라이트 모드 시 곡선 및 토글 색상 복구 (검은색으로 통일)
#             pen_black = make_pen((0, 0, 0))
#             if self.raw_curve: self.raw_curve.setPen(pen_black)
#             if self.sq_curve_original: self.sq_curve_original.setPen(pen_black)
#             if self.fq_smoothed_curve: self.fq_smoothed_curve.setPen(pen_black)
#             if self.gr_smoothed_curve: self.gr_smoothed_curve.setPen(pen_black)
#
#             base_style = "QCheckBox::indicator { width: 12px; height: 12px; border-radius: 6px; border: 1px solid #777; background: #fff; } "
#             black_checked = "QCheckBox::indicator:checked { background: #000000; border:1px solid #000000; }"
#
#             if hasattr(self, 'original_checkbox'):
#                 self.original_checkbox.setStyleSheet(base_style + black_checked)
#             if hasattr(self, 'original_sq_checkbox'):
#                 self.original_sq_checkbox.setStyleSheet(base_style + black_checked)
#             if hasattr(self, 'smoothed_fq_checkbox'):
#                 self.smoothed_fq_checkbox.setStyleSheet(base_style + black_checked)
#
#     def init_bottom_menu(self):
#         self.bottom_menu = QMenuBar()
#         self.bottom_menu.setStyleSheet("margin-top: 6px; background-color:#F0F0F0;")
#
#         save_data_action = QAction("Save data", self)
#         save_data_action.triggered.connect(self._save_data_dialog)
#         self.bottom_menu.addAction(save_data_action)
#
#         save_figure_action = QAction("Save figure", self)
#         save_figure_action.triggered.connect(self.save_combined_graph_image)
#         self.bottom_menu.addAction(save_figure_action)
#
#         self.bottom_menu.addMenu("Option:")
#
#         # I(q) Menu
#         iq_menu = self.bottom_menu.addMenu("I(q)")
#         iq_widget = QWidget()
#         iq_layout = QVBoxLayout()
#         iq_layout.setContentsMargins(10, 10, 10, 10)
#         iq_widget.setLayout(iq_layout)
#
#         self.log_checkbox = QCheckBox("Log Y Scale")
#         self.log_checkbox.setChecked(False)
#         self.log_checkbox.stateChanged.connect(self.toggle_log_scale)
#
#         self.background_checkbox = QCheckBox("Background")
#         self.background_checkbox.setChecked(False)
#         self.background_checkbox.stateChanged.connect(self.show_background_data)
#
#         self.original_checkbox = QCheckBox("Original I(q) Curve")
#         self.original_checkbox.setChecked(False)
#         self.original_checkbox.stateChanged.connect(self.show_original_data)
#
#         for checkbox in [self.log_checkbox, self.background_checkbox, self.original_checkbox]:
#             iq_layout.addWidget(checkbox)
#
#         iq_widget_action = QWidgetAction(self)
#         iq_widget_action.setDefaultWidget(iq_widget)
#         iq_menu.addAction(iq_widget_action)
#
#         # S(q) Menu
#         sq_menu = self.bottom_menu.addMenu("S(q)")
#         sq_widget = QWidget()
#         sq_layout = QVBoxLayout()
#         sq_layout.setContentsMargins(10, 10, 10, 10)
#         sq_widget.setLayout(sq_layout)
#
#         self.original_sq_checkbox = QCheckBox("Original S(q)")
#         self.original_sq_checkbox.setChecked(False)
#         self.original_sq_checkbox.stateChanged.connect(self.toggle_original_sq)
#         sq_layout.addWidget(self.original_sq_checkbox)
#
#         self.polynomial_sq_checkbox = QCheckBox("Polynomial for S(q)")
#         self.polynomial_sq_checkbox.setChecked(False)
#         self.polynomial_sq_checkbox.stateChanged.connect(self.toggle_polynomial_sq)
#         sq_layout.addWidget(self.polynomial_sq_checkbox)
#
#         self.mean_sq_fi_checkbox = QCheckBox("<f^2>")
#         self.mean_sq_fi_checkbox.setChecked(False)
#         self.mean_sq_fi_checkbox.stateChanged.connect(self.toggle_mean_sq_fi)
#         sq_layout.addWidget(self.mean_sq_fi_checkbox)
#
#         self.sq_mean_fi_checkbox = QCheckBox("<f>^2")
#         self.sq_mean_fi_checkbox.setChecked(False)
#         self.sq_mean_fi_checkbox.stateChanged.connect(self.toggle_sq_mean_fi)
#         sq_layout.addWidget(self.sq_mean_fi_checkbox)
#
#         sq_widget_action = QWidgetAction(self)
#         sq_widget_action.setDefaultWidget(sq_widget)
#         sq_menu.addAction(sq_widget_action)
#
#         # F(q) Menu
#         fq_menu = self.bottom_menu.addMenu("F(q)")
#         fq_widget = QWidget()
#         fq_layout = QVBoxLayout()
#         fq_layout.setContentsMargins(10, 10, 10, 10)
#         fq_widget.setLayout(fq_layout)
#
#         self.smoothed_fq_checkbox = QCheckBox("Smoothed F(q)")
#         self.smoothed_fq_checkbox.setChecked(False)
#         self.smoothed_fq_checkbox.stateChanged.connect(self.toggle_smoothed_fq)
#         fq_layout.addWidget(self.smoothed_fq_checkbox)
#
#         fq_widget_action = QWidgetAction(self)
#         fq_widget_action.setDefaultWidget(fq_widget)
#         fq_menu.addAction(fq_widget_action)
#
#         # G(r) Menu -- Lorch-modified variants (experimental data only).
#         # These overlay curves are precomputed; toggling only shows/hides.
#         gr_menu = self.bottom_menu.addMenu("G(r)")
#         gr_widget = QWidget()
#         gr_layout = QVBoxLayout()
#         gr_layout.setContentsMargins(10, 10, 10, 10)
#         gr_widget.setLayout(gr_layout)
#
#         # Lorch controls apply to BOTH F(q) and G(r) panels. The "Smoothed"
#         # axis reuses the existing "Smoothed F(q)" checkbox in the F(q) menu.
#         _lorch_cb_style = (
#             "QCheckBox::indicator { width: 12px; height: 12px; border-radius: 6px; "
#             "border: 1px solid #777; background: #fff; } "
#             "QCheckBox::indicator:checked { background: #FF8C00; border: 1px solid #FF8C00; }")
#         self.apply_lorch_checkbox = QCheckBox("Apply Lorch")
#         self.apply_lorch_checkbox.setStyleSheet(_lorch_cb_style)
#         self.apply_lorch_checkbox.setChecked(False)
#         self.apply_lorch_checkbox.stateChanged.connect(lambda: self._on_lorch_toggle(self.apply_lorch_checkbox))
#         gr_layout.addWidget(self.apply_lorch_checkbox)
#
#         self.compare_lorch_checkbox = QCheckBox("Compare (Lorch on/off)")
#         self.compare_lorch_checkbox.setStyleSheet(_lorch_cb_style)
#         self.compare_lorch_checkbox.setChecked(False)
#         self.compare_lorch_checkbox.stateChanged.connect(lambda: self._on_lorch_toggle(self.compare_lorch_checkbox))
#         gr_layout.addWidget(self.compare_lorch_checkbox)
#
#         self.compare_all_checkbox = QCheckBox("Compare all")
#         self.compare_all_checkbox.setStyleSheet(_lorch_cb_style)
#         self.compare_all_checkbox.setChecked(False)
#         self.compare_all_checkbox.stateChanged.connect(lambda: self._on_lorch_toggle(self.compare_all_checkbox))
#         gr_layout.addWidget(self.compare_all_checkbox)
#
#         gr_widget_action = QWidgetAction(self)
#         gr_widget_action.setDefaultWidget(gr_widget)
#         gr_menu.addAction(gr_widget_action)
#
#         # Indicator Colors
#         base_indicator_style = """
#             QCheckBox::indicator {
#                 width: 12px; height: 12px; border-radius: 6px;
#                 border: 1px solid #777; background: #fff;
#             }
#         """
#         self.background_checkbox.setStyleSheet(
#             base_indicator_style + "QCheckBox::indicator:checked { background: #FF0000; border:1px solid #FF0000; }")
#         self.original_checkbox.setStyleSheet(
#             base_indicator_style + "QCheckBox::indicator:checked { background: #000000; border:1px solid #000000; }")
#         self.original_sq_checkbox.setStyleSheet(
#             base_indicator_style + "QCheckBox::indicator:checked { background: #000000; border:1px solid #000000; }")
#         self.polynomial_sq_checkbox.setStyleSheet(
#             base_indicator_style + "QCheckBox::indicator:checked { background: #FF0000; border:1px solid #FF0000; }")
#         self.mean_sq_fi_checkbox.setStyleSheet(
#             base_indicator_style + "QCheckBox::indicator:checked { background: #009600; border:1px solid #009600; }")
#         self.sq_mean_fi_checkbox.setStyleSheet(
#             base_indicator_style + "QCheckBox::indicator:checked { background: #800080; border:1px solid #800080; }")
#
#         default_checked_style = base_indicator_style + "QCheckBox::indicator:checked { background: #FFA500; border:1px solid #FFA500; }"
#         self.log_checkbox.setStyleSheet(default_checked_style)
#         self.smoothed_fq_checkbox.setStyleSheet(
#             base_indicator_style + "QCheckBox::indicator:checked { background: #000000; border:1px solid #000000; }")
#
#         self.bottom_container_widget = QWidget()
#         bottom_container_layout = QHBoxLayout(self.bottom_container_widget)
#         bottom_container_layout.setContentsMargins(4, 0, 4, 0)
#         bottom_container_layout.setSpacing(6)
#
#         bottom_container_layout.addWidget(self.bottom_menu)
#         bottom_container_layout.addStretch(1)
#
#         self.lock_graph_checkbox = QCheckBox("Lock graph")
#         self.lock_graph_checkbox.setStyleSheet("""
#             QCheckBox { font-weight: bold; color: #B22222; }
#             QCheckBox::indicator { width: 12px; height: 12px; border-radius: 3px; border: 1px solid #B22222; }
#             QCheckBox::indicator:checked { background: #B22222; }
#         """)
#         self.lock_graph_checkbox.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
#         bottom_container_layout.addWidget(self.lock_graph_checkbox)
#
#         if hasattr(self, 'plot_checkboxes'):
#             for cb in self.plot_checkboxes:
#                 bottom_container_layout.addWidget(cb)
#
#         self.layout.addWidget(self.bottom_container_widget)
#
#     def is_locked(self):
#         return self.lock_graph_checkbox.isChecked()
#
#     def _reset_intermediates(self):
#         self.bkg_curve = None
#         self.raw_curve = None
#         self.sq_curve_original = None
#         self.sq_curve_polynomial = None
#         self.mean_sq_fi_curve = None
#         self.sq_mean_fi_curve = None
#         self.fq_smoothed_curve = None
#         self.gr_smoothed_curve = None
#         self.gr_lorch_curve = None
#         self.gr_smoothed_lorch_curve = None
#         self.fq_lorch_curve = None
#         self.fq_smoothed_lorch_curve = None
#         self.fq_main_curve = None
#         self.gr_main_curve = None
#
#         self.raw_iq = None
#         self.background_data = None
#         self.sq_original_data = None
#         self.sq_polynomial_data = None
#         self.fq_smoothed_data = None
#         self.mean_sq_fi = None
#         self.sq_mean_fi = None
#         self.gr_smoothed_data = None
#         self.gr_lorch_data = None
#         self.gr_smoothed_lorch_data = None
#         self.fq_lorch_data = None
#         self.fq_smoothed_lorch_data = None
#
#     def _align_pair(self, x, y):
#         try:
#             if x is None or y is None:
#                 return x, y
#             n = min(len(x), len(y))
#             return x[:n], y[:n]
#         except Exception:
#             return x, y
#
#     def _align_with_x(self, x, y):
#         try:
#             if x is None or y is None:
#                 return y
#             n = min(len(x), len(y))
#             return y[:n]
#         except Exception:
#             return y
#
#     def plot_data(self, xs, ys, bkg_x, bkg_y, raw_x, raw_y,
#                   list_Sq=None, Fq_smoothed=None, mean_sq_fi=None, sq_mean_fi=None,
#                   r_smoothed=None, G_smoothed=None, title="", bring_front=True,
#                   gr_lorch=None, gr_smoothed_lorch=None, **_kwargs):
#         self._reset_intermediates()
#         self.is_compton = False
#
#         self.cached_xs = self.cached_ys = self.cached_titles = None
#
#         raw_x, raw_y = self._align_pair(raw_x, raw_y)
#         bkg_x, bkg_y = self._align_pair(bkg_x, bkg_y)
#         if xs is not None and ys is not None:
#             xs = list(xs)
#             ys = list(ys)
#             for i in range(min(len(xs), len(ys))):
#                 xs[i], ys[i] = self._align_pair(xs[i], ys[i])
#         if xs is not None and len(xs) > 1:
#             x_sq = xs[1]
#             if list_Sq is not None:  list_Sq = self._align_with_x(x_sq, list_Sq)
#             if mean_sq_fi is not None:  mean_sq_fi = self._align_with_x(x_sq, mean_sq_fi)
#             if sq_mean_fi is not None:  sq_mean_fi = self._align_with_x(x_sq, sq_mean_fi)
#         if xs is not None and len(xs) > 2 and Fq_smoothed is not None:
#             Fq_smoothed = self._align_with_x(xs[2], Fq_smoothed)
#         r_smoothed, G_smoothed = self._align_pair(r_smoothed, G_smoothed)
#         _r_main = xs[3] if (xs is not None and len(xs) > 3) else None
#         _r_lorch, gr_lorch = self._align_pair(_r_main, gr_lorch)
#         _r_sm_lorch, gr_smoothed_lorch = self._align_pair(r_smoothed, gr_smoothed_lorch)
#
#         # Only bring the window to the front when a new file is opened here.
#         # During a parameter change every open plot window is replotted; raising
#         # them all would make the windows jump back and forth, so replots pass
#         # bring_front=False and stay wherever the user put them.
#         if bring_front:
#             self.bring_to_front()
#         self.label.setText(f"Displaying: {title}")
#         self.clear_slider_layout()
#
#         self.file_name = title
#         self.xs = xs
#         self.ys = ys
#         self.raw_iq = [raw_x, raw_y]
#         self.background_data = [bkg_x, bkg_y]
#         self.sq_original_data = list_Sq
#         self.sq_polynomial_data = (
#             (list_Sq - ys[1])
#             if (list_Sq is not None and ys is not None and len(ys) > 1 and len(list_Sq) == len(ys[1]))
#             else None
#         )
#         self.fq_smoothed_data = Fq_smoothed
#         self.mean_sq_fi = mean_sq_fi
#         self.sq_mean_fi = sq_mean_fi
#         self.gr_smoothed_data = [r_smoothed, G_smoothed]
#         self.gr_lorch_data = [_r_lorch, gr_lorch]
#         self.gr_smoothed_lorch_data = [_r_sm_lorch, gr_smoothed_lorch]
#
#         x_labels = ['q(1/Å)', 'q(1/Å)', 'q(1/Å)', 'r(Å)']
#         y_labels = ['I(q)', 'S(q)', 'F(q)', 'G(Å⁻²)']
#
#         pen_main = make_pen((0, 0, 255))
#         pen_bkg = make_pen((255, 0, 0))
#         pen_raw = make_pen((0, 0, 0))
#         pen_f2 = make_pen((0, 150, 0))
#         pen_f_avg2 = make_pen((128, 0, 128))
#         pen_lorch = make_pen((255, 140, 0))      # raw + Lorch (orange)
#         pen_lorch2 = make_pen((0, 150, 136))     # smoothed + Lorch (teal)
#
#         is_dark = False
#         app_instance = QApplication.instance()
#         if hasattr(app_instance, 'is_dark_mode') and app_instance.is_dark_mode:
#             is_dark = True
#             pen_raw = make_pen((255, 255, 255))  # White for raw data in dark mode
#
#         for i in range(4):
#             setup_plot(self.plots[i], x_labels[i], y_labels[i])
#             if hasattr(self, '_xy_labels') and len(self._xy_labels) > i and self._xy_labels[i] is not None:
#                 self.plots[i].addItem(self._xy_labels[i])
#                 self._reposition_xy_label(i)
#
#             pi = self.plots[i].getPlotItem()
#             pi.setLabel('bottom', f'<span style="font-size:{LABEL_PT}pt;">{x_labels[i]}</span>')
#             pi.setLabel('left', f'<span style="font-size:{LABEL_PT}pt;">{y_labels[i]}</span>')
#
#             if is_dark:
#                 for axis in ['left', 'bottom']:
#                     pi.getAxis(axis).setPen('#E0E0E0')
#                     pi.getAxis(axis).setTextPen('#E0E0E0')
#
#             _main_curve = self.plots[i].plot(xs[i], ys[i], pen=pen_main)
#             if i == 2:
#                 self.fq_main_curve = _main_curve
#             elif i == 3:
#                 self.gr_main_curve = _main_curve
#
#             if i == 0:
#                 self.plots[i].setLogMode(y=self.log_checkbox.isChecked())
#                 self.bkg_curve = plot_curve(self.plots[i], bkg_x, bkg_y, pen_bkg,
#                                             visible=self.background_checkbox.isChecked())
#                 self.raw_curve = plot_curve(self.plots[i], raw_x, raw_y, pen_raw,
#                                             visible=self.original_checkbox.isChecked())
#             if i == 1:
#                 self.sq_curve_original = plot_curve(self.plots[i], xs[1], list_Sq, pen_raw,
#                                                     visible=self.original_sq_checkbox.isChecked())
#                 self.sq_curve_polynomial = plot_curve(self.plots[i], xs[1], self.sq_polynomial_data, pen_bkg,
#                                                       visible=self.polynomial_sq_checkbox.isChecked())
#                 self.mean_sq_fi_curve = plot_curve(self.plots[i], xs[1], self.mean_sq_fi, pen_f2,
#                                                    visible=self.mean_sq_fi_checkbox.isChecked())
#                 self.sq_mean_fi_curve = plot_curve(self.plots[i], xs[1], self.sq_mean_fi, pen_f_avg2,
#                                                    visible=self.sq_mean_fi_checkbox.isChecked())
#             if i == 2:
#                 self.fq_smoothed_curve = plot_curve(self.plots[i], xs[2], Fq_smoothed, pen_raw,
#                                                     visible=False)
#                 # F(q).Lorch = F(q) * M(q); derived here (no FFT needed).
#                 _qf = np.asarray(xs[2], dtype=float) if (xs[2] is not None and len(xs[2]) > 0) else None
#                 _Mf = lorch_window(_qf, float(_qf[-1])) if _qf is not None else None
#                 _fq_l = (np.asarray(ys[2], dtype=float) * _Mf) if (_Mf is not None and ys[2] is not None and len(ys[2]) == len(_Mf)) else None
#                 _fq_sm_l = (np.asarray(Fq_smoothed, dtype=float) * _Mf) if (_Mf is not None and Fq_smoothed is not None and len(Fq_smoothed) == len(_Mf)) else None
#                 self.fq_lorch_data = [xs[2], _fq_l]
#                 self.fq_smoothed_lorch_data = [xs[2], _fq_sm_l]
#                 self.fq_lorch_curve = plot_curve(self.plots[i], xs[2], _fq_l, pen_lorch, visible=False)
#                 self.fq_smoothed_lorch_curve = plot_curve(self.plots[i], xs[2], _fq_sm_l, pen_lorch2, visible=False)
#             if i == 3:
#                 self.gr_smoothed_curve = plot_curve(self.plots[i], r_smoothed, G_smoothed, pen_raw, visible=False)
#                 self.gr_lorch_curve = plot_curve(self.plots[i], _r_lorch, gr_lorch, pen_lorch, visible=False)
#                 self.gr_smoothed_lorch_curve = plot_curve(self.plots[i], _r_sm_lorch, gr_smoothed_lorch, pen_lorch2, visible=False)
#
#         self.update_visibility()
#         self._apply_lorch_view()
#         for i in range(4):
#             self._autorange_and_fix_x0(i)
#
#     def plot_multiple(self, list_of_xs, list_of_ys, titles=None, bring_front=True):
#         self._reset_intermediates()
#         self.is_compton = False
#
#         if bring_front:
#             self.bring_to_front()
#         self.cached_xs = list_of_xs
#         self.cached_ys = list_of_ys
#         self.cached_titles = titles
#
#         self.xs = None
#         self.ys = None
#
#         if self.slider_layout is None:
#             self.show_offset_sliders()
#
#         num_plots = 4
#         offset_x = self.get_x_offset()
#         offset_y = self.get_y_offset()
#
#         x_labels = ['q(1/Å)', 'q(1/Å)', 'q(1/Å)', 'r(Å)']
#         y_labels = ['I(q)', 'S(q)', 'F(1/Å)', 'G(Å⁻²)']
#         colors = [(0, 0, 255), (255, 0, 0), (0, 150, 0), (255, 165, 0),
#                   (128, 0, 128), (0, 255, 255), (255, 105, 180), (128, 128, 0)]
#
#         is_dark = False
#         app_instance = QApplication.instance()
#         if hasattr(app_instance, 'is_dark_mode') and app_instance.is_dark_mode:
#             is_dark = True
#
#         def shorten_name(name, max_len=80):
#             if not name: return ""
#             if len(name) <= max_len: return name
#             return f"{name[:30]}...{name[-20:]}"
#
#         for i in range(num_plots):
#             plot_item = self.plots[i].getPlotItem()
#
#             # 1. Legend Handling: Keep existing legend object to preserve manual position
#             if plot_item.legend is None:
#                 legend = self.plots[i].addLegend(offset=(-5, 5))
#             else:
#                 legend = plot_item.legend
#                 legend.clear()  # Only clear the labels/items inside, not the position
#
#             # 2. Clear Data Curves ONLY (do not clear the whole plot to save legend/readout)
#             for item in self.plots[i].listDataItems():
#                 self.plots[i].removeItem(item)
#
#             # 3. Setup Plot Appearance (Grid and Labels)
#             self.plots[i].showGrid(x=False, y=False, alpha=0.3)
#
#             # Legend Style
#             legend.setBrush(pg.mkBrush(255, 255, 255, 0))
#             if is_dark:
#                 legend.setLabelTextColor(pg.mkColor(255, 255, 255))
#             else:
#                 legend.setLabelTextColor(pg.mkColor(0, 0, 0))
#
#             legend.setFont(QFont("Segoe UI", 6))
#             try:
#                 legend.layout.setColumnFixedWidth(1, 600)
#                 legend.layout.setColumnSpacing(0, 20)
#                 legend.layout.setVerticalSpacing(0)
#                 legend.layout.setContentsMargins(0, 0, 0, 0)
#             except Exception:
#                 pass
#
#             # 4. Ensure XY Readout is present
#             if hasattr(self, '_xy_labels') and len(self._xy_labels) > i and self._xy_labels[i] is not None:
#                 if self._xy_labels[i] not in self.plots[i].items():
#                     self.plots[i].addItem(self._xy_labels[i])
#                 self._reposition_xy_label(i)
#
#             pi = self.plots[i].getPlotItem()
#             pi.setLabel('bottom', f'<span style="font-size:{LABEL_PT}pt;">{x_labels[i]}</span>')
#             pi.setLabel('left', f'<span style="font-size:{LABEL_PT}pt;">{y_labels[i]}</span>')
#
#             if is_dark:
#                 for axis in ['left', 'bottom']:
#                     pi.getAxis(axis).setPen('#E0E0E0')
#                     pi.getAxis(axis).setTextPen('#E0E0E0')
#
#         valid = 0
#         for file_idx, (xs, ys) in enumerate(zip(list_of_xs, list_of_ys)):
#             try:
#                 # ---------------- [버그 수정 구간 유지] ----------------
#                 if not xs or not ys or len(xs) != len(ys):
#                     continue
#                 # -----------------------------------------------------
#                 color = colors[file_idx % len(colors)]
#                 pen = make_pen(color)
#
#                 raw_name = titles[file_idx] if titles and file_idx < len(titles) else None
#                 legend_name = shorten_name(raw_name)
#
#                 for i in range(min(len(xs), len(ys), num_plots)):
#                     if len(xs[i]) == 0 or len(ys[i]) == 0:
#                         continue
#                     x = xs[i] + valid * offset_x
#                     y = ys[i] + valid * offset_y
#                     self.plots[i].plot(x, y, pen=pen, name=legend_name)
#                 valid += 1
#             except Exception:
#                 continue
#
#         self.label.setText(f"Displaying: {valid} file(s).")
#         self.update_visibility()
#         for i in range(4):
#             self._autorange_and_fix_x0(i)
#
#     def show_offset_sliders(self):
#         if self.slider_layout is None:
#             self.slider_bar_widget = QWidget(self)
#             self.slider_bar_widget.setContentsMargins(0, 0, 0, 0)
#             self.slider_bar_widget.setStyleSheet("margin:0; padding:0; background-color: #F0F0F0;")
#             row = QHBoxLayout(self.slider_bar_widget)
#             row.setContentsMargins(0, 0, 0, 0)
#             row.setSpacing(10)
#             row.setAlignment(Qt.AlignmentFlag.AlignLeft)
#             xlab = QLabel("X Offset")
#             xlab.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
#             self.x_offset_slider = QSlider(Qt.Orientation.Horizontal)
#             self.x_offset_slider.setRange(0, int(10.0 * 100))
#             self.x_offset_slider.setSingleStep(1)
#             self.x_offset_slider.setValue(0)
#             self.x_offset_slider.setMaximumHeight(16)
#             self.x_offset_slider.setMinimumWidth(120)
#             self.x_offset_slider.setMaximumWidth(200)
#             self.x_offset_input = QLineEdit("0.00")
#             self.x_offset_input.setFixedWidth(50)
#             self.x_offset_input.setAlignment(Qt.AlignmentFlag.AlignRight)
#             xgrp = QHBoxLayout()
#             xgrp.setContentsMargins(0, 0, 0, 0)
#             xgrp.setSpacing(4)
#             xgrp.addWidget(xlab)
#             xgrp.addWidget(self.x_offset_slider)
#             xgrp.addWidget(self.x_offset_input)
#             xwrap = QWidget(self.slider_bar_widget)
#             xwrap.setContentsMargins(0, 0, 0, 0)
#             xwrap.setLayout(xgrp)
#             xwrap.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
#             ylab = QLabel("Y Offset")
#             ylab.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
#             self.y_offset_slider = QSlider(Qt.Orientation.Horizontal)
#             self.y_offset_slider.setRange(0, int(10.0 * 100))
#             self.y_offset_slider.setSingleStep(1)
#             self.y_offset_slider.setValue(int(10.0 * 100))
#             self.y_offset_slider.setMaximumHeight(16)
#             self.y_offset_slider.setMinimumWidth(120)
#             self.y_offset_slider.setMaximumWidth(200)
#             self.y_offset_input = QLineEdit("10.00")
#             self.y_offset_input.setFixedWidth(50)
#             self.y_offset_input.setAlignment(Qt.AlignmentFlag.AlignRight)
#             minus_btn = QPushButton("−")
#             plus_btn = QPushButton("+")
#             for b in (minus_btn, plus_btn):
#                 b.setFixedWidth(22)
#                 b.setStyleSheet("margin:0; padding:0;")
#
#             def update_y_max(multiplier):
#                 new_max = max(100, int(self.y_offset_slider.maximum() * multiplier))
#                 self.y_offset_slider.setMaximum(new_max)
#                 if self.y_offset_slider.value() > new_max:
#                     self.y_offset_slider.setValue(new_max)
#
#             minus_btn.clicked.connect(lambda: update_y_max(0.5))
#             plus_btn.clicked.connect(lambda: update_y_max(2.0))
#             ygrp = QHBoxLayout()
#             ygrp.setContentsMargins(0, 0, 0, 0)
#             ygrp.setSpacing(4)
#             ygrp.addWidget(ylab)
#             ygrp.addWidget(self.y_offset_slider)
#             ygrp.addWidget(self.y_offset_input)
#             ygrp.addWidget(minus_btn)
#             ygrp.addWidget(plus_btn)
#             ywrap = QWidget(self.slider_bar_widget)
#             ywrap.setContentsMargins(0, 0, 0, 0)
#             ywrap.setLayout(ygrp)
#             ywrap.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
#             row.addWidget(xwrap)
#             row.addWidget(ywrap)
#             row.addStretch(1)
#
#             def _s2txt(slider, edit):
#                 edit.setText(f"{slider.value() / 100.0:.2f}")
#
#             def _txt2s(edit, slider):
#                 try:
#                     val = float(edit.text())
#                     slider.setValue(int(round(val * 100)))
#                 except Exception:
#                     pass
#
#             self.x_offset_slider.valueChanged.connect(
#                 lambda _: (_s2txt(self.x_offset_slider, self.x_offset_input),
#                            self.update_offsets_and_replot()))
#             self.y_offset_slider.valueChanged.connect(
#                 lambda _: (_s2txt(self.y_offset_slider, self.y_offset_input),
#                            self.update_offsets_and_replot()))
#             self.x_offset_input.editingFinished.connect(
#                 lambda: (_txt2s(self.x_offset_input, self.x_offset_slider),
#                          self.update_offsets_and_replot()))
#             self.y_offset_input.editingFinished.connect(
#                 lambda: (_txt2s(self.y_offset_input, self.y_offset_slider),
#                          self.update_offsets_and_replot()))
#             self.slider_layout = row
#             self.layout.insertWidget(2, self.slider_bar_widget)
#             self._resize_offset_controls()
#             app_instance = QApplication.instance()
#             if hasattr(app_instance, 'is_dark_mode') and app_instance.is_dark_mode:
#                 self.slider_bar_widget.setStyleSheet("margin:0; padding:0; background-color: #2b2b2b;")
#
#     def _resize_offset_controls(self):
#         maxw = max(150, self.width() // 5)
#         if self.x_offset_slider:
#             self.x_offset_slider.setMaximumWidth(min(self.x_offset_slider.maximumWidth(), maxw))
#         if self.y_offset_slider:
#             self.y_offset_slider.setMaximumWidth(min(self.y_offset_slider.maximumWidth(), maxw))
#
#     def clear_slider_layout(self):
#         if self.slider_layout:
#             for sig in (getattr(self, "x_offset_slider", None),
#                         getattr(self, "y_offset_slider", None)):
#                 if sig:
#                     try:
#                         sig.valueChanged.disconnect()
#                     except Exception:
#                         pass
#             for edit in (getattr(self, "x_offset_input", None),
#                          getattr(self, "y_offset_input", None)):
#                 if edit:
#                     try:
#                         edit.editingFinished.disconnect()
#                     except Exception:
#                         pass
#             parent_widget = self.slider_layout.parentWidget()
#             while self.slider_layout.count():
#                 item = self.slider_layout.takeAt(0)
#                 widget = item.widget()
#                 if widget:
#                     widget.setParent(None)
#                     widget.deleteLater()
#             if parent_widget:
#                 parent_widget.setParent(None)
#                 parent_widget.deleteLater()
#             self.slider_layout = None
#             self.slider_bar_widget = None
#             self.x_offset_slider = self.y_offset_slider = None
#             self.x_offset_input = self.y_offset_input = None
#
#     def resizeEvent(self, e):
#         super().resizeEvent(e)
#         self._resize_offset_controls()
#
#     def update_visibility(self):
#         update_visibility(self.plot_layout, self.plot_checkboxes, self.plots, self.plot_container)
#
#     def toggle_log_scale(self):
#         # Preserve the current view across the linear/log switch so zooming in
#         # and toggling the scale does not reset to the full auto-range.
#         keep_view = self._user_zoomed[0] or self._is_zoomed(0)
#         vb = self.plots[0].getViewBox()
#         try:
#             (x0, x1), (y0, y1) = vb.viewRange()
#         except Exception:
#             keep_view = False
#         self.plots[0].setLogMode(y=self.log_checkbox.isChecked())
#         self.enable_graphs(enable_iq=True)
#         if keep_view:
#             # y is in log units when log mode is on; keep x, refit y to the
#             # visible data so the zoomed q-window stays put.
#             try:
#                 vb.enableAutoRange(x=False, y=True)
#                 self.plots[0].setXRange(float(x0), float(x1), padding=0)
#             except Exception:
#                 pass
#         else:
#             self._autorange_and_fix_x0(0)
#
#     def show_background_data(self):
#         if self.bkg_curve:
#             self.bkg_curve.setVisible(self.background_checkbox.isChecked())
#             self.enable_graphs(enable_iq=True)
#             if not (self._user_zoomed[0] or self._is_zoomed(0)):
#                 try:
#                     self.plots[0].getViewBox().enableAutoRange(x=True, y=True)
#                 except Exception:
#                     pass
#                 self._autorange_and_fix_x0(0)
#
#     def show_original_data(self):
#         if self.raw_curve:
#             self.raw_curve.setVisible(self.original_checkbox.isChecked())
#             self.enable_graphs(enable_iq=True)
#             if not (self._user_zoomed[0] or self._is_zoomed(0)):
#                 try:
#                     self.plots[0].getViewBox().enableAutoRange(x=True, y=True)
#                 except Exception:
#                     pass
#                 self._autorange_and_fix_x0(0)
#
#     def toggle_original_sq(self):
#         if self.sq_curve_original:
#             self.sq_curve_original.setVisible(self.original_sq_checkbox.isChecked())
#             self.enable_graphs(enable_sq=True)
#             if not self._user_zoomed[1]:
#                 self._autorange_and_fix_x0(1)
#
#     def toggle_polynomial_sq(self):
#         if self.sq_curve_polynomial:
#             self.sq_curve_polynomial.setVisible(self.polynomial_sq_checkbox.isChecked())
#             self.enable_graphs(enable_sq=True)
#             if not self._user_zoomed[1]:
#                 self._autorange_and_fix_x0(1)
#
#     def toggle_mean_sq_fi(self):
#         if self.mean_sq_fi_curve:
#             self.mean_sq_fi_curve.setVisible(self.mean_sq_fi_checkbox.isChecked())
#             self.enable_graphs(enable_sq=True)
#             if not self._user_zoomed[1]:
#                 self._autorange_and_fix_x0(1)
#
#     def toggle_sq_mean_fi(self):
#         if self.sq_mean_fi_curve:
#             self.sq_mean_fi_curve.setVisible(self.sq_mean_fi_checkbox.isChecked())
#             self.enable_graphs(enable_sq=True)
#             if not self._user_zoomed[1]:
#                 self._autorange_and_fix_x0(1)
#
#     def toggle_smoothed_fq(self):
#         # "Smoothed" is now a state axis feeding _apply_lorch_view (it selects
#         # the smoothed curves as the base instead of overlaying on raw).
#         self._apply_lorch_view()
#
#     def _on_lorch_toggle(self, sender):
#         # The three Lorch modes are mutually exclusive: turning one on turns
#         # the others off; turning all off returns to the raw view.
#         if sender.isChecked():
#             for cb in (self.apply_lorch_checkbox, self.compare_lorch_checkbox,
#                        self.compare_all_checkbox):
#                 if cb is not sender and cb.isChecked():
#                     cb.blockSignals(True)
#                     cb.setChecked(False)
#                     cb.blockSignals(False)
#         self._apply_lorch_view()
#
#     def _apply_lorch_view(self):
#         """Show/hide the F(q) and G(r) curves per the Lorch toggles.
#
#         Axes: Smoothed (smoothed_fq_checkbox), Apply Lorch, Compare, Compare all.
#         The same rule drives both the F(q) and G(r) panels.
#         """
#         # A Compton plot has only an I(q)-type curve; the Lorch/Smoothed toggles
#         # don't apply, so do nothing here. Otherwise this would force-enable the
#         # (empty) F(q) and G(r) panels.
#         if getattr(self, "is_compton", False):
#             return
#         # "Smoothed" only has an effect when smoothed curves actually exist
#         # (they don't for the xyz/CAL path). Otherwise a checked "Smoothed F(q)"
#         # would route Apply Lorch / Compare to non-existent smoothed variants,
#         # making those toggles look broken for xyz.
#         sm_available = (getattr(self, "fq_smoothed_curve", None) is not None
#                         or getattr(self, "gr_smoothed_curve", None) is not None)
#         sm = bool(self.smoothed_fq_checkbox.isChecked()) and sm_available
#         lorch = bool(getattr(self, "apply_lorch_checkbox", None) and self.apply_lorch_checkbox.isChecked())
#         comp = bool(getattr(self, "compare_lorch_checkbox", None) and self.compare_lorch_checkbox.isChecked())
#         compall = bool(getattr(self, "compare_all_checkbox", None) and self.compare_all_checkbox.isChecked())
#
#         show = {"raw": False, "raw_l": False, "sm": False, "sm_l": False}
#         if compall:
#             show = {"raw": True, "raw_l": True, "sm": True, "sm_l": True}
#         elif comp:
#             if sm:
#                 show["sm"] = True; show["sm_l"] = True
#             else:
#                 show["raw"] = True; show["raw_l"] = True
#         else:
#             if sm and lorch:   show["sm_l"] = True
#             elif sm:           show["sm"] = True
#             elif lorch:        show["raw_l"] = True
#             else:              show["raw"] = True
#
#         def _set_panel(curves):
#             # curves: list of (attr_name, show_key); curves[0] must be the raw base.
#             any_shown = False
#             for attr, key in curves:
#                 c = getattr(self, attr, None)
#                 if c is not None:
#                     vis = bool(show[key])
#                     c.setVisible(vis)
#                     any_shown = any_shown or vis
#             # If the requested variant doesn't exist (e.g. an xyz panel without a
#             # smoothed/Lorch curve), fall back to the raw base so the panel is
#             # never left blank.
#             if not any_shown:
#                 base = getattr(self, curves[0][0], None)
#                 if base is not None:
#                     base.setVisible(True)
#
#         _set_panel([("fq_main_curve", "raw"), ("fq_lorch_curve", "raw_l"),
#                     ("fq_smoothed_curve", "sm"), ("fq_smoothed_lorch_curve", "sm_l")])
#         _set_panel([("gr_main_curve", "raw"), ("gr_lorch_curve", "raw_l"),
#                     ("gr_smoothed_curve", "sm"), ("gr_smoothed_lorch_curve", "sm_l")])
#         # Only set curve visibility here; do NOT force the F(q)/G(r) panels on.
#         # Panel visibility is the user's choice (legend checkboxes), and this
#         # method also runs on every re-plot (e.g. moving the background-scale
#         # slider) -- forcing panels on there would resurrect panels the user
#         # hid. Autorange only visible panels that the user has NOT zoomed (a
#         # zoomed panel keeps its view; the toggled curve stays in range anyway).
#         zoomed = getattr(self, "_user_zoomed", [False, False, False, False])
#         if len(self.plots) > 2 and self.plots[2].isVisible() and not zoomed[2]:
#             self._autorange_and_fix_x0(2)
#         if len(self.plots) > 3 and self.plots[3].isVisible() and not zoomed[3]:
#             self._autorange_and_fix_x0(3)
#
#     def plot_under_cursor(self, global_pos):
#         for plot in self.plots:
#             if not plot.isVisible():
#                 continue
#             local_pos = plot.mapFromGlobal(global_pos)
#             if plot.rect().contains(local_pos):
#                 return plot
#         return None
#
#     def keyPressEvent(self, event):
#         if event.isAutoRepeat():
#             return
#         if event.key() == Qt.Key.Key_R:
#             self.reset_zoom_on_hovered_plot()
#         elif event.key() == Qt.Key.Key_Z:
#             self.left_button_pan_enabled = not self.left_button_pan_enabled
#             for plot in self.plots:
#                 vb = plot.getViewBox()
#                 vb.setMouseMode(pg.ViewBox.PanMode if self.left_button_pan_enabled else pg.ViewBox.RectMode)
#
#     def _mark_user_zoomed(self, idx):
#         try:
#             if 0 <= idx < len(self._user_zoomed):
#                 self._user_zoomed[idx] = True
#         except Exception:
#             pass
#
#     def _is_zoomed(self, idx):
#         """True if the panel's current view spans notably less than its data
#         extent (i.e. the user has zoomed in). Deterministic — does not rely on
#         the manual-range signal, which can miss some zoom interactions."""
#         try:
#             vb = self.plots[idx].getViewBox()
#             (vx0, vx1), (vy0, vy1) = vb.viewRange()
#             b = vb.childrenBounds()
#             if not b:
#                 return bool(self._user_zoomed[idx]) if idx < len(self._user_zoomed) else False
#             zoomed = False
#             if b[0] is not None:
#                 dsx = float(b[0][1]) - float(b[0][0])
#                 if dsx > 0 and (float(vx1) - float(vx0)) < 0.98 * dsx:
#                     zoomed = True
#             if b[1] is not None:
#                 dsy = float(b[1][1]) - float(b[1][0])
#                 if dsy > 0 and (float(vy1) - float(vy0)) < 0.98 * dsy:
#                     zoomed = True
#             return zoomed
#         except Exception:
#             try:
#                 return bool(self._user_zoomed[idx])
#             except Exception:
#                 return False
#
#     def reset_zoom_on_hovered_plot(self):
#         global_pos = QCursor.pos()
#         plot = self.plot_under_cursor(global_pos)
#         if not plot:
#             return
#         try:
#             idx = self.plots.index(plot)
#             # R returns this panel to auto-scaling: clear its user-zoom flag so
#             # later parameter changes auto-fit it again.
#             if 0 <= idx < len(self._user_zoomed):
#                 self._user_zoomed[idx] = False
#             self._autorange_and_fix_x0(idx)
#         except Exception:
#             try:
#                 plot.getViewBox().autoRange()
#                 self._force_x_from_zero(plot)
#             except Exception:
#                 pass
#
#     def export_graphs(self):
#         dialog = SaveMenu(self)
#         if dialog.exec():
#             dialog.export_images()
#
#     def export_graph_data(self):
#         dialog = SaveMenu(self)
#         if dialog.exec():
#             dialog.export_files()
#
#     def _save_data_dialog(self):
#         dlg = QDialog(self)
#         dlg.setWindowTitle("Select data to save")
#         v = QVBoxLayout(dlg)
#         CB_ROUND_STYLE = """
#         QCheckBox::indicator { width:12px; height:12px; border-radius:6px;
#             border:1px solid #777; background:#fff; }
#         QCheckBox::indicator:checked { background:#444; }
#         """
#         is_compton = getattr(self, "is_compton", False)
#         is_multi_plot = hasattr(self, 'cached_xs') and self.cached_xs is not None
#         is_single_exp = False
#         if not is_multi_plot:
#             is_single_exp = (self.raw_iq is not None and self.raw_iq[0] is not None) or \
#                             (self.background_data is not None and self.background_data[0] is not None) or \
#                             (self.mean_sq_fi is not None)
#         is_experimental = (is_multi_plot or is_single_exp) and not is_compton
#         # A Compton curve is neither experimental data nor calculated (.xyz)
#         # output; it has its own save option below.
#         is_calculated = not is_experimental and not is_compton
#         cols = QHBoxLayout()
#         exp_col = QVBoxLayout()
#         cal_col = QVBoxLayout()
#         compton_col = QVBoxLayout()
#
#         # --- Experimental Data column ---
#         exp_col.addWidget(QLabel("--- Experimental Data ---", dlg))
#         checks_core = []
#         core_labels = [("I(q)", 0, ".iq"), ("S(q)", 1, ".sq"), ("F(q)", 2, ".fq"), ("G(r)", 3, ".gr")]
#         for text_label, idx, _ext in core_labels:
#             cb = QCheckBox(text_label, dlg)
#             cb.setStyleSheet(CB_ROUND_STYLE)
#             cb.setChecked(False)
#             has_data = False
#             try:
#                 xs_src = self.cached_xs[0] if (is_multi_plot and self.cached_xs) else self.xs
#                 has_data = xs_src is not None and len(xs_src) > idx and xs_src[idx] is not None and len(xs_src[idx]) > 0
#             except Exception:
#                 pass
#             if not (has_data and is_experimental):
#                 cb.setEnabled(False)
#             exp_col.addWidget(cb)
#             checks_core.append(cb)
#         # F(q) variants are listed before G(r) variants.
#         extras_def = [
#             ("Original S(q)", 1, lambda: self.sq_original_data, "_Sq_original", ".sq"),
#             ("Polynomial for S(q)", 1, lambda: self.sq_polynomial_data, "_Sq_polynomial", ".sq"),
#             ("<f^2>", 1, lambda: self.mean_sq_fi, "_f2", ".dat"),
#             ("<f>^2", 1, lambda: self.sq_mean_fi, "_favg2", ".dat"),
#             ("Smoothed F(q)", 2, lambda: self.fq_smoothed_data, "_Fq_smoothed", ".fq"),
#             ("Lorch F(q)", 2, lambda: self.fq_lorch_data[1] if getattr(self, "fq_lorch_data", None) else None,
#              "_Fq_lorch", ".fq"),
#             ("Smoothed Lorch F(q)", 2, lambda: self.fq_smoothed_lorch_data[1] if getattr(self, "fq_smoothed_lorch_data", None) else None,
#              "_Fq_smoothed_lorch", ".fq"),
#             ("Smoothed G(r)", 3, lambda: self.gr_smoothed_data[1] if self.gr_smoothed_data else None,
#              "_Gr_smoothed", ".gr"),
#             ("Lorch G(r)", 3, lambda: self.gr_lorch_data[1] if getattr(self, "gr_lorch_data", None) else None,
#              "_Gr_lorch", ".gr"),
#             ("Smoothed Lorch G(r)", 3, lambda: self.gr_smoothed_lorch_data[1] if getattr(self, "gr_smoothed_lorch_data", None) else None,
#              "_Gr_smoothed_lorch", ".gr"),
#         ]
#         checks_extras = []
#         if not is_multi_plot:
#             for label, _xidx, y_fn, _suf, _ext in extras_def:
#                 cb = QCheckBox(label, dlg)
#                 cb.setStyleSheet(CB_ROUND_STYLE)
#                 cb.setChecked(False)
#                 y_ok = False
#                 if is_experimental:
#                     try:
#                         y = y_fn()
#                         y_ok = y is not None and (isinstance(y, (np.ndarray, list)) and len(y) > 0)
#                     except Exception:
#                         y_ok = False
#                 if not y_ok:
#                     cb.setEnabled(False)
#                 exp_col.addWidget(cb)
#                 checks_extras.append(cb)
#         exp_col.addStretch()
#
#         # --- Calculated (.xyz) column ---
#         cal_col.addWidget(QLabel("--- Calculated (.xyz) ---", dlg))
#         checks_cal = []
#         # Grouped so each Lorch variant follows its base: I(q), S(q),
#         # F(q), F(q) Lorch, G(r), G(r) Lorch. The save mapping in
#         # _save_selected_data uses this exact order.
#         cal_defs = [
#             ("calI(q)", "core", 0),
#             ("calS(q)", "core", 1),
#             ("calF(q)", "core", 2),
#             ("calF(q) Lorch", "lorch", "fq_lorch_data"),
#             ("calG(r)", "core", 3),
#             ("calG(r) Lorch", "lorch", "gr_lorch_data"),
#         ]
#         for text_label, kind, ref in cal_defs:
#             cb = QCheckBox(text_label, dlg)
#             cb.setStyleSheet(CB_ROUND_STYLE)
#             cb.setChecked(False)
#             has_data = False
#             try:
#                 if kind == "core":
#                     src = self.cached_xs[0] if is_multi_plot and self.cached_xs else self.xs
#                     has_data = (src is not None and len(src) > ref
#                                 and src[ref] is not None and len(src[ref]) > 0)
#                 else:
#                     d = getattr(self, ref, None)
#                     has_data = d is not None and d[1] is not None and len(d[1]) > 0
#             except Exception:
#                 pass
#             if not (has_data and is_calculated):
#                 cb.setEnabled(False)
#             cal_col.addWidget(cb)
#             checks_cal.append(cb)
#         cal_col.addStretch()
#
#         # --- Compton column ---
#         compton_col.addWidget(QLabel("--- Compton ---", dlg))
#         cb_compton = QCheckBox("Compton", dlg)
#         cb_compton.setStyleSheet(CB_ROUND_STYLE)
#         cb_compton.setChecked(False)
#         compton_has_data = False
#         try:
#             compton_has_data = (self.xs is not None and len(self.xs) > 0
#                                 and self.xs[0] is not None and len(self.xs[0]) > 0)
#         except Exception:
#             compton_has_data = False
#         if not (is_compton and compton_has_data):
#             cb_compton.setEnabled(False)
#         compton_col.addWidget(cb_compton)
#         compton_col.addStretch()
#
#         cols.addLayout(exp_col)
#         cols.addLayout(cal_col)
#         cols.addLayout(compton_col)
#         v.addLayout(cols)
#         btns = QHBoxLayout()
#         ok = QPushButton("Save", dlg)
#         cancel = QPushButton("Cancel", dlg)
#         btns.addWidget(ok);
#         btns.addWidget(cancel)
#         v.addLayout(btns)
#
#         def do_save():
#             selected_core = [i for i, cb in enumerate(checks_core) if cb.isChecked()]
#             selected_extras = [i for i, cb in enumerate(checks_extras) if cb.isChecked()]
#             selected_cal = [i for i, cb in enumerate(checks_cal) if cb.isChecked()]
#             selected_compton = cb_compton.isChecked()
#             if not selected_core and not selected_extras and not selected_cal and not selected_compton:
#                 QMessageBox.information(self, "Nothing selected", "Please select at least one dataset to save.")
#                 return
#
#             # Remember the last folder used for saving, shared with the other
#             # save dialogs (same QSettings key), so every save reopens where
#             # the previous one left off.
#             settings = QSettings("EZPDF", "EZPDF")
#             last_dir = settings.value("last_dir", "") or ""
#
#             # Compton is a single file, so let the user type the file name
#             # directly (choosing the folder at the same time). On a Compton
#             # plot the other datasets are disabled, so nothing else can be
#             # selected here.
#             if selected_compton:
#                 default_name = self.file_name or "compton"
#                 if default_name.lower().endswith(".compton"):
#                     default_name = default_name[:-len(".compton")]
#                 start_path = os.path.join(last_dir, default_name + ".compton") if last_dir \
#                     else default_name + ".compton"
#                 path, _ = QFileDialog.getSaveFileName(
#                     self, "Save Compton File", start_path,
#                     "Compton Files (*.compton)")
#                 if not path:
#                     return
#                 if not path.lower().endswith(".compton"):
#                     path += ".compton"
#                 written = self._write_compton_file(path)
#                 if written:
#                     settings.setValue("last_dir", os.path.dirname(path))
#                     settings.sync()
#                     QMessageBox.information(self, "Saved", f"Compton file saved to:\n{path}")
#                 else:
#                     QMessageBox.warning(self, "Save failed", "No Compton data was available to save.")
#                 dlg.accept()
#                 return
#
#             folder = QFileDialog.getExistingDirectory(self, "Select folder to save", last_dir)
#             if not folder: return
#             settings.setValue("last_dir", folder)
#             settings.sync()
#             written = self._save_selected_data(folder, selected_core, selected_extras, selected_cal)
#             QMessageBox.information(self, "Saved", f"{written} file(s) saved to:\n{folder}")
#             dlg.accept()
#
#         ok.clicked.connect(do_save)
#         cancel.clicked.connect(dlg.reject)
#         dlg.exec()
#
#     def _write_compton_file(self, path):
#         """Write the Compton I(q) curve (panel 0) to a .compton file.
#
#         Returns 1 on success, 0 if there is no data or the write fails.
#         """
#         try:
#             x = self.xs[0] if (self.xs is not None and len(self.xs) > 0) else None
#             y = self.ys[0] if (self.ys is not None and len(self.ys) > 0) else None
#             if x is None or y is None or len(x) == 0 or len(y) == 0:
#                 return 0
#             n = min(len(x), len(y))
#             arr = np.column_stack([x[:n], y[:n]])
#             np.savetxt(path, arr, fmt="%.10g", header="x y", comments="")
#             return 1
#         except Exception:
#             return 0
#
#     def _save_selected_data(self, folder, selected_core, selected_extras, selected_cal=None):
#         def _write_xy(path, x, y):
#             try:
#                 if x is None or y is None or len(x) == 0 or len(y) == 0: return 0
#                 n = min(len(x), len(y))
#                 arr = np.column_stack([x[:n], y[:n]])
#                 np.savetxt(path, arr, fmt="%.10g", header="x y", comments="")
#                 return 1
#             except Exception:
#                 return 0
#
#         written = 0
#         is_multi = hasattr(self, "cached_xs") and self.cached_xs is not None
#         if is_multi:
#             count = len(self.cached_xs)
#             core_map = {0: ("_I", 0, ".iq"), 1: ("_S", 1, ".sq"), 2: ("_F", 2, ".fq"), 3: ("_G", 3, ".gr")}
#             cal_map = {0: (".caliq", 0), 1: (".calsq", 1), 2: (".calfq", 2), 3: (".calgr", 3)}
#             for k in range(count):
#                 if hasattr(self, "cached_titles") and self.cached_titles and k < len(self.cached_titles):
#                     base_name = self.cached_titles[k]
#                 else:
#                     base_name = f"data_{k}"
#                 base = os.path.splitext(base_name)[0]
#                 xs_src = self.cached_xs[k]
#                 ys_src = self.cached_ys[k]
#                 for idx in selected_core:
#                     suf, ax, ext = core_map[idx]
#                     try:
#                         if xs_src and ys_src and len(xs_src) > ax and len(ys_src) > ax:
#                             written += _write_xy(os.path.join(folder, f"{base}{suf}{ext}"), xs_src[ax], ys_src[ax])
#                     except Exception:
#                         pass
#                 if selected_cal:
#                     for idx in selected_cal:
#                         ext, ax = cal_map[idx]
#                         try:
#                             if xs_src and ys_src and len(xs_src) > ax and len(ys_src) > ax:
#                                 written += _write_xy(os.path.join(folder, f"{base}{ext}"), xs_src[ax], ys_src[ax])
#                         except Exception:
#                             pass
#         else:
#             xs_src = self.xs;
#             ys_src = self.ys;
#             base_name = self.file_name or "data";
#             base = os.path.splitext(base_name)[0]
#             core_map = {0: ("_I", 0, ".iq"), 1: ("_S", 1, ".sq"), 2: ("_F", 2, ".fq"), 3: ("_G", 3, ".gr")}
#             for idx in selected_core:
#                 suf, ax, ext = core_map[idx]
#                 try:
#                     if xs_src and ys_src and len(xs_src) > ax and len(ys_src) > ax:
#                         written += _write_xy(os.path.join(folder, f"{base}{suf}{ext}"), xs_src[ax], ys_src[ax])
#                 except Exception:
#                     pass
#
#             def _xy_or_none(ax, y):
#                 try:
#                     if xs_src is None or len(xs_src) <= ax or y is None: return None, None
#                     return xs_src[ax], y
#                 except Exception:
#                     return None, None
#
#             extra_sources = [
#                 ("_Sq_original", 1, lambda: _xy_or_none(1, self.sq_original_data), ".sq"),
#                 ("_Sq_polynomial", 1, lambda: _xy_or_none(1, self.sq_polynomial_data), ".sq"),
#                 ("_f2", 1, lambda: _xy_or_none(1, self.mean_sq_fi), ".dat"),
#                 ("_favg2", 1, lambda: _xy_or_none(1, self.sq_mean_fi), ".dat"),
#                 ("_Fq_smoothed", 2, lambda: _xy_or_none(2, self.fq_smoothed_data), ".fq"),
#                 ("_Fq_lorch", 2,
#                  lambda: (self.fq_lorch_data[0], self.fq_lorch_data[1]) if getattr(self, "fq_lorch_data", None) else (None, None),
#                  ".fq"),
#                 ("_Fq_smoothed_lorch", 2,
#                  lambda: (self.fq_smoothed_lorch_data[0], self.fq_smoothed_lorch_data[1]) if getattr(self, "fq_smoothed_lorch_data", None) else (None, None),
#                  ".fq"),
#                 ("_Gr_smoothed", 3,
#                  lambda: (self.gr_smoothed_data[0], self.gr_smoothed_data[1]) if self.gr_smoothed_data else (None, None),
#                  ".gr"),
#                 ("_Gr_lorch", 3,
#                  lambda: (self.gr_lorch_data[0], self.gr_lorch_data[1]) if getattr(self, "gr_lorch_data", None) else (None, None),
#                  ".gr"),
#                 ("_Gr_smoothed_lorch", 3,
#                  lambda: (self.gr_smoothed_lorch_data[0], self.gr_smoothed_lorch_data[1]) if getattr(self, "gr_smoothed_lorch_data", None) else (None, None),
#                  ".gr"),
#             ]
#             for i in selected_extras:
#                 suf, _ax, getter, ext = extra_sources[i]
#                 try:
#                     x, y = getter()
#                     if x is not None and y is not None:
#                         written += _write_xy(os.path.join(folder, f"{base}{suf}{ext}"), x, y)
#                 except Exception:
#                     pass
#             if selected_cal:
#                 # Same order as the dialog's cal_defs:
#                 # I(q), S(q), F(q), F(q) Lorch, G(r), G(r) Lorch.
#                 cal_defs = [
#                     (".caliq", "core", 0),
#                     (".calsq", "core", 1),
#                     (".calfq", "core", 2),
#                     ("_lorch.calfq", "lorch", "fq_lorch_data"),
#                     (".calgr", "core", 3),
#                     ("_lorch.calgr", "lorch", "gr_lorch_data"),
#                 ]
#                 for idx in selected_cal:
#                     try:
#                         ext, kind, ref = cal_defs[idx]
#                         if kind == "core":
#                             if xs_src and ys_src and len(xs_src) > ref and len(ys_src) > ref:
#                                 written += _write_xy(os.path.join(folder, f"{base}{ext}"), xs_src[ref], ys_src[ref])
#                         else:
#                             d = getattr(self, ref, None)
#                             if d and d[0] is not None and d[1] is not None:
#                                 written += _write_xy(os.path.join(folder, f"{base}{ext}"), d[0], d[1])
#                     except Exception:
#                         pass
#         return written
#
#     def save_combined_graph_image(self):
#         try:
#             from PySide6.QtSvg import QSvgGenerator
#         except Exception:
#             QSvgGenerator = None
#         filters = "PNG (*.png);;JPEG (*.jpg *.jpeg);;SVG (*.svg);;PDF (*.pdf);;All Supported (*.png *.jpg *.jpeg *.svg *.pdf);;All Files (*)"
#         default_name = (self.file_name or "plot") + "_figure.png"
#         # Reuse the folder remembered by the other save dialogs.
#         settings = QSettings("EZPDF", "EZPDF")
#         last_dir = settings.value("last_dir", "") or ""
#         start_path = os.path.join(last_dir, default_name) if last_dir else default_name
#         file_path, selected_filter = QFileDialog.getSaveFileName(self, "Save figure", start_path, filters)
#         if not file_path: return
#         ext = os.path.splitext(file_path)[1].lower()
#         if not ext:
#             if "SVG" in selected_filter.upper():
#                 ext = ".svg";
#                 file_path += ext
#             elif "PDF" in selected_filter.upper():
#                 ext = ".pdf";
#                 file_path += ext
#             elif "JPEG" in selected_filter.upper():
#                 ext = ".jpg";
#                 file_path += ext
#             else:
#                 ext = ".png";
#                 file_path += ext
#         selected = [pw for cb, pw in zip(self.plot_checkboxes, self.plots) if cb.isChecked()]
#         if not selected:
#             QMessageBox.information(self, "No plots selected", "Please enable at least one plot to save.")
#             return
#         # Remember this folder for the next save.
#         settings.setValue("last_dir", os.path.dirname(file_path))
#         settings.sync()
#         union_rect = self.main_widget.rect()
#         if ext == ".svg" and QSvgGenerator is not None:
#             gen = QSvgGenerator();
#             gen.setFileName(file_path);
#             gen.setSize(QSize(union_rect.width(), union_rect.height()));
#             gen.setViewBox(union_rect)
#             p = QPainter(gen);
#             self.main_widget.render(p, QPoint());
#             p.end();
#             return
#         if ext == ".pdf":
#             writer = QPdfWriter(file_path);
#             writer.setResolution(300);
#             writer.setPageMargins(QMarginsF(0, 0, 0, 0));
#             dpi = writer.resolution();
#             pdf_rect = union_rect;
#             w_pt = pdf_rect.width() * 72.0 / dpi;
#             h_pt = pdf_rect.height() * 72.0 / dpi;
#             writer.setPageSize(QPageSize(QSizeF(w_pt, h_pt), QPageSize.Unit.Point))
#             p = QPainter(writer);
#             self.main_widget.render(p, QPoint());
#             p.end();
#             return
#         # Raster (PNG/JPG): render each selected plot with pyqtgraph's own
#         # ImageExporter at a high target width so lines and text stay vector-
#         # crisp, then stack the panels vertically. (QWidget.render() + painter
#         # scaling just upscales pyqtgraph's cached bitmap, which looks blurry.)
#         target_width = 3600  # px; raises effective DPI of the saved figure
#         panel_imgs = None
#         try:
#             from pyqtgraph.exporters import ImageExporter
#             panel_imgs = []
#             for pw in selected:
#                 exp = ImageExporter(pw.plotItem)
#                 try:
#                     exp.parameters()['width'] = int(target_width)
#                     exp.parameters()['antialias'] = True
#                 except Exception:
#                     pass
#                 qi = exp.export(toBytes=True)
#                 if qi is not None and not qi.isNull():
#                     panel_imgs.append(qi)
#             if not panel_imgs:
#                 panel_imgs = None
#         except Exception:
#             panel_imgs = None
#
#         if panel_imgs:
#             total_h = sum(im.height() for im in panel_imgs)
#             max_w = max(im.width() for im in panel_imgs)
#             img = QImage(max_w, total_h, QImage.Format.Format_ARGB32)
#             img.fill(QColor(255, 255, 255))
#             p = QPainter(img)
#             y = 0
#             for im in panel_imgs:
#                 p.drawImage(0, y, im)
#                 y += im.height()
#             p.end()
#         else:
#             # Fallback: high-scale widget render (still better than 1x).
#             scale = 6.0
#             w = max(1, int(union_rect.width() * scale))
#             h = max(1, int(union_rect.height() * scale))
#             img = QImage(w, h, QImage.Format.Format_ARGB32)
#             img.fill(QColor(255, 255, 255))
#             p = QPainter(img)
#             p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
#             p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
#             p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
#             try:
#                 p.setRenderHint(QPainter.RenderHint.HighQualityAntialiasing, True)
#             except Exception:
#                 pass
#             p.scale(scale, scale)
#             self.main_widget.render(p, QPoint())
#             p.end()
#
#         if ext in (".jpg", ".jpeg"):
#             img.save(file_path, "JPG", quality=100)
#         else:
#             img.save(file_path, "PNG")
#
#     def get_x_offset(self):
#         try:
#             return float(self.x_offset_input.text())
#         except (ValueError, AttributeError):
#             return 0.0
#
#     def get_y_offset(self):
#         try:
#             return float(self.y_offset_input.text())
#         except (ValueError, AttributeError):
#             return 10.0
#
#     def update_offsets_and_replot(self):
#         if hasattr(self, "cached_xs") and hasattr(self, "cached_ys"): self.plot_multiple(self.cached_xs, self.cached_ys,
#                                                                                          self.cached_titles)
#
#     def bring_to_front(self):
#         self.show();
#         self.raise_()
#
#     def enable_graphs(self, enable_iq=None, enable_sq=None, enable_fq=None, enable_gr=None):
#         def set_graph_state(index, enable):
#             if enable is not None:
#                 self.plot_checkboxes[index].blockSignals(True);
#                 self.plot_checkboxes[index].setChecked(enable);
#                 self.plot_checkboxes[index].blockSignals(False)
#                 if enable:
#                     self.plots[index].show()
#                 else:
#                     self.plots[index].hide()
#
#         set_graph_state(0, enable_iq);
#         set_graph_state(1, enable_sq);
#         set_graph_state(2, enable_fq);
#         set_graph_state(3, enable_gr);
#         self.update_visibility()
#
#     def _install_xy_readouts(self):
#         self._xy_labels = [];
#         self._mouse_proxies = [];
#         self._active_idx = None
#         for idx, pw in enumerate(self.plots):
#             vb = pw.getPlotItem().getViewBox();
#             lbl = pg.TextItem("", anchor=(1, 1), color=(40, 40, 40))
#             try:
#                 lbl.setFill(pg.mkBrush(255, 255, 255, 180))
#             except Exception:
#                 pass
#             lbl.setFont(QFont("Arial", HUD_PT));
#             lbl.setZValue(1e6);
#             lbl.setVisible(False);
#             pw.addItem(lbl);
#             self._xy_labels.append(lbl)
#             vb.sigRangeChanged.connect(lambda _vb, _vr, i=idx: self._reposition_xy_label(i))
#             proxy = pg.SignalProxy(pw.scene().sigMouseMoved, rateLimit=60,
#                                    slot=lambda ev, i=idx, viewbox=vb: self._on_mouse_moved(i, viewbox, ev))
#             self._mouse_proxies.append(proxy);
#             self._reposition_xy_label(idx)
#
#     def _ensure_xy_label(self, idx: int):
#         if not hasattr(self, "_xy_labels"): self._xy_labels = [None] * len(self.plots)
#         pw = self.plots[idx];
#         lbl = None if idx >= len(self._xy_labels) else self._xy_labels[idx]
#         if lbl is None or lbl.scene() is None:
#             lbl = pg.TextItem("", anchor=(1, 1), color=(40, 40, 40))
#             try:
#                 lbl.setFill(pg.mkBrush(255, 255, 255, 180))
#             except Exception:
#                 pass
#             lbl.setFont(QFont("Arial", HUD_PT));
#             lbl.setZValue(1e6);
#             lbl.setVisible(False);
#             pw.addItem(lbl)
#             if idx >= len(self._xy_labels): self._xy_labels.extend([None] * (idx + 1 - len(self._xy_labels)))
#             self._xy_labels[idx] = lbl
#         self._reposition_xy_label(idx)
#
#     def _reposition_xy_label(self, idx: int):
#         try:
#             pw = self.plots[idx];
#             vb = pw.getViewBox();
#             (x0, x1), (y0, y1) = vb.viewRange();
#             dx = 0.05 * (x1 - x0);
#             dy = 0.08 * (y1 - y0);
#             self._xy_labels[idx].setPos(x1 - dx, y0 + dy)
#         except Exception:
#             pass
#
#     def _on_mouse_moved(self, idx: int, vb, ev):
#         try:
#             pos = ev[0] if isinstance(ev, (tuple, list)) else ev;
#             inside = vb.sceneBoundingRect().contains(pos)
#             if inside:
#                 for j, lbl in enumerate(self._xy_labels):
#                     if lbl is None: continue
#                     lbl.setVisible(j == idx)
#                 p = vb.mapSceneToView(pos);
#                 val_x = p.x();
#                 val_y = p.y()
#                 if vb.state['logMode'][1]: val_y = 10 ** val_y
#                 self._xy_labels[idx].setText(f"x = {val_x:.3f},  y = {val_y:.3f}");
#                 self._reposition_xy_label(idx);
#                 self._active_idx = idx
#             else:
#                 if self._active_idx == idx and self._xy_labels[idx] is not None: self._xy_labels[idx].setVisible(
#                     False); self._active_idx = None
#         except Exception:
#             pass
#
#     def _force_x_from_zero(self, plot_widget):
#         try:
#             vb = plot_widget.getViewBox();
#             (x0, x1), (_y0, _y1) = vb.viewRange()
#             if x1 <= 0:
#                 plot_widget.setXRange(0, 1, padding=0)
#             else:
#                 plot_widget.setXRange(0, x1, padding=0)
#         except Exception:
#             pass
#
#     def _autorange_and_fix_x0(self, idx: int):
#         try:
#             pw = self.plots[idx]
#             vb = pw.getViewBox()
#             xmax = None
#             ymin = None
#             ymax = None
#
#             # Only include items that are truly visible on screen.
#             # item.isVisible() alone can return True even for items whose
#             # opacity is 0 or whose parent is hidden — use item.opts to
#             # check the pen alpha as a reliable visibility indicator.
#             def _is_truly_visible(item):
#                 try:
#                     if not item.isVisible():
#                         return False
#                     # Check pen alpha: items hidden via setPen(transparent)
#                     # still report isVisible()=True
#                     pen = item.opts.get('pen', None)
#                     if pen is not None:
#                         try:
#                             from pyqtgraph import mkPen
#                             p = mkPen(pen)
#                             if p.color().alpha() == 0:
#                                 return False
#                         except Exception:
#                             pass
#                     return True
#                 except Exception:
#                     return item.isVisible()
#
#             visible_items = [it for it in pw.listDataItems() if _is_truly_visible(it)]
#             if not visible_items:
#                 vb.autoRange()
#                 self._force_x_from_zero(pw)
#                 return
#
#             for it in visible_items:
#                 try:
#                     x, y = it.getData()
#                     if x is None or len(x) == 0 or y is None or len(y) == 0:
#                         continue
#                     finite_mask = np.isfinite(x) & np.isfinite(y)
#                     if not np.any(finite_mask):
#                         continue
#                     x_finite = x[finite_mask]
#                     y_finite = y[finite_mask]
#                     if len(x_finite) == 0 or len(y_finite) == 0:
#                         continue
#                     cur_xmax = float(x_finite.max())
#                     cur_ymin = float(y_finite.min())
#                     cur_ymax = float(y_finite.max())
#                     if xmax is None or cur_xmax > xmax: xmax = cur_xmax
#                     if ymin is None or cur_ymin < ymin: ymin = cur_ymin
#                     if ymax is None or cur_ymax > ymax: ymax = cur_ymax
#                 except Exception:
#                     continue
#
#             if xmax is not None and ymin is not None and ymax is not None:
#                 padding = (ymax - ymin) * 0.05
#                 if padding == 0:
#                     padding = 1.0
#                 pw.setYRange(ymin - padding, ymax + padding, padding=0.0)
#                 pw.setXRange(0.0, xmax, padding=0.0)
#             else:
#                 vb.autoRange()
#                 self._force_x_from_zero(pw)
#         except Exception as e:
#             try:
#                 pw.getViewBox().autoRange()
#                 self._force_x_from_zero(pw)
#             except Exception:
#                 pass