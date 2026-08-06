# ui/control_panel.py
import os
import re

import numpy as np

# [중요] Qt 모듈 임포트
from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator, QFont
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QStyle,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ezpit.gui.controller.graph_controller import (
    calculate_compton,
    update_current_graph,
)
from ezpit.gui.model.helpers import (
    composition_string_from_xyz,
    extract_data,
    parse_composition,
    preview_composition,
)
from ezpit.processing import reset_warning_history

from .ui_helpers import add_form_row, add_slider_field

# Hint shown under the composition field. Kept in one place because the same
# text is used by both the Basic and the Compton tab.
# Covers every accepted style: spaced or compact, a count of 1 that may be
# omitted, and fractional amounts. Scaling every element by the same factor
# describes the same material (Li0.2Co0.36Mn0.37Ni0.07 = Li20Co36Mn37Ni7).
COMPOSITION_EXAMPLE_TEXT = (
    "Examples:  C 1 O 2 P 5   ·   Co38O119P1   ·   SiO2  (a count of 1 may be omitted)\n"
    "Fractions are allowed:  Li0.2Co0.36Mn0.37Ni0.07  =  Li20Co36Mn37Ni7"
)


class ControlPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.main_window = parent

        # The .xyz file that last auto-filled the Compton composition, and the
        # composition string it produced. Used so the Compton Calculate button
        # can ask which source to use even after the file is deselected.
        self._compton_xyz_path = None
        self._compton_xyz_comp = None

        fixed_font = QFont()
        fixed_font.setPointSize(9)
        self.setFont(fixed_font)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        tabs = QTabWidget()
        tabs.addTab(self.basic_controls(), "Basic")
        tabs.addTab(self.pdf_controls(), "PDF")
        tabs.addTab(self.cal_controls(), "CAL")
        self.compton_tab_widget = self.compton_controls()
        tabs.addTab(self.compton_tab_widget, "Compton")

        # Warn about ionic species when the Compton tab is opened, since Compton
        # scattering is defined for neutral atoms only.
        self.control_tabs = tabs
        tabs.currentChanged.connect(self._on_control_tab_changed)

        layout.addWidget(tabs)

    def basic_controls(self):
        basic_tab = QWidget()

        main_v_layout = QVBoxLayout(basic_tab)
        main_v_layout.setContentsMargins(8, 8, 8, 8)

        form_widget = QWidget()
        basic_layout = QFormLayout(form_widget)
        basic_layout.setContentsMargins(0, 0, 0, 0)
        basic_layout.setHorizontalSpacing(10)
        basic_layout.setVerticalSpacing(8)

        self.data_format_group = QButtonGroup()

        self.format_2theta = QRadioButton("2θ (degree)")
        self.format_q_invA = QRadioButton("Q (Å⁻¹)")
        self.format_q_nmn = QRadioButton("Q (nm⁻¹)")
        self.format_q_invA.setChecked(True)

        self.data_format_group.addButton(self.format_2theta)
        self.data_format_group.addButton(self.format_q_invA)
        self.data_format_group.addButton(self.format_q_nmn)

        data_format_layout = QHBoxLayout()
        data_format_layout.addWidget(self.format_2theta)
        data_format_layout.addWidget(self.format_q_invA)
        data_format_layout.addWidget(self.format_q_nmn)
        basic_layout.addRow("Data Format:", data_format_layout)

        self.data_format_group.buttonClicked.connect(self.send_update)

        self.source_type_label = QLabel("Source Type:")
        self.source_type_dropdown = QComboBox()
        self.source_type_dropdown.addItems(["Custom", "Mo K-Alpha"])

        source_type_layout = QHBoxLayout()
        source_type_layout.addWidget(self.source_type_label)
        source_type_layout.addWidget(self.source_type_dropdown)
        source_type_layout.addStretch()
        basic_layout.addRow(source_type_layout)

        self.wavelength_label = QLabel("Wavelength: ")
        self.wavelength_input = QLineEdit()
        self.wavelength_input.setPlaceholderText("Enter wavelength")
        self.wavelength_input.setText("0.1")

        wavelength_layout = QHBoxLayout()
        wavelength_layout.addWidget(self.wavelength_label)
        wavelength_layout.addWidget(self.wavelength_input)
        basic_layout.addRow(wavelength_layout)

        def update_source_visibility():
            visible = self.format_2theta.isChecked()
            self.source_type_dropdown.setVisible(visible)
            self.source_type_label.setVisible(visible)

        self.format_2theta.toggled.connect(update_source_visibility)
        update_source_visibility()

        self.background_edit = QLineEdit()
        self.background_edit.setPlaceholderText("Choose or paste a background file path")
        self.background_edit.setClearButtonEnabled(True)
        self.background_edit.textChanged.connect(self._on_background_text_changed)
        self.background_edit.returnPressed.connect(self._on_background_editing_finished)
        self.background_edit.editingFinished.connect(self._on_background_editing_finished)

        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)
        self.bg_browse_action = self.background_edit.addAction(icon, QLineEdit.ActionPosition.TrailingPosition)
        self.bg_browse_action.setToolTip("Choose Background File")
        self.bg_browse_action.triggered.connect(self.select_background_file)

        background_layout = QHBoxLayout()
        background_layout.addWidget(self.background_edit, 1)

        self.use_bg_checkbox = QCheckBox("Use BG")
        self.use_bg_checkbox.setChecked(False)
        self.use_bg_checkbox.setEnabled(False)
        self.use_bg_checkbox.toggled.connect(self._on_use_bg_toggled)
        background_layout.addWidget(self.use_bg_checkbox)

        basic_layout.addRow("Background:", background_layout)

        self.background_label = QLabel("No file selected")
        self.background_label.setVisible(False)
        self.background_path = None
        self.background_enabled = False

        self.composition_input = QLineEdit()
        self.composition_input.setPlaceholderText("Enter composition (e.g. C 1 O 2  or  Co38O119P1)")
        basic_layout.addRow("Composition:", self.composition_input)

        self.composition_example_label = QLabel(COMPOSITION_EXAMPLE_TEXT)
        font_sm = self.composition_example_label.font()
        font_sm.setPointSize(8)
        self.composition_example_label.setFont(font_sm)
        self.composition_example_label.setStyleSheet("color: #666; font-weight: normal;")
        self.composition_example_label.setWordWrap(True)
        basic_layout.addRow("", self.composition_example_label)

        self.composition_input.clear()
        self.composition_input.editingFinished.connect(self.send_update)
        # Live preview: show how the composition is interpreted (with full
        # element names) so symbol mix-ups like 'C' (carbon) vs 'Co' (cobalt)
        # are caught before calculation. Unknown elements are flagged in red.
        self.composition_input.textChanged.connect(self._update_composition_preview)

        reset_btn = QPushButton("Reset")

        def reset_fields():
            self.source_type_dropdown.setCurrentText("Custom")
            self.composition_input.clear()
            self.wavelength_input.setText("0.1")
            self.background_edit.setText("")
            self.background_label.setText("No file selected")
            self.background_path = None
            self.background_enabled = False
            self.use_bg_checkbox.setEnabled(False)
            self.use_bg_checkbox.setChecked(False)

        reset_btn.clicked.connect(reset_fields)
        reset_btn.setFixedWidth(80)

        reset_layout = QHBoxLayout()
        reset_layout.addStretch()
        reset_layout.addWidget(reset_btn)
        basic_layout.addRow("", reset_layout)

        main_v_layout.addWidget(form_widget)
        main_v_layout.addStretch(1)

        return basic_tab

    def _on_background_text_changed(self, txt: str):
        self.background_path = txt.strip() or None
        has_bg = bool(self.background_path)
        self.use_bg_checkbox.setEnabled(has_bg)
        if not has_bg:
            self.background_enabled = False
            self.use_bg_checkbox.setChecked(False)

    def _on_background_editing_finished(self):
        txt = (self.background_edit.text() or "").strip()
        if txt and os.path.isfile(txt):
            # Warn if the pasted background's q-axis does not match the sample.
            if not self._confirm_background_q_axis(txt):
                # User declined: clear the background field.
                self.background_path = None
                self.background_enabled = False
                self.use_bg_checkbox.setEnabled(False)
                self.use_bg_checkbox.setChecked(False)
                self.background_edit.setText("")
                self.background_label.setText("No file selected")
                self.send_update()
                return
            self.background_path = txt
            self.background_label.setText(os.path.basename(txt))
            self.background_enabled = True
            self.use_bg_checkbox.setEnabled(True)
            self.use_bg_checkbox.setChecked(True)
        else:
            self.background_path = None
            self.background_enabled = False
            self.use_bg_checkbox.setEnabled(False)
            self.use_bg_checkbox.setChecked(False)
            if self.background_edit.text():
                self.background_edit.setText("")
            self.background_label.setText("No file selected")
        self.send_update()

    def select_background_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Background File",
            "",
            "Data Files (*.chi *.iq *.xy *.dat *.txt);;All files(*)",
        )
        if not file_path:
            return

        # Check that the background shares the sample's q-axis. Background
        # subtraction is only physically valid when the sample and background
        # were integrated with the same settings (identical q values). If they
        # differ, warn the user once here and let them decide whether to
        # continue, rather than silently producing a questionable result.
        if not self._confirm_background_q_axis(file_path):
            return

        # A new background is a new situation, so allow the calculation-level
        # warnings to be reported again for it.
        reset_warning_history()

        self.background_edit.setText(file_path)
        self.background_label.setText(os.path.basename(file_path))
        self.background_path = file_path
        self.background_enabled = True
        self.use_bg_checkbox.setEnabled(True)
        self.use_bg_checkbox.setChecked(True)
        self.send_update()

    def _path_from_item(self, obj):
        """Return a file path from a string or a QTreeWidgetItem.

        file_panel.get_selected_file_paths() returns QTreeWidgetItem objects,
        not path strings; the real path is stored under
        Qt.ItemDataRole.UserRole.
        """
        if obj is None:
            return None
        if isinstance(obj, str):
            return obj
        data_fn = getattr(obj, "data", None)
        if callable(data_fn):
            try:
                p = obj.data(0, Qt.ItemDataRole.UserRole)
                if isinstance(p, str):
                    return p
            except Exception:
                pass
        return None

    def _q_axis_mismatch(self, sample_path, bkg_path):
        """Return a short description of the q-axis mismatch, or None if OK."""
        try:
            s = extract_data(sample_path)
            b = extract_data(bkg_path)
            if s is None or b is None:
                return None
            sample_q = np.asarray(s[0], dtype=float)
            bkg_q = np.asarray(b[0], dtype=float)
        except Exception:
            return None

        if len(sample_q) != len(bkg_q):
            return f"Different number of points (sample: {len(sample_q)}, background: {len(bkg_q)})."

        max_abs_diff = float(np.max(np.abs(sample_q - bkg_q)))
        if not np.allclose(sample_q, bkg_q, rtol=1e-5, atol=1e-6):
            return f"Different q values (max difference = {max_abs_diff:.4g})."
        return None

    def _confirm_background_q_axis(self, bkg_path):
        """Compare the background q-axis with the currently selected sample.

        Returns True if it is safe to proceed (q-axes match, or the user chose
        to continue anyway, or the comparison could not be made). Returns False
        only if the user explicitly cancels after being warned.
        """
        # Find the currently selected sample file, if any.
        sample_path = None
        try:
            mw = self.main_window
            fp = getattr(mw, "file_panel", None)
            if fp is not None and hasattr(fp, "get_selected_file_paths"):
                selected = fp.get_selected_file_paths()
                if selected:
                    sample_path = self._path_from_item(selected[0])
            if sample_path is None:
                cur = getattr(mw, "current_path", None)
                if isinstance(cur, (list, tuple)) and cur:
                    sample_path = self._path_from_item(cur[0])
                else:
                    sample_path = self._path_from_item(cur)
        except Exception:
            sample_path = None

        # Without a readable sample path to compare against we cannot check;
        # allow it.
        if not sample_path or not os.path.isfile(sample_path):
            return True

        mismatch_msg = self._q_axis_mismatch(sample_path, bkg_path)
        if mismatch_msg is None:
            return True  # q-axes match; nothing to warn about.

        # This pair has now been reported, so send_update() should not warn
        # about it again.
        if not hasattr(self, "_bkg_warned_pairs"):
            self._bkg_warned_pairs = set()
        self._bkg_warned_pairs.add((sample_path, bkg_path))

        # Warn and let the user choose.
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Background q-axis does not match")
        box.setText(mismatch_msg + "\n\nSubtraction normally requires the same q values.\nUse this background anyway?")
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        return box.exec() == QMessageBox.StandardButton.Yes

    def _on_use_bg_toggled(self, checked: bool):
        self.background_enabled = bool(checked) and bool(self.background_path)
        self.send_update()

    def toggle_rgrid_fields(self, checked):
        self.rgrid_widget.setVisible(checked)

    def pdf_controls(self):
        pdf_tab = QWidget()

        # 메인 레이아웃: 수직 배치
        main_layout = QVBoxLayout(pdf_tab)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # 폼 위젯
        form_widget = QWidget()
        pdf_layout = QFormLayout(form_widget)
        pdf_layout.setContentsMargins(0, 0, 0, 0)
        pdf_layout.setVerticalSpacing(8)

        # ---------------- Checkbox 정의 ----------------
        self.new_window_checkbox = QCheckBox("Open in New Graphs")
        self.new_window_checkbox.setChecked(False)
        # -----------------------------------------------

        self.pdf_description = QLabel("For .chi, .iq, .sq, .fq, .gr, .caliq, .calsq, .calfq, .calgr files")
        f = self.pdf_description.font()
        f.setPointSize(8)
        self.pdf_description.setFont(f)
        self.pdf_description.setStyleSheet("font-weight: normal; color: #444;")
        pdf_layout.addRow(self.pdf_description)

        self.bg_slider, self.bg_input = add_slider_field(
            "Background Scale:",
            0.0,
            1.0,
            5,
            0.00000,
            pdf_layout,
            factor=10,
            with_buttons=True,
        )

        self.pdf_qstep_input = QLineEdit()
        self.pdf_qstep_input.setText("0.01")
        self.pdf_qstep_input.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.pdf_qstep_input.setPlaceholderText("0.01")
        self.pdf_qstep_input.editingFinished.connect(self.send_update)

        qstep_layout = QHBoxLayout()
        qstep_layout.addWidget(self.pdf_qstep_input, 1)

        default_label = QLabel("Default: 0.01")
        default_label.setStyleSheet("color: #666; margin-left: 5px; font-style: italic; font-weight: bold;")

        qstep_layout.addWidget(default_label, 0)

        pdf_layout.addRow("q_step:", qstep_layout)

        self.qmax_slider, self.qmax_input = add_slider_field(
            "q_max:", 0.0, 30, 3, 30.000, pdf_layout, with_buttons=True
        )
        self.qmin_slider, self.qmin_input = add_slider_field("q_min:", 0.0, 2, 3, 0.000, pdf_layout, with_buttons=True)

        self.poly_order_slider, self.poly_order_input = add_slider_field(
            "Poly Order:", 0.0, 20.0, 3, 9.000, pdf_layout, with_buttons=True
        )

        def enforce_qmin_qmax():
            qmin_val = self.qmin_slider.value() / 1000.0
            qmax_val = self.qmax_slider.value() / 1000.0
            if qmin_val > qmax_val:
                sender = self.sender()
                if sender == self.qmin_slider:
                    self.qmax_slider.setValue(int(round(qmin_val * 1000)))
                elif sender == self.qmax_slider:
                    self.qmin_slider.setValue(int(round(qmax_val * 1000)))

        self.qmax_slider.valueChanged.connect(lambda: (enforce_qmin_qmax(), update_rpoly(), self.send_update()))
        self.qmin_slider.valueChanged.connect(lambda: (enforce_qmin_qmax(), update_rpoly(), self.send_update()))

        self.bg_slider.valueChanged.connect(self.send_update)
        self.qmax_slider.valueChanged.connect(self.send_update)
        self.qmin_slider.valueChanged.connect(self.send_update)
        self.poly_order_slider.valueChanged.connect(lambda: (update_rpoly(), self.send_update()))

        self.rpoly_display = QLineEdit()
        self.rpoly_display.setReadOnly(True)
        pdf_layout.addRow("R-poly (calculated):", self.rpoly_display)

        def update_rpoly():
            qmax = self.qmax_slider.value() / 1000.0
            poly_order = self.poly_order_slider.value() / 1000.0
            if qmax > 0:
                rpoly = np.pi * poly_order / qmax
                self.rpoly_display.setText(f"{rpoly:.3f}")
            else:
                self.rpoly_display.setText("N/A")

        self.qmax_slider.valueChanged.connect(update_rpoly)
        self.poly_order_slider.valueChanged.connect(update_rpoly)
        update_rpoly()

        self.r_min_input = QLineEdit()
        pdf_layout.addRow("r_min:", self.r_min_input)

        self.r_max_input = QLineEdit()
        pdf_layout.addRow("r_max:", self.r_max_input)

        self.r_step_input = QLineEdit()
        pdf_layout.addRow("r_step:", self.r_step_input)

        self.wh_smoothing_label = QLabel("Whittaker-Henderson (WH) smoothing at F(q)")
        self.wh_smoothing_label.setStyleSheet("font-weight: bold;")
        pdf_layout.addRow(self.wh_smoothing_label)

        self.lambda_fq_input = QLineEdit()
        self.order_input = QLineEdit()

        self.lambda_label = add_form_row(pdf_layout, "lambda:", self.lambda_fq_input)
        self.order_label = add_form_row(pdf_layout, "Order:", self.order_input)

        int_validator = QDoubleValidator(0, 100000, 0)
        int_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.lambda_fq_input.setValidator(int_validator)
        self.order_input.setValidator(int_validator)

        self.r_min_input.setText("0")
        self.r_max_input.setText("30")
        self.r_step_input.setText("0.01")
        self.lambda_fq_input.setText("1000")
        self.order_input.setText("2")

        float_validator = QDoubleValidator(0.0, 9999999.0, 6)
        float_validator.setNotation(QDoubleValidator.Notation.StandardNotation)

        self.wavelength_input.setValidator(float_validator)
        self.r_min_input.setValidator(float_validator)
        self.r_max_input.setValidator(float_validator)
        self.r_step_input.setValidator(float_validator)
        self.lambda_fq_input.setValidator(float_validator)

        self.pdf_qstep_input.setValidator(float_validator)

        self.r_min_input.editingFinished.connect(self.on_gr_inputs_finished)
        self.r_max_input.editingFinished.connect(self.on_gr_inputs_finished)
        self.r_step_input.editingFinished.connect(self.on_gr_inputs_finished)
        self.lambda_fq_input.editingFinished.connect(self.on_fq_inputs_finished)
        self.order_input.editingFinished.connect(self.on_fq_inputs_finished)

        reset_btn = QPushButton("Reset")

        def reset_fields():
            self.bg_slider.setValue(int(round(0.00000 * 100000)))
            self.pdf_qstep_input.setText("0.01")
            self.qmax_slider.setValue(int(round(30.0 * 1000)))
            self.qmin_slider.setValue(int(round(0.0 * 1000)))
            self.poly_order_slider.setValue(int(round(9.000 * 1000)))
            self.lambda_fq_input.setText("1000")
            self.order_input.setText("2")
            update_rpoly()

        reset_btn.clicked.connect(reset_fields)
        reset_btn.setFixedWidth(80)
        reset_layout = QHBoxLayout()
        reset_layout.addStretch()
        reset_layout.addWidget(reset_btn)

        # Reset 버튼은 폼 레이아웃의 마지막 행에 추가
        pdf_layout.addRow("", reset_layout)

        # ---------------- 레이아웃 조합 및 바닥 배치 ----------------
        main_layout.addWidget(form_widget)
        main_layout.addStretch(1)  # 공간 채우기

        # ⬇⬇⬇ [체크박스 위치 수정: 왼쪽 정렬] ⬇⬇⬇
        cb_layout = QHBoxLayout()
        cb_layout.addWidget(self.new_window_checkbox)  # 체크박스를 먼저 추가
        cb_layout.addStretch()  # 뒤에 여백(Stretch) 추가
        # ⬆⬆⬆ [수정 완료] ⬆⬆⬆

        # 메인 레이아웃 최하단에 추가
        main_layout.addLayout(cb_layout)

        return pdf_tab

    def cal_controls(self):
        cal_tab = QWidget()
        cal_layout = QFormLayout(cal_tab)
        cal_layout.setContentsMargins(8, 8, 8, 8)
        cal_layout.setVerticalSpacing(8)

        self.cal_description = QLabel("For .xyz files")
        f = self.cal_description.font()
        f.setPointSize(8)
        self.cal_description.setFont(f)
        self.cal_description.setStyleSheet("font-weight: normal; color: #444;")
        cal_layout.addRow(self.cal_description)

        self.cal_qmax_slider, self.cal_qmax_input = add_slider_field(
            "q_max:", 0.0, 30, 3, 30.000, cal_layout, with_buttons=True
        )
        self.cal_qmin_slider, self.cal_qmin_input = add_slider_field("q_min:", 0.0, 5, 3, 0.000, cal_layout)

        self.cal_qmax_slider.valueChanged.connect(self.send_update)
        self.cal_qmin_slider.valueChanged.connect(self.send_update)

        self.cal_r_min_input = QLineEdit()
        cal_layout.addRow("r_min:", self.cal_r_min_input)

        self.cal_r_max_input = QLineEdit()
        cal_layout.addRow("r_max:", self.cal_r_max_input)

        self.cal_r_step_input = QLineEdit()
        cal_layout.addRow("r_step:", self.cal_r_step_input)

        float_validator = QDoubleValidator(0.0, 9999999.0, 6)
        float_validator.setNotation(QDoubleValidator.Notation.StandardNotation)

        self.cal_r_min_input.setValidator(float_validator)
        self.cal_r_max_input.setValidator(float_validator)
        self.cal_r_step_input.setValidator(float_validator)

        self.cal_r_min_input.setText("0")
        self.cal_r_max_input.setText("30")
        self.cal_r_step_input.setText("0.01")

        self.cal_r_min_input.editingFinished.connect(self.on_gr_inputs_finished)
        self.cal_r_max_input.editingFinished.connect(self.on_gr_inputs_finished)
        self.cal_r_step_input.editingFinished.connect(self.on_gr_inputs_finished)

        reset_btn = QPushButton("Reset")

        def reset_fields():
            self.cal_qmax_slider.setValue(int(round(30.0 * 1000)))
            self.cal_qmin_slider.setValue(int(round(0.0 * 1000)))
            self.cal_r_min_input.setText("0")
            self.cal_r_max_input.setText("30")
            self.cal_r_step_input.setText("0.01")

        reset_btn.clicked.connect(reset_fields)
        reset_btn.setFixedWidth(80)
        reset_layout = QHBoxLayout()
        reset_layout.addStretch()
        reset_layout.addWidget(reset_btn)
        cal_layout.addRow("", reset_layout)

        return cal_tab

    def compton_controls(self):
        compton_tab = QWidget()
        compton_layout = QFormLayout(compton_tab)
        compton_layout.setContentsMargins(8, 8, 8, 8)
        compton_layout.setVerticalSpacing(8)

        self.compton_wavelength_input = QLineEdit()
        self.compton_wavelength_input.setPlaceholderText("Enter wavelength")
        self.compton_wavelength_input.setText("0.1665")
        compton_layout.addRow("Wavelength:", self.compton_wavelength_input)

        self.alpha_dropdown = QComboBox()
        self.alpha_dropdown.addItems(["2", "3"])
        self.alpha_dropdown.setCurrentText("3")
        compton_layout.addRow("Alpha:", self.alpha_dropdown)

        self.compton_composition_input = QLineEdit()
        self.compton_composition_input.setPlaceholderText("Enter composition (e.g. Co 2 O 2 P 1  or  Co2O2P)")
        self.compton_composition_input.setText("")
        compton_layout.addRow("Composition:", self.compton_composition_input)

        self.compton_composition_example_label = QLabel(COMPOSITION_EXAMPLE_TEXT)
        f = self.compton_composition_example_label.font()
        f.setPointSize(8)
        self.compton_composition_example_label.setFont(f)
        self.compton_composition_example_label.setStyleSheet("font-weight: normal; color: #666;")
        self.compton_composition_example_label.setWordWrap(True)
        compton_layout.addRow("", self.compton_composition_example_label)

        # Live preview for the Compton-tab composition field (its own label).
        self.compton_composition_input.textChanged.connect(self._update_compton_composition_preview)

        self.compton_qmax_slider, self.compton_qmax_input = add_slider_field(
            "q_max:", 0.0, 40, 3, 30.000, compton_layout, with_buttons=True
        )
        self.compton_qmin_slider, self.compton_qmin_input = add_slider_field("q_min:", 0.0, 5, 3, 0.000, compton_layout)

        self.compton_qmax_slider.valueChanged.connect(self.send_update)
        self.compton_qmin_slider.valueChanged.connect(self.send_update)

        self.compton_qstep_input = QLineEdit()
        self.compton_qstep_input.setText("0.01")
        compton_layout.addRow("q_step:", self.compton_qstep_input)

        float_validator = QDoubleValidator(0.0, 9999999.0, 6)
        float_validator.setNotation(QDoubleValidator.Notation.StandardNotation)

        self.compton_wavelength_input.setValidator(float_validator)
        self.compton_qmin_input.setValidator(float_validator)
        self.compton_qmax_input.setValidator(float_validator)
        self.compton_qstep_input.setValidator(float_validator)

        self.compton_wavelength_input.editingFinished.connect(self.send_update)
        self.alpha_dropdown.currentIndexChanged.connect(self.send_update)
        self.compton_composition_input.editingFinished.connect(self.send_update)
        self.compton_qmin_input.editingFinished.connect(self.send_update)
        self.compton_qmax_input.editingFinished.connect(self.send_update)
        self.compton_qstep_input.editingFinished.connect(self.send_update)

        reset_btn = QPushButton("Reset")

        def reset_fields():
            self.compton_wavelength_input.setText("0.1665")
            self.alpha_dropdown.setCurrentText("3")
            self.compton_composition_input.setText("")
            self.compton_qmin_slider.setValue(int(round(0.0 * 1000)))
            self.compton_qmax_slider.setValue(int(round(30.0 * 1000)))
            self.compton_qstep_input.setText("0.01")

        reset_btn.clicked.connect(reset_fields)
        reset_btn.setFixedWidth(80)

        reset_layout = QHBoxLayout()
        reset_layout.addStretch()
        reset_layout.addWidget(reset_btn)

        calculate_btn = QPushButton("Calculate Compton")
        calculate_btn.setFixedWidth(160)

        def run_compton_calculation():
            try:
                # If an .xyz file is selected and the composition field no
                # longer matches it, ask whether to use the file or the typed
                # composition. Returns False if the user cancels.
                if not self._resolve_compton_composition_source():
                    return
                # Compton scattering is defined for neutral atoms only.
                ions = self._find_ion_symbols(self.compton_composition_input.text())
                if ions:
                    shown = ", ".join(ions[:8]) + (" ..." if len(ions) > 8 else "")
                    QMessageBox.warning(
                        self,
                        "Ions in composition",
                        f"The composition contains ions ({shown}).\n\n"
                        "Compton scattering cannot use ions \u2014 please enter "
                        "neutral element symbols only (e.g. 'Fe' not 'Fe2+', "
                        "'O' not 'O2-').",
                    )
                    return
                calculate_compton(self)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Compton Calculation Error",
                    f"An error occurred while calculating Compton scattering:\n{str(e)}",
                )

        calculate_btn.clicked.connect(run_compton_calculation)

        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        buttons_layout.addWidget(calculate_btn)
        buttons_layout.addWidget(reset_btn)
        compton_layout.addRow("", buttons_layout)

        return compton_tab

    def get_pdf_parameters(self):
        return {
            "bg": self.bg_slider.value() / 100000,
            "qstep": self.pdf_qstep_input.text(),
            "qmax": self.qmax_slider.value() / 1000,
            "qmin": self.qmin_slider.value() / 1000,
            "poly_order": self.poly_order_input.text(),
            "lambda": self.lambda_fq_input.text(),
            "order": self.order_input.text(),
            "wh_enabled": True,
        }

    def _apply_composition_preview(self, line_edit, label):
        """Shared logic: read `line_edit`, update `label` with a live,.

        human-readable interpretation. Valid compositions are shown in grey with
        full element names; unknown elements are shown in red. An empty field
        restores the original example hint.
        """
        text = line_edit.text() if line_edit is not None else ""

        if text is None or text.strip() == "":
            label.setText(COMPOSITION_EXAMPLE_TEXT)
            label.setStyleSheet("color: #666; font-weight: normal;")
            return

        try:
            result = preview_composition(text)
        except Exception:
            # Never let a preview failure interfere with normal input.
            label.setText(COMPOSITION_EXAMPLE_TEXT)
            label.setStyleSheet("color: #666; font-weight: normal;")
            return

        if result["ok"]:
            label.setText("\u2192 " + result["message"])
            label.setStyleSheet("color: #666; font-weight: normal;")
        else:
            label.setText("\u26a0 " + result["message"])
            label.setStyleSheet("color: #c0392b; font-weight: normal;")

    def _update_composition_preview(self, text=None):
        """Preview for the Basic-tab composition field."""
        self._apply_composition_preview(self.composition_input, self.composition_example_label)

    def _update_compton_composition_preview(self, text=None):
        """Preview for the Compton-tab composition field.

        If the composition contains ions, show a Compton-specific message
        (ions aren't allowed) instead of the generic 'cannot read' parse error,
        since Compton scattering is defined for neutral atoms only.
        """
        ions = self._find_ion_symbols(self.compton_composition_input.text())
        if ions:
            shown = ", ".join(ions[:5]) + (" ..." if len(ions) > 5 else "")
            label = self.compton_composition_example_label
            label.setText(
                "\u26a0 Compton cannot use ions \u2014 use neutral atoms only (e.g. 'Fe' not 'Fe2+'). Found: " + shown
            )
            label.setStyleSheet("color: #c0392b; font-weight: normal;")
            return
        self._apply_composition_preview(self.compton_composition_input, self.compton_composition_example_label)

    @staticmethod
    def _find_ion_symbols(text):
        """Return the ionic species (e.g. 'Fe2+', 'O2-') found in a composition.

        string. Neutral compositions contain no charge signs, so this is empty
        for them.
        """
        if not text:
            return []
        # Element symbol + optional oxidation number + charge sign.
        ions = re.findall(r"[A-Z][a-z]?\d*[+-]", text)
        # De-duplicate while preserving order.
        seen = set()
        unique = []
        for ion in ions:
            if ion not in seen:
                seen.add(ion)
                unique.append(ion)
        return unique

    def _on_control_tab_changed(self, index):
        """When the Compton tab is opened, warn if the composition has ions."""
        if getattr(self, "control_tabs", None) is None:
            return
        if self.control_tabs.widget(index) is getattr(self, "compton_tab_widget", None):
            self._warn_if_compton_composition_has_ions()

    def _warn_if_compton_composition_has_ions(self):
        """Show a warning if the Compton composition contains ionic species.

        Compton scattering is defined for neutral atoms only.
        """
        ions = self._find_ion_symbols(self.compton_composition_input.text())
        if not ions:
            return
        QMessageBox.warning(
            self,
            "Ions in composition",
            "The composition contains ionic species: {}.\n\n"
            "Compton scattering is defined for neutral atoms only, so ions "
            "cannot be used here. Please enter neutral element symbols "
            "(e.g. 'Fe' instead of 'Fe2+', 'O' instead of 'O2-') for the "
            "Compton calculation.".format(", ".join(ions)),
        )

    def _selected_xyz_path(self):
        """Return the path of the single selected .xyz file, or None.

        None is returned when the file panel is unavailable, nothing (or more
        than one file) is selected, or the selection is not an .xyz file.
        """
        mw = getattr(self, "main_window", None)
        panel = getattr(mw, "file_panel", None) if mw else None
        if panel is None:
            return None
        try:
            items = panel.get_selected_file_paths()
        except Exception:
            return None
        if not items or len(items) != 1:
            return None
        try:
            path = items[0].data(0, Qt.ItemDataRole.UserRole)
        except Exception:
            return None
        if not path or os.path.splitext(path)[1].lower() != ".xyz":
            return None
        return path

    @staticmethod
    def _compositions_equal(a, b):
        """True if two composition strings have the same elements and amounts.

        Whitespace and writing style are ignored (both are parsed first), so
        'Co2O3P1' and 'Co 2 O 3 P 1' are equal. Unlike a normalised compare,
        a scaled edit such as 'Co4O6P2' counts as *different*, so changing the
        numbers always prompts the user.
        """
        try:
            da = parse_composition(a)
            db = parse_composition(b)
        except Exception:
            return False
        if set(da.keys()) != set(db.keys()):
            return False
        for el in da:
            if abs(float(da[el]) - float(db[el])) > 1e-9:
                return False
        return True

    def _resolve_compton_composition_source(self):
        """Decide which composition the Compton calculation should use.

        The reference is the .xyz file that auto-filled the composition: the
        one currently selected, or failing that the last one remembered (so a
        deselected file still prompts). Behaviour:

        - No reference .xyz: use the composition field as-is.
        - Field empty: fill it from the file.
        - Field matches the file: use it (no prompt).
        - Field differs: ask whether to compute from the file or from the
          typed composition.

        Returns True to proceed, False if the user cancels.
        """
        xyz_path = self._selected_xyz_path()
        if not xyz_path:
            remembered = getattr(self, "_compton_xyz_path", None)
            if remembered and os.path.isfile(remembered):
                xyz_path = remembered
        if not xyz_path:
            return True

        xyz_comp = composition_string_from_xyz(xyz_path)
        if not xyz_comp:
            # Couldn't read a composition from the file; fall back to the field.
            return True

        field_text = self.compton_composition_input.text().strip()

        if not field_text:
            self.compton_composition_input.setText(xyz_comp)
            self._compton_xyz_path = xyz_path
            self._compton_xyz_comp = xyz_comp
            return True

        if self._compositions_equal(field_text, xyz_comp):
            return True

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Which composition to use?")
        box.setText(
            "The composition differs from the selected .xyz file.\n\n"
            f"File:  {os.path.basename(xyz_path)}  →  {xyz_comp}\n"
            f"Entered:  {field_text}\n\n"
            "Calculate from the .xyz file or from the composition you entered?"
        )
        from_file_btn = box.addButton("From .xyz file", QMessageBox.ButtonRole.AcceptRole)
        from_input_btn = box.addButton("From entered composition", QMessageBox.ButtonRole.AcceptRole)
        cancel_btn = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(from_input_btn)
        box.exec()

        clicked = box.clickedButton()
        if clicked is cancel_btn:
            return False
        if clicked is from_file_btn:
            self.compton_composition_input.setText(xyz_comp)
            self._compton_xyz_path = xyz_path
            self._compton_xyz_comp = xyz_comp
        else:
            # User chose their own composition: stop prompting for this file.
            self._compton_xyz_path = None
            self._compton_xyz_comp = None
        return True

    def get_basic_parameters(self):
        if self.format_2theta.isChecked():
            data_format = "2theta"
        elif self.format_q_invA.isChecked():
            data_format = "q_invA"
        else:
            data_format = "q_nmn"

        background_path = getattr(self, "background_path", None)

        comp_text = (self.composition_input.text() or "").strip() or "C 1 O 2"

        return {
            "data_format": data_format,
            "source_type": self.source_type_dropdown.currentText(),
            "wavelength": self.wavelength_input.text(),
            "background_file": background_path,
            "background_enabled": bool(getattr(self, "background_enabled", False)),
            "composition": comp_text,
            "rmin": self.r_min_input.text(),
            "rmax": self.r_max_input.text(),
            "rstep": self.r_step_input.text(),
        }

    def get_cal_parameters(self):
        try:
            return {
                "qmax": self.cal_qmax_input.text(),
                "qmin": self.cal_qmin_input.text(),
                "rmin": self.cal_r_min_input.text(),
                "rmax": self.cal_r_max_input.text(),
                "rstep": self.cal_r_step_input.text(),
            }
        except AttributeError as e:
            print(f"[Cal] Error getting input: {e}")
            return {}

    def get_compton_parameters(self):
        return {
            "wavelength": self.compton_wavelength_input.text(),
            "alpha": int(self.alpha_dropdown.currentText()),
            "composition": self.compton_composition_input.text(),
            "qmin": str(self.compton_qmin_slider.value() / 1000),
            "qmax": str(self.compton_qmax_slider.value() / 1000),
            "qstep": self.compton_qstep_input.text(),
        }

    def send_update(self):
        windows = getattr(self.main_window, "plot_windows", [])
        active_windows = [w for w in windows if w.isVisible()]
        self.main_window.plot_windows = active_windows

        for window in active_windows:
            items = getattr(window, "associated_items", None)
            if items:
                # Warn once if this window's sample does not share the
                # background's q-axis. Checking here (rather than only when the
                # background is picked) catches the case where the background
                # was chosen first and a mismatching sample was opened later.
                self._warn_if_bkg_q_mismatch(items)
                update_current_graph(items, self, window)

    def _warn_if_bkg_q_mismatch(self, items):
        """Show the q-axis warning once for each sample/background pair."""
        if not self.background_enabled or not self.background_path:
            return
        try:
            sample_path = self._path_from_item(items[0] if isinstance(items, (list, tuple)) else items)
        except Exception:
            return
        if not sample_path or not os.path.isfile(sample_path):
            return

        pair = (sample_path, self.background_path)
        if not hasattr(self, "_bkg_warned_pairs"):
            self._bkg_warned_pairs = set()
        if pair in self._bkg_warned_pairs:
            return  # already warned about this combination
        self._bkg_warned_pairs.add(pair)

        mismatch = self._q_axis_mismatch(sample_path, self.background_path)
        if mismatch:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Background q-axis does not match")
            box.setText(
                mismatch + "\n\n"
                "Subtraction normally requires the same q values.\n"
                "The background is interpolated onto the sample q-grid; "
                "please check the result."
            )
            box.setStandardButtons(QMessageBox.StandardButton.Ok)
            box.exec()

    def set_basic_parameters(self, params: dict):
        data_format = params.get("data_format", "2theta")
        if data_format == "2theta":
            self.format_2theta.setChecked(True)
        elif data_format == "q_invA":
            self.format_q_invA.setChecked(True)
        elif data_format == "q_nmn":
            self.format_q_nmn.setChecked(True)

        source_type = params.get("source_type", "Custom")
        index = self.source_type_dropdown.findText(source_type)
        if index != -1:
            self.source_type_dropdown.setCurrentIndex(index)

        self.wavelength_input.setText(params.get("wavelength", "0.1"))

        background_file = params.get("background_file")
        if background_file:
            self.background_label.setText(os.path.basename(background_file))
            self.background_edit.setText(background_file)
            self.background_path = background_file
            self.background_enabled = bool(params.get("background_enabled", True))
            self.use_bg_checkbox.setEnabled(True)
            self.use_bg_checkbox.setChecked(self.background_enabled)
        else:
            self.background_label.setText("No file selected")
            self.background_edit.setText("")
            self.background_path = None
            self.background_enabled = False
            self.use_bg_checkbox.setEnabled(False)
            self.use_bg_checkbox.setChecked(False)

        self.composition_input.setText(params.get("composition", ""))
        self.r_min_input.setText(params.get("rmin", "0"))
        self.r_max_input.setText(params.get("rmax", "30"))
        self.r_step_input.setText(params.get("rstep", "0.01"))

    def set_pdf_parameters(self, params: dict):
        # [안전장치 2: 강력한 상태 보호]
        # 이 함수가 호출되기 전(Optimizer가 값을 바꾸기 전) 상태를 저장
        was_checked = self.new_window_checkbox.isChecked()

        current_bg = self.bg_slider.value() / 100000.0
        bg = float(params["bg"]) if "bg" in params else current_bg
        self.bg_slider.setValue(int(round(bg * 100000)))

        self.pdf_qstep_input.setText(str(params["qstep"]) if "qstep" in params else self.pdf_qstep_input.text())

        current_qmax = self.qmax_slider.value() / 1000.0
        qmax = float(params["qmax"]) if "qmax" in params else current_qmax
        self.qmax_slider.setValue(int(round(qmax * 1000)))

        current_qmin = self.qmin_slider.value() / 1000.0
        qmin = float(params["qmin"]) if "qmin" in params else current_qmin
        self.qmin_slider.setValue(int(round(qmin * 1000)))

        poly_text = str(params["poly_order"]) if "poly_order" in params else self.poly_order_input.text()
        self.poly_order_input.setText(poly_text)
        try:
            self.poly_order_slider.setValue(int(round(float(poly_text) * 1000)))
        except Exception:
            pass

        self.lambda_fq_input.setText(str(params["lambda"]) if "lambda" in params else self.lambda_fq_input.text())
        self.order_input.setText(str(params["order"]) if "order" in params else self.order_input.text())

        # [안전장치 2 복구]
        # 어떤 값이 들어왔든 상관없이, 사용자가 설정한 '새 창 열기' 옵션은 강제로 복구함
        self.new_window_checkbox.setChecked(was_checked)

    def set_cal_parameters(self, params: dict):
        qmax = float(params.get("qmax", 30.0))
        self.cal_qmax_slider.setValue(int(round(qmax * 1000)))

        qmin = float(params.get("qmin", 0.1))
        self.cal_qmin_slider.setValue(int(round(qmin * 1000)))

        self.cal_r_min_input.setText(params.get("rmin", "0"))
        self.cal_r_max_input.setText(params.get("rmax", "30"))
        self.r_step_input.setText(params.get("rstep", "0.01"))

    def set_compton_parameters(self, params: dict):
        try:
            self.compton_wavelength_input.setText(params.get("wavelength", "0.1665"))
            self.alpha_dropdown.setCurrentText(str(params.get("alpha", "3")))
            self.compton_composition_input.setText(params.get("composition", ""))
            qmin = float(params.get("qmin", 0.0))
            self.compton_qmin_slider.setValue(int(round(qmin * 1000)))
            qmax = float(params.get("qmax", 30.0))
            self.compton_qmax_slider.setValue(int(round(qmax * 1000)))
            self.compton_qstep_input.setText(params.get("qstep", "0.01"))
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not set all Compton parameters from project file. {e}")

    def enable_graphs(self, enable_fq=None, enable_gr=None):
        if self.main_window.plot_window:
            self.main_window.plot_window.enable_graphs(enable_fq=enable_fq, enable_gr=enable_gr)

    def on_fq_inputs_finished(self):
        self.send_update()
        self.enable_graphs(enable_fq=True, enable_gr=True)

    def on_gr_inputs_finished(self):
        self.send_update()
        self.enable_graphs(enable_gr=True)
