# ui/control_panel.py
import os
import numpy as np

# [중요] Qt 모듈 임포트
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QTabWidget, QLineEdit,
    QPushButton, QLabel, QHBoxLayout, QMessageBox,
    QButtonGroup, QComboBox, QRadioButton, QFileDialog, QCheckBox, QStyle
)
from PySide6.QtGui import QDoubleValidator, QFont
from .ui_helpers import add_slider_field, add_form_row
from ezpit.gui.model.helpers import preview_composition
from ezpit.gui.controller.graph_controller import update_current_graph, calculate_compton


class ControlPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.main_window = parent

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
        tabs.addTab(self.compton_controls(), "Compton")

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
        self.bg_browse_action = self.background_edit.addAction(
            icon, QLineEdit.ActionPosition.TrailingPosition
        )
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
        self.composition_input.setPlaceholderText("Enter composition (e.g. C 1 O 2)")
        basic_layout.addRow("Composition:", self.composition_input)

        self.composition_example_label = QLabel("Example: C 1 O 2 P 5")
        font_sm = self.composition_example_label.font()
        font_sm.setPointSize(8)
        self.composition_example_label.setFont(font_sm)
        self.composition_example_label.setStyleSheet("color: #666; font-weight: normal;")
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
            self, "Select Background File", "",
            "CHI files (*.chi);;All files(*)"
        )
        if not file_path:
            return
        self.background_edit.setText(file_path)
        self.background_label.setText(os.path.basename(file_path))
        self.background_path = file_path
        self.background_enabled = True
        self.use_bg_checkbox.setEnabled(True)
        self.use_bg_checkbox.setChecked(True)
        self.send_update()

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

        self.bg_slider, self.bg_input = add_slider_field("Background Scale:", 0.0, 1.0, 5, 0.00000, pdf_layout,
                                                         factor=10,
                                                         with_buttons=True)

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

        self.qmax_slider, self.qmax_input = add_slider_field("q_max:", 0.0, 30, 3, 30.000, pdf_layout,
                                                             with_buttons=True)
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

        self.wh_smoothing_label = QLabel(
            "Whittaker-Henderson (WH) smoothing at F(q)")
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
        cb_layout.addWidget(self.new_window_checkbox) # 체크박스를 먼저 추가
        cb_layout.addStretch()                        # 뒤에 여백(Stretch) 추가
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
        self.cal_qmin_slider, self.cal_qmin_input = add_slider_field(
            "q_min:", 0.0, 5, 3, 0.000, cal_layout
        )

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
        self.compton_composition_input.setPlaceholderText("Enter composition (e.g. Co 2 O 2 P 1)")
        self.compton_composition_input.setText("")
        compton_layout.addRow("Composition:", self.compton_composition_input)

        self.compton_composition_example_label = QLabel("Example: C 1 O 2 P 5")
        f = self.compton_composition_example_label.font()
        f.setPointSize(8)
        self.compton_composition_example_label.setFont(f)
        self.compton_composition_example_label.setStyleSheet("font-weight: normal; color: #666;")
        compton_layout.addRow("", self.compton_composition_example_label)

        # Live preview for the Compton-tab composition field (its own label).
        self.compton_composition_input.textChanged.connect(
            self._update_compton_composition_preview)

        self.compton_qmax_slider, self.compton_qmax_input = add_slider_field(
            "q_max:", 0.0, 40, 3, 30.000, compton_layout, with_buttons=True)
        self.compton_qmin_slider, self.compton_qmin_input = add_slider_field(
            "q_min:", 0.0, 5, 3, 0.000, compton_layout)

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
                calculate_compton(self)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Compton Calculation Error",
                    f"An error occurred while calculating Compton scattering:\n{str(e)}")

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
            "wh_enabled": True
        }

    def _apply_composition_preview(self, line_edit, label):
        """Shared logic: read `line_edit`, update `label` with a live,
        human-readable interpretation. Valid compositions are shown in grey with
        full element names; unknown elements are shown in red. An empty field
        restores the original example hint."""
        text = line_edit.text() if line_edit is not None else ""

        if text is None or text.strip() == "":
            label.setText("Example: C 1 O 2 P 5")
            label.setStyleSheet("color: #666; font-weight: normal;")
            return

        try:
            result = preview_composition(text)
        except Exception:
            # Never let a preview failure interfere with normal input.
            label.setText("Example: C 1 O 2 P 5")
            label.setStyleSheet("color: #666; font-weight: normal;")
            return

        if result['ok']:
            label.setText("\u2192 " + result['message'])
            label.setStyleSheet("color: #666; font-weight: normal;")
        else:
            label.setText("\u26a0 " + result['message'])
            label.setStyleSheet("color: #c0392b; font-weight: normal;")

    def _update_composition_preview(self, text=None):
        """Preview for the Basic-tab composition field."""
        self._apply_composition_preview(self.composition_input,
                                        self.composition_example_label)

    def _update_compton_composition_preview(self, text=None):
        """Preview for the Compton-tab composition field."""
        self._apply_composition_preview(self.compton_composition_input,
                                        self.compton_composition_example_label)

    def get_basic_parameters(self):
        if self.format_2theta.isChecked():
            data_format = "2theta"
        elif self.format_q_invA.isChecked():
            data_format = "q_invA"
        else:
            data_format = "q_nmn"

        background_path = getattr(self, 'background_path', None)

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
            "rstep": self.r_step_input.text()
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
            "qstep": self.compton_qstep_input.text()
        }

    def send_update(self):
        windows = getattr(self.main_window, "plot_windows", [])
        active_windows = [w for w in windows if w.isVisible()]
        self.main_window.plot_windows = active_windows

        for window in active_windows:
            items = getattr(window, "associated_items", None)
            if items:
                update_current_graph(items, self, window)

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




