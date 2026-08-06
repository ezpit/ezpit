# ui/ui_helpers.py
from PySide6.QtWidgets import (
    QSlider, QLineEdit, QHBoxLayout, QLabel, QPushButton, QFormLayout, QSizePolicy
)
from PySide6.QtCore import Qt, QSize


def get_short_name(filename, max_len):
    if not max_len or max_len <= 0:
        return filename
    if len(filename) > max_len:
        return filename[:max_len] + "..."
    return filename


def update_file_list_numbering(file_list):
    for i in range(file_list.topLevelItemCount()):
        item = file_list.topLevelItem(i)
        item.setText(0, str(i + 1))


def add_slider_field(label, min_val, max_val, decimals, default_val, layout, factor=2.0, with_buttons=False):
    scale = 10 ** decimals
    current_max = [max_val]

    def to_slider_val(real_val):
        return int(real_val * scale)

    def from_slider_val(slider_val):
        return slider_val / scale

    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setMinimum(to_slider_val(min_val))
    slider.setMaximum(to_slider_val(current_max[0]))
    slider.setValue(to_slider_val(default_val))
    slider.setSingleStep(1)
    slider.setTickInterval(1)

    # [수정] 슬라이더 높이 제한 제거 및 최소 높이 확보 (잘림 방지)
    slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    slider.setMinimumHeight(22)  # 핸들(14px) + 여유 공간

    # [수정] 입력창 크기 60, 우측 정렬
    line_edit = QLineEdit(f"{default_val:.{decimals}f}")
    line_edit.setFixedWidth(60)
    line_edit.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    line_edit.setAlignment(Qt.AlignmentFlag.AlignRight)

    min_label = QLabel(f"{min_val}")
    max_label = QLabel(f"{current_max[0]:.{decimals}f}")
    min_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    max_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def slider_to_input(val):
        line_edit.setText(f"{from_slider_val(val):.{decimals}f}")

    slider.valueChanged.connect(slider_to_input)

    def update_slider():
        try:
            val = float(line_edit.text())
            slider.setValue(to_slider_val(val))
        except ValueError:
            pass

    line_edit.editingFinished.connect(update_slider)

    row_layout = QHBoxLayout()
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(4)

    row_layout.addWidget(min_label)
    row_layout.addWidget(slider, 1)
    row_layout.addWidget(max_label)

    if with_buttons:
        minus_button = QPushButton("−")
        plus_button = QPushButton("+")

        minus_button.setFixedSize(20, 20)
        plus_button.setFixedSize(20, 20)
        btn_style = "QPushButton { padding: 0px; margin: 0px; font-weight: bold; }"
        minus_button.setStyleSheet(btn_style)
        plus_button.setStyleSheet(btn_style)

        def update_max(multiplier):
            new_max = current_max[0] * multiplier
            new_max = max(min_val + 1 / scale, new_max)
            current_max[0] = new_max
            slider.setMaximum(to_slider_val(new_max))
            max_label.setText(f"{new_max:.{decimals}f}")
            if from_slider_val(slider.value()) > new_max:
                slider.setValue(to_slider_val(new_max))

        minus_button.clicked.connect(lambda: update_max(1 / factor))
        plus_button.clicked.connect(lambda: update_max(factor))

        row_layout.addWidget(minus_button)
        row_layout.addWidget(plus_button)

    row_layout.addWidget(line_edit)
    layout.addRow(label, row_layout)

    return slider, line_edit


def add_form_row(layout: QFormLayout, label_text: str, input_field: QLineEdit) -> QLabel:
    label = QLabel(label_text)
    layout.addRow(label, input_field)
    return label

