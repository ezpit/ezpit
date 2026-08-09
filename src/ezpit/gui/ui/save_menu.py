import os

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ezpit.gui.model.saver import (
    save_calculated_graph_data,
    save_graph_data,
    save_selected_graphs,
)


class SaveMenu(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Save Menu")
        self.parent_window = parent
        self._settings = QSettings("EZPDF", "EZPDF")

        self.cb_iq_img = QCheckBox("Save I(q) image")
        self.cb_iq_img.setChecked(False)
        self.cb_sq_img = QCheckBox("Save S(q) image")
        self.cb_sq_img.setChecked(False)
        self.cb_fq_img = QCheckBox("Save F(q) image")
        self.cb_fq_img.setChecked(False)
        self.cb_gr_img = QCheckBox("Save G(r) image")
        self.cb_gr_img.setChecked(False)

        if not self._has_data(0):
            self.cb_iq_img.hide()
        if not self._has_data(1):
            self.cb_sq_img.hide()
        if not self._has_data(2):
            self.cb_fq_img.hide()
        if not self._has_data(3):
            self.cb_gr_img.hide()

        self.save_images_button = QPushButton("Save Selected Images")
        self.save_images_button.clicked.connect(self.export_images)

        self.save_files_button = QPushButton("Save Graph Data Files")
        self.save_files_button.clicked.connect(self.export_files)

        layout = QVBoxLayout()
        layout.addWidget(self.cb_iq_img)
        layout.addWidget(self.cb_sq_img)
        layout.addWidget(self.cb_fq_img)
        layout.addWidget(self.cb_gr_img)
        layout.addWidget(self.save_images_button)
        layout.addWidget(self.save_files_button)
        self.setLayout(layout)

    def _has_data(self, index):
        try:
            xs = self.parent_window.xs[index]
            ys = self.parent_window.ys[index]
            return xs is not None and ys is not None and len(xs) > 0 and len(ys) > 0
        except (IndexError, AttributeError, TypeError):
            return False

    def export_images(self):
        selected_plots = []
        selected_checkboxes = []

        if self.cb_iq_img.isVisible() and self.cb_iq_img.isChecked():
            selected_checkboxes.append(self.parent_window.plot_checkboxes[0])
            selected_plots.append(self.parent_window.plots[0])
        if self.cb_sq_img.isVisible() and self.cb_sq_img.isChecked():
            selected_checkboxes.append(self.parent_window.plot_checkboxes[1])
            selected_plots.append(self.parent_window.plots[1])
        if self.cb_fq_img.isVisible() and self.cb_fq_img.isChecked():
            selected_checkboxes.append(self.parent_window.plot_checkboxes[2])
            selected_plots.append(self.parent_window.plots[2])
        if self.cb_gr_img.isVisible() and self.cb_gr_img.isChecked():
            selected_checkboxes.append(self.parent_window.plot_checkboxes[3])
            selected_plots.append(self.parent_window.plots[3])

        last_dir = self._settings.value("last_dir", "")
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder to Save Graph Images", last_dir)
        if not folder_path:
            return
        self._settings.setValue("last_dir", folder_path)
        self._settings.sync()

        if selected_plots:
            # 순수 파일명만 추출 (경로 제외)
            file_name_only = os.path.basename(self.parent_window.file_name)
            base_name = os.path.splitext(file_name_only)[0]

            save_selected_graphs(
                self.parent_window,
                selected_checkboxes,
                selected_plots,
                folder_path,
                base_name=base_name,
            )

    def export_files(self):
        last_dir = self._settings.value("last_dir", "")
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder to Save Graph Data", last_dir)
        if not folder_path:
            return
        self._settings.setValue("last_dir", folder_path)
        self._settings.sync()

        # [수정] 전체 경로에서 '순수 파일명'만 추출 (예: C:/data/image-0.chi -> image-0.chi)
        full_path = self.parent_window.file_name
        file_name_only = os.path.basename(full_path)

        # [수정] 확장자만 제거하고 이름 그대로 사용 (예: image-0.chi -> image-0)
        base_name, ext = os.path.splitext(file_name_only)

        try:
            if ext.lower() in [".caliq", ".calsq", ".calfq", ".calgr", ".xyz"]:
                save_calculated_graph_data(
                    xs=self.parent_window.xs,
                    ys=self.parent_window.ys,
                    folder_name=base_name,
                    path=folder_path,
                    extensions_to_save=[".caliq", ".calsq", ".calfq", ".calgr"],
                )
            else:
                save_graph_data(
                    xs=self.parent_window.xs,
                    ys=self.parent_window.ys,
                    raw_iq=self.parent_window.raw_iq,
                    background_data=self.parent_window.background_data,
                    sq_original_data=self.parent_window.sq_original_data,
                    sq_polynomial_data=self.parent_window.sq_polynomial_data,
                    fq_smoothed_data=self.parent_window.fq_smoothed_data,
                    mean_sq_fi=self.parent_window.mean_sq_fi,
                    sq_mean_fi=self.parent_window.sq_mean_fi,
                    r_smoothed_data=self.parent_window.gr_smoothed_data[0],
                    G_smoothed_data=self.parent_window.gr_smoothed_data[1],
                    folder_name=base_name,
                    path=folder_path,
                )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Error",
                f"An error occurred while exporting files:\n{str(e)}",
            )
