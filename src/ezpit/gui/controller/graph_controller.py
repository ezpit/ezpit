# controller/graph_controller.py

from PySide6.QtWidgets import QTreeWidgetItem, QMessageBox
from PySide6.QtCore import Qt
import os
import numpy as np

from ezpit.gui.ui.ui_helpers import get_short_name
from ezpit.gui.ui.viewer import PlotWindow
from ezpit.gui.model.dataloader import get_graph_data, get_config_data
from ezpit.gui.model.processing import get_compton_values
from ezpit.gui.model.saver import write_compton_file
from .utils import apply_optimal_pdf_settings

VALID_EXTENSIONS = [
    ".chi", ".gr", ".sq", "fq", ".iq", ".xyz",
    ".caliq", ".calsq", ".calfq", ".calgr", ".compton"
]


def load_selected_files(selected_items, label_widget, plot_window_ref, control_panel, max_length_file_name):
    if not selected_items:
        label_widget.setText("No file selected.")
        if plot_window_ref and hasattr(plot_window_ref, 'close'):
            plot_window_ref.close()
        return None

    paths = [item.data(0, Qt.ItemDataRole.UserRole) for item in selected_items]
    extensions = [os.path.splitext(p)[1].lower() for p in paths]

    try:
        if len(paths) == 1:
            path = paths[0]
            ext = extensions[0]
            short_name = get_short_name(os.path.basename(path), max_length_file_name)

            xs, ys, bkg_x, bkg_y, raw_x, raw_y, list_Sq, Fq_smoothed, mean_sq_fi, sq_mean_fi, r_smoothed, G_smoothed = get_graph_data(
                path, ext, control_panel
            )

            if plot_window_ref is None or not plot_window_ref.isVisible():
                plot_window_ref = PlotWindow()
                plot_window_ref.show()

            plot_window_ref.plot_data(
                xs, ys, bkg_x, bkg_y, raw_x, raw_y,
                list_Sq, Fq_smoothed, mean_sq_fi, sq_mean_fi,
                r_smoothed, G_smoothed, os.path.basename(path)
            )
            label_widget.setText(f"Showing plot for {short_name}")

        else:
            legend_names = [os.path.basename(p) for p in paths]
            xs_list, ys_list = [], []

            for path, ext in zip(paths, extensions):
                xs, ys, *_ = get_graph_data(path, ext, control_panel, multiple_graphs=True)
                xs_list.append(xs)
                ys_list.append(ys)

            if plot_window_ref is None:
                plot_window_ref = PlotWindow()
                plot_window_ref.show()

            plot_window_ref.plot_multiple(xs_list, ys_list, titles=legend_names)
            label_widget.setText(f"Showing plots for {len(paths)} files")

        return plot_window_ref

    except Exception as e:
        label_widget.setText(f"Failed to load: {e}")
        if plot_window_ref and hasattr(plot_window_ref, 'close'):
            plot_window_ref.close()
        return None


def get_valid_files(folder, extensions=VALID_EXTENSIONS):
    return sorted(
        [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if any(f.lower().endswith(ext) for ext in extensions)
        ],
        key=lambda f: os.path.getmtime(f),
        reverse=True
    )


def add_files_to_list_widget(file_list_widget, label_widget, source, loaded_files, is_folder=True, max_name_length=25):
    if is_folder:
        valid_file_paths = get_valid_files(source)
    else:
        valid_file_paths = [
            f for f in source if any(f.lower().endswith(ext) for ext in VALID_EXTENSIONS)
        ]

    valid_file_paths.sort(key=lambda f: os.path.getmtime(f), reverse=True)

    new_file_paths = [f for f in valid_file_paths if f not in loaded_files]
    loaded_files.update(new_file_paths)

    starting_index = file_list_widget.topLevelItemCount() + 1
    for idx, full_path in enumerate(new_file_paths, start=starting_index):
        filename = os.path.basename(full_path)
        short_name = get_short_name(filename, max_name_length)

        item = QTreeWidgetItem([str(idx), short_name])
        item.setToolTip(1, filename)
        item.setData(0, Qt.ItemDataRole.UserRole, full_path)
        file_list_widget.addTopLevelItem(item)

    if not file_list_widget.currentItem() and label_widget is not None:
        label_widget.setText("No file selected.")


def _capture_plot_ranges(plot_window):
    ranges = []
    if plot_window is None:
        return ranges

    for plot in getattr(plot_window, 'plots', []):
        try:
            vb = plot.getViewBox()
            x_range, y_range = vb.viewRange()
            vals = np.array([x_range[0], x_range[1], y_range[0], y_range[1]], dtype=float)
            if np.all(np.isfinite(vals)):
                ranges.append(((float(x_range[0]), float(x_range[1])), (float(y_range[0]), float(y_range[1]))))
            else:
                ranges.append(None)
        except Exception:
            ranges.append(None)
    return ranges


def _restore_plot_ranges(plot_window, ranges, restore_y=True):
    """Restore saved x (and optionally y) axis ranges.

    restore_y=False: only x-range is restored; y-axis auto-scales to the
    new data.  Use this when data values may have changed (e.g. qmin/qmax
    update) so the y-axis is not stuck at stale limits.
    """
    if plot_window is None or not ranges:
        return

    for plot, saved in zip(getattr(plot_window, 'plots', []), ranges):
        if not saved:
            continue
        try:
            (x0, x1), (y0, y1) = saved
            x_vals = np.array([x0, x1], dtype=float)
            if not np.all(np.isfinite(x_vals)) or x1 <= x0:
                continue

            vb = plot.getViewBox()
            if restore_y:
                y_vals = np.array([y0, y1], dtype=float)
                if np.all(np.isfinite(y_vals)) and y1 > y0:
                    try:
                        vb.enableAutoRange(x=False, y=False)
                    except Exception:
                        pass
                    plot.setYRange(float(y0), float(y1), padding=0)
            else:
                # Let y-axis auto-scale to the newly plotted data
                try:
                    vb.enableAutoRange(x=False, y=True)
                except Exception:
                    pass

            plot.setXRange(float(x0), float(x1), padding=0)
        except Exception:
            continue


def update_current_graph(selected_items, control_panel, plot_window):
    # 그래프 잠금 상태면 업데이트하지 않음
    if plot_window is not None and hasattr(plot_window, 'is_locked') and plot_window.is_locked():
        return

    try:
        if not selected_items:
            return

        # Guard against "Internal C++ object already deleted" —
        # QTreeWidgetItem references become invalid after file_list.clear().
        paths = []
        valid_items = []
        for item in selected_items:
            if item is None:
                continue
            try:
                p = item.data(0, Qt.ItemDataRole.UserRole)
                if p:
                    paths.append(p)
                    valid_items.append(item)
            except RuntimeError:
                # C++ object deleted — skip stale reference
                continue
        if not paths:
            return

        saved_ranges = _capture_plot_ranges(plot_window)

        if len(paths) == 1:
            path = paths[0]
            if not os.path.isfile(path):
                return

            extension = os.path.splitext(path)[1].lower()
            xs, ys, bkg_x, bkg_y, raw_x, raw_y, list_Sq, Fq_smoothed, mean_sq_fi, sq_mean_fi, r_smoothed, G_smoothed = get_graph_data(
                path, extension, control_panel
            )

            plot_window.plot_data(
                xs, ys, bkg_x, bkg_y, raw_x, raw_y,
                list_Sq, Fq_smoothed, mean_sq_fi, sq_mean_fi,
                r_smoothed, G_smoothed, os.path.basename(path)
            )

        else:
            list_xs = []
            list_ys = []
            legend_names = [os.path.basename(p) for p in paths]

            for path in paths:
                if not os.path.isfile(path):
                    continue
                ext = os.path.splitext(path)[1].lower()
                xs, ys, *_ = get_graph_data(path, ext, control_panel, multiple_graphs=True)
                list_xs.append(xs)
                list_ys.append(ys)

            if list_xs and list_ys:
                plot_window.plot_multiple(list_xs, list_ys, titles=legend_names)

        # Do NOT restore ranges after a parameter-driven update.
        # plot_data() already calls _autorange_and_fix_x0() for every subplot,
        # which correctly scales both x and y to the new data.
        # Restoring saved ranges would re-lock stale axes (e.g. G(r) x-range
        # stuck at old rmax, S(q) y-range stuck at pre-normalisation values).
        # _restore_plot_ranges(plot_window, saved_ranges)  ← intentionally skipped

    except Exception as e:
        msg = QMessageBox(control_panel)
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Error Updating Plot")
        msg.setText("Could not update graph with current parameters.")
        msg.setInformativeText(str(e))
        msg.exec()


def calculate_compton(control_panel):
    q_range, list_compton_scat = get_compton_values(control_panel)

    list_compton_scat = np.array(list_compton_scat)
    q_range = np.array(q_range)

    write_compton_file(q_range, list_compton_scat, control_panel)

