# controller/graph_controller.py

import os
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QTreeWidgetItem

from ezpit.gui.model.dataloader import get_graph_data
from ezpit.gui.model.processing import get_compton_values
from ezpit.gui.ui.ui_helpers import get_short_name
from ezpit.gui.ui.viewer import PlotWindow

VALID_EXTENSIONS: list[str] = [
    ".chi",
    ".gr",
    ".sq",
    ".fq",
    ".iq",
    ".xy",
    ".dat",
    ".txt",
    ".xyz",
    ".caliq",
    ".calsq",
    ".calfq",
    ".calgr",
    ".compton",
]


def load_selected_files(selected_items, label_widget, plot_window_ref, control_panel, max_length_file_name):
    if not selected_items:
        label_widget.setText("No file selected.")
        if plot_window_ref and hasattr(plot_window_ref, "close"):
            plot_window_ref.close()
        return None

    paths = [item.data(0, Qt.ItemDataRole.UserRole) for item in selected_items]
    extensions = [os.path.splitext(p)[1].lower() for p in paths]

    try:
        if len(paths) == 1:
            path = paths[0]
            ext = extensions[0]
            short_name = get_short_name(os.path.basename(path), max_length_file_name)

            (
                xs,
                ys,
                bkg_x,
                bkg_y,
                raw_x,
                raw_y,
                list_Sq,
                Fq_smoothed,
                mean_sq_fi,
                sq_mean_fi,
                r_smoothed,
                G_smoothed,
            ) = get_graph_data(path, ext, control_panel)

            if plot_window_ref is None or not plot_window_ref.isVisible():
                plot_window_ref = PlotWindow()
                plot_window_ref.show()

            plot_window_ref.plot_data(
                xs,
                ys,
                bkg_x,
                bkg_y,
                raw_x,
                raw_y,
                list_Sq,
                Fq_smoothed,
                mean_sq_fi,
                sq_mean_fi,
                r_smoothed,
                G_smoothed,
                os.path.basename(path),
            )

            # A Compton curve only has an I(q)-type panel; S(q), F(q) and G(r)
            # would just be empty, so show the I(q) panel alone. The user can
            # still tick the other boxes if they want them.
            if ext == ".compton":
                plot_window_ref.enable_graphs(enable_iq=True, enable_sq=False, enable_fq=False, enable_gr=False)

            label_widget.setText(f"Showing plot for {short_name}")

        else:
            legend_names = [os.path.basename(p) for p in paths]
            xs_list, ys_list = [], []

            for path, ext in zip(paths, extensions, strict=False):
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
        if plot_window_ref and hasattr(plot_window_ref, "close"):
            plot_window_ref.close()
        return None


def get_valid_files(dir_path: Path, extensions: list[str] = VALID_EXTENSIONS) -> list[Path]:
    return sorted(
        [f for f in dir_path.iterdir() if any(f.name.lower().endswith(ext) for ext in extensions)],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )


def add_files_to_list_widget(
    file_list_widget,
    label_widget,
    source,
    loaded_files,
    is_folder=True,
    max_name_length=25,
):
    if is_folder:
        valid_file_paths = get_valid_files(source)
    else:
        valid_file_paths = [f for f in source if any(f.lower().endswith(ext) for ext in VALID_EXTENSIONS)]

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

    for plot in getattr(plot_window, "plots", []):
        try:
            vb = plot.getViewBox()
            x_range, y_range = vb.viewRange()
            vals = np.array([x_range[0], x_range[1], y_range[0], y_range[1]], dtype=float)
            if np.all(np.isfinite(vals)):
                ranges.append(
                    (
                        (float(x_range[0]), float(x_range[1])),
                        (float(y_range[0]), float(y_range[1])),
                    )
                )
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

    for plot, saved in zip(getattr(plot_window, "plots", []), ranges, strict=False):
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
    if plot_window is not None and hasattr(plot_window, "is_locked") and plot_window.is_locked():
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

        _capture_plot_ranges(plot_window)

        if len(paths) == 1:
            path = paths[0]
            if not os.path.isfile(path):
                return

            extension = os.path.splitext(path)[1].lower()
            (
                xs,
                ys,
                bkg_x,
                bkg_y,
                raw_x,
                raw_y,
                list_Sq,
                Fq_smoothed,
                mean_sq_fi,
                sq_mean_fi,
                r_smoothed,
                G_smoothed,
            ) = get_graph_data(path, extension, control_panel)

            plot_window.plot_data(
                xs,
                ys,
                bkg_x,
                bkg_y,
                raw_x,
                raw_y,
                list_Sq,
                Fq_smoothed,
                mean_sq_fi,
                sq_mean_fi,
                r_smoothed,
                G_smoothed,
                os.path.basename(path),
                bring_front=False,
            )

            # Compton curves only populate the I(q) panel (see load_selected_files).
            if extension == ".compton":
                plot_window.enable_graphs(enable_iq=True, enable_sq=False, enable_fq=False, enable_gr=False)

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
                plot_window.plot_multiple(list_xs, list_ys, titles=legend_names, bring_front=False)

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
    """Compute the Compton scattering curve and plot it directly on the.

    EZPDF Plot window's I(q) panel.

    Previously this opened a save dialog and wrote a .compton file. Now
    pressing "Calculate Compton" (after typing a composition into the
    Compton tab) shows the result immediately in the I(q) panel, matching
    how a loaded .compton file is displayed.
    """
    q_range, list_compton_scat = get_compton_values(control_panel)

    q_range = np.asarray(q_range, dtype=float)
    list_compton_scat = np.asarray(list_compton_scat, dtype=float)

    # PlotWindow.plot_data() expects a 4-panel structure
    # (I(q), S(q), F(q), G(r)). A Compton curve only has an I(q)-type
    # panel, so populate index 0 and leave the rest empty; the empty
    # panels are hidden via enable_graphs() below. This mirrors the
    # .compton file-loading path in get_graph_data().
    xs = [q_range, [], [], []]
    ys = [list_compton_scat, [], [], []]

    main_window = getattr(control_panel, "main_window", None)

    # Choose the target window. Respect the "Open in New Graphs" checkbox if
    # present; otherwise reuse the current EZPDF Plot window, creating one
    # when none is open/visible.
    plot_window = getattr(main_window, "plot_window", None) if main_window else None
    use_new = bool(
        getattr(control_panel, "new_window_checkbox", None) and control_panel.new_window_checkbox.isChecked()
    )

    if use_new or plot_window is None or not plot_window.isVisible():
        plot_window = PlotWindow()
        plot_window.show()

    composition = control_panel.get_compton_parameters().get("composition", "").strip()
    title = f"Compton ({composition})" if composition else "Compton"

    plot_window.plot_data(xs, ys, None, None, None, None, None, None, None, None, None, None, title)
    plot_window.enable_graphs(enable_iq=True, enable_sq=False, enable_fq=False, enable_gr=False)

    # Flag this window as showing a Compton curve. plot_data()/plot_multiple()
    # reset this to False, so it stays True only until the next file is drawn.
    # The "Select data to save" dialog uses it to offer a Compton (.compton)
    # save option.
    plot_window.is_compton = True

    # This window now shows a computed Compton curve, not a loaded file.
    # Dropping the file association stops later parameter-driven
    # send_update() calls from replotting a previous file over the curve.
    plot_window.associated_items = None

    # Register/activate the window with the main window and match the theme.
    if main_window is not None:
        main_window.plot_window = plot_window
        windows = getattr(main_window, "plot_windows", None)
        if windows is None:
            main_window.plot_windows = windows = []
        if plot_window not in windows:
            windows.append(plot_window)
        try:
            plot_window.apply_theme(getattr(main_window, "is_dark_mode", False))
        except Exception:
            pass

    return plot_window
