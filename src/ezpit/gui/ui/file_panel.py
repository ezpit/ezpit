# ui/file_panel.py
# PySide6 version + Drag&Drop (files/folders) while preserving Undo Delete and plot updates.
# Sorting: click "#" header → restore load order
#          click "File Name" header → cycle  ▲(asc) → ▼(desc) → load order
#          Sort key: embedded timestamp in filename (YYYYMMDD-HHMMSS), fallback to alpha.

import os
import re

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ezpit.gui.controller.graph_controller import load_selected_files
from ezpit.gui.model.helpers import composition_string_from_xyz

from .ui_helpers import update_file_list_numbering

# ---------------------------------------------------------------------------
# Regex: capture YYYYMMDD-HHMMSS (or YYYYMMDD_HHMMSS) embedded in a filename
# e.g.  "..._20260331-012503_..."  →  sortable key "20260331012503"
# ---------------------------------------------------------------------------
_TS_RE = re.compile(r"(\d{8})[_\-](\d{6})")


def _filename_sort_key(path: str) -> str:
    """Return a sortable string: embedded timestamp if found, else lowered basename."""
    name = os.path.basename(path)
    m = _TS_RE.search(name)
    if m:
        return m.group(1) + m.group(2)  # "YYYYMMDDHHMMSS" → lexicographic == chronological
    return name.lower()


class _NoAutoScrollTree(QTreeWidget):
    INDEX_WIDTH = 15

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._fix_index_col()
        self.header().sectionResized.connect(self._on_section_resized)
        self.installEventFilter(self)

    def scrollTo(self, index, hint=QAbstractItemView.ScrollHint.EnsureVisible):
        return

    def _fix_index_col(self):
        h = self.header()
        try:
            h.blockSignals(True)
            h.setStretchLastSection(False)
            h.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
            self.setColumnWidth(0, self.INDEX_WIDTH)
        finally:
            h.blockSignals(False)

    def _on_section_resized(self, logical, _old, new):
        if logical == 0 and new != self.INDEX_WIDTH:
            QTimer.singleShot(0, self._fix_index_col)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fix_index_col()

    def showEvent(self, e):
        super().showEvent(e)
        self._fix_index_col()

    def paintEvent(self, e):
        self._fix_index_col()
        super().paintEvent(e)
        self._fix_index_col()

    def eventFilter(self, obj, ev):
        if obj is self and ev.type() in (
            QEvent.Type.LayoutRequest,
            QEvent.Type.UpdateRequest,
            QEvent.Type.PolishRequest,
            QEvent.Type.ShowToParent,
        ):
            self._fix_index_col()
        return super().eventFilter(obj, ev)


class FilePanel(QWidget):
    def __init__(self, label_widget, loaded_files, undo_stack, parent=None):
        super().__init__()

        self.label_widget = label_widget
        self.loaded_files = loaded_files
        self.undo_stack = undo_stack
        self.main_window = parent
        self.control_panel = parent.control_panel if parent else None

        # --- Sort state ---
        # _sort_col : None (load order) | 1 (File Name column)
        # _sort_dir : 'asc' | 'desc'   (only meaningful when _sort_col == 1)
        self._sort_col = None
        self._sort_dir = "asc"

        # Original insertion order (paths), used to restore "Load Order"
        self._load_order = []

        # ----------------------------------------------------------------
        # File list widget
        # ----------------------------------------------------------------
        self.file_list = _NoAutoScrollTree()
        self.file_list.setColumnCount(2)
        self.file_list.setHeaderLabels(["#", "File Name"])
        self.file_list.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.file_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.file_list.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.file_list.setAutoScroll(False)
        self.file_list.setRootIsDecorated(False)
        self.file_list.setIndentation(0)

        # Header: resizing + clickable sorting
        header = self.file_list.header()
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(False)  # shown only while a sort is active
        header.sectionClicked.connect(self._on_header_clicked)

        # Model signals
        m = self.file_list.model()
        m.rowsInserted.connect(self._on_rows_inserted)
        m.rowsRemoved.connect(lambda *_: self._apply_index_width())
        m.columnsInserted.connect(lambda *_: self._apply_index_width())
        m.columnsRemoved.connect(lambda *_: self._apply_index_width())
        self.file_list.itemChanged.connect(lambda *_: self._apply_index_width())

        self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.file_list.itemDoubleClicked.connect(self.on_file_double_clicked)
        # Auto-fill the Compton tab composition when a single .xyz is selected.
        self.file_list.itemSelectionChanged.connect(self._autofill_compton_composition)

        # Drag & drop
        try:
            self.file_list.setAcceptDrops(True)
            self.file_list.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
            self.file_list.setDropIndicatorShown(True)
        except Exception:
            pass

        self.file_list.installEventFilter(self)

        # ----------------------------------------------------------------
        # Buttons
        # ----------------------------------------------------------------
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(2)

        delete_button = QPushButton("Delete Selected")
        delete_button.clicked.connect(self.delete_selected_files)
        btn_layout.addWidget(delete_button)

        self.undo_button = QPushButton("Undo Delete")
        self.undo_button.clicked.connect(self.undo_delete)
        btn_layout.addWidget(self.undo_button)

        # ----------------------------------------------------------------
        # Main layout
        # ----------------------------------------------------------------
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.file_list)
        layout.addLayout(btn_layout)

        self._apply_index_width()

    # ------------------------------------------------------------------
    # Header-click sorting  (Windows Explorer style)
    # ------------------------------------------------------------------

    def _on_header_clicked(self, col: int):
        """
        Col 0  (#)         → always restore load order.

        col 1  (File Name) → cycle:  none → ▲ asc → ▼ desc → none (load order).
        """
        if col == 0:
            self._sort_col = None
            self._sort_dir = "asc"
        elif col == 1:
            if self._sort_col != 1:
                self._sort_col = 1
                self._sort_dir = "asc"
            elif self._sort_dir == "asc":
                self._sort_dir = "desc"
            else:
                # Third click → back to load order
                self._sort_col = None
                self._sort_dir = "asc"

        self._apply_sort()
        self._update_sort_indicator()

    def _update_sort_indicator(self):
        """Show/hide Qt sort arrow on the File Name column header."""
        header = self.file_list.header()
        if self._sort_col == 1:
            header.setSortIndicatorShown(True)
            qt_order = Qt.SortOrder.AscendingOrder if self._sort_dir == "asc" else Qt.SortOrder.DescendingOrder
            header.setSortIndicator(1, qt_order)
        else:
            header.setSortIndicatorShown(False)

    def _apply_sort(self):
        """Re-order tree items according to current sort state."""
        n = self.file_list.topLevelItemCount()
        if n == 0:
            return

        # Snapshot current items
        items_data = []
        for i in range(n):
            item = self.file_list.topLevelItem(i)
            path = item.data(0, Qt.ItemDataRole.UserRole)
            name = item.text(1)
            tip = item.toolTip(1)
            items_data.append((path, name, tip))

        if self._sort_col == 1:
            # Sort by embedded filename timestamp (fallback: alpha)
            reverse = self._sort_dir == "desc"
            items_data.sort(key=lambda x: _filename_sort_key(x[0]), reverse=reverse)
        else:
            # Restore load order
            order_map = {p: i for i, p in enumerate(self._load_order)}
            items_data.sort(key=lambda x: order_map.get(x[0], 999999))

        # Rebuild tree (disconnect _on_rows_inserted so _load_order stays clean)
        m = self.file_list.model()
        m.rowsInserted.disconnect(self._on_rows_inserted)
        self.file_list.clear()
        for idx, (path, name, tip) in enumerate(items_data, start=1):
            new_item = QTreeWidgetItem([str(idx), name])
            new_item.setToolTip(1, tip)
            new_item.setData(0, Qt.ItemDataRole.UserRole, path)
            self.file_list.addTopLevelItem(new_item)
        m.rowsInserted.connect(self._on_rows_inserted)

        # Undo indices are stale after a reorder → clear stack
        self.undo_stack.clear()

        # ------------------------------------------------------------------
        # CRITICAL: after clear() + re-insert, old QTreeWidgetItem C++ objects
        # are deleted.  Any plot_window.associated_items still pointing to them
        # will raise "Internal C++ object already deleted" on the next
        # send_update().  Remap every window's associated_items to the newly
        # created items (matched by UserRole path).
        # ------------------------------------------------------------------
        if self.main_window is not None:
            # Build path → new_item lookup
            path_to_new_item = {}
            for i in range(self.file_list.topLevelItemCount()):
                it = self.file_list.topLevelItem(i)
                p = it.data(0, Qt.ItemDataRole.UserRole)
                if p:
                    path_to_new_item[p] = it

            for pw in getattr(self.main_window, "plot_windows", []):
                old_items = getattr(pw, "associated_items", None)
                if not old_items:
                    continue
                new_items = []
                for old_it in old_items:
                    try:
                        p = old_it.data(0, Qt.ItemDataRole.UserRole)
                        new_it = path_to_new_item.get(p)
                        if new_it:
                            new_items.append(new_it)
                    except RuntimeError:
                        # C++ object already deleted — skip, will be remapped
                        pass
                pw.associated_items = new_items if new_items else old_items

        self._apply_index_width()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _on_rows_inserted(self, parent, first: int, last: int):
        """Record insertion order so load-order restore works correctly."""
        self._apply_index_width()
        for row in range(first, last + 1):
            item = self.file_list.topLevelItem(row)
            if item:
                path = item.data(0, Qt.ItemDataRole.UserRole)
                if path and path not in self._load_order:
                    self._load_order.append(path)

    def _apply_index_width(self):
        QTimer.singleShot(0, self.file_list._fix_index_col)  # noqa: SLF001

    # ------------------------------------------------------------------
    # Existing behaviour (unchanged)
    # ------------------------------------------------------------------

    def on_file_double_clicked(self, *args):
        if not args:
            return
        item = args[0]

        use_new = False
        if self.control_panel and hasattr(self.control_panel, "new_window_checkbox"):
            use_new = self.control_panel.new_window_checkbox.isChecked()

        ref = None if use_new else self.main_window.plot_window

        new_plot_window = load_selected_files(
            selected_items=[item],
            label_widget=self.label_widget,
            plot_window_ref=ref,
            control_panel=self.control_panel,
            max_length_file_name=0,
        )

        if new_plot_window:
            new_plot_window.associated_items = [item]
            self.main_window.plot_window = new_plot_window
            if new_plot_window not in self.main_window.plot_windows:
                self.main_window.plot_windows.append(new_plot_window)

        self.main_window.current_path = [item]

    def _autofill_compton_composition(self):
        """Fill the Compton tab's composition field from a single selected .xyz file.

        Multi-selection is ignored so we never clobber the field during a
        drag-select. The user can still edit the field afterward; the Compton
        Calculate button asks which source to use if it no longer matches.
        """
        cp = self.control_panel
        if cp is None or not hasattr(cp, "compton_composition_input"):
            return
        try:
            items = self.file_list.selectedItems()
        except RuntimeError:
            return
        if len(items) != 1:
            return
        try:
            path = items[0].data(0, Qt.ItemDataRole.UserRole)
        except RuntimeError:
            return
        if not path or os.path.splitext(path)[1].lower() != ".xyz":
            return
        comp = composition_string_from_xyz(path)
        if comp:
            cp.compton_composition_input.setText(comp)
            # Remember which file/composition this came from so the Compton
            # Calculate button can still prompt after the file is deselected.
            cp._compton_xyz_path = path  # noqa: SLF001
            cp._compton_xyz_comp = comp  # noqa: SLF001

    def delete_selected_files(self):
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            return

        confirm = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete {len(selected_items)} file(s) from the list?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        deleted_batch = []
        for item in selected_items:
            index = self.file_list.indexOfTopLevelItem(item)
            name = item.text(1)
            path = item.data(0, Qt.ItemDataRole.UserRole)
            deleted_batch.append((index, name, path))

            if path in self.loaded_files:
                try:
                    self.loaded_files.remove(path)
                except Exception:
                    pass
            self.file_list.takeTopLevelItem(index)

        if deleted_batch:
            self.undo_stack.append(deleted_batch)
            update_file_list_numbering(self.file_list)

        self._apply_index_width()

    def undo_delete(self):
        if not self.undo_stack:
            return

        last_deleted = self.undo_stack.pop()
        for index, name, path in sorted(last_deleted, key=lambda x: x[0]):
            if path:
                try:
                    self.loaded_files.add(path)
                except Exception:
                    pass

            new_item = QTreeWidgetItem()
            new_item.setText(1, name)
            new_item.setData(0, Qt.ItemDataRole.UserRole, path)
            self.file_list.insertTopLevelItem(index, new_item)

        update_file_list_numbering(self.file_list)
        self._apply_index_width()

    def _handle_drop_paths(self, paths):
        """Add dropped files/folders via main_window.populate_file_list."""
        if not paths:
            return

        files = []
        folders = []
        for p in paths:
            p = os.path.normpath(p)
            if os.path.isdir(p):
                folders.append(p)
            elif os.path.isfile(p):
                files.append(p)

        def _dedup(seq):
            seen = set()
            out = []
            for x in seq:
                if x not in seen:
                    out.append(x)
                    seen.add(x)
            return out

        folders = _dedup(folders)
        files = _dedup(files)

        if self.main_window is not None and hasattr(self.main_window, "populate_file_list"):
            for d in folders:
                self.main_window.populate_file_list(d, is_folder=True)
            if files:
                self.main_window.populate_file_list(files, is_folder=False)

        update_file_list_numbering(self.file_list)
        self._apply_index_width()

    def eventFilter(self, source, event):
        if source == self.file_list:
            et = event.type()

            if et == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Delete:
                self.delete_selected_files()
                return True

            if et in (QEvent.Type.DragEnter, QEvent.Type.DragMove):
                md = event.mimeData()
                if md is not None and md.hasUrls():
                    event.acceptProposedAction()
                    return True

            if et == QEvent.Type.Drop:
                md = event.mimeData()
                if md is not None and md.hasUrls():
                    paths = []
                    for url in md.urls():
                        try:
                            if url.isLocalFile():
                                p = url.toLocalFile()
                                if p:
                                    paths.append(p)
                        except Exception:
                            continue
                    self._handle_drop_paths(paths)
                    event.acceptProposedAction()
                    return True

        return super().eventFilter(source, event)

    def get_widget(self):
        return self

    def get_selected_file_paths(self):
        return self.file_list.selectedItems()
