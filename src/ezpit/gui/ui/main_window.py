import json  # JSON data handling - JSON 데이터 처리
import os  # OS path and file handling - OS 경로 및 파일 처리
import sys  # Needed to detect a PyInstaller bundle - PyInstaller 번들 감지용

from PySide6.QtCore import QSettings, QSize, Qt
from PySide6.QtGui import QAction, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QSizePolicy,
    QSplitter,
    QWidget,
)

from ezpit.gui.controller.graph_controller import (
    add_files_to_list_widget,
    load_selected_files,
)

from .control_panel import ControlPanel
from .file_panel import FilePanel


def resource_path(relative_path):
    """Return the absolute path to a bundled resource.

    Paths are resolved relative to the program itself rather than to the
    directory the user happens to launch it from. A bare relative path such as
    "ui/icons/graph2d.png" is resolved against the current working directory,
    so it breaks as soon as the program is started from anywhere else - which
    is always the case for a packaged executable the user double-clicks.

    Works both when running from source and from a PyInstaller bundle.
    `relative_path` is given relative to the project root, e.g.
    "ui/icons/graph2d.png".
    """
    if hasattr(sys, "_MEIPASS"):
        # Running inside a PyInstaller bundle.
        base_path = sys._MEIPASS  # noqa: SLF001
    else:
        # Running from source: ui/ -> project root.
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


# Internal module imports - 내부 모듈 임포트

# PySide6 components for UI - UI 구성을 위한 PySide6 컴포넌트


# Read style.qss (No error if missing) - style.qss 읽기 (파일이 없어도 에러 안 나게 처리)
try:
    with open("style.qss", encoding="utf-8") as f:
        config = f.read()
except Exception:
    config = ""

# Define LIGHT_QSS (Original style with CheckBox Fix) - 라이트 모드 스타일 정의 (기존 스타일 및 체크박스 수정 포함)
LIGHT_QSS = """
            /* Main Window Background: Light Gray - 메인 윈도우 배경: 조금 더 밝은 회색 (#E0E0E0) */
            QMainWindow, QWidget {
                background-color: #E0E0E0;
                color: #000000;
                font-family: 'Segoe UI', 'Malgun Gothic', sans-serif;
            }

            /* Top Menubar: Blue Tones - 상단 메뉴바: 파란색 계열 */
            QMenuBar {
                background-color: #D6E4F0;
                border-bottom: 1px solid #A0A0A0;
            }
            QMenuBar::item {
                background-color: transparent;
                color: #000000;
                padding: 4px 8px;
            }
            QMenuBar::item:selected {
                background-color: #A9Cce3;
            }
            QMenu {
                background-color: #F0F0F0;
                color: #000000;
                border: 1px solid #888;
            }
            QMenu::item:selected {
                background-color: #D6E4F0;
            }

            /* Input fields, lists, tree widgets background - 입력창, 리스트, 트리 위젯 배경 */
            QLineEdit, QTreeWidget, QListWidget, QTableWidget, QAbstractItemView {
                background-color: #FFFFFF;
                border: 1px solid #888888;
                color: #000000;
            }

            QTreeWidget::item:selected, QListWidget::item:selected {
                background-color: #A9Cce3;
                color: #000000;
            }

            /* Combo box style - 콤보박스 스타일 */
            QComboBox {
                background-color: #FFFFFF;
                border: 1px solid #888888;
                padding: 2px;
                color: #000000;
            }

            /* Tab widgets and panes - 탭 위젯 및 패인 스타일 */
            QTabWidget::pane {
                border: 1px solid #888888;
                background-color: #E0E0E0;
            }
            QTabBar::tab {
                background: #D0D0D0;
                border: 1px solid #888888;
                padding: 5px;
                margin-right: 2px;
                color: #000000;
            }
            QTabBar::tab:selected {
                background: #E0E0E0;
                border-bottom-color: #E0E0E0;
                font-weight: bold;
            }

            /* Button Styles - 버튼 스타일 */
            QPushButton {
                background-color: #F0F0F0;
                border: 1px solid #888888;
                padding: 4px;
                border-radius: 2px;
                color: #000000;
            }
            QPushButton:hover {
                background-color: #D6E4F0;
                border: 1px solid #0078D7;
            }
            QPushButton:pressed {
                background-color: #A9Cce3;
            }

            /* Slider handle size and style - 슬라이더 핸들 크기 및 스타일 조정 */
            QSlider {
                min-height: 20px;
            }

            QSlider::handle:horizontal {
                border: 1px solid #009688;
                background: qradialgradient(
                    spread:pad, cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
                    stop:0.0 #009688,
                    stop:0.45 #009688,
                    stop:0.46 #FFFFFF,
                    stop:1.0 #FFFFFF
                );
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                border-color: #00796B;
            }
            QSlider::groove:horizontal {
                background: #B0B0B0;
                height: 4px;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #009688;
                height: 4px;
                border-radius: 2px;
            }

            /* CheckBox Style Fix (Maintains color even when inactive)
               체크박스 스타일 수정 (비활성 시에도 색상 유지) */
            QCheckBox {
                spacing: 5px;
                color: #000000;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #888888;
                background: #FFFFFF;
                border-radius: 2px;
            }
            QCheckBox::indicator:hover {
                border: 1px solid #009688;
            }
            QCheckBox::indicator:checked {
                background-color: #009688;
                border: 1px solid #009688;
            }
            QCheckBox::indicator:checked:disabled {
                background-color: #80CBC4;
            }
"""

# Define DARK_QSS (Fix included) - 다크 모드 스타일 정의 (기존 수정본 포함)
DARK_QSS = """
            /* Main Window Background: Dark Gray - 메인 윈도우 배경: 다크 그레이 (#2b2b2b) */
            QMainWindow, QWidget {
                background-color: #2b2b2b;
                color: #E0E0E0;
                font-family: 'Segoe UI', 'Malgun Gothic', sans-serif;
            }

            /* Top Menubar in Dark Mode - 다크 모드 상단 메뉴바 */
            QMenuBar {
                background-color: #1e1e1e;
                border-bottom: 1px solid #444;
            }
            QMenuBar::item {
                background-color: transparent;
                color: #E0E0E0;
                padding: 4px 8px;
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

            /* Dark theme input, list, tree background - 다크 테마 입력창, 리스트 배경 */
            QLineEdit, QTreeWidget, QListWidget, QTableWidget, QAbstractItemView {
                background-color: #3b3b3b;
                border: 1px solid #555555;
                color: #E0E0E0;
            }

            QTreeWidget::item:selected, QListWidget::item:selected {
                background-color: #009688;
                color: #FFFFFF;
            }

            QLineEdit[readOnly="true"] {
                background-color: #2e2e2e;
                color: #aaa;
            }

            /* Combo box style in dark mode - 다크 모드 콤보박스 스타일 */
            QComboBox {
                background-color: #3b3b3b;
                border: 1px solid #555555;
                padding: 2px;
                color: #E0E0E0;
            }
            QComboBox QAbstractItemView {
                background-color: #3b3b3b;
                color: #E0E0E0;
                selection-background-color: #009688;
            }

            /* Tab widgets and panes in dark mode - 다크 모드 탭 위젯 및 패인 스타일 */
            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background: #3c3f41;
                border: 1px solid #555555;
                padding: 5px;
                margin-right: 2px;
                color: #E0E0E0;
            }
            QTabBar::tab:selected {
                background: #4e5254;
                border-bottom-color: #4e5254;
                font-weight: bold;
            }

            /* Button Styles in dark mode - 다크 모드 버튼 스타일 */
            QPushButton {
                background-color: #3c3f41;
                border: 1px solid #555555;
                padding: 4px;
                border-radius: 2px;
                color: #E0E0E0;
            }
            QPushButton:hover {
                background-color: #4e5254;
                border: 1px solid #009688;
            }
            QPushButton:pressed {
                background-color: #00796B;
            }

            /* Slider handle style in dark mode - 다크 모드 슬라이더 핸들 스타일 */
            QSlider {
                min-height: 20px;
            }
            QSlider::handle:horizontal {
                border: 1px solid #009688;
                background: #b0b0b0;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                border-color: #00796B;
                background: #ffffff;
            }
            QSlider::groove:horizontal {
                background: #555555;
                height: 4px;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #009688;
                height: 4px;
                border-radius: 2px;
            }

            /* CheckBox Style Fix (Dark Mode) - 다크 모드 체크박스 스타일 수정 */
            QCheckBox {
                spacing: 5px;
                color: #E0E0E0;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #777;
                background: #444;
                border-radius: 2px;
            }
            QCheckBox::indicator:hover {
                border: 1px solid #009688;
            }
            QCheckBox::indicator:checked {
                background-color: #009688;
                border: 1px solid #009688;
            }
            QCheckBox::indicator:checked:disabled {
                background-color: #004D40;
                border: 1px solid #555;
            }
"""


class PDFApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # --- 아이콘 설정 코드 (번들 대응 경로 방식) ---
        icon_path = resource_path(os.path.join("ui", "icons", "EZPDF_logo_2x_transparent.png"))
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        # ----------------------------------------

        self.setWindowTitle("EZPDF")
        self.setGeometry(100, 100, 800, 500)

        # State and shared objects - 상태 및 공유 객체 관리
        self.loaded_files = set()  # Prevent duplicate loads - 중복 로드 방지
        self.label = QLabel("No file selected.")
        self.undo_stack = []

        # Multiple window management - 다중 창 관리
        self.plot_windows = []
        self.plot_window = None  # Current active plot window - 현재 활성화된 플롯 창

        self.current_path = None
        self.is_dark_mode = False

        # Restore last-used directory from persistent settings
        self._settings = QSettings("EZPDF", "EZPDF")
        self._last_dir = self._settings.value("last_dir", "")  # Dark mode flag - 다크 모드 플래그

        # Initialize menu and UI - 메뉴 및 UI 초기화
        self.init_menu()
        self.init_help_menu()
        self.init_ui()

        # Apply default Light theme - 기본 라이트 테마 적용
        self.setStyleSheet(LIGHT_QSS)

        # Shortcut for Undo Delete (Ctrl+Z) - 삭제 실행 취소 단축키 (Ctrl+Z)
        undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        undo_shortcut.activated.connect(self.file_panel.undo_delete)

    # ---------------- Menu Setup - 메뉴 설정 ----------------
    def init_menu(self):
        menu_bar = self.menuBar()
        self.init_file(menu_bar)

        # View Menu for theme control - 테마 조절을 위한 View 메뉴
        view_menu = menu_bar.addMenu("View")
        self.dark_mode_action = QAction("Dark Mode", self)
        self.dark_mode_action.setCheckable(True)
        self.dark_mode_action.setChecked(False)
        self.dark_mode_action.triggered.connect(self.toggle_dark_mode)
        view_menu.addAction(self.dark_mode_action)

        menu_bar.setContentsMargins(0, 0, 0, 0)

        # 2D Graph Icon on menubar - 메뉴바의 2D 그래프 아이콘
        graph_action = QAction(QIcon(resource_path(os.path.join("ui", "icons", "graph2d.png"))), "", self)
        graph_action.setToolTip("Plot Selected File(s) (2D)")
        graph_action.triggered.connect(self.graph_files)
        menu_bar.addAction(graph_action)

        self.setIconSize(QSize(32, 32))

    def init_file(self, menu_bar: QMenuBar):
        file_menu = menu_bar.addMenu("File")
        open_menu = QMenu("Open...", self)

        # Actions for Opening files and folders - 파일 및 폴더 열기 액션
        open_file_action = QAction("Open File(s)", self)
        open_file_action.triggered.connect(self.select_files)

        open_folder_action = QAction("Open Folder", self)
        open_folder_action.triggered.connect(self.select_folder)

        # Actions for Project Save/Open - 프로젝트 저장 및 열기 액션
        open_project_action = QAction("Open Project", self)
        open_project_action.triggered.connect(self.open_project)

        save_project_action = QAction("Save Project", self)
        save_project_action.triggered.connect(self.save_project)

        open_menu.addAction(open_file_action)
        open_menu.addAction(open_folder_action)
        open_menu.addAction(open_project_action)
        file_menu.addMenu(open_menu)

        file_menu.addAction(save_project_action)
        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    # ---------------- Theme Switching Logic - 테마 전환 로직 ----------------
    def toggle_dark_mode(self):
        self.is_dark_mode = self.dark_mode_action.isChecked()

        # Update global app style - 전역 앱 스타일 업데이트
        if self.is_dark_mode:
            self.setStyleSheet(DARK_QSS)
        else:
            self.setStyleSheet(LIGHT_QSS)

        QApplication.instance().is_dark_mode = self.is_dark_mode

        # Update style of all open windows - 모든 열려있는 창의 스타일 업데이트
        for pw in self.plot_windows:
            if pw and pw.isVisible():
                pw.apply_theme(self.is_dark_mode)

        # Edge case: If active plot window exists but is not in the list
        # 특이 케이스: 활성 창이 리스트에 없는 경우 업데이트
        if self.plot_window and self.plot_window not in self.plot_windows:
            if self.plot_window.isVisible():
                self.plot_window.apply_theme(self.is_dark_mode)

    # ---------------- Main UI Layout - 메인 UI 레이아웃 ----------------
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Setup panels for parameters and files - 파라미터 및 파일용 패널 설정
        self.control_panel = ControlPanel(parent=self)
        self.file_panel = FilePanel(self.label, self.loaded_files, self.undo_stack, parent=self)

        splitter.addWidget(self.file_panel.get_widget())
        splitter.addWidget(self.control_panel)

        # Layout resize policies - 레이아웃 크기 조정 정책
        self.file_panel.get_widget().setMinimumWidth(0)
        self.file_panel.get_widget().setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        splitter.setCollapsible(0, True)

        self.control_panel.setMinimumWidth(0)
        self.control_panel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        splitter.setCollapsible(1, True)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.setMinimumHeight(0)
        self.centralWidget().setMinimumHeight(0)

        splitter.setSizes([300, 700])

        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(splitter)

    def init_help_menu(self):
        menu_bar = self.menuBar()
        help_menu = menu_bar.addMenu("Help")
        about_action = QAction("About EZPDF", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def show_about_dialog(self):
        # About Dialog Text - 프로그램 정보 텍스트
        about_text = """
        <b>EZPDF (Easy(EZ) Pair Distribution Function (PDF) Version 1.0.2, August 4, 2026)</b>
        <p>: An easy-to-use software developed by the NSLS-II team for processing X-ray
        diffraction/scattering data and generating pair distribution function (PDF) spectra.</p>

        <p><b>Authors (ordered alphabetically by last name):</b><br>
        <b>National Synchrotron Light Source II:</b><br>
        Gihan Kwon</b><br>
        Cheng-Hung Lin</b><br>
        Ajith Pattammattel</b><br>
        Nghia Vo</b><br>
        Hui Zhong<br><br>
        <b>Northwestern University:</b><br>
        Dustin Zhao<br><br>
        <b>Stony Brook University:</b><br>
        Dongyoon Lee</b><br>
        Yichen Liu</p>
        <p><b>Website:</b><br>
        https://github.com/ezpit/ezpit</p>
        """
        QMessageBox.about(self, "About EZPDF", about_text)

    # ---------------- Project Save/Open Logic (with Directory Info) ----------------
    # 프로젝트 저장/열기 (디렉토리 정보 포함)

    def save_project(self):
        # Select project file path - 프로젝트 파일 경로 선택
        proj_filter = "EZPDF Project (*.proj);;All Files (*)"
        default_name = "my_project.proj"

        last_dir = self._settings.value("last_dir", "")
        save_start = os.path.join(last_dir, default_name) if last_dir else default_name
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Project", save_start, proj_filter)

        if not file_path:
            return

        # Collect absolute paths for all loaded files - 모든 로드된 파일의 절대 경로를 수집합니다.
        file_list = []
        for i in range(self.file_panel.file_list.topLevelItemCount()):
            item = self.file_panel.file_list.topLevelItem(i)
            raw_path = item.data(0, Qt.ItemDataRole.UserRole)
            if raw_path:
                full_path = os.path.abspath(raw_path).replace("\\", "/")
                file_list.append(full_path)

        try:
            # Gather current UI parameters from Control Panel - 컨트롤 패널로부터 현재 설정된 파라미터 수집
            parameters = {
                "basic": self.control_panel.get_basic_parameters(),
                "pdf": self.control_panel.get_pdf_parameters(),
                "cal": self.control_panel.get_cal_parameters(),
                "compton": self.control_panel.get_compton_parameters(),
            }
        except Exception as e:
            QMessageBox.critical(self, "Error Collecting Parameters", f"Failed to get parameters: {e}")
            return

        # [NEW/FIXED] Extract directory information matching xPDFsuite style
        # [신규/수정] xPDFsuite 스타일의 디렉토리 정보 추출
        # inputdir: Directory of the loaded files - 입력 데이터 디렉토리
        input_dir = os.path.abspath(os.path.dirname(file_list[0])).replace("\\", "/") if file_list else ""

        # savedir: Directory where the project is saved - 프로젝트 저장 디렉토리
        save_dir = os.path.abspath(os.path.dirname(file_path)).replace("\\", "/")

        # [FIX] Get the correct key 'background_file' from parameters and ensure it's absolute
        # [수정] parameters에서 정확한 키('background_file')를 가져와 절대 경로로 변환합니다.
        bg_path_raw = parameters["basic"].get("background_file", "")
        bg_full = os.path.abspath(bg_path_raw).replace("\\", "/") if bg_path_raw else ""

        # Update the background path inside parameters as well for future loading
        # 나중에 불러오기를 대비해 parameters 내부의 배경 경로도 절대 경로로 업데이트합니다.
        if bg_path_raw:
            parameters["basic"]["background_file"] = bg_full

        # Construct project data object with absolute path fields - 프로젝트 데이터 객체 구성 (절대 경로 필드 포함)
        project_data = {
            "inputdir": input_dir,  # Data input directory - 데이터 입력 디렉토리
            "savedir": save_dir,  # Project save directory - 프로젝트 저장 디렉토리
            "backgroundfiledir": bg_full,  # Full path to background file - 배경 파일 전체 경로
            "loaded_files": file_list,  # List of loaded file paths - 로드된 파일 경로 리스트
            "parameters": parameters,  # All configuration parameters - 모든 설정 파라미터
        }

        try:
            # Save project to JSON file - 프로젝트를 JSON 파일로 저장합니다.
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(project_data, f, indent=4)
            self.label.setText(f"Project saved to {os.path.basename(file_path)}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save project file:\n{e}")

    def open_project(self):
        # Open and load project file - 프로젝트 파일 열기 및 로드
        proj_filter = "EZPDF Project (*.proj);;All Files (*)"
        last_dir = self._settings.value("last_dir", "")
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Project", last_dir, proj_filter)

        if not file_path:
            return
        self._last_dir = os.path.dirname(file_path)
        self._settings.setValue("last_dir", self._last_dir)
        self._settings.sync()

        try:
            with open(file_path, encoding="utf-8") as f:
                project_data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Open Error", f"Failed to read project file:\n{e}")
            return

        # Reset current UI state for new project - 새로운 프로젝트를 위해 현재 UI 상태 초기화
        self.file_panel.file_list.clear()
        self.loaded_files.clear()
        if self.plot_window:
            self.plot_window.close()
            self.plot_window = None
        self.plot_windows.clear()

        # Restore file list using saved paths - 저장된 경로를 사용하여 파일 리스트 복원
        file_paths = project_data.get("loaded_files", [])
        if file_paths:
            self.populate_file_list(file_paths, is_folder=False)
            self.label.setText(f"Loaded {len(file_paths)} files from project.")
        else:
            self.label.setText("Project loaded (no files).")

        # Restore parameters to UI - 파라미터들을 UI에 복원합니다.
        params = project_data.get("parameters", {})
        try:
            if "basic" in params:
                self.control_panel.set_basic_parameters(params["basic"])
            if "pdf" in params:
                self.control_panel.set_pdf_parameters(params["pdf"])
            if "cal" in params:
                self.control_panel.set_cal_parameters(params["cal"])
            if "compton" in params:
                self.control_panel.set_compton_parameters(params["compton"])
        except Exception as e:
            QMessageBox.warning(self, "Parameter Error", f"Failed to apply some parameters:\n{e}")

        QMessageBox.information(self, "Project Loaded", f"Project '{os.path.basename(file_path)}' loaded.")

    # ---------------- File/Folder Selection Logic - 파일 선택 로직 ----------------
    def select_folder(self):
        # Select folder and add files to list - 폴더를 선택하고 파일을 목록에 추가합니다.
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", self._last_dir)
        if folder:
            self._last_dir = folder
            self._settings.setValue("last_dir", folder)
            self._settings.sync()  # Force immediate write to disk
            self.populate_file_list(folder)

    def select_files(self):
        # Select individual files - 개별 파일들을 선택합니다.
        file_filter = (
            "All Supported Files (*.sq *.fq *.iq *.chi *.xy *.dat *.txt "
            "*.gr *.xyz *.calsq *.calfq *.caliq *.calgr *.compton);;"
            "All Files (*)"
        )
        files, _ = QFileDialog.getOpenFileNames(self, "Select Files", self._last_dir, file_filter)
        if files:
            self._last_dir = os.path.dirname(files[0])
            self._settings.setValue("last_dir", self._last_dir)
            self._settings.sync()  # Force immediate write to disk
            self.populate_file_list(files, is_folder=False)

    def populate_file_list(self, source, is_folder=True):
        # Update UI file list widget - 파일 리스트 UI 업데이트
        add_files_to_list_widget(
            self.file_panel.file_list,
            self.label,
            source,
            self.loaded_files,
            is_folder=is_folder,
            max_name_length=0,
        )
        try:
            # Resize columns to fit content - 가시성을 위해 헤더 열 조정
            header = self.file_panel.file_list.header()
            header.setStretchLastSection(False)
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            self.file_panel.file_list.resizeColumnToContents(1)
        except Exception:
            pass

    # ---------------- Graph Plotting Logic - 그래프 생성 로직 ----------------
    def graph_files(self):
        # Plot selected data files - 선택된 데이터 파일들을 그래프로 출력합니다.
        selected_items = self.file_panel.get_selected_file_paths()
        nsel = len(selected_items)

        if nsel == 1:
            self.label.setText("Graphing 1 file…")
        else:
            self.label.setText(f"Graphing {nsel} files in waterfall mode…")

        # Determine plotting window preference - 그래프 출력 창 환경 설정 확인
        use_new = (
            getattr(self.control_panel, "new_window_checkbox", None)
            and self.control_panel.new_window_checkbox.isChecked()
        )
        ref = None if use_new else self.plot_window

        # Call controller to generate the plot window - 컨트롤러를 호출하여 그래프 창을 생성합니다.
        new_plot_window = load_selected_files(
            selected_items=selected_items,
            label_widget=self.label,
            plot_window_ref=ref,
            control_panel=self.control_panel,
            max_length_file_name=0,
        )

        if new_plot_window:
            # Store association for later updates - 업데이트를 위해 데이터 연결 저장
            new_plot_window.associated_items = selected_items
            new_plot_window.apply_theme(self.is_dark_mode)

            self.plot_window = new_plot_window
            if new_plot_window not in self.plot_windows:
                self.plot_windows.append(new_plot_window)

        self.current_path = selected_items
