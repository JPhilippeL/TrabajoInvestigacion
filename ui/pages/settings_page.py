"""
@file settings_page.py
@author Mohamed EL BOUKHIARI
@brief Settings page for the Molecular Analysis System GUI.
@details
This page centralizes persistent paths, runtime defaults and application
resource settings.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ui.utils.app_settings import AppSettings
from ui.widgets.path_selector import PathSelector


class SettingsPage(QWidget):
    """
    @brief Page used to configure persistent application settings.
    """

    def __init__(self) -> None:
        """
        @brief Initialize the settings page.
        """
        super().__init__()

        self.setObjectName("SettingsPage")
        self.app_settings = AppSettings()

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(26, 24, 26, 20)
        root_layout.setSpacing(14)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)

        title_layout = QVBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(4)

        title = QLabel("Settings")
        title.setObjectName("PageTitle")

        subtitle = QLabel(
            "Configure dataset paths, model folders, runtime defaults and application resources."
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)

        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        self.reset_button = QPushButton("Reset defaults")
        self.reset_button.clicked.connect(self.reset_defaults)

        self.save_button = QPushButton("Save settings")
        self.save_button.setObjectName("PrimaryButton")
        self.save_button.clicked.connect(self.save_settings)

        header_layout.addLayout(title_layout)
        header_layout.addStretch(1)
        header_layout.addWidget(self.reset_button)
        header_layout.addWidget(self.save_button)

        root_layout.addLayout(header_layout)

        scroll_area = QScrollArea()
        scroll_area.setObjectName("SettingsScroll")
        scroll_area.setWidgetResizable(True)

        content = QWidget()
        content.setObjectName("SettingsContent")

        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 10, 0)
        content_layout.setSpacing(16)

        self._build_dataset_group(content_layout)
        self._build_model_group(content_layout)
        self._build_runtime_group(content_layout)
        self._build_appearance_group(content_layout)

        content_layout.addStretch(1)

        scroll_area.setWidget(content)
        root_layout.addWidget(scroll_area, 1)

        self.load_settings()

    def _build_dataset_group(self, parent_layout: QVBoxLayout) -> None:
        """
        @brief Build dataset path settings.

        @param parent_layout Parent layout.
        @return None.
        """
        group = QGroupBox("Dataset paths")
        form = QFormLayout(group)
        self.dataset_root_selector = PathSelector("directory", "Select dataset root")
        self.ligand_sdf_selector = PathSelector("directory", "Select Ligand_SDF directory")
        self.protein_pdb_selector = PathSelector("directory", "Select Protein_PDB directory")
        self.pic50_file_selector = PathSelector("file", "Select pIC50 file", "Text files (*.txt);;All files (*)")
        self.splits_folder_selector = PathSelector("directory", "Select splits folder")

        form.addRow("Dataset root:", self.dataset_root_selector)
        form.addRow("Ligand SDF:", self.ligand_sdf_selector)
        form.addRow("Protein PDB:", self.protein_pdb_selector)
        form.addRow("pIC50 file:", self.pic50_file_selector)
        form.addRow("Splits folder:", self.splits_folder_selector)

        parent_layout.addWidget(group)

    def _build_model_group(self, parent_layout: QVBoxLayout) -> None:
        """
        @brief Build model folder settings.

        @param parent_layout Parent layout.
        @return None.
        """
        group = QGroupBox("Model and output paths")
        grid = QGridLayout(group)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)

        self.egnn_root_selector = PathSelector("directory", "Select EGNN root")
        self.ednn_root_selector = PathSelector("directory", "Select EDNN root")
        self.deepdta_root_selector = PathSelector("directory", "Select DeepDTA root")
        self.widedta_root_selector = PathSelector("directory", "Select WideDTA root")
        self.exports_dir_selector = PathSelector("directory", "Select exports directory")

        grid.addWidget(QLabel("EGNN root:"), 0, 0)
        grid.addWidget(self.egnn_root_selector, 0, 1)
        grid.addWidget(QLabel("EDNN root:"), 1, 0)
        grid.addWidget(self.ednn_root_selector, 1, 1)
        grid.addWidget(QLabel("DeepDTA root:"), 2, 0)
        grid.addWidget(self.deepdta_root_selector, 2, 1)
        grid.addWidget(QLabel("WideDTA root:"), 3, 0)
        grid.addWidget(self.widedta_root_selector, 3, 1)
        grid.addWidget(QLabel("Exports directory:"), 4, 0)
        grid.addWidget(self.exports_dir_selector, 4, 1)

        grid.setColumnStretch(1, 1)

        parent_layout.addWidget(group)

    def _build_runtime_group(self, parent_layout: QVBoxLayout) -> None:
        """
        @brief Build runtime default settings.

        @param parent_layout Parent layout.
        @return None.
        """
        group = QGroupBox("Runtime defaults")
        form = QFormLayout(group)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda"])

        self.seed_spinbox = QSpinBox()
        self.seed_spinbox.setRange(0, 999999)
        self.seed_spinbox.setSingleStep(1)

        form.addRow("Default device:", self.device_combo)
        form.addRow("Default seed:", self.seed_spinbox)

        parent_layout.addWidget(group)

    def _build_appearance_group(self, parent_layout: QVBoxLayout) -> None:
        """
        @brief Build appearance and resource settings.

        @param parent_layout Parent layout.
        @return None.
        """
        group = QGroupBox("Application resources")
        form = QFormLayout(group)

        self.urv_logo_selector = PathSelector("file", "Select URV logo", "Images (*.png *.jpg *.jpeg *.svg);;All files (*)")
        self.app_logo_selector = PathSelector("file", "Select application logo", "Images (*.png *.jpg *.jpeg *.svg);;All files (*)")
        self.app_icon_selector = PathSelector("file", "Select application icon", "Images (*.png *.jpg *.jpeg *.svg);;All files (*)")

        form.addRow("URV logo:", self.urv_logo_selector)
        form.addRow("Application logo:", self.app_logo_selector)
        form.addRow("Application icon:", self.app_icon_selector)

        parent_layout.addWidget(group)

    def load_settings(self) -> None:
        """
        @brief Load settings into the page widgets.

        @return None.
        """
        self.dataset_root_selector.set_text(self.app_settings.get_value("paths/dataset_root"))
        self.ligand_sdf_selector.set_text(self.app_settings.get_value("paths/ligand_sdf"))
        self.protein_pdb_selector.set_text(self.app_settings.get_value("paths/protein_pdb"))
        self.pic50_file_selector.set_text(self.app_settings.get_value("paths/pic50_file"))
        self.splits_folder_selector.set_text(self.app_settings.get_value("paths/splits_folder"))

        self.egnn_root_selector.set_text(self.app_settings.get_value("paths/egnn_root"))
        self.ednn_root_selector.set_text(self.app_settings.get_value("paths/ednn_root"))
        self.deepdta_root_selector.set_text(self.app_settings.get_value("paths/deepdta_root"))
        self.widedta_root_selector.set_text(self.app_settings.get_value("paths/widedta_root"))
        self.exports_dir_selector.set_text(self.app_settings.get_value("paths/exports_dir"))

        device = self.app_settings.get_value("runtime/default_device")
        index = self.device_combo.findText(device)

        if index >= 0:
            self.device_combo.setCurrentIndex(index)

        try:
            self.seed_spinbox.setValue(int(self.app_settings.get_value("runtime/default_seed")))
        except ValueError:
            self.seed_spinbox.setValue(42)

        self.urv_logo_selector.set_text(self.app_settings.get_value("appearance/urv_logo"))
        self.app_logo_selector.set_text(self.app_settings.get_value("appearance/app_logo"))
        self.app_icon_selector.set_text(self.app_settings.get_value("appearance/app_icon"))

    def save_settings(self) -> None:
        """
        @brief Save widget values into persistent settings.

        @return None.
        """
        self.app_settings.set_value("paths/dataset_root", self.dataset_root_selector.text())
        self.app_settings.set_value("paths/ligand_sdf", self.ligand_sdf_selector.text())
        self.app_settings.set_value("paths/protein_pdb", self.protein_pdb_selector.text())
        self.app_settings.set_value("paths/pic50_file", self.pic50_file_selector.text())
        self.app_settings.set_value("paths/splits_folder", self.splits_folder_selector.text())

        self.app_settings.set_value("paths/egnn_root", self.egnn_root_selector.text())
        self.app_settings.set_value("paths/ednn_root", self.ednn_root_selector.text())
        self.app_settings.set_value("paths/deepdta_root", self.deepdta_root_selector.text())
        self.app_settings.set_value("paths/widedta_root", self.widedta_root_selector.text())
        self.app_settings.set_value("paths/exports_dir", self.exports_dir_selector.text())

        self.app_settings.set_value("runtime/default_device", self.device_combo.currentText())
        self.app_settings.set_value("runtime/default_seed", self.seed_spinbox.value())

        self.app_settings.set_value("appearance/urv_logo", self.urv_logo_selector.text())
        self.app_settings.set_value("appearance/app_logo", self.app_logo_selector.text())
        self.app_settings.set_value("appearance/app_icon", self.app_icon_selector.text())

        self.app_settings.sync()

        QMessageBox.information(
            self,
            "Settings saved",
            "Application settings were saved successfully.",
        )

    def reset_defaults(self) -> None:
        """
        @brief Reset settings to default values and reload the page.

        @return None.
        """
        self.app_settings.reset_defaults()
        self.app_settings.sync()
        self.load_settings()

        QMessageBox.information(
            self,
            "Settings reset",
            "Application settings were reset to their default values.",
        )
