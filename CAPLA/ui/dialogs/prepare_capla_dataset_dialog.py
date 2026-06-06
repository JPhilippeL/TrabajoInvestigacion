"""
@file prepare_capla_dataset_dialog.py
@author Mohamed EL BOUKHIARI
@brief Dialog used to prepare the official URV v3b CAPLA dataset.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from CAPLA.ui.dialogs._shared import browse_existing_directory, with_button


class PrepareCAPLADatasetDialog(QDialog):
    """Collect paths required by Prepare_URV_V3B_CAPLA_Dataset.py."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prepare CAPLA URV v3b Dataset")
        self.resize(820, 360)
        self.settings = QSettings("ResearchApp", "CAPLA_PrepareDataset")

        self.urv_v3b_dir_input = QLineEdit(
            self.settings.value("prepare/urv_v3b_dir", "CAPLA/data/urv_dataset_v3b")
        )
        self.urv_v3b_dir_btn = QPushButton("Browse...")
        self.urv_v3b_dir_btn.clicked.connect(
            lambda: browse_existing_directory(
                self,
                "Select URV v3b source directory",
                self.urv_v3b_dir_input,
            )
        )

        self.source_dataset_dir_input = QLineEdit(
            self.settings.value("prepare/source_dataset_dir", "CAPLA/data/urv_dataset")
        )
        self.source_dataset_dir_btn = QPushButton("Browse...")
        self.source_dataset_dir_btn.clicked.connect(
            lambda: browse_existing_directory(
                self,
                "Select existing CAPLA feature dataset",
                self.source_dataset_dir_input,
            )
        )

        self.out_dir_input = QLineEdit(
            self.settings.value("prepare/out_dir", "CAPLA/data/urv_dataset_v3b_prepared")
        )
        self.out_dir_btn = QPushButton("Browse...")
        self.out_dir_btn.clicked.connect(
            lambda: browse_existing_directory(
                self,
                "Select prepared-dataset output directory",
                self.out_dir_input,
            )
        )

        self.feature_mode_combo = QComboBox()
        self.feature_mode_combo.addItems(["copy", "symlink"])
        self.feature_mode_combo.setCurrentText(
            self.settings.value("prepare/feature_mode", "copy")
        )

        form = QFormLayout()
        form.addRow(QLabel("<b>Prepare the official URV v3b dataset for CAPLA</b>"))
        form.addRow(
            "URV v3b source:",
            with_button(self.urv_v3b_dir_input, self.urv_v3b_dir_btn),
        )
        form.addRow(
            "Existing CAPLA features:",
            with_button(self.source_dataset_dir_input, self.source_dataset_dir_btn),
        )
        form.addRow(
            "Prepared output directory:",
            with_button(self.out_dir_input, self.out_dir_btn),
        )
        form.addRow("Feature handling:", self.feature_mode_combo)
        form.addRow(
            "Note:",
            QLabel(
                "This operation prepares CSV files and official splits. "
                "It reuses the existing global/ and pocket/ feature matrices."
            ),
        )

        layout = QVBoxLayout()
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def accept(self):
        values = self.get_inputs()
        for key, value in values.items():
            self.settings.setValue(f"prepare/{key}", value)
        super().accept()

    def get_inputs(self) -> dict:
        return {
            "urv_v3b_dir": self.urv_v3b_dir_input.text().strip(),
            "source_dataset_dir": self.source_dataset_dir_input.text().strip(),
            "out_dir": self.out_dir_input.text().strip(),
            "feature_mode": self.feature_mode_combo.currentText(),
        }
