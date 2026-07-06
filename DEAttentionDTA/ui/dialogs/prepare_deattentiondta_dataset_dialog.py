"""Dialog used to prepare the official URV v3b DEAttentionDTA dataset."""

from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout

from DEAttentionDTA.ui.dialogs._shared import browse_existing_directory, with_button


class PrepareDEAttentionDTADatasetDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prepare DEAttentionDTA URV v3b Dataset")
        self.resize(880, 390)
        self.settings = QSettings("ResearchApp", "DEAttentionDTA_PrepareDataset")

        self.urv_v3b_dir_input = QLineEdit(self.settings.value("prepare/urv_v3b_dir", "DEAttentionDTA/data/urv_dataset_v3b"))
        self.urv_v3b_dir_btn = QPushButton("Browse...")
        self.urv_v3b_dir_btn.clicked.connect(lambda: browse_existing_directory(self, "Select URV v3b source directory", self.urv_v3b_dir_input))

        self.urv_v2_dir_input = QLineEdit(self.settings.value("prepare/urv_v2_dir", ""))
        self.urv_v2_dir_btn = QPushButton("Browse...")
        self.urv_v2_dir_btn.clicked.connect(lambda: browse_existing_directory(self, "Select MPro-URV Version2 directory", self.urv_v2_dir_input))

        self.out_dir_input = QLineEdit(self.settings.value("prepare/out_dir", "DEAttentionDTA/data/urv_dataset_v3b_prepared"))
        self.out_dir_btn = QPushButton("Browse...")
        self.out_dir_btn.clicked.connect(lambda: browse_existing_directory(self, "Select prepared-dataset output directory", self.out_dir_input))

        self.distance_cutoff_spin = QDoubleSpinBox()
        self.distance_cutoff_spin.setDecimals(2)
        self.distance_cutoff_spin.setRange(0.1, 30.0)
        self.distance_cutoff_spin.setValue(float(self.settings.value("prepare/distance_cutoff", 4.5)))
        self.distance_cutoff_spin.setSuffix(" Å")

        form = QFormLayout()
        form.addRow(QLabel("<b>Prepare Position/Pocket values required by DEAttentionDTA</b>"))
        form.addRow("URV v3b source:", with_button(self.urv_v3b_dir_input, self.urv_v3b_dir_btn))
        form.addRow("MPro-URV Version2 source:", with_button(self.urv_v2_dir_input, self.urv_v2_dir_btn))
        form.addRow("Prepared output directory:", with_button(self.out_dir_input, self.out_dir_btn))
        form.addRow("CIF fallback distance cutoff:", self.distance_cutoff_spin)
        form.addRow("Note:", QLabel("URV v3b remains the official dataset. Version2 is used only to reconstruct residue positions and pockets."))

        layout = QVBoxLayout()
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def accept(self) -> None:
        for key, value in self.get_inputs().items():
            self.settings.setValue(f"prepare/{key}", value)
        super().accept()

    def get_inputs(self) -> dict:
        return {
            "urv_v3b_dir": self.urv_v3b_dir_input.text().strip(),
            "urv_v2_dir": self.urv_v2_dir_input.text().strip(),
            "out_dir": self.out_dir_input.text().strip(),
            "distance_cutoff": self.distance_cutoff_spin.value(),
        }
