"""
@file prepare_capla_dataset_dialog.py
@author Mohamed EL BOUKHIARI
@brief Dialog used to generate a CAPLA dataset from raw MPro-URV_Version2.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QComboBox,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from CAPLA.ui.dialogs._shared import browse_existing_directory, with_button


class PrepareCAPLADatasetDialog(QDialog):
    """Collect paths required by generate_capla_from_mpro_v2.py."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generate CAPLA Data")
        self.resize(840, 420)
        self.settings = QSettings("ResearchApp", "CAPLA_PrepareDataset")

        self.raw_root_input = QLineEdit(
            self.settings.value("prepare/raw_root", "CAPLA/data/MPro-URV_Version2")
        )
        self.raw_root_btn = QPushButton("Browse...")
        self.raw_root_btn.clicked.connect(
            lambda: browse_existing_directory(
                self,
                "Select raw MPro-URV_Version2 root",
                self.raw_root_input,
            )
        )

        self.output_root_input = QLineEdit(
            self.settings.value("prepare/output_root", "CAPLA/data/mpro_urv_v2_prepared")
        )
        self.output_root_btn = QPushButton("Browse...")
        self.output_root_btn.clicked.connect(
            lambda: browse_existing_directory(
                self,
                "Select prepared-dataset output root",
                self.output_root_input,
            )
        )

        self.feature_source_root_input = QLineEdit(
            self.settings.value("prepare/feature_source_root", "CAPLA/data/urv_dataset")
        )
        self.feature_source_root_btn = QPushButton("Browse...")
        self.feature_source_root_btn.clicked.connect(
            lambda: browse_existing_directory(
                self,
                "Select existing CAPLA feature source root",
                self.feature_source_root_input,
            )
        )

        self.feature_mode_combo = QComboBox()
        self.feature_mode_combo.addItems(["generate", "copy_existing", "symlink_existing", "validate_existing"])
        self.feature_mode_combo.setCurrentText(
            self.settings.value("prepare/feature_mode", "generate")
        )
        self.feature_mode_combo.currentTextChanged.connect(self._update_feature_source_enabled)
        self.secondary_structure_mode_combo = QComboBox()
        self.secondary_structure_mode_combo.addItems(["dssp", "coil_fallback"])
        self.secondary_structure_mode_combo.setCurrentText(
            self.settings.value("prepare/secondary_structure_mode", "dssp")
        )
        self.pocket_cutoff_spin = QDoubleSpinBox()
        self.pocket_cutoff_spin.setDecimals(2)
        self.pocket_cutoff_spin.setRange(0.1, 50.0)
        self.pocket_cutoff_spin.setSingleStep(0.1)
        self.pocket_cutoff_spin.setValue(float(self.settings.value("prepare/pocket_cutoff", 4.5)))
        self.overwrite_check = QCheckBox("Overwrite output root")
        self.overwrite_check.setChecked(
            str(self.settings.value("prepare/overwrite", "false")).lower() in {"true", "1", "yes"}
        )

        form = QFormLayout()
        form.addRow(QLabel("<b>Generate CAPLA data from raw MPro-URV_Version2</b>"))
        form.addRow(
            "Raw MPro-URV_Version2 root:",
            with_button(self.raw_root_input, self.raw_root_btn),
        )
        form.addRow(
            "Output prepared dataset root:",
            with_button(self.output_root_input, self.output_root_btn),
        )
        form.addRow("Overwrite:", self.overwrite_check)
        form.addRow(
            "Existing CAPLA feature source root:",
            with_button(self.feature_source_root_input, self.feature_source_root_btn),
        )
        form.addRow("Pocket cutoff:", self.pocket_cutoff_spin)
        form.addRow("Secondary structure mode:", self.secondary_structure_mode_combo)
        form.addRow("Feature handling:", self.feature_mode_combo)
        form.addRow(
            "Note:",
            QLabel(
                "Default generation uses DSSP/mkdssp for secondary structure. "
                "Existing feature source is not used in generate mode and is only required for fallback feature modes."
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
        self._update_feature_source_enabled(self.feature_mode_combo.currentText())

    def _update_feature_source_enabled(self, mode: str) -> None:
        enabled = mode != "generate"
        self.feature_source_root_input.setEnabled(enabled)
        self.feature_source_root_btn.setEnabled(enabled)

    def accept(self):
        values = self.get_inputs()
        for key, value in values.items():
            self.settings.setValue(f"prepare/{key}", value)
        super().accept()

    def get_inputs(self) -> dict:
        return {
            "raw_root": self.raw_root_input.text().strip(),
            "output_root": self.output_root_input.text().strip(),
            "overwrite": self.overwrite_check.isChecked(),
            "pocket_cutoff": self.pocket_cutoff_spin.value(),
            "secondary_structure_mode": self.secondary_structure_mode_combo.currentText(),
            "feature_mode": self.feature_mode_combo.currentText(),
            "feature_source_root": self.feature_source_root_input.text().strip(),
        }
