"""Generate-data dialog for DEAttentionDTA MPro-v2-like raw datasets."""

from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox, QVBoxLayout

from DEAttentionDTA.ui.dialogs._shared import browse_existing_directory, with_button


def safe_str(value, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() == "none":
        return default
    return text


def safe_int(value, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip()
    if not text or text.lower() == "none":
        return default
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return default


def safe_float(value, default: float) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() == "none":
        return default
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def safe_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text or text == "none":
        return default
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


class PrepareDEAttentionDTADatasetDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DEAttentionDTA Generate Data")
        self.resize(860, 430)
        self.settings = QSettings("ResearchApp", "DEAttentionDTA_PrepareDataset")

        self.raw_root_input = QLineEdit(safe_str(self.settings.value("prepare/raw_root"), "/home/mohamed/Studies/stage/MPro-URV_Version2/MPro-URV_Version2"))
        self.raw_root_btn = QPushButton("Browse...")
        self.raw_root_btn.clicked.connect(lambda: browse_existing_directory(self, "Select raw MPro-v2-like dataset root", self.raw_root_input))

        self.output_root_input = QLineEdit(safe_str(self.settings.value("prepare/output_root"), "DEAttentionDTA/data/urv_dataset_v3b_prepared"))
        self.output_root_btn = QPushButton("Browse...")
        self.output_root_btn.clicked.connect(lambda: browse_existing_directory(self, "Select prepared DEAttentionDTA output root", self.output_root_input))

        self.overwrite_check = QCheckBox("Overwrite output root")
        self.overwrite_check.setChecked(safe_bool(self.settings.value("prepare/overwrite"), False))

        self.max_smiles_spin = QSpinBox()
        self.max_smiles_spin.setRange(0, 10000)
        self.max_smiles_spin.setSpecialValueText("Loader default")
        self.max_smiles_spin.setValue(safe_int(self.settings.value("prepare/max_smiles_len"), 0))

        self.max_protein_spin = QSpinBox()
        self.max_protein_spin.setRange(0, 10000)
        self.max_protein_spin.setSpecialValueText("Loader default")
        self.max_protein_spin.setValue(safe_int(self.settings.value("prepare/max_protein_len"), 0))

        self.strict_check = QCheckBox("Strict mode")
        self.strict_check.setChecked(safe_bool(self.settings.value("prepare/strict"), True))

        form = QFormLayout()
        form.addRow(QLabel("<b>MPro-v2-like source to DEAttentionDTA prepared CSVs</b>"))
        form.addRow("Raw dataset root:", with_button(self.raw_root_input, self.raw_root_btn))
        form.addRow("Output prepared dataset root:", with_button(self.output_root_input, self.output_root_btn))
        form.addRow("Overwrite output root:", self.overwrite_check)
        form.addRow("Max SMILES length:", self.max_smiles_spin)
        form.addRow("Max protein length:", self.max_protein_spin)
        form.addRow("Validation:", self.strict_check)
        form.addRow("Output format:", QLabel("seq_data_all.csv, affinity_all.csv, split_manifest.csv, splits/split_XX CSVs"))

        layout = QVBoxLayout()
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def accept(self) -> None:
        raw_root = self.raw_root_input.text().strip()
        output_root = self.output_root_input.text().strip()
        if not raw_root:
            QMessageBox.warning(self, "DEAttentionDTA Generate Data", "Raw dataset root is required.")
            return
        if not output_root:
            QMessageBox.warning(self, "DEAttentionDTA Generate Data", "Output prepared dataset root is required.")
            return

        for key, value in self._settings_payload().items():
            self.settings.setValue(f"prepare/{key}", value)
        super().accept()

    def _settings_payload(self) -> dict:
        output_root = self.output_root_input.text().strip()
        return {
            "raw_root": self.raw_root_input.text().strip(),
            "output_root": output_root,
            "out_dir": output_root,
            "overwrite": self.overwrite_check.isChecked(),
            "max_smiles_len": self.max_smiles_spin.value(),
            "max_protein_len": self.max_protein_spin.value(),
            "strict": self.strict_check.isChecked(),
        }

    def get_inputs(self) -> dict:
        output_root = self.output_root_input.text().strip()
        return {
            "raw_root": self.raw_root_input.text().strip(),
            "output_root": output_root,
            "out_dir": output_root,
            "overwrite": self.overwrite_check.isChecked(),
            "max_smiles_len": self.max_smiles_spin.value() or None,
            "max_protein_len": self.max_protein_spin.value() or None,
            "strict": self.strict_check.isChecked(),
        }
