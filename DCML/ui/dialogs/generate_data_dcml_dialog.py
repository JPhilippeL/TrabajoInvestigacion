"""
@file generate_data_dcml_dialog.py
@brief Generate Data dialog for raw MPro-v2-like DCML feature generation.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def safe_str(value, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() == "none":
        return default
    return text


def safe_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text or text == "none":
        return default
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def safe_int(value, default: int | None = None) -> int | None:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() == "none":
        return default
    try:
        return int(text)
    except (TypeError, ValueError):
        return default


def safe_float(value, default: float | None = None) -> float | None:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() == "none":
        return default
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


class GenerateDataDCMLDialog(QDialog):
    """Collect inputs for generating DCML-compatible data from raw MPro-v2-like files."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DCML Generate Data")
        self.resize(720, 220)
        self.settings = QSettings("ResearchApp", "DCML_GenerateData")

        self.raw_root_input = QLineEdit(safe_str(self.settings.value("generate/raw_root"), "DCML/MPro-URV_Version2"))
        self.raw_root_btn = QPushButton("Browse...")
        self.raw_root_btn.clicked.connect(lambda: self._browse_dir("Select raw MPro-v2-like dataset root", self.raw_root_input))

        self.output_root_input = QLineEdit(
            safe_str(self.settings.value("generate/output_root"), "DCML/results/generated_data")
        )
        self.output_root_btn = QPushButton("Browse...")
        self.output_root_btn.clicked.connect(lambda: self._browse_dir("Select output prepared dataset root", self.output_root_input))

        self.variant_combo = QComboBox()
        self.variant_combo.addItems(["distance_only", "real_charge", "full"])
        saved_variant = safe_str(self.settings.value("generate/variant"), "distance_only")
        if saved_variant not in {"distance_only", "real_charge", "full"}:
            saved_variant = "distance_only"
        self.variant_combo.setCurrentText(saved_variant)
        self.overwrite_check = QCheckBox("Overwrite output root")
        self.overwrite_check.setChecked(safe_bool(self.settings.value("generate/overwrite"), False))

        form = QFormLayout()
        form.addRow("Raw dataset root (MPro-v2-like):", self._with_button(self.raw_root_input, self.raw_root_btn))
        form.addRow("Output prepared dataset root:", self._with_button(self.output_root_input, self.output_root_btn))
        form.addRow("Variant:", self.variant_combo)
        form.addRow("Overwrite output root:", self.overwrite_check)

        layout = QVBoxLayout()
        layout.addLayout(form)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.setLayout(layout)

    def _with_button(self, line_edit, button):
        container = QWidget()
        hbox = QHBoxLayout(container)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(line_edit)
        hbox.addWidget(button)
        return container

    def _browse_dir(self, title: str, target: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, title)
        if path:
            target.setText(path)

    def accept(self):
        self.settings.setValue("generate/raw_root", self.raw_root_input.text())
        self.settings.setValue("generate/output_root", self.output_root_input.text())
        self.settings.setValue("generate/variant", self.variant_combo.currentText())
        self.settings.setValue("generate/overwrite", self.overwrite_check.isChecked())
        super().accept()

    def get_inputs(self):
        variant = self.variant_combo.currentText()
        return {
            "raw_root": self.raw_root_input.text().strip(),
            "output_root": self.output_root_input.text().strip(),
            "overwrite": self.overwrite_check.isChecked(),
            "variant": variant,
            "protein_charge_method": "pdb2pqr",
            "pdb2pqr_executable": None,
            "ligand_charge_method": "rdkit_gasteiger",
            "openbabel_executable": None,
            "distance_cutoff": None,
            "max_ligand_atoms": 36,
            "max_protein_atoms": 1760,
            "strict": True,
        }
