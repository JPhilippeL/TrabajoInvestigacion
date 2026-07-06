from __future__ import annotations

import os

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout, QWidget

from WideDTA.utils.constants import MODULE_ROOT


class GenerateDataWideDTADialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("WideDTA Generate MPro-URV Data")
        self.resize(720, 260)
        self.settings = QSettings("ResearchApp", "WideDTA_GenerateData")
        self.source_input = QLineEdit(self.settings.value("source_root", ""))
        self.output_input = QLineEdit(self.settings.value("output_root", os.path.join(MODULE_ROOT, "data", "mpro_urv")))
        self.motif_combo = QComboBox()
        self.motif_combo.addItems(["technical motif baseline: motif2.txt = proteins.txt"])
        self.overwrite_check = QCheckBox("Overwrite existing output")
        self.overwrite_check.setChecked(self.settings.value("overwrite", False, type=bool))
        form = QFormLayout()
        form.addRow("Raw MPro-URV or DeepDTA-format root:", self._path_row(self.source_input, self.browse_source))
        form.addRow("Output dataset root:", self._path_row(self.output_input, self.browse_output))
        form.addRow("Motif mode:", self.motif_combo)
        form.addRow("Overwrite:", self.overwrite_check)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _path_row(self, edit, slot):
        button = QPushButton("Browse...")
        button.clicked.connect(slot)
        box = QWidget()
        hbox = QHBoxLayout(box)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(edit)
        hbox.addWidget(button)
        return box

    def browse_source(self):
        path = QFileDialog.getExistingDirectory(self, "Select MPro-URV source root")
        if path:
            self.source_input.setText(path)

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select WideDTA output root")
        if path:
            self.output_input.setText(path)

    def accept(self):
        self.settings.setValue("source_root", self.source_input.text())
        self.settings.setValue("output_root", self.output_input.text())
        self.settings.setValue("overwrite", self.overwrite_check.isChecked())
        super().accept()

    def get_inputs(self) -> dict:
        source = self.source_input.text().strip()
        output = self.output_input.text().strip()
        if not source or not os.path.isdir(source):
            raise ValueError("MPro-URV source root is missing or is not a directory.")
        if not output:
            raise ValueError("Output dataset root is required.")
        return {"source_root": source, "output_root": output, "motif_mode": "technical_motif_baseline", "overwrite": self.overwrite_check.isChecked()}
