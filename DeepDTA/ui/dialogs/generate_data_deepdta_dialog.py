from __future__ import annotations

import os

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout, QWidget

from DeepDTA.utils.constants import MODULE_ROOT


class GenerateDataDeepDTADialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DeepDTA Generate MPro-URV Data")
        self.resize(720, 240)
        self.settings = QSettings("ResearchApp", "DeepDTA_GenerateData")
        self.source_input = QLineEdit(self.settings.value("source_root", ""))
        self.output_input = QLineEdit(self.settings.value("output_root", os.path.join(MODULE_ROOT, "data", "mpro_urv")))
        self.protein_input = QLineEdit(self.settings.value("protein_pdb_path", ""))
        self.overwrite_check = QCheckBox("Overwrite existing output")
        self.overwrite_check.setChecked(self.settings.value("overwrite", False, type=bool))
        form = QFormLayout()
        form.addRow("Raw MPro-URV root:", self._path_row(self.source_input, self.browse_source))
        form.addRow("Output dataset root:", self._path_row(self.output_input, self.browse_output))
        form.addRow("Protein PDB path:", self._path_row(self.protein_input, self.browse_protein))
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
        path = QFileDialog.getExistingDirectory(self, "Select raw MPro-URV root")
        if path:
            self.source_input.setText(path)

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select DeepDTA output root")
        if path:
            self.output_input.setText(path)

    def browse_protein(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select protein PDB", filter="PDB files (*.pdb);;All files (*)")
        if path:
            self.protein_input.setText(path)

    def accept(self):
        for key, edit in (("source_root", self.source_input), ("output_root", self.output_input), ("protein_pdb_path", self.protein_input)):
            self.settings.setValue(key, edit.text())
        self.settings.setValue("overwrite", self.overwrite_check.isChecked())
        super().accept()

    def get_inputs(self) -> dict:
        source = self.source_input.text().strip()
        output = self.output_input.text().strip()
        if not source or not os.path.isdir(source):
            raise ValueError("Raw MPro-URV root is missing or is not a directory.")
        if not output:
            raise ValueError("Output dataset root is required.")
        protein = self.protein_input.text().strip()
        if protein and not os.path.isfile(protein):
            raise ValueError("Protein PDB path does not exist.")
        return {"source_root": source, "output_root": output, "protein_pdb_path": protein or None, "overwrite": self.overwrite_check.isChecked()}
