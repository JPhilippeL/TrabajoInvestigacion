from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QPushButton, QFileDialog,
    QHBoxLayout, QDialogButtonBox, QLabel
)
from PySide6.QtCore import QSettings
import os

from PySide6.QtWidgets import QCheckBox

class SDFSplitDialog(QDialog):
    session_defaults = {
        "sdf_file": "",
        "target_file": "",
        "output_dir": "",
        "force_rename": False
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dividir SDF y generar targets")
        self.settings = QSettings("Investigacion", "Analisis Molecular")

        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # ---------- SDF file ----------
        self.sdf_file_input = QLineEdit()
        self.sdf_file_input.setText(self.settings.value("split/sdf_file", ""))
        self.sdf_file_button = QPushButton("Elegir SDF...")
        self.sdf_file_button.clicked.connect(self.select_sdf_file)
        sdf_layout = QHBoxLayout()
        sdf_layout.addWidget(self.sdf_file_input)
        sdf_layout.addWidget(self.sdf_file_button)
        form_layout.addRow("Archivo SDF:", sdf_layout)

        # ---------- Target file ----------
        self.target_file_input = QLineEdit()
        self.target_file_input.setText(self.settings.value("split/target_file", ""))
        self.target_file_button = QPushButton("Elegir archivo...")
        self.target_file_button.clicked.connect(self.select_target_file)
        target_layout = QHBoxLayout()
        target_layout.addWidget(self.target_file_input)
        target_layout.addWidget(self.target_file_button)
        form_layout.addRow("Archivo de targets (.txt):", target_layout)

        # ---------- Prefijo ----------
        self.prefix_input = QLineEdit()
        self.prefix_input.setText(self.settings.value("split/name_prefix", "mol"))
        form_layout.addRow("Prefijo para moléculas sin nombre:", self.prefix_input)

        # ---------- Force rename checkbox ----------
        force_layout = QHBoxLayout()
        force_label = QLabel("Sobreescribir todos los nombres de moléculas:")
        self.force_rename_checkbox = QCheckBox()
        self.force_rename_checkbox.setChecked(self.settings.value("split/force_rename", False, type=bool))
        force_layout.addWidget(force_label)
        force_layout.addWidget(self.force_rename_checkbox)
        form_layout.addRow(force_layout)

        # ---------- Output directory ----------
        self.output_dir_input = QLineEdit()
        self.output_dir_input.setText(self.settings.value("split/output_dir", ""))
        self.output_dir_button = QPushButton("Elegir carpeta...")
        self.output_dir_button.clicked.connect(self.select_output_dir)
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_dir_input)
        output_layout.addWidget(self.output_dir_button)
        form_layout.addRow("Directorio de salida:", output_layout)

        layout.addLayout(form_layout)

        # Botones Ok / Cancel
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.setLayout(layout)

    def select_sdf_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Seleccionar archivo SDF", filter="SDF files (*.sdf)")
        if file:
            self.sdf_file_input.setText(file)

    def select_target_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Seleccionar archivo de targets", filter="TXT files (*.txt)")
        if file:
            self.target_file_input.setText(file)

    def select_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de salida")
        if folder:
            self.output_dir_input.setText(folder)

    def accept(self):
        # Guardar configuraciones
        self.settings.setValue("split/sdf_file", self.sdf_file_input.text())
        self.settings.setValue("split/target_file", self.target_file_input.text())
        self.settings.setValue("split/name_prefix", self.prefix_input.text())
        self.settings.setValue("split/output_dir", self.output_dir_input.text())
        self.settings.setValue("split/force_rename", self.force_rename_checkbox.isChecked())
        super().accept()

    def get_values(self):
        return {
            "sdf_file": self.sdf_file_input.text(),
            "target_file": self.target_file_input.text(),
            "name_prefix": self.prefix_input.text(),
            "output_dir": self.output_dir_input.text(),
            "force_rename": self.force_rename_checkbox.isChecked()
        }

