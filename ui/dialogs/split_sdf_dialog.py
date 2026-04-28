from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QPushButton, QFileDialog,
    QHBoxLayout, QDialogButtonBox
)
from PySide6.QtCore import QSettings

class SDFSplitDialog(QDialog):
    session_defaults = {
        "sdf_file": "",
        "output_dir": ""
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Split SDF en moléculas individuales")
        self.settings = QSettings("Investigacion", "Analisis Molecular")

        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # ---------- SDF file ----------
        self.sdf_file_input = QLineEdit()
        self.sdf_file_input.setText(self.settings.value("split/sdf_file", ""))
        self.sdf_file_button = QPushButton("Choose SDF...")
        self.sdf_file_button.clicked.connect(self.select_sdf_file)
        sdf_layout = QHBoxLayout()
        sdf_layout.addWidget(self.sdf_file_input)
        sdf_layout.addWidget(self.sdf_file_button)
        form_layout.addRow("SDF File:", sdf_layout)

        # ---------- Output directory ----------
        self.output_dir_input = QLineEdit()
        self.output_dir_input.setText(self.settings.value("split/output_dir", ""))
        self.output_dir_button = QPushButton("Choose folder...")
        self.output_dir_button.clicked.connect(self.select_output_dir)
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_dir_input)
        output_layout.addWidget(self.output_dir_button)
        form_layout.addRow("Output Directory:", output_layout)

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

    def select_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de salida")
        if folder:
            self.output_dir_input.setText(folder)

    def accept(self):
        # Save configuraciones
        self.settings.setValue("split/sdf_file", self.sdf_file_input.text())
        self.settings.setValue("split/output_dir", self.output_dir_input.text())
        super().accept()

    def get_values(self):
        return {
            "sdf_file": self.sdf_file_input.text(),
            "output_dir": self.output_dir_input.text()
        }
