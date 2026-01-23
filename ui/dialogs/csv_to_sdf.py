from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QPushButton, QFileDialog,
    QHBoxLayout, QDialogButtonBox, QMessageBox
)
from PySide6.QtCore import QSettings
import os

class CSVtoSDFDialog(QDialog):
    session_defaults = {
        "csv_file": "",
        "output_dir": ""
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Convertir CSV (SMILES) a SDFs individuales")
        self.settings = QSettings("Investigacion", "Analisis Molecular")

        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # ---------- CSV file ----------
        self.csv_file_input = QLineEdit()
        self.csv_file_input.setText(self.settings.value("csv2sdf/csv_file", ""))
        self.csv_file_button = QPushButton("Elegir CSV...")
        self.csv_file_button.clicked.connect(self.select_csv_file)
        csv_layout = QHBoxLayout()
        csv_layout.addWidget(self.csv_file_input)
        csv_layout.addWidget(self.csv_file_button)
        form_layout.addRow("Archivo CSV:", csv_layout)

        # ---------- Output directory ----------
        self.output_dir_input = QLineEdit()
        self.output_dir_input.setText(self.settings.value("csv2sdf/output_dir", ""))
        self.output_dir_button = QPushButton("Elegir carpeta...")
        self.output_dir_button.clicked.connect(self.select_output_dir)
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_dir_input)
        output_layout.addWidget(self.output_dir_button)
        form_layout.addRow("Directorio de salida:", output_layout)

        layout.addLayout(form_layout)

        # ---------- Botones OK / Cancel ----------
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.setLayout(layout)

    def select_csv_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo CSV", filter="CSV files (*.csv)"
        )
        if file:
            self.csv_file_input.setText(file)

    def select_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de salida")
        if folder:
            self.output_dir_input.setText(folder)

    def accept(self):
        csv_file = self.csv_file_input.text().strip()
        output_dir = self.output_dir_input.text().strip()

        if not os.path.isfile(csv_file):
            QMessageBox.warning(self, "Error", "Debes seleccionar un archivo CSV válido.")
            return
        if not output_dir:
            QMessageBox.warning(self, "Error", "Debes seleccionar un directorio de salida.")
            return

        # Guardar configuraciones
        self.settings.setValue("csv2sdf/csv_file", csv_file)
        self.settings.setValue("csv2sdf/output_dir", output_dir)

        super().accept()

    def get_values(self):
        return {
            "csv_file": self.csv_file_input.text(),
            "output_dir": self.output_dir_input.text()
        }
