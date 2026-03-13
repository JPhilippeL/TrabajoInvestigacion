from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit,
    QPushButton, QFileDialog, QDialogButtonBox, QWidget, QHBoxLayout
)
from PySide6.QtCore import QSettings


class BatchAllModelsTestDialogPT(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Testear TODOS los modelos en un directorio")

        # ---------- QSettings ----------
        self.settings = QSettings("Investigacion", "Analisis Molecular")

        # Inputs
        self.models_dir_input = QLineEdit()
        self.models_dir_input.setText(self.settings.value("batchAllTest/models_dir", ""))
        self.models_browse_btn = QPushButton("Seleccionar...")
        self.models_browse_btn.clicked.connect(self.browse_models_dir)

        # ---------- PT File (Reemplaza SDF y Target) ----------
        self.pt_file_input = QLineEdit()
        self.pt_file_input.setText(self.settings.value("batchAllTest/pt_file", ""))
        self.pt_browse_btn = QPushButton("Seleccionar...")
        self.pt_browse_btn.clicked.connect(self.browse_pt_file)

        # Form layout
        form_layout = QFormLayout()
        form_layout.addRow("Directorio de modelos:", self._with_button(self.models_dir_input, self.models_browse_btn))
        form_layout.addRow("Archivo Dataset (.pt):", self._with_button(self.pt_file_input, self.pt_browse_btn))

        # Main layout
        layout = QVBoxLayout()
        layout.addLayout(form_layout)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.setLayout(layout)

    def _with_button(self, line_edit, button):
        container = QWidget()
        hbox = QHBoxLayout(container)
        hbox.addWidget(line_edit)
        hbox.addWidget(button)
        hbox.setContentsMargins(0, 0, 0, 0)
        return container

    def browse_models_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Seleccionar directorio de modelos")
        if path:
            self.models_dir_input.setText(path)

    def browse_pt_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar dataset preprocesado", "", "PyTorch files (*.pt)")
        if path:
            self.pt_file_input.setText(path)

    def accept(self):
        # Guardar en QSettings
        self.settings.setValue("batchAllTest/models_dir", self.models_dir_input.text())
        self.settings.setValue("batchAllTest/pt_file", self.pt_file_input.text())
        super().accept()

    def get_paths(self):
        # Ahora devuelve solo dos valores
        return (
            self.models_dir_input.text(),
            self.pt_file_input.text()
        )