from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit,
    QPushButton, QFileDialog, QDialogButtonBox, QWidget, QHBoxLayout
)
from PySide6.QtCore import QSettings


class ModelTestDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Seleccionar modelo y molécula")

        # ---------- QSettings ----------
        self.settings = QSettings("Investigacion", "Analisis Molecular")

        # ---------- Model path ----------
        self.model_path_input = QLineEdit()
        self.model_path_input.setText(self.settings.value("modelTest/last_model_path", ""))
        self.model_browse_btn = QPushButton("Seleccionar...")
        self.model_browse_btn.clicked.connect(self.browse_model)

        # ---------- SDF path ----------
        self.sdf_path_input = QLineEdit()
        self.sdf_path_input.setText(self.settings.value("modelTest/last_sdf_path", ""))
        self.sdf_browse_btn = QPushButton("Seleccionar...")
        self.sdf_browse_btn.clicked.connect(self.browse_sdf)

        form_layout = QFormLayout()
        form_layout.addRow("Modelo (.pt):", self._with_button(self.model_path_input, self.model_browse_btn))
        form_layout.addRow("Molécula (.sdf):", self._with_button(self.sdf_path_input, self.sdf_browse_btn))

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

    # ---------- Browse ----------
    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar modelo", "", "Modelos (*.pt)")
        if path:
            self.model_path_input.setText(path)

    def browse_sdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar molécula", "", "Moléculas (*.sdf)")
        if path:
            self.sdf_path_input.setText(path)

    # ---------- Accept ----------
    def accept(self):
        # Guardar rutas en QSettings
        self.settings.setValue("modelTest/last_model_path", self.model_path_input.text())
        self.settings.setValue("modelTest/last_sdf_path", self.sdf_path_input.text())
        super().accept()

    # ---------- Get paths ----------
    def get_paths(self):
        return self.model_path_input.text(), self.sdf_path_input.text()
