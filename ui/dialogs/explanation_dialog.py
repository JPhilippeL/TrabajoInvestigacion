from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit,
    QPushButton, QFileDialog, QDialogButtonBox, QWidget, QHBoxLayout
)
from PySide6.QtCore import QSettings


class ExplanationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Model, Molecule and Targets")
        self.resize(500, 200) # Un poco más grande para que quepa todo bien

        # ---------- QSettings ----------
        self.settings = QSettings("Investigacion", "Analisis Molecular")

        # ---------- Model path ----------
        self.model_path_input = QLineEdit()
        self.model_path_input.setText(self.settings.value("modelTest/last_model_path", ""))
        self.model_browse_btn = QPushButton("Select...")
        self.model_browse_btn.clicked.connect(self.browse_model)

        # ---------- SDF path ----------
        self.sdf_path_input = QLineEdit()
        self.sdf_path_input.setText(self.settings.value("modelTest/last_sdf_path", ""))
        self.sdf_browse_btn = QPushButton("Select...")
        self.sdf_browse_btn.clicked.connect(self.browse_sdf)

        # ---------- Target (TXT) path ----------
        self.target_path_input = QLineEdit()
        self.target_path_input.setText(self.settings.value("modelTest/last_target_path", ""))
        self.target_browse_btn = QPushButton("Select...")
        self.target_browse_btn.clicked.connect(self.browse_target)

        # ---------- Layout ----------
        form_layout = QFormLayout()
        form_layout.addRow("Model (.pt):", self._with_button(self.model_path_input, self.model_browse_btn))
        form_layout.addRow("Molecule (.sdf):", self._with_button(self.sdf_path_input, self.sdf_browse_btn))
        form_layout.addRow("Targets (.txt):", self._with_button(self.target_path_input, self.target_browse_btn))

        layout = QVBoxLayout()
        layout.addLayout(form_layout)

        # ---------- Buttons ----------
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

    # ---------- Browse Methods ----------
    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar modelo", "", "Modelos (*.pt)")
        if path:
            self.model_path_input.setText(path)

    def browse_sdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Molecule", "", "Molecules (*.sdf)")
        if path:
            self.sdf_path_input.setText(path)

    def browse_target(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar targets", "", "Targets (*.txt)")
        if path:
            self.target_path_input.setText(path)

    # ---------- Accept ----------
    def accept(self):
        # Save rutas en QSettings para recordarlas la próxima vez
        self.settings.setValue("modelTest/last_model_path", self.model_path_input.text())
        self.settings.setValue("modelTest/last_sdf_path", self.sdf_path_input.text())
        self.settings.setValue("modelTest/last_target_path", self.target_path_input.text())
        super().accept()

    # ---------- Get paths ----------
    def get_paths(self):
        """Devuelve una tupla con las tres rutas seleccionadas."""
        return (
            self.model_path_input.text(),
            self.sdf_path_input.text(),
            self.target_path_input.text()
        )