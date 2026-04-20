from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit,
    QPushButton, QFileDialog, QDialogButtonBox, QWidget, QHBoxLayout,
    QComboBox  # <-- Importamos QComboBox
)
from PySide6.QtCore import QSettings


class ExplanationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Seleccionar modelo, molécula, targets y explicador")
        self.resize(500, 250) # Un poco más grande para el nuevo campo

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

        # ---------- Target (TXT) path ----------
        self.target_path_input = QLineEdit()
        self.target_path_input.setText(self.settings.value("modelTest/last_target_path", ""))
        self.target_browse_btn = QPushButton("Seleccionar...")
        self.target_browse_btn.clicked.connect(self.browse_target)

        # ---------- Explainer Selector ----------
        self.explainer_combo = QComboBox()
        
        # Lista de todos los explicadores disponibles
        explainers = [
            "GraphExplainer",
            "GNNExplainer",
            "Captum_IntegratedGradients",
            "Captum_InputXGradient",
            "Captum_ShapleyValueSampling",
            "DummyExplainer"
        ]
        self.explainer_combo.addItems(explainers)
        
        # Restaurar la última selección si existe
        last_explainer = self.settings.value("modelTest/last_explainer", "GNNExplainer")
        index = self.explainer_combo.findText(last_explainer)
        if index >= 0:
            self.explainer_combo.setCurrentIndex(index)

        # ---------- Layout ----------
        form_layout = QFormLayout()
        form_layout.addRow("Modelo (.pt):", self._with_button(self.model_path_input, self.model_browse_btn))
        form_layout.addRow("Molécula (.sdf):", self._with_button(self.sdf_path_input, self.sdf_browse_btn))
        form_layout.addRow("Targets (.txt):", self._with_button(self.target_path_input, self.target_browse_btn))
        form_layout.addRow("Explicador:", self.explainer_combo) # <-- Añadimos el selector al formulario

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
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar molécula", "", "Moléculas (*.sdf)")
        if path:
            self.sdf_path_input.setText(path)

    def browse_target(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar targets", "", "Targets (*.txt)")
        if path:
            self.target_path_input.setText(path)

    # ---------- Accept ----------
    def accept(self):
        # Guardar rutas y el explicador en QSettings
        self.settings.setValue("modelTest/last_model_path", self.model_path_input.text())
        self.settings.setValue("modelTest/last_sdf_path", self.sdf_path_input.text())
        self.settings.setValue("modelTest/last_target_path", self.target_path_input.text())
        self.settings.setValue("modelTest/last_explainer", self.explainer_combo.currentText())
        super().accept()

    # ---------- Get paths & settings ----------
    def get_paths(self):
        """Devuelve una tupla con las tres rutas seleccionadas y el explicador."""
        return (
            self.model_path_input.text(),
            self.sdf_path_input.text(),
            self.target_path_input.text(),
            self.explainer_combo.currentText() # <-- Ahora devuelve 4 elementos
        )