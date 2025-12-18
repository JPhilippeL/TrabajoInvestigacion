from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit,
    QPushButton, QFileDialog, QDialogButtonBox, QWidget, QHBoxLayout
)
from PySide6.QtCore import QSettings

class ExplainerComparerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Comparar Explainers")
        self.resize(500, 250) # Un poco más ancho por defecto

        # ---------- QSettings ----------
        # Usamos un prefijo 'explainerComparer' para no mezclar settings con el otro dialog
        self.settings = QSettings("Investigacion", "Analisis Molecular")

        # 1. ---------- Model path ----------
        self.model_path_input = QLineEdit()
        self.model_path_input.setText(self.settings.value("explainerComparer/last_model_path", ""))
        self.model_browse_btn = QPushButton("Seleccionar...")
        self.model_browse_btn.clicked.connect(self.browse_model)

        # 2. ---------- GraphExplainer path (Nuevo) ----------
        self.graph_exp_input = QLineEdit()
        self.graph_exp_input.setPlaceholderText("Pesos de GraphExplainer (.pt)")
        self.graph_exp_input.setText(self.settings.value("explainerComparer/last_graph_exp_path", ""))
        self.graph_exp_btn = QPushButton("Seleccionar...")
        self.graph_exp_btn.clicked.connect(self.browse_graph_exp)

        # 3. ---------- GNNExplainer path (Nuevo) ----------
        self.gnn_exp_input = QLineEdit()
        self.gnn_exp_input.setPlaceholderText("Pesos de GNNExplainer (.pt)")
        self.gnn_exp_input.setText(self.settings.value("explainerComparer/last_gnn_exp_path", ""))
        self.gnn_exp_btn = QPushButton("Seleccionar...")
        self.gnn_exp_btn.clicked.connect(self.browse_gnn_exp)

        # 4. ---------- SDF path ----------
        self.sdf_path_input = QLineEdit()
        self.sdf_path_input.setText(self.settings.value("explainerComparer/last_sdf_path", ""))
        self.sdf_browse_btn = QPushButton("Seleccionar...")
        self.sdf_browse_btn.clicked.connect(self.browse_sdf)

        # ---------- Layout ----------
        form_layout = QFormLayout()
        form_layout.addRow("Modelo Base (.pt):", self._with_button(self.model_path_input, self.model_browse_btn))
        form_layout.addRow("Molécula (.sdf):", self._with_button(self.sdf_path_input, self.sdf_browse_btn))
        form_layout.addRow("GraphExplainer (.pt):", self._with_button(self.graph_exp_input, self.graph_exp_btn))
        form_layout.addRow("GNNExplainer (.pt):", self._with_button(self.gnn_exp_input, self.gnn_exp_btn))

        layout = QVBoxLayout()
        layout.addLayout(form_layout)

        # ---------- Buttons ----------
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.setLayout(layout)

    def _with_button(self, line_edit, button):
        """Helper para poner el input y el botón en la misma celda del form"""
        container = QWidget()
        hbox = QHBoxLayout(container)
        hbox.addWidget(line_edit)
        hbox.addWidget(button)
        hbox.setContentsMargins(0, 0, 0, 0)
        return container

    # ---------- Browse Methods ----------
    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Modelo Base", "", "Modelos (*.pt)")
        if path:
            self.model_path_input.setText(path)

    def browse_graph_exp(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar GraphExplainer", "", "Pesos (*.pt)")
        if path:
            self.graph_exp_input.setText(path)

    def browse_gnn_exp(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar GNNExplainer", "", "Pesos (*.pt)")
        if path:
            self.gnn_exp_input.setText(path)

    def browse_sdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Molécula", "", "Moléculas (*.sdf)")
        if path:
            self.sdf_path_input.setText(path)

    # ---------- Accept ----------
    def accept(self):
        # Guardar todas las rutas en QSettings
        self.settings.setValue("explainerComparer/last_model_path", self.model_path_input.text())
        self.settings.setValue("explainerComparer/last_graph_exp_path", self.graph_exp_input.text())
        self.settings.setValue("explainerComparer/last_gnn_exp_path", self.gnn_exp_input.text())
        self.settings.setValue("explainerComparer/last_sdf_path", self.sdf_path_input.text())
        super().accept()

    # ---------- Get paths ----------
    def get_paths(self):
        """Retorna una tupla con las 4 rutas"""
        return (
            self.model_path_input.text(),
            self.sdf_path_input.text(),
            self.graph_exp_input.text(),
            self.gnn_exp_input.text()
        )