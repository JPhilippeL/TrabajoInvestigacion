from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QForGNNsayout, QLineEdit,
    QPushButton, QFileDialog, QDialogButtonBox, QWidget, QHBoxLayout,
    QComboBox  # <--- Importante: añadir QComboBox
)
from PySide6.QtCore import QSettings

class ExplainerComparerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Comparar Explainers")
        self.resize(550, 300) 

        # ---------- QSettings ----------
        self.settings = QSettings("Investigacion", "Analisis Molecular")

        # 0. ---------- Modo (Nuevo Selector) ----------
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["alfa", "beta", "gamma", "delta"])
        # Recuperar último modo usado o default 'beta'
        last_mode = self.settings.value("explainerComparer/last_mode", "beta")
        self.mode_combo.setCurrentText(last_mode)
        # Conectar señal para cambiar sugerencias visuales
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)

        # 1. ---------- Model path ----------
        self.model_path_input = QLineEdit()
        self.model_path_input.setText(self.settings.value("explainerComparer/last_model_path", ""))
        self.model_browse_btn = QPushButton("Seleccionar...")
        self.model_browse_btn.clicked.connect(self.browse_model)

        # 2. ---------- GraphExplainer path ----------
        self.graph_exp_input = QLineEdit()
        self.graph_exp_input.setPlaceholderText("Tus pesos (.pt)")
        self.graph_exp_input.setText(self.settings.value("explainerComparer/last_graph_exp_path", ""))
        self.graph_exp_btn = QPushButton("Seleccionar...")
        self.graph_exp_btn.clicked.connect(self.browse_graph_exp)

        # 3. ---------- GNNExplainer path ----------
        self.gnn_exp_input = QLineEdit()
        self.gnn_exp_input.setPlaceholderText("Opcional para Gamma (.pt)")
        self.gnn_exp_input.setText(self.settings.value("explainerComparer/last_gnn_exp_path", ""))
        self.gnn_exp_btn = QPushButton("Seleccionar...")
        self.gnn_exp_btn.clicked.connect(self.browse_gnn_exp)

        # 4. ---------- SDF path ----------
        self.sdf_path_input = QLineEdit()
        self.sdf_path_input.setText(self.settings.value("explainerComparer/last_sdf_path", ""))
        self.sdf_browse_btn = QPushButton("Seleccionar...")
        self.sdf_browse_btn.clicked.connect(self.browse_sdf)

        # ---------- Layout ----------
        form_layout = QForGNNsayout()
        form_layout.addRow("Modo de Análisis:", self.mode_combo) # <--- Nuevo campo
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
        
        # Ejecutar lógica inicial de UI
        self.on_mode_changed(last_mode)

    def _with_button(self, line_edit, button):
        container = QWidget()
        hbox = QHBoxLayout(container)
        hbox.addWidget(line_edit)
        hbox.addWidget(button)
        hbox.setContentsMargins(0, 0, 0, 0)
        return container

    def on_mode_changed(self, text):
        """Cambia el placeholder para avisar al usuario sobre Gamma"""
        if text == "gamma":
            self.gnn_exp_input.setPlaceholderText("Generalmente no disponible (Dejar vacío)")
        else:
            self.gnn_exp_input.setPlaceholderText("Pesos de GNNExplainer (.pt)")

    # ---------- Browse Methods ----------
    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Modelo Base", "", "Modelos (*.pt)")
        if path: self.model_path_input.setText(path)

    def browse_graph_exp(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar GraphExplainer", "", "Pesos (*.pt)")
        if path: self.graph_exp_input.setText(path)

    def browse_gnn_exp(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar GNNExplainer", "", "Pesos (*.pt)")
        if path: self.gnn_exp_input.setText(path)

    def browse_sdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Molécula", "", "Moléculas (*.sdf)")
        if path: self.sdf_path_input.setText(path)

    # ---------- Accept ----------
    def accept(self):
        # Guardar settings
        self.settings.setValue("explainerComparer/last_mode", self.mode_combo.currentText())
        self.settings.setValue("explainerComparer/last_model_path", self.model_path_input.text())
        self.settings.setValue("explainerComparer/last_graph_exp_path", self.graph_exp_input.text())
        self.settings.setValue("explainerComparer/last_gnn_exp_path", self.gnn_exp_input.text())
        self.settings.setValue("explainerComparer/last_sdf_path", self.sdf_path_input.text())
        super().accept()

    # ---------- Get Inputs ----------
    def get_inputs(self):
        """
        Retorna una tupla con las 5 variables necesarias.
        Nota: Cambié el nombre de get_paths a get_inputs porque ahora retorna también el modo.
        """
        return (
            self.model_path_input.text(),
            self.sdf_path_input.text(),
            self.graph_exp_input.text(),
            self.gnn_exp_input.text(),
            self.mode_combo.currentText() # <--- Nuevo retorno
        )