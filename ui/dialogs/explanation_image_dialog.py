from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit,
    QPushButton, QFileDialog, QDialogButtonBox, QWidget, QHBoxLayout
)
from PySide6.QtCore import QSettings

class FullExplainerConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Full Configuration (Alpha - Delta)")
        self.resize(550, 400) 

        # ---------- QSettings ----------
        self.settings = QSettings("Investigacion", "Analisis Molecular")

        # 1. ---------- Modelo Base ----------
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("Trained base model (.pt)")
        self.model_path_input.setText(self.settings.value("fullConfig/last_model_path", ""))
        self.model_browse_btn = QPushButton("Select...")
        self.model_browse_btn.clicked.connect(self.browse_model)

        # 2. ---------- Molécula SDF ----------
        self.sdf_path_input = QLineEdit()
        self.sdf_path_input.setPlaceholderText("Structure file (.sdf)")
        self.sdf_path_input.setText(self.settings.value("fullConfig/last_sdf_path", ""))
        self.sdf_browse_btn = QPushButton("Select...")
        self.sdf_browse_btn.clicked.connect(self.browse_sdf)

        # 3. ---------- Alfa Path ----------
        self.alpha_path_input = QLineEdit()
        self.alpha_path_input.setPlaceholderText("Pesos para Alfa (.pt)")
        self.alpha_path_input.setText(self.settings.value("fullConfig/last_alpha_path", ""))
        self.alpha_browse_btn = QPushButton("Select...")
        self.alpha_browse_btn.clicked.connect(lambda: self.browse_generic(self.alpha_path_input))

        # 4. ---------- Beta Path ----------
        self.beta_path_input = QLineEdit()
        self.beta_path_input.setPlaceholderText("Pesos para Beta (.pt)")
        self.beta_path_input.setText(self.settings.value("fullConfig/last_beta_path", ""))
        self.beta_browse_btn = QPushButton("Select...")
        self.beta_browse_btn.clicked.connect(lambda: self.browse_generic(self.beta_path_input))

        # 5. ---------- Gamma Path ----------
        self.gamma_path_input = QLineEdit()
        self.gamma_path_input.setPlaceholderText("Pesos para Gamma (.pt)")
        self.gamma_path_input.setText(self.settings.value("fullConfig/last_gamma_path", ""))
        self.gamma_browse_btn = QPushButton("Select...")
        self.gamma_browse_btn.clicked.connect(lambda: self.browse_generic(self.gamma_path_input))

        # 6. ---------- Delta Path ----------
        self.delta_path_input = QLineEdit()
        self.delta_path_input.setPlaceholderText("Pesos para Delta (.pt)")
        self.delta_path_input.setText(self.settings.value("fullConfig/last_delta_path", ""))
        self.delta_browse_btn = QPushButton("Select...")
        self.delta_browse_btn.clicked.connect(lambda: self.browse_generic(self.delta_path_input))

        # ---------- Layout ----------
        form_layout = QFormLayout()
        
        # Agrupamos visualmente
        form_layout.addRow("<b>Entradas Principales:</b>", QWidget()) # Separador visual vacío
        form_layout.addRow("Base Model (.pt):", self._with_button(self.model_path_input, self.model_browse_btn))
        form_layout.addRow("Molecule (.sdf):", self._with_button(self.sdf_path_input, self.sdf_browse_btn))
        
        form_layout.addRow("<b>Configuraciones Específicas:</b>", QWidget())
        form_layout.addRow("Explainer Alfa:", self._with_button(self.alpha_path_input, self.alpha_browse_btn))
        form_layout.addRow("Explainer Beta:", self._with_button(self.beta_path_input, self.beta_browse_btn))
        form_layout.addRow("Explainer Gamma:", self._with_button(self.gamma_path_input, self.gamma_browse_btn))
        form_layout.addRow("Explainer Delta:", self._with_button(self.delta_path_input, self.delta_browse_btn))

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
        hbox.addWidget(line_edit)
        hbox.addWidget(button)
        hbox.setContentsMargins(0, 0, 0, 0)
        return container

    # ---------- Browse Methods ----------
    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Modelo Base", "", "Modelos (*.pt)")
        if path: self.model_path_input.setText(path)

    def browse_sdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Molecule", "", "Molecules (*.sdf)")
        if path: self.sdf_path_input.setText(path)

    def browse_generic(self, target_input):
        """Método genérico para los 4 explainers"""
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Pesos", "", "Pesos (*.pt)")
        if path: target_input.setText(path)

    # ---------- Accept ----------
    def accept(self):
        # Save settings para la próxima vez
        self.settings.setValue("fullConfig/last_model_path", self.model_path_input.text())
        self.settings.setValue("fullConfig/last_sdf_path", self.sdf_path_input.text())
        self.settings.setValue("fullConfig/last_alpha_path", self.alpha_path_input.text())
        self.settings.setValue("fullConfig/last_beta_path", self.beta_path_input.text())
        self.settings.setValue("fullConfig/last_gamma_path", self.gamma_path_input.text())
        self.settings.setValue("fullConfig/last_delta_path", self.delta_path_input.text())
        super().accept()

    # ---------- Get Inputs ----------
    def get_inputs(self):
        """
        Retorna una tupla con las 6 rutas de archivo en orden.
        """
        return (
            self.model_path_input.text(), # Modelo
            self.sdf_path_input.text(),   # SDF
            self.alpha_path_input.text(), # Alfa
            self.beta_path_input.text(),  # Beta
            self.gamma_path_input.text(), # Gamma
            self.delta_path_input.text()  # Delta
        )