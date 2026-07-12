from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QCheckBox,
    QPushButton, QFileDialog, QDialogButtonBox, QWidget, QHBoxLayout,
    QComboBox  
)
from PySide6.QtCore import QSettings

class ExplainerComparerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Comparar Explainers (Análisis Dinámico)")
        self.resize(550, 250) # Un poco más compacto al quitar campos

        # ---------- QSettings ----------
        self.settings = QSettings("Investigacion", "Analisis Molecular")

        # 0. ---------- Modo ----------
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["alfa", "beta", "gamma", "delta"])
        last_mode = self.settings.value("explainerComparer/last_mode", "beta")
        self.mode_combo.setCurrentText(last_mode)

        # 0.5 ---------- Fidelity Mode (Checkbox) ---------- 
        self.fidelity_check = QCheckBox("Usar RegFidelity+ (Ascendente)")
        self.fidelity_check.setToolTip(
            "Marcado: Elimina lo MENOS importante primero (Curva de estabilidad).\n"
            "Desmarcado: Elimina lo MÁS importante primero (Curva de daño)."
        )
        last_fidelity = self.settings.value("explainerComparer/reg_fidelity_mas", True, type=bool)
        self.fidelity_check.setChecked(last_fidelity)

        # 1. ---------- Model path ----------
        self.model_path_input = QLineEdit()
        self.model_path_input.setText(self.settings.value("explainerComparer/last_model_path", ""))
        self.model_browse_btn = QPushButton("Seleccionar...")
        self.model_browse_btn.clicked.connect(self.browse_model)

        # 2. ---------- SDF path ----------
        self.sdf_path_input = QLineEdit()
        self.sdf_path_input.setText(self.settings.value("explainerComparer/last_sdf_path", ""))
        self.sdf_browse_btn = QPushButton("Seleccionar...")
        self.sdf_browse_btn.clicked.connect(self.browse_sdf)

        # 3. ---------- Directorio Madre de Pesos (NUEVO) ----------
        self.weights_dir_input = QLineEdit()
        self.weights_dir_input.setPlaceholderText("Carpeta raíz de los pesos guardados")
        self.weights_dir_input.setText(self.settings.value("explainerComparer/last_weights_dir", ""))
        self.weights_dir_btn = QPushButton("Seleccionar...")
        self.weights_dir_btn.clicked.connect(self.browse_weights_dir)

        # ---------- Layout ----------
        form_layout = QFormLayout()
        form_layout.addRow("Modo de Análisis:", self.mode_combo) 
        form_layout.addRow("Tipo de Métrica:", self.fidelity_check) 
        form_layout.addRow("Modelo Base (.pt):", self._with_button(self.model_path_input, self.model_browse_btn))
        form_layout.addRow("Molécula (.sdf):", self._with_button(self.sdf_path_input, self.sdf_browse_btn))
        form_layout.addRow("Directorio Raíz Pesos:", self._with_button(self.weights_dir_input, self.weights_dir_btn)) # <--- Nuevo campo

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
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Molécula", "", "Moléculas (*.sdf)")
        if path: self.sdf_path_input.setText(path)

    def browse_weights_dir(self):
        # Usamos getExistingDirectory para seleccionar la carpeta madre
        path = QFileDialog.getExistingDirectory(self, "Seleccionar Directorio Raíz de Pesos")
        if path: self.weights_dir_input.setText(path)

    # ---------- Accept ----------
    def accept(self):
        # Guardar settings
        self.settings.setValue("explainerComparer/last_mode", self.mode_combo.currentText())
        self.settings.setValue("explainerComparer/reg_fidelity_mas", self.fidelity_check.isChecked()) 
        self.settings.setValue("explainerComparer/last_model_path", self.model_path_input.text())
        self.settings.setValue("explainerComparer/last_sdf_path", self.sdf_path_input.text())
        self.settings.setValue("explainerComparer/last_weights_dir", self.weights_dir_input.text())
        super().accept()

    # ---------- Get Inputs ----------
    def get_inputs(self):
        """
        Retorna una tupla con las 5 variables necesarias para la nueva función.
        """
        return (
            self.model_path_input.text(),
            self.sdf_path_input.text(),
            self.weights_dir_input.text(), # <--- Devuelve el directorio en vez de archivos individuales
            self.mode_combo.currentText(),
            self.fidelity_check.isChecked()
        )