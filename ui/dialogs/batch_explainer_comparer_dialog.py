from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit,
    QPushButton, QFileDialog, QDialogButtonBox, QWidget, QHBoxLayout,
    QComboBox, QLabel
)
from PySide6.QtCore import QSettings

class BatchComparerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Comparer: Directorio Completo")
        self.resize(600, 250) 

        # ---------- QSettings ----------
        self.settings = QSettings("Investigacion", "Analisis Molecular")

        # 0. ---------- Modo (Selector) ----------
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["alfa", "beta", "gamma", "delta"])
        # Recuperar último modo usado
        last_mode = self.settings.value("batchComparer/last_mode", "beta")
        self.mode_combo.setCurrentText(last_mode)

        # 1. ---------- Model path (Archivo .pt) ----------
        self.model_path_input = QLineEdit()
        self.model_path_input.setText(self.settings.value("batchComparer/last_model_path", ""))
        self.model_browse_btn = QPushButton("Seleccionar...")
        self.model_browse_btn.clicked.connect(self.browse_model)

        # 2. ---------- SDF Directory (Directorio) ----------
        self.sdf_dir_input = QLineEdit()
        self.sdf_dir_input.setPlaceholderText("Carpeta con archivos .sdf")
        self.sdf_dir_input.setText(self.settings.value("batchComparer/last_sdf_dir", ""))
        self.sdf_dir_btn = QPushButton("Seleccionar...")
        self.sdf_dir_btn.clicked.connect(self.browse_sdf_dir)

        # 3. ---------- Weights Root Directory (Directorio Raíz) ----------
        self.weights_dir_input = QLineEdit()
        self.weights_dir_input.setPlaceholderText("Carpeta padre (debe contener subcarpetas alfa/, beta/...)")
        self.weights_dir_input.setText(self.settings.value("batchComparer/last_weights_dir", ""))
        self.weights_dir_btn = QPushButton("Seleccionar...")
        self.weights_dir_btn.clicked.connect(self.browse_weights_dir)

        # ---------- Layout ----------
        form_layout = QFormLayout()
        
        # Fila Modo
        form_layout.addRow("Modo de Análisis:", self.mode_combo)
        
        # Fila Modelo
        form_layout.addRow("Modelo Base (.pt):", self._with_button(self.model_path_input, self.model_browse_btn))
        
        # Separador visual o espacio
        form_layout.addRow(QLabel("--- Directorios de Datos ---"))
        
        # Fila SDFs
        form_layout.addRow("Directorio SDFs:", self._with_button(self.sdf_dir_input, self.sdf_dir_btn))
        
        # Fila Pesos
        form_layout.addRow("Dir. Raíz Pesos:", self._with_button(self.weights_dir_input, self.weights_dir_btn))

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
        # Selecciona UN ARCHIVO
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Modelo Base", "", "Modelos (*.pt)")
        if path: self.model_path_input.setText(path)

    def browse_sdf_dir(self):
        # Selecciona UN DIRECTORIO
        path = QFileDialog.getExistingDirectory(self, "Seleccionar Directorio de SDFs")
        if path: self.sdf_dir_input.setText(path)

    def browse_weights_dir(self):
        # Selecciona UN DIRECTORIO
        path = QFileDialog.getExistingDirectory(self, "Seleccionar Raíz de Pesos")
        if path: self.weights_dir_input.setText(path)

    # ---------- Accept ----------
    def accept(self):
        # Guardar settings para recordar la próxima vez
        self.settings.setValue("batchComparer/last_mode", self.mode_combo.currentText())
        self.settings.setValue("batchComparer/last_model_path", self.model_path_input.text())
        self.settings.setValue("batchComparer/last_sdf_dir", self.sdf_dir_input.text())
        self.settings.setValue("batchComparer/last_weights_dir", self.weights_dir_input.text())
        super().accept()

    # ---------- Get Inputs ----------
    def get_inputs(self):
        """
        Retorna:
        1. Ruta del modelo (.pt)
        2. Ruta del directorio de SDFs
        3. Ruta del directorio raíz de pesos
        4. Modo seleccionado (str)
        """
        return (
            self.model_path_input.text(),
            self.sdf_dir_input.text(),
            self.weights_dir_input.text(),
            self.mode_combo.currentText()
        )