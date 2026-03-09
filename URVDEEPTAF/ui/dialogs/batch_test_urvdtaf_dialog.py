from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit,
    QPushButton, QFileDialog, QDialogButtonBox, QWidget, QHBoxLayout,
    QCheckBox, QSpinBox, QLabel, QComboBox
)
from PySide6.QtCore import QSettings

class BatchTestDialog(QDialog): # Le cambié el nombre a BatchTestDialog para diferenciarlo
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración de Evaluación Múltiple (Batch Test)")
        self.resize(600, 500)

        # ---------- QSettings ----------
        self.settings = QSettings("Investigacion", "Testing")

        # ================= SECCIÓN 1: RUTAS DE ARCHIVOS =================
        
        # CAMBIO 1: Ahora pedimos un directorio de modelos, no un archivo .pt
        self.models_dir_input = QLineEdit()
        self.models_dir_input.setPlaceholderText("Carpeta madre con los modelos entrenados...")
        self.models_dir_input.setText(self.settings.value("test/last_models_dir", ""))
        self.models_btn = QPushButton("Seleccionar...")
        self.models_btn.clicked.connect(self.browse_models_dir)

        self.data_path_input = QLineEdit()
        self.data_path_input.setPlaceholderText("Directorio de datos (train/val/test)...")
        self.data_path_input.setText(self.settings.value("test/last_data_path", ""))
        self.data_btn = QPushButton("Seleccionar...")
        self.data_btn.clicked.connect(self.browse_data)

        self.output_base_input = QLineEdit()
        self.output_base_input.setPlaceholderText("Carpeta base (Default: visuals/saved)")
        self.output_base_input.setText(self.settings.value("test/last_output_base", "visuals/saved"))
        self.output_btn = QPushButton("Seleccionar...")
        self.output_btn.clicked.connect(self.browse_output)

        # ================= SECCIÓN 2: HARDWARE Y BATCH =================
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu", "cuda:0", "cuda:1"])
        self.device_combo.setCurrentText(self.settings.value("test/last_device", "cuda"))

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setSingleStep(16)
        self.batch_spin.setValue(int(self.settings.value("test/batch_size", 32)))

        # ================= SECCIÓN 3: LÍMITES DE LONGITUD =================
        self.seq_len_spin = QSpinBox()
        self.seq_len_spin.setRange(10, 5000)
        self.seq_len_spin.setValue(int(self.settings.value("test/max_seq_len", 1000)))

        self.pkt_len_spin = QSpinBox()
        self.pkt_len_spin.setRange(10, 5000)
        self.pkt_len_spin.setValue(int(self.settings.value("test/max_pkt_len", 63)))

        self.smi_len_spin = QSpinBox()
        self.smi_len_spin.setRange(10, 5000)
        self.smi_len_spin.setValue(int(self.settings.value("test/max_smi_len", 150)))

        # ================= SECCIÓN 4: OPCIONES DE SALIDA =================
        self.plots_check = QCheckBox("Generar gráficos de diagnóstico (Residuales, etc.)")
        self.plots_check.setChecked(self.settings.value("test/generate_plots", "true") == "true")

        self.predictions_check = QCheckBox("Guardar predicciones en CSV (test_predictions.csv)")
        self.predictions_check.setChecked(self.settings.value("test/predictions", "false") == "true")


        # ---------- Layout Principal ----------
        form_layout = QFormLayout()
        
        form_layout.addRow(QLabel("<b>1. Modelos y Datos</b>"))
        # CAMBIO 2: Actualizamos la etiqueta
        form_layout.addRow("Models Directory:", self._with_button(self.models_dir_input, self.models_btn))
        form_layout.addRow("Data Path:", self._with_button(self.data_path_input, self.data_btn))
        form_layout.addRow("Output Base:", self._with_button(self.output_base_input, self.output_btn))
        
        form_layout.addRow(QLabel("<br><b>2. Configuración de Inferencia</b>"))
        form_layout.addRow("Dispositivo:", self.device_combo)
        form_layout.addRow("Batch Size:", self.batch_spin)

        form_layout.addRow(QLabel("<br><b>3. Límites de Longitud (Padding)</b>"))
        form_layout.addRow("Max Seq Length:", self.seq_len_spin)
        form_layout.addRow("Max Pocket Length:", self.pkt_len_spin)
        form_layout.addRow("Max SMI Length:", self.smi_len_spin)
        
        form_layout.addRow(QLabel("<br><b>4. Resultados</b>"))
        form_layout.addRow(self.plots_check)
        form_layout.addRow(self.predictions_check)

        layout = QVBoxLayout()
        layout.addLayout(form_layout)

        # ---------- Botones de Aceptar/Cancelar ----------
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

    # ---------- Eventos de Selección de Carpetas y Archivos ----------
    
    # CAMBIO 3: Usamos getExistingDirectory en lugar de getOpenFileName
    def browse_models_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta Madre de Modelos")
        if path: self.models_dir_input.setText(path)

    def browse_data(self):
        path = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta de Datos")
        if path: self.data_path_input.setText(path)

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta de Salida")
        if path: self.output_base_input.setText(path)

    # ---------- Guardar Configuración (Accept) ----------
    def accept(self):
        self.settings.setValue("test/last_models_dir", self.models_dir_input.text())
        self.settings.setValue("test/last_data_path", self.data_path_input.text())
        self.settings.setValue("test/last_output_base", self.output_base_input.text())
        self.settings.setValue("test/last_device", self.device_combo.currentText())
        self.settings.setValue("test/batch_size", self.batch_spin.value())
        self.settings.setValue("test/max_seq_len", self.seq_len_spin.value())
        self.settings.setValue("test/max_pkt_len", self.pkt_len_spin.value())
        self.settings.setValue("test/max_smi_len", self.smi_len_spin.value())
        
        self.settings.setValue("test/generate_plots", "true" if self.plots_check.isChecked() else "false")
        self.settings.setValue("test/predictions", "true" if self.predictions_check.isChecked() else "false")
        
        super().accept()

    # ---------- Obtener Parámetros Finales ----------
    def get_inputs(self):
        return {
            "models_dir": self.models_dir_input.text(), # CAMBIO 4: Devolvemos 'models_dir'
            "data_path": self.data_path_input.text(),
            "batch_size": self.batch_spin.value(),
            "device": self.device_combo.currentText(),
            "generate_plots": self.plots_check.isChecked(),
            "predictions": self.predictions_check.isChecked(),
            "output_base": self.output_base_input.text(),
            "max_seq_len": self.seq_len_spin.value(),
            "max_pkt_len": self.pkt_len_spin.value(),
            "max_smi_len": self.smi_len_spin.value()
        }