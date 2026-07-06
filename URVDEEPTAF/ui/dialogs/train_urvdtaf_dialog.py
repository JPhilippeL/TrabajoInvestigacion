from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit,
    QPushButton, QFileDialog, QDialogButtonBox, QWidget, QHBoxLayout,
    QCheckBox, QSpinBox, QDoubleSpinBox, QLabel, QComboBox
)
from PySide6.QtCore import QSettings

# IMPORTANTE: Asegúrate de importar MODEL_DICT desde tu archivo de modelos
from URVDEEPTAF.Core.urvdtaf_model import MODEL_DICT 

class TrainDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración de Entrenamiento")
        self.resize(650, 600)

        # Obtenemos la lista de modelos dinámicamente
        model_list = list(MODEL_DICT.keys())

        # ---------- QSettings ----------
        self.settings = QSettings("Investigacion", "Training")

        # ================= SECCIÓN 1: RUTAS =================
        self.data_path_input = QLineEdit()
        self.data_path_input.setPlaceholderText("Directorio de datos procesados...")
        self.data_path_input.setText(self.settings.value("train/last_data_path", ""))
        self.data_btn = QPushButton("Seleccionar...")
        self.data_btn.clicked.connect(self.browse_data)

        self.output_base_input = QLineEdit()
        self.output_base_input.setPlaceholderText("Carpeta base de salida (Default: runs)")
        self.output_base_input.setText(self.settings.value("train/last_output_base", "runs"))
        self.output_btn = QPushButton("Seleccionar...")
        self.output_btn.clicked.connect(self.browse_output)

        # ================= SECCIÓN 2: MODELO Y HARDWARE =================
        self.model_combo = QComboBox()
        self.model_combo.addItems(model_list) # <-- Llenado automático desde MODEL_DICT
        
        # Seleccionar el último modelo usado si aún existe en el diccionario
        last_model = self.settings.value("train/last_model", model_list[0])
        if last_model in model_list:
            self.model_combo.setCurrentText(last_model)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu", "cuda:0", "cuda:1"])
        self.device_combo.setCurrentText(self.settings.value("train/last_device", "cuda"))

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 32)
        self.workers_spin.setValue(int(self.settings.value("train/num_workers", 4)))

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(int(self.settings.value("train/random_seed", 42)))

        # ================= SECCIÓN 3: HIPERPARÁMETROS =================
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setSingleStep(16)
        self.batch_spin.setValue(int(self.settings.value("train/batch_size", 32)))

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 5000)
        self.epochs_spin.setValue(int(self.settings.value("train/epochs", 100)))

        self.best_epoch_spin = QSpinBox()
        self.best_epoch_spin.setRange(1, 5000)
        self.best_epoch_spin.setValue(int(self.settings.value("train/save_best", 10)))

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(6)
        self.lr_spin.setRange(0.000001, 1.0)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(float(self.settings.value("train/lr", 0.001)))

        # ================= SECCIÓN 4: LÍMITES Y GRÁFICOS =================
        self.seq_len_spin = QSpinBox()
        self.seq_len_spin.setRange(10, 5000)
        self.seq_len_spin.setValue(int(self.settings.value("train/max_seq_len", 1000)))

        self.pkt_len_spin = QSpinBox()
        self.pkt_len_spin.setRange(10, 5000)
        self.pkt_len_spin.setValue(int(self.settings.value("train/max_pkt_len", 63)))

        self.smi_len_spin = QSpinBox()
        self.smi_len_spin.setRange(10, 5000)
        self.smi_len_spin.setValue(int(self.settings.value("train/max_smi_len", 150)))

        self.plots_check = QCheckBox("Generar y guardar gráficos")
        is_checked = self.settings.value("train/generate_plots", "true") == "true"
        self.plots_check.setChecked(is_checked)


        # ---------- Layout Principal ----------
        form_layout = QFormLayout()
        
        form_layout.addRow(QLabel("<b>1. Rutas del Sistema</b>"))
        form_layout.addRow("Data Path:", self._with_button(self.data_path_input, self.data_btn))
        form_layout.addRow("Output Base:", self._with_button(self.output_base_input, self.output_btn))
        
        form_layout.addRow(QLabel("<br><b>2. Configuración General</b>"))
        form_layout.addRow("Modelo:", self.model_combo)
        form_layout.addRow("Dispositivo:", self.device_combo)
        form_layout.addRow("Semilla Aleatoria:", self.seed_spin)
        form_layout.addRow("Workers (CPU):", self.workers_spin)

        form_layout.addRow(QLabel("<br><b>3. Hiperparámetros de Entrenamiento</b>"))
        form_layout.addRow("Learning Rate:", self.lr_spin)
        form_layout.addRow("Batch Size:", self.batch_spin)
        form_layout.addRow("Total Epochs:", self.epochs_spin)
        form_layout.addRow("Save Best After Epoch:", self.best_epoch_spin)

        form_layout.addRow(QLabel("<br><b>4. Límites de Longitud (Padding)</b>"))
        form_layout.addRow("Max Seq Length:", self.seq_len_spin)
        form_layout.addRow("Max Pocket Length:", self.pkt_len_spin)
        form_layout.addRow("Max SMILES Length:", self.smi_len_spin)
        
        form_layout.addRow(QLabel("")) # Espaciador
        form_layout.addRow("Visualización:", self.plots_check)

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

    def browse_data(self):
        path = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta de Datos")
        if path: self.data_path_input.setText(path)

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta de Salida")
        if path: self.output_base_input.setText(path)

    def accept(self):
        self.settings.setValue("train/last_data_path", self.data_path_input.text())
        self.settings.setValue("train/last_output_base", self.output_base_input.text())
        self.settings.setValue("train/last_model", self.model_combo.currentText())
        self.settings.setValue("train/last_device", self.device_combo.currentText())
        self.settings.setValue("train/num_workers", self.workers_spin.value())
        self.settings.setValue("train/random_seed", self.seed_spin.value())
        self.settings.setValue("train/batch_size", self.batch_spin.value())
        self.settings.setValue("train/epochs", self.epochs_spin.value())
        self.settings.setValue("train/save_best", self.best_epoch_spin.value())
        self.settings.setValue("train/lr", self.lr_spin.value())
        self.settings.setValue("train/max_seq_len", self.seq_len_spin.value())
        self.settings.setValue("train/max_pkt_len", self.pkt_len_spin.value())
        self.settings.setValue("train/max_smi_len", self.smi_len_spin.value())
        self.settings.setValue("train/generate_plots", "true" if self.plots_check.isChecked() else "false")
        
        super().accept()

    def get_inputs(self):
        return {
            "data_path": self.data_path_input.text(),
            "model_name": self.model_combo.currentText(),
            "batch_size": self.batch_spin.value(),
            "epochs": self.epochs_spin.value(),
            "save_best_epoch": self.best_epoch_spin.value(),
            "lr": self.lr_spin.value(),
            "device": self.device_combo.currentText(),
            "seed": self.seed_spin.value(),
            "num_workers": self.workers_spin.value(),
            "generate_plots": self.plots_check.isChecked(),
            "output_base": self.output_base_input.text(),
            "max_seq_len": self.seq_len_spin.value(),
            "max_pkt_len": self.pkt_len_spin.value(),
            "max_smi_len": self.smi_len_spin.value()
        }