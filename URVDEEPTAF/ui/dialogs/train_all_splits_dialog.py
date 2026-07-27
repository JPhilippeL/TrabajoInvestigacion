from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit,
    QPushButton, QFileDialog, QDialogButtonBox, QWidget, QHBoxLayout,
    QSpinBox, QLabel
)
from PySide6.QtCore import QSettings

class TrainAllSplitsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración de Entrenamiento por Splits")
        self.resize(500, 300)

        # ---------- QSettings ----------
        self.settings = QSettings("Investigacion", "TrainSplits")

        # ================= SECCIÓN 1: RUTAS =================
        self.mother_dir_input = QLineEdit()
        self.mother_dir_input.setPlaceholderText("Directorio madre de los splits...")
        self.mother_dir_input.setText(self.settings.value("splits/mother_dir", ""))
        self.mother_btn = QPushButton("Seleccionar...")
        self.mother_btn.clicked.connect(self.browse_mother_dir)

        self.output_dir_input = QLineEdit()
        self.output_dir_input.setPlaceholderText("Carpeta raíz de guardado...")
        self.output_dir_input.setText(self.settings.value("splits/output_dir", "./mis_modelos_guardados"))
        self.output_btn = QPushButton("Seleccionar...")
        self.output_btn.clicked.connect(self.browse_output)

        self.target_file_input = QLineEdit()
        self.target_file_input.setPlaceholderText("Nombre del archivo target (ej. targets.csv)")
        self.target_file_input.setText(self.settings.value("splits/target_file", "targets.csv"))

        # ================= SECCIÓN 2: HIPERPARÁMETROS =================
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setSingleStep(16)
        self.batch_spin.setValue(int(self.settings.value("splits/batch_size", 32)))

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 5000)
        self.epochs_spin.setValue(int(self.settings.value("splits/epochs", 50)))

        # ---------- Layout Principal ----------
        form_layout = QFormLayout()
        
        form_layout.addRow(QLabel("<b>1. Rutas del Sistema</b>"))
        form_layout.addRow("Carpeta Madre de Splits:", self._with_button(self.mother_dir_input, self.mother_btn))
        form_layout.addRow("Carpeta de Salida:", self._with_button(self.output_dir_input, self.output_btn))
        form_layout.addRow("Archivo Target:", self.target_file_input)
        
        form_layout.addRow(QLabel("<br><b>2. Parámetros Base</b>"))
        form_layout.addRow("Batch Size:", self.batch_spin)
        form_layout.addRow("Total Epochs:", self.epochs_spin)

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

    def browse_mother_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta Madre de Splits")
        if path: self.mother_dir_input.setText(path)

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta de Salida")
        if path: self.output_dir_input.setText(path)

    def accept(self):
        # Guardar valores para la próxima vez
        self.settings.setValue("splits/mother_dir", self.mother_dir_input.text())
        self.settings.setValue("splits/output_dir", self.output_dir_input.text())
        self.settings.setValue("splits/target_file", self.target_file_input.text())
        self.settings.setValue("splits/batch_size", self.batch_spin.value())
        self.settings.setValue("splits/epochs", self.epochs_spin.value())
        super().accept()

    def get_inputs(self):
        return {
            "mother_dir": self.mother_dir_input.text(),
            "target_file": self.target_file_input.text(),
            "epochs": self.epochs_spin.value(),
            "batch_size": self.batch_spin.value(),
            "output_dir": self.output_dir_input.text(),
        }