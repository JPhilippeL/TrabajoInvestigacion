from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QSpinBox,
    QDoubleSpinBox, QDialogButtonBox, QPushButton, QFileDialog, QHBoxLayout
)
from PySide6.QtCore import QSettings

class TrainMultipleModelsDialog(QDialog):
    session_defaults = {
        "sdf_dir": "",
        "target_file": ""
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración de entrenamiento múltiple")
        self.settings = QSettings("Investigacion", "Analisis Molecular")

        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # ---------- SDF directory ----------
        self.sdf_path_input = QLineEdit()
        self.sdf_path_input.setText(self.settings.value("train_multi/sdf_dir", ""))
        self.sdf_path_button = QPushButton("Elegir carpeta...")
        self.sdf_path_button.clicked.connect(self.select_sdf_folder)
        sdf_layout = QHBoxLayout()
        sdf_layout.addWidget(self.sdf_path_input)
        sdf_layout.addWidget(self.sdf_path_button)
        form_layout.addRow("Directorio de SDFs:", sdf_layout)

        # ---------- Target file ----------
        self.target_file_input = QLineEdit()
        self.target_file_input.setText(self.settings.value("train_multi/target_file", ""))
        self.target_file_button = QPushButton("Elegir archivo...")
        self.target_file_button.clicked.connect(self.select_target_file)
        target_layout = QHBoxLayout()
        target_layout.addWidget(self.target_file_input)
        target_layout.addWidget(self.target_file_button)
        form_layout.addRow("Archivo de targets (.txt):", target_layout)

        # ---------- Otros parámetros ----------
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 10000)
        self.epochs_input.setValue(int(self.settings.value("train_multi/epochs", 100)))

        self.batch_input = QSpinBox()
        self.batch_input.setRange(1, 1024)
        self.batch_input.setValue(int(self.settings.value("train_multi/batch_size", 32)))

        self.lr_input = QDoubleSpinBox()
        self.lr_input.setDecimals(5)
        self.lr_input.setRange(0.00001, 1.0)
        self.lr_input.setSingleStep(0.0001)
        self.lr_input.setValue(float(self.settings.value("train_multi/lr", 0.001)))

        self.valid_split_input = QDoubleSpinBox()
        self.valid_split_input.setDecimals(2)
        self.valid_split_input.setRange(0.0, 0.5)
        self.valid_split_input.setSingleStep(0.05)
        self.valid_split_input.setValue(float(self.settings.value("train_multi/valid_split", 0.2)))

        self.hidden_dim_input = QSpinBox()
        self.hidden_dim_input.setRange(8, 1024)
        self.hidden_dim_input.setValue(int(self.settings.value("train_multi/hidden_dim", 64)))

        self.patience_input = QSpinBox()
        self.patience_input.setRange(0, 100)
        self.patience_input.setValue(int(self.settings.value("train_multi/patience", 0)))

        form_layout.addRow("Épocas:", self.epochs_input)
        form_layout.addRow("Batch size:", self.batch_input)
        form_layout.addRow("Learning rate:", self.lr_input)
        form_layout.addRow("Porcentaje validación:", self.valid_split_input)
        form_layout.addRow("Hidden dim:", self.hidden_dim_input)
        form_layout.addRow("Paciencia Early Stopping (0 desactiva):", self.patience_input)

        layout.addLayout(form_layout)

        # Botones
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.setLayout(layout)

    def select_sdf_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta con SDFs")
        if folder:
            self.sdf_path_input.setText(folder)

    def select_target_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Seleccionar archivo de targets", filter="TXT files (*.txt)")
        if file:
            self.target_file_input.setText(file)

    def accept(self):
        # Guardar configuraciones
        self.settings.setValue("train_multi/sdf_dir", self.sdf_path_input.text())
        self.settings.setValue("train_multi/target_file", self.target_file_input.text())
        self.settings.setValue("train_multi/epochs", self.epochs_input.value())
        self.settings.setValue("train_multi/batch_size", self.batch_input.value())
        self.settings.setValue("train_multi/lr", self.lr_input.value())
        self.settings.setValue("train_multi/valid_split", self.valid_split_input.value())
        self.settings.setValue("train_multi/hidden_dim", self.hidden_dim_input.value())
        self.settings.setValue("train_multi/patience", self.patience_input.value())
        super().accept()

    def get_values(self):
        return {
            "sdf_dir": self.sdf_path_input.text(),
            "target_file": self.target_file_input.text(),
            "epochs": self.epochs_input.value(),
            "batch_size": self.batch_input.value(),
            "lr": self.lr_input.value(),
            "valid_split": self.valid_split_input.value(),
            "hidden_dim": self.hidden_dim_input.value(),
            "patience": self.patience_input.value()
        }
