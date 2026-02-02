from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QSpinBox, QDoubleSpinBox,
    QDialogButtonBox, QPushButton, QFileDialog, QHBoxLayout, QComboBox
)
from PySide6.QtCore import QSettings

class TransferLearningMultipleDialog(QDialog):
    session_defaults = {
        "pretrained_model_directory_path": "",
        "sdf_dir": "",
        "target_file": ""
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración de Transfer Learning")
        self.settings = QSettings("Investigacion", "Analisis Molecular")

        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # ---------- Pretrained model directory ----------
        self.pretrained_dir_input = QLineEdit()
        self.pretrained_dir_input.setText(self.settings.value("transfer/pretrained_model_directory_path", ""))
        self.pretrained_dir_button = QPushButton("Elegir carpeta...")
        self.pretrained_dir_button.clicked.connect(self.select_pretrained_folder)
        pretrained_layout = QHBoxLayout()
        pretrained_layout.addWidget(self.pretrained_dir_input)
        pretrained_layout.addWidget(self.pretrained_dir_button)
        form_layout.addRow("Carpeta de modelos pre-entrenados:", pretrained_layout)

        # ---------- SDF directory ----------
        self.sdf_path_input = QLineEdit()
        self.sdf_path_input.setText(self.settings.value("transfer/sdf_dir", ""))
        self.sdf_path_button = QPushButton("Elegir carpeta...")
        self.sdf_path_button.clicked.connect(self.select_sdf_folder)
        sdf_layout = QHBoxLayout()
        sdf_layout.addWidget(self.sdf_path_input)
        sdf_layout.addWidget(self.sdf_path_button)
        form_layout.addRow("Directorio de SDFs:", sdf_layout)

        # ---------- Target file ----------
        self.target_file_input = QLineEdit()
        self.target_file_input.setText(self.settings.value("transfer/target_file", ""))
        self.target_file_button = QPushButton("Elegir archivo...")
        self.target_file_button.clicked.connect(self.select_target_file)
        target_layout = QHBoxLayout()
        target_layout.addWidget(self.target_file_input)
        target_layout.addWidget(self.target_file_button)
        form_layout.addRow("Archivo de targets (.txt):", target_layout)

        # ---------- Training parameters ----------
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 10000)
        self.epochs_input.setValue(int(self.settings.value("transfer/epochs", 100)))

        self.batch_input = QSpinBox()
        self.batch_input.setRange(1, 1024)
        self.batch_input.setValue(int(self.settings.value("transfer/batch_size", 32)))

        self.lr_input = QDoubleSpinBox()
        self.lr_input.setDecimals(5)
        self.lr_input.setRange(0.00001, 1.0)
        self.lr_input.setSingleStep(0.0001)
        self.lr_input.setValue(float(self.settings.value("transfer/lr", 0.001)))

        self.valid_split_input = QDoubleSpinBox()
        self.valid_split_input.setDecimals(2)
        self.valid_split_input.setRange(0.0, 0.5)
        self.valid_split_input.setSingleStep(0.05)
        self.valid_split_input.setValue(float(self.settings.value("transfer/valid_split", 0.2)))

        self.early_stopping_patience_input = QSpinBox()
        self.early_stopping_patience_input.setRange(0, 100)
        self.early_stopping_patience_input.setValue(int(self.settings.value("transfer/early_stopping_patience", 0)))

        form_layout.addRow("Épocas:", self.epochs_input)
        form_layout.addRow("Batch size:", self.batch_input)
        form_layout.addRow("Learning rate:", self.lr_input)
        form_layout.addRow("Porcentaje validación:", self.valid_split_input)
        form_layout.addRow("Paciencia Early Stopping (0 desactiva):", self.early_stopping_patience_input)

        layout.addLayout(form_layout)

        # Botones OK / Cancel
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.setLayout(layout)

    # ---------- File/folder selectors ----------
    def select_pretrained_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de modelos pre-entrenados")
        if folder:
            self.pretrained_dir_input.setText(folder)

    def select_sdf_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta con SDFs")
        if folder:
            self.sdf_path_input.setText(folder)

    def select_target_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Seleccionar archivo de targets", filter="TXT files (*.txt)")
        if file:
            self.target_file_input.setText(file)

    # ---------- Guardar configuraciones ----------
    def accept(self):
        self.settings.setValue("transfer/pretrained_model_directory_path", self.pretrained_dir_input.text())
        self.settings.setValue("transfer/sdf_dir", self.sdf_path_input.text())
        self.settings.setValue("transfer/target_file", self.target_file_input.text())
        self.settings.setValue("transfer/epochs", self.epochs_input.value())
        self.settings.setValue("transfer/batch_size", self.batch_input.value())
        self.settings.setValue("transfer/lr", self.lr_input.value())
        self.settings.setValue("transfer/valid_split", self.valid_split_input.value())
        self.settings.setValue("transfer/early_stopping_patience", self.early_stopping_patience_input.value())
        super().accept()

    # ---------- Obtener valores ----------
    def get_values(self):
        return {
            "pretrained_model_directory_path": self.pretrained_dir_input.text(),
            "sdf_dir": self.sdf_path_input.text(),
            "target_file": self.target_file_input.text(),
            "epochs": self.epochs_input.value(),
            "batch_size": self.batch_input.value(),
            "lr": self.lr_input.value(),
            "valid_split": self.valid_split_input.value(),
            "patience": self.early_stopping_patience_input.value()
        }
