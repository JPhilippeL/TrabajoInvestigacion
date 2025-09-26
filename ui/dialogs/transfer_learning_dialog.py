from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QComboBox, QSpinBox,
    QDoubleSpinBox, QDialogButtonBox, QPushButton, QFileDialog, QWidget, QHBoxLayout
)

class TransferLearningDialog(QDialog):
    session_defaults = {
        "sdf_dir": "",
        "target_file": "",
        "pretrained_model_path": ""
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración de Transfer Learning")

        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # ---------- SDF directory ----------
        self.sdf_path_input = QLineEdit()
        self.sdf_path_input.setText(self.session_defaults["sdf_dir"])
        self.sdf_path_button = QPushButton("Elegir carpeta...")
        self.sdf_path_button.clicked.connect(self.select_sdf_folder)
        sdf_layout = QHBoxLayout()
        sdf_layout.addWidget(self.sdf_path_input)
        sdf_layout.addWidget(self.sdf_path_button)
        form_layout.addRow("Directorio de SDFs:", sdf_layout)

        # ---------- Target file ----------
        self.target_file_input = QLineEdit()
        self.target_file_input.setText(self.session_defaults["target_file"])
        self.target_file_button = QPushButton("Elegir archivo...")
        self.target_file_button.clicked.connect(self.select_target_file)
        target_layout = QHBoxLayout()
        target_layout.addWidget(self.target_file_input)
        target_layout.addWidget(self.target_file_button)
        form_layout.addRow("Archivo de targets (.txt):", target_layout)

        # ---------- Pretrained model ----------
        self.pretrained_model_input = QLineEdit()
        self.pretrained_model_input.setText(self.session_defaults["pretrained_model_path"])
        self.pretrained_model_button = QPushButton("Elegir modelo preentrenado...")
        self.pretrained_model_button.clicked.connect(self.select_pretrained_model)
        pretrained_layout = QHBoxLayout()
        pretrained_layout.addWidget(self.pretrained_model_input)
        pretrained_layout.addWidget(self.pretrained_model_button)
        form_layout.addRow("Modelo preentrenado (.pt):", pretrained_layout)

        # ---------- Transfer mode ----------
        self.transfer_mode_select = QComboBox()
        self.transfer_mode_select.addItems(["Fine Tuning", "Feature Extraction"])
        form_layout.addRow("Modo Transfer Learning:", self.transfer_mode_select)

        # ---------- Otros parámetros ----------
        self.epochs_input = QSpinBox(); self.epochs_input.setRange(1, 10000); self.epochs_input.setValue(20)
        self.valid_split_input = QDoubleSpinBox(); self.valid_split_input.setDecimals(2)
        self.valid_split_input.setRange(0.0, 0.5); self.valid_split_input.setSingleStep(0.05); self.valid_split_input.setValue(0.2)
        self.early_stopping_patience_input = QSpinBox(); self.early_stopping_patience_input.setRange(0, 100); self.early_stopping_patience_input.setValue(0)
        self.batch_input = QSpinBox(); self.batch_input.setRange(1, 1024); self.batch_input.setValue(32)
        self.lr_input = QDoubleSpinBox(); self.lr_input.setDecimals(5); self.lr_input.setRange(0.00001, 0.1)
        self.lr_input.setSingleStep(0.0001); self.lr_input.setValue(0.0001)
        self.hidden_dim_input = QSpinBox(); self.hidden_dim_input.setRange(8, 1024); self.hidden_dim_input.setValue(64)
        self.num_layers_input = QSpinBox(); self.num_layers_input.setRange(1, 10); self.num_layers_input.setValue(3)
        self.save_name_input = QLineEdit()

        form_layout.addRow("Épocas:", self.epochs_input)
        form_layout.addRow("Porcentaje validación:", self.valid_split_input)
        form_layout.addRow("Paciencia Early Stopping (0 desactiva):", self.early_stopping_patience_input)
        form_layout.addRow("Batch size:", self.batch_input)
        form_layout.addRow("Learning rate:", self.lr_input)
        form_layout.addRow("Hidden dim:", self.hidden_dim_input)
        form_layout.addRow("Número de capas:", self.num_layers_input)
        form_layout.addRow("Nombre del modelo final:", self.save_name_input)

        layout.addLayout(form_layout)

        # ---------- Botones ----------
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.setLayout(layout)

    # ---------- Selectors ----------
    def select_sdf_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta con SDFs")
        if folder:
            self.sdf_path_input.setText(folder)

    def select_target_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Seleccionar archivo de targets", filter="TXT files (*.txt)")
        if file:
            self.target_file_input.setText(file)

    def select_pretrained_model(self):
        file, _ = QFileDialog.getOpenFileName(self, "Seleccionar modelo preentrenado", filter="PT files (*.pt)")
        if file:
            self.pretrained_model_input.setText(file)

    # ---------- Accept ----------
    def accept(self):
        TransferLearningDialog.session_defaults["sdf_dir"] = self.sdf_path_input.text()
        TransferLearningDialog.session_defaults["target_file"] = self.target_file_input.text()
        TransferLearningDialog.session_defaults["pretrained_model_path"] = self.pretrained_model_input.text()
        super().accept()

    # ---------- Get values ----------
    def get_values(self):
        return {
            "sdf_dir": self.sdf_path_input.text(),
            "target_file": self.target_file_input.text(),
            "pretrained_model_path": self.pretrained_model_input.text(),
            "transfer_mode": self.transfer_mode_select.currentText(),
            "epochs": self.epochs_input.value(),
            "batch_size": self.batch_input.value(),
            "lr": self.lr_input.value(),
            "valid_split": self.valid_split_input.value(),
            "save_name": self.save_name_input.text(),
            "hidden_dim": self.hidden_dim_input.value(),
            "num_layers": self.num_layers_input.value(),
            "early_stopping_patience": self.early_stopping_patience_input.value()
        }
