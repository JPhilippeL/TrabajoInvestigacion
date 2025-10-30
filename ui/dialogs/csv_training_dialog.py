from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QComboBox, QSpinBox,
    QDoubleSpinBox, QDialogButtonBox, QPushButton, QFileDialog, QHBoxLayout
)
from PySide6.QtCore import QSettings

class TrainCSVConfigDialog(QDialog):
    session_defaults = {
        "csv_file": ""
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración de entrenamiento (CSV)")
        self.settings = QSettings("Investigacion", "Analisis Molecular")

        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # ---------- CSV file ----------
        self.csv_file_input = QLineEdit()
        self.csv_file_input.setText(self.settings.value("train/csv_file", ""))
        self.csv_file_button = QPushButton("Elegir archivo CSV...")
        self.csv_file_button.clicked.connect(self.select_csv_file)
        csv_layout = QHBoxLayout()
        csv_layout.addWidget(self.csv_file_input)
        csv_layout.addWidget(self.csv_file_button)
        form_layout.addRow("Archivo CSV:", csv_layout)

        # ---------- Modelo y otros ----------
        self.model_select = QComboBox()
        self.model_select.addItems(["GIN", "GINE", "GAT", "EGAT", "GraphTransformer"])
        self.model_select.setCurrentText(self.settings.value("train/modelo", "GIN"))

        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 10000)
        self.epochs_input.setValue(int(self.settings.value("train/epochs", 100)))

        self.valid_split_input = QDoubleSpinBox()
        self.valid_split_input.setDecimals(2)
        self.valid_split_input.setRange(0.0, 0.5)
        self.valid_split_input.setSingleStep(0.05)
        self.valid_split_input.setValue(float(self.settings.value("train/valid_split", 0.2)))

        self.early_stopping_patience_input = QSpinBox()
        self.early_stopping_patience_input.setRange(0, 100)
        self.early_stopping_patience_input.setValue(int(self.settings.value("train/early_stopping_patience", 0)))

        self.batch_input = QSpinBox()
        self.batch_input.setRange(1, 1024)
        self.batch_input.setValue(int(self.settings.value("train/batch_size", 32)))

        self.lr_input = QDoubleSpinBox()
        self.lr_input.setDecimals(5)
        self.lr_input.setRange(0.00001, 1.0)
        self.lr_input.setSingleStep(0.0001)
        self.lr_input.setValue(float(self.settings.value("train/lr", 0.001)))

        self.hidden_dim_input = QSpinBox()
        self.hidden_dim_input.setRange(8, 1024)
        self.hidden_dim_input.setValue(int(self.settings.value("train/hidden_dim", 64)))

        self.num_layers_input = QSpinBox()
        self.num_layers_input.setRange(1, 10)
        self.num_layers_input.setValue(int(self.settings.value("train/num_layers", 3)))

        self.save_name_input = QLineEdit()
        self.save_name_input.setText(self.settings.value("train/save_name", ""))

        form_layout.addRow("Modelo:", self.model_select)
        form_layout.addRow("Épocas:", self.epochs_input)
        form_layout.addRow("Porcentaje validación:", self.valid_split_input)
        form_layout.addRow("Paciencia Early Stopping (0 desactiva):", self.early_stopping_patience_input)
        form_layout.addRow("Batch size:", self.batch_input)
        form_layout.addRow("Learning rate:", self.lr_input)
        form_layout.addRow("Hidden dim:", self.hidden_dim_input)
        form_layout.addRow("Número de capas:", self.num_layers_input)
        form_layout.addRow("Nombre del modelo:", self.save_name_input)

        layout.addLayout(form_layout)

        # Botones
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.setLayout(layout)

    def select_csv_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Seleccionar archivo CSV", filter="CSV files (*.csv)")
        if file:
            self.csv_file_input.setText(file)

    def accept(self):
        self.settings.setValue("train/csv_file", self.csv_file_input.text())
        self.settings.setValue("train/modelo", self.model_select.currentText())
        self.settings.setValue("train/epochs", self.epochs_input.value())
        self.settings.setValue("train/batch_size", self.batch_input.value())
        self.settings.setValue("train/lr", self.lr_input.value())
        self.settings.setValue("train/valid_split", self.valid_split_input.value())
        self.settings.setValue("train/save_name", self.save_name_input.text())
        self.settings.setValue("train/hidden_dim", self.hidden_dim_input.value())
        self.settings.setValue("train/num_layers", self.num_layers_input.value())
        self.settings.setValue("train/early_stopping_patience", self.early_stopping_patience_input.value())
        super().accept()

    def get_values(self):
        return {
            "csv_file": self.csv_file_input.text(),
            "modelo": self.model_select.currentText(),
            "epochs": self.epochs_input.value(),
            "batch_size": self.batch_input.value(),
            "lr": self.lr_input.value(),
            "valid_split": self.valid_split_input.value(),
            "save_name": self.save_name_input.text(),
            "hidden_dim": self.hidden_dim_input.value(),
            "num_layers": self.num_layers_input.value(),
            "early_stopping_patience": self.early_stopping_patience_input.value()
        }
