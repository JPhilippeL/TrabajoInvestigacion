"""
@file train_ednn_dialog.py
@author Mohamed EL BOUKHIARI
@brief Training dialog for the EDNN module.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit,
    QPushButton, QFileDialog, QDialogButtonBox, QWidget, QHBoxLayout,
    QSpinBox, QDoubleSpinBox, QLabel, QComboBox
)
from PySide6.QtCore import QSettings

from EDNN.utils.constants import (
    DEFAULT_GRAPHS_DIR,
    DEFAULT_MODELS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_EPOCHS,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
)


class TrainDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training Configuration - EDNN")
        self.resize(700, 520)

        self.settings = QSettings("Investigacion", "EDNN_Training")

        self.graphs_dir_input = QLineEdit()
        self.graphs_dir_input.setPlaceholderText("Graphs_EDNN directory")
        self.graphs_dir_input.setText(self.settings.value("train/graphs_dir", DEFAULT_GRAPHS_DIR))
        self.graphs_btn = QPushButton("Select...")
        self.graphs_btn.clicked.connect(self.browse_graphs)

        self.train_split_input = QLineEdit()
        self.train_split_input.setPlaceholderText("train_index_folder.txt file")
        self.train_split_input.setText(self.settings.value("train/train_split_file", ""))
        self.train_split_btn = QPushButton("Select...")
        self.train_split_btn.clicked.connect(self.browse_train_split)

        self.val_split_input = QLineEdit()
        self.val_split_input.setPlaceholderText("valid_index_folder.txt file")
        self.val_split_input.setText(self.settings.value("train/val_split_file", ""))
        self.val_split_btn = QPushButton("Select...")
        self.val_split_btn.clicked.connect(self.browse_val_split)

        self.test_split_input = QLineEdit()
        self.test_split_input.setPlaceholderText("test_index_folder.txt file")
        self.test_split_input.setText(self.settings.value("train/test_split_file", ""))
        self.test_split_btn = QPushButton("Select...")
        self.test_split_btn.clicked.connect(self.browse_test_split)

        self.output_base_input = QLineEdit()
        self.output_base_input.setPlaceholderText("Base output folder for models")
        self.output_base_input.setText(self.settings.value("train/output_base", DEFAULT_MODELS_DIR))
        self.output_btn = QPushButton("Select...")
        self.output_btn.clicked.connect(self.browse_output)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu", "cuda:0", "cuda:1"])
        self.device_combo.setCurrentText(self.settings.value("train/device", "cuda"))

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(int(self.settings.value("train/seed", DEFAULT_SEED)))

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setValue(int(self.settings.value("train/batch_size", DEFAULT_BATCH_SIZE)))

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 5000)
        self.epochs_spin.setValue(int(self.settings.value("train/epochs", DEFAULT_EPOCHS)))

        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 1000)
        self.patience_spin.setValue(int(self.settings.value("train/patience", DEFAULT_PATIENCE)))

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(6)
        self.lr_spin.setRange(0.000001, 1.0)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(float(self.settings.value("train/lr", DEFAULT_LR)))

        self.hidden_dim_spin = QSpinBox()
        self.hidden_dim_spin.setRange(1, 4096)
        self.hidden_dim_spin.setValue(int(self.settings.value("train/hidden_dim", DEFAULT_HIDDEN_DIM)))

        form_layout = QFormLayout()
        form_layout.addRow(QLabel("<b>1. System paths</b>"))
        form_layout.addRow("Graphs Path:", self._with_button(self.graphs_dir_input, self.graphs_btn))
        form_layout.addRow("Train Split (.txt):", self._with_button(self.train_split_input, self.train_split_btn))
        form_layout.addRow("Validation Split (.txt):", self._with_button(self.val_split_input, self.val_split_btn))
        form_layout.addRow("Test Split (.txt):", self._with_button(self.test_split_input, self.test_split_btn))
        form_layout.addRow("Output Base:", self._with_button(self.output_base_input, self.output_btn))

        form_layout.addRow(QLabel("<br><b>2. General configuration</b>"))
        form_layout.addRow("Device:", self.device_combo)
        form_layout.addRow("Random Seed:", self.seed_spin)

        form_layout.addRow(QLabel("<br><b>3. Training hyperparameters</b>"))
        form_layout.addRow("Learning Rate:", self.lr_spin)
        form_layout.addRow("Batch Size:", self.batch_spin)
        form_layout.addRow("Total Epochs:", self.epochs_spin)
        form_layout.addRow("Patience:", self.patience_spin)
        form_layout.addRow("Hidden Dim:", self.hidden_dim_spin)

        layout = QVBoxLayout()
        layout.addLayout(form_layout)

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

    def browse_graphs(self):
        path = QFileDialog.getExistingDirectory(self, "Select Graphs_EDNN folder")
        if path:
            self.graphs_dir_input.setText(path)

    def browse_train_split(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select train split", "", "Text Files (*.txt);;All Files (*)")
        if path:
            self.train_split_input.setText(path)

    def browse_val_split(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select validation split", "", "Text Files (*.txt);;All Files (*)")
        if path:
            self.val_split_input.setText(path)

    def browse_test_split(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select test split", "", "Text Files (*.txt);;All Files (*)")
        if path:
            self.test_split_input.setText(path)

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self.output_base_input.setText(path)

    def accept(self):
        self.settings.setValue("train/graphs_dir", self.graphs_dir_input.text())
        self.settings.setValue("train/train_split_file", self.train_split_input.text())
        self.settings.setValue("train/val_split_file", self.val_split_input.text())
        self.settings.setValue("train/test_split_file", self.test_split_input.text())
        self.settings.setValue("train/output_base", self.output_base_input.text())
        self.settings.setValue("train/device", self.device_combo.currentText())
        self.settings.setValue("train/seed", self.seed_spin.value())
        self.settings.setValue("train/batch_size", self.batch_spin.value())
        self.settings.setValue("train/epochs", self.epochs_spin.value())
        self.settings.setValue("train/patience", self.patience_spin.value())
        self.settings.setValue("train/lr", self.lr_spin.value())
        self.settings.setValue("train/hidden_dim", self.hidden_dim_spin.value())
        super().accept()

    def get_inputs(self):
        return {
            "graphs_dir": self.graphs_dir_input.text(),
            "train_split_file": self.train_split_input.text(),
            "val_split_file": self.val_split_input.text(),
            "test_split_file": self.test_split_input.text(),
            "output_base": self.output_base_input.text(),
            "batch_size": self.batch_spin.value(),
            "epochs": self.epochs_spin.value(),
            "patience": self.patience_spin.value(),
            "lr": self.lr_spin.value(),
            "hidden_dim": self.hidden_dim_spin.value(),
            "device": self.device_combo.currentText(),
            "seed": self.seed_spin.value(),
        }
