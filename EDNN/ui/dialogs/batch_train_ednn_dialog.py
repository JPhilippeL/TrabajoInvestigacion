"""
@file batch_train_ednn_dialog.py
@author Mohamed EL BOUKHIARI
@brief Batch training dialog for the EDNN module.
@details
In the EDNN module, batch training is interpreted as hyperparameter search.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit,
    QPushButton, QFileDialog, QDialogButtonBox, QWidget, QHBoxLayout,
    QSpinBox, QLabel, QComboBox
)
from PySide6.QtCore import QSettings

from EDNN.utils.constants import (
    DEFAULT_GRAPHS_DIR,
    DEFAULT_MODELS_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TEMP_RUNS_DIR,
    DEFAULT_EPOCHS,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_LR_VALUES,
    DEFAULT_HIDDEN_DIM_VALUES,
    DEFAULT_BATCH_SIZE_VALUES,
)


class BatchTrainDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EDNN Hyperparameter Search Configuration")
        self.resize(760, 580)

        self.settings = QSettings("Investigacion", "EDNN_BatchTraining")

        self.graphs_dir_input = QLineEdit()
        self.graphs_dir_input.setPlaceholderText("Graphs_EDNN directory")
        self.graphs_dir_input.setText(self.settings.value("batch/graphs_dir", DEFAULT_GRAPHS_DIR))
        self.graphs_btn = QPushButton("Select...")
        self.graphs_btn.clicked.connect(self.browse_graphs)

        self.train_split_input = QLineEdit()
        self.train_split_input.setPlaceholderText("train_index_folder.txt file")
        self.train_split_input.setText(self.settings.value("batch/train_split_file", ""))
        self.train_split_btn = QPushButton("Select...")
        self.train_split_btn.clicked.connect(self.browse_train_split)

        self.val_split_input = QLineEdit()
        self.val_split_input.setPlaceholderText("valid_index_folder.txt file")
        self.val_split_input.setText(self.settings.value("batch/val_split_file", ""))
        self.val_split_btn = QPushButton("Select...")
        self.val_split_btn.clicked.connect(self.browse_val_split)

        self.test_split_input = QLineEdit()
        self.test_split_input.setPlaceholderText("test_index_folder.txt file")
        self.test_split_input.setText(self.settings.value("batch/test_split_file", ""))
        self.test_split_btn = QPushButton("Select...")
        self.test_split_btn.clicked.connect(self.browse_test_split)

        self.models_root_input = QLineEdit()
        self.models_root_input.setPlaceholderText("Root folder for trial models")
        self.models_root_input.setText(self.settings.value("batch/models_root", DEFAULT_MODELS_DIR))
        self.models_root_btn = QPushButton("Select...")
        self.models_root_btn.clicked.connect(self.browse_models_root)

        self.results_root_input = QLineEdit()
        self.results_root_input.setPlaceholderText("Root folder for trial results")
        self.results_root_input.setText(self.settings.value("batch/results_root", DEFAULT_RESULTS_DIR))
        self.results_root_btn = QPushButton("Select...")
        self.results_root_btn.clicked.connect(self.browse_results_root)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu", "cuda:0", "cuda:1"])
        self.device_combo.setCurrentText(self.settings.value("batch/device", "auto"))

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(int(self.settings.value("batch/seed", DEFAULT_SEED)))

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 5000)
        self.epochs_spin.setValue(int(self.settings.value("batch/epochs", DEFAULT_EPOCHS)))

        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 1000)
        self.patience_spin.setValue(int(self.settings.value("batch/patience", DEFAULT_PATIENCE)))

        self.lr_values_input = QLineEdit()
        self.lr_values_input.setPlaceholderText("Example: 5e-5,1e-4,5e-4,1e-3")
        self.lr_values_input.setText(self.settings.value("batch/lr_values", DEFAULT_LR_VALUES))

        self.hidden_dim_values_input = QLineEdit()
        self.hidden_dim_values_input.setPlaceholderText("Example: 32,64,128")
        self.hidden_dim_values_input.setText(self.settings.value("batch/hidden_dim_values", DEFAULT_HIDDEN_DIM_VALUES))

        self.batch_size_values_input = QLineEdit()
        self.batch_size_values_input.setPlaceholderText("Example: 2,4,8")
        self.batch_size_values_input.setText(self.settings.value("batch/batch_size_values", DEFAULT_BATCH_SIZE_VALUES))

        form_layout = QFormLayout()
        form_layout.addRow(QLabel("<b>1. System paths</b>"))
        form_layout.addRow("Graphs Path:", self._with_button(self.graphs_dir_input, self.graphs_btn))
        form_layout.addRow("Train Split (.txt):", self._with_button(self.train_split_input, self.train_split_btn))
        form_layout.addRow("Validation Split (.txt):", self._with_button(self.val_split_input, self.val_split_btn))
        form_layout.addRow("Test Split (.txt):", self._with_button(self.test_split_input, self.test_split_btn))
        form_layout.addRow("Models Root:", self._with_button(self.models_root_input, self.models_root_btn))
        form_layout.addRow("Results Root:", self._with_button(self.results_root_input, self.results_root_btn))

        form_layout.addRow(QLabel("<br><b>2. General configuration</b>"))
        form_layout.addRow("Device:", self.device_combo)
        form_layout.addRow("Random Seed:", self.seed_spin)
        form_layout.addRow("Total Epochs:", self.epochs_spin)
        form_layout.addRow("Patience:", self.patience_spin)

        form_layout.addRow(QLabel("<br><b>3. Search space</b>"))
        form_layout.addRow("Learning Rate Values:", self.lr_values_input)
        form_layout.addRow("Hidden Dim Values:", self.hidden_dim_values_input)
        form_layout.addRow("Batch Size Values:", self.batch_size_values_input)

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

    def browse_models_root(self):
        path = QFileDialog.getExistingDirectory(self, "Select model root folder")
        if path:
            self.models_root_input.setText(path)

    def browse_results_root(self):
        path = QFileDialog.getExistingDirectory(self, "Select result root folder")
        if path:
            self.results_root_input.setText(path)

    def accept(self):
        self.settings.setValue("batch/graphs_dir", self.graphs_dir_input.text())
        self.settings.setValue("batch/train_split_file", self.train_split_input.text())
        self.settings.setValue("batch/val_split_file", self.val_split_input.text())
        self.settings.setValue("batch/test_split_file", self.test_split_input.text())
        self.settings.setValue("batch/models_root", self.models_root_input.text())
        self.settings.setValue("batch/results_root", self.results_root_input.text())
        self.settings.setValue("batch/device", self.device_combo.currentText())
        self.settings.setValue("batch/seed", self.seed_spin.value())
        self.settings.setValue("batch/epochs", self.epochs_spin.value())
        self.settings.setValue("batch/patience", self.patience_spin.value())
        self.settings.setValue("batch/lr_values", self.lr_values_input.text())
        self.settings.setValue("batch/hidden_dim_values", self.hidden_dim_values_input.text())
        self.settings.setValue("batch/batch_size_values", self.batch_size_values_input.text())
        super().accept()

    @staticmethod
    def _parse_float_list(raw_text: str):
        return [float(x.strip()) for x in raw_text.split(",") if x.strip()]

    @staticmethod
    def _parse_int_list(raw_text: str):
        return [int(x.strip()) for x in raw_text.split(",") if x.strip()]

    def get_inputs(self):
        device = self.device_combo.currentText()
        if device == "auto":
            device = None

        return {
            "graphs_dir": self.graphs_dir_input.text(),
            "train_split_file": self.train_split_input.text(),
            "val_split_file": self.val_split_input.text(),
            "test_split_file": self.test_split_input.text(),
            "models_root": self.models_root_input.text(),
            "results_root": self.results_root_input.text(),
            "temp_runs_dir": DEFAULT_TEMP_RUNS_DIR,
            "device": device,
            "seed": self.seed_spin.value(),
            "epochs": self.epochs_spin.value(),
            "patience": self.patience_spin.value(),
            "lr_values": self._parse_float_list(self.lr_values_input.text()),
            "hidden_dim_values": self._parse_int_list(self.hidden_dim_values_input.text()),
            "batch_size_values": self._parse_int_list(self.batch_size_values_input.text()),
        }
