"""
@file batch_train_egnn_dialog.py
@author Mohamed EL BOUKHIARI
@brief Batch training dialog for the EGNN module.
@details
In the EGNN module, batch training is interpreted as hyperparameter search.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QDialogButtonBox,
    QWidget,
    QHBoxLayout,
    QSpinBox,
    QLabel,
    QComboBox,
)

from EGNN.utils.constants import (
    DEFAULT_GRAPHS_DIR,
    DEFAULT_MODELS_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TEMP_RUNS_DIR,
    DEFAULT_EPOCHS,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_DEVICE,
    DEFAULT_LR_VALUES,
    DEFAULT_HIDDEN_DIM_VALUES,
    DEFAULT_BATCH_SIZE_VALUES,
    DEFAULT_TRAIN_SPLIT_FILE,
    DEFAULT_VAL_SPLIT_FILE,
    DEFAULT_TEST_SPLIT_FILE,
)


class BatchTrainDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EGNN Hyperparameter Search Configuration")
        self.resize(780, 640)

        self.settings = QSettings("ResearchApp", "EGNN_HyperparameterSearch")

        self.graphs_dir_input = QLineEdit()
        self.graphs_dir_input.setPlaceholderText("Graphs_EGNN directory")
        self.graphs_dir_input.setText(self.settings.value("search/graphs_dir", DEFAULT_GRAPHS_DIR))
        self.graphs_btn = QPushButton("Browse...")
        self.graphs_btn.clicked.connect(self.browse_graphs)

        self.train_split_input = QLineEdit()
        self.train_split_input.setPlaceholderText("train_index_folder.txt")
        self.train_split_input.setText(self.settings.value("search/train_split_file", DEFAULT_TRAIN_SPLIT_FILE))
        self.train_split_btn = QPushButton("Browse...")
        self.train_split_btn.clicked.connect(self.browse_train_split)

        self.val_split_input = QLineEdit()
        self.val_split_input.setPlaceholderText("valid_index_folder.txt")
        self.val_split_input.setText(self.settings.value("search/val_split_file", DEFAULT_VAL_SPLIT_FILE))
        self.val_split_btn = QPushButton("Browse...")
        self.val_split_btn.clicked.connect(self.browse_val_split)

        self.test_split_input = QLineEdit()
        self.test_split_input.setPlaceholderText("test_index_folder.txt")
        self.test_split_input.setText(self.settings.value("search/test_split_file", DEFAULT_TEST_SPLIT_FILE))
        self.test_split_btn = QPushButton("Browse...")
        self.test_split_btn.clicked.connect(self.browse_test_split)

        self.models_root_input = QLineEdit()
        self.models_root_input.setPlaceholderText("Root directory for trial model folders")
        self.models_root_input.setText(self.settings.value("search/models_root", DEFAULT_MODELS_DIR))
        self.models_root_btn = QPushButton("Browse...")
        self.models_root_btn.clicked.connect(self.browse_models_root)

        self.results_root_input = QLineEdit()
        self.results_root_input.setPlaceholderText("Root directory for trial results")
        self.results_root_input.setText(self.settings.value("search/results_root", DEFAULT_RESULTS_DIR))
        self.results_root_btn = QPushButton("Browse...")
        self.results_root_btn.clicked.connect(self.browse_results_root)

        self.temp_runs_dir_input = QLineEdit()
        self.temp_runs_dir_input.setPlaceholderText("Temporary runs directory")
        self.temp_runs_dir_input.setText(self.settings.value("search/temp_runs_dir", DEFAULT_TEMP_RUNS_DIR))
        self.temp_runs_dir_btn = QPushButton("Browse...")
        self.temp_runs_dir_btn.clicked.connect(self.browse_temp_runs)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu", "cuda:0", "cuda:1"])
        self.device_combo.setCurrentText(self.settings.value("search/device", DEFAULT_DEVICE))

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(int(self.settings.value("search/seed", DEFAULT_SEED)))

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 5000)
        self.epochs_spin.setValue(int(self.settings.value("search/epochs", DEFAULT_EPOCHS)))

        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 1000)
        self.patience_spin.setValue(int(self.settings.value("search/patience", DEFAULT_PATIENCE)))

        self.lr_values_input = QLineEdit()
        self.lr_values_input.setPlaceholderText("Example: 5e-5,1e-4,5e-4,1e-3")
        self.lr_values_input.setText(self.settings.value("search/lr_values", DEFAULT_LR_VALUES))

        self.hidden_dim_values_input = QLineEdit()
        self.hidden_dim_values_input.setPlaceholderText("Example: 32,64,128")
        self.hidden_dim_values_input.setText(self.settings.value("search/hidden_dim_values", DEFAULT_HIDDEN_DIM_VALUES))

        self.batch_size_values_input = QLineEdit()
        self.batch_size_values_input.setPlaceholderText("Example: 2,4,8")
        self.batch_size_values_input.setText(self.settings.value("search/batch_size_values", DEFAULT_BATCH_SIZE_VALUES))

        form_layout = QFormLayout()
        form_layout.addRow(QLabel("<b>1. Paths</b>"))
        form_layout.addRow("Graphs path:", self._with_button(self.graphs_dir_input, self.graphs_btn))
        form_layout.addRow("Train split:", self._with_button(self.train_split_input, self.train_split_btn))
        form_layout.addRow("Validation split:", self._with_button(self.val_split_input, self.val_split_btn))
        form_layout.addRow("Test split:", self._with_button(self.test_split_input, self.test_split_btn))
        form_layout.addRow("Models root:", self._with_button(self.models_root_input, self.models_root_btn))
        form_layout.addRow("Results root:", self._with_button(self.results_root_input, self.results_root_btn))
        form_layout.addRow("Temporary runs directory:", self._with_button(self.temp_runs_dir_input, self.temp_runs_dir_btn))

        form_layout.addRow(QLabel("<br><b>2. General configuration</b>"))
        form_layout.addRow("Device:", self.device_combo)
        form_layout.addRow("Random seed:", self.seed_spin)
        form_layout.addRow("Epochs:", self.epochs_spin)
        form_layout.addRow("Patience:", self.patience_spin)

        form_layout.addRow(QLabel("<br><b>3. Search space</b>"))
        form_layout.addRow("Learning rate values:", self.lr_values_input)
        form_layout.addRow("Hidden dimension values:", self.hidden_dim_values_input)
        form_layout.addRow("Batch size values:", self.batch_size_values_input)

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
        path = QFileDialog.getExistingDirectory(self, "Select Graphs_EGNN directory")
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
        path = QFileDialog.getExistingDirectory(self, "Select models root directory")
        if path:
            self.models_root_input.setText(path)

    def browse_results_root(self):
        path = QFileDialog.getExistingDirectory(self, "Select results root directory")
        if path:
            self.results_root_input.setText(path)

    def browse_temp_runs(self):
        path = QFileDialog.getExistingDirectory(self, "Select temporary runs directory")
        if path:
            self.temp_runs_dir_input.setText(path)

    def accept(self):
        self.settings.setValue("search/graphs_dir", self.graphs_dir_input.text())
        self.settings.setValue("search/train_split_file", self.train_split_input.text())
        self.settings.setValue("search/val_split_file", self.val_split_input.text())
        self.settings.setValue("search/test_split_file", self.test_split_input.text())
        self.settings.setValue("search/models_root", self.models_root_input.text())
        self.settings.setValue("search/results_root", self.results_root_input.text())
        self.settings.setValue("search/temp_runs_dir", self.temp_runs_dir_input.text())
        self.settings.setValue("search/device", self.device_combo.currentText())
        self.settings.setValue("search/seed", self.seed_spin.value())
        self.settings.setValue("search/epochs", self.epochs_spin.value())
        self.settings.setValue("search/patience", self.patience_spin.value())
        self.settings.setValue("search/lr_values", self.lr_values_input.text())
        self.settings.setValue("search/hidden_dim_values", self.hidden_dim_values_input.text())
        self.settings.setValue("search/batch_size_values", self.batch_size_values_input.text())
        super().accept()

    @staticmethod
    def _parse_float_list(raw_text: str):
        return [float(x.strip()) for x in raw_text.split(",") if x.strip()]

    @staticmethod
    def _parse_int_list(raw_text: str):
        return [int(x.strip()) for x in raw_text.split(",") if x.strip()]

    def get_inputs(self):
        return {
            "graphs_dir": self.graphs_dir_input.text(),
            "train_split_file": self.train_split_input.text(),
            "val_split_file": self.val_split_input.text(),
            "test_split_file": self.test_split_input.text(),
            "models_root": self.models_root_input.text(),
            "results_root": self.results_root_input.text(),
            "temp_runs_dir": self.temp_runs_dir_input.text(),
            "device": self.device_combo.currentText(),
            "seed": self.seed_spin.value(),
            "epochs": self.epochs_spin.value(),
            "patience": self.patience_spin.value(),
            "lr_values": self._parse_float_list(self.lr_values_input.text()),
            "hidden_dim_values": self._parse_int_list(self.hidden_dim_values_input.text()),
            "batch_size_values": self._parse_int_list(self.batch_size_values_input.text()),
        }
