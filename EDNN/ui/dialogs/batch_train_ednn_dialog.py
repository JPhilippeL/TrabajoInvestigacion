"""
@file batch_train_ednn_dialog.py
@author Mohamed EL BOUKHIARI
@brief Batch training dialog for the EDNN module.
@details
In the EDNN module, batch training is interpreted as hyperparameter search.
This dialog reads global defaults from AppSettings for paths, device and seed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
from ui.utils.app_settings import AppSettings


DEFAULT_TRAIN_SPLIT_FILE = "train_index_folder.txt"
DEFAULT_VAL_SPLIT_FILE = "valid_index_folder.txt"
DEFAULT_TEST_SPLIT_FILE = "test_index_folder.txt"
DEFAULT_DEVICE = "auto"


class BatchTrainDialog(QDialog):
    """
    @brief Dialog used to configure EDNN hyperparameter search.
    """

    def __init__(self, parent=None) -> None:
        """
        @brief Initialize the EDNN hyperparameter search dialog.

        @param parent Optional parent widget.
        """
        super().__init__(parent)

        self.setWindowTitle("EDNN Hyperparameter Search Configuration")
        self.resize(760, 580)

        self.settings = QSettings("Investigacion", "EDNN_BatchTraining")
        self.app_settings = AppSettings()

        self.graphs_dir_input = QLineEdit()
        self.graphs_dir_input.setPlaceholderText("Graphs_EDNN directory")
        self.graphs_dir_input.setText(self._ednn_subpath_or_default("Graphs_EDNN", DEFAULT_GRAPHS_DIR))
        self.graphs_btn = QPushButton("Select...")
        self.graphs_btn.clicked.connect(self.browse_graphs)

        self.train_split_input = QLineEdit()
        self.train_split_input.setPlaceholderText("train_index_folder.txt file")
        self.train_split_input.setText(self._split_file_or_default(DEFAULT_TRAIN_SPLIT_FILE))
        self.train_split_btn = QPushButton("Select...")
        self.train_split_btn.clicked.connect(self.browse_train_split)

        self.val_split_input = QLineEdit()
        self.val_split_input.setPlaceholderText("valid_index_folder.txt file")
        self.val_split_input.setText(self._split_file_or_default(DEFAULT_VAL_SPLIT_FILE))
        self.val_split_btn = QPushButton("Select...")
        self.val_split_btn.clicked.connect(self.browse_val_split)

        self.test_split_input = QLineEdit()
        self.test_split_input.setPlaceholderText("test_index_folder.txt file")
        self.test_split_input.setText(self._split_file_or_default(DEFAULT_TEST_SPLIT_FILE))
        self.test_split_btn = QPushButton("Select...")
        self.test_split_btn.clicked.connect(self.browse_test_split)

        self.models_root_input = QLineEdit()
        self.models_root_input.setPlaceholderText("Root folder for trial models")
        self.models_root_input.setText(self._ednn_subpath_or_default("Models_EDNN", DEFAULT_MODELS_DIR))
        self.models_root_btn = QPushButton("Select...")
        self.models_root_btn.clicked.connect(self.browse_models_root)

        self.results_root_input = QLineEdit()
        self.results_root_input.setPlaceholderText("Root folder for trial results")
        self.results_root_input.setText(self._ednn_subpath_or_default("Results_EDNN", DEFAULT_RESULTS_DIR))
        self.results_root_btn = QPushButton("Select...")
        self.results_root_btn.clicked.connect(self.browse_results_root)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu", "cuda:0", "cuda:1"])
        self.device_combo.setCurrentText(self._device_default())

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(self._seed_default())

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 5000)
        self.epochs_spin.setValue(self._int_setting("batch/epochs", DEFAULT_EPOCHS))

        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 1000)
        self.patience_spin.setValue(self._int_setting("batch/patience", DEFAULT_PATIENCE))

        self.lr_values_input = QLineEdit()
        self.lr_values_input.setPlaceholderText("Example: 5e-5,1e-4,5e-4,1e-3")
        self.lr_values_input.setText(
            str(self.settings.value("batch/lr_values", DEFAULT_LR_VALUES))
        )

        self.hidden_dim_values_input = QLineEdit()
        self.hidden_dim_values_input.setPlaceholderText("Example: 32,64,128")
        self.hidden_dim_values_input.setText(
            str(self.settings.value("batch/hidden_dim_values", DEFAULT_HIDDEN_DIM_VALUES))
        )

        self.batch_size_values_input = QLineEdit()
        self.batch_size_values_input.setPlaceholderText("Example: 2,4,8")
        self.batch_size_values_input.setText(
            str(self.settings.value("batch/batch_size_values", DEFAULT_BATCH_SIZE_VALUES))
        )

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

    def _with_button(self, line_edit: QLineEdit, button: QPushButton) -> QWidget:
        """
        @brief Wrap a line edit and a button in a horizontal container.

        @param line_edit Path input widget.
        @param button Browse button.
        @return QWidget containing both widgets.
        """
        container = QWidget()
        hbox = QHBoxLayout(container)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(line_edit)
        hbox.addWidget(button)
        return container

    def _ednn_subpath_or_default(self, folder_name: str, fallback: str) -> str:
        """
        @brief Build an EDNN path from the configured EDNN root.

        @param folder_name Folder name inside the EDNN root.
        @param fallback Fallback value.
        @return Selected path.
        """
        ednn_root = self.app_settings.get_value("paths/ednn_root").strip()

        if ednn_root:
            return str(Path(ednn_root) / folder_name)

        return str(fallback)

    def _split_file_or_default(self, fallback: str) -> str:
        """
        @brief Build a split file path from the configured splits folder.

        @param fallback Fallback split file name.
        @return Selected split file path.
        """
        splits_folder = self.app_settings.get_value("paths/splits_folder").strip()

        if splits_folder:
            return str(Path(splits_folder) / Path(fallback).name)

        return ""

    def _device_default(self) -> str:
        """
        @brief Return the default device from AppSettings.

        @return Device text.
        """
        device = self.app_settings.get_value("runtime/default_device").strip()
        return device if device else DEFAULT_DEVICE

    def _seed_default(self) -> int:
        """
        @brief Return the default random seed from AppSettings.

        @return Random seed.
        """
        try:
            return int(self.app_settings.get_value("runtime/default_seed"))
        except ValueError:
            return int(DEFAULT_SEED)

    def _int_setting(self, key: str, fallback: Any) -> int:
        """
        @brief Read an integer value from local dialog settings.

        @param key Local settings key.
        @param fallback Fallback value.
        @return Parsed integer.
        """
        try:
            return int(self.settings.value(key, fallback))
        except (TypeError, ValueError):
            return int(fallback)

    def browse_graphs(self) -> None:
        """
        @brief Browse for the EDNN graphs directory.

        @return None.
        """
        path = QFileDialog.getExistingDirectory(self, "Select Graphs_EDNN folder")

        if path:
            self.graphs_dir_input.setText(path)

    def browse_train_split(self) -> None:
        """
        @brief Browse for the training split file.

        @return None.
        """
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select train split",
            "",
            "Text Files (*.txt);;All Files (*)",
        )

        if path:
            self.train_split_input.setText(path)

    def browse_val_split(self) -> None:
        """
        @brief Browse for the validation split file.

        @return None.
        """
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select validation split",
            "",
            "Text Files (*.txt);;All Files (*)",
        )

        if path:
            self.val_split_input.setText(path)

    def browse_test_split(self) -> None:
        """
        @brief Browse for the test split file.

        @return None.
        """
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select test split",
            "",
            "Text Files (*.txt);;All Files (*)",
        )

        if path:
            self.test_split_input.setText(path)

    def browse_models_root(self) -> None:
        """
        @brief Browse for the EDNN models root directory.

        @return None.
        """
        path = QFileDialog.getExistingDirectory(self, "Select model root folder")

        if path:
            self.models_root_input.setText(path)

    def browse_results_root(self) -> None:
        """
        @brief Browse for the EDNN results root directory.

        @return None.
        """
        path = QFileDialog.getExistingDirectory(self, "Select result root folder")

        if path:
            self.results_root_input.setText(path)

    def accept(self) -> None:
        """
        @brief Persist dialog values and accept the dialog.

        @return None.
        """
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

        self.app_settings.set_value("runtime/default_device", self.device_combo.currentText())
        self.app_settings.set_value("runtime/default_seed", self.seed_spin.value())
        self.app_settings.sync()

        super().accept()

    @staticmethod
    def _parse_float_list(raw_text: str) -> list[float]:
        """
        @brief Parse a comma-separated float list.

        @param raw_text Raw input text.
        @return Parsed float list.
        """
        return [float(x.strip()) for x in raw_text.split(",") if x.strip()]

    @staticmethod
    def _parse_int_list(raw_text: str) -> list[int]:
        """
        @brief Parse a comma-separated integer list.

        @param raw_text Raw input text.
        @return Parsed integer list.
        """
        return [int(x.strip()) for x in raw_text.split(",") if x.strip()]

    def get_inputs(self) -> dict[str, object]:
        """
        @brief Return dialog values as EDNN hyperparameter search parameters.

        @return Input parameter dictionary.
        """
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
