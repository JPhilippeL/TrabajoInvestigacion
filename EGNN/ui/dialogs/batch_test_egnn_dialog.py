"""
@file batch_test_egnn_dialog.py
@author Mohamed EL BOUKHIARI
@brief Batch test dialog for the EGNN module.
@details
This dialog reads global defaults from AppSettings for EGNN paths and runtime
device configuration.
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

from EGNN.utils.constants import (
    DEFAULT_GRAPHS_DIR,
    DEFAULT_MODELS_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_DEVICE,
    DEFAULT_TEST_SPLIT_FILE,
)
from ui.utils.app_settings import AppSettings


class BatchTestDialog(QDialog):
    """
    @brief Dialog used to configure EGNN evaluation over several model runs.
    """

    def __init__(self, parent=None) -> None:
        """
        @brief Initialize the EGNN batch evaluation dialog.

        @param parent Optional parent widget.
        """
        super().__init__(parent)

        self.setWindowTitle("EGNN Batch Evaluation Configuration")
        self.resize(720, 460)

        self.settings = QSettings("ResearchApp", "EGNN_BatchEvaluation")
        self.app_settings = AppSettings()

        self.graphs_dir_input = QLineEdit()
        self.graphs_dir_input.setPlaceholderText("Graphs_EGNN directory")
        self.graphs_dir_input.setText(self._egnn_subpath_or_default("Graphs_EGNN", DEFAULT_GRAPHS_DIR))
        self.graphs_btn = QPushButton("Browse...")
        self.graphs_btn.clicked.connect(self.browse_graphs)

        self.test_split_input = QLineEdit()
        self.test_split_input.setPlaceholderText("test_index_folder.txt")
        self.test_split_input.setText(self._split_file_or_default(DEFAULT_TEST_SPLIT_FILE))
        self.test_split_btn = QPushButton("Browse...")
        self.test_split_btn.clicked.connect(self.browse_test_split)

        self.models_root_input = QLineEdit()
        self.models_root_input.setPlaceholderText("Root folder containing several model runs")
        self.models_root_input.setText(self._egnn_subpath_or_default("Models_EGNN", DEFAULT_MODELS_DIR))
        self.models_root_btn = QPushButton("Browse...")
        self.models_root_btn.clicked.connect(self.browse_models_root)

        self.results_root_input = QLineEdit()
        self.results_root_input.setPlaceholderText("Root folder for evaluation results")
        self.results_root_input.setText(self._egnn_subpath_or_default("Results_EGNN", DEFAULT_RESULTS_DIR))
        self.results_root_btn = QPushButton("Browse...")
        self.results_root_btn.clicked.connect(self.browse_results_root)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu", "cuda:0", "cuda:1"])
        self.device_combo.setCurrentText(self._device_default())

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setValue(self._int_setting("batch_evaluation/batch_size", DEFAULT_BATCH_SIZE))

        self.hidden_dim_spin = QSpinBox()
        self.hidden_dim_spin.setRange(1, 4096)
        self.hidden_dim_spin.setValue(self._int_setting("batch_evaluation/hidden_dim", DEFAULT_HIDDEN_DIM))

        form_layout = QFormLayout()
        form_layout.addRow(QLabel("<b>1. Model and data paths</b>"))
        form_layout.addRow("Graphs path:", self._with_button(self.graphs_dir_input, self.graphs_btn))
        form_layout.addRow("Test split:", self._with_button(self.test_split_input, self.test_split_btn))
        form_layout.addRow("Models root:", self._with_button(self.models_root_input, self.models_root_btn))
        form_layout.addRow("Results root:", self._with_button(self.results_root_input, self.results_root_btn))

        form_layout.addRow(QLabel("<br><b>2. Inference configuration</b>"))
        form_layout.addRow("Device:", self.device_combo)
        form_layout.addRow("Batch size:", self.batch_spin)
        form_layout.addRow("Hidden dimension:", self.hidden_dim_spin)

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

    def _egnn_subpath_or_default(self, folder_name: str, fallback: str) -> str:
        """
        @brief Build an EGNN path from the configured EGNN root.

        @param folder_name Folder name inside the EGNN root.
        @param fallback Fallback value.
        @return Selected path.
        """
        egnn_root = self.app_settings.get_value("paths/egnn_root").strip()

        if egnn_root:
            return str(Path(egnn_root) / folder_name)

        return str(fallback)

    def _split_file_or_default(self, fallback: str) -> str:
        """
        @brief Build a split file path from the configured splits folder.

        @param fallback Fallback split file path.
        @return Selected split file path.
        """
        splits_folder = self.app_settings.get_value("paths/splits_folder").strip()

        if splits_folder:
            return str(Path(splits_folder) / Path(fallback).name)

        return str(fallback)

    def _device_default(self) -> str:
        """
        @brief Return the default device from AppSettings.

        @return Device text.
        """
        device = self.app_settings.get_value("runtime/default_device").strip()
        return device if device else str(DEFAULT_DEVICE)

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
        @brief Browse for the EGNN graphs directory.

        @return None.
        """
        path = QFileDialog.getExistingDirectory(self, "Select Graphs_EGNN directory")

        if path:
            self.graphs_dir_input.setText(path)

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
        @brief Browse for the EGNN models root directory.

        @return None.
        """
        path = QFileDialog.getExistingDirectory(self, "Select models root directory")

        if path:
            self.models_root_input.setText(path)

    def browse_results_root(self) -> None:
        """
        @brief Browse for the EGNN results root directory.

        @return None.
        """
        path = QFileDialog.getExistingDirectory(self, "Select results root directory")

        if path:
            self.results_root_input.setText(path)

    def accept(self) -> None:
        """
        @brief Persist dialog values and accept the dialog.

        @return None.
        """
        self.settings.setValue("batch_evaluation/graphs_dir", self.graphs_dir_input.text())
        self.settings.setValue("batch_evaluation/test_split_file", self.test_split_input.text())
        self.settings.setValue("batch_evaluation/models_root", self.models_root_input.text())
        self.settings.setValue("batch_evaluation/results_root", self.results_root_input.text())
        self.settings.setValue("batch_evaluation/device", self.device_combo.currentText())
        self.settings.setValue("batch_evaluation/batch_size", self.batch_spin.value())
        self.settings.setValue("batch_evaluation/hidden_dim", self.hidden_dim_spin.value())

        self.app_settings.set_value("runtime/default_device", self.device_combo.currentText())
        self.app_settings.sync()

        super().accept()

    def get_inputs(self) -> dict[str, object]:
        """
        @brief Return dialog values as EGNN batch evaluation parameters.

        @return Input parameter dictionary.
        """
        return {
            "graphs_dir": self.graphs_dir_input.text(),
            "test_split_file": self.test_split_input.text(),
            "models_root": self.models_root_input.text(),
            "results_root": self.results_root_input.text(),
            "batch_size": self.batch_spin.value(),
            "device": self.device_combo.currentText(),
            "hidden_dim": self.hidden_dim_spin.value(),
        }
