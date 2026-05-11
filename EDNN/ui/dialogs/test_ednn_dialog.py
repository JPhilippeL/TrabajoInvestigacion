"""
@file test_ednn_dialog.py
@author Mohamed EL BOUKHIARI
@brief Test dialog for the EDNN module.
@details
This dialog reads global defaults from AppSettings for EDNN paths and runtime
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

from EDNN.utils.constants import (
    DEFAULT_GRAPHS_DIR,
    DEFAULT_MODELS_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_HIDDEN_DIM,
)
from ui.utils.app_settings import AppSettings


DEFAULT_TEST_SPLIT_FILE = "test_index_folder.txt"
DEFAULT_DEVICE = "auto"


class TestDialog(QDialog):
    """
    @brief Dialog used to configure EDNN evaluation for one model directory.
    """

    def __init__(self, parent=None) -> None:
        """
        @brief Initialize the EDNN evaluation dialog.

        @param parent Optional parent widget.
        """
        super().__init__(parent)

        self.setWindowTitle("Evaluation Configuration - EDNN")
        self.resize(700, 460)

        self.settings = QSettings("Investigacion", "EDNN_Testing")
        self.app_settings = AppSettings()

        self.graphs_dir_input = QLineEdit()
        self.graphs_dir_input.setPlaceholderText("Graphs_EDNN directory")
        self.graphs_dir_input.setText(self._ednn_subpath_or_default("Graphs_EDNN", DEFAULT_GRAPHS_DIR))
        self.graphs_btn = QPushButton("Select...")
        self.graphs_btn.clicked.connect(self.browse_graphs)

        self.test_split_input = QLineEdit()
        self.test_split_input.setPlaceholderText("test_index_folder.txt file")
        self.test_split_input.setText(self._split_file_or_default(DEFAULT_TEST_SPLIT_FILE))
        self.test_split_btn = QPushButton("Select...")
        self.test_split_btn.clicked.connect(self.browse_test_split)

        self.models_dir_input = QLineEdit()
        self.models_dir_input.setPlaceholderText("Model experiment directory")
        self.models_dir_input.setText(self._ednn_subpath_or_default("Models_EDNN", DEFAULT_MODELS_DIR))
        self.models_btn = QPushButton("Select...")
        self.models_btn.clicked.connect(self.browse_models_dir)

        self.results_dir_input = QLineEdit()
        self.results_dir_input.setPlaceholderText("Results directory")
        self.results_dir_input.setText(self._ednn_subpath_or_default("Results_EDNN", DEFAULT_RESULTS_DIR))
        self.results_btn = QPushButton("Select...")
        self.results_btn.clicked.connect(self.browse_results_dir)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu", "cuda:0", "cuda:1"])
        self.device_combo.setCurrentText(self._device_default())

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setValue(self._int_setting("test/batch_size", DEFAULT_BATCH_SIZE))

        self.hidden_dim_spin = QSpinBox()
        self.hidden_dim_spin.setRange(1, 4096)
        self.hidden_dim_spin.setValue(self._int_setting("test/hidden_dim", DEFAULT_HIDDEN_DIM))

        form_layout = QFormLayout()
        form_layout.addRow(QLabel("<b>1. Model and data</b>"))
        form_layout.addRow("Graphs Path:", self._with_button(self.graphs_dir_input, self.graphs_btn))
        form_layout.addRow("Test Split (.txt):", self._with_button(self.test_split_input, self.test_split_btn))
        form_layout.addRow("Models Directory:", self._with_button(self.models_dir_input, self.models_btn))
        form_layout.addRow("Results Directory:", self._with_button(self.results_dir_input, self.results_btn))

        form_layout.addRow(QLabel("<br><b>2. Inference configuration</b>"))
        form_layout.addRow("Device:", self.device_combo)
        form_layout.addRow("Batch Size:", self.batch_spin)
        form_layout.addRow("Hidden Dim:", self.hidden_dim_spin)

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

    def _selected_device(self) -> str | None:
        """
        @brief Return selected device for the worker.

        @return Device string, or None when auto is selected.
        """
        device = self.device_combo.currentText()
        return None if device == "auto" else device

    def browse_graphs(self) -> None:
        """
        @brief Browse for the EDNN graphs directory.

        @return None.
        """
        path = QFileDialog.getExistingDirectory(self, "Select Graphs_EDNN folder")

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

    def browse_models_dir(self) -> None:
        """
        @brief Browse for the EDNN models directory.

        @return None.
        """
        path = QFileDialog.getExistingDirectory(self, "Select model folder")

        if path:
            self.models_dir_input.setText(path)

    def browse_results_dir(self) -> None:
        """
        @brief Browse for the EDNN results directory.

        @return None.
        """
        path = QFileDialog.getExistingDirectory(self, "Select results folder")

        if path:
            self.results_dir_input.setText(path)

    def accept(self) -> None:
        """
        @brief Persist dialog values and accept the dialog.

        @return None.
        """
        self.settings.setValue("test/graphs_dir", self.graphs_dir_input.text())
        self.settings.setValue("test/test_split_file", self.test_split_input.text())
        self.settings.setValue("test/models_dir", self.models_dir_input.text())
        self.settings.setValue("test/results_dir", self.results_dir_input.text())
        self.settings.setValue("test/device", self.device_combo.currentText())
        self.settings.setValue("test/batch_size", self.batch_spin.value())
        self.settings.setValue("test/hidden_dim", self.hidden_dim_spin.value())

        self.app_settings.set_value("runtime/default_device", self.device_combo.currentText())
        self.app_settings.sync()

        super().accept()

    def get_inputs(self) -> dict[str, object]:
        """
        @brief Return dialog values as EDNN evaluation parameters.

        @return Input parameter dictionary.
        """
        return {
            "graphs_dir": self.graphs_dir_input.text(),
            "test_split_file": self.test_split_input.text(),
            "models_dir": self.models_dir_input.text(),
            "results_dir": self.results_dir_input.text(),
            "batch_size": self.batch_spin.value(),
            "device": self._selected_device(),
            "hidden_dim": self.hidden_dim_spin.value(),
        }
