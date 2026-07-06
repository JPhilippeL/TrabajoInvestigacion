"""
@file test_egnn_dialog.py
@author Mohamed EL BOUKHIARI
@brief Test dialog for the EGNN module.
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
    DEFAULT_BATCH_SIZE,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_DEVICE,
    DEFAULT_TEST_SPLIT_FILE,
)


class TestDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EGNN Evaluation Configuration")
        self.resize(720, 460)

        self.settings = QSettings("ResearchApp", "EGNN_Evaluation")

        self.graphs_dir_input = QLineEdit()
        self.graphs_dir_input.setPlaceholderText("Graphs_EGNN directory")
        self.graphs_dir_input.setText(self.settings.value("evaluation/graphs_dir", DEFAULT_GRAPHS_DIR))
        self.graphs_btn = QPushButton("Browse...")
        self.graphs_btn.clicked.connect(self.browse_graphs)

        self.test_split_input = QLineEdit()
        self.test_split_input.setPlaceholderText("test_index_folder.txt")
        self.test_split_input.setText(self.settings.value("evaluation/test_split_file", DEFAULT_TEST_SPLIT_FILE))
        self.test_split_btn = QPushButton("Browse...")
        self.test_split_btn.clicked.connect(self.browse_test_split)

        self.models_dir_input = QLineEdit()
        self.models_dir_input.setPlaceholderText("Model experiment directory")
        self.models_dir_input.setText(self.settings.value("evaluation/models_dir", DEFAULT_MODELS_DIR))
        self.models_btn = QPushButton("Browse...")
        self.models_btn.clicked.connect(self.browse_models_dir)

        self.results_dir_input = QLineEdit()
        self.results_dir_input.setPlaceholderText("Results directory")
        self.results_dir_input.setText(self.settings.value("evaluation/results_dir", DEFAULT_RESULTS_DIR))
        self.results_btn = QPushButton("Browse...")
        self.results_btn.clicked.connect(self.browse_results_dir)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu", "cuda:0", "cuda:1"])
        self.device_combo.setCurrentText(self.settings.value("evaluation/device", DEFAULT_DEVICE))

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setValue(int(self.settings.value("evaluation/batch_size", DEFAULT_BATCH_SIZE)))

        self.hidden_dim_spin = QSpinBox()
        self.hidden_dim_spin.setRange(1, 4096)
        self.hidden_dim_spin.setValue(int(self.settings.value("evaluation/hidden_dim", DEFAULT_HIDDEN_DIM)))

        form_layout = QFormLayout()
        form_layout.addRow(QLabel("<b>1. Model and data paths</b>"))
        form_layout.addRow("Graphs path:", self._with_button(self.graphs_dir_input, self.graphs_btn))
        form_layout.addRow("Test split:", self._with_button(self.test_split_input, self.test_split_btn))
        form_layout.addRow("Models directory:", self._with_button(self.models_dir_input, self.models_btn))
        form_layout.addRow("Results directory:", self._with_button(self.results_dir_input, self.results_btn))

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

    def browse_test_split(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select test split", "", "Text Files (*.txt);;All Files (*)")
        if path:
            self.test_split_input.setText(path)

    def browse_models_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select models directory")
        if path:
            self.models_dir_input.setText(path)

    def browse_results_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select results directory")
        if path:
            self.results_dir_input.setText(path)

    def accept(self):
        self.settings.setValue("evaluation/graphs_dir", self.graphs_dir_input.text())
        self.settings.setValue("evaluation/test_split_file", self.test_split_input.text())
        self.settings.setValue("evaluation/models_dir", self.models_dir_input.text())
        self.settings.setValue("evaluation/results_dir", self.results_dir_input.text())
        self.settings.setValue("evaluation/device", self.device_combo.currentText())
        self.settings.setValue("evaluation/batch_size", self.batch_spin.value())
        self.settings.setValue("evaluation/hidden_dim", self.hidden_dim_spin.value())
        super().accept()

    def get_inputs(self):
        return {
            "graphs_dir": self.graphs_dir_input.text(),
            "test_split_file": self.test_split_input.text(),
            "models_dir": self.models_dir_input.text(),
            "results_dir": self.results_dir_input.text(),
            "batch_size": self.batch_spin.value(),
            "device": self.device_combo.currentText(),
            "hidden_dim": self.hidden_dim_spin.value(),
        }
