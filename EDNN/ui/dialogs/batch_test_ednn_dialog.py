"""
@file batch_test_ednn_dialog.py
@author Mohamed EL BOUKHIARI
@brief Batch test dialog for the EDNN module.
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
    DEFAULT_BATCH_SIZE,
    DEFAULT_HIDDEN_DIM,
)


class BatchTestDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Evaluation Configuration - EDNN")
        self.resize(700, 460)

        self.settings = QSettings("Investigacion", "EDNN_BatchTesting")

        self.graphs_dir_input = QLineEdit()
        self.graphs_dir_input.setPlaceholderText("Graphs_EDNN directory")
        self.graphs_dir_input.setText(self.settings.value("batch_test/graphs_dir", DEFAULT_GRAPHS_DIR))
        self.graphs_btn = QPushButton("Select...")
        self.graphs_btn.clicked.connect(self.browse_graphs)

        self.test_split_input = QLineEdit()
        self.test_split_input.setPlaceholderText("test_index_folder.txt file")
        self.test_split_input.setText(self.settings.value("batch_test/test_split_file", ""))
        self.test_split_btn = QPushButton("Select...")
        self.test_split_btn.clicked.connect(self.browse_test_split)

        self.models_root_input = QLineEdit()
        self.models_root_input.setPlaceholderText("Root folder containing several models/runs")
        self.models_root_input.setText(self.settings.value("batch_test/models_root", DEFAULT_MODELS_DIR))
        self.models_root_btn = QPushButton("Select...")
        self.models_root_btn.clicked.connect(self.browse_models_root)

        self.results_root_input = QLineEdit()
        self.results_root_input.setPlaceholderText("Root folder for results")
        self.results_root_input.setText(self.settings.value("batch_test/results_root", DEFAULT_RESULTS_DIR))
        self.results_root_btn = QPushButton("Select...")
        self.results_root_btn.clicked.connect(self.browse_results_root)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu", "cuda:0", "cuda:1"])
        self.device_combo.setCurrentText(self.settings.value("batch_test/device", "cuda"))

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setValue(int(self.settings.value("batch_test/batch_size", DEFAULT_BATCH_SIZE)))

        self.hidden_dim_spin = QSpinBox()
        self.hidden_dim_spin.setRange(1, 4096)
        self.hidden_dim_spin.setValue(int(self.settings.value("batch_test/hidden_dim", DEFAULT_HIDDEN_DIM)))

        form_layout = QFormLayout()
        form_layout.addRow(QLabel("<b>1. Models and data</b>"))
        form_layout.addRow("Graphs Path:", self._with_button(self.graphs_dir_input, self.graphs_btn))
        form_layout.addRow("Test Split (.txt):", self._with_button(self.test_split_input, self.test_split_btn))
        form_layout.addRow("Models Root:", self._with_button(self.models_root_input, self.models_root_btn))
        form_layout.addRow("Results Root:", self._with_button(self.results_root_input, self.results_root_btn))

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

    def browse_test_split(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select test split", "", "Text Files (*.txt);;All Files (*)")
        if path:
            self.test_split_input.setText(path)

    def browse_models_root(self):
        path = QFileDialog.getExistingDirectory(self, "Select model root folder")
        if path:
            self.models_root_input.setText(path)

    def browse_results_root(self):
        path = QFileDialog.getExistingDirectory(self, "Select results root folder")
        if path:
            self.results_root_input.setText(path)

    def accept(self):
        self.settings.setValue("batch_test/graphs_dir", self.graphs_dir_input.text())
        self.settings.setValue("batch_test/test_split_file", self.test_split_input.text())
        self.settings.setValue("batch_test/models_root", self.models_root_input.text())
        self.settings.setValue("batch_test/results_root", self.results_root_input.text())
        self.settings.setValue("batch_test/device", self.device_combo.currentText())
        self.settings.setValue("batch_test/batch_size", self.batch_spin.value())
        self.settings.setValue("batch_test/hidden_dim", self.hidden_dim_spin.value())
        super().accept()

    def get_inputs(self):
        return {
            "graphs_dir": self.graphs_dir_input.text(),
            "test_split_file": self.test_split_input.text(),
            "models_root": self.models_root_input.text(),
            "results_root": self.results_root_input.text(),
            "batch_size": self.batch_spin.value(),
            "device": self.device_combo.currentText(),
            "hidden_dim": self.hidden_dim_spin.value(),
        }
