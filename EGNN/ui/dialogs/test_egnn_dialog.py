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


def _safe_text(settings: QSettings, key: str, default: str) -> str:
    value = settings.value(key, default)
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() == "none":
        return default
    return text


def _safe_int(settings: QSettings, key: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(settings.value(key, default))
    except (TypeError, ValueError):
        return default
    return min(max(value, minimum), maximum)


def _safe_choice(settings: QSettings, key: str, default: str, choices: list[str]) -> str:
    value = _safe_text(settings, key, default)
    return value if value in choices else default


class TestDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EGNN Evaluation Configuration")
        self.resize(720, 460)

        self.settings = QSettings("ResearchApp", "EGNN_Evaluation")

        self.graphs_dir_input = QLineEdit()
        self.graphs_dir_input.setPlaceholderText("Graphs_EGNN directory")
        self.graphs_dir_input.setText(_safe_text(self.settings, "evaluation/graphs_dir", DEFAULT_GRAPHS_DIR))
        self.graphs_btn = QPushButton("Browse...")
        self.graphs_btn.clicked.connect(self.browse_graphs)

        self.test_split_input = QLineEdit()
        self.test_split_input.setPlaceholderText("test_index_folder.txt")
        self.test_split_input.setText(_safe_text(self.settings, "evaluation/test_split_file", DEFAULT_TEST_SPLIT_FILE))
        self.test_split_btn = QPushButton("Browse...")
        self.test_split_btn.clicked.connect(self.browse_test_split)

        self.checkpoint_or_run_input = QLineEdit()
        self.checkpoint_or_run_input.setPlaceholderText("Checkpoint file or trained run root")
        checkpoint_or_run = _safe_text(
            self.settings,
            "evaluation/checkpoint_or_run",
            _safe_text(self.settings, "evaluation/models_dir", DEFAULT_MODELS_DIR),
        )
        self.checkpoint_or_run_input.setText(checkpoint_or_run)
        self.checkpoint_file_btn = QPushButton("File...")
        self.checkpoint_file_btn.clicked.connect(self.browse_checkpoint_file)
        self.run_root_btn = QPushButton("Folder...")
        self.run_root_btn.clicked.connect(self.browse_run_root)

        self.results_dir_input = QLineEdit()
        self.results_dir_input.setPlaceholderText("Results directory")
        self.results_dir_input.setText(_safe_text(self.settings, "evaluation/results_dir", DEFAULT_RESULTS_DIR))
        self.results_btn = QPushButton("Browse...")
        self.results_btn.clicked.connect(self.browse_results_dir)

        self.device_combo = QComboBox()
        device_choices = ["auto", "cuda", "cpu", "cuda:0", "cuda:1"]
        self.device_combo.addItems(device_choices)
        self.device_combo.setCurrentText(_safe_choice(self.settings, "evaluation/device", DEFAULT_DEVICE, device_choices))

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setValue(_safe_int(self.settings, "evaluation/batch_size", DEFAULT_BATCH_SIZE, 1, 1024))

        self.hidden_dim_spin = QSpinBox()
        self.hidden_dim_spin.setRange(1, 4096)
        self.hidden_dim_spin.setValue(_safe_int(self.settings, "evaluation/hidden_dim", DEFAULT_HIDDEN_DIM, 1, 4096))

        self.split_index_spin = QSpinBox()
        self.split_index_spin.setRange(0, 999)
        self.split_index_spin.setValue(_safe_int(self.settings, "evaluation/split_index", 0, 0, 999))

        self.scope_combo = QComboBox()
        scope_choices = ["auto", "single checkpoint", "all detected splits"]
        self.scope_combo.addItems(scope_choices)
        self.scope_combo.setCurrentText(_safe_choice(self.settings, "evaluation/scope", "auto", scope_choices))

        form_layout = QFormLayout()
        form_layout.addRow(QLabel("<b>1. Model and data paths</b>"))
        form_layout.addRow("Graphs path:", self._with_button(self.graphs_dir_input, self.graphs_btn))
        form_layout.addRow("Test split:", self._with_button(self.test_split_input, self.test_split_btn))
        form_layout.addRow(
            "Checkpoint file or trained run root:",
            self._with_buttons(self.checkpoint_or_run_input, self.checkpoint_file_btn, self.run_root_btn),
        )
        form_layout.addRow("Output root:", self._with_button(self.results_dir_input, self.results_btn))

        form_layout.addRow(QLabel("<br><b>2. Inference configuration</b>"))
        form_layout.addRow("Evaluation scope:", self.scope_combo)
        form_layout.addRow("Split index:", self.split_index_spin)
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

    def _with_buttons(self, line_edit, *buttons):
        container = QWidget()
        hbox = QHBoxLayout(container)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(line_edit)
        for button in buttons:
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

    def browse_checkpoint_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select checkpoint file", "", "PyTorch Files (*.pt);;All Files (*)")
        if path:
            self.checkpoint_or_run_input.setText(path)

    def browse_run_root(self):
        path = QFileDialog.getExistingDirectory(self, "Select trained run root")
        if path:
            self.checkpoint_or_run_input.setText(path)

    def browse_results_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select results directory")
        if path:
            self.results_dir_input.setText(path)

    def accept(self):
        self.settings.setValue("evaluation/graphs_dir", self.graphs_dir_input.text())
        self.settings.setValue("evaluation/test_split_file", self.test_split_input.text())
        self.settings.setValue("evaluation/checkpoint_or_run", self.checkpoint_or_run_input.text())
        self.settings.setValue("evaluation/results_dir", self.results_dir_input.text())
        self.settings.setValue("evaluation/device", self.device_combo.currentText())
        self.settings.setValue("evaluation/batch_size", self.batch_spin.value())
        self.settings.setValue("evaluation/hidden_dim", self.hidden_dim_spin.value())
        self.settings.setValue("evaluation/split_index", self.split_index_spin.value())
        self.settings.setValue("evaluation/scope", self.scope_combo.currentText())
        super().accept()

    def get_inputs(self):
        return {
            "graphs_dir": self.graphs_dir_input.text(),
            "test_split_file": self.test_split_input.text(),
            "checkpoint_or_run": self.checkpoint_or_run_input.text(),
            "results_dir": self.results_dir_input.text(),
            "batch_size": self.batch_spin.value(),
            "device": self.device_combo.currentText(),
            "hidden_dim": self.hidden_dim_spin.value(),
            "split_index": self.split_index_spin.value(),
            "evaluation_scope": self.scope_combo.currentText().replace(" ", "_"),
        }
