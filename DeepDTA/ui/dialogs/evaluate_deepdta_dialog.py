from __future__ import annotations

import os

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget

from DeepDTA.utils.constants import DEFAULT_DATASET, DEFAULT_DEVICE, MODULE_ROOT


class EvaluateDeepDTADialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DeepDTA Evaluate/Predict Model")
        self.resize(720, 380)
        self.settings = QSettings("ResearchApp", "DeepDTA_Evaluate")
        self.dataset_combo = QComboBox(); self.dataset_combo.addItems(["davis", "kiba", "mpro_urv"]); self.dataset_combo.setCurrentText(self.settings.value("dataset", DEFAULT_DATASET))
        self.checkpoint_input = QLineEdit(self.settings.value("checkpoint_path", ""))
        self.output_input = QLineEdit(self.settings.value("output_dir", os.path.join(MODULE_ROOT, "results", "deepdta_evaluation", "runs")))
        self.device_combo = QComboBox(); self.device_combo.addItems(["auto", "cpu", "cuda", "cuda:0"]); self.device_combo.setCurrentText(self.settings.value("device", DEFAULT_DEVICE))
        self.fold_spin = QSpinBox(); self.fold_spin.setRange(0, 99); self.fold_spin.setValue(int(self.settings.value("fold_index", 0)))
        self.use_folds_check = QCheckBox("Use dataset fold files when available"); self.use_folds_check.setChecked(self.settings.value("use_dataset_folds", True, type=bool))
        self.split_combo = QComboBox(); self.split_combo.addItems(["test", "valid", "train", "all"]); self.split_combo.setCurrentText(self.settings.value("split", "test"))
        self.batch_spin = QSpinBox(); self.batch_spin.setRange(1, 4096); self.batch_spin.setValue(int(self.settings.value("batch_size", 4)))
        form = QFormLayout()
        form.addRow("Dataset:", self.dataset_combo)
        form.addRow("Checkpoint path:", self._path_row(self.checkpoint_input, self.browse_checkpoint))
        form.addRow("Output directory:", self._path_row(self.output_input, self.browse_output))
        form.addRow("Device:", self.device_combo)
        form.addRow("Use dataset folds:", self.use_folds_check)
        form.addRow("Fold index:", self.fold_spin)
        form.addRow("Split:", self.split_combo)
        form.addRow("Batch size:", self.batch_spin)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self); layout.addLayout(form); layout.addWidget(buttons)

    def _path_row(self, edit, slot):
        button = QPushButton("Browse..."); button.clicked.connect(slot)
        box = QWidget(); hbox = QHBoxLayout(box); hbox.setContentsMargins(0, 0, 0, 0); hbox.addWidget(edit); hbox.addWidget(button); return box

    def browse_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select checkpoint", filter="PyTorch checkpoints (*.pt *.pth);;All files (*)")
        if path: self.checkpoint_input.setText(path)

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path: self.output_input.setText(path)

    def accept(self):
        for key, value in self.get_inputs().items(): self.settings.setValue(key, value)
        super().accept()

    def get_inputs(self) -> dict:
        checkpoint = self.checkpoint_input.text().strip()
        output = self.output_input.text().strip()
        if not checkpoint or not os.path.isfile(checkpoint): raise ValueError("Checkpoint path is missing or does not exist.")
        if not output: raise ValueError("Output directory is required.")
        return {
            "dataset_name": self.dataset_combo.currentText(), "checkpoint_path": checkpoint, "output_dir": output,
            "device": self.device_combo.currentText(), "fold_index": self.fold_spin.value(), "use_dataset_folds": self.use_folds_check.isChecked(),
            "split": self.split_combo.currentText(), "batch_size": self.batch_spin.value(),
        }
