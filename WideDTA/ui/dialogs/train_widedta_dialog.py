from __future__ import annotations

import os

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget

from WideDTA.utils.constants import DEFAULT_DATASET, DEFAULT_DEVICE, DEFAULT_EPOCHS, DEFAULT_SEED, DEFAULT_TEST_SPLIT, DEFAULT_VAL_SPLIT, MODULE_ROOT


class TrainWideDTADialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("WideDTA Train Model")
        self.resize(720, 520)
        self.settings = QSettings("ResearchApp", "WideDTA_Train")
        self.dataset_combo = QComboBox(); self.dataset_combo.addItems(["davis", "kiba", "mpro_urv"]); self.dataset_combo.setCurrentText(self.settings.value("dataset", DEFAULT_DATASET))
        self.output_input = QLineEdit(self.settings.value("output_base", os.path.join(MODULE_ROOT, "results", "widedta_runs")))
        self.device_combo = QComboBox(); self.device_combo.addItems(["auto", "cpu", "cuda", "cuda:0"]); self.device_combo.setCurrentText(self.settings.value("device", DEFAULT_DEVICE))
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(0, 999999); self.seed_spin.setValue(int(self.settings.value("seed", DEFAULT_SEED)))
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(1, 5000); self.epochs_spin.setValue(int(self.settings.value("epochs", DEFAULT_EPOCHS)))
        self.batch_spin = QSpinBox(); self.batch_spin.setRange(1, 4096); self.batch_spin.setValue(int(self.settings.value("batch_size", 1)))
        self.lr_spin = QDoubleSpinBox(); self.lr_spin.setDecimals(6); self.lr_spin.setRange(0.000001, 1.0); self.lr_spin.setValue(float(self.settings.value("lr", 0.003)))
        self.dropout_spin = QDoubleSpinBox(); self.dropout_spin.setDecimals(2); self.dropout_spin.setRange(0.0, 0.9); self.dropout_spin.setValue(float(self.settings.value("dropout", 0.3)))
        self.fold_spin = QSpinBox(); self.fold_spin.setRange(0, 99); self.fold_spin.setValue(int(self.settings.value("fold_index", 0)))
        self.fold_spin.setToolTip("Index of the official dataset split to use, usually 0 to 4.")
        self.val_spin = QDoubleSpinBox(); self.val_spin.setDecimals(2); self.val_spin.setRange(0.0, 0.8); self.val_spin.setSingleStep(0.05); self.val_spin.setValue(float(self.settings.value("val_split", DEFAULT_VAL_SPLIT)))
        self.test_spin = QDoubleSpinBox(); self.test_spin.setDecimals(2); self.test_spin.setRange(0.05, 0.8); self.test_spin.setSingleStep(0.05); self.test_spin.setValue(float(self.settings.value("test_split", DEFAULT_TEST_SPLIT)))
        self.max_batches_spin = QSpinBox(); self.max_batches_spin.setRange(0, 100000); self.max_batches_spin.setValue(int(self.settings.value("max_train_batches", 0)))
        form = QFormLayout()
        form.addRow("Dataset:", self.dataset_combo)
        form.addRow("Output directory:", self._path_row(self.output_input))
        form.addRow("Device:", self.device_combo)
        form.addRow("Seed:", self.seed_spin)
        form.addRow("Epochs:", self.epochs_spin)
        form.addRow("Batch size:", self.batch_spin)
        form.addRow("Learning rate:", self.lr_spin)
        form.addRow("Dropout:", self.dropout_spin)
        form.addRow("Split/Fold index:", self.fold_spin)
        form.addRow("Validation split:", self.val_spin)
        form.addRow("Test split:", self.test_spin)
        form.addRow("Max train batches:", self.max_batches_spin)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self); layout.addLayout(form); layout.addWidget(buttons)

    def _path_row(self, edit):
        button = QPushButton("Browse..."); button.clicked.connect(self.browse_output)
        box = QWidget(); hbox = QHBoxLayout(box); hbox.setContentsMargins(0, 0, 0, 0); hbox.addWidget(edit); hbox.addWidget(button); return box

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path: self.output_input.setText(path)

    def accept(self):
        for key, value in self.get_inputs().items():
            if value is not None: self.settings.setValue(key, value)
        super().accept()

    def get_inputs(self) -> dict:
        output = self.output_input.text().strip()
        if not output: raise ValueError("Output directory is required.")
        if self.val_spin.value() + self.test_spin.value() >= 1.0: raise ValueError("Validation split + test split must be lower than 1.")
        max_batches = self.max_batches_spin.value()
        params = {
            "dataset_name": self.dataset_combo.currentText(), "output_base": output, "device": self.device_combo.currentText(),
            "seed": self.seed_spin.value(), "epochs": self.epochs_spin.value(), "batch_size": self.batch_spin.value(), "lr": self.lr_spin.value(),
            "fold_index": self.fold_spin.value(), "use_dataset_folds": True, "val_split": self.val_spin.value(),
            "test_split": self.test_spin.value(), "max_train_batches": None if max_batches == 0 else max_batches,
        }
        params["dropout"] = self.dropout_spin.value()
        return params
