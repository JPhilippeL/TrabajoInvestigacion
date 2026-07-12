"""Checkpoint evaluation dialog for DEAttentionDTA."""

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout

from DEAttentionDTA.ui.dialogs._shared import browse_existing_directory, browse_existing_file, with_button


class TestDEAttentionDTADialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DEAttentionDTA Evaluate")
        self.resize(820, 460)
        self.settings = QSettings("ResearchApp", "DEAttentionDTA_Evaluation")
        self.prepared_dir_input = QLineEdit(self.settings.value("evaluation/prepared_dir", "DEAttentionDTA/data/urv_dataset_v3b_prepared"))
        self.prepared_dir_btn = QPushButton("Browse...")
        self.prepared_dir_btn.clicked.connect(lambda: browse_existing_directory(self, "Select prepared dataset", self.prepared_dir_input))
        self.checkpoint_input = QLineEdit(self.settings.value("evaluation/checkpoint", "DEAttentionDTA/models/pretrained/DEAttentionDTA.pt"))
        self.checkpoint_btn = QPushButton("Browse...")
        self.checkpoint_btn.clicked.connect(lambda: browse_existing_file(self, "Select DEAttentionDTA checkpoint", "PyTorch Checkpoint (*.pt *.ckpt);;All Files (*)", self.checkpoint_input))
        self.results_dir_input = QLineEdit(self.settings.value("evaluation/results_dir", "DEAttentionDTA/outputs/predictions"))
        self.results_dir_btn = QPushButton("Browse...")
        self.results_dir_btn.clicked.connect(lambda: browse_existing_directory(self, "Select output directory", self.results_dir_input))
        self.fold_spin = QSpinBox(); self.fold_spin.setRange(1, 5); self.fold_spin.setValue(int(self.settings.value("evaluation/fold_index", 1)))
        self.fold_spin.setToolTip("Index of the official dataset split to use, usually 0 to 4.")
        self.split_combo = QComboBox(); self.split_combo.addItems(["test", "valid", "train", "all"]); self.split_combo.setCurrentText(self.settings.value("evaluation/split", "test"))
        self.device_combo = QComboBox(); self.device_combo.addItems(["auto", "cpu", "cuda", "cuda:0"]); self.device_combo.setCurrentText(self.settings.value("evaluation/device", "auto"))
        self.batch_size_spin = QSpinBox(); self.batch_size_spin.setRange(1, 1024); self.batch_size_spin.setValue(int(self.settings.value("evaluation/batch_size", 16)))

        form = QFormLayout()
        form.addRow(QLabel("<b>Checkpoint and dataset</b>"))
        form.addRow("Prepared dataset root:", with_button(self.prepared_dir_input, self.prepared_dir_btn))
        form.addRow("Checkpoint path:", with_button(self.checkpoint_input, self.checkpoint_btn))
        form.addRow("Output directory:", with_button(self.results_dir_input, self.results_dir_btn))
        form.addRow("Official split index:", self.fold_spin)
        form.addRow("Split:", self.split_combo)
        form.addRow(QLabel("<br><b>Runtime</b>"))
        form.addRow("Device:", self.device_combo)
        form.addRow("Batch size:", self.batch_size_spin)
        form.addRow("Checkpoint fold:", QLabel("Matching dataset fold"))
        layout = QVBoxLayout(); layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons); self.setLayout(layout)

    def accept(self) -> None:
        for key, value in self.get_inputs().items(): self.settings.setValue(f"evaluation/{key}", value)
        self.settings.setValue("evaluation/fold_index", self.fold_spin.value())
        self.settings.setValue("evaluation/split", self.split_combo.currentText())
        super().accept()

    def get_inputs(self) -> dict:
        fold = str(self.fold_spin.value())
        return {
            "prepared_dir": self.prepared_dir_input.text().strip(),
            "checkpoint": self.checkpoint_input.text().strip(),
            "results_dir": self.results_dir_input.text().strip(),
            "splits": fold,
            "fold_index": self.fold_spin.value(),
            "split": self.split_combo.currentText(),
            "eval_split_mode": self.split_combo.currentText(),
            "pretrained_fold": "matching",
            "device": self.device_combo.currentText(),
            "batch_size": self.batch_size_spin.value(),
            "num_workers": 0,
            "non_strict_pretrained": False,
        }
