"""Checkpoint evaluation dialog for DEAttentionDTA."""

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout

from DEAttentionDTA.ui.dialogs._shared import browse_existing_directory, browse_existing_file, with_button


class TestDEAttentionDTADialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DEAttentionDTA Evaluation Configuration")
        self.resize(860, 500)
        self.settings = QSettings("ResearchApp", "DEAttentionDTA_Evaluation")
        self.checkpoint_input = QLineEdit(self.settings.value("evaluation/checkpoint", "DEAttentionDTA/models/pretrained/DEAttentionDTA.pt")); self.checkpoint_btn = QPushButton("Browse..."); self.checkpoint_btn.clicked.connect(lambda: browse_existing_file(self, "Select DEAttentionDTA checkpoint", "PyTorch Checkpoint (*.pt *.ckpt);;All Files (*)", self.checkpoint_input))
        self.prepared_dir_input = QLineEdit(self.settings.value("evaluation/prepared_dir", "DEAttentionDTA/data/urv_dataset_v3b_prepared")); self.prepared_dir_btn = QPushButton("Browse..."); self.prepared_dir_btn.clicked.connect(lambda: browse_existing_directory(self, "Select prepared dataset", self.prepared_dir_input))
        self.results_dir_input = QLineEdit(self.settings.value("evaluation/results_dir", "DEAttentionDTA/outputs/predictions")); self.results_dir_btn = QPushButton("Browse..."); self.results_dir_btn.clicked.connect(lambda: browse_existing_directory(self, "Select results directory", self.results_dir_input))
        self.splits_input = QLineEdit(self.settings.value("evaluation/splits", "all")); self.splits_input.setPlaceholderText("all or 1,3,5")
        self.pretrained_fold_input = QLineEdit(self.settings.value("evaluation/pretrained_fold", "matching")); self.pretrained_fold_input.setPlaceholderText("matching, first or a fold number")
        self.device_combo = QComboBox(); self.device_combo.addItems(["auto", "cpu", "cuda"]); self.device_combo.setCurrentText(self.settings.value("evaluation/device", "auto"))
        self.batch_size_spin = QSpinBox(); self.batch_size_spin.setRange(1, 1024); self.batch_size_spin.setValue(int(self.settings.value("evaluation/batch_size", 16)))
        self.num_workers_spin = QSpinBox(); self.num_workers_spin.setRange(0, 64); self.num_workers_spin.setValue(int(self.settings.value("evaluation/num_workers", 0)))
        self.non_strict_check = QCheckBox("Allow non-strict checkpoint loading"); self.non_strict_check.setChecked(str(self.settings.value("evaluation/non_strict_pretrained", "false")).lower() in {"true", "1", "yes"})
        form = QFormLayout(); form.addRow(QLabel("<b>1. Checkpoint and prepared dataset</b>")); form.addRow("Checkpoint:", with_button(self.checkpoint_input, self.checkpoint_btn)); form.addRow("Prepared dataset:", with_button(self.prepared_dir_input, self.prepared_dir_btn)); form.addRow("Results directory:", with_button(self.results_dir_input, self.results_dir_btn)); form.addRow("Official splits:", self.splits_input); form.addRow("Checkpoint fold:", self.pretrained_fold_input)
        form.addRow(QLabel("<br><b>2. Runtime</b>")); form.addRow("Device:", self.device_combo); form.addRow("Batch size:", self.batch_size_spin); form.addRow("DataLoader workers:", self.num_workers_spin); form.addRow("Compatibility mode:", self.non_strict_check)
        form.addRow("Note:", QLabel("The selected checkpoint is evaluated without training. Validation and test outputs are stored separately."))
        layout = QVBoxLayout(); layout.addLayout(form); buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel); buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons); self.setLayout(layout)

    def accept(self) -> None:
        for key, value in self.get_inputs().items(): self.settings.setValue(f"evaluation/{key}", value)
        super().accept()

    def get_inputs(self) -> dict:
        return {"checkpoint": self.checkpoint_input.text().strip(), "prepared_dir": self.prepared_dir_input.text().strip(), "results_dir": self.results_dir_input.text().strip(), "splits": self.splits_input.text().strip(), "pretrained_fold": self.pretrained_fold_input.text().strip(), "device": self.device_combo.currentText(), "batch_size": self.batch_size_spin.value(), "num_workers": self.num_workers_spin.value(), "non_strict_pretrained": self.non_strict_check.isChecked()}
