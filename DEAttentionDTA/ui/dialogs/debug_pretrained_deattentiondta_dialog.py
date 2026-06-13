"""Pretrained-checkpoint validation dialog for DEAttentionDTA."""

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout

from DEAttentionDTA.ui.dialogs._shared import browse_existing_directory, browse_existing_file, with_button


class DebugPretrainedDEAttentionDTADialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DEAttentionDTA Pretrained Checkpoint Validation")
        self.resize(860, 390)
        self.settings = QSettings("ResearchApp", "DEAttentionDTA_DebugPretrained")

        self.prepared_dir_input = QLineEdit(self.settings.value("debug_pretrained/prepared_dir", "DEAttentionDTA/data/urv_dataset_v3b_prepared"))
        self.prepared_dir_btn = QPushButton("Browse..."); self.prepared_dir_btn.clicked.connect(lambda: browse_existing_directory(self, "Select prepared dataset", self.prepared_dir_input))
        self.checkpoint_input = QLineEdit(self.settings.value("debug_pretrained/checkpoint", "DEAttentionDTA/models/pretrained/DEAttentionDTA.pt"))
        self.checkpoint_btn = QPushButton("Browse..."); self.checkpoint_btn.clicked.connect(lambda: browse_existing_file(self, "Select DEAttentionDTA checkpoint", "PyTorch Checkpoint (*.pt *.ckpt);;All Files (*)", self.checkpoint_input))
        self.results_dir_input = QLineEdit(self.settings.value("debug_pretrained/results_dir", "DEAttentionDTA/outputs/debug/pretrained"))
        self.results_dir_btn = QPushButton("Browse..."); self.results_dir_btn.clicked.connect(lambda: browse_existing_directory(self, "Select output directory", self.results_dir_input))
        self.pretrained_fold_input = QLineEdit(self.settings.value("debug_pretrained/pretrained_fold", "matching")); self.pretrained_fold_input.setPlaceholderText("matching, first or a fold number")
        self.device_combo = QComboBox(); self.device_combo.addItems(["auto", "cpu", "cuda"]); self.device_combo.setCurrentText(self.settings.value("debug_pretrained/device", "auto"))
        self.batch_size_spin = QSpinBox(); self.batch_size_spin.setRange(1, 128); self.batch_size_spin.setValue(int(self.settings.value("debug_pretrained/batch_size", 2)))
        self.non_strict_check = QCheckBox("Allow non-strict checkpoint loading"); self.non_strict_check.setChecked(str(self.settings.value("debug_pretrained/non_strict_pretrained", "false")).lower() in {"true", "1", "yes"})

        form = QFormLayout(); form.addRow(QLabel("<b>Checkpoint loading and forward-pass validation</b>"))
        form.addRow("Prepared dataset:", with_button(self.prepared_dir_input, self.prepared_dir_btn)); form.addRow("Checkpoint:", with_button(self.checkpoint_input, self.checkpoint_btn)); form.addRow("Output directory:", with_button(self.results_dir_input, self.results_dir_btn)); form.addRow("Checkpoint fold:", self.pretrained_fold_input); form.addRow("Device:", self.device_combo); form.addRow("Batch size:", self.batch_size_spin); form.addRow("Compatibility mode:", self.non_strict_check)
        layout = QVBoxLayout(); layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel); buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons); self.setLayout(layout)

    def accept(self) -> None:
        for key, value in self.get_inputs().items(): self.settings.setValue(f"debug_pretrained/{key}", value)
        super().accept()

    def get_inputs(self) -> dict:
        return {"prepared_dir": self.prepared_dir_input.text().strip(), "checkpoint": self.checkpoint_input.text().strip(), "results_dir": self.results_dir_input.text().strip(), "pretrained_fold": self.pretrained_fold_input.text().strip(), "device": self.device_combo.currentText(), "batch_size": self.batch_size_spin.value(), "non_strict_pretrained": self.non_strict_check.isChecked()}
