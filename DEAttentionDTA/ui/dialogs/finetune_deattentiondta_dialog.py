"""Pretrained versus fine-tuned DEAttentionDTA comparison dialog."""

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout

from DEAttentionDTA.ui.dialogs._shared import browse_existing_directory, browse_existing_file, with_button


class FinetuneDEAttentionDTADialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DEAttentionDTA Pretrained vs Fine-Tuned Configuration")
        self.resize(900, 720)
        self.settings = QSettings("ResearchApp", "DEAttentionDTA_Finetune")
        self.prepared_dir_input = QLineEdit(self.settings.value("finetune/prepared_dir", "DEAttentionDTA/data/urv_dataset_v3b_prepared")); self.prepared_dir_btn = QPushButton("Browse..."); self.prepared_dir_btn.clicked.connect(lambda: browse_existing_directory(self, "Select prepared dataset", self.prepared_dir_input))
        self.checkpoint_input = QLineEdit(self.settings.value("finetune/checkpoint", "DEAttentionDTA/models/pretrained/DEAttentionDTA.pt")); self.checkpoint_btn = QPushButton("Browse..."); self.checkpoint_btn.clicked.connect(lambda: browse_existing_file(self, "Select DEAttentionDTA checkpoint", "PyTorch Checkpoint (*.pt *.ckpt);;All Files (*)", self.checkpoint_input))
        self.results_dir_input = QLineEdit(self.settings.value("finetune/results_dir", "DEAttentionDTA/outputs/pretrained_vs_finetuned")); self.results_dir_btn = QPushButton("Browse..."); self.results_dir_btn.clicked.connect(lambda: browse_existing_directory(self, "Select results directory", self.results_dir_input))
        self.models_dir_input = QLineEdit(self.settings.value("finetune/models_dir", "DEAttentionDTA/models/finetuned")); self.models_dir_btn = QPushButton("Browse..."); self.models_dir_btn.clicked.connect(lambda: browse_existing_directory(self, "Select fine-tuned models directory", self.models_dir_input))
        self.splits_input = QLineEdit(self.settings.value("finetune/splits", "all")); self.splits_input.setPlaceholderText("all or 1,3,5")
        self.pretrained_fold_input = QLineEdit(self.settings.value("finetune/pretrained_fold", "matching")); self.pretrained_fold_input.setPlaceholderText("matching, first or a fold number")
        self.device_combo = QComboBox(); self.device_combo.addItems(["auto", "cpu", "cuda"]); self.device_combo.setCurrentText(self.settings.value("finetune/device", "auto"))
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(1, 10000); self.epochs_spin.setValue(int(self.settings.value("finetune/epochs", 150)))
        self.batch_size_spin = QSpinBox(); self.batch_size_spin.setRange(1, 1024); self.batch_size_spin.setValue(int(self.settings.value("finetune/batch_size", 16)))
        self.lr_spin = QDoubleSpinBox(); self.lr_spin.setDecimals(8); self.lr_spin.setRange(1e-8, 1.0); self.lr_spin.setValue(float(self.settings.value("finetune/lr", 0.00005)))
        self.weight_decay_spin = QDoubleSpinBox(); self.weight_decay_spin.setDecimals(8); self.weight_decay_spin.setRange(0.0, 10.0); self.weight_decay_spin.setValue(float(self.settings.value("finetune/weight_decay", 0.0)))
        self.early_stopping_spin = QSpinBox(); self.early_stopping_spin.setRange(1, 10000); self.early_stopping_spin.setValue(int(self.settings.value("finetune/early_stopping_rounds", 25)))
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(0, 999999999); self.seed_spin.setValue(int(self.settings.value("finetune/seed", 990721)))
        self.num_workers_spin = QSpinBox(); self.num_workers_spin.setRange(0, 64); self.num_workers_spin.setValue(int(self.settings.value("finetune/num_workers", 0)))
        self.non_strict_check = QCheckBox("Allow non-strict checkpoint loading"); self.non_strict_check.setChecked(str(self.settings.value("finetune/non_strict_pretrained", "false")).lower() in {"true", "1", "yes"})
        form = QFormLayout(); form.addRow(QLabel("<b>1. Dataset, checkpoint and outputs</b>")); form.addRow("Prepared dataset:", with_button(self.prepared_dir_input, self.prepared_dir_btn)); form.addRow("Checkpoint:", with_button(self.checkpoint_input, self.checkpoint_btn)); form.addRow("Results directory:", with_button(self.results_dir_input, self.results_dir_btn)); form.addRow("Models directory:", with_button(self.models_dir_input, self.models_dir_btn)); form.addRow("Official splits:", self.splits_input); form.addRow("Checkpoint fold:", self.pretrained_fold_input)
        form.addRow(QLabel("<br><b>2. Runtime</b>")); form.addRow("Device:", self.device_combo); form.addRow("Batch size:", self.batch_size_spin); form.addRow("DataLoader workers:", self.num_workers_spin); form.addRow("Random seed:", self.seed_spin); form.addRow("Compatibility mode:", self.non_strict_check)
        form.addRow(QLabel("<br><b>3. Fine-tuning optimization</b>")); form.addRow("Maximum epochs:", self.epochs_spin); form.addRow("Learning rate:", self.lr_spin); form.addRow("Weight decay:", self.weight_decay_spin); form.addRow("Early-stopping rounds:", self.early_stopping_spin)
        layout = QVBoxLayout(); layout.addLayout(form); buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel); buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons); self.setLayout(layout)

    def accept(self) -> None:
        for key, value in self.get_inputs().items(): self.settings.setValue(f"finetune/{key}", value)
        super().accept()

    def get_inputs(self) -> dict:
        return {"prepared_dir": self.prepared_dir_input.text().strip(), "checkpoint": self.checkpoint_input.text().strip(), "results_dir": self.results_dir_input.text().strip(), "models_dir": self.models_dir_input.text().strip(), "splits": self.splits_input.text().strip(), "pretrained_fold": self.pretrained_fold_input.text().strip(), "device": self.device_combo.currentText(), "epochs": self.epochs_spin.value(), "batch_size": self.batch_size_spin.value(), "lr": self.lr_spin.value(), "weight_decay": self.weight_decay_spin.value(), "early_stopping_rounds": self.early_stopping_spin.value(), "seed": self.seed_spin.value(), "num_workers": self.num_workers_spin.value(), "non_strict_pretrained": self.non_strict_check.isChecked()}
