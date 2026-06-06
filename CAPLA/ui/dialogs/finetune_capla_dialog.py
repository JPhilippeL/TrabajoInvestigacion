"""Original-pretrained versus fine-tuned CAPLA comparison dialog."""

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout
from CAPLA.ui.dialogs._shared import browse_existing_directory, browse_existing_file, with_button


class FinetuneCAPLADialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CAPLA Pretrained vs Fine-Tuned Configuration"); self.resize(860, 720)
        self.settings = QSettings("ResearchApp", "CAPLA_Finetune")
        self.dataset_dir_input = QLineEdit(self.settings.value("finetune/dataset_dir", "CAPLA/data/urv_dataset_v3b_prepared")); self.dataset_dir_btn = QPushButton("Browse..."); self.dataset_dir_btn.clicked.connect(lambda: browse_existing_directory(self, "Select prepared CAPLA dataset", self.dataset_dir_input))
        self.checkpoint_input = QLineEdit(self.settings.value("finetune/checkpoint_original", "CAPLA/models/pretrained/best_model.pt")); self.checkpoint_btn = QPushButton("Browse..."); self.checkpoint_btn.clicked.connect(lambda: browse_existing_file(self, "Select original CAPLA checkpoint", "PyTorch Checkpoint (*.pt);;All Files (*)", self.checkpoint_input))
        self.results_dir_input = QLineEdit(self.settings.value("finetune/results_dir", "CAPLA/outputs/pretrained_vs_finetuned")); self.results_dir_btn = QPushButton("Browse..."); self.results_dir_btn.clicked.connect(lambda: browse_existing_directory(self, "Select comparison output directory", self.results_dir_input))
        self.models_dir_input = QLineEdit(self.settings.value("finetune/models_dir", "CAPLA/models/finetuned")); self.models_dir_btn = QPushButton("Browse..."); self.models_dir_btn.clicked.connect(lambda: browse_existing_directory(self, "Select fine-tuned-model directory", self.models_dir_input))
        self.splits_input = QLineEdit(self.settings.value("finetune/splits", "all")); self.splits_input.setPlaceholderText("all or 1,3,5")
        self.device_combo = QComboBox(); self.device_combo.addItems(["auto", "cpu", "cuda"]); self.device_combo.setCurrentText(self.settings.value("finetune/device", "auto"))
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(1, 10000); self.epochs_spin.setValue(int(self.settings.value("finetune/epochs", 100)))
        self.batch_size_spin = QSpinBox(); self.batch_size_spin.setRange(1, 1024); self.batch_size_spin.setValue(int(self.settings.value("finetune/batch_size", 32)))
        self.lr_spin = QDoubleSpinBox(); self.lr_spin.setDecimals(8); self.lr_spin.setRange(1e-8, 1.0); self.lr_spin.setValue(float(self.settings.value("finetune/lr", 0.00005)))
        self.weight_decay_spin = QDoubleSpinBox(); self.weight_decay_spin.setDecimals(8); self.weight_decay_spin.setRange(0.0, 10.0); self.weight_decay_spin.setValue(float(self.settings.value("finetune/weight_decay", 0.01)))
        self.early_stopping_spin = QSpinBox(); self.early_stopping_spin.setRange(1, 10000); self.early_stopping_spin.setValue(int(self.settings.value("finetune/early_stopping_rounds", 20)))
        self.min_epochs_spin = QSpinBox(); self.min_epochs_spin.setRange(1, 10000); self.min_epochs_spin.setValue(int(self.settings.value("finetune/min_epochs_before_stopping", 15)))
        self.min_delta_spin = QDoubleSpinBox(); self.min_delta_spin.setDecimals(8); self.min_delta_spin.setRange(0.0, 1000.0); self.min_delta_spin.setValue(float(self.settings.value("finetune/min_delta", 0.0)))
        self.grad_clip_spin = QDoubleSpinBox(); self.grad_clip_spin.setDecimals(6); self.grad_clip_spin.setRange(0.0, 100000.0); self.grad_clip_spin.setValue(float(self.settings.value("finetune/grad_clip_norm", 0.0)))
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(0, 999999999); self.seed_spin.setValue(int(self.settings.value("finetune/seed", 42)))
        self.num_workers_spin = QSpinBox(); self.num_workers_spin.setRange(0, 64); self.num_workers_spin.setValue(int(self.settings.value("finetune/num_workers", 0)))
        form = QFormLayout(); form.addRow(QLabel("<b>1. Dataset, checkpoint and outputs</b>")); form.addRow("Prepared dataset:", with_button(self.dataset_dir_input, self.dataset_dir_btn)); form.addRow("Original checkpoint:", with_button(self.checkpoint_input, self.checkpoint_btn)); form.addRow("Results directory:", with_button(self.results_dir_input, self.results_dir_btn)); form.addRow("Models directory:", with_button(self.models_dir_input, self.models_dir_btn)); form.addRow("Official splits:", self.splits_input); form.addRow(QLabel("<br><b>2. Runtime</b>")); form.addRow("Device:", self.device_combo); form.addRow("Batch size:", self.batch_size_spin); form.addRow("DataLoader workers:", self.num_workers_spin); form.addRow("Random seed:", self.seed_spin); form.addRow(QLabel("<br><b>3. Fine-tuning optimization</b>")); form.addRow("Maximum epochs:", self.epochs_spin); form.addRow("Learning rate:", self.lr_spin); form.addRow("Weight decay:", self.weight_decay_spin); form.addRow("Early-stopping rounds:", self.early_stopping_spin); form.addRow("Minimum epochs before stopping:", self.min_epochs_spin); form.addRow("Minimum improvement:", self.min_delta_spin); form.addRow("Gradient clipping norm:", self.grad_clip_spin)
        layout = QVBoxLayout(); layout.addLayout(form); buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons); self.setLayout(layout)

    def accept(self):
        values = self.get_inputs()
        for key, value in values.items(): self.settings.setValue(f"finetune/{key}", value)
        super().accept()

    def get_inputs(self):
        return {"dataset_dir": self.dataset_dir_input.text().strip(), "checkpoint_original": self.checkpoint_input.text().strip(), "results_dir": self.results_dir_input.text().strip(), "models_dir": self.models_dir_input.text().strip(), "splits": self.splits_input.text().strip(), "device": self.device_combo.currentText(), "epochs": self.epochs_spin.value(), "batch_size": self.batch_size_spin.value(), "lr": self.lr_spin.value(), "weight_decay": self.weight_decay_spin.value(), "early_stopping_rounds": self.early_stopping_spin.value(), "min_epochs_before_stopping": self.min_epochs_spin.value(), "min_delta": self.min_delta_spin.value(), "grad_clip_norm": self.grad_clip_spin.value(), "seed": self.seed_spin.value(), "num_workers": self.num_workers_spin.value()}
