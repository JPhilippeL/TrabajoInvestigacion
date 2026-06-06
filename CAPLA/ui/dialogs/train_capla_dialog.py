"""Official-split from-scratch training dialog for CAPLA."""

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout
from CAPLA.ui.dialogs._shared import browse_existing_directory, with_button


class TrainCAPLADialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CAPLA Official-Split Training Configuration")
        self.resize(840, 680)
        self.settings = QSettings("ResearchApp", "CAPLA_TrainOfficialSplits")

        self.dataset_dir_input = QLineEdit(self.settings.value("training/dataset_dir", "CAPLA/data/urv_dataset_v3b_prepared"))
        self.dataset_dir_btn = QPushButton("Browse..."); self.dataset_dir_btn.clicked.connect(lambda: browse_existing_directory(self, "Select prepared CAPLA dataset", self.dataset_dir_input))
        self.output_dir_input = QLineEdit(self.settings.value("training/output_dir", "CAPLA/outputs/from_scratch"))
        self.output_dir_btn = QPushButton("Browse..."); self.output_dir_btn.clicked.connect(lambda: browse_existing_directory(self, "Select training output directory", self.output_dir_input))
        self.models_dir_input = QLineEdit(self.settings.value("training/models_dir", "CAPLA/models/from_scratch"))
        self.models_dir_btn = QPushButton("Browse..."); self.models_dir_btn.clicked.connect(lambda: browse_existing_directory(self, "Select trained-model directory", self.models_dir_input))
        self.splits_input = QLineEdit(self.settings.value("training/splits", "all")); self.splits_input.setPlaceholderText("all or 1,3,5")

        self.device_combo = QComboBox(); self.device_combo.addItems(["auto", "cpu", "cuda"]); self.device_combo.setCurrentText(self.settings.value("training/device", "auto"))
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(1, 10000); self.epochs_spin.setValue(int(self.settings.value("training/epochs", 150)))
        self.batch_size_spin = QSpinBox(); self.batch_size_spin.setRange(1, 1024); self.batch_size_spin.setValue(int(self.settings.value("training/batch_size", 32)))
        self.lr_spin = QDoubleSpinBox(); self.lr_spin.setDecimals(8); self.lr_spin.setRange(1e-8, 1.0); self.lr_spin.setValue(float(self.settings.value("training/lr", 0.0001)))
        self.weight_decay_spin = QDoubleSpinBox(); self.weight_decay_spin.setDecimals(8); self.weight_decay_spin.setRange(0.0, 10.0); self.weight_decay_spin.setValue(float(self.settings.value("training/weight_decay", 0.01)))
        self.early_stopping_spin = QSpinBox(); self.early_stopping_spin.setRange(1, 10000); self.early_stopping_spin.setValue(int(self.settings.value("training/early_stopping_rounds", 25)))
        self.min_epochs_spin = QSpinBox(); self.min_epochs_spin.setRange(1, 10000); self.min_epochs_spin.setValue(int(self.settings.value("training/min_epochs_before_stopping", 20)))
        self.min_delta_spin = QDoubleSpinBox(); self.min_delta_spin.setDecimals(8); self.min_delta_spin.setRange(0.0, 1000.0); self.min_delta_spin.setValue(float(self.settings.value("training/min_delta", 0.0)))
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(0, 999999999); self.seed_spin.setValue(int(self.settings.value("training/seed", 990721)))
        self.num_workers_spin = QSpinBox(); self.num_workers_spin.setRange(0, 64); self.num_workers_spin.setValue(int(self.settings.value("training/num_workers", 0)))
        self.disable_amp_check = QCheckBox("Disable CUDA mixed precision"); self.disable_amp_check.setChecked(str(self.settings.value("training/disable_amp", "false")).lower() in {"true", "1", "yes"})

        form = QFormLayout(); form.addRow(QLabel("<b>1. Dataset and outputs</b>"))
        form.addRow("Prepared dataset:", with_button(self.dataset_dir_input, self.dataset_dir_btn)); form.addRow("Results directory:", with_button(self.output_dir_input, self.output_dir_btn)); form.addRow("Models directory:", with_button(self.models_dir_input, self.models_dir_btn)); form.addRow("Official splits:", self.splits_input)
        form.addRow(QLabel("<br><b>2. Runtime</b>")); form.addRow("Device:", self.device_combo); form.addRow("Batch size:", self.batch_size_spin); form.addRow("DataLoader workers:", self.num_workers_spin); form.addRow("Random seed:", self.seed_spin); form.addRow("CUDA AMP:", self.disable_amp_check)
        form.addRow(QLabel("<br><b>3. Optimization</b>")); form.addRow("Maximum epochs:", self.epochs_spin); form.addRow("Learning rate:", self.lr_spin); form.addRow("Weight decay:", self.weight_decay_spin); form.addRow("Early-stopping rounds:", self.early_stopping_spin); form.addRow("Minimum epochs before stopping:", self.min_epochs_spin); form.addRow("Minimum improvement:", self.min_delta_spin)
        layout = QVBoxLayout(); layout.addLayout(form); buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons); self.setLayout(layout)

    def accept(self):
        values = self.get_inputs()
        for key, value in values.items(): self.settings.setValue(f"training/{key}", value)
        super().accept()

    def get_inputs(self):
        return {"dataset_dir": self.dataset_dir_input.text().strip(), "output_dir": self.output_dir_input.text().strip(), "models_dir": self.models_dir_input.text().strip(), "splits": self.splits_input.text().strip(), "device": self.device_combo.currentText(), "epochs": self.epochs_spin.value(), "batch_size": self.batch_size_spin.value(), "lr": self.lr_spin.value(), "weight_decay": self.weight_decay_spin.value(), "early_stopping_rounds": self.early_stopping_spin.value(), "min_epochs_before_stopping": self.min_epochs_spin.value(), "min_delta": self.min_delta_spin.value(), "seed": self.seed_spin.value(), "num_workers": self.num_workers_spin.value(), "disable_amp": self.disable_amp_check.isChecked()}
