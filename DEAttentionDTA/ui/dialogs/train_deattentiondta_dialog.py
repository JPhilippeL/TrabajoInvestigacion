"""Training dialog for DEAttentionDTA."""

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout

from DEAttentionDTA.ui.dialogs._shared import browse_existing_directory, with_button


class TrainDEAttentionDTADialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DEAttentionDTA Train")
        self.resize(820, 560)
        self.settings = QSettings("ResearchApp", "DEAttentionDTA_TrainOfficialSplits")

        self.prepared_dir_input = QLineEdit(self.settings.value("training/prepared_dir", "DEAttentionDTA/data/urv_dataset_v3b_prepared"))
        self.prepared_dir_btn = QPushButton("Browse...")
        self.prepared_dir_btn.clicked.connect(lambda: browse_existing_directory(self, "Select prepared dataset", self.prepared_dir_input))
        self.output_dir_input = QLineEdit(self.settings.value("training/output_dir", "DEAttentionDTA/outputs/from_scratch"))
        self.output_dir_btn = QPushButton("Browse...")
        self.output_dir_btn.clicked.connect(lambda: browse_existing_directory(self, "Select output directory", self.output_dir_input))

        self.device_combo = QComboBox(); self.device_combo.addItems(["auto", "cpu", "cuda", "cuda:0"]); self.device_combo.setCurrentText(self.settings.value("training/device", "auto"))
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(0, 999999999); self.seed_spin.setValue(int(self.settings.value("training/seed", 990721)))
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(1, 10000); self.epochs_spin.setValue(int(self.settings.value("training/epochs", 100)))
        self.batch_size_spin = QSpinBox(); self.batch_size_spin.setRange(1, 1024); self.batch_size_spin.setValue(int(self.settings.value("training/batch_size", 16)))
        self.lr_spin = QDoubleSpinBox(); self.lr_spin.setDecimals(8); self.lr_spin.setRange(1e-8, 1.0); self.lr_spin.setValue(float(self.settings.value("training/lr", 0.0001)))
        self.weight_decay_spin = QDoubleSpinBox(); self.weight_decay_spin.setDecimals(8); self.weight_decay_spin.setRange(0.0, 10.0); self.weight_decay_spin.setValue(float(self.settings.value("training/weight_decay", 0.0)))
        self.early_stopping_spin = QSpinBox(); self.early_stopping_spin.setRange(1, 10000); self.early_stopping_spin.setValue(int(self.settings.value("training/early_stopping_rounds", 15)))
        self.fold_spin = QSpinBox(); self.fold_spin.setRange(1, 5); self.fold_spin.setValue(int(self.settings.value("training/fold_index", 1)))
        self.use_dataset_folds_check = QCheckBox("Use dataset folds")
        self.use_dataset_folds_check.setChecked(str(self.settings.value("training/use_dataset_folds", "true")).lower() in {"true", "1", "yes"})

        form = QFormLayout()
        form.addRow(QLabel("<b>Dataset and outputs</b>"))
        form.addRow("Prepared dataset root:", with_button(self.prepared_dir_input, self.prepared_dir_btn))
        form.addRow("Output directory:", with_button(self.output_dir_input, self.output_dir_btn))
        form.addRow("Fold index:", self.fold_spin)
        form.addRow("Folds:", self.use_dataset_folds_check)
        form.addRow(QLabel("<br><b>Runtime and optimization</b>"))
        form.addRow("Device:", self.device_combo)
        form.addRow("Seed:", self.seed_spin)
        form.addRow("Epochs:", self.epochs_spin)
        form.addRow("Batch size:", self.batch_size_spin)
        form.addRow("Learning rate:", self.lr_spin)
        form.addRow("Weight decay:", self.weight_decay_spin)
        form.addRow("Patience:", self.early_stopping_spin)

        layout = QVBoxLayout(); layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons); self.setLayout(layout)

    def accept(self) -> None:
        for key, value in self.get_inputs().items(): self.settings.setValue(f"training/{key}", value)
        self.settings.setValue("training/fold_index", self.fold_spin.value())
        self.settings.setValue("training/use_dataset_folds", self.use_dataset_folds_check.isChecked())
        self.settings.setValue("training/output_dir", self.output_dir_input.text().strip())
        super().accept()

    def get_inputs(self) -> dict:
        output_dir = self.output_dir_input.text().strip()
        fold = str(self.fold_spin.value())
        return {
            "prepared_dir": self.prepared_dir_input.text().strip(),
            "output_dir": output_dir,
            "results_dir": output_dir,
            "models_dir": output_dir.rstrip("/") + "/models",
            "splits": fold,
            "fold_index": self.fold_spin.value(),
            "use_dataset_folds": self.use_dataset_folds_check.isChecked(),
            "device": self.device_combo.currentText(),
            "epochs": self.epochs_spin.value(),
            "batch_size": self.batch_size_spin.value(),
            "lr": self.lr_spin.value(),
            "weight_decay": self.weight_decay_spin.value(),
            "early_stopping_rounds": self.early_stopping_spin.value(),
            "seed": self.seed_spin.value(),
            "num_workers": 0,
        }
