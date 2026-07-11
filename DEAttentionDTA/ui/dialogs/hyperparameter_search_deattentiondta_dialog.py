"""Search dialog for DEAttentionDTA hyperparameter tuning."""

from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout

from DEAttentionDTA.ui.dialogs._shared import browse_existing_directory, with_button


class HyperparameterSearchDEAttentionDTADialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DEAttentionDTA Search")
        self.resize(860, 620)
        self.settings = QSettings("ResearchApp", "DEAttentionDTA_HyperparameterSearch")

        self.prepared_dir_input = QLineEdit(self._settings_text("hpo/prepared_dir", "DEAttentionDTA/data/urv_dataset_v3b_prepared"))
        self.prepared_dir_btn = QPushButton("Browse...")
        self.prepared_dir_btn.clicked.connect(lambda: browse_existing_directory(self, "Select prepared dataset", self.prepared_dir_input))
        self.output_dir_input = QLineEdit(self._settings_text("hpo/output_dir", "DEAttentionDTA/outputs/hpo"))
        self.output_dir_btn = QPushButton("Browse...")
        self.output_dir_btn.clicked.connect(lambda: browse_existing_directory(self, "Select search output root", self.output_dir_input))

        self.fold_spin = QSpinBox(); self.fold_spin.setRange(1, 5); self.fold_spin.setValue(int(self.settings.value("hpo/fold_index", 1)))
        self.use_dataset_folds_check = QCheckBox("Use dataset folds")
        self.use_dataset_folds_check.setChecked(str(self.settings.value("hpo/use_dataset_folds", "true")).lower() in {"true", "1", "yes"})
        self.device_combo = QComboBox(); self.device_combo.addItems(["auto", "cpu", "cuda", "cuda:0"]); self.device_combo.setCurrentText(self._settings_text("hpo/device", "auto"))
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(0, 999999999); self.seed_spin.setValue(int(self.settings.value("hpo/seed", 42)))
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(1, 10000); self.epochs_spin.setValue(int(self.settings.value("hpo/epochs", 50)))
        self.early_stopping_spin = QSpinBox(); self.early_stopping_spin.setRange(1, 10000); self.early_stopping_spin.setValue(int(self.settings.value("hpo/early_stopping_rounds", 10)))
        self.lr_values_input = QLineEdit(self._settings_text("hpo/lr_values", "0.00001,0.00005,0.0001"))
        self.batch_size_values_input = QLineEdit(self._settings_text("hpo/batch_sizes", "8,16"))
        self.weight_decay_values_input = QLineEdit(self._settings_text("hpo/weight_decays", "0,0.0001,0.001,0.01"))

        form = QFormLayout()
        form.addRow(QLabel("<b>Dataset and outputs</b>"))
        form.addRow("Prepared dataset root:", with_button(self.prepared_dir_input, self.prepared_dir_btn))
        form.addRow("Output root:", with_button(self.output_dir_input, self.output_dir_btn))
        form.addRow("Fold index:", self.fold_spin)
        form.addRow("Folds:", self.use_dataset_folds_check)
        form.addRow(QLabel("<br><b>Runtime</b>"))
        form.addRow("Device:", self.device_combo)
        form.addRow("Seed:", self.seed_spin)
        form.addRow("Epochs:", self.epochs_spin)
        form.addRow(QLabel("<br><b>Search space</b>"))
        form.addRow("Learning-rate values:", self.lr_values_input)
        form.addRow("Batch-size values:", self.batch_size_values_input)
        form.addRow("Weight-decay values:", self.weight_decay_values_input)
        form.addRow("Patience:", self.early_stopping_spin)
        form.addRow("Selection rule:", QLabel("Validation RMSE first, validation Pearson as tie-breaker."))

        layout = QVBoxLayout(); layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons); self.setLayout(layout)

    @staticmethod
    def _parse_float_list(raw: str) -> list[float]:
        return [float(item.strip()) for item in raw.split(",") if item.strip()]

    @staticmethod
    def _parse_int_list(raw: str) -> list[int]:
        return [int(item.strip()) for item in raw.split(",") if item.strip()]

    def _settings_text(self, key: str, default: str) -> str:
        value = self.settings.value(key, default)
        if value is None:
            return str(default)
        if isinstance(value, (list, tuple)):
            return ",".join(str(item) for item in value)
        return str(value)

    def accept(self) -> None:
        self.settings.setValue("hpo/prepared_dir", self.prepared_dir_input.text().strip())
        self.settings.setValue("hpo/output_dir", self.output_dir_input.text().strip())
        self.settings.setValue("hpo/fold_index", self.fold_spin.value())
        self.settings.setValue("hpo/use_dataset_folds", self.use_dataset_folds_check.isChecked())
        self.settings.setValue("hpo/device", self.device_combo.currentText())
        self.settings.setValue("hpo/epochs", self.epochs_spin.value())
        self.settings.setValue("hpo/early_stopping_rounds", self.early_stopping_spin.value())
        self.settings.setValue("hpo/seed", self.seed_spin.value())
        self.settings.setValue("hpo/lr_values", self.lr_values_input.text().strip())
        self.settings.setValue("hpo/batch_sizes", self.batch_size_values_input.text().strip())
        self.settings.setValue("hpo/weight_decays", self.weight_decay_values_input.text().strip())
        super().accept()

    def get_inputs(self) -> dict:
        output_dir = self.output_dir_input.text().strip()
        fold = str(self.fold_spin.value())
        return {
            "prepared_dir": self.prepared_dir_input.text().strip(),
            "output_dir": output_dir,
            "models_root": output_dir.rstrip("/") + "/models",
            "results_root": output_dir,
            "splits": fold,
            "fold_index": self.fold_spin.value(),
            "use_dataset_folds": self.use_dataset_folds_check.isChecked(),
            "device": self.device_combo.currentText(),
            "epochs": self.epochs_spin.value(),
            "early_stopping_rounds": self.early_stopping_spin.value(),
            "min_epochs_before_stopping": 1,
            "min_delta": 0.0,
            "grad_clip_norm": 0.0,
            "num_workers": 0,
            "seed": self.seed_spin.value(),
            "lr_values": self._parse_float_list(self.lr_values_input.text().strip()),
            "batch_size_values": self._parse_int_list(self.batch_size_values_input.text().strip()),
            "weight_decay_values": self._parse_float_list(self.weight_decay_values_input.text().strip()),
        }
