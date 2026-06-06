"""
@file hyperparameter_search_capla_dialog.py
@author Mohamed EL BOUKHIARI
@brief Hyperparameter-search dialog for CAPLA.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from CAPLA.ui.dialogs._shared import browse_existing_directory, with_button


class HyperparameterSearchCAPLADialog(QDialog):
    """Collect CAPLA HPO parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CAPLA Hyperparameter Search Configuration")
        self.resize(860, 670)
        self.settings = QSettings("ResearchApp", "CAPLA_HyperparameterSearch")

        self.dataset_dir_input = QLineEdit(
            self.settings.value("hpo/dataset_dir", "CAPLA/data/urv_dataset_v3b_prepared")
        )
        self.dataset_dir_btn = QPushButton("Browse...")
        self.dataset_dir_btn.clicked.connect(
            lambda: browse_existing_directory(
                self, "Select prepared CAPLA dataset", self.dataset_dir_input
            )
        )

        self.models_root_input = QLineEdit(
            self.settings.value("hpo/models_root", "CAPLA/models/hpo")
        )
        self.models_root_btn = QPushButton("Browse...")
        self.models_root_btn.clicked.connect(
            lambda: browse_existing_directory(
                self, "Select CAPLA HPO models directory", self.models_root_input
            )
        )

        self.results_root_input = QLineEdit(
            self.settings.value("hpo/results_root", "CAPLA/outputs/hpo")
        )
        self.results_root_btn = QPushButton("Browse...")
        self.results_root_btn.clicked.connect(
            lambda: browse_existing_directory(
                self, "Select CAPLA HPO results directory", self.results_root_input
            )
        )

        self.splits_input = QLineEdit(self.settings.value("hpo/splits", "1"))
        self.splits_input.setPlaceholderText("Tuning splits only, e.g. 1 or 1,2")

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda"])
        self.device_combo.setCurrentText(self.settings.value("hpo/device", "auto"))

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(int(self.settings.value("hpo/epochs", 50)))

        self.early_stopping_spin = QSpinBox()
        self.early_stopping_spin.setRange(1, 10000)
        self.early_stopping_spin.setValue(
            int(self.settings.value("hpo/early_stopping_rounds", 10))
        )

        self.min_epochs_spin = QSpinBox()
        self.min_epochs_spin.setRange(1, 10000)
        self.min_epochs_spin.setValue(
            int(self.settings.value("hpo/min_epochs_before_stopping", 5))
        )

        self.min_delta_spin = QDoubleSpinBox()
        self.min_delta_spin.setDecimals(8)
        self.min_delta_spin.setRange(0.0, 1000.0)
        self.min_delta_spin.setValue(float(self.settings.value("hpo/min_delta", 0.0)))

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999999)
        self.seed_spin.setValue(int(self.settings.value("hpo/seed", 42)))

        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setRange(0, 64)
        self.num_workers_spin.setValue(int(self.settings.value("hpo/num_workers", 0)))

        self.disable_amp_check = QCheckBox("Disable CUDA mixed precision")
        self.disable_amp_check.setChecked(
            str(self.settings.value("hpo/disable_amp", "false")).lower()
            in {"true", "1", "yes"}
        )

        self.lr_values_input = QLineEdit(
            self.settings.value("hpo/lr_values", "0.00005,0.0001")
        )
        self.batch_size_values_input = QLineEdit(
            self.settings.value("hpo/batch_size_values", "8,16")
        )
        self.weight_decay_values_input = QLineEdit(
            self.settings.value("hpo/weight_decay_values", "0.0,0.01")
        )

        form = QFormLayout()
        form.addRow(QLabel("<b>1. Dataset and outputs</b>"))
        form.addRow("Prepared dataset:", with_button(self.dataset_dir_input, self.dataset_dir_btn))
        form.addRow("Trial models root:", with_button(self.models_root_input, self.models_root_btn))
        form.addRow("Results root:", with_button(self.results_root_input, self.results_root_btn))
        form.addRow("Tuning split(s):", self.splits_input)

        form.addRow(QLabel("<br><b>2. Runtime</b>"))
        form.addRow("Device:", self.device_combo)
        form.addRow("Maximum epochs per trial:", self.epochs_spin)
        form.addRow("Early-stopping rounds:", self.early_stopping_spin)
        form.addRow("Minimum epochs before stopping:", self.min_epochs_spin)
        form.addRow("Minimum improvement:", self.min_delta_spin)
        form.addRow("DataLoader workers:", self.num_workers_spin)
        form.addRow("Random seed:", self.seed_spin)
        form.addRow("CUDA AMP:", self.disable_amp_check)

        form.addRow(QLabel("<br><b>3. Search space</b>"))
        form.addRow("Learning rates:", self.lr_values_input)
        form.addRow("Batch sizes:", self.batch_size_values_input)
        form.addRow("Weight decays:", self.weight_decay_values_input)
        form.addRow("Selection rule:", QLabel("min mean validation RMSE, then max mean validation Pearson"))
        form.addRow("Methodology:", QLabel("Official test subsets are never used during HPO."))

        layout = QVBoxLayout()
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)

    @staticmethod
    def _parse_float_list(raw: str) -> list[float]:
        return [float(item.strip()) for item in raw.split(",") if item.strip()]

    @staticmethod
    def _parse_int_list(raw: str) -> list[int]:
        return [int(item.strip()) for item in raw.split(",") if item.strip()]

    def accept(self) -> None:
        values = self.get_inputs()
        for key, value in values.items():
            self.settings.setValue(f"hpo/{key}", value)
        super().accept()

    def get_inputs(self) -> dict:
        return {
            "dataset_dir": self.dataset_dir_input.text().strip(),
            "models_root": self.models_root_input.text().strip(),
            "results_root": self.results_root_input.text().strip(),
            "splits": self.splits_input.text().strip(),
            "device": self.device_combo.currentText(),
            "epochs": self.epochs_spin.value(),
            "early_stopping_rounds": self.early_stopping_spin.value(),
            "min_epochs_before_stopping": self.min_epochs_spin.value(),
            "min_delta": self.min_delta_spin.value(),
            "seed": self.seed_spin.value(),
            "num_workers": self.num_workers_spin.value(),
            "disable_amp": self.disable_amp_check.isChecked(),
            "lr_values": self._parse_float_list(self.lr_values_input.text()),
            "batch_size_values": self._parse_int_list(self.batch_size_values_input.text()),
            "weight_decay_values": self._parse_float_list(self.weight_decay_values_input.text()),
        }
