import logging

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

logger = logging.getLogger(__name__)


class HyperparameterSearchDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("PLANET Hyperparameter Tuning")
        self.resize(760, 560)

        self.settings = QSettings("Investigacion", "PLANETHyperparameterSearch")

        self.data_output_path_input = QLineEdit()
        self.data_output_path_input.setPlaceholderText(
            "PLANET data directory containing metadata/pkl/train.pkl, valid.pkl, core.pkl"
        )
        self.data_output_path_input.setText(
            str(self.settings.value("hpplanet/data_output_path", ""))
        )
        self.data_output_path_btn = QPushButton("Select...")
        self.data_output_path_btn.clicked.connect(self.browse_data_output_path)

        self.save_directory_input = QLineEdit()
        self.save_directory_input.setPlaceholderText("Save directory")
        self.save_directory_input.setText(str(self.settings.value("hpplanet/save_directory", "")))
        self.save_directory_btn = QPushButton("Select...")
        self.save_directory_btn.clicked.connect(self.browse_save_directory)

        self.cpu_per_trials_input = QSpinBox()
        self.cpu_per_trials_input.setRange(1, 100)
        self.cpu_per_trials_input.setValue(int(self.settings.value("hpplanet/cpu_per_trials", 5)))

        self.gpu_per_trials_input = QDoubleSpinBox()
        self.gpu_per_trials_input.setDecimals(2)
        self.gpu_per_trials_input.setRange(0.0, 100.0)
        self.gpu_per_trials_input.setSingleStep(0.25)
        self.gpu_per_trials_input.setValue(
            float(self.settings.value("hpplanet/gpu_per_trials", 0.0))
        )

        self.number_of_trials = QSpinBox()
        self.number_of_trials.setRange(1, 1000)
        self.number_of_trials.setValue(int(self.settings.value("hpplanet/number_of_trials", 20)))

        self.batch_size_min = QSpinBox()
        self.batch_size_min.setRange(1, 64)
        self.batch_size_min.setValue(int(self.settings.value("hpplanet/batch_size_min", 1)))

        self.batch_size_max = QSpinBox()
        self.batch_size_max.setRange(1, 64)
        self.batch_size_max.setValue(int(self.settings.value("hpplanet/batch_size_max", 2)))

        self.lr_min = QDoubleSpinBox()
        self.lr_min.setDecimals(8)
        self.lr_min.setRange(1e-8, 1.0)
        self.lr_min.setSingleStep(1e-5)
        self.lr_min.setValue(float(self.settings.value("hpplanet/lr_min", 1e-5)))

        self.lr_max = QDoubleSpinBox()
        self.lr_max.setDecimals(8)
        self.lr_max.setRange(1e-8, 1.0)
        self.lr_max.setSingleStep(1e-5)
        self.lr_max.setValue(float(self.settings.value("hpplanet/lr_max", 1e-4)))

        self.weight_decay_min = QDoubleSpinBox()
        self.weight_decay_min.setDecimals(10)
        self.weight_decay_min.setRange(1e-10, 1.0)
        self.weight_decay_min.setSingleStep(1e-6)
        self.weight_decay_min.setValue(
            float(self.settings.value("hpplanet/weight_decay_min", 1e-8))
        )

        self.weight_decay_max = QDoubleSpinBox()
        self.weight_decay_max.setDecimals(10)
        self.weight_decay_max.setRange(1e-10, 1.0)
        self.weight_decay_max.setSingleStep(1e-6)
        self.weight_decay_max.setValue(
            float(self.settings.value("hpplanet/weight_decay_max", 1e-4))
        )

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 10000)
        self.epochs.setValue(int(self.settings.value("hpplanet/epochs", 30)))

        self.patience = QSpinBox()
        self.patience.setRange(1, 10000)
        self.patience.setValue(int(self.settings.value("hpplanet/patience", 8)))

        self.seed = QSpinBox()
        self.seed.setRange(0, 1_000_000)
        self.seed.setValue(int(self.settings.value("hpplanet/seed", 42)))

        self.num_workers = QSpinBox()
        self.num_workers.setRange(0, 64)
        self.num_workers.setValue(int(self.settings.value("hpplanet/num_workers", 0)))

        self.feature_dims_min = QSpinBox()
        self.feature_dims_min.setRange(1, 4096)
        self.feature_dims_min.setValue(int(self.settings.value("hpplanet/feature_dims_min", 300)))

        self.feature_dims_max = QSpinBox()
        self.feature_dims_max.setRange(1, 4096)
        self.feature_dims_max.setValue(int(self.settings.value("hpplanet/feature_dims_max", 300)))

        self.nheads_min = QSpinBox()
        self.nheads_min.setRange(1, 64)
        self.nheads_min.setValue(int(self.settings.value("hpplanet/nheads_min", 4)))

        self.nheads_max = QSpinBox()
        self.nheads_max.setRange(1, 64)
        self.nheads_max.setValue(int(self.settings.value("hpplanet/nheads_max", 8)))

        self.key_dims_min = QSpinBox()
        self.key_dims_min.setRange(1, 4096)
        self.key_dims_min.setValue(int(self.settings.value("hpplanet/key_dims_min", 128)))

        self.key_dims_max = QSpinBox()
        self.key_dims_max.setRange(1, 4096)
        self.key_dims_max.setValue(int(self.settings.value("hpplanet/key_dims_max", 300)))

        self.value_dims_min = QSpinBox()
        self.value_dims_min.setRange(1, 4096)
        self.value_dims_min.setValue(int(self.settings.value("hpplanet/value_dims_min", 128)))

        self.value_dims_max = QSpinBox()
        self.value_dims_max.setRange(1, 4096)
        self.value_dims_max.setValue(int(self.settings.value("hpplanet/value_dims_max", 300)))

        self.pro_update_inters_min = QSpinBox()
        self.pro_update_inters_min.setRange(1, 100)
        self.pro_update_inters_min.setValue(
            int(self.settings.value("hpplanet/pro_update_inters_min", 2))
        )

        self.pro_update_inters_max = QSpinBox()
        self.pro_update_inters_max.setRange(1, 100)
        self.pro_update_inters_max.setValue(
            int(self.settings.value("hpplanet/pro_update_inters_max", 3))
        )

        self.lig_update_iters_min = QSpinBox()
        self.lig_update_iters_min.setRange(1, 100)
        self.lig_update_iters_min.setValue(
            int(self.settings.value("hpplanet/lig_update_iters_min", 5))
        )

        self.lig_update_iters_max = QSpinBox()
        self.lig_update_iters_max.setRange(1, 100)
        self.lig_update_iters_max.setValue(
            int(self.settings.value("hpplanet/lig_update_iters_max", 10))
        )

        self.pro_lig_update_iters_min = QSpinBox()
        self.pro_lig_update_iters_min.setRange(1, 100)
        self.pro_lig_update_iters_min.setValue(
            int(self.settings.value("hpplanet/pro_lig_update_iters_min", 1))
        )

        self.pro_lig_update_iters_max = QSpinBox()
        self.pro_lig_update_iters_max.setRange(1, 100)
        self.pro_lig_update_iters_max.setValue(
            int(self.settings.value("hpplanet/pro_lig_update_iters_max", 2))
        )

        self.clip_norm_min = QDoubleSpinBox()
        self.clip_norm_min.setDecimals(4)
        self.clip_norm_min.setRange(1.0, 100000.0)
        self.clip_norm_min.setSingleStep(10.0)
        self.clip_norm_min.setValue(float(self.settings.value("hpplanet/clip_norm_min", 100.0)))

        self.clip_norm_max = QDoubleSpinBox()
        self.clip_norm_max.setDecimals(4)
        self.clip_norm_max.setRange(1.0, 100000.0)
        self.clip_norm_max.setSingleStep(10.0)
        self.clip_norm_max.setValue(float(self.settings.value("hpplanet/clip_norm_max", 200.0)))

        self.beta_start_step_min = QSpinBox()
        self.beta_start_step_min.setRange(0, 1_000_000)
        self.beta_start_step_min.setValue(
            int(self.settings.value("hpplanet/beta_start_step_min", 0))
        )

        self.beta_start_step_max = QSpinBox()
        self.beta_start_step_max.setRange(0, 1_000_000)
        self.beta_start_step_max.setValue(
            int(self.settings.value("hpplanet/beta_start_step_max", 500))
        )

        self.beta_start_step_step = QSpinBox()
        self.beta_start_step_step.setRange(1, 1_000_000)
        self.beta_start_step_step.setValue(
            int(self.settings.value("hpplanet/beta_start_step_step", 250))
        )

        self._connect_min_max()

        main_layout = QVBoxLayout()
        form_layout = QGridLayout()

        row = 0

        form_layout.addWidget(QLabel("PLANET data dir:"), row, 0)
        form_layout.addWidget(self.data_output_path_input, row, 1, 1, 2)
        form_layout.addWidget(self.data_output_path_btn, row, 3)

        row += 1
        form_layout.addWidget(QLabel("Save directory:"), row, 0)
        form_layout.addWidget(self.save_directory_input, row, 1, 1, 2)
        form_layout.addWidget(self.save_directory_btn, row, 3)

        row += 1
        form_layout.addWidget(QLabel("CPU per trial:"), row, 0)
        form_layout.addWidget(self.cpu_per_trials_input, row, 1)
        form_layout.addWidget(QLabel("GPU per trial:"), row, 2)
        form_layout.addWidget(self.gpu_per_trials_input, row, 3)

        row += 1
        form_layout.addWidget(QLabel("Number of trials:"), row, 0)
        form_layout.addWidget(self.number_of_trials, row, 1)

        row += 1
        form_layout.addWidget(QLabel("Batch size:"), row, 0)
        form_layout.addLayout(
            self._min_max_layout(self.batch_size_min, self.batch_size_max),
            row,
            1,
            1,
            3,
        )

        row += 1
        form_layout.addWidget(QLabel("Learning rate:"), row, 0)
        form_layout.addLayout(
            self._min_max_layout(self.lr_min, self.lr_max),
            row,
            1,
            1,
            3,
        )

        row += 1
        form_layout.addWidget(QLabel("Weight decay:"), row, 0)
        form_layout.addLayout(
            self._min_max_layout(self.weight_decay_min, self.weight_decay_max),
            row,
            1,
            1,
            3,
        )

        row += 1
        form_layout.addWidget(QLabel("Epochs:"), row, 0)
        form_layout.addWidget(self.epochs, row, 1)
        form_layout.addWidget(QLabel("Patience:"), row, 2)
        form_layout.addWidget(self.patience, row, 3)

        row += 1
        form_layout.addWidget(QLabel("Seed:"), row, 0)
        form_layout.addWidget(self.seed, row, 1)
        form_layout.addWidget(QLabel("Num workers:"), row, 2)
        form_layout.addWidget(self.num_workers, row, 3)

        row += 1
        form_layout.addWidget(QLabel("Feature dims:"), row, 0)
        form_layout.addLayout(
            self._min_max_layout(self.feature_dims_min, self.feature_dims_max),
            row,
            1,
            1,
            3,
        )

        row += 1
        form_layout.addWidget(QLabel("Attention heads:"), row, 0)
        form_layout.addLayout(
            self._min_max_layout(self.nheads_min, self.nheads_max),
            row,
            1,
            1,
            3,
        )

        row += 1
        form_layout.addWidget(QLabel("Key dims:"), row, 0)
        form_layout.addLayout(
            self._min_max_layout(self.key_dims_min, self.key_dims_max),
            row,
            1,
            1,
            3,
        )

        row += 1
        form_layout.addWidget(QLabel("Value dims:"), row, 0)
        form_layout.addLayout(
            self._min_max_layout(self.value_dims_min, self.value_dims_max),
            row,
            1,
            1,
            3,
        )

        row += 1
        form_layout.addWidget(QLabel("Protein updates:"), row, 0)
        form_layout.addLayout(
            self._min_max_layout(
                self.pro_update_inters_min,
                self.pro_update_inters_max,
            ),
            row,
            1,
            1,
            3,
        )

        row += 1
        form_layout.addWidget(QLabel("Ligand updates:"), row, 0)
        form_layout.addLayout(
            self._min_max_layout(
                self.lig_update_iters_min,
                self.lig_update_iters_max,
            ),
            row,
            1,
            1,
            3,
        )

        row += 1
        form_layout.addWidget(QLabel("Protein-ligand updates:"), row, 0)
        form_layout.addLayout(
            self._min_max_layout(
                self.pro_lig_update_iters_min,
                self.pro_lig_update_iters_max,
            ),
            row,
            1,
            1,
            3,
        )

        row += 1
        form_layout.addWidget(QLabel("Clip norm:"), row, 0)
        form_layout.addLayout(
            self._min_max_layout(self.clip_norm_min, self.clip_norm_max),
            row,
            1,
            1,
            3,
        )

        row += 1
        form_layout.addWidget(QLabel("Beta start step:"), row, 0)
        form_layout.addLayout(
            self._min_max_layout(
                self.beta_start_step_min,
                self.beta_start_step_max,
            ),
            row,
            1,
            1,
            2,
        )
        form_layout.addWidget(QLabel("Step:"), row, 3)

        row += 1
        form_layout.addWidget(QLabel("Beta step value:"), row, 0)
        form_layout.addWidget(self.beta_start_step_step, row, 1)

        main_layout.addLayout(form_layout)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        main_layout.addWidget(self.button_box)

        self.setLayout(main_layout)

    def _connect_min_max(self):
        pairs = [
            (self.batch_size_min, self.batch_size_max),
            (self.lr_min, self.lr_max),
            (self.weight_decay_min, self.weight_decay_max),
            (self.feature_dims_min, self.feature_dims_max),
            (self.nheads_min, self.nheads_max),
            (self.key_dims_min, self.key_dims_max),
            (self.value_dims_min, self.value_dims_max),
            (self.pro_update_inters_min, self.pro_update_inters_max),
            (self.lig_update_iters_min, self.lig_update_iters_max),
            (self.pro_lig_update_iters_min, self.pro_lig_update_iters_max),
            (self.clip_norm_min, self.clip_norm_max),
            (self.beta_start_step_min, self.beta_start_step_max),
        ]

        for min_widget, max_widget in pairs:
            min_widget.valueChanged.connect(
                lambda value, max_widget=max_widget: self._ensure_min_max(
                    value,
                    max_widget,
                )
            )

    def _min_max_layout(self, min_widget, max_widget):
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Min:"))
        layout.addWidget(min_widget)
        layout.addWidget(QLabel("Max:"))
        layout.addWidget(max_widget)
        return layout

    def _ensure_min_max(self, min_value, max_widget):
        if min_value > max_widget.value():
            max_widget.setValue(min_value)

    def browse_data_output_path(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select PLANET data directory",
            "",
        )

        if directory:
            self.data_output_path_input.setText(directory)

    def browse_save_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select save directory",
            "",
        )

        if directory:
            self.save_directory_input.setText(directory)

    def accept(self):
        if not self._validate_inputs():
            return

        self.save_settings()
        super().accept()

    def _validate_inputs(self):
        required_fields = [
            (self.data_output_path_input, "PLANET data directory"),
            (self.save_directory_input, "Save directory"),
        ]

        for widget, name in required_fields:
            if not widget.text().strip():
                QMessageBox.warning(
                    self,
                    "Missing field",
                    f"{name} is required.",
                )
                return False

        ranges = [
            (self.batch_size_min.value(), self.batch_size_max.value(), "batch_size"),
            (self.lr_min.value(), self.lr_max.value(), "learning rate"),
            (
                self.weight_decay_min.value(),
                self.weight_decay_max.value(),
                "weight decay",
            ),
            (
                self.feature_dims_min.value(),
                self.feature_dims_max.value(),
                "feature_dims",
            ),
            (self.nheads_min.value(), self.nheads_max.value(), "nheads"),
            (self.key_dims_min.value(), self.key_dims_max.value(), "key_dims"),
            (self.value_dims_min.value(), self.value_dims_max.value(), "value_dims"),
            (
                self.pro_update_inters_min.value(),
                self.pro_update_inters_max.value(),
                "pro_update_inters",
            ),
            (
                self.lig_update_iters_min.value(),
                self.lig_update_iters_max.value(),
                "lig_update_iters",
            ),
            (
                self.pro_lig_update_iters_min.value(),
                self.pro_lig_update_iters_max.value(),
                "pro_lig_update_iters",
            ),
            (self.clip_norm_min.value(), self.clip_norm_max.value(), "clip_norm"),
            (
                self.beta_start_step_min.value(),
                self.beta_start_step_max.value(),
                "beta_start_step",
            ),
        ]

        for min_value, max_value, name in ranges:
            if min_value > max_value:
                QMessageBox.warning(
                    self,
                    "Invalid range",
                    f"{name} min must be <= {name} max.",
                )
                return False

        return True

    def save_settings(self):
        self.settings.setValue(
            "hpplanet/data_output_path",
            self.data_output_path_input.text().strip(),
        )
        self.settings.setValue(
            "hpplanet/save_directory",
            self.save_directory_input.text().strip(),
        )

        self.settings.setValue(
            "hpplanet/cpu_per_trials",
            self.cpu_per_trials_input.value(),
        )
        self.settings.setValue(
            "hpplanet/gpu_per_trials",
            self.gpu_per_trials_input.value(),
        )
        self.settings.setValue(
            "hpplanet/number_of_trials",
            self.number_of_trials.value(),
        )

        self.settings.setValue("hpplanet/batch_size_min", self.batch_size_min.value())
        self.settings.setValue("hpplanet/batch_size_max", self.batch_size_max.value())

        self.settings.setValue("hpplanet/lr_min", self.lr_min.value())
        self.settings.setValue("hpplanet/lr_max", self.lr_max.value())

        self.settings.setValue(
            "hpplanet/weight_decay_min",
            self.weight_decay_min.value(),
        )
        self.settings.setValue(
            "hpplanet/weight_decay_max",
            self.weight_decay_max.value(),
        )

        self.settings.setValue("hpplanet/epochs", self.epochs.value())
        self.settings.setValue("hpplanet/patience", self.patience.value())
        self.settings.setValue("hpplanet/seed", self.seed.value())
        self.settings.setValue("hpplanet/num_workers", self.num_workers.value())

        self.settings.setValue(
            "hpplanet/feature_dims_min",
            self.feature_dims_min.value(),
        )
        self.settings.setValue(
            "hpplanet/feature_dims_max",
            self.feature_dims_max.value(),
        )

        self.settings.setValue("hpplanet/nheads_min", self.nheads_min.value())
        self.settings.setValue("hpplanet/nheads_max", self.nheads_max.value())

        self.settings.setValue("hpplanet/key_dims_min", self.key_dims_min.value())
        self.settings.setValue("hpplanet/key_dims_max", self.key_dims_max.value())

        self.settings.setValue("hpplanet/value_dims_min", self.value_dims_min.value())
        self.settings.setValue("hpplanet/value_dims_max", self.value_dims_max.value())

        self.settings.setValue(
            "hpplanet/pro_update_inters_min",
            self.pro_update_inters_min.value(),
        )
        self.settings.setValue(
            "hpplanet/pro_update_inters_max",
            self.pro_update_inters_max.value(),
        )

        self.settings.setValue(
            "hpplanet/lig_update_iters_min",
            self.lig_update_iters_min.value(),
        )
        self.settings.setValue(
            "hpplanet/lig_update_iters_max",
            self.lig_update_iters_max.value(),
        )

        self.settings.setValue(
            "hpplanet/pro_lig_update_iters_min",
            self.pro_lig_update_iters_min.value(),
        )
        self.settings.setValue(
            "hpplanet/pro_lig_update_iters_max",
            self.pro_lig_update_iters_max.value(),
        )

        self.settings.setValue("hpplanet/clip_norm_min", self.clip_norm_min.value())
        self.settings.setValue("hpplanet/clip_norm_max", self.clip_norm_max.value())

        self.settings.setValue(
            "hpplanet/beta_start_step_min",
            self.beta_start_step_min.value(),
        )
        self.settings.setValue(
            "hpplanet/beta_start_step_max",
            self.beta_start_step_max.value(),
        )
        self.settings.setValue(
            "hpplanet/beta_start_step_step",
            self.beta_start_step_step.value(),
        )

    def get_values(self):
        return {
            "data_output_path": self.data_output_path_input.text().strip(),
            "save_directory": self.save_directory_input.text().strip(),
            "cpu_per_trials": self.cpu_per_trials_input.value(),
            "gpu_per_trials": self.gpu_per_trials_input.value(),
            "number_of_trials": self.number_of_trials.value(),
            "batch_size_min": self.batch_size_min.value(),
            "batch_size_max": self.batch_size_max.value(),
            "lr_min": self.lr_min.value(),
            "lr_max": self.lr_max.value(),
            "weight_decay_min": self.weight_decay_min.value(),
            "weight_decay_max": self.weight_decay_max.value(),
            "EPOCHS": self.epochs.value(),
            "PATIENCE": self.patience.value(),
            "seed": self.seed.value(),
            "num_workers": self.num_workers.value(),
            "feature_dims_min": self.feature_dims_min.value(),
            "feature_dims_max": self.feature_dims_max.value(),
            "nheads_min": self.nheads_min.value(),
            "nheads_max": self.nheads_max.value(),
            "key_dims_min": self.key_dims_min.value(),
            "key_dims_max": self.key_dims_max.value(),
            "value_dims_min": self.value_dims_min.value(),
            "value_dims_max": self.value_dims_max.value(),
            "pro_update_inters_min": self.pro_update_inters_min.value(),
            "pro_update_inters_max": self.pro_update_inters_max.value(),
            "lig_update_iters_min": self.lig_update_iters_min.value(),
            "lig_update_iters_max": self.lig_update_iters_max.value(),
            "pro_lig_update_iters_min": self.pro_lig_update_iters_min.value(),
            "pro_lig_update_iters_max": self.pro_lig_update_iters_max.value(),
            "clip_norm_min": self.clip_norm_min.value(),
            "clip_norm_max": self.clip_norm_max.value(),
            "beta_start_step_min": self.beta_start_step_min.value(),
            "beta_start_step_max": self.beta_start_step_max.value(),
            "beta_start_step_step": self.beta_start_step_step.value(),
        }
