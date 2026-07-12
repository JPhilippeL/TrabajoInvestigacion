import logging
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

logger = logging.getLogger(__name__)


MODEL_NAMES = [
    "DeepDTAF",
    "DeepDTAF_NoPocket",
    "DeepDTAF_NoProtein",
    "DeepDTAF_OnlyLigand",
    "DeepDTAF_GNN",
    "DeepDTAF_GNN_NoPocket",
    "DeepDTAF_GNN_NoProtein",
    "DeepDTAF_GNN_OnlyLigand",
]


class DeepDTAFHyperparameterSearchDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("DeepDTAF Hyperparameter Tuning")
        self.resize(760, 680)

        self.settings = QSettings(
            "Investigacion",
            "DeepDTAFHyperparameterSearch",
        )

        settings_prefix = "hpdeepdtaf"

        self.data_directory_input = QLineEdit()
        self.data_directory_input.setPlaceholderText("Directory containing the split folders")
        self.data_directory_input.setText(
            str(
                self.settings.value(
                    f"{settings_prefix}/data_directory",
                    "",
                )
            )
        )

        self.data_directory_button = QPushButton("Select...")
        self.data_directory_button.clicked.connect(self.browse_data_directory)

        self.model_input = QComboBox()
        self.model_input.addItems(MODEL_NAMES)

        saved_model = str(
            self.settings.value(
                f"{settings_prefix}/model_name",
                MODEL_NAMES[0],
            )
        )

        model_index = self.model_input.findText(saved_model)

        if model_index >= 0:
            self.model_input.setCurrentIndex(model_index)

        self.output_directory_input = QLineEdit()
        self.output_directory_input.setPlaceholderText(
            "Directory where Ray Tune results will be saved"
        )
        self.output_directory_input.setText(
            str(
                self.settings.value(
                    f"{settings_prefix}/output_directory",
                    "",
                )
            )
        )

        self.output_directory_button = QPushButton("Select...")
        self.output_directory_button.clicked.connect(self.browse_output_directory)

        self.cpu_per_trial_input = QSpinBox()
        self.cpu_per_trial_input.setRange(1, 256)
        self.cpu_per_trial_input.setValue(
            int(
                self.settings.value(
                    f"{settings_prefix}/cpu_per_trial",
                    4,
                )
            )
        )
        self.cpu_per_trial_input.setToolTip("Number of CPU cores reserved for each Ray Tune trial.")

        self.gpu_per_trial_input = QDoubleSpinBox()
        self.gpu_per_trial_input.setDecimals(2)
        self.gpu_per_trial_input.setRange(0.0, 16.0)
        self.gpu_per_trial_input.setSingleStep(0.25)
        self.gpu_per_trial_input.setValue(
            float(
                self.settings.value(
                    f"{settings_prefix}/gpu_per_trial",
                    1.0,
                )
            )
        )
        self.gpu_per_trial_input.setToolTip("Number or fraction of GPU reserved for each trial. .")

        self.number_of_trials_input = QSpinBox()
        self.number_of_trials_input.setRange(1, 10000)
        self.number_of_trials_input.setValue(
            int(
                self.settings.value(
                    f"{settings_prefix}/number_of_trials",
                    50,
                )
            )
        )
        self.number_of_trials_input.setToolTip(
            "Number of hyperparameter combinations tested by Ray Tune."
        )

        self.num_workers_input = QSpinBox()
        self.num_workers_input.setRange(0, 128)
        self.num_workers_input.setValue(
            int(
                self.settings.value(
                    f"{settings_prefix}/num_workers",
                    0,
                )
            )
        )
        self.num_workers_input.setToolTip(
            "Number of DataLoader worker processes used inside each trial. "
            "Use 0 if multiprocessing causes problems."
        )

        self.lr_min_input = QDoubleSpinBox()
        self.lr_min_input.setDecimals(8)
        self.lr_min_input.setRange(1e-8, 1.0)
        self.lr_min_input.setSingleStep(1e-5)
        self.lr_min_input.setKeyboardTracking(False)
        self.lr_min_input.setValue(
            float(
                self.settings.value(
                    f"{settings_prefix}/lr_min",
                    1e-5,
                )
            )
        )

        self.lr_max_input = QDoubleSpinBox()
        self.lr_max_input.setDecimals(8)
        self.lr_max_input.setRange(1e-8, 1.0)
        self.lr_max_input.setSingleStep(1e-4)
        self.lr_max_input.setKeyboardTracking(False)
        self.lr_max_input.setValue(
            float(
                self.settings.value(
                    f"{settings_prefix}/lr_max",
                    1e-3,
                )
            )
        )

        self.weight_decay_min_input = QDoubleSpinBox()
        self.weight_decay_min_input.setDecimals(10)
        self.weight_decay_min_input.setRange(1e-10, 1.0)
        self.weight_decay_min_input.setSingleStep(1e-6)
        self.weight_decay_min_input.setKeyboardTracking(False)
        self.weight_decay_min_input.setValue(
            float(
                self.settings.value(
                    f"{settings_prefix}/weight_decay_min",
                    1e-6,
                )
            )
        )

        self.weight_decay_max_input = QDoubleSpinBox()
        self.weight_decay_max_input.setDecimals(10)
        self.weight_decay_max_input.setRange(1e-10, 1.0)
        self.weight_decay_max_input.setSingleStep(1e-4)
        self.weight_decay_max_input.setKeyboardTracking(False)
        self.weight_decay_max_input.setValue(
            float(
                self.settings.value(
                    f"{settings_prefix}/weight_decay_max",
                    1e-3,
                )
            )
        )

        self.batch_size_min_input = QSpinBox()
        self.batch_size_min_input.setRange(1, 4096)
        self.batch_size_min_input.setValue(
            int(
                self.settings.value(
                    f"{settings_prefix}/batch_size_min",
                    4,
                )
            )
        )
        self.batch_size_min_input.setToolTip(
            "Minimum batch size. The adapter can generate powers of two "
            "between the minimum and maximum values."
        )

        self.batch_size_max_input = QSpinBox()
        self.batch_size_max_input.setRange(1, 4096)
        self.batch_size_max_input.setValue(
            int(
                self.settings.value(
                    f"{settings_prefix}/batch_size_max",
                    64,
                )
            )
        )

        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 10000)
        self.epochs_input.setValue(
            int(
                self.settings.value(
                    f"{settings_prefix}/epochs",
                    50,
                )
            )
        )

        self.patience_input = QSpinBox()
        self.patience_input.setRange(1, 10000)
        self.patience_input.setValue(
            int(
                self.settings.value(
                    f"{settings_prefix}/patience",
                    15,
                )
            )
        )

        self.save_best_epoch_input = QSpinBox()
        self.save_best_epoch_input.setRange(1, 10000)
        self.save_best_epoch_input.setValue(
            int(
                self.settings.value(
                    f"{settings_prefix}/save_best_epoch",
                    1,
                )
            )
        )
        self.save_best_epoch_input.setToolTip(
            "The model can only be saved as the best model starting from this epoch."
        )

        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 2_147_483_647)
        self.seed_input.setValue(
            int(
                self.settings.value(
                    f"{settings_prefix}/seed",
                    42,
                )
            )
        )

        self.max_seq_len_input = QSpinBox()
        self.max_seq_len_input.setMaximum(1000)
        self.max_seq_len_input.setMinimum(1000)
        self.max_seq_len_input.setValue(
            int(
                self.settings.value(
                    f"{settings_prefix}/max_seq_len",
                    1000,
                )
            )
        )

        self.max_seq_len_input.setToolTip("Maximum protein sequence length.")

        self.max_pkt_len_input = QSpinBox()
        self.max_pkt_len_input.setMaximum(63)
        self.max_pkt_len_input.setMinimum(63)
        self.max_pkt_len_input.setValue(
            int(
                self.settings.value(
                    f"{settings_prefix}/max_pkt_len",
                    63,
                )
            )
        )
        self.max_pkt_len_input.setToolTip("Maximum pocket sequence length.")

        self.max_smi_len_input = QSpinBox()
        self.max_smi_len_input.setMaximum(150)
        self.max_smi_len_input.setMinimum(150)
        self.max_smi_len_input.setValue(
            int(
                self.settings.value(
                    f"{settings_prefix}/max_smi_len",
                    150,
                )
            )
        )
        self.max_smi_len_input.setToolTip("Maximum SMILES length.")

        self.lr_min_input.valueChanged.connect(
            lambda value: (
                self.lr_max_input.setValue(value) if value > self.lr_max_input.value() else None
            )
        )

        self.weight_decay_min_input.valueChanged.connect(
            lambda value: (
                self.weight_decay_max_input.setValue(value)
                if value > self.weight_decay_max_input.value()
                else None
            )
        )

        self.batch_size_min_input.valueChanged.connect(
            lambda value: (
                self.batch_size_max_input.setValue(value)
                if value > self.batch_size_max_input.value()
                else None
            )
        )

        general_group = QGroupBox("General configuration")
        general_layout = QGridLayout(general_group)

        general_layout.addWidget(
            QLabel("Data directory:"),
            0,
            0,
        )
        general_layout.addWidget(
            self.data_directory_input,
            0,
            1,
        )
        general_layout.addWidget(
            self.data_directory_button,
            0,
            2,
        )

        general_layout.addWidget(
            QLabel("Model:"),
            1,
            0,
        )
        general_layout.addWidget(
            self.model_input,
            1,
            1,
            1,
            2,
        )

        general_layout.addWidget(
            QLabel("Output directory:"),
            2,
            0,
        )
        general_layout.addWidget(
            self.output_directory_input,
            2,
            1,
        )
        general_layout.addWidget(
            self.output_directory_button,
            2,
            2,
        )

        general_layout.setColumnStretch(1, 1)

        resources_group = QGroupBox("Ray Tune resources")
        resources_layout = QGridLayout(resources_group)

        resources_layout.addWidget(
            QLabel("CPU per trial:"),
            0,
            0,
        )
        resources_layout.addWidget(
            self.cpu_per_trial_input,
            0,
            1,
        )

        resources_layout.addWidget(
            QLabel("GPU per trial:"),
            0,
            2,
        )
        resources_layout.addWidget(
            self.gpu_per_trial_input,
            0,
            3,
        )

        resources_layout.addWidget(
            QLabel("Number of trials:"),
            1,
            0,
        )
        resources_layout.addWidget(
            self.number_of_trials_input,
            1,
            1,
        )

        resources_layout.addWidget(
            QLabel("DataLoader workers:"),
            1,
            2,
        )
        resources_layout.addWidget(
            self.num_workers_input,
            1,
            3,
        )

        search_space_group = QGroupBox("Hyperparameter search space")
        search_space_layout = QGridLayout(search_space_group)

        learning_rate_layout = QHBoxLayout()
        learning_rate_layout.addWidget(QLabel("Min:"))
        learning_rate_layout.addWidget(self.lr_min_input)
        learning_rate_layout.addSpacing(15)
        learning_rate_layout.addWidget(QLabel("Max:"))
        learning_rate_layout.addWidget(self.lr_max_input)

        weight_decay_layout = QHBoxLayout()
        weight_decay_layout.addWidget(QLabel("Min:"))
        weight_decay_layout.addWidget(self.weight_decay_min_input)
        weight_decay_layout.addSpacing(15)
        weight_decay_layout.addWidget(QLabel("Max:"))
        weight_decay_layout.addWidget(self.weight_decay_max_input)

        batch_size_layout = QHBoxLayout()
        batch_size_layout.addWidget(QLabel("Min:"))
        batch_size_layout.addWidget(self.batch_size_min_input)
        batch_size_layout.addSpacing(15)
        batch_size_layout.addWidget(QLabel("Max:"))
        batch_size_layout.addWidget(self.batch_size_max_input)

        search_space_layout.addWidget(
            QLabel("Learning rate:"),
            0,
            0,
        )
        search_space_layout.addLayout(
            learning_rate_layout,
            0,
            1,
        )

        search_space_layout.addWidget(
            QLabel("Weight decay:"),
            1,
            0,
        )
        search_space_layout.addLayout(
            weight_decay_layout,
            1,
            1,
        )

        search_space_layout.addWidget(
            QLabel("Batch size:"),
            2,
            0,
        )
        search_space_layout.addLayout(
            batch_size_layout,
            2,
            1,
        )

        search_space_layout.setColumnStretch(1, 1)

        training_group = QGroupBox("Training configuration")
        training_layout = QGridLayout(training_group)

        training_layout.addWidget(
            QLabel("Epochs:"),
            0,
            0,
        )
        training_layout.addWidget(
            self.epochs_input,
            0,
            1,
        )

        training_layout.addWidget(
            QLabel("Patience:"),
            0,
            2,
        )
        training_layout.addWidget(
            self.patience_input,
            0,
            3,
        )

        training_layout.addWidget(
            QLabel("Save best from epoch:"),
            1,
            0,
        )
        training_layout.addWidget(
            self.save_best_epoch_input,
            1,
            1,
        )

        training_layout.addWidget(
            QLabel("Seed:"),
            1,
            2,
        )
        training_layout.addWidget(
            self.seed_input,
            1,
            3,
        )

        training_layout.addWidget(
            QLabel("Maximum sequence length:"),
            2,
            0,
        )
        training_layout.addWidget(
            self.max_seq_len_input,
            2,
            1,
        )

        training_layout.addWidget(
            QLabel("Maximum pocket length:"),
            2,
            2,
        )
        training_layout.addWidget(
            self.max_pkt_len_input,
            2,
            3,
        )

        training_layout.addWidget(
            QLabel("Maximum SMILES length:"),
            3,
            0,
        )
        training_layout.addWidget(
            self.max_smi_len_input,
            3,
            1,
        )

        main_layout = QVBoxLayout()

        main_layout.addWidget(general_group)
        main_layout.addWidget(resources_group)
        main_layout.addWidget(search_space_group)
        main_layout.addWidget(training_group)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        main_layout.addWidget(self.button_box)

        self.setLayout(main_layout)

    def browse_data_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select the directory containing the data splits",
            self.data_directory_input.text().strip(),
        )

        if directory:
            self.data_directory_input.setText(directory)

    def browse_output_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select the output directory",
            self.output_directory_input.text().strip(),
        )

        if directory:
            self.output_directory_input.setText(directory)

    def accept(self):
        if not self.validate_inputs():
            return

        self.save_settings()
        super().accept()

    def validate_inputs(self):
        data_directory = self.data_directory_input.text().strip()
        output_directory = self.output_directory_input.text().strip()

        if not data_directory:
            QMessageBox.warning(
                self,
                "Missing field",
                "The data directory is required.",
            )
            return False

        if not Path(data_directory).is_dir():
            QMessageBox.warning(
                self,
                "Invalid data directory",
                f"The following directory does not exist:\n{data_directory}",
            )
            return False

        if not output_directory:
            QMessageBox.warning(
                self,
                "Missing field",
                "The output directory is required.",
            )
            return False

        output_path = Path(output_directory)

        try:
            output_path.mkdir(
                parents=True,
                exist_ok=True,
            )
        except OSError as error:
            QMessageBox.warning(
                self,
                "Invalid output directory",
                f"The output directory cannot be created:\n{error}",
            )
            return False

        ranges = [
            (
                self.lr_min_input.value(),
                self.lr_max_input.value(),
                "learning rate",
            ),
            (
                self.weight_decay_min_input.value(),
                self.weight_decay_max_input.value(),
                "weight decay",
            ),
            (
                self.batch_size_min_input.value(),
                self.batch_size_max_input.value(),
                "batch size",
            ),
        ]

        for minimum, maximum, parameter_name in ranges:
            if minimum > maximum:
                QMessageBox.warning(
                    self,
                    "Invalid range",
                    f"The minimum {parameter_name} must be less than or equal to its maximum.",
                )
                return False

        if self.save_best_epoch_input.value() > self.epochs_input.value():
            QMessageBox.warning(
                self,
                "Invalid save epoch",
                "Save best from epoch must be less than or equal to the total number of epochs.",
            )
            return False

        if self.patience_input.value() > self.epochs_input.value():
            QMessageBox.warning(
                self,
                "Invalid patience",
                "Patience must be less than or equal to the total number of epochs.",
            )
            return False

        return True

    def save_settings(self):
        prefix = "hpdeepdtaf"
        values = self.get_inputs()

        for key, value in values.items():
            self.settings.setValue(
                f"{prefix}/{key}",
                value,
            )

        self.settings.sync()

    def get_inputs(self):
        return {
            "model_name": self.model_input.currentText(),
            "data_directory": self.data_directory_input.text().strip(),
            "output_directory": self.output_directory_input.text().strip(),
            "cpu_per_trial": self.cpu_per_trial_input.value(),
            "gpu_per_trial": self.gpu_per_trial_input.value(),
            "number_of_trials": self.number_of_trials_input.value(),
            "num_workers": self.num_workers_input.value(),
            "seed": self.seed_input.value(),
            "save_best_epoch": self.save_best_epoch_input.value(),
            "lr_min": self.lr_min_input.value(),
            "lr_max": self.lr_max_input.value(),
            "weight_decay_min": self.weight_decay_min_input.value(),
            "weight_decay_max": self.weight_decay_max_input.value(),
            "batch_size_min": self.batch_size_min_input.value(),
            "batch_size_max": self.batch_size_max_input.value(),
            "epochs": self.epochs_input.value(),
            "patience": self.patience_input.value(),
            "max_seq_len": self.max_seq_len_input.value(),
            "max_pkt_len": self.max_pkt_len_input.value(),
            "max_smi_len": self.max_smi_len_input.value(),
        }
