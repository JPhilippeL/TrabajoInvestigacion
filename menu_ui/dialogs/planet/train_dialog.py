import logging
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from job_config.planet.PlanetTrainerConfig import PlanetTrainerConfig
from models.planet.architecture.planet import PLANET

logger = logging.getLogger(__name__)


class TrainDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Train PLANET")
        self.resize(700, 620)

        self.settings = QSettings("Investigacion", "Train_PLANET")

        self.data_output_dir_input = QLineEdit()
        self.data_output_dir_input.setPlaceholderText(
            "PLANET data directory containing metadata/pkl/train.pkl, valid.pkl, core.pkl"
        )
        self.data_output_dir_input.setText(
            self.settings.value("train_planet/last_data_output_dir", "")
        )
        self.data_output_dir_btn = QPushButton("Select...")
        self.data_output_dir_btn.clicked.connect(self.browse_data_output_dir)

        self.output_dir_input = QLineEdit()
        self.output_dir_input.setPlaceholderText("Directory to save trained PLANET model")
        self.output_dir_input.setText(self.settings.value("train_planet/last_output_dir", ""))
        self.output_dir_btn = QPushButton("Select...")
        self.output_dir_btn.clicked.connect(self.browse_output_dir)

        self.batch_size_input = QSpinBox()
        self.batch_size_input.setRange(1, 1024)
        self.batch_size_input.setValue(int(self.settings.value("train_planet/batch_size", 1)))

        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 10000)
        self.epochs_input.setValue(int(self.settings.value("train_planet/epochs", 50)))

        self.lr_input = QLineEdit()
        self.lr_input.setPlaceholderText("Learning rate, for example 0.0001")
        self.lr_input.setText(str(self.settings.value("train_planet/lr", "0.0001")))

        self.weight_decay_input = QLineEdit()
        self.weight_decay_input.setPlaceholderText("Weight decay, for example 0.00001")
        self.weight_decay_input.setText(
            str(self.settings.value("train_planet/weight_decay", "0.0"))
        )

        self.patience_input = QSpinBox()
        self.patience_input.setRange(1, 10000)
        self.patience_input.setValue(int(self.settings.value("train_planet/patience", 10)))

        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 1_000_000)
        self.seed_input.setValue(int(self.settings.value("train_planet/seed", 42)))

        self.num_workers_input = QSpinBox()
        self.num_workers_input.setRange(0, 64)
        self.num_workers_input.setValue(int(self.settings.value("train_planet/num_workers", 0)))

        self.feature_dims_input = QSpinBox()
        self.feature_dims_input.setRange(1, 4096)
        self.feature_dims_input.setValue(int(self.settings.value("train_planet/feature_dims", 300)))

        self.nheads_input = QSpinBox()
        self.nheads_input.setRange(1, 64)
        self.nheads_input.setValue(int(self.settings.value("train_planet/nheads", 8)))

        self.key_dims_input = QSpinBox()
        self.key_dims_input.setRange(1, 4096)
        self.key_dims_input.setValue(int(self.settings.value("train_planet/key_dims", 300)))

        self.value_dims_input = QSpinBox()
        self.value_dims_input.setRange(1, 4096)
        self.value_dims_input.setValue(int(self.settings.value("train_planet/value_dims", 300)))

        self.pro_update_inters_input = QSpinBox()
        self.pro_update_inters_input.setRange(1, 100)
        self.pro_update_inters_input.setValue(
            int(self.settings.value("train_planet/pro_update_inters", 3))
        )

        self.lig_update_iters_input = QSpinBox()
        self.lig_update_iters_input.setRange(1, 100)
        self.lig_update_iters_input.setValue(
            int(self.settings.value("train_planet/lig_update_iters", 10))
        )

        self.pro_lig_update_iters_input = QSpinBox()
        self.pro_lig_update_iters_input.setRange(1, 100)
        self.pro_lig_update_iters_input.setValue(
            int(self.settings.value("train_planet/pro_lig_update_iters", 1))
        )

        self.clip_norm_input = QDoubleSpinBox()
        self.clip_norm_input.setDecimals(4)
        self.clip_norm_input.setRange(0.0, 100000.0)
        self.clip_norm_input.setSingleStep(10.0)
        self.clip_norm_input.setValue(float(self.settings.value("train_planet/clip_norm", 200.0)))

        self.beta_start_step_input = QSpinBox()
        self.beta_start_step_input.setRange(0, 1_000_000)
        self.beta_start_step_input.setValue(
            int(self.settings.value("train_planet/beta_start_step", 500))
        )

        form_layout = QFormLayout()

        form_layout.addRow(QLabel("<b>Required directories</b>"))
        form_layout.addRow(
            "PLANET data dir:",
            self._with_button(self.data_output_dir_input, self.data_output_dir_btn),
        )
        form_layout.addRow(
            "Output dir:",
            self._with_button(self.output_dir_input, self.output_dir_btn),
        )

        form_layout.addRow(QLabel("<b>Training parameters</b>"))
        form_layout.addRow("Batch size:", self.batch_size_input)
        form_layout.addRow("Epochs:", self.epochs_input)
        form_layout.addRow("Learning rate:", self.lr_input)
        form_layout.addRow("Weight decay:", self.weight_decay_input)
        form_layout.addRow("Patience:", self.patience_input)
        form_layout.addRow("Seed:", self.seed_input)
        form_layout.addRow("Num workers:", self.num_workers_input)

        form_layout.addRow(QLabel("<b>PLANET architecture parameters</b>"))
        form_layout.addRow("Feature dims:", self.feature_dims_input)
        form_layout.addRow("Attention heads:", self.nheads_input)
        form_layout.addRow("Key dims:", self.key_dims_input)
        form_layout.addRow("Value dims:", self.value_dims_input)
        form_layout.addRow("Protein update iterations:", self.pro_update_inters_input)
        form_layout.addRow("Ligand update iterations:", self.lig_update_iters_input)
        form_layout.addRow(
            "Protein-ligand update iterations:",
            self.pro_lig_update_iters_input,
        )

        form_layout.addRow(QLabel("<b>Loss parameters</b>"))
        form_layout.addRow("Clip norm:", self.clip_norm_input)
        form_layout.addRow("Beta start step:", self.beta_start_step_input)

        layout = QVBoxLayout()
        layout.addLayout(form_layout)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout.addWidget(self.buttons)
        self.setLayout(layout)

    def _with_button(self, line_edit, button):
        container = QWidget()
        hbox = QHBoxLayout(container)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(line_edit)
        hbox.addWidget(button)
        return container

    def _browse_dir(self, title):
        return QFileDialog.getExistingDirectory(self, title)

    def browse_data_output_dir(self):
        path = self._browse_dir("Select PLANET data directory")
        if path:
            self.data_output_dir_input.setText(path)

    def browse_output_dir(self):
        path = self._browse_dir("Select output directory")
        if path:
            self.output_dir_input.setText(path)

    def accept(self):
        if not self.data_output_dir_input.text().strip():
            logger.warning("Missing input: please select the PLANET data directory.")
            return

        if not self.output_dir_input.text().strip():
            logger.warning("Missing input: please select the output directory.")
            return

        lr_text = self.lr_input.text().strip()
        if not lr_text:
            logger.warning("Missing input: please enter the learning rate.")
            return

        try:
            lr = float(lr_text)
        except ValueError:
            logger.warning("Invalid input: learning rate must be a float.")
            return

        if lr <= 0:
            logger.warning("Invalid input: learning rate must be positive.")
            return

        weight_decay_text = self.weight_decay_input.text().strip()
        if not weight_decay_text:
            logger.warning("Missing input: please enter the weight decay.")
            return

        try:
            weight_decay = float(weight_decay_text)
        except ValueError:
            logger.warning("Invalid input: weight decay must be a float.")
            return

        if weight_decay < 0:
            logger.warning("Invalid input: weight decay must be >= 0.")
            return

        if self.clip_norm_input.value() <= 0:
            logger.warning("Invalid input: clip norm must be positive.")
            return

        self.settings.setValue(
            "train_planet/last_data_output_dir",
            self.data_output_dir_input.text().strip(),
        )
        self.settings.setValue(
            "train_planet/last_output_dir",
            self.output_dir_input.text().strip(),
        )

        self.settings.setValue("train_planet/batch_size", self.batch_size_input.value())
        self.settings.setValue("train_planet/epochs", self.epochs_input.value())
        self.settings.setValue("train_planet/lr", lr_text)
        self.settings.setValue("train_planet/weight_decay", weight_decay_text)
        self.settings.setValue("train_planet/patience", self.patience_input.value())
        self.settings.setValue("train_planet/seed", self.seed_input.value())
        self.settings.setValue("train_planet/num_workers", self.num_workers_input.value())

        self.settings.setValue("train_planet/feature_dims", self.feature_dims_input.value())
        self.settings.setValue("train_planet/nheads", self.nheads_input.value())
        self.settings.setValue("train_planet/key_dims", self.key_dims_input.value())
        self.settings.setValue("train_planet/value_dims", self.value_dims_input.value())
        self.settings.setValue(
            "train_planet/pro_update_inters",
            self.pro_update_inters_input.value(),
        )
        self.settings.setValue(
            "train_planet/lig_update_iters",
            self.lig_update_iters_input.value(),
        )
        self.settings.setValue(
            "train_planet/pro_lig_update_iters",
            self.pro_lig_update_iters_input.value(),
        )

        self.settings.setValue("train_planet/clip_norm", self.clip_norm_input.value())
        self.settings.setValue(
            "train_planet/beta_start_step",
            self.beta_start_step_input.value(),
        )

        super().accept()

    def get_inputs(self) -> PlanetTrainerConfig:
        return PlanetTrainerConfig(
            output_path=Path(self.output_dir_input.text().strip()),
            data_output_path=Path(self.data_output_dir_input.text().strip()),
            model=PLANET,
            model_name="PLANET",
            batch_size=self.batch_size_input.value(),
            epochs=self.epochs_input.value(),
            lr=float(self.lr_input.text().strip()),
            weight_decay=float(self.weight_decay_input.text().strip()),
            patience=self.patience_input.value(),
            seed=self.seed_input.value(),
            num_workers=self.num_workers_input.value(),
            feature_dims=self.feature_dims_input.value(),
            nheads=self.nheads_input.value(),
            key_dims=self.key_dims_input.value(),
            value_dims=self.value_dims_input.value(),
            pro_update_inters=self.pro_update_inters_input.value(),
            lig_update_iters=self.lig_update_iters_input.value(),
            pro_lig_update_iters=self.pro_lig_update_iters_input.value(),
            clip_norm=self.clip_norm_input.value(),
            beta_start_step=self.beta_start_step_input.value(),
        )
