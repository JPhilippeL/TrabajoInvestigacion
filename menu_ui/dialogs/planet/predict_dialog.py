import logging
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
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

from job_config.planet.PlanetPredictionConfig import PlanetPredictionConfig
from models.planet.architecture.planet import PLANET

logger = logging.getLogger(__name__)


class PredictDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Predict PLANET")
        self.resize(700, 400)

        self.settings = QSettings("Investigacion", "Predict_PLANET")

        self.data_output_dir_input = QLineEdit()
        self.data_output_dir_input.setPlaceholderText(
            "PLANET data directory containing metadata/pkl/core.pkl"
        )
        self.data_output_dir_input.setText(
            self.settings.value("predict_planet/last_data_output_dir", "")
        )
        self.data_output_dir_btn = QPushButton("Select...")
        self.data_output_dir_btn.clicked.connect(self.browse_data_output_dir)

        self.model_dir_input = QLineEdit()
        self.model_dir_input.setPlaceholderText("Directory containing best_model.pt")
        self.model_dir_input.setText(self.settings.value("predict_planet/last_model_dir", ""))
        self.model_dir_btn = QPushButton("Select...")
        self.model_dir_btn.clicked.connect(self.browse_model_dir)

        self.output_dir_input = QLineEdit()
        self.output_dir_input.setPlaceholderText("Directory to save prediction outputs")
        self.output_dir_input.setText(self.settings.value("predict_planet/last_output_dir", ""))
        self.output_dir_btn = QPushButton("Select...")
        self.output_dir_btn.clicked.connect(self.browse_output_dir)

        self.batch_size_input = QSpinBox()
        self.batch_size_input.setRange(1, 1024)
        self.batch_size_input.setValue(int(self.settings.value("predict_planet/batch_size", 1)))

        self.num_workers_input = QSpinBox()
        self.num_workers_input.setRange(0, 64)
        self.num_workers_input.setValue(int(self.settings.value("predict_planet/num_workers", 0)))

        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 1_000_000)
        self.seed_input.setValue(int(self.settings.value("predict_planet/seed", 42)))

        form_layout = QFormLayout()

        form_layout.addRow(QLabel("<b>Model</b>"))
        form_layout.addRow("Model name:", QLabel("PLANET"))

        form_layout.addRow(QLabel("<b>Required directories</b>"))
        form_layout.addRow(
            "PLANET data dir:",
            self._with_button(self.data_output_dir_input, self.data_output_dir_btn),
        )
        form_layout.addRow(
            "Model dir:",
            self._with_button(self.model_dir_input, self.model_dir_btn),
        )
        form_layout.addRow(
            "Output dir:",
            self._with_button(self.output_dir_input, self.output_dir_btn),
        )

        form_layout.addRow(QLabel("<b>Prediction parameters</b>"))
        form_layout.addRow("Batch size:", self.batch_size_input)
        form_layout.addRow("Num workers:", self.num_workers_input)
        form_layout.addRow("Seed:", self.seed_input)

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

    def browse_model_dir(self):
        path = self._browse_dir("Select PLANET model directory")
        if path:
            self.model_dir_input.setText(path)

    def browse_output_dir(self):
        path = self._browse_dir("Select prediction output directory")
        if path:
            self.output_dir_input.setText(path)

    def accept(self):
        data_output_dir = self.data_output_dir_input.text().strip()
        model_dir = self.model_dir_input.text().strip()
        output_dir = self.output_dir_input.text().strip()

        if not data_output_dir:
            logger.warning("Missing input: please select the PLANET data directory.")
            return

        if not model_dir:
            logger.warning("Missing input: please select the model directory.")
            return

        if not output_dir:
            logger.warning("Missing input: please select the output directory.")
            return

        core_pkl = Path(data_output_dir) / "metadata" / "pkl" / "core.pkl"
        if not core_pkl.exists():
            logger.warning(f"Missing file: {core_pkl}")
            return

        best_model = Path(model_dir) / "best_model.pt"
        if not best_model.exists():
            logger.warning(f"Missing model checkpoint: {best_model}")
            return

        self.settings.setValue(
            "predict_planet/last_data_output_dir",
            data_output_dir,
        )
        self.settings.setValue(
            "predict_planet/last_model_dir",
            model_dir,
        )
        self.settings.setValue(
            "predict_planet/last_output_dir",
            output_dir,
        )
        self.settings.setValue(
            "predict_planet/batch_size",
            self.batch_size_input.value(),
        )
        self.settings.setValue(
            "predict_planet/num_workers",
            self.num_workers_input.value(),
        )
        self.settings.setValue(
            "predict_planet/seed",
            self.seed_input.value(),
        )

        super().accept()

    def get_inputs(self) -> PlanetPredictionConfig:
        return PlanetPredictionConfig(
            data_output_path=Path(self.data_output_dir_input.text().strip()),
            model_path=Path(self.model_dir_input.text().strip()),
            output_path=Path(self.output_dir_input.text().strip()),
            model=PLANET,
            model_name="PLANET",
            batch_size=self.batch_size_input.value(),
            num_workers=self.num_workers_input.value(),
            seed=self.seed_input.value(),
        )
