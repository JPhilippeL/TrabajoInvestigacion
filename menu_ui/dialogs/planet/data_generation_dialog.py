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
    QVBoxLayout,
    QWidget,
)

from job_config.planet.PlanetDataConfig import PlanetDataConfig

logger = logging.getLogger(__name__)


class DBGenerationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("PLANET DB Generation")
        self.resize(650, 450)

        self.settings = QSettings("Investigacion", "PLANET_DB_Generation")

        self.pic50_file_input = QLineEdit()
        self.pic50_file_input.setPlaceholderText("Target file (pic50.txt)")
        self.pic50_file_input.setText(self.settings.value("planetDbGen/last_pic50_file", ""))
        self.pic50_btn = QPushButton("Select...")
        self.pic50_btn.clicked.connect(self.browse_txt)

        self.ligand_dir_input = QLineEdit()
        self.ligand_dir_input.setPlaceholderText("Folder with .sdf")
        self.ligand_dir_input.setText(self.settings.value("planetDbGen/last_ligand_dir", ""))
        self.ligand_btn = QPushButton("Select...")
        self.ligand_btn.clicked.connect(self.browse_ligand)

        self.pdb_dir_input = QLineEdit()
        self.pdb_dir_input.setPlaceholderText("Folder with .pdb")
        self.pdb_dir_input.setText(self.settings.value("planetDbGen/last_pdb_dir", ""))
        self.pdb_btn = QPushButton("Select...")
        self.pdb_btn.clicked.connect(self.browse_pdb)

        self.output_dir_input = QLineEdit()
        self.output_dir_input.setPlaceholderText("Output directory")
        self.output_dir_input.setText(self.settings.value("planetDbGen/last_output_dir", ""))
        self.output_btn = QPushButton("Select...")
        self.output_btn.clicked.connect(self.browse_output)

        self.train_ratio_input = QLineEdit()
        self.train_ratio_input.setPlaceholderText("Example: 0.8")
        self.train_ratio_input.setText(self.settings.value("planetDbGen/train_ratio", "0.8"))

        self.valid_ratio_input = QLineEdit()
        self.valid_ratio_input.setPlaceholderText("Example: 0.1")
        self.valid_ratio_input.setText(self.settings.value("planetDbGen/valid_ratio", "0.1"))

        self.seed_input = QLineEdit()
        self.seed_input.setPlaceholderText("Example: 42")
        self.seed_input.setText(self.settings.value("planetDbGen/seed", "42"))

        form_layout = QFormLayout()
        form_layout.addRow(QLabel("<b>Required Parameters</b>"))

        form_layout.addRow(
            "pIC50 file:",
            self._with_button(self.pic50_file_input, self.pic50_btn),
        )

        form_layout.addRow(
            "Ligands dir:",
            self._with_button(self.ligand_dir_input, self.ligand_btn),
        )

        form_layout.addRow(
            "Proteins dir:",
            self._with_button(self.pdb_dir_input, self.pdb_btn),
        )

        form_layout.addRow(
            "Output dir:",
            self._with_button(self.output_dir_input, self.output_btn),
        )

        form_layout.addRow(QLabel("<b>Split Parameters</b>"))
        form_layout.addRow("Train ratio:", self.train_ratio_input)
        form_layout.addRow("Valid ratio:", self.valid_ratio_input)
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

    def browse_txt(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select pIC50 file",
            "",
            "Text files (*.txt);;All files (*)",
        )

        if path:
            self.pic50_file_input.setText(path)

    def browse_ligand(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Ligands Folder",
        )

        if path:
            self.ligand_dir_input.setText(path)

    def browse_pdb(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select PDB Folder",
        )

        if path:
            self.pdb_dir_input.setText(path)

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
        )

        if path:
            self.output_dir_input.setText(path)

    def accept(self):
        if not self.pic50_file_input.text().strip():
            logger.warning("Missing input, please select the pIC50 file.")
            return

        if not self.ligand_dir_input.text().strip():
            logger.warning("Missing input, please select the ligand directory.")
            return

        if not self.pdb_dir_input.text().strip():
            logger.warning("Missing input, please select the protein directory.")
            return

        try:
            train_ratio = float(self.train_ratio_input.text().strip())
            valid_ratio = float(self.valid_ratio_input.text().strip())
            seed = int(self.seed_input.text().strip())
        except ValueError:
            logger.warning(
                "Invalid split parameters. Ratios must be floats and seed must be an integer."
            )
            return

        if not 0 < train_ratio < 1:
            logger.warning("train_ratio must be between 0 and 1.")
            return

        if not 0 < valid_ratio < 1:
            logger.warning("valid_ratio must be between 0 and 1.")
            return

        if train_ratio + valid_ratio >= 1:
            logger.warning("train_ratio + valid_ratio must be < 1.")
            return

        self.settings.setValue(
            "planetDbGen/last_pic50_file",
            self.pic50_file_input.text(),
        )
        self.settings.setValue(
            "planetDbGen/last_ligand_dir",
            self.ligand_dir_input.text(),
        )
        self.settings.setValue(
            "planetDbGen/last_pdb_dir",
            self.pdb_dir_input.text(),
        )
        self.settings.setValue(
            "planetDbGen/last_output_dir",
            self.output_dir_input.text(),
        )
        self.settings.setValue(
            "planetDbGen/train_ratio",
            self.train_ratio_input.text(),
        )
        self.settings.setValue(
            "planetDbGen/valid_ratio",
            self.valid_ratio_input.text(),
        )
        self.settings.setValue(
            "planetDbGen/seed",
            self.seed_input.text(),
        )

        super().accept()

    def get_inputs(self) -> PlanetDataConfig:
        output_dir = self.output_dir_input.text().strip()

        if not output_dir:
            output_dir = "outputs/planet/data"

        return PlanetDataConfig(
            protein_path=Path(self.pdb_dir_input.text().strip()),
            ligand_path=Path(self.ligand_dir_input.text().strip()),
            pic50_path=Path(self.pic50_file_input.text().strip()),
            output_path=Path(output_dir),
            train_ratio=float(self.train_ratio_input.text().strip()),
            valid_ratio=float(self.valid_ratio_input.text().strip()),
            seed=int(self.seed_input.text().strip()),
        )
