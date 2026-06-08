import logging
import os

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class DBGenerationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("PLANET DB Generation")
        self.resize(650, 420)

        self.settings = QSettings("Investigacion", "DB_Generation")

        self.pic50_file_input = QLineEdit()
        self.pic50_file_input.setPlaceholderText("pIC50.txt")
        self.pic50_file_input.setText(
            self.settings.value("dbGen/last_pic50_file", "")
        )
        self.pic50_btn = QPushButton("Select...")
        self.pic50_btn.clicked.connect(self.browse_txt)

        self.ligand_dir_input = QLineEdit()
        self.ligand_dir_input.setPlaceholderText("Folder containing .sdf files")
        self.ligand_dir_input.setText(
            self.settings.value("dbGen/last_ligand_dir", "")
        )
        self.ligand_btn = QPushButton("Select...")
        self.ligand_btn.clicked.connect(self.browse_ligand)

        self.pdb_dir_input = QLineEdit()
        self.pdb_dir_input.setPlaceholderText("Folder containing .pdb files")
        self.pdb_dir_input.setText(
            self.settings.value("dbGen/last_pdb_dir", "")
        )
        self.pdb_btn = QPushButton("Select...")
        self.pdb_btn.clicked.connect(self.browse_pdb)

        self.output_dir_input = QLineEdit()
        self.output_dir_input.setPlaceholderText("Output directory")
        self.output_dir_input.setText(
            self.settings.value("dbGen/last_output_dir", "")
        )
        self.output_btn = QPushButton("Select...")
        self.output_btn.clicked.connect(self.browse_output)

        self.train_ratio_input = QDoubleSpinBox()
        self.train_ratio_input.setRange(0.1, 0.95)
        self.train_ratio_input.setSingleStep(0.05)
        self.train_ratio_input.setDecimals(2)
        self.train_ratio_input.setValue(
            float(self.settings.value("dbGen/train_ratio", 0.8))
        )

        self.val_ratio_input = QDoubleSpinBox()
        self.val_ratio_input.setRange(0.01, 0.5)
        self.val_ratio_input.setSingleStep(0.05)
        self.val_ratio_input.setDecimals(2)
        self.val_ratio_input.setValue(
            float(self.settings.value("dbGen/val_ratio", 0.1))
        )

        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 999999)
        self.seed_input.setValue(
            int(self.settings.value("dbGen/seed", 42))
        )

        self.overwrite_checkbox = QCheckBox("Overwrite copied raw files")
        self.overwrite_checkbox.setChecked(
            self.settings.value("dbGen/overwrite", "true") == "true"
        )

        self.overwrite_pockets_checkbox = QCheckBox("Overwrite pocket.pkl files")
        self.overwrite_pockets_checkbox.setChecked(
            self.settings.value("dbGen/overwrite_pockets", "true") == "true"
        )

        form_layout = QFormLayout()
        form_layout.addRow(QLabel("<b>Required parameters</b>"))

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

        form_layout.addRow(QLabel("<b>Split parameters</b>"))
        form_layout.addRow("Train ratio:", self.train_ratio_input)
        form_layout.addRow("Validation ratio:", self.val_ratio_input)
        form_layout.addRow("Seed:", self.seed_input)

        form_layout.addRow(QLabel("<b>Options</b>"))
        form_layout.addRow("", self.overwrite_checkbox)
        form_layout.addRow("", self.overwrite_pockets_checkbox)

        layout = QVBoxLayout()
        layout.addLayout(form_layout)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
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
            "Text files (*.txt);;CSV files (*.csv);;All files (*)",
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
        pic50_file = self.pic50_file_input.text().strip()
        sdf_dir = self.ligand_dir_input.text().strip()
        pdb_dir = self.pdb_dir_input.text().strip()
        output_dir = self.output_dir_input.text().strip()

        if not pic50_file:
            logger.warning("Missing input: please select the pIC50 file.")
            return

        if not os.path.isfile(pic50_file):
            logger.warning("Invalid pIC50 file: %s", pic50_file)
            return

        if not sdf_dir:
            logger.warning("Missing input: please select the ligand directory.")
            return

        if not os.path.isdir(sdf_dir):
            logger.warning("Invalid ligand directory: %s", sdf_dir)
            return

        if not pdb_dir:
            logger.warning("Missing input: please select the protein directory.")
            return

        if not os.path.isdir(pdb_dir):
            logger.warning("Invalid protein directory: %s", pdb_dir)
            return

        if not output_dir:
            logger.warning("Missing input: please select the output directory.")
            return

        if not os.path.isdir(output_dir):
            logger.warning("Invalid output directory: %s", output_dir)
            return

        train_ratio = self.train_ratio_input.value()
        val_ratio = self.val_ratio_input.value()

        if train_ratio + val_ratio >= 1.0:
            logger.warning(
                "Invalid split ratios: train_ratio + val_ratio must be < 1.0"
            )
            return

        self.settings.setValue("dbGen/last_pic50_file", pic50_file)
        self.settings.setValue("dbGen/last_ligand_dir", sdf_dir)
        self.settings.setValue("dbGen/last_pdb_dir", pdb_dir)
        self.settings.setValue("dbGen/last_output_dir", output_dir)

        self.settings.setValue("dbGen/train_ratio", train_ratio)
        self.settings.setValue("dbGen/val_ratio", val_ratio)
        self.settings.setValue("dbGen/seed", self.seed_input.value())

        self.settings.setValue(
            "dbGen/overwrite",
            "true" if self.overwrite_checkbox.isChecked() else "false",
        )
        self.settings.setValue(
            "dbGen/overwrite_pockets",
            "true" if self.overwrite_pockets_checkbox.isChecked() else "false",
        )

        super().accept()

    def get_inputs(self):
        return {
            "pic50_path": self.pic50_file_input.text().strip(),
            "sdf_dir": self.ligand_dir_input.text().strip(),
            "pdb_dir": self.pdb_dir_input.text().strip(),
            "output_dir": self.output_dir_input.text().strip(),
            "train_ratio": self.train_ratio_input.value(),
            "val_ratio": self.val_ratio_input.value(),
            "seed": self.seed_input.value(),
            "overwrite": self.overwrite_checkbox.isChecked(),
            "overwrite_pockets": self.overwrite_pockets_checkbox.isChecked(),
        }