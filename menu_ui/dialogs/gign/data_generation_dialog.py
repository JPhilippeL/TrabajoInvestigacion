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

from job_config.gign.GIGNDataConfig import GIGNDataConfig

logger = logging.getLogger(__name__)


class DBGenerationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DB Generation")
        self.resize(600, 400)

        self.settings = QSettings("Investigacion", "DB_Generation")

        self.pic50_file_input = QLineEdit()
        self.pic50_file_input.setPlaceholderText("Target file (pic50.txt)")
        self.pic50_file_input.setText(self.settings.value("dbGen/last_pic50_file", ""))
        self.pic50_btn = QPushButton("Select...")
        self.pic50_btn.clicked.connect(self.browse_txt)

        self.ligand_dir_input = QLineEdit()
        self.ligand_dir_input.setPlaceholderText("Folder with .sdf")
        self.ligand_dir_input.setText(self.settings.value("dbGen/last_ligand_dir", ""))
        self.ligand_btn = QPushButton("Select...")
        self.ligand_btn.clicked.connect(self.browse_ligand)

        self.pdb_dir_input = QLineEdit()
        self.pdb_dir_input.setPlaceholderText("Folder with .pdb")
        self.pdb_dir_input.setText(self.settings.value("dbGen/last_pdb_dir", ""))
        self.pdb_btn = QPushButton("Select...")
        self.pdb_btn.clicked.connect(self.browse_pdb)

        self.output_dir_input = QLineEdit()
        self.output_dir_input.setPlaceholderText("Output directory")
        self.output_dir_input.setText(self.settings.value("dbGen/last_output_dir", ""))
        self.output_btn = QPushButton("Select...")
        self.output_btn.clicked.connect(self.browse_output)

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
        form_layout.addRow("Proteins dir:", self._with_button(self.pdb_dir_input, self.pdb_btn))

        form_layout.addRow(
            "Output dir:",
            self._with_button(self.output_dir_input, self.output_btn),
        )

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
            self, "Select pIC50 file", "", "Text files (*.txt);;All files (*)"
        )
        if path:
            self.pic50_file_input.setText(path)

    def browse_ligand(self):
        path = QFileDialog.getExistingDirectory(self, "Select Ligands Folder")
        if path:
            self.ligand_dir_input.setText(path)

    def browse_pdb(self):
        path = QFileDialog.getExistingDirectory(self, "Select PDB Folder")
        if path:
            self.pdb_dir_input.setText(path)

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.output_dir_input.setText(path)

    def accept(self):
        if not self.pic50_file_input.text().strip():
            logger.warning("Missing input,Please select the pic50 file.")
            return

        if not self.ligand_dir_input.text().strip():
            logger.warning("Missing input,Please select the ligand directory.")

            return

        if not self.pdb_dir_input.text().strip():
            logger.warning("Missing input,Please select the protein directory.")
            return

        self.settings.setValue("dbGen/last_pic50_file", self.pic50_file_input.text())
        self.settings.setValue("dbGen/last_ligand_dir", self.ligand_dir_input.text())
        self.settings.setValue("dbGen/last_pdb_dir", self.pdb_dir_input.text())
        self.settings.setValue("dbGen/last_output_dir", self.output_dir_input.text())

        super().accept()

    def get_inputs(self) -> GIGNDataConfig:
        output_dir = self.output_dir_input.text().strip()

        if not output_dir:
            output_dir = "outputs/cheapnet/data"

        return GIGNDataConfig(
            pic50_path=Path(self.pic50_file_input.text().strip()),
            ligand_path=Path(self.ligand_dir_input.text().strip()),
            protein_path=Path(self.pdb_dir_input.text().strip()),
            output_path=Path(output_dir),
        )
