"""
@file generate_data_dialog.py
@author Mohamed EL BOUKHIARI
@brief Data generation dialog for the EGNN module.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QDialogButtonBox,
    QWidget,
    QHBoxLayout,
    QDoubleSpinBox,
    QLabel,
)

from EGNN.utils.constants import (
    DEFAULT_GRAPHS_DIR,
    DEFAULT_CUTOFF_EDGES,
    DEFAULT_CUTOFF_PROT,
    DEFAULT_PIC50_FILE,
    DEFAULT_LIGAND_SDF_DIR,
    DEFAULT_PROTEIN_PDB_DIR,
)


class DBGenerationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EGNN Data Generation Configuration")
        self.resize(720, 320)

        self.settings = QSettings("ResearchApp", "EGNN_DataGeneration")

        self.pic50_file_input = QLineEdit()
        self.pic50_file_input.setPlaceholderText("pIC50.txt file")
        self.pic50_file_input.setText(self.settings.value("data_generation/pic50_file", DEFAULT_PIC50_FILE))
        self.pic50_btn = QPushButton("Browse...")
        self.pic50_btn.clicked.connect(self.browse_pic50)

        self.ligand_sdf_dir_input = QLineEdit()
        self.ligand_sdf_dir_input.setPlaceholderText("Ligand_SDF directory")
        self.ligand_sdf_dir_input.setText(self.settings.value("data_generation/ligand_sdf_dir", DEFAULT_LIGAND_SDF_DIR))
        self.ligand_sdf_btn = QPushButton("Browse...")
        self.ligand_sdf_btn.clicked.connect(self.browse_ligand_sdf)

        self.protein_pdb_dir_input = QLineEdit()
        self.protein_pdb_dir_input.setPlaceholderText("Protein_PDB directory")
        self.protein_pdb_dir_input.setText(self.settings.value("data_generation/protein_pdb_dir", DEFAULT_PROTEIN_PDB_DIR))
        self.protein_pdb_btn = QPushButton("Browse...")
        self.protein_pdb_btn.clicked.connect(self.browse_protein_pdb)

        self.graphs_dir_input = QLineEdit()
        self.graphs_dir_input.setPlaceholderText("Output directory for Graphs_EGNN")
        self.graphs_dir_input.setText(self.settings.value("data_generation/graphs_dir", DEFAULT_GRAPHS_DIR))
        self.graphs_dir_btn = QPushButton("Browse...")
        self.graphs_dir_btn.clicked.connect(self.browse_graphs_dir)

        self.cutoff_edges_spin = QDoubleSpinBox()
        self.cutoff_edges_spin.setRange(0.1, 20.0)
        self.cutoff_edges_spin.setSingleStep(0.1)
        self.cutoff_edges_spin.setDecimals(2)
        self.cutoff_edges_spin.setValue(float(self.settings.value("data_generation/cutoff_edges", DEFAULT_CUTOFF_EDGES)))

        self.cutoff_prot_spin = QDoubleSpinBox()
        self.cutoff_prot_spin.setRange(0.1, 20.0)
        self.cutoff_prot_spin.setSingleStep(0.1)
        self.cutoff_prot_spin.setDecimals(2)
        self.cutoff_prot_spin.setValue(float(self.settings.value("data_generation/cutoff_prot", DEFAULT_CUTOFF_PROT)))

        form_layout = QFormLayout()
        form_layout.addRow(QLabel("<b>Required inputs</b>"))
        form_layout.addRow("pIC50 file:", self._with_button(self.pic50_file_input, self.pic50_btn))
        form_layout.addRow("Ligand_SDF directory:", self._with_button(self.ligand_sdf_dir_input, self.ligand_sdf_btn))
        form_layout.addRow("Protein_PDB directory:", self._with_button(self.protein_pdb_dir_input, self.protein_pdb_btn))
        form_layout.addRow("Output graphs directory:", self._with_button(self.graphs_dir_input, self.graphs_dir_btn))

        form_layout.addRow(QLabel("<b>Graph parameters</b>"))
        form_layout.addRow("Edge cutoff:", self.cutoff_edges_spin)
        form_layout.addRow("Protein cutoff:", self.cutoff_prot_spin)

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

    def browse_pic50(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select pIC50.txt", "", "Text Files (*.txt);;All Files (*)")
        if path:
            self.pic50_file_input.setText(path)

    def browse_ligand_sdf(self):
        path = QFileDialog.getExistingDirectory(self, "Select Ligand_SDF directory")
        if path:
            self.ligand_sdf_dir_input.setText(path)

    def browse_protein_pdb(self):
        path = QFileDialog.getExistingDirectory(self, "Select Protein_PDB directory")
        if path:
            self.protein_pdb_dir_input.setText(path)

    def browse_graphs_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select output graphs directory")
        if path:
            self.graphs_dir_input.setText(path)

    def accept(self):
        self.settings.setValue("data_generation/pic50_file", self.pic50_file_input.text())
        self.settings.setValue("data_generation/ligand_sdf_dir", self.ligand_sdf_dir_input.text())
        self.settings.setValue("data_generation/protein_pdb_dir", self.protein_pdb_dir_input.text())
        self.settings.setValue("data_generation/graphs_dir", self.graphs_dir_input.text())
        self.settings.setValue("data_generation/cutoff_edges", self.cutoff_edges_spin.value())
        self.settings.setValue("data_generation/cutoff_prot", self.cutoff_prot_spin.value())
        super().accept()

    def get_inputs(self):
        return {
            "pic50_file": self.pic50_file_input.text(),
            "ligand_sdf_dir": self.ligand_sdf_dir_input.text(),
            "protein_pdb_dir": self.protein_pdb_dir_input.text(),
            "graphs_dir": self.graphs_dir_input.text(),
            "cutoff_edges": self.cutoff_edges_spin.value(),
            "cutoff_prot": self.cutoff_prot_spin.value(),
        }
