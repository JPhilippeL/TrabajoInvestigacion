"""
@file generate_data_dialog.py
@author Mohamed EL BOUKHIARI
@brief Data generation dialog for the EGNN module.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit,
    QPushButton, QFileDialog, QDialogButtonBox, QWidget, QHBoxLayout,
    QDoubleSpinBox, QLabel
)
from PySide6.QtCore import QSettings

from EGNN.utils.constants import (
    DEFAULT_GRAPHS_DIR,
    DEFAULT_CUTOFF_EDGES,
    DEFAULT_CUTOFF_PROT,
)


class DBGenerationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generación de Base de Datos - EGNN")
        self.resize(700, 320)

        self.settings = QSettings("Investigacion", "EGNN_DB_Generation")

        self.pic50_file_input = QLineEdit()
        self.pic50_file_input.setPlaceholderText("Archivo pIC50.txt")
        self.pic50_file_input.setText(self.settings.value("dbGen/pic50_file", ""))
        self.pic50_btn = QPushButton("Seleccionar...")
        self.pic50_btn.clicked.connect(self.browse_pic50)

        self.ligand_sdf_dir_input = QLineEdit()
        self.ligand_sdf_dir_input.setPlaceholderText("Carpeta Ligand_SDF")
        self.ligand_sdf_dir_input.setText(self.settings.value("dbGen/ligand_sdf_dir", ""))
        self.ligand_sdf_btn = QPushButton("Seleccionar...")
        self.ligand_sdf_btn.clicked.connect(self.browse_ligand_sdf)

        self.protein_pdb_dir_input = QLineEdit()
        self.protein_pdb_dir_input.setPlaceholderText("Carpeta Protein_PDB")
        self.protein_pdb_dir_input.setText(self.settings.value("dbGen/protein_pdb_dir", ""))
        self.protein_pdb_btn = QPushButton("Seleccionar...")
        self.protein_pdb_btn.clicked.connect(self.browse_protein_pdb)

        self.graphs_dir_input = QLineEdit()
        self.graphs_dir_input.setPlaceholderText("Directorio de salida Graphs_EGNN")
        self.graphs_dir_input.setText(self.settings.value("dbGen/graphs_dir", DEFAULT_GRAPHS_DIR))
        self.graphs_dir_btn = QPushButton("Seleccionar...")
        self.graphs_dir_btn.clicked.connect(self.browse_graphs_dir)

        self.cutoff_edges_spin = QDoubleSpinBox()
        self.cutoff_edges_spin.setRange(0.1, 20.0)
        self.cutoff_edges_spin.setSingleStep(0.1)
        self.cutoff_edges_spin.setDecimals(2)
        self.cutoff_edges_spin.setValue(float(self.settings.value("dbGen/cutoff_edges", DEFAULT_CUTOFF_EDGES)))

        self.cutoff_prot_spin = QDoubleSpinBox()
        self.cutoff_prot_spin.setRange(0.1, 20.0)
        self.cutoff_prot_spin.setSingleStep(0.1)
        self.cutoff_prot_spin.setDecimals(2)
        self.cutoff_prot_spin.setValue(float(self.settings.value("dbGen/cutoff_prot", DEFAULT_CUTOFF_PROT)))

        form_layout = QFormLayout()
        form_layout.addRow(QLabel("<b>Parámetros requeridos</b>"))
        form_layout.addRow("Archivo pIC50:", self._with_button(self.pic50_file_input, self.pic50_btn))
        form_layout.addRow("Dir. Ligand_SDF:", self._with_button(self.ligand_sdf_dir_input, self.ligand_sdf_btn))
        form_layout.addRow("Dir. Protein_PDB:", self._with_button(self.protein_pdb_dir_input, self.protein_pdb_btn))
        form_layout.addRow("Output Graphs:", self._with_button(self.graphs_dir_input, self.graphs_dir_btn))

        form_layout.addRow(QLabel("<b>Opcionales</b>"))
        form_layout.addRow("Cutoff Edges:", self.cutoff_edges_spin)
        form_layout.addRow("Cutoff Protein:", self.cutoff_prot_spin)

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
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar pIC50.txt", "", "Text Files (*.txt);;All Files (*)")
        if path:
            self.pic50_file_input.setText(path)

    def browse_ligand_sdf(self):
        path = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta Ligand_SDF")
        if path:
            self.ligand_sdf_dir_input.setText(path)

    def browse_protein_pdb(self):
        path = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta Protein_PDB")
        if path:
            self.protein_pdb_dir_input.setText(path)

    def browse_graphs_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de salida de grafos")
        if path:
            self.graphs_dir_input.setText(path)

    def accept(self):
        self.settings.setValue("dbGen/pic50_file", self.pic50_file_input.text())
        self.settings.setValue("dbGen/ligand_sdf_dir", self.ligand_sdf_dir_input.text())
        self.settings.setValue("dbGen/protein_pdb_dir", self.protein_pdb_dir_input.text())
        self.settings.setValue("dbGen/graphs_dir", self.graphs_dir_input.text())
        self.settings.setValue("dbGen/cutoff_edges", self.cutoff_edges_spin.value())
        self.settings.setValue("dbGen/cutoff_prot", self.cutoff_prot_spin.value())
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
