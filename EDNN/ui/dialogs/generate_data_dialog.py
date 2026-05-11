"""
@file generate_data_dialog.py
@author Mohamed EL BOUKHIARI
@brief Data generation dialog for the EDNN module.
@details
This dialog reads global defaults from AppSettings for dataset paths and EDNN
graph output paths, while preserving local QSettings for EDNN-specific graph
generation parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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

from EDNN.utils.constants import (
    DEFAULT_GRAPHS_DIR,
    DEFAULT_CUTOFF_EDGES,
    DEFAULT_CUTOFF_PROT,
)
from ui.utils.app_settings import AppSettings


class DBGenerationDialog(QDialog):
    """
    @brief Dialog used to configure EDNN graph generation.
    """

    def __init__(self, parent=None) -> None:
        """
        @brief Initialize the EDNN data generation dialog.

        @param parent Optional parent widget.
        """
        super().__init__(parent)

        self.setWindowTitle("Data Generation Configuration - EDNN")
        self.resize(700, 320)

        self.settings = QSettings("Investigacion", "EDNN_DB_Generation")
        self.app_settings = AppSettings()

        self.pic50_file_input = QLineEdit()
        self.pic50_file_input.setPlaceholderText("pIC50.txt file")
        self.pic50_file_input.setText(self._global_path_or_default("paths/pic50_file", ""))
        self.pic50_btn = QPushButton("Select...")
        self.pic50_btn.clicked.connect(self.browse_pic50)

        self.ligand_sdf_dir_input = QLineEdit()
        self.ligand_sdf_dir_input.setPlaceholderText("Ligand_SDF folder")
        self.ligand_sdf_dir_input.setText(self._global_path_or_default("paths/ligand_sdf", ""))
        self.ligand_sdf_btn = QPushButton("Select...")
        self.ligand_sdf_btn.clicked.connect(self.browse_ligand_sdf)

        self.protein_pdb_dir_input = QLineEdit()
        self.protein_pdb_dir_input.setPlaceholderText("Protein_PDB folder")
        self.protein_pdb_dir_input.setText(self._global_path_or_default("paths/protein_pdb", ""))
        self.protein_pdb_btn = QPushButton("Select...")
        self.protein_pdb_btn.clicked.connect(self.browse_protein_pdb)

        self.graphs_dir_input = QLineEdit()
        self.graphs_dir_input.setPlaceholderText("Output directory Graphs_EDNN")
        self.graphs_dir_input.setText(self._ednn_subpath_or_default("Graphs_EDNN", DEFAULT_GRAPHS_DIR))
        self.graphs_dir_btn = QPushButton("Select...")
        self.graphs_dir_btn.clicked.connect(self.browse_graphs_dir)

        self.cutoff_edges_spin = QDoubleSpinBox()
        self.cutoff_edges_spin.setRange(0.1, 20.0)
        self.cutoff_edges_spin.setSingleStep(0.1)
        self.cutoff_edges_spin.setDecimals(2)
        self.cutoff_edges_spin.setValue(self._float_setting("dbGen/cutoff_edges", DEFAULT_CUTOFF_EDGES))

        self.cutoff_prot_spin = QDoubleSpinBox()
        self.cutoff_prot_spin.setRange(0.1, 20.0)
        self.cutoff_prot_spin.setSingleStep(0.1)
        self.cutoff_prot_spin.setDecimals(2)
        self.cutoff_prot_spin.setValue(self._float_setting("dbGen/cutoff_prot", DEFAULT_CUTOFF_PROT))

        form_layout = QFormLayout()
        form_layout.addRow(QLabel("<b>Required parameters</b>"))
        form_layout.addRow("pIC50 file:", self._with_button(self.pic50_file_input, self.pic50_btn))
        form_layout.addRow(
            "Ligand_SDF dir:",
            self._with_button(self.ligand_sdf_dir_input, self.ligand_sdf_btn),
        )
        form_layout.addRow(
            "Protein_PDB dir:",
            self._with_button(self.protein_pdb_dir_input, self.protein_pdb_btn),
        )
        form_layout.addRow(
            "Output Graphs:",
            self._with_button(self.graphs_dir_input, self.graphs_dir_btn),
        )

        form_layout.addRow(QLabel("<b>Optional parameters</b>"))
        form_layout.addRow("Cutoff Edges:", self.cutoff_edges_spin)
        form_layout.addRow("Cutoff Protein:", self.cutoff_prot_spin)

        layout = QVBoxLayout()
        layout.addLayout(form_layout)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.setLayout(layout)

    def _with_button(self, line_edit: QLineEdit, button: QPushButton) -> QWidget:
        """
        @brief Wrap a line edit and a button in a horizontal container.

        @param line_edit Path input widget.
        @param button Browse button.
        @return QWidget containing both widgets.
        """
        container = QWidget()
        hbox = QHBoxLayout(container)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(line_edit)
        hbox.addWidget(button)
        return container

    def _global_path_or_default(self, key: str, fallback: str) -> str:
        """
        @brief Return a path from global AppSettings or fallback.

        @param key AppSettings key.
        @param fallback Fallback value.
        @return Selected path.
        """
        value = self.app_settings.get_value(key).strip()
        return value if value else str(fallback)

    def _ednn_subpath_or_default(self, folder_name: str, fallback: str) -> str:
        """
        @brief Build an EDNN path from the configured EDNN root.

        @param folder_name Folder name inside the EDNN root.
        @param fallback Fallback value.
        @return Selected path.
        """
        ednn_root = self.app_settings.get_value("paths/ednn_root").strip()

        if ednn_root:
            return str(Path(ednn_root) / folder_name)

        return str(fallback)

    def _float_setting(self, key: str, fallback: Any) -> float:
        """
        @brief Read a floating-point value from local dialog settings.

        @param key Local settings key.
        @param fallback Fallback value.
        @return Parsed float.
        """
        try:
            return float(self.settings.value(key, fallback))
        except (TypeError, ValueError):
            return float(fallback)

    def browse_pic50(self) -> None:
        """
        @brief Browse for the pIC50 file.

        @return None.
        """
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select pIC50.txt",
            "",
            "Text Files (*.txt);;All Files (*)",
        )

        if path:
            self.pic50_file_input.setText(path)

    def browse_ligand_sdf(self) -> None:
        """
        @brief Browse for the Ligand_SDF directory.

        @return None.
        """
        path = QFileDialog.getExistingDirectory(self, "Select Ligand_SDF folder")

        if path:
            self.ligand_sdf_dir_input.setText(path)

    def browse_protein_pdb(self) -> None:
        """
        @brief Browse for the Protein_PDB directory.

        @return None.
        """
        path = QFileDialog.getExistingDirectory(self, "Select Protein_PDB folder")

        if path:
            self.protein_pdb_dir_input.setText(path)

    def browse_graphs_dir(self) -> None:
        """
        @brief Browse for the EDNN graph output directory.

        @return None.
        """
        path = QFileDialog.getExistingDirectory(self, "Select graph output folder")

        if path:
            self.graphs_dir_input.setText(path)

    def accept(self) -> None:
        """
        @brief Persist dialog values and accept the dialog.

        @return None.
        """
        self.settings.setValue("dbGen/pic50_file", self.pic50_file_input.text())
        self.settings.setValue("dbGen/ligand_sdf_dir", self.ligand_sdf_dir_input.text())
        self.settings.setValue("dbGen/protein_pdb_dir", self.protein_pdb_dir_input.text())
        self.settings.setValue("dbGen/graphs_dir", self.graphs_dir_input.text())
        self.settings.setValue("dbGen/cutoff_edges", self.cutoff_edges_spin.value())
        self.settings.setValue("dbGen/cutoff_prot", self.cutoff_prot_spin.value())

        self.app_settings.set_value("paths/pic50_file", self.pic50_file_input.text())
        self.app_settings.set_value("paths/ligand_sdf", self.ligand_sdf_dir_input.text())
        self.app_settings.set_value("paths/protein_pdb", self.protein_pdb_dir_input.text())
        self.app_settings.sync()

        super().accept()

    def get_inputs(self) -> dict[str, object]:
        """
        @brief Return dialog values as EDNN graph generation parameters.

        @return Input parameter dictionary.
        """
        return {
            "pic50_file": self.pic50_file_input.text(),
            "ligand_sdf_dir": self.ligand_sdf_dir_input.text(),
            "protein_pdb_dir": self.protein_pdb_dir_input.text(),
            "graphs_dir": self.graphs_dir_input.text(),
            "cutoff_edges": self.cutoff_edges_spin.value(),
            "cutoff_prot": self.cutoff_prot_spin.value(),
        }
