"""
@file test_dcml_dialog.py
@author Mohamed EL BOUKHIARI
@brief Evaluation dialog for the DCML module.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
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


class TestDCMLDialog(QDialog):
    """Collect parameters needed to evaluate one DCML model."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DCML Evaluation Configuration")
        self.resize(760, 460)
        self.settings = QSettings("ResearchApp", "DCML_Evaluation")

        self.model_pt_input = QLineEdit(self.settings.value("evaluation/model_pt", "DCML/results/DCML.pt"))
        self.model_pt_input.setPlaceholderText("DCML.pt")
        self.model_pt_btn = QPushButton("Browse...")
        self.model_pt_btn.clicked.connect(self.browse_model_pt)

        self.feature_zip_input = QLineEdit(self.settings.value("evaluation/feature_zip", ""))
        self.feature_zip_input.setPlaceholderText("test_feature.zip or validation_feature.zip")
        self.feature_zip_btn = QPushButton("Browse...")
        self.feature_zip_btn.clicked.connect(self.browse_feature_zip)

        self.label_npy_input = QLineEdit(self.settings.value("evaluation/label_npy", ""))
        self.label_npy_input.setPlaceholderText("test_label.npy or validation_label.npy")
        self.label_npy_btn = QPushButton("Browse...")
        self.label_npy_btn.clicked.connect(self.browse_label_npy)

        self.output_dir_input = QLineEdit(self.settings.value("evaluation/output_dir", "DCML/results/predict"))
        self.output_dir_input.setPlaceholderText("Evaluation output directory")
        self.output_dir_btn = QPushButton("Browse...")
        self.output_dir_btn.clicked.connect(self.browse_output_dir)

        self.dataset_name_input = QLineEdit(self.settings.value("evaluation/dataset_name", "test"))
        self.dataset_name_input.setPlaceholderText("test, validation, MPro-URV...")

        self.split_id_input = QLineEdit(self.settings.value("evaluation/split_id", ""))
        self.split_id_input.setPlaceholderText("Optional, e.g. 00")

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "auto", "cuda", "cuda:0", "cuda:1"])
        self.device_combo.setCurrentText(self.settings.value("evaluation/device", "cpu"))

        self.cast_float32_check = QCheckBox("Cast features to float32")
        self.cast_float32_check.setChecked(str(self.settings.value("evaluation/cast_float32", "true")).lower() in {"true", "1", "yes"})

        form_layout = QFormLayout()
        form_layout.addRow(QLabel("<b>1. Model and dataset</b>"))
        form_layout.addRow("Model bundle:", self._with_button(self.model_pt_input, self.model_pt_btn))
        form_layout.addRow("Feature ZIP:", self._with_button(self.feature_zip_input, self.feature_zip_btn))
        form_layout.addRow("Label NPY:", self._with_button(self.label_npy_input, self.label_npy_btn))
        form_layout.addRow("Output directory:", self._with_button(self.output_dir_input, self.output_dir_btn))

        form_layout.addRow(QLabel("<br><b>2. Metadata</b>"))
        form_layout.addRow("Dataset name:", self.dataset_name_input)
        form_layout.addRow("Split ID:", self.split_id_input)

        form_layout.addRow(QLabel("<br><b>3. Runtime</b>"))
        form_layout.addRow("Device:", self.device_combo)
        form_layout.addRow("Memory:", self.cast_float32_check)
        form_layout.addRow("Note:", QLabel("DCML uses scikit-learn. Inference runs on CPU even if CUDA is selected."))

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

    def browse_model_pt(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select DCML model bundle", "", "PyTorch Bundle (*.pt);;All Files (*)")
        if path:
            self.model_pt_input.setText(path)

    def browse_feature_zip(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select feature ZIP", "", "ZIP Files (*.zip);;All Files (*)")
        if path:
            self.feature_zip_input.setText(path)

    def browse_label_npy(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select label NPY", "", "NumPy Files (*.npy);;All Files (*)")
        if path:
            self.label_npy_input.setText(path)

    def browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self.output_dir_input.setText(path)

    def accept(self):
        self.settings.setValue("evaluation/model_pt", self.model_pt_input.text())
        self.settings.setValue("evaluation/feature_zip", self.feature_zip_input.text())
        self.settings.setValue("evaluation/label_npy", self.label_npy_input.text())
        self.settings.setValue("evaluation/output_dir", self.output_dir_input.text())
        self.settings.setValue("evaluation/dataset_name", self.dataset_name_input.text())
        self.settings.setValue("evaluation/split_id", self.split_id_input.text())
        self.settings.setValue("evaluation/device", self.device_combo.currentText())
        self.settings.setValue("evaluation/cast_float32", self.cast_float32_check.isChecked())
        super().accept()

    def get_inputs(self):
        split_id = self.split_id_input.text().strip() or None
        dataset_name = self.dataset_name_input.text().strip() or None
        return {
            "model_pt": self.model_pt_input.text(),
            "feature_zip": self.feature_zip_input.text(),
            "label_npy": self.label_npy_input.text(),
            "output_dir": self.output_dir_input.text(),
            "device": self.device_combo.currentText(),
            "split_id": split_id,
            "dataset_name": dataset_name,
            "cast_float32": self.cast_float32_check.isChecked(),
        }
