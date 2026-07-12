"""
@file test_dcml_dialog.py
@brief Evaluation dialog for the DCML module.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)


class TestDCMLDialog(QDialog):
    """Collect parameters needed to evaluate one DCML model."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DCML Evaluate")
        self.resize(760, 440)
        self.settings = QSettings("ResearchApp", "DCML_Evaluation")

        self.prepared_root_input = QLineEdit(self.settings.value("evaluation/prepared_root", "DCML/datasets"))
        self.prepared_root_btn = QPushButton("Browse...")
        self.prepared_root_btn.clicked.connect(lambda: self._browse_dir("Select prepared feature root", self.prepared_root_input))

        self.model_pt_input = QLineEdit(self.settings.value("evaluation/model_pt", "DCML/results/train/DCML.pt"))
        self.model_pt_btn = QPushButton("Browse...")
        self.model_pt_btn.clicked.connect(lambda: self._browse_file("Select DCML model bundle", "PyTorch Bundle (*.pt);;All Files (*)", self.model_pt_input))

        self.output_dir_input = QLineEdit(self.settings.value("evaluation/output_dir", "DCML/results/evaluate"))
        self.output_dir_btn = QPushButton("Browse...")
        self.output_dir_btn.clicked.connect(lambda: self._browse_dir("Select output directory", self.output_dir_input))

        self.labels_input = QLineEdit(self.settings.value("evaluation/labels_path", ""))
        self.labels_btn = QPushButton("Browse...")
        self.labels_btn.clicked.connect(lambda: self._browse_file("Select labels file", "Data Files (*.npy *.csv);;All Files (*)", self.labels_input))

        self.sample_ids_input = QLineEdit(self.settings.value("evaluation/sample_ids_path", ""))
        self.sample_ids_btn = QPushButton("Browse...")
        self.sample_ids_btn.clicked.connect(lambda: self._browse_file("Select sample IDs file", "CSV Files (*.csv);;All Files (*)", self.sample_ids_input))

        self.variant_combo = QComboBox(); self.variant_combo.addItems(["distance_only", "real_charge", "full"]); self.variant_combo.setCurrentText(self.settings.value("evaluation/variant", "distance_only"))
        self.split_combo = QComboBox(); self.split_combo.addItems(["train", "valid", "test", "all"]); self.split_combo.setCurrentText(self.settings.value("evaluation/split", "test"))
        self.fold_spin = QSpinBox(); self.fold_spin.setRange(0, 9999); self.fold_spin.setValue(int(self.settings.value("evaluation/fold_index", 0)))
        self.fold_spin.setToolTip("Index of the official dataset split to use, usually 0 to 4.")

        form = QFormLayout()
        form.addRow("Prepared feature root:", self._with_button(self.prepared_root_input, self.prepared_root_btn))
        form.addRow("Model/checkpoint path:", self._with_button(self.model_pt_input, self.model_pt_btn))
        form.addRow("Output directory:", self._with_button(self.output_dir_input, self.output_dir_btn))
        form.addRow("Labels path:", self._with_button(self.labels_input, self.labels_btn))
        form.addRow("Sample IDs path:", self._with_button(self.sample_ids_input, self.sample_ids_btn))
        form.addRow("Variant:", self.variant_combo)
        form.addRow("Split/Fold index:", self.fold_spin)
        form.addRow("Split:", self.split_combo)
        form.addRow("Note:", QLabel("Metrics are computed only on the selected split."))

        layout = QVBoxLayout(); layout.addLayout(form)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept); self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons); self.setLayout(layout)

    def _with_button(self, line_edit, button):
        container = QWidget(); hbox = QHBoxLayout(container); hbox.setContentsMargins(0, 0, 0, 0); hbox.addWidget(line_edit); hbox.addWidget(button); return container

    def _browse_dir(self, title: str, target: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, title)
        if path: target.setText(path)

    def _browse_file(self, title: str, file_filter: str, target: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(self, title, "", file_filter)
        if path: target.setText(path)

    def accept(self):
        for key, widget in (("prepared_root", self.prepared_root_input), ("model_pt", self.model_pt_input), ("output_dir", self.output_dir_input), ("labels_path", self.labels_input), ("sample_ids_path", self.sample_ids_input)):
            self.settings.setValue(f"evaluation/{key}", widget.text())
        self.settings.setValue("evaluation/variant", self.variant_combo.currentText())
        self.settings.setValue("evaluation/split", self.split_combo.currentText())
        self.settings.setValue("evaluation/fold_index", self.fold_spin.value())
        super().accept()

    def get_inputs(self):
        return {
            "prepared_feature_root": self.prepared_root_input.text(),
            "model_pt": self.model_pt_input.text(),
            "output_dir": self.output_dir_input.text(),
            "labels_path": self.labels_input.text().strip() or None,
            "sample_ids_path": self.sample_ids_input.text().strip() or None,
            "variant": self.variant_combo.currentText(),
            "fold_index": self.fold_spin.value(),
            "split": self.split_combo.currentText(),
            "cast_float32": True,
        }
