"""Prepared-dataset evaluation dialog for CAPLA."""

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout
from CAPLA.ui.dialogs._shared import browse_existing_directory, browse_existing_file, with_button


class TestCAPLADialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CAPLA Evaluation Configuration"); self.resize(820, 460)
        self.settings = QSettings("ResearchApp", "CAPLA_Evaluation")
        self.model_pt_input = QLineEdit(self.settings.value("evaluation/model_pt", "CAPLA/models/pretrained/best_model.pt")); self.model_pt_btn = QPushButton("Browse..."); self.model_pt_btn.clicked.connect(lambda: browse_existing_file(self, "Select CAPLA checkpoint", "PyTorch Checkpoint (*.pt);;All Files (*)", self.model_pt_input))
        self.dataset_dir_input = QLineEdit(self.settings.value("evaluation/dataset_dir", "CAPLA/data/urv_dataset_v3b_prepared")); self.dataset_dir_btn = QPushButton("Browse..."); self.dataset_dir_btn.clicked.connect(lambda: browse_existing_directory(self, "Select prepared CAPLA dataset", self.dataset_dir_input))
        self.output_dir_input = QLineEdit(self.settings.value("evaluation/output_dir", "CAPLA/outputs/predictions")); self.output_dir_btn = QPushButton("Browse..."); self.output_dir_btn.clicked.connect(lambda: browse_existing_directory(self, "Select prediction output directory", self.output_dir_input))
        self.dataset_name_input = QLineEdit(self.settings.value("evaluation/dataset_name", "URV_v3b")); self.split_id_input = QLineEdit(self.settings.value("evaluation/split_id", "")); self.split_id_input.setPlaceholderText("Optional suffix, e.g. 01")
        self.device_combo = QComboBox(); self.device_combo.addItems(["auto", "cpu", "cuda"]); self.device_combo.setCurrentText(self.settings.value("evaluation/device", "auto"))
        self.batch_size_spin = QSpinBox(); self.batch_size_spin.setRange(1, 1024); self.batch_size_spin.setValue(int(self.settings.value("evaluation/batch_size", 32)))
        self.num_workers_spin = QSpinBox(); self.num_workers_spin.setRange(0, 64); self.num_workers_spin.setValue(int(self.settings.value("evaluation/num_workers", 0)))
        form = QFormLayout(); form.addRow(QLabel("<b>1. Model and prepared dataset</b>")); form.addRow("Model checkpoint:", with_button(self.model_pt_input, self.model_pt_btn)); form.addRow("Prepared dataset:", with_button(self.dataset_dir_input, self.dataset_dir_btn)); form.addRow("Output directory:", with_button(self.output_dir_input, self.output_dir_btn)); form.addRow(QLabel("<br><b>2. Output metadata</b>")); form.addRow("Dataset label:", self.dataset_name_input); form.addRow("Optional split suffix:", self.split_id_input); form.addRow(QLabel("<br><b>3. Runtime</b>")); form.addRow("Device:", self.device_combo); form.addRow("Batch size:", self.batch_size_spin); form.addRow("DataLoader workers:", self.num_workers_spin)
        layout = QVBoxLayout(); layout.addLayout(form); buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons); self.setLayout(layout)

    def accept(self):
        values = self.get_inputs()
        for key, value in values.items(): self.settings.setValue(f"evaluation/{key}", "" if value is None else value)
        super().accept()

    def get_inputs(self):
        return {"model_pt": self.model_pt_input.text().strip(), "dataset_dir": self.dataset_dir_input.text().strip(), "output_dir": self.output_dir_input.text().strip(), "dataset_name": self.dataset_name_input.text().strip() or None, "split_id": self.split_id_input.text().strip() or None, "device": self.device_combo.currentText(), "batch_size": self.batch_size_spin.value(), "num_workers": self.num_workers_spin.value()}
