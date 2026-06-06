"""Prepared-dataset validation dialog for CAPLA."""

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout
from CAPLA.ui.dialogs._shared import browse_existing_directory, with_button


class DebugCAPLADialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CAPLA Dataset Validation")
        self.resize(760, 260)
        self.settings = QSettings("ResearchApp", "CAPLA_Debug")

        self.dataset_dir_input = QLineEdit(self.settings.value("debug/dataset_dir", "CAPLA/data/urv_dataset_v3b_prepared"))
        self.dataset_dir_btn = QPushButton("Browse...")
        self.dataset_dir_btn.clicked.connect(lambda: browse_existing_directory(self, "Select prepared CAPLA dataset", self.dataset_dir_input))

        self.output_dir_input = QLineEdit(self.settings.value("debug/output_dir", "CAPLA/outputs/from_scratch"))
        self.output_dir_btn = QPushButton("Browse...")
        self.output_dir_btn.clicked.connect(lambda: browse_existing_directory(self, "Select debug output directory", self.output_dir_input))

        self.device_combo = QComboBox(); self.device_combo.addItems(["auto", "cpu", "cuda"])
        self.device_combo.setCurrentText(self.settings.value("debug/device", "auto"))
        self.batch_size_spin = QSpinBox(); self.batch_size_spin.setRange(1, 1024)
        self.batch_size_spin.setValue(int(self.settings.value("debug/batch_size", 2)))

        form = QFormLayout()
        form.addRow(QLabel("<b>Prepared dataset sanity check</b>"))
        form.addRow("Prepared dataset:", with_button(self.dataset_dir_input, self.dataset_dir_btn))
        form.addRow("Output directory:", with_button(self.output_dir_input, self.output_dir_btn))
        form.addRow("Device:", self.device_combo)
        form.addRow("Batch size:", self.batch_size_spin)
        form.addRow("Note:", QLabel("Runs a short forward pass. It does not train the model."))
        layout = QVBoxLayout(); layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addWidget(buttons); self.setLayout(layout)

    def accept(self):
        values = self.get_inputs()
        for key, value in values.items(): self.settings.setValue(f"debug/{key}", value)
        super().accept()

    def get_inputs(self):
        return {"dataset_dir": self.dataset_dir_input.text().strip(), "output_dir": self.output_dir_input.text().strip(), "device": self.device_combo.currentText(), "batch_size": self.batch_size_spin.value()}
