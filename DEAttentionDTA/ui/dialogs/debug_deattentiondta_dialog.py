"""Prepared-dataset validation dialog for DEAttentionDTA."""

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout

from DEAttentionDTA.ui.dialogs._shared import browse_existing_directory, with_button


class DebugDEAttentionDTADialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DEAttentionDTA Dataset Validation")
        self.resize(800, 280)
        self.settings = QSettings("ResearchApp", "DEAttentionDTA_Debug")

        self.prepared_dir_input = QLineEdit(self.settings.value("debug/prepared_dir", "DEAttentionDTA/data/urv_dataset_v3b_prepared"))
        self.prepared_dir_btn = QPushButton("Browse...")
        self.prepared_dir_btn.clicked.connect(lambda: browse_existing_directory(self, "Select prepared DEAttentionDTA dataset", self.prepared_dir_input))
        self.results_dir_input = QLineEdit(self.settings.value("debug/results_dir", "DEAttentionDTA/outputs/debug/prepared_dataset"))
        self.results_dir_btn = QPushButton("Browse...")
        self.results_dir_btn.clicked.connect(lambda: browse_existing_directory(self, "Select debug output directory", self.results_dir_input))
        self.device_combo = QComboBox(); self.device_combo.addItems(["auto", "cpu", "cuda"]); self.device_combo.setCurrentText(self.settings.value("debug/device", "auto"))
        self.batch_size_spin = QSpinBox(); self.batch_size_spin.setRange(1, 128); self.batch_size_spin.setValue(int(self.settings.value("debug/batch_size", 2)))

        form = QFormLayout()
        form.addRow(QLabel("<b>Prepared dataset sanity check</b>"))
        form.addRow("Prepared dataset:", with_button(self.prepared_dir_input, self.prepared_dir_btn))
        form.addRow("Output directory:", with_button(self.results_dir_input, self.results_dir_btn))
        form.addRow("Device:", self.device_combo)
        form.addRow("Batch size:", self.batch_size_spin)
        form.addRow("Note:", QLabel("Runs one forward pass. It does not train the model."))
        layout = QVBoxLayout(); layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addWidget(buttons); self.setLayout(layout)

    def accept(self) -> None:
        for key, value in self.get_inputs().items(): self.settings.setValue(f"debug/{key}", value)
        super().accept()

    def get_inputs(self) -> dict:
        return {"prepared_dir": self.prepared_dir_input.text().strip(), "results_dir": self.results_dir_input.text().strip(), "device": self.device_combo.currentText(), "batch_size": self.batch_size_spin.value()}
