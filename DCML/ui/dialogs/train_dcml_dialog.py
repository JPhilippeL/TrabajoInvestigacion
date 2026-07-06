"""
@file train_dcml_dialog.py
@author Mohamed EL BOUKHIARI
@brief Training dialog for the DCML module.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class TrainDCMLDialog(QDialog):
    """Collect all parameters needed to train one DCML model."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DCML Training Configuration")
        self.resize(760, 560)
        self.settings = QSettings("ResearchApp", "DCML_Training")

        self.train_feature_zip_input = QLineEdit(self.settings.value("training/train_feature_zip", ""))
        self.train_feature_zip_input.setPlaceholderText("train_feature.zip")
        self.train_feature_zip_btn = QPushButton("Browse...")
        self.train_feature_zip_btn.clicked.connect(self.browse_train_feature_zip)

        self.train_label_npy_input = QLineEdit(self.settings.value("training/train_label_npy", ""))
        self.train_label_npy_input.setPlaceholderText("train_label.npy")
        self.train_label_npy_btn = QPushButton("Browse...")
        self.train_label_npy_btn.clicked.connect(self.browse_train_label_npy)

        self.output_model_input = QLineEdit(self.settings.value("training/output_model", "DCML/results/DCML.pt"))
        self.output_model_input.setPlaceholderText("Output DCML.pt")
        self.output_model_btn = QPushButton("Browse...")
        self.output_model_btn.clicked.connect(self.browse_output_model)

        self.output_dir_input = QLineEdit(self.settings.value("training/output_dir", "DCML/results/train"))
        self.output_dir_input.setPlaceholderText("Output directory for summaries")
        self.output_dir_btn = QPushButton("Browse...")
        self.output_dir_btn.clicked.connect(self.browse_output_dir)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "auto", "cuda", "cuda:0", "cuda:1"])
        self.device_combo.setCurrentText(self.settings.value("training/device", "cpu"))

        self.cast_float32_check = QCheckBox("Cast features to float32")
        self.cast_float32_check.setChecked(str(self.settings.value("training/cast_float32", "true")).lower() in {"true", "1", "yes"})

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(int(self.settings.value("training/seed", 42)))

        self.n_estimators_spin = QSpinBox()
        self.n_estimators_spin.setRange(1, 10000)
        self.n_estimators_spin.setValue(int(self.settings.value("training/n_estimators", 300)))

        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setRange(1, 100)
        self.max_depth_spin.setValue(int(self.settings.value("training/max_depth", 6)))

        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setDecimals(6)
        self.learning_rate_spin.setRange(0.000001, 10.0)
        self.learning_rate_spin.setSingleStep(0.01)
        self.learning_rate_spin.setValue(float(self.settings.value("training/learning_rate", 0.01)))

        self.min_samples_split_spin = QSpinBox()
        self.min_samples_split_spin.setRange(2, 10000)
        self.min_samples_split_spin.setValue(int(self.settings.value("training/min_samples_split", 2)))

        self.subsample_spin = QDoubleSpinBox()
        self.subsample_spin.setDecimals(3)
        self.subsample_spin.setRange(0.001, 1.0)
        self.subsample_spin.setSingleStep(0.05)
        self.subsample_spin.setValue(float(self.settings.value("training/subsample", 0.7)))

        self.max_features_combo = QComboBox()
        self.max_features_combo.addItems(["sqrt", "log2", "none"])
        self.max_features_combo.setCurrentText(self.settings.value("training/max_features", "sqrt"))

        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["squared_error", "absolute_error", "huber", "quantile"])
        self.loss_combo.setCurrentText(self.settings.value("training/loss", "squared_error"))

        form_layout = QFormLayout()
        form_layout.addRow(QLabel("<b>1. Dataset and outputs</b>"))
        form_layout.addRow("Train feature ZIP:", self._with_button(self.train_feature_zip_input, self.train_feature_zip_btn))
        form_layout.addRow("Train label NPY:", self._with_button(self.train_label_npy_input, self.train_label_npy_btn))
        form_layout.addRow("Output model:", self._with_button(self.output_model_input, self.output_model_btn))
        form_layout.addRow("Output directory:", self._with_button(self.output_dir_input, self.output_dir_btn))

        form_layout.addRow(QLabel("<br><b>2. Runtime</b>"))
        form_layout.addRow("Device:", self.device_combo)
        form_layout.addRow("Memory:", self.cast_float32_check)
        form_layout.addRow("Random seed:", self.seed_spin)
        form_layout.addRow("Note:", QLabel("DCML uses scikit-learn. Training runs on CPU even if CUDA is selected."))

        form_layout.addRow(QLabel("<br><b>3. GradientBoostingRegressor hyperparameters</b>"))
        form_layout.addRow("n_estimators:", self.n_estimators_spin)
        form_layout.addRow("max_depth:", self.max_depth_spin)
        form_layout.addRow("learning_rate:", self.learning_rate_spin)
        form_layout.addRow("min_samples_split:", self.min_samples_split_spin)
        form_layout.addRow("subsample:", self.subsample_spin)
        form_layout.addRow("max_features:", self.max_features_combo)
        form_layout.addRow("loss:", self.loss_combo)

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

    def browse_train_feature_zip(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select train feature ZIP", "", "ZIP Files (*.zip);;All Files (*)")
        if path:
            self.train_feature_zip_input.setText(path)

    def browse_train_label_npy(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select train label NPY", "", "NumPy Files (*.npy);;All Files (*)")
        if path:
            self.train_label_npy_input.setText(path)

    def browse_output_model(self):
        path, _ = QFileDialog.getSaveFileName(self, "Select output model", self.output_model_input.text(), "PyTorch Bundle (*.pt);;All Files (*)")
        if path:
            self.output_model_input.setText(path)

    def browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self.output_dir_input.setText(path)

    def accept(self):
        self.settings.setValue("training/train_feature_zip", self.train_feature_zip_input.text())
        self.settings.setValue("training/train_label_npy", self.train_label_npy_input.text())
        self.settings.setValue("training/output_model", self.output_model_input.text())
        self.settings.setValue("training/output_dir", self.output_dir_input.text())
        self.settings.setValue("training/device", self.device_combo.currentText())
        self.settings.setValue("training/cast_float32", self.cast_float32_check.isChecked())
        self.settings.setValue("training/seed", self.seed_spin.value())
        self.settings.setValue("training/n_estimators", self.n_estimators_spin.value())
        self.settings.setValue("training/max_depth", self.max_depth_spin.value())
        self.settings.setValue("training/learning_rate", self.learning_rate_spin.value())
        self.settings.setValue("training/min_samples_split", self.min_samples_split_spin.value())
        self.settings.setValue("training/subsample", self.subsample_spin.value())
        self.settings.setValue("training/max_features", self.max_features_combo.currentText())
        self.settings.setValue("training/loss", self.loss_combo.currentText())
        super().accept()

    def get_inputs(self):
        max_features = self.max_features_combo.currentText()
        return {
            "train_feature_zip": self.train_feature_zip_input.text(),
            "train_label_npy": self.train_label_npy_input.text(),
            "output_model": self.output_model_input.text(),
            "output_dir": self.output_dir_input.text(),
            "device": self.device_combo.currentText(),
            "seed": self.seed_spin.value(),
            "cast_float32": self.cast_float32_check.isChecked(),
            "hyperparameters": {
                "n_estimators": self.n_estimators_spin.value(),
                "max_depth": self.max_depth_spin.value(),
                "learning_rate": self.learning_rate_spin.value(),
                "min_samples_split": self.min_samples_split_spin.value(),
                "subsample": self.subsample_spin.value(),
                "max_features": None if max_features == "none" else max_features,
                "loss": self.loss_combo.currentText(),
            },
        }
