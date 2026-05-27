"""
@file hyperparameter_search_dcml_dialog.py
@author Mohamed EL BOUKHIARI
@brief Hyperparameter-search dialog for the DCML module.
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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class HyperparameterSearchDCMLDialog(QDialog):
    """Collect parameters for DCML hyperparameter search."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DCML Hyperparameter Search Configuration")
        self.resize(820, 660)
        self.settings = QSettings("ResearchApp", "DCML_HyperparameterSearch")

        self.train_feature_zip_input = QLineEdit(self.settings.value("search/train_feature_zip", ""))
        self.train_feature_zip_btn = QPushButton("Browse...")
        self.train_feature_zip_btn.clicked.connect(self.browse_train_feature_zip)

        self.train_label_npy_input = QLineEdit(self.settings.value("search/train_label_npy", ""))
        self.train_label_npy_btn = QPushButton("Browse...")
        self.train_label_npy_btn.clicked.connect(self.browse_train_label_npy)

        self.validation_feature_zip_input = QLineEdit(self.settings.value("search/validation_feature_zip", ""))
        self.validation_feature_zip_btn = QPushButton("Browse...")
        self.validation_feature_zip_btn.clicked.connect(self.browse_validation_feature_zip)

        self.validation_label_npy_input = QLineEdit(self.settings.value("search/validation_label_npy", ""))
        self.validation_label_npy_btn = QPushButton("Browse...")
        self.validation_label_npy_btn.clicked.connect(self.browse_validation_label_npy)

        self.models_root_input = QLineEdit(self.settings.value("search/models_root", "DCML/results/hpo_models"))
        self.models_root_btn = QPushButton("Browse...")
        self.models_root_btn.clicked.connect(self.browse_models_root)

        self.results_root_input = QLineEdit(self.settings.value("search/results_root", "DCML/results/hpo"))
        self.results_root_btn = QPushButton("Browse...")
        self.results_root_btn.clicked.connect(self.browse_results_root)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "auto", "cuda", "cuda:0", "cuda:1"])
        self.device_combo.setCurrentText(self.settings.value("search/device", "cpu"))

        self.cast_float32_check = QCheckBox("Cast features to float32")
        self.cast_float32_check.setChecked(str(self.settings.value("search/cast_float32", "true")).lower() in {"true", "1", "yes"})

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(int(self.settings.value("search/seed", 42)))

        self.n_estimators_values_input = QLineEdit(self.settings.value("search/n_estimators_values", "100,300"))
        self.max_depth_values_input = QLineEdit(self.settings.value("search/max_depth_values", "3,6"))
        self.learning_rate_values_input = QLineEdit(self.settings.value("search/learning_rate_values", "0.01,0.05"))
        self.min_samples_split_values_input = QLineEdit(self.settings.value("search/min_samples_split_values", "2"))
        self.subsample_values_input = QLineEdit(self.settings.value("search/subsample_values", "0.7,1.0"))
        self.max_features_values_input = QLineEdit(self.settings.value("search/max_features_values", "sqrt,none"))
        self.loss_values_input = QLineEdit(self.settings.value("search/loss_values", "squared_error"))

        form_layout = QFormLayout()
        form_layout.addRow(QLabel("<b>1. Train/validation datasets</b>"))
        form_layout.addRow("Train feature ZIP:", self._with_button(self.train_feature_zip_input, self.train_feature_zip_btn))
        form_layout.addRow("Train label NPY:", self._with_button(self.train_label_npy_input, self.train_label_npy_btn))
        form_layout.addRow("Validation feature ZIP:", self._with_button(self.validation_feature_zip_input, self.validation_feature_zip_btn))
        form_layout.addRow("Validation label NPY:", self._with_button(self.validation_label_npy_input, self.validation_label_npy_btn))

        form_layout.addRow(QLabel("<br><b>2. Outputs</b>"))
        form_layout.addRow("Trial models root:", self._with_button(self.models_root_input, self.models_root_btn))
        form_layout.addRow("Results root:", self._with_button(self.results_root_input, self.results_root_btn))

        form_layout.addRow(QLabel("<br><b>3. Runtime</b>"))
        form_layout.addRow("Device:", self.device_combo)
        form_layout.addRow("Memory:", self.cast_float32_check)
        form_layout.addRow("Random seed:", self.seed_spin)
        form_layout.addRow("Selection rule:", QLabel("min validation RMSE, then max validation Pearson"))

        form_layout.addRow(QLabel("<br><b>4. Search space</b>"))
        form_layout.addRow("n_estimators:", self.n_estimators_values_input)
        form_layout.addRow("max_depth:", self.max_depth_values_input)
        form_layout.addRow("learning_rate:", self.learning_rate_values_input)
        form_layout.addRow("min_samples_split:", self.min_samples_split_values_input)
        form_layout.addRow("subsample:", self.subsample_values_input)
        form_layout.addRow("max_features:", self.max_features_values_input)
        form_layout.addRow("loss:", self.loss_values_input)

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

    def _browse_file(self, title: str, file_filter: str, target: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(self, title, "", file_filter)
        if path:
            target.setText(path)

    def browse_train_feature_zip(self):
        self._browse_file("Select train feature ZIP", "ZIP Files (*.zip);;All Files (*)", self.train_feature_zip_input)

    def browse_train_label_npy(self):
        self._browse_file("Select train label NPY", "NumPy Files (*.npy);;All Files (*)", self.train_label_npy_input)

    def browse_validation_feature_zip(self):
        self._browse_file("Select validation feature ZIP", "ZIP Files (*.zip);;All Files (*)", self.validation_feature_zip_input)

    def browse_validation_label_npy(self):
        self._browse_file("Select validation label NPY", "NumPy Files (*.npy);;All Files (*)", self.validation_label_npy_input)

    def browse_models_root(self):
        path = QFileDialog.getExistingDirectory(self, "Select models root directory")
        if path:
            self.models_root_input.setText(path)

    def browse_results_root(self):
        path = QFileDialog.getExistingDirectory(self, "Select results root directory")
        if path:
            self.results_root_input.setText(path)

    def accept(self):
        self.settings.setValue("search/train_feature_zip", self.train_feature_zip_input.text())
        self.settings.setValue("search/train_label_npy", self.train_label_npy_input.text())
        self.settings.setValue("search/validation_feature_zip", self.validation_feature_zip_input.text())
        self.settings.setValue("search/validation_label_npy", self.validation_label_npy_input.text())
        self.settings.setValue("search/models_root", self.models_root_input.text())
        self.settings.setValue("search/results_root", self.results_root_input.text())
        self.settings.setValue("search/device", self.device_combo.currentText())
        self.settings.setValue("search/cast_float32", self.cast_float32_check.isChecked())
        self.settings.setValue("search/seed", self.seed_spin.value())
        self.settings.setValue("search/n_estimators_values", self.n_estimators_values_input.text())
        self.settings.setValue("search/max_depth_values", self.max_depth_values_input.text())
        self.settings.setValue("search/learning_rate_values", self.learning_rate_values_input.text())
        self.settings.setValue("search/min_samples_split_values", self.min_samples_split_values_input.text())
        self.settings.setValue("search/subsample_values", self.subsample_values_input.text())
        self.settings.setValue("search/max_features_values", self.max_features_values_input.text())
        self.settings.setValue("search/loss_values", self.loss_values_input.text())
        super().accept()

    @staticmethod
    def _parse_int_list(raw_text: str) -> list[int]:
        return [int(value.strip()) for value in raw_text.split(",") if value.strip()]

    @staticmethod
    def _parse_float_list(raw_text: str) -> list[float]:
        return [float(value.strip()) for value in raw_text.split(",") if value.strip()]

    @staticmethod
    def _parse_string_list(raw_text: str) -> list[str]:
        return [value.strip() for value in raw_text.split(",") if value.strip()]

    @staticmethod
    def _parse_max_features_list(raw_text: str):
        values = []
        for value in raw_text.split(","):
            text = value.strip()
            if not text:
                continue
            values.append(None if text.lower() in {"none", "null"} else text)
        return values

    def get_inputs(self):
        return {
            "train_feature_zip": self.train_feature_zip_input.text(),
            "train_label_npy": self.train_label_npy_input.text(),
            "validation_feature_zip": self.validation_feature_zip_input.text(),
            "validation_label_npy": self.validation_label_npy_input.text(),
            "models_root": self.models_root_input.text(),
            "results_root": self.results_root_input.text(),
            "device": self.device_combo.currentText(),
            "seed": self.seed_spin.value(),
            "cast_float32": self.cast_float32_check.isChecked(),
            "n_estimators_values": self._parse_int_list(self.n_estimators_values_input.text()),
            "max_depth_values": self._parse_int_list(self.max_depth_values_input.text()),
            "learning_rate_values": self._parse_float_list(self.learning_rate_values_input.text()),
            "min_samples_split_values": self._parse_int_list(self.min_samples_split_values_input.text()),
            "subsample_values": self._parse_float_list(self.subsample_values_input.text()),
            "max_features_values": self._parse_max_features_list(self.max_features_values_input.text()),
            "loss_values": self._parse_string_list(self.loss_values_input.text()),
        }
