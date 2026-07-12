"""
@file hyperparameter_search_dcml_dialog.py
@brief Search dialog for the DCML module.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)


class HyperparameterSearchDCMLDialog(QDialog):
    """Collect parameters for DCML model-parameter search."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DCML Search")
        self.resize(900, 850)
        self.setMinimumSize(760, 620)
        self.settings = QSettings("ResearchApp", "DCML_Search")

        self.prepared_root_input = QLineEdit(self.settings.value("search/prepared_root", "DCML/datasets"))
        self.prepared_root_btn = QPushButton("Browse...")
        self.prepared_root_btn.clicked.connect(lambda: self._browse_dir("Select prepared feature root", self.prepared_root_input))

        self.output_root_input = QLineEdit(self.settings.value("search/output_root", "DCML/results/search"))
        self.output_root_btn = QPushButton("Browse...")
        self.output_root_btn.clicked.connect(lambda: self._browse_dir("Select output root", self.output_root_input))

        self.labels_input = QLineEdit(self.settings.value("search/labels_path", ""))
        self.labels_btn = QPushButton("Browse...")
        self.labels_btn.clicked.connect(lambda: self._browse_file("Select labels file", "Data Files (*.npy *.csv);;All Files (*)", self.labels_input))

        self.sample_ids_input = QLineEdit(self.settings.value("search/sample_ids_path", ""))
        self.sample_ids_btn = QPushButton("Browse...")
        self.sample_ids_btn.clicked.connect(lambda: self._browse_file("Select sample IDs file", "CSV Files (*.csv);;All Files (*)", self.sample_ids_input))

        self.variant_combo = QComboBox(); self.variant_combo.addItems(["distance_only", "real_charge", "full"]); self.variant_combo.setCurrentText(self.settings.value("search/variant", "distance_only"))
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(0, 999999); self.seed_spin.setValue(int(self.settings.value("search/seed", 42)))
        self.fold_spin = QSpinBox(); self.fold_spin.setRange(0, 9999); self.fold_spin.setValue(int(self.settings.value("search/fold_index", 0)))
        self.fold_spin.setToolTip("Index of the official dataset split to use, usually 0 to 4.")

        self.n_estimators_values_input = QLineEdit(self.settings.value("search/n_estimators_values", "100,300"))
        self.max_depth_values_input = QLineEdit(self.settings.value("search/max_depth_values", "3,6"))
        self.learning_rate_values_input = QLineEdit(self.settings.value("search/learning_rate_values", "0.01,0.05"))
        self.min_samples_split_values_input = QLineEdit(self.settings.value("search/min_samples_split_values", "2"))
        self.subsample_values_input = QLineEdit(self.settings.value("search/subsample_values", "0.7,1.0"))
        self.max_features_values_input = QLineEdit(self.settings.value("search/max_features_values", "sqrt,none"))
        self.loss_values_input = QLineEdit(self.settings.value("search/loss_values", "squared_error"))

        form = QFormLayout()
        form.addRow("Prepared feature root:", self._with_button(self.prepared_root_input, self.prepared_root_btn))
        form.addRow("Output root:", self._with_button(self.output_root_input, self.output_root_btn))
        form.addRow("Labels path:", self._with_button(self.labels_input, self.labels_btn))
        form.addRow("Sample IDs path:", self._with_button(self.sample_ids_input, self.sample_ids_btn))
        form.addRow("Variant:", self.variant_combo)
        form.addRow("Seed:", self.seed_spin)
        form.addRow("Split/Fold index:", self.fold_spin)
        form.addRow("Selection rule:", QLabel("min validation RMSE, then max validation Pearson; test metrics are not used."))
        form.addRow(QLabel("<b>Search space</b>"))
        form.addRow("n_estimators:", self.n_estimators_values_input)
        form.addRow("max_depth:", self.max_depth_values_input)
        form.addRow("learning_rate:", self.learning_rate_values_input)
        form.addRow("min_samples_split:", self.min_samples_split_values_input)
        form.addRow("subsample:", self.subsample_values_input)
        form.addRow("max_features:", self.max_features_values_input)
        form.addRow("loss:", self.loss_values_input)

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
        for key, widget in (("prepared_root", self.prepared_root_input), ("output_root", self.output_root_input), ("labels_path", self.labels_input), ("sample_ids_path", self.sample_ids_input)):
            self.settings.setValue(f"search/{key}", widget.text())
        self.settings.setValue("search/variant", self.variant_combo.currentText())
        self.settings.setValue("search/seed", self.seed_spin.value())
        self.settings.setValue("search/fold_index", self.fold_spin.value())
        self.settings.setValue("search/n_estimators_values", self.n_estimators_values_input.text())
        self.settings.setValue("search/max_depth_values", self.max_depth_values_input.text())
        self.settings.setValue("search/learning_rate_values", self.learning_rate_values_input.text())
        self.settings.setValue("search/min_samples_split_values", self.min_samples_split_values_input.text())
        self.settings.setValue("search/subsample_values", self.subsample_values_input.text())
        self.settings.setValue("search/max_features_values", self.max_features_values_input.text())
        self.settings.setValue("search/loss_values", self.loss_values_input.text())
        super().accept()

    @staticmethod
    def _parse_int_list(raw_text: str) -> list[int]: return [int(value.strip()) for value in raw_text.split(",") if value.strip()]
    @staticmethod
    def _parse_float_list(raw_text: str) -> list[float]: return [float(value.strip()) for value in raw_text.split(",") if value.strip()]
    @staticmethod
    def _parse_string_list(raw_text: str) -> list[str]: return [value.strip() for value in raw_text.split(",") if value.strip()]
    @staticmethod
    def _parse_max_features_list(raw_text: str):
        values = []
        for value in raw_text.split(","):
            text = value.strip()
            if text: values.append(None if text.lower() in {"none", "null"} else text)
        return values

    def get_inputs(self):
        return {
            "prepared_feature_root": self.prepared_root_input.text(),
            "output_root": self.output_root_input.text(),
            "labels_path": self.labels_input.text().strip() or None,
            "sample_ids_path": self.sample_ids_input.text().strip() or None,
            "variant": self.variant_combo.currentText(),
            "seed": self.seed_spin.value(),
            "fold_index": self.fold_spin.value(),
            "use_dataset_folds": True,
            "cast_float32": True,
            "n_estimators_values": self._parse_int_list(self.n_estimators_values_input.text()),
            "max_depth_values": self._parse_int_list(self.max_depth_values_input.text()),
            "learning_rate_values": self._parse_float_list(self.learning_rate_values_input.text()),
            "min_samples_split_values": self._parse_int_list(self.min_samples_split_values_input.text()),
            "subsample_values": self._parse_float_list(self.subsample_values_input.text()),
            "max_features_values": self._parse_max_features_list(self.max_features_values_input.text()),
            "loss_values": self._parse_string_list(self.loss_values_input.text()),
        }
