"""
@file train_dcml_dialog.py
@brief Training dialog for the DCML module.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)


class TrainDCMLDialog(QDialog):
    """Collect parameters needed to train one DCML model."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DCML Train")
        self.resize(760, 560)
        self.settings = QSettings("ResearchApp", "DCML_Training")

        self.prepared_root_input = QLineEdit(self.settings.value("training/prepared_root", "DCML/datasets"))
        self.prepared_root_btn = QPushButton("Browse...")
        self.prepared_root_btn.clicked.connect(lambda: self._browse_dir("Select prepared feature root", self.prepared_root_input))

        self.labels_input = QLineEdit(self.settings.value("training/labels_path", ""))
        self.labels_btn = QPushButton("Browse...")
        self.labels_btn.clicked.connect(lambda: self._browse_file("Select labels file", "Data Files (*.npy *.csv);;All Files (*)", self.labels_input))

        self.sample_ids_input = QLineEdit(self.settings.value("training/sample_ids_path", ""))
        self.sample_ids_btn = QPushButton("Browse...")
        self.sample_ids_btn.clicked.connect(lambda: self._browse_file("Select sample IDs file", "CSV Files (*.csv);;All Files (*)", self.sample_ids_input))

        self.output_dir_input = QLineEdit(self.settings.value("training/output_dir", "DCML/results/train"))
        self.output_dir_btn = QPushButton("Browse...")
        self.output_dir_btn.clicked.connect(lambda: self._browse_dir("Select output directory", self.output_dir_input))

        self.variant_combo = QComboBox()
        self.variant_combo.addItems(["distance_only", "real_charge", "full"])
        self.variant_combo.setCurrentText(self.settings.value("training/variant", "distance_only"))

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(int(self.settings.value("training/seed", 42)))

        self.fold_spin = QSpinBox()
        self.fold_spin.setRange(0, 9999)
        self.fold_spin.setValue(int(self.settings.value("training/fold_index", 0)))
        self.fold_spin.setToolTip("Index of the official dataset split to use, usually 0 to 4.")

        self.n_estimators_spin = QSpinBox(); self.n_estimators_spin.setRange(1, 10000); self.n_estimators_spin.setValue(int(self.settings.value("training/n_estimators", 300)))
        self.max_depth_spin = QSpinBox(); self.max_depth_spin.setRange(1, 100); self.max_depth_spin.setValue(int(self.settings.value("training/max_depth", 6)))
        self.learning_rate_spin = QDoubleSpinBox(); self.learning_rate_spin.setDecimals(6); self.learning_rate_spin.setRange(0.000001, 10.0); self.learning_rate_spin.setSingleStep(0.01); self.learning_rate_spin.setValue(float(self.settings.value("training/learning_rate", 0.01)))
        self.min_samples_split_spin = QSpinBox(); self.min_samples_split_spin.setRange(2, 10000); self.min_samples_split_spin.setValue(int(self.settings.value("training/min_samples_split", 2)))
        self.subsample_spin = QDoubleSpinBox(); self.subsample_spin.setDecimals(3); self.subsample_spin.setRange(0.001, 1.0); self.subsample_spin.setSingleStep(0.05); self.subsample_spin.setValue(float(self.settings.value("training/subsample", 0.7)))
        self.max_features_combo = QComboBox(); self.max_features_combo.addItems(["sqrt", "log2", "none"]); self.max_features_combo.setCurrentText(self.settings.value("training/max_features", "sqrt"))
        self.loss_combo = QComboBox(); self.loss_combo.addItems(["squared_error", "absolute_error", "huber", "quantile"]); self.loss_combo.setCurrentText(self.settings.value("training/loss", "squared_error"))

        form = QFormLayout()
        form.addRow("Prepared feature root:", self._with_button(self.prepared_root_input, self.prepared_root_btn))
        form.addRow("Labels path:", self._with_button(self.labels_input, self.labels_btn))
        form.addRow("Sample IDs path:", self._with_button(self.sample_ids_input, self.sample_ids_btn))
        form.addRow("Output directory:", self._with_button(self.output_dir_input, self.output_dir_btn))
        form.addRow("Variant:", self.variant_combo)
        form.addRow("Seed:", self.seed_spin)
        form.addRow("Split/Fold index:", self.fold_spin)
        form.addRow("Note:", QLabel("DCML uses scikit-learn GradientBoostingRegressor on prepared feature matrices."))
        form.addRow(QLabel("<b>Model parameters</b>"))
        form.addRow("n_estimators:", self.n_estimators_spin)
        form.addRow("max_depth:", self.max_depth_spin)
        form.addRow("learning_rate:", self.learning_rate_spin)
        form.addRow("min_samples_split:", self.min_samples_split_spin)
        form.addRow("subsample:", self.subsample_spin)
        form.addRow("max_features:", self.max_features_combo)
        form.addRow("loss:", self.loss_combo)

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
        self.settings.setValue("training/prepared_root", self.prepared_root_input.text())
        self.settings.setValue("training/labels_path", self.labels_input.text())
        self.settings.setValue("training/sample_ids_path", self.sample_ids_input.text())
        self.settings.setValue("training/output_dir", self.output_dir_input.text())
        self.settings.setValue("training/variant", self.variant_combo.currentText())
        self.settings.setValue("training/seed", self.seed_spin.value())
        self.settings.setValue("training/fold_index", self.fold_spin.value())
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
            "prepared_feature_root": self.prepared_root_input.text(),
            "labels_path": self.labels_input.text().strip() or None,
            "sample_ids_path": self.sample_ids_input.text().strip() or None,
            "output_dir": self.output_dir_input.text(),
            "variant": self.variant_combo.currentText(),
            "seed": self.seed_spin.value(),
            "fold_index": self.fold_spin.value(),
            "use_dataset_folds": True,
            "cast_float32": True,
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
