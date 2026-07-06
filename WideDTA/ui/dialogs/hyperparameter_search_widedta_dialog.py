"""
@file hyperparameter_search_widedta_dialog.py
@author Mohamed EL BOUKHIARI
@brief Hyperparameter search configuration dialog for the WideDTA module.
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

from WideDTA.utils.constants import (
    DEFAULT_BATCH_SIZE_VALUES,
    DEFAULT_DATASET,
    DEFAULT_DEVICE,
    DEFAULT_DROPOUT_VALUES,
    DEFAULT_EPOCHS,
    DEFAULT_LR_VALUES,
    DEFAULT_MAX_TRAIN_BATCHES,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEED,
    DEFAULT_TEST_SPLIT,
    DEFAULT_VAL_SPLIT,
)


class HyperparameterSearchWideDTADialog(QDialog):
    """
    @brief Dialog used to configure WideDTA hyperparameter search.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("WideDTA Hyperparameter Search Configuration")
        self.resize(760, 600)

        self.settings = QSettings("ResearchApp", "WideDTA_HyperparameterSearch")

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["mpro_urv", "davis", "kiba"])
        self.dataset_combo.setCurrentText(self.settings.value("search/dataset_name", DEFAULT_DATASET))

        self.output_root_input = QLineEdit()
        self.output_root_input.setPlaceholderText("Root directory for WideDTA HPO runs")
        self.output_root_input.setText(self.settings.value("search/output_root", DEFAULT_OUTPUT_ROOT))

        self.output_root_btn = QPushButton("Browse...")
        self.output_root_btn.clicked.connect(self.browse_output_root)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu", "cuda:0", "cuda:1"])
        self.device_combo.setCurrentText(self.settings.value("search/device", DEFAULT_DEVICE))

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(int(self.settings.value("search/seed", DEFAULT_SEED)))

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 5000)
        self.epochs_spin.setValue(int(self.settings.value("search/epochs", DEFAULT_EPOCHS)))

        self.val_split_spin = QDoubleSpinBox()
        self.val_split_spin.setRange(0.0, 0.8)
        self.val_split_spin.setDecimals(2)
        self.val_split_spin.setSingleStep(0.05)
        self.val_split_spin.setValue(float(self.settings.value("search/val_split", DEFAULT_VAL_SPLIT)))
        self.val_split_spin.setToolTip("Used only when dataset fold files are not used.")

        self.test_split_spin = QDoubleSpinBox()
        self.test_split_spin.setRange(0.05, 0.8)
        self.test_split_spin.setDecimals(2)
        self.test_split_spin.setSingleStep(0.05)
        self.test_split_spin.setValue(float(self.settings.value("search/test_split", DEFAULT_TEST_SPLIT)))
        self.test_split_spin.setToolTip("Used only when dataset fold files are not used.")

        self.use_dataset_folds_checkbox = QCheckBox("Use dataset fold files when available")
        self.use_dataset_folds_checkbox.setChecked(
            self.settings.value("search/use_dataset_folds", True, type=bool)
        )
        self.use_dataset_folds_checkbox.setToolTip(
            "Uses train/valid/test fold files if they exist. Otherwise the trainer falls back to random split."
        )

        self.fold_index_spin = QSpinBox()
        self.fold_index_spin.setRange(0, 20)
        self.fold_index_spin.setValue(int(self.settings.value("search/fold_index", 0)))

        self.max_train_batches_spin = QSpinBox()
        self.max_train_batches_spin.setRange(0, 100000)
        self.max_train_batches_spin.setValue(
            int(self.settings.value("search/max_train_batches", DEFAULT_MAX_TRAIN_BATCHES))
        )
        self.max_train_batches_spin.setToolTip("0 means no limit. Use a small value only for debugging.")

        self.lr_values_input = QLineEdit()
        self.lr_values_input.setPlaceholderText("Example: 0.003,0.001,0.0005")
        self.lr_values_input.setText(self.settings.value("search/lr_values", DEFAULT_LR_VALUES))

        self.batch_size_values_input = QLineEdit()
        self.batch_size_values_input.setPlaceholderText("Example: 1,2,4")
        self.batch_size_values_input.setText(
            self.settings.value("search/batch_size_values", DEFAULT_BATCH_SIZE_VALUES)
        )

        self.dropout_values_input = QLineEdit()
        self.dropout_values_input.setPlaceholderText("Example: 0.2,0.3,0.4")
        self.dropout_values_input.setText(self.settings.value("search/dropout_values", DEFAULT_DROPOUT_VALUES))

        form_layout = QFormLayout()

        form_layout.addRow(QLabel("<b>1. Dataset and output</b>"))
        form_layout.addRow("Dataset:", self.dataset_combo)
        form_layout.addRow("Output root:", self._with_button(self.output_root_input, self.output_root_btn))

        form_layout.addRow(QLabel("<br><b>2. General configuration</b>"))
        form_layout.addRow("Device:", self.device_combo)
        form_layout.addRow("Random seed:", self.seed_spin)
        form_layout.addRow("Epochs:", self.epochs_spin)

        form_layout.addRow(QLabel("<br><b>3. Split configuration</b>"))
        form_layout.addRow("Use dataset folds:", self.use_dataset_folds_checkbox)
        form_layout.addRow("Fold index:", self.fold_index_spin)
        form_layout.addRow("Validation split:", self.val_split_spin)
        form_layout.addRow("Test split:", self.test_split_spin)

        form_layout.addRow(QLabel("<br><b>4. Debug configuration</b>"))
        form_layout.addRow("Max train batches:", self.max_train_batches_spin)

        form_layout.addRow(QLabel("<br><b>5. Search space</b>"))
        form_layout.addRow("Learning rate values:", self.lr_values_input)
        form_layout.addRow("Batch size values:", self.batch_size_values_input)
        form_layout.addRow("Dropout values:", self.dropout_values_input)

        layout = QVBoxLayout()
        layout.addLayout(form_layout)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout.addWidget(self.buttons)
        self.setLayout(layout)

    def _with_button(self, line_edit: QLineEdit, button: QPushButton) -> QWidget:
        container = QWidget()
        hbox = QHBoxLayout(container)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(line_edit)
        hbox.addWidget(button)
        return container

    def browse_output_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select WideDTA HPO output root")

        if path:
            self.output_root_input.setText(path)

    def accept(self) -> None:
        self.settings.setValue("search/dataset_name", self.dataset_combo.currentText())
        self.settings.setValue("search/output_root", self.output_root_input.text())
        self.settings.setValue("search/device", self.device_combo.currentText())
        self.settings.setValue("search/seed", self.seed_spin.value())
        self.settings.setValue("search/epochs", self.epochs_spin.value())
        self.settings.setValue("search/val_split", self.val_split_spin.value())
        self.settings.setValue("search/test_split", self.test_split_spin.value())
        self.settings.setValue("search/use_dataset_folds", self.use_dataset_folds_checkbox.isChecked())
        self.settings.setValue("search/fold_index", self.fold_index_spin.value())
        self.settings.setValue("search/max_train_batches", self.max_train_batches_spin.value())
        self.settings.setValue("search/lr_values", self.lr_values_input.text())
        self.settings.setValue("search/batch_size_values", self.batch_size_values_input.text())
        self.settings.setValue("search/dropout_values", self.dropout_values_input.text())

        super().accept()

    @staticmethod
    def _parse_float_list(raw_text: str, field_name: str) -> list[float]:
        values = [value.strip() for value in raw_text.split(",") if value.strip()]

        if not values:
            raise ValueError(f"At least one {field_name} value is required.")

        return [float(value) for value in values]

    @staticmethod
    def _parse_int_list(raw_text: str, field_name: str) -> list[int]:
        values = [value.strip() for value in raw_text.split(",") if value.strip()]

        if not values:
            raise ValueError(f"At least one {field_name} value is required.")

        return [int(value) for value in values]

    def get_inputs(self) -> dict:
        max_train_batches = self.max_train_batches_spin.value()

        return {
            "dataset_name": self.dataset_combo.currentText(),
            "output_root": self.output_root_input.text(),
            "device": self.device_combo.currentText(),
            "seed": self.seed_spin.value(),
            "epochs": self.epochs_spin.value(),
            "lr_values": self._parse_float_list(self.lr_values_input.text(), "learning rate"),
            "batch_size_values": self._parse_int_list(self.batch_size_values_input.text(), "batch size"),
            "dropout_values": self._parse_float_list(self.dropout_values_input.text(), "dropout"),
            "val_split": self.val_split_spin.value(),
            "test_split": self.test_split_spin.value(),
            "use_dataset_folds": self.use_dataset_folds_checkbox.isChecked(),
            "fold_index": self.fold_index_spin.value(),
            "max_train_batches": None if max_train_batches == 0 else max_train_batches,
        }
