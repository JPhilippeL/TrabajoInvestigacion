"""
@file hyperparameter_search_widedta_dialog.py
@author Mohamed EL BOUKHIARI
@brief Hyperparameter search configuration dialog for the WideDTA module.
@details
This dialog uses AppSettings as the global source for module paths, runtime
device and random seed, while preserving local QSettings for WideDTA-specific
hyperparameter-search values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
from ui.utils.app_settings import AppSettings


class HyperparameterSearchWideDTADialog(QDialog):
    """
    @brief Dialog used to configure WideDTA hyperparameter search.
    """

    def __init__(self, parent=None) -> None:
        """
        @brief Initialize the WideDTA hyperparameter-search dialog.

        @param parent Optional parent widget.
        """
        super().__init__(parent)

        self.setWindowTitle("WideDTA Hyperparameter Search Configuration")
        self.resize(760, 600)

        self.settings = QSettings("ResearchApp", "WideDTA_HyperparameterSearch")
        self.app_settings = AppSettings()

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["mpro_urv", "davis", "kiba"])
        self.dataset_combo.setCurrentText(
            str(self.settings.value("search/dataset_name", DEFAULT_DATASET))
        )

        self.output_root_input = QLineEdit()
        self.output_root_input.setPlaceholderText("Root directory for WideDTA HPO runs")
        self.output_root_input.setText(self._output_root_default())

        self.output_root_btn = QPushButton("Browse...")
        self.output_root_btn.clicked.connect(self.browse_output_root)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu", "cuda:0", "cuda:1"])
        self.device_combo.setCurrentText(self._device_default())

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(self._seed_default())

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 5000)
        self.epochs_spin.setValue(self._int_setting("search/epochs", DEFAULT_EPOCHS))

        self.val_split_spin = QDoubleSpinBox()
        self.val_split_spin.setRange(0.0, 0.8)
        self.val_split_spin.setDecimals(2)
        self.val_split_spin.setSingleStep(0.05)
        self.val_split_spin.setValue(
            self._float_setting("search/val_split", DEFAULT_VAL_SPLIT)
        )
        self.val_split_spin.setToolTip(
            "Used only when dataset fold files are not used."
        )

        self.test_split_spin = QDoubleSpinBox()
        self.test_split_spin.setRange(0.05, 0.8)
        self.test_split_spin.setDecimals(2)
        self.test_split_spin.setSingleStep(0.05)
        self.test_split_spin.setValue(
            self._float_setting("search/test_split", DEFAULT_TEST_SPLIT)
        )
        self.test_split_spin.setToolTip(
            "Used only when dataset fold files are not used."
        )

        self.use_dataset_folds_checkbox = QCheckBox(
            "Use dataset fold files when available"
        )
        self.use_dataset_folds_checkbox.setChecked(
            self._bool_setting("search/use_dataset_folds", True)
        )
        self.use_dataset_folds_checkbox.setToolTip(
            "Uses train/valid/test fold files if they exist. Otherwise the trainer "
            "falls back to random split."
        )

        self.fold_index_spin = QSpinBox()
        self.fold_index_spin.setRange(0, 20)
        self.fold_index_spin.setValue(self._int_setting("search/fold_index", 0))

        self.max_train_batches_spin = QSpinBox()
        self.max_train_batches_spin.setRange(0, 100000)
        self.max_train_batches_spin.setValue(
            self._int_setting(
                "search/max_train_batches",
                DEFAULT_MAX_TRAIN_BATCHES,
            )
        )
        self.max_train_batches_spin.setToolTip(
            "0 means no limit. Use a small value only for debugging."
        )

        self.lr_values_input = QLineEdit()
        self.lr_values_input.setPlaceholderText("Example: 0.003,0.001,0.0005")
        self.lr_values_input.setText(
            str(self.settings.value("search/lr_values", DEFAULT_LR_VALUES))
        )

        self.batch_size_values_input = QLineEdit()
        self.batch_size_values_input.setPlaceholderText("Example: 1,2,4")
        self.batch_size_values_input.setText(
            str(
                self.settings.value(
                    "search/batch_size_values",
                    DEFAULT_BATCH_SIZE_VALUES,
                )
            )
        )

        self.dropout_values_input = QLineEdit()
        self.dropout_values_input.setPlaceholderText("Example: 0.2,0.3,0.4")
        self.dropout_values_input.setText(
            str(self.settings.value("search/dropout_values", DEFAULT_DROPOUT_VALUES))
        )

        form_layout = QFormLayout()

        form_layout.addRow(QLabel("<b>1. Dataset and output</b>"))
        form_layout.addRow("Dataset:", self.dataset_combo)
        form_layout.addRow(
            "Output root:",
            self._with_button(self.output_root_input, self.output_root_btn),
        )

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
        """
        @brief Wrap a line edit and a button in a horizontal widget.

        @param line_edit Path input widget.
        @param button Browse button.
        @return QWidget containing both widgets.
        """
        container = QWidget()
        hbox = QHBoxLayout(container)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(line_edit)
        hbox.addWidget(button)
        return container

    def _output_root_default(self) -> str:
        """
        @brief Return the default WideDTA HPO output root.

        @details
        The global WideDTA root from AppSettings has priority. If it is not
        available, the local dialog setting is used. If neither is available,
        the module constant is used.

        @return Output root path.
        """
        widedta_root = self.app_settings.get_value("paths/widedta_root").strip()

        if widedta_root:
            return str(Path(widedta_root) / "results" / "widedta_hpo" / "runs")

        local_value = str(self.settings.value("search/output_root", "")).strip()

        if local_value:
            return local_value

        return str(DEFAULT_OUTPUT_ROOT)

    def _device_default(self) -> str:
        """
        @brief Return the default runtime device.

        @return Device text.
        """
        device = self.app_settings.get_value("runtime/default_device").strip()

        if device:
            return device

        return str(DEFAULT_DEVICE)

    def _seed_default(self) -> int:
        """
        @brief Return the default random seed.

        @return Random seed.
        """
        try:
            return int(self.app_settings.get_value("runtime/default_seed"))
        except ValueError:
            return int(DEFAULT_SEED)

    def _int_setting(self, key: str, fallback: Any) -> int:
        """
        @brief Read an integer value from local dialog settings.

        @param key Local settings key.
        @param fallback Fallback value.
        @return Parsed integer.
        """
        try:
            return int(self.settings.value(key, fallback))
        except (TypeError, ValueError):
            return int(fallback)

    def _float_setting(self, key: str, fallback: Any) -> float:
        """
        @brief Read a floating-point value from local dialog settings.

        @param key Local settings key.
        @param fallback Fallback value.
        @return Parsed float.
        """
        try:
            return float(self.settings.value(key, fallback))
        except (TypeError, ValueError):
            return float(fallback)

    def _bool_setting(self, key: str, fallback: bool) -> bool:
        """
        @brief Read a boolean value from local dialog settings.

        @param key Local settings key.
        @param fallback Fallback value.
        @return Parsed boolean.
        """
        return bool(self.settings.value(key, fallback, type=bool))

    def _selected_device(self) -> str | None:
        """
        @brief Return selected device for the worker.

        @return Device string, or None when auto is selected.
        """
        device = self.device_combo.currentText()
        return None if device == "auto" else device

    def browse_output_root(self) -> None:
        """
        @brief Select the output root directory for WideDTA HPO results.

        @return None.
        """
        path = QFileDialog.getExistingDirectory(
            self,
            "Select WideDTA HPO output root",
        )

        if path:
            self.output_root_input.setText(path)

    def accept(self) -> None:
        """
        @brief Save dialog settings before accepting.

        @return None.
        """
        self.settings.setValue("search/dataset_name", self.dataset_combo.currentText())
        self.settings.setValue("search/output_root", self.output_root_input.text())
        self.settings.setValue("search/device", self.device_combo.currentText())
        self.settings.setValue("search/seed", self.seed_spin.value())
        self.settings.setValue("search/epochs", self.epochs_spin.value())
        self.settings.setValue("search/val_split", self.val_split_spin.value())
        self.settings.setValue("search/test_split", self.test_split_spin.value())
        self.settings.setValue(
            "search/use_dataset_folds",
            self.use_dataset_folds_checkbox.isChecked(),
        )
        self.settings.setValue("search/fold_index", self.fold_index_spin.value())
        self.settings.setValue(
            "search/max_train_batches",
            self.max_train_batches_spin.value(),
        )
        self.settings.setValue("search/lr_values", self.lr_values_input.text())
        self.settings.setValue(
            "search/batch_size_values",
            self.batch_size_values_input.text(),
        )
        self.settings.setValue("search/dropout_values", self.dropout_values_input.text())

        self.app_settings.set_value("runtime/default_device", self.device_combo.currentText())
        self.app_settings.set_value("runtime/default_seed", self.seed_spin.value())
        self.app_settings.sync()

        super().accept()

    @staticmethod
    def _parse_float_list(raw_text: str, field_name: str) -> list[float]:
        """
        @brief Parse comma-separated float values.

        @param raw_text Raw input text.
        @param field_name Field name used in validation errors.
        @return Parsed float list.
        """
        values = [value.strip() for value in raw_text.split(",") if value.strip()]

        if not values:
            raise ValueError(f"At least one {field_name} value is required.")

        return [float(value) for value in values]

    @staticmethod
    def _parse_int_list(raw_text: str, field_name: str) -> list[int]:
        """
        @brief Parse comma-separated integer values.

        @param raw_text Raw input text.
        @param field_name Field name used in validation errors.
        @return Parsed integer list.
        """
        values = [value.strip() for value in raw_text.split(",") if value.strip()]

        if not values:
            raise ValueError(f"At least one {field_name} value is required.")

        return [int(value) for value in values]

    def get_inputs(self) -> dict[str, object]:
        """
        @brief Return validated WideDTA HPO parameters.

        @return Input parameter dictionary.
        """
        max_train_batches = self.max_train_batches_spin.value()

        return {
            "dataset_name": self.dataset_combo.currentText(),
            "output_root": self.output_root_input.text(),
            "device": self._selected_device(),
            "seed": self.seed_spin.value(),
            "epochs": self.epochs_spin.value(),
            "lr_values": self._parse_float_list(
                self.lr_values_input.text(),
                "learning rate",
            ),
            "batch_size_values": self._parse_int_list(
                self.batch_size_values_input.text(),
                "batch size",
            ),
            "dropout_values": self._parse_float_list(
                self.dropout_values_input.text(),
                "dropout",
            ),
            "val_split": self.val_split_spin.value(),
            "test_split": self.test_split_spin.value(),
            "use_dataset_folds": self.use_dataset_folds_checkbox.isChecked(),
            "fold_index": self.fold_index_spin.value(),
            "max_train_batches": None if max_train_batches == 0 else max_train_batches,
        }
