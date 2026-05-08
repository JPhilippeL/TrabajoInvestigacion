from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import QSettings


class HyperparameterSearchDialog(QDialog):
    MODELS = ["GIN", "GINE", "GAT", "EGAT", "GraphTransformer"]

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Hyperparameter Search GNN")
        self.resize(720, 650)

        self.settings = QSettings("Investigacion", "Analisis Molecular")

        self.train_sdf_dir_input = QLineEdit(
            self.settings.value("hyperparameterSearch/train_sdf_dir", "")
        )
        self.target_file_input = QLineEdit(
            self.settings.value("hyperparameterSearch/target_file", "")
        )
        self.eval_sdf_dir_input = QLineEdit(
            self.settings.value("hyperparameterSearch/eval_sdf_dir", "")
        )
        self.eval_targets_file_input = QLineEdit(
            self.settings.value("hyperparameterSearch/eval_targets_file", "")
        )
        self.output_root_input = QLineEdit(
            self.settings.value("hyperparameterSearch/output_root", "hyperparameter_Search")
        )

        self.lr_values_input = QLineEdit(
            self.settings.value("hyperparameterSearch/lr_values", "0.001,0.0005,0.0001")
        )
        self.batch_size_values_input = QLineEdit(
            self.settings.value("hyperparameterSearch/batch_size_values", "16,32")
        )
        self.hidden_dim_values_input = QLineEdit(
            self.settings.value("hyperparameterSearch/hidden_dim_values", "64,128")
        )
        self.num_layers_values_input = QLineEdit(
            self.settings.value("hyperparameterSearch/num_layers_values", "2,3")
        )
        self.atom_emb_dim_values_input = QLineEdit(
            self.settings.value("hyperparameterSearch/atom_emb_dim_values", "0.4")
        )
        self.hibrid_emb_dim_values_input = QLineEdit(
            self.settings.value("hyperparameterSearch/hibrid_emb_dim_values", "0.5")
        )
        self.bond_emb_dim_values_input = QLineEdit(
            self.settings.value("hyperparameterSearch/bond_emb_dim_values", "1")
        )

        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 100000)
        self.epochs_input.setValue(int(self.settings.value("hyperparameterSearch/epochs", 50)))

        self.patience_input = QSpinBox()
        self.patience_input.setRange(0, 100000)
        self.patience_input.setValue(int(self.settings.value("hyperparameterSearch/patience", 10)))

        self.valid_split_input = QDoubleSpinBox()
        self.valid_split_input.setRange(0.01, 0.90)
        self.valid_split_input.setSingleStep(0.05)
        self.valid_split_input.setDecimals(2)
        self.valid_split_input.setValue(float(self.settings.value("hyperparameterSearch/valid_split", 0.2)))

        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 2147483647)
        self.seed_input.setValue(int(self.settings.value("hyperparameterSearch/seed", 42)))

        self.objective_metric_input = QComboBox()
        self.objective_metric_input.addItems(["RMSE", "MSE", "MAE", "Pearson", "Spearman", "R2"])
        self.objective_metric_input.setCurrentText(
            self.settings.value("hyperparameterSearch/objective_metric", "RMSE")
        )

        self.objective_mode_input = QComboBox()
        self.objective_mode_input.addItems(["min", "max"])
        self.objective_mode_input.setCurrentText(
            self.settings.value("hyperparameterSearch/objective_mode", "min")
        )

        self.resume_input = QCheckBox("Resume completed trials")
        self.resume_input.setChecked(
            self._settings_bool("hyperparameterSearch/resume", True)
        )

        self.rerun_failed_input = QCheckBox("Rerun failed trials")
        self.rerun_failed_input.setChecked(
            self._settings_bool("hyperparameterSearch/rerun_failed", False)
        )

        self.model_checkboxes = {}

        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        main_layout.addWidget(self._build_dataset_group())
        main_layout.addWidget(self._build_models_group())
        main_layout.addWidget(self._build_search_space_group())
        main_layout.addWidget(self._build_execution_group())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(main_widget)

        layout = QVBoxLayout()
        layout.addWidget(scroll)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout.addWidget(self.buttons)
        self.setLayout(layout)

    def _build_dataset_group(self):
        group = QGroupBox("Dataset paths")
        form = QFormLayout()

        form.addRow(
            "Train SDF directory:",
            self._with_button(self.train_sdf_dir_input, "Select...", self.browse_train_sdf_dir),
        )
        form.addRow(
            "Target file:",
            self._with_button(self.target_file_input, "Select...", self.browse_target_file),
        )
        form.addRow(
            "Eval SDF directory:",
            self._with_button(self.eval_sdf_dir_input, "Select...", self.browse_eval_sdf_dir),
        )
        form.addRow(
            "Eval targets file:",
            self._with_button(self.eval_targets_file_input, "Select...", self.browse_eval_targets_file),
        )
        form.addRow(
            "Output root:",
            self._with_button(self.output_root_input, "Select...", self.browse_output_root),
        )

        group.setLayout(form)
        return group

    def _build_models_group(self):
        group = QGroupBox("Models")
        grid = QGridLayout()

        for index, model_name in enumerate(self.MODELS):
            checkbox = QCheckBox(model_name)
            checkbox.setChecked(
                self._settings_bool(f"hyperparameterSearch/model_{model_name}", True)
            )

            self.model_checkboxes[model_name] = checkbox
            grid.addWidget(checkbox, index // 3, index % 3)

        group.setLayout(grid)
        return group

    def _build_search_space_group(self):
        group = QGroupBox("Search space")
        form = QFormLayout()

        form.addRow("Learning rates:", self.lr_values_input)
        form.addRow("Batch sizes:", self.batch_size_values_input)
        form.addRow("Hidden dims:", self.hidden_dim_values_input)
        form.addRow("Num layers:", self.num_layers_values_input)
        form.addRow("Atom emb dims:", self.atom_emb_dim_values_input)
        form.addRow("Hibrid emb dims:", self.hibrid_emb_dim_values_input)
        form.addRow("Bond emb dims:", self.bond_emb_dim_values_input)

        group.setLayout(form)
        return group

    def _build_execution_group(self):
        group = QGroupBox("Execution")
        form = QFormLayout()

        form.addRow("Epochs:", self.epochs_input)
        form.addRow("Patience:", self.patience_input)
        form.addRow("Validation split:", self.valid_split_input)
        form.addRow("Objective metric:", self.objective_metric_input)
        form.addRow("Objective mode:", self.objective_mode_input)
        form.addRow("Seed:", self.seed_input)
        form.addRow("", self.resume_input)
        form.addRow("", self.rerun_failed_input)

        group.setLayout(form)
        return group

    def _with_button(self, line_edit, button_text, callback):
        container = QWidget()
        hbox = QHBoxLayout(container)

        button = QPushButton(button_text)
        button.clicked.connect(callback)

        hbox.addWidget(line_edit)
        hbox.addWidget(button)
        hbox.setContentsMargins(0, 0, 0, 0)

        return container

    def _settings_bool(self, key, default):
        value = self.settings.value(key, default)

        if isinstance(value, bool):
            return value

        return str(value).strip().lower() in {"1", "true", "yes", "y", "si", "sí"}

    def _selected_models(self):
        return [
            model_name
            for model_name, checkbox in self.model_checkboxes.items()
            if checkbox.isChecked()
        ]

    def browse_train_sdf_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select train SDF directory")
        if path:
            self.train_sdf_dir_input.setText(path)

    def browse_target_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select target file", "", "Text files (*.txt);;All files (*)")
        if path:
            self.target_file_input.setText(path)

    def browse_eval_sdf_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select eval SDF directory")
        if path:
            self.eval_sdf_dir_input.setText(path)

    def browse_eval_targets_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select eval targets file", "", "Text files (*.txt);;All files (*)")
        if path:
            self.eval_targets_file_input.setText(path)

    def browse_output_root(self):
        path = QFileDialog.getExistingDirectory(self, "Select output root")
        if path:
            self.output_root_input.setText(path)

    def accept(self):
        if not self.train_sdf_dir_input.text().strip():
            self._show_error("Train SDF directory is required.")
            return

        if not self.target_file_input.text().strip():
            self._show_error("Target file is required.")
            return

        if not self.output_root_input.text().strip():
            self._show_error("Output root is required.")
            return

        if not self._selected_models():
            self._show_error("Select at least one GNN model.")
            return

        self._save_settings()
        super().accept()

    def _show_error(self, message):
        QMessageBox.warning(self, "Invalid configuration", message)

    def _save_settings(self):
        self.settings.setValue("hyperparameterSearch/train_sdf_dir", self.train_sdf_dir_input.text())
        self.settings.setValue("hyperparameterSearch/target_file", self.target_file_input.text())
        self.settings.setValue("hyperparameterSearch/eval_sdf_dir", self.eval_sdf_dir_input.text())
        self.settings.setValue("hyperparameterSearch/eval_targets_file", self.eval_targets_file_input.text())
        self.settings.setValue("hyperparameterSearch/output_root", self.output_root_input.text())

        self.settings.setValue("hyperparameterSearch/lr_values", self.lr_values_input.text())
        self.settings.setValue("hyperparameterSearch/batch_size_values", self.batch_size_values_input.text())
        self.settings.setValue("hyperparameterSearch/hidden_dim_values", self.hidden_dim_values_input.text())
        self.settings.setValue("hyperparameterSearch/num_layers_values", self.num_layers_values_input.text())
        self.settings.setValue("hyperparameterSearch/atom_emb_dim_values", self.atom_emb_dim_values_input.text())
        self.settings.setValue("hyperparameterSearch/hibrid_emb_dim_values", self.hibrid_emb_dim_values_input.text())
        self.settings.setValue("hyperparameterSearch/bond_emb_dim_values", self.bond_emb_dim_values_input.text())

        self.settings.setValue("hyperparameterSearch/epochs", self.epochs_input.value())
        self.settings.setValue("hyperparameterSearch/patience", self.patience_input.value())
        self.settings.setValue("hyperparameterSearch/valid_split", self.valid_split_input.value())
        self.settings.setValue("hyperparameterSearch/objective_metric", self.objective_metric_input.currentText())
        self.settings.setValue("hyperparameterSearch/objective_mode", self.objective_mode_input.currentText())
        self.settings.setValue("hyperparameterSearch/resume", self.resume_input.isChecked())
        self.settings.setValue("hyperparameterSearch/rerun_failed", self.rerun_failed_input.isChecked())
        self.settings.setValue("hyperparameterSearch/seed", self.seed_input.value())

        for model_name, checkbox in self.model_checkboxes.items():
            self.settings.setValue(
                f"hyperparameterSearch/model_{model_name}",
                checkbox.isChecked(),
            )

    def get_config(self):
        return {
            "train_sdf_dir": self.train_sdf_dir_input.text().strip(),
            "target_file": self.target_file_input.text().strip(),
            "eval_sdf_dir": self.eval_sdf_dir_input.text().strip(),
            "eval_targets_file": self.eval_targets_file_input.text().strip(),
            "output_root": self.output_root_input.text().strip(),

            "model_names": ",".join(self._selected_models()),

            "lr_values": self.lr_values_input.text().strip(),
            "batch_size_values": self.batch_size_values_input.text().strip(),
            "hidden_dim_values": self.hidden_dim_values_input.text().strip(),
            "num_layers_values": self.num_layers_values_input.text().strip(),
            "atom_emb_dim_values": self.atom_emb_dim_values_input.text().strip(),
            "hibrid_emb_dim_values": self.hibrid_emb_dim_values_input.text().strip(),
            "bond_emb_dim_values": self.bond_emb_dim_values_input.text().strip(),

            "epochs": self.epochs_input.value(),
            "patience": self.patience_input.value(),
            "valid_split": self.valid_split_input.value(),
            "objective_metric": self.objective_metric_input.currentText(),
            "objective_mode": self.objective_mode_input.currentText(),
            "resume": self.resume_input.isChecked(),
            "rerun_failed": self.rerun_failed_input.isChecked(),
            "seed": self.seed_input.value(),
        }
