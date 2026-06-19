import logging
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
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

from job_config.graphdta.DTATrainerConfig import DTATrainerConfig
from models.graphdta.registry import get_graph_dta_model

logger = logging.getLogger(__name__)


class TrainDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Train GraphDTA")
        self.resize(700, 520)

        self.settings = QSettings("Investigacion", "Train_GraphDTA")

        self.model_name_input = QComboBox()
        self.model_name_input.addItems(
            [
                "GAT",
                "GAT_GCN",
                "GINConvNet",
                "GCN",
            ]
        )

        last_model_name = self.settings.value("train_graphdta/model_name", "GINConvNet")
        index = self.model_name_input.findText(str(last_model_name))
        if index >= 0:
            self.model_name_input.setCurrentIndex(index)

        self.train_file_input = QLineEdit()
        self.train_file_input.setPlaceholderText("Train split file (.txt)")
        self.train_file_input.setText(self.settings.value("train_graphdta/last_train_file", ""))
        self.train_file_btn = QPushButton("Select...")
        self.train_file_btn.clicked.connect(self.browse_train_file)

        self.test_file_input = QLineEdit()
        self.test_file_input.setPlaceholderText("Test split file (.txt)")
        self.test_file_input.setText(self.settings.value("train_graphdta/last_test_file", ""))
        self.test_file_btn = QPushButton("Select...")
        self.test_file_btn.clicked.connect(self.browse_test_file)

        self.val_file_input = QLineEdit()
        self.val_file_input.setPlaceholderText("Validation split file (.txt)")
        self.val_file_input.setText(self.settings.value("train_graphdta/last_val_file", ""))
        self.val_file_btn = QPushButton("Select...")
        self.val_file_btn.clicked.connect(self.browse_val_file)

        self.graph_dir_input = QLineEdit()
        self.graph_dir_input.setPlaceholderText("Directory containing GraphDTA .pt files")
        self.graph_dir_input.setText(self.settings.value("train_graphdta/last_graph_dir", ""))
        self.graph_dir_btn = QPushButton("Select...")
        self.graph_dir_btn.clicked.connect(self.browse_graph_dir)

        self.output_dir_input = QLineEdit()
        self.output_dir_input.setPlaceholderText("Directory to save trained GraphDTA models")
        self.output_dir_input.setText(self.settings.value("train_graphdta/last_output_dir", ""))
        self.output_dir_btn = QPushButton("Select...")
        self.output_dir_btn.clicked.connect(self.browse_output_dir)

        self.batch_size_input = QSpinBox()
        self.batch_size_input.setRange(1, 100000)
        self.batch_size_input.setValue(int(self.settings.value("train_graphdta/batch_size", 32)))
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 1000)
        self.epochs_input.setValue(int(self.settings.value("train_graphdta/epochs", 50)))

        self.n_filters_input = QSpinBox()
        self.n_filters_input.setRange(1, 2048)
        self.n_filters_input.setValue(int(self.settings.value("train_graphdta/n_filters", 32)))

        self.lr_input = QLineEdit()
        self.lr_input.setPlaceholderText("Learning rate, for example 0.001")
        self.lr_input.setText(str(self.settings.value("train_graphdta/lr", "0.001")))

        self.weight_decay_input = QLineEdit()
        self.weight_decay_input.setPlaceholderText("Weight decay, for example 0.00001")
        self.weight_decay_input.setText(
            str(self.settings.value("train_graphdta/weight_decay", "0.00001"))
        )

        self.dropout_input = QDoubleSpinBox()
        self.dropout_input.setDecimals(4)
        self.dropout_input.setRange(0.0, 1.0)
        self.dropout_input.setSingleStep(0.05)
        self.dropout_input.setValue(float(self.settings.value("train_graphdta/dropout", 0.2)))

        form_layout = QFormLayout()

        form_layout.addRow(QLabel("<b>Model</b>"))
        form_layout.addRow("Model name:", self.model_name_input)

        form_layout.addRow(QLabel("<b>Required files/directories</b>"))
        form_layout.addRow(
            "Train split:",
            self._with_button(self.train_file_input, self.train_file_btn),
        )
        form_layout.addRow(
            "Test split:",
            self._with_button(self.test_file_input, self.test_file_btn),
        )
        form_layout.addRow(
            "Validation split:",
            self._with_button(self.val_file_input, self.val_file_btn),
        )
        form_layout.addRow(
            "Graph dir:",
            self._with_button(self.graph_dir_input, self.graph_dir_btn),
        )
        form_layout.addRow(
            "Output dir:",
            self._with_button(self.output_dir_input, self.output_dir_btn),
        )

        form_layout.addRow(QLabel("<b>Training parameters</b>"))
        form_layout.addRow("Batch size:", self.batch_size_input)
        form_layout.addRow("Epochs:", self.epochs_input)
        form_layout.addRow("Number of filters:", self.n_filters_input)
        form_layout.addRow("Learning rate:", self.lr_input)
        form_layout.addRow("Weight decay:", self.weight_decay_input)
        form_layout.addRow("Dropout:", self.dropout_input)
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

    def _browse_txt_file(self, title):
        path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            "",
            "Text files (*.txt);;All files (*)",
        )
        return path

    def _browse_dir(self, title):
        return QFileDialog.getExistingDirectory(self, title)

    def browse_train_file(self):
        path = self._browse_txt_file("Select train split file")
        if path:
            self.train_file_input.setText(path)

    def browse_test_file(self):
        path = self._browse_txt_file("Select test split file")
        if path:
            self.test_file_input.setText(path)

    def browse_val_file(self):
        path = self._browse_txt_file("Select validation split file")
        if path:
            self.val_file_input.setText(path)

    def browse_graph_dir(self):
        path = self._browse_dir("Select graph directory")
        if path:
            self.graph_dir_input.setText(path)

    def browse_output_dir(self):
        path = self._browse_dir("Select output directory")
        if path:
            self.output_dir_input.setText(path)

    def accept(self):
        if not self.train_file_input.text().strip():
            logger.warning("Missing input: please select the train split file.")
            return

        if not self.test_file_input.text().strip():
            logger.warning("Missing input: please select the test split file.")
            return

        if not self.val_file_input.text().strip():
            logger.warning("Missing input: please select the validation split file.")
            return

        if not self.graph_dir_input.text().strip():
            logger.warning("Missing input: please select the graph directory.")
            return

        if not self.output_dir_input.text().strip():
            logger.warning("Missing input: please select the output directory.")
            return

        lr_text = self.lr_input.text().strip()
        if not lr_text:
            logger.warning("Missing input: please enter the learning rate.")
            return

        try:
            float(lr_text)
        except ValueError:
            logger.warning("Invalid input: learning rate must be a float.")
            return

        weight_decay_text = self.weight_decay_input.text().strip()
        if not weight_decay_text:
            logger.warning("Missing input: please enter the weight decay.")
            return

        try:
            float(weight_decay_text)
        except ValueError:
            logger.warning("Invalid input: weight decay must be a float.")
            return

        self.settings.setValue(
            "train_graphdta/model_name",
            self.model_name_input.currentText(),
        )
        self.settings.setValue(
            "train_graphdta/last_train_file",
            self.train_file_input.text().strip(),
        )
        self.settings.setValue(
            "train_graphdta/last_test_file",
            self.test_file_input.text().strip(),
        )
        self.settings.setValue(
            "train_graphdta/last_val_file",
            self.val_file_input.text().strip(),
        )
        self.settings.setValue(
            "train_graphdta/last_graph_dir",
            self.graph_dir_input.text().strip(),
        )
        self.settings.setValue(
            "train_graphdta/last_output_dir",
            self.output_dir_input.text().strip(),
        )

        self.settings.setValue(
            "train_graphdta/batch_size",
            self.batch_size_input.value(),
        )
        self.settings.setValue(
            "train_graphdta/n_filters",
            self.n_filters_input.value(),
        )
        self.settings.setValue("train_graphdta/lr", lr_text)
        self.settings.setValue(
            "train_graphdta/weight_decay",
            weight_decay_text,
        )
        self.settings.setValue(
            "train_graphdta/dropout",
            self.dropout_input.value(),
        )
        self.settings.setValue(
            "train_graphdta/epochs",
            self.epochs_input.value(),
        )

        super().accept()

    def get_inputs(self):
        model_name = self.model_name_input.currentText()
        model_class = get_graph_dta_model(model_name)

        return DTATrainerConfig(
            model=model_class,
            model_name=model_name,
            train_split_file=Path(self.train_file_input.text().strip()),
            test_split_file=Path(self.test_file_input.text().strip()),
            val_split_file=Path(self.val_file_input.text().strip()),
            graphs_path=Path(self.graph_dir_input.text().strip()),
            output_path=Path(self.output_dir_input.text().strip()),
            batch_size=self.batch_size_input.value(),
            lr=float(self.lr_input.text().strip()),
            number_of_filters=self.n_filters_input.value(),
            dropout=self.dropout_input.value(),
            weight_decay=float(self.weight_decay_input.text().strip()),
            epochs=self.epochs_input.value(),
        )
