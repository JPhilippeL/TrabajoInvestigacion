import logging

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
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

logger = logging.getLogger(__name__)


class TrainDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Train GIGN")
        self.resize(700, 560)

        self.settings = QSettings("Investigacion", "Train_GIGN")

        self.train_file_input = QLineEdit()
        self.train_file_input.setPlaceholderText("Train split file (.txt)")
        self.train_file_input.setText(
            self.settings.value("train_gign/last_train_file", "")
        )
        self.train_file_btn = QPushButton("Select...")
        self.train_file_btn.clicked.connect(self.browse_train_file)

        self.test_file_input = QLineEdit()
        self.test_file_input.setPlaceholderText("Test split file (.txt)")
        self.test_file_input.setText(
            self.settings.value("train_gign/last_test_file", "")
        )
        self.test_file_btn = QPushButton("Select...")
        self.test_file_btn.clicked.connect(self.browse_test_file)

        self.val_file_input = QLineEdit()
        self.val_file_input.setPlaceholderText("Validation split file (.txt)")
        self.val_file_input.setText(
            self.settings.value("train_gign/last_val_file", "")
        )
        self.val_file_btn = QPushButton("Select...")
        self.val_file_btn.clicked.connect(self.browse_val_file)

        self.graph_dir_input = QLineEdit()
        self.graph_dir_input.setPlaceholderText(
            "Directory containing graph .pt files"
        )
        self.graph_dir_input.setText(
            self.settings.value("train_gign/last_graph_dir", "")
        )
        self.graph_dir_btn = QPushButton("Select...")
        self.graph_dir_btn.clicked.connect(self.browse_graph_dir)

        self.save_dir_input = QLineEdit()
        self.save_dir_input.setPlaceholderText(
            "Directory to save trained models"
        )
        self.save_dir_input.setText(
            self.settings.value("train_gign/last_save_dir", "")
        )
        self.save_dir_btn = QPushButton("Select...")
        self.save_dir_btn.clicked.connect(self.browse_save_dir)

        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 999999)
        self.seed_input.setValue(int(self.settings.value("train_gign/seed", 42)))

        self.node_dim_input = QSpinBox()
        self.node_dim_input.setRange(14, 14)
        self.node_dim_input.setValue(14)
        self.node_dim_input.setReadOnly(True)
        self.node_dim_input.setButtonSymbols(QSpinBox.NoButtons)

        self.hidden_dim_input = QSpinBox()
        self.hidden_dim_input.setRange(1, 100000)
        self.hidden_dim_input.setValue(
            int(self.settings.value("train_gign/hidden_dim", 128))
        )

        self.batch_size_input = QSpinBox()
        self.batch_size_input.setRange(1, 100000)
        self.batch_size_input.setValue(
            int(self.settings.value("train_gign/batch_size", 32))
        )

        self.lr_input = QLineEdit()
        self.lr_input.setPlaceholderText(
            "Learning rate, for example 0.0001"
        )
        self.lr_input.setText(str(self.settings.value("train_gign/lr", "0.001")))

        self.weight_decay_input = QLineEdit()
        self.weight_decay_input.setPlaceholderText("Weight decay, for exemple 0.00001")
        self.weight_decay_input.setText(
            str(self.settings.value("train_gign/weight_decay", "0.00001"))
        )
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 200)
        self.epochs_input.setValue(
            int(self.settings.value("train_gign/epochs", 50))
        )

        self.patience_input = QSpinBox()
        self.patience_input.setRange(1, 100)
        self.patience_input.setValue(
            int(self.settings.value("train_gign/patience", 15))
        )

        self.drop_out_input = QDoubleSpinBox()
        self.drop_out_input.setDecimals(4)
        self.drop_out_input.setRange(0.0, 1.0)
        self.drop_out_input.setSingleStep(0.05)
        self.drop_out_input.setValue(
            float(self.settings.value("train_gign/drop_out", 0.1))
        )

        form_layout = QFormLayout()
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
            "Save dir:",
            self._with_button(self.save_dir_input, self.save_dir_btn),
        )

        form_layout.addRow(QLabel("<b>Training parameters</b>"))
        form_layout.addRow("Seed:", self.seed_input)
        form_layout.addRow("Node dim:", self.node_dim_input)
        form_layout.addRow("Hidden dim:", self.hidden_dim_input)
        form_layout.addRow("Batch size:", self.batch_size_input)
        form_layout.addRow("Learning rate:", self.lr_input)
        form_layout.addRow("Weight decay:", self.weight_decay_input)
        form_layout.addRow("Epochs:", self.epochs_input)
        form_layout.addRow("Patience:", self.patience_input)
        form_layout.addRow("Drop out:", self.drop_out_input)

        layout = QVBoxLayout()
        layout.addLayout(form_layout)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
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
            self, title, "", "Text files (*.txt);;All files (*)"
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

    def browse_save_dir(self):
        path = self._browse_dir("Select save directory")
        if path:
            self.save_dir_input.setText(path)

    def accept(self):
        if not self.train_file_input.text().strip():
            logger.warning("Missing input: please select the train split file.")
            return

        if not self.test_file_input.text().strip():
            logger.warning("Missing input: please select the test split file.")
            return

        if not self.val_file_input.text().strip():
            logger.warning(
                "Missing input: please select the validation split file."
            )
            return

        if not self.graph_dir_input.text().strip():
            logger.warning("Missing input: please select the graph directory.")
            return

        if not self.save_dir_input.text().strip():
            logger.warning("Missing input: please select the save directory.")
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
            "train_gign/last_train_file", self.train_file_input.text().strip()
        )
        self.settings.setValue(
            "train_gign/last_test_file", self.test_file_input.text().strip()
        )
        self.settings.setValue(
            "train_gign/last_val_file", self.val_file_input.text().strip()
        )
        self.settings.setValue(
            "train_gign/last_graph_dir", self.graph_dir_input.text().strip()
        )
        self.settings.setValue(
            "train_gign/last_save_dir", self.save_dir_input.text().strip()
        )

        self.settings.setValue("train_gign/seed", self.seed_input.value())
        self.settings.setValue(
            "train_gign/hidden_dim", self.hidden_dim_input.value()
        )
        self.settings.setValue(
            "train_gign/batch_size", self.batch_size_input.value()
        )
        self.settings.setValue("train_gign/lr", lr_text)
        self.settings.setValue("train_gign/weight_decay", weight_decay_text)
        self.settings.setValue("train_gign/epochs", self.epochs_input.value())
        self.settings.setValue("train_gign/patience", self.patience_input.value())
        self.settings.setValue("train_gign/drop_out", self.drop_out_input.value())

        super().accept()

    def get_inputs(self):
        return {
            "train_file": self.train_file_input.text().strip(),
            "test_file": self.test_file_input.text().strip(),
            "val_file": self.val_file_input.text().strip(),
            "epochs": self.epochs_input.value(),
            "seed": self.seed_input.value(),
            "node_dim": self.node_dim_input.value(),
            "hidden_dim": self.hidden_dim_input.value(),
            "batch_size": self.batch_size_input.value(),
            "lr": float(self.lr_input.text().strip()),
            "weight_decay": float(self.weight_decay_input.text().strip()),
            "patience": self.patience_input.value(),
            "drop_out": self.drop_out_input.value(),
            "save_dir": self.save_dir_input.text().strip(),
            "graph_dir": self.graph_dir_input.text().strip(),
        }
