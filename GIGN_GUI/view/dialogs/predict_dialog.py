from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit,
    QPushButton, QFileDialog, QDialogButtonBox, QWidget,
    QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox
)
from PySide6.QtCore import QSettings
import logging

logger = logging.getLogger(__name__)


class PredictDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Predict GIGN")
        self.resize(700, 560)

        self.settings = QSettings("Investigacion", "Predict_GIGN")

        self.train_split_input = QLineEdit()
        self.train_split_input.setPlaceholderText("Train split file (.txt)")
        self.train_split_input.setText(self.settings.value("predict/last_train_split_file", ""))
        self.train_split_btn = QPushButton("Select...")
        self.train_split_btn.clicked.connect(self.browse_train_split)

        self.test_split_input = QLineEdit()
        self.test_split_input.setPlaceholderText("Test split file (.txt)")
        self.test_split_input.setText(self.settings.value("predict/last_test_split_file", ""))
        self.test_split_btn = QPushButton("Select...")
        self.test_split_btn.clicked.connect(self.browse_test_split)

        self.val_split_input = QLineEdit()
        self.val_split_input.setPlaceholderText("Validation split file (.txt)")
        self.val_split_input.setText(self.settings.value("predict/last_val_split_file", ""))
        self.val_split_btn = QPushButton("Select...")
        self.val_split_btn.clicked.connect(self.browse_val_split)

        self.model_dir_input = QLineEdit()
        self.model_dir_input.setPlaceholderText("Directory containing split_xx/best_model.pt")
        self.model_dir_input.setText(self.settings.value("predict/last_model_dir", ""))
        self.model_dir_btn = QPushButton("Select...")
        self.model_dir_btn.clicked.connect(self.browse_model_dir)

        self.pic50_file_input = QLineEdit()
        self.pic50_file_input.setPlaceholderText("File pic50.txt")
        self.pic50_file_input.setText(self.settings.value("pic50.txt", ""))
        self.pic50_btn = QPushButton("Select...")
        self.pic50_btn.clicked.connect(self.browse_pic50_file)
        self.graph_dir_input = QLineEdit()
        self.graph_dir_input.setPlaceholderText("Directory containing graph .pt files")
        self.graph_dir_input.setText(self.settings.value("predict/last_graph_dir", ""))
        self.graph_dir_btn = QPushButton("Select...")
        self.graph_dir_btn.clicked.connect(self.browse_graph_dir)

        self.output_dir_input = QLineEdit()
        self.output_dir_input.setPlaceholderText("Directory to save prediction outputs")
        self.output_dir_input.setText(self.settings.value("predict/last_output_dir", ""))
        self.output_dir_btn = QPushButton("Select...")
        self.output_dir_btn.clicked.connect(self.browse_output_dir)

        self.batch_size_input = QSpinBox()
        self.batch_size_input.setRange(1, 100000)
        self.batch_size_input.setValue(int(self.settings.value("predict/batch_size", 32)))

        self.node_dim_input = QSpinBox()
        self.node_dim_input.setRange(14, 14)
        self.node_dim_input.setValue(14)
        self.node_dim_input.setReadOnly(True)
        self.node_dim_input.setButtonSymbols(QSpinBox.NoButtons)

        self.hidden_dim_input = QSpinBox()
        self.hidden_dim_input.setRange(1, 100000)
        self.hidden_dim_input.setValue(int(self.settings.value("predict/hidden_dim", 128)))

        self.drop_out_input = QDoubleSpinBox()
        self.drop_out_input.setDecimals(4)
        self.drop_out_input.setRange(0.0, 1.0)
        self.drop_out_input.setSingleStep(0.05)
        self.drop_out_input.setValue(float(self.settings.value("predict/drop_out", 0.2)))

        form_layout = QFormLayout()

        form_layout.addRow(QLabel("<b>Required files/directories</b>"))
        form_layout.addRow("Train split:", self._with_button(self.train_split_input, self.train_split_btn))
        form_layout.addRow("Test split:", self._with_button(self.test_split_input, self.test_split_btn))
        form_layout.addRow("Validation split:", self._with_button(self.val_split_input, self.val_split_btn))
        form_layout.addRow("Model dir:", self._with_button(self.model_dir_input, self.model_dir_btn))
        form_layout.addRow("Graph dir:", self._with_button(self.graph_dir_input, self.graph_dir_btn))
        form_layout.addRow("Output dir:", self._with_button(self.output_dir_input, self.output_dir_btn))
        form_layout.addRow("Target file:", self._with_button(self.pic50_file_input, self.pic50_btn))

        form_layout.addRow(QLabel("<b>Prediction parameters</b>"))
        form_layout.addRow("Batch size:", self.batch_size_input)
        form_layout.addRow("Node dim:", self.node_dim_input)
        form_layout.addRow("Hidden dim:", self.hidden_dim_input)
        form_layout.addRow("Drop out:", self.drop_out_input)

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
            "Text files (*.txt);;All files (*)"
        )
        return path

    def _browse_dir(self, title):
        return QFileDialog.getExistingDirectory(self, title)

    def browse_train_split(self):
        path = self._browse_txt_file("Select train split file")
        if path:
            self.train_split_input.setText(path)

    def browse_test_split(self):
        path = self._browse_txt_file("Select test split file")
        if path:
            self.test_split_input.setText(path)

    def browse_val_split(self):
        path = self._browse_txt_file("Select validation split file")
        if path:
            self.val_split_input.setText(path)

    def browse_pic50_file(self):
        path = self._browse_txt_file("Select pic50 file")
        if path:
            self.pic50_file_input.setText(path)

    def browse_model_dir(self):
        path = self._browse_dir("Select model directory")
        if path:
            self.model_dir_input.setText(path)

    def browse_graph_dir(self):
        path = self._browse_dir("Select graph directory")
        if path:
            self.graph_dir_input.setText(path)

    def browse_output_dir(self):
        path = self._browse_dir("Select output directory")
        if path:
            self.output_dir_input.setText(path)

    def accept(self):
        if not self.train_split_input.text().strip():
            logger.warning("Missing input: please select the train split file.")
            return

        if not self.test_split_input.text().strip():
            logger.warning("Missing input: please select the test split file.")
            return

        if not self.val_split_input.text().strip():
            logger.warning("Missing input: please select the validation split file.")
            return

        if not self.model_dir_input.text().strip():
            logger.warning("Missing input: please select the model directory.")
            return

        if not self.graph_dir_input.text().strip():
            logger.warning("Missing input: please select the graph directory.")
            return

        if not self.output_dir_input.text().strip():
            logger.warning("Missing input: please select the output directory.")
            return

        self.settings.setValue("predict/last_train_split_file", self.train_split_input.text().strip())
        self.settings.setValue("predict/last_test_split_file", self.test_split_input.text().strip())
        self.settings.setValue("predict/last_val_split_file", self.val_split_input.text().strip())
        self.settings.setValue("predict/last_model_dir", self.model_dir_input.text().strip())
        self.settings.setValue("predict/last_graph_dir", self.graph_dir_input.text().strip())
        self.settings.setValue("predict/last_output_dir", self.output_dir_input.text().strip())

        self.settings.setValue("predict/batch_size", self.batch_size_input.value())
        self.settings.setValue("predict/hidden_dim", self.hidden_dim_input.value())
        self.settings.setValue("predict/drop_out", self.drop_out_input.value())
        self.settings.setValue("predict/pic50_txt", self.pic50_file_input.text().strip())

        super().accept()

    def get_inputs(self):
        return {
            "batch_size": self.batch_size_input.value(),
            "node_dim": self.node_dim_input.value(),
            "hidden_dim": self.hidden_dim_input.value(),
            "model_dir": self.model_dir_input.text().strip(),
            "drop_out": self.drop_out_input.value(),
            "graph_dir": self.graph_dir_input.text().strip(),
            "train_split_file": self.train_split_input.text().strip(),
            "test_split_file": self.test_split_input.text().strip(),
            "val_split_file": self.val_split_input.text().strip(),
            "output_dir": self.output_dir_input.text().strip(),
            "pic50_txt": self.pic50_file_input.text().strip(),
        }
