from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QComboBox, QSpinBox,
    QDoubleSpinBox, QDialogButtonBox, QPushButton, QFileDialog, QWidget, QHBoxLayout
)
from PySide6.QtCore import QSettings

# Asegúrate de importar tus valores por defecto:
from ui.utils.constants import ATOM_EMB_PR, HYBRID_EMB_PR, BOND_EMB_PR, GNN_ARCHITECTURES


class TrainConfigDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración de entrenamiento")
        self.settings = QSettings("Investigacion", "Analisis Molecular")

        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # ---------- Train SDF directory ----------
        self.train_sdf_input = QLineEdit()
        self.train_sdf_input.setText(self.settings.value("train/train_sdf_dir", ""))
        self.train_sdf_button = QPushButton("Elegir carpeta...")
        self.train_sdf_button.clicked.connect(self.select_train_sdf_folder)
        train_layout = QHBoxLayout()
        train_layout.addWidget(self.train_sdf_input)
        train_layout.addWidget(self.train_sdf_button)
        form_layout.addRow("Directorio SDF (Train):", train_layout)

        # ---------- Validation SDF directory ----------
        self.val_sdf_input = QLineEdit()
        self.val_sdf_input.setText(self.settings.value("train/val_sdf_dir", ""))
        self.val_sdf_button = QPushButton("Elegir carpeta...")
        self.val_sdf_button.clicked.connect(self.select_val_sdf_folder)
        val_layout = QHBoxLayout()
        val_layout.addWidget(self.val_sdf_input)
        val_layout.addWidget(self.val_sdf_button)
        form_layout.addRow("Directorio SDF (Validation):", val_layout)

        # ---------- Target file ----------
        self.target_file_input = QLineEdit()
        self.target_file_input.setText(self.settings.value("train/target_file", ""))
        self.target_file_button = QPushButton("Elegir archivo...")
        self.target_file_button.clicked.connect(self.select_target_file)
        target_layout = QHBoxLayout()
        target_layout.addWidget(self.target_file_input)
        target_layout.addWidget(self.target_file_button)
        form_layout.addRow("Archivo de targets (.txt):", target_layout)

        # ---------- Modelo y configuraciones ----------
        self.model_select = QComboBox()
        # Se convierte la tupla en lista para agregarla al QComboBox
        self.model_select.addItems(list(GNN_ARCHITECTURES))
        self.model_select.setCurrentText(self.settings.value("train/modelo", GNN_ARCHITECTURES[0]))

        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 10000)
        self.epochs_input.setValue(int(self.settings.value("train/epochs", 100)))

        self.early_stopping_patience_input = QSpinBox()
        self.early_stopping_patience_input.setRange(0, 1000)
        self.early_stopping_patience_input.setValue(int(self.settings.value("train/early_stopping_patience", 0)))

        self.batch_input = QSpinBox()
        self.batch_input.setRange(1, 1024)
        self.batch_input.setValue(int(self.settings.value("train/batch_size", 32)))

        self.lr_input = QDoubleSpinBox()
        self.lr_input.setDecimals(5)
        self.lr_input.setRange(0.00001, 1.0)
        self.lr_input.setSingleStep(0.0001)
        self.lr_input.setValue(float(self.settings.value("train/lr", 0.001)))

        self.hidden_dim_input = QSpinBox()
        self.hidden_dim_input.setRange(8, 1024)
        self.hidden_dim_input.setValue(int(self.settings.value("train/hidden_dim", 64)))

        self.num_layers_input = QSpinBox()
        self.num_layers_input.setRange(1, 10)
        self.num_layers_input.setValue(int(self.settings.value("train/num_layers", 3)))

        self.save_name_input = QLineEdit()
        self.save_name_input.setText(self.settings.value("train/save_name", ""))

        # ---------- NUEVOS: Porcentajes de embedding ----------
        self.atom_emb_pr_input = QDoubleSpinBox()
        self.atom_emb_pr_input.setDecimals(3)
        self.atom_emb_pr_input.setRange(0.0, 1.0)
        self.atom_emb_pr_input.setSingleStep(0.01)
        self.atom_emb_pr_input.setValue(float(self.settings.value("train/atom_emb_pr", ATOM_EMB_PR)))

        self.hibrid_emb_pr_input = QDoubleSpinBox()
        self.hibrid_emb_pr_input.setDecimals(3)
        self.hibrid_emb_pr_input.setRange(0.0, 1.0)
        self.hibrid_emb_pr_input.setSingleStep(0.01)
        self.hibrid_emb_pr_input.setValue(float(self.settings.value("train/hibrid_emb_pr", HYBRID_EMB_PR)))

        self.bond_emb_pr_input = QDoubleSpinBox()
        self.bond_emb_pr_input.setDecimals(3)
        self.bond_emb_pr_input.setRange(0.0, 1.0)
        self.bond_emb_pr_input.setSingleStep(0.01)
        self.bond_emb_pr_input.setValue(float(self.settings.value("train/bond_emb_pr", BOND_EMB_PR)))

        # ---------- Añadir al formulario ----------
        form_layout.addRow("Modelo:", self.model_select)
        form_layout.addRow("Épocas:", self.epochs_input)
        form_layout.addRow("Paciencia Early Stopping:", self.early_stopping_patience_input)
        form_layout.addRow("Batch size:", self.batch_input)
        form_layout.addRow("Learning rate:", self.lr_input)
        form_layout.addRow("Hidden dim:", self.hidden_dim_input)
        form_layout.addRow("Número de capas:", self.num_layers_input)
        form_layout.addRow("Nombre del modelo:", self.save_name_input)

        # ---- NUEVOS ----
        form_layout.addRow("Atom Embedding %:", self.atom_emb_pr_input)
        form_layout.addRow("Hybrid Embedding %:", self.hibrid_emb_pr_input)
        form_layout.addRow("Bond Embedding %:", self.bond_emb_pr_input)

        layout.addLayout(form_layout)

        # Botones
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.setLayout(layout)

    # --------------------
    # Selección de archivos
    # --------------------

    def select_train_sdf_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta SDF de Entrenamiento")
        if folder:
            self.train_sdf_input.setText(folder)

    def select_val_sdf_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta SDF de Validación")
        if folder:
            self.val_sdf_input.setText(folder)

    def select_target_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Seleccionar archivo de targets", filter="TXT files (*.txt)")
        if file:
            self.target_file_input.setText(file)

    # --------------------
    # Guardar configuraciones
    # --------------------

    def accept(self):
        self.settings.setValue("train/train_sdf_dir", self.train_sdf_input.text())
        self.settings.setValue("train/val_sdf_dir", self.val_sdf_input.text())
        self.settings.setValue("train/target_file", self.target_file_input.text())
        self.settings.setValue("train/modelo", self.model_select.currentText())
        self.settings.setValue("train/epochs", self.epochs_input.value())
        self.settings.setValue("train/batch_size", self.batch_input.value())
        self.settings.setValue("train/lr", self.lr_input.value())
        self.settings.setValue("train/save_name", self.save_name_input.text())
        self.settings.setValue("train/hidden_dim", self.hidden_dim_input.value())
        self.settings.setValue("train/num_layers", self.num_layers_input.value())
        self.settings.setValue("train/early_stopping_patience", self.early_stopping_patience_input.value())

        # Nuevos
        self.settings.setValue("train/atom_emb_pr", self.atom_emb_pr_input.value())
        self.settings.setValue("train/hibrid_emb_pr", self.hibrid_emb_pr_input.value())
        self.settings.setValue("train/bond_emb_pr", self.bond_emb_pr_input.value())

        super().accept()

    # --------------------
    # Devolver valores
    # --------------------

    def get_values(self):
        return {
            "train_sdf_dir": self.train_sdf_input.text(),
            "val_sdf_dir": self.val_sdf_input.text(),
            "target_file": self.target_file_input.text(),
            "modelo": self.model_select.currentText(),
            "epochs": self.epochs_input.value(),
            "batch_size": self.batch_input.value(),
            "lr": self.lr_input.value(),
            "save_name": self.save_name_input.text(),
            "hidden_dim": self.hidden_dim_input.value(),
            "num_layers": self.num_layers_input.value(),
            "early_stopping_patience": self.early_stopping_patience_input.value(),
            "atom_emb_pr": self.atom_emb_pr_input.value(),
            "hibrid_emb_pr": self.hibrid_emb_pr_input.value(),
            "bond_emb_pr": self.bond_emb_pr_input.value()
        }