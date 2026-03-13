from PySide6.QtWidgets import QMessageBox, QDialog, QMenu
from PySide6.QtGui import QAction
import os
import logging

from ui.dialogs.train_config_dialog import TrainConfigDialog
from ui.dialogs.train_multiple_models import TrainMultipleModelsDialog
from ui.dialogs.train_config_dialog_from_pt import TrainConfigDialogPT

logger = logging.getLogger(__name__)

class MenuTrainGNN(QMenu):
    def __init__(self, parent_window):
        # "parent_window" es tu MainWindow, lo necesitamos para los diálogos
        super().__init__("Train GNN", parent_window)
        self.main_window = parent_window 

        self.init_actions()

    def init_actions(self):
        # Menu de Entrenamiento

        # Entrenamiento de IA
        entrenar_action = QAction("Entrenar Modelo con SDFs", self)
        entrenar_action.triggered.connect(self.entrenar_gnn)
        self.addAction(entrenar_action)

        entrenar_from_pt_action = QAction("Entrenar Modelo con .pt", self)
        entrenar_from_pt_action.triggered.connect(self.entrenar_gnn_pt)
        self.addAction(entrenar_from_pt_action)

        # Entrenamiento de múltiples modelos
        entrenar_multiple_action = QAction("Entrenar Múltiples Modelos", self)
        entrenar_multiple_action.triggered.connect(self.entrenar_multiples_modelos_gnn)
        self.addAction(entrenar_multiple_action)

    def entrenar_gnn(self):
        dialog = TrainConfigDialog(self.main_window)

        if dialog.exec() != QDialog.Accepted:
            return

        config = dialog.get_values()

        # ----- Validaciones -----

        if not config["target_file"] or not os.path.isfile(config["target_file"]):
            QMessageBox.warning(self.main_window, "Error", "Debes seleccionar un archivo .txt válido con los targets.")
            return

        if not config["save_name"]:
            QMessageBox.warning(self.main_window, "Nombre inválido", "El nombre del archivo no puede estar vacío.")
            return

        # ----- Ejecutar entrenamiento -----
        self.main_window.training_controller.entrenar(
            train_dir=config["train_sdf_dir"],
            val_dir=config["val_sdf_dir"],
            target_file=config["target_file"],
            model_type=config["modelo"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            lr=config["lr"],
            model_name=config['save_name'],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            patience=config["early_stopping_patience"],
            atom_emb_dim = config["atom_emb_pr"],
            hibrid_emb_dim = config["hibrid_emb_pr"],
            bond_emb_dim = config["bond_emb_pr"]
        )

    def entrenar_gnn_pt(self):
        dialog = TrainConfigDialogPT(self.main_window)

        if dialog.exec() != QDialog.Accepted:
            return

        config = dialog.get_values()

        # ----- Validaciones -----
        if not config["save_name"]:
            QMessageBox.warning(self.main_window, "Nombre inválido", "El nombre del archivo no puede estar vacío.")
            return

        # Validar early stopping y validación
        if config["early_stopping_patience"] > 0 and config["valid_split"] <= 0:
            QMessageBox.warning(
                self.main_window,
                "Configuración inválida",
                "Para usar Early Stopping, el porcentaje de validación debe ser mayor que 0."
            )
            return

        # ----- Ejecutar entrenamiento -----
        self.main_window.training_controller.entrenar_desde_pt(
            pt_file = config["pt_file"],
            model_type=config["modelo"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            lr=config["lr"],
            valid_split=config["valid_split"],
            model_name=config['save_name'],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            patience=config["early_stopping_patience"],
            atom_emb_dim = config["atom_emb_pr"],
            hibrid_emb_dim = config["hibrid_emb_pr"],
            bond_emb_dim = config["bond_emb_pr"]
        )

    def entrenar_multiples_modelos_gnn(self):
        dialog = TrainMultipleModelsDialog(self.main_window)

        if dialog.exec() != QDialog.Accepted:
            return

        config = dialog.get_values()

        # ----- Validaciones -----
        if not config.get("target_file") or not os.path.isfile(config["target_file"]):
            QMessageBox.warning(self.main_window, "Error", "Debes seleccionar un archivo .txt válido con los targets.")
            return

        # ----- Ejecutar entrenamiento múltiple -----
        self.main_window.training_controller.train_multiple_models(
            train_dir=config["train_sdf_dir"],
            val_dir=config["val_sdf_dir"],
            target_file=config["target_file"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            lr=config["lr"],
            hidden_dim=config["hidden_dim"],
            patience=config["patience"],
            atom_emb_dim = config["atom_emb_pr"],
            hibrid_emb_dim = config["hibrid_emb_pr"],
            bond_emb_dim = config["bond_emb_pr"]
        )