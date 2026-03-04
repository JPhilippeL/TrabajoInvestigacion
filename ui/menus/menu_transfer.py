from PySide6.QtWidgets import QMessageBox, QDialog, QMenu
from PySide6.QtGui import QAction
import os
import logging

from ui.dialogs.transfer_learning_dialog import TransferLearningDialog
from ui.dialogs.transfer_learning_multiple_models import TransferLearningMultipleDialog

logger = logging.getLogger(__name__)

class MenuTransferGNN(QMenu):
    def __init__(self, parent_window):
        # "parent_window" es tu MainWindow, lo necesitamos para los diálogos
        super().__init__("Transfer GNN", parent_window)
        self.main_window = parent_window 

        self.init_actions()

    def init_actions(self):
        # Menu de Transfer Learning

        # Transfer Learning
        transfer_action = QAction("Transfer Learning", self)
        transfer_action.triggered.connect(self.transfer_learning_ia)
        self.addAction(transfer_action)

        # Transfer Learning con múltiples modelos
        transfer_multiple_action = QAction("Transfer Learning Múltiples Modelos", self)
        transfer_multiple_action.triggered.connect(self.transfer_learning_multiple_modelos)
        self.addAction(transfer_multiple_action)

        # # Feature Extraction con múltiples modelos
        # feature_extraction_multiple_action = QAction("Feature Extraction Múltiples Modelos", self)
        # feature_extraction_multiple_action.triggered.connect(self.feature_extraction_multiples_modelos)
        # self.addAction(feature_extraction_multiple_action)

        # # Fine Tuning con múltiples modelos
        # fine_tuning_multiple_action = QAction("Fine Tuning Múltiples Modelos", self)
        # fine_tuning_multiple_action.triggered.connect(self.fine_tuning_multiples_modelos)
        # self.addAction(fine_tuning_multiple_action)

    def transfer_learning_ia(self):
        dialog = TransferLearningDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return

        config = dialog.get_values()

        # ---------- Validaciones básicas ----------
        if not config["sdf_dir"] or not os.path.isdir(config["sdf_dir"]):
            QMessageBox.warning(self.main_window, "Error", "Debes seleccionar un directorio válido con archivos SDF.")
            return

        if not config["target_file"] or not os.path.isfile(config["target_file"]):
            QMessageBox.warning(self.main_window, "Error", "Debes seleccionar un archivo .txt válido con los targets.")
            return

        if not config["pretrained_model_path"] or not os.path.isfile(config["pretrained_model_path"]):
            QMessageBox.warning(self.main_window, "Error", "Debes seleccionar un modelo preentrenado válido (.pt).")
            return

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

        # ---------- Ejecutar Transfer Learning ----------
        self.main_window.training_controller.transfer_learning(
            sdf_dir=config["sdf_dir"],
            target_file=config["target_file"],
            pretrained_model_path=config["pretrained_model_path"],
            transfer_mode=config["transfer_mode"].lower().replace(" ", "_"),  # fine_tuning o feature_extraction
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            lr=config["lr"],
            valid_split=config["valid_split"],
            model_name=config["save_name"],
            patience=config["early_stopping_patience"],
            # atom_emb_dim = config["atom_emb_pr"],
            # hibrid_emb_dim = config["hibrid_emb_pr"],
            # bond_emb_dim = config["bond_emb_pr"]
        )

    def transfer_learning_multiple_modelos(self):
        dialog = TransferLearningMultipleDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return

        config = dialog.get_values()

        # ---------- Validaciones básicas ----------
        if not config["sdf_dir"] or not os.path.isdir(config["sdf_dir"]):
            QMessageBox.warning(self.main_window, "Error", "Debes seleccionar un directorio válido con archivos SDF.")
            return

        if not config["target_file"] or not os.path.isfile(config["target_file"]):
            QMessageBox.warning(self.main_window, "Error", "Debes seleccionar un archivo .txt válido con los targets.")
            return

        if not config["pretrained_model_directory_path"] or not os.path.isdir(config["pretrained_model_directory_path"]):
            QMessageBox.warning(self.main_window, "Error", "Debes seleccionar un directorio de modelos preentrenados válido.")
            return

        # Validar early stopping y validación
        if config["patience"] > 0 and config["valid_split"] <= 0:
            QMessageBox.warning(
                self.main_window,
                "Configuración inválida",
                "Para usar Early Stopping, el porcentaje de validación debe ser mayor que 0."
            )
            return

        # ---------- Ejecutar Transfer Learning múltiple ----------
        self.main_window.training_controller.transfer_train_multiple_models(
            pretrained_model_directory_path=config["pretrained_model_directory_path"],
            sdf_dir=config["sdf_dir"],
            target_file=config["target_file"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            lr=config["lr"],
            valid_split=config["valid_split"],
            patience=config["patience"]
        )

    # Esto es opcional, si se quiere hacer solo uno sobre muchos modelos, 
    # con el de transfer learning muchos modelos ya se hacen los dos.

    # def feature_extraction_multiples_modelos(self):
    #     dialog = TransferLearningMultipleDialog(self)
    #     if dialog.exec_() != QDialog.Accepted:
    #         return

    #     config = dialog.get_values()

    #     # ---------- Validaciones básicas ----------
    #     if not config["sdf_dir"] or not os.path.isdir(config["sdf_dir"]):
    #         QMessageBox.warning(self.main_window, "Error", "Debes seleccionar un directorio válido con archivos SDF.")
    #         return

    #     if not config["target_file"] or not os.path.isfile(config["target_file"]):
    #         QMessageBox.warning(self.main_window, "Error", "Debes seleccionar un archivo .txt válido con los targets.")
    #         return

    #     if not config["pretrained_model_directory_path"] or not os.path.isdir(config["pretrained_model_directory_path"]):
    #         QMessageBox.warning(self.main_window, "Error", "Debes seleccionar un directorio de modelos preentrenados válido.")
    #         return

    #     # Validar early stopping y validación
    #     if config["patience"] > 0 and config["valid_split"] <= 0:
    #         QMessageBox.warning(
    #             self.main_window,
    #             "Configuración inválida",
    #             "Para usar Early Stopping, el porcentaje de validación debe ser mayor que 0."
    #         )
    #         return

    #     # ---------- Ejecutar Transfer Learning múltiple ----------
    #     self.main_window.training_controller.feature_extraction_multiple_models(
    #         pretrained_model_directory_path=config["pretrained_model_directory_path"],
    #         sdf_dir=config["sdf_dir"],
    #         target_file=config["target_file"],
    #         epochs=config["epochs"],
    #         batch_size=config["batch_size"],
    #         lr=config["lr"],
    #         valid_split=config["valid_split"],
    #         patience=config["patience"]
    #     )

    # def fine_tuning_multiples_modelos(self):
    #     dialog = TransferLearningMultipleDialog(self)
    #     if dialog.exec_() != QDialog.Accepted:
    #         return

    #     config = dialog.get_values()

    #     # ---------- Validaciones básicas ----------
    #     if not config["sdf_dir"] or not os.path.isdir(config["sdf_dir"]):
    #         QMessageBox.warning(self.main_window, "Error", "Debes seleccionar un directorio válido con archivos SDF.")
    #         return

    #     if not config["target_file"] or not os.path.isfile(config["target_file"]):
    #         QMessageBox.warning(self.main_window, "Error", "Debes seleccionar un archivo .txt válido con los targets.")
    #         return

    #     if not config["pretrained_model_directory_path"] or not os.path.isdir(config["pretrained_model_directory_path"]):
    #         QMessageBox.warning(self.main_window, "Error", "Debes seleccionar un directorio de modelos preentrenados válido.")
    #         return

    #     # Validar early stopping y validación
    #     if config["patience"] > 0 and config["valid_split"] <= 0:
    #         QMessageBox.warning(
    #             self.main_window,
    #             "Configuración inválida",
    #             "Para usar Early Stopping, el porcentaje de validación debe ser mayor que 0."
    #         )
    #         return

    #     # ---------- Ejecutar Transfer Learning múltiple ----------
    #     self.main_window.training_controller.fine_tuning_multiple_models(
    #         pretrained_model_directory_path=config["pretrained_model_directory_path"],
    #         sdf_dir=config["sdf_dir"],
    #         target_file=config["target_file"],
    #         epochs=config["epochs"],
    #         batch_size=config["batch_size"],
    #         lr=config["lr"],
    #         valid_split=config["valid_split"],
    #         patience=config["patience"]
    #     )