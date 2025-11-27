from PySide6.QtWidgets import QMenuBar, QFileDialog, QMessageBox, QInputDialog, QDialog
from PySide6.QtGui import QAction
import torch
import os
from rdkit import Chem
import logging

from ui.dialogs.train_config_dialog import TrainConfigDialog
from ui.dialogs.model_test_dialog import ModelTestDialog
from ui.dialogs.batch_model_test_dialog import BatchModelTestDialog
from ui.dialogs.transfer_learning_dialog import TransferLearningDialog
from ui.dialogs.split_sdf_dialog import SDFSplitDialog
from ui.dialogs.test_all_models_dialog import BatchAllModelsTestDialog
from ui.dialogs.train_multiple_models import TrainMultipleModelsDialog
from ui.dialogs.transfer_learning_multiple_models import TransferLearningMultipleDialog
from ui.dialogs.image_dialog import ImageDialog
from ui.dialogs.csv_to_sdf import CSVtoSDFDialog

from ML.model_tester import test_model_on_directory
from ML.model_tester import obtener_info_checkpoint
from ML.model_explainer import obtener_lime
from ML.model_tester import cargar_y_predecir
from core.sdf_converter import graph_to_mol, save_graph_as_sdf, split_sdf, smiles_csv_to_sdf_dir

logger = logging.getLogger(__name__)

class MenuBar(QMenuBar):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent  # Referencia a MainWindow

        #Menu de Archivo    
        menu_molecula = self.addMenu("Molécula")

        nuevo_action = QAction("Nueva", self)
        nuevo_action.triggered.connect(self.nuevo_archivo)
        menu_molecula.addAction(nuevo_action)

        cargar_action = QAction("Cargar", self)
        cargar_action.triggered.connect(self.cargar_archivo)
        menu_molecula.addAction(cargar_action)

        guardar_action = QAction("Guardar", self)
        guardar_action.triggered.connect(self.guardar_archivo)
        menu_molecula.addAction(guardar_action)

        verificar_action = QAction("Verificar Molécula", self)
        verificar_action.triggered.connect(self.verificar_molecula)
        menu_molecula.addAction(verificar_action)

        dividir_action = QAction("Dividir SDF", self)
        dividir_action.triggered.connect(self.dividir_sdf)
        menu_molecula.addAction(dividir_action)

        dividir_csv_action = QAction("CSV a SDF", self)
        dividir_csv_action.triggered.connect(self.csv_a_sdf)
        menu_molecula.addAction(dividir_csv_action)

        # Menu de Entrenamiento
        menu_train = self.addMenu("Train GNN")

        # Entrenamiento de IA
        entrenar_action = QAction("Entrenar Modelo", self)
        entrenar_action.triggered.connect(self.entrenar_ia)
        menu_train.addAction(entrenar_action)

        # Entrenamiento de múltiples modelos
        entrenar_multiple_action = QAction("Entrenar Múltiples Modelos", self)
        entrenar_multiple_action.triggered.connect(self.entrenar_multiples_modelos)
        menu_train.addAction(entrenar_multiple_action)

        # Menu de Transfer Learning
        menu_transfer = self.addMenu("Transfer Learning GNN")

        # Transfer Learning
        transfer_action = QAction("Transfer Learning", self)
        transfer_action.triggered.connect(self.transfer_learning_ia)
        menu_transfer.addAction(transfer_action)

        # Transfer Learning con múltiples modelos
        transfer_multiple_action = QAction("Transfer Learning Múltiples Modelos", self)
        transfer_multiple_action.triggered.connect(self.transfer_learning_multiple_modelos)
        menu_transfer.addAction(transfer_multiple_action)

        # Feature Extraction con múltiples modelos
        feature_extraction_multiple_action = QAction("Feature Extraction Múltiples Modelos", self)
        feature_extraction_multiple_action.triggered.connect(self.feature_extraction_multiples_modelos)
        menu_transfer.addAction(feature_extraction_multiple_action)

        # Fine Tuning con múltiples modelos
        fine_tuning_multiple_action = QAction("Fine Tuning Múltiples Modelos", self)
        fine_tuning_multiple_action.triggered.connect(self.fine_tuning_multiples_modelos)
        menu_transfer.addAction(fine_tuning_multiple_action)    

        menu_test = self.addMenu("Test GNN")

        # Testeo de IA con un solo SDF
        testeo_action = QAction("Predecir SDF", self)
        testeo_action.triggered.connect(self.testear_modelo)
        menu_test.addAction(testeo_action)

        # Testeo de IA con múltiples SDF
        testeo_batch_action = QAction("Testear modelo", self)
        testeo_batch_action.triggered.connect(self.testear_modelo_en_batch)
        menu_test.addAction(testeo_batch_action)

        # Testeo de TODOS los modelos en un directorio
        testeo_all_models_action = QAction("Testear todos los modelos", self)
        testeo_all_models_action.triggered.connect(self.testear_directorio_modelos)
        menu_test.addAction(testeo_all_models_action)

        # Consultar parámetros modelo
        consultar_params_action = QAction("Consultar modelo", self)
        consultar_params_action.triggered.connect(self.consultar_parametros_modelo)
        menu_test.addAction(consultar_params_action)

        # LIME
        menu_lime = self.addMenu("Explicador LIME")
        lime_action =QAction("Obtener explicación LIME", self)
        lime_action.triggered.connect(self.get_explanation_LIME)
        menu_lime.addAction(lime_action)

    def nuevo_archivo(self):
        self.parent.create_new_graph()

    def cargar_archivo(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.parent,
            "Seleccionar archivo SDF",
            "",
            "Archivos SDF (*.sdf);;Todos los archivos (*)"
        )
        if file_path:
            self.parent.load_graph_from_file(file_path)


    def guardar_archivo(self):
        if not self.parent.graph_view:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self.parent,
            "Guardar archivo SDF",
            "",
            "Archivos SDF (*.sdf)"
        )
        if not file_path:
            return

        graph = self.parent.graph_view.scene().graph

        try:
            save_graph_as_sdf(graph, file_path)
            mensaje = f"Archivo guardado correctamente en: {file_path}"
            logger.info(mensaje)
            # self.parent.load_graph_from_file(file_path)  # Recargar el grafo guardado
        except Exception as e:
            QMessageBox.critical(
                self.parent,
                "Error al guardar",
                str(e)
            )

    def verificar_molecula(self):
        if not self.parent.graph_view:
            mensaje = "No hay una molécula cargada para verificar."
            QMessageBox.warning(self.parent, "Verificación", mensaje)
            return

        try:
            mol = graph_to_mol(self.parent.graph_view.scene().graph)
            Chem.SanitizeMol(mol)
            mensaje = "La molécula no contiene errores químicos detectables."
            logger.info(mensaje)
        except Exception as e:
            mensaje = f"Se detectaron errores químicos en la molécula:\n{str(e)}"
            logger.error(mensaje)

    def dividir_sdf(self):
        dialog = SDFSplitDialog(self.parent)
        if dialog.exec_() != QDialog.Accepted:
            return

        config = dialog.get_values()

        # Validaciones básicas
        if not config["sdf_file"] or not os.path.isfile(config["sdf_file"]):
            QMessageBox.warning(self.parent, "Error", "Debes seleccionar un archivo SDF válido.")
            return

        if not config["output_dir"]:
            QMessageBox.warning(self.parent, "Error", "Debes seleccionar un directorio de salida.")
            return

        try:
            split_sdf(config["sdf_file"], config["output_dir"])
            mensaje = f"SDF dividido correctamente. Archivos guardados en: {config['output_dir']}"
            logger.info(mensaje)
        except Exception as e:
            logger.error(f"Error al dividir SDF: {str(e)}", exc_info=True)

    def csv_a_sdf(self):
        dialog = CSVtoSDFDialog(self.parent)
        if dialog.exec_() != QDialog.Accepted:
            return

        config = dialog.get_values()

        # Validaciones
        if not config["csv_file"] or not os.path.isfile(config["csv_file"]):
            QMessageBox.warning(self.parent, "Error", "Debes seleccionar un archivo CSV válido.")
            return

        if not config["output_dir"]:
            QMessageBox.warning(self.parent, "Error", "Debes seleccionar un directorio de salida.")
            return

        try:
            smiles_csv_to_sdf_dir(config["csv_file"], config["output_dir"])
            mensaje = f"CSV dividido correctamente. SDFs guardados en: {config['output_dir']}"
            logger.info(mensaje)
        except Exception as e:
            QMessageBox.critical(self.parent, "Error al dividir CSV", str(e))
            logger.error(f"Error al dividir CSV: {str(e)}", exc_info=True)

        

    def entrenar_ia(self):
        dialog = TrainConfigDialog(self.parent)

        if dialog.exec() != QDialog.Accepted:
            return

        config = dialog.get_values()

        # ----- Validaciones -----
        if not config["sdf_dir"] or not os.path.isdir(config["sdf_dir"]):
            QMessageBox.warning(self.parent, "Error", "Debes seleccionar un directorio válido con archivos SDF.")
            return

        if not config["target_file"] or not os.path.isfile(config["target_file"]):
            QMessageBox.warning(self.parent, "Error", "Debes seleccionar un archivo .txt válido con los targets.")
            return

        if not config["save_name"]:
            QMessageBox.warning(self.parent, "Nombre inválido", "El nombre del archivo no puede estar vacío.")
            return

        # Validar early stopping y validación
        if config["early_stopping_patience"] > 0 and config["valid_split"] <= 0:
            QMessageBox.warning(
                self.parent,
                "Configuración inválida",
                "Para usar Early Stopping, el porcentaje de validación debe ser mayor que 0."
            )
            return

        # ----- Ejecutar entrenamiento -----
        self.parent.training_controller.entrenar(
            sdf_dir=config["sdf_dir"],
            target_file=config["target_file"],
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


    def transfer_learning_ia(self):
        dialog = TransferLearningDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return

        config = dialog.get_values()

        # ---------- Validaciones básicas ----------
        if not config["sdf_dir"] or not os.path.isdir(config["sdf_dir"]):
            QMessageBox.warning(self.parent, "Error", "Debes seleccionar un directorio válido con archivos SDF.")
            return

        if not config["target_file"] or not os.path.isfile(config["target_file"]):
            QMessageBox.warning(self.parent, "Error", "Debes seleccionar un archivo .txt válido con los targets.")
            return

        if not config["pretrained_model_path"] or not os.path.isfile(config["pretrained_model_path"]):
            QMessageBox.warning(self.parent, "Error", "Debes seleccionar un modelo preentrenado válido (.pt).")
            return

        if not config["save_name"]:
            QMessageBox.warning(self.parent, "Nombre inválido", "El nombre del archivo no puede estar vacío.")
            return

        # Validar early stopping y validación
        if config["early_stopping_patience"] > 0 and config["valid_split"] <= 0:
            QMessageBox.warning(
                self.parent,
                "Configuración inválida",
                "Para usar Early Stopping, el porcentaje de validación debe ser mayor que 0."
            )
            return

        # ---------- Ejecutar Transfer Learning ----------
        self.parent.training_controller.transfer_learning(
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
            atom_emb_dim = config["atom_emb_pr"],
            hibrid_emb_dim = config["hibrid_emb_pr"],
            bond_emb_dim = config["bond_emb_pr"]
        )

    def entrenar_multiples_modelos(self):
        dialog = TrainMultipleModelsDialog(self.parent)

        if dialog.exec() != QDialog.Accepted:
            return

        config = dialog.get_values()

        # ----- Validaciones -----
        if not config.get("sdf_dir") or not os.path.isdir(config["sdf_dir"]):
            QMessageBox.warning(self.parent, "Error", "Debes seleccionar un directorio válido con archivos SDF.")
            return
        if not config.get("target_file") or not os.path.isfile(config["target_file"]):
            QMessageBox.warning(self.parent, "Error", "Debes seleccionar un archivo .txt válido con los targets.")
            return

        # Validar early stopping y porcentaje de validación
        if config.get("patience", 0) > 0 and config.get("valid_split", 0) <= 0:
            QMessageBox.warning(
                self.parent,
                "Configuración inválida",
                "Para usar Early Stopping, el porcentaje de validación debe ser mayor que 0."
            )
            return

        # ----- Ejecutar entrenamiento múltiple -----
        self.parent.training_controller.train_multiple_models(
            sdf_dir=config["sdf_dir"],
            target_file=config["target_file"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            lr=config["lr"],
            valid_split=config["valid_split"],
            hidden_dim=config["hidden_dim"],
            patience=config["patience"],
            atom_emb_dim = config["atom_emb_pr"],
            hibrid_emb_dim = config["hibrid_emb_pr"],
            bond_emb_dim = config["bond_emb_pr"]
        )


    def transfer_learning_multiple_modelos(self):
        dialog = TransferLearningMultipleDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return

        config = dialog.get_values()

        # ---------- Validaciones básicas ----------
        if not config["sdf_dir"] or not os.path.isdir(config["sdf_dir"]):
            QMessageBox.warning(self.parent, "Error", "Debes seleccionar un directorio válido con archivos SDF.")
            return

        if not config["target_file"] or not os.path.isfile(config["target_file"]):
            QMessageBox.warning(self.parent, "Error", "Debes seleccionar un archivo .txt válido con los targets.")
            return

        if not config["pretrained_model_directory_path"] or not os.path.isdir(config["pretrained_model_directory_path"]):
            QMessageBox.warning(self.parent, "Error", "Debes seleccionar un directorio de modelos preentrenados válido.")
            return

        # Validar early stopping y validación
        if config["patience"] > 0 and config["valid_split"] <= 0:
            QMessageBox.warning(
                self.parent,
                "Configuración inválida",
                "Para usar Early Stopping, el porcentaje de validación debe ser mayor que 0."
            )
            return

        # ---------- Ejecutar Transfer Learning múltiple ----------
        self.parent.training_controller.transfer_train_multiple_models(
            pretrained_model_directory_path=config["pretrained_model_directory_path"],
            sdf_dir=config["sdf_dir"],
            target_file=config["target_file"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            lr=config["lr"],
            valid_split=config["valid_split"],
            patience=config["patience"]
        )

    def feature_extraction_multiples_modelos(self):
        dialog = TransferLearningMultipleDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return

        config = dialog.get_values()

        # ---------- Validaciones básicas ----------
        if not config["sdf_dir"] or not os.path.isdir(config["sdf_dir"]):
            QMessageBox.warning(self.parent, "Error", "Debes seleccionar un directorio válido con archivos SDF.")
            return

        if not config["target_file"] or not os.path.isfile(config["target_file"]):
            QMessageBox.warning(self.parent, "Error", "Debes seleccionar un archivo .txt válido con los targets.")
            return

        if not config["pretrained_model_directory_path"] or not os.path.isdir(config["pretrained_model_directory_path"]):
            QMessageBox.warning(self.parent, "Error", "Debes seleccionar un directorio de modelos preentrenados válido.")
            return

        # Validar early stopping y validación
        if config["patience"] > 0 and config["valid_split"] <= 0:
            QMessageBox.warning(
                self.parent,
                "Configuración inválida",
                "Para usar Early Stopping, el porcentaje de validación debe ser mayor que 0."
            )
            return

        # ---------- Ejecutar Transfer Learning múltiple ----------
        self.parent.training_controller.feature_extraction_multiple_models(
            pretrained_model_directory_path=config["pretrained_model_directory_path"],
            sdf_dir=config["sdf_dir"],
            target_file=config["target_file"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            lr=config["lr"],
            valid_split=config["valid_split"],
            patience=config["patience"]
        )

    def fine_tuning_multiples_modelos(self):
        dialog = TransferLearningMultipleDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return

        config = dialog.get_values()

        # ---------- Validaciones básicas ----------
        if not config["sdf_dir"] or not os.path.isdir(config["sdf_dir"]):
            QMessageBox.warning(self.parent, "Error", "Debes seleccionar un directorio válido con archivos SDF.")
            return

        if not config["target_file"] or not os.path.isfile(config["target_file"]):
            QMessageBox.warning(self.parent, "Error", "Debes seleccionar un archivo .txt válido con los targets.")
            return

        if not config["pretrained_model_directory_path"] or not os.path.isdir(config["pretrained_model_directory_path"]):
            QMessageBox.warning(self.parent, "Error", "Debes seleccionar un directorio de modelos preentrenados válido.")
            return

        # Validar early stopping y validación
        if config["patience"] > 0 and config["valid_split"] <= 0:
            QMessageBox.warning(
                self.parent,
                "Configuración inválida",
                "Para usar Early Stopping, el porcentaje de validación debe ser mayor que 0."
            )
            return

        # ---------- Ejecutar Transfer Learning múltiple ----------
        self.parent.training_controller.fine_tuning_multiple_models(
            pretrained_model_directory_path=config["pretrained_model_directory_path"],
            sdf_dir=config["sdf_dir"],
            target_file=config["target_file"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            lr=config["lr"],
            valid_split=config["valid_split"],
            patience=config["patience"]
        )



    def testear_modelo(self):
        dialog = ModelTestDialog(self.parent)
        if dialog.exec():
            model_path, sdf_path = dialog.get_paths()
            try:
                pred, target_name = cargar_y_predecir(model_path, sdf_path)

                model_name = os.path.basename(model_path)
                sdf_name = os.path.basename(sdf_path)
                msg = f"Predicción de '{target_name}' con el modelo '{model_name}' en la molécula '{sdf_name}': {pred:.4f}"
                logger.info(msg)

            except Exception as e:
                logger.error(f"Error en testear modelo: {str(e)}", exc_info=True)

    def testear_modelo_en_batch(self):
        
        dialog = BatchModelTestDialog(self.parent)
        if dialog.exec():
            model_path, sdf_dir, targets_file = dialog.get_paths()

            try:
                # Ejecutar función de testeo
                plot_path = test_model_on_directory(model_path, sdf_dir, targets_file)

                # Mostrar scatter plot
                self.image_dialog = ImageDialog(plot_path, self.parent)
                self.image_dialog.show()

            except Exception as e:
                logger.error("Error en testeo por lotes: " + str(e), exc_info=True)

    def testear_directorio_modelos(self):
        dialog = BatchAllModelsTestDialog(self.parent)
        if dialog.exec():
            models_dir, sdf_dir, targets_file = dialog.get_paths()

            try:
                # Ejecutamos testing con el proceso
                self.parent.testing_controller.testear_modelos(models_dir, sdf_dir, targets_file)
            except Exception as e:
                logger.error("Error en testeo de todos los modelos: " + str(e), exc_info=True)

    def get_explanation_LIME(self):
        dialog = ModelTestDialog(self.parent)
        if dialog.exec():
            model_path, sdf_path = dialog.get_paths()
            try:
                # Obtener explicación LIME
                # feature_mask espera: [Atom, Degree, Arom, Hybrid, BondType, BondDist]
                plot_path = obtener_lime(model_path, sdf_path, num_samples=200, noise_level=0.1, device='cpu')

                # mostrar el sdf por pantalla
                self.parent.load_graph_from_file(sdf_path)

                # Mostrar la imagen en un diálogo
                self.image_dialog = ImageDialog(plot_path, self.parent)
                self.image_dialog.show()

            except Exception as e:
                logger.error(f"Error en explicación LIME: {str(e)}", exc_info=True)


    def consultar_parametros_modelo(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.parent,
            "Seleccionar archivo de modelo (.pt)",
            "",
            "Modelos (*.pt)"
        )
        if not file_path:
            return

        info = obtener_info_checkpoint(file_path)
        logger.info(info)