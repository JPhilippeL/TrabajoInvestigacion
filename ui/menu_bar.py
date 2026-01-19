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
from ui.dialogs.explanation_dialog import ExplanationDialog
from ui.dialogs.explainer_comparer_dialog import ExplainerComparerDialog
from ui.dialogs.batch_explainer_comparer_dialog import BatchComparerDialog

from ML.model_tester import test_model_on_directory,cargar_y_predecir, cargar_modelo, obtener_info_checkpoint
from ML.explainers.model_Graph_explainer import obtener_graph_explainer
from ML.explainers.model_GNNExplainer import obtener_GNN_Explainer
from ML.explainers.explanation_fidelity import generar_comparativa_fidelity, save_auc_results_csv, calcular_aucs_fidelity_batch
from ML.data_processing import read_targets, mol_to_graph_data
from graph_managment.sdf_converter import graph_to_mol, save_graph_as_sdf, split_sdf, smiles_csv_to_sdf_dir

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


        # Explicaciones
        menu_explicacion = self.addMenu("Explicadores")
        
        # Graph_explainer
        graph_explainer_action =QAction("Obtener GraphExplainer", self)
        graph_explainer_action.triggered.connect(self.get_explanation_GraphExplainer)
        menu_explicacion.addAction(graph_explainer_action)

        # GNN Explainer
        gnn_explainer_action = QAction("Obtener GNNExplainer", self)
        gnn_explainer_action.triggered.connect(self.get_explanation_GNNExplainer)
        menu_explicacion.addAction(gnn_explainer_action)

        # Batch Graph_explainer
        batch_graph_explainer_action =QAction("Obtener GraphExplainer de Directorio", self)
        batch_graph_explainer_action.triggered.connect(self.get_batch_explanation_GraphExplainer)
        menu_explicacion.addAction(batch_graph_explainer_action)

        # Batch GNN Explainer
        batch_gnn_explainer_action = QAction("Obtener GNNExplainer de Directorio", self)
        batch_gnn_explainer_action.triggered.connect(self.get_batch_explanation_GNNExplainer)
        menu_explicacion.addAction(batch_gnn_explainer_action)

        # Comparador
        explanation_comparer_action = QAction("Comparar Explicadores", self)
        explanation_comparer_action.triggered.connect(self.get_explanation_comparer)
        menu_explicacion.addAction(explanation_comparer_action)

        # batch_Comparador
        batch_explanation_comparer_action = QAction("Comparar Explicadores Batch", self)
        batch_explanation_comparer_action.triggered.connect(self.get_batch_explanation_comparer)
        menu_explicacion.addAction(batch_explanation_comparer_action)

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
            logger.warning(mensaje)
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

    def get_explanation_GraphExplainer(self):
        dialog = ExplanationDialog(self.parent)
        if dialog.exec():
            model_path, sdf_path, target_path = dialog.get_paths()
            try:
                # Obtener explicación GraphExplanation
                # feature_mask espera: [Atom, Degree, Arom, Hybrid, BondType, BondDist]
                plot_path = obtener_graph_explainer(model_path, sdf_path, target_path, num_samples=1000, noise_level=0.01, device='cpu')

                # mostrar el sdf por pantalla
                self.parent.load_graph_from_file(sdf_path)

                # Mostrar la imagen en un diálogo
                self.image_dialog = ImageDialog(plot_path, self.parent)
                self.image_dialog.show()

            except Exception as e:
                logger.error(f"Error en explicación GraphExplanation: {str(e)}", exc_info=True)
    
    def get_explanation_GNNExplainer(self):
        dialog = ExplanationDialog(self.parent)
        if dialog.exec():
            model_path, sdf_path, target_path = dialog.get_paths()
            try:
                # Obtener explicación Explain er
                plot_path = obtener_GNN_Explainer(model_path, sdf_path, target_path)

                # mostrar el sdf por pantalla
                self.parent.load_graph_from_file(sdf_path)

                # Mostrar la imagen en un diálogo
                self.image_dialog = ImageDialog(plot_path, self.parent)
                self.image_dialog.show()

            except Exception as e:
                logger.error(f"Error en explicación GNNExplainer: {str(e)}", exc_info=True)

    def get_batch_explanation_GraphExplainer(self):
        """
        Procesa un directorio completo de archivos SDF usando GraphExplainer.
        Reutiliza BatchModelTestDialog para seleccionar Modelo, Directorio y Targets.
        """
        dialog = BatchModelTestDialog(self.parent)
        if dialog.exec():
            model_path, directory_path, target_path = dialog.get_paths()
            try:
                # Filtrar archivos .sdf
                sdf_files = [f for f in os.listdir(directory_path) if f.endswith('.sdf')]

                if not sdf_files:
                    logger.warning(f"No se encontraron archivos .sdf en {directory_path}")
                    return

                logger.info(f"Iniciando procesamiento batch GraphExplainer para {len(sdf_files)} archivos.")

                for sdf_file in sdf_files:
                    full_sdf_path = os.path.join(directory_path, sdf_file)
                    
                    # Llamada a la función explicadora
                    # Asumimos que esta función ya gestiona el guardado de la imagen internamente
                    obtener_graph_explainer(
                        model_path, 
                        full_sdf_path, 
                        target_path, 
                        num_samples=1000, 
                        noise_level=0.01, 
                        device='cpu',
                        imagen = False
                    )
                    logger.info(f"Procesado: {sdf_file}")

                logger.info("Procesamiento batch GraphExplainer finalizado.")

            except Exception as e:
                logger.error(f"Error en batch GraphExplainer: {str(e)}", exc_info=True)

    def get_batch_explanation_GNNExplainer(self):
        """
        Procesa un directorio completo de archivos SDF usando GNNExplainer.
        Reutiliza BatchModelTestDialog para seleccionar Modelo, Directorio y Targets.
        """
        dialog = BatchModelTestDialog(self.parent)
        if dialog.exec():
            model_path, directory_path, target_path = dialog.get_paths()
            try:
                # Filtrar archivos .sdf
                sdf_files = [f for f in os.listdir(directory_path) if f.endswith('.sdf')]

                if not sdf_files:
                    logger.warning(f"No se encontraron archivos .sdf en {directory_path}")
                    return

                logger.info(f"Iniciando procesamiento batch GNNExplainer para {len(sdf_files)} archivos.")

                for sdf_file in sdf_files:
                    full_sdf_path = os.path.join(directory_path, sdf_file)

                    # Llamada a la función explicadora
                    # Asumimos que esta función ya gestiona el guardado de la imagen internamente
                    obtener_GNN_Explainer(model_path, full_sdf_path, target_path, imagen=False)
                    
                    logger.info(f"Procesado: {sdf_file}")

                logger.info("Procesamiento batch GNNExplainer finalizado.")

            except Exception as e:
                logger.error(f"Error en batch GNNExplainer: {str(e)}", exc_info=True)

    def get_explanation_comparer(self):
        dialog = ExplainerComparerDialog(self.parent)
        if dialog.exec():
            # 1. Recuperar los 5 valores
            model_path, sdf_path, graphexplanation_path, gnn_path_raw, mode = dialog.get_inputs()
            
            # 2. Lógica para manejar GNNExplainer como opcional/None
            # Si el string está vacío (usuario no seleccionó nada), pasamos None.
            if not gnn_path_raw.strip():
                gnnexplanation_path = None
            else:
                gnnexplanation_path = gnn_path_raw

            try:
                # 3. Llamar a la función generadora pasando el MODO y el path (que puede ser None)
                plot_path, auc_graph_explainer, auc_gnn_explainer = generar_comparativa_fidelity(
                    model_path, 
                    sdf_path, 
                    graphexplanation_path, 
                    gnnexplanation_path, 
                    mode=mode  # <--- Pasamos el modo seleccionado
                )

                # 4. Mostrar resultado si se generó
                if plot_path:
                    self.image_dialog = ImageDialog(plot_path, self.parent)
                    self.image_dialog.show()

            except Exception as e:
                logger.error(f"Error en explicación Comparativa ({mode}): {str(e)}", exc_info=True)

    def get_batch_explanation_comparer(self):
        """
        Calcula métricas de fidelidad para todo un directorio de SDFs comparando
        GraphExplainer vs GNNExplainer (si existe), buscando los pesos automáticamente.
        """
        # Suponemos que este diálogo devuelve:
        # model_path: Ruta al modelo .pt
        # sdfs_dir: Ruta al directorio con los .sdf
        # weights_root_dir: Ruta raíz de los pesos (dentro debe haber carpetas alpha, beta, etc.)
        # mode: String 'alpha', 'beta', 'gamma' o 'delta'
        dialog = BatchComparerDialog(self.parent)

        UMBRAL_ERROR = 0.6767
        
        if dialog.exec():
            model_path, sdfs_dir, weights_root_dir, targets_path, mode = dialog.get_inputs()
            
            # Construir la ruta específica del modo (ej: .../pesos/alpha)
            weights_mode_dir = os.path.join(weights_root_dir, mode)
            
            if not os.path.exists(weights_mode_dir):
                logger.error(f"No existe el directorio de pesos para el modo {mode}: {weights_mode_dir}")
                return

            results = []  # Lista para guardar diccionarios: {'name': str, 'auc_graph': float, 'auc_gnn': float}
            
            try:
                # Filtrar solo archivos .sdf
                sdf_files = [f for f in os.listdir(sdfs_dir) if f.endswith('.sdf')]
                
                if not sdf_files:
                    logger.warning(f"No hay archivos .sdf en {sdfs_dir}")
                    return

                logger.info(f"Iniciando comparativa Batch ({mode}). Total archivos: {len(sdf_files)}")

                # Listar todos los archivos de pesos una sola vez para no leer disco en cada iteración
                # Esto mejora el rendimiento si hay muchos archivos.
                all_weight_files = os.listdir(weights_mode_dir)

                # Cargar modelo
                model, device, targetname = cargar_modelo(model_path)
                targets_dict = read_targets(targets_path)

                for sdf_file in sdf_files:
                    mol_name = os.path.splitext(sdf_file)[0] # Nombre sin extensión (el "componente")
                    full_sdf_path = os.path.join(sdfs_dir, sdf_file)

                    # --- FILTRO DE ERROR ---
                    
                    # A) Obtener Valor Real
                    if mol_name not in targets_dict:
                        logger.warning(f"Saltando {mol_name}: No tiene valor target asociado.")
                        continue
                    y_real = targets_dict[mol_name]

                    # B) Obtener Valor Predicho (Inferencia rápida)
                    try:
                        mol = Chem.SDMolSupplier(full_sdf_path, removeHs=False)[0] # Ojo con removeHs
                        if mol is None: continue
                        data = mol_to_graph_data(mol).to(device)
                        
                        with torch.no_grad():
                            pred_tensor = model(data.x, data.edge_index, data.edge_attr, data.batch)
                            y_pred = pred_tensor.item()
                    except Exception as e:
                        logger.error(f"Error en inferencia {mol_name}: {e}")
                        continue

                    # C) Calcular Error y Filtrar
                    error_abs = abs(y_real - y_pred)
                    
                    if error_abs >= UMBRAL_ERROR:
                        # Si el error es grande, saltamos esta molécula
                        # logger.info(f"Saltando {mol_name}: Error {error_abs:.4f} > {UMBRAL_ERROR}")
                        continue

                    matches = []
                    for w in all_weight_files:
                        if w.endswith(f"_{mol_name}.pt"):
                            matches.append(w)
                    
                    if not matches:
                        logger.warning(f"Saltando {mol_name}: No se encontraron pesos en {mode}.")
                        continue

                    # Identificar cuál es cual
                    path_graph_explainer = None
                    path_gnn_explainer = None

                    for w_file in matches:
                        full_w_path = os.path.join(weights_mode_dir, w_file)
                        if "GraphExplainer" in w_file:
                            path_graph_explainer = full_w_path
                        elif "GNNExplainer" in w_file: 
                            # Asumimos que si no es GraphExplainer y hizo match, es el GNNExplainer
                            # O buscamos explícitamente el string si tus archivos lo tienen.
                            path_gnn_explainer = full_w_path
                    
                    # Verificar requisitos mínimos
                    if not path_graph_explainer and not path_gnn_explainer:
                         logger.warning(f"Saltando {mol_name}: Archivos encontrados pero no se identificó el tipo de explainer.")
                         continue

                    # --- Llamada a la función generadora ---
                    try:
                        # LLAMADA NUEVA OPTIMIZADA
                        auc_graph, auc_gnn = calcular_aucs_fidelity_batch(
                            model, device,
                            full_sdf_path, 
                            path_graph_explainer, 
                            path_gnn_explainer, 
                            mode=mode
                        )
                        
                        # Guardar en memoria solo si el cálculo fue exitoso
                        if auc_graph is not None:
                            results.append({
                                "name": mol_name,
                                "auc_graph": auc_graph,
                                "auc_gnn": auc_gnn if auc_gnn is not None else "N/A"
                            })
                            logger.info(f"Procesado {mol_name} | G: {auc_graph:.4f}")
                        else:
                             logger.warning(f"Fallo cálculo para {mol_name}")

                    except Exception as e_inner:
                        logger.error(f"Error procesando {mol_name}: {e_inner}")

                # --- Guardar resultados finales ---
                if results:
                    model_name_clean = os.path.splitext(os.path.basename(model_path))[0]

                    # Llamada a la función externa actualizada
                    save_auc_results_csv(results, mode, model_name_clean)
                else:
                    logger.warning("No se generaron resultados para guardar.")

            except Exception as e:
                logger.error(f"Error global en Batch Comparer: {str(e)}", exc_info=True)


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