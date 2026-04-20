from PySide6.QtWidgets import QMenu
from PySide6.QtGui import QAction
import torch
import os
from rdkit import Chem
import logging

from ui.dialogs.batch_explanation_dialog import BatchExplanationDialog
from ui.dialogs.image_dialog import ImageDialog
from ui.dialogs.explanation_dialog import ExplanationDialog
from ui.dialogs.explainer_comparer_dialog import ExplainerComparerDialog
from ui.dialogs.batch_explainer_comparer_dialog import BatchComparerDialog

from GNNs.explainers.graph_explainer_onehot import obtener_graph_explainer
from GNNs.explainers.model_TorchExplainers import obtener_Dummy_Explainer, obtener_Captum_Explainer, obtener_GNN_Explainer
from GNNs.explainers.explanation_fidelity import generar_comparativa_fidelity, obtener_aucs_directorio

logger = logging.getLogger(__name__)

class MenuExplainerGNN(QMenu):
    def __init__(self, parent_window):
        # "parent_window" es tu MainWindow, lo necesitamos para los diálogos
        super().__init__("Explainer GNN", parent_window)
        self.main_window = parent_window 

        self.init_actions()

    def init_actions(self):
        # Obtener explicador
        explicador_action = QAction("Obtener Explicación", self)
        explicador_action.triggered.connect(self.get_explanation)
        self.addAction(explicador_action)

        # Batch Graph_explainer
        batch_explainer_action =QAction("Obtener Explicacion de Directorio", self)
        batch_explainer_action.triggered.connect(self.get_batch_explanation)
        self.addAction(batch_explainer_action)

        # Comparador
        explanation_comparer_action = QAction("Comparar Explicadores", self)
        explanation_comparer_action.triggered.connect(self.get_explanation_comparer)
        self.addAction(explanation_comparer_action)

        # batch_Comparador
        batch_explanation_comparer_action = QAction("Comparar Explicadores Batch", self)
        batch_explanation_comparer_action.triggered.connect(self.get_batch_explanation_comparer)
        self.addAction(batch_explanation_comparer_action)

    def get_explanation(self):
        dialog = ExplanationDialog(self.main_window)
        if dialog.exec():
            # 1. Recuperamos las 4 variables del diálogo actualizado
            model_path, sdf_path, target_path, explainer_name = dialog.get_paths()
            
            try:
                plot_path = None
                logger.info(f"Iniciando flujo de explicación con: {explainer_name}")

                # 2. Enrutamiento según el explicador seleccionado
                if explainer_name == "GraphExplainer":
                    plot_path = obtener_graph_explainer(
                        model_path, sdf_path, target_path, 
                        num_samples=1000, noise_level=0.01, device='cpu'
                    )
                
                elif explainer_name == "GNNExplainer":
                    plot_path = obtener_GNN_Explainer(
                        model_path, sdf_path, target_path
                    )
                
                elif explainer_name.startswith("Captum_"):
                    # Extraemos la segunda parte del string (ej: "IntegratedGradients")
                    captum_method = explainer_name.split("_")[1]
                    # Asume que importaste obtener_Captum_Explainer al inicio del archivo
                    plot_path = obtener_Captum_Explainer(
                        model_path, sdf_path, target_path, 
                        imagen=True, captum_method=captum_method
                    )
                    
                elif explainer_name == "DummyExplainer":
                    plot_path = obtener_Dummy_Explainer(
                        model_path, sdf_path, target_path
                    )

                else:
                    logger.error(f"Explicador no reconocido: {explainer_name}")
                    return

                # 3. Código común para la interfaz (solo si se generó el plot)
                if plot_path:
                    # Mostrar el sdf por pantalla
                    self.main_window.load_graph_from_file(sdf_path)

                    # Mostrar la imagen generada en un diálogo
                    self.image_dialog = ImageDialog(plot_path, self.main_window)
                    self.image_dialog.show()

            except Exception as e:
                # El log ahora registra dinámicamente en qué explicador falló
                logger.error(f"Error en explicación con {explainer_name}: {str(e)}", exc_info=True)

    def get_batch_explanation(self):
        """
        Procesa un directorio completo de archivos SDF usando el explicador seleccionado.
        Utiliza BatchExplanationDialog para seleccionar Modelo, Directorio, Targets y Explicador.
        """
        # Asegúrate de haber importado BatchExplanationDialog al principio del archivo
        dialog = BatchExplanationDialog(self.main_window)
        if dialog.exec():
            # Recuperamos las 4 variables del diálogo
            model_path, directory_path, target_path, explainer_name = dialog.get_paths()
            
            try:
                # Filtrar archivos .sdf
                sdf_files = [f for f in os.listdir(directory_path) if f.endswith('.sdf')]

                if not sdf_files:
                    logger.warning(f"No se encontraron archivos .sdf en {directory_path}")
                    return

                logger.info(f"Iniciando procesamiento batch con {explainer_name} para {len(sdf_files)} archivos.")

                for sdf_file in sdf_files:
                    full_sdf_path = os.path.join(directory_path, sdf_file)
                    
                    # Try-except interno: Si una molécula falla, pasamos a la siguiente sin abortar el batch completo
                    try:
                        # Enrutamiento según el explicador seleccionado
                        if explainer_name == "GraphExplainer":
                            obtener_graph_explainer(
                                model_path, 
                                full_sdf_path, 
                                target_path, 
                                num_samples=1000, 
                                noise_level=0.01, 
                                device='cpu',
                                imagen=False
                            )
                        
                        elif explainer_name == "GNNExplainer":
                            obtener_GNN_Explainer(
                                model_path, 
                                full_sdf_path, 
                                target_path,
                                imagen=False
                            )
                        
                        elif explainer_name.startswith("Captum_"):
                            captum_method = explainer_name.split("_")[1]
                            obtener_Captum_Explainer(
                                model_path, 
                                full_sdf_path, 
                                target_path, 
                                imagen=False, 
                                captum_method=captum_method
                            )
                            
                        elif explainer_name == "DummyExplainer":
                            obtener_Dummy_Explainer(
                                model_path, 
                                full_sdf_path, 
                                target_path,
                                imagen=False
                            )

                        else:
                            logger.error(f"Explicador no reconocido en batch: {explainer_name}")
                            return

                        logger.info(f"Procesado correctamente: {sdf_file}")

                    except Exception as inner_e:
                        logger.error(f"Error procesando {sdf_file} con {explainer_name}: {str(inner_e)}", exc_info=True)
                        # El bucle continúa automáticamente con el siguiente archivo

                logger.info(f"Procesamiento batch con {explainer_name} finalizado.")

            except Exception as e:
                logger.error(f"Error crítico en el directorio de batch: {str(e)}", exc_info=True)

    def get_explanation_comparer(self):
        dialog = ExplainerComparerDialog(self.main_window)
        if dialog.exec():
            # 1. Recuperar los 5 valores
            model_path, sdf_path, graphexplanation_path, gnn_path_raw, mode, reg_fidelity_mas = dialog.get_inputs()
            
            # 2. Lógica para manejar GNNExplainer como opcional/None
            # Si el string está vacío (usuario no seleccionó nada), pasamos None.
            if not gnn_path_raw.strip():
                gnnexplanation_path = None
            else:
                gnnexplanation_path = gnn_path_raw

            try:
                # 3. Llamar a la función generadora pasando el MODO y el path (que puede ser None)
                plot_path = generar_comparativa_fidelity(
                    model_path, 
                    sdf_path, 
                    graphexplanation_path, 
                    gnnexplanation_path, 
                    mode=mode,  # <--- Pasamos el modo seleccionado
                    reg_fidelity_mas=reg_fidelity_mas
                )

                # 4. Mostrar resultado si se generó
                if plot_path:
                    self.image_dialog = ImageDialog(plot_path, self.main_window)
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
        dialog = BatchComparerDialog(self.main_window)

        UMBRAL_ERROR = 0.6767
        
        if dialog.exec():
            model_path, sdfs_dir, weights_root_dir, targets_path, mode, reg_fidelity_mas = dialog.get_inputs()
            try:
                obtener_aucs_directorio(
                    model_path,
                    sdfs_dir,
                    weights_root_dir,
                    targets_path,
                    mode,
                    reg_fidelity_mas
                )
            except Exception as e:
                logger.error(f"Error global en Batch Comparer: {str(e)}", exc_info=True)