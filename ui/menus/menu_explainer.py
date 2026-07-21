from PySide6.QtWidgets import QMenu
from PySide6.QtGui import QAction
import sys
import os
import logging

dir_actual = os.path.dirname(os.path.abspath(__file__))
dir_padre = os.path.abspath(os.path.join(dir_actual, "../.."))
sys.path.insert(0, dir_padre)

from ui.dialogs.batch_explanation_dialog import BatchExplanationDialog
from ui.dialogs.image_dialog import ImageDialog
from ui.dialogs.explanation_dialog import ExplanationDialog
from ui.dialogs.explainer_comparer_dialog import ExplainerComparerDialog
from ui.dialogs.batch_explainer_comparer_dialog import BatchComparerDialog

from GNNs.explainers.explanation_helper import guardar_pesos_batch
from GNNs.explainers.graph_explainer_onehot import obtener_graph_explainer
from GNNs.explainers.model_TorchExplainers import obtener_Dummy_Explainer, obtener_Captum_Explainer, obtener_GNN_Explainer, obtener_SubgraphX_Explainer
from GNNs.explainers.explanation_fidelity import generar_comparativa_fidelity, obtener_aucs_directorio
from GNNs.explainers.GraphSVX.obtener_GraphSVX_explainer import obtener_GraphSVX_Explainer
from ui.utils.constants import EXPLAINERS

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
                        batch_mode=False, captum_method=captum_method
                    )
                    
                elif explainer_name == "DummyExplainer":
                    plot_path = obtener_Dummy_Explainer(
                        model_path, sdf_path, target_path
                    )

                elif explainer_name == "SubgraphX":
                    plot_path = obtener_SubgraphX_Explainer(
                        model_path, sdf_path, target_path
                    )

                elif explainer_name == "GraphSVX":
                    plot_path = obtener_GraphSVX_Explainer(
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
        dialog = BatchExplanationDialog(self.main_window)
        if dialog.exec():
            model_path, directory_path, target_path, explainer_name = dialog.get_paths()
            
            # Diccionario para acumular todos los resultados del batch
            batch_results_dict = {}
            model_folder_name = os.path.basename(model_path).split('.')[0]
            
            try:
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
                            pesos_dict = obtener_graph_explainer(
                                model_path, 
                                full_sdf_path, 
                                target_path, 
                                num_samples=1000, 
                                noise_level=0.01, 
                                device='cpu',
                                batch_mode=True
                            )
                        
                        elif explainer_name == "GNNExplainer":
                            pesos_dict = obtener_GNN_Explainer(
                                model_path, 
                                full_sdf_path, 
                                target_path,
                                batch_mode=True
                            )
                        
                        elif explainer_name.startswith("Captum_"):
                            captum_method = explainer_name.split("_")[1]
                            pesos_dict = obtener_Captum_Explainer(
                                model_path, 
                                full_sdf_path, 
                                target_path, 
                                batch_mode=True, 
                                captum_method=captum_method
                            )
                            
                        elif explainer_name == "DummyExplainer":
                            pesos_dict = obtener_Dummy_Explainer(
                                model_path, 
                                full_sdf_path, 
                                target_path,
                                batch_mode=True
                            )
                        
                        else:
                            logger.error(f"Explicador no reconocido en batch: {explainer_name}")
                            return

                        # Si el explainer devolvió datos, los guardamos en el diccionario gigante
                        if pesos_dict:
                            mol_name = pesos_dict.pop('mol_name') # Sacamos el nombre para usarlo de llave
                            batch_results_dict[mol_name] = pesos_dict
                            logger.info(f"Procesado en memoria correctamente: {sdf_file}")

                    except Exception as inner_e:
                        logger.error(f"Error procesando {sdf_file} con {explainer_name}: {str(inner_e)}", exc_info=True)

                # --- GUARDADO FINAL ---
                if batch_results_dict:
                    ruta_guardada = guardar_pesos_batch(batch_results_dict, model_folder_name, explainer_name)
                    logger.info(f"Procesamiento batch finalizado. Todo guardado en: {ruta_guardada}")
                else:
                    logger.warning("No se generaron resultados para guardar en este batch.")

            except Exception as e:
                logger.error(f"Error crítico en el directorio de batch: {str(e)}", exc_info=True)

    def get_explanation_comparer(self):
        dialog = ExplainerComparerDialog(self.main_window)
        if dialog.exec():
            model_path, sdf_path, weights_root_dir, mode, reg_fidelity_mas = dialog.get_inputs()
            
            try:
                import os
                
                mol_name = os.path.splitext(os.path.basename(sdf_path))[0]
                weights_mode_dir = os.path.join(weights_root_dir, mode)
                
                if not os.path.exists(weights_mode_dir):
                    logger.error(f"El directorio de pesos no existe: {weights_mode_dir}")
                    return

                weights_paths_dict = {}
                suffix = f"_{mol_name}.pt"
                
                for w_file in os.listdir(weights_mode_dir):
                    if w_file.endswith(suffix):
                        # Buscar a qué explicador conocido pertenece este archivo
                        clean_explainer_name = None
                        for known in EXPLAINERS:
                            if known in w_file:
                                clean_explainer_name = known
                                break # Encontrado, dejamos de buscar
                        
                        # Si lo reconoció, lo añade con la llave LIMPIA
                        if clean_explainer_name:
                            weights_paths_dict[clean_explainer_name] = os.path.join(weights_mode_dir, w_file)
                        else:
                            logger.warning(f"Archivo ignorado (explicador desconocido): {w_file}")
                
                if not weights_paths_dict:
                    logger.warning(f"No se encontraron pesos válidos para la molécula '{mol_name}' en {weights_mode_dir}")
                    return

                logger.info(f"Comparando {len(weights_paths_dict)} explicadores: {list(weights_paths_dict.keys())}")

                # Llamada a la función generadora
                plot_path = generar_comparativa_fidelity(
                    model_path, 
                    sdf_path, 
                    weights_paths_dict, 
                    mode=mode,
                    reg_fidelity_mas=reg_fidelity_mas
                )

                if plot_path:
                    self.image_dialog = ImageDialog(plot_path, self.main_window)
                    self.image_dialog.show()

            except Exception as e:
                logger.error(f"Error en explicación Comparativa ({mode}): {str(e)}", exc_info=True)

    def get_batch_explanation_comparer(self):
        """
        Calcula métricas de fidelidad para todo un directorio de SDFs comparando
        MÚLTIPLES explicadores de forma dinámica, buscando los pesos automáticamente.
        """
        dialog = BatchComparerDialog(self.main_window)
        
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



# ====================================================================
# 2. FUNCIÓN ORQUESTADORA
# ====================================================================
def generar_todos_los_explainers_masivo(model_path, sdfs_dir, targets_path):
    """
    Toma un modelo, un directorio de SDFs y ejecuta los 6 explicadores
    para cada molécula, guardando solo los tensores (.pt).
    """
    # 1. Filtrar los archivos SDF
    sdf_files = [f for f in os.listdir(sdfs_dir) if f.endswith('.sdf')]
    if not sdf_files:
        logging.warning(f"No se encontraron archivos .sdf en {sdfs_dir}")
        return

    # 2. Lista de los explicadores a procesar

    logging.info(f"Iniciando generación de pesos para {len(sdf_files)} moléculas con {len(EXPLAINERS)} explicadores.")

    # 3. Doble bucle: Por cada explicador, procesamos todas las moléculas
    for explainer_name in EXPLAINERS:
        logging.info(f"==================================================")
        logging.info(f"--- PROCESANDO LOTE: {explainer_name} ---")
        logging.info(f"==================================================")
        
        for sdf_file in sdf_files:
            full_sdf_path = os.path.join(sdfs_dir, sdf_file)
            
            try:
                # Enrutador igual al de la interfaz, pero automático
                if explainer_name == "GraphExplainer":
                    obtener_graph_explainer(
                        model_path, 
                        full_sdf_path, 
                        targets_path, 
                        num_samples=1000, 
                        noise_level=0.01, 
                        device='cpu',
                        imagen=False  # <-- MUY IMPORTANTE
                    )
                    
                elif explainer_name == "GNNExplainer":
                    obtener_GNN_Explainer(
                        model_path, 
                        full_sdf_path, 
                        targets_path,
                        imagen=False
                    )
                    
                elif explainer_name.startswith("Captum_"):
                    captum_method = explainer_name.split("_")[1]
                    obtener_Captum_Explainer(
                        model_path, 
                        full_sdf_path, 
                        targets_path, 
                        imagen=False, 
                        captum_method=captum_method
                    )
                    
                elif explainer_name == "DummyExplainer":
                    obtener_Dummy_Explainer(
                        model_path, 
                        full_sdf_path, 
                        targets_path,
                        imagen=False
                    )

            except Exception as e:
                # Si falla una molécula, lo registra y pasa a la siguiente
                logging.error(f"Error procesando {sdf_file} con {explainer_name}: {str(e)}")

# ====================================================================
# 3. BLOQUE PRINCIPAL DE EJECUCIÓN
# ====================================================================
if __name__ == "__main__":
    # Configuración de logs
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # ------------------- RUTAS DE CONFIGURACIÓN ------------------------------
    MODELO_PT = "Modelos/ZGINE_paper.pt" 
    DATOS_MADRE = "/home/andromeda/Documentos/Philippe/Datos Philippe/AqSol_Test_2000" # Carpeta exacta donde están los .sdf a explicar
    TARGETS = "/home/andromeda/Documentos/Philippe/Datos Philippe/AqSol_Test_2000/Solubility.txt"
    
    print("🚀 Iniciando la generación masiva de pesos de explicabilidad...")
    
    generar_todos_los_explainers_masivo(
        model_path=MODELO_PT,
        sdfs_dir=DATOS_MADRE,
        targets_path=TARGETS
    )

    print("✅ ¡Generación de pesos finalizada! Ya puedes correr la comparativa AUC.")