from PySide6.QtWidgets import QMenu
from PySide6.QtGui import QAction
import torch
import os
from rdkit import Chem
import logging

from ui.dialogs.batch_model_test_dialog import BatchModelTestDialog
from ui.dialogs.image_dialog import ImageDialog
from ui.dialogs.explanation_dialog import ExplanationDialog
from ui.dialogs.explainer_comparer_dialog import ExplainerComparerDialog
from ui.dialogs.batch_explainer_comparer_dialog import BatchComparerDialog

from GNNs.model_tester import cargar_modelo
from GNNs.explainers.graph_explainer_onehot import obtener_graph_explainer
from GNNs.explainers.model_GNNExplainer import obtener_GNN_Explainer
from GNNs.explainers.explanation_fidelity import generar_comparativa_fidelity, save_auc_results_csv, calcular_aucs_fidelity_batch
from GNNs.data_processing import read_targets, mol_to_graph_data

logger = logging.getLogger(__name__)

class MenuExplainerGNN(QMenu):
    def __init__(self, parent_window):
        # "parent_window" es tu MainWindow, lo necesitamos para los diálogos
        super().__init__("Explainer GNN", parent_window)
        self.main_window = parent_window 

        self.init_actions()

    def init_actions(self):
        # Graph_explainer
        graph_explainer_action =QAction("Obtener GraphExplainer", self)
        graph_explainer_action.triggered.connect(self.get_explanation_GraphExplainer)
        self.addAction(graph_explainer_action)

        # GNN Explainer
        gnn_explainer_action = QAction("Obtener GNNExplainer", self)
        gnn_explainer_action.triggered.connect(self.get_explanation_GNNExplainer)
        self.addAction(gnn_explainer_action)

        # Batch Graph_explainer
        batch_graph_explainer_action =QAction("Obtener GraphExplainer de Directorio", self)
        batch_graph_explainer_action.triggered.connect(self.get_batch_explanation_GraphExplainer)
        self.addAction(batch_graph_explainer_action)

        # Batch GNN Explainer
        batch_gnn_explainer_action = QAction("Obtener GNNExplainer de Directorio", self)
        batch_gnn_explainer_action.triggered.connect(self.get_batch_explanation_GNNExplainer)
        self.addAction(batch_gnn_explainer_action)

        # Comparador
        explanation_comparer_action = QAction("Comparar Explicadores", self)
        explanation_comparer_action.triggered.connect(self.get_explanation_comparer)
        self.addAction(explanation_comparer_action)

        # batch_Comparador
        batch_explanation_comparer_action = QAction("Comparar Explicadores Batch", self)
        batch_explanation_comparer_action.triggered.connect(self.get_batch_explanation_comparer)
        self.addAction(batch_explanation_comparer_action)

    def get_explanation_GraphExplainer(self):
        dialog = ExplanationDialog(self.main_window)
        if dialog.exec():
            model_path, sdf_path, target_path = dialog.get_paths()
            try:
                # Obtener explicación GraphExplanation
                # feature_mask espera: [Atom, Degree, Arom, Hybrid, BondType, BondDist]
                plot_path = obtener_graph_explainer(model_path, sdf_path, target_path, num_samples=1000, noise_level=0.01, device='cpu')

                # mostrar el sdf por pantalla
                self.main_window.load_graph_from_file(sdf_path)

                # Mostrar la imagen en un diálogo
                self.image_dialog = ImageDialog(plot_path, self.main_window)
                self.image_dialog.show()

            except Exception as e:
                logger.error(f"Error en explicación GraphExplanation: {str(e)}", exc_info=True)
    
    def get_explanation_GNNExplainer(self):
        dialog = ExplanationDialog(self.main_window)
        if dialog.exec():
            model_path, sdf_path, target_path = dialog.get_paths()
            try:
                # Obtener explicación Explain er
                plot_path = obtener_GNN_Explainer(model_path, sdf_path, target_path)

                # mostrar el sdf por pantalla
                self.main_window.load_graph_from_file(sdf_path)

                # Mostrar la imagen en un diálogo
                self.image_dialog = ImageDialog(plot_path, self.main_window)
                self.image_dialog.show()

            except Exception as e:
                logger.error(f"Error en explicación GNNExplainer: {str(e)}", exc_info=True)

    def get_batch_explanation_GraphExplainer(self):
        """
        Procesa un directorio completo de archivos SDF usando GraphExplainer.
        Reutiliza BatchModelTestDialog para seleccionar Modelo, Directorio y Targets.
        """
        dialog = BatchModelTestDialog(self.main_window)
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
        dialog = BatchModelTestDialog(self.main_window)
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
        dialog = ExplainerComparerDialog(self.main_window)
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