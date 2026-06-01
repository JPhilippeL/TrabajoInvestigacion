import csv
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import statistics  # <--- IMPORTANTE: Importar esto
from rdkit import Chem
from ui.utils.constants import (
    RESULTADOS_DIR, EXPLAINERS,
    EMBEDDING_INDICES, CATEGORICAL_INDICES, 
    EDGE_EMBEDDING_INDICES,
)
from ui.utils.plot_style import apply_paper_style, save_paper_figure
from GNNs.data_processing import mol_to_graph_data, onehot_to_indices, read_targets
from GNNs.model_tester import cargar_modelo
from GNNs.explainers.k_elimination_functions import (
    ocultar_features_aristas_indices, ocultar_features_aristas_onehot, ocultar_features_nodos_indices, ocultar_features_nodos_onehot, 
    eliminar_aristas_selectivas, eliminar_nodos_y_conexiones
)
import logging
logger = logging.getLogger(__name__)

PORCENTAJE_K = 0.25

# === CONFIGURACIÓN DE ESTILOS PARA LA GRÁFICA ===
# Asigna colores y marcadores únicos para identificar fácilmente a los 6 explicadores
PLOT_STYLES = {
    'GraphExplainer': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'},        # Azul
    'GNNExplainer': {'color': '#ff7f0e', 'marker': 'x', 'linestyle': '--'},       # Naranja
    'Captum_IntegratedGradients': {'color': '#2ca02c', 'marker': 's', 'linestyle': '-.'}, # Verde
    'Captum_InputXGradient': {'color': '#d62728', 'marker': '^', 'linestyle': ':'},       # Rojo
    'Captum_ShapleyValueSampling': {'color': '#9467bd', 'marker': 'v', 'linestyle': '-'}, # Morado
    'Captum_Saliency': {'color': '#17becf', 'marker': 'p', 'linestyle': '--'},            # Cian (Pentágono)
    'Captum_Deconvolution': {'color': '#8c564b', 'marker': 'h', 'linestyle': '-.'},       # Marrón (Hexágono)
    'Captum_GuidedBackprop': {'color': '#bcbd22', 'marker': '*', 'linestyle': ':'},       # Verde Oliva (Estrella)
    'DummyExplainer': {'color': '#7f7f7f', 'marker': 'D', 'linestyle': '--'},             # Gris
}

# FORMULA QUE HACE TODO PARA UN SOLO COMPONENTE
# CARGA, CALCULA, GENERA IMAGEN
def generar_comparativa_fidelity(
    model_path, 
    sdf_path, 
    weights_paths_dict, # <-- AHORA RECIBE UN DICCIONARIO: {'GraphExplainer': path, 'GNNExplainer': path, ...}
    mode = "delta",
    reg_fidelity_mas = True,
):
    model_folder_name = model_path.split('/')[-1].split('.')[0]
    mol_id = os.path.basename(sdf_path).split('.')[0]

    try:
        model, device, _ = cargar_modelo(model_path)
        model.eval()
    except Exception as e:
        logger.error(f"Error cargando el modelo desde {model_path}: {e}")
        return None
    
    if not os.path.exists(sdf_path):
        logger.error(f"Error: No se encontró el archivo SDF en {sdf_path}")
        return None

    mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
    if mol is None:
        logger.error(f"Error: No se pudo leer la molécula del SDF.")
        return None

    mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else mol_id
    logger.info(f"--- Comparativa ({mode}) para {mol_name} ---")

    # Calcula las métricas dinámicamente
    metrics_dict = calcular_metricas_comparativas(
        model, device, mol, 
        weights_paths_dict, 
        mode, reg_fidelity_mas
    )

    if not metrics_dict:
        logger.warning(f"No se pudieron calcular métricas para {mol_name}")
        return None

    # Genera el gráfico iterando sobre los resultados
    plot_path = guardar_plot_fidelity_comparativo(
        metrics_dict=metrics_dict,
        model_name=model_folder_name,
        mol_name=mol_name,
        mode=mode,
        reg_fidelity_mas=reg_fidelity_mas
    )
    
    return plot_path

# FUNCION PARA CARGAR PESOS Y CALCULAR LOS DATOS DE LOS DOS
def calcular_metricas_comparativas(
    model, device, mol, 
    weights_dict, # <-- Cambiado: ahora recibe el diccionario con los datos, no las rutas
    mode, reg_fidelity_mas, data=None, usar_porcentaje=False
):
    """
    Retorna un diccionario: 
    { 'Explainer1': {'k_vals': [...], 'fiab': [...], 'auc': float}, ... }
    """
    metrics_dict = {}

    # Iteramos directamente sobre el nombre y los pesos en memoria
    for explainer_name, tensor_weights in weights_dict.items():
        
        # Omitimos GNNExplainer en gamma (no soporta feature de aristas)
        if explainer_name == 'GNNExplainer' and mode == 'gamma':
            continue
            
        # GraphExplainer es el único que usa lógica one-hot en este contexto
        is_onehot = (explainer_name == 'GraphExplainer')

        try:
            # 1. Preparar datos: pasamos tensor_weights DIRECTAMENTE a importance
            graph_model, graph_masking, sorted_indices, limit = preparar_datos_fidelity(
                importance=tensor_weights, 
                device=device, 
                mol=mol, 
                data=data, 
                mode=mode, 
                usar_porcentaje=usar_porcentaje, 
                is_onehot_explainer=is_onehot, 
                reg_fidelity_mas=reg_fidelity_mas
            )

            # Condición de salida temprana (ej. modo gamma sin atributos de arista)
            if graph_model is None:
                # SOLUCIÓN DE BUG: Hacemos continue en lugar de retornar [], [] 
                # para no romper el tipo de dato que espera recibir la función principal
                continue

            # 2. Ejecutar el modelo predictivo y generar la curva
            k_vals, fiab = ejecutar_bucle_perturbacion(
                model=model, 
                device=device, 
                graph_model=graph_model, 
                graph_masking=graph_masking, 
                sorted_indices=sorted_indices, 
                limit=limit, 
                mode=mode, 
                is_onehot_explainer=is_onehot, 
                reg_fidelity_mas=reg_fidelity_mas
            )
            
            auc_val = _calcular_auc_simple(k_vals, fiab)
            
            metrics_dict[explainer_name] = {
                'k_vals': k_vals,
                'fiab': fiab,
                'auc': auc_val
            }
        except Exception as e:
            # Si un explicador falla (ej. error de dimensiones), no aborta el resto
            logger.error(f"Fallo al calcular curvas para {explainer_name}: {e}")

    return metrics_dict

# Se encarga puramente del procesamiento de tensores, máscaras y ordenamiento de índices.
def preparar_datos_fidelity(
    importance, 
    device, 
    mol=None,        
    data=None,       
    mode="beta", 
    usar_porcentaje=False,
    is_onehot_explainer=False, 
    reg_fidelity_mas=True
):
    """
    Prepara los grafos y calcula los índices a perturbar antes de entrar al bucle.
    Retorna: graph_model, graph_masking, sorted_indices, limit
    """
    # === 1. PREPARACIÓN DE DATOS ===
    if is_onehot_explainer and data is None:
        if mol is None:
            raise ValueError("Modo One-Hot requiere pasar el objeto 'mol' de RDKit.")
        
        graph_model = mol_to_graph_data(mol).to(device)
        graph_masking = mol_to_graph_data(mol, 'one_hot') 
    else:
        if data is not None:
            graph_model = data.clone().to(device)
        elif mol is not None:
            graph_model = mol_to_graph_data(mol).to(device)
        else:
            raise ValueError("Modo Indices requiere 'data' o 'mol'.")
        
        graph_masking = graph_model 

    if graph_model.batch is None:
        graph_model.batch = torch.zeros(graph_model.x.shape[0], dtype=torch.long, device=device)

    # === 2. DETERMINAR TOTAL DE ELEMENTOS ===
    if mode == 'alfa':
        total_elements = graph_masking.x.shape[1] 
    elif mode == 'beta':
        total_elements = graph_model.x.shape[0] 
    elif mode == 'gamma':
        if graph_model.edge_attr is None: 
            return None, None, [], 0 # Señal para abortar anticipadamente
        total_elements = graph_masking.edge_attr.shape[1] 
    elif mode == 'delta':
        total_elements = graph_model.edge_index.shape[1] 
    else:
        raise ValueError(f"Modo {mode} no reconocido.")

    # Procesar Importancia
    imp = importance.detach().cpu().numpy().flatten() if torch.is_tensor(importance) else np.array(importance).flatten()
    imp = np.abs(imp)

    if mode in ['beta', 'delta'] and len(imp) != total_elements:
        raise ValueError(f"MISMATCH: Molécula tiene {total_elements} elementos ({mode}), pero hay {len(imp)} pesos.")
    
    sorted_indices = np.argsort(imp).copy() if reg_fidelity_mas else np.argsort(imp)[::-1].copy()

    # === 3. LÓGICA DE FILTRADO (DIVERGENCIA) ===
    indices_activos_reales = sorted_indices 

    if is_onehot_explainer:
        if mode == 'alfa':
            col_is_active = (graph_masking.x != 0).any(dim=0).cpu().numpy()
            indices_activos_reales = [idx for idx in sorted_indices if col_is_active[idx]]
        elif mode == 'gamma' and graph_masking.edge_attr is not None:
            col_is_active = (graph_masking.edge_attr != 0).any(dim=0).cpu().numpy()
            indices_activos_reales = [idx for idx in sorted_indices if col_is_active[idx]]
    else:
        if mode == 'alfa':
            cat_cols = CATEGORICAL_INDICES
            filtered = []
            x_vals = graph_masking.x.cpu() 
            
            for idx in sorted_indices:
                if idx in cat_cols or (x_vals[:, idx] != 0).any(): 
                    filtered.append(idx)
            indices_activos_reales = filtered

    sorted_indices = np.array(indices_activos_reales)
    limit = max(1, round(len(sorted_indices) * PORCENTAJE_K)) if usar_porcentaje else (len(sorted_indices) - 1 if mode == 'beta' else len(sorted_indices))

    return graph_model, graph_masking, sorted_indices, limit

# Toma los datos ya procesados, realiza la inferencia original y ejecuta el bucle de perturbaciones.
def ejecutar_bucle_perturbacion(
    model, 
    device, 
    graph_model, 
    graph_masking, 
    sorted_indices, 
    limit, 
    mode="beta", 
    is_onehot_explainer=False, 
    reg_fidelity_mas=True
):
    """
    Ejecuta el bucle iterativo ocultando características y evaluando la respuesta del modelo.
    """
    model.eval()
    
    # === 4. PREDICCIÓN ORIGINAL ===
    with torch.no_grad():
        pred_original = model(graph_model.x, graph_model.edge_index, graph_model.edge_attr, graph_model.batch)
        val_orig = pred_original.item()

    fiab_list = []
    k_values = []

    # === 5. BUCLE PRINCIPAL DE PERTURBACIÓN ===
    for k in range(limit + 1):
        k_values.append(k)
        current_indices = sorted_indices[:k]

        if k == 0:
            data_minus = graph_model # Baseline intacto
        else:
            # === DESPACHADOR DE PERTURBACIÓN ===
            if mode == 'beta':
                data_minus = eliminar_nodos_y_conexiones(graph_model, current_indices)
            elif mode == 'delta':
                data_minus = eliminar_aristas_selectivas(graph_model, current_indices)
            elif is_onehot_explainer:
                if mode == 'alfa':
                    data_aux = ocultar_features_nodos_onehot(graph_masking, current_indices)
                    data_minus = onehot_to_indices(data_aux)
                elif mode == 'gamma':
                    data_aux = ocultar_features_aristas_onehot(graph_masking, current_indices)
                    data_minus = onehot_to_indices(data_aux)
            else: 
                if mode == 'alfa':
                    data_minus = ocultar_features_nodos_indices(graph_model, current_indices)
                elif mode == 'gamma':
                    data_minus = ocultar_features_aristas_indices(graph_model, current_indices)

        # Unificación: Mover siempre el resultado al dispositivo del modelo
        data_minus = data_minus.to(device)
        
        if data_minus.batch is None:
             data_minus.batch = torch.zeros(data_minus.x.shape[0], dtype=torch.long, device=device)

        # Inferencia
        with torch.no_grad():
            if data_minus.x.shape[0] == 0:
                val_minus = 0.0
            elif mode == 'delta' and data_minus.edge_index.shape[1] == 0:
                 pred_minus = model(data_minus.x, data_minus.edge_index, data_minus.edge_attr, data_minus.batch)
                 val_minus = pred_minus.item()
            else:
                pred_minus = model(data_minus.x, data_minus.edge_index, data_minus.edge_attr, data_minus.batch)
                val_minus = pred_minus.item()

        # Cálculo de Métrica
        if reg_fidelity_mas:
            fiab_list.append(np.exp(-abs(val_orig - val_minus))) # Fidelidad
        else:
            fiab_list.append(1.0 - np.exp(-abs(val_orig - val_minus))) # Infidelidad

    return k_values, fiab_list

# Función auxiliar para calcular aucs
def _calcular_auc_simple(k_vals, fiab_list):
    """Helper interno para cálculo matemático de AUC normalizado."""
    if not k_vals or not fiab_list:
        return 0.0
    max_k = k_vals[-1]
    if max_k == 0: return 0.0
    
    try:
        # Numpy >= 2.0
        area = np.trapezoid(fiab_list, k_vals)
    except AttributeError:
        # Numpy < 2.0
        area = np.trapz(fiab_list, k_vals)
        
    return area / max_k

def cargar_pesos_tensor(path, device='cpu'):
    """
    Carga un tensor guardado en .pt y asegura que esté en el formato correcto.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de pesos: {path}")
    
    weights = torch.load(path, map_location=device)
            
    return weights

# FUNCION PARA HACER GRAFICA
def guardar_plot_fidelity_comparativo(
        metrics_dict, 
        model_name, 
        mol_name,
        mode,
        reg_fidelity_mas = True
    ):
    
    apply_paper_style()

    safe_mol_name = "".join([c for c in mol_name if c.isalnum() or c in (' ', '_', '-')]).strip()
    
    if reg_fidelity_mas:
        filename = f"COMPARATIVA_FIDELITY_MAS_{safe_mol_name}_{mode}.png"
    else:
        filename = f"COMPARATIVA_FIDELITY_MENOS_{safe_mol_name}_{mode}.png"

    base_model_dir = os.path.join(RESULTADOS_DIR, model_name)
    fidelity_dir = os.path.join(base_model_dir, "Fidelity_Comparison")
    os.makedirs(fidelity_dir, exist_ok=True)
    full_save_path = os.path.join(fidelity_dir, filename)

    plt.figure()
    
    max_total_k = 0

    # Iterar sobre todos los explicadores en metrics_dict
    for explainer_name, data in metrics_dict.items():
        k_vals = data['k_vals']
        fiab = data['fiab']
        
        if not k_vals: continue
            
        # Actualizamos el K máximo para ajustar el eje X después
        max_total_k = max(max_total_k, k_vals[-1])
        
        # Extraer estilos si existen, si no usar un fallback por defecto
        style = PLOT_STYLES.get(explainer_name, {'color': np.random.rand(3,), 'marker': 'o', 'linestyle': '-'})

        plt.plot(k_vals, fiab, 
                 marker=style['marker'], color=style['color'], 
                 linestyle=style['linestyle'], linewidth=2, alpha=0.9,
                 label=f'{explainer_name}')

    subscript_map = {'alfa': 'n_a', 'beta': 'n', 'gamma': 'e_a', 'delta': 'e'}
    sub = subscript_map.get(mode, 'u')
    if reg_fidelity_mas:
        ylabel_text = rf"$\mathrm{{RegFidelity}}_{{({sub})}}^{{+k}}$"
    else:
        ylabel_text = rf"$\mathrm{{RegFidelity}}_{{({sub})}}^{{-k}}$"

    plt.ylabel(ylabel_text)
    plt.xlabel("K")
    
    plt.ylim(-0.05, 1.05) 
    plt.axhline(1, color='gray', linestyle=':', alpha=0.5)
    
    # Legend adaptado
    plt.legend(loc="best", frameon=True, fontsize='small')
    plt.grid(True, linestyle='-', alpha=0.3)

    if max_total_k < 15:
        plt.xticks(range(max_total_k + 1))

    save_paper_figure(full_save_path)
    logger.info(f"Gráfico comparativo guardado en: {full_save_path}")
    
    return full_save_path

# OBTENER TODOS LOS DATOS DE UN DIRECTORIO ENTERO
def obtener_aucs_directorio(
        model_path, sdfs_dir, weights_root_dir, 
        targets_path, mode, reg_fidelity_mas
):
    weights_mode_dir = os.path.join(weights_root_dir, mode)
    
    if not os.path.exists(weights_mode_dir):
        logger.error(f"No existe el directorio de pesos para el modo {mode}: {weights_mode_dir}")
        return

    results = [] 
    
    try:
        sdf_files = [f for f in os.listdir(sdfs_dir) if f.endswith('.sdf')]
        if not sdf_files: 
            logger.warning(f"No se encontraron archivos .sdf en {sdfs_dir}")
            return

        all_weight_files = os.listdir(weights_mode_dir)
        model, device, _ = cargar_modelo(model_path)
        targets_dict = read_targets(targets_path)

        for sdf_file in sdf_files:
            mol_name = os.path.splitext(sdf_file)[0] 
            full_sdf_path = os.path.join(sdfs_dir, sdf_file)

            # Nos aseguramos de que la molécula esté en nuestro dataset de targets
            if mol_name not in targets_dict: 
                continue

            weights_paths_dict = {}
            suffix = f"_{mol_name}.pt"
            
            # Búsqueda y limpieza dinámica de nombres
            for w in all_weight_files:
                if w.endswith(suffix):
                    clean_explainer_name = None
                    for known in EXPLAINERS:
                        if known in w:
                            clean_explainer_name = known
                            break 
                    
                    if clean_explainer_name:
                        weights_paths_dict[clean_explainer_name] = os.path.join(weights_mode_dir, w)

            # Si no hay ningún peso para esta molécula, la saltamos
            if not weights_paths_dict: 
                continue

            try:
                # Cargamos la molécula solo cuando sabemos que vamos a procesarla
                mol = Chem.SDMolSupplier(full_sdf_path, removeHs=False)[0] 
                if mol is None: 
                    continue
                
                # Calculamos todas las métricas para los explicadores encontrados
                metrics_dict = calcular_metricas_comparativas(
                    model, device, mol, 
                    weights_paths_dict, 
                    mode, reg_fidelity_mas, usar_porcentaje=True
                )
                
                if metrics_dict:
                    row_data = {"name": mol_name}
                    for exp_name, val in metrics_dict.items():
                        row_data[exp_name] = val['auc']
                        
                    results.append(row_data)
                    logger.info(f"Procesado {mol_name} | Explainers: {list(metrics_dict.keys())}")

            except Exception as e_inner:
                logger.error(f"Error procesando {mol_name}: {e_inner}")

        if results:
            model_name_clean = os.path.splitext(os.path.basename(model_path))[0]
            save_auc_results_csv(results, mode, model_name_clean, reg_fidelity_mas)
        else:
            logger.warning("No se generaron resultados para el batch.")

    except Exception as e:
        logger.error(f"Error global en Batch Comparer: {str(e)}", exc_info=True)

# Asumo que tienes definido EXPLAINERS en algún lugar de tu script
# EXPLAINERS = ["GraphExplainer", "GNNExplainer", "Captum_IG", ...]

def obtener_aucs_pt(
        model_path, data_list, weights_path, 
        mode, reg_fidelity_mas
):
    if not os.path.exists(weights_path):
        logger.error(f"No existe la ruta de pesos: {weights_path}")
        return

    results = [] 
    
    try:
        if not data_list: 
            logger.warning("La lista de datos (data_list) recibida está vacía.")
            return

        model, device, _ = cargar_modelo(model_path)
        
        # =====================================================================
        # 1. PRE-CARGA DE DICCIONARIOS BATCH EN MEMORIA
        # =====================================================================
        loaded_explainers_data = {}
        
        # Adaptación: Verificamos si es un archivo directo o un directorio
        if os.path.isfile(weights_path):
            all_weight_files = [weights_path] # Convertimos a lista de un solo elemento
            logger.info(f"Cargando archivo único de pesos: {weights_path}")
        else:
            all_weight_files = [os.path.join(weights_path, f) for f in os.listdir(weights_path) if f.endswith('.pt')]
            logger.info(f"Cargando archivos batch de pesos desde directorio: {weights_path}")
        
        for filepath in all_weight_files:
            filename = os.path.basename(filepath)
            clean_explainer_name = None
            
            # Identificamos el explicador buscando en el nombre del archivo
            for known in EXPLAINERS:
                if known in filename:
                    clean_explainer_name = known
                    break 
            
            # Fallback de seguridad: si no lo encuentra en EXPLAINERS, usamos el nombre del archivo
            if not clean_explainer_name:
                clean_explainer_name = filename.replace('.pt', '')
                
            try:
                # Cargamos el diccionario gigante en memoria
                loaded_explainers_data[clean_explainer_name] = torch.load(filepath)
                logger.info(f"Cargado exitosamente: {clean_explainer_name} ({len(loaded_explainers_data[clean_explainer_name])} moléculas)")
            except Exception as e:
                logger.error(f"Error cargando el archivo batch {filename}: {e}")

        if not loaded_explainers_data:
            logger.warning("No se pudo cargar ningún archivo batch de explicadores válido.")
            return

        # =====================================================================
        # 2. BUCLE SOBRE LOS GRAFOS
        # =====================================================================
        for idx, graph_data in enumerate(data_list):
            
            mol_name = getattr(graph_data, 'name', f'mol_idx_{idx}')

            # Extraemos los pesos específicos para ESTA molécula y ESTE modo
            molecule_weights_dict = {}
            
            for exp_name, exp_dict in loaded_explainers_data.items():
                if mol_name in exp_dict:
                    if mode in exp_dict[mol_name]:
                        molecule_weights_dict[exp_name] = exp_dict[mol_name][mode]

            # Si no hay ningún peso para esta molécula en ningún explicador, la saltamos
            if not molecule_weights_dict: 
                continue

            try:
                # 3. Llamada a métricas comparativas
                metrics_dict = calcular_metricas_comparativas(
                    model=model, 
                    device=device, 
                    mol=None,
                    data=graph_data, 
                    weights_dict=molecule_weights_dict, 
                    mode=mode, 
                    reg_fidelity_mas=reg_fidelity_mas, 
                    usar_porcentaje=False
                )
                
                if metrics_dict:
                    row_data = {"name": mol_name}
                    for exp_name, val in metrics_dict.items():
                        row_data[exp_name] = val['auc']
                        
                    results.append(row_data)
                    logger.info(f"Procesado {mol_name} | Explainers: {list(metrics_dict.keys())}")

            except Exception as e_inner:
                logger.error(f"Error procesando {mol_name}: {e_inner}", exc_info=True)

        # =====================================================================
        # --- GUARDADO FINAL ---
        # =====================================================================
        if results:
            model_name_clean = os.path.splitext(os.path.basename(model_path))[0]
            save_auc_results_csv(results, mode, model_name_clean, reg_fidelity_mas)
            logger.info(f"Resultados AUC guardados exitosamente para {len(results)} moléculas.")
        else:
            logger.warning("No se generaron resultados para el batch.")

    except Exception as e:
        logger.error(f"Error global en Batch Comparer desde .pt: {str(e)}", exc_info=True)

# GUARDARLO EL CSV
def save_auc_results_csv(results, mode, model_name, reg_fidelity_mas):
    try:
        output_folder = os.path.join(RESULTADOS_DIR, model_name, "auc_results")
        os.makedirs(output_folder, exist_ok=True)

        metrica = "RegFidelityMas" if reg_fidelity_mas else "RegFidelityMenos"
        csv_filename = f"{metrica}_{mode}_{PORCENTAJE_K}.csv"
        csv_path = os.path.join(output_folder, csv_filename)

        # 1. Averiguar todas las columnas (explicadores) existentes en los resultados
        # Unimos las keys de todos los diccionarios por si a una molécula le faltó un explainer
        all_explainers = set()
        for row in results:
            all_explainers.update([k for k in row.keys() if k != 'name'])
            
        all_explainers = sorted(list(all_explainers)) # Ordenar alfabéticamente
        fieldnames = ["name"] + all_explainers

        # 2. Calcular estadísticas dinámicas para cada columna
        stats_avg = {"name": "AVERAGE"}
        stats_std = {"name": "STD_DEV"}

        for exp in all_explainers:
            # Filtrar valores válidos numéricos de la columna
            vals = [r.get(exp) for r in results if isinstance(r.get(exp), (int, float))]
            
            stats_avg[exp] = statistics.mean(vals) if vals else 0.0
            stats_std[exp] = statistics.stdev(vals) if len(vals) > 1 else 0.0

        # 3. Escribir CSV
        with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, restval="N/A")
            writer.writeheader()
            
            for data in results:
                writer.writerow(data)
            
            writer.writerow(stats_avg)
            writer.writerow(stats_std)
                
        logger.info(f"Resultados AUC guardados con promedios dinámicos en: {csv_path}")

    except Exception as e:
        logger.error(f"Error al guardar CSV: {str(e)}", exc_info=True)