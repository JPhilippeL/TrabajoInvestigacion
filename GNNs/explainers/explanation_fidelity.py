import csv
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import statistics  # <--- IMPORTANTE: Importar esto
from rdkit import Chem
from ui.utils.constants import (
    RESULTADOS_DIR,
    EMBEDDING_INDICES, 
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
    'GraphExplainer': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'},
    'GNNExplainer': {'color': '#ff7f0e', 'marker': 'x', 'linestyle': '--'},
    'Captum_IntegratedGradients': {'color': '#2ca02c', 'marker': 's', 'linestyle': '-.'},
    'Captum_InputXGradient': {'color': '#d62728', 'marker': '^', 'linestyle': ':'},
    'Captum_ShapleyValueSampling': {'color': '#9467bd', 'marker': 'v', 'linestyle': '-'},
    'DummyExplainer': {'color': '#7f7f7f', 'marker': 'D', 'linestyle': '--'},
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
    weights_paths_dict, 
    mode, reg_fidelity_mas, usar_porcentaje=False
):
    """
    Retorna un diccionario: 
    { 'Explainer1': {'k_vals': [...], 'fiab': [...], 'auc': float}, ... }
    """
    metrics_dict = {}

    for explainer_name, path in weights_paths_dict.items():
        try:
            tensor_weights = cargar_pesos_tensor(path, device)
        except Exception as e:
            logger.warning(f"Error cargando pesos de {explainer_name}: {e}")
            continue
        
        # Omitimos GNNExplainer en gamma (no soporta feature de aristas)
        if explainer_name == 'GNNExplainer' and mode == 'gamma':
            continue
            
        # GraphExplainer es el único que usa lógica one-hot en este contexto
        is_onehot = (explainer_name == 'GraphExplainer')

        try:
            k_vals, fiab = calcular_curvas_fidelity(
                model=model, importance=tensor_weights, device=device, 
                mol=mol, is_onehot_explainer=is_onehot, 
                mode=mode, usar_porcentaje=usar_porcentaje, 
                reg_fidelity_mas=reg_fidelity_mas
            )
            
            auc_val = _calcular_auc_simple(k_vals, fiab)
            
            metrics_dict[explainer_name] = {
                'k_vals': k_vals,
                'fiab': fiab,
                'auc': auc_val
            }
        except Exception as e:
            logger.error(f"Fallo al calcular curvas para {explainer_name}: {e}")

    return metrics_dict

# FUNCION PARA CALCULAR DATOS DE UNO
def calcular_curvas_fidelity(
    model, 
    importance, 
    device, 
    mol=None,        
    data=None,       
    mode="beta", 
    usar_porcentaje = False,
    is_onehot_explainer=False, 
    reg_fidelity_mas=True # True: Fidelidad (Ascendente), False: Infidelidad/Daño (Descendente)
):
    model.eval()

    # === 1. PREPARACIÓN DE DATOS ROBUSTA ===
    # El objetivo es tener dos copias INDEPENDIENTES:
    # - data_gpu: Para inferencia en el modelo (GPU)
    # - data_cpu: Para analizar ceros y filtrar features (CPU)

    if is_onehot_explainer:
        if mol is None:
            raise ValueError("Modo One-Hot requiere pasar el objeto 'mol' de RDKit.")
        
        # Generamos instancias frescas para evitar problemas de referencia
        data_gpu = mol_to_graph_data(mol).to(device)
        data_cpu = mol_to_graph_data(mol, 'one_hot') # Se queda en CPU
    else:
        if data is None:
            if mol is not None:
                # Generamos desde cero si tenemos mol
                data_gpu = mol_to_graph_data(mol).to(device)
                data_cpu = mol_to_graph_data(mol) # CPU por defecto
            else:
                raise ValueError("Modo Indices requiere 'data' o 'mol'.")
        else:
            # Si viene 'data', CLONAMOS para romper referencias antes de mover
            data_gpu = data.clone().to(device)
            data_cpu = data.clone().cpu()

    # Manejo explícito del batch si es None (para evitar errores en modelos sensibles)
    if data_gpu.batch is None:
        data_gpu.batch = torch.zeros(data_gpu.x.shape[0], dtype=torch.long, device=device)

    # === 2. DETERMINAR TOTAL ELEMENTOS ===
    # Usamos data_cpu para ver dimensiones y contenido
    if mode == 'alfa':
        total_elements = data_cpu.x.shape[1] 
    elif mode == 'beta':
        total_elements = data_gpu.x.shape[0] 
    elif mode == 'gamma':
        if data_gpu.edge_attr is None: return [], []
        total_elements = data_cpu.edge_attr.shape[1] 
    elif mode == 'delta':
        total_elements = data_gpu.edge_index.shape[1] 
    else:
        raise ValueError(f"Modo {mode} no reconocido.")

    # Procesar Importancia
    if torch.is_tensor(importance):
        imp = importance.detach().cpu().numpy().flatten()
    else:
        imp = np.array(importance).flatten()

    imp = np.abs(imp)

    # === En calcular_curvas_fidelity, después de imp = np.abs(imp) ===
    if mode in ['beta', 'delta'] and len(imp) != total_elements:
        raise ValueError(
            f"MISMATCH DE DIMENSIONES: La molécula tiene {total_elements} elementos ({mode}), "
            f"pero el tensor de pesos tiene {len(imp)}."
        )
    
    # === LÓGICA DE ORDENAMIENTO (Ascendente vs Descendente) ===
    if reg_fidelity_mas:
        # Fidelity+: Borramos lo MENOS importante primero.
        # Esperamos que la curva se mantenga alta (1.0) y caiga al final.
        sorted_indices = np.argsort(imp).copy()
    else:
        # Fidelity-: Borramos lo MÁS importante primero.
        # Esperamos que la curva (de impacto) suba rápido a 1.0.
        sorted_indices = np.argsort(imp)[::-1].copy()

    # =========================================================================
    # 3. LÓGICA DE FILTRADO (DIVERGENCIA)
    # =========================================================================
    
    indices_activos_reales = sorted_indices 

    # --- RAMA A: LOGICA ONE-HOT (GraphExplainer) ---
    if is_onehot_explainer:
        if mode == 'alfa':
            # Filtrar columnas todas a cero
            col_is_active = (data_cpu.x != 0).any(dim=0).cpu().numpy()
            indices_activos_reales = [idx for idx in sorted_indices if col_is_active[idx]]
            
        elif mode == 'gamma':
            if data_cpu.edge_attr is not None:
                col_is_active = (data_cpu.edge_attr != 0).any(dim=0).cpu().numpy()
                indices_activos_reales = [idx for idx in sorted_indices if col_is_active[idx]]
    
    # --- RAMA B: LOGICA INDICES (GNNExplainer) ---
    else:
        if mode == 'alfa':
            cat_cols = [EMBEDDING_INDICES["ATOM_SYMBOL"], EMBEDDING_INDICES["HYBRIDIZATION"]]
            filtered = []
            x_vals = data_cpu.x # Ya está en CPU
            
            for idx in sorted_indices:
                if idx in cat_cols:
                    filtered.append(idx) 
                elif (x_vals[:, idx] != 0).any(): 
                    filtered.append(idx)
            indices_activos_reales = filtered

    # Aplicar filtro
    sorted_indices = np.array(indices_activos_reales)
    limit = len(sorted_indices)
    
    if usar_porcentaje:
        limit = max(1, round(limit * PORCENTAJE_K))
    elif mode == 'beta':
        limit = limit - 1 

    # =========================================================================

    # 4. PREDICCIÓN ORIGINAL
    with torch.no_grad():
        # Usamos data_gpu explícitamente
        pred_original = model(data_gpu.x, data_gpu.edge_index, data_gpu.edge_attr, data_gpu.batch)
        val_orig = pred_original.item()

    fiab_list = []
    k_values = []

    # === 5. BUCLE PRINCIPAL ===
    for k in range(limit + 1):
        k_values.append(k)
        current_indices = sorted_indices[:k]

        if k == 0:
            data_minus = data_gpu
        else:
            # === DESPACHADOR DE PERTURBACIÓN ===
            
            # --- MODOS ESTRUCTURALES ---
            if mode == 'beta':
                data_minus = eliminar_nodos_y_conexiones(data_gpu, current_indices)
            elif mode == 'delta':
                data_minus = eliminar_aristas_selectivas(data_gpu, current_indices)
            
            # --- MODOS FEATURES ---
            elif is_onehot_explainer:
                if mode == 'alfa':
                    # data_cpu se usa para enmascarar en CPU, luego conversion
                    data_aux = ocultar_features_nodos_onehot(data_cpu, current_indices)
                    data_minus = onehot_to_indices(data_aux)
                elif mode == 'gamma':
                    data_aux = ocultar_features_aristas_onehot(data_cpu, current_indices)
                    data_minus = onehot_to_indices(data_aux)
            
            else: 
                if mode == 'alfa':
                    # Aquí data_gpu se modifica en GPU directamente (si la funcion soporta tensores)
                    # Ocultar features indices usa tensores, mantiene device.
                    data_minus = ocultar_features_nodos_indices(data_gpu, current_indices)
                elif mode == 'gamma':
                    data_minus = ocultar_features_aristas_indices(data_gpu, current_indices)

        # SEGURO FINAL: Asegurar que todo esté en GPU antes de entrar al modelo
        data_minus = data_minus.to(device)
        
        # Parche de seguridad para batch si se perdió en la perturbación
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

        # === CÁLCULO DE LA MÉTRICA ===
        if reg_fidelity_mas:
            # FIDELIDAD (Similitud): Empieza en 1.0, baja si el modelo sufre.
            fiab_list.append(np.exp(-abs(val_orig - val_minus)))
        else:
            # INFIDELIDAD (Daño): Empieza en 0.0, sube si el modelo sufre.
            diff_minus = abs(val_orig - val_minus)
            fidelity_score = np.exp(-diff_minus) 
            metric_to_plot = 1.0 - fidelity_score
            fiab_list.append(metric_to_plot)

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

    # --- LISTA MAESTRA PARA LIMPIEZA DE NOMBRES ---
    KNOWN_EXPLAINERS = [
        "GraphExplainer",
        "GNNExplainer",
        "Captum_IntegratedGradients",
        "Captum_InputXGradient",
        "Captum_ShapleyValueSampling",
        "DummyExplainer"
    ]

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
                    for known in KNOWN_EXPLAINERS:
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