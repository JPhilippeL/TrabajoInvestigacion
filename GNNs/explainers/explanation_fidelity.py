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

# FORMULA QUE HACE TODO PARA UN SOLO COMPONENTE
# CARGA, CALCULA, GENERA IMAGEN
def generar_comparativa_fidelity(
    model_path, 
    sdf_path, 
    graphexp_weights_path, 
    gnnexp_weights_path, # Puede ser None
    mode = "delta",
    reg_fidelity_mas = True,
):
    """
    Función orquestadora completa.
    """
    
    # --- 1. Procesamiento de Strings y Nombres ---
    model_folder_name = model_path.split('/')[-1].split('.')[0]
    mol_id = os.path.basename(sdf_path).split('.')[0]

    # --- 2. Carga del Modelo ---
    try:
        model, device, _ = cargar_modelo(model_path)
        model.eval()
    except Exception as e:
        print(f"Error cargando el modelo desde {model_path}: {e}")
        return None
    
    # --- 3. Carga de Molécula y Conversión a Grafo ---
    if not os.path.exists(sdf_path):
        print(f"Error: No se encontró el archivo SDF en {sdf_path}")
        return None

    mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
    
    if mol is None:
        print(f"Error: No se pudo leer la molécula del SDF.")
        return None

    mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else mol_id

    print(f"--- Comparativa ({mode}) para {mol_name} ---")

    # --- 4. Obtener metricas ---

    k_vals_graph, fiab_graphexp, k_vals_gnn, fiab_gnn, auc_graph, auc_gnn = calcular_metricas_comparativas(
        model, device, mol, 
        graphexp_weights_path, gnnexp_weights_path, 
        mode, reg_fidelity_mas
    )

    # --- 7. Generar Gráfico ---
    plot_path = guardar_plot_fidelity_comparativo(
        k_values_graph=k_vals_graph,      # <--- K de GraphExplainer
        fiab_graph=fiab_graphexp,
        k_values_gnn=k_vals_gnn,          # <--- K de GNNExplainer
        fiab_gnn=fiab_gnn,
        auc_graph=auc_graph,
        auc_gnn=auc_gnn,
        model_name=model_folder_name,
        mol_name=mol_name,
        mode=mode,
        reg_fidelity_mas=reg_fidelity_mas
    )
    
    return plot_path

# FUNCION PARA CARGAR PESOS Y CALCULAR LOS DATOS DE LOS DOS
def calcular_metricas_comparativas(
    model, 
    device, 
    mol, 
    graphexp_weights_path, 
    gnnexp_weights_path, 
    mode, 
    reg_fidelity_mas
):
    """
    Función NÚCLEO. Carga pesos y calcula las curvas y AUCs para ambos explainers.
    Retorna datos puros, sin generar gráficos.
    """
    
    # --- 1. Carga de Tensores ---
    try:
        tensor_graphexp = cargar_pesos_tensor(graphexp_weights_path, device)
    except Exception:
        # Si falla GraphExplainer, no podemos comparar nada. Retornamos vacío.
        return None, None, None, None, 0.0, None

    tensor_gnn = None
    if gnnexp_weights_path is not None:
        try:
            tensor_gnn = cargar_pesos_tensor(gnnexp_weights_path, device)
        except Exception:
            tensor_gnn = None

    # --- 2. GraphExplainer (One-Hot) ---
    k_vals_graph, fiab_graph = calcular_curvas_fidelity(
        model=model, 
        importance=tensor_graphexp, 
        device=device, 
        mol=mol, 
        is_onehot_explainer=True, 
        mode=mode, 
        reg_fidelity_mas=reg_fidelity_mas
    )
    
    auc_graph = _calcular_auc_simple(k_vals_graph, fiab_graph)

    # --- 3. GNNExplainer (Indices) ---
    k_vals_gnn = []
    fiab_gnn = None
    auc_gnn = None

    if tensor_gnn is not None:
        try:
            k_vals_gnn, fiab_gnn = calcular_curvas_fidelity(
                model=model, 
                importance=tensor_gnn, 
                device=device, 
                mol=mol, # Pasamos mol, la función genera data internamente
                is_onehot_explainer=False, 
                mode=mode, 
                reg_fidelity_mas=reg_fidelity_mas
            )
            auc_gnn = _calcular_auc_simple(k_vals_gnn, fiab_gnn)
        except ValueError:
            k_vals_gnn = []
            fiab_gnn = None
            auc_gnn = None

    return k_vals_graph, fiab_graph, k_vals_gnn, fiab_gnn, auc_graph, auc_gnn

# FUNCION PARA CALCULAR DATOS DE UNO
def calcular_curvas_fidelity(
    model, 
    importance, 
    device, 
    mol=None,        
    data=None,       
    mode="beta", 
    max_steps=None,
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

        elif mode == 'gamma':
            if data_gpu.edge_attr is not None:
                cat_cols = [EDGE_EMBEDDING_INDICES["BOND_TYPE"]]
                filtered = []
                e_vals = data_cpu.edge_attr
                
                for idx in sorted_indices:
                    if idx in cat_cols:
                        filtered.append(idx)
                    elif (e_vals[:, idx] != 0).any():
                        filtered.append(idx)
                indices_activos_reales = filtered

    # Aplicar filtro
    sorted_indices = np.array(indices_activos_reales)
    limit = len(sorted_indices)
    if max_steps is not None:
        limit = min(limit, max_steps)

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
        k_values_graph,      # K values específicos para tu explainer
        fiab_graph, 
        k_values_gnn,        # K values específicos para GNNExplainer
        fiab_gnn,
        auc_graph,           # <--- NUEVO ARGUMENTO: Ya viene calculado
        auc_gnn,             # <--- NUEVO ARGUMENTO: Ya viene calculado 
        model_name, 
        mol_name,
        mode,
        reg_fidelity_mas = True
    ):
    """
    Genera un gráfico comparativo. Si fiab_gnn es None,
    solo grafica GraphExplainer Explainer.
    """
    apply_paper_style()

    # 1. Sanitizar nombre
    safe_mol_name = "".join([c for c in mol_name if c.isalnum() or c in (' ', '_', '-')]).strip()
    
    # 2. Configurar Rutas
    if reg_fidelity_mas:
        filename = f"COMPARATIVA_FIDELITY_MAS_{safe_mol_name}_{mode}.png"
    else:
        filename = f"COMPARATIVA_FIDELITY_MENOS_{safe_mol_name}_{mode}.png"

    base_model_dir = os.path.join(RESULTADOS_DIR, model_name) # Asegúrate que RESULTADOS_DIR es accesible
    fidelity_dir = os.path.join(base_model_dir, "Fidelity_Comparison")
    os.makedirs(fidelity_dir, exist_ok=True)
    full_save_path = os.path.join(fidelity_dir, filename)

    # === CÁLCULO DE AUC NORMALIZADO ===
    
    # === PLOTTING ===
    plt.figure()
    
    # Plot GraphExplainer (Eje X largo)
    plt.plot(k_values_graph, fiab_graph, 
             marker='o', color='#1f77b4', linestyle='-', linewidth=2.5,
             label=f'GraphExplainer')
    
    if fiab_gnn is not None and len(fiab_gnn) > 0:
        has_gnn = True
    else:
        has_gnn = False
    # Plot GNNExplainer (Eje X corto)
    if has_gnn:
        plt.plot(k_values_gnn, fiab_gnn, 
                 marker='x', color='#ff7f0e', linestyle='--', linewidth=2, alpha=0.9,
                 label=f'GNNExplainer')

    # Decoración
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
    
    plt.legend(loc="best", frameon=True)
    plt.grid(True, linestyle='-', alpha=0.3)

    # Ajuste de ticks para que no se vea saturado
    # Usamos el K más largo para definir el eje X
    max_total_k = max(k_values_graph[-1], k_values_gnn[-1] if has_gnn else 0)
    if max_total_k < 15:
        plt.xticks(range(max_total_k + 1))

    save_paper_figure(full_save_path)
    print(f"Gráfico comparativo guardado en: {full_save_path}")
    
    return full_save_path

# OBTENER TODOS LOS DATOS DE UN DIRECTORIO ENTERO
def obtener_aucs_directorio(
        model_path,
        sdfs_dir,
        weights_root_dir,
        targets_path,
        mode,
        reg_fidelity_mas,
):
    UMBRAL_ERROR = 100

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
                # Si ya tienes 'mol' validado del paso de inferencia, úsalo:
                k_vals_g, fiab_g, k_vals_n, fiab_n, auc_graph, auc_gnn = calcular_metricas_comparativas(
                    model, device, mol, 
                    path_graph_explainer, path_gnn_explainer, 
                    mode, reg_fidelity_mas
                )
                
                if auc_graph is not None: # Si hubo resultado válido
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
            save_auc_results_csv(results, mode, model_name_clean, reg_fidelity_mas)
        else:
            logger.warning("No se generaron resultados para guardar.")

    except Exception as e:
        logger.error(f"Error global en Batch Comparer: {str(e)}", exc_info=True)

# GUARDARLO EL CSV
def save_auc_results_csv(results, mode, model_name, reg_fidelity_mas):
    """
    Guarda la lista de resultados en: RESULTADOS_DIR / model_name / auc_results / {metrica}_{mode}.csv
    Calcula el PROMEDIO de las columnas numéricas y lo añade al final.
    """
    try:
        # 1. Definir rutas
        # Asegúrate de tener RESULTADOS_DIR importado o definido globalmente
        output_folder = os.path.join(RESULTADOS_DIR, model_name, "auc_results")
        os.makedirs(output_folder, exist_ok=True)

        if reg_fidelity_mas:
            metrica = "RegFidelityMas"
        else:
            metrica = "RegFidelityMenos"

        csv_filename = f"{metrica}_{mode}.csv"
        csv_path = os.path.join(output_folder, csv_filename)

        # 2. RECOPILAR VALORES (Para estadísticas)
        # Extraemos solo los números, ignorando "N/A" o None
        vals_graph = [
            r["auc_graph"] for r in results 
            if isinstance(r.get("auc_graph"), (int, float))
        ]
        
        vals_gnn = [
            r["auc_gnn"] for r in results 
            if isinstance(r.get("auc_gnn"), (int, float))
        ]

        # 3. CALCULAR ESTADÍSTICAS
        # -- Promedio --
        avg_graph = statistics.mean(vals_graph) if vals_graph else 0.0
        avg_gnn = statistics.mean(vals_gnn) if vals_gnn else 0.0

        # -- Desviación Estándar (Sample Stdev) --
        # Requiere al menos 2 datos para calcularse
        std_graph = statistics.stdev(vals_graph) if len(vals_graph) > 1 else 0.0
        std_gnn = statistics.stdev(vals_gnn) if len(vals_gnn) > 1 else 0.0

        # 3. ESCRIBIR CSV
        fieldnames = ["name", "auc_graph", "auc_gnn"]
        
        with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # A) Escribir todas las filas de datos
            for data in results:
                writer.writerow(data)
            
            # B) Escribir fila de separación (opcional, visualmente útil) o ir directo al promedio
            # writer.writerow({}) 
            
            # C) Escribir la fila AVERAGE
            writer.writerow({
                "name": "AVERAGE",
                "auc_graph": avg_graph,
                "auc_gnn": avg_gnn
            })

            # D) Fila DESVIACIÓN ESTÁNDAR
            writer.writerow({
                "name": "STD_DEV",
                "auc_graph": std_graph,
                "auc_gnn": std_gnn
            })
                
        logging.getLogger(__name__).info(f"Resultados AUC guardados con promedio en: {csv_path}")

    except Exception as e:
        logging.getLogger(__name__).error(f"Error al guardar CSV: {str(e)}", exc_info=True)