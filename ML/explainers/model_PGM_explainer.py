from torch_geometric.contrib.explain import PGMExplainer
from torch_geometric.explain import Explainer
from ML.model_tester import cargar_modelo, predecir_molecula
from ML.data_processing import mol_to_graph_data
import torch
from ui.utils import RESULTADOS_DIR
from rdkit import Chem
import os
from ui.utils import RegressionToClassificationWrapper
from ML.explainers.model_GNNExplainer import visualizar_custom_gnn

def obtener_PGM_Explainer(checkpoint_path, sdf_path):
    # 1. CARGA DE MODELO Y DATOS
    reg_model, device, target_name = cargar_modelo(checkpoint_path)

    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None: raise ValueError(f"Error leyendo: {sdf_path}")
    
    data = mol_to_graph_data(mol, mode='embedding')
    data = data.to(device)
    batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)

    # 2. CALCULAR PREDICCIÓN REAL (Para el título de la imagen)
    with torch.no_grad():
        pred_tensor = reg_model(data.x, data.edge_index, data.edge_attr, batch)
        pred_val = pred_tensor.item()

    # 3. CREAR WRAPPER (Usando la versión nueva de utils.py)
    model = RegressionToClassificationWrapper(
        regression_model=reg_model,
        edge_attr_static=data.edge_attr,
        batch_static=batch
    ).to(device)
    model.eval()

    # 4. CONFIGURACIÓN PGM (CORREGIDA)
    explainer = Explainer(
        model=model,
        algorithm=PGMExplainer(num_samples=20, perturbation_mode="mean"), 
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type=None, # PGM no soporta aristas
        model_config=dict(
            # --- CAMBIO CLAVE: MULTICLASE ---
            # Esto hace que target = argmax(pred), devolviendo un solo índice (0 o 1).
            mode='multiclass_classification',
            task_level='graph',
            return_type='probs', 
        ),
    )

    # 5. EJECUTAR (SIN TARGET)
    # Dejamos que el explainer infiera el target automáticamente.
    # Al ser multiclass, argmax([0.1, 0.9]) -> 1 (Escalar). ¡Funciona!
    explanation = explainer(
        x=data.x, 
        edge_index=data.edge_index, 
        edge_attr=data.edge_attr, 
        batch=batch
        # target=... ¡ELIMINADO!
    )

    # 6. GUARDAR Y VISUALIZAR
    model_name = checkpoint_path.split('/')[-1].split('.')[0]
    mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else os.path.basename(sdf_path).split('.')[0]
    
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    save_dir = os.path.join(RESULTADOS_DIR, model_name)
    os.makedirs(save_dir, exist_ok=True)
    plotfilename = os.path.join(save_dir, f"{model_name}_{mol_name}_pgmexplainer.png")

    visualizar_custom_gnn(
        explanation=explanation, 
        sdf_path=sdf_path, 
        save_path=plotfilename,
        pred_val=pred_val,       
        target_name=target_name, 
        mol_name=mol_name,       
        algo_name="PGMExplainer" 
    )
    
    print(f"Explicación PGM guardada en: {plotfilename}")
    return plotfilename