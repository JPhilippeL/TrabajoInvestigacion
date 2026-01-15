# sdf_converter.py
# Leer/Guardar archivos SDF y convertirlos a un grafo de NetworkX

import logging
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import networkx as nx
import os
import pandas as pd
logger = logging.getLogger(__name__)


SCALE = 50
MINNODES = 2

def parse_sdf(file_path):
    suppl = Chem.SDMolSupplier(file_path, removeHs=False)
    mol = next((m for m in suppl if m is not None), None)

    if mol is None:
        raise ValueError("No se pudo leer una molécula válida desde el archivo SDF.")
    
    # Guardamos los datos de las posiciones 3D originales
    conf = mol.GetConformer()

    # Copia para calcular 2D
    mol2d = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol2d)
    conf2d = mol2d.GetConformer()

    graph = nx.Graph()

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos3D = conf.GetAtomPosition(idx)  # x,y,z originales
        pos2D = conf2d.GetAtomPosition(idx)  # x,y 2D generados para un layout mejor
        graph.add_node(str(idx), 
               element=atom.GetSymbol(), 
                # 3D sin escalar para poder reescribir el SDF
                coords3d = (float(pos3D.x), float(pos3D.y), float(pos3D.z)),
                # 2D escalado para visualización
                pos = (float(pos2D.x) * SCALE, float(pos2D.y) * SCALE)
        )
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        graph.add_edge(str(start), str(end), bond_type=str(bond_type))

    return graph

def graph_to_mol(graph):
    mol = Chem.RWMol()

    # Mapear ID de nodos a nuevos índices de átomos
    node_to_idx = {}

    for node_id in sorted(graph.nodes, key=int):
        element = graph.nodes[node_id]["element"]
        atom = Chem.Atom(element)
        idx = mol.AddAtom(atom)
        node_to_idx[node_id] = idx

    for u, v, data in graph.edges(data=True):
        bond_type_str = data.get("bond_type", "SINGLE").upper()
        bond_type = getattr(Chem.rdchem.BondType, bond_type_str, Chem.rdchem.BondType.SINGLE)
        mol.AddBond(node_to_idx[u], node_to_idx[v], bond_type)

    mol = mol.GetMol()
    Chem.SanitizeMol(mol)

    node_ids = [nid for nid in sorted(graph.nodes, key=int)]
    conf = Chem.Conformer(mol.GetNumAtoms())

    # Asignar posiciones 3D si existen, sino z = 0
    any_3d = False
    for i, nid in enumerate(node_ids):
        if "coords3d" in graph.nodes[nid]:
            x, y, *z = graph.nodes[nid]["coords3d"]
            z_val = float(z[0]) if z else 0.0
            conf.SetAtomPosition(i, Point3D(float(x), float(y), z_val))
            if len(z) == 1:  # tenía z explícita
                any_3d = True
        elif "pos" in graph.nodes[nid]:
            x, y = graph.nodes[nid]["pos"]
            conf.SetAtomPosition(i, Point3D(float(x) / SCALE, float(y) / SCALE, 0.0))
        else:
            # Poner en origen
            conf.SetAtomPosition(i, Point3D(0.0, 0.0, 0.0))

    try:
        conf.Set3D(any_3d)
    except AttributeError:
        pass

    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)

    return mol

def save_graph_as_sdf(graph, file_path):
    mol = graph_to_mol(graph)
    try:
        Chem.SanitizeMol(mol)
        writer = Chem.SDWriter(file_path)
        writer.write(mol)
        writer.close()
    except Exception as e:
        raise RuntimeError(f"Error al guardar la molécula: {str(e)}")

def split_sdf(sdf_file, output_dir):
    """
    Divide un archivo SDF con múltiples moléculas en SDF individuales,
    guardándolas en el directorio de salida.
    """
    os.makedirs(output_dir, exist_ok=True)

    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
    for idx, mol in enumerate(suppl):
        if mol is None:
            logger.warning(f"MOL {idx}: INVÁLIDA, se saltará")
            continue

         # Obtener nombre de la molécula
        if mol.HasProp("_Name"):
            mol_name = mol.GetProp("_Name")
        else:
            logger.warning(f"MOL {idx}: NO TIENE NOMBRE")
            mol_name = f"mol_{idx+1}"  # Todavía necesitamos un nombre para guardar el archivo

        # Guardar SDF individual
        out_sdf_path = os.path.join(output_dir, f"{mol_name}.sdf")
        writer = Chem.SDWriter(out_sdf_path)
        writer.write(mol)
        writer.close()

    logger.info(f"Se han generado {len(suppl)} moléculas individuales en '{output_dir}'.")

def smiles_csv_to_sdf_dir(csv_path, output_dir, minimoNodos = MINNODES):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    if minimoNodos < MINNODES: minimoNodos = MINNODES

    # (A) --- INICIALIZAR LISTAS PARA ESTADÍSTICAS ---
    stats_nodos = [] # Para guardar num átomos por molécula
    stats_edges = [] # Para guardar num enlaces por molécula

    # 1. Identificar columnas clave
    smiles_col = next((col for col in df.columns if col.lower() == "smiles"), None)
    if smiles_col is None:
        raise ValueError("El CSV debe tener una columna 'SMILES'.")

    name_col = next((col for col in df.columns if col.lower() in ["id", "name"]), None)
    
    # Buscamos la columna que contenga la palabra 'target'. Si no existe, avisaremos.
    target_col = next((col for col in df.columns if "target" in col.lower()), None)
    if target_col is None:
        logger.warning("No se encontró ninguna columna que contenga 'target'. El .txt tendrá 'N/A' en ese campo.")

    # 2. Preparar el archivo de texto de salida
    # Obtenemos el nombre base del archivo (ej: 'dataset.csv' -> 'dataset')
    csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
    txt_filename = f"targets_{csv_basename}.txt"
    txt_path = os.path.join(output_dir, txt_filename)

    files_created = 0

    logger.info(f"Generando SDFs y archivo de reporte en: {txt_path}")

    # Abrimos el archivo de texto. Usamos 'w' para escribir (sobrescribe si existe).
    with open(txt_path, "w", encoding="utf-8") as f_txt:
        
        # Escribimos cabecera (opcional, pero recomendado)
        # f_txt.write("ID\tTarget\n")

        for i, row in df.iterrows():
            smiles = str(row[smiles_col]).strip()
            if not smiles:
                logger.warning(f"Fila {i}: SMILES vacío, se omite.")
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Fila {i}: SMILES inválido '{smiles}', se omite.")
                continue

            # ---------------------------------------------------------
            # ### BLOQUE DE LIMPIEZA (Keep Largest Fragment) ###
            # ---------------------------------------------------------

            # Obtener todos los fragmentos individuales
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)

            if len(frags) > 1:
                mol_principal = max(frags, key=lambda m: m.GetNumAtoms())
                logger.info(f"Fila {i}: Se eliminaron fragmentos pequeños/desconectados (Sales/Iones).")
                mol = mol_principal
            
            # Validamos de nuevo si lo que quedó es válido
            if mol.GetNumAtoms() < minimoNodos:
                logger.warning(f"Fila {i}: Es menor que el threshold de nodos tras limpieza, se omite.")
                continue

            if mol.GetNumBonds() == 0:
                logger.warning(f"Fila {i}: Molécula sin conexiones tras limpieza, se omite.")
                continue
            
            # --- CORRECCIÓN CRÍTICA AQUÍ ---
            # Debemos sanitizar aquí para recuperar RingInfo antes de pasar a 3D/UFF.
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                logger.warning(f"Fila {i}: Error sanitización tras limpieza de fragmentos ({e}), se omite.")
                continue
            # -------------------------------

            # ---------------------------------------------------------
            # ### FIN BLOQUE DE LIMPIEZA ###
            # ---------------------------------------------------------

            # Procesamiento 3D
            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            success = AllChem.EmbedMolecule(mol, params)
            if success != 0:
                logger.warning(f"Fila {i}: Falló generación 3D, se omite.")
                continue

            AllChem.UFFOptimizeMolecule(mol)

            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                logger.warning(f"Fila {i}: Error sanitización ({e}), se omite.")
                continue

            # (B) --- RECOLECTAR ESTADÍSTICAS ---
            # Es importante hacerlo sobre la molécula FINAL que se guarda (con Hidrógenos)
            num_atoms = mol.GetNumAtoms()
            num_bonds = mol.GetNumBonds()
            
            stats_nodos.append(num_atoms)
            stats_edges.append(num_bonds)
            # -----------------------------------

            # Definir Nombre (ID)
            name = (
                str(row[name_col]).strip().replace(" ", "_")
                if name_col and not pd.isna(row[name_col])
                else f"mol_{i+1}"
            )
            mol.SetProp("_Name", name)

            # Definir Target
            target_val = "N/A"
            if target_col and not pd.isna(row[target_col]):
                # Convertimos a string y reemplazamos espacios para no romper el formato
                target_val = str(row[target_col]).strip()

            # 3. Escribir SDF
            out_path = os.path.join(output_dir, f"{name}.sdf")
            with Chem.SDWriter(out_path) as writer:
                writer.write(mol)
            
            # 4. Escribir en el .txt (Solo si llegamos hasta aquí)
            # Usamos tabulador (\t) como separador, es más seguro que el espacio.
            f_txt.write(f"{name}\t{target_val}\n")
            
            files_created += 1

    # (C) --- CALCULAR Y MOSTRAR RESULTADOS ---
    if files_created > 0:
        avg_nodes = sum(stats_nodos) / len(stats_nodos)
        avg_edges = sum(stats_edges) / len(stats_edges)
        
        print("\n" + "="*40)
        print(f" RESULTADOS ESTADÍSTICOS ({files_created} moléculas)")
        print("="*40)
        print(f"Promedio de Nodos (Átomos):   {avg_nodes:.2f}")
        print(f"Promedio de Edges (Enlaces):  {avg_edges:.2f}")
        print(f"Mínimo Nodos: {min(stats_nodos)} | Máximo Nodos: {max(stats_nodos)}")
        print("="*40 + "\n")
        
        logger.info(f"Stats: Avg Nodes={avg_nodes:.2f}, Avg Edges={avg_edges:.2f}")
    else:
        logger.warning("No se procesó ninguna molécula correctamente.")



