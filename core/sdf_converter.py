# sdf_parser.py
# Leer/Guardar archivos SDF y convertirlos a un grafo de NetworkX

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import networkx as nx

SCALE = 50

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