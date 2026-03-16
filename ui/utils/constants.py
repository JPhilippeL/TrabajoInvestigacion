from rdkit import Chem

# Directorios de Guardado
RESULTADOS_DIR = "Resultados"
MODELOS_DIR = "Modelos"

# Features Training
periodic_elements = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe',
                     'As', 'Al', 'I', 'B', 'V', 'K','Sb', 'Sn', 'Ag', 'Pd',
                     'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Ni', 'Cd', 'In',
                     'Mn', 'Zr','Cr', 'Pt', 'Pb','Unknown']

hybridization_types = ['S', 'SP', 'SP2', 'SP2D','SP3','SP3D', 'OTHER','UNSPECIFIED']

# DICCIONARIO DE INDICES PARA MODO EMBEDDING
# Basado en el orden de tu función get_atom_features(mode='embedding')
# EMBEDDING_INDICES = {
#     "ATOM_SYMBOL": 0,
#     "HYBRIDIZATION": 1,
#     "DEGREE": 2,
#     "TOTAL_HS": 3,
#     "IS_AROMATIC": 4,
#     "FORMAL_CHARGE": 5,
#     "GASTEIGER": 6,
#     "IS_DONOR": 7,
#     "IS_ACCEPTOR": 8
# }

EMBEDDING_INDICES = {
    "ATOM_SYMBOL": 0,
    "HYBRIDIZATION": 1,
    "DEGREE": 2,
    "TOTAL_HS": 3,
    "IS_AROMATIC": 4,
    "IS_DONOR": 5,
    "IS_ACCEPTOR": 6
}

EDGE_EMBEDDING_INDICES = {
    "BOND_TYPE": 0,
    "DISTANCE": 1
}

# Definimos los valores de "Unknown" para las categorías
# Asumimos que 'Unknown' es el último elemento de tus listas
UNKNOWN_ATOM_IDX = len(periodic_elements) - 1
UNKNOWN_HYBRID_IDX = len(hybridization_types) - 1

# Grupos para facilitar la lógica
CATEGORICAL_INDICES = [EMBEDDING_INDICES["ATOM_SYMBOL"], EMBEDDING_INDICES["HYBRIDIZATION"]]
# Definimos cuáles son categóricas para los enlaces
CATEGORICAL_EDGE_INDICES = [EDGE_EMBEDDING_INDICES["BOND_TYPE"]]

# --- ENLACES
BOND_TYPE_TO_INT = {
    # --- ENLACES COVALENTES ESTÁNDAR ---
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
    
    # --- ENLACES NO COVALENTES (Los agrupamos en el índice 4) ---
    Chem.rdchem.BondType.IONIC: 4,
    Chem.rdchem.BondType.HYDROGEN: 4,
    Chem.rdchem.BondType.DATIVE: 4,
    Chem.rdchem.BondType.ZERO: 4, # A veces usado para interacciones metálicas o de van der Waals
    
    # --- OTROS / DESCONOCIDOS ---
    Chem.rdchem.BondType.OTHER: 5,
    Chem.rdchem.BondType.UNSPECIFIED: 5
}

EDGE_FEATURE_NAMES = ["Single", "Double", "Triple", "Aromatic", "No_Covalent", "Other", "Distance"]

# Variable auxiliar para el índice de "Unknown Bond"
UNKNOWN_BOND_IDX = BOND_TYPE_TO_INT[Chem.rdchem.BondType.OTHER]

# Mapas inversos y rápidos
ATOM_TYPE_TO_IDX = {el: i for i, el in enumerate(periodic_elements)}
HYBRID_TO_IDX = {h: i for i, h in enumerate(hybridization_types)}

N_BOND_TYPES = len(BOND_TYPE_TO_INT)

# Porcentaje Estandard de Reduccion del Embedding (40 * 0.4 = 16), (8 * 0.5 = 4), (6 * 0.5 = 3)
ATOM_EMB_PR = 0.4
HYBRID_EMB_PR = 0.5
BOND_EMB_PR = 0.5

# Total: 7. Categóricas: 2 (Symbol, Hybridization). Resto: 5.
OTHER_NODE_FEATURES = len(EMBEDDING_INDICES) - len(CATEGORICAL_INDICES)

# Total: 2. Categóricas: 1. Resto (Distance): 1.
OTHER_EDGE_FEATURES = len(EDGE_EMBEDDING_INDICES) - len(CATEGORICAL_EDGE_INDICES)

GNN_ARCHITECTURES = (
    "GIN",
    "GINE",
    "GAT",
    "GraphTransformer",
    "EGAT",
    "NNConv"  # <-- Puedes agregar fácilmente el modelo que definiste antes
)