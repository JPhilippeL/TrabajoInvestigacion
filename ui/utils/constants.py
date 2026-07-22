from rdkit import Chem

# Directorios de Guardado
RESULTADOS_DIR = "Resultados"
MODELOS_DIR = "Modelos"

EXPLAINERS = [
        "GraphExplainer",
        "GNNExplainer",
        # "Captum_IntegratedGradients",
        # "Captum_InputXGradient",
        "Captum_ShapleyValueSampling",
        # "Captum_Saliency",          # <--- NUEVO
        # "Captum_Deconvolution",     # <--- NUEVO
        # "Captum_GuidedBackprop",    # <--- NUEVO
        "DummyExplainer",
    ]

# Features Training
periodic_elements = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe',
                     'As', 'Al', 'I', 'B', 'V', 'K','Sb', 'Sn', 'Ag', 'Pd',
                     'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Ni', 'Cd', 'In',
                     'Mn', 'Zr','Cr', 'Pt', 'Pb','Unknown']

hybridization_types = ['S', 'SP', 'SP2', 'SP2D','SP3','SP3D', 'OTHER','UNSPECIFIED']

EMBEDDING_INDICES = {
    "ATOM_SYMBOL": 0,
    "HYBRIDIZATION": 1,
    "DEGREE": 2,
    "TOTAL_HS": 3,
    "IS_AROMATIC": 4,
    "IS_DONOR": 5,
    "IS_ACCEPTOR": 6
}

L_A = len(periodic_elements)

ONE_HOT_INDICES = {
    "ATOM_SYMBOL": slice(0, L_A),                 # Ocupa los primeros L_A índices
    "DEGREE": L_A + 0,                            # Justo después del One-Hot del átomo
    "TOTAL_HS": L_A + 1,
    "IS_AROMATIC": L_A + 2,
    "IS_DONOR": L_A + 3,
    "IS_ACCEPTOR": L_A + 4,
    "HYBRIDIZATION": slice(L_A + 5, L_A + 5 + len(hybridization_types)) # Al final
}

# Definimos los valores de "Unknown" para las categorías
# Asumimos que 'Unknown' es el último elemento de tus listas
UNKNOWN_ATOM_IDX = len(periodic_elements) - 1
UNKNOWN_HYBRID_IDX = len(hybridization_types) - 1

# Suponiendo que tienes acceso a periodic_elements y hybridization_types
NODE_FEATURE_NAMES_ONE_HOT = (
    [f"Atom_{atom}" for atom in periodic_elements] + 
    ["Degree", "Total_Hs", "Is_Aromatic", "Is_Donor", "Is_Acceptor"] + 
    [f"Hybrid_{h}" for h in hybridization_types]
)

NODE_FEATURES_NAMES_EMBEDDING = (
    ["AtomSymbol"] + 
    ["Degree", "Total_Hs", "Is_Aromatic", "Is_Donor", "Is_Acceptor"] + 
    ["Hybridization"]
)

# Grupos para facilitar la lógica
CATEGORICAL_INDICES = [EMBEDDING_INDICES["ATOM_SYMBOL"], EMBEDDING_INDICES["HYBRIDIZATION"]]

# --- ENLACES
BOND_TYPE_TO_INT = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
    Chem.rdchem.BondType.OTHER: 4,
    Chem.rdchem.BondType.UNSPECIFIED: 4,
    "OTHER": 4,
    "PEPTIDE": 5
}

# 2. INTERACCIONES NO COVALENTES (Vector Multi-Hot de 25 dimensiones) - NUEVO
BOND_TYPES_NON_COVALENT = [
    "AMIDERING", "hydrophobic", "CARBONPI", "DONORPI", "weak_polar",
    "weak_hbond", "aromatic", "METSULPHURPI", "EF",
    "polar", "FE", "hbond", "FF", "FT", "OT", "OE", "OF",
    "carbonyl", "ET", "xbond", "HALOGENPI", "ionic", "CATIONPI"
]

N_NON_COV = len(BOND_TYPES_NON_COVALENT) # 25

EDGE_EMBEDDING_INDICES = {
    "BOND_TYPE": 0,                                 # Categórica (Pasa por nn.Embedding)
    "NON_COVALENT": slice(1, 1 + N_NON_COV),        # Multi-hot (Pasa por nn.Linear)
    "DISTANCE": 1 + N_NON_COV,                      # Continua (Columna 26)
    "BOND_FLEXIBILITY": 1 + N_NON_COV + 1           # Continua (Columna 27)
}

# Cuenta los valores únicos (del 0 al 15 = 16 clases)
L_B = max(BOND_TYPE_TO_INT.values()) + 1  # Esto dará 16

EDGE_ONE_HOT_INDICES = {
    "BOND_TYPE": slice(0, L_B),                             # Ej: 0 a 5
    "NON_COVALENT": slice(L_B, L_B + N_NON_COV),            # Ej: 6 a 30
    "DISTANCE": L_B + N_NON_COV,                            # Ej: 31
    "FLEXIBILITY": L_B + N_NON_COV + 1                      # Ej: 32
}

CATEGORICAL_EDGE_INDICES = ["BOND_TYPE", "NON_COVALENT"]

# Nombres exactos mapeados a los índices del 0 al 15
BOND_CLASS_NAMES = [
    "SINGLE",          # 0
    "DOUBLE",          # 1
    "TRIPLE",          # 2
    "AROMATIC",        # 3
    "OTHER",       # 4 (Agrupa OTHER, UNSPECIFIED)
    "PEPTIDE"
]

# Y ahora tu lista final de features para aristas será perfecta:
EDGE_FEATURE_NAMES = BOND_CLASS_NAMES + BOND_TYPES_NON_COVALENT + ["Distance", "Flexibility"]

EDGE_FEATURE_NAMES_EMBEDDING = ["BondType"] + BOND_TYPES_NON_COVALENT + ["Distance", "Flexibility"]

# Variable auxiliar para el índice de "Unknown Bond"
UNKNOWN_BOND_IDX = BOND_TYPE_TO_INT[Chem.rdchem.BondType.OTHER]

# Mapas inversos y rápidos
ATOM_TYPE_TO_IDX = {el: i for i, el in enumerate(periodic_elements)}
HYBRID_TO_IDX = {h: i for i, h in enumerate(hybridization_types)}

N_BOND_TYPES = len(BOND_TYPE_TO_INT)

# Porcentaje Estandard de Reduccion del Embedding (40 * 0.4 = 16), (8 * 0.5 = 4), (6 * 0.5 = 3)
ATOM_EMB_PR = 0.3
HYBRID_EMB_PR = 0.3
BOND_EMB_PR = 0.3
NON_COV_EMB_PR = 0.3 # <--- AÑADIR: Para la nueva capa Linear de la GNN

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
    "NNConv"
)