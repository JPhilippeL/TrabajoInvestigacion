# Directorios de Guardado
RESULTADOS_DIR = "Resultados"
MODELOS_DIR = "Modelos"

# Features Training
periodic_elements = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe',
                     'As', 'Al', 'I', 'B', 'V', 'K','Sb', 'Sn', 'Ag', 'Pd',
                     'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Ni', 'Cd', 'In',
                     'Mn', 'Zr','Cr', 'Pt', 'Pb','Unknown']

hybridization_types = ['S', 'SP', 'SP2', 'SP2D','SP3','SP3D', 'OTHER','UNSPECIFIED']

N_BOND_TYPES = 4

# Porcentaje Estandard de Reduccion del Embedding (40 * 0.4 = 16), (8 * 0.5 = 4), (4 * 1 = 4)
ATOM_EMB_PR = 0.4
HYBRID_EMB_PR = 0.5
BOND_EMB_PR = 1

OTHER_NODE_FEATURES = 7
OTHER_EDGE_FEATURES = 1