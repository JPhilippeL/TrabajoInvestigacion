ATOM_COLORS_DEFAULT = "#909090"  # Gris medio por defecto
ATOM_TEXT_COLORS_DEFAULT = "#FFFFFF"
BACKGROUND_COLOR = "#2E2E2E"  # Color de fondo
ATOM_COLORS = {
    "C":  "#909090",  # Gris medio para Carbono
    "H":  "#FFFFFF",  # Blanco para Hidrógeno
    "O":  "#FF4C4C",  # Rojo brillante para Oxígeno
    "N":  "#4A90E2",  # Azul claro para Nitrógeno
    "S":  "#FFFF66",  # Amarillo pálido para Azufre
    "P":  "#FF9933",  # Naranja para Fósforo
    "Cl": "#33FF33",  # Verde brillante para Cloro
    "F":  "#99FF33",  # Verde lima para Flúor
    "Br": "#B5651D",  # Marrón anaranjado para Bromo
    "I":  "#800080",  # Púrpura para Yodo
    "Fe": "#FF6600",  # Naranja fuerte para Hierro
    "Au": "#FFD700",  # Dorado para Oro
}

ATOM_TEXT_COLORS = {
    "C":  ATOM_TEXT_COLORS_DEFAULT,
    "H":  BACKGROUND_COLOR,
    "O":  ATOM_TEXT_COLORS_DEFAULT,
    "N":  ATOM_TEXT_COLORS_DEFAULT,
    "S":  BACKGROUND_COLOR,
    "P":  ATOM_TEXT_COLORS_DEFAULT,
    "Cl": ATOM_TEXT_COLORS_DEFAULT,
    "F":  BACKGROUND_COLOR,
    "Br": ATOM_TEXT_COLORS_DEFAULT,
    "I":  ATOM_TEXT_COLORS_DEFAULT,
    "Fe": ATOM_TEXT_COLORS_DEFAULT,
    "Au": ATOM_TEXT_COLORS_DEFAULT,
}

RESULTADOS_DIR = "Resultados"
MODELOS_DIR = "Modelos"

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