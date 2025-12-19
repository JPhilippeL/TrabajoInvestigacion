import matplotlib.pyplot as plt
import os

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

# === 2. CONSTANTES DE TAMAÑO Y ESTILO ===
FIG_SIZE_PAPER = (8, 5)      # Tamaño ideal para papers (column width)
LINE_WIDTH = 3.0             # Grosor de línea principal
MARKER_SIZE = 9              # Tamaño de puntos
DPI_PAPER = 300              # Resolución de impresión (Estándar IEEE/ACM)

# === 3. CONFIGURACIÓN GLOBAL DE MATPLOTLIB (RC PARAMS) ===
def apply_paper_style():
    """
    Aplica una configuración global a matplotlib para que todos los gráficos
    tengan estilo de publicación científica (Serif, fuentes grandes, etc).
    """
    plt.rcParams.update({
        # Fuentes: Times New Roman o similar (Serif)
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
        'font.size': 16,
        
        # Tamaños de texto específicos
        'axes.labelsize': 20,     # Ejes X e Y
        'axes.titlesize': 18,     # Título del gráfico
        'xtick.labelsize': 14,    # Números en X
        'ytick.labelsize': 14,    # Números en Y
        'legend.fontsize': 16,    # Leyenda
        
        # Grosores para que se vea bien al reducir la imagen
        'lines.linewidth': LINE_WIDTH,
        'lines.markersize': MARKER_SIZE,
        'axes.linewidth': 1.5,     # Grosor del marco del gráfico
        'grid.linewidth': 0.8,
        
        # Estética general
        'figure.figsize': FIG_SIZE_PAPER,
        'figure.dpi': 100,         # DPI en pantalla (el de guardar será 300)
        'savefig.dpi': DPI_PAPER,  # DPI por defecto al guardar
        'savefig.bbox': 'tight',   # Evita cortar etiquetas
        
        # LaTeX (Opcional: poner en True solo si tienes TeX instalado en el sistema)
        'text.usetex': False       
    })

def save_paper_figure(path):
    """
    Función helper para guardar asegurando los parámetros correctos.
    """
    # Crear directorio si no existe (extra safety)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(path, dpi=DPI_PAPER, bbox_inches='tight')
    plt.close()
    print(f"--> Figura guardada: {path}")