import matplotlib.pyplot as plt
import os

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