# main.py
import os
import sys
import multiprocessing as mp

# 1) Reducir hilos para BLAS/OpenMP
# (Es ideal dejar esto arriba del todo antes de que se importe torch o numpy)
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from ui.main_window import MainWindow
from ui.styles.theme import application_stylesheet

def main():
    # 2) Forzar spawn (Lo metemos aquí dentro para que solo lo haga el proceso principal)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # ya se había fijado; ok
        pass

    # 3) Inicialización de la Interfaz Gráfica
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(application_stylesheet())
    app.setApplicationName("Sistema de Análisis Molecular")
    icon_path = os.path.join(os.path.dirname(__file__), "assets", "icono2.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

# ==============================================================================
# EL ESCUDO DE PROTECCIÓN MULTIPROCESO
# Todo lo que esté aquí dentro SOLO lo ejecutará el proceso principal (padre).
# Los workers de PyTorch ignorarán esta sección.
# ==============================================================================
if __name__ == '__main__':
    main()
