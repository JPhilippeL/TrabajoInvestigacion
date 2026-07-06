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
    app.setStyleSheet("""
        QWidget {
            background-color: #2b2b2b;
            color: white;
        }
    """)
    app.setApplicationName("Sistema de Análisis Molecular")
    app.setWindowIcon(QIcon("assets/icono2.png"))
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