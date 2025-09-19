# main.py
import os
import multiprocessing as mp

# 1) Forzar spawn
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # ya se había fijado; ok
    pass

# 2) Reducir hilos para BLAS/OpenMP
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from ui.main_window import MainWindow
import sys

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
