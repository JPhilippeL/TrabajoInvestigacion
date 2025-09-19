# main.py
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
