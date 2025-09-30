from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QMessageBox, QSplitter
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from ui.graph.graph_view import MoleculeGraphView
from core.sdf_converter import parse_sdf
from ui.file_selector import FileSelector
from ui.welcome_screen import WelcomeScreen
from ui.menu_bar import MenuBar
from PySide6.QtWidgets import QTextEdit
from PySide6.QtGui import QTextCursor
import networkx as nx
import logging
from ui.logger import QtHandler
from ML.training_controller_process import TrainingControllerProcess

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Análisis Molecular")
        self.resize(900, 600)

        # Contenedor central permanente
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.menu_bar = MenuBar(self)
        self.setMenuBar(self.menu_bar)

        # Creamos un splitter vertical
        self.splitter = QSplitter(Qt.Vertical, self.central_widget)

        # Layout principal solo con el splitter
        layout = QVBoxLayout(self.central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.splitter)

        # Welcome screen inicial
        self.splitter.addWidget(WelcomeScreen())

        # Área de log 
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Mensajes del sistema...")
        self.splitter.addWidget(self.log_output)

        # Ajustar proporciones: 75% para el contenido, 25% para el log
        self.splitter.setStretchFactor(0, 3)  # índice 0: WelcomeScreen o GraphView
        self.splitter.setStretchFactor(1, 1)  # índice 1: log

        self.qt_handler = QtHandler(self.log)  # pasamos función directamente
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.qt_handler.setFormatter(formatter)

        logger = logging.getLogger()
        logger.addHandler(self.qt_handler)
        logger.setLevel(logging.DEBUG)

        # Controlador de entrenamiento
        #self.training_controller = TrainingController(self)
        self.training_controller = TrainingControllerProcess(self)


    def create_new_graph(self):
        graph = nx.Graph()
        new_graph_view = MoleculeGraphView(graph)

        self.graph_view = new_graph_view
        self.clear_splitter()

        self.splitter.insertWidget(0, self.graph_view)

        QMessageBox.information(
            self,
            "Consejo",
            "Haz clic derecho sobre el área para añadir un nodo."
        )


    def load_graph_from_file(self, file_path):
        if not file_path:
            return

        try:
            graph = parse_sdf(file_path)
        except Exception as e:
            QMessageBox.critical(self, "Error al cargar", f"No se pudo cargar el archivo:\n{str(e)}")
            return

        new_graph_view = MoleculeGraphView(graph)
        self.clear_splitter()

        self.graph_view = new_graph_view

        self.splitter.insertWidget(0, self.graph_view)


    def clear_splitter(self):
        widget = self.splitter.widget(0)
        widget.setParent(None)


    def log(self, message):
        scrollbar = self.log_output.verticalScrollBar()
        # Proteger por si el scrollbar no está listo o tiene rango 0
        if scrollbar.maximum() == 0:
            self.log_output.append(message)
            return

        self.log_output.append(message)

        cursor = self.log_output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_output.setTextCursor(cursor)
        self.log_output.ensureCursorVisible()


