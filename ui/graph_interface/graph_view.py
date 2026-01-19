from PySide6.QtWidgets import QGraphicsView
from PySide6.QtGui import QPainter
from ui.graph_interface.graph_scene import MoleculeGraphScene

class MoleculeGraphView(QGraphicsView):
    def __init__(self, graph):
        super().__init__()
        self.setScene(MoleculeGraphScene(graph))
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

