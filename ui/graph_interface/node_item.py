from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsItem, QMenu
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtCore import Signal, QObject, Qt
from ui.utils.gui_style import ATOM_COLORS, ATOM_TEXT_COLORS, ATOM_COLORS_DEFAULT, ATOM_TEXT_COLORS_DEFAULT

NODE_RADIUS = 20

class NodeItem(QGraphicsEllipseItem, QObject):
    modify_node_requested = Signal(object)
    delete_node_requested = Signal(object)
    add_edge_requested = Signal(object)
    position_changed = Signal(object)  
    
    def __init__(self, x, y, radius, element, node_id):
        QObject.__init__(self)
        QGraphicsEllipseItem.__init__(self, -radius, -radius, 2 * radius, 2 * radius)
        
        # Estilo según el elemento
        color = QColor(ATOM_COLORS.get(element, ATOM_COLORS_DEFAULT))
        text_color = QColor(ATOM_TEXT_COLORS.get(element, ATOM_TEXT_COLORS_DEFAULT))

        self.node_id = node_id
        self.setBrush(QBrush(QColor(color)))
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        self.setZValue(10)

        # Crear etiqueta de texto
        self.label = QGraphicsTextItem(element, self)
        font = QFont("Arial", 13)
        font.setBold(True)
        self.label.setFont(font)
        self.label.setDefaultTextColor(text_color)
        self.label.setZValue(20)

        # Calcular centro del nodo y ajustar el texto
        text_rect = self.label.boundingRect()
        self.label.setPos(-text_rect.width() / 2, -text_rect.height() / 2)
        self.label.setAcceptedMouseButtons(Qt.NoButton)  # No permitir interacción directa con el texto

        self.setToolTip(f"ID: {self.node_id}")

        # Lista de conexiones (enlaces)
        self.edges = []

        # Posición inicial
        self.setPos(x, y)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            for edge in self.edges:
                edge.update_position()
        elif change == QGraphicsItem.ItemPositionHasChanged:
            # Después de moverse, actualizar la posición en el grafo lógico
            pos = self.pos()
            self.position_changed.emit(self)

        return super().itemChange(change, value)
    
    def update_element(self, new_element):
        self.element = new_element

        # Actualizar color de fondo y texto
        color = QColor(ATOM_COLORS.get(new_element, ATOM_COLORS_DEFAULT))
        text_color = QColor(ATOM_TEXT_COLORS.get(new_element, ATOM_TEXT_COLORS_DEFAULT))
        self.setBrush(QBrush(color))
        self.label.setDefaultTextColor(text_color)

        # Actualizar el texto del nodo
        self.label.setPlainText(new_element)

        # Reajustar posición del texto centrado
        text_rect = self.label.boundingRect()
        self.label.setPos(-text_rect.width() / 2, -text_rect.height() / 2)
    
    def contextMenuEvent(self, event):
        menu = QMenu()
        modify_action = menu.addAction("Modificar nodo")
        delete_action = menu.addAction("Eliminar nodo")
        add_edge_action = menu.addAction("Añadir enlace")

        action = menu.exec(event.screenPos())

        if action == modify_action:
            self.modify_node_requested.emit(self)
        elif action == delete_action:
            self.delete_node_requested.emit(self)
        elif action == add_edge_action:
            self.add_edge_requested.emit(self)

    def get_id(self):
        return self.node_id