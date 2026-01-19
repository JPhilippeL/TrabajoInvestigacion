from PySide6.QtWidgets import QGraphicsLineItem, QGraphicsItemGroup, QMenu
from PySide6.QtGui import QPen
from PySide6.QtCore import Signal, QObject, Qt


class EdgeItem(QGraphicsItemGroup, QObject):
    modify_edge_requested = Signal(object)
    delete_edge_requested = Signal(object)
    add_node_requested = Signal(object)  

    def __init__(self, source_node, target_node, bond_type):
        QObject.__init__(self)
        QGraphicsItemGroup.__init__(self)

        self.source = source_node
        self.target = target_node
        self.bond_type = bond_type.upper()

        self.lines = []
        self.hitboxes = []

        self._create_lines()
        self.update_position()

        self.source.edges.append(self)
        self.target.edges.append(self)

    def _create_lines(self):
        # Determina desplazamientos según el tipo de enlace
        if self.bond_type == 'DOUBLE':
            offsets = [-3, 3]
        elif self.bond_type == 'TRIPLE':
            offsets = [-6, 0, 6]
        else:
            offsets = [0]  # SINGLE o AROMATIC

        for offset in offsets:
            line = QGraphicsLineItem()
            pen = QPen(Qt.white, 4)

            if self.bond_type == 'AROMATIC':
                pen.setStyle(Qt.DashLine)

            line.setPen(pen)
            line.setZValue(1)
            line.offset = offset
            self.addToGroup(line)
            self.lines.append(line)

            # Línea invisible más gruesa para detección
            hitbox = QGraphicsLineItem()
            hit_pen = QPen(Qt.transparent, 15)  # invisible pero con ancho
            hitbox.setPen(hit_pen)
            hitbox.setZValue(2)  # por encima, pero invisible
            hitbox.setAcceptedMouseButtons(Qt.RightButton)  # necesario para context menu
            hitbox.offset = 0
            hitbox.contextMenuEvent = self.contextMenuEvent  # delega el evento
            self.addToGroup(hitbox)
            self.hitboxes.append(hitbox)

    def _delete_lines(self):
        for line in self.lines:
            self.removeFromGroup(line)
            if line.scene():
                line.scene().removeItem(line)
        for hitbox in self.hitboxes:
            self.removeFromGroup(hitbox)
            if hitbox.scene():
                hitbox.scene().removeItem(hitbox)

        self.lines.clear()
        self.hitboxes.clear()

    def update_position(self):
        src = self.source.sceneBoundingRect().center()
        tgt = self.target.sceneBoundingRect().center()

        dx = tgt.x() - src.x()
        dy = tgt.y() - src.y()
        length = (dx ** 2 + dy ** 2) ** 0.5
        if length == 0:
            return

        # Vector perpendicular normalizado
        norm_x = -dy / length
        norm_y = dx / length

        for line, hitbox in zip(self.lines, self.hitboxes):
            offset = line.offset  # usan el mismo
            ox = norm_x * offset
            oy = norm_y * offset

            x1, y1 = src.x() + ox, src.y() + oy
            x2, y2 = tgt.x() + ox, tgt.y() + oy

            line.setLine(x1, y1, x2, y2)
            hitbox.setLine(x1, y1, x2, y2)

    def update_bond_type(self, new_bond_type):
        self.bond_type = new_bond_type.upper()
        self._delete_lines()
        self._create_lines()
        self.update_position()

    def contextMenuEvent(self, event):
        menu = QMenu()
        modify_action = menu.addAction("Modificar enlace")
        delete_action = menu.addAction("Eliminar enlace")
        add_node_action = menu.addAction("Añadir nodo")

        action = menu.exec(event.screenPos())

        if action == modify_action:
            self.modify_edge_requested.emit(self)
        elif action == delete_action:
            self.delete_edge_requested.emit(self)
        elif action == add_node_action:
            self.add_node_requested.emit(self)

    def remove(self):
        self._delete_lines()
        self.source.edges.remove(self)
        self.target.edges.remove(self)
        if self.scene():
            self.scene().removeItem(self)
        super().remove()