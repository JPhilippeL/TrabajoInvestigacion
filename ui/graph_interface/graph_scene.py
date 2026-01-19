from PySide6.QtWidgets import QGraphicsScene, QInputDialog, QMessageBox, QGraphicsLineItem
from PySide6.QtGui import QColor, QPen, Qt, QTransform
from PySide6.QtCore import QPointF
from ui.utils.gui_style import BACKGROUND_COLOR
from ui.graph_interface.edge_item import EdgeItem
from ui.graph_interface.node_item import NodeItem
from graph_managment.graph_manager import GraphManager

class MoleculeGraphScene(QGraphicsScene):

    def __init__(self, graph):
        super().__init__()
        self.setBackgroundBrush(QColor(BACKGROUND_COLOR))
        self.graph = graph
        self._draw_graph()

        self.edge_mode = False
        self.edge_source_node = None
        self.temp_edge_line = None

    def _draw_graph(self):
        self.node_items = {}
        
        # Crear nodos
        for node_id, data in self.graph.nodes(data=True):
            x, y = data["pos"]
            element = data["element"]
            node_id = data.get("idx", node_id)
            node = NodeItem(x, y, 20, element, node_id)

            self.add_node(node)

        # Crear enlaces
        for source_id, target_id, data in self.graph.edges(data=True):
            bond_type = data.get('bond_type').upper()
            source_node = self.node_items[source_id]
            target_node = self.node_items[target_id]
            edge = EdgeItem(source_node, target_node, bond_type)

            self.add_edge(edge)

    def add_node(self, node):
        node.modify_node_requested.connect(self.on_modify_node)
        node.delete_node_requested.connect(self.on_delete_node)
        node.add_edge_requested.connect(self.start_temporary_edge)
        node.position_changed.connect(self.update_node_position)
        id = node.get_id()

        self.addItem(node)
        self.node_items[id] = node

    def add_edge(self, edge):
        edge.modify_edge_requested.connect(self.on_modify_edge)
        edge.delete_edge_requested.connect(self.on_delete_edge)
        edge.add_node_requested.connect(self.on_add_node_to_edge)

        self.addItem(edge)
        

    def on_modify_node(self, node_item):
        node_id = self._get_node_id_from_item(node_item)
        new_element, ok = QInputDialog.getText(None, "Modificar nodo", "Nuevo símbolo químico:")
        if ok and new_element:
            new_element = self.fix_element_symbol(new_element.strip())
            GraphManager.modify_node(self.graph, node_id, new_element)
            node_item.update_element(new_element)

    def on_delete_node(self, node_item):
        node_id = self._get_node_id_from_item(node_item)
        GraphManager.delete_node(self.graph, node_id)
        for edge in node_item.edges[:]:
            self.removeItem(edge)
        self.removeItem(node_item)
        del self.node_items[node_id]

    def _get_node_id_from_item(self, item):
        for node_id, node_item in self.node_items.items():
            if node_item == item:
                return node_id
            
    def start_temporary_edge(self, source_node):
        self.edge_mode = True
        self.edge_source_node = source_node
        self.temp_edge_line = QGraphicsLineItem()
        self.temp_edge_line.setPen(QPen(Qt.white, 2, Qt.SolidLine))
        self.addItem(self.temp_edge_line)

    def cancel_temporary_edge(self):
        if self.temp_edge_line:
            self.removeItem(self.temp_edge_line)
            self.temp_edge_line = None
        self.edge_mode = False
        self.edge_source_node = None

    def mouseMoveEvent(self, event):
        if self.edge_mode and self.temp_edge_line:
            start = self.edge_source_node.sceneBoundingRect().center()
            end = event.scenePos()
            self.temp_edge_line.setLine(start.x(), start.y(), end.x(), end.y())
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if self.edge_mode:
            if event.button() == Qt.RightButton:
                self.cancel_temporary_edge()
                return

            if event.button() == Qt.LeftButton:
                clicked_item = self.itemAt(event.scenePos(), QTransform())
                # Subir la jerarquía hasta encontrar un NodeItem
                while clicked_item and not isinstance(clicked_item, NodeItem):
                    clicked_item = clicked_item.parentItem()

                if isinstance(clicked_item, NodeItem) and clicked_item != self.edge_source_node:
                    source_id = self._get_node_id_from_item(self.edge_source_node)
                    target_id = self._get_node_id_from_item(clicked_item)

                    # Verificar si ya hay un enlace entre los nodos
                    if self.graph.has_edge(source_id, target_id) or self.graph.has_edge(target_id, source_id):
                        QMessageBox.information(None, "Enlace ya existe", f"Ya hay un enlace entre {source_id} y {target_id}.")
                    else:
                        GraphManager.add_edge(self.graph, source_id, target_id, bond_type="SINGLE")
                        new_edge = EdgeItem(self.edge_source_node, clicked_item, "SINGLE")
                        self.add_edge(new_edge)
                    self.cancel_temporary_edge()
                    return
                
        # Detectar clic derecho en el fondo añade nodo
        if not self.edge_mode and event.button() == Qt.RightButton:
            clicked_item = self.itemAt(event.scenePos(), QTransform())
            if not clicked_item:
                self._create_node_at(event.scenePos())
                return


        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if self.edge_mode and event.key() == Qt.Key_Escape:
            self.cancel_temporary_edge()
            return
        super().keyPressEvent(event)

    # Método para modificar un enlace
    def on_modify_edge(self, edge_item):
        source_id = self._get_node_id_from_item(edge_item.source)
        target_id = self._get_node_id_from_item(edge_item.target)
        bond_type, ok = QInputDialog.getItem(None, "Modificar enlace", "Tipo de enlace:",
                                             ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"], 0, False)
        if ok and bond_type:
            GraphManager.modify_edge(self.graph, source_id, target_id, bond_type)
            edge_item.update_bond_type(bond_type)

    # Método para eliminar un enlace
    def on_delete_edge(self, edge_item):
        source_id = self._get_node_id_from_item(edge_item.source)
        target_id = self._get_node_id_from_item(edge_item.target)
        GraphManager.delete_edge(self.graph, source_id, target_id)
        self.removeItem(edge_item)
        edge_item.source.edges.remove(edge_item)
        edge_item.target.edges.remove(edge_item)

    # Método para añadir un nodo en medio de una arista existente
    def on_add_node_to_edge(self, edge_item):
        # 1. Preguntar al usuario el símbolo químico
        symbol, ok = QInputDialog.getText(self.views()[0], "Nuevo nodo", "Símbolo del nuevo átomo (Ej: C, O, N):")
        if not ok or not symbol.strip():
            return

        symbol = self.fix_element_symbol(symbol.strip())

        # 2. Calcular posición media entre source y target
        src = edge_item.source.sceneBoundingRect().center()
        tgt = edge_item.target.sceneBoundingRect().center()
        mid = QPointF((src.x() + tgt.x()) / 2, (src.y() + tgt.y()) / 2)

        # 3. Crear nuevo nodo
        new_node_id = str(GraphManager.obtain_highest_node_id(self.graph))
        new_node_item = NodeItem(mid.x(), mid.y(), 20, symbol, new_node_id)

        self.add_node(new_node_item)
        GraphManager.add_node(self.graph, new_node_id, symbol, position=(mid.x(), mid.y()))

        # 4. Eliminar la arista original
        source_id = self._get_node_id_from_item(edge_item.source)
        target_id = self._get_node_id_from_item(edge_item.target)
        if self.graph.has_edge(source_id, target_id):
            GraphManager.delete_edge(self.graph, source_id, target_id)
            self.removeItem(edge_item)
            edge_item.source.edges.remove(edge_item)
            edge_item.target.edges.remove(edge_item)

        # 5. Añadir dos nuevas aristas
        self.graph.add_edge(source_id, new_node_id, bond_type=edge_item.bond_type)
        GraphManager.add_edge(self.graph, source_id, new_node_id, bond_type=edge_item.bond_type)
        self.graph.add_edge(new_node_id, target_id, bond_type=edge_item.bond_type)
        GraphManager.add_edge(self.graph, new_node_id, target_id, bond_type=edge_item.bond_type)

        edge1 = EdgeItem(edge_item.source, new_node_item, edge_item.bond_type)
        edge2 = EdgeItem(new_node_item, edge_item.target, edge_item.bond_type)

        self.add_edge(edge1)
        self.add_edge(edge2)

    # Metodo para crear un nuevo nodo en una posición específica
    def _create_node_at(self, pos):
        symbol, ok = QInputDialog.getText(None, "Nuevo nodo", "Símbolo del nuevo átomo (Ej: C, O, N):")
        if not ok or not symbol.strip():
            return

        symbol = self.fix_element_symbol(symbol.strip())
        new_node_id = str(GraphManager.obtain_highest_node_id(self.graph))

        new_node_item = NodeItem(pos.x(), pos.y(), 20, symbol, new_node_id)
        self.add_node(new_node_item)
        GraphManager.add_node(self.graph, new_node_id, symbol, position=(pos.x(), pos.y()))

    # Metodo para actualizar la posición de un nodo en graph_manager
    def update_node_position(self, node_item):
        node_id = self._get_node_id_from_item(node_item)
        new_position = (node_item.pos().x(), node_item.pos().y())
        GraphManager.update_node_position(self.graph, node_id, new_position)

    def fix_element_symbol(self, element: str) -> str:
        if len(element) == 1:
            return element.upper()
        elif len(element) > 1:
            return element[0].upper() + element[1:].lower()
        else:
            return element



    