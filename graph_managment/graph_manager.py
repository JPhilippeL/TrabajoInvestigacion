class GraphManager:
    @staticmethod
    def add_node(graph, node_id, element, position=None):
        if node_id not in graph:
            graph.add_node(node_id, element=element, pos=position)
        else:
            raise ValueError(f"Node with ID {node_id} already exists.")

    @staticmethod
    def delete_node(graph, node_id):
        if node_id in graph:
            graph.remove_node(node_id)

    @staticmethod
    def modify_node(graph, node_id, new_element):
        if node_id in graph:
            graph.nodes[node_id]['element'] = new_element

    @staticmethod
    def update_node_position(graph, node_id, new_position):
        if node_id in graph:
            graph.nodes[node_id]['pos'] = new_position
        else:
            raise ValueError(f"Node with ID {node_id} does not exist.")

    @staticmethod
    def add_edge(graph, source_id, target_id, bond_type="SINGLE"):
        if not graph.has_edge(source_id, target_id):
            graph.add_edge(source_id, target_id, bond_type=bond_type)

    @staticmethod
    def delete_edge(graph, source_id, target_id):
        if graph.has_edge(source_id, target_id):
            graph.remove_edge(source_id, target_id)

    @staticmethod
    def modify_edge(graph, source_id, target_id, new_bond_type):
        if graph.has_edge(source_id, target_id):
            graph.edges[source_id, target_id]['bond_type'] = new_bond_type.upper()

    @staticmethod
    def obtain_highest_node_id(graph):
        if not graph.nodes:
            return 0
        # Convertir los IDs a int para comparar
        max_id = max(int(node_id) for node_id in graph.nodes)
        return max_id + 1

    
