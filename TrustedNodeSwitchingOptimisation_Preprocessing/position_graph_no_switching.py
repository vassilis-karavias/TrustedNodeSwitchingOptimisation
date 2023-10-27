from generate_graph import *
from trusted_nodes_utils import *
import random
# from optimal_solution_finding import *
from capacity_graph import CapacityGraph
import graph_tool as gt
import os
import csv


class Position_Graph_No_Switching(Graph):

    # def __init__(self, box_size, no_nodes, no_source_nodes, no_of_conns_av):
    #     if no_source_nodes > no_nodes:
    #         print("The number of source nodes cannot exceed the number of total nodes")
    #         raise ValueError
    #     xcoords = np.random.uniform(low=0.0, high=box_size, size=(no_nodes))
    #     ycoords = np.random.uniform(low=0.0, high=box_size, size=(no_nodes))
    #     position_vectors = {}
    #     for i in range(len(xcoords)):
    #         position_i = Vector(np.array([xcoords[i], ycoords[i]]))
    #         position_vectors[i] = position_i
    #     # we generate no_source_nodes random directions and select the furthest node in this direction
    #     # should the furthest node already be a source node we chose a different direction. If after a certain number of
    #     # attempts a new furthest node is not found then a random node is selected to be the source node
    #     position_in_array_source_nodes = []
    #     for j in range(no_source_nodes):
    #         n = 0
    #         while n < 10:
    #             xcoord = np.random.uniform(low=0.0, high=1)
    #             ycoord = np.random.uniform(low=0.0, high=1)
    #             position_direction = Vector(np.array([xcoord, ycoord]))
    #             position_direction = position_direction.normalise()
    #             current_max_value = np.NINF
    #             current_max_key = None
    #             for key in position_vectors:
    #                 dot_prod = position_direction.dot_prod(position_vectors[key])
    #                 if current_max_value < dot_prod:
    #                     current_max_value = dot_prod
    #                     current_max_key = key
    #             if current_max_key not in position_in_array_source_nodes:
    #                 position_in_array_source_nodes.append(current_max_key)
    #                 break
    #             else:
    #                 n += 1
    #         if n == 10:
    #             set_keys = set(position_vectors.keys())
    #             set_currently_source = set(position_in_array_source_nodes)
    #             key_to_be_source = random.choice(list(set_keys - set_currently_source))
    #             position_in_array_source_nodes.append(key_to_be_source)
    #     nodetypes = []
    #     for i in range(no_nodes):
    #         if i in position_in_array_source_nodes:
    #             nodetypes.append(NodeType(0))
    #         else:
    #             nodetypes.append(NodeType(3))
    #     node_names = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    #                   "11", "12", "13", "14", "15"]
    #     for i in range(16, 500):
    #         node_names.append(str(i))
    #     g = MeshNetwork(xcoords, ycoords, nodetypes, no_of_conns_av, box_size, node_names, dbswitch = 0)
    #     # store_optimal_solution_for_trusted_node_inv_distances_bb84(g, distance=100, xcoords=xcoords,
    #     #                                                             ycoords=ycoords, data_save_name="new_data/1_edges_graph_bb84_network",
    #     #                                                             node_data_save_name="new_data/1_nodes_graph_bb84_network")
    #     self.x_coord = g.new_vertex_property(value_type="double")
    #     self.y_coord = g.new_vertex_property(value_type="double")
    #     self.vertex_type = g.new_vertex_property(value_type="object")
    #     self.label = g.new_vertex_property(value_type="string")
    #     vertices = g.get_vertices()
    #     for vertex in vertices:
    #         self.vertex_type[vertices[vertex]] = nodetypes[vertex]
    #         self.x_coord[vertices[vertex]] = xcoords[vertex]
    #         self.y_coord[vertices[vertex]] = ycoords[vertex]
    #         self.label[vertices[vertex]] = node_names[vertex]
    #     self.g = g
    #     # self.lengths_of_connections[edge] = get_length(x_coord, y_coord, source_node, target_node).item()
    #     super().__init__(g = g, directed = False)

    def __init__(self, node_dataframe, edge_dataframe, no_source_nodes, db_loss = 1):
        node_dataframe = node_dataframe.sort_values(by=["node"])
        self.node_dataframe = node_dataframe
        # change the edge info to appropriate form to add into graph directly with inbuilt functions
        edges = edge_dataframe.drop(["weight"], axis=1)
        # create an undirected graph
        g = Graph(directed=False)
        self.db_loss = db_loss
        self.edge_dataframe = edge_dataframe
        xcoords = self.node_dataframe["xcoord"].values
        ycoords = self.node_dataframe["ycoord"].values
        position_vectors = {}
        for i in range(len(xcoords)):
            position_i = Vector(np.array([xcoords[i], ycoords[i]]))
            position_vectors[i] = position_i
        # we generate no_source_nodes random directions and select the furthest node in this direction
        # should the furthest node already be a source node we chose a different direction. If after a certain number of
        # attempts a new furthest node is not found then a random node is selected to be the source node
        position_in_array_source_nodes = []
        for j in range(no_source_nodes):
            n = 0
            while n < 10:
                xcoord = np.random.uniform(low=0.0, high=1)
                ycoord = np.random.uniform(low=0.0, high=1)
                position_direction = Vector(np.array([xcoord, ycoord]))
                position_direction = position_direction.normalise()
                current_max_value = np.NINF
                current_max_key = None
                for key in position_vectors:
                    dot_prod = position_direction.dot_prod(position_vectors[key])
                    if current_max_value < dot_prod:
                        current_max_value = dot_prod
                        current_max_key = key
                if current_max_key not in position_in_array_source_nodes:
                    position_in_array_source_nodes.append(current_max_key)
                    break
                else:
                    n += 1
            if n == 10:
                set_keys = set(position_vectors.keys())
                set_currently_source = set(position_in_array_source_nodes)
                key_to_be_source = random.choice(list(set_keys - set_currently_source))
                position_in_array_source_nodes.append(key_to_be_source)
        node_types = []
        for i in range(len(xcoords)):
            if i in position_in_array_source_nodes:
                node_types.append(NodeType(0))
            else:
                node_types.append(NodeType(3))
        # add all edges
        ## only need this if base 1
        # edges = edges.values
        # new_edges = []
        # for edge in edges:
        #     new_edges.append([edge[0]-1, edge[1]-1])
        # edges = new_edges
        # vertex_ids = g.add_edge_list(edges)
        ## if base 0 use this
        vertex_ids = g.add_edge_list(edges.values)
        self.x_coord = g.new_vertex_property(value_type="double")
        self.y_coord = g.new_vertex_property(value_type="double")
        self.vertex_type = g.new_vertex_property(value_type="object")
        self.label = g.new_vertex_property(value_type="string")
        g.vertex_properties["name"] = self.label
        # nodetypes = node_dataframe["type"]
        xcoords = node_dataframe["xcoord"]
        ycoords = node_dataframe["ycoord"]
        distances = edge_dataframe["weight"]
        # set up the positions in the Network that each node is at
        # also set up coordinates of the vertices and then names of the vertices

        vertices = g.get_vertices()
        for vertex in vertices:
            self.vertex_type[vertices[vertex]] = node_types[vertex]
            self.x_coord[vertices[vertex]] = xcoords.values[vertex]
            self.y_coord[vertices[vertex]] = ycoords.values[vertex]
            self.label[vertices[vertex]] = node_types[vertex]
        self.lengths_of_connections = g.new_edge_property(value_type="double", vals=distances.values)
        self.connection_uses_Oband = None
        self.g = g

        # self.lengths_of_connections[edge] = get_length(x_coord, y_coord, source_node, target_node).item()
        super().__init__(g = g, directed = False)


    def set_new_node_types(self, node_types):
        if len(node_types)!= len(self.g.get_vertices()):
            raise ValueError
        else:
            node_types_2 = []
            for i in self.g.get_vertices():
                if node_types[i] == 0:
                    node_types_2.append(NodeType(0))
                else:
                    node_types_2.append(NodeType(3))
        self.vertex_type = self.g.new_vertex_property(value_type="object")
        vertices = self.g.get_vertices()
        for vertex in vertices:
            self.vertex_type[vertices[vertex]] = node_types_2[vertex]


    def set_o_band_set(self, oband_set):
        self.connection_uses_Oband = oband_set


    def set_db_switch(self, new_db_switch):
        self.db_loss = new_db_switch

    def store_position_graph(self, node_data_store_location, edge_data_store_location, graph_id = 0):
        # storage distances for plotting purposes - use true distances not adjusted distances to accomodate switch loss
        edges = self.g.get_edges(eprops=[self.lengths_of_connections])
        dictionaries = []
        dictionary_fieldnames = ["ID", "source", "target", "distance"]
        for edge in range(len(edges)):
            source = edges[edge][0]
            target = edges[edge][1]
            distance = edges[edge][2]
            dictionaries.append(
                {"ID": graph_id, "source": source, "target": target, "distance": distance})
        nodes = self.g.get_vertices(vprops =[self.x_coord, self.y_coord])
        dictionary_fieldnames_nodes = ["ID", "node", "xcoord", "ycoord", "type"]
        dict_nodes = []
        for node in range(len(nodes)):
            node_label = nodes[node][0]
            xcoord = nodes[node][1]
            ycoord = nodes[node][2]
            type = self.vertex_type[node]
            dict_nodes.append({"ID": graph_id, "node": node_label, "xcoord": xcoord, "ycoord": ycoord ,"type": type})

        if os.path.isfile(node_data_store_location + '.csv'):
            with open(node_data_store_location + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames_nodes)
                writer.writerows(dict_nodes)
        else:
            with open(node_data_store_location + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames_nodes)
                writer.writeheader()
                writer.writerows(dict_nodes)

        if os.path.isfile(edge_data_store_location + '.csv'):
            with open(edge_data_store_location + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writerows(dictionaries)
        else:
            with open(edge_data_store_location + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
                writer.writerows(dictionaries)

    def get_required_connections(self):
        required_connections = []
        vertices = self.get_vertices()
        for vertex_1 in range(len(vertices)):
            for vertex_2 in range(vertex_1, len(vertices)):
                if self.vertex_type[vertices[vertex_1]] == NodeType.S and self.vertex_type[
                    vertices[vertex_2]] == NodeType.S and vertex_1 != vertex_2:
                    required_connections.append((vertex_1, vertex_2))
        return required_connections

    def store_n_k_for_n_state(self, n,store_file_location, graph_id=0):
        key_capacities = []
        required_connections = self.get_required_connections()
        for key in required_connections:
            key_capacities.append({"ID": graph_id, "source": key[0], "target": key[1], "N_k": n})
        dictionary_fieldnames = ["ID", "source", "target", "N_k"]
        if os.path.isfile(store_file_location + '.csv'):
            with open(store_file_location + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writerows(key_capacities)
        else:
            with open(store_file_location + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
                writer.writerows(key_capacities)

    def generate_capacity_graph_trusted_nodes_bb84(self,  dictionary_bb84):
        """
        generate capacity graph with trusted nodes connections only - capacities between trusted nodes are calculated
        using BB84 Decoy protocol
        :param dictionary_bb84: The dictionary that defines the capacities according to length of connections [source,
         target, capacity] - BB84
        :return: The CapacityGraph with only trusted nodes connections
        """
        # get the capacities for trusted nodes connections
        edges = self.g.get_edges(eprops=[self.lengths_of_connections])
        edge_data = []
        for edge in edges:
            length = edge[2]
            if self.connection_uses_Oband != None:
                if (edge[0],edge[1]) in self.connection_uses_Oband:
                    length = length * 0.35 / 0.2
            distance_actual = round(length,2)
            # distance_actual = round(length)
            if distance_actual > 999:
                capacity = 0.0
            else:
                # from the look-up table
                capacity = dictionary_bb84["L" + str(distance_actual)]
            edge_data.append([edge[0], edge[1], capacity])
        # create the CapacityGraph
        return CapacityGraph(edge_data)

    def generate_capacity_graph_trusted_nodes_bb84_with_switching(self,  dictionary_bb84):
        """
        generate capacity graph with trusted nodes connections only - capacities between trusted nodes are calculated
        using BB84 Decoy protocol
        :param dictionary_bb84: The dictionary that defines the capacities according to length of connections [source,
         target, capacity] - BB84
        :return: The CapacityGraph with only trusted nodes connections
        """
        edges = self.g.get_edges(eprops=[self.lengths_of_connections])
        edge_data = []
        for edge in edges:
            if self.connection_uses_Oband != None:
                if (int(edge[0]),int(edge[1])) in self.connection_uses_Oband or (int(edge[1]),int(edge[0])) in self.connection_uses_Oband:
                    length = edge[2] * 0.35 / 0.2
                else:
                    length = edge[2]
            else:
                length = edge[2]
            length = length + 5 * self.db_loss
            distance_actual = round(length,2)
            # distance_actual = round(length)
            if distance_actual > 999:
                capacity = 0.0
            else:
                # from the look-up table
                capacity = dictionary_bb84["L" + str(distance_actual)]
            edge_data.append([edge[0], edge[1], capacity])
        # create the CapacityGraph
        return CapacityGraph(edge_data)


    def generate_capacity_graph_trusted_nodes_tfqkd_with_switching(self,  dictionary_tf):
        """
        generate capacity graph with trusted nodes connections only - capacities between trusted nodes are calculated
        using BB84 Decoy protocol
        :param dictionary_bb84: The dictionary that defines the capacities according to length of connections [source,
         target, capacity] - BB84
        :return: The CapacityGraph with only trusted nodes connections
        """
        edges = self.g.get_edges(eprops=[self.lengths_of_connections])
        edge_data = []
        for edge in edges:
            if self.connection_uses_Oband != None:
                if (int(edge[0]),int(edge[1])) in self.connection_uses_Oband or (int(edge[1]),int(edge[0])) in self.connection_uses_Oband:
                    length = edge[2] * 0.35 / 0.2
                else:
                    length = edge[2]
            else:
                length = edge[2]
            length = length + 5 * self.db_loss

            distance_actual = round(length, 2)
            if distance_actual > 999:
                capacity = 0.0
            else:
                # from the look-up table
                capacity = dictionary_tf["L" + str(int(distance_actual)) + "LB" + str(int(distance_actual/2))]
            edge_data.append([edge[0], edge[1], capacity])
        # create the CapacityGraph
        return CapacityGraph(edge_data)


    def generate_capacity_graph_trusted_nodes_tfqkd(self,  dictionary_tf):
        """
        generate capacity graph with trusted nodes connections only - capacities between trusted nodes are calculated
        using BB84 Decoy protocol
        :param dictionary_bb84: The dictionary that defines the capacities according to length of connections [source,
         target, capacity] - BB84
        :return: The CapacityGraph with only trusted nodes connections
        """
        # get the capacities for trusted nodes connections
        edges = self.g.get_edges(eprops=[self.lengths_of_connections])
        edge_data = []
        for edge in edges:
            length = edge[2]
            if self.connection_uses_Oband != None:
                if (edge[0],edge[1]) in self.connection_uses_Oband:
                    length = length * 0.35 / 0.2
            distance_actual = round(length, 2)
            if distance_actual > 999:
                capacity = 0.0
            else:
                # from the look-up table
                capacity = dictionary_tf["L" + str(int(distance_actual)) + "LB" + str(int(distance_actual/2))]
            edge_data.append([edge[0], edge[1], capacity])
        # create the CapacityGraph
        return CapacityGraph(edge_data)



class Position_Graph_No_Switching_Set_Distances():

    def __init__(self):
        self.pos_graphs = {}
        self.distances = {}

    def create_graphs(self,import_path_nodes, import_path_edges, no_source_nodes, db_switch = 1):
        """
        import information of graphs from csv files that contain many different graphs segragated by ID - keep a
        dictionary of PositionGraphs of these graphs internally
        :param import_path_nodes: The path to the node data csv file
        :param import_path_edges: The path to the edge data csv file
        """
        # import data into Dataframes
        node_data = pd.read_csv(import_path_nodes)
        edge_data = pd.read_csv(import_path_edges)
        # get all possible ids for each different graph
        possible_ids = node_data["ID"].unique()
        # separate each graph based on ID and add to dictionary
        if bool(self.pos_graphs):
            current_max_id = max(key for key, value in self.pos_graphs.items())
        else:
            current_max_id = -1
        for id in possible_ids:
            id_to_insert = id + current_max_id + 1
            self.distances[id_to_insert] = edge_data[edge_data["ID"]== id]["distance"][edge_data[edge_data["ID"] == id]["distance"].keys()[0]]
            node_data_id = node_data[node_data["ID"] == id].drop(["ID"], axis=1)
            edge_data_id = edge_data[edge_data["ID"] == id].drop(["ID", "distance"], axis=1)
            self.pos_graphs[id_to_insert] = Position_Graph_No_Switching(node_dataframe=node_data_id, edge_dataframe=edge_data_id, no_source_nodes = no_source_nodes, db_loss = db_switch)


    def create_graphs_with_o_band_loss(self,import_path_nodes, import_path_edges, no_source_nodes, db_switch = 1):
        """
        import information of graphs from csv files that contain many different graphs segragated by ID - keep a
        dictionary of PositionGraphs of these graphs internally
        :param import_path_nodes: The path to the node data csv file
        :param import_path_edges: The path to the edge data csv file
        """
        # import data into Dataframes
        node_data = pd.read_csv(import_path_nodes)
        edge_data = pd.read_csv(import_path_edges)
        # get all possible ids for each different graph
        possible_ids = node_data["ID"].unique()
        # separate each graph based on ID and add to dictionary
        if bool(self.pos_graphs):
            current_max_id = max(key for key, value in self.pos_graphs.items())
        else:
            current_max_id = -1
        for id in possible_ids:
            id_to_insert = id + current_max_id + 1
            self.distances[id_to_insert] = edge_data[edge_data["ID"]== id]["distance"][edge_data[edge_data["ID"] == id]["distance"].keys()[0]]
            node_data_id = node_data[node_data["ID"] == id].drop(["ID"], axis=1)
            edge_data_id = edge_data[edge_data["ID"] == id].drop(["ID", "distance"], axis=1)
            self.pos_graphs[id_to_insert] = Position_Graph_No_Switching(node_dataframe=node_data_id, edge_dataframe=edge_data_id, no_source_nodes = no_source_nodes, db_loss = db_switch)
            for graph in self.pos_graphs.keys():
                edges_with_o_band = []
                g = self.pos_graphs[graph].g
                nodes = g.vertices()
                for node in nodes:
                    if self.pos_graphs[graph].label[node] == "S" or self.pos_graphs[graph].label[node] == "NodeType.S":
                        for e in node.out_edges():
                            edges_with_o_band.append((g.vertex_index[e.source()],g.vertex_index[e.target()]))
                        for e in node.in_edges():
                            edges_with_o_band.append((g.vertex_index[e.source()],g.vertex_index[e.target()]))
                self.pos_graphs[graph].set_o_band_set(edges_with_o_band)

    def store_n_k_for_n_state(self, n=2, store_file_location="cap_needed"):
        for key in self.pos_graphs.keys():
            self.pos_graphs[key].store_n_k_for_n_state(n,store_file_location, graph_id=key)


    def set_new_o_band_req_to_none(self):
        for graph in self.pos_graphs.keys():
            edges_with_o_band = []
            self.pos_graphs[graph].set_o_band_set(edges_with_o_band)

    def store_position_graph(self, node_data_store_location, edge_data_store_location):
        for key in self.pos_graphs.keys():
            self.pos_graphs[key].store_position_graph(node_data_store_location =node_data_store_location,
                                                                  edge_data_store_location = edge_data_store_location,
                                                                  graph_id = key)


    def store_capacity_edge_graph_distance_bb84(self, dictionary_bb84, store_file_location,
                                                node_data_store_location):
        self.trusted_nodes_graphs = {}
        for key in self.pos_graphs.keys():
            self.trusted_nodes_graphs[key] = self.pos_graphs[key].generate_capacity_graph_trusted_nodes_bb84_with_switching(dictionary_bb84=dictionary_bb84)

            self.trusted_nodes_graphs[key].store_capacity_edge_graph_distances(
                    store_file_location=store_file_location,
                    node_types=self.pos_graphs[key].vertex_type,
                    node_data_store_location=node_data_store_location,
                    graph_id=key, distance=self.distances[key])


    def store_capacity_edge_graph_distance_bb84_no_switching(self, dictionary_bb84, store_file_location,
                                                node_data_store_location):
        self.trusted_nodes_graphs = {}
        for key in self.pos_graphs.keys():
            self.trusted_nodes_graphs[key] = self.pos_graphs[key].generate_capacity_graph_trusted_nodes_bb84(dictionary_bb84=dictionary_bb84)

            self.trusted_nodes_graphs[key].store_capacity_edge_graph_distances(
                    store_file_location=store_file_location,
                    node_types=self.pos_graphs[key].vertex_type,
                    node_data_store_location=node_data_store_location,
                    graph_id=key, distance=self.distances[key])


    def store_capacity_edge_graph_distance_tf_qkd(self, dictionary_tfqkd, store_file_location,
                                                node_data_store_location):
        self.trusted_nodes_graphs = {}
        for key in self.pos_graphs.keys():
            self.trusted_nodes_graphs[key] = self.pos_graphs[key].generate_capacity_graph_trusted_nodes_tfqkd_with_switching(dictionary_tf=dictionary_tfqkd)

            self.trusted_nodes_graphs[key].store_capacity_edge_graph_distances(
                    store_file_location=store_file_location,
                    node_types=self.pos_graphs[key].vertex_type,
                    node_data_store_location=node_data_store_location,
                    graph_id=key, distance=self.distances[key])


    def store_capacity_edge_graph_distance_tf_qkd_no_switching(self, dictionary_tf_qkd, store_file_location,
                                                node_data_store_location):
        self.trusted_nodes_graphs = {}
        for key in self.pos_graphs.keys():
            self.trusted_nodes_graphs[key] = self.pos_graphs[key].generate_capacity_graph_trusted_nodes_tfqkd(dictionary_tf=dictionary_tf_qkd)

            self.trusted_nodes_graphs[key].store_capacity_edge_graph_distances(
                    store_file_location=store_file_location,
                    node_types=self.pos_graphs[key].vertex_type,
                    node_data_store_location=node_data_store_location,
                    graph_id=key, distance=self.distances[key])