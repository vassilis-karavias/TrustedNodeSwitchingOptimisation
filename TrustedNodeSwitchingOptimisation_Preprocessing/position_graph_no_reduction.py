import pandas as pd
import numpy as np
from generate_graph import NodeType
from graph_tool.all import *
from utils_graph import get_length
from vector import Vector
import random
from minimum_length import get_minimum_to_any_bob, get_k_minimum_to_any_bob
from capacity_calculation import calculate_capacity_efficient_all_distances, calculate_capacity_for_n_highest_capacities_efficient_corrected, calculate_capacity_for_k_highest_capacities_multiple_paths_per_detector_allowed, calculate_capacities_for_k_highest_connections_bb84
from trusted_nodes_utils import *
from capacity_graph import CapacityGraph
import os
import csv


class Position_Graph(Graph):

    def __init__(self, node_dataframe, edge_dataframe, no_source_nodes, db_switch = 1, connection_uses_Oband = None):
        """
        Creates a graph of nodes connected with edges defined by their positions and distances between each other
        :param node_dataframe: The dataframe that contains the node information  Dataframe(node, type, xcoord, ycoord)
        :param edge_dataframe: The dataframe that contains the edge information  Dataframe(source, target, weight(dist))
        """
        # sort the node dataframe with respect to the nodes
        node_dataframe = node_dataframe.sort_values(by = ["node"])
        self.node_dataframe = node_dataframe
        # change the edge info to appropriate form to add into graph directly with inbuilt functions
        edges = edge_dataframe.drop(["weight"], axis = 1)
        # create an undirected graph
        g = Graph(directed=False)
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
        vertex_ids = g.add_edge_list(edges.values)
        self.connection_uses_Oband = connection_uses_Oband
        # add all vertex properties of the graph (x,y) coords vertex_type, label
        self.x_coord = g.new_vertex_property(value_type="double")
        self.y_coord = g.new_vertex_property(value_type="double")
        self.vertex_type = g.new_vertex_property(value_type="object")
        self.label = g.new_vertex_property(value_type="string")
        # get a list of all vertices
        vertices = g.get_vertices()
        g.vertex_properties["name"] = self.label
        nodetypes = node_dataframe["type"]
        xcoords = node_dataframe["xcoord"]
        ycoords = node_dataframe["ycoord"]
        distances = edge_dataframe["weight"]
        # set up the positions in the Network that each node is at
        # also set up coordinates of the vertices and then names of the vertices
        for vertex in vertices:
            self.vertex_type[vertices[vertex]] = node_types[vertex]
            self.x_coord[vertices[vertex]] = xcoords.values[vertex]
            self.y_coord[vertices[vertex]] = ycoords.values[vertex]
            self.label[vertices[vertex]] = node_types[vertex]
        self.db_switch = db_switch
        # add the edge property: the distance between the states
        self.lengths_of_connections = g.new_edge_property(value_type="double", vals=distances.values)
        edges = g.get_edges(eprops = [self.lengths_of_connections])
        lengths_of_switched = []
        for edge in edges:
            ## Take a look at the lengths with and without switching
            length_with_switch = edge[2]
            lengths_of_switched.append(length_with_switch)
        self.lengths_with_switch = g.new_edge_property(value_type="double", vals=lengths_of_switched)
        self.g = g
        # self.lengths_of_connections[edge] = get_length(x_coord, y_coord, source_node, target_node).item()
        super().__init__(g = g, directed = False)


    def set_db_switch(self, new_db_switch):
        self.db_switch = new_db_switch

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
                    vertices[vertex_2]] == NodeType.S \
                        and vertex_1 != vertex_2:
                    required_connections.append((vertex_1, vertex_2))
        return required_connections

    def store_n_k_for_n_state(self, n,store_file_location, graph_id=0):
        key_capacities = []
        required_connections = self.get_required_connections()
        for key in required_connections:
            key_capacities.append({"ID": graph_id, "source": key[0], "target": key[1], "N_k": n })
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

    def add_n_random_trusted_nodes(self, n, size_graph):
        """
        Add n  randomly placed trusted nodes in the graph with at least 3 connections to nodes in the graph
        :param n: The number of trusted nodes to add to the graph
        :param size_graph: The size of the graph
        :return: PositionGraph with trusted nodes added.
        """
        trusted_node_dict = {"node" : list(range(0,n)), "xcoord": size_graph * np.random.random_sample(n), "ycoord": size_graph * np.random.random_sample(n)}
        trusted_node_edge_dict = {"source": [], "target": []}
        trusted_node_edge_dict_to_other_nodes = {"source": [], "target": []}
        vertices = self.get_vertices()
        distance_for_ith_node = {}
        for node in trusted_node_dict["node"]:
            m = np.random.randint(low = 1, high=5)
            distances_for_node = {}
            for vertex in vertices:
                x_coord_vertex, y_coord_vertex = self.x_coord[vertex], self.y_coord[vertex]
                x_coord_trusted_node = trusted_node_dict["xcoord"][node]
                y_coord_trusted_node = trusted_node_dict["ycoord"][node]
                posn_vertex = Vector(np.array([x_coord_vertex, y_coord_vertex]))
                posn_trusted_node = Vector(np.array([x_coord_trusted_node, y_coord_trusted_node]))
                distance = (posn_vertex - posn_trusted_node).magnitude()
                distances_for_node[vertex] = distance
            distances_for_node = {k: v for k, v in sorted(distances_for_node.items(), key=lambda item: item[1])}
            for i in range(len(distances_for_node.keys())):
                if i < m or distances_for_node[list(distances_for_node.keys())[i]] < 40:
                    trusted_node_edge_dict_to_other_nodes["source"].append(node)
                    trusted_node_edge_dict_to_other_nodes["target"].append(list(distances_for_node.keys())[i])
                    distance_for_ith_node[node] = distances_for_node[list(distances_for_node.keys())[i]]
        for i in range(len(trusted_node_dict["node"])):
            for j in range(i + 1, len(trusted_node_dict["node"])):
                position_of_trusted_node_1 = Vector(np.array([trusted_node_dict["xcoord"][i], trusted_node_dict["ycoord"][i]]))
                position_of_trusted_node_2 = Vector(np.array([trusted_node_dict["xcoord"][j], trusted_node_dict["ycoord"][j]]))
                distance = (position_of_trusted_node_1 - position_of_trusted_node_2).magnitude()
                if distance < distance_for_ith_node[trusted_node_dict["node"][i]] or distance < distance_for_ith_node[trusted_node_dict["node"][j]]:
                    trusted_node_edge_dict["source"].append(trusted_node_dict["node"][i])
                    trusted_node_edge_dict["target"].append(trusted_node_dict["node"][j])
        trusted_node_list = pd.DataFrame.from_dict(trusted_node_dict)
        trusted_node_edge_list = pd.DataFrame.from_dict(trusted_node_edge_dict)
        trusted_node_edge_list_to_other_nodes = pd.DataFrame.from_dict(trusted_node_edge_dict_to_other_nodes)
        return self.add_trusted_node_sets(trusted_node_list=trusted_node_list,
                                          trusted_node_edge_list=trusted_node_edge_list,
                                          trusted_node_edge_list_to_other_nodes=trusted_node_edge_list_to_other_nodes)

    def generate_capacity_graph_trusted_nodes_bb84(self,  dictionary_bb84):
        """
        generate capacity graph with trusted nodes connections only - capacities between trusted nodes are calculated
        using BB84 Decoy protocol
        :param dictionary_bb84: The dictionary that defines the capacities according to length of connections [source,
         target, capacity] - BB84
        :return: The CapacityGraph with only trusted nodes connections
        """
        # get the capacities for trusted nodes connections
        capacities = self.get_capacities_for_trusted_nodes_with_bb84(dictionary_bb84 = dictionary_bb84)
        # change the data to the form desired by the CapacityGraph input
        restructured_capacities = []
        for source, target, detector, capacity in capacities:
            restructured_capacities.append([source, target, capacity])
        # create the CapacityGraph
        return CapacityGraph(restructured_capacities)

    def get_capacities_for_trusted_nodes_with_bb84(self, dictionary_bb84):
        """
        get the capacity for every trusted node paired with a source node in the graph using TF-QKD
        get the capacity between trusted node pairs using Decoy BB84

        :param dictionary_tf: The dictionary that defines the capacities according to length of connections for TF-QKD
        :param dictionary_bb84: The dictionary that defines the capacities according to length of connection for Decoy BB84
        :return: The capacity array for each connection between source pairs in the graph in the form
         [(source, target, bob_used, capacity)]
        """
        # get a list of all the source vertex connections to trusted nodes in the graph
        required_connections_bb = []
        vertices = self.get_vertices()
        for vertex_1 in range(len(vertices)):
            for vertex_2 in range(len(vertices)):
                if (self.vertex_type[vertices[vertex_1]] == NodeType.T and self.vertex_type[
                    vertices[vertex_2]] == NodeType.S):
                    required_connections_bb.append((vertex_1, vertex_2))
                elif (self.vertex_type[vertices[vertex_1]] == NodeType.T and self.vertex_type[
                        vertices[vertex_2]] == NodeType.T and vertex_1 > vertex_2):
                    required_connections_bb.append((vertex_1, vertex_2))
        capacities_bb = self.get_capacities_for_each_required_connection_bb84(required_connections=required_connections_bb,
                                                                dictionary=dictionary_bb84)
        return capacities_bb

    def get_capacities_for_each_required_connection_bb84(self, required_connections, dictionary):
        """
        Get the capacities for a list of required connections where the capacity values based on distance are determined
        by dictionary - for BB84 protcol - should use appropriate dictionary
        :param required_connections: A list of required connections of the form [(source, target)]
        :param dictionary: The dictionary that defines the capacities according to length of connections
        :return: The capacity array for the connections required in the form [(source, target, bob_used - filler to keep
        same form as TF-QKD, capacity)]
        """
        rates =[]
        for source_node, target_node in required_connections:
            distance = self.get_shortest_distance_between_two_nodes(source_node, target_node) + 5 * self.db_switch
            if self.connection_uses_Oband != None:
                if (source_node, target_node) in self.connection_uses_Oband:
                    distance = distance * 0.35 / 0.2
            distance_actual = round(distance, 2)
            if distance_actual > 999:
                capacity = 0.0
            else:
                # from the look-up table
                capacity = dictionary["L" + str(distance_actual)]
            rates.append((source_node, target_node, 0, capacity))
        return rates


    def generate_capacity_graph_trusted_nodes_bb84_no_switching(self,  dictionary_bb84):
        """
        generate capacity graph with trusted nodes connections only - capacities between trusted nodes are calculated
        using BB84 Decoy protocol
        :param dictionary_bb84: The dictionary that defines the capacities according to length of connections [source,
         target, capacity] - BB84
        :return: The CapacityGraph with only trusted nodes connections
        """
        # get the capacities for trusted nodes connections
        capacities = self.get_capacities_for_trusted_nodes_with_bb84_no_switching(dictionary_bb84 = dictionary_bb84)
        # change the data to the form desired by the CapacityGraph input
        restructured_capacities = []
        for source, target, detector, capacity in capacities:
            restructured_capacities.append([source, target, capacity])
        # create the CapacityGraph
        return CapacityGraph(restructured_capacities)


    def get_capacities_for_trusted_nodes_with_bb84_no_switching(self, dictionary_bb84):
        """
        get the capacity for every trusted node paired with a source node in the graph using TF-QKD
        get the capacity between trusted node pairs using Decoy BB84

        :param dictionary_tf: The dictionary that defines the capacities according to length of connections for TF-QKD
        :param dictionary_bb84: The dictionary that defines the capacities according to length of connection for Decoy BB84
        :return: The capacity array for each connection between source pairs in the graph in the form
         [(source, target, bob_used, capacity)]
        """
        # get a list of all the source vertex connections to trusted nodes in the graph
        required_connections_bb = []
        vertices = self.get_vertices()
        for vertex_1 in range(len(vertices)):
            for vertex_2 in range(len(vertices)):
                if (self.vertex_type[vertices[vertex_1]] == NodeType.T and self.vertex_type[
                    vertices[vertex_2]] == NodeType.S):
                    required_connections_bb.append((vertex_1, vertex_2))
                elif (self.vertex_type[vertices[vertex_1]] == NodeType.T and self.vertex_type[
                        vertices[vertex_2]] == NodeType.T and vertex_1 > vertex_2):
                    required_connections_bb.append((vertex_1, vertex_2))
        capacities_bb = self.get_capacities_for_each_required_connection_bb84_no_switching(required_connections=required_connections_bb,
                                                                dictionary=dictionary_bb84)
        return capacities_bb

    def get_capacities_for_each_required_connection_bb84_no_switching(self, required_connections, dictionary):
        """
        Get the capacities for a list of required connections where the capacity values based on distance are determined
        by dictionary - for BB84 protcol - should use appropriate dictionary
        :param required_connections: A list of required connections of the form [(source, target)]
        :param dictionary: The dictionary that defines the capacities according to length of connections
        :return: The capacity array for the connections required in the form [(source, target, bob_used - filler to keep
        same form as TF-QKD, capacity)]
        """
        rates =[]
        for source_node, target_node in required_connections:
            distance = self.get_shortest_distance_between_two_nodes(source_node, target_node)
            if self.connection_uses_Oband != None:
                if (source_node, target_node) in self.connection_uses_Oband:
                    distance = distance * 0.35 / 0.2
            distance_actual = round(distance, 2)
            if distance_actual > 999:
                capacity = 0.0
            else:
                # from the look-up table
                capacity = dictionary["L" + str(distance_actual)]
            rates.append((source_node, target_node, 0, capacity))
        return rates


    def get_capacities_for_each_required_connection(self, required_connections, dictionary):
        """
        Get the capacities for a list of required connections where the capacity values based on distance are determined
        by dictionary
        :param required_connections: A list of required connections of the form [(source, target)]
        :param dictionary: The dictionary that defines the capacities according to length of connections
        :return: The capacity array for the connections required in the form [(source, target, bob_used, capacity)]
        """
        # get the shortest distances from every node to every bob
        true_distances = self.get_shortest_distance_of_source_nodes()
        # get the array of minimum distance for every source node and detector in the form [(source, target, bob_min, distance_array, distance_to_detector)]
        minimum_distance_array = []
        for source_node, target_node in required_connections:
            if (source_node, target_node) in self.connection_uses_Oband:
                true_distances = true_distances * 0.35 / 0.2
            distance_array, distance_to_detector = self.get_shortest_distances_for_given_pair(true_distances, source_node, target_node)
            bobs_minimum = []
            for k in list(true_distances.keys()):
                bobs_minimum.append(int(self.vertex_properties["name"][k]))
            minimum_distance_array.append((source_node, target_node, bobs_minimum, distance_array, distance_to_detector))
        # get the capacities of for each of the pairs of sources
        capacity, distance_array = calculate_capacity_efficient_all_distances(minimum_distance_array, dictionary)
        return capacity

    def get_shortest_distance_of_source_nodes(self):
        """
        Get the shortest distance of all source and trusted nodes to each of the Bobs. Distances between bobs are set to
        infty
        return: The distances from each of the Bobs for each source node as a dictionary of EdgePropertyMaps
        """
        # get the minimum distances from all Bobs to all nodes
        distances_from_bob_of_all_nodes = self.get_shortest_distance_from_bobs()
        # iterate over all vertices
        vertices = self.get_vertices()
        # will keep the dictionary of minimum distances from all Bobs to all source nodes
        true_distances = {}
        for vertex in range(len(vertices)):
            # for each bob
            if self.vertex_type[vertices[vertex]] == NodeType.B:
                distances_for_source_nodes = self.new_vertex_property(value_type="double")
                for v in range(len(vertices)):
                    # if not a Bob node then keep same key as before
                    if self.vertex_type[vertices[v]] == NodeType.S or self.vertex_type[
                        vertices[v]] == NodeType.T:
                        distances_for_source_nodes[vertices[v]] = \
                        distances_from_bob_of_all_nodes[vertices[vertex]].a[v]
                    else:
                        # no key can be created between these two nodes.- both detectors - as such min distance is not
                        # of interest. - set to infty
                        distances_for_source_nodes[vertices[v]] = np.infty
                true_distances[vertex] = distances_for_source_nodes
        return true_distances

    def get_shortest_distance_between_two_nodes(self, node_1, node_2):
        dist = shortest_distance(g = self.g, source = node_1, target = node_2, weights = self.lengths_with_switch)
        return dist


    def get_shortest_distance_from_bobs(self):
        """
        Get the shortest distance from each of the Bobs for each of the vertices - including other Bob vertices
        :return: The distances from each of the Bobs for each node as a dictionary of EdgePropertyMaps
        """
        # get a list of all vertices
        vertices = self.get_vertices()
        # This will hold the Bobs shortest distance
        distances_from_bob = {}
        # iterate through all the vertices
        for vertex in range(len(vertices)):
            if self.vertex_type[vertices[vertex]] ==NodeType.B:
                # if vertex is a Bob look for shortest path to all nodes - Dijkstra's Algorithm
                dist, pred = dijkstra_search(self.g, weight=self.lengths_with_switch, source=vertices[vertex])
                # add this to the dictionary - will be a dictionary of EdgePropertyMap
                distances_from_bob[vertex] = dist
        return distances_from_bob

    def get_shortest_distances_for_given_pair(self, true_distances, source, target):
        """
        Get the minimum distance and additional information for given pair of nodes
        :param true_distances: The distances from each of the Bobs for each source node as a dictionary of EdgePropertyMaps
        :param source: The source node: int
        :param target: The target node: int
        :return: the mimimum total distance (L_1+L_2), the position of the Bob at this minimum:int, the length to bob
                from one node (L_1)
        """

        if self.vertex_type[source] == NodeType(2).name or self.vertex_type[source] == NodeType(1).name or \
            self.vertex_type[target] == NodeType(2).name or self.vertex_type[target] == NodeType(1).name:
            print("Cannot generate key between a Detector Node and other Nodes")
            raise ValueError
        else:
            return get_minimum_to_any_bob(source, target, true_distances)


    def add_n_random_trusted_nodes(self, n, size_graph):
        """
        Add n  randomly placed trusted nodes in the graph with at least 3 connections to nodes in the graph
        :param n: The number of trusted nodes to add to the graph
        :param size_graph: The size of the graph
        :return: PositionGraph with trusted nodes added.
        """
        trusted_node_dict = {"node": list(range(0, n)), "xcoord": size_graph * np.random.random_sample(n),
                             "ycoord": size_graph * np.random.random_sample(n)}
        trusted_node_edge_dict = {"source": [], "target": []}
        trusted_node_edge_dict_to_other_nodes = {"source": [], "target": []}
        vertices = self.get_vertices()
        distance_for_ith_node = {}
        for node in trusted_node_dict["node"]:
            m = np.random.randint(low=1, high=5)
            distances_for_node = {}
            for vertex in vertices:
                x_coord_vertex, y_coord_vertex = self.x_coord[vertex], self.y_coord[vertex]
                x_coord_trusted_node = trusted_node_dict["xcoord"][node]
                y_coord_trusted_node = trusted_node_dict["ycoord"][node]
                posn_vertex = Vector(np.array([x_coord_vertex, y_coord_vertex]))
                posn_trusted_node = Vector(np.array([x_coord_trusted_node, y_coord_trusted_node]))
                distance = (posn_vertex - posn_trusted_node).magnitude()
                distances_for_node[vertex] = distance
            distances_for_node = {k: v for k, v in sorted(distances_for_node.items(), key=lambda item: item[1])}
            for i in range(len(distances_for_node.keys())):
                if i < m or distances_for_node[list(distances_for_node.keys())[i]] < 40:
                    trusted_node_edge_dict_to_other_nodes["source"].append(node)
                    trusted_node_edge_dict_to_other_nodes["target"].append(list(distances_for_node.keys())[i])
                    distance_for_ith_node[node] = distances_for_node[list(distances_for_node.keys())[i]]
        for i in range(len(trusted_node_dict["node"])):
            for j in range(i + 1, len(trusted_node_dict["node"])):
                position_of_trusted_node_1 = Vector(
                    np.array([trusted_node_dict["xcoord"][i], trusted_node_dict["ycoord"][i]]))
                position_of_trusted_node_2 = Vector(
                    np.array([trusted_node_dict["xcoord"][j], trusted_node_dict["ycoord"][j]]))
                distance = (position_of_trusted_node_1 - position_of_trusted_node_2).magnitude()
                if distance < distance_for_ith_node[trusted_node_dict["node"][i]] or distance < distance_for_ith_node[
                    trusted_node_dict["node"][j]]:
                    trusted_node_edge_dict["source"].append(trusted_node_dict["node"][i])
                    trusted_node_edge_dict["target"].append(trusted_node_dict["node"][j])
        trusted_node_list = pd.DataFrame.from_dict(trusted_node_dict)
        trusted_node_edge_list = pd.DataFrame.from_dict(trusted_node_edge_dict)
        trusted_node_edge_list_to_other_nodes = pd.DataFrame.from_dict(trusted_node_edge_dict_to_other_nodes)
        return self.add_trusted_node_sets(trusted_node_list=trusted_node_list,
                                          trusted_node_edge_list=trusted_node_edge_list,
                                          trusted_node_edge_list_to_other_nodes=trusted_node_edge_list_to_other_nodes)


class Position_Graph_Set_Distances():

    def __init__(self):
        self.pos_graphs = {}
        self.distances = {}

    def import_graphs(self, import_path_nodes, import_path_edges, no_source_nodes, db_switch = 1):
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
            self.pos_graphs[id_to_insert] = Position_Graph(node_dataframe=node_data_id, edge_dataframe=edge_data_id, no_source_nodes = no_source_nodes, db_switch = db_switch)

    def add_trusted_nodes_to_graphs(self, p = 0.4, dist = 40):
        self.trusted_node_pos_graph = {}
        for key in self.pos_graphs.keys():
            self.trusted_node_pos_graph[key] = self.pos_graphs[key].add_trusted_nodes_at_midpoints(p,dist)

    def add_random_trusted_nodes_to_graphs(self, n, fraction = 1):
        self.trusted_node_pos_graph = {}
        for key in self.pos_graphs.keys():
            self.trusted_node_pos_graph[key] = self.pos_graphs[key].add_n_random_trusted_nodes(n = n, size_graph = self.distances[key] / fraction)

    def store_n_k_for_n_state(self, n=2, store_file_location="cap_needed"):
        for key in self.pos_graphs.keys():
            self.pos_graphs[key].store_n_k_for_n_state(n,store_file_location, graph_id=key)


    def store_position_graph(self, node_data_store_location, edge_data_store_location):
        for key in self.pos_graphs.keys():
            self.pos_graphs[key].store_position_graph(node_data_store_location =node_data_store_location,
                                                                  edge_data_store_location = edge_data_store_location,
                                                                  graph_id = key)


    def store_capacity_edge_graph(self, dictionary_bb84, store_file_location, node_data_store_location):
        self.trusted_nodes_graphs = {}
        for key in self.pos_graphs.keys():
            self.trusted_nodes_graphs[key] = self.pos_graphs[
                key].generate_capacity_graph_trusted_nodes_bb84(dictionary_bb84=dictionary_bb84)
            self.trusted_nodes_graphs[key].store_capacity_edge_graph(store_file_location=store_file_location,
                                                                     node_types=self.pos_graphs[
                                                                         key].vertex_type,
                                                                     node_data_store_location=node_data_store_location,
                                                                     graph_id=key)

    def store_capacity_edge_graph_distance_bb84(self, dictionary_bb84, store_file_location,
                                                node_data_store_location):
        self.trusted_nodes_graphs = {}
        for key in self.pos_graphs.keys():
            try:
                self.trusted_nodes_graphs[key] = self.pos_graphs[
                    key].generate_capacity_graph_trusted_nodes_bb84(dictionary_bb84=dictionary_bb84)

                self.trusted_nodes_graphs[key].store_capacity_edge_graph_distances(
                    store_file_location=store_file_location,
                    node_types=self.pos_graphs[
                        key].vertex_type,
                    node_data_store_location=node_data_store_location,
                    graph_id=key, distance=self.distances[key])
            except:
                self.trusted_node_pos_graph[key] = self.pos_graphs[key].add_n_random_trusted_nodes(n=2, size_graph=
                self.distances[key])
                self.trusted_nodes_graphs[key] = self.pos_graphs[
                    key].generate_capacity_graph_trusted_nodes_bb84(dictionary_bb84=dictionary_bb84)

                self.trusted_nodes_graphs[key].store_capacity_edge_graph_distances(
                    store_file_location=store_file_location,
                    node_types=self.pos_graphs[
                        key].vertex_type,
                    node_data_store_location=node_data_store_location,
                    graph_id=key, distance=self.distances[key])


    def store_capacity_edge_graph_distance_bb84_no_switching(self, dictionary_bb84, store_file_location,
                                                node_data_store_location):
        self.trusted_nodes_graphs_no_switching = {}
        for key in self.pos_graphs.keys():
            try:
                self.trusted_nodes_graphs_no_switching[key] = self.pos_graphs[
                    key].generate_capacity_graph_trusted_nodes_bb84_no_switching(dictionary_bb84=dictionary_bb84)

                self.trusted_nodes_graphs_no_switching[key].store_capacity_edge_graph_distances(
                    store_file_location=store_file_location,
                    node_types=self.pos_graphs[
                        key].vertex_type,
                    node_data_store_location=node_data_store_location,
                    graph_id=key, distance=self.distances[key])
            except:
                self.trusted_node_pos_graph[key] = self.pos_graphs[key].add_n_random_trusted_nodes(n=2, size_graph=
                self.distances[key])
                self.trusted_nodes_graphs_no_switching[key] = self.pos_graphs[
                    key].generate_capacity_graph_trusted_nodes_bb84_no_switching(dictionary_bb84=dictionary_bb84)

                self.trusted_nodes_graphs_no_switching[key].store_capacity_edge_graph_distances(
                    store_file_location=store_file_location,
                    node_types=self.pos_graphs[
                        key].vertex_type,
                    node_data_store_location=node_data_store_location,
                    graph_id=key, distance=self.distances[key])
