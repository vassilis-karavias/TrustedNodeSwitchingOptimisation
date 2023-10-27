import pandas as pd
import numpy as np
from generate_graph import NodeType
from graph_tool.all import *
from utils_graph import get_length
from vector import Vector
from minimum_length import get_minimum_to_any_bob, get_k_minimum_to_any_bob
from capacity_calculation import calculate_capacity_efficient_all_distances, \
    calculate_capacity_for_n_highest_capacities_efficient_corrected, \
    calculate_capacity_for_k_highest_capacities_multiple_paths_per_detector_allowed, \
    calculate_capacities_for_k_highest_connections_bb84
from trusted_nodes_utils import *
from capacity_graph import CapacityGraph
import os
import csv


class Position_Graph(Graph):

    def __init__(self, node_dataframe, edge_dataframe, db_switch=1, connection_uses_Oband = None):
        """
        Creates a graph of nodes connected with edges defined by their positions and distances between each other
        :param node_dataframe: The dataframe that contains the node information  Dataframe(node, type, xcoord, ycoord)
        :param edge_dataframe: The dataframe that contains the edge information  Dataframe(source, target, weight(dist))
        """
        # sort the node dataframe with respect to the nodes
        node_dataframe = node_dataframe.sort_values(by=["node"])
        self.node_dataframe = node_dataframe
        # change the edge info to appropriate form to add into graph directly with inbuilt functions
        edges = edge_dataframe.drop(["weight"], axis=1)
        # create an undirected graph
        g = Graph(directed=False)
        self.edge_dataframe = edge_dataframe
        # add all edges
        vertex_ids = g.add_edge_list(edges.values)
        # add all vertex properties of the graph (x,y) coords vertex_type, label
        self.x_coord = g.new_vertex_property(value_type="double")
        self.y_coord = g.new_vertex_property(value_type="double")
        self.vertex_type = g.new_vertex_property(value_type="object")
        self.label = g.new_vertex_property(value_type="string")
        self.connection_uses_Oband = connection_uses_Oband
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
            self.vertex_type[vertices[vertex]] = nodetypes.values[vertex]
            self.x_coord[vertices[vertex]] = xcoords.values[vertex]
            self.y_coord[vertices[vertex]] = ycoords.values[vertex]
            self.label[vertices[vertex]] = node_dataframe["node"].values[vertex]

        # add the edge property: the distance between the states
        self.lengths_of_connections_orig = g.new_edge_property(value_type="double", vals=distances.values)
        edges = g.get_edges(eprops=[self.lengths_of_connections_orig])
        lengths_of_switched = []
        for edge in edges:
            ## Take a look at the lengths with and without switching
            length_with_switch = edge[2] + 5 * db_switch
            lengths_of_switched.append(length_with_switch)
        self.lengths_with_switch = g.new_edge_property(value_type="double", vals=lengths_of_switched)
        self.g = g
        # self.lengths_of_connections[edge] = get_length(x_coord, y_coord, source_node, target_node).item()
        super().__init__(g=g, directed=False)

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
            if self.vertex_type[vertices[vertex]] == NodeType(2).name:
                # if vertex is a Bob look for shortest path to all nodes - Dijkstra's Algorithm
                dist, pred = dijkstra_search(self.g, weight=self.lengths_with_switch, source=vertices[vertex])
                # add this to the dictionary - will be a dictionary of EdgePropertyMap
                distances_from_bob[vertex] = dist
        return distances_from_bob

    def get_k_shortest_paths_from_bobs(self, k):
        """
        Returns the k shortest path to any detector from all source nodes - i.e. find the k shortest unique (no edge is
        reused once used) paths to each detector from any source node and find the distance of these k paths.
        :param k: Number of paths to search for
        :return: dictionary of path lengths: {key [detector_node, other_node]: [distance of paths for path in k_shortest_paths]}
        """
        # get a list of all vertices
        vertices = self.get_vertices()
        # This will hold the Bobs shortest distance
        distances_from_bob = {}
        for vertex_1 in vertices:
            if self.vertex_type[vertex_1] == NodeType(2).name:
                # vertex_1 is a detector vertex
                for vertex_2_index in range(len(vertices)):
                    # get the node label
                    vertex_2 = vertices[vertex_2_index]
                    # make a copy of the graph
                    copy_graph = Graph(self.g)
                    edges = self.get_edges([self.lengths_with_switch])
                    # to store the lengths of the edges and the map contains the value in the array corresponding to
                    # edge [source, target]
                    lengths = []
                    map = {}
                    n = 0
                    for source, target, length in edges:
                        lengths.append(length)
                        map[source, target] = n
                        map[target, source] = n
                        n += 1
                    # loop over the following until k shortest paths have been explored: set the new edge lengths to the
                    # value of lengths (first loop will be the original graph lengths) - use dijkstra's algorithm to find
                    # the minimum distance path between detector node and other nodes on copy_graph and get an iterator
                    # of all shortest paths - for the shortest path set the edge lengths of this path to 10e9 to ensure
                    # the edges used in the path are not picked again by this algorithm and check that the minimum
                    # distance of the current path from vertex_1 -> vertex_2 < 10e9 (this means there's a unique path
                    # from vertex_1 -> vertex_2) In this case add path distance to the dictionary holding the information
                    # and recalculate edge_lengths (start the loop over) else continue
                    for l in range(k):
                        edge_lengths = copy_graph.new_edge_property(value_type="double")
                        try:
                            copy_edges = copy_graph.edges()
                            while True:
                                edge = copy_edges.next()
                                n = map[int(edge.source()), int(edge.target())]
                                edge_lengths[edge] = lengths[n]
                        except StopIteration:
                            copy_graph.edge_properties["lengths"] = edge_lengths
                            # get the shortest distance and shortest path
                            # distances, pred = dijkstra_search(copy_graph, weight= edge_lengths, source=copy_graph.vertex(vertex_1))

                            path_iterator = all_shortest_paths(copy_graph, source=copy_graph.vertex(vertex_1),
                                                               target=copy_graph.vertex(vertex_2), weights=edge_lengths,
                                                               epsilon=1e-10)
                            for path in path_iterator:
                                dist = 0.0
                                i = None
                                for j in path:
                                    if i == None:
                                        i = j
                                    else:
                                        n = map[i, j]
                                        dist += lengths[n]
                                        lengths[n] = 10e9
                                        i = j
                                if l == 0 and dist < 10e8:
                                    distances_from_bob[vertex_1, vertex_2] = [dist]
                                elif dist < 10e8:
                                    distances_from_bob[vertex_1, vertex_2].append(dist)
                                del copy_graph.edge_properties["lengths"]
                                break
        return distances_from_bob

    def get_k_shortest_distance_of_source_nodes(self, k):
        """
        Returns the k shortest path to any detector from all source nodes - i.e. find the k shortest unique (no edge is
        reused once used) paths to each detector from any source node and find the distance of these k paths. Any path
        from detector to detector is set to np.infty
        :param k:  Number of paths to search for
        :return: dictionary of path lengths: {key [detector_node, other_node]: [distance of paths for path in k_shortest_paths]}
        """
        distances_from_detectors = self.get_k_shortest_paths_from_bobs(k)

        vertices = self.get_vertices()
        true_distances = {}
        for vertex in range(len(vertices)):
            if self.vertex_type[vertices[vertex]] == NodeType(2).name:
                distances_for_source_nodes = {}
                for v in range(len(vertices)):
                    # if not a Bob node then keep same key as before
                    if self.vertex_type[vertices[v]] == NodeType(0).name or self.vertex_type[vertices[v]] == NodeType(
                            3).name:
                        distances_for_source_nodes[vertices[v]] = distances_from_detectors[
                            vertices[vertex], vertices[v]]
                    else:
                        # no key can be created between these two nodes.- both detectors - as such min distance is not
                        # of interest. - set to infty
                        distances_for_source_nodes[vertices[v]] = [np.infty]
                true_distances[vertex] = distances_for_source_nodes
        return true_distances

    def get_k_shortest_distances_for_given_pair(self, true_distances, source_node, target_node):
        """
        Get the shortest distances and distances to detectors of the source node and target node (must both be sources)
        In the form {key detector: {key [path_source, path_target] : distances}}
        :param true_distances: dictionary of path lengths: {key [detector_node, other_node]: [distance of paths for path in k_shortest_paths]}
        :param source_node: label of the source node
        :param target_node: label of the target node (both ints) - or position of them in the array if these are different
        :return:
        """
        if self.vertex_type[source_node] == NodeType(2).name or self.vertex_type[source_node] == NodeType(1).name or \
                self.vertex_type[target_node] == NodeType(2).name or self.vertex_type[target_node] == NodeType(1).name:
            print("Cannot generate key between a Detector Node and other Nodes")
            raise ValueError
        else:
            return get_k_minimum_to_any_bob(source_node, target_node, true_distances)

    def get_k_largest_capacities_for_each_required_connection(self, required_connections, dictionary, k):
        """
        Gets the k largest capacities for each connection in the required connections.
        :param required_connections: A list of required connections of the form [(source, target)]
        :param dictionary: The dictionary that defines the capacities according to length of connections
        :param k: Number of paths to search for
        :return: The capacities of the connections as a list [(source, target, detector, capacity)]
        """
        # get the k shortest distances from every node to every bob
        true_distances = self.get_k_shortest_distance_of_source_nodes(k)
        # get the array of minimum distance for every source node and detector in the form [(source, target, bob_min, distance_array, distance_to_detector)]
        minimum_distance_array = []
        for source_node, target_node in required_connections:
            if (source_node, target_node) in self.connection_uses_Oband:
                true_distances = true_distances * 0.35 / 0.2
            distance_array, distance_to_detector = self.get_k_shortest_distances_for_given_pair(true_distances,
                                                                                                source_node,
                                                                                                target_node)
            bobs_minimum = []
            for l in list(true_distances.keys()):
                bobs_minimum.append(int(self.vertex_properties["name"][l]))
            minimum_distance_array.append(
                (source_node, target_node, bobs_minimum, distance_array, distance_to_detector))
        # get the capacities of for each of the pairs of sources
        capacity, distance_array = calculate_capacity_for_k_highest_capacities_multiple_paths_per_detector_allowed(
            minimum_distance_array, dictionary, k)
        # now need to keep only the k largest capacities for each source-target pair

        return capacity

    # trusted node - trusted node -> use BB84 so search directly for k shortest paths:
    # trusted node - source node -> run get k shortest paths to bob
    # then run get_k_largest_capacities_for_each_required_connection  --> All of these need to be done for very large k
    # to ensure there are no paths that are skipped on the loop.
    # we can then remove the paths that have a capacity of 0.0 (Length of > L_{max}).

    def get_k_shortest_paths_between_two_nodes(self, node_1, node_2, k):
        # get a list of all vertices
        distances = []
        copy_graph = Graph(self.g)
        edges = self.get_edges([self.lengths_with_switch])
        # to store the lengths of the edges and the map contains the value in the array corresponding to
        # edge [source, target]
        lengths = []
        map = {}
        n = 0
        for source, target, length in edges:
            lengths.append(length)
            map[source, target] = n
            map[target, source] = n
            n += 1
        # loop over the following until k shortest paths have been explored: set the new edge lengths to the
        # value of lengths (first loop will be the original graph lengths) - use dijkstra's algorithm to find
        # the minimum distance path between detector node and other nodes on copy_graph and get an iterator
        # of all shortest paths - for the shortest path set the edge lengths of this path to 10e9 to ensure
        # the edges used in the path are not picked again by this algorithm and check that the minimum
        # distance of the current path from vertex_1 -> vertex_2 < 10e9 (this means there's a unique path
        # from vertex_1 -> vertex_2) In this case add path distance to the dictionary holding the information
        # and recalculate edge_lengths (start the loop over) else continue
        for l in range(k):
            if (node_1, node_2) not in self.connection_uses_Oband:
                edge_lengths = copy_graph.new_edge_property(value_type="double")
                try:
                    copy_edges = copy_graph.edges()
                    while True:
                        edge = copy_edges.next()
                        n = map[int(edge.source()), int(edge.target())]
                        edge_lengths[edge] = lengths[n]
                except StopIteration:

                    copy_graph.edge_properties["lengths"] = edge_lengths
                    # get the shortest distance and shortest path
                    distance_investigation, pred = dijkstra_search(copy_graph, weight=edge_lengths,
                                                               source=copy_graph.vertex(node_1))

                    path_iterator = all_shortest_paths(copy_graph, source=copy_graph.vertex(node_1),
                                                   target=copy_graph.vertex(node_2), weights=edge_lengths,
                                                   epsilon=1e-10)
                    for path in path_iterator:
                        i = None
                        dist = 0.0
                        for j in path:
                            if i == None:
                                i = j
                            else:
                                n = map[i, j]
                                dist += lengths[n]
                                lengths[n] = 10e9
                                i = j
                        if l == 0 and dist < 10e8:
                            distances = [dist]
                        elif dist < 10e8:
                            distances.append(dist)
                        del copy_graph.edge_properties["lengths"]
                        break
            else:
                edge_lengths = copy_graph.new_edge_property(value_type="double")
                try:
                    copy_edges = copy_graph.edges()
                    while True:
                        edge = copy_edges.next()
                        n = map[int(edge.source()), int(edge.target())]
                        edge_lengths[edge] = lengths[n] * 0.35/0.2
                except StopIteration:

                    copy_graph.edge_properties["lengths"] = edge_lengths
                    # get the shortest distance and shortest path
                    distance_investigation, pred = dijkstra_search(copy_graph, weight=edge_lengths,
                                                                   source=copy_graph.vertex(node_1))

                    path_iterator = all_shortest_paths(copy_graph, source=copy_graph.vertex(node_1),
                                                       target=copy_graph.vertex(node_2), weights=edge_lengths,
                                                       epsilon=1e-10)
                    for path in path_iterator:
                        i = None
                        dist = 0.0
                        for j in path:
                            if i == None:
                                i = j
                            else:
                                n = map[i, j]
                                dist += lengths[n]
                                lengths[n] = 10e9
                                i = j
                        if l == 0 and dist < 10e8:
                            distances = [dist]
                        elif dist < 10e8:
                            distances.append(dist)
                        del copy_graph.edge_properties["lengths"]
                        break
        return distances

    def get_k_largest_capacities_trusted_nodes(self, required_connections, dictionary, k):
        minimum_distance_array = []
        for node_1, node_2 in required_connections:
            distances = self.get_k_shortest_paths_between_two_nodes(node_1, node_2, k)
            # Add O-band Loss terms: 0.35dB/km vs 0.2dB/km
            if (node_1, node_2) in self.connection_uses_Oband:
                distances = distances * 0.35/0.2
            bobs_minimum = []
            minimum_distance_array.append((node_1, node_2, distances))
            # get the capacities of for each of the pairs of sources
        capacity = calculate_capacities_for_k_highest_connections_bb84(minimum_distance_array, dictionary, k)
        return capacity

    def get_k_largest_capacities_for_capacity_graph_to_store_capacity_constrains(self, dictionary_bb84, dictionary_tf,
                                                                                 k):
        # get a list of all the trusted node vertex connections in the graph - will be needed for trusted node ->
        # trusted node connections.
        required_connections = []
        vertices = self.get_vertices()
        for vertex_1 in range(len(vertices)):
            for vertex_2 in range(vertex_1, len(vertices)):
                if self.vertex_type[vertices[vertex_1]] == NodeType(3).name and self.vertex_type[
                    vertices[vertex_2]] == NodeType(3).name \
                        and vertex_1 != vertex_2:
                    required_connections.append((vertex_1, vertex_2))
        # get the required capacities
        capacities_trusted_nodes = self.get_k_largest_capacities_trusted_nodes(required_connections, dictionary_bb84, k)
        capacities_non_zero = []
        for source, target, bobs, capacity in capacities_trusted_nodes:
            if capacity > 0.0000001:
                capacities_non_zero.append((source, target, bobs, capacity))
        required_connections_trusted_source = []
        for vertex_1 in range(len(vertices)):
            if self.vertex_type[vertices[vertex_1]] == NodeType(3).name:
                for vertex_2 in range(len(vertices)):
                    if self.vertex_type[vertices[vertex_2]] == NodeType(0).name:
                        required_connections_trusted_source.append((vertex_1, vertex_2))
        capacities_tf = self.get_k_largest_capacities_for_each_required_connection(required_connections_trusted_source,
                                                                                   dictionary_tf, k)
        capacities_non_zero.extend(capacities_tf)
        return capacities_non_zero

    def generate_capacity_graph_trusted_nodes_bb84_multipath_between_connections(self, dictionary_tf, dictionary_bb84,
                                                                                 k):
        """
        generate capacity graph with trusted nodes connections only - capacities between trusted nodes are calculated
        using BB84 Decoy protocol
        :param dictionary_tf: The dictionary that defines the capacities according to length of connections [source,
         target, detector, capacity] - TF-QKD
        :param dictionary_bb84: The dictionary that defines the capacities according to length of connections [source,
         target, capacity] - BB84
        :return: The CapacityGraph with only trusted nodes connections
        """
        # get the capacities for trusted nodes connections
        capacities = self.get_k_largest_capacities_for_capacity_graph_to_store_capacity_constrains(
            dictionary_tf=dictionary_tf, dictionary_bb84=dictionary_bb84, k=k)
        # change the data to the form desired by the CapacityGraph input
        restructured_capacities = []
        for source, target, detector, capacity in capacities:
            restructured_capacities.append([source, target, capacity])
        # create the CapacityGraph
        return CapacityGraph(restructured_capacities)

    def get_shortest_distance_between_two_nodes(self, node_1, node_2):
        dist = shortest_distance(g=self.g, source=node_1, target=node_2, weights=self.lengths_with_switch)
        return dist

    def get_shortest_distance_of_source_nodes(self):
        """
        Get the shortest distance of all source and trusted nodes to each of the Bobs. Distances between bobs are set to
        infty
        :return: The distances from each of the Bobs for each source node as a dictionary of EdgePropertyMaps
        """
        # get the minimum distances from all Bobs to all nodes
        distances_from_bob_of_all_nodes = self.get_shortest_distance_from_bobs()
        # iterate over all vertices
        vertices = self.get_vertices()
        # will keep the dictionary of minimum distances from all Bobs to all source nodes
        true_distances = {}
        for vertex in range(len(vertices)):
            # for each bob
            if self.vertex_type[vertices[vertex]] == NodeType(2).name:
                distances_for_source_nodes = self.new_vertex_property(value_type="double")
                for v in range(len(vertices)):
                    # if not a Bob node then keep same key as before
                    if self.vertex_type[vertices[v]] == NodeType(0).name or self.vertex_type[vertices[v]] == NodeType(
                            3).name:
                        distances_for_source_nodes[vertices[v]] = distances_from_bob_of_all_nodes[vertices[vertex]].a[v]
                    else:
                        # no key can be created between these two nodes.- both detectors - as such min distance is not
                        # of interest. - set to infty
                        distances_for_source_nodes[vertices[v]] = np.infty
                true_distances[vertex] = distances_for_source_nodes
        return true_distances

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
            distance_array, distance_to_detector = self.get_shortest_distances_for_given_pair(true_distances,
                                                                                              source_node, target_node)
            bobs_minimum = []
            for k in list(true_distances.keys()):
                bobs_minimum.append(int(self.vertex_properties["name"][k]))
            minimum_distance_array.append(
                (source_node, target_node, bobs_minimum, distance_array, distance_to_detector))
        # get the capacities of for each of the pairs of sources
        capacity, distance_array = calculate_capacity_efficient_all_distances(minimum_distance_array, dictionary)
        return capacity

    def get_n_top_capacities_for_each_required_connection(self, required_connections, dictionary, n):
        """
        Get the n highest capacities for a list of required connections where the capacity values based on distance are
        determined by dictionary
        :param required_connections: A list of required connections of the form [(source, target)]
        :param dictionary: The dictionary that defines the capacities according to length of connections
        :param n: The number of connections to get the minimum capacities for
        :return: The capacity array for the connections required in the form [(source, target, bob_used, capacity)]
        """
        # get the shortest distances from every node to every bob
        true_distances = self.get_shortest_distance_of_source_nodes()
        # get the array of minimum distance for every source node and detector in the form [(source, target, bob_min, distance_array, distance_to_detector)]
        minimum_distance_array = []
        for source_node, target_node in required_connections:
            if (source_node, target_node) in self.connection_uses_Oband:
                true_distances = true_distances * 0.35 / 0.2
            distance_array, distance_to_detector = self.get_shortest_distances_for_given_pair(true_distances,
                                                                                              source_node, target_node)
            bobs_minimum = []
            for k in list(true_distances.keys()):
                bobs_minimum.append(int(self.vertex_properties["name"][k]))
            minimum_distance_array.append(
                (source_node, target_node, bobs_minimum, distance_array, distance_to_detector))
        # get the capacities of for each of the pairs of sources
        capacity, distance_array = calculate_capacity_for_n_highest_capacities_efficient_corrected(
            minimum_distance_array, dictionary, n=n)
        return capacity

    def get_capacities_for_each_required_connection_bb84(self, required_connections, dictionary):
        """
        Get the capacities for a list of required connections where the capacity values based on distance are determined
        by dictionary - for BB84 protcol - should use appropriate dictionary
        :param required_connections: A list of required connections of the form [(source, target)]
        :param dictionary: The dictionary that defines the capacities according to length of connections
        :return: The capacity array for the connections required in the form [(source, target, bob_used - filler to keep
        same form as TF-QKD, capacity)]
        """
        rates = []
        for source_node, target_node in required_connections:
            distance = self.get_shortest_distance_between_two_nodes(source_node, target_node)
            distance_actual = round(distance, 2)
            if distance_actual > 999:
                capacity = 0.0
            else:
                # from the look-up table
                capacity = dictionary["L" + str(distance_actual)]
            rates.append((source_node, target_node, 0, capacity))
        return rates

    def get_capacities_for_every_source_pair(self, dictionary):
        """
        get the capacity for every source pair in the graph
        :param dictionary: The dictionary that defines the capacities according to length of connections
        :return: The capacity array for each connection between source pairs in the graph in the form
         [(source, target, bob_used, capacity)]
        """
        # get a list of all the source vertex connections in the graph
        required_connections = []
        vertices = self.get_vertices()
        for vertex_1 in range(len(vertices)):
            for vertex_2 in range(vertex_1, len(vertices)):
                if self.vertex_type[vertices[vertex_1]] == NodeType(0).name and self.vertex_type[
                    vertices[vertex_2]] == NodeType(0).name \
                        and vertex_1 != vertex_2:
                    required_connections.append((vertex_1, vertex_2))
        # get the required capacities
        return self.get_capacities_for_each_required_connection(required_connections=required_connections,
                                                                dictionary=dictionary)

    def get_n_top_capacities_for_every_source_pair(self, dictionary, n):
        """
        get the n highest capacity for every source pair in the graph
        :param dictionary: The dictionary that defines the capacities according to length of connections
        :param n: The number of connections to get the minimum capacities for
        :return: The capacity array for each connection between source pairs in the graph in the form
         [(source, target, bob_used, capacity)]
        """
        # get a list of all the source vertex connections in the graph
        required_connections = []
        vertices = self.get_vertices()
        for vertex_1 in range(len(vertices)):
            for vertex_2 in range(vertex_1, len(vertices)):
                if self.vertex_type[vertices[vertex_1]] == NodeType(0).name and self.vertex_type[
                    vertices[vertex_2]] == NodeType(0).name \
                        and vertex_1 != vertex_2:
                    required_connections.append((vertex_1, vertex_2))
        # get the required capacities
        return self.get_n_top_capacities_for_each_required_connection(required_connections=required_connections,
                                                                      dictionary=dictionary, n=n)

    def get_k_top_capacities_for_every_source_pair_allowing_multiple_connections_to_detectors(self, dictionary, k):
        """
        get the n highest capacity for every source pair in the graph - TF-QKD system
        :param dictionary: The dictionary that defines the capacities according to length of connections
        :param n: The number of connections to get the minimum capacities for
        :return: The capacity array for each connection between source pairs in the graph in the form
         [(source, target, bob_used, capacity)]
        """
        # get a list of all the source vertex connections in the graph
        required_connections = []
        vertices = self.get_vertices()
        for vertex_1 in range(len(vertices)):
            for vertex_2 in range(vertex_1, len(vertices)):
                if self.vertex_type[vertices[vertex_1]] == NodeType(0).name and self.vertex_type[
                    vertices[vertex_2]] == NodeType(0).name \
                        and vertex_1 != vertex_2:
                    required_connections.append((vertex_1, vertex_2))
        # get the required capacities
        return self.get_k_largest_capacities_for_each_required_connection(required_connections=required_connections,
                                                                          dictionary=dictionary, k=k)

    def get_k_top_capacities_for_every_source_pair_allowing_multiple_paths_bb84(self, dictionary, k):
        """
        get the n highest capacity for every source pair in the graph - BB84 system
        :param dictionary: The dictionary that defines the capacities according to length of connections
        :param n: The number of connections to get the minimum capacities for
        :return: The capacity array for each connection between source pairs in the graph in the form
         [(source, target, bob_used, capacity)]
        """
        # get a list of all the source vertex connections in the graph
        required_connections = []
        vertices = self.get_vertices()
        for vertex_1 in range(len(vertices)):
            for vertex_2 in range(vertex_1, len(vertices)):
                if self.vertex_type[vertices[vertex_1]] == NodeType(0).name and self.vertex_type[
                    vertices[vertex_2]] == NodeType(0).name \
                        and vertex_1 != vertex_2:
                    required_connections.append((vertex_1, vertex_2))
        # get the required capacities
        return self.get_k_largest_capacities_trusted_nodes(required_connections=required_connections,
                                                           dictionary=dictionary, k=k)

    def add_trusted_node(self, capacity_graph):
        """
        add the trusted node to the graph, create new graph with the trusted node
        :param capacity_graph: The capacity graph of the network defining the links that need more capacity and how much
        more capacity is needed: Capacity Graph
        :return: Position_Graph that has the connections of current Position_Graph with added trusted node
        """
        # edges = capacity_graph.get_edges(eprops = [self.capacities])
        ### deal with how to deal with endpoint - empty capacity graph
        # get the vector of the position to place the trusted node
        position_vector = position_of_next_trusted_node(self, capacity_graph)

        # get the largest value of the previous node
        max_prev_node = self.node_dataframe["node"].max()
        # create copy of node information and edge information of current graph.
        node_data = self.node_dataframe.copy()
        edge_data = self.edge_dataframe.copy()
        # add new node - Trusted node NodeType(3) with position at position_vector
        max_index = node_data.index.max(axis=0)
        new_node_dataframe = node_data.append(pd.DataFrame(
            {"node": max_prev_node + 1, "type": NodeType(3).name, "xcoord": position_vector.get_ith_element(0),
             "ycoord": position_vector.get_ith_element(1)}, index=[max_index + 1]))
        # add new connections to all edges
        vertices = self.get_vertices()
        for vertex in vertices:
            posn_of_vertex = Vector(np.array([self.x_coord[vertex], self.y_coord[vertex]]))
            edge_data_indices = edge_data.index
            max_indx = edge_data_indices.max()
            edge_data = edge_data.append(pd.DataFrame({"source": max_prev_node + 1, "target": vertex,
                                                       "weight": (position_vector - posn_of_vertex).magnitude()},
                                                      index=[max_indx]))
        return Position_Graph(node_dataframe=new_node_dataframe, edge_dataframe=edge_data)

    def get_capacities_for_trusted_nodes(self, dictionary):
        """
         get the capacity for every trusted node paired with a source node in the graph (or trusted node to trusted node)
        :param dictionary: The dictionary that defines the capacities according to length of connections
        :return: The capacity array for each connection between source pairs in the graph in the form
         [(source, target, bob_used, capacity)]
        """
        # get a list of all the source vertex connections to trusted nodes in the graph
        required_connections = []
        vertices = self.get_vertices()
        for vertex_1 in range(len(vertices)):
            for vertex_2 in range(len(vertices)):
                if (self.vertex_type[vertices[vertex_1]] == NodeType(3).name and self.vertex_type[
                    vertices[vertex_2]] == NodeType(0).name) \
                        or (self.vertex_type[vertices[vertex_1]] == NodeType(3).name and self.vertex_type[
                    vertices[vertex_2]] == NodeType(3).name and vertex_1 > vertex_2):
                    required_connections.append((vertex_1, vertex_2))
        # get the required capacities
        return self.get_capacities_for_each_required_connection(required_connections=required_connections,
                                                                dictionary=dictionary)

    def get_capacities_for_trusted_nodes_with_bb84(self, dictionary_tf, dictionary_bb84):
        """
        get the capacity for every trusted node paired with a source node in the graph using TF-QKD
        get the capacity between trusted node pairs using Decoy BB84

        :param dictionary_tf: The dictionary that defines the capacities according to length of connections for TF-QKD
        :param dictionary_bb84: The dictionary that defines the capacities according to length of connection for Decoy BB84
        :return: The capacity array for each connection between source pairs in the graph in the form
         [(source, target, bob_used, capacity)]
        """
        # get a list of all the source vertex connections to trusted nodes in the graph
        required_connections_tf = []
        required_connections_bb = []
        vertices = self.get_vertices()
        for vertex_1 in range(len(vertices)):
            for vertex_2 in range(len(vertices)):
                if (self.vertex_type[vertices[vertex_1]] == NodeType(3).name and self.vertex_type[
                    vertices[vertex_2]] == NodeType(0).name):
                    required_connections_tf.append((vertex_1, vertex_2))
                elif (self.vertex_type[vertices[vertex_1]] == NodeType(3).name and self.vertex_type[
                    vertices[vertex_2]] == NodeType(3).name and vertex_1 > vertex_2):
                    required_connections_bb.append((vertex_1, vertex_2))
        capacities_tf = self.get_capacities_for_each_required_connection(required_connections=required_connections_tf,
                                                                         dictionary=dictionary_tf)
        capacities_bb = self.get_capacities_for_each_required_connection_bb84(
            required_connections=required_connections_bb,
            dictionary=dictionary_bb84)
        capacities_tf_bb = self.get_capacities_for_each_required_connection(
            required_connections=required_connections_bb,
            dictionary=dictionary_tf)
        capacities_tf.extend(capacities_bb)
        return capacities_tf

    def get_capacities_for_trusted_nodes_with_bb84_for_all_connections(self, dictionary_bb84):
        """
        get the capacity for every trusted node paired with a source node in the graph using TF-QKD
        get the capacity between all connections using Decoy BB84

        :param dictionary_bb84: The dictionary that defines the capacities according to length of connection for Decoy BB84
        :return: The capacity array for each connection between source pairs in the graph in the form
         [(source, target, bob_used, capacity)]
        """
        # get a list of all the source vertex connections to trusted nodes in the graph
        required_connections_bb = []
        vertices = self.get_vertices()
        for vertex_1 in range(len(vertices)):
            for vertex_2 in range(len(vertices)):
                if (self.vertex_type[vertices[vertex_1]] == NodeType(3).name and self.vertex_type[
                    vertices[vertex_2]] == NodeType(0).name):
                    required_connections_bb.append((vertex_1, vertex_2))
                elif (self.vertex_type[vertices[vertex_1]] == NodeType(3).name and self.vertex_type[
                    vertices[vertex_2]] == NodeType(3).name and vertex_1 > vertex_2):
                    required_connections_bb.append((vertex_1, vertex_2))
        capacities_bb = self.get_capacities_for_each_required_connection_bb84(
            required_connections=required_connections_bb,
            dictionary=dictionary_bb84)
        return capacities_bb

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

    def add_trusted_nodes_at_midpoints(self, p, dist):
        """
        Add trusted nodes at the midpoints of edges with probability p - connected with each other if distance between
        them < dist
        :param p: The probability to add trusted nodes on edge
        :param dist: The distance at which to connect the trusted nodes together
        :return: PositionGraph with trusted nodes added.
        """
        edges = self.get_edges()
        trusted_node_dict = {"node": [], "xcoord": [], "ycoord": []}
        trusted_node_edge_dict = {"source": [], "target": []}
        trusted_node_edge_dict_to_other_nodes = {"source": [], "target": []}
        current_trusted_node = 0
        for edge in edges:
            if (np.random.uniform() < p):
                # add the trusted node with probability p
                source = edge[0]
                target = edge[1]
                x_coord_source, y_coord_source = self.x_coord[source], self.y_coord[source]
                x_coord_target, y_coord_target = self.x_coord[target], self.y_coord[target]
                posn_of_source = Vector(np.array([x_coord_source, y_coord_source]))
                posn_of_target = Vector(np.array([x_coord_target, y_coord_target]))
                midpoint = (posn_of_source + posn_of_target).scalar_mult(0.5)
                xcoord_trusted_node, ycoord_trusted_node = midpoint.get_ith_element(0), midpoint.get_ith_element(1)
                trusted_node_dict["node"].append(current_trusted_node)

                trusted_node_dict["xcoord"].append(xcoord_trusted_node)
                trusted_node_dict["ycoord"].append(ycoord_trusted_node)
                trusted_node_edge_dict_to_other_nodes["source"].append(current_trusted_node)
                trusted_node_edge_dict_to_other_nodes["target"].append(source)
                trusted_node_edge_dict_to_other_nodes["source"].append(current_trusted_node)
                trusted_node_edge_dict_to_other_nodes["target"].append(target)
                current_trusted_node += 1
        for i in range(len(trusted_node_dict["node"])):
            for j in range(i + 1, len(trusted_node_dict["node"])):
                position_of_trusted_node_1 = Vector(
                    np.array([trusted_node_dict["xcoord"][i], trusted_node_dict["ycoord"][i]]))
                position_of_trusted_node_2 = Vector(
                    np.array([trusted_node_dict["xcoord"][j], trusted_node_dict["ycoord"][j]]))
                distance = (position_of_trusted_node_1 - position_of_trusted_node_2).magnitude()
                if distance < dist:
                    trusted_node_edge_dict["source"].append(trusted_node_dict["node"][i])
                    trusted_node_edge_dict["target"].append(trusted_node_dict["node"][j])
        trusted_node_list = pd.DataFrame.from_dict(trusted_node_dict)
        trusted_node_edge_list = pd.DataFrame.from_dict(trusted_node_edge_dict)
        trusted_node_edge_list_to_other_nodes = pd.DataFrame.from_dict(trusted_node_edge_dict_to_other_nodes)
        return self.add_trusted_node_sets(trusted_node_list=trusted_node_list,
                                          trusted_node_edge_list=trusted_node_edge_list,
                                          trusted_node_edge_list_to_other_nodes=trusted_node_edge_list_to_other_nodes)

    def add_trusted_node_sets(self, trusted_node_list, trusted_node_edge_list, trusted_node_edge_list_to_other_nodes):
        """
        Create new Position Graph with added trusted nodes defined by trusted_node_list. The node names can be
        anything - they will be changed - as long as they are consistent. trusted_node_edge_list is the list of edges
        that exists between trusted nodes and trusted_node_edge_list_to_other_nodes is the list of edges from trusted
        nodes to other types of nodes
        :param trusted_node_list: List of node names for trusted nodes: pd dataframe with [node, xcoord, ycoord]
        :param trusted_node_edge_list: List of edges between trusted nodes [source, target]
        :param trusted_node_edge_list_to_other_nodes: List of edges between trusted and other types of nodes
        [source, target]: source is the trusted node and target is the other node type
        :return: PositionGraph with trusted nodes added.
        """
        # edges = capacity_graph.get_edges(eprops = [self.capacities])
        ### deal with how to deal with endpoint - empty capacity graph
        # get the vector of the position to place the trusted node
        # get the largest value of the previous node
        max_prev_node = self.node_dataframe["node"].max()
        # create copy of node information and edge information of current graph.
        node_data = self.node_dataframe.copy()
        edge_data = self.edge_dataframe.copy()
        # add new node - Trusted node NodeType(3) with position at position_vector
        max_index = node_data.index.max(axis=0)
        gc = Graph_Collector()

        node_data["node"] = node_data.apply(lambda x: gc.map_nodes_for_node_set(0, x["node"]), axis=1)
        trusted_node_list["node"] = trusted_node_list.apply(lambda x: gc.map_nodes_for_node_set(1, x["node"]), axis=1)
        trusted_node_list["type"] = NodeType(3).name
        new_node_dataframe = pd.concat([node_data, trusted_node_list])
        trusted_node_edge_list["source"] = trusted_node_edge_list.apply(
            lambda x: gc.map_nodes_for_edges(1, x["source"]), axis=1)
        trusted_node_edge_list["target"] = trusted_node_edge_list.apply(
            lambda x: gc.map_nodes_for_edges(1, x["target"]), axis=1)
        trusted_node_edge_list_to_other_nodes["source"] = trusted_node_edge_list_to_other_nodes.apply(
            lambda x: gc.map_nodes_for_edges(1, x["source"]), axis=1)
        trusted_node_edge_list_to_other_nodes["target"] = trusted_node_edge_list_to_other_nodes.apply(
            lambda x: gc.map_nodes_for_edges(0, x["target"]), axis=1)
        if not trusted_node_edge_list.empty:
            trusted_node_edge_list["weight"] = trusted_node_edge_list.apply(
                lambda x: get_weight(x["source"], x["target"], new_node_dataframe), axis=1)
        if not trusted_node_edge_list_to_other_nodes.empty:
            trusted_node_edge_list_to_other_nodes["weight"] = trusted_node_edge_list_to_other_nodes.apply(
                lambda x: get_weight(x["source"], x["target"], new_node_dataframe), axis=1)
        edge_data = pd.concat([edge_data, trusted_node_edge_list, trusted_node_edge_list_to_other_nodes])
        return Position_Graph(node_dataframe=new_node_dataframe, edge_dataframe=edge_data)

    def get_key_for_n_capacities(self, capacities):
        capacities_for_pairs = {}
        for source, target, bob_used, capacity in capacities:
            if (source, target) in capacities_for_pairs.keys():
                capacities_for_pairs[(source, target)].append(capacity)
                capacities_for_pairs[(source, target)].sort(reverse=True)
            else:
                capacities_for_pairs[(source, target)] = [capacity]
        return capacities_for_pairs

    def store_position_graph(self, node_data_store_location, edge_data_store_location, graph_id=0):
        # storage distances for plotting purposes - use true distances not adjusted distances to accomodate switch loss
        edges = self.g.get_edges(eprops=[self.lengths_of_connections_orig])
        dictionaries = []
        dictionary_fieldnames = ["ID", "source", "target", "distance"]
        for edge in range(len(edges)):
            source = edges[edge][0]
            target = edges[edge][1]
            distance = edges[edge][2]
            dictionaries.append(
                {"ID": graph_id, "source": source, "target": target, "distance": distance})
        nodes = self.g.get_vertices(vprops=[self.x_coord, self.y_coord])
        dictionary_fieldnames_nodes = ["ID", "node", "xcoord", "ycoord", "type"]
        dict_nodes = []
        for node in range(len(nodes)):
            node_label = nodes[node][0]
            xcoord = nodes[node][1]
            ycoord = nodes[node][2]
            type = self.vertex_type[node]
            dict_nodes.append({"ID": graph_id, "node": node_label, "xcoord": xcoord, "ycoord": ycoord, "type": type})

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

    def store_n_k_for_n_state_tfqkd(self, dictionary, n, c_min, store_file_location, graph_id=0,
                                    allow_multiple_connections_per_bob=True):
        """
                Finds and stores the nk = number of paths needed for each connection in the graph using TF-QKD underlying network
                :param dictionary: BB84 capacity dictionary
                :param n: number of paths needed with c >= c_min
                :param c_min: The minimum allowed capacity
                :param store_file_location: File name to store the N_k results
                :param graph_id: current graph id
                :param allow_multiple_connections_per_bob: whether to allow multiple paths to each detector location allowed
                """
        if allow_multiple_connections_per_bob:
            capacities = self.get_k_top_capacities_for_every_source_pair_allowing_multiple_connections_to_detectors(
                dictionary, n)
        else:
            capacities = self.get_n_top_capacities_for_every_source_pair(dictionary, n)
        capacities_for_pairs = self.get_key_for_n_capacities(capacities)
        key_capacities = []
        for key in capacities_for_pairs.keys():
            capacity = capacities_for_pairs[key]
            for i in range(len(capacity)):
                if isinstance(c_min, dict):
                    if capacity[i] < c_min[key]:
                        key_capacities.append({"ID": graph_id, "source": key[0], "target": key[1], "N_k": n - i})
                        break
                    elif i == len(capacity) - 1 and n > len(capacity):
                        key_capacities.append(
                            {"ID": graph_id, "source": key[0], "target": key[1], "N_k": max(n - len(capacity), 0)})
                else:
                    if capacity[i] < c_min:
                        key_capacities.append({"ID": graph_id, "source": key[0], "target": key[1], "N_k": n - i})
                        break
                    elif i == len(capacity) - 1 and n > len(capacity):
                        key_capacities.append(
                            {"ID": graph_id, "source": key[0], "target": key[1], "N_k": max(n - len(capacity), 0)})
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

    def store_n_k_for_n_state_bb84_used(self, dictionary, n, c_min, store_file_location, graph_id=0):
        """
        Finds and stores the nk = number of paths needed for each connection in the graph using BB84 underlying network
        :param dictionary: BB84 capacity dictionary
        :param n: number of paths needed with c >= c_min
        :param c_min: The minimum allowed capacity
        :param store_file_location: File name to store the N_k results
        :param graph_id: current graph id
        """
        capacities = self.get_k_top_capacities_for_every_source_pair_allowing_multiple_paths_bb84(dictionary, n)
        capacities_for_pairs = self.get_key_for_n_capacities(capacities)
        key_capacities = []
        for key in capacities_for_pairs.keys():
            capacity = capacities_for_pairs[key]
            for i in range(len(capacity)):
                if isinstance(c_min, dict):
                    if capacity[i] < c_min[key]:
                        key_capacities.append({"ID": graph_id, "source": key[0], "target": key[1], "N_k": n - i})
                        break
                    elif i == len(capacity) - 1 and n > len(capacity):
                        key_capacities.append(
                            {"ID": graph_id, "source": key[0], "target": key[1], "N_k": max(n - len(capacity), 0)})
                else:
                    if capacity[i] < c_min:
                        key_capacities.append({"ID": graph_id, "source": key[0], "target": key[1], "N_k": n - i})
                        break
                    elif i == len(capacity) - 1 and n > len(capacity):
                        key_capacities.append(
                            {"ID": graph_id, "source": key[0], "target": key[1], "N_k": max(n - len(capacity), 0)})
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

    def generate_capacity_graph_sources(self, dictionary):
        """
        generate capacity graph with source nodes connections only
        :param dictionary: The dictionary that defines the capacities according to length of connections
        :return: The CapacityGraph with only source nodes connections
        """
        # get the capacities for source nodes connections
        capacities = self.get_capacities_for_every_source_pair(dictionary)
        # change the data to the form desired by the CapacityGraph input
        restructured_capacities = []
        for source, target, detector, capacity in capacities:
            restructured_capacities.append([source, target, capacity])
        # create the CapacityGraph
        return CapacityGraph(restructured_capacities)

    def generate_capacity_graph_trusted_nodes(self, dictionary):
        """
        generate capacity graph with trusted nodes connections only
        :param dictionary: The dictionary that defines the capacities according to length of connections [source, target, detector, capacity]
        :return: The CapacityGraph with only trusted nodes connections
        """
        # get the capacities for trusted nodes connections
        capacities = self.get_capacities_for_trusted_nodes(dictionary)
        # change the data to the form desired by the CapacityGraph input
        restructured_capacities = []
        for source, target, detector, capacity in capacities:
            restructured_capacities.append([source, target, capacity])
        # create the CapacityGraph
        return CapacityGraph(restructured_capacities)

    def generate_capacity_graph_trusted_nodes_bb84(self, dictionary_tf, dictionary_bb84):
        """
        generate capacity graph with trusted nodes connections and connections between trusted and untrusted nodes
         - capacities between trusted nodes are calculated using BB84 Decoy protocol, between trusted and
         untrusted nodes the connections are calculated using TF-QKD
        :param dictionary_tf: The dictionary that defines the capacities according to length of connections [source,
         target, detector, capacity] - TF-QKD
        :param dictionary_bb84: The dictionary that defines the capacities according to length of connections [source,
         target, capacity] - BB84
        :return: The CapacityGraph with only trusted nodes connections
        """
        # get the capacities for trusted nodes connections
        capacities = self.get_capacities_for_trusted_nodes_with_bb84(dictionary_tf=dictionary_tf,
                                                                     dictionary_bb84=dictionary_bb84)
        # change the data to the form desired by the CapacityGraph input
        restructured_capacities = []
        for source, target, detector, capacity in capacities:
            restructured_capacities.append([source, target, capacity])
        # create the CapacityGraph
        return CapacityGraph(restructured_capacities)

    def generate_capacity_graph_trusted_nodes_bb84_full_graph(self, dictionary_bb84):
        """
        generate capacity graph with trusted nodes connections and connections between trusted and untrusted nodes
         - capacities between trusted nodes are calculated using BB84 Decoy protocol, between trusted and
         untrusted nodes the connections are calculated using TF-QKD- capacities between connections are calculated
        using BB84 Decoy protocol
        :param dictionary_bb84: The dictionary that defines the capacities according to length of connections [source,
         target, capacity] - BB84
        :return: The CapacityGraph with only trusted nodes connections
        """
        # get the capacities for trusted nodes connections
        capacities = self.get_capacities_for_trusted_nodes_with_bb84_for_all_connections(
            dictionary_bb84=dictionary_bb84)
        # change the data to the form desired by the CapacityGraph input
        restructured_capacities = []
        for source, target, detector, capacity in capacities:
            restructured_capacities.append([source, target, capacity])
        # create the CapacityGraph
        return CapacityGraph(restructured_capacities)


class Position_Graph_Set():

    def import_graphs(self, import_path_nodes, import_path_edges, db_switch = 1):
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
        self.pos_graphs = {}
        for id in possible_ids:
            node_data_id = node_data[node_data["ID"] == id].drop(["ID"], axis=1)
            edge_data_id = edge_data[edge_data["ID"] == id].drop(["ID"], axis=1)
            self.pos_graphs[id] = Position_Graph(node_dataframe=node_data_id, edge_dataframe=edge_data_id, db_switch = db_switch)

    def add_trusted_nodes_to_graphs(self, p=0.4, dist=40):
        self.trusted_node_pos_graph = {}
        for key in self.pos_graphs.keys():
            self.trusted_node_pos_graph[key] = self.pos_graphs[key].add_trusted_nodes_at_midpoints(p, dist)

    def store_n_k_for_n_state_tfqkd(self, dictionary, n=2, c_min=10000, store_file_location="cap_needed",
                                    allow_multiple_connections_per_bob=True):
        for key in self.pos_graphs.keys():
            self.pos_graphs[key].store_n_k_for_n_state_tfqkd(dictionary, n, c_min, store_file_location, graph_id=key,
                                                             allow_multiple_connections_per_bob=allow_multiple_connections_per_bob)

    def store_n_k_for_n_state_bb84(self, dictionary, n=2, c_min=10000, store_file_location="cap_needed"):
        for key in self.pos_graphs.keys():
            self.pos_graphs[key].store_n_k_for_n_state_bb84_used(dictionary, n, c_min, store_file_location,
                                                                 graph_id=key)

    def store_required_capacity(self, dictionary, cap_min=10000, store_file_location="cap_needed"):
        self.capacity_graphs = {}
        self.cap_min_graphs = {}
        for key in self.pos_graphs.keys():
            self.capacity_graphs[key] = self.pos_graphs[key].generate_capacity_graph_sources(dictionary)
            self.cap_min_graphs[key] = self.capacity_graphs[key].generate_needed_capacity_graph(cap_min=cap_min)
            self.cap_min_graphs[key].store_capacity_edge_graph(store_file_location, graph_id=key)

    def store_required_capacity_distance(self, dictionary, distances, cap_min=10000, store_file_location="cap_needed"):
        self.capacity_graphs = {}
        self.cap_min_graphs = {}
        for key in self.pos_graphs.keys():
            self.capacity_graphs[key] = self.pos_graphs[key].generate_capacity_graph_sources(dictionary)
            self.cap_min_graphs[key] = self.capacity_graphs[key].generate_needed_capacity_graph(cap_min=cap_min)
            self.cap_min_graphs[key].store_capacity_edge_graph_distances(store_file_location, distance=distances[key],
                                                                         graph_id=key)

    def store_capacity_edge_graph(self, dictionary_tf, dictionary_bb84, store_file_location, node_data_store_location):
        self.trusted_nodes_graphs = {}
        for key in self.pos_graphs.keys():
            self.trusted_nodes_graphs[key] = self.trusted_node_pos_graph[
                key].generate_capacity_graph_trusted_nodes_bb84(dictionary_tf=dictionary_tf,
                                                                dictionary_bb84=dictionary_bb84)
            self.trusted_nodes_graphs[key].store_capacity_edge_graph(store_file_location=store_file_location,
                                                                     node_types=self.trusted_node_pos_graph[
                                                                         key].vertex_type,
                                                                     node_data_store_location=node_data_store_location,
                                                                     graph_id=key)

    def store_capacity_edge_graph_distance_tfqkd(self, dictionary_tf, dictionary_bb84, store_file_location,
                                                 node_data_store_location, distances):
        self.trusted_nodes_graphs = {}
        for key in self.pos_graphs.keys():
            self.trusted_nodes_graphs[key] = self.trusted_node_pos_graph[
                key].generate_capacity_graph_trusted_nodes_bb84(dictionary_tf=dictionary_tf,
                                                                dictionary_bb84=dictionary_bb84)
            self.trusted_nodes_graphs[key].store_capacity_edge_graph_distances(store_file_location=store_file_location,
                                                                               node_types=self.trusted_node_pos_graph[
                                                                                   key].vertex_type,
                                                                               node_data_store_location=node_data_store_location,
                                                                               graph_id=key, distance=distances[key])

    def store_capacity_edge_graph_distance_bb84(self, dictionary_bb84, store_file_location,
                                                node_data_store_location):
        self.trusted_nodes_graphs = {}
        for key in self.pos_graphs.keys():
            try:
                self.trusted_nodes_graphs[key] = self.trusted_node_pos_graph[
                    key].generate_capacity_graph_trusted_nodes_bb84_full_graph(dictionary_bb84=dictionary_bb84)

                self.trusted_nodes_graphs[key].store_capacity_edge_graph_distances(
                    store_file_location=store_file_location,
                    node_types=self.trusted_node_pos_graph[
                        key].vertex_type,
                    node_data_store_location=node_data_store_location,
                    graph_id=key, distance=self.distances[key])
            except:
                self.trusted_node_pos_graph[key] = self.pos_graphs[key].add_n_random_trusted_nodes(n=2, size_graph=
                self.distances[key])
                self.trusted_nodes_graphs[key] = self.trusted_node_pos_graph[
                    key].generate_capacity_graph_trusted_nodes_bb84_full_graph(dictionary_bb84=dictionary_bb84)

                self.trusted_nodes_graphs[key].store_capacity_edge_graph_distances(
                    store_file_location=store_file_location,
                    node_types=self.trusted_node_pos_graph[
                        key].vertex_type,
                    node_data_store_location=node_data_store_location,
                    graph_id=key, distance=self.distances[key])

    def store_capacity_edge_graph_multiedge_graph_distance(self, dictionary_tf, dictionary_bb84, store_file_location,
                                                           node_data_store_location, distances, k=4):
        self.trusted_nodes_graphs = {}
        for key in self.pos_graphs.keys():
            self.trusted_nodes_graphs[key] = self.trusted_node_pos_graph[
                key].generate_capacity_graph_trusted_nodes_bb84_multipath_between_connections(self, dictionary_tf,
                                                                                              dictionary_bb84, k=k)
            self.trusted_nodes_graphs[key].store_capacity_edge_graph_distances(store_file_location=store_file_location,
                                                                               node_types=self.trusted_node_pos_graph[
                                                                                   key].vertex_type,
                                                                               node_data_store_location=node_data_store_location,
                                                                               graph_id=key, distance=distances[key])


class Position_Graph_Set_Distances():

    def import_graphs(self, import_path_nodes, import_path_edges, db_switch = 1):
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
        self.pos_graphs = {}
        self.distances = {}
        for id in possible_ids:
            self.distances[id] = edge_data[edge_data["ID"] == id]["distance"][
                edge_data[edge_data["ID"] == id]["distance"].keys()[0]]
            node_data_id = node_data[node_data["ID"] == id].drop(["ID"], axis=1)
            edge_data_id = edge_data[edge_data["ID"] == id].drop(["ID", "distance"], axis=1)
            self.pos_graphs[id] = Position_Graph(node_dataframe=node_data_id, edge_dataframe=edge_data_id, db_switch = db_switch)

    def add_trusted_nodes_to_graphs(self, p=0.4, dist=40):
        self.trusted_node_pos_graph = {}
        for key in self.pos_graphs.keys():
            self.trusted_node_pos_graph[key] = self.pos_graphs[key].add_trusted_nodes_at_midpoints(p, dist)

    def add_random_trusted_nodes_to_graphs(self, n):
        self.trusted_node_pos_graph = {}
        for key in self.pos_graphs.keys():
            self.trusted_node_pos_graph[key] = self.pos_graphs[key].add_n_random_trusted_nodes(n=n, size_graph=
            self.distances[key])

    def store_n_k_for_n_state_tfqkd(self, dictionary, n=2, c_min=10000, store_file_location="cap_needed",
                                    allow_multiple_connections_per_bob=True):
        for key in self.pos_graphs.keys():
            self.pos_graphs[key].store_n_k_for_n_state_tfqkd(dictionary, n, c_min, store_file_location, graph_id=key,
                                                             allow_multiple_connections_per_bob=allow_multiple_connections_per_bob)

    def store_n_k_for_n_state_bb84(self, dictionary, n=2, c_min=10000, store_file_location="cap_needed"):
        for key in self.pos_graphs.keys():
            self.pos_graphs[key].store_n_k_for_n_state_bb84_used(dictionary, n, c_min, store_file_location,
                                                                 graph_id=key)

    def store_required_capacity(self, dictionary, cap_min=10000, store_file_location="cap_needed"):
        self.capacity_graphs = {}
        self.cap_min_graphs = {}
        for key in self.pos_graphs.keys():
            self.capacity_graphs[key] = self.pos_graphs[key].generate_capacity_graph_sources(dictionary)
            self.cap_min_graphs[key] = self.capacity_graphs[key].generate_needed_capacity_graph(cap_min=cap_min)
            self.cap_min_graphs[key].store_capacity_edge_graph(store_file_location, graph_id=key)

    def store_required_capacity_distance(self, dictionary, cap_min=10000, store_file_location="cap_needed"):
        self.capacity_graphs = {}
        self.cap_min_graphs = {}
        for key in self.pos_graphs.keys():
            self.capacity_graphs[key] = self.pos_graphs[key].generate_capacity_graph_sources(dictionary)
            self.cap_min_graphs[key] = self.capacity_graphs[key].generate_needed_capacity_graph(cap_min=cap_min)
            try:
                self.cap_min_graphs[key].store_capacity_edge_graph_distances(store_file_location,
                                                                             distance=self.distances[key],
                                                                             graph_id=key)
            except:
                continue

    def store_position_graph(self, node_data_store_location, edge_data_store_location):
        for key in self.pos_graphs.keys():
            self.trusted_node_pos_graph[key].store_position_graph(node_data_store_location=node_data_store_location,
                                                                  edge_data_store_location=edge_data_store_location,
                                                                  graph_id=key)

    def store_capacity_edge_graph(self, dictionary_tf, dictionary_bb84, store_file_location, node_data_store_location):
        self.trusted_nodes_graphs = {}
        for key in self.pos_graphs.keys():
            self.trusted_nodes_graphs[key] = self.trusted_node_pos_graph[
                key].generate_capacity_graph_trusted_nodes_bb84(dictionary_tf=dictionary_tf,
                                                                dictionary_bb84=dictionary_bb84)
            self.trusted_nodes_graphs[key].store_capacity_edge_graph(store_file_location=store_file_location,
                                                                     node_types=self.trusted_node_pos_graph[
                                                                         key].vertex_type,
                                                                     node_data_store_location=node_data_store_location,
                                                                     graph_id=key)

    def store_capacity_edge_graph_distance_bb84(self, dictionary_bb84, store_file_location,
                                                node_data_store_location):
        self.trusted_nodes_graphs = {}
        for key in self.pos_graphs.keys():
            try:
                self.trusted_nodes_graphs[key] = self.trusted_node_pos_graph[
                    key].generate_capacity_graph_trusted_nodes_bb84_full_graph(dictionary_bb84=dictionary_bb84)

                self.trusted_nodes_graphs[key].store_capacity_edge_graph_distances(
                    store_file_location=store_file_location,
                    node_types=self.trusted_node_pos_graph[
                        key].vertex_type,
                    node_data_store_location=node_data_store_location,
                    graph_id=key, distance=self.distances[key])
            except:
                self.trusted_node_pos_graph[key] = self.pos_graphs[key].add_n_random_trusted_nodes(n=2, size_graph=
                self.distances[key])
                self.trusted_nodes_graphs[key] = self.trusted_node_pos_graph[
                    key].generate_capacity_graph_trusted_nodes_bb84_full_graph(dictionary_bb84=dictionary_bb84)

                self.trusted_nodes_graphs[key].store_capacity_edge_graph_distances(
                    store_file_location=store_file_location,
                    node_types=self.trusted_node_pos_graph[
                        key].vertex_type,
                    node_data_store_location=node_data_store_location,
                    graph_id=key, distance=self.distances[key])

    def store_capacity_edge_graph_distance_tfqkd(self, dictionary_tf, dictionary_bb84, store_file_location,
                                                 node_data_store_location):
        self.trusted_nodes_graphs = {}
        for key in self.pos_graphs.keys():
            try:
                self.trusted_nodes_graphs[key] = self.trusted_node_pos_graph[
                    key].generate_capacity_graph_trusted_nodes_bb84(dictionary_tf=dictionary_tf,
                                                                    dictionary_bb84=dictionary_bb84)

                self.trusted_nodes_graphs[key].store_capacity_edge_graph_distances(
                    store_file_location=store_file_location,
                    node_types=self.trusted_node_pos_graph[
                        key].vertex_type,
                    node_data_store_location=node_data_store_location,
                    graph_id=key, distance=self.distances[key])
            except:
                self.trusted_node_pos_graph[key] = self.pos_graphs[key].add_n_random_trusted_nodes(n=2, size_graph=
                self.distances[key])
                self.trusted_nodes_graphs[key] = self.trusted_node_pos_graph[
                    key].generate_capacity_graph_trusted_nodes_bb84(dictionary_tf=dictionary_tf,
                                                                    dictionary_bb84=dictionary_bb84)

                self.trusted_nodes_graphs[key].store_capacity_edge_graph_distances(
                    store_file_location=store_file_location,
                    node_types=self.trusted_node_pos_graph[
                        key].vertex_type,
                    node_data_store_location=node_data_store_location,
                    graph_id=key, distance=self.distances[key])

    def store_capacity_edge_graph_multiedge_graph_distance(self, dictionary_tf, dictionary_bb84, store_file_location,
                                                           node_data_store_location, k=4):
        self.trusted_nodes_graphs = {}
        for key in self.pos_graphs.keys():
            try:
                self.trusted_nodes_graphs[key] = self.trusted_node_pos_graph[
                    key].generate_capacity_graph_trusted_nodes_bb84_multipath_between_connections(
                    dictionary_tf=dictionary_tf,
                    dictionary_bb84=dictionary_bb84, k=k)

                self.trusted_nodes_graphs[key].store_capacity_edge_graph_distances(
                    store_file_location=store_file_location,
                    node_types=self.trusted_node_pos_graph[
                        key].vertex_type,
                    node_data_store_location=node_data_store_location,
                    graph_id=key, distance=self.distances[key])
            except:
                self.trusted_node_pos_graph[key] = self.pos_graphs[key].add_n_random_trusted_nodes(n=2, size_graph=
                self.distances[key])
                self.trusted_nodes_graphs[key] = self.trusted_node_pos_graph[
                    key].generate_capacity_graph_trusted_nodes_bb84_multipath_between_connections(
                    dictionary_tf=dictionary_tf,
                    dictionary_bb84=dictionary_bb84, k=k)

                self.trusted_nodes_graphs[key].store_capacity_edge_graph_distances(
                    store_file_location=store_file_location,
                    node_types=self.trusted_node_pos_graph[
                        key].vertex_type,
                    node_data_store_location=node_data_store_location,
                    graph_id=key, distance=self.distances[key])
