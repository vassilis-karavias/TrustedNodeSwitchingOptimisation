from graph_tool.all import *
from utils_graph import *
from enum import Enum, unique
import numpy as np
import copy
import math

@unique
class NodeType(Enum):
    """
    Enum to keep track of each node property in the network - S = 0 for source node, C=1 for Bob node without a Bob
    B=2 for Bob node with a Bob. T = 3 is for Trusted nodes - used in Trusted node analysis, I= 4 is for imaginary
    nodes that are just connections
    """
    S = 0
    C = 1
    B = 2
    T = 3
    I = 4


class VisitorExample(DijkstraVisitor):

    def __init__(self, name, vertex_type):
        """
        Visitor for the Dijkstra search to see what is going on when the search is happening - not useful, just for
        checking
        """
        self.name = name
        self.vertex_type = vertex_type

    # def discover_vertex(self, u):
    #     print("-->", self.name[u], "has been discovered!")

    # def examine_edge(self, e):
        # print("edge (%s, %s) has been examined..." % \
        #       (self.name[e.source()], self.name[e.target()]))

    # def edge_relaxed(self, e):
    #     print("edge (%s, %s) has been relaxed..." % \
    #           (self.name[e.source()], self.name[e.target()]))




class Network_Setup(Graph):

    def __init__(self, n, nodetypes, xcoords, ycoords, node_names, fully_connected = True, p =0.8, geometric_graph = False, no_of_connected_nodes = 3.5, box_size = 100, dbswitch  = 1, hubspoke = False, db_switch = 1):
        """
        This class is the network setup- keeps track of the network topology, all of the nodes, distances etc
        :param n: The number of nodes in the network: int
        :param nodetypes: Array of the types of the nodes 0 == Source, 1 == Possible bob with no bob, 2 == Possible bob
        with bob: list(n)
        :param xcoords: The xcoordinates of the nodes: list(n)
        :param ycoords: The ycoordinates of the nodes: list(n)
        :param fully_connected: whether to use fully connected graph
        :param p: the network connectivity
        :param node_names: The names of the nodes: list(n)
        :param geometric_graph: True to use geometric graph topology: boolean - default = False
        :param no_of_connected_nodes: no. of nodes connected on average default 3.5
        :param box_size: size of the box: default = 100
        :param hubspoke: whether to use hub and spoke - in this case the nodetypes must be in order: all detector node
        positions first then sources
        """
        # generate the graph with n nodes - here we generate a fully connected graph - this will be changed in future
        # g = generate_random_graph(n, n *(n-1)/2 - 2)
        if not geometric_graph:
            if fully_connected:
                g = generate_fully_connected_graph(n)
            else:
                g  = generate_graph_with_random_connections(n=n, p = p)
            # add vertex properties for the type of vertex, the coordinates and labels
            self.vertex_type = g.new_vertex_property(value_type = "object")
            x_coord = g.new_vertex_property(value_type = "double")
            y_coord = g.new_vertex_property(value_type = "double")
            self.label = g.new_vertex_property(value_type = "string")
            self.x_coord = x_coord
            self.y_coord = y_coord
            # get a list of all vertices
            vertices = g.get_vertices()
            # add these propeties as internal to the graph
            g.vertex_properties["name"] = self.label
            g.vertex_properties["node_type"] = self.vertex_type
            # set up the positions in the Network that Bob's and Alices can be placed.
            # S = only Alice, C = only Bob, B = Node C occupied By Bob
            # also set up coordinates of the vertices and then names of the vertices
            for vertex in range(len(vertices)):
                self.vertex_type[vertices[vertex]] = NodeType(nodetypes[vertex]).name
                x_coord[vertices[vertex]] = xcoords[vertex]
                y_coord[vertices[vertex]] = ycoords[vertex]
                self.label[vertices[vertex]] = node_names[vertex]
            # self.lengths_of_connections = g.new_edge_property(value_type = "double", vals = None)
            # set up the length of each of the connections in the network as an edge propetry
            edges = []
            edges_rounded = []
            for edge in g.iter_edges():
                # get the source and end node of each edge
                source_node = edge[0]
                target_node = edge[1]
                # get the length of the connection
                length_of_connection = get_length(x_coord, y_coord, source_node, target_node).item()
                length_rounded = int(round(length_of_connection))
                edges_rounded.append(length_rounded)
                edges.append(length_of_connection)
            # add the length as an edge property
            self.lengths_of_connections = g.new_edge_property(value_type = "double", vals = edges)
            self.lengths_rounded = g.new_edge_property(value_type = "int", vals = edges_rounded)
            # g.add_edge_list(edges, eprops = [self.lengths_of_connections])
            # add this as an internal property of the graph
            g.edge_properties["length_of_connections"] = self.lengths_of_connections
            self.g = g
        elif not hubspoke:
            g, lengths_of_connections, lengths_rounded, x_coords, y_coords, lengths_with_switch = get_graph_with_connections_to_nearest_nodes(no_of_conns_av = no_of_connected_nodes, xcoords = xcoords, ycoords = ycoords, box_size= box_size, db_switch = db_switch)
            self.vertex_type = g.new_vertex_property(value_type = "object")
            self.label = g.new_vertex_property(value_type = "string")
            self.x_coord = x_coords
            self.y_coord = y_coords
            # get a list of all vertices
            vertices = g.get_vertices()
            # add these propeties as internal to the graph
            g.vertex_properties["name"] = self.label
            g.vertex_properties["node_type"] = self.vertex_type
            # set up the positions in the Network that Bob's and Alices can be placed.
            # S = only Alice, C = only Bob, B = Node C occupied By Bob
            # also set up coordinates of the vertices and then names of the vertices
            for vertex in range(len(vertices)):
                self.vertex_type[vertices[vertex]] = NodeType(nodetypes[vertex]).name
                self.label[vertices[vertex]] = node_names[vertex]
            self.lengths_of_connections = lengths_of_connections
            self.lengths_rounded = lengths_rounded
            g.edge_properties["length_of_connections"] = self.lengths_of_connections
            self.lengths_with_switch = lengths_with_switch
            g.edge_properties["length_with_switch"] = self.lengths_with_switch

            self.g = g
        else:
            # hubspoke
            g, lengths_of_connections, lengths_rounded, x_coords, y_coords, lengths_with_switch = generate_hub_and_spoke_network(xcoords, ycoords, nodetypes, box_size, db_switch = db_switch)
            self.vertex_type = g.new_vertex_property(value_type="object")
            self.label = g.new_vertex_property(value_type="string")
            self.x_coord = x_coords
            self.y_coord = y_coords
            # get a list of all vertices
            vertices = g.get_vertices()
            # add these propeties as internal to the graph
            g.vertex_properties["name"] = self.label
            g.vertex_properties["node_type"] = self.vertex_type
            # set up the positions in the Network that Bob's and Alices can be placed.
            # S = only Alice, C = only Bob, B = Node C occupied By Bob
            # also set up coordinates of the vertices and then names of the vertices
            for vertex in range(len(vertices)):
                self.vertex_type[vertices[vertex]] = NodeType(nodetypes[vertex]).name
                self.label[vertices[vertex]] = node_names[vertex]
            self.lengths_of_connections = lengths_of_connections
            self.lengths_rounded = lengths_rounded
            g.edge_properties["length_of_connections"] = self.lengths_of_connections
            self.lengths_with_switch = lengths_with_switch
            g.edge_properties["length_with_switch"] = self.lengths_with_switch

            self.g = g
        self.dbswitch = dbswitch
        # self.lengths_of_connections[edge] = get_length(x_coord, y_coord, source_node, target_node).item()
        super().__init__(g = g, directed = False)


    def position_bobs(self, nodeposns):
        """
        Move the position of Bobs to the values given: 1 in nodeposns is no Bob, 2 is place bob
        :param nodeposns: The array of the reordered Bobs: list(total no. of nodes in C)
        """
        # get a list of the vertices of the array
        vertices = self.get_vertices()
        # keeps track of the position in the nodeposns array
        i =0
        # iterate over every vertex
        for vertex in range(len(vertices)):
            # if vertex is a place without Bob and needs to be a Bob then change the node to have a Bob -
            if self.vertex_type[vertices[vertex]] == NodeType(1).name:
                if nodeposns[i] == 2:
                    self.vertex_type[vertices[vertex]] = NodeType(2).name
                i += 1
            # if vertex is a place with Bob and no Bob should be in location then remove Bob
            elif self.vertex_type[vertices[vertex]] == NodeType(2).name:
                if nodeposns[i] == 1:
                    self.vertex_type[vertices[vertex]] = NodeType(1).name
                i += 1

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
                dist, pred = dijkstra_search(self.g, weight = self.lengths_of_connections, source = vertices[vertex],
                                             visitor = VisitorExample(name = self.label, vertex_type = self.vertex_type))
                # add this to the dictionary - will be a dictionary of EdgePropertyMap
                distances_from_bob[vertex]  = dist
        return distances_from_bob

    def get_shortest_distance_from_bobs_with_switches(self):
        """
        Get the shortest distance from each of the Bobs for each of the vertices - including other Bob vertices
        includes a 2.5dB loss due to switches
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
                dist, pred = dijkstra_search(self.g, weight = self.lengths_with_switch, source = vertices[vertex],
                                             visitor = VisitorExample(name = self.label, vertex_type = self.vertex_type))

                # add this to the dictionary - will be a dictionary of EdgePropertyMap
                distances_from_bob[vertex]  = dist
        return distances_from_bob

    def get_shortest_distance_of_source_nodes(self):
        """
        Get the shortest distance of all source nodes to each of the Bobs. Distances between bobs are set to infty
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
                distances_for_source_nodes = self.new_vertex_property(value_type = "double")
                for v in range(len(vertices)):
                    # if not a Bob node then keep same key as before
                    if self.vertex_type[vertices[v]] == NodeType(0).name:
                        distances_for_source_nodes[vertices[v]] = distances_from_bob_of_all_nodes[vertices[vertex]].a[v]
                    else:
                        # no key can be created between these two nodes.- both detectors - as such min distance is not
                        # of interest. - set to infty
                        distances_for_source_nodes[vertices[v]] = np.infty
                true_distances[vertex] = distances_for_source_nodes
        return true_distances

    def get_shortest_distance_of_source_nodes_with_switches(self):
        """
        Get the shortest distance of all source nodes to each of the Bobs. Distances between bobs are set to infty
        :return: The distances from each of the Bobs for each source node as a dictionary of EdgePropertyMaps
        """
        # get the minimum distances from all Bobs to all nodes
        distances_from_bob_of_all_nodes = self.get_shortest_distance_from_bobs_with_switches()
        # iterate over all vertices
        vertices = self.get_vertices()
        # will keep the dictionary of minimum distances from all Bobs to all source nodes
        true_distances = {}
        for vertex in range(len(vertices)):
            # for each bob
            if self.vertex_type[vertices[vertex]] == NodeType(2).name:
                distances_for_source_nodes = self.new_vertex_property(value_type = "double")
                for v in range(len(vertices)):
                    # if not a Bob node then keep same key as before
                    if self.vertex_type[vertices[v]] == NodeType(0).name:
                        distances_for_source_nodes[vertices[v]] = distances_from_bob_of_all_nodes[vertices[vertex]].a[v] - 5 * self.dbswitch
                    else:
                        # no key can be created between these two nodes.- both detectors - as such min distance is not
                        # of interest. - set to infty
                        distances_for_source_nodes[vertices[v]] = np.infty
                true_distances[vertex] = distances_for_source_nodes
        return true_distances

    def get_shortest_distance_between_source_nodes(self):
        vertices = self.get_vertices()
        distances = []
        for vertex_1 in range(len(vertices)):
            for vertex_2 in range(len(vertices)):
                if self.vertex_type[vertices[vertex_1]] == NodeType(0).name and self.vertex_type[vertices[vertex_2]] == NodeType(0).name\
                    and vertex_1 != vertex_2:
                    dist = shortest_distance(g = self.g, source = vertices[vertex_1], target = vertices[vertex_2], weights = self.lengths_of_connections)
                    source_vertex = self.label[vertices[vertex_1]]
                    target_vertex = self.label[vertices[vertex_2]]
                    distances.append(((source_vertex, target_vertex, dist)))
        return distances


    def get_shortest_distance_between_source_nodes_switch_loss(self):
        vertices = self.get_vertices()
        distances = []
        for vertex_1 in range(len(vertices)):
            for vertex_2 in range(len(vertices)):
                if self.vertex_type[vertices[vertex_1]] == NodeType(0).name and self.vertex_type[vertices[vertex_2]] == NodeType(0).name\
                    and vertex_1 != vertex_2:
                    dist = shortest_distance(g = self.g, source = vertices[vertex_1], target = vertices[vertex_2], weights = self.lengths_with_switch)
                    source_vertex = self.label[vertices[vertex_1]]
                    target_vertex = self.label[vertices[vertex_2]]
                    distances.append(((source_vertex, target_vertex, dist - 5* self.dbswitch)))
        return distances


class Network_Setup_With_Edges_Input(Network_Setup):

    def __init__(self, n, nodetypes, xcoords, ycoords, lengths_of_connections, node_names, fully_connected = True, p = 0.8, db_switch = 1):
        """
        This class is the network setup- keeps track of the network topology, all of the nodes, distances etc
        This class uses the lengths_of_connections list instead of the true distance to label the lenghts of the
        connections
        :param n: The number of nodes in the network: int
        :param nodetypes: Array of the types of the nodes 0 == Source, 1 == Possible bob with no bob, 2 == Possible bob
        with bob: list(n)
        :param xcoords: The xcoordinates of the nodes: list(n)
        :param ycoords: The ycoordinates of the nodes: list(n)
        :param lengths_of_connections: the length of the edge connections: list(connections)
        :param fully_connected: whether to use fully connected graph
        :param node_names: The names of the nodes: list(n)
        :param p: The network connectivity
        """
        # generate the graph with n nodes - here we generate a fully connected graph - this will be changed in future
        # g = generate_random_graph(n, n *(n-1)/2 - 2)
        super().__init__(n = n , nodetypes = nodetypes, xcoords = xcoords, ycoords = ycoords, node_names = node_names, db_switch = db_switch)
        if fully_connected:
            g = generate_fully_connected_graph(n)
        else:
            g = generate_graph_with_random_connections(n=n, p=p)
        # add vertex properties for the type of vertex, the coordinates and labels
        self.vertex_type = g.new_vertex_property(value_type = "object")
        x_coord = g.new_vertex_property(value_type = "double")
        y_coord = g.new_vertex_property(value_type = "double")
        self.label = g.new_vertex_property(value_type = "string")
        self.x_coord = x_coord
        self.y_coord = y_coord
        # get a list of all vertices
        vertices = g.get_vertices()
        # add these propeties as internal to the graph
        g.vertex_properties["name"] = self.label
        g.vertex_properties["node_type"] = self.vertex_type
        # set up the positions in the Network that Bob's and Alices can be placed.
        # S = only Alice, C = only Bob, B = Node C occupied By Bob
        # also set up coordinates of the vertices and then names of the vertices
        for vertex in range(len(vertices)):
            self.vertex_type[vertices[vertex]] = NodeType(nodetypes[vertex]).name
            x_coord[vertices[vertex]] = xcoords[vertex]
            y_coord[vertices[vertex]] = ycoords[vertex]
            self.label[vertices[vertex]] = node_names[vertex]
        # round the values of the connections
        edges_rounded = []
        for length in lengths_of_connections:
            edges_rounded.append(int(round(length)))
        # add the length as an edge property - this will come from the input of lengths_of_connections instead
        self.lengths_of_connections = g.new_edge_property(value_type = "double", vals = lengths_of_connections)
        self.lengths_rounded = g.new_edge_property(value_type = "int", vals = edges_rounded)
        # g.add_edge_list(edges, eprops = [self.lengths_of_connections])
        # add this as an internal property of the graph
        g.edge_properties["length_of_connections"] = self.lengths_of_connections
        self.g = g



class Network_Setup_with_priority_and_random_desired_connections(Network_Setup):

    def __init__(self, n, nodetypes, xcoords, ycoords, node_names, p_customer, fully_connected = True, p =0.8, db_switch = 1):
        """
        This class is the network setup- keeps track of the network topology, all of the nodes, distances etc
        :param n: The number of nodes in the network: int
        :param nodetypes: Array of the types of the nodes 0 == Source, 1 == Possible bob with no bob, 2 == Possible bob
        with bob: list(n)
        :param xcoords: The xcoordinates of the nodes: list(n)
        :param ycoords: The ycoordinates of the nodes: list(n)
        :param p_customer: The probability that a pair of customers will want to communicate with each other
        :param fully_connected: whether to use fully connected graph
        :param p: the network connectivity
        :param node_names: The names of the nodes: list(n)
        """
        super().__init__(n=n, nodetypes=nodetypes, xcoords=xcoords, ycoords=ycoords, node_names=node_names, fully_connected = fully_connected, p = p, db_switch = db_switch)
        priority = {}
        # get a list of all vertices
        vertices = self.g.get_vertices()
        for vertex in vertices:
            if self.vertex_type[vertices[vertex]] != NodeType(2).name and self.vertex_type[vertices[vertex]] != NodeType(1).name:
                priority[vertex] = np.random.uniform(low = 0.01, high = 1)
            else:
                # no priority for detector nodes needed
                priority[vertex] = 1
        self.priority = priority
        connections = {}
        for i in range(len(vertices)):
            for j in range(len(vertices) - i):
                if self.vertex_type[vertices[vertices[i]]] != NodeType(2).name and self.vertex_type[vertices[vertices[i]]] != NodeType(1).name:
                    if self.vertex_type[vertices[vertices[i+j]]] != NodeType(2).name and self.vertex_type[vertices[vertices[i+j]]] != NodeType(1).name:
                        if j != 0:
                            sample = np.random.uniform()
                            if sample < p_customer:
                                connections[str(i) + "-" + str(j + i)] = 1
                            else:
                                connections[str(i) + "-" + str(j + i)] = 0
                else:
                    # no connection to detector nodes
                    connections[str(i) + "-" + str(j+i)] = 0
        self.connections = connections

    def get_min_distance_for_node_to_other_nodes(self, node):
        vertices = self.g.get_vertices()
        distances = []
        for i in range(len(vertices)):
            if i != node:
                if str(node) + "-" + str(i) in self.connections:
                    if self.connections[str(node) + "-" + str(i)] == 1:
                        dist = shortest_distance(g = self.g, source = vertices[vertices[node]], target = vertices[vertices[i]], weights = self.lengths_of_connections)
                        distances.append(dist)
                elif str(i) + "-" + str(node) in self.connections:
                    if self.connections[str(i) + "-" + str(node)] == 1:
                        dist = shortest_distance(g=self.g, source=vertices[vertices[node]],
                                                 target=vertices[vertices[i]], weights=self.lengths_of_connections)
                        distances.append(dist)
        dist_mean = np.mean(distances)
        dist_std = np.std(distances, ddof = 1)
        return dist_mean, dist_std

    def get_n_shortest_distances_to_bobs(self, node, n):
        distances = self.get_shortest_distance_from_bobs()
        shortest_dist = []
        for key in distances:
            if len(shortest_dist) < n:
                shortest_dist.append(distances[key].a[node])
                shortest_dist.sort()
            else:
                for i in range(len(shortest_dist)):
                    if distances[key].a[node] < shortest_dist[i]:
                        shortest_dist[i] = distances[key].a[node]
                        shortest_dist.sort()
        return shortest_dist

    def get_no_customers(self, node):
        no_customers = 0
        vertices = self.g.get_vertices()
        for i in range(len(vertices)):
            if i != node:
                if str(i) + "-" + str(node) in self.connections:
                    if self.connections[str(i) + "-" + str(node)] == 1:
                        no_customers += 1
                elif str(node) + "-" + str(i) in self.connections:
                    if self.connections[str(node) + "-" + str(i)] == 1:
                        no_customers += 1
        return no_customers


class GeneralNetwork(Graph):

    def __init__(self, g, label, dbswitch):
        self.g = g
        self.label = g.new_vertex_property(value_type="string")
        vertices = g.get_vertices()
        # add these propeties as internal to the graph
        for vertex in range(len(vertices)):
            self.label[vertices[vertex]] = label[vertex]
        g.vertex_properties["name"] = self.label
        self.dbswitch = dbswitch
        super().__init__(g=self.g, directed=False)

    def copy(self):
        pass


    def update_db_switch(self, new_dbswitch):
        """
        Update the value of db_switch to the new value
        :param new_dbswitch: New value of bd_switch to be used
        """
        self.dbswitch = new_dbswitch
        edges = self.g.get_edges(eprops = [self.lengths_of_connections])
        edge_lengths_with_switch = []
        for edge in edges:
            edge_lengths_with_switch.append(edge[2] + 5 * new_dbswitch)
        self.lengths_with_switch = self.g.new_edge_property(value_type="double", vals=edge_lengths_with_switch)


    def position_bobs(self, nodeposns):
        """
        Move the position of Bobs to the values given: 1 in nodeposns is no Bob, 2 is place bob
        :param nodeposns: The array of the reordered Bobs: list(total no. of nodes in C)
        """
        # get a list of the vertices of the array
        vertices = self.get_vertices()
        # keeps track of the position in the nodeposns array
        i =0
        # iterate over every vertex
        for vertex in range(len(vertices)):
            # if vertex is a place without Bob and needs to be a Bob then change the node to have a Bob -
            if self.vertex_type[vertices[vertex]] == NodeType(1).name:
                if nodeposns[i] == 2:
                    self.vertex_type[vertices[vertex]] = NodeType(2).name
                i += 1
            # if vertex is a place with Bob and no Bob should be in location then remove Bob
            elif self.vertex_type[vertices[vertex]] == NodeType(2).name:
                if nodeposns[i] == 1:
                    self.vertex_type[vertices[vertex]] = NodeType(1).name
                i += 1

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
                dist, pred = dijkstra_search(self.g, weight=self.lengths_of_connections, source=vertices[vertex],
                                             visitor=VisitorExample(name=self.label, vertex_type=self.vertex_type))
                # add this to the dictionary - will be a dictionary of EdgePropertyMap
                distances_from_bob[vertex] = dist
        return distances_from_bob

    def get_shortest_distance_from_bobs_with_switches(self):
        """
        Get the shortest distance from each of the Bobs for each of the vertices - including other Bob vertices
        includes a 2.5dB loss due to switches
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
                dist, pred = dijkstra_search(self.g, weight=self.lengths_with_switch, source=vertices[vertex],
                                             visitor=VisitorExample(name=self.label, vertex_type=self.vertex_type))

                # add this to the dictionary - will be a dictionary of EdgePropertyMap
                distances_from_bob[vertex] = dist
        return distances_from_bob

    def get_shortest_distance_from_bobs_with_switches_channel_allocation_routing(self, db_switch):
        """
        Get the shortest distance from each of the Bobs for each of the vertices - including other Bob vertices
        includes a 2.5dB loss due to switches
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
                dist, pred = dijkstra_search(self.g, weight=self.lengths_of_connections, source=vertices[vertex],
                                             visitor=VisitorExample(name=self.label, vertex_type=self.vertex_type))
                for i in range(len(dist.a)):
                    if dist.a[i] < 0.00001:
                        dist.a[i] = 0.0
                    else:
                        dist.a[i] = dist.a[i] + 5 * db_switch
                # add this to the dictionary - will be a dictionary of EdgePropertyMap
                distances_from_bob[vertex] = dist
        return distances_from_bob

    def get_shortest_distance_of_source_nodes(self):
        """
        Get the shortest distance of all source nodes to each of the Bobs. Distances between bobs are set to infty
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
                    if self.vertex_type[vertices[v]] == NodeType(0).name:
                        distances_for_source_nodes[vertices[v]] = distances_from_bob_of_all_nodes[vertices[vertex]].a[v]
                    else:
                        # no key can be created between these two nodes.- both detectors - as such min distance is not
                        # of interest. - set to infty
                        distances_for_source_nodes[vertices[v]] = np.infty
                true_distances[vertex] = distances_for_source_nodes
        return true_distances

    def get_shortest_distance_of_source_nodes_with_switches(self):
        """
        Get the shortest distance of all source nodes to each of the Bobs. Distances between bobs are set to infty
        :return: The distances from each of the Bobs for each source node as a dictionary of EdgePropertyMaps
        """
        # get the minimum distances from all Bobs to all nodes
        distances_from_bob_of_all_nodes = self.get_shortest_distance_from_bobs_with_switches()
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
                    if self.vertex_type[vertices[v]] == NodeType(0).name:
                        distances_for_source_nodes[vertices[v]] = distances_from_bob_of_all_nodes[vertices[vertex]].a[
                                                                      v] - 5 * self.dbswitch
                    else:
                        # no key can be created between these two nodes.- both detectors - as such min distance is not
                        # of interest. - set to infty
                        distances_for_source_nodes[vertices[v]] = np.infty
                true_distances[vertex] = distances_for_source_nodes
        return true_distances

    def get_shortest_distance_of_source_nodes_with_switches_channel_allocation_routing(self, db_switch):
        """
        Get the shortest distance of all source nodes to each of the Bobs. Distances between bobs are set to infty
        :return: The distances from each of the Bobs for each source node as a dictionary of EdgePropertyMaps
        """
        # get the minimum distances from all Bobs to all nodes
        distances_from_bob_of_all_nodes = self.get_shortest_distance_from_bobs_with_switches_channel_allocation_routing(db_switch)
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
                    if self.vertex_type[vertices[v]] == NodeType(0).name:
                        distances_for_source_nodes[vertices[v]] = distances_from_bob_of_all_nodes[vertices[vertex]].a[ v]
                    else:
                        # no key can be created between these two nodes.- both detectors - as such min distance is not
                        # of interest. - set to infty
                        distances_for_source_nodes[vertices[v]] = np.infty
                true_distances[vertex] = distances_for_source_nodes
        return true_distances

    def get_shortest_distance_between_source_nodes(self):
        vertices = self.get_vertices()
        distances = []
        for vertex_1 in range(len(vertices)):
            for vertex_2 in range(len(vertices)):
                if self.vertex_type[vertices[vertex_1]] == NodeType(0).name and self.vertex_type[
                    vertices[vertex_2]] == NodeType(0).name \
                        and vertex_1 != vertex_2:
                    dist = shortest_distance(g=self.g, source=vertices[vertex_1], target=vertices[vertex_2],
                                             weights=self.lengths_of_connections)
                    source_vertex = self.label[vertices[vertex_1]]
                    target_vertex = self.label[vertices[vertex_2]]
                    distances.append(((source_vertex, target_vertex, dist)))
        return distances

    def get_shortest_distance_between_source_nodes_switch_loss(self):
        vertices = self.get_vertices()
        distances = []
        for vertex_1 in range(len(vertices)):
            for vertex_2 in range(len(vertices)):
                if self.vertex_type[vertices[vertex_1]] == NodeType(0).name and self.vertex_type[
                    vertices[vertex_2]] == NodeType(0).name \
                        and vertex_1 != vertex_2:
                    dist = shortest_distance(g=self.g, source=vertices[vertex_1], target=vertices[vertex_2],
                                             weights=self.lengths_with_switch)
                    source_vertex = self.label[vertices[vertex_1]]
                    target_vertex = self.label[vertices[vertex_2]]
                    distances.append(((source_vertex, target_vertex, dist - 5 * self.dbswitch)))
        return distances

    def draw_graph(self):
        pos = self.g.new_vertex_property("vector<double>")
        vertices = self.get_vertices(vprops = [self.x_coord, self.y_coord])
        for node in vertices:
            pos[node[0]] = [node[1], node[2]]
        graph_tool.draw.graph_draw(g= self.g, pos= pos)



class BusNetwork(GeneralNetwork):

    def __init__(self, xcoords, ycoords, nodetypes, label, dbswitch):
        node_types = []
        for type in nodetypes:
            node_types.append(NodeType(type).name)
        g, lengths_of_connections, lengths_rounded, x_coords, y_coords, lengths_with_switch, vertex_type = bustopology(xcoords, ycoords, node_types, dbswitch)
        self.lengths_of_connections = lengths_of_connections
        self.lengths_rounded = lengths_rounded
        self.lengths_with_switch  = lengths_with_switch
        self.x_coord = x_coords
        self.y_coord = y_coords
        self.vertex_type = vertex_type
        g.vertex_properties["node_type"] = self.vertex_type
        g.edge_properties["length_of_connections"] = self.lengths_of_connections
        g.edge_properties["length_with_switch"] = self.lengths_with_switch
        super().__init__(g=g, label = label, dbswitch = dbswitch)

    def copy(self):
        vertextype = []
        labels = []
        vertices = self.g.get_vertices()
        n = 0
        for vertex in vertices:
            vertextype.append(NodeType[self.vertex_type[vertex]].value)
            if NodeType[self.vertex_type[vertex]].value != 4:
                n += 1
            labels.append(copy.deepcopy(self.label[vertex]))
        return BusNetwork(xcoords = copy.deepcopy(self.x_coord.a[:n]), ycoords = copy.deepcopy(self.y_coord.a[:n]), nodetypes = vertextype[:n], label = labels, dbswitch = copy.deepcopy(self.dbswitch))


class RingNetwork(GeneralNetwork):

    def __init__(self, radius, no_nodes, nodetypes, label, dbswitch):
        node_types = []
        for type in nodetypes:
            node_types.append(NodeType(type).name)
        g, lengths_of_connections, lengths_rounded, x_coords, y_coords, lengths_with_switch = ring_topology(radius, no_nodes, node_types, dbswitch)
        self.lengths_of_connections = lengths_of_connections
        self.lengths_rounded = lengths_rounded
        self.lengths_with_switch = lengths_with_switch
        self.x_coord = x_coords
        self.y_coord = y_coords
        self.vertex_type = g.new_vertex_property(value_type="object")
        vertices = g.get_vertices()
        for vertex in vertices:
            self.vertex_type[vertices[vertex]] = node_types[vertex]
        g.vertex_properties["node_type"] = self.vertex_type
        g.edge_properties["length_of_connections"] = self.lengths_of_connections
        g.edge_properties["length_with_switch"] = self.lengths_with_switch
        super().__init__(g=g, label = label, dbswitch = dbswitch)

    def copy(self):
        print("Not possible to copy ring networks")
        raise ValueError

class StarNetwork(GeneralNetwork):

    def __init__(self, xcoords, ycoords, nodetypes, central_node, label, dbswitch):
        node_types = []
        for type in nodetypes:
            node_types.append(NodeType(type).name)
        g, lengths_of_connections, lengths_rounded, x_coords, y_coords, lengths_with_switch = star_topology(xcoords, ycoords, node_types, central_node, dbswitch)
        self.lengths_of_connections = lengths_of_connections
        self.lengths_rounded = lengths_rounded
        self.lengths_with_switch = lengths_with_switch
        self.x_coord = x_coords
        self.y_coord = y_coords
        self.centralnode = central_node
        self.vertex_type = g.new_vertex_property(value_type="object")
        vertices = g.get_vertices()
        for vertex in vertices:
            self.vertex_type[vertices[vertex]] = node_types[vertex]
        g.vertex_properties["node_type"] = self.vertex_type
        g.edge_properties["length_of_connections"] = self.lengths_of_connections
        g.edge_properties["length_with_switch"] = self.lengths_with_switch
        super().__init__(g=g, label=label, dbswitch=dbswitch)

    def copy(self):
        vertextype = []
        labels = []
        vertices = self.g.get_vertices()
        for vertex in vertices:
            vertextype.append(NodeType[self.vertex_type[vertex]].value)
            labels.append(copy.deepcopy(self.label[vertex]))
        return StarNetwork(xcoords=copy.deepcopy(self.x_coord.a), ycoords=copy.deepcopy(self.y_coord.a),
                          nodetypes=vertextype, central_node = copy.deepcopy(self.centralnode), label=labels, dbswitch=copy.deepcopy(self.dbswitch))

class MeshNetwork(GeneralNetwork):

    def __init__(self, xcoords, ycoords, nodetypes, no_of_conns_av, box_size, label, dbswitch):
        node_types = []
        for type in nodetypes:
            node_types.append(NodeType(type).name)
        graph = True
        while graph:
            try:
                g, lengths_of_connections, lengths_rounded, x_coords, y_coords, lengths_with_switch = mesh_topology(xcoords, ycoords, node_types, no_of_conns_av, box_size, dbswitch)
                graph = False
            except ValueError:
                print("This Graph is not fully connected. Try a different Graph.")
                raise ValueError
        self.lengths_of_connections = lengths_of_connections
        self.lengths_rounded = lengths_rounded
        self.lengths_with_switch = lengths_with_switch
        self.x_coord = x_coords
        self.y_coord = y_coords
        self.no_conns_av = no_of_conns_av
        self.box_size = box_size
        self.vertex_type = g.new_vertex_property(value_type="object")
        vertices = g.get_vertices()
        for vertex in vertices:
            self.vertex_type[vertices[vertex]] = node_types[vertex]
        g.vertex_properties["node_type"] = self.vertex_type
        g.edge_properties["length_of_connections"] = self.lengths_of_connections
        g.edge_properties["length_with_switch"] = self.lengths_with_switch
        super().__init__(g=g, label=label, dbswitch=dbswitch)


    def swap_db_switch(self, new_db_switch):
        edges_new_switch = []
        edges = self.g.get_edges(eprops = ["length_of_connections"])
        for edge in edges:
            edges_new_switch = edges[2] + 5 * new_db_switch
        lengths_with_switch = self.g.new_edge_property(value_type="double", vals=edges_new_switch)
        self.g.edge_properties["length_with_switch"] = lengths_with_switch
        self.lengths_with_switch = lengths_with_switch



    def copy(self):
        vertextype = []
        labels = []
        vertices = self.g.get_vertices()
        for vertex in vertices:
            vertextype.append(NodeType[self.vertex_type[vertex]].value)
            labels.append(copy.deepcopy(self.label[vertex]))
        return MeshNetwork(xcoords=copy.deepcopy(self.x_coord.a), ycoords=copy.deepcopy(self.y_coord.a),
                          nodetypes=vertextype, label=labels, dbswitch=copy.deepcopy(self.dbswitch), no_of_conns_av = copy.deepcopy(self.no_conns_av), box_size = copy.deepcopy(self.box_size))




class ExistingNetwork(GeneralNetwork):

    def __init__(self, g,lengths_of_connections, lengths_rounded, lengths_with_switch, x_coords, y_coords, node_types,label, dbswitch):
        self.lengths_of_connections = lengths_of_connections
        self.lengths_rounded = lengths_rounded
        self.lengths_with_switch = lengths_with_switch
        self.x_coord = x_coords
        self.y_coord = y_coords
        no_of_conns_av = 3.5
        box_size = 100
        self.no_conns_av = no_of_conns_av
        self.box_size = box_size
        self.vertex_type = g.new_vertex_property(value_type="object")
        vertices = g.get_vertices()
        for vertex in vertices:
            self.vertex_type[vertices[vertex]] = node_types[vertex]
        g.vertex_properties["node_type"] = self.vertex_type
        g.edge_properties["length_of_connections"] = self.lengths_of_connections
        g.edge_properties["length_with_switch"] = self.lengths_with_switch
        super().__init__(g=g, label=label, dbswitch=dbswitch)


    def swap_db_switch(self, new_db_switch):
        edges_new_switch = []
        edges = self.g.get_edges(eprops = ["length_of_connections"])
        for edge in edges:
            edges_new_switch = edges[2] + 5 * new_db_switch
        lengths_with_switch = self.g.new_edge_property(value_type="double", vals=edges_new_switch)
        self.g.edge_properties["length_with_switch"] = lengths_with_switch
        self.lengths_with_switch = lengths_with_switch



    def copy(self):
        vertextype = []
        labels = []
        vertices = self.g.get_vertices()
        for vertex in vertices:
            vertextype.append(NodeType[self.vertex_type[vertex]].value)
            labels.append(copy.deepcopy(self.label[vertex]))
        return MeshNetwork(xcoords=copy.deepcopy(self.x_coord.a), ycoords=copy.deepcopy(self.y_coord.a),
                          nodetypes=vertextype, label=labels, dbswitch=copy.deepcopy(self.dbswitch), no_of_conns_av = copy.deepcopy(self.no_conns_av), box_size = copy.deepcopy(self.box_size))








class HubSpokeNetwork(GeneralNetwork):

    def __init__(self, xcoords, ycoords, nodetypes, nodes_for_mesh, no_of_conns_av, box_size,  label, dbswitch, mesh_in_centre = True):
        node_types = []
        for type in nodetypes:
            node_types.append(NodeType(type).name)
        try:
            g, lengths_of_connections, lengths_rounded, x_coords, y_coords, lengths_with_switch = mesh_star_topology(xcoords, ycoords, node_types, nodes_for_mesh, no_of_conns_av, box_size, mesh_in_centre, dbswitch)
        except ValueError:
            print("This Graph is not fully connected. Try a different Graph.")
            raise ValueError
        self.lengths_of_connections = lengths_of_connections
        self.lengths_rounded = lengths_rounded
        self.lengths_with_switch = lengths_with_switch
        self.x_coord = x_coords
        self.y_coord = y_coords
        self.no_conns_av = no_of_conns_av
        self.box_size = box_size
        self.nodes_for_mesh = nodes_for_mesh
        self.mesh_in_centre = mesh_in_centre
        self.vertex_type = g.new_vertex_property(value_type="object")
        vertices = g.get_vertices()
        for vertex in vertices:
            self.vertex_type[vertices[vertex]] = node_types[vertex]
        g.vertex_properties["node_type"] = self.vertex_type
        g.edge_properties["length_of_connections"] = self.lengths_of_connections
        g.edge_properties["length_with_switch"] = self.lengths_with_switch
        super().__init__(g=g, label=label, dbswitch=dbswitch)


    def copy(self):
        vertextype = []
        labels = []
        vertices = self.g.get_vertices()
        for vertex in vertices:
            vertextype.append(NodeType[self.vertex_type[vertex]].value)
            labels.append(copy.deepcopy(self.label[vertex]))
        return HubSpokeNetwork(xcoords=copy.deepcopy(self.x_coord.a), ycoords=copy.deepcopy(self.y_coord.a), nodes_for_mesh = copy.deepcopy(self.nodes_for_mesh),
                          nodetypes=vertextype, label=labels, dbswitch=copy.deepcopy(self.dbswitch), no_of_conns_av = copy.deepcopy(self.no_conns_av), box_size = copy.deepcopy(self.box_size), mesh_in_centre = self.mesh_in_centre)#



class LatticeNetwork(GeneralNetwork):

    def __init__(self, shape, nodetypes, box_size,  label, dbswitch):
        node_types = []
        for type in nodetypes:
            node_types.append(NodeType(type).name)
        if math.prod(shape) != len(nodetypes):
            print("Shape of the graph does not correspond to length of nodetypes")
            raise ValueError
        g, lengths_of_connections, lengths_rounded, x_coords, y_coords, lengths_with_switch = lattice_topology(shape, node_types, box_size, dbswitch)
        self.shape = shape
        self.lengths_of_connections = lengths_of_connections
        self.lengths_rounded = lengths_rounded
        self.lengths_with_switch = lengths_with_switch
        self.x_coord = x_coords
        self.y_coord = y_coords
        self.box_size = box_size
        self.vertex_type = g.new_vertex_property(value_type="object")
        vertices = g.get_vertices()
        for vertex in vertices:
            self.vertex_type[vertices[vertex]] = node_types[vertex]
        g.vertex_properties["node_type"] = self.vertex_type
        g.edge_properties["length_of_connections"] = self.lengths_of_connections
        g.edge_properties["length_with_switch"] = self.lengths_with_switch
        super().__init__(g=g, label=label, dbswitch=dbswitch)


    def copy(self):
        vertextype = []
        labels = []
        vertices = self.g.get_vertices()
        for vertex in vertices:
            vertextype.append(NodeType[self.vertex_type[vertex]].value)
            labels.append(copy.deepcopy(self.label[vertex]))
        return LatticeNetwork(shape = self.shape, nodetypes=vertextype, label=labels, dbswitch=copy.deepcopy(self.dbswitch), box_size = copy.deepcopy(self.box_size))