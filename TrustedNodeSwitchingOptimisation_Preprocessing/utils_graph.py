import graph_tool.topology
from graph_tool.all import *
import numpy as np
import math
import random

def generate_fully_connected_graph(n):
    """
    Generates a fully connected n node graph
    :param n: number of nodes: int
    :return: a fully connected n node graph: Graph
    """
    # start with a graph - in the topology of TF-QKD this graph will be undirected
    g = Graph(directed = False)
    # add n vertices
    g.add_vertex(n = n)
    # add the fully connected edges
    for s in range(n):
        for t in range(s,n):
            if s != t:
                g.add_edge(g.vertex(s),g.vertex(t))
    return g

def generate_random_graph(n, connections):
    """
    Generate a completely random graph with n nodes, connections == connections:
    :param n: number of nodes
    :param connections: no. of connections
    :return: random Graph
    """
    # random undirected graph of n nodes
    # counter to ensure the right number of unique connections are added
    i = 0
    g = Graph(directed = False)
    g.add_vertex(n=n)
    # list of connections - keeps track of which connections are already present in the graph
    conn = []
    while i < connections:
       # generate random connection and check that it is valid - not generated before
       random_1 = np.random.randint(n)
       random_2 = np.random.randint(n)
       if random_1 != random_2:
            for c_1, c_2 in conn:
                if c_1 == random_1 and c_2 == random_2:
                    continue
                elif c_1 == random_2 and c_2 == random_1:
                    continue
            # if unique then add the vertex to the set
            g.add_edge(g.vertex(random_1), g.vertex(random_2))
            conn.append((c_1,c_2))
            i += 1
    return g

def generate_graph_with_random_connections(n, p):
    """
    This will generate a random n node graph with each node having at least one connection and with each connection
    having probability p to turn on
    :param n: no. of nodes of the graph
    :param p: probability of turning on the connection
    :return: The graph required
    """
    random_connections = np.random.randint(low = 1, high = n-1, size = (n))
    g = Graph(directed=False)
    g.add_vertex(n=n)
    on_edges = np.zeros((n,n-1))
    for i in range(len(on_edges)):
        random_on_vertex = np.random.randint(low = 0, high = n-1, size = (1))
        if random_on_vertex < i:
            # look for the [random_on_vertex, i-1] pos as this is the same position in opposite direction
            if on_edges[random_on_vertex, i - 1] == 1:
                random_on_vertex = (random_on_vertex + np.random.randint(low=1, high=n-2, size=(1))) % (n-1)
                on_edges[i][random_on_vertex] = 1
            else:
                on_edges[i][random_on_vertex] = 1
        else:
            #  look for the [random_on_vertex+1, i] pos as this is the same position in opposite direction
            if on_edges[random_on_vertex+1, i] == 1:
                random_on_vertex = (random_on_vertex + np.random.randint(low=1, high=n - 2, size=(1))) % (n - 1)
                on_edges[i][random_on_vertex] = 1
            else:
                on_edges[i][random_on_vertex] = 1
    ## need to add a check to ensure the graph is fully connected before continuing:
    need_to_check = True
    while need_to_check:
        g_test = Graph(directed=False)
        g_test.add_vertex(n=n)
        for i in range(len(on_edges)):
            for j in range(len(on_edges[i])):
                if on_edges[i][j] == 1:
                    if j < i:
                        g_test.add_edge(g_test.vertex(i), g_test.vertex(j))
                    else:
                        g_test.add_edge(g.vertex(i), g_test.vertex(j + 1))
        vertices = g_test.get_vertices()
        # This will hold the Bobs shortest distance
        distances_from_bob = {}
        # iterate through all the vertices
        has_inf = False
        vertex_1 = np.random.randint(len(vertices))
        for vertex_2 in range(len(vertices)):
            if vertex_1 != vertex_2:
                dist = shortest_distance(g=g_test, source=vertices[vertex_1], target=vertices[vertex_2])
                if dist.item() == 2147483647:
                    # add connection between these two connections:
                    on_edges[vertices[vertex_1]][vertices[vertex_2]] = 1
                    has_inf = True
                    break
                    break
        need_to_check = has_inf
    for i in range(len(on_edges)):
        for j in range(len(on_edges[i])):
            if j < i:
                if on_edges[i][j] == 1 or on_edges[j][i-1] == 1:
                    # edge already turned on
                    continue
                else:
                    prob = np.random.uniform()
                    if (prob < p / 2):
                        # turn on edge
                        on_edges[i][j] = 1
            else:
                if on_edges[i][j] == 1 or on_edges[j+1][i] == 1:
                    # edge already turned on
                    continue
                else:
                    prob = np.random.uniform()
                    if (prob < p / 2):
                        # turn on edge
                        on_edges[i][j] = 1
    # now to add on edges to the network
    for i in range(len(on_edges)):
        for j in range(len(on_edges[i])):
            if on_edges[i][j] == 1:
                if j < i:
                    g.add_edge(g.vertex(i), g.vertex(j))
                else:
                    g.add_edge(g.vertex(i), g.vertex(j+1))
    return g


def get_graph_with_connections_to_nearest_nodes(no_of_conns_av, xcoords, ycoords, box_size, db_switch = 1):
    """
    Generate a geometric graph with average connectivity given by no_of_conns_av
    :param no_of_conns_av: The average no. of connections per node
    :param xcoords: The list of x coordinates of each of the nodes
    :param ycoords: The list of y coordinates of each of the nodes
    :param box_size: The size of the box (box_size * box_size)
    :return: The graph, lengths_of_connections as an EdgeProperty, lengths_rounded as an EdgeProperty,
     x_coords as a VertexProperty, y_coords as a VertexProperty
    """
    # get a list of position in the form [[x,y]]
    positions = [[xcoords[i], ycoords[i]] for i in range(len(xcoords))]
    # get the radius of the connection - any connection with distance smaller than this will get connected
    radius = np.sqrt(no_of_conns_av / (len(xcoords) * np.pi)) * box_size
    # get the geometric graph with r and positions
    g, pos = graph_tool.generation.geometric_graph(points = positions, radius = radius)
    # adds coordinates to the graph as VertexProperties
    x_coords = g.new_vertex_property(value_type = "double")
    y_coords = g.new_vertex_property(value_type = "double")#
    vertices = g.get_vertices()
    for vertex in vertices:
        x_coords[vertex] = pos[vertex][0]
        y_coords[vertex] = pos[vertex][1]
    g.vertex_properties["x_coord"] = x_coords
    g.vertex_properties["y_coord"] = y_coords
    # check the graph is fully connected - if not then add connections to fully connect the graph
    # algorithm: for a vertex find min path to each other vertex- if this min path exists then add the node to the set
    # S_v that is the set containing all vertices reachable by starting at vertex v. If all vertices in the graph are
    # contained in S_v we have a complete graph else start from a vertex not in S_v and carry out the same analysis
    # adding vertices reachable by the new vertex u into set S_u. Continue until the union of all sets is the set of all
    # nodes. Now for every pair of sets S_v, S_u find the distance between each of the nodes and store them in an array
    # Then connect the minimum connection between the sets and any connections with distance <1.12x min distance - check
    # if the graph is connected if yes stop else continue

    # for an initial node check the min path to all other nodes
    vertex_current = 0
    not_fully_connected = True
    sets = {}
    while not_fully_connected:
        sets[vertex_current] = []
        dist = graph_tool.topology.shortest_distance(g, source = g.vertex(vertex_current))
        vertex_current_new = None
        # for all distances check the connectivity - if distance is 2147483647 then it means there is no connection -
        # do not add to the set with node vertex_current else add to vertex_current - use the latest vertex with
        # no connection as the nex vertex to check for set
        for i in range(len(dist.a)):
            if dist.a[i] == 2147483647:
                indict = False
                # check we have not already looked at the key before using it as new key for next set
                for key in sets.keys():
                    set_key = sets[key]
                    for j in set_key:
                        if j == i:
                            indict = True
                if not indict:
                    vertex_current_new = i
            else:
                sets[vertex_current].append(i)
        # if there is only 1 set for the whole graph no need to continue checking sets - so break out of loop
        if vertex_current_new == None:
            break
        else:
            # checkl whether the nodes in the sets looked at are all the nodes- if yes then no need to check more sets
            # break, else continue looking at sets
            vertex_current = vertex_current_new
            current_elements = []
            for key in sets.keys():
                current_elements.append(sets[key])
            current_elements = np.concatenate(current_elements).flat.base
            if len(current_elements) == len(dist.a):
                not_fully_connected = False
    # now need to connect the graph in such a way that the closest pair of nodes from each disjoint set are connected
    # in particular any connection smaller than 1.12 x distance of shortest pair in disjoint sets are also connected
    # if only one element in checked sets than graph already fully connected so just add edges with appropriate lengths
    # to the graph and return the required terms
    if len(list(sets.keys())) == 1:
        edges = g.get_edges()
        edge_non_rounded = []
        edges_rounded = []
        edges_with_switch = []
        for edge in edges:
            source_node = edge[0]
            target_node = edge[1]
            length_of_connection = get_length(x_coords.a, y_coords.a, source_node, target_node).item()
            length_rounded = int(length_of_connection)
            edges_rounded.append(length_rounded)
            edge_non_rounded.append(length_of_connection)
            length_with_switch = length_of_connection + 5 * db_switch
            edges_with_switch.append(length_with_switch)
            # add the length as an edge property
        lengths_of_connections = g.new_edge_property(value_type="double", vals=edge_non_rounded)
        lengths_rounded = g.new_edge_property(value_type="int", vals=edges_rounded)
        lengths_with_switch = g.new_edge_property(value_type = "double", vals = edges_with_switch)
        return g, lengths_of_connections, lengths_rounded, x_coords, y_coords, lengths_with_switch
    else:
        # if more than 1 element in set need to connect the graph fully. To do so follow protocol above
        # loop over pairs of sets
        break_out_forloop = False
        for i in range(len(list(sets.keys()))):
            # break if graph fully connected
            if break_out_forloop:
                break
            for j in range(i+1, len(list(sets.keys()))):
                # get distances between every pair of elements in sets: (i,j) forall i \in U, j \in V
                length_between_sets = {}
                for k in sets[list(sets.keys())[i]]:
                    for l in sets[list(sets.keys())[j]]:
                        distance = get_length(xcoords, ycoords, k, l)
                        length_between_sets[str(k) + "," + str(l)] = distance
                # add connections that have a distance of at most 1.12* the shortest distance between sets
                min_dist = min(length_between_sets, key = length_between_sets.get)
                tolerable_dist = 1.12 * length_between_sets[min_dist]
                for key in length_between_sets.keys():
                    if length_between_sets[key] < tolerable_dist:
                        key_list = key.split(",")
                        key_list = [int(key_list[0]), int(key_list[1])]
                        g.add_edge(g.vertex(key_list[0]), g.vertex(key_list[1]))
                # check whether the graph is fully connected, if it is break out of the for loop
                dist = graph_tool.topology.shortest_distance(g, source=g.vertex(0))
                is_complete = True
                for distance in dist.a:
                    if distance == 2147483647:
                        is_complete = False
                if is_complete:
                    break_out_forloop = True
                    break
        # add edges with appropriate lengths to the graph and return the required terms
        edges = g.get_edges()
        edge_non_rounded = []
        edges_rounded = []
        edges_with_switch = []
        for edge in edges:
            source_node = edge[0]
            target_node = edge[1]
            length_of_connection = get_length(x_coords.a, y_coords.a, source_node, target_node).item()
            length_rounded = int(length_of_connection)
            edges_rounded.append(length_rounded)
            edge_non_rounded.append(length_of_connection)
            length_with_switch = length_of_connection + 5 * db_switch
            edges_with_switch.append(length_with_switch)
            # add the length as an edge property
        lengths_of_connections = g.new_edge_property(value_type="double", vals=edge_non_rounded)
        lengths_rounded = g.new_edge_property(value_type="int", vals=edges_rounded)
        lengths_with_switch = g.new_edge_property(value_type="double", vals=edges_with_switch)
        return g, lengths_of_connections, lengths_rounded, x_coords, y_coords, lengths_with_switch



def check_feasible(no_of_conns_av, xcoords, ycoords, box_size, dbswitch = 1):
    """
    GChecks whether fully connected graph is feeasible
    :param no_of_conns_av: The average no. of connections per node
    :param xcoords: The list of x coordinates of each of the nodes
    :param ycoords: The list of y coordinates of each of the nodes
    :param box_size: The size of the box (box_size * box_size)
    :return: boolean whether graph is feasible
    """
    # get a list of position in the form [[x,y]]
    # positions = [[xcoords[i], ycoords[i]] for i in range(len(xcoords))]
    # # get the radius of the connection - any connection with distance smaller than this will get connected
    # radius = np.sqrt(no_of_conns_av / (len(xcoords) * np.pi)) * box_size
    # # get the geometric graph with r and positions
    # g, pos = graph_tool.generation.geometric_graph(points = positions, radius = radius)
    # # adds coordinates to the graph as VertexProperties
    # x_coords = g.new_vertex_property(value_type = "double")
    # y_coords = g.new_vertex_property(value_type = "double")#
    # vertices = g.get_vertices()
    # for vertex in vertices:
    #     x_coords[vertex] = pos[vertex][0]
    #     y_coords[vertex] = pos[vertex][1]
    # g.vertex_properties["x_coord"] = x_coords
    # g.vertex_properties["y_coord"] = y_coords
    # check the graph is fully connected - if not then add connections to fully connect the graph
    # algorithm: for a vertex find min path to each other vertex- if this min path exists then add the node to the set
    # S_v that is the set containing all vertices reachable by starting at vertex v. If all vertices in the graph are
    # contained in S_v we have a complete graph else start from a vertex not in S_v and carry out the same analysis
    # adding vertices reachable by the new vertex u into set S_u. Continue until the union of all sets is the set of all
    # nodes. Now for every pair of sets S_v, S_u find the distance between each of the nodes and store them in an array
    # Then connect the minimum connection between the sets and any connections with distance <1.12x min distance - check
    # if the graph is connected if yes stop else continue


    # get a list of position in the form [[x,y]]
    positions = [[xcoords[i], ycoords[i]] for i in range(len(xcoords))]
    # get the radius of the connection - any connection with distance smaller than this will get connected
    radius = np.sqrt(no_of_conns_av / (len(xcoords) * np.pi)) * box_size
    # get the geometric graph with r and positions
    g, pos = graph_tool.generation.geometric_graph(points=positions, radius=radius)
    # adds coordinates to the graph as VertexProperties
    x_coords = g.new_vertex_property(value_type="double")
    y_coords = g.new_vertex_property(value_type="double")  #
    vertices = g.get_vertices()
    for vertex in vertices:
        x_coords[vertex] = pos[vertex][0]
        y_coords[vertex] = pos[vertex][1]
    g.vertex_properties["x_coord"] = x_coords
    g.vertex_properties["y_coord"] = y_coords
    vertex_current = 0
    dist = graph_tool.topology.shortest_distance(g, source = g.vertex(vertex_current))
    feasible = True
    for i in range(len(dist.a)):
        if dist.a[i] == 2147483647:
           feasible = False
    return feasible


def generate_hub_and_spoke_network(xcoords, ycoords, nodetypes, box_size):
    """
    To get this method to work all detector nodes must be in the first section of xcoords, ycoords
    :param xcoords:
    :param ycoords:
    :param nodetypes:
    :param box_size:
    :return:
    """
    # get a list of position in the form [[x,y]]
    positions = []
    for i in range(len(xcoords)):
        if nodetypes[i]== 1 or nodetypes[i]== 2:
            positions.append([xcoords[i], ycoords[i]])
    # get the radius of the connection - any connection with distance smaller than this will get connected
    radius = 3 * box_size/4
    # get the geometric graph with r and positions
    g, pos = graph_tool.generation.geometric_graph(points=positions, radius=radius)
    new_edges = []
    for i in range(len(xcoords)):
        if nodetypes[i] == 0:
            current_shortest_distance_position = 0
            current_shortest_distance = np.inf
            for j in range(len(positions)):
                distance = get_length(xcoords, ycoords, vertex_i = i, vertex_j = j)
                if distance < current_shortest_distance:
                    current_shortest_distance = distance
                    current_shortest_distance_position = j
            new_edges.append((i,current_shortest_distance_position))
    for edge in new_edges:
        g.add_edge(edge[0], edge[1])
    x_coords = g.new_vertex_property(value_type="double")
    y_coords = g.new_vertex_property(value_type="double")  #
    vertices = g.get_vertices()
    for vertex in vertices:
        x_coords[vertex] = xcoords[vertex]
        y_coords[vertex] = ycoords[vertex]
    g.vertex_properties["x_coord"] = x_coords
    g.vertex_properties["y_coord"] = y_coords
    # add edges with appropriate lengths to the graph and return the required terms
    edges = g.get_edges()
    edge_non_rounded = []
    edges_rounded = []
    edges_with_switch = []
    for edge in edges:
        source_node = edge[0]
        target_node = edge[1]
        length_of_connection = get_length(x_coords.a, y_coords.a, source_node, target_node).item()
        length_rounded = int(length_of_connection)
        edges_rounded.append(length_rounded)
        edge_non_rounded.append(length_of_connection)
        length_with_switch = length_of_connection + 5
        edges_with_switch.append(length_with_switch)
        # add the length as an edge property
    lengths_of_connections = g.new_edge_property(value_type="double", vals=edge_non_rounded)
    lengths_rounded = g.new_edge_property(value_type="int", vals=edges_rounded)
    lengths_with_switch = g.new_edge_property(value_type="double", vals=edges_with_switch)
    return g, lengths_of_connections, lengths_rounded, x_coords, y_coords, lengths_with_switch






def get_length(xcoords, ycoords, vertex_i, vertex_j):
    """
    Get the length of the connection vertex_i to vertex_j from the coordinates of the graph
    l = ((x_1-x_0) ** 2 + (y_1 - y_0) ** 2) ** 1/2
    :param xcoords: xcoordinate list
    :param ycoords: ycoordinate list
    :param vertex_i: The start vertex of the connection
    :param vertex_j: The end vertex of the connection
    :return: The length of the connection
    """
    xdist = (xcoords[vertex_i] - xcoords[vertex_j]) ** 2
    ydist = (ycoords[vertex_i] - ycoords[vertex_j]) ** 2
    return np.sqrt(xdist + ydist)


def centre_of_mass_metric(capacities):
    """
    Calculates the centre of mass metric of the network  = sum_(i,j) \in S c^{max}_{i,j}(L_1,L_2)
    :param capacities: The set of capacities which is a list [(source, target, detector, capacity)]
    :return: The centre of mass parameter
    """
    # factor of 1/2 accounted for as we are only taking half of the capacities:
    centre_of_mass = 0
    for source_node, target_node, detector_node, capacity in capacities:
        centre_of_mass += capacity
    return centre_of_mass

def generalised_moment_of_inertia(capacities, distances, n):
    """
    Calculates the generalised moment of inertia =  sum_(i,j) \in S  (L_1+L_2)^n c^{max}_{i,j}(L_1,L_2)
    :param capacities: The set of capacities which is a list [(source, target, detector, capacity)]
    :param distances: The set of the distances of the network is a list [(source, target, detector, total length,
    length to detector)]
    :param n: The order of the moment of inertia: int or double
    :return: the value for the moment of inertia
    """
    moment_of_inertia = 0
    for i in range(len(capacities)):
        source_node, target_node, detector_node, capacity = capacities[i]
        source_node, target_node, detector_node, total_length, length_to_detector = distances[i]
        moment_of_inertia += np.power((total_length),n) * capacity
    return moment_of_inertia

def general_metric(beta, gamma, capacities, distances, n):
    """
    Calculate the general metric of the capacities = sum_(i,j) \in S beta_{ij} c^{max}_{i,j}(L_1,L_2)
    + gamma_{ij}(L_1+L_2)^n c^{max}_{i,j}(L_1,L_2)
    :param beta: The coefficient for the centre of mass term : double
    :param gamma: The coefficient for the general moment of inertia term: double
    :param capacities: The set of capacities which is a list [(source, target, detector, capacity)]
    :param distances: The set of the distances of the network is a list [(source, target, detector, total length,
    length to detector)]
    :param n: The order of the moment of inertia: int or double
    :return: the value for the  general metric
    """
    # get both terms
    centre_of_mass = centre_of_mass_metric(capacities)
    generalised_moi = generalised_moment_of_inertia(capacities, distances, n)
    # need coefficients to be sufficiently large as otherwise there will be large errors from division
    if (beta < 0.000001) and (gamma < 0.000001):
        print("beta and gamma cannot be smaller than the precision. Increase one of them")
        raise ValueError
    else:
        return (beta **2 + gamma ** 2) ** (-1/2) * (beta * centre_of_mass + gamma * generalised_moi)


def log_metric(capacities):
    """
    Use a log metric to test quality of graphs - here use sum_(i,j) \in S ln(c^{max}_{i,j}(L_1,L_2)+1)
    :param capacities:
    :return:
    """
    # factor of 1/2 accounted for as we are only taking half of the capacities:
    centre_of_mass = 0
    for source_node, target_node, detector_node, capacity in capacities:
        centre_of_mass += np.log10(capacity + 1)
    return centre_of_mass


def centre_of_mass_metric_node_features(capacities, graph):
    """
    Calculates the centre of mass metric of the network  = sum_(i,j) \in S 1/min{lambda_1, lambda_2} c^{max}_{i,j}(L_1,L_2)
    :param capacities: The set of capacities which is a list [(source, target, detector, capacity)]
    :param graph: The graph with information of the priority
    :return: The centre of mass parameter
    """
    # factor of 1/2 accounted for as we are only taking half of the capacities:
    centre_of_mass = 0
    for source_node, target_node, detector_node, capacity in capacities:
        centre_of_mass += (np.min([graph.priority[int(source_node)], graph.priority[int(target_node)]]) ** -1) *capacity
    return centre_of_mass


def generalised_moment_of_inertia_node_features(capacities, distances, n, graph):
    """
    Calculates the generalised moment of inertia =  sum_(i,j) \in S 1/min{lambda_1, lambda_2} (L_1+L_2)^n c^{max}_{i,j}(L_1,L_2)
    :param capacities: The set of capacities which is a list [(source, target, detector, capacity)]
    :param distances: The set of the distances of the network is a list [(source, target, detector, total length,
    length to detector)]
    :param n: The order of the moment of inertia: int or double
    :param graph: The graph with information of the priority
    :return: the value for the moment of inertia
    """
    moment_of_inertia = 0
    for i in range(len(capacities)):
        source_node, target_node, detector_node, capacity = capacities[i]
        source_node, target_node, detector_node, total_length, length_to_detector = distances[i]
        moment_of_inertia += (np.min([graph.priority[int(source_node)], graph.priority[int(target_node)]]) ** -1) *\
                             np.power((total_length), n) * capacity
    return moment_of_inertia


def general_metric_node_features(beta, gamma, capacities, distances, n, graph):
    """
    Calculate the general metric of the capacities = sum_(i,j) \in S 1/min{lambda_1, lambda_2} beta_{ij} c^{max}_{i,j}(L_1,L_2)
    + gamma_{ij}(L_1+L_2)^n c^{max}_{i,j}(L_1,L_2)
    :param beta: The coefficient for the centre of mass term : double
    :param gamma: The coefficient for the general moment of inertia term: double
    :param capacities: The set of capacities which is a list [(source, target, detector, capacity)]
    :param distances: The set of the distances of the network is a list [(source, target, detector, total length,
    length to detector)]
    :param n: The order of the moment of inertia: int or double
    :param graph: The graph with information of the priority
    :return: the value for the  general metric
    """
    # get both terms
    centre_of_mass = centre_of_mass_metric_node_features(capacities, graph)
    generalised_moi = generalised_moment_of_inertia_node_features(capacities, distances, n, graph)
    # need coefficients to be sufficiently large as otherwise there will be large errors from division
    if (beta < 0.000001) and (gamma < 0.000001):
        print("beta and gamma cannot be smaller than the precision. Increase one of them")
        raise ValueError
    else:
        return (beta ** 2 + gamma ** 2) ** (-1 / 2) * (beta * centre_of_mass + gamma * generalised_moi)






##### Generate different topologies for graphs:

#### bus topology

def min_parameter(xcoords, ycoords):
    n = len(xcoords)
    x_mean = np.mean(xcoords)
    x_var = np.var(xcoords)
    if x_var < 0.00000001:
        raise ValueError
    else:
        y_mean = np.mean(ycoords)
        xy_mean = np.mean(np.multiply(xcoords, ycoords))
        m = (xy_mean - x_mean * y_mean)/ x_var
        b = y_mean - m * x_mean
        return m,b


def findpositionofmindistance(xcoord, ycoord, m, b):
    if m == 0:
        y = b
        x = xcoord
    else:
        x = (ycoord + xcoord/m -b)/ (m + 1/m)
        y = m * x +b
    return x,y

def bustopology(xcoords, ycoords, nodetypes, dbswitch):
    """
    Creates bus topology graph - assumes that the y axis is narrower than the x axis and the main cable is parallel to
    x axis -
    :param xcoords:
    :param ycoords:
    :param nodetypes: NodeTypes of style name of the nodetype not the number i.e. S not 1
    :param box_size:
    :return:
    """
    m, b = min_parameter(xcoords, ycoords)
    # algorithm: find the coordinates (x_j, y_j) where the nodes connect to the bus line and add these to the graph as
    # virtual nodes and add an edge between the node and the virtual node and connect the virtual nodes together via
    # their next neighbour
    virtual_node_positions = []
    edges = []
    length = len(xcoords)
    for i in range(length):
        x,y = findpositionofmindistance(xcoords[i], ycoords[i], m ,b)
        virtual_node_positions.append((x,y))
        xcoords = np.append(xcoords, [x])
        ycoords = np.append(ycoords, [y])
        nodetypes.append("I")
        ## check -1
        edges.append((i, length + i))
    for i in range(len(virtual_node_positions)):
        node_connect = None
        for j in range(len(virtual_node_positions)):
            if i != j:
                if j ==0 or (i == 0 and j == 1):
                    x = virtual_node_positions[j][0] - virtual_node_positions[i][0]
                    if x < 0:
                        x = np.infty
                    else:
                        node_connect = j
                else:
                    y = virtual_node_positions[j][0] -virtual_node_positions[i][0]
                    if y < x and y > 0:
                        x = y
                        node_connect = j
        if node_connect != None:
            edges.append((i + length , node_connect + length ))
    positions = [[xcoords[i], ycoords[i]] for i in range(len(xcoords))]
    # get the radius of the connection - any connection with distance smaller than this will get connected
    radius = 0.0
    # get the geometric graph with r and positions
    g, pos = graph_tool.generation.geometric_graph(points=positions, radius=radius)
    # adds coordinates to the graph as VertexProperties
    x_coords = g.new_vertex_property(value_type="double")
    y_coords = g.new_vertex_property(value_type="double")  #
    vertex_type  = g.new_vertex_property(value_type = "object")
    g.vertex_properties["node_type"] = vertex_type
    vertices = g.get_vertices()
    for vertex in vertices:
        vertex_type[vertices[vertex]] = nodetypes[vertex]
        x_coords[vertex] = pos[vertex][0]
        y_coords[vertex] = pos[vertex][1]
    g.vertex_properties["x_coord"] = x_coords
    g.vertex_properties["y_coord"] = y_coords
    for edge in edges:
        g.add_edge(edge[0], edge[1])
    # add edges with appropriate lengths to the graph and return the required terms
    edges = g.get_edges()
    edge_non_rounded = []
    edges_rounded = []
    edges_with_switch = []
    for edge in edges:
        source_node = edge[0]
        target_node = edge[1]
        length_of_connection = get_length(x_coords.a, y_coords.a, source_node, target_node).item()
        length_rounded = int(length_of_connection)
        edges_rounded.append(length_rounded)
        edge_non_rounded.append(length_of_connection)
        length_with_switch = length_of_connection + 5 * dbswitch
        edges_with_switch.append(length_with_switch)
        # add the length as an edge property
    lengths_of_connections = g.new_edge_property(value_type="double", vals=edge_non_rounded)
    lengths_rounded = g.new_edge_property(value_type="int", vals=edges_rounded)
    lengths_with_switch = g.new_edge_property(value_type="double", vals=edges_with_switch)
    return g, lengths_of_connections, lengths_rounded, x_coords, y_coords, lengths_with_switch, vertex_type


#### ring topology
def get_coords(radius, no_nodes):
    if no_nodes == 0:
        print("cannot have no nodes")
        raise ValueError

    xcoords = []
    ycoords = []
    for m in range(no_nodes):
        sigma = np.random.randn() * np.pi * 0.25/ no_nodes
        xcoords.append(radius * np.cos(sigma + 2 * np.pi * m / no_nodes ))
        ycoords.append(radius * np.sin(sigma + 2 * np.pi * m / no_nodes))
    return xcoords, ycoords


def get_arc_length(radius, xcoords, ycoords, vertex_i, vertex_j):
    lsqrd = (xcoords[vertex_i] - xcoords[vertex_j]) ** 2 + (ycoords[vertex_i] - ycoords[vertex_j]) ** 2
    theta = np.arccos(1-(lsqrd / (2 * radius ** 2)))
    return radius * theta


def ring_topology(radius, no_nodes, node_types, dbswitch):
    xcoords, ycoords = get_coords(radius, no_nodes)
    edges = []
    for i in range(len(xcoords)):
        edges.append((i, (i+1) % (len(xcoords))))
    positions = [[xcoords[i], ycoords[i]] for i in range(len(xcoords))]
    # get the radius of the connection - any connection with distance smaller than this will get connected
    r = 0.0
    # get the geometric graph with r and positions
    g, pos = graph_tool.generation.geometric_graph(points=positions, radius=r)
    # adds coordinates to the graph as VertexProperties
    x_coords = g.new_vertex_property(value_type="double")
    y_coords = g.new_vertex_property(value_type="double")  #
    vertex_type  = g.new_vertex_property(value_type = "object")

    vertices = g.get_vertices()
    for vertex in vertices:
        vertex_type[vertices[vertex]] = node_types[vertex]
        x_coords[vertex] = pos[vertex][0]
        y_coords[vertex] = pos[vertex][1]
    g.vertex_properties["x_coord"] = x_coords
    g.vertex_properties["y_coord"] = y_coords
    g.vertex_properties["node_type"] = vertex_type
    for edge in edges:
        g.add_edge(edge[0], edge[1])
    # add edges with appropriate lengths to the graph and return the required terms
    edges = g.get_edges()
    edge_non_rounded = []
    edges_rounded = []
    edges_with_switch = []
    for edge in edges:
        source_node = edge[0]
        target_node = edge[1]
        length_of_connection = get_arc_length(radius, x_coords.a, y_coords.a, source_node, target_node).item()
        length_rounded = int(length_of_connection)
        edges_rounded.append(length_rounded)
        edge_non_rounded.append(length_of_connection)
        length_with_switch = length_of_connection + 5 * dbswitch
        edges_with_switch.append(length_with_switch)
        # add the length as an edge property
    lengths_of_connections = g.new_edge_property(value_type="double", vals=edge_non_rounded)
    lengths_rounded = g.new_edge_property(value_type="int", vals=edges_rounded)
    lengths_with_switch = g.new_edge_property(value_type="double", vals=edges_with_switch)
    return g, lengths_of_connections, lengths_rounded, x_coords, y_coords, lengths_with_switch

#### star topology

def star_topology(xcoords, ycoords, node_types, central_node, dbswitch):
    edges = []
    positions = [[xcoords[i], ycoords[i]] for i in range(len(xcoords))]
    radius = 0.0
    g, pos = graph_tool.generation.geometric_graph(points=positions, radius=radius)
    vertices = g.get_vertices()
    for node in vertices:
        if node != central_node:
            edges.append((node, central_node))
    x_coords = g.new_vertex_property(value_type="double")
    y_coords = g.new_vertex_property(value_type="double")  #
    vertex_type = g.new_vertex_property(value_type="object")
    g.vertex_properties["node_type"] = vertex_type
    vertices = g.get_vertices()
    for vertex in vertices:
        vertex_type[vertices[vertex]] = node_types[vertex]
        x_coords[vertex] = pos[vertex][0]
        y_coords[vertex] = pos[vertex][1]
    g.vertex_properties["x_coord"] = x_coords
    g.vertex_properties["y_coord"] = y_coords
    for edge in edges:
        g.add_edge(edge[0], edge[1])
    # add edges with appropriate lengths to the graph and return the required terms
    edges = g.get_edges()
    edge_non_rounded = []
    edges_rounded = []
    edges_with_switch = []
    for edge in edges:
        source_node = edge[0]
        target_node = edge[1]
        length_of_connection = get_length(x_coords.a, y_coords.a, source_node, target_node).item()
        length_rounded = int(length_of_connection)
        edges_rounded.append(length_rounded)
        edge_non_rounded.append(length_of_connection)
        length_with_switch = length_of_connection + 5 * dbswitch
        edges_with_switch.append(length_with_switch)
        # add the length as an edge property
    lengths_of_connections = g.new_edge_property(value_type="double", vals=edge_non_rounded)
    lengths_rounded = g.new_edge_property(value_type="int", vals=edges_rounded)
    lengths_with_switch = g.new_edge_property(value_type="double", vals=edges_with_switch)
    return g, lengths_of_connections, lengths_rounded, x_coords, y_coords, lengths_with_switch

#### mesh topology

def get_partitions(g):
    partitions = {}
    number_nodes = len(g.get_vertices())
    vertex_current = 0
    while True:
        in_current_partition = []
        dist = graph_tool.topology.shortest_distance(g, source=g.vertex(vertex_current))
        for i in range(len(dist.a)):
            if dist.a[i] != 2147483647:
                in_current_partition.append(i)
        partitions[vertex_current] = in_current_partition
        explored_vertices = []
        for key in partitions.keys():
            explored_vertices.extend(partitions[key])
        if len(explored_vertices) == number_nodes:
            break
        else:
            explored_vertices.sort()
            vertices = g.get_vertices()
            vertices_not_explored = list(set(vertices) - set(explored_vertices)) # get the list of vertices not explored in current sets
            vertex_current = vertices_not_explored[0]
    return partitions

def calculate_distances_between_partitions(x_coords, y_coords, partitions):
    distances_between_partitions = {}
    for i in range(len(partitions.keys())):
        for j in range(i+1, len(partitions.keys())):
            distance = {}
            key_1 = list(partitions.keys())[i]
            key_2 = list(partitions.keys())[j]
            for k in partitions[key_1]:
                for m in partitions[key_2]:
                    length_of_connection = get_length(x_coords.a, y_coords.a, k, m).item()
                    distance[k,m] = length_of_connection
            if key_1 not in distances_between_partitions.keys():
                distances_between_partitions[key_1] = {key_2: distance}
            else:
                distances_between_partitions[key_1][key_2] = distance
    return distances_between_partitions

def connect_m_shortest_connections_between_partitions(g, partition, distances_between_partitions, no_of_conns_av):
    current_connections_to_set = {}
    for key_1 in distances_between_partitions.keys():
        distances_min = []
        sigma = 1.25
        m = int(np.random.normal(no_of_conns_av * min(np.power(len(partition[key_1]), 0.33), np.power(len(g.get_vertices()) - len(partition[key_1]), 0.33)), sigma))
        if m < 1:
            m = 1
        n=0
        for key_2 in distances_between_partitions[key_1].keys():
            if key_1 in current_connections_to_set.keys():
                if m > current_connections_to_set[key_1]:
                    if n == 0:
                        m = m - current_connections_to_set[key_1]
                        n += 1
                    distances = distances_between_partitions[key_1][key_2]
                    distances = dict(sorted(distances.items(), key=lambda item: item[1]))
                    if len(distances_min) < m:
                        for key in distances.keys():
                            if len(distances_min) >= m:
                                break
                            distances_min.append((key_2,distances[key],key))
                            distances_min = sorted(distances_min, key = lambda item: item[1])
                    else:
                        for key in distances.keys():
                            if distances[key] < distances_min[-1][1]:
                                distances_min[-1] = (key_2, distances[key], key)
                                distances_min = sorted(distances_min, key=lambda item: item[1])
                            else:
                                # both arrays are sorted so if min of array 1 is not smaller than max of
                                # array 2 then no element is smaller
                                break
            else:
                distances = distances_between_partitions[key_1][key_2]
                distances = dict(sorted(distances.items(), key=lambda item: item[1]))
                if len(distances_min) < m:
                    for key in distances.keys():
                        if len(distances_min) >= m:
                            break
                        distances_min.append((key_2, distances[key], key))
                        distances_min = sorted(distances_min, key=lambda item: item[1])
                else:
                    for key in distances.keys():
                        if distances[key] < distances_min[-1][1]:
                            distances_min[-1] = (key_2, distances[key], key)
                            distances_min = sorted(distances_min, key=lambda item: item[1])
                        else:
                            # both arrays are sorted so if min of array 1 is not smaller than max of
                            # array 2 then no element is smaller
                            break
        for key_partition, distance, key_connection in distances_min:
            if key_partition not in current_connections_to_set.keys():
                current_connections_to_set[key_partition] = 1
            else:
                current_connections_to_set[key_partition] += 1
            g.add_edge(key_connection[0], key_connection[1])



def mesh_topology(xcoords, ycoords, node_types, no_of_conns_av, box_size, dbswitch):
    # get a list of position in the form [[x,y]]
    is_feasible = check_feasible(no_of_conns_av, xcoords, ycoords, box_size)
    if not is_feasible:
        print("mesh topology not feasible")
        raise ValueError
    positions = [[xcoords[i], ycoords[i]] for i in range(len(xcoords))]
    # get the radius of the connection - any connection with distance smaller than this will get connected
    radius = np.sqrt(no_of_conns_av / (len(xcoords) * np.pi)) * box_size
    # get the geometric graph with r and positions
    g, pos = graph_tool.generation.geometric_graph(points=positions, radius=radius)
    # adds coordinates to the graph as VertexProperties
    x_coords = g.new_vertex_property(value_type="double")
    y_coords = g.new_vertex_property(value_type="double")  #
    vertex_type = g.new_vertex_property(value_type="object")
    g.vertex_properties["node_type"] = vertex_type
    vertices = g.get_vertices()
    for vertex in vertices:
        vertex_type[vertices[vertex]] = node_types[vertex]
        x_coords[vertex] = pos[vertex][0]
        y_coords[vertex] = pos[vertex][1]
    g.vertex_properties["x_coord"] = x_coords
    g.vertex_properties["y_coord"] = y_coords
    if not is_feasible:
        partitions = get_partitions(g)
        distances_between_partitions = calculate_distances_between_partitions(x_coords, y_coords, partitions)
        connect_m_shortest_connections_between_partitions(g, partitions, distances_between_partitions, no_of_conns_av)
    dist = graph_tool.topology.shortest_distance(g, source=g.vertex(0))
    is_feasible = True
    for i in range(len(dist.a)):
        if dist.a[i] == 2147483647:
            is_feasible = False
    if not is_feasible:
        print("mesh topology not feasible")
        raise ValueError
    # add edges with appropriate lengths to the graph and return the required terms
    edges = g.get_edges()
    edge_non_rounded = []
    edges_rounded = []
    edges_with_switch = []
    for edge in edges:
        source_node = edge[0]
        target_node = edge[1]
        length_of_connection = get_length(x_coords.a, y_coords.a, source_node, target_node).item()
        length_rounded = int(length_of_connection)
        edges_rounded.append(length_rounded)
        edge_non_rounded.append(length_of_connection)
        length_with_switch = length_of_connection + 5 * dbswitch
        edges_with_switch.append(length_with_switch)
        # add the length as an edge property
    lengths_of_connections = g.new_edge_property(value_type="double", vals=edge_non_rounded)
    lengths_rounded = g.new_edge_property(value_type="int", vals=edges_rounded)
    lengths_with_switch = g.new_edge_property(value_type="double", vals=edges_with_switch)
    return g, lengths_of_connections, lengths_rounded, x_coords, y_coords, lengths_with_switch



#### mesh + star topology (hub and spokes)

def mesh_star_topology(xcoords, ycoords, node_types, nodes_for_mesh, no_of_conns_av, box_size, mesh_in_centre = True, db_switch = 1):
    # only capable of using the first set of n nodes to the mesh grid and the remaining nodes into star topology
    if mesh_in_centre:
        is_feasible = check_feasible(no_of_conns_av, xcoords[:nodes_for_mesh], ycoords[:nodes_for_mesh], box_size / 3)
    else:
        is_feasible = check_feasible(no_of_conns_av, xcoords[:nodes_for_mesh], ycoords[:nodes_for_mesh], box_size)
    if not is_feasible:
        print("mesh topology not feasible")
        raise ValueError
    positions = [[xcoords[i], ycoords[i]] for i in range(nodes_for_mesh)]
    # get the radius of the connection - any connection with distance smaller than this will get connected
    if mesh_in_centre:
        radius = np.sqrt(no_of_conns_av / (nodes_for_mesh * np.pi)) * box_size/3
    else:
        radius = np.sqrt(no_of_conns_av / (nodes_for_mesh * np.pi)) * box_size
    # get the geometric graph with r and positions
    g, pos = graph_tool.generation.geometric_graph(points=positions, radius=radius)
    position_rest = [[xcoords[i], ycoords[i]] for i in range(nodes_for_mesh, len(xcoords))]
    for node in range(len(position_rest)):
        current_node_to_connect = 0
        current_distance = np.infty
        for mesh_node in range(nodes_for_mesh):
            distance_to_node = get_length(xcoords, ycoords, mesh_node, node + nodes_for_mesh).item()
            if distance_to_node < current_distance:
                current_distance = distance_to_node
                current_node_to_connect = mesh_node
        v = g.add_vertex()
        g.add_edge(v, current_node_to_connect)
    x_coords = g.new_vertex_property(value_type="double")
    y_coords = g.new_vertex_property(value_type="double")  #
    vertex_type = g.new_vertex_property(value_type="object")
    g.vertex_properties["node_type"] = vertex_type
    all_positions = [[xcoords[i], ycoords[i]] for i in range(len(xcoords))]
    vertices = g.get_vertices()
    for vertex in vertices:
        vertex_type[vertices[vertex]] = node_types[vertex]
        x_coords[vertex] = all_positions[vertex][0]
        y_coords[vertex] = all_positions[vertex][1]
    g.vertex_properties["x_coord"] = x_coords
    g.vertex_properties["y_coord"] = y_coords
    # add edges with appropriate lengths to the graph and return the required terms
    edges = g.get_edges()
    edge_non_rounded = []
    edges_rounded = []
    edges_with_switch = []
    for edge in edges:
        source_node = edge[0]
        target_node = edge[1]
        length_of_connection = get_length(x_coords.a, y_coords.a, source_node, target_node).item()
        length_rounded = int(length_of_connection)
        edges_rounded.append(length_rounded)
        edge_non_rounded.append(length_of_connection)
        length_with_switch = length_of_connection + 5 * db_switch
        edges_with_switch.append(length_with_switch)
        # add the length as an edge property
    lengths_of_connections = g.new_edge_property(value_type="double", vals=edge_non_rounded)
    lengths_rounded = g.new_edge_property(value_type="int", vals=edges_rounded)
    lengths_with_switch = g.new_edge_property(value_type="double", vals=edges_with_switch)
    return g, lengths_of_connections, lengths_rounded, x_coords, y_coords, lengths_with_switch

#### lattice topology: with positions and everything

def lattice_topology(shape, node_types, box_size, db_switch):
    g = graph_tool.generation.lattice(shape = shape)
    x_coords = g.new_vp("double", (np.arange(g.num_vertices()) % shape[0]) * box_size / (shape[0] - 1))
    y_coords = g.new_vp("double", (np.arange(g.num_vertices()) // shape[0]) * box_size / (shape[1] -1))
    pos = group_vector_property([x_coords, y_coords])

    graph_draw(g, pos, output="lattice.png")

    vertex_type = g.new_vertex_property(value_type="object")
    g.vertex_properties["node_type"] = vertex_type
    vertices = g.get_vertices()
    for vertex in vertices:
        vertex_type[vertices[vertex]] = node_types[vertex]

    edges = g.get_edges()
    edge_non_rounded = []
    edges_rounded = []
    edges_with_switch = []
    for edge in edges:
        source_node = edge[0]
        target_node = edge[1]
        length_of_connection = get_length(x_coords.a, y_coords.a, source_node, target_node).item()
        length_rounded = int(length_of_connection)
        edges_rounded.append(length_rounded)
        edge_non_rounded.append(length_of_connection)
        # account for the loss due to fibres
        length_with_switch = length_of_connection + 5 * db_switch
        edges_with_switch.append(length_with_switch)
        # add the length as an edge property
    lengths_of_connections = g.new_edge_property(value_type="double", vals=edge_non_rounded)
    lengths_rounded = g.new_edge_property(value_type="int", vals=edges_rounded)
    lengths_with_switch = g.new_edge_property(value_type="double", vals=edges_with_switch)
    return g, lengths_of_connections, lengths_rounded, x_coords, y_coords, lengths_with_switch


# lattice_topology(shape = [6,15], node_types =np.random.randint(low = 0, high = 3, size = [11 * 21]), box_size = 100)