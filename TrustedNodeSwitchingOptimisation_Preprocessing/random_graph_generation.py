import copy
import math
from capacity_calculation import *
from graph_tool.all import *
import numpy as np
from capacity_main import capacities_for_graph_efficient_corrected, capacities_for_graph_using_m_shortest_connections_efficient_corrected, capacities_for_graph_with_node_features, perm_unique
from utils_graph import check_feasible
from generate_graph import *
import csv
from minimum_length import get_minimum_length_for_any_source_pair, get_n_minimum_length_for_any_source_pair, get_minimum_length_to_each_bob_for_any_source_pair

def generate_random_position_graph(n, no_of_bobs, no_of_bob_locations, output_hot = False, p = 0.8, no_connected_nodes = None, size = 100):
    """
    Generates a random fully connected graph with n nodes and the specified no. of bobs and locations for bobs. 100km
    box size
    :param n: The total number of nodes: int
    :param no_of_bobs: The totsl number of bobs in the network: int
    :param no_of_bob_locations: The number of bob locations in the network
    :param output_hot: Whether to return the hot Bob output
    :param p: The connectivity of the network: int or double
    :param no_connected_nodes: The number of connected nodes average per connection if using geometric graph: default
    None - if you want to use geometric graph must pass this and box_size in
    :param size: The size of the network in km
    :return: The output of the whole analysis in form [(metric, node_types, capacities)], the graph optimised:
     Network_Setup, the xcoords and ycoords of the positions [(doubles)]
    """
    # list of names for the nodes
    node_names = ["1", "2", "3","4", "5", "6", "7", "8", "9", "10",
                  "11", "12", "13", "14", "15"]
    for i in range(16, 500):
        node_names.append(str(i))
    # check to see the input parameters are valid - e.g. are there enough nodes to account for the no. of bob locations
    if no_of_bob_locations > n or no_of_bobs > no_of_bob_locations:
        print("Parameters invalid")
        raise ValueError
    # currently only able to do this many nodes
    elif n > len(node_names):
        print("number of nodes too many. Currently only able to do " + str(len(node_names)) + " nodes")
        raise ValueError
    # Get the random x and y coordinates

    xcoords = np.random.uniform(low = 0.0, high = size, size = (n))
    ycoords = np.random.uniform(low = 0.0, high = size, size = (n))

    # get an array of the appropriate number of nodes, bob nodes, and bob locations
    array_for_perturbation = np.concatenate((np.full(shape=no_of_bobs, fill_value=2),
                                             np.full(shape=no_of_bob_locations - no_of_bobs, fill_value=1)))
    array_for_perturbation = np.concatenate((array_for_perturbation, np.full(shape = n - no_of_bob_locations, fill_value = 0)))
    # randomly permute this set of nodes to generate random graph
    node_types = np.random.permutation(array_for_perturbation)
    # get the full information for this graph
    output, graph = capacities_for_graph_efficient_corrected(n=n, nodetypes=node_types,
                                                             xcoords=xcoords, ycoords=ycoords,
                                                             node_names=node_names[:n], p = p,
                                                             no_connected_nodes = no_connected_nodes, box_size = 100)
    if output_hot:
        output_hot, graph = capacities_for_graph_efficient_corrected(n=n, nodetypes=node_types,
                                                                 xcoords=xcoords, ycoords=ycoords,
                                                                 node_names=node_names[:n], cold_bobs = False, p = p,
                                                                 no_connected_nodes = no_connected_nodes, box_size = 100)
        return output, graph, xcoords, ycoords, output_hot
    else:
        return output, graph, xcoords, ycoords


def generate_random_position_graph_m_best_bobs(n, no_of_bobs, no_of_bob_locations, m, output_hot = False, no_connected_nodes = None):
    """
    Generates a random fully connected graph with n nodes and the specified no. of bobs and locations for bobs.
    :param n: The total number of nodes: int
    :param no_of_bobs: The totsl number of bobs in the network: int
    :param no_of_bob_locations: The number of bob locations in the network
    :param m: The no. of Bobs to take the rates from
    :param output_hot: Whether to return the hot Bob output
    :param no_connected_nodes: The number of connected nodes average per connection if using geometric graph: default
    None - if you want to use geometric graph must pass this and box_size in
    :return: The output of the whole analysis in form [(metric, node_types, capacities)], the graph optimised:
     Network_Setup, the xcoords and ycoords of the positions [(doubles)]
    """
    # list of names for the nodes
    node_names = ["Alice", "Bob", "Charlie", "Dave", "Eric", "Felicity", "Gary", "Harry", "Irene", "Juan", "Kale",
                  "Lavender", "Molly", "Nigel", "Oliver", "Pat", "Q", "Raul", "Sam", "Thomas", "Uma", "Vincent",
                  "Wanda", "Xavier", "Yvonnie", "Zoe", "Anna", "Bert", "Chloe", "Dan", "Elena", "Fred", "Gavin",
                  "Holly", "Io", "Jan", "Katherine", "Laura", "Mike", "Neil", "Omar", "Pedro", "Rodrick", "Selene",
                  "Trevor", "Utah", "Vas", "Wally", "Xena", "Zero", "1", "2", "3","4", "5", "6", "7", "8", "9", "10",
                  "11", "12", "13", "14", "15"]
    for i in range(16, 500):
        node_names.append(str(i))
    # check to see the input parameters are valid - e.g. are there enough nodes to account for the no. of bob locations
    if no_of_bob_locations > n or no_of_bobs > no_of_bob_locations:
        print("Parameters invalid")
        raise ValueError
    # currently only able to do this many nodes
    elif n > len(node_names):
        print("number of nodes too many. Currently only able to do " + str(len(node_names)) + " nodes")
        raise ValueError
    elif m > no_of_bobs:
        print("Number of bobs to take the rates from set to larger than the no. of Bobs. Setting m = no_of_Bobs")
        m = no_of_bobs
    # Get the random x and y coordinates
    xcoords = np.random.uniform(low = 0.0, high = 100.0, size = (n))
    ycoords = np.random.uniform(low = 0.0, high = 100.0, size = (n))
    # get an array of the appropriate number of nodes, bob nodes, and bob locations
    array_for_perturbation = np.concatenate((np.full(shape=no_of_bobs, fill_value=2),
                                             np.full(shape=no_of_bob_locations - no_of_bobs, fill_value=1)))
    array_for_perturbation = np.concatenate((array_for_perturbation, np.full(shape = n - no_of_bob_locations, fill_value = 0)))
    # randomly permute this set of nodes to generate random graph
    node_types = np.random.permutation(array_for_perturbation)
    # get the full information for this graph
    output, graph = capacities_for_graph_using_m_shortest_connections_efficient_corrected(n=n, nodetypes=node_types,
                                                             xcoords=xcoords, ycoords=ycoords,
                                                             node_names=node_names[:n], m = m, no_connected_nodes = no_connected_nodes, box_size = 100)
    if output_hot:
        output_hot, graph = capacities_for_graph_using_m_shortest_connections_efficient_corrected(n=n, nodetypes=node_types,
                                                                 xcoords=xcoords, ycoords=ycoords,
                                                                 node_names=node_names[:n], cold_bobs = False, m = m, no_connected_nodes = no_connected_nodes, box_size = 100)
        return output, graph, xcoords, ycoords, output_hot
    else:
        return output, graph, xcoords, ycoords


def generate_random_position_graph_node_features(n, no_of_bobs, no_of_bob_locations, p_connection, output_hot = False):
    """
    Generates a random fully connected graph with n nodes and the specified no. of bobs and locations for bobs.
    :param n: The total number of nodes: int
    :param no_of_bobs: The totsl number of bobs in the network: int
    :param no_of_bob_locations: The number of bob locations in the network
    :param output_hot: Whether to return the hot Bob output
    :param p_connection: The probability that the nodes are connected
    :return: The output of the whole analysis in form [(metric, node_types, capacities)], the graph optimised:
     Network_Setup, the xcoords and ycoords of the positions [(doubles)]
    """
    # list of names for the nodes
    node_names = ["0", "1", "2", "3","4", "5", "6", "7", "8", "9", "10",
                  "11", "12", "13", "14", "15"]
    for i in range(16, 500):
        node_names.append(str(i))
    # check to see the input parameters are valid - e.g. are there enough nodes to account for the no. of bob locations
    if no_of_bob_locations > n or no_of_bobs > no_of_bob_locations:
        print("Parameters invalid")
        raise ValueError
    # currently only able to do this many nodes
    elif n > len(node_names):
        print("number of nodes too many. Currently only able to do " + str(len(node_names)) + " nodes")
        raise ValueError
    # Get the random x and y coordinates
    xcoords = np.random.uniform(low = 0.0, high = 100.0, size = (n))
    ycoords = np.random.uniform(low = 0.0, high = 100.0, size = (n))
    # get an array of the appropriate number of nodes, bob nodes, and bob locations
    array_for_perturbation = np.concatenate((np.full(shape=no_of_bobs, fill_value=2),
                                             np.full(shape=no_of_bob_locations - no_of_bobs, fill_value=1)))
    array_for_perturbation = np.concatenate((array_for_perturbation, np.full(shape = n - no_of_bob_locations, fill_value = 0)))
    # randomly permute this set of nodes to generate random graph
    node_types = np.random.permutation(array_for_perturbation)
    # get the full information for this graph
    output, graph = capacities_for_graph_with_node_features(n=n, nodetypes=node_types,
                                                             xcoords=xcoords, ycoords=ycoords,
                                                             node_names=node_names[:n], p_connection=p_connection)
    if output_hot:
        output_hot, graph = capacities_for_graph_with_node_features(n=n, nodetypes=node_types,
                                                                 xcoords=xcoords, ycoords=ycoords,
                                                                 node_names=node_names[:n], cold_bobs = False, p_connection=p_connection)
        return output, graph, xcoords, ycoords, output_hot
    else:
        return output, graph, xcoords, ycoords


def generate_random_graph_no_solving(n, no_of_bobs, no_of_bob_locations, p = 0.8, no_connected_nodes = None, size = 100):
    """
    Generates a random fully connected graph with n nodes and the specified no. of bobs and locations for bobs. 100km
    box size
    :param n: The total number of nodes: int
    :param no_of_bobs: The totsl number of bobs in the network: int
    :param no_of_bob_locations: The number of bob locations in the network
    :param output_hot: Whether to return the hot Bob output
    :param p: The connectivity of the network: int or double
    :param no_connected_nodes: The number of connected nodes average per connection if using geometric graph: default
    None - if you want to use geometric graph must pass this and box_size in
    :param size: The size of the network in km
    :return: The output of the whole analysis in form [(metric, node_types, capacities)], the graph optimised:
     Network_Setup, the xcoords and ycoords of the positions [(doubles)]
    """
    # list of names for the nodes
    node_names = ["1", "2", "3","4", "5", "6", "7", "8", "9", "10",
                  "11", "12", "13", "14", "15"]
    for i in range(16, 500):
        node_names.append(str(i))
    # check to see the input parameters are valid - e.g. are there enough nodes to account for the no. of bob locations
    if no_of_bob_locations > n or no_of_bobs > no_of_bob_locations:
        print("Parameters invalid")
        raise ValueError
    # currently only able to do this many nodes
    elif n > len(node_names):
        print("number of nodes too many. Currently only able to do " + str(len(node_names)) + " nodes")
        raise ValueError
    # Get the random x and y coordinates

    xcoords = np.random.uniform(low = 0.0, high = size, size = (n))
    ycoords = np.random.uniform(low = 0.0, high = size, size = (n))

    # get an array of the appropriate number of nodes, bob nodes, and bob locations
    array_for_perturbation = np.concatenate((np.full(shape=no_of_bobs, fill_value=2),
                                             np.full(shape=no_of_bob_locations - no_of_bobs, fill_value=1)))
    array_for_perturbation = np.concatenate((array_for_perturbation, np.full(shape = n - no_of_bob_locations, fill_value = 0)))
    # randomly permute this set of nodes to generate random graph
    node_types = np.random.permutation(array_for_perturbation)
    if no_connected_nodes == None:
        graph = Network_Setup(n=n, nodetypes=node_types, xcoords=xcoords,
                              ycoords=ycoords, node_names=
                              node_names, fully_connected=False, p=p)
    else:
        graph = Network_Setup(n=n, nodetypes=node_types, xcoords=xcoords,
                          ycoords=ycoords, node_names=
                          node_names, fully_connected=False, geometric_graph=True,
                          no_of_connected_nodes=no_connected_nodes, box_size=size)
    return graph


class SpecifiedTopologyGraph:

    def __init__(self, graph = None):
        if graph != None:
            self.graph = graph


    def bus_topology_graph(self, xcoords, ycoords, nodetypes, label, dbswitch):
        self.graph = BusNetwork(xcoords, ycoords, nodetypes, label, dbswitch)

    def ring_topology_graph(self, radius, no_nodes, node_types, label, dbswitch):
        self.graph = RingNetwork(radius, no_nodes, node_types, label, dbswitch)

    def star_topology_graph(self, xcoords, ycoords, node_types, central_node, label, dbswitch):
        self.graph = StarNetwork(xcoords, ycoords, node_types, central_node, label, dbswitch)

    def mesh_topology_graph(self, xcoords, ycoords, node_types, no_of_conns_av, box_size, label, dbswitch):
        self.graph = MeshNetwork(xcoords, ycoords, node_types, no_of_conns_av, box_size, label, dbswitch)

    def hub_spoke_graph(self, xcoords, ycoords, node_types, nodes_for_mesh, no_of_conns_av, box_size,  label, dbswitch, mesh_in_centre):
        self.graph = HubSpokeNetwork(xcoords, ycoords, node_types, nodes_for_mesh, no_of_conns_av, box_size,  label, dbswitch, mesh_in_centre)

    def lattice_graph(self, shape, nodetypes, box_size,  label, dbswitch):
        self.graph = LatticeNetwork(shape, nodetypes, box_size,  label, dbswitch)

    def make_standard_labels(self):
        node_names = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                      "11", "12", "13", "14", "15"]
        for i in range(16, 500):
            node_names.append(str(i))
        return node_names


    def generate_random_coordinates(self, n, box_size):
        xcoords = np.random.uniform(low=0.0, high=box_size, size=(n))
        ycoords = np.random.uniform(low=0.0, high=box_size, size=(n))
        self.xcoords, self.ycoords = xcoords, ycoords
        return xcoords, ycoords

    def generate_random_detector_perturbation(self, n, no_of_bobs, no_of_bob_locations):
        # get an array of the appropriate number of nodes, bob nodes, and bob locations
        array_for_perturbation = np.concatenate((np.full(shape=no_of_bobs, fill_value=2),
                                                 np.full(shape=no_of_bob_locations - no_of_bobs, fill_value=1)))
        array_for_perturbation = np.concatenate(
            (array_for_perturbation, np.full(shape=n - no_of_bob_locations, fill_value=0)))
        # randomly permute this set of nodes to generate random graph
        node_types = np.random.permutation(array_for_perturbation)
        self.node_types = node_types
        return node_types

    def generate_random_bus_graph(self, n, no_of_bobs, no_of_bob_locations, dbswitch, box_size):
        label = self.make_standard_labels()
        xcoords, ycoords = self.generate_random_coordinates(n, box_size)
        node_types = self.generate_random_detector_perturbation(n, no_of_bobs, no_of_bob_locations)
        self.bus_topology_graph(xcoords, ycoords, node_types, label, dbswitch)

    def generate_random_ring_graph(self, n, no_of_bobs, no_of_bob_locations, radius, dbswitch):
        label = self.make_standard_labels()
        node_types = self.generate_random_detector_perturbation(n, no_of_bobs, no_of_bob_locations)
        self.ring_topology_graph(radius, n, node_types, label, dbswitch)

    def generate_random_star_graph(self, n, no_of_bobs, no_of_bob_locations, dbswitch, box_size, central_node_is_detector):
        label = self.make_standard_labels()
        xcoords, ycoords = self.generate_random_coordinates(n, box_size)
        node_types = self.generate_random_detector_perturbation(n, no_of_bobs, no_of_bob_locations)
        central_node = 0
        if central_node_is_detector:
            for i in range(len(node_types)):
                if node_types[i] == 1 or node_types[i] == 2:
                    central_node = i
                    break
        else:
            for i in range(len(node_types)):
                if node_types[i] == 0:
                    central_node = i
                    break
        self.star_topology_graph(xcoords, ycoords, node_types, central_node, label, dbswitch)

    def generate_random_mesh_graph(self, n, no_of_bobs, no_of_bob_locations, dbswitch, box_size, no_of_conns_av):
        i = 0
        while True:
            try:
                label = self.make_standard_labels()
                xcoords, ycoords = self.generate_random_coordinates(n, box_size)
                node_types = self.generate_random_detector_perturbation(n, no_of_bobs, no_of_bob_locations)
                self.mesh_topology_graph(xcoords, ycoords, node_types, no_of_conns_av, box_size, label, dbswitch)
            except:
                if i < 200:
                    pass
                else:
                    i += 1
                    raise ValueError
            else:
                break

    def generate_random_hub_spoke_graph(self, n, no_of_bobs, no_of_bob_locations, dbswitch, box_size, mesh_composed_of_only_detectors, nodes_for_mesh, no_of_conns_av):
        i = 0
        while True:
            try:
                label = self.make_standard_labels()
                xcoords, ycoords = self.generate_random_coordinates(n, box_size)
                if mesh_composed_of_only_detectors:
                    if no_of_bob_locations < nodes_for_mesh:
                        print("For all mesh nodes to be detectors the number of detectors must be bigger than the number of nodes in the mesh grid.")
                        raise ValueError
                    else:
                        array_for_perturbation = np.concatenate((np.full(shape=no_of_bobs, fill_value=2),
                                                                 np.full(shape=no_of_bob_locations - no_of_bobs, fill_value=1)))
                        array_for_perturbation = np.concatenate(
                            (array_for_perturbation, np.full(shape=n - no_of_bob_locations, fill_value=0)))
                        node_types = array_for_perturbation
                else:
                    node_types = self.generate_random_detector_perturbation(n, no_of_bobs, no_of_bob_locations)
                self.hub_spoke_graph(xcoords, ycoords, node_types, nodes_for_mesh, no_of_conns_av, box_size,  label, dbswitch, mesh_in_centre = False)
            except ValueError:
                if i < 50:
                    pass
                else:
                    i += 1
                    raise ValueError
            else:
                break

    def generate_hub_spoke_with_hub_in_centre(self, n, no_of_bobs, no_of_bob_locations, dbswitch, box_size, mesh_composed_of_only_detectors, nodes_for_mesh, no_of_conns_av):
        i = 0
        while True:
            try:
                label = self.make_standard_labels()
                xcoords_mesh, ycoords_mesh = self.generate_random_coordinates(nodes_for_mesh, box_size/3)
                xcoords_mesh = xcoords_mesh + box_size/3
                ycoords_mesh = ycoords_mesh + box_size/3 # add term to centre mesh parts
                xcoords_rest, ycoords_rest = self.generate_random_coordinates(n - nodes_for_mesh, box_size)
                xcoords = np.concatenate((xcoords_mesh, xcoords_rest))
                ycoords = np.concatenate((ycoords_mesh, ycoords_rest))
                self.xcoords, self.ycoords = xcoords, ycoords
                if mesh_composed_of_only_detectors:
                    if no_of_bob_locations < nodes_for_mesh:
                        print(
                            "For all mesh nodes to be detectors the number of detectors must be bigger than the number of nodes in the mesh grid.")
                        raise ValueError
                    else:
                        array_for_perturbation = np.concatenate((np.full(shape=no_of_bobs, fill_value=2),
                                                                 np.full(shape=no_of_bob_locations - no_of_bobs, fill_value=1)))
                        array_for_perturbation = np.concatenate(
                            (array_for_perturbation, np.full(shape=n - no_of_bob_locations, fill_value=0)))
                        node_types = array_for_perturbation
                else:
                    node_types = self.generate_random_detector_perturbation(n, no_of_bobs, no_of_bob_locations)
                self.hub_spoke_graph(xcoords, ycoords, node_types, nodes_for_mesh, no_of_conns_av, box_size, label, dbswitch, mesh_in_centre = True)
            except ValueError:
                if i < 50:
                    pass
                else:
                    i += 1
                    raise ValueError
            else:
                break


    def generate_random_lattice_graph(self, shape, no_of_bobs, no_of_bob_locations, box_size, dbswitch):
        label = self.make_standard_labels()
        n = math.prod(shape)
        node_types = self.generate_random_detector_perturbation(n, no_of_bobs, no_of_bob_locations)
        self.lattice_graph(shape, node_types, box_size, label, dbswitch)
        self.xcoords = (np.arange(self.graph.num_vertices()) % shape[0]) * box_size / (shape[0] - 1)
        self.ycoords = (np.arange(self.graph.num_vertices()) // shape[0]) * box_size / (shape[1] - 1)

    def get_capacities_for_graph_tfqkd(self, cold_bobs = True, beta = 1, gamma = 0, q =0.1, log_met = False, switches = True):

        no_of_bobs = 0
        no_of_locations_for_bob = 0
        for i in range(len(self.node_types)):
            if self.node_types[i] == 2:
                no_of_bobs += 1
                no_of_locations_for_bob += 1
            elif self.node_types[i] == 1:
                no_of_locations_for_bob += 1
        # generate an array of perturbations with the correct no. of Bobs and free locations
        array_for_perturbation = np.concatenate((np.full(shape=no_of_bobs, fill_value=2),
                                                 np.full(shape=no_of_locations_for_bob - no_of_bobs, fill_value=1)))
        # find all possible unique perturbations of this setup
        perturbations = list(perm_unique(list(array_for_perturbation)))
        # to store the requested data
        output = []
        # open and hold the precalculated capacities
        dictionary = {}
        if cold_bobs:
            with open('rates_coldbob_new.csv', mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        print(f'Column names are {", ".join(row)}')
                        line_count += 1
                    dictionary["L" + row["L"] + "LB" + row["LB"]] = float(row['rate'])
                    line_count += 1
                print(f'Processed {line_count} lines.')
        else:

            with open('rates_hotbob_new.csv', mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        print(f'Column names are {", ".join(row)}')
                        line_count += 1
                    dictionary["L" + row["L"] + "LB" + row["LB"]] = float(row['rate'])
                    line_count += 1
                print(f'Processed {line_count} lines.')
        # for each perturbation calculate the capacities and the metric
        for perturbation in perturbations:
            # set the nodes to the perturbation states
            j = 0
            new_node_types = copy.deepcopy(self.node_types)
            for i in range(len(self.node_types)):
                if self.node_types[i] == 1 or self.node_types[i] == 2:
                    new_node_types[i] = perturbation[j]
                    j += 1
            perturbed_graph =  SpecifiedTopologyGraph(graph = self.graph.copy())
            perturbed_graph.graph.position_bobs(new_node_types)
            if switches:
                true_distance = perturbed_graph.graph.get_shortest_distance_of_source_nodes_with_switches()
            else:
                true_distance = perturbed_graph.graph.get_shortest_distance_of_source_nodes()
            distance_array = get_minimum_length_to_each_bob_for_any_source_pair(true_distance, perturbed_graph.graph.g)
            # get the capacities of for each of the pairs of sources
            capacity, distance_array = calculate_capacity_efficient_all_distances(distance_array, dictionary)
            # calculate the metric
            if log_met:
                metric = log_metric(capacity)
            else:
                metric = general_metric(beta=beta, gamma=gamma, capacities=capacity, distances=distance_array, n=q)
            # place the requested info into array
            output.append((metric, perturbation, capacity))
        return output

