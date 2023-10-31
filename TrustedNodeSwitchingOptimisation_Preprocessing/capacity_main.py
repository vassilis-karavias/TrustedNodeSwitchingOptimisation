from __future__ import division, absolute_import, print_function
import numpy as np
from numpy.random import *  # for random sampling
from utils_graph import generate_fully_connected_graph, general_metric, general_metric_node_features, log_metric
# We need to import the graph_tool module itself
from graph_tool.all import *
from generate_graph import Network_Setup, Network_Setup_With_Edges_Input, Network_Setup_with_priority_and_random_desired_connections
from minimum_length import get_minimum_length_for_any_source_pair, get_n_minimum_length_for_any_source_pair, get_minimum_length_to_each_bob_for_any_source_pair
from capacity_calculation import *
from server_client_capacities import total_rate_server_client, total_rate_server_client_efficient
from hot_tf_qkd_capacities import total_rate_tf_efficient
import time
import csv

class unique_element:
    def __init__(self,value,occurrences):
        """
        To find the possible unique permutations - taken from
        https://stackoverflow.com/questions/6284396/permutations-with-unique-values
        """
        self.value = value
        self.occurrences = occurrences

def perm_unique(elements):
    """
    To find the possible unique permutations - taken from
    https://stackoverflow.com/questions/6284396/permutations-with-unique-values
    """
    eset=set(elements)
    listunique = [unique_element(i,elements.count(i)) for i in eset]
    u=len(elements)
    return perm_unique_helper(listunique,[0]*u,u-1)

def perm_unique_helper(listunique,result_list,d):
    """
    To find the possible unique permutations - taken from
    https://stackoverflow.com/questions/6284396/permutations-with-unique-values
    """
    if d < 0:
        yield tuple(result_list)
    else:
        for i in listunique:
            if i.occurrences > 0:
                result_list[d]=i.value
                i.occurrences-=1
                for g in  perm_unique_helper(listunique,result_list,d-1):
                    yield g
                i.occurrences+=1


# def capacities_for_graph(n, nodetypes, xcoords, ycoords, node_names, cold_bobs = True, beta = 1, gamma = 0, p = 1):
#     """
#     Calculate the total metric for the graph in question
#     :param n: no. of nodes in the graph: int
#     :param nodetypes: Array of the types of the nodes 0 == Source, 1 == Possible bob with no bob, 2 == Possible bob
#         with bob: list(n)
#     :param xcoords: The xcoordinates of the nodes: list(n)
#     :param ycoords: The ycoordinates of the nodes: list(n)
#     :param node_names: The names of the nodes: list(n)
#     :param cold_bob: Whether to use cold bob topologies or not: Boolean
#     :param beta: The coefficient for the centre of mass term : double
#     :param gamma: The coefficient for the general moment of inertia term: double
#     :param p: The order of the moment of inertia: int or double
#     :return: The metric for each possible orientation of Bob in the setup desired in form [(metric, perturbation of bobs,
#     capacity for each connection)]
#     """
#     # here we will look at how many Bobs and locations for Bobs there are in the network
#     no_of_bobs = 0
#     no_of_locations_for_bob = 0
#     for i in range(len(nodetypes)):
#         if nodetypes[i] == 2:
#             no_of_bobs += 1
#             no_of_locations_for_bob += 1
#         elif nodetypes[i] == 1:
#             no_of_locations_for_bob += 1
#     # generate an array of perturbations with the correct no. of Bobs and free locations
#     array_for_perturbation = np.concatenate((np.full(shape = no_of_bobs, fill_value = 2),
#                                             np.full(shape = no_of_locations_for_bob - no_of_bobs, fill_value = 1)))
#     # find all possible unique perturbations of this setup
#     perturbations = list(perm_unique(list(array_for_perturbation)))
#     # to store the requested data
#     output = []
#     # for each perturbation calculate the capacities and the metric
#     for perturbation in perturbations:
#         # set the nodes to the perturbation states
#         t_0 = time.time()
#         j = 0
#         for i in range(len(nodetypes)):
#             if nodetypes[i] == 1 or nodetypes[i] == 2:
#                 nodetypes[i] = perturbation[j]
#                 j += 1
#         # generate this graph
#         graph = Network_Setup(n=n, nodetypes=nodetypes, xcoords=xcoords,
#                                 ycoords=ycoords, node_names=
#                                 node_names)
#         # get the shortest distances between the nodes and each of the Bobs
#         t_2 = time.time()
#         true_distance = graph.get_shortest_distance_of_source_nodes()
#         t_3 = time.time()
#         print("Time for shortest path calculation:" + str(t_3 - t_2))
#         # get the shortest distance to any Bob
#         distance_array = get_minimum_length_for_any_source_pair(true_distance, graph.g)
#         # get the capacities of for each of the pairs of sources
#         capacity = calculate_capacity(distance_array, cold_bob=cold_bobs)
#         # calculate the metric
#         metric = general_metric(beta=beta, gamma=gamma, capacities=capacity, distances=distance_array, n=p)
#         # place the requested info into array
#         output.append((metric, perturbation, capacity))
#         t_1 = time.time()
#         print("Time for full perturbation: " + str(t_1 - t_0))
#     return output, graph

# def capacities_for_graph_using_m_shortest_connections(n, nodetypes, xcoords, ycoords, node_names, m , cold_bobs = True, beta = 1, gamma = 0, p = 1):
#     """
#     Calculate the total metric for the graph in question where the m shortest connections are used
#     :param n: no. of nodes in the graph: int
#     :param nodetypes: Array of the types of the nodes 0 == Source, 1 == Possible bob with no bob, 2 == Possible bob
#         with bob: list(n)
#     :param xcoords: The xcoordinates of the nodes: list(n)
#     :param ycoords: The ycoordinates of the nodes: list(n)
#     :param node_names: The names of the nodes: list(n)
#     :param m: no. of shortest connections used: int
#     :param cold_bob: Whether to use cold bob topologies or not: Boolean
#     :param beta: The coefficient for the centre of mass term : double
#     :param gamma: The coefficient for the general moment of inertia term: double
#     :param p: The order of the moment of inertia: int or double
#     :return: The metric for each possible orientation of Bob in the setup desired in form [(metric, perturbation of bobs,
#     capacity for each connection)]
#     """
#     # here we will look at how many Bobs and locations for Bobs there are in the network
#     no_of_bobs = 0
#     no_of_locations_for_bob = 0
#     for i in range(len(nodetypes)):
#         if nodetypes[i] == 2:
#             no_of_bobs += 1
#             no_of_locations_for_bob += 1
#         elif nodetypes[i] == 1:
#             no_of_locations_for_bob += 1
#     # generate an array of perturbations with the correct no. of Bobs and free locations
#     array_for_perturbation = np.concatenate((np.full(shape = no_of_bobs, fill_value = 2),
#                                             np.full(shape = no_of_locations_for_bob - no_of_bobs, fill_value = 1)))
#     # find all possible unique perturbations of this setup
#     perturbations = list(perm_unique(list(array_for_perturbation)))
#     # to store the requested data
#     output = []
#     # for each perturbation calculate the capacities and the metric
#     for perturbation in perturbations:
#         # set the nodes to the perturbation states
#         j = 0
#         for i in range(len(nodetypes)):
#             if nodetypes[i] == 1 or nodetypes[i] == 2:
#                 nodetypes[i] = perturbation[j]
#                 j += 1
#         # generate this graph
#         graph = Network_Setup(n=n, nodetypes=nodetypes, xcoords=xcoords,
#                                 ycoords=ycoords, node_names=
#                                 node_names)
#         # get the shortest distances between the nodes and each of the Bobs
#         true_distance = graph.get_shortest_distance_of_source_nodes()
#         # get the shortest distance to any Bob
#         distance_array = get_n_minimum_length_for_any_source_pair(true_distance, graph.g, n = m)
#         # get the capacities of for each of the pairs of sources
#         capacity = calculate_capacity_for_n_shortest_distance(distance_array, cold_bob=cold_bobs)
#         # separate out the distance_array:
#         distances = []
#         for source_node, target_node, detector_nodes, total_lengths, length_to_detectors in distance_array:
#             # get the capacity for each connection based on the asymmetric twin field calculation
#             for i in range(len(total_lengths)):
#                 distances.append((source_node, target_node, detector_nodes[i], total_lengths[i], length_to_detectors[i]))
#         # calculate the metric
#         metric = general_metric(beta=beta, gamma=gamma, capacities=capacity, distances=distances, n=p)
#         # place the requested info into array
#         output.append((metric, perturbation, capacity))
#     return output, graph



# def capacities_for_graph_efficient(n, nodetypes, xcoords, ycoords, node_names, cold_bobs = True, beta = 1, gamma = 0, p = 1):
#     """
#     Calculate the total metric for the graph in question - this uses precalculated capacities in files rates_coldbob_20_eff.csv
#     and rates_hotbob_20_eff.csv
#     :param n: no. of nodes in the graph: int
#     :param nodetypes: Array of the types of the nodes 0 == Source, 1 == Possible bob with no bob, 2 == Possible bob
#         with bob: list(n)
#     :param xcoords: The xcoordinates of the nodes: list(n)
#     :param ycoords: The ycoordinates of the nodes: list(n)
#     :param node_names: The names of the nodes: list(n)
#     :param cold_bob: Whether to use cold bob topologies or not: Boolean
#     :param beta: The coefficient for the centre of mass term : double
#     :param gamma: The coefficient for the general moment of inertia term: double
#     :param p: The order of the moment of inertia: int or double
#     :return: The metric for each possible orientation of Bob in the setup desired in form [(metric, perturbation of bobs,
#     capacity for each connection)]
#     """
#     # here we will look at how many Bobs and locations for Bobs there are in the network
#     no_of_bobs = 0
#     no_of_locations_for_bob = 0
#     for i in range(len(nodetypes)):
#         if nodetypes[i] == 2:
#             no_of_bobs += 1
#             no_of_locations_for_bob += 1
#         elif nodetypes[i] == 1:
#             no_of_locations_for_bob += 1
#     # generate an array of perturbations with the correct no. of Bobs and free locations
#     array_for_perturbation = np.concatenate((np.full(shape = no_of_bobs, fill_value = 2),
#                                             np.full(shape = no_of_locations_for_bob - no_of_bobs, fill_value = 1)))
#     # find all possible unique perturbations of this setup
#     perturbations = list(perm_unique(list(array_for_perturbation)))
#     # to store the requested data
#     output = []
#     # open and hold the precalculated capacities
#     dictionary = {}
#     if cold_bobs:
#         with open('rates_coldbob_20_eff.csv', mode='r') as csv_file:
#             csv_reader = csv.DictReader(csv_file)
#             line_count = 0
#             for row in csv_reader:
#                 if line_count == 0:
#                     print(f'Column names are {", ".join(row)}')
#                     line_count += 1
#                 dictionary["L" + row["L"] + "LB" + row["LB"]] = float(row['rate'])
#                 line_count += 1
#             print(f'Processed {line_count} lines.')
#     else:
#
#         with open('rates_hotbob_20_eff.csv', mode='r') as csv_file:
#             csv_reader = csv.DictReader(csv_file)
#             line_count = 0
#             for row in csv_reader:
#                 if line_count == 0:
#                     print(f'Column names are {", ".join(row)}')
#                     line_count += 1
#                 dictionary["L" + row["L"] + "LB" + row["LB"]] = float(row['rate'])
#                 line_count += 1
#             print(f'Processed {line_count} lines.')
#     # for each perturbation calculate the capacities and the metric
#     for perturbation in perturbations:
#         # set the nodes to the perturbation states
#         t_0 = time.time()
#         j = 0
#         for i in range(len(nodetypes)):
#             if nodetypes[i] == 1 or nodetypes[i] == 2:
#                 nodetypes[i] = perturbation[j]
#                 j += 1
#         # generate this graph
#         graph = Network_Setup(n=n, nodetypes=nodetypes, xcoords=xcoords,
#                                 ycoords=ycoords, node_names=
#                                 node_names)
#         # get the shortest distances between the nodes and each of the Bobs
#         t_2 = time.time()
#         true_distance = graph.get_shortest_distance_of_source_nodes()
#         t_3 = time.time()
#         print("Time for shortest path calculation:" + str(t_3 - t_2))
#         # get the shortest distance to any Bob
#         distance_array = get_minimum_length_for_any_source_pair(true_distance, graph.g)
#
#         # get the capacities of for each of the pairs of sources
#         capacity = calculate_capacity_efficient(distance_array, dictionary)
#         # calculate the metric
#         metric = general_metric(beta=beta, gamma=gamma, capacities=capacity, distances=distance_array, n=p)
#         # place the requested info into array
#         output.append((metric, perturbation, capacity))
#         t_1 = time.time()
#         print("Time for full perturbation: " + str(t_1 - t_0))
#     return output, graph


# def capacities_for_graph_using_m_shortest_connections_efficient(n, nodetypes, xcoords, ycoords, node_names, m , cold_bobs = True, beta = 1, gamma = 0, p = 1):
#     """
#     Calculate the total metric for the graph in question where the m shortest connections are used- this uses
#     precalculated capacities in files rates_coldbob.csvand rates_hotbob_20_eff.csv
#     :param n: no. of nodes in the graph: int
#     :param nodetypes: Array of the types of the nodes 0 == Source, 1 == Possible bob with no bob, 2 == Possible bob
#         with bob: list(n)
#     :param xcoords: The xcoordinates of the nodes: list(n)
#     :param ycoords: The ycoordinates of the nodes: list(n)
#     :param node_names: The names of the nodes: list(n)
#     :param m: no. of shortest connections used: int
#     :param cold_bob: Whether to use cold bob topologies or not: Boolean
#     :param beta: The coefficient for the centre of mass term : double
#     :param gamma: The coefficient for the general moment of inertia term: double
#     :param p: The order of the moment of inertia: int or double
#     :return: The metric for each possible orientation of Bob in the setup desired in form [(metric, perturbation of bobs,
#     capacity for each connection)]
#     """
#     # here we will look at how many Bobs and locations for Bobs there are in the network
#     no_of_bobs = 0
#     no_of_locations_for_bob = 0
#     for i in range(len(nodetypes)):
#         if nodetypes[i] == 2:
#             no_of_bobs += 1
#             no_of_locations_for_bob += 1
#         elif nodetypes[i] == 1:
#             no_of_locations_for_bob += 1
#     # generate an array of perturbations with the correct no. of Bobs and free locations
#     array_for_perturbation = np.concatenate((np.full(shape = no_of_bobs, fill_value = 2),
#                                             np.full(shape = no_of_locations_for_bob - no_of_bobs, fill_value = 1)))
#     # find all possible unique perturbations of this setup
#     perturbations = list(perm_unique(list(array_for_perturbation)))
#     # to store the requested data
#     output = []
#     # open and hold the precalculated capacities
#     dictionary = {}
#     if cold_bobs:
#         with open('rates_coldbob_20_eff.csv', mode='r') as csv_file:
#             csv_reader = csv.DictReader(csv_file)
#             line_count = 0
#             for row in csv_reader:
#                 if line_count == 0:
#                     print(f'Column names are {", ".join(row)}')
#                     line_count += 1
#                 dictionary["L" + row["L"] + "LB" + row["LB"]] = float(row['rate'])
#                 line_count += 1
#             print(f'Processed {line_count} lines.')
#     else:
#
#         with open('rates_hotbob_20_eff.csv', mode='r') as csv_file:
#             csv_reader = csv.DictReader(csv_file)
#             line_count = 0
#             for row in csv_reader:
#                 if line_count == 0:
#                     print(f'Column names are {", ".join(row)}')
#                     line_count += 1
#                 dictionary["L" + row["L"] + "LB" + row["LB"]] = float(row['rate'])
#                 line_count += 1
#             print(f'Processed {line_count} lines.')
#     # for each perturbation calculate the capacities and the metric
#     for perturbation in perturbations:
#         # set the nodes to the perturbation states
#         j = 0
#         for i in range(len(nodetypes)):
#             if nodetypes[i] == 1 or nodetypes[i] == 2:
#                 nodetypes[i] = perturbation[j]
#                 j += 1
#         # generate this graph
#         graph = Network_Setup(n=n, nodetypes=nodetypes, xcoords=xcoords,
#                                 ycoords=ycoords, node_names=
#                                 node_names)
#         # get the shortest distances between the nodes and each of the Bobs
#         true_distance = graph.get_shortest_distance_of_source_nodes()
#         # get the shortest distance to any Bob
#         distance_array = get_n_minimum_length_for_any_source_pair(true_distance, graph.g, n = m)
#         # get the capacities of for each of the pairs of sources
#         capacity = calculate_capacity_for_n_shortest_distance_efficient(distance_array, dictionary)
#         # separate out the distance_array:
#         distances = []
#         for source_node, target_node, detector_nodes, total_lengths, length_to_detectors in distance_array:
#             # get the capacity for each connection based on the asymmetric twin field calculation
#             for i in range(len(total_lengths)):
#                 distances.append((source_node, target_node, detector_nodes[i], total_lengths[i], length_to_detectors[i]))
#         # calculate the metric
#         metric = general_metric(beta=beta, gamma=gamma, capacities=capacity, distances=distances, n=p)
#         # place the requested info into array
#         output.append((metric, perturbation, capacity))
#     return output, graph




def capacities_for_graph_efficient_corrected(n, nodetypes, xcoords, ycoords, node_names, cold_bobs = True, beta = 1, gamma = 0, p = 0.8, q = 1, lengths_of_connections = None, no_connected_nodes = None, box_size = None, log_met = False, switches = True, hubspoke = False):
    """
    Calculate the total metric for the graph in question - this uses precalculated capacities in files rates_coldbob_20_eff.csv
    and rates_hotbob_20_eff.csv - this is the corrected version which accounts for all paths and finds the max capacity
    :param n: no. of nodes in the graph: int
    :param nodetypes: Array of the types of the nodes 0 == Source, 1 == Possible bob with no bob, 2 == Possible bob
        with bob: list(n)
    :param xcoords: The xcoordinates of the nodes: list(n)
    :param ycoords: The ycoordinates of the nodes: list(n)
    :param node_names: The names of the nodes: list(n)
    :param cold_bob: Whether to use cold bob topologies or not: Boolean
    :param beta: The coefficient for the centre of mass term : double
    :param gamma: The coefficient for the general moment of inertia term: double
    :param p: The connectivity of the network: int or double
    :param q: The order of the moment of inertia: int or double
    :param no_connected_nodes: The number of connected nodes average per connection if using geometric graph: default
    None - if you want to use geometric graph must pass this and box_size in
    :param box_size: The size of the box we're working with: default None - if you want to use geometric graph must pass
    this with no_connected_nodes in
    :param log_met: Boolean of whether to use a log metric instead of linear
    :param switches: whether or not to use the switches loss default True
    :return: The metric for each possible orientation of Bob in the setup desired in form [(metric, perturbation of bobs,
    capacity for each connection)]
    """
    # here we will look at how many Bobs and locations for Bobs there are in the network
    no_of_bobs = 0
    no_of_locations_for_bob = 0
    for i in range(len(nodetypes)):
        if nodetypes[i] == 2:
            no_of_bobs += 1
            no_of_locations_for_bob += 1
        elif nodetypes[i] == 1:
            no_of_locations_for_bob += 1
    # generate an array of perturbations with the correct no. of Bobs and free locations
    array_for_perturbation = np.concatenate((np.full(shape = no_of_bobs, fill_value = 2),
                                            np.full(shape = no_of_locations_for_bob - no_of_bobs, fill_value = 1)))
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
        t_0 = time.time()
        j = 0
        for i in range(len(nodetypes)):
            if nodetypes[i] == 1 or nodetypes[i] == 2:
                nodetypes[i] = perturbation[j]
                j += 1
        # generate this graph - if there is no lengths_of_connections use distances between nodes as edge lengths
        if no_connected_nodes == None or box_size == None:
            if lengths_of_connections == None:
                graph = Network_Setup(n=n, nodetypes=nodetypes, xcoords=xcoords,
                                        ycoords=ycoords, node_names=
                                        node_names, fully_connected = False, p = p)
            else:
                graph = Network_Setup_With_Edges_Input(n=n, nodetypes=nodetypes, xcoords=xcoords,
                                        ycoords=ycoords, node_names=
                                        node_names, lengths_of_connections = lengths_of_connections, fully_connected = False, p= p)
        else:
            if hubspoke == True:
                graph = Network_Setup(n=n, nodetypes=nodetypes, xcoords=xcoords,
                                      ycoords=ycoords, node_names=
                                      node_names, fully_connected=False, geometric_graph=True,
                                      no_of_connected_nodes=no_connected_nodes, box_size=box_size, hubspoke = True)
            else:
                graph = Network_Setup(n=n, nodetypes=nodetypes, xcoords=xcoords,
                                        ycoords=ycoords, node_names=
                                        node_names, fully_connected = False, geometric_graph = True, no_of_connected_nodes = no_connected_nodes, box_size = box_size)
        # get the shortest distances between the nodes and each of the Bobs
        t_2 = time.time()
        if switches:
            true_distance = graph.get_shortest_distance_of_source_nodes_with_switches()
        else:
            true_distance = graph.get_shortest_distance_of_source_nodes()
        t_3 = time.time()
        print("Time for shortest path calculation:" + str(t_3 - t_2))
        # get the shortest distance to any Bob
        distance_array = get_minimum_length_to_each_bob_for_any_source_pair(true_distance, graph.g)
        # get the capacities of for each of the pairs of sources
        capacity, distance_array = calculate_capacity_efficient_all_distances(distance_array, dictionary)
        # calculate the metric
        if log_met:
            metric = log_metric(capacity)
        else:
            metric = general_metric(beta=beta, gamma=gamma, capacities=capacity, distances=distance_array, n=q)
        # place the requested info into array
        output.append((metric, perturbation, capacity))
        t_1 = time.time()
        print("Time for full perturbation: " + str(t_1 - t_0))
    return output, graph


def capacities_for_graph_using_m_shortest_connections_efficient_corrected(n, nodetypes, xcoords, ycoords, node_names, m , cold_bobs = True, beta = 1, gamma = 0, p = 0.8, q = 1,  lengths_of_connections = None, no_connected_nodes = None, box_size = None, log_met = False, switches  = True):
    """
    Calculate the total metric for the graph in question where the m shortest connections are used- this uses
    precalculated capacities in files rates_coldbob_20_eff.csv and rates_hotbob_20_eff.csv - this is the corrected version which
    accounts for all paths and finds the max capacity
    :param n: no. of nodes in the graph: int
    :param nodetypes: Array of the types of the nodes 0 == Source, 1 == Possible bob with no bob, 2 == Possible bob
        with bob: list(n)
    :param xcoords: The xcoordinates of the nodes: list(n)
    :param ycoords: The ycoordinates of the nodes: list(n)
    :param node_names: The names of the nodes: list(n)
    :param m: no. of shortest connections used: int
    :param cold_bob: Whether to use cold bob topologies or not: Boolean
    :param beta: The coefficient for the centre of mass term : double
    :param gamma: The coefficient for the general moment of inertia term: double
    :param q:- The order of the moment of inertia: int or double
    :param p: The connectivity
    :param no_connected_nodes: The number of connected nodes average per connection if using geometric graph: default
    None - if you want to use geometric graph must pass this and box_size in
    :param box_size: The size of the box we're working with: default None - if you want to use geometric graph must pass
    this with no_connected_nodes in
    :param log_met: Boolean of whether to use a log metric instead of linear
    :param switches: whether or not to use the switches loss default True
    :return: The metric for each possible orientation of Bob in the setup desired in form [(metric, perturbation of bobs,
    capacity for each connection)]
    """
    # here we will look at how many Bobs and locations for Bobs there are in the network
    no_of_bobs = 0
    no_of_locations_for_bob = 0
    for i in range(len(nodetypes)):
        if nodetypes[i] == 2:
            no_of_bobs += 1
            no_of_locations_for_bob += 1
        elif nodetypes[i] == 1:
            no_of_locations_for_bob += 1
    # generate an array of perturbations with the correct no. of Bobs and free locations
    array_for_perturbation = np.concatenate((np.full(shape = no_of_bobs, fill_value = 2),
                                            np.full(shape = no_of_locations_for_bob - no_of_bobs, fill_value = 1)))
    # find all possible unique perturbations of this setup
    perturbations = list(perm_unique(list(array_for_perturbation)))
    # to store the requested data
    output = []
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
        for i in range(len(nodetypes)):
            if nodetypes[i] == 1 or nodetypes[i] == 2:
                nodetypes[i] = perturbation[j]
                j += 1
        # generate this graph
        if no_connected_nodes == None or box_size == None:
            if lengths_of_connections == None:
                graph = Network_Setup(n=n, nodetypes=nodetypes, xcoords=xcoords,
                                        ycoords=ycoords, node_names=
                                        node_names, p = p)
            else:
                graph = Network_Setup(n=n, nodetypes=nodetypes, xcoords=xcoords,
                                      ycoords=ycoords, node_names=
                                      node_names, lengths_of_connections = lengths_of_connections, p = p)
        else:
            graph = Network_Setup(n=n, nodetypes=nodetypes, xcoords=xcoords,
                                  ycoords=ycoords, node_names=
                                  node_names, fully_connected=False, geometric_graph=True,
                                  no_of_connected_nodes=no_connected_nodes, box_size=box_size)
        # get the shortest distances between the nodes and each of the Bobs
        if switches:
            true_distance = graph.get_shortest_distance_of_source_nodes_with_switches()
        else:
            true_distance = graph.get_shortest_distance_of_source_nodes()
        # get the shortest distance to any Bob
        distance_array = get_minimum_length_to_each_bob_for_any_source_pair(true_distance, graph.g)
        # get the capacities of for each of the pairs of sources
        capacity, distance_array = calculate_capacity_for_n_highest_capacities_efficient_corrected(distance_array, dictionary,n=m)
        # calculate the metric
        if log_met:
            metric = log_metric(capacity)
        else:
            metric = general_metric(beta=beta, gamma=gamma, capacities=capacity, distances=distance_array, n=q)
        # place the requested info into array
        output.append((metric, perturbation, capacity))
    return output, graph


def capacities_for_graph_with_node_features(n, nodetypes, xcoords, ycoords, node_names, p_connection, cold_bobs = True, beta = 1, gamma = 0, p = 0.8, q = 1):
    """
    Calculate the total metric for the graph in question - this uses precalculated capacities in files rates_coldbob_20_eff.csv
    and rates_hotbob_20_eff.csv - this is the corrected version which accounts for all paths and finds the max capacity with features
    :param n: no. of nodes in the graph: int
    :param nodetypes: Array of the types of the nodes 0 == Source, 1 == Possible bob with no bob, 2 == Possible bob
        with bob: list(n)
    :param xcoords: The xcoordinates of the nodes: list(n)
    :param ycoords: The ycoordinates of the nodes: list(n)
    :param node_names: The names of the nodes: list(n)
    :param p_connection: The probability that the nodes are connected
    :param cold_bob: Whether to use cold bob topologies or not: Boolean
    :param beta: The coefficient for the centre of mass term : double
    :param gamma: The coefficient for the general moment of inertia term: double
    :param p: The connectivity of the network: int or double
    :param q: The order of the moment of inertia: int or double
    :return: The metric for each possible orientation of Bob in the setup desired in form [(metric, perturbation of bobs,
    capacity for each connection)]
    """
    # here we will look at how many Bobs and locations for Bobs there are in the network
    no_of_bobs = 0
    no_of_locations_for_bob = 0
    for i in range(len(nodetypes)):
        if nodetypes[i] == 2:
            no_of_bobs += 1
            no_of_locations_for_bob += 1
        elif nodetypes[i] == 1:
            no_of_locations_for_bob += 1
    # generate an array of perturbations with the correct no. of Bobs and free locations
    array_for_perturbation = np.concatenate((np.full(shape = no_of_bobs, fill_value = 2),
                                            np.full(shape = no_of_locations_for_bob - no_of_bobs, fill_value = 1)))
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
        t_0 = time.time()
        j = 0
        for i in range(len(nodetypes)):
            if nodetypes[i] == 1 or nodetypes[i] == 2:
                nodetypes[i] = perturbation[j]
                j += 1
        # generate this graph -
        graph = Network_Setup_with_priority_and_random_desired_connections(n=n, nodetypes=nodetypes, xcoords=xcoords,
                                ycoords=ycoords, node_names=
                                node_names, fully_connected = False, p = p, p_customer = p_connection)
        # get the shortest distances between the nodes and each of the Bobs
        t_2 = time.time()
        true_distance = graph.get_shortest_distance_of_source_nodes()
        t_3 = time.time()
        print("Time for shortest path calculation:" + str(t_3 - t_2))
        # get the shortest distance to any Bob
        distance_array = get_minimum_length_to_each_bob_for_any_source_pair(true_distance, graph.g)
        # get the capacities of for each of the pairs of sources
        capacity, distance_array = calculate_capacity_efficient_all_distances_node_features(distance_array, dictionary, graph)
        # calculate the metric

        ##### need new metric to calculate
        metric = general_metric_node_features(beta=beta, gamma=gamma, capacities=capacity, distances=distance_array, n=q, graph = graph)
        # place the requested info into array
        output.append((metric, perturbation, capacity))
        t_1 = time.time()
        print("Time for full perturbation: " + str(t_1 - t_0))
    return output, graph





def server_client_rate(graph, cold_bobs = True, switches = True):
    """
    Get the rates for each connection and the total rates for the graph for the server client protocol
    :param graph: The graph to use
    :param cold_bobs: whether to use cold bobs or not
    :param switches: whether to include switches loss
    :return: rate per connection and total_rates
    """
    dictionary = {}
    if cold_bobs:
        with open('rates_coldbob_bb84_20_eff.csv', mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                dictionary["L" + str(round(float(row["L"]), 2))] = float(row['rate'])
                line_count += 1
            print(f'Processed {line_count} lines.')
    else:
        with open('rates_hotbob_bb84_20_eff.csv', mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                dictionary["L" + str(round(float(row["L"]), 2))] = float(row['rate'])
                line_count += 1
            print(f'Processed {line_count} lines.')
    rates, total_rate =total_rate_server_client_efficient(graph, dictionary, switches)
    return rates, total_rate


def get_hot_tfqkd_rates_solution(graph, switches = True):
    """
    Get the rates for each connection and the total rates for the graph for the TF-QKD protocol with hot Bobs
    here we assume we can place the Bobs in the centre of each path with negligible cost - requires 1/2N(N-1)
    Bobs total
    :param graph: The graph to use
    :param switches: whether to include loss due to switches: default True
    :return: rate per connection and total_rates
    """
    dictionary = {}
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
    rates, total_rate = total_rate_tf_efficient(graph, dictionary, switches)
    return rates, total_rate