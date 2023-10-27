from graph_tool.all import *
from utils_graph import generate_fully_connected_graph, get_length, generate_random_graph
import numpy as np
from rates_estimates.utils import get_rate
import time
import copy

def calculate_capacity(distance_array, cold_bob):
    """
    This will calculate the capacities of the system for AsymmetricTwinField given an array of distances and based
    on the use of cold_bobs or not
    :param distance_array: The set of the distances of the network is a list [(source, target, detector, total length,
    length to detector)]
    :param cold_bob: Whether to use cold bob topologies or not: Boolean
    :return: The capacities of the connections as a list [(source, target, detector, capacity)]
    """
    capacities = []
    t_0 = time.time()
    for source_node, target_node, detector_node, total_length, length_to_detector in distance_array:
        # get the capacity for each connection based on the asymmetric twin field calculation

        capacity = get_rate(total_length, protocol = "AsymmetricTwinField", coldalice = False, coldbob = cold_bob, length_1 = length_to_detector)
        capacities.append((source_node, target_node, detector_node, capacity))
    t_1 = time.time()
    print("Time for capacity calculation:" + str(t_1 - t_0))
    return capacities

def calculate_capacity_efficient(distance_array, dictionary):
    """
    This will calculate the capacities of the system for AsymmetricTwinField given an array of distances and based
    on the use of cold_bobs or not - this is done efficiently based on the lookup dictionary calculated previously
    (cold_bob comes in to the data in the dictionary)
    :param distance_array: The set of the distances of the network is a list [(source, target, detector, total length,
    length to detector)]
    :param dictionary: The dictionary of the capacities for a given length and a given length to bob: dictionary {float}
    :return: The capacities of the connections as a list [(source, target, detector, capacity)]
    """
    capacities = []
    t_0 = time.time()
    for source_node, target_node, detector_node, total_length, length_to_detector in distance_array:
        # get the capacity for each connection based on the asymmetric twin field calculation
        total_length = int(round(total_length))
        length_to_detector = int(round(length_to_detector))
        # capacities are in the dictionary - simply extract them from the dictionary
        capacity = dictionary["L" + str(total_length) + "LB" + str(length_to_detector)]
        capacities.append((source_node, target_node, detector_node, capacity))
    t_1 = time.time()
    print("Time for capacity calculation:" + str(t_1 - t_0))
    return capacities

def calculate_capacity_for_n_shortest_distance(distance_array, cold_bob):
    """
    This will calculate the capacities of the system for AsymmetricTwinField given an array of distances and based
    on the use of cold_bobs or not - in this case we consider the n shortest distances
    :param distance_array: The set of the distances of the network is a list [(source, target, detector, total length,
    length to detector)]
    :param cold_bob: Whether to use cold bob topologies or not: Boolean
    :return: The capacities of the connections as a list [(source, target, detector, capacity)]
    """
    capacities = []
    for source_node, target_node, detector_nodes, total_lengths, length_to_detectors in distance_array:
        # get the capacity for each connection based on the asymmetric twin field calculation
        for i in range(len(total_lengths)):

            capacity = get_rate(total_lengths[i], protocol = "AsymmetricTwinField", coldalice = False, coldbob = cold_bob, length_1 = length_to_detectors[i])
            capacities.append((source_node, target_node, detector_nodes[i], capacity))
    return capacities


def calculate_capacity_for_n_shortest_distance_efficient(distance_array, dictionary):
    """
    This will calculate the capacities of the system for AsymmetricTwinField given an array of distances and based
    on the use of cold_bobs or not - in this case we consider the n shortest distances - this is done efficiently based
    on the lookup dictionary calculated previously
    (cold_bob comes in to the data in the dictionary)
    :param distance_array: The set of the distances of the network is a list [(source, target, detector, total length,
    length to detector)]
    :param dictionary: The dictionary of the capacities for a given length and a given length to bob: dictionary {float}
    :return: The capacities of the connections as a list [(source, target, detector, capacity)]
    """
    capacities = []
    for source_node, target_node, detector_nodes, total_lengths, length_to_detectors in distance_array:
        # get the capacity for each connection based on the asymmetric twin field calculation
        for i in range(len(total_lengths)):
            # get the capacity for each connection based on the asymmetric twin field calculation
            total_length = int(round(total_lengths[i]))
            length_to_detector = int(round(length_to_detectors[i]))
            # capacities are in the dictionary - simply extract them from the dictionary
            capacity = dictionary["L" + str(total_length) + "LB" + str(length_to_detector)]
            # capacity = get_rate(total_lengths[i], protocol = "AsymmetricTwinField", coldalice = False, coldbob = cold_bob, length_1 = length_to_detectors[i])
            capacities.append((source_node, target_node, detector_nodes[i], capacity))
    return capacities


def calculate_capacity_efficient_all_distances(distance_array, dictionary):
    """
    This will calculate the capacities of the system for AsymmetricTwinField given an array of distances for each bob
    and based on the use of cold_bobs or not - this is done efficiently based on the lookup dictionary calculated
    previously (cold_bob comes in to the data in the dictionary) - this is the corrected algorithm with finding the
    maximum after calculating the capacity for all possible Bobs
    :param distance_array: The set of the distances of the network is a list [(source, target, detectors, total lengths,
    length to detectors)]
    :param dictionary: The dictionary of the capacities for a given length and a given length to bob: dictionary {float}
    :return: The capacities of the connections as a list [(source, target, detector, capacity)] and the distance array
            as a list [(source, target, detector, total length, length to detector)]
    """
    # array to hold the capacities for each bob and the new expanded distance array
    capacities = []
    t_0 = time.time()
    distances_array = []
    # for all sources and target nodes and for all bobs in the array - calculate capacities and find the biggest value
    # of these
    for source_node, target_node, detector_nodes, total_lengths, length_to_detectors in distance_array:
        # to hold the current largest capacity and the detector for which this capacity corresponds to
        current_max_capacity = 0.0
        detector_node = 0
        for i in range(len(detector_nodes)):
            # get the capacity for each connection based on the asymmetric twin field calculation
            total_length = int(round(total_lengths[i]))
            length_to_detector = int(round(length_to_detectors[i]))
            if total_length > 990:
                capacity = 0.0
            else:
                # from the look-up table
                capacity = dictionary["L" + str(total_length) + "LB" + str(length_to_detector)]
            # replace with current capacity if it is bigger than the old one - also replace the detector node location
            if current_max_capacity < capacity:
                current_max_capacity = capacity
                detector_node = i
        # append each of these terms to the capacities and the appropriate distances to the distance array
        capacities.append((source_node, target_node, detector_nodes[detector_node], current_max_capacity))
        distances_array.append((source_node, target_node, detector_nodes[detector_node], total_lengths[detector_node], length_to_detectors[detector_node]))
    t_1 = time.time()
    print("Time for capacity calculation:" + str(t_1 - t_0))
    return capacities, distances_array

def calculate_capacity_for_n_highest_capacities_efficient_corrected(distance_array, dictionary, n):
    """
    This will calculate the capacities of the system for AsymmetricTwinField given an array of distances and based
    on the use of cold_bobs or not - in this case we consider the n best capacities - this is done efficiently based
    on the lookup dictionary calculated previously- this is the corrected algorithm with finding the
    maximum after calculating the capacity for all possible Bobs
    (cold_bob comes in to the data in the dictionary)
    :param distance_array: The set of the distances of the network is a list [(source, target, detector, total length,
    length to detector)]
    :param dictionary: The dictionary of the capacities for a given length and a given length to bob: dictionary {float}
    :return: The capacities of the connections as a list [(source, target, detector, capacity)] and the distance array
            as a list [(source, target, detector, total length, length to detector)]
    """
    # array to hold the capacities for each bob and the new expanded distance array
    capacities = []
    distances_array = []
    for source_node, target_node, detector_nodes, total_lengths, length_to_detectors in distance_array:
        # array to hold the best capacities in order of best to worse
        capacities_best = []
        for i in range(len(total_lengths)):
            # get the capacity for each connection based on the asymmetric twin field calculation
            total_length = int(round(total_lengths[i]))
            length_to_detector = int(round(length_to_detectors[i]))
            if total_length > 990:
                capacity = 0.0
            else:
                capacity = dictionary["L" + str(total_length) + "LB" + str(length_to_detector)]
            # capacity = get_rate(total_lengths[i], protocol = "AsymmetricTwinField", coldalice = False, coldbob = cold_bob, length_1 = length_to_detectors[i])
            # if there are too few elements add the next one and sort in terms of the capacity to have highest capacity
            # at end
            if len(capacities_best)< n:
                capacities_best.append([capacity,detector_nodes[i],total_length, length_to_detector])
                capacities_best.sort(key = lambda x: x[0])
            else:
                # check from largest element to smallest if one element is smaller than terms replace smallest element
                # and sort
                if capacity > capacities_best[0][0]:
                    capacities_best[0][0] = copy.deepcopy(capacity)
                    capacities_best[0][1] = detector_nodes[i]
                    capacities_best[0][2] = copy.deepcopy(total_length)
                    capacities_best[0][3] = copy.deepcopy(length_to_detector)
                    capacities_best.sort(key=lambda x: x[0])
        # add the best elements into capacities and keep their distance information too
        for j in range(len(capacities_best)):
            capacities.append((source_node, target_node, capacities_best[j][1], capacities_best[j][0]))
            distances_array.append((source_node, target_node, capacities_best[j][1], capacities_best[j][2], capacities_best[j][3]))
    return capacities, distances_array


def order_paths_in_terms_of_capacities(total_lengths, length_to_detectors, dictionary):
    """
    Arranges the paths in order of decreasing capacity pairs - excludes paths that have already been used (i.e. orders
    the set of unique connections)
    :param total_lengths: {key = [path_source, path_detector] : length}
    :param length_to_detectors: {key = [path_source, path_detector]: length}
    :param dictionary: The dictionary of the capacities for a given length and a given length to bob: dictionary {float}
    :return: {key = [path_source, path_detector]: (capacity, total_length, length_to_detector)}
    """
    capacities = {}
    capacities_for_keys = {}
    lengths = {}
    for i,j in total_lengths.keys():
        total_length = int(round(total_lengths[i,j]))
        length_to_detector = int(round(length_to_detectors[i,j]))
        if total_length > 1190:
            capacity = 0.0
        else:
            capacity = dictionary["L" + str(total_length) + "LB" + str(length_to_detector)]
        capacities[i,j] = capacity
        lengths[i,j] = (total_length, length_to_detector)
    capacities_sorted = dict(sorted(capacities.items(), key=lambda item: item[1],  reverse=True))
    used_paths_source = []
    used_paths_target = []
    for i,j in capacities_sorted.keys():
        if i not in used_paths_source and j not in used_paths_target:
            capacities_for_keys[i,j] = (capacities_sorted[i,j], lengths[i,j][0], lengths[i,j][1])
            used_paths_source.append(i)
            used_paths_target.append(j)
    return capacities_for_keys



def calculate_capacity_for_k_highest_capacities_multiple_paths_per_detector_allowed(distance_array, dictionary, k):
    """
    This will calculate the capacities of the system for AsymmetricTwinField given an array of distances and based
    on the use of cold_bobs or not - in this case we consider the n best capacities - this is done efficiently based
    on the lookup dictionary calculated previously- this is the corrected algorithm with finding the
    maximum after calculating the capacity for all possible Bobs
    (cold_bob comes in to the data in the dictionary)
    Only k highest capacity paths per [source, target] pair are in the solutions array.
    :param distance_array: The set of the distances of the network is a list [(source, target, detector[detectors], total length-{key= detector :{key = [path_source, path_detector] : length}},
    length to detector- {key = detector: {key = [path_source, path_detector] : length}})]
    :param dictionary: The dictionary of the capacities for a given length and a given length to bob: dictionary {float}
    :return: The capacities of the connections as a list [(source, target, detector, capacity)] and the distance array
            as a list [(source, target, detector, total length, length to detector)]
    """
    # array to hold the capacities for each bob and the new expanded distance array
    capacities = []
    distances_array = []
    for source_node, target_node, detector_nodes, total_lengths, length_to_detectors in distance_array:
        # array to hold the best capacities in order of best to worse
        capacities_best = []
        for key in total_lengths.keys():
            capacities_for_current_detector = order_paths_in_terms_of_capacities(total_lengths[key], length_to_detectors[key], dictionary)
            for i,j in capacities_for_current_detector.keys():
                if len(capacities_best) < k:
                    capacities_best.append([capacities_for_current_detector[i,j][0], key, capacities_for_current_detector[i,j][1], capacities_for_current_detector[i,j][2]])
                    capacities_best.sort(key=lambda x: x[0])
                else:
                    # check from largest element to smallest if one element is smaller than terms replace smallest element
                    # and sort
                    if capacities_for_current_detector[i,j][0] > capacities_best[0][0]:
                        capacities_best[0][0] = copy.deepcopy(capacities_for_current_detector[i,j][0])
                        capacities_best[0][1] = key  # detector node
                        capacities_best[0][2] = copy.deepcopy(capacities_for_current_detector[i,j][1])
                        capacities_best[0][3] = copy.deepcopy(capacities_for_current_detector[i,j][2])
                        capacities_best.sort(key=lambda x: x[0])
        for j in range(len(capacities_best)):
            capacities.append((source_node, target_node, capacities_best[j][1], capacities_best[j][0]))
            distances_array.append(
                (source_node, target_node, capacities_best[j][1], capacities_best[j][2], capacities_best[j][3]))
    return capacities, distances_array


def calculate_capacities_for_k_highest_connections_bb84(distance_array, dictionary, k):
    rates = []
    for source_node, target_node, distances in distance_array:
        for distance in distances:
            distance_actual = round(distance, 2)
            if distance_actual > 999:
                capacity = 0.0
            else:
                # from the look-up table
                capacity = dictionary["L" + str(distance_actual)]
            if capacity > 0.00000001:
                rates.append((source_node, target_node, 0, capacity))
    return rates


def calculate_capacity_efficient_all_distances_node_features(distance_array, dictionary, graph):
    """
    This will calculate the capacities of the system for AsymmetricTwinField given an array of distances for each bob
    and based on the use of cold_bobs or not - this is done efficiently based on the lookup dictionary calculated
    previously (cold_bob comes in to the data in the dictionary) - this is the corrected algorithm with finding the
    maximum after calculating the capacity for all possible Bobs
    :param distance_array: The set of the distances of the network is a list [(source, target, detectors, total lengths,
    length to detectors)]
    :param dictionary: The dictionary of the capacities for a given length and a given length to bob: dictionary {float}
    :param graph: the graph of type Network_Setup_with_priority_and_random_desired_connections
    :return: The capacities of the connections as a list [(source, target, detector, capacity)] and the distance array
            as a list [(source, target, detector, total length, length to detector)]
    """
    # array to hold the capacities for each bob and the new expanded distance array
    capacities = []
    t_0 = time.time()
    distances_array = []
    # for all sources and target nodes and for all bobs in the array - calculate capacities and find the biggest value
    # of these
    for source_node, target_node, detector_nodes, total_lengths, length_to_detectors in distance_array:
        # to hold the current largest capacity and the detector for which this capacity corresponds to
        current_max_capacity = 0.0
        detector_node = 0
        if str(source_node) + "-" + str(target_node) in graph.connections:
            if graph.connections[str(source_node) + "-" + str(target_node)] == 1:
                for i in range(len(detector_nodes)):
                    # get the capacity for each connection based on the asymmetric twin field calculation
                    total_length = int(round(total_lengths[i]))
                    length_to_detector = int(round(length_to_detectors[i]))
                    # from the look-up table
                    capacity = dictionary["L" + str(total_length) + "LB" + str(length_to_detector)]
                    # replace with current capacity if it is bigger than the old one - also replace the detector node location
                    if current_max_capacity < capacity:
                        current_max_capacity = capacity
                        detector_node = i
        else:
            if graph.connections[str(target_node) + "-" + str(source_node)] == 1:
                for i in range(len(detector_nodes)):
                    # get the capacity for each connection based on the asymmetric twin field calculation
                    total_length = int(round(total_lengths[i]))
                    length_to_detector = int(round(length_to_detectors[i]))
                    # from the look-up table
                    capacity = dictionary["L" + str(total_length) + "LB" + str(length_to_detector)]
                    # replace with current capacity if it is bigger than the old one - also replace the detector node location
                    if current_max_capacity < capacity:
                        current_max_capacity = capacity
                        detector_node = i
        # append each of these terms to the capacities and the appropriate distances to the distance array
        capacities.append((source_node, target_node, detector_nodes[detector_node], current_max_capacity))
        distances_array.append((source_node, target_node, detector_nodes[detector_node], total_lengths[detector_node], length_to_detectors[detector_node]))
    t_1 = time.time()
    print("Time for capacity calculation:" + str(t_1 - t_0))
    return capacities, distances_array