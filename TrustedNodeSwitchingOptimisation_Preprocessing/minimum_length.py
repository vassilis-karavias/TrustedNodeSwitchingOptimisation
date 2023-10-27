from graph_tool.all import *
from utils_graph import generate_fully_connected_graph, get_length, generate_random_graph
import numpy as np
from generate_graph import Network_Setup, NodeType


def get_minimum_length_via_specific_bob(node_a, node_b, bob, distance_list):
    """
    Get the min length between a pair of nodes via a given node bob
    :param node_a: initial node location in array: int
    :param node_b: final node location in array: int
    :param bob: The Bob we are interested in: int/string
    :param distance_list: The dictionary of EdgePropertyMaps that gives the distances
    :return: total length (L_1+L_2) and length to bob (L_1)
    """
    # find the length of node_a from Bob
    length_a = distance_list[bob].a[node_a]
    # find the length of node_b from Bob
    length_b = distance_list[bob].a[node_b]
    # check if the nodes are two Bobs - if there are no keys can be generated
    if length_a == np.infty or length_b == np.infty:
        print("Cannot generate keys between two detectors")
        raise ValueError
    else:
        # return total length and length to one bob
        return length_a + length_b, length_a



def get_minimum_length_for_any_bob(node_a, node_b, distance_list):
    """
    Get the minimum length for a single bob for all bobs for a given pair of nodes
    :param node_a: initial node location in array: int
    :param node_b: final node location in array: int
    :param distance_list: The dictionary of EdgePropertyMaps that gives the distances
    :return: the mimimum total distance (L_1+L_2), the position of the Bob at this minimum:int, the length to bob
    from one node (L_1)
    """
    # arrays to keep track of the total distance and the distance to bob from one node
    distance_array = []
    distance_to_detector = []
    # for every element in the dictionary find the smallest distance to this bob and distance to bob from one node
    # add these to the arrays
    for key in distance_list:
        distance_i, distance_1 = get_minimum_length_via_specific_bob(node_a, node_b, key, distance_list)
        distance_array.append(distance_i)
        distance_to_detector.append(distance_1)
    # find the position of the minimum of this minimum
    min_pos = np.argmin(distance_array)
    return distance_array[min_pos], min_pos, distance_to_detector[min_pos]

def get_minimum_length_for_any_source_pair(distance_list, graph):
    """
    Get the minimum distance and additional information for each pair of nodes
    :param distance_list: The dictionary of EdgePropertyMaps that gives the distances
    :param graph: The Graph we are considering
    :return: An array with the structure: [(source, target, bob for min, min total distance, distance to detector)]
    """
    # to keep all the required information
    minimum_distance_array = []
    # The first Bob in the list - needed to carry out the for loop
    first_bob = list(distance_list.keys())[0]
    # for every element in the distance dictionary
    for i in range(len(distance_list[first_bob].a)):
        for j in range(i, len(distance_list[first_bob].a)):
            # if the nodes are both sources and they are not the same source then calculate all the required info and
            # place it into the array
            if graph.vertex_properties["node_type"][i] == NodeType(0).name \
                and graph.vertex_properties["node_type"][j] == NodeType(0).name and i != j:
                minimum_distance, bob_for_min, distance_to_detector = get_minimum_length_for_any_bob(i,j, distance_list)
                source_vertex = graph.vertex_properties["name"][i]
                target_vertex = graph.vertex_properties["name"][j]
                bob_minimum = graph.vertex_properties["name"][list(distance_list.keys())[bob_for_min]]
                minimum_distance_array.append((source_vertex, target_vertex, bob_minimum, minimum_distance, distance_to_detector))
    return minimum_distance_array


def get_minimum_to_any_bob(node_a, node_b, distance_list):
    """
    Get the minimum length for all bobs for a given pair of nodes
    :param node_a: initial node location in array: int
    :param node_b: final node location in array: int
    :param distance_list: The dictionary of EdgePropertyMaps that gives the distances
    :return: the mimimum total distance (L_1+L_2), the position of the Bob at this minimum:int, the length to bob
    from one node (L_1)
    """
    # arrays to keep track of the total distance and the distance to bob from one node
    distance_array = []
    distance_to_detector = []
    # for every element in the dictionary find the smallest distance to this bob and distance to bob from one node
    # add these to the arrays
    for key in distance_list:
        distance_i, distance_1 = get_minimum_length_via_specific_bob(node_a, node_b, key, distance_list)
        distance_array.append(distance_i)
        distance_to_detector.append(distance_1)
    # find the position of the minimum of this minimum
    # min_pos = np.argmin(distance_array)
    return distance_array, distance_to_detector



def get_k_minimum_length_via_specific_bob(node_a, node_b, bob, distance_list):
    """
    Get the total_lengths and lengths_to_a_bob for each of the k paths in dictionaries {key [path_1,path_2]: length}
    This ensures we get the label for which path used in each element
    :param node_a: initial node location in array: int
    :param node_b: final node location in array: int
    :param bob: The Bob we are interested in: int/string
    :param distance_list: dictionary of path lengths: {key [detector_node, other_node]: [distance of paths for path in k_shortest_paths]}
    :return: dictionaries of total lengths and lengths to bob using for {key [path_a, path_b]: length}
    """
    total_lengths = {}
    lengths_a_to_bob = {}
    # find the lengths of node_a from Bob
    lengths_a = distance_list[bob][node_a]
    # find the lengths of node_b from Bob
    lengths_b = distance_list[bob][node_b]
    # check if the nodes are two Bobs - if there are no keys can be generated
    for i in range(len(lengths_a)):
        for j in range(len(lengths_b)):
            total_lengths[i,j] = lengths_a[i] + lengths_b[j]
            lengths_a_to_bob[i,j] = lengths_a[i]
    return total_lengths, lengths_a_to_bob

def get_k_minimum_to_any_bob(node_a, node_b, distance_list):
    """
     Get the total_lengths and lengths_to_a_bob for each of the k paths in dictionaries {key bob :{key [path_1,path_2]: length}}
    This ensures we get the label for which path used in each element. This is obtained for all bobs.
    :param node_a: initial node location in array: int
    :param node_b: final node location in array: int
    :param distance_list: dictionary of path lengths: {key [detector_node, other_node]: [distance of paths for path in k_shortest_paths]}
    :return: dictionaries of total lengths and lengths to bob using {key bob: {key [path_a, path_b]: length}}
    """
    # arrays to keep track of the total distance and the distance to bob from one node
    distance_array = {}
    distance_to_detector = {}
    # for every element in the dictionary find the smallest distance to this bob and distance to bob from one node
    # add these to the arrays
    for key in distance_list:
        distance_i, distance_1 = get_k_minimum_length_via_specific_bob(node_a, node_b, key, distance_list)
        distance_array[key] = distance_i
        distance_to_detector[key] = distance_1
        # find the position of the minimum of this minimum
        # min_pos = np.argmin(distance_array)
    return distance_array, distance_to_detector









def get_minimum_length_to_each_bob_for_any_source_pair(distance_list, graph):
    """
    Get the minimum distance and additional information for each pair of nodes
    :param distance_list: The dictionary of EdgePropertyMaps that gives the distances
    :param graph: The Graph we are considering
    :return: An array with the structure: [(source, target, bobs for min, min total distance for each bob, distance to detectors)]
    """
    # to keep all the required information
    minimum_distance_array = []
    # The first Bob in the list - needed to carry out the for loop
    first_bob = list(distance_list.keys())[0]
    # for every element in the distance dictionary
    for i in range(len(distance_list[first_bob].a)):
        for j in range(i, len(distance_list[first_bob].a)):
            # if the nodes are both sources and they are not the same source then calculate all the required info and
            # place it into the array
            if graph.vertex_properties["node_type"][i] == NodeType(0).name \
                and graph.vertex_properties["node_type"][j] == NodeType(0).name and i != j:
                # keep track of all the minimum distances and the distances to detector for each detector
                minimum_distances, distance_to_detectors = get_minimum_to_any_bob(i,j, distance_list)
                source_vertex = graph.vertex_properties["name"][i]
                target_vertex = graph.vertex_properties["name"][j]
                # add the array of the bobs keeping track of the correct bob for each distance added into the array of
                # minimum distances
                bobs_minimum = []
                for k in list(distance_list.keys()):
                    bobs_minimum.append(graph.vertex_properties["name"][k])
                # add all requested information
                minimum_distance_array.append((source_vertex, target_vertex, bobs_minimum, minimum_distances, distance_to_detectors))
    return minimum_distance_array




def get_n_minimum_lengths_for_any_bob(node_a, node_b, distance_list, n):
    """
    Get the n shortest lengths for all bobs for a given pair of nodes
    :param node_a: initial node location in array: int
    :param node_b: final node location in array: int
    :param distance_list: The dictionary of EdgePropertyMaps that gives the distances
    :param n: the number of minimum lengths to look for: int
    :return: the mimimum total distance (L_1+L_2), the position of the Bob at this minimum:int, the length to bob
    from one node (L_1)
    """
    # arrays to keep track of the total distance and the distance to bob from one node
    distance_array = []
    distance_to_detector = []
    # for every element in the dictionary find the smallest distance to this bob and distance to bob from one node
    # add these to the arrays
    for key in distance_list:
        distance_i, distance_1 = get_minimum_length_via_specific_bob(node_a, node_b, key, distance_list)
        distance_array.append(distance_i)
        distance_to_detector.append(distance_1)
    # find the positions of the n smallest elements in the minimum
    min_posns = np.argpartition(distance_array, n)
    # add these elements into the arrays to be returned
    min_distance_array = []
    distance_to_detector_min = []
    min_posn = []
    for i in range(n):
        min_distance_array.append(distance_array[min_posns[i]])
        distance_to_detector_min.append(distance_to_detector[min_posns[i]])
        min_posn.append(min_posns[i])
    return min_distance_array, min_posn, distance_to_detector_min

def get_n_minimum_length_for_any_source_pair(distance_list, graph, n):
    """
    Get the n minimum distances and additional information for each pair of nodes
    :param distance_list: The dictionary of EdgePropertyMaps that gives the distances
    :param graph: The Graph we are considering
    :param n: the number of minimum lengths to look for: int
    :return: An array with the structure: [(source, target, bobs for min, min total distances, distances to detector)]
    """
    # to keep all the required information
    minimum_distance_array = []
    # The first Bob in the list - needed to carry out the for loop
    first_bob = list(distance_list.keys())[0]
    # for every element in the distance dictionary
    for i in range(len(distance_list[first_bob].a)):
        for j in range(i, len(distance_list[first_bob].a)):
            # if the nodes are both sources and they are not the same source then calculate all the required info and
            # place it into the array
            if graph.vertex_properties["node_type"][i] == NodeType(0).name \
                and graph.vertex_properties["node_type"][j] == NodeType(0).name and i != j:
                # get the arrays for mimimum distances to each bob, the bobs corresponding to these distances
                minimum_distances, bobs_for_min, distance_to_detectors = get_n_minimum_lengths_for_any_bob(i,j, distance_list, n)
                source_vertex = graph.vertex_properties["name"][i]
                target_vertex = graph.vertex_properties["name"][j]
                # to store the list of the minimum bobs using name and not labels
                bob_minimum = []
                for k in range(len(bobs_for_min)):
                    bob_minimum.append(graph.vertex_properties["name"][list(distance_list.keys())[bobs_for_min[k]]])
                # add all requested information
                minimum_distance_array.append((source_vertex, target_vertex, bob_minimum, minimum_distances, distance_to_detectors))
    return minimum_distance_array