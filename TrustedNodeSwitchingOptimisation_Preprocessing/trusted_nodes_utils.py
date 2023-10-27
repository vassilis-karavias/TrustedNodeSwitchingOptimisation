import pandas as pd
import numpy as np
from generate_graph import NodeType
from graph_tool.all import *
from trusted_nodes.vector import Vector

def position_of_next_trusted_node(position_graph, capacity_graph):
    """
    find the position of the next trusted node to add to the graph: guess given in notes
    :param position_graph: The position graph of the network to add the trusted node: Position Graph
    :param capacity_graph: The capacity graph of the network defining the links that need more capacity and how much
    more capacity is needed: Capacity Graph
    :return: the vector with the guess of the position of the next trusted node - returns None if the capacity graph is
    empty, i.e. no need for further trusted nodes
    """
    # get a list of the edges that need more capacity and the value of this capacity
    edges = capacity_graph.get_edges(eprops=[capacity_graph.capacities])
    # set up the average position and normalisation criterion
    position = Vector(np.array([0,0]))
    capacity_normalisation = 0.0
    for i in range(len(edges)):
        # sum over all terms the values as defined in notes
        source_node = edges[i][0]
        target_node = edges[i][1]
        capacity = int(edges[i][2])
        x_coord_1 = position_graph.x_coord[source_node]
        y_coord_1 = position_graph.y_coord[source_node]
        x_coord_2 = position_graph.x_coord[target_node]
        y_coord_2 = position_graph.y_coord[target_node]
        vector_1 = Vector(np.array([x_coord_1, y_coord_1]))
        vector_2 = Vector(np.array([x_coord_2, y_coord_2]))
        average_position = (vector_1 + vector_2).scalar_mult(0.5)
        if capacity != 0:
            position = position + average_position.scalar_mult(np.log(capacity))
            capacity_normalisation += np.log(capacity)
    # if capacity_normalisation == 0.0 then there is no capacity needed thus return None - no need to place new trusted
    # node
    if capacity_normalisation > 0.00000000000001:
        position = position.scalar_mult(1/capacity_normalisation)
    else:
        position = None
    return position

def get_weight(source, target, node_data):
    x_pos_source = node_data[node_data["node"] == source]["xcoord"]
    y_pos_source = node_data[node_data["node"] == source]["ycoord"]
    x_pos_target = node_data[node_data["node"] == target]["xcoord"]
    y_pos_target = node_data[node_data["node"] == target]["ycoord"]
    distance = (Vector(np.asarray([x_pos_source, y_pos_source])) - Vector(np.asarray([x_pos_target, y_pos_target]))).magnitude()
    return distance


def get_key_dict(capacity_graph):
    # get a list of the edges that need more capacity and the value of this capacity
    edges = capacity_graph.get_edges(eprops=[capacity_graph.capacities])
    key_dict = {}
    for source, target, capacity in edges:
        key_dict[(source, target)] = int(round(capacity))
    return key_dict

def make_key_dict_bidirectional(key_dict):
    missing_entries = [(k[1],k[0]) for k in key_dict if (k[1],k[0]) not in key_dict]
    for idx in missing_entries:
        key_dict[idx] = 0
    return key_dict


class Graph_Collector():

    def __init__(self):
        """
        Initialises class for Graph Collector, here will take a set of data of form [Graph_ID, node] and [Graph_ID,
        source_node, target_node] and collect each of the graphs into a single large graph by assigning different IDs
        to each different node
        """
        self.maps = {}


    def map_nodes_for_node_set(self, graph_id, node_graph):
        """
        returns the new node label for the mapped node based on input graph_id and node_id in the graph - if this is an
        unseen node then add the new node to the map dictionary that maps graph_id, node_id -> node_id of collective
        graph

        Parameters
        ----------
        graph_id : ID of the graph (int)
        node_graph : ID of node in the graph (int)

        Returns : ID of node in the new combined graph (int)
        -------

        """
        # look for entry in self.maps - if it exists return the value in maps
        if str(graph_id) + "," + str(int(node_graph)) in self.maps:
            return self.maps[str(graph_id) + "," + str(int(node_graph))]
        else:
            # if not in maps then add the entry with value the next biggest value to that in maps
            if not bool(self.maps):
                self.maps[str(graph_id) + "," + str(int(node_graph))] = 0
                return 0
            else:
                highest = max(self.maps.values())
                self.maps[str(graph_id)+ "," + str(int(node_graph))] = highest + 1
                return highest + 1

    def map_nodes_for_edges(self, graph_id, node_graph):
        """
        Returns the node ID in the new graph for each element in an edge
        Parameters
        ----------
        graph_id : ID of the graph (int)
        node_graph : ID of node in the graph (int)

        Returns : ID of node in the new combined graph (int)
        -------

        """
        # if in self.maps return the value
        if str(int(graph_id)) + "," + str(int(node_graph)) in self.maps:
            return self.maps[str(int(graph_id)) + "," + str(int(node_graph))]
        else:
            # if not in self.maps add new value to self.maps and return this
            print("Warning: Key not found - means there is a node in edges not in node set")
            return self.map_nodes_for_node_set(graph_id, node_graph)