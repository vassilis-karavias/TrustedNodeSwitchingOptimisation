import pandas as pd
import numpy as np
from generate_graph import NodeType
from graph_tool.all import *
from utils_graph import get_length
from minimum_length import get_minimum_to_any_bob
from capacity_calculation import calculate_capacity_efficient_all_distances
import os
import csv


class CapacityGraph(Graph):

    def __init__(self, capacities):
        """
        Generates a graph of capacities with nodes connected by edges telling them about the capacity between these
        nodes in the network
        :param capacities: The list of capacities between the nodes in the network in the form of a list
         [(source, target, capacity)]
        """
        # generate an undirected graph with edge properties equal to the capacities
        g = Graph(directed=False)
        if len(capacities) != 0:
            # edges = []
            # for source, target, capacity in capacities:
            #     edges.append((source, target))
            #
            # vertex_ids = g.add_edge_list(edges)
            self.capacities = g.new_edge_property(value_type="double")
            g.add_edge_list(capacities, eprops = [self.capacities])
            g.edge_properties["capacity"] = self.capacities
            self.g = g
            # edges = g.get_edges()
            # for i in range(len(edges)):
            #     assert(edges[i][0] == capacities[i][0] and edges[i][1] == capacities[i][1])
            #     self.capacities[i] = capacities[i][2]
        super().__init__(g = g, directed = False)

    def generate_needed_capacity_graph(self, cap_min):
        """
        Creates a CapacityGraph of the needed capacity given some minimum capacity cap_min - what is the capacity needed
        such that all connections in the graph have cap_min
        :param cap_min: The minimum acceptable capacity for any connection: double
        :return: The CapacityGraph where the edges represent the capacity needed to reach cap_min
        """
        # get the edges and the capacity of each of the edges in the list of the current CapacityGraph
        edges = self.get_edges(eprops = [self.capacities])
        # store the edges where there is not enough capacity as well as how much capacity is needed to make sure the
        # connection has cap_min
        new_graph_edges = []
        for i in range(len(edges)):
            source_node = int(edges[i][0])
            target_node = int(edges[i][1])
            capacity = edges[i][2]
            if capacity < cap_min:
                new_graph_edges.append((source_node, target_node, cap_min - capacity))
        # generate the new graph
        return CapacityGraph(new_graph_edges)


    def store_capacity_edge_graph(self, store_file_location, node_types = None, node_data_store_location = None, graph_id = 0):
        """
        Store the capacity edge list of the current capacity graph in location provided
        :param store_file_location: Location to store the data, string
        :param node_types: list of node_types
        :param node_data_store_location:Location to store the node data, string
        """
        # get the edges and the capacity of each of the edges in the list of the current CapacityGraph
        edges = self.g.get_edges(eprops= [self.capacities])

        dictionaries = []
        dictionary_fieldnames = ["ID", "source", "target", "capacity"]
        for edge in range(len(edges)):
            source = edges[edge][0]
            target = edges[edge][1]
            capacity = edges[edge][2]
            dictionaries.append({"ID": graph_id, "source": source, "target": target, "capacity": capacity})
        if node_types != None and node_data_store_location != None:
            nodes = self.g.get_vertices()
            # if len(nodes) != len(node_types):
            #     print("Node_types length must be same size as nodes")
            #     raise ValueError
            dictionary_fieldnames_nodes = ["ID", "node", "type"]
            dict_nodes = []
            for node in nodes:
                type = node_types[node]
                if type == "S" or type == "T":
                    dict_nodes.append({"ID": graph_id, "node": node, "type": type})
            if os.path.isfile(node_data_store_location + '.csv'):
                with open(node_data_store_location + '.csv', mode='a') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames_nodes)
                    writer.writerows(dict_nodes)
            else:
                with open(node_data_store_location + '.csv', mode='a') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames_nodes)
                    writer.writeheader()
                    writer.writerows(dict_nodes)

        if os.path.isfile(store_file_location + '.csv'):
            with open(store_file_location + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writerows(dictionaries)
        else:
            with open(store_file_location + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
                writer.writerows(dictionaries)

    def store_capacity_edge_graph_distances(self, store_file_location, distance, node_types = None, node_data_store_location = None, graph_id = 0):
        """
        Store the capacity edge list of the current capacity graph in location provided
        :param store_file_location: Location to store the data, string
        :param distance: The distance of the network
        :param node_types: list of node_types
        :param node_data_store_location:Location to store the node data, string
        """
        # get the edges and the capacity of each of the edges in the list of the current CapacityGraph
        edges = self.g.get_edges(eprops= [self.capacities])

        dictionaries = []
        dictionary_fieldnames = ["ID", "source", "target", "capacity", "distance"]
        for edge in range(len(edges)):
            source = edges[edge][0]
            target = edges[edge][1]
            capacity = edges[edge][2]
            dictionaries.append({"ID": graph_id, "source": source + 1, "target": target + 1, "capacity": capacity, "distance": distance})
        if node_types != None and node_data_store_location != None:
            nodes = self.g.get_vertices()
            # if len(nodes) != len(node_types):
            #     print("Node_types length must be same size as nodes")
            #     raise ValueError
            dictionary_fieldnames_nodes = ["ID", "node", "type"]
            dict_nodes = []
            for node in nodes:
                type = node_types[node]
                if type == "S" or type == "T" or type == NodeType(0) or type == NodeType(3):
                    dict_nodes.append({"ID": graph_id, "node": node+1, "type": type})
            if os.path.isfile(node_data_store_location + '.csv'):
                with open(node_data_store_location + '.csv', mode='a') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames_nodes)
                    writer.writerows(dict_nodes)
            else:
                with open(node_data_store_location + '.csv', mode='a') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames_nodes)
                    writer.writeheader()
                    writer.writerows(dict_nodes)

        if os.path.isfile(store_file_location + '.csv'):
            with open(store_file_location + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writerows(dictionaries)
        else:
            with open(store_file_location + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
                writer.writerows(dictionaries)