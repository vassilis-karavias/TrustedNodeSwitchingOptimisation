import copy
import os
import csv
import pandas as pd
import networkx as nx


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


def halve_capacities(capacity):
    """
    halves the capacity input into the function
    """
    return capacity

def convert_to_sc_form(capacity):
    """

    Parameters
    ----------
    capacity

    Returns
    -------
     """

    return str("{:.0e}".format(capacity))

def get_key_dict(cap_needed):
    """
    get the key dictionary in form {(source, target): capacity} - this lists the needed capacity of the connection
    Parameters
    ----------
    cap_needed - pandas Dataframe with needed capacity in form [source, target, capacity]
    -------

    """
    # get a list of the edges that need more capacity and the value of this capacity
    key_dict = {}
    for index, row in cap_needed.iterrows():
        source = int(row["source"])
        target = int(row["target"])
        capacity = row["capacity"]
        key_dict[(source, target)] = int(round(capacity))
    return key_dict


def get_key_dict_flexibility(cap_needed):
    """
    get the key dictionary in form {(source, target): N_k} - this lists the needed number of paths needed per
    connection - for flexibility incestigation

    Parameters
    ----------
    cap_needed- pandas Dataframe with needed capacity in form [source, target, capacity]
    -------

    """
    # get a list of the edges that need more capacity and the value of this capacity
    key_dict = {}
    for index, row in cap_needed.iterrows():
        source = int(row["source"])
        target = int(row["target"])
        nk = row["N_k"]
        key_dict[(source, target)] = int(round(nk))
    return key_dict


def make_key_dict_bidirectional(key_dict):
    """
    make the key dict bidirectional : if (source, target) in key_dict then (target, source) should be too
    """
    missing_entries = [(k[1],k[0]) for k in key_dict if (k[1],k[0]) not in key_dict]
    key_dict_copy = copy.deepcopy(key_dict)
    for idx in missing_entries:
        key_dict_copy[idx] = 0
    return key_dict_copy#

def import_problem_from_files(cap_needed_file, capacity_values_file, node_type_file):
    """

    Parameters
    ----------
    cap_needed_file
    capacity_values_file
    node_type_file

    Returns
    -------

    """
    cap_needed = pd.read_csv(cap_needed_file)

    # bidirectional_key_dict = make_key_dict_bidirectional(key_dict)
    capacity_values = pd.read_csv(capacity_values_file)
    ## need to map keys to new values which are correctly labelled as the detector nodes should not be included
    gc = Graph_Collector()
    node_types = pd.read_csv(node_type_file)
    node_types["node"] = node_types.apply(lambda x: gc.map_nodes_for_edges(0, x["node"]), axis = 1)

    capacity_values["source"] = capacity_values.apply(lambda x: gc.map_nodes_for_edges(0, x["source"]), axis=1)
    capacity_values["target"] = capacity_values.apply(lambda x: gc.map_nodes_for_edges(0, x["target"]), axis=1)
    capacity_values["capacity"] = capacity_values.apply(lambda x: halve_capacities(x["capacity"]), axis = 1)
    capacity_values["capacity_sc"] = capacity_values.apply(lambda x: convert_to_sc_form(x["capacity"]), axis=1)

    graph = nx.from_pandas_edgelist(capacity_values, "source", "target", ["capacity", "capacity_sc"])
    graph = graph.to_undirected()
    graph = graph.to_directed()
    node_attr = node_types.set_index("node").to_dict("index")
    nx.set_node_attributes(graph, node_attr)
    cap_needed["source"] = cap_needed.apply(lambda x: gc.map_nodes_for_edges(0, x["source"]), axis=1)
    cap_needed["target"] = cap_needed.apply(lambda x: gc.map_nodes_for_edges(0, x["target"]), axis=1)
    key_dict = get_key_dict(cap_needed)
    return key_dict, graph


def import_problem_from_files_flexibility(nk_file, capacity_values_file, node_type_file):
    nk_needed = pd.read_csv(nk_file)

    # bidirectional_key_dict = make_key_dict_bidirectional(key_dict)
    capacity_values = pd.read_csv(capacity_values_file)
    ## need to map keys to new values which are correctly labelled as the detector nodes should not be included
    gc = Graph_Collector()
    node_types = pd.read_csv(node_type_file)
    node_types["node"] = node_types.apply(lambda x: gc.map_nodes_for_edges(0, x["node"]), axis = 1)
    capacity_values["source"] = capacity_values.apply(lambda x: gc.map_nodes_for_edges(0, x["source"]), axis=1)
    capacity_values["target"] = capacity_values.apply(lambda x: gc.map_nodes_for_edges(0, x["target"]), axis=1)
    capacity_values["capacity"] = capacity_values.apply(lambda x: halve_capacities(x["capacity"]), axis = 1)
    capacity_values["capacity_sc"] = capacity_values.apply(lambda x: convert_to_sc_form(x["capacity"]), axis = 1)

    graph = nx.from_pandas_edgelist(capacity_values, "source", "target", ["capacity", "capacity_sc"])
    graph = graph.to_undirected()
    graph = graph.to_directed()
    node_attr = node_types.set_index("node").to_dict("index")
    nx.set_node_attributes(graph, node_attr)
    nk_needed["source"] = nk_needed.apply(lambda x: gc.map_nodes_for_edges(0, x["source"]), axis=1)
    nk_needed["target"] = nk_needed.apply(lambda x: gc.map_nodes_for_edges(0, x["target"]), axis=1)
    key_dict = get_key_dict_flexibility(nk_needed)
    return key_dict, graph



def import_problem_from_files_multiple_graphs(cap_needed_file, capacity_values_file, node_type_file):
    """

    Parameters
    ----------
    cap_needed_file
    capacity_values_file
    node_type_file

    Returns
    -------

    """
    cap_needed = pd.read_csv(cap_needed_file)

    # bidirectional_key_dict = make_key_dict_bidirectional(key_dict)
    capacity_values = pd.read_csv(capacity_values_file)
    ## need to map keys to new values which are correctly labelled as the detector nodes should not be included

    node_types = pd.read_csv(node_type_file)
    possible_ids = cap_needed["ID"].unique()
    # separate each graph based on ID and add to dictionary
    capacity_values_dict = {}
    node_types_dict = {}
    graphs = {}
    key_dict = {}
    for id in possible_ids:
        gc = Graph_Collector()
        node_types_set = node_types[node_types["ID"] == id].drop(["ID"], axis = 1)
        node_types_set["node"] = node_types_set.apply(lambda x: gc.map_nodes_for_edges(0, x["node"]), axis = 1)
        node_types_dict[id] = node_types_set

        capacity_values_set = capacity_values[capacity_values["ID"] == id].drop(["ID"], axis = 1)
        capacity_values_set["source"] = capacity_values_set.apply(lambda x: gc.map_nodes_for_edges(0, x["source"]), axis=1)
        capacity_values_set["target"] = capacity_values_set.apply(lambda x: gc.map_nodes_for_edges(0, x["target"]), axis=1)
        capacity_values_set["capacity"] = capacity_values_set.apply(lambda x: halve_capacities(x["capacity"]), axis = 1)
        capacity_values_set["capacitysc"] = capacity_values_set.apply(lambda x: convert_to_sc_form(x["capacity"]), axis=1)

        capacity_values_dict[id] = capacity_values_set



        graph = nx.from_pandas_edgelist(capacity_values_set, "source", "target", ["capacity", "capacity_sc"])
        graph = graph.to_undirected()
        graph = graph.to_directed()
        node_attr = node_types_set.set_index("node").to_dict("index")
        nx.set_node_attributes(graph, node_attr)
        graphs[id] = graph

        cap_needed_set = cap_needed[cap_needed["ID"] == id].drop(["ID"], axis = 1)
        cap_needed_set["source"] = cap_needed_set.apply(lambda x: gc.map_nodes_for_edges(0, x["source"]), axis=1)
        cap_needed_set["target"] = cap_needed_set.apply(lambda x: gc.map_nodes_for_edges(0, x["target"]), axis=1)
        key_dict[id] = get_key_dict(cap_needed_set)
    return key_dict, graphs


def import_problem_from_files_multiple_graphs_distances(cap_needed_file, capacity_values_file, node_type_file):
    """

    Parameters
    ----------
    cap_needed_file
    capacity_values_file
    node_type_file

    Returns
    -------

    """
    cap_needed = pd.read_csv(cap_needed_file)

    # bidirectional_key_dict = make_key_dict_bidirectional(key_dict)
    capacity_values = pd.read_csv(capacity_values_file)
    ## need to map keys to new values which are correctly labelled as the detector nodes should not be included

    node_types = pd.read_csv(node_type_file)
    possible_ids = cap_needed["ID"].unique()
    # separate each graph based on ID and add to dictionary
    capacity_values_dict = {}
    node_types_dict = {}
    graphs = {}
    key_dict = {}
    distances = {}
    for id in possible_ids:
        gc = Graph_Collector()
        node_types_set = node_types[node_types["ID"] == id].drop(["ID"], axis = 1)
        node_types_set["node"] = node_types_set.apply(lambda x: gc.map_nodes_for_edges(0, x["node"]), axis = 1)
        node_types_dict[id] = node_types_set


        distances[id] = capacity_values[capacity_values["ID"] == id]["distance"][capacity_values[capacity_values["ID"] == id]["distance"].keys()[0]]
        capacity_values_set = capacity_values[capacity_values["ID"] == id].drop(["ID"], axis = 1)
        capacity_values_set["source"] = capacity_values_set.apply(lambda x: gc.map_nodes_for_edges(0, x["source"]), axis=1)
        capacity_values_set["target"] = capacity_values_set.apply(lambda x: gc.map_nodes_for_edges(0, x["target"]), axis=1)
        capacity_values_set["capacity"] = capacity_values_set.apply(lambda x: halve_capacities(x["capacity"]), axis = 1)
        capacity_values_set["capacitysc"] = capacity_values_set.apply(lambda x: convert_to_sc_form(x["capacity"]), axis=1)

        capacity_values_dict[id] = capacity_values_set



        graph = nx.from_pandas_edgelist(capacity_values_set, "source", "target", ["capacity", "capacity_sc"])
        graph = graph.to_undirected()
        graph = graph.to_directed()
        node_attr = node_types_set.set_index("node").to_dict("index")
        nx.set_node_attributes(graph, node_attr)
        graphs[id] = graph

        cap_needed_set = cap_needed[cap_needed["ID"] == id].drop(["ID"], axis = 1)
        cap_needed_set["source"] = cap_needed_set.apply(lambda x: gc.map_nodes_for_edges(0, x["source"]), axis=1)
        cap_needed_set["target"] = cap_needed_set.apply(lambda x: gc.map_nodes_for_edges(0, x["target"]), axis=1)
        key_dict[id] = get_key_dict(cap_needed_set)
    return key_dict, graphs, distances



def import_problem_from_files_flexibility_multiple_graphs(nk_file, capacity_values_file, node_type_file):
    nk_needed = pd.read_csv(nk_file)

    # bidirectional_key_dict = make_key_dict_bidirectional(key_dict)
    capacity_values = pd.read_csv(capacity_values_file)
    possible_ids = nk_needed["ID"].unique()
    node_types = pd.read_csv(node_type_file)


    capacity_values_dict = {}
    node_types_dict = {}
    graphs = {}
    key_dict = {}

    for id in possible_ids:
        ## need to map keys to new values which are correctly labelled as the detector nodes should not be included
        gc = Graph_Collector()
        node_types_set = node_types[node_types["ID"] == id].drop(["ID"], axis = 1)
        node_types_set["node"] = node_types_set.apply(lambda x: gc.map_nodes_for_edges(0, x["node"]), axis = 1)
        node_types_dict[id] = node_types_set


        capacity_values_set = capacity_values[capacity_values["ID"] == id].drop(["ID"], axis = 1)
        capacity_values_set["source"] = capacity_values_set.apply(lambda x: gc.map_nodes_for_edges(0, x["source"]), axis=1)
        capacity_values_set["target"] = capacity_values_set.apply(lambda x: gc.map_nodes_for_edges(0, x["target"]), axis=1)
        capacity_values_set["capacity"] = capacity_values_set.apply(lambda x: halve_capacities(x["capacity"]), axis = 1)
        capacity_values_set["capacitysc"] = capacity_values_set.apply(lambda x: convert_to_sc_form(x["capacity"]), axis=1)

        capacity_values_dict[id] = capacity_values_set



        graph = nx.from_pandas_edgelist(capacity_values_set, "source", "target", ["capacity", "capacitysc"])
        graph = graph.to_undirected()
        graph = graph.to_directed()
        node_attr = node_types_set.set_index("node").to_dict("index")
        nx.set_node_attributes(graph, node_attr)
        graphs[id] = graph

        nk_needed_set = nk_needed[nk_needed["ID"] == id].drop(["ID"], axis = 1)
        nk_needed_set["source"] = nk_needed_set.apply(lambda x: gc.map_nodes_for_edges(0, x["source"]), axis=1)
        nk_needed_set["target"] = nk_needed_set.apply(lambda x: gc.map_nodes_for_edges(0, x["target"]), axis=1)
        key_dict[id] = get_key_dict_flexibility(nk_needed_set)
    return key_dict, graphs


def import_problem_from_files_flexibility_multiple_graphs_position_graph(nk_file, capacity_values_file, node_type_file, position_node_file, position_edge_file):
    nk_needed = pd.read_csv(nk_file)

    # bidirectional_key_dict = make_key_dict_bidirectional(key_dict)
    capacity_values = pd.read_csv(capacity_values_file)
    possible_ids = nk_needed["ID"].unique()
    node_types = pd.read_csv(node_type_file)
    position_nodes = pd.read_csv(position_node_file)
    position_edges = pd.read_csv(position_edge_file)

    capacity_values_dict = {}
    node_types_dict = {}
    graphs = {}
    key_dict = {}
    position_graphs = {}

    for id in possible_ids:
        ## need to map keys to new values which are correctly labelled as the detector nodes should not be included
        gc = Graph_Collector()
        node_types_set = node_types[node_types["ID"] == id].drop(["ID"], axis = 1)
        position_node_set = position_nodes[position_nodes["ID"] == id].drop(["ID"], axis = 1)
        node_types_set["node"] = node_types_set.apply(lambda x: gc.map_nodes_for_edges(0, x["node"]), axis = 1)
        node_types_dict[id] = node_types_set
        position_node_set["node"] = position_node_set.apply(lambda x: gc.map_nodes_for_edges(0, x["node"]), axis = 1)

        capacity_values_set = capacity_values[capacity_values["ID"] == id].drop(["ID"], axis = 1)
        position_edges_set = position_edges[position_edges["ID"] == id].drop(["ID"], axis =1)
        capacity_values_set["source"] = capacity_values_set.apply(lambda x: gc.map_nodes_for_edges(0, x["source"]), axis=1)
        capacity_values_set["target"] = capacity_values_set.apply(lambda x: gc.map_nodes_for_edges(0, x["target"]), axis=1)
        capacity_values_set["capacity"] = capacity_values_set.apply(lambda x: halve_capacities(x["capacity"]), axis = 1)
        capacity_values_set["capacitysc"] = capacity_values_set.apply(lambda x: convert_to_sc_form(x["capacity"]), axis=1)
        position_edges_set["source"] = position_edges_set.apply(lambda x: gc.map_nodes_for_edges(0, x["source"]), axis=1)
        position_edges_set["target"] = position_edges_set.apply(lambda x: gc.map_nodes_for_edges(0, x["target"]), axis=1)
        capacity_values_dict[id] = capacity_values_set




        graph = nx.from_pandas_edgelist(capacity_values_set, "source", "target", ["capacity", "capacitysc"])
        graph = graph.to_undirected()
        graph = graph.to_directed()
        node_attr = node_types_set.set_index("node").to_dict("index")
        nx.set_node_attributes(graph, node_attr)
        graphs[id] = graph

        position_graph = nx.from_pandas_edgelist(position_edges_set, "source", "target", ["distance"])
        position_graph = position_graph.to_undirected()
        position_graph = position_graph.to_directed()
        pos_node_attr = position_node_set.set_index("node").to_dict("index")
        nx.set_node_attributes(position_graph, pos_node_attr)
        position_graphs[id] = position_graph

        nk_needed_set = nk_needed[nk_needed["ID"] == id].drop(["ID"], axis = 1)
        nk_needed_set["source"] = nk_needed_set.apply(lambda x: gc.map_nodes_for_edges(0, x["source"]), axis=1)
        nk_needed_set["target"] = nk_needed_set.apply(lambda x: gc.map_nodes_for_edges(0, x["target"]), axis=1)
        key_dict[id] = get_key_dict_flexibility(nk_needed_set)
    return key_dict, graphs, position_graphs



def import_problem_from_files_multigraph_multiple_graphs_position_graph(nk_file, capacity_values_file, node_type_file, position_node_file, position_edge_file):
    nk_needed = pd.read_csv(nk_file)

    # bidirectional_key_dict = make_key_dict_bidirectional(key_dict)
    capacity_values = pd.read_csv(capacity_values_file)
    possible_ids = nk_needed["ID"].unique()
    node_types = pd.read_csv(node_type_file)
    position_nodes = pd.read_csv(position_node_file)
    position_edges = pd.read_csv(position_edge_file)

    capacity_values_dict = {}
    node_types_dict = {}
    graphs = {}
    key_dict = {}
    position_graphs = {}

    for id in possible_ids:
        ## need to map keys to new values which are correctly labelled as the detector nodes should not be included
        gc = Graph_Collector()

        node_types_set = node_types[node_types["ID"] == id].drop(["ID"], axis = 1)
        position_node_set = position_nodes[position_nodes["ID"] == id].drop(["ID"], axis = 1)
        node_types_set["node"] = node_types_set.apply(lambda x: gc.map_nodes_for_edges(0, x["node"]), axis = 1)
        node_types_dict[id] = node_types_set

        position_node_set["node"] = position_node_set.apply(lambda x: gc.map_nodes_for_edges(0, x["node"]), axis = 1)
        capacity_values_set = capacity_values[capacity_values["ID"] == id].drop(["ID"], axis = 1)
        position_edges_set = position_edges[position_edges["ID"] == id].drop(["ID"], axis =1)
        capacity_values_set["source"] = capacity_values_set.apply(lambda x: gc.map_nodes_for_edges(0, x["source"]), axis=1)
        capacity_values_set["target"] = capacity_values_set.apply(lambda x: gc.map_nodes_for_edges(0, x["target"]), axis=1)
        capacity_values_set["capacity"] = capacity_values_set.apply(lambda x: halve_capacities(x["capacity"]), axis = 1)
        capacity_values_set["capacitysc"] = capacity_values_set.apply(lambda x: convert_to_sc_form(x["capacity"]), axis=1)
        position_edges_set["source"] = position_edges_set.apply(lambda x: gc.map_nodes_for_edges(0, x["source"]), axis=1)
        position_edges_set["target"] = position_edges_set.apply(lambda x: gc.map_nodes_for_edges(0, x["target"]), axis=1)
        capacity_values_dict[id] = capacity_values_set




        graph = nx.from_pandas_edgelist(capacity_values_set, "source", "target", edge_attr = ["capacity", "capacitysc"], create_using=nx.MultiGraph())
        graph = graph.to_undirected()
        graph = graph.to_directed()
        node_attr = node_types_set.set_index("node").to_dict("index")
        nx.set_node_attributes(graph, node_attr)
        graphs[id] = graph

        position_graph = nx.from_pandas_edgelist(position_edges_set, "source", "target", ["distance"])
        position_graph = position_graph.to_undirected()
        position_graph = position_graph.to_directed()
        pos_node_attr = position_node_set.set_index("node").to_dict("index")
        nx.set_node_attributes(position_graph, pos_node_attr)
        position_graphs[id] = position_graph

        nk_needed_set = nk_needed[nk_needed["ID"] == id].drop(["ID"], axis = 1)
        nk_needed_set["source"] = nk_needed_set.apply(lambda x: gc.map_nodes_for_edges(0, x["source"]), axis=1)
        nk_needed_set["target"] = nk_needed_set.apply(lambda x: gc.map_nodes_for_edges(0, x["target"]), axis=1)
        key_dict[id] = get_key_dict_flexibility(nk_needed_set)
    return key_dict, graphs, position_graphs


def import_cmin(cmin_file):
    cmin_df = pd.read_csv(cmin_file)
    possible_ids = cmin_df["ID"].unique()
    cmin_values_dict = {}
    for id in possible_ids:
        cmin_dict = {}
        cmin_set = cmin_df[cmin_df["ID"] == id].drop(["ID"], axis=1)
        for index, row in cmin_set.iterrows():
            cmin_dict[row["source"] -1, row["target"] -1] = row["key"]
        cmin_values_dict[id] = cmin_dict
    return cmin_dict