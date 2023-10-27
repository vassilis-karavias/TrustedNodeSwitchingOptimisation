import networkx
import networkx as nx
import pandas
import numpy as np
from time import time



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


def get_new_paths_and_capacities(graph, paths_and_capacities, n):
    paths_and_capacities_new = paths_and_capacities[:n]
    paths_to_get = paths_and_capacities[n:]
    new_paths = []
    for path, capacity in paths_to_get:
        capacity = np.infty
        for i,j in path:
            if graph.edges[i,j]["capacity"] < capacity:
                capacity = graph.edges[i,j]["capacity"]
        new_paths.append([path, capacity])
    paths_and_capacities_new = paths_and_capacities_new + new_paths
    return paths_and_capacities_new

class RecursivePaths():

    def __init__(self, initial_graph, key_dict):
        """
        The class for the recursive algorithm
        Parameters
        ----------
        initial_graph: the initial graph with node types and capacities
        key_dict: a key dictionary that contains the information of the keys.
        """
        # for each connection get all simple paths - perhaps this needs to be changed to the key dictionary pairs
        paths = []
        for i in initial_graph.nodes:
            for j in initial_graph.nodes:
                if i < j and initial_graph.nodes[i]["type"] == "S" and initial_graph.nodes[j]["type"] == "S" and (i,j) in key_dict.keys():
                    # array paths = [(all i,j pairs)[(all paths for the pair)[[the path in form (a,b)], capacity
                    # source, target]]]
                    simple_paths = [[path, 0, i, j] for path in nx.all_simple_edge_paths(initial_graph, source=i, target=j)]
                    ##### need to check that the simple path does not contain an S node in the intermediate
                    simple_paths_that_are_valid = []
                    for path in simple_paths:
                        add_path_to_list = True
                        for k in range(len(path[0])):
                            if k != len(path[0])-1 and initial_graph.nodes[path[0][k][1]]["type"] == "S":
                                add_path_to_list = False
                        if add_path_to_list:
                            simple_paths_that_are_valid.append(path)
                    paths.append(simple_paths_that_are_valid)
        self.paths_and_capacities = paths

    def update_paths_and_capacities(self, graph, n):
        """
        update the paths_and_capacities capacities to match those in the remaining graph based on capacity field of
        graph
        Parameters
        ----------
        graph: graph with capacities and node types
        n: edge we are currently on (int to get the outer term in the paths and capacities array
        -------

        """
        # make copy of the paths_and_capacities array - no need to change the part before the nth term
        paths_and_capacities_new = self.paths_and_capacities[:n].copy()
        # get array from nth term onwards
        paths_to_get = self.paths_and_capacities[n:].copy()
        paths_new = []
        # for all paths in this array find the new capacity of this array and change the value in the array to this
        # capacity
        for paths in paths_to_get:
            new_paths = []
            for path, capacity, source, target in paths:
                capacity = np.infty
                for i,j in path:
                    if graph.edges[i,j]["capacity"] < capacity:
                       capacity = graph.edges[i,j]["capacity"]
                new_paths.append([path, capacity, source, target])
            paths_new.append(new_paths)
        # append the two arrays (unchanged and changed) and then update paths_and_capacities
        paths_and_capacities_new = paths_and_capacities_new + paths_new
        self.paths_and_capacities = paths_and_capacities_new


    def recursive_paths(self, graph, n, k, key_dict):
        """
        The recursion: for the graph from the previous iteration we want to carry the recursion on the nth connection
        (i,j) for the kth largest capacity (checking the key_dict is satisfied)
        Parameters
        ----------
        graph: graph for the previous iteration with node type and capacity
        n: connection to look at
        k: what number of capacity are we at
        key_dict: the required connectivity

        Returns Whether this is a solution or not
        -------

        """
        # if we are on the last connection then we have edge case: if the total available capacity is more than the total
        # required capacity then this is a solution else not a solution
        if n == len(self.paths_and_capacities) - 1:
            # update the capacities
            self.update_paths_and_capacities(graph, n=n)
            # get current connection we are looking at
            current_path = self.paths_and_capacities[n]
            # sort the capacities in ascending order
            current_path = sorted(current_path, key = lambda x: x[1])
            # need to check there exists a combination of paths that can create the required capacities
            new_graph = graph.copy()
            # get the capacity needed by edge
            cap_needed = key_dict[(current_path[-1][3], current_path[-1][2])] + key_dict[
                (current_path[-1][2], current_path[-1][3])]
            # need to update the capacities after every added path thus need to store this in a new array
            current_path_update = current_path
            for i in range(len(current_path)):
                #  get the current largest path - this is the only approximation in the whole algorithm
                path = current_path_update[-1]
                # if largest path has enough capacity then this is a solution as this is the last connection
                if path[1] >= cap_needed:
                    return True
                # if the largest capacity path == 0 then return not a solution
                elif path[1] < 0.000001:
                    return False
                else:
                    # if not add the largest capacity to the graph - remove this used capacity from the graph and
                    # update the available capacities.
                    for i, j in path[0]:
                        new_graph.edges[i, j]["capacity"] = new_graph.edges[i, j]["capacity"] - path[1]
                    cap_needed = cap_needed - path[1]
                    self.update_paths_and_capacities(new_graph, n=n)
                    current_path_update = self.paths_and_capacities[n]
                    current_path_update = sorted(current_path_update, key=lambda x: x[1])
            return False
        # other edge case: if we're on the first connection in the array and have gone past last path then no solution
        elif n == 0 and k == len(self.paths_and_capacities[0]):
            return False
        else:
            self.update_paths_and_capacities(graph, n=n)
            # else get current connection and sort them in ascending capacity
            current_path = self.paths_and_capacities[n]
            current_path = sorted(current_path, key=lambda x: x[1])
            # get kth largest capacity path
            current_path_to_use = current_path[-(k+1):]
            current_path_to_use = sorted(current_path_to_use, key=lambda x: x[1])
            # need to check there exists a combination of paths that can create the required capacities
            new_graph = graph.copy()
            # get the capacity needed by edge
            cap_needed = key_dict[(current_path_to_use[-1][3], current_path_to_use[-1][2])] + key_dict[
                (current_path_to_use[-1][2], current_path_to_use[-1][3])]
            # need to update the capacities after every added path thus need to store this in a new array
            current_path_update = current_path_to_use
            for i in range(len(current_path_to_use) + 1):
                #  get the current largest path - this is the only approximation in the whole algorithm
                path = current_path_update[-1]
                # if largest path has enough capacity then this is a solution for the current term so continue
                if path[1] >= cap_needed:
                    for i, j in path[0]:
                        new_graph.edges[i, j]["capacity"] = new_graph.edges[i, j]["capacity"] - cap_needed
                    break
                # if the largest capacity path == 0 then return not a solution
                elif path[1] < 0.000001:
                    return False
                else:
                    # if not add the largest capacity to the graph - remove this used capacity from the graph and
                    # update the available capacities.
                    for i, j in path[0]:
                        new_graph.edges[i, j]["capacity"] = new_graph.edges[i, j]["capacity"] - path[1]
                    cap_needed = cap_needed - path[1]
                    self.update_paths_and_capacities(new_graph, n=n)
                    current_path_update = self.paths_and_capacities[n]
                    current_path_update = sorted(current_path_update, key=lambda x: x[1])
            self.update_paths_and_capacities(new_graph, n=n)
            # the connection uses up some capacity- this must be removed from the graph and the paths and capacity
            # array must be updated
            # recurse down the connections for all paths to check if any are solution - if one is return true
            # else return false
            value = False
            for ks in range(len(self.paths_and_capacities[n])):
                solution =  self.recursive_paths(new_graph, n = n+1, k = ks, key_dict = key_dict)
                if solution:
                    value = solution
                    break
            return value


def nth_change_to_kth_largest_capacity(graph, paths_and_capacities, n, k, cmin):
    # paths_and_capacities has form [[[path], capacities]] where outer array is for connection i,j - only works if
    # graph has full capacity info with everything removed till the nth path
    if n == 0:
        return paths_and_capacities
    else:
        current_path = paths_and_capacities[-n]
        current_path_sorted = current_path.sort(reverse = True)
        current_path_taken = current_path_sorted[k]
        new_graph = graph.copy()
        for i,j in current_path_taken[0]:
            new_graph.edges[i,j]["capacity"] = new_graph.edges[i,j]["capacity"] - cmin
            ####### need a get new paths_and_capacities from graph function
            paths_and_capacities_new = get_new_paths_and_capacities(graph, paths_and_capacities, n)
        paths_and_capacities_to_return = nth_change_to_kth_largest_capacity(new_graph, paths_and_capacities_new, n = n-1, k = 0, cmin = cmin)
        return paths_and_capacities_to_return




def is_solution(graph, key_dict):
    paths = []
    for i in graph.nodes:
        for j in graph.nodes:
            if i < j and graph.nodes[i]["type"] == "S" and graph.nodes[j]["type"] == "S":
                paths.append([[path for path in nx.all_simple_edge_paths(graph, source = i, target= j, cutoff=20)],0])
    paths_with_capacities = get_new_paths_and_capacities(graph, paths_and_capacities = paths, n = 0)






def permutation_is_solution(graph, key_dict):
    solution = True
    for pair, num_keys in key_dict.items():
        cut_value, partition = networkx.algorithms.flow.minimum_cut(graph, pair[0], pair[1])
        if cut_value < key_dict[pair]:
            solution = False
            break
    return solution


def get_graph_for_permutation(graph_full):
    """Here C denotes on trusted nodes"""
    graph = networkx.Graph()
    node_data = {}
    for edge in graph_full.edges:
        source = edge[0]
        target = edge[1]
        if graph_full.nodes[source]["type"] == "S" and graph_full.nodes[target]["type"] == "S":
            if source not in node_data.keys():
                node_data[source] = {"type": graph_full.nodes[source]["type"]}
            if target not in node_data.keys():
                node_data[target] = {"type": graph_full.nodes[target]["type"]}
            graph.add_edge(source, target, capacity = graph_full.edges[edge]["capacity"] * 2)
        elif graph_full.nodes[source]["type"] == "S" and graph_full.nodes[target]["type"] == "C" or \
            graph_full.nodes[source]["type"] == "C" and graph_full.nodes[target]["type"] == "S" or \
            graph_full.nodes[source]["type"] == "C" and graph_full.nodes[target]["type"] == "C":
            if source not in node_data.keys():
                node_data[source] = {"type": graph_full.nodes[source]["type"]}
            if target not in node_data.keys():
                node_data[target] = {"type": graph_full.nodes[target]["type"]}
            graph.add_edge(source, target, capacity = graph_full.edges[edge]["capacity"] * 2)
    networkx.set_node_attributes(graph, node_data)
    return graph


def generate_permutations(graph, key_dict):
    no_of_trusted_nodes = 0
    for node in graph.nodes():
        if graph.nodes[node]["type"] == "C" or graph.nodes[node]["type"] == "T":
            no_of_trusted_nodes += 1
    t_1 = time.time()
    for i in range(1,no_of_trusted_nodes+1):
        # generate an array of perturbations with the correct no. of Bobs and free locations
        array_for_perturbation = np.concatenate((np.full(shape=no_of_trusted_nodes - i, fill_value="T"),
                                                 np.full(shape=i, fill_value="C")))
        # find all possible unique perturbations of this setup
        perturbations = list(perm_unique(list(array_for_perturbation)))
        for pert in perturbations:
            j = 0
            for node in graph.nodes():
                if graph.nodes[node]["type"] == "C" or graph.nodes[node]["type"] == "T":
                    graph.nodes[node]["type"] = pert[j]
                    j += 1
            graph_small = get_graph_for_permutation(graph)
            recusion = RecursivePaths(graph_small, key_dict)
            solution = recusion.recursive_paths(graph_small, n = 0, k = len(recusion.paths_and_capacities[0]) - 1, key_dict = key_dict)
            # solution = permutation_is_solution(graph_small, key_dict)
            if solution:
                t_2 = time.time()
                return i, t_2 - t_1
    return -1


