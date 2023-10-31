from generate_graph import NodeType
from rates_estimates.utils import get_rate
import csv

def get_server_client_capacities(graph):
    """
    Get the capacities for each connection of the graph for the BB84 Decoy protocol using calculation
    :param graph: The graph to calculate the capacities on
    :return: An array of capacities of each pair of conenctions in form [(source_nodes, target_nodes, capacities)]
    """
    # get the shortest distances between all source nodes
    distances = graph.get_shortest_distance_between_source_nodes()
    # get the rate for each connection and store in the array
    rates = []
    for source_node, target_node, distance in distances:
        rate = get_rate(length= distance, protocol = "Decoy", coldalice = False, coldbob = False)
        rates.append((source_node, target_node, rate))
    return rates

def get_total_rate(rates):
    """
    Get the total rate of the graph as sum of all rates
    :param rates: List of all the rates of the connections in array [(source_nodes, target_nodes, capacities)]
    :return: The total rate of the graph
    """
    total_rate = 0
    for source_node, target_node, rate in rates:
        total_rate += rate
    return total_rate

def total_rate_server_client(graph):
    """
    Get the rates of the connections and total rate and print them out
    :param graph: The graph to calculate the capacities on
    """
    rates = get_server_client_capacities(graph)
    total_rate = get_total_rate(rates)
    print("Capacities of configuration - BB84 Decoy State Protocol: ")
    for i in range(len(rates)):
        print(str(rates[i]))
    print("Total capacity: " + str(total_rate))

def get_server_client_capacities_efficient(graph, dictionary, switches):
    """
    Get the capacities for each connection of the graph for the BB84 Decoy protocol using efficient precalculated rates
    :param graph: The graph to calculate the capacities on
    :param dictionary: Dictionary containing the rates for each of the different lengths
    :param switches: whether to include switches loss or not
    :return: An array of capacities of each pair of conenctions in form [(source_nodes, target_nodes, capacities)]
    """
    if switches:
        distances = graph.get_shortest_distance_between_source_nodes_switch_loss()
    else:
        distances = graph.get_shortest_distance_between_source_nodes()
    rates = []
    for source_node, target_node, distance in distances:
        distance_actual = round(distance, 2)
        if distance_actual > 999:
            capacity = 0.0
        else:
            # from the look-up table
            capacity = dictionary["L" + str(distance_actual)]
        rates.append((source_node,target_node,capacity))
    return rates

def total_rate_server_client_efficient(graph, dictionary, switches = True):
    """
    Get total capacity of graph and each capacity of the connection for the graph
    :param graph: The graph to calculate the capacities on
    :param dictionary: Dictionary containing the rates for each of the different lengths
    :param switches: whether to include switches loss or not
    :return: An array of capacities of each pair of conenctions in form [(source_nodes, target_nodes, capacities)] and
    the total rates of the graph
    """
    rates = get_server_client_capacities_efficient(graph, dictionary, switches)
    total_rate = get_total_rate(rates)
    # print("Capacities of configuration - BB84 Decoy State Protocol: ")
    # for i in range(len(rates)):
    #     print(str(rates[i]))
    # print("Total capacity: " + str(total_rate))
    return rates, total_rate