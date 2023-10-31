from generate_graph import NodeType
import csv


def get_tf_capacities(graph, dictionary, switches = True):
    if switches:
        distances = graph.get_shortest_distance_between_source_nodes_switch_loss()
    else:
        distances = graph.get_shortest_distance_between_source_nodes()
    rates = []
    for source_node, target_node, distance in distances:
        distance_to_detector = int(round(distance / 2))
        total_distance = int(round(distance))
        capacity = dictionary["L" + str(total_distance) + "LB" + str(distance_to_detector)]
        rates.append((source_node, target_node, capacity))
    return rates

def get_total_rate(rates):
    total_rate = 0
    for source_node, target_node, rate in rates:
        total_rate += rate
    return total_rate

def total_rate_tf_efficient(graph, dictionary, switches = True):
    rates = get_tf_capacities(graph, dictionary, switches)
    total_rate = get_total_rate(rates)
    return rates, total_rate