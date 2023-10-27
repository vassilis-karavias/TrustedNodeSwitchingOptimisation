import pandas as pd

from trusted_nodes.position_graph_no_reduction import Position_Graph, Position_Graph_Set_Distances
from trusted_nodes.position_graph_no_switching import Position_Graph_No_Switching_Set_Distances
import csv
from trusted_nodes.trusted_nodes_utils import *

# new_graphs = Position_Graph_No_Switching_Set_Distances()
# new_graphs.create_graphs(import_path_nodes = "~/anaconda3/envs/gt/sources/new_data/7_nodes_graph_bb84_network.csv", import_path_edges = "~/anaconda3/envs/gt/sources/new_data/7_edges_graph_bb84_network.csv", no_source_nodes=6)

new_graphs_2 = Position_Graph_No_Switching_Set_Distances()
new_graphs_2.create_graphs(import_path_nodes = "~/anaconda3/envs/gt/sources/new_data/7_nodes_graph_bb84_network.csv", import_path_edges = "~/anaconda3/envs/gt/sources/new_data/7_edges_graph_bb84_network.csv", no_source_nodes=5)
# node_type = [0,3,0,0,3,0,3,3,3,3,3,3,3] + [3 for i in range(27)]
# graph_list = new_graphs.pos_graphs
graph_list_2 = new_graphs_2.pos_graphs
# graph_list_2[0].set_new_node_types(node_types = node_type)
### use O-band for connections between users and trusted nodes in no routing case....


#
# new_graphs_2 = Position_Graph_No_Switching_Set_Distances()
# new_graphs_2.create_graphs(import_path_nodes = "~/anaconda3/envs/gt/sources/new_data/4_nodes_graph_bb84_network.csv", import_path_edges = "~/anaconda3/envs/gt/sources/new_data/4_edges_graph_bb84_network.csv", no_source_nodes=8)
#
# graph_list = new_graphs.pos_graphs

# dictionary = {}
# with open('rates_coldbob.csv', mode='r') as csv_file:
#     csv_reader = csv.DictReader(csv_file)
#     line_count = 0
#     for row in csv_reader:
#         if line_count == 0:
#             print(f'Column names are {", ".join(row)}')
#             line_count += 1
#         dictionary["L" + row["L"] + "LB" + row["LB"]] = float(row['rate'])
#         line_count += 1
#     print(f'Processed {line_count} lines.')
dictionary_bb84_15_eff = {}
with open('rates_hotbob_bb84_15_eff.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        dictionary_bb84_15_eff["L" + str(round(float(row["L"]), 2))] = float(row['rate'])
        line_count += 1
    print(f'Processed {line_count} lines.')

dictionary_tf_15_eff = {}
with open('rates_hotbob_15_eff.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if round(float(row["L"])/2) == round(float(row["LB"])):
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            dictionary_tf_15_eff["L" + str(round(float(row["L"])))] = float(row['rate'])
            line_count += 1
    print(f'Processed {line_count} lines.')

# dictionary_bb84_15_eff = {}
# with open('rates_hotbob_bb84_15_eff.csv', mode='r') as csv_file:
#     csv_reader = csv.DictReader(csv_file)
#     line_count = 0
#     for row in csv_reader:
#         if line_count == 0:
#             print(f'Column names are {", ".join(row)}')
#             line_count += 1
#         dictionary_bb84_15_eff["L" + str(round(float(row["L"]), 2))] = float(row['rate'])
#         line_count += 1
#     print(f'Processed {line_count} lines.')
# dictionary_bb84_10_eff = {}
# with open('rates_hotbob_bb84_20_eff.csv', mode='r') as csv_file:
#     csv_reader = csv.DictReader(csv_file)
#     line_count = 0
#     for row in csv_reader:
#         if line_count == 0:
#             print(f'Column names are {", ".join(row)}')
#             line_count += 1
#         dictionary_bb84_10_eff["L" + str(round(float(row["L"]), 2))] = float(row['rate'])
#         line_count += 1
#     print(f'Processed {line_count} lines.')
# dictionary_bb84_25_eff = {}
# with open('rates_hotbob_bb84_25_eff.csv', mode='r') as csv_file:
#     csv_reader = csv.DictReader(csv_file)
#     line_count = 0
#     for row in csv_reader:
#         if line_count == 0:
#             print(f'Column names are {", ".join(row)}')
#             line_count += 1
#         dictionary_bb84_25_eff["L" + str(round(float(row["L"]), 2))] = float(row['rate'])
#         line_count += 1
#     print(f'Processed {line_count} lines.')
# dictionary_bb84_cold = {}
# with open('rates_coldbob_bb84.csv', mode='r') as csv_file:
#     csv_reader = csv.DictReader(csv_file)
#     line_count = 0
#     for row in csv_reader:
#         if line_count == 0:
#             print(f'Column names are {", ".join(row)}')
#             line_count += 1
#         dictionary_bb84_cold["L" + str(round(float(row["L"]), 2))] = float(row['rate'])
#         line_count += 1
#     print(f'Processed {line_count} lines.')


# dictionary_hot = {}
# with open('rates_hotbob_new.csv', mode='r') as csv_file:
#     csv_reader = csv.DictReader(csv_file)
#     line_count = 0
#     for row in csv_reader:
#         if line_count == 0:
#             print(f'Column names are {", ".join(row)}')
#             line_count += 1
#         dictionary_hot["L" + row["L"] + "LB" + row["LB"]] = float(row['rate'])
#         line_count += 1
#     print(f'Processed {line_count} lines.')

# new_graphs.add_random_trusted_nodes_to_graphs(n = 10)
# new_graphs.store_position_graph(node_data_store_location = "one_path_small_graphs_for_test_position_graph_nodes", edge_data_store_location = "one_path_small_graphs_for_test_position_graph_edges")
# new_graphs.store_n_k_for_n_state_tfqkd(dictionary, c_min=10000, store_file_location="one_path_small_graphs_for_test_cap_needed")
# new_graphs.store_capacity_edge_graph_distance_tfqkd(dictionary_tf = dictionary, dictionary_bb84 = dictionary_bb84, store_file_location ="one_path_small_graphs_for_test_edge_data", node_data_store_location ="one_path_small_graphs_for_test_node_types")

# new_graphs.store_capacity_edge_graph_multiedge_graph_distance(dictionary_tf = dictionary, dictionary_bb84 = dictionary_bb84, store_file_location = "small_graphs_for_test_edge_data", node_data_store_location = "small_graphs_for_test_node_types")

##### CURRENT SETUP: ALL SOURCE NODES ARE FOUND ON EDGES OF GRAPH AND SWITCHING AVAILABLE FOR NO REDUCTION GRAPH BUT NOT
##### FOR POINT-TO-POINT GRAPH
# new_graphs.add_random_trusted_nodes_to_graphs(n = 10)


# new_graphs_2.store_position_graph(node_data_store_location = f"18_nodes_bb84_network_position_graph", edge_data_store_location = f"18_edges_bb84_network_position_graph")
# new_graphs_2.store_n_k_for_n_state(n = 1,store_file_location=f"17_cap_needed_bb84_graph_n_1")
# new_graphs_2.store_n_k_for_n_state(n = 2,store_file_location=f"18_cap_needed_bb84_graph_n_2")
# new_graphs_2.store_n_k_for_n_state(n = 3,store_file_location=f"17_cap_needed_bb84_graph_n_3")
# new_graphs_2.store_n_k_for_n_state(n = 4,store_file_location=f"17_cap_needed_bb84_graph_n_4")
# new_graphs_2.store_n_k_for_n_state(n = 5,store_file_location=f"17_cap_needed_bb84_graph_n_5")
    # here is where you can do a parameter sweep - run using multiple different dictionaries.
    # new_graphs.store_n_k_for_n_state_bb84(dictionary_bb84, c_min=1000, store_file_location="t_cap_needed_bb84_graph")
# new_graphs_2.store_capacity_edge_graph_distance_bb84_no_switching(dictionary_bb84_15_eff, store_file_location = "18_edge_data_capacity_graph_bb84_network_no_switching",  node_data_store_location = "18_node_data_capacity_graph_bb84_network_no_switching")
# new_graphs_2.store_capacity_edge_graph_distance_bb84(dictionary_bb84_15_eff, store_file_location = f"18_edge_data_capacity_graph_bb84_network",  node_data_store_location = f"18_node_data_capacity_graph_bb84_network")
new_graphs_2.store_capacity_edge_graph_distance_bb84_no_switching(dictionary_bb84_15_eff, store_file_location = "18_edge_data_capacity_graph_bb84_network_no_switching",  node_data_store_location = "18_node_data_capacity_graph_bb84_network_no_switching")
new_graphs_2.store_capacity_edge_graph_distance_bb84(dictionary_bb84_15_eff, store_file_location = f"18_edge_data_capacity_graph_bb84_network",  node_data_store_location = f"18_node_data_capacity_graph_bb84_network")
# new_graphs_2.store_capacity_edge_graph_distance_bb84_no_switching(dictionary_bb84_15_eff, store_file_location = "14_edge_data_capacity_graph_bb84_network_no_switching_15_eff",  node_data_store_location = "14_node_data_capacity_graph_bb84_network_no_switching_15_eff")
# new_graphs_2.store_capacity_edge_graph_distance_bb84(dictionary_bb84_20_eff, store_file_location = f"real_edge_data_2",  node_data_store_location = f"real_node_data_2")
# new_graphs_2.store_capacity_edge_graph_distance_bb84_no_switching(dictionary_bb84_20_eff, store_file_location = "real_edge_data_no_switching_2",  node_data_store_location = "real_node_data_no_switching_2")
# new_graphs_2.set_new_o_band_req_to_none()
# new_graphs_2.store_capacity_edge_graph_distance_bb84(dictionary_bb84_20_eff, store_file_location = f"18_edge_data_capacity_graph_bb84_network_no_o_band",  node_data_store_location = f"18_node_data_capacity_graph_bb84_network_no_o_band")
# new_graphs_2.store_capacity_edge_graph_distance_bb84_no_switching(dictionary_bb84_20_eff, store_file_location = "18_edge_data_capacity_graph_bb84_network_no_switching_no_o_band",  node_data_store_location = "18_node_data_capacity_graph_bb84_network_no_switching_no_o_band")

# new_graphs_2.store_capacity_edge_graph_distance_bb84(dictionary_bb84_25_eff, store_file_location = f"14_edge_data_capacity_graph_bb84_network_25_eff",  node_data_store_location = f"14_node_data_capacity_graph_bb84_network_25_eff")
# new_graphs_2.store_capacity_edge_graph_distance_bb84_no_switching(dictionary_bb84_25_eff, store_file_location = "14_edge_data_capacity_graph_bb84_network_no_switching_25_eff",  node_data_store_location = "14_node_data_capacity_graph_bb84_network_no_switching_25_eff")



# new_graphs.store_capacity_edge_graph_distance_bb84(dictionary_bb84, store_file_location = "10_edge_data_capacity_graph_bb84_network",  node_data_store_location = "10_node_data_capacity_graph_bb84_network")
# new_graphs.store_capacity_edge_graph_distance_bb84_no_switching(dictionary_bb84, store_file_location = "10_edge_data_capacity_graph_bb84_network_no_switching",  node_data_store_location = "10_node_data_capacity_graph_bb84_network_no_switching")
# new_graphs_2.store_position_graph(node_data_store_location = "5_no_switching_nodes_bb84_network_position_graph", edge_data_store_location = "5_no_switching_edges_bb84_network_position_graph")
# new_graphs_2.store_n_k_for_n_state(n = 2,store_file_location="5_no_switching_cap_needed_bb84_graph")
#
# # here is where you can do a parameter sweep - run using multiple different dictionaries.
# # new_graphs.store_n_k_for_n_state_bb84(dictionary_bb84, c_min=1000, store_file_location="t_cap_needed_bb84_graph")
# new_graphs_2.store_capacity_edge_graph_distance_bb84(dictionary_bb84, store_file_location = "5_no_switching_edge_data_capacity_graph_bb84_network",  node_data_store_location = "5_no_switching_node_data_capacity_graph_bb84_network")





# new_graphs = Position_Graph_Set()
# new_graphs.import_graphs(import_path_nodes= "~/anaconda3/envs/gt/sources/data/node_info_for_trusted_node_15_node.csv", import_path_edges= "~/anaconda3/envs/gt/sources/data/edge_info_for_trusted_node_15_node.csv")
# graph_list = new_graphs.pos_graphs
# pos_graph_0 = graph_list[4]
# dictionary = {}
# with open('rates_coldbob_20_eff.csv', mode='r') as csv_file:
#     csv_reader = csv.DictReader(csv_file)
#     line_count = 0
#     for row in csv_reader:
#         if line_count == 0:
#             print(f'Column names are {", ".join(row)}')
#             line_count += 1
#         dictionary["L" + row["L"] + "LB" + row["LB"]] = float(row['rate'])
#         line_count += 1
#     print(f'Processed {line_count} lines.')11
# dictionary_bb84 = {}
# with open('rates_coldbob_bb84_20_eff.csv', mode='r') as csv_file:
#     csv_reader = csv.DictReader(csv_file)
#     line_count = 0
#     for row in csv_reader:
#         if line_count == 0:
#             print(f'Column names are {", ".join(row)}')
#             line_count += 1
#         dictionary_bb84["L" + str(round(float(row["L"]), 2))] = float(row['rate'])
#         line_count += 1
#     print(f'Processed {line_count} lines.')
# # add trusted nodes to graph
# trusted_node_pos_graph = pos_graph_0.add_trusted_nodes_at_midpoints(p =0.4, dist = 40)
# pos_graph_0.store_n_k_for_n_state(dictionary, n = 2, c_min = 10000, store_file_location = "cap_needed_multi_15")
#
# # capacity_graph = pos_graph_0.generate_capacity_graph_sources(dictionary)
# # cap_min_graph = capacity_graph.generate_needed_capacity_graph(cap_min=10000)
# # cap_min_graph.store_capacity_edge_graph("cap_needed_23")
# # trusted_node_data = pd.concat([pd.DataFrame({"node": 0, "xcoord": 38.2, "ycoord": 32.2}, index = [0]), pd.DataFrame({"node": 1, "xcoord": 49.9, "ycoord": 69.6}, index = [1]), pd.DataFrame({"node": 2, "xcoord": 59.2, "ycoord": 72.0}, index = [2]), pd.DataFrame({"node": 3, "xcoord": 75.4, "ycoord": 77.4}, index = [3]), pd.DataFrame({"node": 4, "xcoord": 92.7, "ycoord": 29.0}, index = [4])])
# # trusted_node_edge_list = pd.DataFrame({"source": 0, "target": 1}, index = [0])
# # trusted_node_edge_list_to_other_nodes = pd.concat([pd.DataFrame({"source": 0, "target": 0}, index = [0]), pd.DataFrame({"source":0, "target": 9}, index = [1]), pd.DataFrame({"source":1, "target": 0}, index = [1]), pd.DataFrame({"source":1, "target": 11}, index = [1]), pd.DataFrame({"source":2, "target": 4}, index = [1]), pd.DataFrame({"source":2, "target": 9}, index = [1]), pd.DataFrame({"source":3, "target": 6}, index = [1]), pd.DataFrame({"source":3, "target": 12}, index = [1]), pd.DataFrame({"source":4, "target": 3}, index = [1]), pd.DataFrame({"source":4, "target": 4}, index = [1])])
#
# trusted_node_capacity_graph = trusted_node_pos_graph.generate_capacity_graph_trusted_nodes_bb84(dictionary_tf = dictionary, dictionary_bb84=dictionary_bb84)
# trusted_node_capacity_graph.store_capacity_edge_graph("capacity_values_multi_15", node_types = trusted_node_pos_graph.vertex_type, node_data_store_location="node_types_multi_15")
# # key_dict = get_key_dict(cap_min_graph)
#
#
# # TO DO: add functionality to calculate capacity between trusted nodes using BB84 Decoy protocol - DONE
# #    .   Convert from cap_min_graph which is c_{min}-c_{i,j} for all pairs that need to have a commidity associated with
# #        the flow - in particular #commodities = 2* no. of edges in graph
# #    .   Convert from trusted_node_capacity_graph to capacity constraint: c_{i,j} given
# #    .   Write rest of Linear program in cplex
#
#
#
# # position = position_of_next_trusted_node(pos_graph_0, capacity_graph)
# new_graph = pos_graph_0.add_trusted_node(capacity_graph=cap_min_graph)
# cap_graph_trusted_node = new_graph.generate_capacity_graph_trusted_nodes(dictionary)
