# TrustedNodeSwitchingOptimisation
Repository for TN switching optimisation

# dependencies
numpy: 1.20.1+  
graph-tool: 2.37+  
pandas: 1.5.2+  
scipy: 1.7.3+  
cplex: V20.1  
matplotlib: 3.6.2+  
networkx: 2.8.8+  

# How to use  
## Preprocessing  
To generate random graphs use the python file  
**TrustedNodeSwitchingOptimisation_Preprocessing/test.py**  
In this you can select what topology the graphs should be in, the size of the box that constains the graph in km and other parameters specific to the topology. The current method will generate 1 random graph with 25 nodes although this can be changed as required. The file will store the graph using the  
**store_optimal_solution_for_trusted_node_inv_distances_bb84(graph_node.graph, distance, xcoords,  ycoords, data_save_name,  node_data_save_name)**
method. graph_node is a class of SpecifiedTopologyGraph(). It allows for easy generation of the random type of graph desired by calling the desired function in the class with appropriate input parameters. In test.py, this is already set up to generate the appropriate random graphs required. The graph can be extracted by graph_node.graph, the distance is the size of the box that contains the graph in km. xcoords and ycoords can be extracted by graph_node.xcoords, graph_node.ycoords respecively. The method will save 2 output files under the names given by data_save_name and node_data_save_name. The first file stores the edge data in a csv file with the form [ID, source, target, distance] while the second file stores the nodes in a csv file with the form [ID, node, type, xcoord, ycoord]. ID represents the label for the current graph and allows multiple graphs to be stored in a single file.  
To convert these graphs into the appropriate graphs for the optimisation the python file  
**TrustedNodeSwitchingOptimisation_Preprocessing/main.py**  
is provided. The no routing model, where connections are considered point to point only can be generated using the class  
**Position_Graph_No_Switching_Set_Distances()**  
The method in this class  
**create_graphs(import_path_nodes, import_path_edges, no_source_nodes, db_switch)**
can be used to import the graphs into the class. import_path_nodes is the location of the node csv file generated in test.py and import_path_edges is the location of the edge csv file generated in test.py. no_source_nodes specifies how many of the graph nodes are to be user nodes. All remaining nodes are trusted nodes. db_switch is the db_switch loss of the graph.
To investigate the routing model, the same can be done but now the class  
**Position_Graph_Set_Distances()**  
should be used. Graphs can be stored with and without switching in csv files using the methods  
**new_graphs_2.store_capacity_edge_graph_distance_bb84_no_switching(dictionary_bb84, store_file_location,  node_data_store_location)**  
**new_graphs_2.store_capacity_edge_graph_distance_bb84(dictionary_bb84, store_file_location,  node_data_store_location)**
The with switchind storage, i.e. the latter method, accounts for the switch loss. dictionary_bb84 is the location of a csv file with the key rate dictionary for the decoy BB84 used in [length, capacity] format. We provide some dictionaries. However your own can also be used. store_file_location is the location to store the edge data file needed for the optimisation in a csv file of the form [ID, source, target, capacity, distance] where ID is the graph ID. node_data_store_location is the location to store the node data file in csv of the form: [ID, node, xcoord,ycoord, type] where type denotes whether the node is a user "S" or a trusted node "T".  
To store the files neded for the cap_needed, use the method:  
**new_graphs_2.store_n_k_for_n_state(n,store_file_location)**  
where n is the number of unique paths needed with cmin capacity and store_file_location is the location to store the csv file with the capacity required in the form [ID, source, target, N_k]  
If you also want to store the physical position data, this can be done by:  
**new_graphs_2.store_position_graph(node_data_store_location, edge_data_store_location )**
where node_data_store_location is the path to store the node data in form [ID, node, xcoord, ycoord, type] and edge_data_store_location is the path to store the edge data in the form [ID, source, target, distance].
## Optimisations
### Importing graphs
To carry out the optimisations, the folder TristedNodeSwitchingOptimisation_Optimisation is provided. trusted_node_utils provides several methods to import the data into graphs.   
**trusted_node_utils.import_problem_from_files_flexibility_multiple_graphs(nk_file, capacity_values_file, node_type_file)**  
imports the data into an array of key_dicts and graphs. The import deals with multiple graphs based on ID and outputs dictionaries with key given by the graph ID. nk_file is the cap_needed file location, capacity_values_file is the edge data file generated in preprocessing and node_type_file is the node data file generated in preprocessing. If you would like to import data for a single graph, you can instead use:  
**trusted_node_utils.import_problem_from_files_flexibility(nk_file, capacity_values_file, node_type_file)**  
now the files no longer have an ID column. If you would like the physical position graphs as well use:  
**trusted_node_utils.import_problem_from_files_flexibility_multiple_graphs_position_graph(nk_file, capacity_values_file, node_type_file, position_node_file, position_edge_file)**  
where position_node_file is the position node data file location and position_edge_file is the position edge data file location generated in preprocessing.  
If the graph has more than one edge between any two pair of nodes use:  
**trusted_node_utils.import_problem_from_files_multigraph_multiple_graphs_position_graph(nk_file, capacity_values_file, node_type_file, position_node_file, position_edge_file)**  
It is also possible to use a file with cmin values instead of N_k, i.e. use a file where instead of storing the number of unique paths that need capacity cmin (which has yet to be specified), you could instead use a file where this column is changed to "cmin" aka the minimum required capacity between edges, using the method:
**trusted_node_utils.import_problem_from_files_multiple_graphs(cap_needed_file, capacity_values_file, node_type_file)**  
cap_needed_file is the location of the file with [ID, source, target, cmin].  
### Optimisations
For the switching optimisation the class  
**optim = optimisation_switching_model_with_calibration.Optimisation_Switching_Calibration_fixed_frac_calibration_time(prob, g,  key_dict)**
is provided. prob is the cplex problem class that can be generated as cplex.Cplex(). g is the current graph of the problem, you can extract this from the imported graph dictionary by graphs[key], key_dict is the dictionary of n_k values needed for each edge of the graph, this can be obtained by key_dict[key]. It should be noted that the key_dict should be bidirectional in the switching case: Tij contains i,j and j,i. If it is not then you can make it bidirectional by calling:  
**trusted_nodes_utils.make_key_dict_bidirectional(key_dict[key])**  
To run the optimisation and get the results use:  
**sol_dict, prob, time_taken = optim.initial_optimisation_cost_reduction(cmin,
                                                                                           time_limit,
                                                                                           cost_on_trusted_node,
                                                                                           cost_detector,
                                                                                           cost_source,
                                                                                           f_switch,
                                                                                           Lambda)**  
the output sol_dict is a dictionary of {variable: value} for the optimal solution, prob is the cplex problem class and time_taken is the time the optimisation took to solve. cmin is the value of cmin desired this can either be an int for cmin to be equal for all user pairs or can be a dict of {(i,j): cmin} for user pairs (i,j). time_limit is how long to run the optimisation before terminating, cost_on_trusted_node is the cost of turning the trusted node on, cost_detector is the detector cost, cost_source is the source cost, f_switch is the switching calibration time, and Lambda is the maximum number of devices on each edge.  
For the no switching model the class:  
**optim = optimisation_no_reverse_commodity.Optimisation_Problem_No_Switching(prob,
                                                                                            g,
                                                                                            key_dict)**  
is provided. In this case key_dict should not be bidirectional. The no switch files generated should be used to import the values of g and key_dict instead. To run the optimisation and get the results use:  
**sol_dict, prob, time = optim.initial_optimisation_cost_reduction(cmin, time_limit,
                                                                                 cost_node,
                                                                                 cost_connection
                                                                                 Lambda)**
cost_node is the cost of turning on the trusted node, cost_connection is the cost of the detector and source combined.  
### Methods for In Depth Analysis  
Methods to do in depth analysis are provided in optimisation_switching_model_with_calibration. For example to investigate the time variation use:  
**time_variation_analysis(cap_needed_location, edge_data_location, node_type_location, position_node_file, position_edge_file,
                            cmin, time_limit=1e5, cost_on_trusted_node=1,cost_detector=0.1, cost_source=0.01, f_switch=0.1, Lambda=100, data_storage_location = None)**
cap_needed_location is nk_file. data_storage_location is optional and if given will store the data in each iteration.
