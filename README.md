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
The with switchind storage, i.e. the latter method, accounts for the switch loss. dictionary_bb84 is the location of a csv file with the key rate dictionary for the decoy BB84 used in [length, capacity] format. We provide some dictionaries. However your own can also be used. store_file_location is the location to store the edge data file needed for the optimisation in a csv file of the form [ID, source, target, distance] where ID is the graph ID. node_data_store_location is the location to store the node data file in csv of the form: [ID, node, xcoord,ycoord, type] where type denotes whether the node is a user "S" or a trusted node "T". 
