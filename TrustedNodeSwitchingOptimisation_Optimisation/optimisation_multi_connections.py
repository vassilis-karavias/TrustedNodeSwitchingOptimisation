import cplex
import networkx as nx
import matplotlib.pyplot as plt
import brute_force
import trusted_nodes_utils
import numpy as np
import time
from optimisationmodel import *


def add_flow_conservation_constraint_multigraph(prob, g, key_dict, cmin):
    """Applies constraints to cplex problem for flow conservation"""
    num_comm=len(key_dict)# checks number of key exchange pairs i.e. number of commodities
    num_nodes=g.number_of_nodes()
    variable_names=[f'x{i}_{j}_{n}_k{k[0]}_{k[1]}' for k in key_dict for i,j,n in list(g.edges(keys = True)) ] # name convention x1_2k3_4 is flow in direction 1 to 2 for key 3 to 4
    flow=[[[int(i+k*num_nodes),int(j+k*num_nodes)], [1.0,-1.0]] for k in range(num_comm) for i,j,n in list(g.edges(keys = True))] # tails are positive, heads negative, ensures source, sink balance

    sn=np.zeros(num_nodes*num_comm) # zeros ensure flow conservation
    count=0
    for pair,num_keys in key_dict.items():
        active_commodity=count*num_nodes
        sn[int(pair[0])+active_commodity]=int(num_keys * cmin) # sets source
        sn[int(pair[1])+active_commodity]=-int(num_keys * cmin) # sets sink
        count+=1 # counter moves forward for next commodity1
    my_senses='E'*num_nodes*num_comm
    prob.linear_constraints.add(senses=my_senses,rhs=sn.tolist())
    prob.variables.add(names=variable_names, columns=flow, types=[prob.variables.type.continuous]*len(variable_names), ub = [cmin/2] * len(variable_names)) # add variables and flow conservation

def add_bidirectional_flow_conservation_constraint_multigraph(prob,g,key_dict, cmin):
    num_nodes=g.number_of_nodes()
    key_dict=trusted_nodes_utils.make_key_dict_bidirectional(key_dict)
    add_flow_conservation_constraint_multigraph(prob, g, key_dict, cmin)#add flow conservation everywhere (incl. souces/sinks)
    rows_to_delete=[key[i]+count*num_nodes for count, key in enumerate(key_dict) for i in range(2)] #list of row indices for constrain matrix which are sinks/sources
    prob.linear_constraints.delete(rows_to_delete) # remove flow conservation at sinks/sources due to unknown source,sink values
    for k in key_dict:
        total_keys=float(key_dict[k] * cmin+key_dict[k[::-1]] * cmin) # total number of keys needed to exchange for pair k
        forward_flow_variables=[f'x{k[0]}_{j}_{n}_k{k[0]}_{k[1]}' for j in g.adj[k[0]] for n in g.adj[k[0]][j] ] # flow out of source for forward commodity
        backwards_flow_variables=[f'x{j}_{k[0]}_{n}_k{k[1]}_{k[0]}' for j in g.adj[k[0]] for n in g.adj[k[0]][j]] # flow into source for backwards commodity
        flow_into_source_variables=[f'x{j}_{k[0]}_{n}_k{k[0]}_{k[1]}' for j in g.adj[k[0]] for n in g.adj[k[0]][j]] # flow into source from forwards commodity (must be zero to avoid loops)
        flow_out_sink_variables=[f'x{k[1]}_{j}_{n}_k{k[0]}_{k[1]}' for j in g.adj[k[1]] for n in g.adj[k[1]][j]] # flow out of sink from forwards commodity (must be zero to avoid loops)
        names=forward_flow_variables+backwards_flow_variables
        lin_expressions=[cplex.SparsePair(ind=names, val=[1.]*len(names)),
                         cplex.SparsePair(ind=flow_into_source_variables, val=[1.]*len(flow_into_source_variables)),
                         cplex.SparsePair(ind=flow_out_sink_variables, val=[1.]*len(flow_out_sink_variables))]
        prob.linear_constraints.add(lin_expr=lin_expressions, senses=['EEE'], rhs=[total_keys, 0., 0.])



def add_capacity_constraint_to_ensure_multigraph(prob, g, key_dict, cmin):
    num_comm = len(key_dict)
    for i,j,n in g.edges(keys = True):
        for k in key_dict:
            # get the indices of the flow variables on the selected edge
            ind_flow = [f'x{i}_{j}_{n}_k{k[0]}_{k[1]}']
            # get the capacity of the edge
            cap_const_1 = cplex.SparsePair(ind=ind_flow, val=[1] * len(ind_flow))
            prob.linear_constraints.add(lin_expr=[cap_const_1], senses='L', rhs=[cmin/2])


def lambda_constraint_multipath(prob, g, key_dict, Lambda):
    num_comm = len(key_dict)
    binary_trusted_variables = []
    # add the lambdas to the variables of problem
    for n in g.nodes:
        if g.nodes[n]["type"] == "T":
            binary_trusted_variables.append(f"delta{n}")
    prob.variables.add(names=binary_trusted_variables,
                          types=[prob.variables.type.binary] * len(binary_trusted_variables))
    explored_edges = {}
    for i,j,n in g.edges:
        if (i,j) not in explored_edges.keys():
            if g.nodes[i]["type"] == "T":
                lambda_constraint_1 = cplex.SparsePair(ind = [f"lambda{i, j}"] + [f"delta{i}"], val = [1] + [-Lambda])
                prob.linear_constraints.add(lin_expr = [lambda_constraint_1], senses = "L", rhs = [0])
            elif g.nodes[j]["type"] == "T":
                lambda_constraint_1 = cplex.SparsePair(ind=[f"lambda{i, j}"] + [f"delta{j}"], val=[1] + [-Lambda])
                prob.linear_constraints.add(lin_expr=[lambda_constraint_1], senses="L", rhs=[0])
            explored_edges[i,j] = 1

def add_capacity_constraint_for_lambda_multi(prob, g, key_dict, Lambda):
    """
    Add the capacity constraint that constraints the number of detectors and sources on the nodes - defined by lambda_{i,j}
    The capacity across an edge cannot be more than the capacity of the connections times the number of devices in the
    connection

    """

    num_comm = len(key_dict)
    binary_trusted_variables = []
    # add the lambdas to the variables of problem
    explored_edges = {}
    for i,j,n in g.edges:
        if (i,j) not in explored_edges.keys():
            binary_trusted_variables.append(f"lambda{i,j}")
            explored_edges[i,j] = 1
    prob.variables.add(names=binary_trusted_variables,
                          types=[prob.variables.type.integer] * len(binary_trusted_variables), ub = [Lambda] * len(binary_trusted_variables))
    explored_edges = {}
    for i,j,n in g.edges:
        if (i,j) not in explored_edges.keys():
            # get the indices of the flow variables on the selected edge
            ind_flow = [f'x{i}_{j}_{m}_k{k[0]}_{k[1]}' for m in range(g.number_of_edges(i,j)) for k in key_dict]
            lambda_ij = [f'lambda{i,j}']
            # get the capacity of the edge
            capacity = [(1/int(g.edges[i,j,m]["capacity"])) for m in range(g.number_of_edges(i,j)) for k in key_dict]
            # add capacity constraint
            cap_const_1 = cplex.SparsePair(ind=ind_flow + lambda_ij, val=capacity + [-1])
            prob.linear_constraints.add(lin_expr=[cap_const_1], senses='L', rhs=[0])
            explored_edges[i,j] = 1


def add_constraint_source_nodes(prob, g, key_dict):
    """
    Adds constraint that ensures that inflow into source nodes is 0 unless it is for the commodity
    required by the source node
    """
    num_comm = len(key_dict)

    for i, nodes in enumerate(g.edges):
        source_node_type = g.nodes[nodes[0]]["type"]
        target_node_type = g.nodes[nodes[1]]["type"]
        # if target node is a source node then only if the commidity if for target node can the flow in be non-zero
        if target_node_type == "S":
            source_node = nodes[0]
            target_node = nodes[1]
            n = nodes[2]
            ind_flow = []
            for j, k in enumerate(key_dict):
                # find the keys with commodity that is not for current target node
                if k[1] != target_node:
                    ind_flow.append(f"x{source_node}_{target_node}_{n}_k{k[0]}_{k[1]}")
            cap_const_2 = []
            # set constraints of these commodities to 0
            for j in range(len(ind_flow)):
                cap_const_2.append(cplex.SparsePair(ind=[ind_flow[j]], val=[1]))
            prob.linear_constraints.add(lin_expr=cap_const_2, senses='E' * len(cap_const_2),
                                           rhs=[0] * len(cap_const_2))



def add_minimise_overall_cost_objective_multigraph(prob, g, cost_node, cost_connection):
    """
    Add the objective to minimise the cost of the network sum_{j in T} c_{j}deta_j + sum_{i,j in E} C_{i,j} lambda_{i,j}

    Parameters
    ----------
    cost_node : The cost of turning on the node
    cost_connection : The cost of adding extra connection
    -------
    """
    obj_vals = []
    for n in g.nodes:
        if g.nodes[n]["type"] == "T":
            obj_vals.append((f"delta{n}", cost_node))
    explored_edges = {}
    for i,j,n in g.edges:
        if (i, j) not in explored_edges.keys():
            obj_vals.append((f"lambda{i, j}", cost_connection))
            explored_edges[i,j] = 1
    prob.objective.set_linear(obj_vals)
    prob.objective.set_sense(prob.objective.sense.minimize)



def initial_optimisation_cost_multi_graph(g, key_dict, cmin, time_limit = 1e5, cost_node = 1, cost_connection= 0.1, Lambda = 100):
    """
    set up and solve the problem for minimising the overall cost of the network
    """
    t_0 = time.time()
    print("Start Optimisation")
    prob = cplex.Cplex()
    add_bidirectional_flow_conservation_constraint_multigraph(prob, g, key_dict, cmin)

    # add_capacity_constraint(prob, g, key_dict, Lambda=Lambda)
    add_capacity_constraint_for_lambda_multi(prob, g, key_dict, Lambda=Lambda)
    lambda_constraint_multipath(prob, g, key_dict, Lambda)
    # add_minimise_trusted_nodes_objective(prob, g)
    add_capacity_constraint_to_ensure_multigraph(prob, g, key_dict, cmin)
    add_minimise_overall_cost_objective_multigraph(prob, g, cost_node, cost_connection)

    prob.write("test_multi.lp")
    prob.parameters.lpmethod.set(3)
    prob.parameters.mip.limits.cutpasses.set(1)
    prob.parameters.mip.strategy.probe.set(-1)
    prob.parameters.mip.strategy.variableselect.set(4)
    prob.parameters.mip.strategy.kappastats.set(1)
    prob.parameters.mip.tolerances.mipgap.set(float(0.01))
    # prob.parameters.simplex.limits.iterations = 50
    print(prob.parameters.get_changed())
    prob.parameters.timelimit.set(time_limit)
    t_1 = time.time()
    print("Time to set up problem: " + str(t_1-t_0))
    prob.solve()
    t_2 = time.time()
    print("Time to solve problem: " + str(t_2 - t_1))
    print(f"The minimum Cost of Network: {prob.solution.get_objective_value()}")
    print(f"Number of Variables = {prob.variables.get_num()}")
    print(f"Number of Conditions = {prob.linear_constraints.get_num()}")
    sol_dict = create_sol_dict(prob)
    flow_dict, binary_dict, lambda_dict = split_sol_to_flow_delta_lambda(sol_dict)
    trusted_nodes = 0
    for key in binary_dict:
        trusted_nodes += binary_dict[key]
    print(f"Number of Trusted Nodes = {trusted_nodes}")
    return sol_dict, prob



key_dict, g, position_graph = trusted_nodes_utils.import_problem_from_files_multigraph_multiple_graphs_position_graph(nk_file = "small_graphs_for_test_cap_needed.csv", capacity_values_file = "small_graphs_for_test_edge_data.csv", node_type_file = "small_graphs_for_test_node_types.csv", position_node_file= "small_graphs_for_test_position_graph_nodes.csv", position_edge_file = "small_graphs_for_test_position_graph_edges.csv")
for key in g.keys():
    initial_optimisation_cost_multi_graph(g[key], key_dict[key], cmin=10000, time_limit=1e5, cost_node=10,
                                        cost_connection=1,
                                        Lambda=100)