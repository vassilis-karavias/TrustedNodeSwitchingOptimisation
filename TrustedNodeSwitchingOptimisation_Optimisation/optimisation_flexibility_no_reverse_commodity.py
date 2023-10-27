import cplex
import networkx as nx
import matplotlib.pyplot as plt
import scipy.optimize as opt
import optimisationmodel
from optimisation_no_reverse_commodity import *
import trusted_nodes_utils
import numpy as np
import time


def get_number_of_nodes(key_dict):
    number_nodes = 0
    for key in key_dict.keys():
        if key[0] > number_nodes:
            number_nodes = key[0]
        if key[1] > number_nodes:
            number_nodes = key[1]
    return number_nodes # not measured from 0.


def add_flow_conservation_constraint_flex(prob, g, key_dict, cmin):
    """Applies constraints to cplex problem for flow conservation
    key_dict contains only keys with terms k[0] < k[1]
    """
    num_comm=len(key_dict)# checks number of key exchange pairs i.e. number of commodities
    num_nodes=g.number_of_nodes()
    variable_names=[f'x{i}_{j}_k{k[0]}_{k[1]}' for k in key_dict for i,j in list(g.edges) ] # name convention x1_2k3_4 is flow in direction 1 to 2 for key 3 to 4
    flow=[[[int(i+k*num_nodes),int(j+k*num_nodes)], [1.0,-1.0]] for k in range(num_comm) for i,j in g.edges ] # tails are positive, heads negative, ensures source, sink balance

    sn=np.zeros(num_nodes*num_comm) # zeros ensure flow conservation
    count=0
    for pair,num_keys in key_dict.items():
        active_commodity=count*num_nodes
        sn[int(pair[0]) + active_commodity] = int(num_keys * cmin)  # sets source
        sn[int(pair[1]) + active_commodity] = -int(num_keys * cmin)  # sets sink
        count+=1 # counter moves forward for next commodity
    my_senses='E'*num_nodes*num_comm
    prob.linear_constraints.add(senses=my_senses,rhs=sn.tolist())
    prob.variables.add(names=variable_names, columns=flow, types=[prob.variables.type.continuous] * len(variable_names), ub=[cmin] * len(variable_names))  # add variables and flow conservation



def add_capacity_constraint_to_ensure_multipath(prob, g, key_dict, cmin):
    for i in g.nodes:
        for k in key_dict:
            if k[0] != i:
                ind_flow = []
                val = []
                for j in g.adj[i]:
                    if k[0] < k[1]:
                        ind_flow.extend([f"x{i}_{j}_k{k[0]}_{k[1]}"])
                        val.extend([1])
                lin_expr = [cplex.SparsePair(ind=ind_flow, val=val)]
                if isinstance(cmin, dict):
                    prob.linear_constraints.add(lin_expr=lin_expr, senses='L' * len(lin_expr),
                                                     rhs=[cmin[k]] * len(lin_expr))
                else:
                    prob.linear_constraints.add(lin_expr=lin_expr, senses='L' * len(lin_expr),
                                                     rhs=[cmin] * len(lin_expr))


def initial_optimisation_cost_reduction(g, key_dict, cmin, time_limit = 1e5, cost_node = 1, cost_connection= 0.1, Lambda = 100):
    """
    set up and solve the problem for minimising the overall cost of the network
    """
    t_0 = time.time()
    print("Start Optimisation")
    prob = cplex.Cplex()
    add_flow_conservation_constraint_flex(prob, g, key_dict, cmin)
    add_flow_in_source_out_sink_zero(prob, g, key_dict)
    # add_capacity_constraint(prob, g, key_dict, Lambda=Lambda)
    add_lambda_constraint(prob, g, key_dict, Lambda=Lambda)
    add_constraint_source_nodes(prob, g, key_dict)
    lambda_constraint(prob, g, key_dict, Lambda)
    # add_minimise_trusted_nodes_objective(prob, g)
    add_minimise_overall_cost_objective(prob, g, cost_node, cost_connection)
    add_capacity_constraint_to_ensure_multipath(prob, g, key_dict, cmin)
    prob.write("test_1.lp")
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
    sol_dict = optimisationmodel.create_sol_dict(prob)
    flow_dict, binary_dict, lambda_dict = optimisationmodel.split_sol_to_flow_delta_lambda(sol_dict)
    trusted_nodes = 0
    for key in binary_dict:
        trusted_nodes += binary_dict[key]
    print(f"Number of Trusted Nodes = {trusted_nodes}")
    return sol_dict, prob




key_dict, g = trusted_nodes_utils.import_problem_from_files_flexibility_multiple_graphs("one_path_small_graphs_for_test_cap_needed.csv", "one_path_small_graphs_for_test_edge_data.csv", "one_path_small_graphs_for_test_node_types.csv")
for key in g.keys():
    sol_dict, prob = initial_optimisation_cost_reduction(g[key], key_dict[key], cmin = 50000, time_limit=1e5, cost_node=10, cost_connection=3,
                                                Lambda=24)
