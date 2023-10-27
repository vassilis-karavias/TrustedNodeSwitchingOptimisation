import cplex
import networkx as nx
import matplotlib.pyplot as plt
import scipy.optimize as opt

from optimisationmodel import *
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
    """Applies constraints to cplex problem for flow conservation"""
    num_comm=len(key_dict)# checks number of key exchange pairs i.e. number of commodities
    num_nodes=g.number_of_nodes()
    variable_names=[f'x{i}_{j}_k{k[0]}_{k[1]}' for k in key_dict for i,j in list(g.edges) ] # name convention x1_2k3_4 is flow in direction 1 to 2 for key 3 to 4
    flow=[[[int(i+k*num_nodes),int(j+k*num_nodes)], [1.0,-1.0]] for k in range(num_comm) for i,j in g.edges ] # tails are positive, heads negative, ensures source, sink balance

    sn=np.zeros(num_nodes*num_comm) # zeros ensure flow conservation
    count=0
    for pair,num_keys in key_dict.items():
        active_commodity=count*num_nodes
        sn[int(pair[0])+active_commodity]=int(num_keys * cmin) # sets source
        sn[int(pair[1])+active_commodity]=-int(num_keys * cmin) # sets sink
        count+=1 # counter moves forward for next commodity
    my_senses='E'*num_nodes*num_comm
    prob.linear_constraints.add(senses=my_senses,rhs=sn.tolist())
    prob.variables.add(names=variable_names, columns=flow, types=[prob.variables.type.continuous]*len(variable_names), ub = [cmin] * len(variable_names)) # add variables and flow conservation

def add_bidirectional_flow_conservation_constraint_flex(prob,g,key_dict, cmin):
    num_nodes=g.number_of_nodes()
    key_dict=trusted_nodes_utils.make_key_dict_bidirectional(key_dict)
    add_flow_conservation_constraint_flex(prob, g, key_dict, cmin)#add flow conservation everywhere (incl. souces/sinks)
    rows_to_delete=[key[i]+count*num_nodes for count, key in enumerate(key_dict) for i in range(2)] #list of row indices for constrain matrix which are sinks/sources
    prob.linear_constraints.delete(rows_to_delete) # remove flow conservation at sinks/sources due to unknown source,sink values
    for k in key_dict:
        total_keys=float(key_dict[k] * cmin+key_dict[k[::-1]] * cmin) # total number of keys needed to exchange for pair k
        forward_flow_variables=[f'x{k[0]}_{j}_k{k[0]}_{k[1]}' for j in g.adj[k[0]]] # flow out of source for forward commodity
        backwards_flow_variables=[f'x{j}_{k[0]}_k{k[1]}_{k[0]}' for j in g.adj[k[0]]] # flow into source for backwards commodity
        flow_into_source_variables=[f'x{j}_{k[0]}_k{k[0]}_{k[1]}' for j in g.adj[k[0]]] # flow into source from forwards commodity (must be zero to avoid loops)
        flow_out_sink_variables=[f'x{k[1]}_{j}_k{k[0]}_{k[1]}' for j in g.adj[k[1]]] # flow out of sink from forwards commodity (must be zero to avoid loops)
        names=forward_flow_variables+backwards_flow_variables
        lin_expressions=[cplex.SparsePair(ind=names, val=[1.]*len(names)),
                         cplex.SparsePair(ind=flow_into_source_variables, val=[1.]*len(flow_into_source_variables)),
                         cplex.SparsePair(ind=flow_out_sink_variables, val=[1.]*len(flow_out_sink_variables))]
        prob.linear_constraints.add(lin_expr=lin_expressions, senses=['EEE'], rhs=[total_keys, 0., 0.])


def add_bidirectional_flow_conservation_constraint_flex_rescaled(prob,g,key_dict, cmin):
    num_nodes=g.number_of_nodes()
    key_dict=trusted_nodes_utils.make_key_dict_bidirectional(key_dict)
    add_flow_conservation_constraint_flex(prob, g, key_dict, cmin)#add flow conservation everywhere (incl. souces/sinks)
    rows_to_delete=[key[i]+count*num_nodes for count, key in enumerate(key_dict) for i in range(2)] #list of row indices for constrain matrix which are sinks/sources
    prob.linear_constraints.delete(rows_to_delete) # remove flow conservation at sinks/sources due to unknown source,sink values
    for k in key_dict:
        total_keys=float(key_dict[k] * 10 +key_dict[k[::-1]]  * 10) # total number of keys needed to exchange for pair k
        forward_flow_variables=[f'x{k[0]}_{j}_k{k[0]}_{k[1]}' for j in g.adj[k[0]]] # flow out of source for forward commodity
        backwards_flow_variables=[f'x{j}_{k[0]}_k{k[1]}_{k[0]}' for j in g.adj[k[0]]] # flow into source for backwards commodity
        flow_into_source_variables=[f'x{j}_{k[0]}_k{k[0]}_{k[1]}' for j in g.adj[k[0]]] # flow into source from forwards commodity (must be zero to avoid loops)
        flow_out_sink_variables=[f'x{k[1]}_{j}_k{k[0]}_{k[1]}' for j in g.adj[k[1]]] # flow out of sink from forwards commodity (must be zero to avoid loops)
        names=forward_flow_variables+backwards_flow_variables
        lin_expressions=[cplex.SparsePair(ind=names, val=[1.]*len(names)),
                         cplex.SparsePair(ind=flow_into_source_variables, val=[1.]*len(flow_into_source_variables)),
                         cplex.SparsePair(ind=flow_out_sink_variables, val=[1.]*len(flow_out_sink_variables))]
        prob.linear_constraints.add(lin_expr=lin_expressions, senses=['EEE'], rhs=[total_keys, 0., 0.])




def add_capacity_constraint_to_ensure_multipath_old(prob, g, key_dict, cmin):
    num_comm = len(key_dict)
    for i, nodes in enumerate(g.edges):
        for k in range(num_comm):
            # get the indices of the flow variables on the selected edge
            ind_flow = [i + k * g.number_of_edges()]
            # get the capacity of the edge
            cap_const_1 = cplex.SparsePair(ind=ind_flow, val=[1] * len(ind_flow))
            prob.linear_constraints.add(lin_expr=[cap_const_1], senses='L', rhs=[cmin/2])


def add_capacity_constraint_to_ensure_multipath(prob, g, key_dict, cmin):
    num_comm = len(key_dict)
    for i, nodes in enumerate(g.edges):
        for k in key_dict:
            if nodes[0] < nodes[1] and k[0] < k[1]:
                # get the indices of the flow variables on the selected edge for given commodity
                ind_flow = [f"x_{nodes[0]}_{nodes[1]}_k{k[0]}_{k[1]}", f"x_{nodes[0]}_{nodes[1]}_k{k[1]}_{k[0]}", f"x_{nodes[1]}_{nodes[0]}_k{k[0]}_{k[1]}"
                            , f"x_{nodes[1]}_{nodes[0]}_k{k[1]}_{k[0]}"]
                # get the capacity of the edge
                cap_const_1 = cplex.SparsePair(ind=ind_flow, val=[1] * len(ind_flow))
                prob.linear_constraints.add(lin_expr=[cap_const_1], senses='L', rhs=[cmin])

def add_capacity_constraint_to_ensure_multipath_rescaled(prob, g, key_dict):
    num_comm = len(key_dict)
    for i, nodes in enumerate(g.edges):
        for k in range(num_comm):
            # get the indices of the flow variables on the selected edge
            ind_flow = [i + k * g.number_of_edges()]
            # get the capacity of the edge
            cap_const_1 = cplex.SparsePair(ind=ind_flow, val=[1] * len(ind_flow))
            prob.linear_constraints.add(lin_expr=[cap_const_1], senses='L', rhs=[5])



def get_difference_in_cost_by_difference_in_solution(soln_pool):
    """
    gets the difference between the values of the cost (objective function) for different solutions based on the
    distance between the solutions.
    """
    # separate the pools into binary values and lambda solutions (the number of detectors on each node)
    binary_soln_pool = []
    lambda_soln_pool = []
    for soln in soln_pool:
        flow_dict, binary_dict, lambda_dict = split_sol_to_flow_delta_lambda(soln)
        binary_soln_pool.append(binary_dict)
        lambda_soln_pool.append(lambda_dict)
    # find optimal solution and et the optimal values for the solution.
    optimal_soln_position, optimal_soln_value = get_optimal_soln(lambda_soln_pool)
    binary_optimal_soln, lambda_optimal_soln = binary_soln_pool[optimal_soln_position], lambda_soln_pool[
        optimal_soln_position]
    difference_solution_dictionary = {}
    for i in range(len(binary_soln_pool)):
        current_binary_soln, current_lambda_soln = binary_soln_pool[i], lambda_soln_pool[i]
        delta = 0.0
        for key in current_binary_soln.keys():
            delta += np.abs(current_binary_soln[key] - binary_optimal_soln[key])
        difference_objective_function = current_lambda_soln["objective"] - lambda_optimal_soln["objective"]
        if delta not in difference_solution_dictionary.keys():
            difference_solution_dictionary[delta] = [difference_objective_function]
        else:
            difference_solution_dictionary[delta].append(difference_objective_function)
    return difference_solution_dictionary






def initial_optimisation_cost_reduction(g, key_dict, cmin, time_limit = 1e5, cost_node = 1, cost_connection= 0.1, Lambda = 100):
    """
    set up and solve the problem for minimising the overall cost of the network
    """
    t_0 = time.time()
    print("Start Optimisation")
    prob = cplex.Cplex()
    add_bidirectional_flow_conservation_constraint_flex(prob, g, key_dict, cmin)

    # add_capacity_constraint(prob, g, key_dict, Lambda=Lambda)
    add_capacity_constraint_for_lambda(prob, g, key_dict, Lambda=Lambda)
    lambda_constraint(prob, g, key_dict, Lambda)
    add_constraint_source_nodes(prob, g, key_dict)
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
    sol_dict = create_sol_dict(prob)
    flow_dict, binary_dict, lambda_dict = split_sol_to_flow_delta_lambda(sol_dict)
    trusted_nodes = 0
    for key in binary_dict:
        trusted_nodes += binary_dict[key]
    print(f"Number of Trusted Nodes = {trusted_nodes}")
    return sol_dict, prob




def initial_optimisation_cost_reduction_rescaled(g, key_dict, cmin, time_limit = 1e5, cost_node = 1, cost_connection= 0.1, Lambda = 100):
    """
    set up and solve the problem for minimising the overall cost of the network
    """
    t_0 = time.time()
    print("Start Optimisation")
    prob = cplex.Cplex()
    add_bidirectional_flow_conservation_constraint_flex_rescaled(prob, g, key_dict, cmin)

    # add_capacity_constraint_rescaled(prob, g, key_dict, cmin, Lambda=Lambda)
    add_capacity_constraint_for_lambda_rescaled(prob, g, key_dict, Lambda = Lambda, cmin = cmin)#
    lambda_constraint(prob, g, key_dict, Lambda)
    add_constraint_source_nodes(prob, g, key_dict)
    # add_minimise_trusted_nodes_objective(prob, g)
    add_capacity_constraint_to_ensure_multipath_rescaled(prob, g, key_dict)
    add_minimise_overall_cost_objective(prob, g, cost_node, cost_connection)

    prob.write("test_1.lp")
    prob.parameters.lpmethod.set(prob.parameters.lpmethod.values.network)
    prob.parameters.mip.limits.cutpasses.set(1)
    prob.parameters.mip.strategy.probe.set(-1)
    prob.parameters.mip.strategy.variableselect.set(4)
    prob.parameters.mip.strategy.kappastats.set(1)
    prob.parameters.mip.tolerances.mipgap.set(float(0.01))
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


def initial_optimisation_cost_reduction_multiple_solns(g, key_dict, cmin, time_limit = 1e5, cost_node = 1, cost_connection= 0.1, Lambda = 100):
    """
    set up and solve the problem for minimising the overall cost of the network - finds multiple solns.
    """
    t_0 = time.time()
    print("Start Optimisation")
    prob = cplex.Cplex()
    add_bidirectional_flow_conservation_constraint_flex(prob, g, key_dict, cmin)
    add_capacity_constraint(prob, g, key_dict, Lambda=Lambda)
    add_capacity_constraint_for_lambda(prob, g, key_dict, Lambda=Lambda)
    # add_minimise_trusted_nodes_objective(prob, g)
    add_minimise_overall_cost_objective(prob, g, cost_node, cost_connection)
    # add_capacity_constraint_to_ensure_multipath(prob, g, key_dict, cmin)
    prob.write("test_1.lp")
    prob.parameters.lpmethod.set(prob.parameters.lpmethod.values.network)
    prob.parameters.mip.limits.cutpasses.set(1)
    prob.parameters.mip.strategy.probe.set(-1)
    prob.parameters.mip.pool.intensity = 4
    prob.parameters.mip.tolerances.mipgap = 0.6
    prob.parameters.mip.limits.populate = 100000000
    prob.parameters.mip.strategy.variableselect.set(0)
    prob.parameters.mip.strategy.kappastats.set(1)
    print(prob.parameters.get_changed())
    prob.parameters.timelimit.set(time_limit)

    try:
        prob.populate_solution_pool()
        numsol = prob.solution.pool.get_num()
        print(numsol)
        sol_pool = []
        for i in range(numsol):
            # get the solution dictionary for each possible solution
            soln_dict = prob.variables.get_names()
            x_i = prob.solution.pool.get_values(i)
            # objective_value = {"objective": prob.solution.pool.get_objective_value(i)}
            sol_dict = {soln_dict[idx]: (x_i[idx]) for idx in range(prob.variables.get_num())}
            sol_dict["objective"] = prob.solution.pool.get_objective_value(i)
            sol_pool.append(sol_dict)
        return sol_pool

    except:
        print("Exception raised during populate")
        return []


def get_region_solution(cap_needed, capacity_values, node_types, cmin, time_limit = 1e5, cost_node = 1, cost_connection= 0.1, Lambda = 100):
    key_dict, g = trusted_nodes_utils.import_problem_from_files_flexibility_multiple_graphs(cap_needed, capacity_values, node_types)
    i = 0
    for key in g.keys():
        try:
            t_0 = time.time()
            soln_pool = initial_optimisation_cost_reduction_multiple_solns(g = g[key], key_dict = key_dict[key], cmin = cmin,
                                                                               time_limit = time_limit, cost_node = cost_node,
                                                                               cost_connection= cost_connection, Lambda = Lambda)
            get_difference_in_cost_by_difference_in_solution(soln_pool)

            # binary_soln_pool = []
            # lambda_soln_pool = []
            # for soln in soln_pool:
            #     flow_dict, binary_dict, lambda_dict = split_sol_to_flow_delta_lambda(soln)
            #     binary_soln_pool.append(binary_dict)
            #     lambda_soln_pool.append(lambda_dict)
            # optimal_soln_position, optimal_soln_value = get_optimal_soln(lambda_soln_pool)
            # binary_optimal_soln, lambda_optimal_soln = binary_soln_pool[optimal_soln_position], lambda_soln_pool[optimal_soln_value]

        except:
            i += 1
            print("Error in Optimisation Program")


def early_stop_comparison(cap_needed, capacity_values, node_types, cmin, time_limit_early_stop = 1e3, cost_node = 1, cost_connection= 0.1, Lambda = 100):
    key_dict, g = trusted_nodes_utils.import_problem_from_files_flexibility_multiple_graphs(cap_needed, capacity_values,
                                                                                            node_types)
    i = 0
    difference_early_stop_array = []
    for key in g.keys():
        try:
            sol_dict, prob = initial_optimisation_cost_reduction(g[key], key_dict[key], cmin = cmin, time_limit=1e5, cost_node=cost_node, cost_connection=cost_connection,
                                                Lambda=Lambda)
            sol_dict_early_stop, prob_early_stop = initial_optimisation_cost_reduction(g[key], key_dict[key], cmin = cmin, time_limit=time_limit_early_stop, cost_node=cost_node, cost_connection=cost_connection,
                                                Lambda=Lambda)

            difference_early_stop = prob_early_stop.solution.get_objective_value() - prob.solution.get_objective_value()
            difference_early_stop_array.append(difference_early_stop)
        except:
            print("Error 404: URL not found - not really, just something went wrong with the program")



def time_taken_with_increasing_number_of_nodes(cap_needed, capacity_values, node_types, cmin, cost_node = 1, cost_connection= 0.1, Lambda = 5):
    key_dict, g = trusted_nodes_utils.import_problem_from_files_flexibility_multiple_graphs(cap_needed, capacity_values,
                                                                                            node_types)

    time_taken = {}
    for key in key_dict.keys():
        try:
            t_0 = time.time()
            sol_dict, prob = initial_optimisation_cost_reduction(g[key], key_dict[key], cmin = cmin, time_limit=1e5, cost_node=cost_node, cost_connection=cost_connection,
                                                Lambda=Lambda)
            if check_solvable(prob):
                t_1 = time.time()
                number_nodes = get_number_of_nodes(key_dict[key])
            if number_nodes in time_taken.keys():
                time_taken[number_nodes].append(t_1 - t_0)
            else:
                time_taken[number_nodes] = [t_1 - t_0]
        except:
            continue
    time_costs_mean_std = {}
    x = []
    y = []
    yerr = []
    for key in time_taken:
        time_costs_mean_std[key] = [np.mean(time_taken[key]), np.std(time_taken[key])]
        x.append(key)
        y.append(time_costs_mean_std[key][0])
        yerr.append(time_costs_mean_std[key][1])
        # x                     y               yerr
    plt.errorbar(x, y, yerr=yerr, color="r")
    plt.xlabel("No. Nodes in Graph", fontsize=10)
    plt.ylabel("Time/s", fontsize=10)
    plt.legend(loc='upper right', fontsize='medium')
    plt.savefig("time_investigation.png")
    plt.show()



# objective function
def objective(x, a, b, c):
    return a * np.power(x,2) + b * x + c



def time_taken_vs_number_on_trusted_nodes(cap_needed, capacity_values, node_types, position_nodes, position_edges, cmin, cost_node = 1, cost_connection= 0.1, Lambda = 100):
    key_dict, g, position_g = trusted_nodes_utils.import_problem_from_files_flexibility_multiple_graphs_position_graph(cap_needed, capacity_values,
                                                                                            node_types, position_node_file = position_nodes, position_edge_file = position_edges)

    time_taken = {}
    for key in key_dict.keys():
        try:
            t_0 = time.time()
            sol_dict, prob = initial_optimisation_cost_reduction(g[key], key_dict[key], cmin = cmin, time_limit=1e5, cost_node=cost_node, cost_connection=cost_connection,
                                                    Lambda=Lambda)
            if check_solvable(prob):
                t_1 = time.time()
                flow_dict, binary_dict, lambda_dict = split_sol_to_flow_delta_lambda(sol_dict)
                plot_graph_solution(g[key], binary_dict, key_dict[key])
                plot_position_graph(position_g[key], binary_dict)
                trusted_nodes = 0
                for key in binary_dict:
                    trusted_nodes += binary_dict[key]
            if trusted_nodes in time_taken.keys():
                time_taken[trusted_nodes].append(t_1 - t_0)
            else:
                time_taken[trusted_nodes] = [t_1 - t_0]
        except:
            continue
    time_costs_mean_std = {}
    x = []
    y = []
    yerr = []
    time_taken = dict(sorted(time_taken.items()))
    for key in time_taken:
        time_costs_mean_std[key] = [np.mean(time_taken[key]), np.std(time_taken[key])]
        x.append(key)
        y.append(time_costs_mean_std[key][0])
        yerr.append(time_costs_mean_std[key][1])
        # x                     y               yerr
    # fit curve
    popt, _ = opt.curve_fit(objective, x, y)
    a, b, c = popt
    x_line = np.arange(min(x), max(x)+0.1, 0.1)
    y_line = objective(x_line, a, b, c)
    plt.errorbar(x, y, yerr=yerr, fmt='.', color="r")
    plt.plot(x_line, y_line, '-', color='red')
    plt.xlabel("No. On Trusted Nodes in Graph", fontsize=10)
    plt.ylabel("Time/s", fontsize=10)
    plt.legend(loc='upper right', fontsize='medium')
    plt.savefig("time_investigation_trusted_nodes.png")
    plt.show()


def time_taken_recaling_comp(cap_needed, capacity_values, node_types, cmin, cost_node = 1, cost_connection= 0.1, Lambda = 100):
    key_dict, g = trusted_nodes_utils.import_problem_from_files_flexibility_multiple_graphs(cap_needed, capacity_values,
                                                                                            node_types)

    time_taken = []
    values = []
    for key in key_dict.keys():
        try:
            t_0 = time.time()
            sol_dict, prob = initial_optimisation_cost_reduction(g[key], key_dict[key], cmin = cmin, time_limit=1e5, cost_node=cost_node, cost_connection=cost_connection,
                                                Lambda=Lambda)
            value_1 = prob.solution.get_objective_value()
            t_1 = time.time()
            sol_dict, prob = initial_optimisation_cost_reduction_rescaled(g[key], key_dict[key], cmin = cmin, time_limit=1e5, cost_node=cost_node, cost_connection=cost_connection,
                                                Lambda=Lambda)
            value_2 = prob.solution.get_objective_value()
            t_2 = time.time()
            # +ve means rescaled takes longer, -ve means normal takes longer
            time_taken.append((t_2 + t_0 -2 * t_1))
            values.append((value_1,value_2))
        except:
            continue
    print(np.mean(time_taken))
    print(time_taken)
    print(values)



def get_metric_for_heuristic(key_dict, g):
    nodes = []
    for key in key_dict:
        if key[0] not in nodes:
            nodes.append(key[0])
        if key[1] not in nodes:
            nodes.append(key[1])
    Ni = {}
    for node in nodes:
        Ni[node] = 0
    for key in key_dict:
        Ni[key[0]] += key_dict[key]
        Ni[key[1]] += key_dict[key]
    node_types = nx.get_node_attributes(g, "type")
    metrics = {}
    for node in g.nodes():
        if node_types[node] == "T":
            metrics[node] = 0.0
            for node_2 in g.nodes():
                if node_types[node_2] == "S":
                    capacity = g.edges[(node,node_2)]["capacity"]
                    if Ni[node_2] != 0:
                        metrics[node] += Ni[node_2]/(capacity + 0.0001)
    for key in metrics.keys():
        metrics[key] = str("{:.2e}".format(metrics[key]))
    return metrics




def plot_graph_solution(g, binary_dict, key_dict):
    """
    Plot the graph of the solution of flow for commodity_to_plot and all 'on' trusted nodes shaded in green.
    Parameters
    ----------
    binary_dict : dictionary of binary variables (delta_{i}) only
    flow_dict : dictionary of flow variables (x^{k}_{(i,j)}
    -------

    """
    pos = nx.spring_layout(g, k = 3)
    plt.figure()
    # nx.draw_networkx_edges(g, pos, width = 1, edge_color="k")
    options = {"node_size": 500, "alpha": 0.8}
    node_list = []
    trusted_node_list = []
    for n in g.nodes:
        if g.nodes[n]["type"] != "T":
            node_list.append(n)
        else:
            trusted_node_list.append(n)
    nx.draw_networkx_nodes(g, pos, nodelist = node_list, node_color="k")
    off_nodes = []
    on_nodes = []
    for key in binary_dict:
        current_node = int(key[5:])
        on_off = int(binary_dict[key])

        if on_off == 0:
            off_nodes.append(current_node)
        else:
            on_nodes.append(current_node)
    nx.draw_networkx_nodes(g, pos, nodelist=off_nodes, node_color="b")
    nx.draw_networkx_nodes(g, pos, nodelist=on_nodes, node_color="r")
    labels = nx.get_edge_attributes(g,'capacitysc')
    nx.draw_networkx_edges(g, pos, edge_color="g")
    # nx.draw_networkx_edge_labels(g, pos, edge_labels = labels, font_size = 7)
    metrics = get_metric_for_heuristic(key_dict, g)
    nx.set_node_attributes(g, metrics, name = "metric")
    # nx.set_node_attributes()
    nx.draw_networkx_labels(g, pos, labels = metrics)
    plt.axis("off")
    plt.show()

    # # get the keys for the largest metric where the nodes are on
    # my_keys = sorted(metrics, key=metrics.get, reverse=True)[0]
    # # Now need to figure out how to minimise sum_{N_i} from the set and then remove these connections from needed
    # # connection - then repeat the process until the connection is 0.0


def plot_position_graph(g, binary_dict):
    pos = {}
    for node in g.nodes:
        pos[node] = [g.nodes[node]["xcoord"], g.nodes[node]["ycoord"]]
    node_list = []
    detectors = []
    trusted_node_list = []
    for n in g.nodes:
        if g.nodes[n]["type"] == "S":
            node_list.append(n)
        elif g.nodes[n]["type"] == "T":
            trusted_node_list.append(n)
        else:
            detectors.append(n)
    nx.draw_networkx_nodes(g, pos, nodelist = node_list, node_color="k")
    nx.draw_networkx_nodes(g, pos, nodelist= detectors, node_color="g")
    off_nodes = []
    on_nodes = []
    for key in binary_dict:
        current_node = int(key[5:])
        on_off = int(binary_dict[key])

        if on_off == 0:
            off_nodes.append(current_node)
        else:
            on_nodes.append(current_node)
    nx.draw_networkx_nodes(g, pos, nodelist=off_nodes, node_color="b")
    nx.draw_networkx_nodes(g, pos, nodelist=on_nodes, node_color="r")
    nx.draw_networkx_edges(g, pos, edge_color="g")
    plt.axis("off")
    plt.show()


time_taken_vs_number_on_trusted_nodes(cap_needed = "15_nodes_cap_needed.csv", capacity_values = "15_nodes_edge_data.csv", node_types = "15_nodes_node_types.csv", position_nodes= "15_nodes_position_graph_nodes.csv", position_edges="15_nodes_position_graph_edges.csv", cmin = 10000, cost_connection=0.01 ,Lambda=10)

# time_taken_vs_number_on_trusted_nodes(cap_needed = "cap_needed_10_nodes.csv", capacity_values = "capacity_values_10_nodes.csv", node_types = "node_types_10_nodes.csv", cmin = 10000, cost_connection=0.01 ,Lambda=10)


# key_dict, g = trusted_nodes_utils.import_problem_from_files_flexibility_multiple_graphs("cap_needed_multiple_graphs.csv", "capacity_values_multiple_graphs.csv", "node_types_multiple_graphs.csv")
# # key_dict = {(12,14): 500}
# for key in key_dict.keys():
#     try:
#         sol_dict, prob = initial_optimisation_cost_reduction_multiple_solns(g[key], key_dict[key], cmin = 10)
#     except:
#         continue