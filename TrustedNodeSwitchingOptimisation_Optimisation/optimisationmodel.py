import cplex
import networkx as nx
import matplotlib.pyplot as plt
import brute_force
import trusted_nodes_utils
import numpy as np
import time

#############################Preprocessing##################################################

def get_on_solutions(soln_dict, key_dict):
    f"""
    Get a dictionary of the commodities used in each edge. If edge does not contain commodity then will not be added to
    dictionary
    Parameters
    ----------
    soln_dict: The dictionary of solutions for the primary commodity - (flow: value)
    key_dict: dictionary containing the commodities and required values - (commodity: value)

    Returns Dictionary of commodities using edge (edge: [commodities_using_edge])
    -------

    """

    ##### THINK ABOUT REVERSE COMMODITY TOO - NEED TO ADD THIS AS A USED PATH FOR REVERSE COMMODITY ######
    dict_flow_primary_commodity = {}
    for flow in soln_dict:
        # if the flow is 0 then no need to include in on solns
        if soln_dict[flow]> 0.001:
            # convert flow commodity to key (source_node, target_node) form
            k_pos = flow.find("k")
            substring_to_compare = flow[k_pos+1:]
            underscore_pos = substring_to_compare.find("_")
            commodity = (int(substring_to_compare[:underscore_pos]), int(substring_to_compare[underscore_pos+1:]))
            # find the position of the commodity in key_dict keys (python  3.6+)
            position_commodity = list(key_dict).index(commodity)
            reverse_commodity_position = list(key_dict).index((commodity[1], commodity[0]))
            # get the current edeg of flow in form (start, end)
            substring_for_edge = flow[1:k_pos-1]
            underscore_pos = substring_for_edge.find("_")
            flow_edge = (int(substring_for_edge[:underscore_pos]), int(substring_for_edge[underscore_pos+1:]))
            # store this in a dictionary with reverse commodity
            if not dict_flow_primary_commodity:
                dict_flow_primary_commodity = {flow_edge: [position_commodity, reverse_commodity_position]}
            elif flow_edge in dict_flow_primary_commodity.keys():
                # check the commodity is not in the dictionary before adding it into dictionary
                current_flow_elements = dict_flow_primary_commodity[flow_edge]
                add_commodity = True
                add_reverse_commodity = True
                for element in current_flow_elements:
                    if element == position_commodity:
                        add_commodity = False
                    if element == reverse_commodity_position:
                        add_reverse_commodity = False
                if add_commodity:
                    dict_flow_primary_commodity[flow_edge].append(position_commodity)
                if add_reverse_commodity:
                    dict_flow_primary_commodity[flow_edge].append(reverse_commodity_position)
            else:
                dict_flow_primary_commodity[flow_edge] = [position_commodity, reverse_commodity_position]
    return dict_flow_primary_commodity


def get_on_solution_all_edges(soln_dict, key_dict, g):
    """
    Get a dictionary of the commodities used in each edge. If edge does not contain commodity then will be added to
    dictionary
    Parameters
    ----------
    soln_dict: The dictionary of solutions for the primary commodity - (flow: value) (only flows no deltas)
    key_dict: dictionary containing the commodities and required values - (commodity: value)
    g: graph of the network

    Returns Dictionary of commodities using edge (edge: [commodities_using_edge])
    -------
    """
    # get a dictionary with all edges that have at least one commodity using the edge and the list of commodities using the edge
    dict_flow_primary_commodity = get_on_solutions(soln_dict, key_dict)
    # add the edges that have no commodity used
    for i,j in list(g.edges):
        if (i,j) not in dict_flow_primary_commodity:
            dict_flow_primary_commodity[(i,j)] = []
    return dict_flow_primary_commodity


def get_list_of_off_trusted_node(sol_dict):
    """
    Get list of trusted nodes that are still off from sol_dict of delta's binaries only
    Parameters
    ----------
    sol_dict: dictionary of solutions for binaries (deltas) n from {deltas: value}

    Returns List of the off trusted nodes
    -------

    """
    off_nodes = []
    for key in sol_dict:
        # get the number of the trusted node
        current_node = int(key[5:])
        # get the value of the solution (1 = on, 0 = off)
        on_off = int(sol_dict[key])
        if on_off == 0:
            off_nodes.append(current_node)
    return off_nodes

def split_sol_to_flow_delta(sol_dict):
    """
    Split the solution dictionary into 2 dictionaries containing the flow variables only and the binary variables only
    Parameters
    ----------
    sol_dict : The solution dictionary containing solutions to the primary flow problem

    Returns
    -------

    """
    flow_dict = {}
    binary_dict = {}
    for key in sol_dict:
        # get all keys that are flow and add to dictionary
        if key[0] == "x":
            flow_dict[key] = sol_dict[key]
        else:
            # get the keys that are binary 'on' 'off' and add to dictionary
            binary_dict[key] = sol_dict[key]
    return flow_dict, binary_dict

def split_sol_to_flow_delta_lambda(sol_dict):
    """
    Split the solution dictionary into 2 dictionaries containing the flow variables only and the binary variables only
    Parameters
    ----------
    sol_dict : The solution dictionary containing solutions to the primary flow problem

    Returns
    -------

    """
    flow_dict = {}
    binary_dict = {}
    lambda_dict = {}
    for key in sol_dict:
        # get all keys that are flow and add to dictionary
        if key[0] == "x":
            flow_dict[key] = sol_dict[key]
        elif key[0] == "d":
            # get the keys that are binary 'on' 'off' and add to dictionary
            binary_dict[key] = sol_dict[key]
        else:
            # get the keys that represent the lambda_{i,j} - representing number of detectors and add to dictionary
            lambda_dict[key] = sol_dict[key]
    return flow_dict, binary_dict, lambda_dict

def get_optimal_soln(binary_soln_pool):
    current_optimal_soln = 0
    current_optimal_value = np.infty
    for i in range(len(binary_soln_pool)):
        if binary_soln_pool[i]["objective"] < current_optimal_value:
            current_optimal_soln = i
            current_optimal_value = binary_soln_pool[i]["objective"]
    return current_optimal_soln, current_optimal_value


#############################CPLEX relevant functions #######################################



def add_capacity_constraint(problem, graph, key_dict, Lambda = 10):
    """
    Add the capacity constraints to the initial flow problem - ensures the capacity is at most lambda c_{i,j} delta_{i,j}
    where the delta ensures no capacity if the node is not on...
    :param problem: the cplex class with the minimisation problem
    :param graph: The graph in capacity space with the edges and vertices for the trusted nodes
    :param key_dict: The list of commodities needed in {(source, target): capacity_needed}
    :param Lambda: The max number of connections per edge
    """
    num_comm = len(key_dict)
    binary_trusted_variables = []
    for n in graph.nodes:
        if graph.nodes[n]["type"] == "T":
            binary_trusted_variables.append(f"delta{n}")
    problem.variables.add(names = binary_trusted_variables, types = [problem.variables.type.binary]*len(binary_trusted_variables))
    for i, nodes in enumerate(graph.edges):
        # get the indices of the flow variables on the selected edge
        ind_flow = [i + k * graph.number_of_edges() for k in range(num_comm)]
        # get the capacity of the edge
        capacity = int(graph.edges[nodes]["capacity"])
        if graph.nodes[nodes[1]]["type"] == "T":
            binary_value = [f'delta{nodes[1]}']
            # for capacity constraint with sum of all flow across channel cannot exceed capacity
            cap_const_1 = cplex.SparsePair(ind = ind_flow + binary_value, val = [1]*len(ind_flow) + [-capacity * Lambda])
            # for capacity constraint where we assume we have enough for each commodity.
            cap_const_2 = []
            # for j in range(len(ind_flow)):
            #     cap_const_2.append(cplex.SparsePair(ind = [ind_flow[j]] + binary_value, val = [1, -capacity]))
            problem.linear_constraints.add(lin_expr=[cap_const_1], senses='L', rhs=[0])
        else:
            # for capacity constraint with sum of all flow across channel cannot exceed capacity
            cap_const_1 = cplex.SparsePair(ind=ind_flow, val=[1] * len(ind_flow))
            # for capacity constraint where we assume we have enough for each commodity.
            # cap_const_2 = []
            # for j in range(len(ind_flow)):
            #     cap_const_2.append(cplex.SparsePair(ind=[ind_flow[j]], val=[1]))
            problem.linear_constraints.add(lin_expr = [cap_const_1], senses = 'L', rhs = [capacity* Lambda])


def add_capacity_constraint_rescaled(problem, graph, key_dict, cmin, Lambda = 10):
    """
    Add the capacity constraints to the initial flow problem - ensures the capacity is at most lambda c_{i,j} delta_{i,j}
    where the delta ensures no capacity if the node is not on...
    :param problem: the cplex class with the minimisation problem
    :param graph: The graph in capacity space with the edges and vertices for the trusted nodes
    :param key_dict: The list of commodities needed in {(source, target): capacity_needed}
    :param Lambda: The max number of connections per edge
    """
    num_comm = len(key_dict)
    binary_trusted_variables = []
    for n in graph.nodes:
        if graph.nodes[n]["type"] == "T":
            binary_trusted_variables.append(f"delta{n}")
    problem.variables.add(names = binary_trusted_variables, types = [problem.variables.type.binary]*len(binary_trusted_variables))
    for i, nodes in enumerate(graph.edges):
        # get the indices of the flow variables on the selected edge
        ind_flow = [i + k * graph.number_of_edges() for k in range(num_comm)]
        # get the capacity of the edge
        capacity = int(graph.edges[nodes]["capacity"])
        if graph.nodes[nodes[1]]["type"] == "T":
            binary_value = [f'delta{nodes[1]}']
            # for capacity constraint with sum of all flow across channel cannot exceed capacity
            cap_const_1 = cplex.SparsePair(ind = ind_flow + binary_value, val = [1]*len(ind_flow) + [-capacity * Lambda/(cmin)])
            # for capacity constraint where we assume we have enough for each commodity.
            cap_const_2 = []
            # for j in range(len(ind_flow)):
            #     cap_const_2.append(cplex.SparsePair(ind = [ind_flow[j]] + binary_value, val = [1, -capacity]))
            problem.linear_constraints.add(lin_expr=[cap_const_1], senses='L', rhs=[0])
        else:
            # for capacity constraint with sum of all flow across channel cannot exceed capacity
            cap_const_1 = cplex.SparsePair(ind=ind_flow, val=[1] * len(ind_flow))
            # for capacity constraint where we assume we have enough for each commodity.
            # cap_const_2 = []
            # for j in range(len(ind_flow)):
            #     cap_const_2.append(cplex.SparsePair(ind=[ind_flow[j]], val=[1]))
            problem.linear_constraints.add(lin_expr = [cap_const_1], senses = 'L', rhs = [capacity* Lambda/(cmin)])


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
            ind_flow = []
            for j, k in enumerate(key_dict):
                # find the keys with commodity that is not for current target node
                if k[1] != target_node:
                    ind_flow.append(i + j * g.number_of_edges())
            cap_const_2 = []
            # set constraints of these commodities to 0
            for j in range(len(ind_flow)):
                cap_const_2.append(cplex.SparsePair(ind=[ind_flow[j]], val=[1]))
            prob.linear_constraints.add(lin_expr=cap_const_2, senses='E' * len(cap_const_2),
                                           rhs=[0] * len(cap_const_2))


def add_capacity_constraint_for_lambda(prob, g, key_dict, Lambda):
    """
    Add the capacity constraint that constraints the number of detectors and sources on the nodes - defined by lambda_{i,j}
    The capacity across an edge cannot be more than the capacity of the connections times the number of devices in the
    connection

    """

    num_comm = len(key_dict)
    binary_trusted_variables = []
    # add the lambdas to the variables of problem
    for i,j in g.edges:
        binary_trusted_variables.append(f"lambda{i,j}")
    prob.variables.add(names=binary_trusted_variables,
                          types=[prob.variables.type.integer] * len(binary_trusted_variables), ub = [Lambda] * len(binary_trusted_variables))
    for i, nodes in enumerate(g.edges):
        # get the indices of the flow variables on the selected edge
        ind_flow = [i + k * g.number_of_edges() for k in range(num_comm)]
        lambda_ij = [f'lambda{nodes[0],nodes[1]}']
        # get the capacity of the edge
        capacity = int(g.edges[nodes]["capacity"])
        # add capacity constraint
        cap_const_1 = cplex.SparsePair(ind=ind_flow + lambda_ij, val=[1] * len(ind_flow) + [-capacity])
        prob.linear_constraints.add(lin_expr=[cap_const_1], senses='L', rhs=[0])


def lambda_constraint(prob, g, key_dict, Lambda):
    num_comm = len(key_dict)
    binary_trusted_variables = []
    # add the lambdas to the variables of problem
    for n in g.nodes:
        if g.nodes[n]["type"] == "T":
            binary_trusted_variables.append(f"delta{n}")
    prob.variables.add(names=binary_trusted_variables,
                          types=[prob.variables.type.binary] * len(binary_trusted_variables))
    for i, j in g.edges:
        if g.nodes[i]["type"] == "T":
            lambda_constraint_1 = cplex.SparsePair(ind = [f"lambda{i, j}"] + [f"delta{i}"], val = [1] + [-Lambda])
            prob.linear_constraints.add(lin_expr = [lambda_constraint_1], senses = "L", rhs = [0])
        elif g.nodes[j]["type"] == "T":
            lambda_constraint_1 = cplex.SparsePair(ind=[f"lambda{i, j}"] + [f"delta{j}"], val=[1] + [-Lambda])
            prob.linear_constraints.add(lin_expr=[lambda_constraint_1], senses="L", rhs=[0])


def add_capacity_constraint_for_lambda_rescaled(prob, g, key_dict, Lambda, cmin):
    """
    Add the capacity constraint that constraints the number of detectors and sources on the nodes - defined by lambda_{i,j}
    The capacity across an edge cannot be more than the capacity of the connections times the number of devices in the
    connection

    """

    num_comm = len(key_dict)
    binary_trusted_variables = []
    # add the lambdas to the variables of problem
    for i,j in g.edges:
        binary_trusted_variables.append(f"lambda{i,j}")
    prob.variables.add(names=binary_trusted_variables,
                          types=[prob.variables.type.integer] * len(binary_trusted_variables), ub = [Lambda] * len(binary_trusted_variables))
    for i, nodes in enumerate(g.edges):
        # get the indices of the flow variables on the selected edge
        ind_flow = [i + k * g.number_of_edges() for k in range(num_comm)]
        lambda_ij = [f'lambda{nodes[0],nodes[1]}']
        # get the capacity of the edge
        capacity = int(g.edges[nodes]["capacity"])
        # add capacity constraint
        cap_const_1 = cplex.SparsePair(ind=ind_flow + lambda_ij, val=[1] * len(ind_flow) + [-10 * capacity/cmin])
        prob.linear_constraints.add(lin_expr=[cap_const_1], senses='L', rhs=[0])


def add_flow_conservation_constraint(prob, g, key_dict):
    """Applies constraints to cplex problem for flow conservation"""
    num_comm=len(key_dict)# checks number of key exchange pairs i.e. number of commodities
    num_nodes=g.number_of_nodes()
    variable_names=[f'x{i}_{j}_k{k[0]}_{k[1]}' for k in key_dict for i,j in list(g.edges) ] # name convention x1_2k3_4 is flow in direction 1 to 2 for key 3 to 4
    flow=[[[int(i+k*num_nodes),int(j+k*num_nodes)], [1.0,-1.0]] for k in range(num_comm) for i,j in g.edges ] # tails are positive, heads negative, ensures source, sink balance

    sn=np.zeros(num_nodes*num_comm) # zeros ensure flow conservation
    count=0
    for pair,num_keys in key_dict.items():
        active_commodity=count*num_nodes
        sn[int(pair[0])+active_commodity]=int(num_keys) # sets source
        sn[int(pair[1])+active_commodity]=-int(num_keys) # sets sink
        count+=1 # counter moves forward for next commodity
    my_senses='E'*num_nodes*num_comm
    prob.linear_constraints.add(senses=my_senses,rhs=sn.tolist())
    prob.variables.add(names=variable_names, columns=flow, types=[prob.variables.type.continuous]*len(variable_names)) # add variables and flow conservation

def add_bidirectional_flow_conservation_constraint(prob,g,key_dict):
    """
    adds the bidirectional constraint to cplex problem for flow conservation without considering direction

    """
    num_nodes=g.number_of_nodes()
    key_dict=trusted_nodes_utils.make_key_dict_bidirectional(key_dict)
    add_flow_conservation_constraint(prob, g, key_dict)#add flow conservation everywhere (incl. souces/sinks)
    rows_to_delete=[key[i]+count*num_nodes for count, key in enumerate(key_dict) for i in range(2)] #list of row indices for constrain matrix which are sinks/sources
    prob.linear_constraints.delete(rows_to_delete) # remove flow conservation at sinks/sources due to unknown source,sink values
    for k in key_dict:
        total_keys=float(key_dict[k]+key_dict[k[::-1]]) # total number of keys needed to exchange for pair k
        forward_flow_variables=[f'x{k[0]}_{j}_k{k[0]}_{k[1]}' for j in g.adj[k[0]]] # flow out of source for forward commodity
        backwards_flow_variables=[f'x{j}_{k[0]}_k{k[1]}_{k[0]}' for j in g.adj[k[0]]] # flow into source for backwards commodity
        flow_into_source_variables=[f'x{j}_{k[0]}_k{k[0]}_{k[1]}' for j in g.adj[k[0]]] # flow into source from forwards commodity (must be zero to avoid loops)
        flow_out_sink_variables=[f'x{k[1]}_{j}_k{k[0]}_{k[1]}' for j in g.adj[k[1]]] # flow out of sink from forwards commodity (must be zero to avoid loops)
        names=forward_flow_variables+backwards_flow_variables
        lin_expressions=[cplex.SparsePair(ind=names, val=[1.]*len(names)),
                         cplex.SparsePair(ind=flow_into_source_variables, val=[1.]*len(flow_into_source_variables)),
                         cplex.SparsePair(ind=flow_out_sink_variables, val=[1.]*len(flow_out_sink_variables))]
        prob.linear_constraints.add(lin_expr=lin_expressions, senses=['EEE'], rhs=[total_keys, 0., 0.])


def add_minimise_trusted_nodes_objective(prob, g):
    """
    Add the objective to minimise the number of trusted nodes that are "on" (sum_{i in T} delta_i
    """
    obj_vals = []
    for n in g.nodes:
        if g.nodes[n]["type"] == "T":
            obj_vals.append((f"delta{n}", 1.))
    prob.objective.set_linear(obj_vals)
    prob.objective.set_sense(prob.objective.sense.minimize)

def add_minimise_overall_cost_objective(prob, g, cost_node, cost_connection):
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
    for i,j in g.edges:
        obj_vals.append((f"lambda{i, j}", cost_connection))
    prob.objective.set_linear(obj_vals)
    prob.objective.set_sense(prob.objective.sense.minimize)


def add_capacity_constraints_rigidity_test(problem, graph, key_dict, flow_sol, off_trusted_nodes):
    """
    add capacity constraints for the rigidity test - (add the constraint that no flow on the edges that already have
    flow for the commodity)

    :param problem: the cplex class with the minimisation problem
    :param graph: The graph in capacity space with the edges and vertices for the trusted nodes
    :param key_dict: The list of commodities needed in {(source, target): capacity_needed}
    :param flow_sol: The flow solution: for commodity k a dictionary with {(i,j): [k]} which contains all commodities
    that use path (i,j)
    :param off_trusted_nodes: A list of the trusted nodes that are still off
    """
    num_comm = len(key_dict)
    binary_trusted_variables = [f"delta{n}" for n in off_trusted_nodes]
    problem.variables.add(names=binary_trusted_variables,
                          types=[problem.variables.type.binary] * len(off_trusted_nodes))
    for i, nodes in enumerate(graph.edges):
        # get the indices of the flow variables on the selected edge
        ind_flow = [i + k * graph.number_of_edges() for k in range(num_comm)]
        # get the capacity of the edge  *** NOTE- MIGHT NEED TO ADD A CAPACITY DRAIN TO LOST CAPACITY FROM FIRST SOLN
        capacity = graph.edges[nodes]["capacity"]
        in_off_trusted_nodes = False
        for j in off_trusted_nodes:
            if nodes[1] == j:
                in_off_trusted_nodes = True
        if in_off_trusted_nodes:
            binary_value = [f'delta{nodes[1]}']
        commodities_to_turn_off = flow_sol[(nodes[0], nodes[1])]
        # for capacity constraint where we assume we have enough for each commodity.
        cap_const_2 = []
        rhs = []
        for j in range(len(ind_flow)):
            turn_off = False
            for l in commodities_to_turn_off:
                if i + l * graph.number_of_edges() == ind_flow[j]:
                    turn_off = True
            if turn_off:
                if in_off_trusted_nodes:
                    rhs.append([0])
                    cap_const_2.append(cplex.SparsePair(ind=[ind_flow[j]] + binary_value, val=[1, 0]))
                else:
                    rhs.append([0])
                    cap_const_2.append(cplex.SparsePair(ind = [ind_flow[j]], val = [1]))
            else:
                if in_off_trusted_nodes:
                    rhs.append([0])
                    cap_const_2.append(cplex.SparsePair(ind_flow[j] + binary_value, val = [1, -capacity]))
                else:
                    ## need to add the capacity to the right hand side of the equation in this case
                    rhs.append([capacity])
                    cap_const_2.append(cplex.SparsePair(ind_flow[j], val=[1]))
        problem.linear_constraints.add(lin_expr=cap_const_2, senses='L' * len(cap_const_2), rhs=rhs)

def create_sol_dict(prob):
    """
    Create a dictionary with the solution of the parameters
    """
    names=prob.variables.get_names()
    values=prob.solution.get_values()
    sol_dict={names[idx]: (values[idx]) for idx in range(prob.variables.get_num())}
    return sol_dict


############################## Post-Processing ##########################################

def plot_graph_solution(g, binary_dict, flow_dict, commodity_to_plot):
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
    edge_labels = {}
    edge_list = []
    for flow in flow_dict:
        # if the flow is 0 then no need to include in on solns
        if flow_dict[flow]> 0.001:
            # convert flow commodity to key (source_node, target_node) form
            k_pos = flow.find("k")
            substring_to_compare = flow[k_pos + 1:]
            underscore_pos = substring_to_compare.find("_")
            commodity = (int(substring_to_compare[:underscore_pos]), int(substring_to_compare[underscore_pos + 1:]))
            if (commodity[0] == commodity_to_plot[0] and commodity[1] == commodity_to_plot[1]) :
                # or (commodity[0] == commodity_to_plot[1] and commodity[1] == commodity_to_plot[0])
                # get the current edeg of flow in form (start, end)
                substring_for_edge = flow[1:k_pos-1]
                underscore_pos = substring_for_edge.find("_")
                flow_edge = (int(substring_for_edge[:underscore_pos]), int(substring_for_edge[underscore_pos+1:]))
                edge_list.append(flow_edge)
                if not edge_labels:
                    edge_labels[flow_edge] = str(int(flow_dict[flow]))
                elif flow_edge in edge_labels.keys():
                    edge_labels[flow_edge] =  str(int(flow_dict[flow]))
                else:
                    edge_labels[flow_edge] =  str(int(flow_dict[flow]))
    nx.draw_networkx_edges(g, pos, edgelist = edge_list)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size = 8, label_pos = 0.4)
    plt.axis("off")
    plt.show()


############################## Optimisation Setup and Run ################################

def check_infeasible(prob):
    """Checks solved cplex problem for infeasibility. Returns True when infeasible, otherwise false."""
    status = prob.solution.get_status()
    if status == 3:
        return True
    else:
        return False


def check_timeout(prob):
    """Checks solved cplex problem for timeout. Returns True when timed out, otherwise false."""
    status = prob.solution.get_status()
    if status == 11:
        return True
    else:
        return False


def check_solvable(prob):
    """Checks solved cplex problem for timeout, infeasibility or optimal solution. Returns True when feasible solution obtained."""
    status = prob.solution.get_status()
    if status == 101 or status == 102:
        return True
    elif status == 103:  # proven infeasible or Timeout
        return False
    elif status == 107:  # timed out
        print("Optimiser Timed out - assuming infeasible")
        return True
    else:
        print(f"Unknown Solution Status: {status} - assuming infeasible")
        return False






def initial_optimisation(g, key_dict, time_limit = 300):
    """
    set up and solve the problem for mimising the number of trusted nodes

    """
    t_0 = time.time()
    print("Start Optimisation")
    prob = cplex.Cplex()
    # add constraints to the problem
    add_bidirectional_flow_conservation_constraint(prob, g, key_dict)
    add_capacity_constraint(prob, g, key_dict)
    # add objective to minimise trusted nodes
    add_minimise_trusted_nodes_objective(prob, g)
    add_constraint_source_nodes(prob, g, key_dict)
    # prob.write("test.lp")
    # set the parameters of the solver to improve speed of solution
    prob.parameters.lpmethod.set(prob.parameters.lpmethod.values.network)
    prob.parameters.mip.tolerances.integrality(1e-08)
    prob.parameters.mip.limits.cutpasses.set(1)
    prob.parameters.mip.strategy.probe.set(-1)
    prob.parameters.mip.strategy.variableselect.set(4)
    print(prob.parameters.get_changed())
    prob.parameters.timelimit.set(time_limit)
    t_1 = time.time()
    print("Time to set up problem: " + str(t_1-t_0))
    prob.solve()
    t_2 = time.time()
    print("Time to solve problem: " + str(t_2 - t_1))
    print(f"The Minimum Number of Trusted Nodes: {prob.solution.get_objective_value()}")

    print(f"Number of Variables = {prob.variables.get_num()}")
    print(f"Number of Conditions = {prob.linear_constraints.get_num()}")
    sol_dict = create_sol_dict(prob)

    return sol_dict, prob


def initial_optimisation_cost_reduction(g, key_dict, time_limit = 300, cost_node = 1, cost_connection= 0.1, Lambda = 5):
    """
    set up and solve the problem for minimising the overall cost of the network
    """
    t_0 = time.time()
    print("Start Optimisation")
    prob = cplex.Cplex()
    # add constraints to the problem
    add_bidirectional_flow_conservation_constraint(prob, g, key_dict)
    add_capacity_constraint(prob, g, key_dict, Lambda=Lambda)
    add_capacity_constraint_for_lambda(prob, g, key_dict, Lambda=Lambda)
    add_constraint_source_nodes(prob, g, key_dict)
    # add_minimise_trusted_nodes_objective(prob, g)
    # add objective to the problem
    add_minimise_overall_cost_objective(prob, g, cost_node, cost_connection)
    prob.write("test.lp")
    # set the parameters of the solver to improve speed of solution
    prob.parameters.lpmethod.set(prob.parameters.lpmethod.values.network)
    prob.parameters.mip.tolerances.integrality= 1e-08
    prob.parameters.mip.limits.cutpasses.set(1)
    prob.parameters.mip.strategy.probe.set(-1)
    prob.parameters.mip.strategy.variableselect.set(0)
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





############## Set of Functions for Investigations ##################################


def get_number_of_nodes_in_graph(graph):
    return len(graph.nodes)

def plot_cost_of_network_against_number_of_nodes_both_methods(cap_needed, capacity_values, node_types):
    key_dicts, g = trusted_nodes_utils.import_problem_from_files_multiple_graphs(cap_needed_file=cap_needed,
                                                        capacity_values_file=capacity_values, node_type_file=node_types)
    network_costs_difference = {}
    for key in key_dicts.keys():
        try:
            sol_dict, prob = initial_optimisation_cost_reduction(g[key], key_dicts[key], cost_connection=0)
            if check_solvable(prob):
                number_nodes = get_number_of_nodes_in_graph(g[key])
                number_trusted_nodes, time = brute_force.generate_permutations(g[key], key_dicts[key])
                if number_nodes in network_costs_difference.keys():
                    network_costs_difference[number_nodes].append(prob.solution.get_objective_value() - number_trusted_nodes)
                else:
                    network_costs_difference[number_nodes] = [prob.solution.get_objective_value()-number_trusted_nodes]
        except:
            continue
    network_costs_difference_mean_std = {}
    x = []
    y = []
    yerr = []
    for key in network_costs_difference:
        network_costs_difference_mean_std[key] = [np.mean(network_costs_difference[key]), np.std(network_costs_difference[key])]
        x.append(key)
        y.append(network_costs_difference_mean_std[key][0])
        yerr.append(network_costs_difference_mean_std[key][1])
    plt.errorbar(x, y, yerr=yerr)
    plt.xlabel("No. Nodes in Graph", fontsize=10)
    plt.ylabel("Cost Linear Program - Cost Brute Force", fontsize=10)
    plt.savefig("difference_brute_force_normal_network.png")
    plt.show()

def plot_cost_network_with_increasing_number_nodes(cap_needed, capacity_values, node_types):
    key_dicts, g = trusted_nodes_utils.import_problem_from_files_multiple_graphs(cap_needed_file=cap_needed,
                                                                                 capacity_values_file=capacity_values,
                                                                                 node_type_file=node_types)
    network_costs = {}
    for key in key_dicts.keys():
        try:
            sol_dict, prob = initial_optimisation_cost_reduction(g[key], key_dicts[key])
            if check_solvable(prob):
                number_nodes = get_number_of_nodes_in_graph(g[key])
                if number_nodes in network_costs.keys():
                    network_costs[number_nodes].append(prob.solution.get_objective_value())
                else:
                    network_costs[number_nodes] = [prob.solution.get_objective_value()]
        except:
            continue
    network_costs_mean_std = {}
    x = []
    y = []
    yerr = []
    for key in network_costs:
        network_costs_mean_std[key] = [np.mean(network_costs[key]), np.std(network_costs[key])]
        x.append(key)
        y.append(network_costs_mean_std[key][0])
        yerr.append(network_costs_mean_std[key][1])
    plt.errorbar(x, y, yerr=yerr)
    plt.xlabel("No. Nodes in Graph", fontsize=10)
    plt.ylabel("Cost of Network", fontsize=10)
    plt.savefig("cost_of_network_with_increasing_no_nodes.png")
    plt.show()


def plot_time_with_increasing_nodes(cap_needed, capacity_values, node_types):
    ## also needs method in other file to write the distance into the data
    key_dicts, g = trusted_nodes_utils.import_problem_from_files_multiple_graphs(cap_needed_file=cap_needed,
                                                                                 capacity_values_file=capacity_values,
                                                                                 node_type_file=node_types)
    time_costs = {}
    time_costs_brute_force = {}
    for key in key_dicts.keys():
        try:
            t_0 = time.time()
            sol_dict, prob = initial_optimisation_cost_reduction(g[key], key_dicts[key], cost_connection=0)
            if check_solvable(prob):
                t_1 = time.time()
                number_nodes = get_number_of_nodes_in_graph(g[key])
                number_trusted_nodes, time = brute_force.generate_permutations(g[key], key_dicts[key])
                if number_nodes in time_costs.keys():
                    time_costs[number_nodes].append(t_1-t_0)
                    time_costs_brute_force[number_nodes].append(time)
                else:
                    time_costs[number_nodes] = [t_1-t_0]
                    time_costs_brute_force[number_nodes].append(time)
        except:
            continue
    time_costs_mean_std = {}
    time_costs_brute_force_mean_std = {}
    x = []
    y = []
    yerr = []
    y_bf = []
    yerr_bf = []
    for key in time_costs:
        time_costs_mean_std[key] = [np.mean(time_costs[key]), np.std(time_costs[key])]
        time_costs_brute_force_mean_std[key] = [np.mean(time_costs_brute_force[key]),  np.std(time_costs_brute_force[key])]
        x.append(key)
        y.append(time_costs_mean_std[key][0])
        yerr.append(time_costs_mean_std[key][1])
        y_bf.append(time_costs_brute_force_mean_std[key][0])
        yerr_bf.append(time_costs_brute_force_mean_std[key][1])
        # x                     y               yerr
    plt.errorbar(x, y, yerr=yerr, color = "r", label = 'Linear Program')
    plt.errorbar(x, y_bf, yerr = yerr_bf, color = "b", label = 'Brute Force')
    plt.xlabel("No. Nodes in Graph", fontsize=10)
    plt.ylabel("Time/s", fontsize=10)
    plt.legend(loc = 'upper right', fontsize = 'medium')
    plt.savefig("time_investigation.png")
    plt.show()



def plot_cost_network_with_increasing_distance(cap_needed, capacity_values, node_types):
    ## also needs method in other file to write the distance into the data
    key_dicts, g, distances = trusted_nodes_utils.import_problem_from_files_multiple_graphs_distances(cap_needed_file=cap_needed,
                                                                                 capacity_values_file=capacity_values,
                                                                                 node_type_file=node_types)
    network_costs = {}
    i = 0
    for key in key_dicts.keys():
        try:
            print("Iteration: " + str(i))
            i += 1
            sol_dict, prob = initial_optimisation_cost_reduction(g[key], key_dicts[key])

            flow_dict, binary_dict, lambda_dict = split_sol_to_flow_delta_lambda(sol_dict)
            if check_solvable(prob):
                distance = distances[key]
                if distance in network_costs.keys():
                    network_costs[distance].append(prob.solution.get_objective_value())
                else:
                    network_costs[distance] = [prob.solution.get_objective_value()]
        except:
            continue
    network_costs_mean_std = {}
    x = []
    y = []
    yerr = []
    for key in network_costs:
        network_costs_mean_std[key] = [np.mean(network_costs[key]), np.std(network_costs[key])]
        x.append(key)
        y.append(network_costs_mean_std[key][0])
        yerr.append(network_costs_mean_std[key][1])
    plt.errorbar(x, y, yerr = yerr)
    plt.xlabel("Distance of Network/km", fontsize=10)
    plt.ylabel("Cost of Network", fontsize=10)
    plt.savefig("cost_of_network_with_increasing_distance.png")
    plt.show()


# plot_cost_network_with_increasing_distance(cap_needed=  "cap_needed_different_distances_7_nodes.csv", capacity_values = "capacity_values_different_distances_7_nodes.csv", node_types= "node_types_different_distances_7_nodes.csv")
# key_dict, graphs, distances = trusted_nodes_utils.import_problem_from_files_multiple_graphs_distances(cap_needed_file=  "cap_needed_different_distances.csv", capacity_values_file = "capacity_values_different_distances.csv", node_type_file = "node_types_different_distances.csv")





# key_dict, g = trusted_nodes_utils.import_problem_from_files("cap_needed_test.csv", "capacity_values_test.csv", "node_types_test.csv")
#
# # key_dict = {(12,14): 500}
# sol_dict, prob = initial_optimisation_cost_reduction(g, key_dict)
# flow_dict, binary_dict, lambda_dict = split_sol_to_flow_delta_lambda(sol_dict)
# plot_graph_solution(g, flow_dict=flow_dict, binary_dict = binary_dict, commodity_to_plot=next(iter(key_dict)))
# number_trusted_nodes, time = brute_force.generate_permutations(g, key_dict)
#
# dict_flow_primary_commodity =get_on_solution_all_edges(flow_dict, key_dict, g)
# off_nodes = get_list_of_off_trusted_node(binary_dict)