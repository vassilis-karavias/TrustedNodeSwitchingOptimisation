import cplex
import networkx as nx
import matplotlib.pyplot as plt
import scipy.optimize as opt
import optimisationmodel
import trusted_nodes_utils
import numpy as np
import time

def split_soln_dict(sol_dict):
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
    detector_dict = {}
    source_dict = {}
    for key in sol_dict:
        # get all keys that are flow and add to dictionary
        if key[0] == "x":
            flow_dict[key] = sol_dict[key]
        elif key[0] == "d" and key[1] == "e":
            # get the keys that are binary 'on' 'off' and add to dictionary
            binary_dict[key] = sol_dict[key]
        elif key[0] == "d" and key[1] == "_":
            detector_dict[key] = sol_dict[key]
        elif key[0] == "s":
            source_dict[key] = sol_dict[key]
    return flow_dict, binary_dict, detector_dict, source_dict


def ensure_capacity_does_not_exceed_source_capacity(prob, g, key_dict, Lambda, cmin):

    variable_names = [f'x{i}_{j}_k{k[0]}_{k[1]}' for k in key_dict for i, j in list(g.edges)]
    prob.variables.add(names=variable_names, types=[prob.variables.type.binary] * len(variable_names))
    sigma_terms = []
    for n in g.nodes:
        if g.nodes[n]["type"] == "T":
            sigma_terms.append(f"sigma_{n}")
    prob.variables.add(names=sigma_terms, types=[prob.variables.type.binary] * len(sigma_terms))
    for i in g.nodes:
        # if the node is a source node there is always a source term on it so the limit is just the number of sources
        # limit here:
        if g.nodes[i]["type"] == "T":
            variables = []
            vals = []
            for j in g.adj[i]:
                for k in key_dict:
                    variables.append(f"x{i}_{j}_k{k[0]}_{k[1]}")
                    vals.append(cmin/int(g.edges[[i,j]]["capacity"] + 1))
            variables.append(f"sigma_{i}")
            vals.append(-Lambda)
            lin_expressions = [cplex.SparsePair(ind=variables, val=vals)]
            prob.linear_constraints.add(lin_expr=lin_expressions, senses=['L'], rhs=[0])
        else:
            variables = []
            vals = []
            for j in g.adj[i]:
                for k in key_dict:
                    variables.append(f"x{i}_{j}_k{k[0]}_{k[1]}")
                    vals.append(cmin / int(g.edges[[i, j]]["capacity"] + 1))
            lin_expressions = [cplex.SparsePair(ind=variables, val=vals)]
            prob.linear_constraints.add(lin_expr=lin_expressions, senses=['L'], rhs=[Lambda])

def ensure_capacity_does_not_exceed_detector_capacity(prob, g, key_dict, Lambda, cmin):
    d_terms = []
    for n in g.nodes:
        if g.nodes[n]["type"] == "T":
            d_terms.append(f"d_{n}")
    prob.variables.add(names=d_terms, types=[prob.variables.type.binary] * len(d_terms))
    for i in g.nodes:
        # if the node is a source node we consider the approximation that there is no detectors on these nodes for the
        # network.
        if g.nodes[i]["type"] == "T":
            variables = []
            vals = []
            for j in g.adj[i]:
                for k in key_dict:
                    variables.append(f"x{j}_{i}_k{k[0]}_{k[1]}")
                    vals.append(cmin/int(g.edges[[i,j]]["capacity"] + 1))
            variables.append(f"d_{i}")
            vals.append(-Lambda)
            lin_expressions = [cplex.SparsePair(ind=variables, val=vals)]
            prob.linear_constraints.add(lin_expr=lin_expressions, senses=['L'], rhs=[0])
        else:
            variables = []
            vals = []
            for j in g.adj[i]:
                for k in key_dict:
                    variables.append(f"x{j}_{i}_k{k[0]}_{k[1]}")
                    vals.append(cmin / int(g.edges[[i, j]]["capacity"] + 1))
            lin_expressions = [cplex.SparsePair(ind=variables, val=vals)]
            prob.linear_constraints.add(lin_expr=lin_expressions, senses=['L'], rhs=[0])

def conservation_of_flow(prob, g, key_dict):
    for k in key_dict:
        for n in g.nodes:
            if k[0] < k[1] and n != k[0] and n != k[1]:
                variables = []
                vals = []
                for m in g.adj[n]:
                    variables.extend([f"x{n}_{m}_k{k[0]}_{k[1]}", f"x{m}_{n}_k{k[1]}_{k[0]}", f"x{m}_{n}_k{k[0]}_{k[1]}",
                                      f"x{n}_{m}_k{k[1]}_{k[0]}"])
                    vals.extend([1,1,-1,-1])
                lin_expressions = [cplex.SparsePair(ind=variables, val=vals)]
                prob.linear_constraints.add(lin_expr=lin_expressions, senses=['E'], rhs=[0])

def flow_requirement_constraint(prob, g, key_dict, N):
    for k in key_dict:
        if k[0] < k[1]:
            variables = []
            vals = []
            for j in g.adj[k[0]]:
                variables.extend([f"x{k[0]}_{j}_k{k[0]}_{k[1]}", f"x{j}_{k[0]}_k{k[1]}_{k[0]}"])
                vals.extend([1,1])
            lin_expressions = [cplex.SparsePair(ind=variables, val=vals)]
            prob.linear_constraints.add(lin_expr=lin_expressions, senses=['G'], rhs=[N])


def add_flow_into_source_out_sink(prob, g, key_dict):
    for k in key_dict:
        if k[0] < k[1]:
            for n in g.adj[k[0]]:
                variables = [f"x{n}_{k[0]}_k{k[0]}_{k[1]}", f"x{k[0]}_{n}_k{k[1]}_{k[0]}"]
                coeffs = [1, 1]
                lin_expressions = [cplex.SparsePair(ind=variables, val=coeffs)]
                prob.linear_constraints.add(lin_expr=lin_expressions, senses=['E'], rhs=[0])
            for n in g.adj[k[1]]:
                variables = [f"x{k[1]}_{n}_k{k[0]}_{k[1]}", f"x{n}_{k[1]}_k{k[1]}_{k[0]}"]
                coeffs = [1,1]
                lin_expressions = [cplex.SparsePair(ind=variables, val=coeffs)]
                prob.linear_constraints.add(lin_expr=lin_expressions, senses=['E'], rhs=[0])

def add_no_flow_into_untrusted_node(prob, g, key_dict):
    for k in key_dict:
        for j in g.nodes:
            if g.nodes[j]["type"] == "S" and k[1] != j and k[0] < k[1]:
                for i in g.adj[j]:
                    variables = [f"x{i}_{j}_k{k[0]}_{k[1]}", f"x{j}_{i}_k{k[1]}_{k[0]}"]
                    coeffs = [1, 1]
                    lin_expressions = [cplex.SparsePair(ind=variables, val=coeffs)]
                    prob.linear_constraints.add(lin_expr=lin_expressions, senses=['E'], rhs=[0])

def add_limited_flow_through_connection_old(prob, g, key_dict):
    """
    Adds the constraint that ensures the flow along an edge is limited by cmin ensuring at least N_k paths are used

    x_{i,j}^{k} + x_{i,j}^{k_R} + x_{j,i}^{k} + x_{j,i}^{k_R} \leq 1
    """
    for i,j in g.edges:
        for k in key_dict:
            if k[0] < k[1] and i < j:
                ind_flow = [f"x{i}_{j}_k{k[0]}_{k[1]}", f"x{i}_{j}_k{k[1]}_{k[0]}", f"x{j}_{i}_k{k[0]}_{k[1]}", f"x{j}_{i}_k{k[1]}_{k[0]}"]
                val = [1,1,1,1]
                lin_expr = [cplex.SparsePair(ind=ind_flow, val=val)]
                prob.linear_constraints.add(lin_expr=lin_expr, senses='L' * len(lin_expr),
                                            rhs=[1] * len(lin_expr))



def add_limited_flow_through_connection(prob, g, key_dict):
    """
    Adds the constraint that ensures the flow along an edge is limited by cmin ensuring at least N_k paths are used

    \sum_{j \in \mathcal{N}(i)} x_{i,j}^{k} + x_{j,i}^{k_R} \leq c_{min}
    """
    for i in g.nodes:
        for k in key_dict:
            if k[0] != i:
                ind_flow = []
                val = []
                for j in g.adj[i]:
                    if k[0] < k[1]:
                        ind_flow.extend([f"x{i}_{j}_k{k[0]}_{k[1]}", f"x{j}_{i}_k{k[1]}_{k[0]}"])
                        val.extend([1,1])
                lin_expr = [cplex.SparsePair(ind=ind_flow, val=val)]
                prob.linear_constraints.add(lin_expr=lin_expr, senses='L' * len(lin_expr),
                                             rhs=[1] * len(lin_expr))


def constaint_on_node_on(prob, g):
    delta_vals = []
    for i in g.nodes:
        if g.nodes[i]["type"] == "T":
            delta_vals.append(f"delta_{i}")
    prob.variables.add(names=delta_vals, types=[prob.variables.type.binary] * len(delta_vals))
    for i in g.nodes:
        if g.nodes[i]["type"] == "T":
            ind = [f"sigma_{i}", f"delta_{i}"]
            val = [1, -1]
            lin_expr = [cplex.SparsePair(ind=ind, val=val)]
            prob.linear_constraints.add(lin_expr=lin_expr, senses='L' * len(lin_expr),
                                rhs=[0] * len(lin_expr))
            ind = [f"d_{i}", f"delta_{i}"]
            val = [1, -1]
            lin_expr = [cplex.SparsePair(ind=ind, val=val)]
            prob.linear_constraints.add(lin_expr=lin_expr, senses='L' * len(lin_expr),
                                        rhs=[0] * len(lin_expr))



def add_objective_constraint(prob, g, key_dict, C_on, C_source, C_det, cmin):
    obj_vals = []
    for i in g.nodes:
        if g.nodes[i]["type"] == "T":
            obj_vals.append((f"delta_{i}", C_on))
    for i, j in g.edges:
        for k in key_dict:
            # add 1 to the capacity to ensure divide by 0 errors are not an issue - this will cause the cost of such an edge
            # to be massive compared to other edges
            obj_vals.append(
                (f"x{i}_{j}_k{k[0]}_{k[1]}", C_source * cmin / int(g.edges[[i, j]]["capacity"] + 1) + C_det * cmin / int(g.edges[[i, j]]["capacity"] + 1)))
    prob.objective.set_linear(obj_vals)
    prob.objective.set_sense(prob.objective.sense.minimize)



def initial_optimisation_cost_reduction(g, key_dict, cmin, time_limit = 1e5,  Lambda = 100, N = 2, C_on = 10, C_source = 0.1, C_det = 1.3):
    """
    set up and solve the problem for minimising the overall cost of the network
    """
    t_0 = time.time()
    print("Start Optimisation")
    prob = cplex.Cplex()
    key_dict = trusted_nodes_utils.make_key_dict_bidirectional(key_dict)
    ensure_capacity_does_not_exceed_source_capacity(prob, g, key_dict, Lambda, cmin)
    ensure_capacity_does_not_exceed_detector_capacity(prob, g, key_dict, Lambda, cmin)
    # add_capacity_constraint(prob, g, key_dict, Lambda=Lambda)
    conservation_of_flow(prob, g, key_dict)
    flow_requirement_constraint(prob, g, key_dict, N)
    # add_untransformed_transformed_relationship(prob, g, key_dict, NT)
    add_flow_into_source_out_sink(prob, g, key_dict)
    add_no_flow_into_untrusted_node(prob, g, key_dict)
    add_limited_flow_through_connection(prob, g, key_dict)
    constaint_on_node_on(prob, g)
    add_objective_constraint(prob, g, key_dict, C_on, C_source, C_det, cmin)

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
    flow_dict, binary_dict, detector_dict, source_dict = split_soln_dict(sol_dict)
    # log_optimal_solution_to_problem_column_format(prob, save_file="solution_dictionary_column_format", graph_id=0)
    trusted_nodes = 0
    for key in binary_dict:
        trusted_nodes += binary_dict[key]
    print(f"Number of Trusted Nodes = {trusted_nodes}")
    return sol_dict, prob


if __name__ == "__main__":
    key_dict, g, position_graphs =  trusted_nodes_utils.import_problem_from_files_flexibility_multiple_graphs_position_graph("15_nodes_cap_needed.csv", "15_nodes_edge_data.csv", "15_nodes_node_types.csv", position_node_file = "15_nodes_position_graph_nodes.csv", position_edge_file="15_nodes_position_graph_edges.csv")
    # key_dict, g, position_graphs = trusted_nodes_utils.import_problem_from_files_flexibility_multiple_graphs_position_graph("small_graphs_for_test_cap_needed.csv", "small_graphs_for_test_edge_data.csv", "small_graphs_for_test_node_types.csv", position_node_file = "small_graphs_for_test_position_graph_nodes.csv", position_edge_file="small_graphs_for_test_position_graph_edges.csv")
    for key in g.keys():
        try:
            sol_dict, prob = initial_optimisation_cost_reduction(g[key], key_dict[key], cmin = 3000, Lambda =6)
            flow_dict, binary_dict, detector_dict, source_dict = split_soln_dict(sol_dict)
            # print(sol_dict)
        except:
            continue