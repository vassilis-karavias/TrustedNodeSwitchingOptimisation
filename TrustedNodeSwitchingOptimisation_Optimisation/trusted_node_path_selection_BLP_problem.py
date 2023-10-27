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
    for key in sol_dict:
        # get all keys that are flow and add to dictionary
        if key[0] == "x":
            flow_dict[key] = sol_dict[key]
        elif key[0] == "d":
            # get the keys that are binary 'on' 'off' and add to dictionary
            binary_dict[key] = sol_dict[key]
    return flow_dict, binary_dict




def add_capacity_constraint(prob, g, key_dict, Lambda, c_min):
    variable_names = [f'x{i}_{j}_k{k[0]}_{k[1]}' for k in key_dict for i, j in list(g.edges)]
    prob.variables.add(names=variable_names, types=[prob.variables.type.binary] * len(variable_names))
    delta_terms = []
    for n in g.nodes:
        if g.nodes[n]["type"] == "T":
            delta_terms.append(f"delta_{n}")
    prob.variables.add(names=delta_terms, types=[prob.variables.type.binary] * len(delta_terms))
    for i,j in g.edges:
        variables = []
        coeffs = []
        for k in key_dict:
            variables.append(f"x{i}_{j}_k{k[0]}_{k[1]}")
            coeffs.append(1)
        # need to consider the fact that if j is not a trusted node then it does not have a delta term
        # However the flow into the node from the connection is still upper bounded by the number of devices on the
        # edge => LHS becomes Lambda c_{i,j}/c_{min}
        if g.nodes[j]["type"] == "T":
            variables.append(f"delta_{j}")
            coeffs.append(-Lambda *int(g.edges[[i,j]]["capacity"])/c_min)
            lin_expressions = [cplex.SparsePair(ind=variables, val=coeffs)]
            prob.linear_constraints.add(lin_expr=lin_expressions, senses=['L'], rhs=[0.])
        else:
            lin_expressions = [cplex.SparsePair(ind=variables, val=coeffs)]
            prob.linear_constraints.add(lin_expr=lin_expressions, senses=['L'], rhs=[Lambda *int(g.edges[[i,j]]["capacity"])/c_min])


def conservation_of_flow_constraint(prob, g, key_dict):
    for k in key_dict:
        for n in g.nodes:
            if n != k[0] and n != k[1]:
                variables = []
                coeffs = []
                for m in g.adj[n]:
                    variables.append(f"x{n}_{m}_k{k[0]}_{k[1]}")
                    coeffs.append(1)
                    variables.append(f"x{m}_{n}_k{k[0]}_{k[1]}")
                    coeffs.append(-1)
                lin_expressions = [cplex.SparsePair(ind=variables, val=coeffs)]
                prob.linear_constraints.add(lin_expr=lin_expressions, senses=['E'], rhs=[0.])

def capacity_requirement_constraint(prob, g, key_dict, N):
    for k in key_dict:
        variables = []
        coeffs = []
        for m in g.adj[k[1]]:
            variables.append(f"x{m}_{k[1]}_k{k[0]}_{k[1]}")
            coeffs.append(1)
        lin_expressions = [cplex.SparsePair(ind=variables, val=coeffs)]
        prob.linear_constraints.add(lin_expr=lin_expressions, senses=['G'], rhs=[N])

def add_flow_into_source_out_sink(prob, g, key_dict):
        for k in key_dict:
            for n in g.adj[k[0]]:
                variables = [f"x{n}_{k[0]}_k{k[0]}_{k[1]}"]
                coeffs = [1]
                lin_expressions = [cplex.SparsePair(ind=variables, val=coeffs)]
                prob.linear_constraints.add(lin_expr=lin_expressions, senses=['E'], rhs=[0])
            for n in g.adj[k[1]]:
                variables = [f"x{k[1]}_{n}_k{k[0]}_{k[1]}"]
                coeffs = [1]
                lin_expressions = [cplex.SparsePair(ind=variables, val=coeffs)]
                prob.linear_constraints.add(lin_expr=lin_expressions, senses=['E'], rhs=[0])

def add_no_flow_into_untrusted_node(prob, g, key_dict):
    for k in key_dict:
        for j in g.nodes:
            if g.nodes[j]["type"] == "S" and k[1] != j:
                for i in g.adj[j]:
                    variables = [f"x{i}_{j}_k{k[0]}_{k[1]}"]
                    coeffs = [1]
                    lin_expressions = [cplex.SparsePair(ind=variables, val=coeffs)]
                    prob.linear_constraints.add(lin_expr=lin_expressions, senses=['E'], rhs=[0])


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
                        ind_flow.extend([f"x{i}_{j}_k{k[0]}_{k[1]}"])
                        val.extend([1])
                lin_expr = [cplex.SparsePair(ind=ind_flow, val=val)]
                prob.linear_constraints.add(lin_expr=lin_expr, senses='L' * len(lin_expr),
                                             rhs=[1] * len(lin_expr))


def add_objective_approximation(prob, g, key_dict, C_on, C_det_source_pair, c_min):
    obj_vals = []
    for i in g.nodes:
        if g.nodes[i]["type"] == "T":
            obj_vals.append((f"delta_{i}", C_on))
    for i,j in g.edges:
        for k in key_dict:
            # add 1 to the capacity to ensure divide by 0 errors are not an issue - this will cause the cost of such an edge
            # to be massive compared to other edges
            obj_vals.append((f"x{i}_{j}_k{k[0]}_{k[1]}", C_det_source_pair * c_min / int(g.edges[[i,j]]["capacity"] + 1)))
    prob.objective.set_linear(obj_vals)
    prob.objective.set_sense(prob.objective.sense.minimize)

def initial_optimisation_cost_reduction(g, key_dict, cmin, time_limit = 1e5,  Lambda = 100, N = 2, C_on = 10, C_det_source_pair = 1.5):
    """
    set up and solve the problem for minimising the overall cost of the network
    """
    t_0 = time.time()
    print("Start Optimisation")
    prob = cplex.Cplex()
    add_capacity_constraint(prob, g, key_dict, Lambda, cmin)
    conservation_of_flow_constraint(prob, g, key_dict)
    # add_capacity_constraint(prob, g, key_dict, Lambda=Lambda)
    capacity_requirement_constraint(prob, g, key_dict, N)
    add_flow_into_source_out_sink(prob, g, key_dict)
    # add_untransformed_transformed_relationship(prob, g, key_dict, NT)
    add_no_flow_into_untrusted_node(prob, g, key_dict)
    add_limited_flow_through_connection(prob, g, key_dict)
    add_objective_approximation(prob, g, key_dict, C_on, C_det_source_pair, cmin)
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
    flow_dict, binary_dict = split_soln_dict(sol_dict)
    # log_optimal_solution_to_problem_column_format(prob, save_file="solution_dictionary_column_format", graph_id=0)
    trusted_nodes = 0
    for key in binary_dict:
        trusted_nodes += binary_dict[key]
    print(f"Number of Trusted Nodes = {trusted_nodes}")
    return sol_dict, prob


if __name__ == "__main__":
    key_dict, g, position_graphs = trusted_nodes_utils.import_problem_from_files_flexibility_multiple_graphs_position_graph("15_nodes_cap_needed.csv", "15_nodes_edge_data.csv", "15_nodes_node_types.csv", position_node_file = "15_nodes_position_graph_nodes.csv", position_edge_file="15_nodes_position_graph_edges.csv")
    for key in g.keys():
        try:
            sol_dict, prob = initial_optimisation_cost_reduction(g[key], key_dict[key], cmin = 10000, Lambda =6)
        except:
            continue
