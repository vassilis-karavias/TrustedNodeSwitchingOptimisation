import cplex
import trusted_nodes_utils
import numpy as np

def add_capacity_constraint(problem, graph, key_dict):
    """
    Add the capacity constraints to the initial flow problem
    :param problem: the cplex class with the minimisation problem
    :param graph: The graph in capacity space with the edges and vertices for the trusted nodes
    :param key_dict: The list of commodities needed in {(source, target): capacity_needed}
    """
    num_comm = len(key_dict)
    binary_trusted_variables = [f"delta{n}" for n in graph.nodes]
    problem.variables.add(names = binary_trusted_variables, types = [problem.variables.type.binary]*graph.number_of_nodes())
    for i, nodes in enumerate(graph.edges):
        # get the indices of the flow variables on the selected edge
        ind_flow = [i + k * graph.number_of_edges() for k in range(num_comm)]
        # get the capacity of the edge
        capacity = graph.edges[nodes]["capacity"]
        binary_value = [f'delta{nodes[1]}']
        # for capacity constraint with sum of all flow across channel cannot exceed capacity
        cap_const_1 = cplex.SparsePair(ind = ind_flow + binary_value, val = [1]*len(ind_flow) + [-capacity])
        # for capacity constraint where we assume we have enough for each commodity.
        cap_const_2 = []
        for j in range(len(ind_flow)):
            cap_const_2.append(cplex.SparsePair(ind = [ind_flow[j]] + binary_value, val = [1, -capacity]))
        problem.linear_constraints.add(lin_expr = cap_const_2, senses = 'L'*len(cap_const_2), rhs = [0]* len(cap_const_2))


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
        sn[pair[0]+active_commodity]=num_keys # sets source
        sn[pair[1]+active_commodity]=-num_keys # sets sink
        count+=1 # counter moves forward for next commodity
    my_senses='E'*num_nodes*num_comm
    prob.linear_constraints.add(senses=my_senses,rhs=sn.tolist())
    prob.variables.add(names=variable_names, columns=flow, types=[prob.variables.type.integer]*len(variable_names)) # add variables and flow conservation

def add_bidirectional_flow_conservation_constraint(prob,g,key_dict):
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
    obj_vals = [(f"delta{n}", 1.) for n in g.nodes]
    prob.objective.set_linear(obj_vals)
    prob.objective.set_sense(prob.objective.sense.minimise)




def add_capacity_constraints_rigidity_test(problem, graph, key_dict, flow_sol, off_trusted_nodes):
    """

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
