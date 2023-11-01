import cplex
import networkx as nx
import matplotlib.pyplot as plt
import brute_force
import trusted_nodes_utils
from optimisation_switching_model_trusted_node import Optimisation_Switching_Problem
import numpy as np
import time
import optimisationmodel


"""
FILE FOR NO SWITCHING SITUATION
"""

class Optimisation_Problem_No_Switching(Optimisation_Switching_Problem):

    def __init__(self, prob, g, key_dict):
        self.prob = prob
        self.g = g
        self.key_dict = key_dict
        super().__init__()


    def add_lambda_constraint(self, Lambda = 10):
        """
        adds the constraint \sum_{k \in K: n>m} X_{(i,j)}^{k=(n,m)} + X_{(j,i)}^{k=(n,m)} \leq \lambda_{i,j} c_{i,j} for i<j
        """
        num_comm = len(self.key_dict)
        binary_trusted_variables = []
        # add the lambdas to the variables of problem
        for i, j in self.g.edges:
            if i < j:
                binary_trusted_variables.append(f"lambda{i, j}")
        self.prob.variables.add(names=binary_trusted_variables,
                           types=[self.prob.variables.type.integer] * len(binary_trusted_variables),
                           ub=[Lambda] * len(binary_trusted_variables))
        for i, nodes in enumerate(self.g.edges):
            if nodes[0] < nodes[1]:
                # f'x{i}_{j}_k{k[0]}_{k[1]}'
                flow_variables = []
                for k in self.key_dict:
                    flow_variables.append(f"x{nodes[0]}_{nodes[1]}_k{k[0]}_{k[1]}")
                    flow_variables.append(f"x{nodes[1]}_{nodes[0]}_k{k[0]}_{k[1]}")
                capacity = int(self.g.edges[nodes]["capacity"])
                lambda_ij = [f'lambda{nodes[0], nodes[1]}']
                cap_const_1 = cplex.SparsePair(ind=flow_variables + lambda_ij, val=[1] * len(flow_variables) + [-capacity])
                self.prob.linear_constraints.add(lin_expr=[cap_const_1], senses='L', rhs=[0])

    def lambda_constraint(self, Lambda):
        """
        \lambda_{i,j} \leq \Lambda \delta_{j}
        """
        num_comm = len(self.key_dict)
        binary_trusted_variables = []
        # add the lambdas to the variables of problem
        for n in self.g.nodes:
            if self.g.nodes[n]["type"] == "T" or self.g.nodes[n]["type"] == "NodeType.T":
                binary_trusted_variables.append(f"delta{n}")
        self.prob.variables.add(names=binary_trusted_variables,
                              types=[self.prob.variables.type.binary] * len(binary_trusted_variables))
        for i, j in self.g.edges:
            if i < j:
                if self.g.nodes[i]["type"] == "T" or self.g.nodes[i]["type"] == "NodeType.T":
                    lambda_constraint_1 = cplex.SparsePair(ind = [f"lambda{i, j}"] + [f"delta{i}"], val = [1] + [-Lambda])
                    self.prob.linear_constraints.add(lin_expr = [lambda_constraint_1], senses = "L", rhs = [0])
                if self.g.nodes[j]["type"] == "T" or self.g.nodes[j]["type"] == "NodeType.T":
                    lambda_constraint_1 = cplex.SparsePair(ind=[f"lambda{i, j}"] + [f"delta{j}"], val=[1] + [-Lambda])
                    self.prob.linear_constraints.add(lin_expr=[lambda_constraint_1], senses="L", rhs=[0])



    def add_flow_conservation_constraint(self, cmin):
        """Applies constraints to cplex problem for flow conservation
        key_dict contains only keys with terms k[0] < k[1]
        """
        num_comm=len(self.key_dict)# checks number of key exchange pairs i.e. number of commodities
        num_nodes= self.g.number_of_nodes()
        variable_names=[f'x{i}_{j}_k{k[0]}_{k[1]}' for k in self.key_dict for i,j in list(self.g.edges) ] # name convention x1_2k3_4 is flow in direction 1 to 2 for key 3 to 4
        flow=[[[int(i+k*num_nodes),int(j+k*num_nodes)], [1.0,-1.0]] for k in range(num_comm) for i,j in self.g.edges ] # tails are positive, heads negative, ensures source, sink balance

        sn=np.zeros(num_nodes*num_comm) # zeros ensure flow conservation
        count=0
        for pair,num_keys in self.key_dict.items():
            active_commodity=count*num_nodes
            sn[int(pair[0])+active_commodity]=int(num_keys * cmin) # sets source
            sn[int(pair[1])+active_commodity]=-int(num_keys * cmin) # sets sink
            count+=1 # counter moves forward for next commodity
        my_senses='E'*num_nodes*num_comm
        self.prob.linear_constraints.add(senses=my_senses,rhs=sn.tolist())
        self.prob.variables.add(names=variable_names, columns=flow, types=[self.prob.variables.type.integer]*len(variable_names)) # add variables and flow conservation


    def add_flow_into_source(self):
        num_nodes = self.g.number_of_nodes()
        for k in self.key_dict:
            flow_into_source_variables = [f'x{j}_{k[0]}_k{k[0]}_{k[1]}' for j in self.g.adj[k[0]]]  # flow into source from forwards commodity (must be zero to avoid loops)
            flow_out_sink_variables = [f'x{k[1]}_{j}_k{k[0]}_{k[1]}' for j in self.g.adj[k[1]]]  # flow out of sink from forwards commodity (must be zero to avoid loops)
            lin_expressions = [cplex.SparsePair(ind=flow_into_source_variables, val=[1.] * len(flow_into_source_variables)),
                               cplex.SparsePair(ind=flow_out_sink_variables, val=[1.] * len(flow_out_sink_variables))]
            self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['EE'], rhs=[0., 0.])

    def add_constraint_source_nodes(self):
        """
        Adds constraint that ensures that inflow into source nodes is 0 unless it is for the commodity
        required by the source node
        """
        num_comm = len(self.key_dict)

        for i, nodes in enumerate(self.g.edges):
            source_node_type = self.g.nodes[nodes[0]]["type"]
            target_node_type = self.g.nodes[nodes[1]]["type"]
            # if target node is a source node then only if the commidity is for target node can the flow in be non-zero
            if target_node_type == "S" or target_node_type == "NodeType.S":
                source_node = nodes[0]
                target_node = nodes[1]
                ind_flow = []
                for j, k in enumerate(self.key_dict):
                    # find the keys with commodity that is not for current target node
                    if k[1] != target_node:
                        ind_flow.append(f"x{source_node}_{target_node}_k{k[0]}_{k[1]}")
                cap_const_2 = []
                # set constraints of these commodities to 0
                for j in range(len(ind_flow)):
                    cap_const_2.append(cplex.SparsePair(ind=[ind_flow[j]], val=[1]))
                self.prob.linear_constraints.add(lin_expr=cap_const_2, senses='E' * len(cap_const_2),
                                               rhs=[0] * len(cap_const_2))


    def add_limited_flow_through_connection(self, cmin):
        for i in self.g.nodes:
            for k in self.key_dict:
                if k[0] != i:
                    ind_flow = []
                    val = []
                    for j in self.g.adj[i]:
                        ind_flow.extend([f"x{i}_{j}_k{k[0]}_{k[1]}"])
                        val.extend([1])
                    lin_expr = [cplex.SparsePair(ind=ind_flow, val=val)]
                    if isinstance(cmin, dict):
                        self.prob.linear_constraints.add(lin_expr=lin_expr, senses='L' * len(lin_expr),
                                                         rhs=[float(cmin[k])] * len(lin_expr))
                    else:
                        self.prob.linear_constraints.add(lin_expr=lin_expr, senses='L' * len(lin_expr),
                                                         rhs=[float(cmin)] * len(lin_expr))


    def add_minimise_overall_cost_objective(self, cost_node, cost_connection):
        """
        Add the objective to minimise the cost of the network sum_{j in T} c_{j}deta_j + sum_{i,j in E} C_{i,j} lambda_{i,j}

        Parameters
        ----------
        cost_node : The cost of turning on the node
        cost_connection : The cost of adding extra connection
        -------
        """
        obj_vals = []
        for n in self.g.nodes:
            if self.g.nodes[n]["type"] == "T" or self.g.nodes[n]["type"] == "NodeType.T":
                obj_vals.append((f"delta{n}", cost_node))
        for i,j in self.g.edges:
            if i < j:
                obj_vals.append((f"lambda{i, j}", cost_connection))
        self.prob.objective.set_linear(obj_vals)
        self.prob.objective.set_sense(self.prob.objective.sense.minimize)


    def initial_optimisation_cost_reduction(self, time_limit = 3e7, cost_node = 1, cost_connection= 0.1, Lambda = 5, cmin = 1000):
        """
        set up and solve the problem for minimising the overall cost of the network
        """
        t_0 = time.time()
        print("Start Optimisation")
        # add constraints to the problem
        self.add_flow_conservation_constraint(cmin)
        self.add_flow_into_source()
        self.add_lambda_constraint(Lambda=Lambda)
        self.lambda_constraint(Lambda=Lambda)
        self.add_constraint_source_nodes()
        self.add_limited_flow_through_connection(cmin)
        # add_minimise_trusted_nodes_objective(prob, g)
        # add objective to the problem
        self.add_minimise_overall_cost_objective(cost_node, cost_connection)
        self.prob.write("test.lp")
        # set the parameters of the solver to improve speed of solution
        self.prob.parameters.lpmethod.set(self.prob.parameters.lpmethod.values.network)
        self.prob.parameters.mip.tolerances.integrality= 1e-08
        self.prob.parameters.mip.limits.cutpasses.set(1)
        self.prob.parameters.mip.strategy.probe.set(-1)
        self.prob.parameters.mip.strategy.variableselect.set(0)
        print(self.prob.parameters.get_changed())
        self.prob.parameters.timelimit.set(time_limit)
        t_1 = time.time()
        print("Time to set up problem: " + str(t_1-t_0))
        self.prob.solve()
        t_2 = time.time()
        print("Time to solve problem: " + str(t_2 - t_1))
        print(f"The minimum Cost of Network: {self.prob.solution.get_objective_value()}")
        print(f"Number of Variables = {self.prob.variables.get_num()}")
        print(f"Number of Conditions = {self.prob.linear_constraints.get_num()}")
        sol_dict = optimisationmodel.create_sol_dict(self.prob)
        flow_dict, binary_dict, lambda_dict = optimisationmodel.split_sol_to_flow_delta_lambda(sol_dict)
        trusted_nodes = 0
        for key in binary_dict:
            trusted_nodes += binary_dict[key]
        print(f"Number of Trusted Nodes = {trusted_nodes}")
        return sol_dict, self.prob, t_2- t_1



class Optimisation_Problem_No_Switching_No_Trusted_Nodes(Optimisation_Problem_No_Switching):

    def __init__(self, prob, g, key_dict):
        super().__init__(prob, g, key_dict)

    def add_minimise_overall_cost_objective(self, cost_connection):
        """
        Add the objective to minimise the cost of the network sum_{j in T} c_{j}deta_j + sum_{i,j in E} C_{i,j} lambda_{i,j}

        Parameters
        ----------
        cost_connection : The cost of adding extra connection
        -------
        """
        obj_vals = []
        for i,j in self.g.edges:
            if i < j:
                obj_vals.append((f"lambda{i, j}", cost_connection))
        self.prob.objective.set_linear(obj_vals)
        self.prob.objective.set_sense(self.prob.objective.sense.minimize)


    def initial_optimisation_cost_reduction(self, time_limit = 3e7, cost_connection= 0.1, Lambda = 5, cmin = 1000):
        """
        set up and solve the problem for minimising the overall cost of the network
        """
        t_0 = time.time()
        print("Start Optimisation")
        # add constraints to the problem
        self.add_flow_conservation_constraint(cmin)
        self.add_flow_into_source()
        self.add_lambda_constraint(Lambda=Lambda)
        self.add_constraint_source_nodes()
        self.add_limited_flow_through_connection(cmin)
        # add_minimise_trusted_nodes_objective(prob, g)
        # add objective to the problem
        self.add_minimise_overall_cost_objective(cost_connection)
        self.prob.write("test.lp")
        # set the parameters of the solver to improve speed of solution
        self.prob.parameters.lpmethod.set(self.prob.parameters.lpmethod.values.network)
        self.prob.parameters.mip.tolerances.integrality= 1e-08
        self.prob.parameters.mip.limits.cutpasses.set(1)
        self.prob.parameters.mip.strategy.probe.set(-1)
        self.prob.parameters.mip.strategy.variableselect.set(0)
        print(self.prob.parameters.get_changed())
        self.prob.parameters.timelimit.set(time_limit)
        t_1 = time.time()
        print("Time to set up problem: " + str(t_1-t_0))
        self.prob.solve()
        t_2 = time.time()
        print("Time to solve problem: " + str(t_2 - t_1))
        print(f"The minimum Cost of Network: {self.prob.solution.get_objective_value()}")
        print(f"Number of Variables = {self.prob.variables.get_num()}")
        print(f"Number of Conditions = {self.prob.linear_constraints.get_num()}")
        sol_dict = optimisationmodel.create_sol_dict(self.prob)
        flow_dict, binary_dict, lambda_dict = optimisationmodel.split_sol_to_flow_delta_lambda(sol_dict)
        trusted_nodes = 0
        for key in binary_dict:
            trusted_nodes += binary_dict[key]
        print(f"Number of Trusted Nodes = {trusted_nodes}")
        return sol_dict, self.prob, t_2- t_1