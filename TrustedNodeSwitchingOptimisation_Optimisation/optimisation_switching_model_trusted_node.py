import cplex
import networkx as nx
import matplotlib.pyplot as plt
import scipy.optimize as opt
import optimisationmodel
import trusted_nodes_utils
import numpy as np
import time
import csv

"""
Optimisation of trusted node locations using the switching model where N_i^S and N_i^D are considered separately. But 
no calibration times are accounted for here.
 """



class Optimisation_Switching_Problem():

    def log_optimal_solution_to_problem(self, prob, save_file, graph_id):
        sol_dict = optimisationmodel.create_sol_dict(prob)
        flow_dict, binary_dict, lambda_dict = optimisationmodel.split_sol_to_flow_delta_lambda(sol_dict)
        ## if file does not exist - we wish to store information of q_{i,j,d}^{m}, w_{i,j,d}^{m}, lambda_{d}^{m}, delta_{i,j,d}^{m}
        ## need to generate all possible values of i,j,d available. Need to think of the most appropriate way to store this data.
        dict = {"ID": graph_id}
        dict.update(flow_dict)
        dict.update(binary_dict)
        dict.update(lambda_dict)
        dictionary = [dict]
        dictionary_fieldnames = list(dict.keys())
        with open(save_file + '.csv', mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
            writer.writeheader()
            writer.writerows(dictionary)

    def plot_graph(self, position_graph, binary_dict, save_name):
        graph = position_graph.to_undirected()
        pos = {}
        for node in graph.nodes:
            pos[node] = [graph.nodes[node]["xcoord"], graph.nodes[node]["ycoord"]]
        plt.figure()
        consumer_nodes_list = []
        trusted_node_list = []
        for node in graph.nodes:
            if graph.nodes[node]["type"] == "S" or self.g.nodes[node]["type"] == "NodeType.S":
                consumer_nodes_list.append(node)
            else:
                trusted_node_list.append(node)
        nx.draw_networkx_nodes(graph, pos, nodelist=consumer_nodes_list, node_color="k", label = "Consumer Nodes")
        all_nodes = []
        on_trusted_nodes = []
        off_trusted_nodes = []
        #### consider this but for cold and hot.....
        for key in binary_dict:
            current_node = int(key[6:])
            on_off = int(binary_dict[key])
            all_nodes.append(current_node)
            if on_off == 1:
                on_trusted_nodes.append(current_node)
            elif on_off == 0:
                off_trusted_nodes.append(current_node)
        nx.draw_networkx_nodes(graph, pos, nodelist=on_trusted_nodes, node_shape="d", node_color= "b", label="On Trusted Node")
        nx.draw_networkx_nodes(graph, pos, nodelist=off_trusted_nodes, node_shape="o", node_color= "b", label="Off Trusted Node")
        nx.draw_networkx_edges(graph, pos, edge_color="k")
        plt.axis("off")
        plt.legend(loc="best", fontsize="small")
        plt.savefig(save_name)
        plt.show()


    def add_source_capacity_on_edge_constraint(self, Lambda):
        pass

    def add_detector_capacity_on_edge_constraint(self, Lambda):
        pass

    def add_source_detector_max_constraint(self, Lambda):
        pass

    def add_flow_conservation_constraint(self):
        pass

    def add_flow_requirement_constraint(self, cmin):
        pass

    def add_flow_into_source(self):
        pass

    def add_flow_into_untrusted_node(self):
        pass

    def add_limited_flow_through_connection(self, cmin):
        pass

    def add_objective_function(self, *args, **kwargs):
        pass

    def initial_optimisation_cost_reduction(self, cmin, time_limit=1e5, *args, **kwargs):
        pass


class Optimisation_Switching_No_Calibration(Optimisation_Switching_Problem):

    def __init__(self, prob, g, key_dict):
        self.prob = prob
        self.g = g
        self.key_dict = key_dict
        super().__init__()



    def add_source_capacity_on_edge_constraint(self, Lambda):
        """
        adds constriant that ensures capacity cannot exceed capabilities of sources on the edge:

        \sum_{j \in N(i)} \frac{\sum_{k \in K} x_{i,j}^{k}}{c_{i,j}} \leq N_i^{S}
        """

        source_terms = []
        for n in self.g.nodes:
            source_terms.append(f"N_{n}_S")
        variable_names = [f'x{i}_{j}_k{k[0]}_{k[1]}' for k in self.key_dict for i, j in list(self.g.edges)]
        self.prob.variables.add(names=variable_names, types=[self.prob.variables.type.continuous] * len(variable_names))
        self.prob.variables.add(names=source_terms, types=[self.prob.variables.type.integer] * len(source_terms),
                           ub=[Lambda] * len(source_terms))
        for i in self.g.nodes:
            flow_along_edge = []
            coeff = []
            for j in self.g.adj[i]:
                capacity = int(self.g.edges[[i, j]]["capacity"])
                if capacity > 0.0000001:
                    for k in self.key_dict:
                        flow_along_edge.append(f"x{i}_{j}_k{k[0]}_{k[1]}")
                        coeff.append(1 / capacity)
            flow_along_edge.append(f"N_{i}_S")
            coeff.append(-1)
            lin_expressions = [cplex.SparsePair(ind=flow_along_edge, val=coeff)]
            self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['L'], rhs=[0.])

    def add_detector_capacity_on_edge_constraint(self, Lambda):
        """
        adds constriant that ensures capacity cannot exceed capabilities of detectors on the edge:

        \sum_{j \in N(i)} \frac{\sum_{k \in K} x_{j,i}^{k}}{c_{i,j}} \leq N_i^{D} for trusted nodes i \in T
        \sum_{j \in N(i)} \frac{\sum_{k \in K} x_{j,i}^{k}}{c_{i,j}} \leq 0 for i \in S
        """
        detector_terms = []
        for n in self.g.nodes:
            if self.g.nodes[n]["type"] == "T" or self.g.nodes[n]["type"] == "NodeType.T":
                detector_terms.append(f"N_{n}_D")
        self.prob.variables.add(names=detector_terms, types=[self.prob.variables.type.integer] * len(detector_terms),
                           ub=[Lambda] * len(detector_terms))
        for i in self.g.nodes:
            if self.g.nodes[i]["type"] == "T" or self.g.nodes[i]["type"] == "NodeType.T":
                flow_along_edge = []
                coeff = []
                for j in self.g.adj[i]:
                    capacity = int(self.g.edges[[i, j]]["capacity"])
                    if capacity > 0.0000001:
                        for k in self.key_dict:
                            flow_along_edge.append(f"x{j}_{i}_k{k[0]}_{k[1]}")
                            coeff.append(1 / capacity)
                flow_along_edge.append(f"N_{i}_D")
                coeff.append(-1)
                lin_expressions = [cplex.SparsePair(ind=flow_along_edge, val=coeff)]
                self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['L'], rhs=[0.])
            else:
                # flow into source nodes/untrusted nodes is 0 as there are no detectors
                flow_along_edge = []
                coeff = []
                for j in self.g.adj[i]:
                    capacity = int(self.g.edges[[i, j]]["capacity"])
                    if capacity > 0.0000001:
                        for k in self.key_dict:
                            flow_along_edge.append(f"x{j}_{i}_k{k[0]}_{k[1]}")
                            coeff.append(1 / capacity)
                lin_expressions = [cplex.SparsePair(ind=flow_along_edge, val=coeff)]
                self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['L'], rhs=[0.])

    def add_source_detector_max_constraint(self, Lambda):
        """
        Add constraint that maximises the source and detector numbers on trusted nodes:
            N_{i}^{S/D} \leq \Lambda \delta_i \forall i \in T
        """
        delta_terms = []
        for n in self.g.nodes:
            if self.g.nodes[n]["type"] == "T" or self.g.nodes[n]["type"] == "NodeType.T":
                delta_terms.append(f"delta_{n}")
        self.prob.variables.add(names=delta_terms, types=[self.prob.variables.type.binary] * len(delta_terms))
        for i in self.g.nodes:
            if self.g.nodes[i]["type"] == "T" or self.g.nodes[i]["type"] == "NodeType.T":
                detector_terms = [f"N_{i}_D", f"delta_{i}"]
                source_terms = [f"N_{i}_S", f"delta_{i}"]
                lin_expressions = [cplex.SparsePair(ind=detector_terms, val=[1, -Lambda]),
                                   cplex.SparsePair(ind=source_terms, val=[1, -Lambda])]
                self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['LL'], rhs=[0., 0.])

    def add_flow_conservation_constraint(self):
        """
        Add the conservation of flow constraint:

        \sum_{m \in N(n)} x^{k}_{(n,m)} + x^{k_R}_{(m,n)} - x^{k}_{(m,n)} - x^{k_R}_{(n,m)} = 0
        """
        for i in self.g.nodes:
            for k in self.key_dict:
                if k[0] < k[1] and k[1] != i and k[0] != i:
                    flow = []
                    val = []
                    for n in self.g.neighbors(i):
                        flow.extend([f"x{i}_{n}_k{k[0]}_{k[1]}", f"x{n}_{i}_k{k[1]}_{k[0]}", f"x{n}_{i}_k{k[0]}_{k[1]}",
                                     f"x{i}_{n}_k{k[1]}_{k[0]}"])
                        val.extend([1, 1, -1, -1])
                    lin_expressions = [cplex.SparsePair(ind=flow, val=val)]
                    self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=["E"], rhs=[0.])

    def add_flow_requirement_constraint(self, cmin):
        """
        Adds the constraint to ensure the total flow into sink is greater than the required flow

        \sum_{m \in N(j)} x_{(m,j)}^{k=(i,j)} + x_{(j,m)}^{k_R= (j,i)}  \geq N'_k c_{min}
        """
        for k in self.key_dict:
            if k[0] < k[1]:
                ind = []
                val = []
                for n in self.g.neighbors(k[1]):
                    ind.extend([f"x{n}_{k[1]}_k{k[0]}_{k[1]}", f"x{k[1]}_{n}_k{k[1]}_{k[0]}"])
                    val.extend([1, 1])
                lin_expressions = [cplex.SparsePair(ind=ind, val=val)]
                if isinstance(cmin, dict):
                    self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=["G"], rhs=[float(self.key_dict[k] * cmin[k])])
                else:
                    self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=["G"],
                                                     rhs=[float(self.key_dict[k] * cmin)])

    def add_flow_into_source(self):
        """
        adds the constraint that ensures no flow into the source/ no flow out of sink

        x_{(i,j)}^{k=(m,i)} + x_{(j,i)}^{k_R=(i,m)} = 0
        x_{(j,m)}^{k=(n,m)} + x_{(m,j)}^{k_R=(m,n)} = 0
        """
        for k in self.key_dict:
            if k[0] < k[1]:
                for n in self.g.neighbors(k[1]):
                    ind = [f"x{k[1]}_{n}_k{k[0]}_{k[1]}", f"x{n}_{k[1]}_k{k[1]}_{k[0]}"]
                    val = [1, 1]
                    lin_expressions = [cplex.SparsePair(ind=ind, val=val)]
                    self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=["E"], rhs=[0])
                for n in self.g.neighbors(k[0]):
                    ind = [f"x{n}_{k[0]}_k{k[0]}_{k[1]}", f"x{k[0]}_{n}_k{k[1]}_{k[0]}"]
                    val = [1, 1]
                    lin_expressions = [cplex.SparsePair(ind=ind, val=val)]
                    self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=["E"], rhs=[0])

    def add_flow_into_untrusted_node(self):
        """
        Ensures the flow of commodities into an untrusted node that is not the sink of the commodity is = 0. i.e. no
        flow can be routed through an untrusted node:

        x_{(i,j)}^{k=(n,m)} + x_{(j,i)}^{k_R = (m,n)} = 0 \forall j \in S j \neq m, i \neq n
        """
        for i, nodes in enumerate(self.g.edges):
            source_node_type = self.g.nodes[nodes[0]]["type"]
            target_node_type = self.g.nodes[nodes[1]]["type"]
            # if target node is a source node then only if the commodity is for target node can the flow in be non-zero
            if target_node_type == "S" or target_node_type == "NodeType.S":
                source_node = nodes[0]
                target_node = nodes[1]
                ind_flow = []
                for j, k in enumerate(self.key_dict):
                    # find the keys with commodity that is not for current target node
                    if k[0] < k[1] and k[1] != target_node:
                        ind_flow.append([f"x{source_node}_{target_node}_k{k[0]}_{k[1]}",
                                         f"x{target_node}_{source_node}_k{k[1]}_{k[0]}"])
                cap_const_2 = []
                # set constraints of these commodities to 0
                for j in range(len(ind_flow)):
                    cap_const_2.append(cplex.SparsePair(ind=ind_flow[j], val=[1, 1]))
                self.prob.linear_constraints.add(lin_expr=cap_const_2, senses='E' * len(cap_const_2),
                                            rhs=[0] * len(cap_const_2))


    def add_limited_flow_through_connection_old(self, cmin):
        """
        Adds the constraint that ensures the flow along an edge is limited by cmin ensuring at least N_k paths are used

        x_{i,j}^{k} + x_{i,j}^{k_R} + x_{j,i}^{k} + x_{j,i}^{k_R} \leq c_{min}
        """
        for i, j in self.g.edges:
            for k in self.key_dict:
                if k[0] < k[1]:
                    ind_flow = [f"x{i}_{j}_k{k[0]}_{k[1]}", f"x{i}_{j}_k{k[1]}_{k[0]}", f"x{j}_{i}_k{k[0]}_{k[1]}",
                                f"x{j}_{i}_k{k[1]}_{k[0]}"]
                    val = [1, 1, 1, 1]
                    lin_expr = [cplex.SparsePair(ind=ind_flow, val=val)]
                    self.prob.linear_constraints.add(lin_expr=lin_expr, senses='L' * len(lin_expr),
                                                rhs=[cmin] * len(lin_expr))

    def add_limited_flow_through_connection(self, cmin):
        """
        Adds the constraint that ensures the flow along an edge is limited by cmin ensuring at least N_k paths are used

        \sum_{j \in \mathcal{N}(i)} x_{i,j}^{k} + x_{j,i}^{k_R} \leq c_{min} except if i is source
        """
        for i in self.g.nodes:
            for k in self.key_dict:
                if k[0] != i:
                    ind_flow = []
                    val = []
                    for j in self.g.adj[i]:
                        if k[0] < k[1]:
                            ind_flow.extend([f"x{i}_{j}_k{k[0]}_{k[1]}", f"x{j}_{i}_k{k[1]}_{k[0]}"])
                            val.extend([1,1])
                    lin_expr = [cplex.SparsePair(ind=ind_flow, val=val)]
                    if isinstance(cmin, dict):
                        self.prob.linear_constraints.add(lin_expr=lin_expr, senses='L' * len(lin_expr),
                                             rhs=[float(cmin[k])] * len(lin_expr))
                    else:
                        self.prob.linear_constraints.add(lin_expr=lin_expr, senses='L' * len(lin_expr),
                                                         rhs=[float(cmin)] * len(lin_expr))


    def add_objective_function(self, cost_on_trusted_node, cost_detector, cost_source, *args, **kwargs):
        """
        adds the objective function: minimise the cost of the network

        \sum_{j \in T} C_{tn, j}\delta_{j} + \sum_{i \in N} C_{i}^{D} N_{i}^{D} + C_{i}^{S} N_{i}^{S}
        """
        obj_vals = []
        for i in self.g.nodes:
            obj_vals.append((f"N_{i}_S", cost_source))
            if self.g.nodes[i]["type"] == "T" or self.g.nodes[i]["type"] == "NodeType.T":
                obj_vals.append((f"N_{i}_D", cost_detector))
                obj_vals.append((f"delta_{i}", cost_on_trusted_node))
        self.prob.objective.set_linear(obj_vals)
        self.prob.objective.set_sense(self.prob.objective.sense.minimize)

    def initial_optimisation_cost_reduction(self, cmin, time_limit=1e5, cost_on_trusted_node=1,
                                            cost_detector=0.1, cost_source=0.01, Lambda=100):
        """
        set up and solve the problem for minimising the overall cost of the network
        """
        t_0 = time.time()
        print("Start Optimisation")
        self.add_source_capacity_on_edge_constraint(Lambda)
        self.add_detector_capacity_on_edge_constraint(Lambda)
        self.add_source_detector_max_constraint(Lambda)
        self.add_flow_conservation_constraint()
        self.add_flow_requirement_constraint(cmin)
        self.add_flow_into_source()
        self.add_flow_into_untrusted_node()
        self.add_limited_flow_through_connection(cmin)
        self.add_objective_function(cost_on_trusted_node, cost_detector, cost_source)
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
        print("Time to set up problem: " + str(t_1 - t_0))
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



if __name__ == "__main__":
    key_dict, g = trusted_nodes_utils.import_problem_from_files_flexibility_multiple_graphs("one_path_small_graphs_for_test_cap_needed.csv", "one_path_small_graphs_for_test_edge_data.csv", "one_path_small_graphs_for_test_node_types.csv")
    for key in g.keys():
        key_dict_temp = trusted_nodes_utils.make_key_dict_bidirectional(key_dict[key])
        prob = cplex.Cplex()
        optim = Optimisation_Switching_No_Calibration(prob=prob, g = g[key], key_dict = key_dict_temp)
        sol_dict, prob = optim.initial_optimisation_cost_reduction(cmin = 200, Lambda =24)
