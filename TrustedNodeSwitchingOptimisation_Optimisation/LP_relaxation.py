from docplex.mp.model import Model
import numpy as np
from trusted_nodes_utils import *


class LP_relaxation_Trusted_Nodes_fixed_switching_time_relaxation():

    def __init__(self, name, g, key_dict):
        self.model = Model(name = name)
        self.g = g
        self.key_dict = key_dict

    def add_source_capacity_on_edge_constraint(self, Lambda, f_switch):
        """
        adds constriant that ensures capacity cannot exceed capabilities of sources on the edge:

        \sum_{j \in N(i)} \frac{\sum_{k \in K} x'_{i,j}^{k}}{c_{i,j}} \leq N_i^{S}

        """
        source_terms = []
        detector_terms = []
        for n in self.g.nodes:
            source_terms.append(f"N_{n}_S")
            if self.g.nodes[n]["type"] == "T" or self.g.nodes[n]["type"] == "NodeType.T":
                detector_terms.append(f"N_{n}_D")
        variable_names = [f'x{i}_{j}_k{k[0]}_{k[1]}' for k in self.key_dict for i, j in list(self.g.edges)]
        self.model.flow_variables = self.model.continuous_var_dict(keys = variable_names, lb = 0.0)
        self.model.source_variables = self.model.continuous_var_dict(keys = source_terms, lb = 0.0, ub = Lambda)
        self.model.detector_variables = self.model.continuous_var_dict(keys = detector_terms, lb = 0.0, ub = Lambda)

        for i in self.g.nodes:
            flow_along_edge = []
            coeff = []
            for j in self.g.adj[i]:
                capacity = int(self.g.edges[[i, j]]["capacity"])
                if capacity > 0.0000001:
                    for k in self.key_dict:
                        flow_along_edge.append(f"x{i}_{j}_k{k[0]}_{k[1]}")

                        coeff.append(1 / capacity)
                else:
                    for k in self.key_dict:
                        self.model.add_constraint(self.model.flow_variables[f"x{i}_{j}_k{k[0]}_{k[1]}"] <= 0)
            self.model.total_flow = self.model.sum(self.model.flow_variables[flow_along_edge[k]] * coeff[k] for k in range(len(flow_along_edge)))
            self.model.add_constraint(self.model.total_flow <= (1-f_switch) * self.model.source_variables[f"N_{i}_S"])


    def add_detector_capacity_on_edge_constraint(self, f_switch):
        """
        adds constriant that ensures capacity cannot exceed capabilities of detectors on the edge:

        \sum_{j \in N(i)} \frac{\sum_{k \in K} x'_{j,i}^{k}}{c_{i,j}} \leq N_i^{D} for trusted nodes i \in T
        \sum_{j \in N(i)} \frac{\sum_{k \in K} x'_'{j,i}^{k}}{c_{i,j}} \leq 0 for i \in S
        """

        for i in self.g.nodes:
            if self.g.nodes[i]["type"] == "T" or self.g.nodes[i]["type"] == "NodeType.T":
                flow_along_edge = []
                coeff = []
                total_nj_coeff = 0.0
                for j in self.g.adj[i]:
                    capacity = int(self.g.edges[[i, j]]["capacity"])
                    if capacity > 0.0000001:
                        for k in self.key_dict:
                            flow_along_edge.append(f"x{j}_{i}_k{k[0]}_{k[1]}")
                            coeff.append(1 / capacity)
                    else:
                        for k in self.key_dict:
                            self.model.add_constraint(self.model.flow_variables[f"x{j}_{i}_k{k[0]}_{k[1]}"] <= 0)
                self.model.total_flow = self.model.sum(self.model.flow_variables[flow_along_edge[k]] * coeff[k] for k in range(len(flow_along_edge)))
                self.model.add_constraint(self.model.total_flow <= (1 - f_switch) * self.model.detector_variables[f"N_{i}_D"])
            else:
                # flow into source nodes/untrusted nodes is 0 as there are no detectors
                flow_along_edge = []
                coeff = []
                for j in self.g.adj[i]:
                    capacity = int(self.g.edges[[i, j]]["capacity"])
                    for k in self.key_dict:
                        flow_along_edge.append(f"x{j}_{i}_k{k[0]}_{k[1]}")
                        coeff.append(1)
                self.model.total_flow = self.model.sum(self.model.flow_variables[flow_along_edge[k]] * coeff[k] for k in range(len(flow_along_edge)))
                self.model.add_constraint(self.model.total_flow <= 0)


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
                    self.model.total_flow = self.model.sum(self.model.flow_variables[flow[l]] * val[l] for l in range(len(flow)))
                    self.model.add_constraint(self.model.total_flow == 0.0)


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
                self.model.total_flow = self.model.sum(self.model.flow_variables[ind[i]] for i in range(len(ind)))
                self.model.add_constraint(self.model.total_flow >= self.key_dict[k]* cmin)


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
                    self.model.total_flow = self.model.sum(self.model.flow_variables[ind[i]] for i in range(len(ind)))
                    self.model.add_constraint(self.model.total_flow == 0.0)
                for n in self.g.neighbors(k[0]):
                    ind = [f"x{n}_{k[0]}_k{k[0]}_{k[1]}", f"x{k[0]}_{n}_k{k[1]}_{k[0]}"]
                    val = [1, 1]
                    self.model.total_flow = self.model.sum(self.model.flow_variables[ind[i]] for i in range(len(ind)))
                    self.model.add_constraint(self.model.total_flow == 0.0)


    def add_flow_into_untrusted_node(self):
        """
        Ensures the flow of commodities into an untrusted node that is not the sink of the commodity is = 0. i.e. no
        flow can be routed through an untrusted node:

        x_{(i,j)}^{k=(n,m)} + x_{(j,i)}^{k_R = (m,n)} = 0 \forall j \in S j \neq m, i \neq n
        """
        for i, nodes in enumerate(self.g.edges):
            source_node_type = self.g.nodes[nodes[0]]["type"]
            target_node_type = self.g.nodes[nodes[1]]["type"]
            # if target node is a source node then only if the commidity if for target node can the flow in be non-zero
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
                    self.model.total_flow = self.model.sum(self.model.flow_variables[ind_flow[j][k]] for k in range(len(ind_flow[j])))
                    self.model.add_constraint(self.model.total_flow == 0.0)

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
                    self.model.total_flow = self.model.sum(self.model.flow_variables[ind_flow[l]] for l in range(len(ind_flow)))
                    self.model.add_constraint(self.model.total_flow <= cmin)

    def add_objective_function(self, C_det, C_source):
        self.model.detector_cost = self.model.sum(self.model.detector_variables[key] * C_det for key in self.model.detector_variables.keys())
        self.model.source_cost = self.model.sum(self.model.source_variables[key] * C_source for key in self.model.source_variables.keys())
        self.model.minimize(self.model.detector_cost + self.model.source_cost)


    def set_up_problem(self, Lambda, f_switch, C_det, C_source, cmin):
        self.add_source_capacity_on_edge_constraint(Lambda=Lambda, f_switch=f_switch)
        self.add_detector_capacity_on_edge_constraint(f_switch=f_switch)
        self.add_flow_conservation_constraint()
        self.add_flow_requirement_constraint(cmin = cmin)
        self.add_flow_into_source()
        self.add_flow_into_untrusted_node()
        self.add_limited_flow_through_connection(cmin = cmin)
        self.add_objective_function(C_det = C_det, C_source= C_source)



if __name__ == "__main__":
    key_dict, g = import_problem_from_files_flexibility_multiple_graphs("one_path_small_graphs_for_test_cap_needed.csv", "one_path_small_graphs_for_test_edge_data.csv", "one_path_small_graphs_for_test_node_types.csv")
    for key in g.keys():
        key_dict_temp = make_key_dict_bidirectional(key_dict[key])
        model = LP_relaxation_Trusted_Nodes_fixed_switching_time_relaxation(name = f"problem_{key}", g = g[key], key_dict= key_dict_temp)
        model.set_up_problem(Lambda=6, f_switch=0.1, C_det=5, C_source=1, cmin=1000)
        model.model.print_information()
        if model.model.solve():
            obj = model.model.objective_value
            # for key in model.model.cold_capacity_variables.keys():
            #     print(str(key) + " solution value:" + str(model.model.cold_capacity_variables[key].solution_value))
            # for key in model.model.hot_capacity_variables.keys():
            #     print(str(key) + " solution value:" + str(model.model.hot_capacity_variables[key].solution_value))
            for key in model.model.source_variables.keys():
                print(str(key) + " solution value:" + str(model.model.source_variables[key].solution_value))
            for key in model.model.detector_variables.keys():
                print(str(key) + " solution value:" + str(model.model.detector_variables[key].solution_value))



