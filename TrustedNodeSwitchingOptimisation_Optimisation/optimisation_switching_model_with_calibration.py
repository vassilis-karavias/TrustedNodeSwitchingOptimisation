import cplex
import networkx as nx
import matplotlib.pyplot as plt
import scipy.optimize as opt
import optimisationmodel
import trusted_nodes_utils
import numpy as np
import time
from copy import deepcopy
import os
import pandas as pd
import optimisation_no_reverse_commodity
import LP_relaxation
import Heuristic_Model
import Heuristic_Genetic_Model
from optimisation_switching_model_trusted_node import *


"""
FILE FOR SWITCHED TRUSTED NODE INVESTIGATION
"""

def check_solvable(prob):
    """Checks solved cplex problem for timeout, infeasibility or optimal solution. Returns True when feasible solution obtained."""
    status = prob.solution.get_status()
    if status == 101 or status == 102: # proven optimal or optimal within tolerance
        return True
    elif status == 103:  # proven infeasible or Timeout
        return False
    elif status == 107:  # timed out
        print("Optimiser Timed out - assuming infeasible")
        return True
    else:
        print(f"Unknown Solution Status: {status} - assuming infeasible")
        return False



def split_source_detector_nodes(n_flow):
    source_nodes = {}
    detector_nodes = {}
    for key in n_flow:
        if key[-1] == "S":
            source_nodes[key] = n_flow[key]
        elif key[-1] == "D":
            detector_nodes[key] = n_flow[key]
    return source_nodes, detector_nodes

class Optimisation_Switching_Calibration_with_W_terms(Optimisation_Switching_No_Calibration):

    def __init__(self, prob, g, key_dict):
        super().__init__(prob = prob, g = g, key_dict = key_dict)

    def add_source_capacity_on_edge_constraint(self, Lambda):
        """
        adds constriant that ensures capacity cannot exceed capabilities of sources on the edge:

        \sum_{j \in N(i)} \frac{\sum_{k \in K} x'_{i,j}^{k}}{c_{i,j}} \leq N_i^{S}

        """
        source_terms = []
        neighbours = []
        for n in self.g.nodes:
            source_terms.append(f"N_{n}_S")
            neighbours.append(len(self.g.adj[n])* Lambda)
        variable_names = [f'x{i}_{j}_k{k[0]}_{k[1]}' for k in self.key_dict for i, j in list(self.g.edges)]
        variable_names_2 = [f'w{i}_{j}' for i, j in list(self.g.edges)]
        self.prob.variables.add(names=variable_names, types=[self.prob.variables.type.continuous] * len(variable_names))
        self.prob.variables.add(names=variable_names_2, types=[self.prob.variables.type.continuous] * len(variable_names_2))
        self.prob.variables.add(names=source_terms, types=[self.prob.variables.type.integer] * len(source_terms),
                           ub=neighbours)
        for i in self.g.nodes:
            flow_along_edge = []
            coeff = []
            for j in self.g.adj[i]:
                capacity = int(self.g.edges[[i, j]]["capacity"])
                if capacity > 0.0000001:
                    flow_along_edge.append(f"w{i}_{j}")
                    coeff.append(1 / capacity)
                else:
                    self.prob.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=[f"w{i}_{j}"], val=[1])],
                                                senses="E", rhs=[0.])
            flow_along_edge.append(f"N_{i}_S")
            coeff.append(-1)
            lin_expressions = [cplex.SparsePair(ind=flow_along_edge, val=coeff)]
            self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['L'], rhs=[0.])

    def add_detector_capacity_on_edge_constraint(self, Lambda):
        """
        adds constriant that ensures capacity cannot exceed capabilities of detectors on the edge:

        \sum_{j \in N(i)} \frac{\sum_{k \in K} x'_{j,i}^{k}}{c_{i,j}} \leq N_i^{D} for trusted nodes i \in T
        \sum_{j \in N(i)} \frac{\sum_{k \in K} x'_'{j,i}^{k}}{c_{i,j}} \leq 0 for i \in S
        """
        detector_terms = []
        neighbours = []
        for n in self.g.nodes:
            if self.g.nodes[n]["type"] == "T" or self.g.nodes[n]["type"] == "NodeType.T":
                detector_terms.append(f"N_{n}_D")
                neighbours.append(len(self.g.adj[n]) * Lambda)
        self.prob.variables.add(names=detector_terms, types=[self.prob.variables.type.integer] * len(detector_terms),
                           ub=neighbours)
        for i in self.g.nodes:
            if self.g.nodes[i]["type"] == "T" or self.g.nodes[i]["type"] == "NodeType.T":
                flow_along_edge = []
                coeff = []
                for j in self.g.adj[i]:
                    capacity = int(self.g.edges[[i, j]]["capacity"])
                    if capacity > 0.0000001:
                        flow_along_edge.append(f"w{j}_{i}")
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

    def add_untransformed_transformed_relationship(self, NT):
        """
        add the constraint that accounts for the calibration time:

        \sum_{k \in K} x_{i,j}^{k} = \sum_{k \in K} x'_{i,j}^{k} - \frac{N}{T}N_{j}^{D}
        """
        for i, j in self.g.edges:
            if self.g.nodes[j]["type"] == "T" or self.g.nodes[j]["type"] == "NodeType.T":
                ind = []
                val = []
                for k in self.key_dict:
                    ind.append(f"x{i}_{j}_k{k[0]}_{k[1]}")
                    val.append(1)
                ind.append(f"w{i}_{j}")
                val.append(-1)
                ind.append(f"N_{j}_D")
                val.append(NT)
                lin_expressions = [cplex.SparsePair(ind=ind, val=val)]
                self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['E'], rhs=[0.])
            else:
                ind = []
                val = []
                for k in self.key_dict:
                    ind.append(f"x{i}_{j}_k{k[0]}_{k[1]}")
                    val.append(1)
                ind.append(f"w{i}_{j}")
                val.append(-1)
                lin_expressions = [cplex.SparsePair(ind=ind, val=val)]
                self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['E'], rhs=[0.])

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
                neighbours = len(self.g.adj[i])
                lin_expressions = [cplex.SparsePair(ind=detector_terms, val=[1, -Lambda * neighbours])]
                self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['L'], rhs=[0.])
            for j in self.g.neighbors(i):
                if self.g.nodes[j]["type"] == "T" or self.g.nodes[j]["type"] == "NodeType.T":
                    terms = [f"N_{i}_S", f"N_{j}_D"]
                    lin_expressions = [cplex.SparsePair(ind=terms, val=[-1, 1])]
                    self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['L'], rhs=[0.])

    def initial_optimisation_cost_reduction(self, cmin, time_limit=1e5, cost_on_trusted_node=1,
                                            cost_detector=0.1, cost_source=0.01, NT=0.01, Lambda=100,
                                            average_connectivity=3.5):
        """
        set up and solve the problem for minimising the overall cost of the network
        """
        t_0 = time.time()
        print("Start Optimisation")
        prob = cplex.Cplex()
        self.add_source_capacity_on_edge_constraint(Lambda)
        self.add_detector_capacity_on_edge_constraint(Lambda)
        # add_capacity_constraint(prob, self.g, key_dict, Lambda=Lambda)
        self.add_source_detector_max_constraint(Lambda)
        # self.add_transformed_relationship_constant_calibration_time(NT, average_connectivity)
        self.add_untransformed_transformed_relationship(NT)
        self.add_flow_conservation_constraint()
        self.add_flow_requirement_constraint(cmin)
        self.add_flow_into_source()
        self.add_flow_into_untrusted_node()
        self.add_limited_flow_through_connection(cmin)
        # add_minimise_trusted_nodes_objective(prob, self.g)
        self.add_objective_function(cost_on_trusted_node, cost_detector, cost_source)
        self.prob.write("test_1.lp")
        self.prob.parameters.lpmethod.set(3)
        self.prob.parameters.mip.limits.cutpasses.set(1)
        self.prob.parameters.mip.strategy.probe.set(-1)
        self.prob.parameters.mip.strategy.variableselect.set(4)
        self.prob.parameters.mip.strategy.kappastats.set(1)
        self.prob.parameters.mip.tolerances.mipgap.set(float(0.01))
        # prob.parameters.simplex.limits.iterations = 50
        print(self.prob.parameters.get_changed())
        self.prob.parameters.timelimit.set(time_limit)
        t_1 = time.time()
        print("Time to set up problem: " + str(t_1 - t_0))
        self.prob.solve()
        t_2 = time.time()
        print("Time to solve problem: " + str(t_2 - t_1))
        print(f"The minimum Cost of Network: {self.prob.solution.get_objective_value()}")
        print(f"Number of Variables = {self.prob.variables.get_num()}")
        print(f"Number of Conditions = {self.prob.linear_constraints.get_num()}")
        sol_dict = optimisationmodel.create_sol_dict(self.prob)
        flow_dict, binary_dict, lambda_dict = optimisationmodel.split_sol_to_flow_delta_lambda(sol_dict)
        self.log_optimal_solution_to_problem(self.prob, save_file="solution_dictionary_column_format", graph_id=0)
        trusted_nodes = 0
        for key in binary_dict:
            trusted_nodes += binary_dict[key]
        print(f"Number of Trusted Nodes = {trusted_nodes}")
        return sol_dict, self.prob

class Optimisation_Switching_Calibration_no_W_terms(Optimisation_Switching_No_Calibration):

    def __init__(self, prob, g, key_dict):
        super().__init__(prob = prob, g = g, key_dict = key_dict)


    def add_source_capacity_on_edge_constraint(self, Lambda, NT):
        """
        adds constriant that ensures capacity cannot exceed capabilities of sources on the edge:
        \sum_{j \in N(i)} \frac{\sum_{k \in K} x'_{i,j}^{k}}{c_{i,j}} \leq N_i^{S}

        """
        source_terms = []
        detector_terms = []
        neighbours_source = []
        neighbours_det = []
        for n in self.g.nodes:
            source_terms.append(f"N_{n}_S")
            neighbours_source.append(len(self.g.adj[n]) * Lambda)
            if self.g.nodes[n]["type"] == "T" or self.g.nodes[n]["type"] == "NodeType.T":
                detector_terms.append(f"N_{n}_D")
                neighbours_det.append(len(self.g.adj[n]) * Lambda)
        variable_names = [f'x{i}_{j}_k{k[0]}_{k[1]}' for k in self.key_dict for i, j in list(self.g.edges)]
        self.prob.variables.add(names=variable_names, types=[self.prob.variables.type.continuous] * len(variable_names))
        self.prob.variables.add(names=source_terms, types=[self.prob.variables.type.integer] * len(source_terms),
                           ub=neighbours_source)
        self.prob.variables.add(names=detector_terms, types=[self.prob.variables.type.integer] * len(detector_terms),
                           ub=neighbours_det)
        for i in self.g.nodes:
            flow_along_edge = []
            coeff = []
            for j in self.g.adj[i]:
                capacity = int(self.g.edges[[i, j]]["capacity"])
                if capacity > 0.0000001:
                    for k in self.key_dict:
                        flow_along_edge.append(f"x{i}_{j}_k{k[0]}_{k[1]}")
                        coeff.append(1 / capacity)
                    if self.g.nodes[j]["type"] == "T" or self.g.nodes[j]["type"] == "NodeType.T":
                        flow_along_edge.append(f"N_{j}_D")
                        coeff.append(NT / capacity)
                else:
                    for k in self.key_dict:
                        self.prob.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=[f"x{i}_{j}_k{k[0]}_{k[1]}"], val=[1])],
                                                senses="E", rhs=[0.])
            flow_along_edge.append(f"N_{i}_S")
            coeff.append(-1)
            lin_expressions = [cplex.SparsePair(ind=flow_along_edge, val=coeff)]
            self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['L'], rhs=[0.])

    def add_detector_capacity_on_edge_constraint(self, NT):
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
                        if self.g.nodes[i]["type"] == "T" or self.g.nodes[i]["type"] == "NodeType.T":
                            total_nj_coeff += NT / capacity
                    else:
                        flow_along_zero_edge = []
                        coeff_along_zero_edge = []
                        for k in self.key_dict:
                            flow_along_edge.append(f"x{j}_{i}_k{k[0]}_{k[1]}")
                            flow_along_zero_edge.append(f"x{j}_{i}_k{k[0]}_{k[1]}")
                            coeff.append(0.0)
                            coeff_along_zero_edge.append(1.0)
                        lin_expressions = [cplex.SparsePair(ind=flow_along_zero_edge, val=coeff_along_zero_edge)]
                        self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['E'], rhs=[0.])
                flow_along_edge.append(f"N_{i}_D")
                coeff.append(-1 + total_nj_coeff)
                lin_expressions = [cplex.SparsePair(ind=flow_along_edge, val=coeff)]
                self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['L'], rhs=[0.])
            else:
                # flow into source nodes/untrusted nodes is 0 as there are no detectors
                flow_along_edge = []
                coeff = []
                for j in self.g.adj[i]:
                    for k in self.key_dict:
                        flow_along_edge.append(f"x{j}_{i}_k{k[0]}_{k[1]}")
                        coeff.append(1)
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
                lin_expressions = [cplex.SparsePair(ind=detector_terms, val=[1, -Lambda * len(self.g.adj[i])])]
                self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['L'], rhs=[0.])
            for j in self.g.neighbors(i):
                if self.g.nodes[j]["type"] == "T" or self.g.nodes[j]["type"] == "NodeType.T":
                    terms = [f"N_{i}_S", f"N_{j}_D"]
                    lin_expressions = [cplex.SparsePair(ind=terms, val=[-1, 1])]
                    self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['L'], rhs=[0.])

    def initial_optimisation_cost_reduction(self, cmin, time_limit=1e5, cost_on_trusted_node=1,
                                                       cost_detector=0.1, cost_source=0.01, NT=0.01, Lambda=100):
        """
        set up and solve the problem for minimising the overall cost of the network
        """
        t_0 = time.time()
        print("Start Optimisation")
        prob = cplex.Cplex()

        self.add_source_capacity_on_edge_constraint(Lambda, NT)
        self.add_detector_capacity_on_edge_constraint(NT)
        # add_capacity_constraint(prob, self.g, self.key_dict, Lambda=Lambda)
        self.add_source_detector_max_constraint(Lambda)
        self.add_flow_conservation_constraint()
        self.add_flow_requirement_constraint(cmin)
        self.add_flow_into_source()
        self.add_flow_into_untrusted_node()
        self.add_limited_flow_through_connection(cmin)
        # add_minimise_trusted_nodes_objective(prob, self.g)
        self.add_objective_function(cost_on_trusted_node, cost_detector, cost_source)
        self.prob.write("test_1.lp")
        self.prob.parameters.lpmethod.set(3)
        self.prob.parameters.mip.limits.cutpasses.set(1)
        self.prob.parameters.mip.strategy.probe.set(-1)
        self.prob.parameters.mip.strategy.variableselect.set(4)
        self.prob.parameters.mip.strategy.kappastats.set(1)
        self.prob.parameters.mip.tolerances.mipgap.set(float(0.001))
        # prob.parameters.simplex.limits.iterations = 50
        print(self.prob.parameters.get_changed())
        self.prob.parameters.timelimit.set(time_limit)
        t_1 = time.time()
        print("Time to set up problem: " + str(t_1 - t_0))
        self.prob.solve()
        t_2 = time.time()
        print("Time to solve problem: " + str(t_2 - t_1))
        print(f"The minimum Cost of Network: {self.prob.solution.get_objective_value()}")
        print(f"Number of Variables = {self.prob.variables.get_num()}")
        print(f"Number of Conditions = {self.prob.linear_constraints.get_num()}")
        sol_dict = optimisationmodel.create_sol_dict(self.prob)
        flow_dict, binary_dict, lambda_dict = optimisationmodel.split_sol_to_flow_delta_lambda(sol_dict)
        self.log_optimal_solution_to_problem(self.prob, save_file="solution_dictionary_column_format", graph_id=0)
        trusted_nodes = 0
        for key in binary_dict:
            trusted_nodes += binary_dict[key]
        print(f"Number of Trusted Nodes = {trusted_nodes}")
        return sol_dict, self.prob


class Optimisation_Switching_Calibration_fixed_frac_calibration_time(Optimisation_Switching_No_Calibration):


    def __init__(self, prob, g, key_dict):
        super().__init__(prob = prob, g = g, key_dict = key_dict)

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
                lin_expressions = [cplex.SparsePair(ind=detector_terms, val=[1, -Lambda * len(self.g.adj[i])])]
                self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['L'], rhs=[0.])
            if self.g.nodes[i]["type"] == "T" or self.g.nodes[i]["type"] == "NodeType.T":
                terms = [f"N_{i}_S", f"delta_{i}"]
                lin_expressions = [cplex.SparsePair(ind=terms, val=[1, -Lambda* len(self.g.adj[i])])]
                self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['L'], rhs=[0.])

    def add_source_capacity_on_edge_constraint(self, Lambda, f_switch):
        """
        adds constriant that ensures capacity cannot exceed capabilities of sources on the edge:

        \sum_{j \in N(i)} \frac{\sum_{k \in K} x'_{i,j}^{k}}{c_{i,j}} \leq N_i^{S}

        """
        source_terms = []
        detector_terms = []
        neighbours_source =[]
        neighbours_det = []
        for n in self.g.nodes:
            source_terms.append(f"N_{n}_S")
            neighbours_source.append(len(self.g.adj[n]) * Lambda)
            if self.g.nodes[n]["type"] == "T" or  self.g.nodes[n]["type"]== "NodeType.T":
                detector_terms.append(f"N_{n}_D")
                neighbours_det.append(len(self.g.adj[n]) * Lambda)
        variable_names = [f'x{i}_{j}_k{k[0]}_{k[1]}' for k in self.key_dict for i, j in list(self.g.edges)]
        self.prob.variables.add(names=variable_names, types=[self.prob.variables.type.continuous] * len(variable_names))
        self.prob.variables.add(names=source_terms, types=[self.prob.variables.type.integer] * len(source_terms),
                           ub=neighbours_source)
        self.prob.variables.add(names=detector_terms, types=[self.prob.variables.type.integer] * len(detector_terms),
                           ub=neighbours_det)
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
                        self.prob.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=[f"x{i}_{j}_k{k[0]}_{k[1]}"], val=[1])],
                                                senses="E", rhs=[0.])
            flow_along_edge.append(f"N_{i}_S")
            coeff.append(-(1-f_switch))
            lin_expressions = [cplex.SparsePair(ind=flow_along_edge, val=coeff)]
            self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['L'], rhs=[0.])

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
                        flow_along_zero_edge = []
                        coeff_along_zero_edge = []
                        for k in self.key_dict:
                            flow_along_edge.append(f"x{j}_{i}_k{k[0]}_{k[1]}")
                            flow_along_zero_edge.append(f"x{j}_{i}_k{k[0]}_{k[1]}")
                            coeff.append(0.0)
                            coeff_along_zero_edge.append(1.0)
                        lin_expressions = [cplex.SparsePair(ind=flow_along_zero_edge, val=coeff_along_zero_edge)]
                        self.prob.linear_constraints.add(lin_expr=lin_expressions, senses=['E'], rhs=[0.])
                flow_along_edge.append(f"N_{i}_D")
                coeff.append(-(1-f_switch))
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


    def initial_optimisation_cost_reduction(self, cmin, time_limit=1e5, cost_on_trusted_node=1,
                                                       cost_detector=0.1, cost_source=0.01, f_switch=0.1, Lambda=100):
        """
        set up and solve the problem for minimising the overall cost of the network
        """
        t_0 = time.time()
        print("Start Optimisation")
        prob = cplex.Cplex()

        self.add_source_capacity_on_edge_constraint(Lambda, f_switch)
        self.add_detector_capacity_on_edge_constraint(f_switch)
        # add_capacity_constraint(prob, self.g, self.key_dict, Lambda=Lambda)
        self.add_source_detector_max_constraint(Lambda)
        self.add_flow_conservation_constraint()
        self.add_flow_requirement_constraint(cmin)
        self.add_flow_into_source()
        # self.add_flow_into_untrusted_node()
        self.add_limited_flow_through_connection(cmin)
        # add_minimise_trusted_nodes_objective(prob, self.g)
        self.add_objective_function(cost_on_trusted_node, cost_detector, cost_source)
        self.prob.write("test_1.lp")
        self.prob.parameters.lpmethod.set(3)
        self.prob.parameters.mip.limits.cutpasses.set(1)
        self.prob.parameters.mip.strategy.probe.set(-1)
        self.prob.parameters.mip.strategy.variableselect.set(4)
        self.prob.parameters.mip.strategy.kappastats.set(1)
        self.prob.parameters.mip.tolerances.mipgap.set(float(0.001))
        # prob.parameters.simplex.limits.iterations = 50
        print(self.prob.parameters.get_changed())
        self.prob.parameters.timelimit.set(time_limit)
        t_1 = time.time()
        print("Time to set up problem: " + str(t_1 - t_0))
        self.prob.solve()
        t_2 = time.time()
        print("Time to solve problem: " + str(t_2 - t_1))
        print(f"The minimum Cost of Network: {self.prob.solution.get_objective_value()}")
        print(f"Number of Variables = {self.prob.variables.get_num()}")
        print(f"Number of Conditions = {self.prob.linear_constraints.get_num()}")
        sol_dict = optimisationmodel.create_sol_dict(self.prob)
        flow_dict, binary_dict, lambda_dict = optimisationmodel.split_sol_to_flow_delta_lambda(sol_dict)
        self.log_optimal_solution_to_problem(self.prob, save_file="solution_dictionary_column_format", graph_id=0)
        trusted_nodes = 0
        for key in binary_dict:
            trusted_nodes += binary_dict[key]
        print(f"Number of Trusted Nodes = {trusted_nodes}")
        return sol_dict, self.prob, t_2 - t_1





def time_variation_analysis(cap_needed_location, edge_data_location, node_type_location, position_node_file, position_edge_file,
                            cmin, time_limit=1e5, cost_on_trusted_node=1,cost_detector=0.1, cost_source=0.01, f_switch=0.1, Lambda=100):
    key_dict, g, position_graphs = trusted_nodes_utils.import_problem_from_files_flexibility_multiple_graphs_position_graph(
        cap_needed_location, edge_data_location, node_type_location,
        position_node_file=position_node_file, position_edge_file=position_edge_file)
    time_taken = {}
    for key in g.keys():
        try:
            # key_dict_temp = trusted_nodes_utils.make_key_dict_bidirectional(key_dict[key])
            prob = cplex.Cplex()
            optim = optimisation_no_reverse_commodity.Optimisation_Problem_No_Switching_No_Trusted_Nodes(prob = prob, g = g[key], key_dict = key_dict[key])
            sol_dict, prob, time_taken = optim.initial_optimisation_cost_reduction(time_limit = time_limit, cost_connection= cost_source + cost_detector, Lambda = Lambda, cmin = cmin)


            # optim = Optimisation_Switching_Calibration_fixed_frac_calibration_time(prob=prob, g=g[key],
            #                                                                        key_dict=key_dict_temp)
            # sol_dict, prob, time_taken = optim.initial_optimisation_cost_reduction(cmin = cmin, time_limit=time_limit, cost_on_trusted_node=cost_on_trusted_node,
            #                                            cost_detector=cost_detector, cost_source=cost_source, f_switch=f_switch, Lambda=Lambda)
            number_nodes = g[key].number_of_nodes()
            if number_nodes in time_taken.keys():
                time_taken[number_nodes].append(time_taken)
            else:
                time_taken[number_nodes] = [time_taken]
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
    # fit of exponential curve to initial points
    # x_exponential = x
    # y_exponential = y
    # popt, pcov = curve_fit(optimisation_switched.exponential_fit, x_exponential, y_exponential)
    # x_exponential = np.arange(x[0], x[-1], 0.1)
    # y_exponential = [optimisation_switched.exponential_fit(a, popt[0], popt[1], popt[2]) for a in x_exponential]
    # # fit of polynomial
    # popt_poly, pcov_poly = curve_fit(optimisation_switched.polynomial, x, y)
    # y_poly = [optimisation_switched.polynomial(a, popt_poly[0], popt_poly[1]) for a in x_exponential]

    plt.errorbar(x, y, yerr=yerr, color="r")
    # plt.plot(x_exponential[:int(np.ceil(len(x_exponential)/1.25))], y_exponential[:int(np.ceil(len(x_exponential)/1.25))], color = "b")
    # plt.plot(x_exponential, y_poly, color = "k")
    plt.xlabel("Number of Nodes in Graph", fontsize=10)
    plt.ylabel("Time/s", fontsize=10)
    # plt.legend(loc='upper right', fontsize='medium')
    plt.savefig("time_investigation_mesh_topology.png")
    plt.show()


def plot_cost_with_increasing_cmin(cap_needed_location, edge_data_location, node_type_location, position_node_file, position_edge_file,
                         time_limit=1e5, cost_on_trusted_node=1,cost_detector=0.1, cost_source=0.01, f_switch=0.1, Lambda=100, data_storage_location_keep_each_loop = None):
    key_dict, g, position_graphs = trusted_nodes_utils.import_problem_from_files_flexibility_multiple_graphs_position_graph(
        cap_needed_location, edge_data_location, node_type_location,
        position_node_file=position_node_file, position_edge_file=position_edge_file)
    if data_storage_location_keep_each_loop != None:
        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            last_ratio_done = last_row_explored["cmin"]
            dataframe_of_cmin_done = plot_information[plot_information["cmin"] == last_ratio_done.iloc[0]]
            current_key = last_row_explored["Graph key"].iloc[0]
            cmin_current = last_ratio_done.iloc[0]
        else:
            cmin_current = None
            current_key = None
            dictionary_fieldnames = ["cmin", "Graph key", "objective_value"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        cmin_current = None
        current_key = None
    no_soln_set = []
    objective_value_at_cmin_1000 = {}
    objective_values_cmin = {}
    lambda_values = {}
    for cmin in np.arange(start=100, stop=5000, step=100):
        if cmin_current != None:
            if cmin != cmin_current:
                continue
            else:
                cmin_current = None
        for key in g.keys():
            if key not in no_soln_set:
                if current_key != key and current_key != None:
                    continue
                elif current_key == key:
                    current_key = None
                try:
                    key_dict_temp = trusted_nodes_utils.make_key_dict_bidirectional(key_dict[key])
                    prob = cplex.Cplex()
                    optim = Optimisation_Switching_Calibration_fixed_frac_calibration_time(prob=prob, g=g[key],
                                                                                           key_dict=key_dict_temp)
                    sol_dict, prob, time_taken = optim.initial_optimisation_cost_reduction(cmin=cmin,
                                                                                           time_limit=time_limit,
                                                                                           cost_on_trusted_node=cost_on_trusted_node,
                                                                                           cost_detector=cost_detector,
                                                                                           cost_source=cost_source,
                                                                                           f_switch=f_switch,
                                                                                           Lambda=Lambda)
                except:
                    no_soln_set.append(key)
                    continue
                if check_solvable(prob):
                    objective_value = prob.solution.get_objective_value()
                    if abs(cmin - 1000) < 0.0001:
                        objective_value_at_cmin_1000[key] = objective_value
                    if cmin not in objective_values_cmin.keys():
                        objective_values_cmin[cmin] = [(objective_value, key)]
                    else:
                        objective_values_cmin[cmin].append((objective_value, key))
                    # print results for on nodes:
                    flow_dict, binary_dict, lambda_dict =  optimisationmodel.split_sol_to_flow_delta_lambda(sol_dict)
                    for lambda_key in lambda_dict.keys():
                        if lambda_key not in lambda_values.keys():
                            lambda_values[lambda_key] = {cmin: lambda_dict[lambda_key]}
                        else:
                            lambda_values[lambda_key][cmin] = lambda_dict[lambda_key]
                    print("Results for cmin:" + str(cmin))
                    for value in lambda_dict:
                        print("Value for " + str(value) + ": " + str(lambda_dict[value]))
                    for binary in binary_dict:
                        print("Value for " + str(binary) + ": " + str(binary_dict[binary]))
                    if data_storage_location_keep_each_loop != None:
                        dictionary = [
                            {"cmin": cmin, "Graph key": key, "objective_value": objective_value}]
                        dictionary_fieldnames = ["cmin", "Graph key", "objective_value"]
                        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
                            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writerows(dictionary)
                        else:
                            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writeheader()
                                writer.writerows(dictionary)
    if data_storage_location_keep_each_loop != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
        for index, row in plot_information.iterrows():
            if row["cmin"] not in objective_values_cmin.keys():
                objective_values_cmin[row["cmin"]] = [(row["objective_value"], row["Graph key"])]
            else:
                objective_values_cmin[row["cmin"]].append((row["objective_value"], row["Graph key"]))
        objective_value_at_cmin_1000 = {}
        for obj_value, key in objective_values_cmin[1000.0]:
            objective_value_at_cmin_1000[key] = obj_value
    objective_values_cmin = dict(sorted(objective_values_cmin.items()))
    objective_values = {}
    for cmin in objective_values_cmin.keys():
        for objective_value, key in objective_values_cmin[cmin]:
            if key in objective_value_at_cmin_1000.keys():
                if cmin not in objective_values.keys():
                    objective_values[cmin] = [objective_value / objective_value_at_cmin_1000[key]]
                else:
                    objective_values[cmin].append(objective_value / objective_value_at_cmin_1000[key])

    for key in lambda_values.keys():
        plt.plot(list(lambda_values[key].keys()), list(lambda_values[key].values()))
    plt.xlabel("Minimum Capacity Necessary (cmin)", fontsize=10)
    plt.ylabel("Number of Detectors On Site", fontsize=10)
    plt.show()
    mean_objectives = {}
    std_objectives = {}
    for key in objective_values.keys():
        mean_objectives[key] = np.mean(objective_values[key])
        std_objectives[key] = np.std(objective_values[key])
    mean_differences = []
    std_differences = []
    # topologies
    x = []
    for key in mean_objectives.keys():
        mean_differences.append(mean_objectives[key])
        std_differences.append(std_objectives[key])
        x.append(key)
    plt.errorbar(x, mean_differences, yerr=std_differences, color="r")
    plt.xlabel("Minimum Capacity Necessary (cmin)", fontsize=10)
    plt.ylabel("Cost of Network/Cost of Network at cmin = 1000", fontsize=10)
    plt.savefig("cmin_mesh_topology_single_graph_single_graph_3.png")
    plt.show()


def f_switch_parameter_sweep(cap_needed_location, edge_data_location, node_type_location, position_node_file, position_edge_file,
                            cap_needed_location_no_switch, edge_data_location_no_switch, node_type_location_no_switch, position_node_file_no_switch, position_edge_file_no_switch,
                         cmin, time_limit=1e5, cost_on_trusted_node=1,cost_detector=0.1, cost_source=0.01, Lambda=100, data_storage_location_keep_each_loop = None, data_storage_location_keep_each_loop_no_switch = None):
    key_dict, g, position_graphs = trusted_nodes_utils.import_problem_from_files_flexibility_multiple_graphs_position_graph(
        cap_needed_location, edge_data_location, node_type_location,
        position_node_file=position_node_file, position_edge_file=position_edge_file)
    key_dict_no_switching, g_no_switching, position_graphs_no_switching = trusted_nodes_utils.import_problem_from_files_flexibility_multiple_graphs_position_graph(
        cap_needed_location_no_switch, edge_data_location_no_switch, node_type_location_no_switch,
        position_node_file=position_node_file_no_switch, position_edge_file=position_edge_file_no_switch)

    # graphs = import_graph_structure(node_information=graph_node_data_file, edge_information=graph_edge_data_file)
    if data_storage_location_keep_each_loop != None:
        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            last_ratio_done = last_row_explored["f_switch"]
            dataframe_of_fswitch_done = plot_information[plot_information["f_switch"] == last_ratio_done.iloc[0]]
            current_key = last_row_explored["Graph key"].iloc[0]
            fswitch_current = last_ratio_done.iloc[0]
        else:
            fswitch_current = None
            current_key = None
            dictionary_fieldnames = ["f_switch", "Graph key", "objective_value"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        fswitch_current = None
        current_key = None

    no_soln_set = []
    objective_value_at_frac_switch_01 = {}
    objective_values_frac_switch= {}
    for frac_switch in np.arange(start=0.0, stop=0.92, step=0.02):
        if fswitch_current != None:
            if frac_switch != fswitch_current:
                continue
            else:
                fswitch_current = None
        for key in g.keys():
            if current_key != None:
                if current_key == key:
                    current_key = None
                    continue
                else:
                    continue
            if key not in no_soln_set:
                try:
                    key_dict_temp = trusted_nodes_utils.make_key_dict_bidirectional(key_dict[key])
                    prob = cplex.Cplex()
                    optim = Optimisation_Switching_Calibration_fixed_frac_calibration_time(prob=prob, g=g[key],
                                                                                           key_dict=key_dict_temp)
                    sol_dict, prob, time_taken = optim.initial_optimisation_cost_reduction(cmin=cmin,
                                                                                           time_limit=time_limit,
                                                                                           cost_on_trusted_node=cost_on_trusted_node,
                                                                                           cost_detector=cost_detector,
                                                                                           cost_source=cost_source,
                                                                                           f_switch=frac_switch,
                                                                                           Lambda=Lambda)

                except:
                    no_soln_set.append(key)
                    continue
                if check_solvable(prob):
                    objective_value = prob.solution.get_objective_value()
                    if abs(frac_switch - 0.1) < 0.0001:
                        objective_value_at_frac_switch_01[key] = objective_value
                    if frac_switch not in objective_values_frac_switch.keys():
                        objective_values_frac_switch[frac_switch] = [(objective_value,key)]
                    else:
                        objective_values_frac_switch[frac_switch].append((objective_value,key))
                if data_storage_location_keep_each_loop != None:
                    dictionary = [
                        {"f_switch": frac_switch, "Graph key": key, "objective_value": objective_value}]
                    dictionary_fieldnames = ["f_switch", "Graph key", "objective_value"]
                    if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
                        with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writerows(dictionary)
                    else:
                        with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writeheader()
                            writer.writerows(dictionary)
    no_soln_set = []
    if data_storage_location_keep_each_loop_no_switch != None:
        if os.path.isfile(data_storage_location_keep_each_loop_no_switch + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop_no_switch + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            current_key = last_row_explored["Graph key"].iloc[0]
        else:
            current_key = None
            dictionary_fieldnames = ["Graph key", "objective_value"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_key = None

    for key in g_no_switching.keys():
        if current_key != None:
            if current_key == key:
                current_key = None
                continue
            else:
                continue
        if key not in no_soln_set:
            try:
                sol_dict, prob = optimisation_no_reverse_commodity.initial_optimisation_cost_reduction(g = g_no_switching[key],
                                        key_dict = key_dict_no_switching[key], cmin = cmin, time_limit = time_limit, cost_node = cost_on_trusted_node, cost_connection= cost_detector + cost_source, Lambda = Lambda)
            except:
                no_soln_set.append(key)
                continue
            if check_solvable(prob):
                objective_value = prob.solution.get_objective_value()
            if data_storage_location_keep_each_loop_no_switch != None:
                dictionary = [
                    {"Graph key": key, "objective_value": objective_value}]
                dictionary_fieldnames = ["Graph key", "objective_value"]
                if os.path.isfile(data_storage_location_keep_each_loop_no_switch + '.csv'):
                    with open(data_storage_location_keep_each_loop_no_switch + '.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writerows(dictionary)
                else:
                    with open(data_storage_location_keep_each_loop_no_switch + '.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writeheader()
                        writer.writerows(dictionary)
    if data_storage_location_keep_each_loop != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
        for index, row in plot_information.iterrows():
            if row["f_switch"] not in objective_values_frac_switch.keys():
                objective_values_frac_switch[row["f_switch"]] = {row["Graph key"] :row["objective_value"]}
            else:
                objective_values_frac_switch[row["f_switch"]][row["Graph key"]] = row["objective_value"]
        objective_value_at_frac_switch_01 = {}
        for key in objective_values_frac_switch[0.1]:
            objective_value_at_frac_switch_01[key] = objective_values_frac_switch[0.1][key]
        if data_storage_location_keep_each_loop_no_switch != None:
            objective_values_no_switch = {}
            plot_information_no_switching = pd.read_csv(data_storage_location_keep_each_loop_no_switch + ".csv")
            for index, row in plot_information_no_switching.iterrows():
                if row["Graph key"] not in objective_values_no_switch.keys():
                    objective_values_no_switch[row["Graph key"]] = [row["objective_value"]]
                else:
                    objective_values_no_switch[row["Graph key"]].append(row["objective_value"])
            objective_values = {}
            for frac_switch in objective_values_frac_switch.keys():
                for key in objective_values_frac_switch[frac_switch].keys():
                    if frac_switch not in objective_values.keys():
                        objective_values[frac_switch] = [objective_values_frac_switch[frac_switch][key] / objective_values_no_switch[key]]
                    else:
                        objective_values[frac_switch].append(objective_values_frac_switch[frac_switch][key] / objective_values_no_switch[key])
            mean_objectives = {}
            std_objectives = {}
            for key in objective_values.keys():
                mean_objectives[key] = np.mean([objective_values[key][12]])
                std_objectives[key] = np.std([objective_values[key][12]])
            mean_differences = []
            std_differences = []
            # topologies
            x = []
            for key in mean_objectives.keys():
                mean_differences.append(mean_objectives[key])
                std_differences.append(std_objectives[key])
                x.append(key)
            plt.errorbar(x, mean_differences, yerr=std_differences, color="r", capsize= 0, label = "Normalised Cost of Network")
            plt.axhline(y=1, color='b', linestyle='-', label = "Cost of Network without Switching")
            plt.legend()
            plt.xlabel("Fraction of time calibrating", fontsize=10)
            plt.ylabel("Cost of Network/Cost of Network without Switching", fontsize=10)
            # plt.legend(loc='upper right', fontsize='medium')
            plt.savefig("frac_switch_mesh_topology_single_graph_one_graph_9.png")
            plt.show()
        else:
            objective_values = {}
            for frac_switch in objective_values_frac_switch.keys():
                for key in objective_values_frac_switch[frac_switch]:
                    if frac_switch not in objective_values.keys():
                        objective_values[frac_switch] = [objective_values_frac_switch[frac_switch][key] / objective_value_at_frac_switch_01[key]]
                    else:
                        objective_values[frac_switch].append(objective_values_frac_switch[frac_switch][key] / objective_value_at_frac_switch_01[key])
            mean_objectives = {}
            std_objectives = {}
            for key in objective_values.keys():
                mean_objectives[key] = np.mean(objective_values[key])
                std_objectives[key] = np.std(objective_values[key])
            mean_differences = []
            std_differences = []
            # topologies
            x = []
            for key in mean_objectives.keys():
                mean_differences.append(mean_objectives[key])
                std_differences.append(std_objectives[key])
                x.append(key)
            plt.errorbar(x, mean_differences, yerr=std_differences, color="r", capsize= 0)
            plt.xlabel("Fraction of time calibrating", fontsize=10)
            plt.ylabel("Cost of Network/Cost of Network at 0.1 Fraction of Time Calibrating" , fontsize=10)
            # plt.legend(loc='upper right', fontsize='medium')
            plt.savefig("frac_switch_mesh_topology_single_graph_one_graph_2.png")
            plt.show()


def switch_loss_cost_comparison(cmin, f_switch = 0.1, time_limit=1e5, cost_on_trusted_node=1,cost_detector=0.1, cost_source=0.01, Lambda=100, data_storage_location_keep_each_loop = None, data_storage_location_keep_each_loop_no_switch = None):
    objective_values_switch_loss = {}
    objective_value_at_1_dB = {}
    no_solution_list = []
    if data_storage_location_keep_each_loop != None:
        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            last_ratio_done = last_row_explored["db_switch"]
            dataframe_of_fswitch_done = plot_information[plot_information["db_switch"] == last_ratio_done.iloc[0]]
            current_key = last_row_explored["Graph key"].iloc[0]
            dbswitch_current = last_ratio_done.iloc[0]
        else:
            dbswitch_current = None
            current_key = None
            dictionary_fieldnames = ["db_switch", "Graph key", "objective_value"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        dbswitch_current = None
        current_key = None

    for switch_loss in np.arange(start=0.5, stop=6, step=0.25):
        if dbswitch_current != None:
            if switch_loss != dbswitch_current:
                continue
            else:
                dbswitch_current = None
        key_dict, g, position_graphs = trusted_nodes_utils.import_problem_from_files_flexibility_multiple_graphs_position_graph(
            cap_needed_location= f"11_nodes_mesh_topology_35_capacity_needed_{round(switch_loss,2)}", edge_data_location = f"11_nodes_mesh_topology_35_edge_data_{round(switch_loss,2)}", node_type_location = f"11_nodes_mesh_topology_35_node_data_{round(switch_loss,2)}",
            position_node_file=f"11_nodes_mesh_topology_35_position_nodes", position_edge_file=f"11_nodes_mesh_topology_35_position_edges")
        for key in g.keys():
            if current_key != None:
                if current_key == key:
                    current_key = None
                    continue
                else:
                    continue
            if key not in no_solution_list:
                try:
                    key_dict_temp = trusted_nodes_utils.make_key_dict_bidirectional(key_dict[key])
                    prob = cplex.Cplex()
                    optim = Optimisation_Switching_Calibration_fixed_frac_calibration_time(prob=prob, g=g[key],
                                                                                           key_dict=key_dict_temp)
                    sol_dict, prob, time_taken = optim.initial_optimisation_cost_reduction(cmin=cmin,
                                                                                           time_limit=time_limit,
                                                                                           cost_on_trusted_node=cost_on_trusted_node,
                                                                                           cost_detector=cost_detector,
                                                                                           cost_source=cost_source,
                                                                                           f_switch=f_switch,
                                                                                           Lambda=Lambda)
                except:
                    no_solution_list.append(key)
                    continue
                if check_solvable(prob):
                    objective_value = prob.solution.get_objective_value()
                    if switch_loss == 1:
                        objective_value_at_1_dB[key] = objective_value
                    if switch_loss not in objective_values_switch_loss.keys():
                        objective_values_switch_loss[switch_loss] = {key: objective_value}
                    else:
                        objective_values_switch_loss[switch_loss][key] = objective_value

                    if data_storage_location_keep_each_loop != None:
                        dictionary = [
                            {"db_switch": round(switch_loss,2), "Graph key": key, "objective_value": objective_value}]
                        dictionary_fieldnames = ["db_switch", "Graph key", "objective_value"]
                        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
                            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writerows(dictionary)
                        else:
                            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                                writer.writeheader()
                                writer.writerows(dictionary)
                    no_soln_set = []
    if data_storage_location_keep_each_loop_no_switch != None:
        if os.path.isfile(data_storage_location_keep_each_loop_no_switch + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop_no_switch + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            current_key = last_row_explored["Graph key"].iloc[0]
        else:
            current_key = None
            dictionary_fieldnames = ["Graph key", "objective_value"]
            with open(data_storage_location_keep_each_loop_no_switch + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_key = None
    key_dict_no_switching, g_no_switching, position_graphs = trusted_nodes_utils.import_problem_from_files_flexibility_multiple_graphs_position_graph(
        cap_needed_location=f"11_nodes_mesh_topology_35_capacity_needed_no_switch",
        edge_data_location=f"11_nodes_mesh_topology_35_edge_data_no_switch",
        node_type_location=f"11_nodes_mesh_topology_35_node_data_no_switch",
        position_node_file=f"11_nodes_mesh_topology_35_position_nodes",
        position_edge_file=f"11_nodes_mesh_topology_35_position_edges")
    no_solution_list =[]
    objective_value_no_switch = {}
    for key in g_no_switching.keys():
        if current_key != None:
            if current_key == key:
                current_key = None
                continue
            else:
                continue
        if key not in no_solution_list:
            try:
                sol_dict, prob = optimisation_no_reverse_commodity.initial_optimisation_cost_reduction(
                                                        g=g_no_switching[key],
                                                        key_dict=key_dict[key], cmin=cmin, time_limit=time_limit, cost_node=cost_on_trusted_node,
                                                        cost_connection=cost_detector + cost_source, Lambda=Lambda)
            except:
                no_solution_list.append(key)
                continue
            if check_solvable(prob):
                objective_value = prob.solution.get_objective_value()
                objective_value_no_switch[key] = objective_value
                if data_storage_location_keep_each_loop_no_switch != None:
                    dictionary = [
                        {"Graph key": key, "objective_value": objective_value}]
                    dictionary_fieldnames = ["Graph key", "objective_value"]
                    if os.path.isfile(data_storage_location_keep_each_loop_no_switch + '.csv'):
                        with open(data_storage_location_keep_each_loop_no_switch + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writerows(dictionary)
                    else:
                        with open(data_storage_location_keep_each_loop_no_switch + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writeheader()
                            writer.writerows(dictionary)
    if data_storage_location_keep_each_loop != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
        for index, row in plot_information.iterrows():
            if row["db_switch"] not in objective_values_switch_loss.keys():
                objective_values_switch_loss[row["db_switch"]] = {row["Graph key"] :row["objective_value"]}
            else:
                objective_values_switch_loss[row["db_switch"]][row["Graph key"]] = row["objective_value"]
        if data_storage_location_keep_each_loop_no_switch != None:
            objective_values_no_switch = {}
            plot_information_no_switching = pd.read_csv(data_storage_location_keep_each_loop_no_switch + ".csv")
            for index, row in plot_information_no_switching.iterrows():
                if row["Graph key"] not in objective_values_no_switch.keys():
                    objective_values_no_switch[row["Graph key"]] = [row["objective_value"]]
                else:
                    objective_values_no_switch[row["Graph key"]].append(row["objective_value"])
            objective_values = {}
            for db_switch in objective_values_switch_loss.keys():
                for key in objective_values_switch_loss[db_switch].keys():
                    if db_switch not in objective_values.keys():
                        objective_values[db_switch] = [objective_values_switch_loss[db_switch][key] / objective_values_no_switch[key]]
                    else:
                        objective_values[db_switch].append(objective_values_switch_loss[db_switch][key] / objective_values_no_switch[key])
    mean_objectives = {}
    std_objectives = {}
    for key in objective_values.keys():
        mean_objectives[key] = np.mean(objective_values[key])
        std_objectives[key] = np.std(objective_values[key])
    mean_differences = []
    std_differences = []
    # topologies
    x = []
    for key in mean_objectives.keys():
        mean_differences.append(mean_objectives[key])
        std_differences.append(std_objectives[key])
        x.append(key)
    plt.errorbar(x, mean_differences, yerr=std_differences, color="r", label = "Normalised Cost of Network")
    plt.axhline(y=1, color='b', linestyle='-', label="Cost of Network without Switching")
    plt.legend()
    plt.xlabel("Switch dB Loss", fontsize=10)
    plt.ylabel("Cost of Network/Cost of Network Without Switching", fontsize=10)
    # plt.legend(loc='upper right', fontsize='medium')
    plt.savefig("switch_loss_mesh_topology_single_graph_6.png")
    plt.show()

def compare_different_detector_parameter(cmin, f_switch = 0.1, time_limit=1e5, cost_on_trusted_node=1,cost_detector=0.1, cost_source=0.01, Lambda=100):
    objective_values = {}
    no_solution_list = []
    for eff in np.arange(start = 10, stop = 30, step = 5):
        if eff != 15:
            key_dict, g, position_graphs = trusted_nodes_utils.import_problem_from_files_flexibility_multiple_graphs_position_graph(
                cap_needed_location=f"8_nodes_mesh_topology_35_capacity_needed_{eff}_eff",
                edge_data_location=f"8_nodes_mesh_topology_35_edge_data_{eff}_eff",
                node_type_location=f"8_nodes_mesh_topology_35_node_data_{eff}_eff",
                position_node_file=f"8_nodes_mesh_topology_35_position_nodes",
                position_edge_file=f"8_nodes_mesh_topology_35_position_edges")
        else:
            key_dict, g, position_graphs = trusted_nodes_utils.import_problem_from_files_flexibility_multiple_graphs_position_graph(
                cap_needed_location=f"8_nodes_mesh_topology_35_capacity_needed",
                edge_data_location=f"8_nodes_mesh_topology_35_edge_data",
                node_type_location=f"8_nodes_mesh_topology_35_node_data",
                position_node_file=f"8_nodes_mesh_topology_35_position_nodes",
                position_edge_file=f"8_nodes_mesh_topology_35_position_edges")
        for key in g.keys():
            if key not in no_solution_list:
                try:
                    key_dict_temp = trusted_nodes_utils.make_key_dict_bidirectional(key_dict[key])
                    prob = cplex.Cplex()
                    optim = Optimisation_Switching_Calibration_fixed_frac_calibration_time(prob=prob, g=g[key],
                                                                                           key_dict=key_dict_temp)
                    sol_dict, prob, time_taken = optim.initial_optimisation_cost_reduction(cmin=cmin,
                                                                                           time_limit=time_limit,
                                                                                           cost_on_trusted_node=cost_on_trusted_node,
                                                                                           cost_detector=cost_detector,
                                                                                           cost_source=cost_source,
                                                                                           f_switch=f_switch,
                                                                                           Lambda=Lambda)
                except:
                    no_solution_list.append(key)
                    continue
                if check_solvable(prob):
                    objective_value = prob.solution.get_objective_value()
                    if eff not in objective_values.keys():
                        objective_values[eff] = {key: objective_value}
                    else:
                        objective_values[eff][key] = objective_value
    objective_value_ratios = {}
    for eff in objective_values:
        for key in objective_values[eff]:
            if eff not in objective_value_ratios.keys():
                objective_value_ratios[eff] = [objective_values[eff][key] / objective_values[15][key]]
    mean_objectives = {}
    std_objectives = {}
    for key in objective_value_ratios.keys():
        mean_objectives[key] = np.mean(objective_value_ratios[key])
        std_objectives[key] = np.std(objective_value_ratios[key])
    for key in mean_objectives.keys():
        print(f"The solution ratio for {key} is {mean_objectives[key]}. The standard deviation is {std_objectives[key]}")

def cold_vs_hot_detectors_cost(cap_needed_location, edge_data_location, node_type_location, position_node_file, position_edge_file,
                            cap_needed_location_cold_det, edge_data_location_cold_det, node_type_location_cold_det,
                         cmin, time_limit=1e5, cost_on_trusted_node=1,cost_detector=0.1, cost_on_trusted_node_cold=3.5, cost_detector_cold=0.5, cost_source=0.01, Lambda=100, f_switch = 0.1, data_storage_location_keep_each_loop = None, data_storage_location_keep_each_loop_cold_det = None):
    key_dict, g, position_graphs = trusted_nodes_utils.import_problem_from_files_flexibility_multiple_graphs_position_graph(
        cap_needed_location, edge_data_location, node_type_location,
        position_node_file=position_node_file, position_edge_file=position_edge_file)
    key_dict_cold_det, g_cold_det, position_graphs_cold_det = trusted_nodes_utils.import_problem_from_files_flexibility_multiple_graphs_position_graph(
        cap_needed_location_cold_det, edge_data_location_cold_det, node_type_location_cold_det,
        position_node_file=position_node_file, position_edge_file=position_edge_file)

    # graphs = import_graph_structure(node_information=graph_node_data_file, edge_information=graph_edge_data_file)
    if data_storage_location_keep_each_loop != None:
        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            current_key = last_row_explored["Graph key"].iloc[0]
        else:
            current_key = None
            dictionary_fieldnames = ["Graph key", "objective_value"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_key = None

    objective_values_hot = {}
    no_solution_list = []
    for key in g.keys():
        if current_key != None and key != current_key:
            continue
        elif current_key != None and key == current_key:
            current_key = None
            continue
        try:
            key_dict_temp = trusted_nodes_utils.make_key_dict_bidirectional(key_dict[key])
            prob = cplex.Cplex()
            optim = Optimisation_Switching_Calibration_fixed_frac_calibration_time(prob=prob, g=g[key],
                                                                                   key_dict=key_dict_temp)
            sol_dict, prob, time_taken = optim.initial_optimisation_cost_reduction(cmin=cmin,
                                                                                   time_limit=time_limit,
                                                                                   cost_on_trusted_node=cost_on_trusted_node,
                                                                                   cost_detector=cost_detector,
                                                                                   cost_source=cost_source,
                                                                                   f_switch=f_switch,
                                                                                   Lambda=Lambda)
        except:
            no_solution_list.append(key)
            continue
        if check_solvable(prob):
            objective_value = prob.solution.get_objective_value()
            objective_values_hot[key] = objective_value
            if data_storage_location_keep_each_loop != None:
                dictionary = [
                    {"Graph key": key, "objective_value": objective_value}]
                dictionary_fieldnames = ["Graph key", "objective_value"]
                if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
                    with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writerows(dictionary)
                else:
                    with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writeheader()
                        writer.writerows(dictionary)

    if data_storage_location_keep_each_loop_cold_det != None:
        if os.path.isfile(data_storage_location_keep_each_loop_cold_det + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop_cold_det + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            last_ratio_done = last_row_explored["ratio_hot_cold"]
            current_key = last_row_explored["Graph key"].iloc[0]
        else:
            last_ratio_done = 0.0
            current_key = None
            dictionary_fieldnames = ["ratio_hot_cold", "Graph key", "objective_value"]
            with open(data_storage_location_keep_each_loop_cold_det + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        last_ratio_done = 0.0
        current_key = None

    objective_values_cold = {}
    no_solution_list = []
    for ratio_hot_cold in np.arange(last_ratio_done, 5, 0.2):
        # change these as required
        cost_on_trusted_node_cold = cost_on_trusted_node * ratio_hot_cold
        cost_detector_cold = cost_detector * ratio_hot_cold
        for key in g_cold_det.keys():
            if current_key != None and key != current_key:
                continue
            elif current_key != None and key == current_key:
                current_key = None
                continue
            try:
                key_dict_temp = trusted_nodes_utils.make_key_dict_bidirectional(key_dict_cold_det[key])
                prob = cplex.Cplex()
                optim = Optimisation_Switching_Calibration_fixed_frac_calibration_time(prob=prob, g=g_cold_det[key],
                                                                                       key_dict=key_dict_temp)
                sol_dict, prob, time_taken = optim.initial_optimisation_cost_reduction(cmin=cmin,
                                                                                       time_limit=time_limit,
                                                                                       cost_on_trusted_node=cost_on_trusted_node_cold,
                                                                                       cost_detector=cost_detector_cold,
                                                                                       cost_source=cost_source,
                                                                                       f_switch=f_switch,
                                                                                       Lambda=Lambda)
            except:
                no_solution_list.append(key)
                continue
            if check_solvable(prob):
                objective_value = prob.solution.get_objective_value()
                if ratio_hot_cold not in objective_values_cold.keys():
                    objective_values_cold[ratio_hot_cold] = {key: objective_value}
                else:
                    objective_values_cold[ratio_hot_cold][key] = objective_value

                if data_storage_location_keep_each_loop_cold_det != None:
                    dictionary = [
                        {"ratio_hot_cold": ratio_hot_cold, "Graph key": key, "objective_value": objective_value}]
                    dictionary_fieldnames = ["ratio_hot_cold", "Graph key", "objective_value"]
                    if os.path.isfile(data_storage_location_keep_each_loop_cold_det + '.csv'):
                        with open(data_storage_location_keep_each_loop_cold_det + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writerows(dictionary)
                    else:
                        with open(data_storage_location_keep_each_loop_cold_det + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writeheader()
                            writer.writerows(dictionary)
    if data_storage_location_keep_each_loop_cold_det != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop_cold_det + ".csv")
        for index, row in plot_information.iterrows():
            if row["ratio_hot_cold"] not in objective_values_cold.keys():
                objective_values_cold[row["ratio_hot_cold"]] = {row["Graph key"]: row["objective_value"]}
            else:
                objective_values_cold[row["ratio_hot_cold"]][row["Graph key"]] = row["objective_value"]
        if data_storage_location_keep_each_loop != None:
            objective_values_hot_detectors = {}
            plot_information_hot_detectors = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            for index, row in plot_information_hot_detectors.iterrows():
                if row["Graph key"] not in objective_values_hot_detectors.keys():
                    objective_values_hot_detectors[row["Graph key"]] = [row["objective_value"]]
                else:
                    objective_values_hot_detectors[row["Graph key"]].append(row["objective_value"])
            objective_values = {}
            for hot_ratio in objective_values_cold.keys():
                for key in objective_values_cold[hot_ratio].keys():
                    if hot_ratio not in objective_values.keys():
                        objective_values[hot_ratio] = [
                            objective_values_cold[hot_ratio][key] / objective_values_hot_detectors[key]]
                    else:
                        objective_values[hot_ratio].append(
                            objective_values_cold[hot_ratio][key] / objective_values_hot_detectors[key])
    mean_objectives = {}
    std_objectives = {}
    for key in objective_values.keys():
        mean_objectives[key] = np.mean(objective_values[key])
        std_objectives[key] = np.std(objective_values[key])
    mean_differences = []
    std_differences = []
    # topologies
    x = []
    for key in mean_objectives.keys():
        mean_differences.append(mean_objectives[key])
        std_differences.append(std_objectives[key])
        x.append(key)
    plt.errorbar(x, mean_differences, yerr=std_differences, color="r", label="Normalised Cost of Network")
    plt.axhline(y=1, color='b', linestyle='-', label="Cost of Network without Switching")
    plt.legend()
    plt.xlabel("Ratio of Cold Terms to Hot Terms", fontsize=10)
    plt.ylabel("Cost of Network with Cooling/Cost of Network Without Cooling", fontsize=10)
    # plt.legend(loc='upper right', fontsize='medium')
    plt.savefig("cooling_mesh_topology.png")
    plt.show()

def cost_detector_cost_on_ratio(cap_needed_location, edge_data_location, node_type_location, position_node_file, position_edge_file,
                         cmin, time_limit=1e5, cost_on_trusted_node=1, cost_source=0.01, f_switch=0.1, Lambda=100, data_storage_location_keep_each_loop = None):
    key_dict, g, position_graphs = trusted_nodes_utils.import_problem_from_files_flexibility_multiple_graphs_position_graph(
        cap_needed_location, edge_data_location, node_type_location,
        position_node_file=position_node_file, position_edge_file=position_edge_file)
    if data_storage_location_keep_each_loop != None:
        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            last_ratio_done = last_row_explored["c_on_ratio"]
            dataframe_of_cmin_done = plot_information[plot_information["c_on_ratio"] == last_ratio_done.iloc[0]]
            current_key = last_row_explored["Graph key"].iloc[0]
            c_on_current = last_ratio_done.iloc[0]
        else:
            c_on_current = 0.0
            current_key = None
            dictionary_fieldnames = ["c_on_ratio", "Graph key", "objective_value"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        c_on_current = 0.0
        current_key = None#
    objective_values_c_on = {}
    no_solution_list = []
    for c_on_ratio in np.arange(c_on_current, 5, 0.2):
        cost_detector = c_on_ratio * cost_on_trusted_node
        for key in g.keys():
            if current_key != None and key != current_key:
                continue
            elif current_key != None and key == current_key:
                current_key = None
                continue
            try:
                key_dict_temp = trusted_nodes_utils.make_key_dict_bidirectional(key_dict[key])
                prob = cplex.Cplex()
                optim = Optimisation_Switching_Calibration_fixed_frac_calibration_time(prob=prob, g=g[key],
                                                                                       key_dict=key_dict_temp)
                sol_dict, prob, time_taken = optim.initial_optimisation_cost_reduction(cmin=cmin,
                                                                                       time_limit=time_limit,
                                                                                       cost_on_trusted_node=cost_on_trusted_node,
                                                                                       cost_detector=cost_detector,
                                                                                       cost_source=cost_source,
                                                                                       f_switch=f_switch,
                                                                                       Lambda=Lambda)
            except:
                no_solution_list.append(key)
                continue
            if check_solvable(prob):
                objective_value = prob.solution.get_objective_value()
                if c_on_ratio not in objective_values_c_on.keys():
                    objective_values_c_on[c_on_ratio] = {key: objective_value}
                else:
                    objective_values_c_on[c_on_ratio][key] = objective_value
                if data_storage_location_keep_each_loop != None:
                    dictionary = [
                        {"c_on_ratio": c_on_ratio, "Graph key": key, "objective_value": objective_value}]
                    dictionary_fieldnames = ["c_on_ratio", "Graph key", "objective_value"]
                    if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
                        with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writerows(dictionary)
                    else:
                        with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writeheader()
                            writer.writerows(dictionary)

    if data_storage_location_keep_each_loop != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
        for index, row in plot_information.iterrows():
            if row["c_on_ratio"] not in objective_values_c_on.keys():
                objective_values_c_on[row["c_on_ratio"]] = {row["Graph key"]: row["objective_value"]}
            else:
                objective_values_c_on[row["c_on_ratio"]][row["Graph key"]] = row["objective_value"]
    objective_values_at_c_on_1 = objective_values_c_on[1.0]
    objective_values = {}
    for c_on_ratio in objective_values_c_on.keys():
        for key in objective_values_c_on[c_on_ratio].keys():
            if c_on_ratio not in objective_values.keys():
                objective_values[c_on_ratio] = [
                    objective_values_c_on[c_on_ratio][key] / objective_values_at_c_on_1[key]]
            else:
                objective_values[c_on_ratio].append(
                    objective_values_c_on[c_on_ratio][key] / objective_values_at_c_on_1[key])
    mean_objectives = {}
    std_objectives = {}
    for key in objective_values.keys():
        mean_objectives[key] = np.mean(objective_values[key])
        std_objectives[key] = np.std(objective_values[key])
    mean_differences = []
    std_differences = []
    # topologies
    x = []
    for key in mean_objectives.keys():
        mean_differences.append(mean_objectives[key])
        std_differences.append(std_objectives[key])
        x.append(key)
    plt.errorbar(x, mean_differences, yerr=std_differences, color="r")
    plt.xlabel("Ratio of Detector Cost to Cost of Turning Node On", fontsize=10)
    plt.ylabel("Cost of Network/Cost of Network Without Ratio 1", fontsize=10)
    # plt.legend(loc='upper right', fontsize='medium')
    plt.savefig("c_on_ratio_mesh_topology.png")
    plt.show()

def plot_graphs_m_variation(cap_needed_location, edge_data_location, node_type_location, position_node_file, position_edge_file,
                         cmin, time_limit=1e5, cost_on_trusted_node=1, cost_source=0.01, cost_detector = 0.1, f_switch=0.1, Lambda=100, data_storage_location_keep_each_loop = None):
    key_dict, g, position_graphs = trusted_nodes_utils.import_problem_from_files_flexibility_multiple_graphs_position_graph(
        cap_needed_location, edge_data_location, node_type_location,
        position_node_file=position_node_file, position_edge_file=position_edge_file)
    no_solution_list = []
    for M in range(1,6):
        for key in g.keys():
            try:
                key_dict_copy = deepcopy(key_dict[key])
                for conn in key_dict_copy.keys():
                    key_dict_copy[conn] = M
                key_dict_temp = trusted_nodes_utils.make_key_dict_bidirectional(key_dict_copy)
                prob = cplex.Cplex()
                optim = Optimisation_Switching_Calibration_fixed_frac_calibration_time(prob=prob, g=g[key],
                                                                                       key_dict=key_dict_temp)
                sol_dict, prob, time_taken = optim.initial_optimisation_cost_reduction(cmin=cmin/M,
                                                                                       time_limit=time_limit,
                                                                                       cost_on_trusted_node=cost_on_trusted_node,
                                                                                       cost_detector=cost_detector,
                                                                                       cost_source=cost_source,
                                                                                       f_switch=f_switch,
                                                                                       Lambda=Lambda)
            except:
                no_solution_list.append(key)
                continue
            if check_solvable(prob):
                flow_dict, binary_dict, lambda_dict = optimisationmodel.split_sol_to_flow_delta_lambda(sol_dict)
                optim.plot_graph(position_graph = position_graphs[key], binary_dict = binary_dict, save_name= f"graph_{key}_M_value_{M}")


def compare_heuristic_to_perfect_model(cap_needed_location, edge_data_location, node_type_location, position_node_file, position_edge_file,
                         cmin, time_limit=1e5, cost_on_trusted_node=1, cost_source=0.01, cost_detector = 0.1, f_switch=0.1, Lambda=100, data_storage_location_keep_each_loop = None):
    key_dict, g, position_graphs = trusted_nodes_utils.import_problem_from_files_flexibility_multiple_graphs_position_graph(
        cap_needed_location, edge_data_location, node_type_location,
        position_node_file=position_node_file, position_edge_file=position_edge_file)
    if data_storage_location_keep_each_loop != None:
        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            current_key = last_row_explored["Graph key"].iloc[0]
        else:
            current_key = None
            dictionary_fieldnames = ["Graph key", "objective_value", "objective_value_heuristic"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_key = None
    current_key = None
    optimisation_heuristic = {}
    for key in g.keys():
        if current_key != None and current_key != key:
            continue
        elif current_key == key:
            current_key = None
            continue

        try:
            key_dict_temp = trusted_nodes_utils.make_key_dict_bidirectional(key_dict[key])
            prob = cplex.Cplex()
            optim = Optimisation_Switching_Calibration_fixed_frac_calibration_time(prob=prob, g=g[key],
                                                                                   key_dict=key_dict_temp)
            sol_dict, prob, time_taken = optim.initial_optimisation_cost_reduction(cmin=cmin,
                                                                                   time_limit=time_limit,
                                                                                   cost_on_trusted_node=cost_on_trusted_node,
                                                                                   cost_detector=cost_detector,
                                                                                   cost_source=cost_source,
                                                                                   f_switch=f_switch,
                                                                                   Lambda=Lambda)
            model = LP_relaxation.LP_relaxation_Trusted_Nodes_fixed_switching_time_relaxation(name = f"problem_{key}", g = g[key], key_dict=key_dict_temp)
            heuristic = Heuristic_Model.Heuristic(Lambda = Lambda, f_switch = f_switch, C_det = cost_detector, C_source = cost_source, c_on = cost_on_trusted_node, cmin = cmin)
            model_best = heuristic.full_recursion(initial_model=model)
            heuristic_objective = heuristic.calculate_current_solution_cost(model_best)
            objective = prob.solution.get_objective_value()
            optimisation_heuristic[key] = [(objective, heuristic_objective)]
            total_flow_in = {}
            for k in key_dict_temp:
                if k[0]< k[1]:
                    ind = []
                    for n in g[key].neighbors(k[1]):
                        ind.extend([f"x{n}_{k[1]}_k{k[0]}_{k[1]}", f"x{k[1]}_{n}_k{k[1]}_{k[0]}"])
                    total_flow_in[k] = sum([sol_dict[ind[i]] for i in range(len(ind))])
            for k in key_dict_temp:
                if k[0] < k[1]:
                    print("Sum flow for commodity " + str(k) + ":" + str(total_flow_in[k]))
            total_flow = {}
            total_flow_out = {}
            for n in g[key].nodes():
                ind = []
                ind_out = []
                for k in key_dict_temp:
                    if k[0]< k[1]:
                        for m in g[key].neighbors(n):
                            ind.extend([f"x{m}_{n}_k{k[0]}_{k[1]}",f"x{n}_{m}_k{k[1]}_{k[0]}"])
                            ind_out.extend([f"x{n}_{m}_k{k[0]}_{k[1]}",f"x{m}_{n}_k{k[1]}_{k[0]}"])
                        total_flow[n] = sum([sol_dict[ind[i]] for i in range(len(ind))])
                        total_flow_out[n] = sum([sol_dict[ind_out[i]] for i in range(len(ind_out))])
            for n in g[key].nodes():
                print("Sum of flow into node " + str(n) + ":" + str(total_flow[n]))
                print("Sum of flow out of node " + str(n) + ":" + str(total_flow_out[n]))
                if g[key].nodes[n]["type"] == "NodeType.T":
                    print(f"Value of N_{n}^D:" + str(sol_dict[f"N_{n}_D"]))
                    print(f"Value of d_{n}: " + str(sol_dict[f"delta_{n}"]))
                print(f"Value of N_{n}^S:" + str(sol_dict[f"N_{n}_S"]))
            for val in sol_dict.keys():
                if sol_dict[val] > 0.0000001:
                    print("Non-zero variable " + val + ": " + str(sol_dict[val]))
            if data_storage_location_keep_each_loop != None:
                dictionary = [
                    {"Graph key": key, "objective_value": objective, "objective_value_heuristic": heuristic_objective}]
                dictionary_fieldnames = ["Graph key", "objective_value", "objective_value_heuristic"]
                if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
                    with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writerows(dictionary)
                else:
                    with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writeheader()
                        writer.writerows(dictionary)


        except:
            print("No solution")
            continue

    if data_storage_location_keep_each_loop != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
        for index, row in plot_information.iterrows():
            if row["Graph key"] not in optimisation_heuristic.keys():
                optimisation_heuristic[row["Graph key"]] = [(row["objective_value"], row["objective_value_heuristic"])]

    percentage_difference_for_number_nodes = {}
    for key in optimisation_heuristic.keys():
        number_nodes = g[int(key)].number_of_nodes()
        if number_nodes not in percentage_difference_for_number_nodes.keys():
            percentage_difference_for_number_nodes[number_nodes] = [
                100 * (optimisation_heuristic[key][i][1] - optimisation_heuristic[key][i][0]) /
                optimisation_heuristic[key][i][0] for i in range(len(optimisation_heuristic[key]))]
        else:
            percentage_difference_for_number_nodes[number_nodes].extend([
                100 * (optimisation_heuristic[key][i][1] - optimisation_heuristic[key][i][0]) /
                optimisation_heuristic[key][i][0] for i in range(len(optimisation_heuristic[key]))])
    mean_objectives = {}
    std_objectives = {}
    percentiles = {}
    median = {}
    for key in percentage_difference_for_number_nodes.keys():
        mean_objectives[key] = np.mean(percentage_difference_for_number_nodes[key])
        percentiles_current = np.percentile(percentage_difference_for_number_nodes[key], [25, 50, 75])
        percentiles[key] = [percentiles_current[1] - percentiles_current[0], percentiles_current[2] - percentiles_current[1]]
        median[key] = percentiles_current[1]
        std_objectives[key] = np.std(percentage_difference_for_number_nodes[key])
    mean_differences = []
    std_differences = []
    percentile_diffs = []
    median_differences = []
    # topologies
    x = []
    for key in mean_objectives.keys():
        median_differences.append(median[key])
        mean_differences.append(mean_objectives[key])
        std_differences.append(std_objectives[key])
        percentile_diffs.append(percentiles[key])
        x.append(key)
    percentile_diffs = np.transpose(percentile_diffs)
    plt.errorbar(x, mean_differences, yerr = std_differences, color="r", capsize = 10)
    plt.xlabel("Number of Nodes in the Graph", fontsize=10)
    plt.ylabel("Percentage Difference of Heuristic to Optimal Solution", fontsize=10)
    plt.savefig("heuristic_quality_plot_no_switching.png")
    plt.show()

    percentage_difference ={}
    for key in optimisation_heuristic.keys():
        percentage_difference[key] = (optimisation_heuristic[key][1] - optimisation_heuristic[key][0])/optimisation_heuristic[key][0]
    percentage =  np.mean(list(percentage_difference.values()))
    print("Percentage Difference between model and heuristic is " + str(percentage * 100) + "%")


def compare_heuristic_to_perfect_model_genetic(cap_needed_location, edge_data_location, node_type_location, position_node_file, position_edge_file,
                         cmin, time_limit=1e5, cost_on_trusted_node=1, cost_source=0.01, cost_detector = 0.1, f_switch=0.1, Lambda=100, data_storage_location_keep_each_loop = None):
    key_dict, g, position_graphs = trusted_nodes_utils.import_problem_from_files_flexibility_multiple_graphs_position_graph(
        cap_needed_location, edge_data_location, node_type_location,
        position_node_file=position_node_file, position_edge_file=position_edge_file)
    if data_storage_location_keep_each_loop != None:
        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            current_key = last_row_explored["Graph key"].iloc[0]
        else:
            current_key = None
            dictionary_fieldnames = ["Graph key", "objective_value", "objective_value_heuristic"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_key = None
    optimisation_heuristic = {}
    for key in g.keys():
        if current_key != None and current_key != key:
            continue
        elif current_key == key:
            current_key = None
            continue

        try:
            key_dict_temp = trusted_nodes_utils.make_key_dict_bidirectional(key_dict[key])
            prob = cplex.Cplex()
            optim = Optimisation_Switching_Calibration_fixed_frac_calibration_time(prob=prob, g=g[key],
                                                                                   key_dict=key_dict_temp)
            sol_dict, prob, time_taken = optim.initial_optimisation_cost_reduction(cmin=cmin,
                                                                                   time_limit=time_limit,
                                                                                   cost_on_trusted_node=cost_on_trusted_node,
                                                                                   cost_detector=cost_detector,
                                                                                   cost_source=cost_source,
                                                                                   f_switch=f_switch,
                                                                                   Lambda=Lambda)
            heuristic = Heuristic_Genetic_Model.Heuristic_Genetic(graph=g[key], key_dict=key_dict_temp, Lambda=Lambda, f_switch=f_switch, C_det=cost_detector,
                                          C_source=cost_source, c_on=cost_on_trusted_node, cmin=cmin)
            chromosome, fitness_value = heuristic.full_recursion(number_parents_in_next_population=10,
                                                                     next_population_size=50, p_cross=0.9,
                                                                     prob_mutation=0.1, number_steps=10)

            objective = prob.solution.get_objective_value()
            optimisation_heuristic[key] = [(objective, fitness_value)]
            total_flow_in = {}
            for k in key_dict_temp:
                if k[0]< k[1]:
                    ind = []
                    for n in g[key].neighbors(k[1]):
                        ind.extend([f"x{n}_{k[1]}_k{k[0]}_{k[1]}", f"x{k[1]}_{n}_k{k[1]}_{k[0]}"])
                    total_flow_in[k] = sum([sol_dict[ind[i]] for i in range(len(ind))])
            for k in key_dict_temp:
                if k[0] < k[1]:
                    print("Sum flow for commodity " + str(k) + ":" + str(total_flow_in[k]))
            total_flow = {}
            total_flow_out = {}
            for n in g[key].nodes():
                ind = []
                ind_out = []
                for k in key_dict_temp:
                    if k[0]< k[1]:
                        for m in g[key].neighbors(n):
                            ind.extend([f"x{m}_{n}_k{k[0]}_{k[1]}",f"x{n}_{m}_k{k[1]}_{k[0]}"])
                            ind_out.extend([f"x{n}_{m}_k{k[0]}_{k[1]}",f"x{m}_{n}_k{k[1]}_{k[0]}"])
                        total_flow[n] = sum([sol_dict[ind[i]] for i in range(len(ind))])
                        total_flow_out[n] = sum([sol_dict[ind_out[i]] for i in range(len(ind_out))])
            for n in g[key].nodes():
                print("Sum of flow into node " + str(n) + ":" + str(total_flow[n]))
                print("Sum of flow out of node " + str(n) + ":" + str(total_flow_out[n]))
                if g[key].nodes[n]["type"] == "NodeType.T":
                    print(f"Value of N_{n}^D:" + str(sol_dict[f"N_{n}_D"]))
                    print(f"Value of d_{n}: " + str(sol_dict[f"delta_{n}"]))
                print(f"Value of N_{n}^S:" + str(sol_dict[f"N_{n}_S"]))
            for val in sol_dict.keys():
                if sol_dict[val] > 0.0000001:
                    print("Non-zero variable " + val + ": " + str(sol_dict[val]))
            if data_storage_location_keep_each_loop != None:
                dictionary = [
                    {"Graph key": key, "objective_value": objective, "objective_value_heuristic": fitness_value}]
                dictionary_fieldnames = ["Graph key", "objective_value", "objective_value_heuristic"]
                if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
                    with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writerows(dictionary)
                else:
                    with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writeheader()
                        writer.writerows(dictionary)


        except:
            print("No solution")
            continue

    if data_storage_location_keep_each_loop != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
        for index, row in plot_information.iterrows():
            if row["Graph key"] not in optimisation_heuristic.keys():
                optimisation_heuristic[row["Graph key"]] = [(row["objective_value"], row["objective_value_heuristic"])]

    percentage_difference_for_number_nodes = {}
    for key in optimisation_heuristic.keys():
        number_nodes = g[int(key)].number_of_nodes()
        if number_nodes not in percentage_difference_for_number_nodes.keys():
            percentage_difference_for_number_nodes[number_nodes] = [
                100 * (optimisation_heuristic[key][i][1] - optimisation_heuristic[key][i][0]) /
                optimisation_heuristic[key][i][0] for i in range(len(optimisation_heuristic[key]))]
        else:
            percentage_difference_for_number_nodes[number_nodes].extend([
                100 * (optimisation_heuristic[key][i][1] - optimisation_heuristic[key][i][0]) /
                optimisation_heuristic[key][i][0] for i in range(len(optimisation_heuristic[key]))])
    mean_objectives = {}
    std_objectives = {}
    percentiles = {}
    median = {}
    for key in percentage_difference_for_number_nodes.keys():
        mean_objectives[key] = np.mean(percentage_difference_for_number_nodes[key])
        percentiles_current = np.percentile(percentage_difference_for_number_nodes[key], [25, 50, 75])
        percentiles[key] = [percentiles_current[1] - percentiles_current[0], percentiles_current[2] - percentiles_current[1]]
        median[key] = percentiles_current[1]
        std_objectives[key] = np.std(percentage_difference_for_number_nodes[key])
    mean_differences = []
    std_differences = []
    percentile_diffs = []
    median_differences = []
    # topologies
    x = []
    for key in mean_objectives.keys():
        median_differences.append(median[key])
        mean_differences.append(mean_objectives[key])
        std_differences.append(std_objectives[key])
        percentile_diffs.append(percentiles[key])
        x.append(key)
    percentile_diffs = np.transpose(percentile_diffs)
    plt.errorbar(x, mean_differences, yerr = std_differences, color="r", capsize = 10)
    plt.xlabel("Number of Nodes in the Graph", fontsize=10)
    plt.ylabel("Percentage Difference of Heuristic to Optimal Solution", fontsize=10)
    plt.savefig("heuristic_quality_plot_genetic_three_point_crossover.png")
    plt.show()

    percentage_difference ={}
    for key in optimisation_heuristic.keys():
        percentage_difference[key] = (optimisation_heuristic[key][1] - optimisation_heuristic[key][0])/optimisation_heuristic[key][0]
    percentage =  np.mean(list(percentage_difference.values()))
    print("Percentage Difference between model and heuristic is " + str(percentage * 100) + "%")




### SPADS + Link Cost (5k *2 + 100k)
### cryostat - 40k

### so turning on : 40 + Building  fees etc.
### adding new connection : 110k
### source cost 1.5k-3k

### C_i^{det} = 110k
### C_i^{source}= 2k
### C_{tn_i} = building fees + security




if __name__ == "__main__":
    # time_variation_analysis(cap_needed_location = "l_cap_needed_bb84_graph.csv", edge_data_location = "l_edge_data_capacity_graph_bb84_network.csv", node_type_location = "l_node_data_capacity_graph_bb84_network.csv", position_node_file = "l_nodes_bb84_network_position_graph.csv",
    #                         position_edge_file = "l_edges_bb84_network_position_graph.csv",
    #                         cmin = 1000, time_limit=1e5, cost_on_trusted_node=1, cost_detector=0.1, cost_source=0.01,
    #                         f_switch=0.1, Lambda=100)

    # compare_heuristic_to_perfect_model_genetic(cap_needed_location = "2_cap_needed_bb84_graph.csv", edge_data_location = "2_edge_data_capacity_graph_bb84_network.csv", node_type_location = "2_node_data_capacity_graph_bb84_network.csv", position_node_file = "2_nodes_bb84_network_position_graph.csv", position_edge_file = "2_edges_bb84_network_position_graph.csv",
    #                                    cmin = 1000, time_limit=1e5, cost_on_trusted_node=1, cost_source=0.02,
    #                                    cost_detector=0.1, f_switch=0.0, Lambda=100,
    #                                    data_storage_location_keep_each_loop="heuristic_genetic_data_half_uniform_crossover")
    data_storage_location_keep_each_loop = "real_graph_data_detectors_1"
    if data_storage_location_keep_each_loop != None:
        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            current_cmin = last_row_explored["scale_factor"].iloc[0]
            current_key = last_row_explored["Graph key"].iloc[0]
        else:
            current_key = None
            current_cmin = 0.00
            dictionary_fieldnames = ["Graph key", "scale_factor", "objective_value", "objective_value_no_switching", "average_number_detectors"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    key_dict, g, position_graphs = trusted_nodes_utils.import_problem_from_files_flexibility_multiple_graphs_position_graph("real_network_n_k_1.csv", "real_edge_data_1.csv", "real_node_data_1.csv", position_node_file = "real_graph_node_positions.csv", position_edge_file="real_graph_edge_positions.csv")
    key_dict_no_switch, g_no_switch, position_graphs_no_switch = trusted_nodes_utils.import_problem_from_files_flexibility_multiple_graphs_position_graph(
        "real_network_n_k_1.csv", "real_edge_data_no_switching_1.csv", "real_node_data_no_switching_1.csv",
        position_node_file="real_graph_node_positions_1.csv", position_edge_file="real_graph_edge_positions_1.csv")
    cmin_solutions = {}
    for key in g.keys():
        try:
            key_dict_temp = trusted_nodes_utils.make_key_dict_bidirectional(key_dict[key])

            for i in np.arange(current_cmin + 0.1, 2.3, 0.1):
                prob = cplex.Cplex()
                optim = Optimisation_Switching_Calibration_fixed_frac_calibration_time(prob=prob, g=g[key], key_dict=key_dict_temp)
                cmin = trusted_nodes_utils.import_cmin(cmin_file="real_network_key_requirements_1.csv")
                for ckey in cmin.keys():
                    cmin[ckey] = cmin[ckey]* i
                sol_dict, prob, time_taken = optim.initial_optimisation_cost_reduction(cmin = cmin, Lambda =24, time_limit=2e2, cost_on_trusted_node=1, cost_source=0.02,
                                           cost_detector=0.1, f_switch=0.1)
                prob_2 = cplex.Cplex()
                optim_no_switching = optimisation_no_reverse_commodity.Optimisation_Problem_No_Switching(prob=prob_2, g=g_no_switch[key], key_dict=key_dict_no_switch[key])
                sol_dict_no_switch, prob_no_switch, time_taken = optim_no_switching.initial_optimisation_cost_reduction(
                 cmin=cmin, time_limit=2e2, cost_node=1,
                    cost_connection=0.1 + 0.02, Lambda=24)

                flow_dict, binary_dict, lambda_dict = optimisationmodel.split_sol_to_flow_delta_lambda(sol_dict)
                source_flow, detector_dict = split_source_detector_nodes(lambda_dict)
                total_sites_used = sum(binary_dict.values())
                average_detectors_per_site = sum(detector_dict.values()) / (3.18 )
                if i not in cmin_solutions.keys():
                    cmin_solutions[i]= {key: prob.solution.get_objective_value() / prob_2.solution.get_objective_value()}
                else:
                    cmin_solutions[i][key] = prob.solution.get_objective_value()/ prob_2.solution.get_objective_value()
                if data_storage_location_keep_each_loop != None:
                    dictionary = [
                        {"Graph key": key, "scale_factor": i, "objective_value": prob.solution.get_objective_value(), "objective_value_no_switching": prob_2.solution.get_objective_value(), "average_number_detectors": average_detectors_per_site}]
                    dictionary_fieldnames = ["Graph key", "scale_factor", "objective_value", "objective_value_no_switching", "average_number_detectors"]
                    if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
                        with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writerows(dictionary)
                    else:
                        with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writeheader()
                            writer.writerows(dictionary)



    #         flow_dict, binary_dict, lambda_dict = optimisationmodel.split_sol_to_flow_delta_lambda(sol_dict)
    #         source_terms, detector_terms = split_source_detector_nodes(lambda_dict)
    #         trusted_nodes = 0
    #         for key in binary_dict:
    #             trusted_nodes += binary_dict[key]
    #         print(f"Cost of Trusted Nodes = {trusted_nodes}")
    #         sources = 0
    #         for key in source_terms:
    #             sources += source_terms[key] * 0.01
    #         print(f"Cost of sources = {sources}")
    #         detectors = 0
    #         for key in detector_terms:
    #             detectors += detector_terms[key] * 0.1
    #         print(f"Cost of detectors = {detectors}")
        except:
            break
    # detector_average ={}
    # if data_storage_location_keep_each_loop != None:
    #     plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
    #     for index, row in plot_information.iterrows():
    #         if row["scale_factor"] not in cmin_solutions.keys():
    #             cmin_solutions[row["scale_factor"]] = {row["Graph key"]: row["objective_value"] /row["objective_value_no_switching"]}
    #         elif row["Graph key"] not in cmin_solutions[row["scale_factor"]].keys():
    #             cmin_solutions[row["scale_factor"]][row["Graph key"]] = row["objective_value"] / row["objective_value_no_switching"]
    #         else:
    #             cmin_solutions[row["scale_factor"]][row["Graph key"]] = row["objective_value"] / row["objective_value_no_switching"]
    #         if row["scale_factor"] not in detector_average.keys():
    #             detector_average[row["scale_factor"]] = {
    #                 row["Graph key"]: row["average_number_detectors"]/2}
    #         elif row["Graph key"] not in detector_average["scale_factor"].keys():
    #             detector_average[row["scale_factor"]][row["Graph key"]] = row["average_number_detectors"]/2
    #         else:
    #             detector_average[row["scale_factor"]][row["Graph key"]] = row["average_number_detectors"]/2
    # mean_objectives = {}
    # std_objectives = {}
    #
    # for key in cmin_solutions.keys():
    #     mean_objectives[key] = cmin_solutions[key][0]
    #     std_objectives[key] = 0.0
    # mean_differences = []
    # std_differences = []
    # percentile_diffs = []
    # median_differences = []
    # # topologies
    # x = []
    # for key in mean_objectives.keys():
    #     mean_differences.append(mean_objectives[key])
    #     std_differences.append(std_objectives[key])
    #     x.append(key)
    #
    # mean_detectors = {}
    # std_detectors= {}
    #
    # for key in detector_average.keys():
    #     mean_detectors[key] = detector_average[key][0]
    #     std_detectors[key] = 0.0
    # mean_differences_det = []
    # std_differences = []
    # percentile_diffs = []
    # median_differences = []
    # # topologies
    # x_det = []
    # for key in mean_detectors.keys():
    #     mean_differences_det.append(mean_detectors[key])
    #     std_differences.append(std_detectors[key])
    #     x_det.append(key)
    #
    # fig, ax1 = plt.subplots()
    # color = 'tab:red'
    # ax1.set_xlabel("Scale Factor Capacities", fontsize=10)
    # ax1.set_ylabel("Cost of Switching Solution/Cost of No Switching Solution", fontsize=10, color=color)
    # ax1.plot(x, mean_differences, color=color)
    #
    # ax2 = ax1.twinx()
    #
    # color = 'tab:blue'
    # ax2.set_ylabel("Total Number of Detectors/ 2|E|/(|S|+|T|)", fontsize=10, color=color)
    # ax2.plot(x_det, mean_differences_det, color=color)
    # fig.tight_layout()
    # plt.savefig("real_graph_plot_data_with_detector_2.png")
    # plt.show()
