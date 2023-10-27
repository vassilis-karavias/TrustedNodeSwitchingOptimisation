import LP_relaxation
import numpy as np
from copy import deepcopy
import trusted_nodes_utils


def remove_trusted_node(g, trusted_node_to_remove):
    new_g = deepcopy(g)
    new_g.remove_node(trusted_node_to_remove)
    return new_g

def get_trusted_node_index_from_key(key):
    new_string = key.split("_")[1]
    return int(new_string)

class Heuristic():

    def __init__(self, Lambda, f_switch, C_det, C_source, c_on, cmin):
        self.Lambda = Lambda
        self.f_switch = f_switch
        self.C_det = C_det
        self.C_source = C_source
        self.cmin =cmin
        self.c_on = c_on

    def calculate_current_solution_cost(self, model):
        cost = 0.0
        on_trusted_nodes = []
        for key in model.model.detector_variables:
            if model.model.detector_variables[key].solution_value > 0.00000001:
                cost += self.C_det * np.ceil(model.model.detector_variables[key].solution_value)
                if get_trusted_node_index_from_key(key) not in on_trusted_nodes:
                    on_trusted_nodes.append(get_trusted_node_index_from_key(key))
        for key in model.model.source_variables:
            if model.model.source_variables[key].solution_value > 0.00000001:
                cost += self.C_source * np.ceil(model.model.source_variables[key].solution_value)
                if get_trusted_node_index_from_key(key) not in on_trusted_nodes and model.g.nodes[get_trusted_node_index_from_key(key)]["type"] == "NodeType.T":
                    on_trusted_nodes.append(get_trusted_node_index_from_key(key))
        cost += len(on_trusted_nodes) * self.c_on
        return cost

    def print_current_solution_cost_breakdown(self, model):
        cost = 0.0
        on_trusted_nodes = []
        for key in model.model.detector_variables:
            if model.model.detector_variables[key].solution_value > 0.00000001:
                cost += self.C_det * np.ceil(model.model.detector_variables[key].solution_value)
                if get_trusted_node_index_from_key(key) not in on_trusted_nodes:
                    on_trusted_nodes.append(get_trusted_node_index_from_key(key))
        print("Cost Attributed To Detectors: " + str(cost))
        cost = 0.0
        for key in model.model.source_variables:
            if model.model.source_variables[key].solution_value > 0.00000001:
                cost += self.C_source * np.ceil(model.model.source_variables[key].solution_value)
                if get_trusted_node_index_from_key(key) not in on_trusted_nodes and \
                        model.g.nodes[get_trusted_node_index_from_key(key)]["type"] == "NodeType.T":
                    on_trusted_nodes.append(get_trusted_node_index_from_key(key))
        print("Cost Attributed To Sources: " + str(cost))
        cost = len(on_trusted_nodes) * self.c_on
        print("Cost Attributed To On Trusted Nodes: " + str(cost))

    def calculate_cost_difference(self, model_1, model_2):
        cost_model_1 = self.calculate_current_solution_cost(model_1)
        cost_model_2 = self.calculate_current_solution_cost(model_2)
        return cost_model_1 - cost_model_2

    def remove_trusted_nodes_not_in_use(self, model):
        trusted_nodes_to_remove = {}
        # only trusted nodes have detectors
        for key in model.model.detector_variables:
            if get_trusted_node_index_from_key(key) not in trusted_nodes_to_remove.keys():
                trusted_nodes_to_remove[get_trusted_node_index_from_key(key)] = 1
        for key in model.model.detector_variables:
            if model.model.detector_variables[key].solution_value > 0.00000001 and get_trusted_node_index_from_key(key) in trusted_nodes_to_remove.keys():
                trusted_nodes_to_remove.pop(get_trusted_node_index_from_key(key))
        for key in model.model.source_variables:
            if model.model.source_variables[key].solution_value > 0.00000001 and get_trusted_node_index_from_key(key) in trusted_nodes_to_remove.keys():
                trusted_nodes_to_remove.pop(get_trusted_node_index_from_key(key))
        new_graph = model.g
        for i in trusted_nodes_to_remove:
            new_graph = remove_trusted_node(new_graph, i)
        return new_graph

    def single_step_down(self, model):
        new_graph = self.remove_trusted_nodes_not_in_use(model)
        ## need to get the full number of remaining trusted nodes.
        remaining_trusted_nodes = {}
        for i in new_graph.nodes:
            if new_graph.nodes[i]["type"] == "T" or new_graph.nodes[i]["type"] == "NodeType.T":
                remaining_trusted_nodes[i] = 1
        models = []
        for node in remaining_trusted_nodes.keys():
            current_new_graph = remove_trusted_node(new_graph, node)
            new_model = LP_relaxation.LP_relaxation_Trusted_Nodes_fixed_switching_time_relaxation(name = model.model.name
                                                                  , g =  current_new_graph, key_dict=model.key_dict)
            new_model.set_up_problem(Lambda = self.Lambda, f_switch = self.f_switch, C_det = self.C_det,
                                     C_source = self.C_source, cmin = self.cmin)
            if new_model.model.solve():
                obj = new_model.model.objective_value
                models.append(new_model)
        current_best_model = None
        current_cost_improvement = 0.0
        for model_new in models:
            cost_improvement = self.calculate_cost_difference(model_new, model)
            if cost_improvement < current_cost_improvement:
                current_best_model = model_new
                current_cost_improvement = cost_improvement
        if current_best_model == None:
            # in this case all viable solutions of removing a node yield an increase in cost and thus the most
            # cost effective solution is to use the solution of the model:
            return model, True
        else:
            # otherwise there is an improved model - return the best improved model and the fact that we haven't
            # finished yet
            return current_best_model, False



    def full_recursion(self, initial_model):
        initial_model.set_up_problem(Lambda = self.Lambda, f_switch = self.f_switch, C_det = self.C_det,
                                     C_source = self.C_source, cmin = self.cmin)
        # initial_model.model.export_as_lp(path = "~/PycharmProjects/GraphTF/trusted_node_optimisation", basename = "lp_heuristic")
        if initial_model.model.solve():
            model = initial_model
            not_complete = True
            while not_complete:
                new_best_model, complete = self.single_step_down(model)
                if new_best_model != None:
                    model = new_best_model
                not_complete = not complete
            return model



if __name__ == "__main__":
    key_dict, g, position_graphs = trusted_nodes_utils.import_problem_from_files_flexibility_multiple_graphs_position_graph("1_cap_needed_bb84_graph.csv", "1_edge_data_capacity_graph_bb84_network.csv", "1_node_data_capacity_graph_bb84_network.csv", position_node_file = "1_nodes_bb84_network_position_graph.csv", position_edge_file="1_edges_bb84_network_position_graph.csv")
    for key in g.keys():
        key_dict_temp = trusted_nodes_utils.make_key_dict_bidirectional(key_dict[key])
        model = LP_relaxation.LP_relaxation_Trusted_Nodes_fixed_switching_time_relaxation(name = f"problem_{key}", g = g[key], key_dict=key_dict_temp)
        heuristic = Heuristic(Lambda = 24, f_switch = 0.1, C_det = 0.1, C_source = 0.01, c_on = 1, cmin = 1000)
        model_best = heuristic.full_recursion(initial_model=model)
        try:
            print(str(heuristic.calculate_current_solution_cost(model_best)))
            heuristic.print_current_solution_cost_breakdown(model_best)
        except:
            print("No solution")
            continue
