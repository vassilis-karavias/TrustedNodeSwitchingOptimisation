import copy

import LP_relaxation
import numpy as np
from copy import deepcopy
import trusted_nodes_utils
from random import randint, uniform
import Heuristic_Model


class Fitness_Function():


    def __init__(self, chromosome, origin_graph, key_dict):
        self.chromosome = chromosome
        new_graph = deepcopy(origin_graph)
        for node in chromosome.dict_form.keys():
            if chromosome.dict_form[node] == 0:
                new_graph = Heuristic_Model.remove_trusted_node(new_graph, node)
        self.graph = new_graph
        self.key_dict = key_dict
        self.model = LP_relaxation.LP_relaxation_Trusted_Nodes_fixed_switching_time_relaxation(name = "model_i", g = self.graph, key_dict = self.key_dict)


    def _calculate_fitness_function(self, C_det, C_source, c_on):
        cost = 0.0
        on_trusted_nodes = []
        for key in self.model.model.detector_variables:
            if self.model.model.detector_variables[key].solution_value > 0.00000001:
                cost += C_det * np.ceil(self.model.model.detector_variables[key].solution_value)
                if Heuristic_Model.get_trusted_node_index_from_key(key) not in on_trusted_nodes:
                    on_trusted_nodes.append(Heuristic_Model.get_trusted_node_index_from_key(key))
        for key in self.model.model.source_variables:
            if self.model.model.source_variables[key].solution_value > 0.00000001:
                cost += C_source * np.ceil(self.model.model.source_variables[key].solution_value)
                if Heuristic_Model.get_trusted_node_index_from_key(key) not in on_trusted_nodes and \
                        self.model.g.nodes[Heuristic_Model.get_trusted_node_index_from_key(key)]["type"] == "NodeType.T":
                    on_trusted_nodes.append(Heuristic_Model.get_trusted_node_index_from_key(key))
        cost += len(on_trusted_nodes) * c_on
        return cost

    def get_fitness_value(self, Lambda, f_switch, C_det, C_source, c_on, cmin):
        self.model.set_up_problem(Lambda=Lambda, f_switch=f_switch, C_det=C_det, C_source=C_source, cmin=cmin)
        if self.model.model.solve():
            fitness_value = self._calculate_fitness_function(C_det, C_source, c_on)
            return fitness_value
        else:
            ## if no solution possible the fitness function is infty
            return np.infty




class Chromosome():


    def __init__(self, dict_chromosome, graph):
        for node in graph.nodes:
            if graph.nodes[node]["type"] == "NodeType.T" or graph.nodes[node]["type"] == "T":
                if node not in dict_chromosome.keys():
                    raise ValueError
            else:
                if node in dict_chromosome.keys():
                    raise ValueError
        self.dict_form = dict_chromosome
        self.graph = graph

    def crossover(self, chromosome_2):
        dict_crossover_value = {}
        dict_crossover_value_2 = {}
        crossover_point = randint(2,len(self.dict_form) - 2)
        i = 0
        for node in self.dict_form.keys():
            if node not in chromosome_2.dict_form.keys():
                raise ValueError
            else:
                if i < crossover_point:
                    dict_crossover_value[node] = self.dict_form[node]
                    dict_crossover_value_2[node] = chromosome_2.dict_form[node]
                else:
                    dict_crossover_value[node] = chromosome_2.dict_form[node]
                    dict_crossover_value_2[node] = self.dict_form[node]
                i += 1
        return Chromosome(dict_chromosome=dict_crossover_value, graph=self.graph), Chromosome(dict_chromosome=dict_crossover_value_2, graph = self.graph)


    def crossover_2_point(self, chromosome_2):
        dict_crossover_value = {}
        dict_crossover_value_2 = {}
        crossover_point_1 = randint(1,len(self.dict_form) - 1)
        crossover_point_2 = randint(1, len(self.dict_form) - 2)
        print("Obtained Crossover Points")
        if crossover_point_1 == crossover_point_2:
            crossover_point_2 += 1
            print("Crossover Points Same: Correction made")
        i = 0
        if crossover_point_2 < crossover_point_1:
            print("crossover 2 smaller than 1. Starting swap")
            cp = crossover_point_2
            crossover_point_2 = crossover_point_1
            crossover_point_1 = cp
            print("Swap Made")
        for node in self.dict_form.keys():
            if node not in chromosome_2.dict_form.keys():
                raise ValueError
            else:
                if i < crossover_point_1:
                    dict_crossover_value[node] = self.dict_form[node]
                    dict_crossover_value_2[node] = chromosome_2.dict_form[node]
                elif i < crossover_point_2:
                    dict_crossover_value_2[node] = self.dict_form[node]
                    dict_crossover_value[node] = chromosome_2.dict_form[node]
                else:
                    dict_crossover_value[node] = self.dict_form[node]
                    dict_crossover_value_2[node] = chromosome_2.dict_form[node]
                i += 1
        return Chromosome(dict_chromosome=dict_crossover_value, graph=self.graph), Chromosome(dict_chromosome=dict_crossover_value_2, graph=self.graph)

    def crossover_3_point(self, chromosome_2):
        dict_crossover_value = {}
        dict_crossover_value_2 = {}
        crossover_point_1 = randint(1,len(self.dict_form) - 1)
        crossover_point_2 = randint(1, len(self.dict_form) - 2)
        crossover_point_3 = randint(1, len(self.dict_form) - 3)
        if crossover_point_1 == crossover_point_2:
            crossover_point_2 += 1
        if crossover_point_1 == crossover_point_3:
            crossover_point_3 += 2
        if crossover_point_2 == crossover_point_3:
            crossover_point_3 += 1
        i = 0

        if crossover_point_2 < crossover_point_1:
            cp = crossover_point_2
            crossover_point_2 = crossover_point_1
            crossover_point_1 = cp
        if crossover_point_3 < crossover_point_2:
            cp = crossover_point_2
            crossover_point_2 = crossover_point_3
            crossover_point_3 = cp
        if crossover_point_2 < crossover_point_1:
            cp = crossover_point_2
            crossover_point_2 = crossover_point_1
            crossover_point_1 = cp
        for node in self.dict_form.keys():
            if node not in chromosome_2.dict_form.keys():
                raise ValueError
            else:
                if i < crossover_point_1:
                    dict_crossover_value[node] = self.dict_form[node]
                    dict_crossover_value_2[node] = chromosome_2.dict_form[node]
                elif i < crossover_point_2:
                    dict_crossover_value_2[node] = self.dict_form[node]
                    dict_crossover_value[node] = chromosome_2.dict_form[node]
                elif i < crossover_point_3:
                    dict_crossover_value[node] = self.dict_form[node]
                    dict_crossover_value_2[node] = chromosome_2.dict_form[node]
                else:
                    dict_crossover_value_2[node] = self.dict_form[node]
                    dict_crossover_value[node] = chromosome_2.dict_form[node]
                i += 1
        return Chromosome(dict_chromosome=dict_crossover_value, graph=self.graph), Chromosome(dict_chromosome=dict_crossover_value_2, graph=self.graph)

    def crossover_half_uniform(self, chromosome_2):
        ## might need a different crossover method?
        dict_crossover_value = {}
        for node in self.dict_form.keys():
            if node not in chromosome_2.dict_form.keys():
                raise ValueError
            else:
                if self.dict_form[node] == chromosome_2.dict_form[node]:
                    dict_crossover_value[node] = self.dict_form[node]
                else:
                    dict_crossover_value[node] = randint(0,1)
        return Chromosome(dict_chromosome=dict_crossover_value, graph = self.graph)

    def crossover_three_parents(self, chromosome_2, chromosome_3):
        ## might need a different crossover method?
        dict_crossover_value = {}
        for node in self.dict_form.keys():
            if node not in chromosome_2.dict_form.keys():
                raise ValueError
            else:
                if self.dict_form[node] == chromosome_2.dict_form[node]:
                    dict_crossover_value[node] = self.dict_form[node]
                else:
                    dict_crossover_value[node] = chromosome_3.dict_form[node]
        return Chromosome(dict_chromosome=dict_crossover_value, graph = self.graph)


    def get_next_best_with_less_values(self, key_dict,Lambda, f_switch, C_det,
                                                          C_source, c_on, cmin, fitness_value_this_chromosome):
        models = []
        best_fitness_value = 1000000000
        best_model = None
        for node in self.dict_form.keys():
            if self.dict_form[node] == 1:
                new_dict_form = copy.deepcopy(self.dict_form)
                new_dict_form[node] = 0
                new_chromosome = Chromosome(new_dict_form, self.graph)
                fitness = Fitness_Function(chromosome=new_chromosome, origin_graph=self.graph, key_dict=key_dict)
                fitness_value = fitness.get_fitness_value(Lambda=Lambda, f_switch=f_switch, C_det= C_det,
                                                          C_source=C_source, c_on=c_on, cmin=cmin)
                if fitness_value < best_fitness_value:
                    best_fitness_value = fitness_value
                    best_model = new_chromosome
        if best_fitness_value <= fitness_value_this_chromosome and best_model != None:
            return best_model
        else:
            return None


    def mutation(self, prob_mutation):
        if prob_mutation > 1:
            print("prob_mutation > 1")
            raise ValueError
        new_chromosome_dict = {}
        for node in self.dict_form.keys():
            print("Getting Random Value")
            random_value = np.random.uniform()
            if random_value <= prob_mutation:
                print("Mutation Occuring")
                new_chromosome_dict[node] = (self.dict_form[node] + 1)% 2
            else:
                print("No Mutation")
                new_chromosome_dict[node] = self.dict_form[node]
        print("Mutation Completed")
        return Chromosome(dict_chromosome= new_chromosome_dict, graph = self.graph)


def takeSecond(elem):
    return elem[1]

class Population():

    def __init__(self, list_chromosomes, graph, key_dict, Lambda, f_switch, C_det, C_source, c_on, cmin):
        self.list_chromosomes = list_chromosomes
        self.graph = graph
        self.key_dict = key_dict
        self.Lambda = Lambda
        self.f_switch = f_switch
        self.C_det = C_det
        self.C_source = C_source
        self.c_on = c_on
        self.cmin = cmin


    def selection(self, number_parents_in_next_population):
        if number_parents_in_next_population > len(self.list_chromosomes):
            raise ValueError
        chromosomes_to_keep = []
        current_needed_fitness_value = 1000000000 ### we want to exclude only solutions that are not feasible
        i =0
        for chromosome in self.list_chromosomes:
            fitness = Fitness_Function(chromosome= chromosome, origin_graph= self.graph, key_dict=self.key_dict)
            fitness_value = fitness.get_fitness_value(Lambda= self.Lambda, f_switch= self.f_switch, C_det= self.C_det, C_source = self.C_source, c_on = self.c_on, cmin= self.cmin)
            if fitness_value < current_needed_fitness_value:
                if len(chromosomes_to_keep) < number_parents_in_next_population:
                    chromosomes_to_keep.append((chromosome, fitness_value))
                    chromosomes_to_keep.sort(key=takeSecond)
                else:
                    chromosomes_to_keep = chromosomes_to_keep[:-1] + [(chromosome, fitness_value)]
                    chromosomes_to_keep.sort(key=takeSecond)
                    current_needed_fitness_value = chromosomes_to_keep[-1][1]
            # print("Finished chromosome " + str(i))
            # i += 1
        chromosomes = []
        for chromosome, fitness_value in chromosomes_to_keep:
            chromosomes.append(chromosome)
        return chromosomes


    def selection_fitness_proportionate_selection(self, number_parents_in_next_population):
        if number_parents_in_next_population > len(self.list_chromosomes):
            raise ValueError
        chromosomes_to_keep = []
        total_fitness = 0.0
        i=0
        for chromosome in self.list_chromosomes:
            fitness = Fitness_Function(chromosome= chromosome, origin_graph= self.graph, key_dict=self.key_dict)
            fitness_value = fitness.get_fitness_value(Lambda= self.Lambda, f_switch= self.f_switch, C_det= self.C_det, C_source = self.C_source, c_on = self.c_on, cmin= self.cmin)
            if i == 0:
                fitness_value_for_largest = fitness_value
            chromosomes_to_keep.append((chromosome,fitness_value))
            if fitness_value != np.infty:
                total_fitness += fitness_value
            # print("Finished chromosome " + str(i))
            i += 1
        chromosomes = []
        ### chromomsome 1 is always the one for reducing number by one:
        chromosome = self.list_chromosomes[0]

        ### This is for the chromosome reduction selection
        # new_chromosome = chromosome.get_next_best_with_less_values(key_dict= self.key_dict,Lambda = self.Lambda, f_switch = self.f_switch, C_det= self.C_det,
        #                                                   C_source= self.C_source, c_on= self.c_on, cmin= self.cmin, fitness_value_this_chromosome = fitness_value_for_largest)
        # if new_chromosome == None:
        #     chromosomes = []
        #     return chromosomes, None
        # else:
        #     chromosomes.append(new_chromosome)
        # use stochastic acceptance
        fittest_chromosome = min(chromosomes_to_keep, key = lambda t: t[1])
        while len(chromosomes) < number_parents_in_next_population:
            total_reduce_fitness = 0.0
            for chromosome, fitness_value in chromosomes_to_keep:
                prob = uniform(0,1)
                if fitness_value != np.infty:
                    if prob < fitness_value/total_fitness:
                        chromosomes.append(chromosome)
                        chromosomes_to_keep.remove((chromosome, fitness_value))
                        total_reduce_fitness += fitness_value
            total_fitness = total_fitness - total_reduce_fitness
            if total_fitness < 0.0001:
                break
        return chromosomes, fittest_chromosome


    def generate_next_population(self, number_parents_in_next_population, next_population_size, p_cross, prob_mutation):
        ## p_cross -> probability of using crossing, prob_mutation is the probability that a gene undergoes a mutation
        parent_chromosomes, fittest_chromosome = self.selection_fitness_proportionate_selection(number_parents_in_next_population=number_parents_in_next_population)
        print("Obtained Parent Chromosomes")
        if fittest_chromosome == None:
            return None, None
        next_generation = []
        p_mut = 1-p_cross
        while len(next_generation) < next_population_size:
            prob = uniform(0, 1)
            if prob < p_mut:
                print("Mutation Started")
                parent = randint(0,len(parent_chromosomes)-1)
                print("Mutation Function Entered")
                child_chromosome = parent_chromosomes[parent].mutation(prob_mutation)
                next_generation.append(child_chromosome)
                print("Mutated")
            else:
                print("Crossover Started")
                parent_1 = randint(0, len(parent_chromosomes) - 1)
                parent_2 = randint(0, len(parent_chromosomes) - 2)
                if parent_2 == parent_1 and parent_2 != len(parent_chromosomes) - 1:
                    parent_2 += 1
                elif parent_2 == parent_1:
                    parent_2 -= 1
                child_chromosome_1= parent_chromosomes[parent_1].crossover_half_uniform(parent_chromosomes[parent_2])
                next_generation.extend([child_chromosome_1])
                print("Crossover Made")
        print("Next Generation Population Made")
        return Population(list_chromosomes = next_generation, graph = self.graph, key_dict = self.key_dict, Lambda = self.Lambda, f_switch = self.f_switch, C_det = self.C_det, C_source = self.C_source, c_on = self.c_on, cmin = self.cmin), fittest_chromosome

    def select_best_member_of_population(self):
        return self.selection(number_parents_in_next_population=1)


class Heuristic_Genetic():

    def __init__(self, graph, key_dict, Lambda, f_switch, C_det, C_source, c_on, cmin):
        self.graph = graph
        self.key_dict = key_dict
        self.Lambda = Lambda
        self.f_switch = f_switch
        self.C_det = C_det
        self.C_source = C_source
        self.cmin =cmin
        self.c_on = c_on

    def generate_initial_population(self, population_size):
        #### full node on is a member of all initial populations and used to check existence of solution Returns None
        #### if no solution exists
        chromosome_keys = []
        chromosomes = []
        for node in self.graph.nodes:
            if self.graph.nodes[node]["type"] == "NodeType.T" or self.graph.nodes[node]["type"] == "T":
                chromosome_keys.append(node)
        full_gene = {}
        for node in chromosome_keys:
            full_gene[node] = 1
        chromosome_1 = Chromosome(dict_chromosome = full_gene, graph = self.graph)

        fitness = Fitness_Function(chromosome = chromosome_1, origin_graph = self.graph, key_dict = self.key_dict)
        fitness_value = fitness.get_fitness_value(Lambda = self.Lambda, f_switch = self.f_switch, C_det= self.C_det, C_source = self.C_source, c_on = self.c_on, cmin = self.cmin)
        if fitness_value == np.infty:
            return None
        chromosomes.append(chromosome_1)
        while len(chromosomes) < population_size:
            chromosome_dict = {node: randint(0,1) for node in chromosome_keys}
            chromosome = Chromosome(dict_chromosome = chromosome_dict, graph = self.graph)
            chromosomes.append(chromosome)
        return Population(list_chromosomes = chromosomes, graph = self.graph, key_dict = self.key_dict, Lambda = self.Lambda, f_switch = self.f_switch, C_det = self.C_det, C_source = self.C_source, c_on = self.c_on, cmin = self.cmin)



    def single_step(self, current_population, number_parents_in_next_population, next_population_size, p_cross, prob_mutation):
        return current_population.generate_next_population(number_parents_in_next_population, next_population_size, p_cross, prob_mutation)


    def full_recursion(self, number_parents_in_next_population, next_population_size, p_cross, prob_mutation, number_steps):
        if number_parents_in_next_population > next_population_size:
            print("Number of parents for next generation cannot be bigger than size of parent population")
            raise ValueError
        if p_cross > 1:
            print("CrossOver probability cannot exceed 1")
            raise ValueError
        if prob_mutation >1:
            print("Probability of gene mutation cannot exceed 1")
            raise ValueError
        initial_population = self.generate_initial_population(population_size=next_population_size)
        print("Initial Population Generated")
        current_population = initial_population
        if initial_population == None:
            print("No solution exists")
            raise ValueError
        else:
            prev_best = 100000000
            fittest_chromosomes = []
            for i in range(number_steps):
                print("Starting Step " + str(i))
                current_population_curr, fittest_chromosome_curr = self.single_step(current_population = current_population, number_parents_in_next_population = number_parents_in_next_population, next_population_size = next_population_size, p_cross = p_cross, prob_mutation = prob_mutation)
                print("Ending Step " + str(i))
                if current_population_curr == None:
                    return min(fittest_chromosomes, key = lambda t: t[1])
                current_population = current_population_curr
                fittest_chromosomes.append(fittest_chromosome_curr)
                # print("Current Difference: " + str(prev_best - fittest_chromosome[1]))
                # if prev_best - fittest_chromosome[1] < 0.05 and prev_best - fittest_chromosome[1] >= 0.0:
                #     break
                # if prev_best - fittest_chromosome[1] > 0:
                #     prev_best = fittest_chromosome[1]
            print("Finished Steps. Getting best chromosome")
            best_chromosome = current_population.select_best_member_of_population()
            fitness = Fitness_Function(chromosome= best_chromosome[0], origin_graph= self.graph, key_dict=self.key_dict)
            fitness_value = fitness.get_fitness_value(Lambda= self.Lambda, f_switch= self.f_switch, C_det= self.C_det, C_source = self.C_source, c_on = self.c_on, cmin= self.cmin)
            return best_chromosome, fitness_value



if __name__ == "__main__":
    key_dict, g, position_graphs = trusted_nodes_utils.import_problem_from_files_flexibility_multiple_graphs_position_graph("2_cap_needed_bb84_graph.csv", "2_edge_data_capacity_graph_bb84_network.csv", "2_node_data_capacity_graph_bb84_network.csv", position_node_file = "2_nodes_bb84_network_position_graph.csv", position_edge_file="2_edges_bb84_network_position_graph.csv")
    for key in g.keys():
        key_dict_temp = trusted_nodes_utils.make_key_dict_bidirectional(key_dict[key])
        heuristic = Heuristic_Genetic(graph = g[key], key_dict = key_dict_temp, Lambda = 24, f_switch = 0.1, C_det = 0.1, C_source = 0.01, c_on = 1, cmin = 1000)
        try:
            chromosome, fitness_value = heuristic.full_recursion(number_parents_in_next_population = 10, next_population_size = 50, p_cross= 0.7, prob_mutation= 0.6, number_steps =100)
            print("Best Value: " + str(fitness_value))
        except:
            print("No solution")
            continue
