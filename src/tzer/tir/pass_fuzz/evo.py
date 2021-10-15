from copy import deepcopy
from random import choice, randint, random
import time

from tzer.evolution.fitness import CENTER, MAX, MIN
from tzer.evolution.fitness import FitnessList, Fitness, Replacement
from .genotype import Genotype

class Evolution:
    def __init__(self):
        self._pre_selected = []
        self.history = []
        self.population = []
        self.children = []
        self.fitness_selections = []
        self.replacement_selections = []        
        
        self._crossover_rate = 0.4
        self._children_per_crossover = 2
        self._mutation_rate = 0.04
        self._max_fitness_rate = 0.5

        self._start_gene_length = None
        self._max_gene_length = None
        self._max_passes_length = None

        self.fitness_list = FitnessList(MAX)
        
        self._generation = 0
        self._fitness_fail = -1000
        self._maintain_history = True

        self._population_size = 0
        

        self.mutationCount = 1
        self.crossoverCount = 1
        self._multiple_rate = 0
        self._max_depth = 0
        self._generative_mutation_rate=0.5

        self.crossover_bias_rate=0
        self.max_generations = 20


    def set_crossover_bias_rate(self,percentage):
        self.crossover_bias_rate=percentage

    def set_max_depth(self,depth):
        self._max_depth=depth

    def set_generative_mutation_rate(self,rate):
        self._generative_mutation_rate=rate

    def set_multiple_rate(self, rate):
        self._multiple_rate = rate

    def set_execution_timeout(self, timeout):
        self.execution_timeout = timeout

    def set_population_size(self, size):
        if isinstance(size, int) and size > 0:
            self._population_size = size
            i = len(self.fitness_list)
            while i < size:
                self.fitness_list.append([0.0, i])
                i += 1
        else:
            raise ValueError("""
                population size, %s, must be a long above 0""" % (size))


    def set_maintain_history(self, true_false):
            self._maintain_history = true_false

    def set_max_passes_length(self, max_passes_length):
        self._max_passes_length = max_passes_length

    def set_fitness_fail(self, fitness_fail):
        fitness_fail = float(fitness_fail)
        self._fitness_fail = fitness_fail

    def set_mutation_rate(self, mutation_rate):
        self._mutation_rate = mutation_rate

    def set_crossover_rate(self, crossover_rate):
        self._crossover_rate = crossover_rate

    def set_children_per_crossover(self, children_per_crossover):
        self._children_per_crossover = children_per_crossover

    def set_max_generations(self, generations):
            self.max_generations = generations

    def get_max_generations(self):
        return self.max_generations

    def set_fitness_type(self, fitness_type, target_value=0.0):
        self.fitness_list.set_fitness_type(fitness_type)
        self.fitness_list.set_target_value(target_value)

    def get_fitness_type(self):
        return self.fitness_list.get_fitness_type()

    def set_max_fitness_rate(self, max_fitness_rate):
        self._max_fitness_rate = max_fitness_rate

    def set_fitness_selections(self, *params):
        for fitness_selection in params:
            if isinstance(fitness_selection, Fitness):
                self.fitness_selections.append(fitness_selection)
            else:
                raise ValueError("Invalid fitness selection")

    def set_replacement_selections(self, *params):
        for replacement_selection in params:
            if isinstance(replacement_selection, Replacement):
                self.replacement_selections.append(replacement_selection)
            else:
                raise ValueError("Invalid replacement selection")

    def get_fitness_history(self, statistic='best_value'):
        hist_list = []
        for fitness_list in self.history:
            hist_list.append(fitness_list.__getattribute__(statistic)())
        return hist_list

    def get_best_member(self):
        return self.population[self.fitness_list.best_member()]

    def get_worst_member(self):
        return self.population[self.fitness_list.worst_member()]


    def get_genotypes(self):
        available_genotypes = self.population + self.children
        for i in range(len(available_genotypes)):
            if available_genotypes[i].blocks <= 0:
                return i, available_genotypes[i].genes

        self._generation += 1
        # self._pre_selected = self._evaluate_fitness(True)

        if self._continue_processing() and self.fitness_list.best_value() != self._fitness_fail:
            if self._generation > 1:
                self._perform_replacements(self.children)
            self.calculate_simple_fitness()

            self._pre_selected = self._evaluate_fitness(True)
            self.children = []
            remaining_count = self._population_size - len(self._pre_selected)
            while len(self.children) < remaining_count:
                limit = round(random(),1) <= 0.7

                fitness_pool = self._evaluate_fitness(limit)
                tmp_children = self._perform_crossovers(fitness_pool)
                tmp_children = self._perform_mutations(tmp_children, len(tmp_children))
                
                if tmp_children is not None:
                    self.children.extend(tmp_children)

            print(self.fitness_list)

        available_genotypes = self.population + self.children

        for i in available_genotypes:
            i.blocks = -1

        for i in range(len(available_genotypes)):
            if available_genotypes[i].blocks <= 0:
                return i, available_genotypes[i].genes


    def calculate_fitness(self):
        print(f'calc fitness, {time.time()}')
        self.mean_length = 1
        
        def calculate_covariance(mean_fitness):
            value = 0
            for gene in self.population:
                value += (gene.get_fitness() - mean_fitness ) * ( gene.genes_length - self.mean_length )
            return value

        def variance():
            value = 0
            for gene in self.population:
                value += ( gene.genes_length - self.mean_length ) ** 2 
            return value
        
        total_length = 0
        total_fitness = 0.0
        for gene in self.population:
            # gene._generation = self._generation

            self.population[gene.member_no]=gene
            self.fitness_list[gene.member_no][0] = gene.get_fitness()
            total_length += gene.genes_length
            total_fitness += gene.get_fitness()

        self.mean_length = total_length / self._population_size
        mean_fitness = total_fitness / self._population_size

        varianceValue = variance()
        if varianceValue == 0:
            self.parsimony_constant = 0
        else:
            self.parsimony_constant = calculate_covariance(mean_fitness)/varianceValue

        print(f'finish calc fitness, {time.time()}')


    def calculate_simple_fitness(self):
        for gene in self.population:
            self.population[gene.member_no]=gene
            self.fitness_list[gene.member_no][0] = gene.get_fitness()

        print(f'finish calc simple fitness, {time.time()}')



    def set_genotypes_result(self, i, result):
        available_genotypes = self.population + self.children
        genotype = available_genotypes[i]
        genotype.blocks = result['blocks']
        genotype.execute_time = result['execute_time']
        genotype.inc_cov = result['inc_cov']


    def create_genotypes(self):
        member_no = 0
        while member_no < self._population_size:
            gene = Genotype(member_no)
            self.population.append(gene)
            member_no += 1


    def _perform_endcycle(self):
        print(f'endcycle, {time.time()}')
        self._pre_selected = self._evaluate_fitness(True)
        print(f'pre_selected {len(self._pre_selected)}')
        print(f'pre_selected: {[gene._fitness for gene in self._pre_selected]}')
        childrens = []
        remaining_count = self._population_size - len(self._pre_selected)
        print('remaining_count', remaining_count)
        while len(childrens) < remaining_count:
            limit = round(random(),1) <= 0.7

            fitness_pool = self._evaluate_fitness(limit)

            child_list1 = self._perform_crossovers(fitness_pool)

            child_list = self._perform_mutations(child_list1, len(child_list1))
            # child_list = self._perform_mutations(fitness_pool,(remainingPopCount-len(childList)))
            if child_list is not None:
                childrens.extend(child_list)
        self._perform_replacements(childrens)


    def _evaluate_fitness(self, limit=False): 
        parents = []
        population = self.population
        
        if limit:
            population = []
            total = int(round(self._max_fitness_rate * float(self._population_size)))
            count = 0

            for fsel in self.fitness_selections:
                fsel.set_fitness_list(self.fitness_list)
                selected = fsel.select()
                for i in selected:
                    if count == total:
                        break
                    population.append(self.population[i])
                    count += 1

        for gene in population:
            if gene._fitness != self._fitness_fail:
                parents.append(deepcopy(gene))
    
        print(f'finish evaluate fitness {time.time()}')
        return parents

    # crossovers
    def _perform_crossovers(self, parents):
        childrens = []
        length = int(round(self._crossover_rate * float(self._population_size)))
        """
        If no of fitness selections is less than no of indv undergoing crossover, than only no equal to no of fitness selections are allowed to undergo process.
        """
        length = min(length, len(parents))
        if length % 2 == 1:
            length -= 1

        if length >= 2:
            while len(parents) >= 2 :
                child1 = choice(parents)
                parents.remove(child1)
                child2 = choice(parents)
                parents.remove(child2)

                child1.crossover(child2.genes)

                child1.blocks = -1
                child2.blocks = -1

                childrens.append(child1)
                childrens.append(child2)

        return childrens
          

    def _perform_mutations(self, genes, count):
        mutated_genes = []
        for gene in genes:
            if random() < self._mutation_rate:
                gene.mutate()
                gene.blocks = -1
            mutated_genes.append(gene)
            if len(mutated_genes) == count:
                break
        return mutated_genes


    def _perform_replacements(self, fitness_pool):
        position = 0
        for gene in self._pre_selected:
            gene.member_no = position
            self.population[position] = gene
            self.fitness_list[position][0] = gene.get_fitness()
            position += 1
        
        fitness_pool.sort(key=Genotype.get_fitness, reverse=True)
        for gene in fitness_pool:
            if position < self._population_size:
                gene.member_no = position
                self.population[position] = gene
                self.fitness_list[position][0] = gene.get_fitness()
                position += 1
            else:
                break


    def _continue_processing(self):

        """
        This method analyzes the fitness list against the stopping_criteria defined over target_value and max generations
        """
        fitl = self.fitness_list

        if fitl.get_target_value() is not None and fitl.best_value() == fitl.get_target_value():
            return False
        else:
            return True