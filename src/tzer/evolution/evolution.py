# rewrite from https://github.com/vspandan/IFuzzer/blob/master/codegen/GrammaticalEvolution.py
from copy import deepcopy
from random import choice, randint, random
import time
from tzer import context
from tzer.report import Reporter


from .fitness import CENTER, MAX, MIN
from .fitness import FitnessList, Fitness, Replacement
from .gene import Genotype

from tzer.fuzz import make_context
from tzer.context import _RELAY_FUNCTION_HARD_PASSES_, _ALL_DIR_PASS_NODES_

class Evolution:
    def __init__(self, reporter):
        self.shrink_mutation_rate=0
        self.FUNCTION_EXEC_TIMEOUT=5
        self._pre_selected = []
        self.history = []
        self.population = []
        self.fitness_selections = []
        self.replacement_selections = []        
        
        self._crossover_rate = 0.4
        self._children_per_crossover = 2
        self._mutation_rate = 0.02
        self._max_fitness_rate = 0.5

        self._start_gene_length = None
        self._max_gene_length = None
        self._max_passes_length = None

        self.fitness_list = FitnessList(MAX)
        
        self._generation = 0
        self._fitness_fail = -1000
        self._maintain_history = True

        self._bnf = {}
        self._population_size = 0
        
        self.dynamic_mutation = 0
        self.dynamic_crossover = 0

        self.mutationCount = 1
        self.crossoverCount = 1
        self._multiple_rate = 0
        self._max_depth = 0
        self._generative_mutation_rate=0.5
        self.parsimony_constant=0
        self.mean_length=1
        self.crossover_break=False
        self.mutation_break=False
        self.function_break=False
        self.crossover_bias_rate=0
        self.max_generations = 100

        self.reporter = reporter
        if self.reporter == None:
            self.reporter = Reporter()

    def set_crossover_bias_rate(self,percentage):
        self.crossover_bias_rate=percentage

    def set_max_depth(self,depth):
        self._max_depth=depth

    def set_generative_mutation_rate(self,rate):
        self._generative_mutation_rate=rate

    def set_multiple_rate(self, rate):
        self._multiple_rate = rate

    def set_crossover_count(self, count):
        self.crossoverCount=count

    def set_mutation_count(self, count):
        self.mutationCount=count

    def set_execution_timeout(self, timeout):
        self.execution_timeout = timeout

    def dynamic_mutation_rate(self, ind):
        self.dynamic_mutation = ind
    
    def dynamic_crossover_rate(self, ind):
        self.dynamic_crossover = ind

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
            gene._generation = self._generation
            if self._generation == 0:
                self.compute_fitness(gene)

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


    def run(self, starting_generation=0, max_time=5*60):
        self.reporter.record_coverage()
        self._generation = starting_generation
        self.calculate_fitness()
        self._generation += 1
        start_time = time.time()

        while True:
            print(f'generation: {self._generation}')
            if self._maintain_history:
                self.history.append(deepcopy(self.fitness_list))

            current_time = time.time()
            self.reporter.record_coverage()

            if current_time - start_time > max_time:
                break
            
            if self._generation > self.max_generations:
                break
            
            if self._continue_processing() and self.fitness_list.best_value() != self._fitness_fail:
                self._perform_endcycle()
                self._generation += 1
                self.calculate_fitness()
            else:
                break

            print(self.fitness_list)


    def create_genotypes(self, model):
        member_no = 0
        while member_no < self._population_size:
            genotype = Genotype(member_no)
            ctx = make_context(model)
            genotype.append_genes(ctx.compile.relay_pass_types, _RELAY_FUNCTION_HARD_PASSES_)
            genotype.append_genes(ctx.compile.tir_pass_nodes, _ALL_DIR_PASS_NODES_)
            genotype.ctx = ctx
            self.population.append(genotype)
            member_no += 1
        self.reporter.record_coverage()


    def compute_fitness(self, gene):
        if gene._fitness <= 0:
            gene.evaluate()
            self.reporter.record_coverage()

            if gene.err != '' and isinstance(gene.err, Exception):
                self.reporter.report_bug(gene.err, gene.ctx, str(gene.err))

            score, length = self.compute_subscore(gene)
            gene.score = score
            gene._fitness =  gene.score
            gene.genes_length = length

        # if gene.exitcode > 0: # @Jiawei
        #     gene._fitness = self.fitness_list.get_target_value()


    def compute_subscore(self, genotype):
        print(f'coverage: {genotype.cur_cov}, time: {genotype.execute_time}')
        score = genotype.cur_cov
        length = sum([len(genes.genes) for genes in genotype.genes_list])
        return score, length


    def _perform_endcycle(self):
        print(f'endcycle, {time.time()}')
        self._pre_selected = self._evaluate_fitness(True)
        print(f'pre_selected {len(self._pre_selected)}')
        print(f'pre_selected: {[gene._fitness for gene in self._pre_selected]}')
        children = []
        remaining_count = self._population_size - len(self._pre_selected)
        print('remaining_count', remaining_count)
        while len(children) < remaining_count:
            limit = round(random(),1) <= 0.7

            fitness_pool = self._evaluate_fitness(limit)
            current_children = self._perform_crossovers(fitness_pool)

            current_children = self._perform_mutations(current_children, len(current_children))
            if current_children is not None:
                children.extend(current_children)

        for child in children:
            self.compute_fitness(child)

        self._perform_replacements(children)


    def _evaluate_fitness(self, limit=False): 
        print(f'evaluate fitness {time.time()}')
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
        print(f'start crossover, {time.time()}')
        childrens = []
        length = int(round(self._crossover_rate * float(self._population_size)))
        """
        If no of fitness selections is less than no of indv undergoing crossover, than only no equal to no of fitness selections are allowed to undergo process.
        """
        length = min(length, len(parents))
        if length % 2 == 1:
            length -= 1

        if length >= 2:
            print("_perform_crossovers " + str(len(parents)) + " individuals are participating in the crossover")
            while len(parents) >= 2 :
                print("_perform_crossovers - Remaining " + str(len(parents)) + " individuals are participating in the crossover")
                child1 = choice(parents)
                parents.remove(child1)
                child2 = choice(parents)
                parents.remove(child2)

                if child1.crossover(child2):
                    child1._fitness = -1
                    child2._fitness = -1

                childrens.append(child1)
                childrens.append(child2)

            print(f'finish crossover, {time.time()}')

            return childrens    


    def _perform_mutations(self, genotypes, count):
        mutated_genotypes = []
        for genotype in genotypes:
            if genotype.mutate():
                genotype._fitnesss = -1
            mutated_genotypes.append(genotype)

            if len(mutated_genotypes) == count:
                break

        return mutated_genotypes


    def _perform_replacements(self, fitness_pool):
        position = 0
        for gene in self._pre_selected:
            gene.member_no = position
            self.population[position] = gene
            position += 1
        
        fitness_pool.sort(key=Genotype.get_fitness, reverse=True)
        for gene in fitness_pool:
            if position < self._population_size:
                gene.member_no = position
                self.population[position] = gene
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
