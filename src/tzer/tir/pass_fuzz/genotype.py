from random import choice, choices, randint, random
from time import time
from multiprocessing import Process, Manager

class Genotype:
    def __init__(self, member_no):
        self._generation = 0
        self.member_no = member_no
        self._fitness = None
        self._max_depth = 0
        self.err = ""
        self.out = ""
        self.score = 0
        self.genes = []
        self.other_genes = []
        self.genes_length = 0
        self._initial_member_no = -1
        self.execute_time = 120
        self.blocks = 0
        self.inc_cov = 0
        self.passes_len_range = (1, 16)
    
    def set_genes(self, genes):
        self.genes = genes
        self.genes_length = len(genes)

    def set_other_genes(self, other_genes):
        self.other_genes = other_genes

    def get_fitness(self):
        return self.blocks

    def new(self):
        return choice(self.genes)

    def new(self):
        possible_genes = self.genes + self.other_genes
        return choices(possible_genes, k=randint(1, len(possible_genes)))


    def add(self, new_genes):
        genes_length = len(self.genes)
        if genes_length >= 1:  
            pos = randint(0, genes_length - 1)
            self.genes = self.genes[:pos:] + new_genes + self.genes[pos::]
        else:
            self.genes = new_genes


    def delete(self):
        genes_length = len(self.genes)
        if genes_length >= 1:  
            start_pos = randint(0, genes_length - 1)
            end_pos = randint(start_pos, genes_length - 1)
            self.genes = self.genes[:start_pos] + self.genes[end_pos+1:]
        else:
            self.genes = []

    def replace(self, new_genes):
        genes_length = len(self.genes)

        if genes_length >= 1:  
            start_pos = randint(0, genes_length - 1)
            end_pos = randint(start_pos, genes_length - 1)
            self.genes = self.genes[:start_pos:] + new_genes + self.genes[end_pos+1::]

    # def mutate(self, new_genes):
    #     c = random()
    #     if c < 0.25:
    #         self.add(new_genes)
    #     elif c < 0.5 and c >= 0.25:
    #         self.delete()
    #     elif c < 0.75 and c >= 0.5:
    #         self.replace(new_genes)
    #     return c >= 0.75


    def set_add(self, add_function):
        self.add = add_function

    def set_delete(self, delete_function):
        self.delete = delete_function

    def set_replace(self, replace_function):
        self.replace = replace_function

    def set_mutate(self, mutate_function):
        self.mutate = mutate_function


    def single_point_mutate(self, passes):
        new_pass = choice(self.other_genes)

        def add(passes, new_pass):
            pass_length = len(passes)
            if pass_length >= 1:  
                pos = randint(0, pass_length - 1)
                return passes[:pos:] + [new_pass] + passes[pos::]
            else:
                return [new_pass]

        def delete(passes, new_pass):
            pass_length = len(passes)
            if pass_length >= 1:  
                pos = randint(0, pass_length - 1)
                return passes[:pos-1] + passes[pos+1:]
            else:
                return []

        def replace(passes, new_pass):
            pass_length = len(passes)

            if pass_length >= 1:  
                pos = randint(0, pass_length - 1)
                return passes[:pos:] + [new_pass] + passes[pos+1::]
            else:
                return []

        mutator = choice([add, delete, replace])
        mutator(passes, new_pass)

    def subseq_mutate(self):
        passes = self.genes
        pass_nodes = self.other_genes
        new_passes = choices(pass_nodes, k=randint(*self.passes_len_range))

        def add(passes, new_passes):
            passes_length = len(passes)
            if passes_length >= 1:  
                pos = randint(0, passes_length - 1)
                return passes[:pos:] + new_passes + passes[pos::]
            else:
                return new_passes

        def delete(passes,new_passes):
            passes_length = len(passes)
            if passes_length >= 1:  
                start_pos = randint(0, passes_length - 1)
                end_pos = randint(start_pos, passes_length - 1)
                return passes[:start_pos] + passes[end_pos+1:]
            else:
                return []


        def replace(passes, new_passes):
            passes_length = len(passes)

            if passes_length >= 1:  
                start_pos = randint(0, passes_length - 1)
                end_pos = randint(start_pos, passes_length - 1)
                return passes[:start_pos:] + new_passes + passes[end_pos+1::]
            else:
                return passes

        mutator = choice([add, delete, replace])
        mutator(passes, new_passes)


    def single_point_crossover(self, other_passes):
        passes = self.genes
        min_length = min(len(passes), len(other_passes))
        if min_length > 0:
            pos = randint(0, min_length - 1)
            passes[:pos],other_passes[:pos] = other_passes[:pos], passes[:pos]


    def two_point_crossover(self, other_passes):
        passes = self.genes
        min_length = min(len(passes), len(other_passes))
        if min_length > 1:
            start_pos = randint(0, min_length - 2)
            end_pos = randint(start_pos + 1, min_length - 1)
            passes[start_pos:end_pos+1], other_passes[start_pos:end_pos+1] = other_passes[start_pos:end_pos+1], passes[start_pos:end_pos+1]

    def uniform_crossover(self, other_passes):
        passes = self.genes
        min_length = min(len(passes), len(other_passes))
        if min_length == 0:
            return passes
        else:
            for i in range(min_length):
                if random() < 0.2333:
                    passes[i], other_passes[i] = other_passes[i], passes[i]
    

    def crossover(self, other_passes):
        mutator = choice([self.single_point_crossover, self.two_point_crossover, self.uniform_crossover])
        mutator(other_passes)


    def mutate(self):
        mutator = choice([self.single_point_mutate, self.subseq_mutate])
        mutator()

