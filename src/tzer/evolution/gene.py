from dataclasses import is_dataclass
from random import choice, randint, random, sample, shuffle
from time import time
from multiprocessing import Process, Manager
from tzer.error import MaybeDeadLoop, RuntimeFailure

from tzer.template import execute_both_mode
from tzer.seed_eval import SimpleLSTMEvaluator
from tzer.context import _RELAY_FUNCTION_HARD_PASSES_

try:
    from tvm.contrib import coverage
except Exception as e:
    print(f'No coverage in linked TVM. {e}')

class Genes:
    def __init__(self, genes, other_genes=[]):
        self.genes = genes
        self.other_genes = other_genes

    def other_genes(self, other_genes):
        self.other_genes = other_genes

    
    def new(self):
        possible_genes = self.genes + self.other_genes
        return [choice(possible_genes) for _ in range(randint(1, len(possible_genes)))]


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

    def mutate(self, new_genes):
        c = random()
        if c < 0.25:
            self.add(new_genes)
        elif c < 0.5 and c >= 0.25:
            self.delete()
        elif c < 0.75 and c >= 0.5:
            self.replace(new_genes)
        return c < 0.75

    # crossover
    def exchage(self, other):
        self.genes, other.genes = other.genes, self.genes
    
    def pairwise_exchange(self, other):
        length = min(len(self.genes), len(other.genes))
        for i in range(0, length, 2):
            self.genes[i], other.genes[i] = other.genes[i], self.genes[i]
    
    def splice(self, other):
        length = min(len(self.genes), len(other.genes))
        mid = length // 2

        parts = [self.genes[:mid:], self.genes[mid::], other.genes[:mid:], other.genes[mid::]]
        shuffle(parts)

        self.genes = parts[0] + parts[1]
        other.genes = parts[2] + parts[3]

    def crossover(self, other):
        c = random()
        if c < 0.25:
            self.exchage(other)
        elif c < 0.5 and c >= 0.25:
            self.pairwise_exchange(other)
        elif c < 0.75 and c >= 0.5:
            self.splice(other)
        return c < 0.75


pass_to_id = {p: i for i, p in enumerate(_RELAY_FUNCTION_HARD_PASSES_)}
evaluator = SimpleLSTMEvaluator(len(_RELAY_FUNCTION_HARD_PASSES_))

class Genotype:
    def __init__(self, member_no):
        self._generation = 0
        self.member_no = member_no
        self._fitness = -1
        self._max_depth = 0
        self.err = ""
        self.out = ""
        self.score = 0
        self.genes_list = []
        self.genes_length = 0
        self._initial_member_no = -1
        self.execute_time = -1
        self.timeout = 600
        self.cur_cov = 0
        self.inc_cov = 0
        self.exitcode = 0
    
    def set_genes_list(self, genes_list):
        self.genes_list = genes_list
        self.genes_length = sum([len(i.genes) for i in genes_list])

    def append_genes(self, genes, other_genes):
        genes = Genes(genes, other_genes)
        self.genes_list.append(genes)
        self.genes_length += len(genes.genes)

    def get_fitness(self):
        return self.cur_cov

    def mutate(self):
        status = False
        for genes in self.genes_list:
            new_gene = genes.new()
            status |= genes.mutate(new_gene)
        return status
    
    def crossover(self, other):
        status = False
        for genes1, genes2 in zip(self.genes_list, other.genes_list):
            status |= genes1.crossover(genes2)
        return status


    def evaluate(self):
        old_coverage = coverage.get_now()
        coverage.push()
        start_time = time()

        try:
            execute_both_mode(self.ctx)
        except Exception as e:
            self.err = e

        self.cur_cov = coverage.get_now()
        self.execute_time = time() - start_time
        coverage.pop()
        self.inc_cov = coverage.get_now() - old_coverage
        # manager = Manager()
        # return_dict = manager.dict()

        # old_coverage = coverage.get_now()

        # def run(ctx, return_dict):
        #     coverage.push()

        #     start_time = time()
        #     return_dict['err'] = ''

        #     try:
        #         execute_both_mode(ctx)
        #     except Exception as e:
        #         return_dict['err'] = e

        #     return_dict['cur_cov'] = coverage.get_now()
        #     return_dict['execute_time'] = time() - start_time
        #     coverage.pop()
        #     return_dict['cov'] = coverage.get_now()
        #     return_dict['hitmap'] = coverage.get_hitmap()

        # p = Process(target=run, args=(self.ctx, return_dict))
        # p.start()
        # p.join(timeout=self.timeout)


        # if p.is_alive():
        #     p.terminate()
        #     self.execute_time = self.timeout
        #     self.err = MaybeDeadLoop()
        # else:
        #     self.exitcode = p.exitcode
        
        # if p.exitcode > 0:
        #     if self.err == None:
        #         self.err = RuntimeFailure()

        # if 'cur_cov' in return_dict:
        #     self.cur_cov = return_dict['cur_cov']
        # if  'err' in return_dict:
        #     self.err = return_dict['err']
        # if 'execute_time' in return_dict:
        #     self.execute_time = return_dict['execute_time']

        # if 'cov' in return_dict:
        #     coverage.set_now(return_dict['cov'])
        # if 'hitmap' in return_dict:
        #     coverage.set_hitmap(return_dict['hitmap'])

        # self.inc_cov = coverage.get_now() - old_coverage

