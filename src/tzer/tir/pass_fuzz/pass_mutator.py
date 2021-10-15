from multiprocessing import Manager
from multiprocessing.context import Process
import queue
import tvm

from time import time
from random import choice, choices, randint, random, shuffle
from tvm import tir
from tzer.tir.domain.construct import *
from tzer.tir.domain.node import LazyDomNode
from tzer.tir.error import *

from tzer.tvmpass import PassDependenceGraph
from tzer.evolution.fitness import MAX, FitnessElites

try:
    from tvm.contrib import coverage
except Exception as e:
    print(f'No coverage in linked TVM. {e}')

target = tvm.target.Target('llvm')
tir_pass_graph = PassDependenceGraph(target)


def random_tir_passes():
    return tir_pass_graph.random_tir_passes(randint(1, 50))

def export_tir_pass(nodes):
    return tir_pass_graph.export_name(nodes)

def lower_tir_primfunc(f: tir.PrimFunc) -> tvm.ir.IRModule:
    return tvm.IRModule({"main": f})

def opt_tir_module(ir_m, tir_pass_nodes) -> tir.PrimFunc:
    if isinstance(ir_m, tir.PrimFunc):
        ir_m = tvm.IRModule({"main": ir_m})

    with tvm.transform.PassContext(opt_level=4):
        seq = tvm.transform.Sequential(
            passes = [node.mutate() for node in tir_pass_nodes],
            opt_level = 4
        )
        ir_m = seq(ir_m)

    return ir_m['main']

def build_opt_tir_module(ir_m, tir_pass_nodes) -> tir.PrimFunc:
    if isinstance(ir_m, tir.PrimFunc):
        ir_m = tvm.IRModule({"main": ir_m})

    with tvm.transform.PassContext(opt_level=4):
        seq = tvm.transform.Sequential(
            passes = [node.mutate() for node in tir_pass_nodes],
            opt_level = 4
        )
        ir_m = seq(ir_m)

    return tvm.build(ir_m)


def infer_pass_from_tir(root: LazyDomNode):
        tree = queue.Queue()
        tree.put(root)
        nodes = []
        while not tree.empty():
            n = tree.get()
            for c in n.children:
                tree.put(c)
            nodes.append(n)


        common_nodes = [
            tir_pass_graph.tir_pass_nodes['CoProcSync'],
            tir_pass_graph.tir_pass_nodes['CombineContextCall'],
            tir_pass_graph.tir_pass_nodes['Apply'],
            tir_pass_graph.tir_pass_nodes['InferFragment'],
            tir_pass_graph.tir_pass_nodes['RemoveNoOp'],
            tir_pass_graph.tir_pass_nodes['Simplify'],
            tir_pass_graph.tir_pass_nodes['VerifyMemory'],
            tir_pass_graph.tir_pass_nodes['InjectPrefetch'],
            tir_pass_graph.tir_pass_nodes['InstrumentBoundCheckers'],
            tir_pass_graph.tir_pass_nodes['LiftAttrScope'],
            tir_pass_graph.tir_pass_nodes['LowerCustomDatatypes'],
            tir_pass_graph.tir_pass_nodes['LowerInitBlock'],
            tir_pass_graph.tir_pass_nodes['LowerIntrin'],
            tir_pass_graph.tir_pass_nodes['LowerTVMBuiltin'],
            tir_pass_graph.tir_pass_nodes['LowerThreadAllreduce'],
            tir_pass_graph.tir_pass_nodes['LowerWarpMemory'],
            tir_pass_graph.tir_pass_nodes['NarrowDataType'],
            tir_pass_graph.tir_pass_nodes['SplitHostDevice']
        ]

        specific_nodes = []

        node_types = [type(i.cons) for i in nodes]

        for t in node_types:
            if t == While or t == For:
                specific_nodes.append([
                    tir_pass_graph.tir_pass_nodes['InjectVirtualThread'],
                    tir_pass_graph.tir_pass_nodes['LoopPartition'],
                    tir_pass_graph.tir_pass_nodes['UnrollLoop'],
                    tir_pass_graph.tir_pass_nodes['VectorizeLoop'],
                ])
            elif t == AssertStmt:
                specific_nodes.append([tir_pass_graph.tir_pass_nodes['SkipAssert']])    
            elif t == Store or t == Load:
                specific_nodes.append([
                    tir_pass_graph.tir_pass_nodes['InjectDoubleBuffer'],
                    tir_pass_graph.tir_pass_nodes['CompactBufferAllocation'],
                    tir_pass_graph.tir_pass_nodes['StorageFlatten'],
                    tir_pass_graph.tir_pass_nodes['FlattenBuffer'],
                    tir_pass_graph.tir_pass_nodes['StorageRewrite'],
                    tir_pass_graph.tir_pass_nodes['PlanAndUpdateBufferAllocationLocation']
                ])
            elif t == Select:
                specific_nodes.append([tir_pass_graph.tir_pass_nodes['RewriteUnsafeSelect']])
            elif t == FloatImmInj or t == FloatImmInj:
                specific_nodes.append([
                    tir_pass_graph.tir_pass_nodes['BF16Legalize'],
                    tir_pass_graph.tir_pass_nodes['BF16Promote'],
                    tir_pass_graph.tir_pass_nodes['BF16TypeLowering']
                ])
            elif t == Cast:
                specific_nodes.append([tir_pass_graph.tir_pass_nodes['BF16CastElimination']])
            else:
                pass
            
        selected_common_nodes = choices(common_nodes, k=randint(1, 50))
        selected_specific_nodes = []
        for s in specific_nodes:
            selected_specific_nodes.extend(choices(s, k=randint(1, len(specific_nodes)+len(s))))
        nodes = selected_common_nodes + selected_specific_nodes
        shuffle(nodes)

        return nodes



class CFPassMutator:
    def __init__(self) -> None:
        from .evo import Evolution

        self.evolution = Evolution()
        self.timeout = 120
        self.pass_nodes = list(tir_pass_graph.tir_pass_nodes.values())
        self.pass_nodes = [node for node in self.pass_nodes if not node.disable]
        self.set_evolution()


    def set_evolution(self):
        self.evolution.set_population_size(200)
        self.evolution.create_genotypes()
        for p in self.evolution.population:
            p.set_genes(random_tir_passes())
            p.set_other_genes(self.pass_nodes)
        self.evolution.set_fitness_type(MAX, 10000000.0)
        self.evolution.set_fitness_selections(FitnessElites(self.evolution.fitness_list, 0.5))


    def mutate(self, func):
        index, tir_pass_nodes = self.evolution.get_genotypes()
        
        manager = Manager()
        return_dict = manager.dict()

        def run(func, args, kwargs, return_dict):

            coverage.push()

            start_time = time()
            return_dict['err'] = ''
            return_dict['result'] = None

            try:
                result = func(*args, **kwargs)
                return_dict['result'] = result
            except Exception as e:
                return_dict['err'] = e

            return_dict['blocks'] = coverage.get_now()
            return_dict['execute_time'] = time() - start_time
            coverage.pop()
            return_dict['cov'] = coverage.get_now()
            return_dict['hitmap'] = coverage.get_hitmap()

        old_coverage = coverage.get_now()

        p = Process(target=run, args=(build_opt_tir_module, (func, tir_pass_nodes), {}, return_dict))
        p.start()
        p.join(timeout=self.timeout)

        coverage.set_now(return_dict['cov'])
        coverage.set_hitmap(return_dict['hitmap'])

        return_dict['inc_cov'] = coverage.get_now() - old_coverage
        self.evolution.set_genotypes_result(index, return_dict)

        if p.is_alive():
            p.terminate()
            raise MaybeDeadLoop
        if not p.exitcode == 0:
            msg = f'{func.__name__} terminated abnormally with exit code: {p.exitcode}, tir pass: {self.graph.export_name(tir_pass_nodes)}'
            raise RuntimeFailure(msg)

        if return_dict['result'] != None:
            return return_dict['result']
        else:
            raise return_dict['err']


class SimplePassMutator:
    def __init__(self, pool_size=50, passes_len_range=(4,50)) -> None:
        self.pass_pools = []
        self.pass_nodes = list(tir_pass_graph.tir_pass_nodes.values())
        self.pass_nodes = [node for node in self.pass_nodes if not node.disable]
        self.passes_len_range = passes_len_range

        for _ in range(pool_size):
            self.pass_pools.append(tir_pass_graph.random_tir_passes(randint(*self.passes_len_range)))

        self.pass_sequences = [tir_pass_graph.random_tir_passes(randint(*self.passes_len_range)) for _ in range(pool_size)]

    def single_point_mutate(self, passes):
        new_pass = choice(self.pass_nodes)

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
        return mutator(passes, new_pass)

    def subseq_mutate(self, passes):
        new_passes = choices(self.pass_nodes, k=randint(*self.passes_len_range))

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
        return mutator(passes, new_passes)


    def single_point_crossover(self, passes):
        other_passes = choice(self.pass_pools)
        min_length = min(len(passes), len(other_passes))
        if min_length == 0:
            return passes
        else:
            pos = randint(0, min_length - 1)
            passes[pos:], other_passes[pos:] = other_passes[pos:], passes[pos:]
            return passes

    def two_point_crossover(self, passes):
        other_passes = choice(self.pass_pools)
        min_length = min(len(passes), len(other_passes))
        if min_length <= 1:
            return passes
        else:
            start_pos = randint(0, min_length - 2)
            end_pos = randint(start_pos + 1, min_length - 1)
            passes[start_pos:end_pos+1], other_passes[start_pos:end_pos+1] = other_passes[start_pos:end_pos+1], passes[start_pos:end_pos+1]
            return passes


    def uniform_crossover(self, passes):
        other_passes = choice(self.pass_pools)
        min_length = min(len(passes), len(other_passes))
        if min_length == 0:
            return passes
        else:
            for i in range(min_length):
                if random() < 0.2333:
                    passes[i], other_passes[i] = other_passes[i], passes[i]
            return passes
    
    def mutate(self, passes):
        mutator = choice([self.single_point_mutate, self.subseq_mutate, self.single_point_crossover, self.two_point_crossover, self.uniform_crossover])
        new_passes = mutator(passes)
        valid_passes = tir_pass_graph.fix_target(new_passes)
        return valid_passes

    def get_pass_sequence(self):
        return choice(self.pass_sequences)

    def put_pass_sequence(self, pass_sequence):
        self.pass_sequences.append(pass_sequence)


class GeneralPassMutator:
    def __init__(self, passes_len_range=(4, 50)) -> None:
        self.pass_nodes = list(tir_pass_graph.tir_pass_nodes.values())
        self.pass_nodes = [node for node in self.pass_nodes if not node.disable]
        self.passes_len_range = passes_len_range

    def single_point_mutate(self, passes):
        new_pass = choice(self.pass_nodes)

        def add(passes, new_pass):
            pass_length = len(passes)
            if pass_length >= 1:  
                pos = randint(0, pass_length - 1)
                return passes[:pos:] + [new_pass] + passes[pos::]
            else:
                return [new_pass]

        def delete(passes, _):
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
        return mutator(passes, new_pass)

    def subseq_mutate(self, passes):
        len_min, len_max = self.passes_len_range
        len_min = max(1, min(len_min, len(passes)))
        len_max = max(1, min(len_max, len(passes)))
        new_passes = choices(self.pass_nodes, k=randint(len_min, len_max))

        def add(passes, new_passes):
            passes_length = len(passes)
            if passes_length >= 1:  
                pos = randint(0, passes_length - 1)
                return passes[:pos:] + new_passes + passes[pos::]
            else:
                return new_passes

        def delete(passes, _):
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
        return mutator(passes, new_passes)


    def single_point_crossover(self, l_passes, r_passes):
        min_length = min(len(l_passes), len(r_passes))
        if min_length == 0:
            return l_passes
        else:
            pos = randint(0, min_length - 1)
            l_passes[pos:], r_passes[pos:] = r_passes[pos:], l_passes[pos:]
            return l_passes

    def two_point_crossover(self, l_passes, r_passes):
        min_length = min(len(l_passes), len(r_passes))
        if min_length <= 1:
            return l_passes
        else:
            start_pos = randint(0, min_length - 2)
            end_pos = randint(start_pos + 1, min_length - 1)
            l_passes[start_pos:end_pos+1], r_passes[start_pos:end_pos+1] = r_passes[start_pos:end_pos+1], l_passes[start_pos:end_pos+1]
            return l_passes


    def uniform_crossover(self, l_passes, r_passes):
        min_length = min(len(l_passes), len(r_passes))
        if min_length == 0:
            return l_passes
        else:
            for i in range(min_length):
                if random() < 0.2333:
                    l_passes[i], r_passes[i] = r_passes[i], l_passes[i]
            return l_passes
