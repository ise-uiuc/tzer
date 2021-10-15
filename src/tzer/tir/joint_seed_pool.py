"""Joint IR-Pass seed pool to mutate them together.
"""

from dataclasses import dataclass
from typing import List
import random
import os

from tvm import tir

from ..tvmpass import PassNode
from .pass_fuzz.pass_mutator import random_tir_passes, GeneralPassMutator, tir_pass_graph
from .mutate import Flipper, Nilizer, Deletor, Insertor, SizedGenerator, RecursiveMutatorCombinator, WeightedIRMutatorCombinator
from .mutate.specific import SpecificMutator

__USE_RANDOM_PASS_GEN__ = os.getenv('RANDOM_PASS') is not None
__USE_FULL_PASS__ = os.getenv('FULL_PASS') is not None

assert not (__USE_RANDOM_PASS_GEN__ and __USE_FULL_PASS__)

__PASS_BASELINE_TESTING__ = __USE_FULL_PASS__ or __USE_RANDOM_PASS_GEN__

@dataclass
class JointSeed:
    tir_func: tir.PrimFunc
    pass_seq: List[PassNode]    # Abstract pass wrapper concretized by `.mutate()`
    n_ir_cont_fail:   int = 0   # #Continous failure time after IR mutation.
    n_pass_cont_fail: int = 0 if not __PASS_BASELINE_TESTING__ else 1000000000000
    # #Continous failure time after Pass mutation.

class JointSeedPool:
    def __init__(self, max_gen_size = 1024, general_cfg_mut = False, use_none = False, tir_func_list = None) -> None:
        self.seeds: List[JointSeed] = [JointSeed(tir_func=tir.PrimFunc([], tir.Evaluate(tir.const(0))), pass_seq=random_tir_passes())] # Must be an init seed.
        self.pass_mutator = GeneralPassMutator()

        ir_generator = SizedGenerator(
            lambda: random.randint(0, max_gen_size))

        general_purpose_mutator = RecursiveMutatorCombinator([
            (1, Insertor(ir_generator)),
            (1, Deletor()),
            (1, Flipper()),
        ], general_cfg_mut)

        if use_none:
            general_purpose_mutator.weighted_mutators.append((1, Nilizer()))

        self.ir_mutator = WeightedIRMutatorCombinator([
            (len(general_purpose_mutator.weighted_mutators), general_purpose_mutator),
        ])

        if os.getenv('LOW') is not None:
            specific_mutator = RecursiveMutatorCombinator([
                (1, SpecificMutator())
            ])
            self.ir_mutator.weighted_ir_mutators.append((1, specific_mutator))

        if tir_func_list is not None:
            for f in tir_func_list:
                self.seeds.append(JointSeed(tir_func=f, pass_seq=random_tir_passes()))

        self.initial_pool_size = len(self.seeds)

    def put(self, tir_func = tir.PrimFunc([], tir.Evaluate(tir.const(0))), pass_seq = None):
        if pass_seq is None:
            pass_seq = random_tir_passes()
        self.seeds.append(JointSeed(tir_func=tir_func, pass_seq=pass_seq))

    def mutate_ir(self, tir_func):
        return self.ir_mutator.mutate_ir(tir_func)

    def mutate_pass(self, pass_seq):
        def make_rhs(binary_func):
            if len(self.seeds) > 0:
                rhs = random.choice(self.seeds).pass_seq
            else:
                rhs = random_tir_passes()
            return lambda lhs : binary_func(lhs, rhs)

        mutator = random.choice([
            self.pass_mutator.single_point_mutate, 
            self.pass_mutator.subseq_mutate, 
            make_rhs(self.pass_mutator.single_point_crossover), 
            make_rhs(self.pass_mutator.two_point_crossover), 
            make_rhs(self.pass_mutator.uniform_crossover)])
        new_passes = mutator(pass_seq)
        valid_passes = tir_pass_graph.fix_target(new_passes)

        return valid_passes

    def random_pick(self):
        idx = random.randint(0, len(self.seeds) - 1)
        return idx, self.seeds[idx]

    def delete(self, idx):
        self.seeds.pop(idx)

    def size(self) -> int:
        return len(self.seeds)
