from typing import List, Tuple
from tvm import tir
from .mutator import IRMutator
from tzer.tir import util


class WeightedIRMutatorCombinator(IRMutator):
    def __init__(self, weighted_ir_mutators: List[Tuple[int, IRMutator]]) -> None:
        self.weighted_ir_mutators = weighted_ir_mutators

    def mutate_ir(self, op: tir.PrimFunc) -> tir.PrimFunc:
        return util.weighted_select(self.weighted_ir_mutators).mutate_ir(op)
