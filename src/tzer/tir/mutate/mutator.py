from tvm import tir
from tzer.tir.util import TIRNode
from tzer.tir.semantic import Context


class IRMutator:
    def mutate_ir(self, op: tir.PrimFunc) -> tir.PrimFunc:
        raise NotImplementedError


class Mutator:
    def mutate(self, op: TIRNode, context: Context) -> TIRNode:
        raise NotImplementedError

    def will_modify(self, op: TIRNode, context: Context) -> bool:
        """See if the mutator can really modify op by mutation instead of just returning it.
        NOTE: this methods should have an O(1) complexity, and if not possible, just return True"""
        # raise NotImplementedError
        return True
