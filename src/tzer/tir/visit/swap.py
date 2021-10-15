from .abstract import TIRAbstractTransformer
from tzer.tir.util import TIRNode
from typing import Any, NamedTuple


class ArgSwap(NamedTuple):
    # node to be swapped
    old: TIRNode
    new: TIRNode


class TIRSwapper(TIRAbstractTransformer[TIRNode]):
    def decorate(visit_func: Any):
        def wrapper_visit(self, op, arg: ArgSwap):
            if op.same_as(arg.old):
                return arg.new
            else:
                return visit_func(self, op, arg)
        return wrapper_visit


TIR_SWAPPER = TIRSwapper()


def swap_tir(root: TIRNode, old: TIRNode, new: TIRNode) -> TIRNode:
    return TIR_SWAPPER(root, ArgSwap(old, new))
