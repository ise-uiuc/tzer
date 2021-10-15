from .mutator import Mutator
from tzer.tir.util import TIRNode
from tzer.tir.semantic import Context


class Nilizer(Mutator):
    """any -> None"""

    def mutate(self, op: TIRNode, context: Context) -> TIRNode:
        return None  # type: ignore

    def will_modify(self, op: TIRNode, context: Context) -> bool:
        return True
