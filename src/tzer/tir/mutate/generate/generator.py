from tzer.tir.util import TIRNode
from tzer.tir.semantic import Context
# from tzer.tir.mutate.mutator import Mutator


class Generator:
    def generate(self, context: Context) -> TIRNode:
        """Generate the node according to the context"""
        raise NotImplementedError

    def can_generate(self, op_type: type, context: Context) -> bool:
        return True
