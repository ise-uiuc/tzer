from tzer.tir.semantic.context import Context
from tzer.tir.util import TIRNode
from .mutator import Mutator
from .generate import Generator, SizedGenerator
from tvm import tir


class Insertor(Mutator):
    def __init__(self, generator: SizedGenerator) -> None:
        self.generator = generator

    def will_modify(self, op: TIRNode, context: Context) -> bool:
        return self.generator.can_generate(type(op), context)

    def mutate(self, op: TIRNode, context: Context) -> TIRNode:
        # Do not generate leaf nodes for leaf nodes, which replacement would do.
        if isinstance(op, (
            tir.IntImm,
            tir.Var,
            tir.SizeVar,
            tir.FloatImm,
            tir.StringImm,
        )):
            generator = SizedGenerator(lambda: self.generator.get_size() + 1)
        else:
            generator = self.generator
        return generator.generate(context)
