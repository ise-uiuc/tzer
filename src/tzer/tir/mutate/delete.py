from typing import Callable, List
from tvm import tir
from tzer.tir.semantic.constraint import satisfies
from tzer.tir.util import TIRNode
import random
from tzer.tir.semantic import Constraint, Context
from tzer.tir import semantic
from tzer.tir.visit.abstract import TIRVisitor
from .mutator import Mutator


def filter_satisfied_options(
    options: List[TIRNode],
    constraint: Constraint,
) -> List[TIRNode]:
    return [option for option in options if satisfies(option, constraint)]


class SubIRFilter(TIRVisitor[List[TIRNode], Constraint]):
    """Returns sub-IRs that satisfy the constraint"""
    def decorate(visit_func: Callable[['SubIRFilter', TIRNode, Constraint], List[TIRNode]]):  # type: ignore
        def wrapper_visit(self: 'SubIRFilter', op: TIRNode, constraint: Constraint):
            options = visit_func(self, op, constraint)
            return [option for option in options if satisfies(option, constraint)]
        return wrapper_visit

    def visit_primfunc(self, op: tir.PrimFunc, constraint: Constraint) -> List[TIRNode]:
        return []

    def visit_var(self, op: tir.Var, constraint: Constraint) -> List[TIRNode]:
        return []

    def visit_sizevar(self, op: tir.SizeVar, constraint: Constraint) -> List[TIRNode]:
        return []

    def visit_itervar(self, op: tir.IterVar, constraint: Constraint) -> List[TIRNode]:
        return []

    def visit_load(self, op: tir.Load, constraint: Constraint) -> List[TIRNode]:
        return [op.index, op.predicate]

    def visit_bufferload(self, op: tir.BufferLoad, constraint: Constraint) -> List[TIRNode]:
        return list(op.indices)

    def visit_producerload(self, op: tir.ProducerLoad, constraint: Constraint) -> List[TIRNode]:
        return list(op.indices)

    def visit_let(self, op: tir.Let, constraint: Constraint) -> List[TIRNode]:
        return [op.value]

    def visit_call(self, op: tir.Call, constraint: Constraint) -> List[TIRNode]:
        return list(op.args)

    def visit_binop(self, op, constraint: Constraint) -> List[TIRNode]:
        return [op.a, op.b]

    def visit_add(self, op: tir.Add, constraint: Constraint) -> List[TIRNode]:
        return self.visit_binop(op, constraint)

    def visit_sub(self, op: tir.Sub, constraint: Constraint) -> List[TIRNode]:
        return self.visit_binop(op, constraint)

    def visit_mul(self, op: tir.Mul, constraint: Constraint) -> List[TIRNode]:
        return self.visit_binop(op, constraint)

    def visit_div(self, op: tir.Div, constraint: Constraint) -> List[TIRNode]:
        return self.visit_binop(op, constraint)

    def visit_mod(self, op: tir.Mod, constraint: Constraint) -> List[TIRNode]:
        return self.visit_binop(op, constraint)

    def visit_floordiv(self, op: tir.FloorDiv, constraint: Constraint) -> List[TIRNode]:
        return self.visit_binop(op, constraint)

    def visit_floormod(self, op: tir.FloorMod, constraint: Constraint) -> List[TIRNode]:
        return self.visit_binop(op, constraint)

    def visit_min(self, op: tir.Min, constraint: Constraint) -> List[TIRNode]:
        return self.visit_binop(op, constraint)

    def visit_max(self, op: tir.Max, constraint: Constraint) -> List[TIRNode]:
        return self.visit_binop(op, constraint)

    def visit_eq(self, op: tir.EQ, constraint: Constraint) -> List[TIRNode]:
        return self.visit_binop(op, constraint)

    def visit_ne(self, op: tir.NE, constraint: Constraint) -> List[TIRNode]:
        return self.visit_binop(op, constraint)

    def visit_lt(self, op: tir.LT, constraint: Constraint) -> List[TIRNode]:
        return self.visit_binop(op, constraint)

    def visit_le(self, op: tir.LE, constraint: Constraint) -> List[TIRNode]:
        return self.visit_binop(op, constraint)

    def visit_gt(self, op: tir.GT, constraint: Constraint) -> List[TIRNode]:
        return self.visit_binop(op, constraint)

    def visit_ge(self, op: tir.GE, constraint: Constraint) -> List[TIRNode]:
        return self.visit_binop(op, constraint)

    def visit_and(self, op: tir.And, constraint: Constraint) -> List[TIRNode]:
        return self.visit_binop(op, constraint)

    def visit_or(self, op: tir.Or, constraint: Constraint) -> List[TIRNode]:
        return self.visit_binop(op, constraint)

    def visit_reduce(self, op: tir.Reduce, constraint: Constraint) -> List[TIRNode]:
        return [
            op.condition,
            op.init,
            *list(op.src),
        ]

    def visit_cast(self, op: tir.Cast, constraint: Constraint) -> List[TIRNode]:
        return [op.value]

    def visit_not(self, op: tir.Not, constraint: Constraint) -> List[TIRNode]:
        return [op.a]

    def visit_select(self, op: tir.Select, constraint: Constraint) -> List[TIRNode]:
        return [op.condition, op.true_value, op.false_value]

    def visit_ramp(self, op: tir.Ramp, constraint: Constraint) -> List[TIRNode]:
        return [op.base]

    def visit_broadcast(self, op: tir.Broadcast, constraint: Constraint) -> List[TIRNode]:
        return [op.value]

    def visit_shuffle(self, op: tir.Shuffle, constraint: Constraint) -> List[TIRNode]:
        return [
            *list(op.vectors),
            *list(op.indices),
        ]

    def visit_intimm(self, op: tir.IntImm, constraint: Constraint) -> List[TIRNode]:
        return []

    def visit_floatimm(self, op: tir.FloatImm, constraint: Constraint) -> List[TIRNode]:
        return []

    def visit_stringimm(self, op: tir.StringImm, constraint: Constraint) -> List[TIRNode]:
        return []

    def visit_any(self, op: tir.Any, constraint: Constraint) -> List[TIRNode]:
        return []

    def visit_attrstmt(self, op: tir.AttrStmt, constraint: Constraint) -> List[TIRNode]:
        return [op.value]

    def visit_ifthenelse(self, op: tir.IfThenElse, constraint: Constraint) -> List[TIRNode]:
        return [op.then_case] + [op.else_case] if op.else_case is not None else []

    def visit_letstmt(self, op: tir.LetStmt, constraint: Constraint) -> List[TIRNode]:
        return []

    def visit_for(self, op: tir.For, constraint: Constraint) -> List[TIRNode]:
        return [op.body]

    def visit_while(self, op: tir.stmt.While, constraint: Constraint) -> List[TIRNode]:
        return [op.body]

    def visit_allocate(self, op: tir.Allocate, constraint: Constraint) -> List[TIRNode]:
        return [op.body]

    def visit_store(self, op: tir.Store, constraint: Constraint) -> List[TIRNode]:
        return []

    def visit_bufferstore(self, op: tir.BufferStore, constraint: Constraint) -> List[TIRNode]:
        return []

    def visit_bufferrealize(self, op: tir.BufferRealize, constraint: Constraint) -> List[TIRNode]:
        return [op.body]

    def visit_assertstmt(self, op: tir.AssertStmt, constraint: Constraint) -> List[TIRNode]:
        return [op.body]

    def visit_producerstore(self, op: tir.ProducerStore, constraint: Constraint) -> List[TIRNode]:
        return []

    def visit_producerrealize(self, op: tir.ProducerRealize, constraint: Constraint) -> List[TIRNode]:
        return [op.body]

    def visit_prefetch(self, op: tir.Prefetch, constraint: Constraint) -> List[TIRNode]:
        return []

    def visit_seqstmt(self, op: tir.SeqStmt, constraint: Constraint) -> List[TIRNode]:
        return list(op.seq)

    def visit_evaluate(self, op: tir.Evaluate, constraint: Constraint) -> List[TIRNode]:
        return []

    def visit_block(self, op: tir.Block, constraint: Constraint) -> List[TIRNode]:
        return [op.body]

    def visit_blockrealize(self, op: tir.BlockRealize, constraint: Constraint) -> List[TIRNode]:
        return [op.block]


SUB_IR_FILTER = SubIRFilter()


class Deletor(Mutator):
    def mutate(self, op: TIRNode, context: Context) -> TIRNode:
        satisfied = SUB_IR_FILTER(op, context.constraint)
        return op if len(satisfied) == 0 else random.choice(satisfied)

    def will_modify(self, op: TIRNode, context: Context) -> bool:
        return len(SUB_IR_FILTER(op, context.constraint)) > 0
