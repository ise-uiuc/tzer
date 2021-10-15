from typing import Any, List
from tzer.tir.util import TIRNode, list_union
from .abstract import TIRVisitor
from tvm import tir


class TIRNodeTraverser(TIRVisitor[List[TIRNode], None]):
    def append_op(func: Any):
        def wrapper(self, op, arg):
            return [op] + func(self, op)
        return wrapper

    @append_op
    def visit_primfunc(self, op: tir.PrimFunc, arg=None) -> List[TIRNode]:
        return self(op.body, arg)

    @append_op
    def visit_var(self, op: tir.Var, arg=None) -> List[TIRNode]:
        return []

    @append_op
    def visit_sizevar(self, op: tir.SizeVar, arg=None) -> List[TIRNode]:
        return []

    @append_op
    def visit_itervar(self, op: tir.IterVar, arg=None) -> List[TIRNode]:
        return [
            *self(op.var, arg),
        ]

    @append_op
    def visit_load(self, op: tir.Load, arg=None) -> List[TIRNode]:
        return [
            *self(op.buffer_var, arg),
            *self(op.index, arg),
            *self(op.predicate, arg),
        ]

    @append_op
    def visit_let(self, op: tir.Let, arg=None) -> List[TIRNode]:
        return [
            *self(op.var, arg),
            *self(op.value, arg),
            *self(op.body, arg),
        ]

    @append_op
    def visit_bufferload(self, op: tir.BufferLoad, arg=None) -> List[TIRNode]:
        return list_union([self(index, arg) for index in op.indices])

    @append_op
    def visit_producerload(self, op: tir.ProducerLoad, arg=None) -> List[TIRNode]:
        return list_union([self(index, arg) for index in op.indices])

    @append_op
    def visit_call(self, op: tir.Call, arg=None) -> List[TIRNode]:
        return list_union([self(a, arg) for a in op.args])

    @append_op
    def visit_add(self, op: tir.Add, arg=None) -> List[TIRNode]:
        return [*self(op.a, arg), *self(op.b, arg)]

    @append_op
    def visit_sub(self, op: tir.Sub, arg=None) -> List[TIRNode]:
        return [*self(op.a, arg), *self(op.b, arg)]

    @append_op
    def visit_mul(self, op: tir.Mul, arg=None) -> List[TIRNode]:
        return [*self(op.a, arg), *self(op.b, arg)]

    @append_op
    def visit_div(self, op: tir.Div, arg=None) -> List[TIRNode]:
        return [*self(op.a, arg), *self(op.b, arg)]

    @append_op
    def visit_mod(self, op: tir.Mod, arg=None) -> List[TIRNode]:
        return [*self(op.a, arg), *self(op.b, arg)]

    @append_op
    def visit_floordiv(self, op: tir.FloorDiv, arg=None) -> List[TIRNode]:
        return [*self(op.a, arg), *self(op.b, arg)]

    @append_op
    def visit_floormod(self, op: tir.FloorMod, arg=None) -> List[TIRNode]:
        return [*self(op.a, arg), *self(op.b, arg)]

    @append_op
    def visit_min(self, op: tir.Min, arg=None) -> List[TIRNode]:
        return [*self(op.a, arg), *self(op.b, arg)]

    @append_op
    def visit_max(self, op: tir.Max, arg=None) -> List[TIRNode]:
        return [*self(op.a, arg), *self(op.b, arg)]

    @append_op
    def visit_eq(self, op: tir.EQ, arg=None) -> List[TIRNode]:
        return [*self(op.a, arg), *self(op.b, arg)]

    @append_op
    def visit_ne(self, op: tir.NE, arg=None) -> List[TIRNode]:
        return [*self(op.a, arg), *self(op.b, arg)]

    @append_op
    def visit_lt(self, op: tir.LT, arg=None) -> List[TIRNode]:
        return [*self(op.a, arg), *self(op.b, arg)]

    @append_op
    def visit_le(self, op: tir.LE, arg=None) -> List[TIRNode]:
        return [*self(op.a, arg), *self(op.b, arg)]

    @append_op
    def visit_gt(self, op: tir.GT, arg=None) -> List[TIRNode]:
        return [*self(op.a, arg), *self(op.b, arg)]

    @append_op
    def visit_ge(self, op: tir.GE, arg=None) -> List[TIRNode]:
        return [*self(op.a, arg), *self(op.b, arg)]

    @append_op
    def visit_and(self, op: tir.And, arg=None) -> List[TIRNode]:
        return [*self(op.a, arg), *self(op.b, arg)]

    @append_op
    def visit_or(self, op: tir.Or, arg=None) -> List[TIRNode]:
        return [*self(op.a, arg), *self(op.b, arg)]

    @append_op
    def visit_reduce(self, op: tir.Reduce, arg=None) -> List[TIRNode]:
        return [
            *list_union([self(s, arg) for s in op.src]),
            *self(op.condition, arg),
            self(op.value_index, arg),
            *list_union([self(i, arg) for i in op.init]),
        ]

    @append_op
    def visit_cast(self, op: tir.Cast, arg=None) -> List[TIRNode]:
        return self(op.value, arg)

    @append_op
    def visit_not(self, op: tir.Not, arg=None) -> List[TIRNode]:
        return self(op.a, arg)

    @append_op
    def visit_select(self, op: tir.Select, arg=None) -> List[TIRNode]:
        return [
            *self(op.condition, arg),
            *self(op.true_value, arg),
            *self(op.false_value, arg),
        ]

    @append_op
    def visit_ramp(self, op: tir.Ramp, arg=None) -> List[TIRNode]:
        return self(op.base, arg)

    @append_op
    def visit_broadcast(self, op: tir.Broadcast, arg=None) -> List[TIRNode]:
        return self(op.value, arg)

    @append_op
    def visit_shuffle(self, op: tir.Shuffle, arg=None) -> List[TIRNode]:
        return list_union([self(vec, arg) for vec in op.vectors])

    @append_op
    def visit_intimm(self, op: tir.IntImm, arg=None) -> List[TIRNode]:
        return []

    @append_op
    def visit_floatimm(self, op: tir.FloatImm, arg=None) -> List[TIRNode]:
        return []

    @append_op
    def visit_stringimm(self, op: tir.StringImm, arg=None) -> List[TIRNode]:
        return []

    @append_op
    def visit_any(self, op: tir.Any, arg=None) -> List[TIRNode]:
        return []

    @append_op
    def visit_attrstmt(self, op: tir.AttrStmt, arg=None) -> List[TIRNode]:
        return [
            *self(op.value, arg),
            *self(op.body, arg),
        ]

    @append_op
    def visit_ifthenelse(self, op: tir.IfThenElse, arg=None) -> List[TIRNode]:
        if op.else_case is None:
            return [*self(op.condition, arg), *self(op.then_case, arg)]
        return [
            *self(op.condition, arg),
            *self(op.then_case, arg),
            *self(op.else_case, arg),
        ]

    @append_op
    def visit_letstmt(self, op: tir.LetStmt, arg=None) -> List[TIRNode]:
        return [
            *self(op.var, arg),
            *self(op.value, arg),
            *self(op.body, arg),
        ]

    @append_op
    def visit_for(self, op: tir.For, arg=None) -> List[TIRNode]:
        return [
            *self(op.loop_var, arg),
            *self(op.min, arg),
            *self(op.extent, arg),
            *self(op.body, arg),
        ]

    @append_op
    def visit_while(self, op: tir.stmt.While, arg=None) -> List[TIRNode]:
        return [
            *self(op.condition, arg),
            *self(op.body, arg),
        ]

    @append_op
    def visit_allocate(self, op: tir.Allocate, arg=None) -> List[TIRNode]:
        return [
            *self(op.buffer_var, arg),
            *list_union([self(extent, arg) for extent in op.extents]),
            *self(op.condition, arg),
            *self(op.body, arg),
        ]

    @append_op
    def visit_store(self, op: tir.Store, arg=None) -> List[TIRNode]:
        return [
            *self(op.buffer_var, arg),
            *self(op.value, arg),
            *self(op.index, arg),
            *self(op.predicate, arg),
        ]

    @append_op
    def visit_bufferstore(self, op: tir.BufferStore, arg=None) -> List[TIRNode]:
        return [
            *self(op.value, arg),
            *list_union([self(index, arg) for index in op.indices]),
        ]

    @append_op
    def visit_bufferrealize(self, op: tir.BufferRealize, arg=None) -> List[TIRNode]:
        return [
            *self(op.condition, arg),
            *self(op.body, arg),
        ]

    @append_op
    def visit_assertstmt(self, op: tir.AssertStmt, arg=None) -> List[TIRNode]:
        return [
            *self(op.condition, arg),
            *self(op.message, arg),
            *self(op.body, arg),
        ]

    @append_op
    def visit_producerstore(self, op: tir.ProducerStore, arg=None) -> List[TIRNode]:
        return [
            *self(op.value, arg),
            *list_union([self(index, arg) for index in op.indices]),
        ]

    @append_op
    def visit_producerrealize(self, op: tir.ProducerRealize, arg=None) -> List[TIRNode]:
        return [
            *self(op.condition, arg),
            *self(op.body, arg),
        ]

    @append_op
    def visit_prefetch(self, op: tir.Prefetch, arg=None) -> List[TIRNode]:
        return []

    @append_op
    def visit_seqstmt(self, op: tir.SeqStmt, arg=None) -> List[TIRNode]:
        return list_union([self(s, arg) for s in op.seq])

    @append_op
    def visit_evaluate(self, op: tir.Evaluate, arg=None) -> List[TIRNode]:
        return self(op.value, arg)

    @append_op
    def visit_block(self, op: tir.Block, arg=None) -> List[TIRNode]:
        if op.init is None:
            return self(op.body, arg)
        else:
            return [
                *self(op.body, arg),
                *self(op.init, arg)
            ]

    @append_op
    def visit_blockrealize(self, op: tir.BlockRealize, arg=None) -> List[TIRNode]:
        return [
            *list_union([self(v, arg) for v in op.iter_values]),
            *self(op.predicate, arg),
            *self(op.block, arg),
        ]


TRAVERSER = TIRNodeTraverser()


def get_all_nodes(root: TIRNode) -> List[TIRNode]:
    return TRAVERSER(root, None)
