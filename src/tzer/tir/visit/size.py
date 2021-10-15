from typing import Any, Dict, Optional
from tzer.tir.util import TIRNode, get_id
from .abstract import TIRVisitor
from tvm import tir


class MemoizedGetSize(TIRVisitor[int, None]):
    def __init__(self) -> None:
        self.size_map: Dict[Optional[int], int] = {}

    def decorate(visit_func: Any):
        def wrapper_visit(self, op, arg):
            id = get_id(op)
            # memoization
            if id in self.size_map:
                return self.size_map[id]
            else:
                size = visit_func(self, op, arg)
                self.size_map[id] = size
                return size
        return wrapper_visit

    def visit_primfunc(self, op: tir.PrimFunc, arg: None) -> int:
        return 1 + self(op.body, None)

    def visit_var(self, op: tir.Var, arg: None) -> int:
        return 1

    def visit_sizevar(self, op: tir.SizeVar, arg: None) -> int:
        return 1

    def visit_itervar(self, op: tir.IterVar, arg: None) -> int:
        return 2

    def visit_load(self, op: tir.Load, arg: None) -> int:
        return 1 + self(op.buffer_var, None) + self(op.index, None) + self(op.predicate, None)

    def visit_let(self, op: tir.Let, arg: None) -> int:
        return 1 + self(op.var, None) + self(op.value, None) + self(op.body, None)

    def visit_bufferload(self, op: tir.BufferLoad, arg: None) -> int:
        sum = 1
        for index in op.indices:
            sum += self(index, None)
        return sum

    def visit_producerload(self, op: tir.ProducerLoad, arg: None) -> int:
        sum = 1
        for index in op.indices:
            sum += self(index, None)
        return sum

    def visit_call(self, op: tir.Call, arg: None) -> int:
        sum = 1
        for arg in op.args:
            sum += self(arg, None)
        return sum

    def visit_add(self, op: tir.Add, arg: None) -> int:
        return self(op.a, None) + self(op.b, None) + 1

    def visit_sub(self, op: tir.Sub, arg: None) -> int:
        return self(op.a, None) + self(op.b, None) + 1

    def visit_mul(self, op: tir.Mul, arg: None) -> int:
        return self(op.a, None) + self(op.b, None) + 1

    def visit_div(self, op: tir.Div, arg: None) -> int:
        return self(op.a, None) + self(op.b, None) + 1

    def visit_mod(self, op: tir.Mod, arg: None) -> int:
        return self(op.a, None) + self(op.b, None) + 1

    def visit_floordiv(self, op: tir.FloorDiv, arg: None) -> int:
        return self(op.a, None) + self(op.b, None) + 1

    def visit_floormod(self, op: tir.FloorMod, arg: None) -> int:
        return self(op.a, None) + self(op.b, None) + 1

    def visit_min(self, op: tir.Min, arg: None) -> int:
        return self(op.a, None) + self(op.b, None) + 1

    def visit_max(self, op: tir.Max, arg: None) -> int:
        return self(op.a, None) + self(op.b, None) + 1

    def visit_eq(self, op: tir.EQ, arg: None) -> int:
        return self(op.a, None) + self(op.b, None) + 1

    def visit_ne(self, op: tir.NE, arg: None) -> int:
        return self(op.a, None) + self(op.b, None) + 1

    def visit_lt(self, op: tir.LT, arg: None) -> int:
        return self(op.a, None) + self(op.b, None) + 1

    def visit_le(self, op: tir.LE, arg: None) -> int:
        return self(op.a, None) + self(op.b, None) + 1

    def visit_gt(self, op: tir.GT, arg: None) -> int:
        return self(op.a, None) + self(op.b, None) + 1

    def visit_ge(self, op: tir.GE, arg: None) -> int:
        return self(op.a, None) + self(op.b, None) + 1

    def visit_and(self, op: tir.And, arg: None) -> int:
        return self(op.a, None) + self(op.b, None) + 1

    def visit_or(self, op: tir.Or, arg: None) -> int:
        return self(op.a, None) + self(op.b, None) + 1

    def visit_reduce(self, op: tir.Reduce, arg: None) -> int:
        sum = 1
        for src in op.src:
            sum += self(src, None)
        sum += self(op.condition, None)
        sum += self(op.value_index, None)
        for init in op.init:
            sum += self(init, None)
        return sum

    def visit_cast(self, op: tir.Cast, arg: None) -> int:
        return 1 + self(op.value, None)

    def visit_not(self, op: tir.Not, arg: None) -> int:
        return 1 + self(op.a, None)

    def visit_select(self, op: tir.Select, arg: None) -> int:
        return 1 + self(op.condition, arg) + self(op.true_value, arg) + self(op.false_value, arg)

    def visit_ramp(self, op: tir.Ramp, arg: None) -> int:
        return 1 + self(op.base, arg)

    def visit_broadcast(self, op: tir.Broadcast, arg: None) -> int:
        return 1 + self(op.value, arg)

    def visit_shuffle(self, op: tir.Shuffle, arg: None) -> int:
        sum = 1
        for vec in op.vectors:
            sum += self(vec, None)
        for index in op.indices:
            sum += self(index, None)
        return sum

    def visit_intimm(self, op: tir.IntImm, arg: None) -> int:
        return 1

    def visit_floatimm(self, op: tir.FloatImm, arg: None) -> int:
        return 1

    def visit_stringimm(self, op: tir.StringImm, arg: None) -> int:
        return 1

    def visit_any(self, op: tir.Any, arg: None) -> int:
        return 1

    def visit_attrstmt(self, op: tir.AttrStmt, arg: None) -> int:
        return 1 + self(op.value, arg) + self(op.body, arg)

    def visit_ifthenelse(self, op: tir.IfThenElse, arg: None) -> int:
        sum = 1 + self(op.condition, None) + self(op.then_case, None)
        if op.else_case is not None:
            sum += self(op.else_case, None)
        return sum

    def visit_letstmt(self, op: tir.LetStmt, arg: None) -> int:
        return 1 + self(op.var, arg) + self(op.value, arg) + self(op.body, arg)

    def visit_for(self, op: tir.For, arg: None) -> int:
        return 1 + self(op.loop_var, arg) + self(op.min, arg) + self(op.extent, arg) + self(op.body, arg)

    def visit_while(self, op: tir.stmt.While, arg: None) -> int:
        return 1 + self(op.condition, arg) + self(op.body, arg)

    def visit_allocate(self, op: tir.Allocate, arg: None) -> int:
        sum = 1 + self(op.buffer_var, arg) + \
            self(op.condition, arg) + self(op.body, arg)
        for extent in op.extents:
            sum += self(extent, None)
        return sum

    def visit_store(self, op: tir.Store, arg: None) -> int:
        return 1 + self(op.buffer_var, arg) + self(op.value, arg) + self(op.index, arg) + self(op.predicate, arg)

    def visit_bufferstore(self, op: tir.BufferStore, arg: None) -> int:
        sum = 1 + self(op.value, None)
        for index in op.indices:
            sum += self(index, None)
        return sum

    def visit_bufferrealize(self, op: tir.BufferRealize, arg: None) -> int:
        return 1 + self(op.condition, arg) + self(op.body, arg)

    def visit_assertstmt(self, op: tir.AssertStmt, arg: None) -> int:
        return 1 + self(op.condition, arg) + self(op.message, arg) + self(op.body, arg)

    def visit_producerstore(self, op: tir.ProducerStore, arg: None) -> int:
        sum = 1 + self(op.value, arg)
        for index in op.indices:
            sum += self(index, None)
        return sum

    def visit_producerrealize(self, op: tir.ProducerRealize, arg: None) -> int:
        return 1 + self(op.condition, arg) + self(op.body, arg)

    def visit_prefetch(self, op: tir.Prefetch, arg: None) -> int:
        return 1

    def visit_seqstmt(self, op: tir.SeqStmt, arg: None) -> int:
        sum = 1
        for seq in op.seq:
            sum += self(seq, None)
        return sum

    def visit_evaluate(self, op: tir.Evaluate, arg: None) -> int:
        return 1 + self(op.value, arg)

    def visit_block(self, op: tir.Block, arg: None) -> int:
        sum = 1 + self(op.body, arg)
        if op.init is not None:
            sum += self(op.init, arg)
        return sum

    def visit_blockrealize(self, op: tir.BlockRealize, arg: None) -> int:
        sum = 1 + self(op.predicate, None) + self(op.block, None)
        for iter in op.iter_values:
            sum += self(iter, None)
        return sum


def get_node_size(root: TIRNode) -> int:
    return MemoizedGetSize()(root, None)
