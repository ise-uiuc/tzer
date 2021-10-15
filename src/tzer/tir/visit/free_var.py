from typing import NamedTuple, Set
from tvm import tir
from tzer.tir.util import TIRNode
from .abstract import TIRVisitor


class ArgBound(NamedTuple):
    """Current known bound variables"""
    bound_vars: Set[tir.Var]


class TIRFreeVar(TIRVisitor[Set[tir.Var], ArgBound]):
    """Returns a set of free variables given bound variables"""

    def visit_primfunc(self, op: tir.PrimFunc, arg: ArgBound) -> Set[tir.Var]:
        return self(op.body, ArgBound(set(op.params)))

    def visit_var(self, op: tir.Var, arg: ArgBound) -> Set[tir.Var]:
        return {op} if not op in arg.bound_vars else set()

    def visit_sizevar(self, op: tir.SizeVar, arg: ArgBound) -> Set[tir.Var]:
        return self.visit_var(op, arg)

    def visit_itervar(self, op: tir.IterVar, arg: ArgBound) -> Set[tir.IterVar]:
        return self.visit_var(op.var, arg)

    def visit_load(self, op: tir.Load, arg: ArgBound) -> Set[tir.Var]:
        return self(op.buffer_var, arg) | self(op.index, arg) | self(op.predicate, arg)

    def visit_let(self, op: tir.Let, arg: ArgBound) -> Set[tir.Var]:
        new_arg = ArgBound(arg.bound_vars | {op.var})
        return self(op.var, arg) | self(op.value, arg) | self(op.body, new_arg)

    def visit_bufferload(self, op: tir.BufferLoad, arg: ArgBound) -> Set[tir.Var]:
        res: Set[tir.Var] = set()
        for index in op.indices:
            res |= self(index, arg)
        return res

    def visit_producerload(self, op: tir.ProducerLoad, arg: ArgBound) -> Set[tir.Var]:
        res: Set[tir.Var] = set()
        for index in op.indices:
            res |= self(index, arg)
        return res

    def visit_call(self, op: tir.Call, arg: ArgBound) -> Set[tir.Var]:
        res: Set[tir.Var] = set()
        for a in op.args:
            res |= self(a, arg)
        return res

    def visit_add(self, op: tir.Add, arg: ArgBound) -> Set[tir.Var]:
        return self(op.a, arg) | self(op.b, arg)

    def visit_sub(self, op: tir.Sub, arg: ArgBound) -> Set[tir.Var]:
        return self(op.a, arg) | self(op.b, arg)

    def visit_mul(self, op: tir.Mul, arg: ArgBound) -> Set[tir.Var]:
        return self(op.a, arg) | self(op.b, arg)

    def visit_div(self, op: tir.Div, arg: ArgBound) -> Set[tir.Var]:
        return self(op.a, arg) | self(op.b, arg)

    def visit_mod(self, op: tir.Mod, arg: ArgBound) -> Set[tir.Var]:
        return self(op.a, arg) | self(op.b, arg)

    def visit_floordiv(self, op: tir.FloorDiv, arg: ArgBound) -> Set[tir.Var]:
        return self(op.a, arg) | self(op.b, arg)

    def visit_floormod(self, op: tir.FloorMod, arg: ArgBound) -> Set[tir.Var]:
        return self(op.a, arg) | self(op.b, arg)

    def visit_min(self, op: tir.Min, arg: ArgBound) -> Set[tir.Var]:
        return self(op.a, arg) | self(op.b, arg)

    def visit_max(self, op: tir.Max, arg: ArgBound) -> Set[tir.Var]:
        return self(op.a, arg) | self(op.b, arg)

    def visit_eq(self, op: tir.EQ, arg: ArgBound) -> Set[tir.Var]:
        return self(op.a, arg) | self(op.b, arg)

    def visit_ne(self, op: tir.NE, arg: ArgBound) -> Set[tir.Var]:
        return self(op.a, arg) | self(op.b, arg)

    def visit_lt(self, op: tir.LT, arg: ArgBound) -> Set[tir.Var]:
        return self(op.a, arg) | self(op.b, arg)

    def visit_le(self, op: tir.LE, arg: ArgBound) -> Set[tir.Var]:
        return self(op.a, arg) | self(op.b, arg)

    def visit_gt(self, op: tir.GT, arg: ArgBound) -> Set[tir.Var]:
        return self(op.a, arg) | self(op.b, arg)

    def visit_ge(self, op: tir.GE, arg: ArgBound) -> Set[tir.Var]:
        return self(op.a, arg) | self(op.b, arg)

    def visit_and(self, op: tir.And, arg: ArgBound) -> Set[tir.Var]:
        return self(op.a, arg) | self(op.b, arg)

    def visit_or(self, op: tir.Or, arg: ArgBound) -> Set[tir.Var]:
        return self(op.a, arg) | self(op.b, arg)

    def visit_reduce(self, op: tir.Reduce, arg: ArgBound) -> Set[tir.Var]:
        res: Set[tir.Var] = set()
        for s in op.src:
            res |= self(s, arg)
        res |= self(op.condition, arg)
        res |= self(op.value_index, arg)
        for i in op.init:
            res |= self(i, arg)
        return res

    def visit_cast(self, op: tir.Cast, arg: ArgBound) -> Set[tir.Var]:
        return self(op.value, arg)

    def visit_not(self, op: tir.Not, arg: ArgBound) -> Set[tir.Var]:
        return self(op.a, arg)

    def visit_select(self, op: tir.Select, arg: ArgBound) -> Set[tir.Var]:
        return self(op.condition, arg) | self(op.true_value, arg) | self(op.false_value, arg)

    def visit_ramp(self, op: tir.Ramp, arg: ArgBound) -> Set[tir.Var]:
        return self(op.base, arg)

    def visit_broadcast(self, op: tir.Broadcast, arg: ArgBound) -> Set[tir.Var]:
        return self(op.value, arg)

    def visit_shuffle(self, op: tir.Shuffle, arg: ArgBound) -> Set[tir.Var]:
        res: Set[tir.Var] = set()
        for v in op.vectors:
            res |= self(v, arg)
        return res

    def visit_intimm(self, op: tir.IntImm, arg: ArgBound) -> Set[tir.Var]:
        return set()

    def visit_floatimm(self, op: tir.FloatImm, arg: ArgBound) -> Set[tir.Var]:
        return set()

    def visit_stringimm(self, op: tir.StringImm, arg: ArgBound) -> Set[tir.Var]:
        return set()

    def visit_any(self, op: tir.Any, arg: ArgBound) -> Set[tir.Var]:
        return set()

    def visit_attrstmt(self, op: tir.AttrStmt, arg: ArgBound) -> Set[tir.Var]:
        return self(op.value, arg) | self(op.body, arg)

    def visit_ifthenelse(self, op: tir.IfThenElse, arg: ArgBound) -> Set[tir.Var]:
        if op.else_case is None:
            return self(op.condition, arg) | self(op.then_case, arg)
        return self(op.condition, arg) | self(op.then_case, arg) | self(op.else_case, arg)

    def visit_letstmt(self, op: tir.LetStmt, arg: ArgBound) -> Set[tir.Var]:
        new_arg = ArgBound(arg.bound_vars | {op.var})
        return self(op.var, arg) | self(op.value, arg) | self(op.body, new_arg)

    def visit_for(self, op: tir.For, arg: ArgBound) -> Set[tir.Var]:
        new_arg = ArgBound(arg.bound_vars | {op.loop_var})
        return self(op.min, arg) | self(op.extent, arg) | self(op.body, new_arg)

    def visit_while(self, op: tir.stmt.While, arg: ArgBound) -> Set[tir.Var]:
        return self(op.condition, arg) | self(op.body, arg)

    def visit_allocate(self, op: tir.Allocate, arg: ArgBound) -> Set[tir.Var]:
        new_arg = ArgBound(arg.bound_vars | {op.buffer_var})
        res: Set[tir.Var] = set()
        res |= self(op.buffer_var, new_arg)
        for extent in op.extents:
            res |= self(extent, arg)
        res |= self(op.condition, arg)
        res |= self(op.body, new_arg)
        return res

    def visit_store(self, op: tir.Store, arg: ArgBound) -> Set[tir.Var]:
        return self(op.buffer_var, arg) | self(op.value, arg) | self(op.index, arg) | self(op.predicate, arg)

    def visit_bufferstore(self, op: tir.BufferStore, arg: ArgBound) -> Set[tir.Var]:
        res = self(op.value, arg)
        for index in op.indices:
            res |= self(index, arg)
        return res

    def visit_bufferrealize(self, op: tir.BufferRealize, arg: ArgBound) -> Set[tir.Var]:
        return self(op.condition, arg) | self(op.body, arg)

    def visit_assertstmt(self, op: tir.AssertStmt, arg: ArgBound) -> Set[tir.Var]:
        return self(op.condition, arg) | self(op.message, arg) | self(op.body, arg)

    def visit_producerstore(self, op: tir.ProducerStore, arg: ArgBound) -> Set[tir.Var]:
        res = self(op.value, arg)
        for index in op.indices:
            res |= self(index, arg)
        return res

    def visit_producerrealize(self, op: tir.ProducerRealize, arg: ArgBound) -> Set[tir.Var]:
        return self(op.condition, arg) | self(op.body, arg)

    def visit_prefetch(self, op: tir.Prefetch, arg: ArgBound) -> Set[tir.Var]:
        return set()

    def visit_seqstmt(self, op: tir.SeqStmt, arg: ArgBound) -> Set[tir.Var]:
        res = set()
        for s in op.seq:
            res |= self(s, arg)
        return res

    def visit_evaluate(self, op: tir.Evaluate, arg: ArgBound) -> Set[tir.Var]:
        return self(op.value, arg)

    def visit_block(self, op: tir.Block, arg: ArgBound) -> Set[tir.Var]:
        new_arg = ArgBound(
            arg.bound_vars | {v.var for v in op.iter_vars})
        return self(op.body, new_arg) if op.init is None else self(op.body, arg) | self(op.init, arg)

    def visit_blockrealize(self, op: tir.BlockRealize, arg: ArgBound) -> Set[tir.Var]:
        res: Set[tir.Var] = set()
        for v in op.iter_values:
            res |= self(v, arg)

        if not isinstance(op.predicate, bool):
            res |= self(op.predicate, arg)
        res |= self(op.block, arg)
        return res


TIR_FREE_VAR = TIRFreeVar()


def get_free_vars(root: TIRNode, bound_vars: Set[tir.Var] = set()) -> Set[tir.Var]:
    return TIR_FREE_VAR(root, ArgBound(bound_vars))


def primfunc_with_new_body(old_func: tir.PrimFunc, body: tir.Stmt) -> tir.PrimFunc:
    # vars in the buffer_map should be bound
    buffer_vars = {v.data for v in old_func.buffer_map.values()}

    params = list((get_free_vars(body) | {k for k in old_func.buffer_map}) - buffer_vars)

    return tir.PrimFunc(
        params,
        body,
        old_func.ret_type,
        old_func.buffer_map,
        old_func.attrs,
        old_func.span
    )
