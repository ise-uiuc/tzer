from typing import Any, Dict, List
from tvm import ir, tir
from .node import LazyDomNode
from .basic_type import *
from . import construct as cons
from tzer.tir.visit import TIRVisitor


def _block_cons(iter_vars, reads,
                writes, name_hint,
                body, init, alloc_buffers,
                match_buffers, annotations):
    return tir.Block(
        iter_vars,
        reads,
        writes,
        name_hint,
        body,
        init,
        alloc_buffers,
        match_buffers,
        annotations
    )


def _list(*args):
    return [*args]


def _producerrealize_cons(producer, bounds, condition, body):
    return tir.ProducerRealize(
        producer, bounds, condition, body)


def _producerstore_cons(prod, value, indices):
    return tir.ProducerStore(
        prod, value, indices)


def _bufferrealize_cons(buffer, bounds, condition, body):
    return tir.BufferRealize(
        buffer, bounds, condition, body)


def _bufferstore_cons(buffer, value, indices):
    return tir.BufferStore(
        buffer, value, indices)


def _attrstmt_cons(node, attr_key, value, body):
    return tir.AttrStmt(node, attr_key, value, body)


def _broadcast_cons(value, lanes):
    return tir.Broadcast(value, lanes)


def _primfunc_cons(params, body, ret_type, buffer_map, attrs):
    return tir.PrimFunc(
        params,
        body,
        ret_type,
        buffer_map,
        attrs,
    )


def _for_cons(loop_var, min_val, extent, kind, body, thread_binding, annotations):
    return tir.For(
        loop_var,
        min_val,
        extent,
        kind,
        body,
        thread_binding,
        annotations,
    )


def _inj(op):
    return op


def _ramp_cons(base, stride, lanes):
    return tir.Ramp(base, stride, lanes)


def _blockrealize_cons(iters, predicate, block):
    return tir.BlockRealize(
        iters,
        predicate,
        block
    )


class Laze(TIRVisitor[LazyDomNode, Any]):
    """TIR -> LazyDomNode"""

    def visit_primfunc(self, op: tir.PrimFunc, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimFunc, cons.SomeCons(
            id='PrimFunc',
            target_dom=tir.PrimFunc,
            src_doms=[List[tir.Var], tir.Stmt, cons.SomeDom(
                'ret_type'), Dict[tir.Var, tir.Buffer], cons.SomeDom('attrs')],
            cons_def=_primfunc_cons), None, [
            LazyDomNode(List[tir.Var], None, op.params, []),
            self(op.body, None),
            LazyDomNode(cons.SomeDom('ret_type'), None, op.ret_type, []),
            LazyDomNode(Dict[tir.Var, tir.Buffer], None, op.buffer_map, []),
            LazyDomNode(cons.SomeDom('attrs'), None, op.attrs, []),
        ])

    def visit_var(self, op: tir.Var, arg) -> LazyDomNode:
        return LazyDomNode.make_nonleaf(tir.PrimExpr, cons.VarInj(), [
            LazyDomNode(tir.Var, None, op, [])
        ])

    def visit_sizevar(self, op: tir.SizeVar, arg) -> LazyDomNode:
        return self.visit_var(op, arg)

    def visit_itervar(self, op: tir.IterVar, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, cons.IterVar(), None, [
            LazyDomNode(ir.Range, None, op.dom, []),
            self(op.var, None),
            LazyDomNode(IterType, None, op.iter_type, []),
            LazyDomNode(ThreadTag, None, op.thread_tag, []),
        ])

    def visit_load(self, op: tir.Load, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, cons.Load(), None, [
            LazyDomNode(str, None, op.dtype, []),
            self(op.buffer_var, None),
            self(op.index, None),
            self(op.predicate, None)
        ])

    def visit_let(self, op: tir.Let, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, cons.Let(), None, [
            self(op.var, None),
            self(op.value, None),
            self(op.body, None)
        ])

    def visit_bufferload(self, op: tir.BufferLoad, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, cons.BufferLoad(), None, [
            LazyDomNode(tir.Buffer, None, op.buffer, []),
            self.make_node_for_list(op.indices)
        ])

    def visit_producerload(self, op: tir.ProducerLoad, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, None, op, [])

    def visit_call(self, op: tir.Call, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, None, op, [])

    def visit_add(self, op: tir.Add, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, cons.Add(), None, [
            self(op.a, None),
            self(op.b, None)
        ])

    def visit_sub(self, op: tir.Sub, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, cons.Sub(), None, [
            self(op.a, None),
            self(op.b, None)
        ])

    def visit_mul(self, op: tir.Mul, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, cons.Mul(), None, [
            self(op.a, None),
            self(op.b, None)
        ])

    def visit_div(self, op: tir.Div, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, cons.Div(), None, [
            self(op.a, None),
            self(op.b, None)
        ])

    def visit_mod(self, op: tir.Mod, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, cons.Mod(), None, [
            self(op.a, None),
            self(op.b, None)
        ])

    def visit_floordiv(self, op: tir.FloorDiv, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, cons.FloorDiv(), None, [
            self(op.a, None),
            self(op.b, None)
        ])

    def visit_floormod(self, op: tir.FloorMod, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, cons.FloorMod(), None, [
            self(op.a, None),
            self(op.b, None)
        ])

    def visit_min(self, op: tir.Min, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, cons.Min(), None, [
            self(op.a, None),
            self(op.b, None)
        ])

    def visit_max(self, op: tir.Max, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, cons.Max(), None, [
            self(op.a, None),
            self(op.b, None)
        ])

    def visit_eq(self, op: tir.EQ, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, cons.EQ(), None, [
            self(op.a, None),
            self(op.b, None)
        ])

    def visit_ne(self, op: tir.NE, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, cons.NE(), None, [
            self(op.a, None),
            self(op.b, None)
        ])

    def visit_lt(self, op: tir.LT, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, cons.LT(), None, [
            self(op.a, None),
            self(op.b, None)
        ])

    def visit_le(self, op: tir.LE, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, cons.LE(), None, [
            self(op.a, None),
            self(op.b, None)
        ])

    def visit_gt(self, op: tir.GT, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, cons.GT(), None, [
            self(op.a, None),
            self(op.b, None)
        ])

    def visit_ge(self, op: tir.GE, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, cons.GE(), None, [
            self(op.a, None),
            self(op.b, None)
        ])

    def visit_and(self, op: tir.And, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, cons.And(), None, [
            self(op.a, None),
            self(op.b, None)
        ])

    def visit_or(self, op: tir.Or, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, cons.Or(), None, [
            self(op.a, None),
            self(op.b, None)
        ])

    def visit_reduce(self, op: tir.Reduce, arg) -> LazyDomNode:
        return LazyDomNode(tir.Reduce, None, op, [])

    def visit_cast(self, op: tir.Cast, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, cons.Cast(), None, [
            LazyDomNode(str, None, op.dtype, []),
            self(op.value, None)
        ])

    def visit_not(self, op: tir.Not, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, cons.Not(), None, [
            self(op.a, None)])

    def visit_select(self, op: tir.Select, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, cons.Select(), None, [
            self(op.condition, None),
            self(op.true_value, None),
            self(op.false_value, None)
        ])

    def visit_ramp(self, op: tir.Ramp, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, cons.Ramp(), None, [
            self(op.base, None),
            self(op.stride, None),
            LazyDomNode(Lanes, None, op.lanes, []),
        ])

    def visit_broadcast(self, op: tir.Broadcast, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, cons.Broadcast(), None, [
            self(op.value, None),
            LazyDomNode(Lanes, None, op.lanes, []),
        ])

    def visit_shuffle(self, op: tir.Shuffle, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, None, op, [])

    def visit_intimm(self, op: tir.IntImm, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, None, op, [])

    def visit_floatimm(self, op: tir.FloatImm, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, None, op, [])

    def visit_stringimm(self, op: tir.StringImm, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, None, op, [])

    def visit_any(self, op: tir.Any, arg) -> LazyDomNode:
        return LazyDomNode(tir.PrimExpr, None, op, [])

    def visit_attrstmt(self, op: tir.AttrStmt, arg) -> LazyDomNode:
        return LazyDomNode(tir.Stmt, cons.AttrStmt, None, [
            self(op.node, None),
            LazyDomNode(AttrKey, None, op.attr_key, []),
            self(op.value, None),
            self(op.body, None)
        ])

    def visit_ifthenelse(self, op: tir.IfThenElse, arg) -> LazyDomNode:
        return LazyDomNode(tir.Stmt, cons.IfThenElse(), None, [
            self(op.condition, None),
            self(op.then_case, None),
            self(op.else_case, None) if op.else_case is not None else LazyDomNode(
                tir.Stmt, None, tir.Evaluate(tir.const(0)), []
            )
        ])

    def visit_letstmt(self, op: tir.LetStmt, arg) -> LazyDomNode:
        return LazyDomNode(tir.Stmt, cons.LetStmt(), None, [
            LazyDomNode(tir.Var, None, op.var, []),
            self(op.value, None),
            self(op.body, None)
        ])

    def visit_for(self, op: tir.For, arg) -> LazyDomNode:
        return LazyDomNode(tir.Stmt, cons.SomeCons(
            'For',
            target_dom=tir.Stmt,
            src_doms=[tir.Var, tir.PrimExpr, tir.PrimExpr, tir.ForKind, tir.Stmt, cons.SomeDom('thread_binding'),
                      cons.SomeDom('annotations')],
            cons_def=_for_cons), None, [
            LazyDomNode(tir.Var, None, op.loop_var, []),
            self(op.min, None),
            self(op.extent, None),
            LazyDomNode(tir.ForKind, None, op.kind, []),
            self(op.body, None),
            LazyDomNode(cons.SomeDom('thread_binding'),
                        None, op.thread_binding, []),
            LazyDomNode(cons.SomeDom('annotations'),
                        None, op.annotations, []),
        ])

    def visit_while(self, op: tir.stmt.While, arg) -> LazyDomNode:
        return LazyDomNode(tir.stmt.Stmt, cons.While(), None, [
            self(op.condition, None),
            self(op.body, None)
        ])

    def visit_allocate(self, op: tir.Allocate, arg) -> LazyDomNode:
        return LazyDomNode(tir.Stmt, cons.Allocate(), None, [
            LazyDomNode(tir.Var, None, op.buffer_var, []),
            LazyDomNode(str, None, op.dtype, []),
            self.make_node_for_list(op.extents),
            self(op.condition, None),
            self(op.body, None)
        ])

    def visit_store(self, op: tir.Store, arg) -> LazyDomNode:
        return LazyDomNode(tir.Stmt, cons.Store(), None, [
            LazyDomNode(tir.Var, None, op.buffer_var, []),
            self(op.value, None),
            self(op.index, None),
            self(op.predicate, None)
        ])

    def visit_bufferstore(self, op: tir.BufferStore, arg) -> LazyDomNode:
        return LazyDomNode(tir.Stmt, cons.SomeCons(
            'BufferStore',
            tir.Stmt,
            [tir.Buffer, tir.PrimExpr, List[tir.PrimExpr]],
            _bufferstore_cons,
        ), None, [
            LazyDomNode(tir.Buffer, None, op.buffer, []),
            self(op.value, None),
            self.make_node_for_list(op.indices),
        ])

    def visit_bufferrealize(self, op: tir.BufferRealize, arg) -> LazyDomNode:
        return LazyDomNode(tir.Stmt, cons.SomeCons(
            'BufferRealize',
            tir.Stmt,
            [tir.Buffer, cons.SomeDom('bounds'), tir.PrimExpr, tir.Stmt],
            _bufferrealize_cons
        ), None, [
            LazyDomNode(tir.Buffer, None, op.buffer, []),
            LazyDomNode(cons.SomeDom('bounds'), None, op.bounds, []),
            self(op.condition, None),
            self(op.body, None)
        ])

    def visit_assertstmt(self, op: tir.AssertStmt, arg) -> LazyDomNode:
        return LazyDomNode(tir.Stmt, cons.AssertStmt(), None, [
            self(op.condition, None),
            self(op.message, None),
            self(op.body, None)
        ])

    def visit_producerstore(self, op: tir.ProducerStore, arg) -> LazyDomNode:
        return LazyDomNode(tir.Stmt, cons.SomeCons(
            'ProducerStore',
            tir.Stmt,
            [cons.SomeDom('producer'), tir.PrimExpr, List[tir.PrimExpr]],
            _producerstore_cons
        ), op, [
            LazyDomNode(cons.SomeDom('producer'), None, op.producer, []),
            self(op.value, None),
            self.make_node_for_list(op.indices)
        ])

    def visit_producerrealize(self, op: tir.ProducerRealize, arg) -> LazyDomNode:
        return LazyDomNode(tir.Stmt, cons.SomeCons(
            'ProducerRealize',
            tir.ProducerRealize,
            [cons.SomeDom('producer'), cons.SomeDom(
                'bounds'), tir.PrimExpr, tir.Stmt],
            _producerrealize_cons
        ), op, [
            self(op.producer, None),
            self(op.bounds, None),
            self(op.condition, None),
            self(op.body, None)
        ])

    def visit_prefetch(self, op: tir.Prefetch, arg) -> LazyDomNode:
        return LazyDomNode(tir.Stmt, None, op, [])

    def visit_seqstmt(self, op: tir.SeqStmt, arg) -> LazyDomNode:
        return LazyDomNode(tir.Stmt, cons.SeqStmt(), None, [
            self.make_node_for_list(op.seq, List[tir.Stmt], tir.Stmt),
        ])

    def visit_evaluate(self, op: tir.Evaluate, arg) -> LazyDomNode:
        return LazyDomNode(tir.Stmt, cons.Evaluate(), None, [
            self(op.value, None)
        ])

    def visit_block(self, op: tir.Block, arg) -> LazyDomNode:
        """Returns tir.Stmt if arg is None; otherwise tir.Block"""
        nop = LazyDomNode.make_leaf(tir.Evaluate, tir.Evaluate(tir.const(0)))
        node = LazyDomNode(tir.Block, cons.SomeCons(
            'Block',
            tir.Block,
            [cons.SomeDom('iter_vars'),
             cons.SomeDom('reads'),
             cons.SomeDom('writes'),
             cons.SomeDom('name_hint'),
             tir.Stmt,
             tir.Stmt,
             cons.SomeDom('alloc_buffers'),
             cons.SomeDom('match_buffers'),
             cons.SomeDom('annotations'), ],
            _block_cons
        ), None, [
            LazyDomNode(cons.SomeDom('iter_vars'), None, op.iter_vars, []),
            LazyDomNode(cons.SomeDom('reads'), None, op.reads, []),
            LazyDomNode(cons.SomeDom('writes'), None, op.writes, []),
            LazyDomNode(cons.SomeDom('name_hint'), None, op.name_hint, []),
            self(op.body, None),
            self(op.init, None) if op.init is not None else nop,
            LazyDomNode(cons.SomeDom('alloc_buffers'),
                        None, op.alloc_buffers, []),
            LazyDomNode(cons.SomeDom('match_buffers'),
                        None, op.match_buffers, []),
            LazyDomNode(cons.SomeDom('annotations'), None, op.annotations, []),
        ])
        return LazyDomNode(tir.Stmt, cons.SomeCons(
            'BlockInj',
            tir.Stmt,
            [tir.Block],
            _inj
        ), None, [node]) if arg is None else node

    def visit_blockrealize(self, op: tir.BlockRealize, arg) -> LazyDomNode:
        return LazyDomNode(tir.Stmt, cons.SomeCons(
            'BlockRealize',
            tir.BlockRealize,
            [List[tir.PrimExpr], tir.PrimExpr, tir.Block],
            _blockrealize_cons,
        ), None, [
            self.make_node_for_list(op.iter_values),
            self(op.predicate, None) if not isinstance(
                op, bool) else self(tir.const(op.predicate, dtype='bool'), None),  # type: ignore
            self(op.block, True)
        ])

    def make_node_for_list(self, lst: list, lst_type: type = List[tir.PrimExpr], item_type: type = tir.PrimExpr) -> LazyDomNode:
        return LazyDomNode(lst_type, cons.SomeCons(
            f'{lst_type}: {len(lst)}',
            lst_type,
            [item_type for _ in lst],
            _list
        ), None, [self(index, None) for index in lst])


_LAZE = Laze()


def laze(func: tir.PrimFunc) -> LazyDomNode:
    return _LAZE(func, None)
