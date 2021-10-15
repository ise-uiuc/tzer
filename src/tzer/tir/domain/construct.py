from typing import Any, Callable, Generic, List, Set, Type, Dict, TypeVar, Union, no_type_check
from tvm import ir, tir

from tzer.tir.domain.basic_type import *


class SomeDom:
    def __init__(self, id: str) -> None:
        self.id = id

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, SomeDom):
            return False
        return self.id == o.id

    def __hash__(self) -> int:
        return self.id.__hash__()

    def __str__(self) -> str:
        return f'SomeDom({self.id})'

    def __repr__(self) -> str:
        return self.__str__()


DomValue = Any
Dom = Union[type, SomeDom]


class Cons:
    target_dom: Dom
    src_doms: List[Dom]

    @no_type_check
    def __call__(self, *args: DomValue, **kwds: DomValue) -> DomValue:
        ...

    def __eq__(self, o: object) -> bool:
        return self.__class__ == o.__class__

    def __hash__(self) -> int:
        return hash(self.__class__)


class IntImmCons(Cons):
    target_dom = tir.IntImm
    src_doms = [tir.IntImm]

    def __call__(self, op: tir.IntImm) -> tir.IntImm:  # type: ignore
        ...


class ExprCons(Cons):
    target_dom = tir.PrimExpr


class IntImmInj(ExprCons):
    src_doms = [tir.IntImm]

    def __call__(self, op: tir.IntImm) -> tir.PrimExpr:  # type: ignore
        return op


class FloatImmInj(ExprCons):
    src_doms = [tir.FloatImm]

    def __call__(self, op: tir.FloatImm) -> tir.PrimExpr:  # type: ignore
        return op


class StringImmInj(ExprCons):
    src_doms = [tir.StringImm]

    def __call__(self, op: tir.StringImm) -> tir.PrimExpr:  # type: ignore
        return op


class VarInj(ExprCons):
    src_doms = [tir.Var]

    def __call__(self, op: tir.Var) -> tir.PrimExpr:  # type: ignore
        return op


class IterVar(ExprCons):
    src_doms = [ir.Range, tir.Var, IterType, ThreadTag]
    target_dom = tir.IterVar

    def __call__(self, dom: ir.Range, var: tir.Var, iter_type: IterType, thread_tag: ThreadTag) -> tir.PrimExpr:  # type: ignore
        return tir.IterVar(dom, var, iter_type, thread_tag)


class Cast(ExprCons):
    src_doms = [DType, tir.PrimExpr]

    def __call__(self, dtype: DType, op: tir.PrimExpr) -> tir.Cast:  # type: ignore
        return tir.Cast(dtype, op)


_T = TypeVar('_T')


class BinOp(Generic[_T], Cons):
    src_doms = [tir.PrimExpr, tir.PrimExpr]
    target_dom = tir.PrimExpr

    def __call__(self, lhs: tir.PrimExpr, rhs: tir.PrimExpr) -> _T:  # type: ignore
        return self.cls(lhs, rhs)  # type: ignore


class Add(BinOp[tir.Add]):
    cls = tir.Add


class Sub(BinOp[tir.Sub]):
    cls = tir.Sub


class Mul(BinOp[tir.Mul]):
    cls = tir.Mul


class Div(BinOp[tir.Div]):
    cls = tir.Div


class Mod(BinOp[tir.Mod]):
    cls = tir.Mod


class FloorDiv(BinOp[tir.FloorDiv]):
    cls = tir.FloorDiv


class FloorMod(BinOp[tir.FloorMod]):
    cls = tir.FloorMod


class Min(BinOp[tir.Min]):
    cls = tir.Min


class Max(BinOp[tir.Max]):
    cls = tir.Max


class EQ(BinOp[tir.EQ]):
    cls = tir.EQ


class NE(BinOp[tir.NE]):
    cls = tir.NE


class LT(BinOp[tir.LT]):
    cls = tir.LT


class LE(BinOp[tir.LE]):
    cls = tir.LE


class GT(BinOp[tir.GT]):
    cls = tir.GT


class GE(BinOp[tir.GE]):
    cls = tir.GE


class And(BinOp[tir.And]):
    cls = tir.And


class Or(BinOp[tir.Or]):
    cls = tir.Or


class Not(ExprCons):
    src_doms = [tir.PrimExpr]

    def __call__(self, op: tir.PrimExpr) -> tir.Not:  # type: ignore
        return tir.Not(op)


class Select(ExprCons):
    src_doms = [tir.PrimExpr, tir.PrimExpr, tir.PrimExpr]

    def __call__(self, cond: tir.PrimExpr, true_val: tir.PrimExpr, false_val: tir.PrimExpr) -> tir.Select:  # type: ignore
        return tir.Select(tir.Cast('bool', cond), true_val, false_val)


class Load(ExprCons):
    src_doms = [DType, tir.Var, tir.PrimExpr, tir.PrimExpr]

    def __call__(self, dtype: DType, bfvar: tir.Var, index: tir.PrimExpr, predicate: tir.PrimExpr) -> tir.Load:  # type: ignore
        return tir.Load(dtype, bfvar, index, predicate)


class Let(ExprCons):
    src_doms = [tir.Var, tir.PrimExpr, tir.PrimExpr]

    def __call__(self, var: tir.Var, value: tir.PrimExpr, body: tir.PrimExpr) -> tir.Let:  # type: ignore
        return tir.Let(var, value, body)


class Ramp(ExprCons):
    src_doms = [tir.PrimExpr, tir.PrimExpr, Lanes]

    def __call__(self, base: tir.PrimExpr, stride: tir.PrimExpr, lanes: Lanes) -> tir.Ramp:
        return tir.Ramp(base, stride, lanes)


class Broadcast(ExprCons):
    src_doms = [tir.PrimExpr, Lanes]

    def __call__(self, value: tir.PrimExpr, lanes: Lanes) -> tir.Broadcast:
        return tir.Broadcast(value, lanes)


class BufferLoad(ExprCons):
    src_doms = [tir.Buffer, List[tir.PrimExpr]]

    def __call__(self, buffer: tir.Buffer, indices: List[tir.PrimExpr]) -> tir.BufferLoad:  # type: ignore # noqa
        return tir.BufferLoad(buffer, indices)


class CallInj(ExprCons):
    src_doms = [tir.Call]
    target_dom = tir.PrimExpr

    def __call__(self, op):
        return op


class CallOneArg(Cons):
    src_doms = [tir.PrimExpr]
    target_dom = tir.Call


class CallFloor(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.floor(x)


class CallCeil(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.ceil(x)


class CallTrunc(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.trunc(x)


class CallAbs(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.abs(x)


class CallRound(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.round(x)


class CallNearbyint(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.nearbyint(x)


class CallIsnan(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.isnan(x)


class CallIsfinite(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.isfinite(x)


class CallIsinf(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.isinf(x)


class CallExp(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.Cast(x.dtype, tir.exp(tir.Cast('float', x)))


class CallExp2(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.Cast(x.dtype, tir.exp2(tir.Cast('float', x)))


class CallExp10(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.Cast(x.dtype, tir.exp10(tir.Cast('float', x)))


class CallErf(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.Cast(x.dtype, tir.erf(tir.Cast('float', x)))


class CallTanh(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.Cast(x.dtype, tir.tanh(tir.Cast('float', x)))


class CallSigmoid(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.Cast(x.dtype, tir.sigmoid(tir.Cast('float', x)))


class CallLog(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.Cast(x.dtype, tir.log(tir.Cast('float', x)))


class CallLog2(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.Cast(x.dtype, tir.log2(tir.Cast('float', x)))


class CallLog10(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.Cast(x.dtype, tir.log10(tir.Cast('float', x)))


class CallLog1p(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.Cast(x.dtype, tir.log1p(tir.Cast('float', x)))


class CallTan(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.Cast(x.dtype, tir.tan(tir.Cast('float', x)))


class CallCos(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.Cast(x.dtype, tir.cos(tir.Cast('float', x)))


class CallCosh(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.Cast(x.dtype, tir.cosh(tir.Cast('float', x)))


class CallAcos(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.Cast(x.dtype, tir.acos(tir.Cast('float', x)))


class CallAcosh(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.Cast(x.dtype, tir.acosh(tir.Cast('float', x)))


class CallSin(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.Cast(x.dtype, tir.sin(tir.Cast('float', x)))


class CallSinh(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.Cast(x.dtype, tir.sinh(tir.Cast('float', x)))


class CallAsin(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.Cast(x.dtype, tir.asin(tir.Cast('float', x)))


class CallAsinh(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.Cast(x.dtype, tir.asinh(tir.Cast('float', x)))


class CallAtan(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.Cast(x.dtype, tir.atan(tir.Cast('float', x)))


class CallAtanh(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.Cast(x.dtype, tir.atanh(tir.Cast('float', x)))


class CallSqrt(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.Cast(x.dtype, tir.sqrt(tir.Cast('float', x)))


class CallRsqrt(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.Cast(x.dtype, tir.rsqrt(tir.Cast('float', x)))


class CallClz(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.Cast(x.dtype, tir.clz(tir.Cast('int64', x)))


class CallRet(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.ret(x)


class CallPopcount(CallOneArg):
    def __call__(self, x: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.Cast(x.dtype, tir.popcount(tir.Cast('int64', x)))


class CallTwoArgs(ExprCons):
    src_doms = [tir.PrimExpr, tir.PrimExpr]


class CallAtan2(CallTwoArgs):
    def __call__(self, x1: tir.PrimExpr, x2: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.Cast(x1.dtype, tir.atan2(tir.Cast('float', x1), tir.Cast('float', x2)))


class CallNextafter(CallTwoArgs):
    def __call__(self, x1: tir.PrimExpr, x2: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.nextafter(x1, x2)


class CallHypot(CallTwoArgs):
    def __call__(self, x1: tir.PrimExpr, x2: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.hypot(x1, x2)


class CallCopysign(CallTwoArgs):
    def __call__(self, x1: tir.PrimExpr, x2: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.copysign(x1, x2)


class CallLdexp(CallTwoArgs):
    def __call__(self, x1: tir.PrimExpr, x2: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.Cast(x1.dtype, tir.ldexp(tir.Cast('float', x1), tir.Cast('float', x2)))


class CallPower(CallTwoArgs):
    def __call__(self, x1: tir.PrimExpr, x2: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.Cast(x1.dtype, tir.power(tir.Cast('float', x1), tir.Cast('float', x2)))


class CallDiv(CallTwoArgs):
    def __call__(self, x1: tir.PrimExpr, x2: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.div(x1, x2)


class CallIndexdiv(CallTwoArgs):
    def __call__(self, x1: tir.PrimExpr, x2: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.indexdiv(x1, x2)


class CallIndexmod(CallTwoArgs):
    def __call__(self, x1: tir.PrimExpr, x2: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.indexmod(x1, x2)


class CallTruncdiv(CallTwoArgs):
    def __call__(self, x1: tir.PrimExpr, x2: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.truncdiv(x1, x2)


class CallTruncmod(CallTwoArgs):
    def __call__(self, x1: tir.PrimExpr, x2: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.truncmod(x1, x2)


class CallFloordiv(CallTwoArgs):
    def __call__(self, x1: tir.PrimExpr, x2: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.floordiv(x1, x2)


class CallFloormod(CallTwoArgs):
    def __call__(self, x1: tir.PrimExpr, x2: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.floormod(x1, x2)


class CallFmod(Cons):
    src_doms = [tir.PrimExpr, tir.PrimExpr]
    target_dom = tir.PrimExpr

    def __call__(self, x: tir.PrimExpr, y: tir.PrimExpr):  # type: ignore
        return tir.fmod(x, y)


class CallQ_multiply_shift(Cons):
    src_doms = [tir.PrimExpr, tir.PrimExpr, tir.PrimExpr, tir.PrimExpr]
    target_dom = tir.PrimExpr

    def __call__(self, x: tir.PrimExpr, y: tir.PrimExpr, q: tir.PrimExpr, s: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.q_multiply_shift(x, y, q, s)


class CallIf_then_else(ExprCons):
    src_doms = [tir.PrimExpr, tir.PrimExpr, tir.PrimExpr]
    target_dom = tir.PrimExpr

    def __call__(self, cond: tir.PrimExpr, t: tir.PrimExpr, f: tir.PrimExpr) -> tir.Call:  # type: ignore
        return tir.if_then_else(tir.Cast('bool', cond), t, f)


# statements

class StmtCons(Cons):
    target_dom = tir.Stmt


class LetStmt(StmtCons):
    src_doms = [tir.Var, tir.PrimExpr, tir.Stmt]

    def __call__(self, var: tir.Var, value: tir.PrimExpr, body: tir.Stmt) -> tir.stmt.LetStmt:  # type: ignore
        return tir.LetStmt(var, tir.Cast(var.dtype, value), body)


class AssertStmt(StmtCons):
    src_doms = [tir.PrimExpr, tir.PrimExpr, tir.Stmt]

    def __call__(self, cond: tir.PrimExpr, msg: tir.PrimExpr, body: tir.Stmt) -> tir.stmt.AssertStmt:  # type: ignore
        return tir.AssertStmt(tir.Cast('bool', cond), msg, body)


class AttrStmt(StmtCons):
    src_doms = [tir.IterVar, AttrKey, tir.PrimExpr, tir.Stmt]

    def __call__(self, node: tir.IterVar, attr_key: AttrKey, value: tir.PrimExpr, body: tir.Stmt) -> tir.stmt.AttrStmt:  # type: ignore
        return tir.AttrStmt(node, attr_key, value, body)


class For(StmtCons):
    src_doms = [tir.Var, tir.PrimExpr, tir.PrimExpr, tir.ForKind, tir.Stmt]

    def __call__(self, loop_var: tir.Var, min_val: tir.PrimExpr, extent: tir.PrimExpr, kind: tir.ForKind, body: tir.Stmt) -> tir.stmt.For:  # type: ignore
        return tir.For(loop_var, min_val, extent, kind, body)


class While(StmtCons):
    src_doms = [tir.PrimExpr, tir.Stmt]

    def __call__(self, cond: tir.PrimExpr, body: tir.Stmt) -> tir.stmt.While:  # type: ignore
        return tir.stmt.While(tir.Cast('bool', cond), body)


class Store(StmtCons):
    src_doms = [tir.Var, tir.PrimExpr, tir.PrimExpr, tir.PrimExpr]

    def __call__(self, bfvar: tir.Var, val: tir.PrimExpr, index: tir.PrimExpr, pred: tir.PrimExpr) -> tir.Store:  # type: ignore
        return tir.Store(bfvar, val, index, pred)


class Allocate(StmtCons):
    src_doms = [tir.Var, DType, List[tir.PrimExpr], tir.PrimExpr, tir.Stmt]

    def __call__(self, bfvar: tir.Var, dtype: DType, extents: List[tir.PrimExpr], cond: tir.PrimExpr, body: tir.Stmt) -> tir.Allocate:  # type: ignore # noqa
        return tir.Allocate(bfvar, dtype, extents, tir.Cast('bool', cond), body)


class SeqStmt(StmtCons):
    src_doms = [List[tir.Stmt]]

    def __call__(self, seq: List[tir.Stmt]) -> tir.SeqStmt:  # type: ignore
        return tir.SeqStmt(seq)


class IfThenElse(StmtCons):
    src_doms = [tir.PrimExpr, tir.Stmt, tir.Stmt]

    def __call__(self, cond: tir.PrimExpr, then: tir.Stmt, else_stmt: tir.Stmt) -> tir.IfThenElse:  # type: ignore
        return tir.IfThenElse(tir.Cast('bool', cond), then, else_stmt)


class Evaluate(StmtCons):
    src_doms = [tir.PrimExpr]

    def __call__(self, op: tir.PrimExpr) -> tir.Evaluate:  # type: ignore
        return tir.Evaluate(op)


class FuncCons(Cons):
    target_dom = tir.PrimFunc


class PrimFunc(FuncCons):
    src_doms = [List[tir.Var], tir.Stmt, Dict[tir.Var, tir.Buffer]]

    def __call__(self, params: List[tir.Var], body: tir.Stmt, buffer_map: Dict[tir.Var, tir.Buffer]) -> tir.PrimFunc:  # type: ignore
        return tir.PrimFunc(
            params,
            body,
            buffer_map=buffer_map
        )


class ListExprCons(Cons):
    src_doms = [List[tir.PrimExpr], tir.PrimExpr]
    target_dom = List[tir.PrimExpr]

    def __call__(self, lst: List[tir.PrimExpr], new: tir.PrimExpr) -> List[tir.PrimExpr]:  # type: ignore # noqa
        return [*lst, new]


class ListStmtCons(Cons):
    src_doms = [List[tir.Stmt], tir.Stmt]
    target_dom = List[tir.Stmt]

    def __call__(self, lst: List[tir.Stmt], new: tir.Stmt) -> List[tir.Stmt]:  # type: ignore # noqa
        return [*lst, new]


class ListVarCons(Cons):
    src_doms = [List[tir.Var], tir.Var]
    target_dom = List[tir.Var]

    def __call__(self, lst: List[tir.Var], new: tir.Var) -> List[tir.Var]:  # type: ignore
        return [*lst, new]


class SomeCons(Cons):
    def __init__(self, id: str, target_dom: Dom, src_doms: List[Dom], cons_def: Callable):
        self.id = id
        self.src_doms = src_doms
        self.target_dom = target_dom
        self.cons_def = cons_def

    def __call__(self, *args: DomValue, **kwds: DomValue) -> DomValue:  # type: ignore
        return self.cons_def(*args, **kwds)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, SomeCons):
            return False
        return self.id.__eq__(o.id)

    def __hash__(self) -> int:
        return self.id.__hash__()

    def __str__(self) -> str:
        return f'SomeCons({self.id})'

    def __repr__(self) -> str:
        return self.__str__()


ABSTRACT_CONSS: List[Type[Cons]] = [
    Cons,
    IntImmCons,
    ExprCons,
    BinOp,
    CallOneArg,
    CallTwoArgs,
    StmtCons,
    FuncCons,
    SomeCons,
]


def _is_concrete_cons(var) -> bool:
    return isinstance(var, type) and issubclass(var, Cons) and var not in ABSTRACT_CONSS


ALL_CONSS: List[Cons] = [v() for v in map(
    lambda vname: globals()[vname], dir()) if _is_concrete_cons(v)]


def _get_all_doms() -> List[Dom]:
    res: Set[Dom] = set()
    for c in ALL_CONSS:
        for dom in c.src_doms:
            res.add(dom)
        res.add(c.target_dom)
    return list(res)


ALL_DOMS: List[Dom] = _get_all_doms()
