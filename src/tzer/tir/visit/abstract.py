from typing import Generic, TypeVar
from typing import Any
from abc import abstractmethod
from tvm import tir
from tzer.tir.util import TIRNode


class NoDispatchPatternError(Warning):
    pass


TIRReturn = TypeVar('TIRReturn')
Arg = TypeVar('Arg')


class TIRVisitor(Generic[TIRReturn, Arg]):
    def visit(self, op, arg: Arg) -> TIRReturn:
        if isinstance(op, tir.PrimFunc):
            return self._visit_primfunc(op, arg)
        elif isinstance(op, tir.Var):
            return self._visit_var(op, arg)
        elif isinstance(op, tir.SizeVar):
            return self._visit_sizevar(op, arg)
        elif isinstance(op, tir.IterVar):
            return self._visit_itervar(op, arg)
        elif isinstance(op, tir.Load):
            return self._visit_load(op, arg)
        elif isinstance(op, tir.BufferLoad):
            return self._visit_bufferload(op, arg)
        elif isinstance(op, tir.ProducerLoad):
            return self._visit_producerload(op, arg)
        elif isinstance(op, tir.Let):
            return self._visit_let(op, arg)
        elif isinstance(op, tir.Call):
            return self._visit_call(op, arg)
        elif isinstance(op, tir.Add):
            return self._visit_add(op, arg)
        elif isinstance(op, tir.Sub):
            return self._visit_sub(op, arg)
        elif isinstance(op, tir.Mul):
            return self._visit_mul(op, arg)
        elif isinstance(op, tir.Div):
            return self._visit_div(op, arg)
        elif isinstance(op, tir.Mod):
            return self._visit_mod(op, arg)
        elif isinstance(op, tir.FloorDiv):
            return self._visit_floordiv(op, arg)
        elif isinstance(op, tir.FloorMod):
            return self._visit_floormod(op, arg)
        elif isinstance(op, tir.Min):
            return self._visit_min(op, arg)
        elif isinstance(op, tir.Max):
            return self._visit_max(op, arg)
        elif isinstance(op, tir.EQ):
            return self._visit_eq(op, arg)
        elif isinstance(op, tir.NE):
            return self._visit_ne(op, arg)
        elif isinstance(op, tir.LT):
            return self._visit_lt(op, arg)
        elif isinstance(op, tir.LE):
            return self._visit_le(op, arg)
        elif isinstance(op, tir.GT):
            return self._visit_gt(op, arg)
        elif isinstance(op, tir.GE):
            return self._visit_ge(op, arg)
        elif isinstance(op, tir.And):
            return self._visit_and(op, arg)
        elif isinstance(op, tir.Or):
            return self._visit_or(op, arg)
        elif isinstance(op, tir.Reduce):
            return self._visit_reduce(op, arg)
        elif isinstance(op, tir.Cast):
            return self._visit_cast(op, arg)
        elif isinstance(op, tir.Not):
            return self._visit_not(op, arg)
        elif isinstance(op, tir.Select):
            return self._visit_select(op, arg)
        elif isinstance(op, tir.Ramp):
            return self._visit_ramp(op, arg)
        elif isinstance(op, tir.Broadcast):
            return self._visit_broadcast(op, arg)
        elif isinstance(op, tir.Shuffle):
            return self._visit_shuffle(op, arg)
        elif isinstance(op, tir.IntImm):
            return self._visit_intimm(op, arg)
        elif isinstance(op, tir.FloatImm):
            return self._visit_floatimm(op, arg)
        elif isinstance(op, tir.StringImm):
            return self._visit_stringimm(op, arg)
        elif isinstance(op, tir.Any):
            return self._visit_any(op, arg)
        if isinstance(op, tir.AttrStmt):
            return self._visit_attrstmt(op, arg)
        elif isinstance(op, tir.IfThenElse):
            return self._visit_ifthenelse(op, arg)
        elif isinstance(op, tir.LetStmt):
            return self._visit_letstmt(op, arg)
        elif isinstance(op, tir.For):
            return self._visit_for(op, arg)
        elif isinstance(op, tir.stmt.While):
            return self._visit_while(op, arg)
        elif isinstance(op, tir.Allocate):
            return self._visit_allocate(op, arg)
        elif isinstance(op, tir.Store):
            return self._visit_store(op, arg)
        elif isinstance(op, tir.BufferStore):
            return self._visit_bufferstore(op, arg)
        elif isinstance(op, tir.BufferRealize):
            return self._visit_bufferrealize(op, arg)
        elif isinstance(op, tir.AssertStmt):
            return self._visit_assertstmt(op, arg)
        elif isinstance(op, tir.ProducerStore):
            return self._visit_producerstore(op, arg)
        elif isinstance(op, tir.ProducerRealize):
            return self._visit_producerrealize(op, arg)
        elif isinstance(op, tir.Prefetch):
            return self._visit_prefetch(op, arg)
        elif isinstance(op, tir.SeqStmt):
            return self._visit_seqstmt(op, arg)
        elif isinstance(op, tir.Evaluate):
            return self._visit_evaluate(op, arg)
        elif isinstance(op, tir.Block):
            return self._visit_block(op, arg)
        elif isinstance(op, tir.BlockRealize):
            return self._visit_blockrealize(op, arg)
        else:
            raise NoDispatchPatternError(type(op), op)

    def decorate(visit_func: Any):
        def wrapper_visit(self, op, arg):
            return visit_func(self, op, arg)
        return wrapper_visit

    def __call__(self, op: TIRNode, arg: Arg) -> TIRReturn:
        return self.visit(op, arg)

    def _visit_primfunc(self, op: tir.PrimFunc, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_primfunc)(self, op, arg)

    def _visit_var(self, op: tir.Var, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_var)(self, op, arg)

    def _visit_sizevar(self, op: tir.SizeVar, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_sizevar)(self, op, arg)

    def _visit_itervar(self, op: tir.IterVar, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_itervar)(self, op, arg)

    def _visit_load(self, op: tir.Load, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_load)(self, op, arg)

    def _visit_bufferload(self, op: tir.BufferLoad, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_bufferload)(self, op, arg)

    def _visit_producerload(self, op: tir.ProducerLoad, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_producerload)(self, op, arg)

    def _visit_let(self, op: tir.Let, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_let)(self, op, arg)

    def _visit_call(self, op: tir.Call, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_call)(self, op, arg)

    def _visit_add(self, op: tir.Add, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_add)(self, op, arg)

    def _visit_sub(self, op: tir.Sub, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_sub)(self, op, arg)

    def _visit_mul(self, op: tir.Mul, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_mul)(self, op, arg)

    def _visit_div(self, op: tir.Div, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_div)(self, op, arg)

    def _visit_mod(self, op: tir.Mod, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_mod)(self, op, arg)

    def _visit_floordiv(self, op: tir.FloorDiv, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_floordiv)(self, op, arg)

    def _visit_floormod(self, op: tir.FloorMod, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_floormod)(self, op, arg)

    def _visit_min(self, op: tir.Min, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_min)(self, op, arg)

    def _visit_max(self, op: tir.Max, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_max)(self, op, arg)

    def _visit_eq(self, op: tir.EQ, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_eq)(self, op, arg)

    def _visit_ne(self, op: tir.NE, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_ne)(self, op, arg)

    def _visit_lt(self, op: tir.LT, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_lt)(self, op, arg)

    def _visit_le(self, op: tir.LE, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_le)(self, op, arg)

    def _visit_gt(self, op: tir.GT, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_gt)(self, op, arg)

    def _visit_ge(self, op: tir.GE, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_ge)(self, op, arg)

    def _visit_and(self, op: tir.And, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_and)(self, op, arg)

    def _visit_or(self, op: tir.Or, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_or)(self, op, arg)

    def _visit_reduce(self, op: tir.Reduce, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_reduce)(self, op, arg)

    def _visit_cast(self, op: tir.Cast, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_cast)(self, op, arg)

    def _visit_not(self, op: tir.Not, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_not)(self, op, arg)

    def _visit_select(self, op: tir.Select, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_select)(self, op, arg)

    def _visit_ramp(self, op: tir.Ramp, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_ramp)(self, op, arg)

    def _visit_broadcast(self, op: tir.Broadcast, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_broadcast)(self, op, arg)

    def _visit_shuffle(self, op: tir.Shuffle, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_shuffle)(self, op, arg)

    def _visit_intimm(self, op: tir.IntImm, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_intimm)(self, op, arg)

    def _visit_floatimm(self, op: tir.FloatImm, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_floatimm)(self, op, arg)

    def _visit_stringimm(self, op: tir.StringImm, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_stringimm)(self, op, arg)

    def _visit_any(self, op: tir.Any, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_any)(self, op, arg)

    def _visit_attrstmt(self, op: tir.AttrStmt, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_attrstmt)(self, op, arg)

    def _visit_ifthenelse(self, op: tir.IfThenElse, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_ifthenelse)(self, op, arg)

    def _visit_letstmt(self, op: tir.LetStmt, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_letstmt)(self, op, arg)

    def _visit_for(self, op: tir.For, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_for)(self, op, arg)

    def _visit_while(self, op: tir.stmt.While, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_while)(self, op, arg)

    def _visit_allocate(self, op: tir.Allocate, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_allocate)(self, op, arg)

    def _visit_store(self, op: tir.Store, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_store)(self, op, arg)

    def _visit_bufferstore(self, op: tir.BufferStore, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_bufferstore)(self, op, arg)

    def _visit_bufferrealize(self, op: tir.BufferRealize, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_bufferrealize)(self, op, arg)

    def _visit_assertstmt(self, op: tir.AssertStmt, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_assertstmt)(self, op, arg)

    def _visit_producerstore(self, op: tir.ProducerStore, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_producerstore)(self, op, arg)

    def _visit_producerrealize(self, op: tir.ProducerRealize, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_producerrealize)(self, op, arg)

    def _visit_prefetch(self, op: tir.Prefetch, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_prefetch)(self, op, arg)

    def _visit_seqstmt(self, op: tir.SeqStmt, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_seqstmt)(self, op, arg)

    def _visit_evaluate(self, op: tir.Evaluate, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_evaluate)(self, op, arg)

    def _visit_block(self, op: tir.Block, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_block)(self, op, arg)

    def _visit_blockrealize(self, op: tir.BlockRealize, arg: Arg) -> TIRReturn:
        return self.__class__.decorate(self.__class__.visit_blockrealize)(self, op, arg)

    @abstractmethod
    def visit_primfunc(self, op: tir.PrimFunc, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_var(self, op: tir.Var, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_sizevar(self, op: tir.SizeVar, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_itervar(self, op: tir.IterVar, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_load(self, op: tir.Load, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_bufferload(self, op: tir.BufferLoad, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_producerload(self, op: tir.ProducerLoad, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_let(self, op: tir.Let, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_call(self, op: tir.Call, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_add(self, op: tir.Add, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_sub(self, op: tir.Sub, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_mul(self, op: tir.Mul, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_div(self, op: tir.Div, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_mod(self, op: tir.Mod, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_floordiv(self, op: tir.FloorDiv, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_floormod(self, op: tir.FloorMod, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_min(self, op: tir.Min, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_max(self, op: tir.Max, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_eq(self, op: tir.EQ, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_ne(self, op: tir.NE, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_lt(self, op: tir.LT, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_le(self, op: tir.LE, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_gt(self, op: tir.GT, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_ge(self, op: tir.GE, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_and(self, op: tir.And, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_or(self, op: tir.Or, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_reduce(self, op: tir.Reduce, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_cast(self, op: tir.Cast, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_not(self, op: tir.Not, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_select(self, op: tir.Select, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_ramp(self, op: tir.Ramp, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_broadcast(self, op: tir.Broadcast, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_shuffle(self, op: tir.Shuffle, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_intimm(self, op: tir.IntImm, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_floatimm(self, op: tir.FloatImm, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_stringimm(self, op: tir.StringImm, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_any(self, op: tir.Any, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_attrstmt(self, op: tir.AttrStmt, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_ifthenelse(self, op: tir.IfThenElse, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_letstmt(self, op: tir.LetStmt, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_for(self, op: tir.For, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_while(self, op: tir.stmt.While, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_allocate(self, op: tir.Allocate, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_store(self, op: tir.Store, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_bufferstore(self, op: tir.BufferStore, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_bufferrealize(self, op: tir.BufferRealize, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_assertstmt(self, op: tir.AssertStmt, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_producerstore(self, op: tir.ProducerStore, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_producerrealize(self, op: tir.ProducerRealize, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_prefetch(self, op: tir.Prefetch, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_seqstmt(self, op: tir.SeqStmt, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_evaluate(self, op: tir.Evaluate, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_block(self, op: tir.Block, arg: Arg) -> TIRReturn:
        ...

    @abstractmethod
    def visit_blockrealize(self, op: tir.BlockRealize, arg: Arg) -> TIRReturn:
        ...


class TIRAbstractTransformer(Generic[Arg], TIRVisitor[TIRNode, Arg]):
    def visit_primfunc(self, op: tir.PrimFunc, arg) -> TIRNode:
        return tir.PrimFunc(
            op.params,
            self(op.body, arg),
            op.ret_type,
            op.buffer_map,
            op.attrs
        )

    def visit_var(self, op: tir.Var, arg) -> TIRNode:
        return op

    def visit_sizevar(self, op: tir.SizeVar, arg) -> TIRNode:
        return op

    def visit_itervar(self, op: tir.IterVar, arg) -> TIRNode:
        return tir.IterVar(
            dom=op.dom,
            var=self(op.var, arg),
            iter_type=op.iter_type,
            thread_tag=op.thread_tag,
            span=op.span
        )

    def visit_load(self, op: tir.Load, arg) -> TIRNode:
        return tir.Load(
            dtype=op.dtype,
            buffer_var=self(op.buffer_var, arg),
            index=self(op.index, arg),
            predicate=self(op.predicate, arg),
            span=op.span
        )

    def visit_let(self, op: tir.Let, arg) -> TIRNode:
        return tir.Let(
            self(op.var, arg),
            self(op.value, arg),
            self(op.body, arg),
            op.span
        )

    def visit_bufferload(self, op: tir.BufferLoad, arg) -> TIRNode:
        return tir.BufferLoad(
            op.buffer,
            [self(index, arg) for index in op.indices],
            op.span
        )

    def visit_producerload(self, op: tir.ProducerLoad, arg) -> TIRNode:
        return tir.ProducerLoad(
            op.producer,
            [self(index, arg) for index in op.indices],
            op.span
        )

    def visit_call(self, op: tir.Call, arg) -> TIRNode:
        return tir.Call(
            dtype=op.dtype,
            op=op.op,
            args=[self(a, arg) for a in op.args],
            span=op.span
        )

    def visit_add(self, op: tir.Add, arg) -> TIRNode:
        return tir.Add(
            a=self(op.a, arg),
            b=self(op.b, arg),
        )

    def visit_sub(self, op: tir.Sub, arg) -> TIRNode:
        return tir.Sub(
            a=self(op.a, arg),
            b=self(op.b, arg),
        )

    def visit_mul(self, op: tir.Mul, arg) -> TIRNode:
        return tir.Mul(
            a=self(op.a, arg),
            b=self(op.b, arg),
        )

    def visit_div(self, op: tir.Div, arg) -> TIRNode:
        return tir.Div(
            a=self(op.a, arg),
            b=self(op.b, arg),
        )

    def visit_mod(self, op: tir.Mod, arg) -> TIRNode:
        return tir.Mod(
            a=self(op.a, arg),
            b=self(op.b, arg),
        )

    def visit_floordiv(self, op: tir.FloorDiv, arg) -> TIRNode:
        return tir.FloorDiv(
            a=self(op.a, arg),
            b=self(op.b, arg),
        )

    def visit_floormod(self, op: tir.FloorMod, arg) -> TIRNode:
        return tir.FloorMod(
            a=self(op.a, arg),
            b=self(op.b, arg),
        )

    def visit_min(self, op: tir.Min, arg) -> TIRNode:
        return tir.Min(
            a=self(op.a, arg),
            b=self(op.b, arg),
        )

    def visit_max(self, op: tir.Max, arg) -> TIRNode:
        return tir.Max(
            a=self(op.a, arg),
            b=self(op.b, arg),
        )

    def visit_eq(self, op: tir.EQ, arg) -> TIRNode:
        return tir.EQ(
            a=self(op.a, arg),
            b=self(op.b, arg),
        )

    def visit_ne(self, op: tir.NE, arg) -> TIRNode:
        return tir.NE(
            a=self(op.a, arg),
            b=self(op.b, arg),
        )

    def visit_lt(self, op: tir.LT, arg) -> TIRNode:
        return tir.LT(
            a=self(op.a, arg),
            b=self(op.b, arg),
        )

    def visit_le(self, op: tir.LE, arg) -> TIRNode:
        return tir.LE(
            a=self(op.a, arg),
            b=self(op.b, arg),
        )

    def visit_gt(self, op: tir.GT, arg) -> TIRNode:
        return tir.GT(
            a=self(op.a, arg),
            b=self(op.b, arg),
        )

    def visit_ge(self, op: tir.GE, arg) -> TIRNode:
        return tir.GE(
            a=self(op.a, arg),
            b=self(op.b, arg),
        )

    def visit_and(self, op: tir.And, arg) -> TIRNode:
        return tir.Add(
            a=self(op.a, arg),
            b=self(op.b, arg),
        )

    def visit_or(self, op: tir.Or, arg) -> TIRNode:
        return tir.Or(
            a=self(op.a, arg),
            b=self(op.b, arg),
        )

    def visit_reduce(self, op: tir.Reduce, arg) -> TIRNode:
        return tir.Reduce(
            op.combiner,
            [self(s, arg) for s in op.src],
            op.rdom,
            self(op.condition, arg),
            self(op.value_index, arg),
            [self(i, arg) for i in op.init],
            op.span
        )

    def visit_cast(self, op: tir.Cast, arg) -> TIRNode:
        return tir.Cast(
            dtype=op.dtype,
            value=self(op.value, arg),
            span=op.span
        )

    def visit_not(self, op: tir.Not, arg) -> TIRNode:
        return tir.Not(
            a=self(op.a, arg),
            span=op.span
        )

    def visit_select(self, op: tir.Select, arg) -> TIRNode:
        return tir.Select(
            condition=self(op.condition, arg),
            true_value=self(op.true_value, arg),
            false_value=self(op.false_value, arg),
            span=op.span
        )

    def visit_ramp(self, op: tir.Ramp, arg) -> TIRNode:
        return tir.Ramp(
            self(op.base, arg),
            op.stride,
            op.lanes,
            op.span
        )

    def visit_broadcast(self, op: tir.Broadcast, arg) -> TIRNode:
        return tir.Broadcast(
            self(op.value, arg),
            op.lanes,
            op.span
        )

    def visit_shuffle(self, op: tir.Shuffle, arg) -> TIRNode:
        return tir.Shuffle(
            [self(vec, arg) for vec in op.vectors],
            op.indices,
            op.span
        )

    def visit_intimm(self, op: tir.IntImm, arg) -> TIRNode:
        return op

    def visit_floatimm(self, op: tir.FloatImm, arg) -> TIRNode:
        return op

    def visit_stringimm(self, op: tir.StringImm, arg) -> TIRNode:
        return op

    def visit_any(self, op: tir.Any, arg) -> TIRNode:
        return op

    def visit_attrstmt(self, op: tir.AttrStmt, arg) -> TIRNode:
        return tir.AttrStmt(
            op.node,
            op.attr_key,
            self(op.value, arg),
            self(op.body, arg),
            op.span
        )

    def visit_ifthenelse(self, op: tir.IfThenElse, arg) -> TIRNode:
        return tir.IfThenElse(
            self(op.condition, arg),
            self(op.then_case, arg),
            None if op.else_case is None else self(op.else_case, arg),
            op.span
        )

    def visit_letstmt(self, op: tir.LetStmt, arg) -> TIRNode:
        return tir.LetStmt(
            self(op.var, arg),
            self(op.value, arg),
            self(op.body, arg),
            op.span
        )

    def visit_for(self, op: tir.For, arg) -> TIRNode:
        return tir.For(
            self(op.loop_var, arg),
            self(op.min, arg),
            self(op.extent, arg),
            op.kind,
            self(op.body, arg),
            op.thread_binding,
            op.annotations,
            op.span,
        )

    def visit_while(self, op: tir.stmt.While, arg) -> TIRNode:
        return tir.stmt.While(
            self(op.condition, arg),
            self(op.body, arg),
            op.span
        )

    def visit_allocate(self, op: tir.Allocate, arg) -> TIRNode:
        return tir.Allocate(
            self(op.buffer_var, arg),
            op.dtype,
            [self(extent, arg) for extent in op.extents],
            self(op.condition, arg),
            self(op.body, arg),
            op.span
        )

    def visit_store(self, op: tir.Store, arg) -> TIRNode:
        return tir.Store(
            self(op.buffer_var, arg),
            self(op.value, arg),
            self(op.index, arg),
            self(op.predicate, arg),
            op.span
        )

    def visit_bufferstore(self, op: tir.BufferStore, arg) -> TIRNode:
        return tir.BufferStore(
            op.buffer,
            self(op.value, arg),
            [self(index, arg) for index in op.indices],
            op.span
        )

    def visit_bufferrealize(self, op: tir.BufferRealize, arg) -> TIRNode:
        return tir.BufferRealize(
            op.buffer,
            op.bounds,
            self(op.condition, arg),
            self(op.body, arg),
            op.span
        )

    def visit_assertstmt(self, op: tir.AssertStmt, arg) -> TIRNode:
        return tir.AssertStmt(
            self(op.condition, arg),
            self(op.message, arg),
            self(op.body, arg),
            op.span
        )

    def visit_producerstore(self, op: tir.ProducerStore, arg) -> TIRNode:
        return tir.ProducerStore(
            op.producer,
            self(op.value, arg),
            [self(index, arg) for index in op.indices],
            op.span
        )

    def visit_producerrealize(self, op: tir.ProducerRealize, arg) -> TIRNode:
        return tir.ProducerRealize(
            op.producer,
            op.bounds,
            self(op.condition, arg),
            self(op.body, arg),
            op.span
        )

    def visit_prefetch(self, op: tir.Prefetch, arg) -> TIRNode:
        return tir.Prefetch(
            op.buffer,
            op.bounds,
            op.span
        )

    def visit_seqstmt(self, op: tir.SeqStmt, arg) -> TIRNode:
        return tir.SeqStmt(
            [self(s, arg) for s in op.seq],
            op.span
        )

    def visit_evaluate(self, op: tir.Evaluate, arg) -> TIRNode:
        return tir.Evaluate(
            self(op.value, arg),
            op.span
        )

    def visit_block(self, op: tir.Block, arg) -> TIRNode:
        return tir.Block(
            op.iter_vars,
            op.reads,
            op.writes,
            op.name_hint,
            self(op.body, arg),
            self(op.init, arg) if op.init is not None else None,
            op.alloc_buffers,
            op.match_buffers,
            op.annotations
        )

    def visit_blockrealize(self, op: tir.BlockRealize, arg) -> TIRNode:
        return tir.BlockRealize(
            [self(v, arg) for v in op.iter_values],
            op.predicate if isinstance(
                op.predicate, bool) else self(op.predicate, arg),
            self(op.block, arg)
        )
