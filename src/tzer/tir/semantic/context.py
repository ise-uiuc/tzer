from dataclasses import dataclass
from typing import Any, Callable, Generic, List, NamedTuple, Optional, Type, TypeVar, Union
from tvm import tir
from tvm.tir import call_intrin
import tvm
from tvm._ffi.runtime_ctypes import DataType
from tvm.ir.expr import PrimExpr
from tvm.tir.stmt import AssertStmt
from tzer.tir.util import TIRNode
from .constraint import Constraint, PrimExprConstraint, PrimFuncConstraint, StmtConstraint
from tzer.tir.visit import TIRVisitor
import random
from tzer.tir import util

_T = TypeVar('_T')
_U = TypeVar('_U')


class Context(Generic[_T]):
    """The semantic context of the hole being visited (e.g. necessary info about [ ] in [ ] + 2)."""
    Extension = Callable[[Callable[['Context'], TIRNode]], TIRNode]

    def __init__(
        self,
        # Constraints of the hole (e.g. type requirement)
        constraint: Constraint,
        # Variables bound in the current evaluation context (e.g. for an expression like
        # let x = 3 in x*x + 5, we perform mutation on x*x, intending to change it to another
        # generated expression. We then should make sure all variables we used are
        # bound, say x, and x would in this value.)
        bound_variables: List[tir.Var] = [],
        buffers: List[tir.Buffer] = [],
        # Additional information of the context which could be derived from the
        # current context, i.e., this property does not make a difference to the context.
        # Defined here for convenience use.
        additional_information: Optional[_T] = None
    ) -> None:
        self.constraint = constraint
        self.bound_variables = bound_variables
        self.buffers = buffers
        self.additional_information = additional_information

    def with_constraint_being(self, constraint: Constraint) -> 'Context':
        return Context(
            constraint,
            self.bound_variables,
            self.buffers,
            self.additional_information,
        )

    def with_bound_vars_being(self, bound_variables: List[tir.Var]) -> 'Context':
        return Context(
            self.constraint,
            bound_variables,
            self.buffers,
            self.additional_information,
        )

    def with_buffers_being(self, buffers: List[tir.Buffer]) -> 'Context':
        return Context(
            self.constraint,
            self.bound_variables,
            buffers,
            self.additional_information,
        )

    def with_additional_information(self, additional_information: _U) -> 'Context[_U]':
        return Context(
            self.constraint,
            self.bound_variables,
            self.buffers,
            additional_information,
        )

    @staticmethod
    def prim_func_context() -> 'Context':
        return Context(PrimFuncConstraint())

    # def extend(self, node_type: type) -> Extension:
    #     if node_type == tir.PrimFunc:
    #         return self.extend_primfunc()
    #     elif node_type == tir.Var:
    #         return self.extend_var()
    #     elif node_type == tir.SizeVar:
    #         return self.extend_sizevar()
    #     elif node_type == tir.IterVar:
    #         return self.extend_itervar()
    #     elif node_type == tir.Load:
    #         return self.extend_load()
    #     elif node_type == tir.BufferLoad:
    #         return self.extend_bufferload()
    #     elif node_type == tir.ProducerLoad:
    #         return self.extend_producerload()
    #     elif node_type == tir.Let:
    #         return self.extend_let()
    #     elif node_type == tir.Call:
    #         return self.extend_call()
    #     elif node_type == tir.Add:
    #         return self.extend_add()
    #     elif node_type == tir.Sub:
    #         return self.extend_sub()
    #     elif node_type == tir.Mul:
    #         return self.extend_mul()
    #     elif node_type == tir.Div:
    #         return self.extend_div()
    #     elif node_type == tir.Mod:
    #         return self.extend_mod()
    #     elif node_type == tir.FloorDiv:
    #         return self.extend_floordiv()
    #     elif node_type == tir.FloorMod:
    #         return self.extend_floormod()
    #     elif node_type == tir.Min:
    #         return self.extend_min()
    #     elif node_type == tir.Max:
    #         return self.extend_max()
    #     elif node_type == tir.EQ:
    #         return self.extend_eq()
    #     elif node_type == tir.NE:
    #         return self.extend_ne()
    #     elif node_type == tir.LT:
    #         return self.extend_lt()
    #     elif node_type == tir.LE:
    #         return self.extend_le()
    #     elif node_type == tir.GT:
    #         return self.extend_gt()
    #     elif node_type == tir.GE:
    #         return self.extend_ge()
    #     elif node_type == tir.And:
    #         return self.extend_and()
    #     elif node_type == tir.Or:
    #         return self.extend_or()
    #     elif node_type == tir.Reduce:
    #         return self.extend_reduce()
    #     elif node_type == tir.Cast:
    #         return self.extend_cast()
    #     elif node_type == tir.Not:
    #         return self.extend_not()
    #     elif node_type == tir.Select:
    #         return self.extend_select()
    #     elif node_type == tir.Ramp:
    #         return self.extend_ramp()
    #     elif node_type == tir.Broadcast:
    #         return self.extend_broadcast()
    #     elif node_type == tir.Shuffle:
    #         return self.extend_shuffle()
    #     elif node_type == tir.IntImm:
    #         return self.extend_intimm()
    #     elif node_type == tir.FloatImm:
    #         return self.extend_floatimm()
    #     elif node_type == tir.StringImm:
    #         return self.extend_stringimm()
    #     elif node_type == tir.Any:
    #         return self.extend_any()
    #     elif node_type == tir.AttrStmt:
    #         return self.extend_attrstmt()
    #     elif node_type == tir.IfThenElse:
    #         return self.extend_ifthenelse()
    #     elif node_type == tir.LetStmt:
    #         return self.extend_letstmt()
    #     elif node_type == tir.For:
    #         return self.extend_for()
    #     elif node_type == tir.stmt.While:
    #         return self.extend_while()
    #     elif node_type == tir.Allocate:
    #         return self.extend_allocate()
    #     elif node_type == tir.Store:
    #         return self.extend_store()
    #     elif node_type == tir.BufferStore:
    #         return self.extend_bufferstore()
    #     elif node_type == tir.BufferRealize:
    #         return self.extend_bufferrealize()
    #     elif node_type == tir.AssertStmt:
    #         return self.extend_assertstmt()
    #     elif node_type == tir.ProducerStore:
    #         return self.extend_producerstore()
    #     elif node_type == tir.ProducerRealize:
    #         return self.extend_producerrealize()
    #     elif node_type == tir.Prefetch:
    #         return self.extend_prefetch()
    #     elif node_type == tir.SeqStmt:
    #         return self.extend_seqstmt()
    #     elif node_type == tir.Evaluate:
    #         return self.extend_evaluate()
    #     elif node_type == tir.Block:
    #         return self.extend_block()
    #     elif node_type == tir.BlockRealize:
    #         return self.extend_blockrealize()
    #     else:
    #         raise NotImplementedError(node_type)

    def extend_primfunc(self) -> Extension:
        raise NotImplementedError

    def extend_var(self) -> Extension:
        raise NotImplementedError

    def extend_sizevar(self) -> Extension:
        raise NotImplementedError

    def extend_load(self) -> Extension:
        bf_var: tir.Var = random.choice(
            self.additional_information)  # type: ignore
        return lambda match: tir.Load(
            self.constraint.dtype,  # type: ignore
            bf_var,
            match(self),
            match(self),
        )

    def extend_bufferload(self) -> Extension:
        buffer: tir.Buffer = random.choice(
            self.additional_information)  # type: ignore
        return lambda match: tir.BufferLoad(
            buffer,
            [match(self)],
        )

    def extend_producerload(self) -> Extension:
        raise NotImplementedError

    def extend_let(self) -> Extension:
        dtype = random.choice(['float', 'bool', 'uint', 'int'])
        var = tir.Var(util.fresh_name(), dtype)
        return lambda match: tir.Let(
            var,
            match(self.with_constraint_being(PrimExprConstraint(dtype))),
            match(self.with_bound_vars_being([*self.bound_variables, var])),
        )

    def extend_call(self) -> Extension:
        raise NotImplementedError

    def extend_binop(self, cls) -> Extension:
        return lambda match: cls(
            match(self),
            match(self),
        )

    def extend_add(self) -> Extension:
        return self.extend_binop(tir.Add)

    def extend_sub(self) -> Extension:
        return self.extend_binop(tir.Sub)

    def extend_mul(self) -> Extension:
        return self.extend_binop(tir.Mul)

    def extend_div(self) -> Extension:
        return self.extend_binop(tir.Div)

    def extend_mod(self) -> Extension:
        return self.extend_binop(tir.Mod)

    def extend_floordiv(self) -> Extension:
        return self.extend_binop(tir.FloorDiv)

    def extend_floormod(self) -> Extension:
        return self.extend_binop(tir.FloorMod)

    def extend_min(self) -> Extension:
        return self.extend_binop(tir.Min)

    def extend_max(self) -> Extension:
        return self.extend_binop(tir.Max)

    def extend_eq(self) -> Extension:
        return self.extend_binop(tir.EQ)

    def extend_ne(self) -> Extension:
        return self.extend_binop(tir.NE)

    def extend_lt(self) -> Extension:
        return self.extend_binop(tir.LT)

    def extend_le(self) -> Extension:
        return self.extend_binop(tir.LE)

    def extend_gt(self) -> Extension:
        return self.extend_binop(tir.GT)

    def extend_ge(self) -> Extension:
        return self.extend_binop(tir.GE)

    def extend_and(self) -> Extension:
        return self.extend_binop(tir.And)

    def extend_or(self) -> Extension:
        return self.extend_binop(tir.Or)

    def extend_reduce(self) -> Extension:
        raise NotImplementedError

    def extend_cast(self) -> Extension:
        return lambda match: tir.Cast(
            self.constraint.dtype,  # type: ignore
            match(self.with_constraint_being(
                PrimExprConstraint(random.choice(
                    ['uint', 'int', 'bool', 'float']))
            ))
        )

    def extend_not(self) -> Extension:
        return lambda match: tir.Not(
            match(self)
        )

    def extend_select(self) -> Extension:
        return lambda match: tir.Select(
            match(self.with_constraint_being(PrimExprConstraint('bool'))),
            match(self),
            match(self),
        )

    def extend_ramp(self) -> Extension:
        raise NotImplementedError

    def extend_broadcast(self) -> Extension:
        raise NotImplementedError

    def extend_shuffle(self) -> Extension:
        raise NotImplementedError

    def extend_intimm(self) -> Extension:
        raise NotImplementedError

    def extend_floatimm(self) -> Extension:
        raise NotImplementedError

    def extend_stringimm(self) -> Extension:
        raise NotImplementedError

    def extend_any(self) -> Extension:
        raise NotImplementedError

    def extend_attrstmt(self) -> Extension:
        raise NotImplementedError

    def extend_ifthenelse(self) -> Extension:
        return lambda match: tir.IfThenElse(
            match(self.with_constraint_being(PrimExprConstraint('bool'))),
            match(self),
            match(self),
        )

    def extend_letstmt(self) -> Extension:
        dtype = random.choice(['uint', 'int', 'bool', 'float'])
        var = tir.Var(util.fresh_name(), dtype)
        return lambda match: tir.LetStmt(
            var,
            match(self.with_constraint_being(PrimExprConstraint(dtype))),
            match(self.with_bound_vars_being([*self.bound_variables, var])),
        )

    def extend_itervar(self) -> Extension:
        raise NotImplementedError

    def extend_for(self) -> Extension:
        raise NotImplementedError

    def extend_while(self) -> Extension:
        return lambda match: tir.stmt.While(
            match(self.with_constraint_being(PrimExprConstraint('bool'))),
            match(self),
        )

    def extend_allocate(self) -> Extension:
        raise NotImplementedError

    def extend_store(self) -> Extension:
        bf_var = random.choice(self.additional_information)  # type: ignore
        return lambda match: tir.Store(
            bf_var,
            match(self.with_constraint_being(PrimExprConstraint(
                bf_var.type_annotation.element_type.dtype))),
            match(self.with_constraint_being(PrimExprConstraint(random.choice(
                ['int', 'uint']
            ))))
        )

    def extend_bufferstore(self) -> Extension:
        buffer = random.choice(self.additional_information)  # type: ignore
        return lambda match: tir.Store(
            buffer,
            match(self.with_constraint_being(
                PrimExprConstraint(buffer.data.dtype))),
            match(self.with_constraint_being(PrimExprConstraint(random.choice(
                ['int', 'uint']
            )))),
            match(self.with_constraint_being(PrimExprConstraint('bool'))),
        )

    def extend_bufferrealize(self) -> Extension:
        raise NotImplementedError

    def extend_assertstmt(self) -> Extension:
        return lambda match: tir.AssertStmt(
            match(self.with_constraint_being(PrimExprConstraint('bool'))),
            tir.StringImm(util.random_string()),
            match(self),
        )

    def extend_producerstore(self) -> Extension:
        raise NotImplementedError

    def extend_producerrealize(self) -> Extension:
        raise NotImplementedError

    def extend_prefetch(self) -> Extension:
        raise NotImplementedError

    def extend_seqstmt(self) -> Extension:
        raise NotImplementedError

    def extend_evaluate(self) -> Extension:
        return lambda match: tir.Evaluate(
            self.with_constraint_being(PrimExprConstraint(random.choice([
                'bool', 'int', 'uint', 'float',
            ])))
        )

    def extend_block(self) -> Extension:
        raise NotImplementedError

    def extend_blockrealize(self) -> Extension:
        raise NotImplementedError

    def candidates(self) -> List[Callable[[], Extension]]:
        options = []
        if isinstance(self.constraint, StmtConstraint):
            options = [
                self.extend_ifthenelse,
                self.extend_letstmt,
                self.extend_while,
                self.extend_assertstmt,
                # tir.AttrStmt,
                # tir.IfThenElse,
                # tir.LetStmt,
                # # tir.For,
                # tir.stmt.While,
                # # tir.Allocate,
                # # tir.Bufferrealize,
                # tir.AssertStmt,
                # # tir.Producerstore,
                # # tir.Producerrealize,
                # # tir.SeqStmt,
                # tir.Evaluate,
                # # tir.Block,
                # # tir.Blockrealize,
            ]
            possible_vars = [
                v for v in self.bound_variables
                if v.dtype == 'handle'
                and 'type_annotation' in dir(v)
                and isinstance(v.type_annotation, tvm.ir.PointerType)
            ]
            if len(possible_vars) > 0:
                options.append(self.with_additional_information(
                    possible_vars).extend_store)

            possible_buffers = self.buffers
            if len(possible_buffers) > 0:
                options.append(self.with_additional_information(
                    possible_buffers).extend_bufferstore)
        elif isinstance(self.constraint, PrimExprConstraint):
            """Returns a list of TIR constructors or that could satisfy the constraint"""
            boolean_only_returns = [
                self.extend_eq,
                self.extend_ne,
                self.extend_lt,
                self.extend_le,
                self.extend_gt,
                self.extend_ge,
                self.extend_and,
                self.extend_or,
                self.extend_not,
            ]

            int_uint_only_returns = [
                self.extend_floordiv,
                self.extend_floormod,
            ]

            numeric_returns = [
                self.extend_add,
                self.extend_sub,
                self.extend_mul,
                self.extend_div,
                self.extend_mod,
                self.extend_min,
                self.extend_max,
                self.extend_cast,
                self.extend_let,
            ]

            options = []
            if self.constraint.is_dtype_bool:
                options = boolean_only_returns + numeric_returns
            elif self.constraint.is_dtype_int or self.constraint.is_dtype_uint:
                options = int_uint_only_returns + numeric_returns
            elif self.constraint.is_dtype_float:
                options = numeric_returns
            else:
                raise NotImplementedError(
                    "Should not happen!", self.constraint.dtype)

            possible_vars = [
                v for v in self.bound_variables
                if v.dtype == 'handle'
                and 'type_annotation' in dir(v)
                and isinstance(v.type_annotation, tvm.ir.PointerType)
                and DataType(v.type_annotation.element_type.dtype) == self.constraint.dtype
            ]
            if len(possible_vars) > 0:
                options.append(self.with_additional_information(
                    possible_vars).extend_load)

            possible_buffers = [buffer for buffer in self.buffers
                                if DataType(buffer.dtype) == self.constraint.dtype]
            if len(possible_buffers) > 0:
                options.append(self.with_additional_information(
                    possible_buffers).extend_bufferload)
        else:
            raise NotImplementedError(self.constraint)
        return options
