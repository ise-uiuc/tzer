from typing import Callable, Dict, List, Optional, Tuple
from tvm import tir
from tzer.tir import util
from tzer.tir.semantic.constraint import PrimExprConstraint, StmtConstraint
from tzer.tir.visit import TIRAbstractTransformer
from tzer.tir.util import TIRNode
from dataclasses import dataclass
import random
from tzer.tir.visit.size import MemoizedGetSize, get_node_size
from tzer.tir.semantic import Context, context
from tzer.tir import semantic
from tzer.tir.semantic import Constraint
from .mutator import IRMutator, Mutator


class RecursiveMutatorCombinator(TIRAbstractTransformer[Context], Mutator, IRMutator):
    """A mutator combinator that recursively select one node to mutate, wherein
    the nodes should be selected (roughly) uniformly.

    NOTE(now fixed roughly): the `Context` design is ad-hoc and not good, since it breaks the do-one-thing
    principle in that `RecursiveMutate` helps to perform variable rebinding, which makes it couple
    with other mutators. To fix this, we'd better have a global bind-table/semantic-table for
    all mutators to access.

    NOTE(now fixed roughly): one possible improvement for this combinator when selecting mutators to apply
    is to select only the mutators that will really mutate `op`. This feature could be achieved
    if we have a more detailed modeling of mutators.

    If we could fix the problems mentioned in the above notes,
    the mutator could be *general* instead of *ad-hoc*.
    """

    def __init__(
        self,
        weighted_mutators: List[Tuple[int, Mutator]],
        touch_control_flow: bool = True,
    ) -> None:
        self.touch_control_flow = touch_control_flow
        self.weighted_mutators = weighted_mutators
        self.reset_memoized_size_object()

    def mutate_ir(self, op: tir.PrimFunc) -> tir.PrimFunc:
        return self.mutate(op, Context.prim_func_context())

    # def visit(self, op, arg):
    #     print(type(op))
    #     return super().visit(op, arg)

    def mutate(self, op: TIRNode, context: Context) -> TIRNode:
        result = self(op, context)
        self.reset_memoized_size_object()
        return result

    def will_modify(self, op: TIRNode, context: Context) -> bool:
        return len([_ for _, mutator in self.weighted_mutators
                    if mutator.will_modify(op, context)]) > 0

    def reset_memoized_size_object(self):
        self._get_node_size = MemoizedGetSize()

    def get_node_size(self, op) -> int:
        return self._get_node_size(op, None)

    def mutate_node(self, op: TIRNode, context: Context) -> TIRNode:
        # if isinstance(op, tir.Var):
        #     """Bad thing here: it's coupled with other mutators. Code is written as such
        #     since we *know* that no mutators modify variables."""
        #     bound_variables = list(context.bound_variables)
        #     return random.choice(bound_variables) if len(bound_variables) > 0 else op
        mutators = [(weight, mutator) for weight,
                    mutator in self.weighted_mutators if mutator.will_modify(op, context)]
        if len(mutators) == 0:
            return op
        else:
            return util.weighted_select(mutators).mutate(op, context)

    def visit_primfunc(self, op: tir.PrimFunc, context: Context) -> TIRNode:
        options = [
            (1, lambda: self.mutate_node(op, context)),
            (self.get_node_size(op.body), lambda: tir.PrimFunc(
                params=op.params,
                body=self(op.body, context
                     .with_constraint_being(semantic.StmtConstraint())
                     .with_bound_vars_being([*op.params, *context.bound_variables])),
                ret_type=op.ret_type,
                buffer_map=op.buffer_map,
                attrs=op.attrs,
            )),
        ]
        return util.weighted_select(options)()

    def visit_var(self, op: tir.Var, context: Context) -> TIRNode:
        return self.mutate_node(op, context)

    def visit_sizevar(self, op: tir.SizeVar, context: Context) -> TIRNode:
        return self.mutate_node(op, context)

    def visit_load(self, op: tir.Load, context: Context) -> TIRNode:
        options = [(1, lambda: self.mutate_node(op, context))]
        options.append((self.get_node_size(op.index), lambda: tir.Load(
            op.dtype,
            op.buffer_var,
            self(op.index, context),
            op.predicate,
        )))
        if op.predicate is not None:
            options.append((self.get_node_size(op.predicate), lambda: tir.Load(
                op.dtype,
                op.buffer_var,
                op.index,
                self(op.predicate, context),
            )))
        return util.weighted_select(options)()

    def visit_bufferload(self, op: tir.BufferLoad, context: Context) -> TIRNode:
        options = [(1, lambda: self.mutate_node(op, context))]
        indices = list(op.indices)
        for index_to_mutate in indices:
            options.append((self.get_node_size(index_to_mutate), lambda: tir.BufferLoad(
                op.buffer,
                [index if index is not index_to_mutate
                 else self(index_to_mutate, context.with_constraint_being(
                     PrimExprConstraint(index_to_mutate.dtype),
                 )) for index in indices]
            )))
        return util.weighted_select(options)()

    def visit_producerload(self, op: tir.ProducerLoad, context: Context) -> TIRNode:
        options = [(1, lambda: self.mutate_node(op, context))]
        indices = list(op.indices)
        for index_to_mutate in indices:
            options.append((self.get_node_size(index_to_mutate), lambda: tir.ProducerLoad(
                op.producer,
                [index if index is not index_to_mutate
                 else self(index_to_mutate, context.with_constraint_being(
                     PrimExprConstraint(index_to_mutate.dtype),
                 )) for index in indices]
            )))
        return util.weighted_select(options)()

    def visit_let(self, op: tir.Let, context: Context) -> TIRNode:
        options = [
            (1, lambda: self.mutate_node(op, context)),
            (self.get_node_size(op.value), lambda: tir.Let(
                op.var,
                self(op.value, context.with_constraint_being(
                    PrimExprConstraint(op.value.dtype)
                )),
                op.body,
            )),
            (self.get_node_size(op.body), lambda: tir.Let(
                op.var,
                op.value,
                self(op.body, context
                     .with_constraint_being(PrimExprConstraint(op.body.dtype))
                     .with_bound_vars_being([*context.bound_variables, op.var])
                     )
            ))
        ]
        return util.weighted_select(options)()

    def visit_call(self, op: tir.Call, context: Context) -> TIRNode:
        options = [(1, lambda: self.mutate_node(op, context))]
        args = list(op.args)
        for arg_to_mutate in args:
            options.append((self.get_node_size(arg_to_mutate), lambda: tir.Call(
                op.dtype,
                op.op,
                [a if a is not arg_to_mutate else self(arg_to_mutate, context.with_constraint_being(
                    PrimExprConstraint(arg_to_mutate.dtype)))
                 for a in args]
            )))
        return util.weighted_select(options)()

    def visit_binop(self, op: tir.expr.BinaryOpExpr, context: Context) -> TIRNode:
        options = [
            (1, lambda: self.mutate_node(op, context)),
            (self.get_node_size(op.a), lambda: op.__class__(
                self(op.a, context.with_constraint_being(
                    PrimExprConstraint(op.a.dtype)
                )), op.b)),
            (self.get_node_size(op.b), lambda: op.__class__(
                op.a, self(op.b, context.with_constraint_being(
                    PrimExprConstraint(op.b.dtype)
                )))),
        ]
        return util.weighted_select(options)()

    def visit_add(self, op: tir.Add, context: Context) -> TIRNode:
        return self.visit_binop(op, context)

    def visit_sub(self, op: tir.Sub, context: Context) -> TIRNode:
        return self.visit_binop(op, context)

    def visit_mul(self, op: tir.Mul, context: Context) -> TIRNode:
        return self.visit_binop(op, context)

    def visit_div(self, op: tir.Div, context: Context) -> TIRNode:
        return self.visit_binop(op, context)

    def visit_mod(self, op: tir.Mod, context: Context) -> TIRNode:
        return self.visit_binop(op, context)

    def visit_floordiv(self, op: tir.FloorDiv, context: Context) -> TIRNode:
        return self.visit_binop(op, context)

    def visit_floormod(self, op: tir.FloorMod, context: Context) -> TIRNode:
        return self.visit_binop(op, context)

    def visit_min(self, op: tir.Min, context: Context) -> TIRNode:
        return self.visit_binop(op, context)

    def visit_max(self, op: tir.Max, context: Context) -> TIRNode:
        return self.visit_binop(op, context)

    def visit_eq(self, op: tir.EQ, context: Context) -> TIRNode:
        return self.visit_binop(op, context)

    def visit_ne(self, op: tir.NE, context: Context) -> TIRNode:
        return self.visit_binop(op, context)

    def visit_lt(self, op: tir.LT, context: Context) -> TIRNode:
        return self.visit_binop(op, context)

    def visit_le(self, op: tir.LE, context: Context) -> TIRNode:
        return self.visit_binop(op, context)

    def visit_gt(self, op: tir.GT, context: Context) -> TIRNode:
        return self.visit_binop(op, context)

    def visit_ge(self, op: tir.GE, context: Context) -> TIRNode:
        return self.visit_binop(op, context)

    def visit_and(self, op: tir.And, context: Context) -> TIRNode:
        return self.visit_binop(op, context)

    def visit_or(self, op: tir.Or, context: Context) -> TIRNode:
        return self.visit_binop(op, context)

    def visit_reduce(self, op: tir.Reduce, context: Context) -> TIRNode:
        options = [
            (1, lambda: self.mutate_node(op, context)),
            (self.get_node_size(op.condition), lambda: tir.Reduce(
                op.combinder,
                op.src,
                op.rdom,
                self(op.condition, context.with_constraint_being(
                    PrimExprConstraint(op.condition.dtype)
                )),
                op.value_index,
                op.init,
            )),
        ]
        if op.init is not None:
            options.append((self.get_node_size(op.init), lambda: tir.Reduce(
                op.combiner,
                op.src,
                op.rdom,
                op.condition,
                op.value_index,
                self(op.init, context.with_constraint_being(
                    PrimExprConstraint(op.init.dtype),
                )),
            )))
        srcs = list(op.src)
        for src_to_mutate in srcs:
            options.append((self.get_node_size(src_to_mutate), lambda: tir.Reduce(
                op.combiner,
                [src if src is not src_to_mutate else self(
                    src_to_mutate, context.with_constraint_being(
                        PrimExprConstraint(src_to_mutate.dtype)
                    )) for src in srcs],
                op.rdom,
                op.condition,
                op.value_index,
                op.init,
            )))
        return util.weighted_select(options)()

    def visit_cast(self, op: tir.Cast, context: Context) -> TIRNode:
        options = [
            (1, lambda: self.mutate_node(op, context)),
            (self.get_node_size(op.value), lambda: tir.Cast(
                context.constraint.dtype,  # type: ignore
                self(op.value, context.with_constraint_being(
                     PrimExprConstraint(op.value.dtype))))),
        ]
        return util.weighted_select(options)()

    def visit_not(self, op: tir.Not, context: Context) -> TIRNode:
        options = [
            (1, lambda: self.mutate_node(op, context)),
            (self.get_node_size(op.a), lambda: tir.Not(self(op.a, context.with_constraint_being(
                PrimExprConstraint(op.a.dtype)
            ))))
        ]
        return util.weighted_select(options)()

    def visit_select(self, op: tir.Select, context: Context) -> TIRNode:
        options = [
            (1, lambda: self.mutate_node(op, context)),
            (self.get_node_size(op.condition), lambda: tir.Select(
                self(op.condition, context.with_constraint_being(
                    PrimExprConstraint(op.condition.dtype)
                )),
                op.true_value,
                op.false_value,
            )),
            (self.get_node_size(op.true_value), lambda: tir.Select(
                op.condition,
                self(op.true_value, context.with_constraint_being(
                    PrimExprConstraint(op.true_value.dtype)
                )),
                op.false_value,
            )),
            (self.get_node_size(op.false_value), lambda: tir.Select(
                op.condition,
                op.true_value,
                self(op.false_value, context.with_constraint_being(
                    PrimExprConstraint(op.false_value.dtype)
                )),
            )),
        ]
        return util.weighted_select(options)()

    def visit_ramp(self, op: tir.Ramp, context: Context) -> TIRNode:
        options = [
            (1, lambda: self.mutate_node(op, context)),
            (self.get_node_size(op.base), lambda: tir.Ramp(
                self(op.base, context.with_constraint_being(
                    PrimExprConstraint(op.base.dtype)
                )),
                op.stride,
                op.lanes,
            ))
        ]
        return util.weighted_select(options)()

    def visit_broadcast(self, op: tir.Broadcast, context: Context) -> TIRNode:
        options = [
            (1, lambda: self.mutate_node(op, context)),
            (self.get_node_size(op.value), lambda: tir.Broadcast(
                self(op.value, context.with_constraint_being(
                    PrimExprConstraint(op.value.dtype)
                )),
                op.lanes,
            ))
        ]
        return util.weighted_select(options)()

    def visit_shuffle(self, op: tir.Shuffle, context: Context) -> TIRNode:
        options = [(1, lambda: self.mutate_node(op, context))]
        vectors = list(op.vectors)
        indices = list(op.indices)
        for vector_to_mutate in vectors:
            options.append((self.get_node_size(vector_to_mutate), lambda: tir.Shuffle(
                [vector if vector is not vector_to_mutate
                 else self(vector_to_mutate, context.with_constraint_being(
                     PrimExprConstraint(vector_to_mutate.dtype)
                 )) for vector in vectors],
                op.indices,
            )))
        for index_to_mutate in indices:
            options.append((self.get_node_size(index_to_mutate), lambda: tir.Shuffle(
                op.vectors,
                [index if index is not index_to_mutate
                 else self(index_to_mutate, context.with_constraint_being(
                     PrimExprConstraint(index_to_mutate.dtype)
                 )) for index in indices],
            )))
        return util.weighted_select(options)()

    def visit_intimm(self, op: tir.IntImm, context: Context) -> TIRNode:
        return self.mutate_node(op, context)

    def visit_floatimm(self, op: tir.FloatImm, context: Context) -> TIRNode:
        return self.mutate_node(op, context)

    def visit_stringimm(self, op: tir.StringImm, context: Context) -> TIRNode:
        return self.mutate_node(op, context)

    def visit_any(self, op: tir.Any, context: Context) -> TIRNode:
        return self.mutate_node(op, context)

    def visit_attrstmt(self, op: tir.AttrStmt, context: Context) -> TIRNode:
        options = [
            (1, lambda: self.mutate_node(op, context)),
            (self.get_node_size(op.value), lambda: tir.AttrStmt(
                op.node,
                op.attr_key,
                self(op.value, context.with_constraint_being(
                    PrimExprConstraint(op.value.dtype)
                )),
                op.body,
            )),
            (self.get_node_size(op.body), lambda: tir.AttrStmt(
                op.node,
                op.attr_key,
                op.value,
                self(op.body, context.with_constraint_being(
                    StmtConstraint()
                )),
            ))
        ]
        return util.weighted_select(options)()

    def visit_ifthenelse(self, op: tir.IfThenElse, context: Context) -> TIRNode:
        options = [(1, lambda: self.mutate_node(op, context))
                   ] if self.touch_control_flow else []
        options.extend([
            (self.get_node_size(op.condition), lambda: tir.IfThenElse(
                self(op.condition, context.with_constraint_being(
                    PrimExprConstraint(op.condition.dtype)
                )),
                op.then_case,
                op.else_case,
            )),
            (self.get_node_size(op.then_case), lambda: tir.IfThenElse(
                op.condition,
                self(op.then_case, context.with_constraint_being(
                    StmtConstraint()
                )),
                op.else_case,
            )),
        ])
        if op.else_case is not None:
            options.append(self.get_node_size(op.else_case), lambda: tir.IfThenElse(
                op.condition,
                op.then_case,
                self(op.else_case, context.with_constraint_being(
                    StmtConstraint()
                )),
            ))
        return util.weighted_select(options)()

    def visit_letstmt(self, op: tir.LetStmt, context: Context) -> TIRNode:
        options = [
            (1, lambda: self.mutate_node(op, context)),
            (self.get_node_size(op.value), lambda: tir.LetStmt(
                op.var,
                self(op.value, context.with_constraint_being(
                    PrimExprConstraint(op.value.dtype)
                )),
                op.body,
            )),
            (self.get_node_size(op.body), lambda: tir.LetStmt(
                op.var,
                op.value,
                self(op.body, context
                     .with_constraint_being(StmtConstraint())
                     .with_bound_vars_being([*context.bound_variables, op.var]))
            )),
        ]
        return util.weighted_select(options)()

    def visit_for(self, op: tir.For, context: Context) -> TIRNode:
        options = [(1, lambda: self.mutate_node(op, context))
                   ] if self.touch_control_flow else []
        options.extend([
            (self.get_node_size(op.min), lambda: tir.For(
                op.loop_var,
                self(op.min, context.with_constraint_being(
                    PrimExprConstraint(op.min.dtype)
                )),
                op.extent,
                op.kind,
                op.body,
                op.thread_binding,
                op.annotations,
            )),
            (self.get_node_size(op.extent), lambda: tir.For(
                op.loop_var,
                op.min,
                self(op.extent, context.with_constraint_being(
                    PrimExprConstraint(op.extent.dtype)
                )),
                op.kind,
                op.body,
                op.thread_binding,
                op.annotations,
            )),
            (self.get_node_size(op.body), lambda: tir.For(
                op.loop_var,
                op.min,
                op.extent,
                op.kind,
                self(op.body, context.with_constraint_being(
                    StmtConstraint()
                )),
                op.thread_binding,
                op.annotations,
            )),
        ])
        return util.weighted_select(options)()

    def visit_while(self, op: tir.stmt.While, context: Context) -> TIRNode:
        options = [(1, lambda: self.mutate_node(op, context))
                   ] if self.touch_control_flow else []
        options.extend([
            (self.get_node_size(op.condition), lambda: tir.stmt.While(
                self(op.condition, context.with_constraint_being(
                    PrimExprConstraint(op.condition.dtype)
                )),
                op.body,
            )),
            (self.get_node_size(op.body), lambda: tir.stmt.While(
                op.condition,
                self(op.body, context.with_constraint_being(
                    StmtConstraint()
                )),
            ))
        ])
        return util.weighted_select(options)()

    def visit_allocate(self, op: tir.Allocate, context: Context) -> TIRNode:
        options = [
            (1, lambda: self.mutate_node(op, context)),
            (self.get_node_size(op.condition), lambda: tir.Allocate(
                op.buffer_var,
                op.dtype,
                op.extents,
                self(op.condition, context.with_constraint_being(
                    PrimExprConstraint(op.condition.dtype)
                )),
                op.body,
            )),
            (self.get_node_size(op.body), lambda: tir.Allocate(
                op.buffer_var,
                op.dtype,
                op.extents,
                op.condition,
                self(op.body, context
                     .with_constraint_being(StmtConstraint())
                     .with_bound_vars_being([*context.bound_variables, op.buffer_var]))
            ))
        ]
        extents = list(op.extents)
        for extent_to_mutate in extents:
            options.append((self.get_node_size(extent_to_mutate), lambda: tir.Allocate(
                op.buffer_var,
                op.dtype,
                [extent if extent is not extent_to_mutate
                 else self(extent_to_mutate, context.with_constraint_being(
                     PrimExprConstraint(extent_to_mutate.dtype)
                 )) for extent in extents],
                op.condition,
                op.body,
            )))
        return util.weighted_select(options)()

    def visit_store(self, op: tir.Store, context: Context) -> TIRNode:
        options = [
            (1, lambda: self.mutate_node(op, context)),
            (self.get_node_size(op.value), lambda: tir.Store(
                op.buffer_var,
                self(op.value, context.with_constraint_being(
                    PrimExprConstraint(op.value.dtype)
                )),
                op.index,
                op.predicate
            )),
            (self.get_node_size(op.index), lambda: tir.Store(
                op.buffer_var,
                op.value,
                self(op.index, context.with_constraint_being(
                    PrimExprConstraint(op.value.dtype),
                )),
                op.predicate
            )),
        ]
        if op.predicate is not None:
            options.append((self.get_node_size(op.predicate), lambda: tir.Store(
                op.buffer_var,
                op.value,
                op.index,
                self(op.predicate, context.with_constraint_being(
                    PrimExprConstraint(op.predicate.dtype)
                )),
            )))
        return util.weighted_select(options)()

    def visit_bufferstore(self, op: tir.BufferStore, context: Context) -> TIRNode:
        options = [
            (1, lambda: self.mutate_node(op, context)),
            (self.get_node_size(op.value), lambda: tir.BufferStore(
                op.buffer,
                self(op.value, context.with_constraint_being(
                    PrimExprConstraint(op.value.dtype)
                )),
                op.indices,
            )),
        ]
        indices = list(op.indices)
        for index_to_mutate in indices:
            options.append((self.get_node_size(index_to_mutate), lambda: tir.BufferStore(
                op.buffer,
                op.value,
                [index if index is not index_to_mutate
                 else self(index_to_mutate, context.with_constraint_being(
                     PrimExprConstraint(index_to_mutate.dtype)
                 )) for index in indices],
            )))
        return util.weighted_select(options)()

    def visit_bufferrealize(self, op: tir.BufferRealize, context: Context) -> TIRNode:
        options = [
            (1, lambda: self.mutate_node(op, context)),
            (self.get_node_size(op.condition), lambda: tir.BufferRealize(
                op.buffer,
                op.bounds,
                self(op.condition, context.with_constraint_being(
                    PrimExprConstraint(op.condition.dtype)
                )),
                op.body,
            )),
            (self.get_node_size(op.body), lambda: tir.BufferRealize(
                op.buffer,
                op.bounds,
                op.condition,
                self(op.body, context.with_constraint_being(
                    StmtConstraint()
                )),
            )),
        ]
        return util.weighted_select(options)()

    def visit_assertstmt(self, op: tir.AssertStmt, context: Context) -> TIRNode:
        options = [
            (1, lambda: self.mutate_node(op, context)),
            (self.get_node_size(op.condition), lambda: tir.AssertStmt(
                self(op.condition, context.with_constraint_being(
                    PrimExprConstraint(op.condition.dtype)
                )),
                op.message,
                op.body,
            )),
            (self.get_node_size(op.message), lambda: tir.AssertStmt(
                op.condition,
                self(op.message, context.with_constraint_being(
                    PrimExprConstraint(op.message.dtype)
                )),
                op.body,
            )),
            (self.get_node_size(op.body), lambda: tir.AssertStmt(
                op.condition,
                op.message,
                self(op.body, context.with_constraint_being(
                    StmtConstraint()
                )),
            ))
        ]
        return util.weighted_select(options)()

    def visit_producerstore(self, op: tir.ProducerStore, context: Context) -> TIRNode:
        options = [
            (1, lambda: self.mutate_node(op, context)),
            (self.get_node_size(op.value), lambda: tir.ProducerStore(
                op.producer,
                self(op.value, context.with_constraint_being(
                    PrimExprConstraint(op.value.dtype)
                )),
                op.indices,
            )),
        ]
        indices = list(op.indices)
        for index_to_mutate in indices:
            options.append((self.get_node_size(index_to_mutate), lambda: tir.ProducerStore(
                op.producer,
                op.value,
                [index if index is not index_to_mutate
                 else self(index_to_mutate, context.with_constraint_being(
                     PrimExprConstraint(index_to_mutate.dtype)
                 )) for index in indices],
            )))
        return util.weighted_select(options)()

    def visit_producerrealize(self, op: tir.ProducerRealize, context: Context) -> TIRNode:
        options = [
            (1, lambda: self.mutate_node(op, context)),
            (self.get_node_size(op.condition), lambda: tir.ProducerRealize(
                op.producer,
                op.bounds,
                self(op.condition, context.with_constraint_being(
                    PrimExprConstraint(op.condition.dtype)
                )),
                op.body,
            )),
            (self.get_node_size(op.body), lambda: tir.ProducerRealize(
                op.producer,
                op.bounds,
                op.condition,
                self(op.body, context.with_constraint_being(
                    StmtConstraint()
                )),
            )),
        ]
        return util.weighted_select(options)()

    def visit_prefetch(self, op: tir.Prefetch, context: Context) -> TIRNode:
        return self.mutate_node(op, context)

    def visit_seqstmt(self, op: tir.SeqStmt, context: Context) -> TIRNode:
        options = [(1, lambda: self.mutate_node(op, context))]
        seq = list(op.seq)
        for stmt_to_mutate in seq:
            options.append((self.get_node_size(stmt_to_mutate), lambda: tir.SeqStmt(
                [stmt if stmt is not stmt_to_mutate
                 else self(stmt_to_mutate, context.with_constraint_being(
                     StmtConstraint()
                 )) for stmt in seq]
            )))
        return util.weighted_select(options)()

    def visit_evaluate(self, op: tir.Evaluate, context: Context) -> TIRNode:
        options = [
            (1, lambda: self.mutate_node(op, context)),
            (self.get_node_size(op.value), lambda: tir.Evaluate(
                self(op.value, context.with_constraint_being(
                    PrimExprConstraint(op.value.dtype)
                ))
            ))
        ]
        return util.weighted_select(options)()

    def visit_block(self, op: tir.Block, context: Context) -> TIRNode:
        options = [
            (1, lambda: self.mutate_node(op, context)),
            (self.get_node_size(op.body), lambda: tir.Block(
                op.iter_vars,
                op.reads,
                op.writes,
                op.name_hint,
                self(op.body, context
                     .with_constraint_being(semantic.StmtConstraint())
                     .with_bound_vars_being([*context.bound_variables, *[v.var for v in op.iter_vars]])),
                op.init,
                op.alloc_buffers,
                op.match_buffers,
                op.annotations,
            ))
        ]
        if op.init is not None:
            options.append((self.get_node_size(op.init), lambda: tir.Block(
                op.iter_vars,
                op.reads,
                op.writes,
                op.name_hint,
                op.body,
                self(op.init, context.with_constraint_being(
                    PrimExprConstraint(op.init.dtype),
                )),
                op.alloc_buffers,
                op.match_buffers,
                op.annotations,
            )))
        return util.weighted_select(options)()

    def visit_blockrealize(self, op: tir.BlockRealize, context: Context) -> TIRNode:
        options = [
            (1, lambda: self.mutate_node(op, context)),
            (self.get_node_size(op.block), lambda: tir.BlockRealize(
                op.iter_values,
                op.predicate,
                self(op.block, context.with_constraint_being(
                    semantic.BlockConstraint())),
            ))
        ]
        if isinstance(op.predicate, tir.PrimExpr):
            options.append((self.get_node_size(op.predicate), lambda: tir.BlockRealize(
                op.iter_values,
                self(op.predicate, context.with_constraint_being(
                    PrimExprConstraint(op.predicate.dtype))),
                op.block,
            )))

        iter_values = list(op.iter_values)
        for iter_value_to_mutate in iter_values:
            options.append((self.get_node_size(iter_value_to_mutate), lambda: tir.BlockRealize(
                [iter_value if iter_value is not iter_value_to_mutate
                 else self(iter_value_to_mutate, context.with_constraint_being(
                     PrimExprConstraint(iter_value_to_mutate.dtype)))
                 for iter_value in iter_values],
                op.predicate,
                op.block,
            )))
        return util.weighted_select(options)()
