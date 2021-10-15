from typing import Callable, List, Tuple
from tvm import tir
from tvm._ffi.runtime_ctypes import DataType
from tzer.tir import util
from tzer.tir.mutate.generate.generator import Generator
from tzer.tir.semantic.constraint import PrimExprConstraint, satisfies
from tzer.tir.visit.size import MemoizedGetSize
from .mutator import Mutator
from tzer.tir.semantic import Context
from tzer.tir.visit import TIRAbstractTransformer
from tzer.tir.util import TIRNode
from tzer.tir.domain import primitive
from .generate import SizedGenerator
import random
import numpy as np

CALL_ONE_ARG = [
    'tir.exp',
    'tir.exp2',
    'tir.exp10',
    'tir.erf',
    'tir.tanh',
    'tir.sigmoid',
    'tir.log',
    'tir.log2',
    'tir.log10',
    'tir.log1p',
    'tir.tan',
    'tir.cos',
    'tir.cosh',
    'tir.acos',
    'tir.acosh',
    'tir.sin',
    'tir.sinh',
    'tir.asin',
    'tir.asinh',
    'tir.atan',
    'tir.atanh',
    'tir.sqrt',
    'tir.rsqrt',
    'tir.ceil'
    'tir.fabs'
    'tir.round'
    'tir.nearbyint'
    'tir.isnan'
    'tir.isfinite'
    'tir.isinf'
    'tir.popcount',
    'tir.clz',
    'tir.ret',
]

CALL_TWO_ARGS = [
    'tir.ldexp',
    'tir.pow',
    'tir.atan2',
    'tir.hypot'
    'tir.copysign'
    # 'tir.fmod'
]

CALL_FOUR_ARGS = [
    'tir.q_multiply_shift'
]


def match_getter(ops: List[TIRNode]) -> Callable[[Context], TIRNode]:
    assert len(ops) > 0

    def match(context: Context) -> List[TIRNode]:
        options = [op for op in ops if satisfies(op, context.constraint)]
        if len(options) > 0:
            return random.choice(options)
        else:
            get_size = MemoizedGetSize()
            average_size = int(np.average([get_size(op, None) for op in ops]))
            generator = SizedGenerator(lambda: average_size)
            return generator.generate(context)
    return match


class Flipper(TIRAbstractTransformer[Context], Mutator):
    def auto_derive_from_options(self, context: Context, options: List[TIRNode]) -> TIRNode:
        extension = random.choice(context.candidates())()
        match = match_getter(options)
        return extension(match)

    def auto_derive(visit_func: Callable[['Flipper', TIRNode, Context], List[TIRNode]]):  # type: ignore # noqa
        """Automatically use a compatible constructor to build-up subterms."""

        def wrapper_visit(self: 'Flipper', op: TIRNode, context: Context) -> TIRNode:
            options = visit_func(self, op, context)
            return self.auto_derive_from_options(context, options)
        return wrapper_visit

    def mutate(self, op: TIRNode, context: Context) -> TIRNode:
        return self(op, context)

    def will_modify(self, op: TIRNode, context: Context) -> bool:
        return not isinstance(op, (
            tir.PrimFunc,
            tir.Any,
            # tir.For,
            # tir.Var,
            # tir.SizeVar,
        ))

    def visit_primfunc(self, op: tir.PrimFunc, context: Context) -> List[TIRNode]:
        return op

    def constant(self, context: Context) -> TIRNode:
        generator = SizedGenerator(lambda: 0)
        return generator.generate(context)


    def visit_var(self, op: tir.Var, context: Context) -> List[TIRNode]:
        return self.constant(context)

    def visit_sizevar(self, op: tir.SizeVar, context: Context) -> List[TIRNode]:
        return self.constant(context)

    def visit_itervar(self, op: tir.IterVar, context: Context) -> List[TIRNode]:
        return tir.IterVar(
            random.choice(primitive.RANGES),
            op.var,
            util.random_integer(),
            random.choice(primitive.THREADTAGS),
        )

    @auto_derive
    def visit_load(self, op: tir.Load, context: Context) -> List[TIRNode]:
        return [op.index, op.predicate]

    @auto_derive
    def visit_bufferload(self, op: tir.BufferLoad, context: Context) -> List[TIRNode]:
        return list(op.indices)

    def visit_producerload(self, op: tir.ProducerLoad, context: Context) -> TIRNode:
        return tir.ProducerLoad(
            op.producer,
            util.permutation(list(op.indices))
        )

    @auto_derive
    def visit_let(self, op: tir.Let, context: Context) -> List[TIRNode]:
        return [op.value]

    def visit_call(self, op: tir.Call, context: Context) -> List[TIRNode]:
        def get_category():
            intrinsic_name = str(op.op)
            for category in [CALL_ONE_ARG, CALL_TWO_ARGS, CALL_FOUR_ARGS]:
                if intrinsic_name in category:
                    return category
            return None
        category = get_category()
        return tir.Call(
            op.dtype, 
            random.choice(category) if category is not None else op.op, 
            util.permutation(list(op.args))
        )

    @auto_derive
    def visit_add(self, op: tir.Add, context: Context) -> List[TIRNode]:
        return [op.a, op.b]

    @auto_derive
    def visit_sub(self, op: tir.Sub, context: Context) -> List[TIRNode]:
        return [op.a, op.b]

    @auto_derive
    def visit_mul(self, op: tir.Mul, context: Context) -> List[TIRNode]:
        return [op.a, op.b]

    @auto_derive
    def visit_div(self, op: tir.Div, context: Context) -> List[TIRNode]:
        return [op.a, op.b]

    @auto_derive
    def visit_mod(self, op: tir.Mod, context: Context) -> List[TIRNode]:
        return [op.a, op.b]

    @auto_derive
    def visit_floordiv(self, op: tir.FloorDiv, context: Context) -> List[TIRNode]:
        return [op.a, op.b]

    @auto_derive
    def visit_floormod(self, op: tir.FloorMod, context: Context) -> List[TIRNode]:
        return [op.a, op.b]

    @auto_derive
    def visit_min(self, op: tir.Min, context: Context) -> List[TIRNode]:
        return [op.a, op.b]

    @auto_derive
    def visit_max(self, op: tir.Max, context: Context) -> List[TIRNode]:
        return [op.a, op.b]

    @auto_derive
    def visit_eq(self, op: tir.EQ, context: Context) -> List[TIRNode]:
        return [op.a, op.b]

    @auto_derive
    def visit_ne(self, op: tir.NE, context: Context) -> List[TIRNode]:
        return [op.a, op.b]

    @auto_derive
    def visit_lt(self, op: tir.LT, context: Context) -> List[TIRNode]:
        return [op.a, op.b]

    @auto_derive
    def visit_le(self, op: tir.LE, context: Context) -> List[TIRNode]:
        return [op.a, op.b]

    @auto_derive
    def visit_gt(self, op: tir.GT, context: Context) -> List[TIRNode]:
        return [op.a, op.b]

    @auto_derive
    def visit_ge(self, op: tir.GE, context: Context) -> List[TIRNode]:
        return [op.a, op.b]

    @auto_derive
    def visit_and(self, op: tir.And, context: Context) -> List[TIRNode]:
        return [op.a, op.b]

    @auto_derive
    def visit_or(self, op: tir.Or, context: Context) -> List[TIRNode]:
        return [op.a, op.b]

    def visit_reduce(self, op: tir.Reduce, context: Context) -> TIRNode:
        return tir.Reduce(
            util.permutation(list(op.src)),
            util.permutation(list(op.rdom)),
            op.condition,
            util.random_integer(),
            None if op.init is None else util.permutation(list(op.init))
        )

    @auto_derive
    def visit_cast(self, op: tir.Cast, context: Context) -> List[TIRNode]:
        return [op]

    @auto_derive
    def visit_not(self, op: tir.Not, context: Context) -> List[TIRNode]:
        return [op.a]

    @auto_derive
    def visit_select(self, op: tir.Select, context: Context) -> List[TIRNode]:
        return [op.condition, op.true_value, op.false_value]

    @auto_derive
    def visit_ramp(self, op: tir.Ramp, context: Context) -> List[TIRNode]:
        return [op.base]

    @auto_derive
    def visit_broadcast(self, op: tir.Broadcast, context: Context) -> List[TIRNode]:
        return [op.value]

    @auto_derive
    def visit_shuffle(self, op: tir.Shuffle, context: Context) -> TIRNode:
        return [*list(op.vectors), *list(op.indices)]

    def visit_intimm(self, op: tir.IntImm, context: Context) -> List[TIRNode]:
        assert isinstance(context.constraint, PrimExprConstraint)
        num = (util.random_uint() if context.constraint.is_dtype_uint
               else random.choice([True, False]) if context.constraint.is_dtype_bool
               else random.choice([
                   lambda x: x + 1,
                   lambda x: x - 1,
                   lambda x: x * -1,
               ])(op.value))
        return random.choice([tir.IntImm(op.dtype, num), self.constant(context)])

    def visit_floatimm(self, op: tir.FloatImm, context: Context) -> List[TIRNode]:
        return self.constant(context)

    def visit_stringimm(self, op: tir.StringImm, context: Context) -> List[TIRNode]:
        return tir.StringImm(util.random_string())

    def visit_any(self, op: tir.Any, context: Context) -> List[TIRNode]:
        return op

    def visit_attrstmt(self, op: tir.AttrStmt, context: Context) -> TIRNode:
        return tir.AttrStmt(
            op.node,
            random.choice(primitive.ATTRKEYS),
            op.value,
            op.body,
        )

    @auto_derive
    def visit_ifthenelse(self, op: tir.IfThenElse, context: Context) -> List[TIRNode]:
        return [op.condition, op.then_case] + ([] if op.else_case is None else [op.else_case])

    @auto_derive
    def visit_letstmt(self, op: tir.LetStmt, context: Context) -> List[TIRNode]:
        return [op.value]

    @auto_derive
    def visit_for(self, op: tir.For, context: Context) -> TIRNode:
        return [op.min, op.extent, op.body]

    @auto_derive
    def visit_while(self, op: tir.stmt.While, context: Context) -> List[TIRNode]:
        return [op.condition, op.body]

    @auto_derive
    def visit_allocate(self, op: tir.Allocate, context: Context) -> List[TIRNode]:
        return [*list(op.extents), op.condition, op.body]

    @auto_derive
    def visit_store(self, op: tir.Store, context: Context) -> List[TIRNode]:
        return [op.value, op.predicate, op.index]

    @auto_derive
    def visit_bufferstore(self, op: tir.BufferStore, context: Context) -> List[TIRNode]:
        return [op.value, *list(op.indices)]

    @auto_derive
    def visit_bufferrealize(self, op: tir.BufferRealize, context: Context) -> List[TIRNode]:
        return [op.condition, op.body]

    @auto_derive
    def visit_assertstmt(self, op: tir.AssertStmt, context: Context) -> List[TIRNode]:
        return [op.condition, op.body]

    @auto_derive
    def visit_producerstore(self, op: tir.ProducerStore, context: Context) -> List[TIRNode]:
        return [op.value, *list(op.indices)]

    @auto_derive
    def visit_producerrealize(self, op: tir.ProducerRealize, context: Context) -> List[TIRNode]:
        return [op.condition, op.body]

    def visit_prefetch(self, op: tir.Prefetch, context: Context) -> List[TIRNode]:
        return tir.Prefetch(
            op.buffer,
            util.permutation(op.bounds),
        )

    @auto_derive
    def visit_seqstmt(self, op: tir.SeqStmt, context: Context) -> TIRNode:
        return list(op.seq)

    @auto_derive
    def visit_evaluate(self, op: tir.Evaluate, context: Context) -> List[TIRNode]:
        return [op.value]

    def visit_block(self, op: tir.Block, context: Context) -> TIRNode:
        return random.choice([lambda: tir.Block(
            util.permutation(list(op.iter_vars)),
            util.permutation(list(op.reads)),
            util.permutation(list(op.writes)),
            util.random_string(),
            op.body,
            op.init,
            util.permutation(list(op.alloc_buffers)),
            op.annotations
        ),
            lambda: self.auto_derive_from_options(
                context, [op.body] + [] if op.init is None else [op.init])
        ])()

    @auto_derive
    def visit_blockrealize(self, op: tir.BlockRealize, context: Context) -> List[TIRNode]:
        return [op.predicate, op.block]
