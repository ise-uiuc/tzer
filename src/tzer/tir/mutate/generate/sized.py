from typing import Callable, List, Optional, Tuple
from tvm import tir
import tvm
from tvm.runtime import DataType
from tvm.tir.op import call_intrin
from tzer.tir import util
from tzer.tir.semantic.constraint import PrimExprConstraint, StmtConstraint, VarConstraint
from tzer.tir.util import TIRNode
import random
from tzer.tir import semantic
from tzer.tir.semantic import Context
from .generator import Generator
from tzer.tir import domain


class SizedGenerator(Generator):
    def __init__(self, get_size: Callable[[], int]) -> None:
        self._get_size = get_size

    def can_generate(self, op_type: type, context: Context) -> bool:
        return not op_type in [
            tir.ProducerLoad,
            tir.Reduce,
            tir.ProducerStore,
            tir.ProducerRealize,
            tir.Block,
            tir.BlockRealize,
            tir.Any,
        ]

    def get_size(self) -> int:
        return self._get_size()

    def generate(self, context: Context) -> TIRNode:
        return self.generate_sized(context, self.get_size())

    def generate_sized(self, context: Context, size: int) -> TIRNode:
        variant = context.constraint
        if isinstance(variant, semantic.PrimExprConstraint):
            return self.generate_sized_prim_expr(context, size)
        elif isinstance(variant, semantic.StmtConstraint):
            return self.generate_sized_stmt(context, size)
        elif isinstance(variant, semantic.VarConstraint):
            return self.generate_var(context)
        elif isinstance(variant, semantic.BlockConstraint):
            raise NotImplementedError
            # return self.generate_sized_block(context, size)
        elif isinstance(variant, semantic.PrimFuncConstraint):
            return self.generate_sized_prim_func(size)
        else:
            raise NotImplementedError

    # Methods below are private. Please don't call them directly

    def get_satisfied_options_for_prim_expr(self, context: Context, size: int) -> List[Callable[[Context, int], TIRNode]]:
        """The core analysis"""
        constraint = context.constraint
        assert isinstance(constraint, PrimExprConstraint)
        if size == 0:
            satisfied: List[Tuple[int, Callable[[Context, int], TIRNode]]] = []  # noqa
            if constraint.is_dtype_numeric and constraint.var_should_be_bound:
                satisfied_vars = [
                    v for v in context.bound_variables
                    if DataType(v.dtype) == constraint.dtype
                ]
                if len(satisfied_vars) > 0:
                    satisfied.append((10, lambda ctx, size: self.generate_var(
                        ctx.with_additional_information(satisfied_vars)
                    )))
            if constraint.is_dtype_int or constraint.is_dtype_uint or constraint.is_dtype_bool:
                satisfied.append(
                    (1, lambda ctx, size: self.generate_intimm(ctx)))
            if constraint.is_dtype_float:
                satisfied.append(
                    (1, lambda ctx, size: self.generate_floatimm(ctx)))
            scalar_option = util.weighted_select(satisfied)
            return [scalar_option] if constraint.dtype.lanes == 1 else [lambda ctx, size: tir.Broadcast(
                scalar_option(ctx.with_constraint_being(
                    PrimExprConstraint(str(constraint.dtype).split('x')[  # type: ignore
                                       0])
                ), size),
                constraint.dtype.lanes,  # type: ignore
            )]
        else:
            boolean_only_returns: List[Callable[[Context, int], TIRNode]] = [
                self.generate_sized_eq,
                self.generate_sized_ne,
                self.generate_sized_lt,
                self.generate_sized_le,
                self.generate_sized_gt,
                self.generate_sized_ge,
                self.generate_sized_and,
                self.generate_sized_or,
                self.generate_sized_not,
            ]
            # float_only_returns: List[Callable[[Context, int], TIRNode]] = [
            #     lambda ctx, size: self.generate_floatimm(ctx),
            # ]
            # numeric_without_float_returns: List[Callable[[Context, int], TIRNode]] = [
            #     lambda ctx, size: self.generate_intimm(ctx),
            # ]
            int_uint_only_returns: List[Callable[[Context, int], TIRNode]] = [
                self.generate_sized_floordiv,
                self.generate_sized_floormod,
            ]

            numeric_returns: List[Callable[[Context, int], TIRNode]] = [
                self.generate_sized_add,
                self.generate_sized_sub,
                self.generate_sized_mul,
                self.generate_sized_div,
                self.generate_sized_mod,
                self.generate_sized_min,
                self.generate_sized_max,
                self.generate_sized_cast,
                self.generate_sized_let,
            ]

            # _1 here to pass the type checker
            satisfied_1: List[Callable[[Context, int], TIRNode]] = []
            if constraint.is_dtype_bool:
                satisfied_1 = boolean_only_returns + numeric_returns
            elif constraint.is_dtype_int or constraint.is_dtype_uint:
                satisfied_1 = int_uint_only_returns + numeric_returns
            elif constraint.is_dtype_float:
                satisfied_1 = numeric_returns
            else:
                raise NotImplementedError(
                    "Should not happen!", constraint.dtype)
            # for v in context.bound_variables:
            #     if v.dtype == 'handle':
            #         if 'type_annotation' in dir(v):
            #             if isinstance(v.type_annotation, tvm.ir.PointerType):
            #                 print(v.type_annotation.element_type, constraint.dtype)
            possible_vars = [
                v for v in context.bound_variables
                if v.dtype == 'handle'
                and 'type_annotation' in dir(v)
                and isinstance(v.type_annotation, tvm.ir.PointerType)
                and DataType(v.type_annotation.element_type.dtype) == constraint.dtype
            ]
            if len(possible_vars) > 0:
                satisfied_1.append(lambda ctx, size: self.generate_sized_load(
                    ctx.with_additional_information(possible_vars),
                    size,
                ))
            possible_buffers = [buffer for buffer in context.buffers
                                if DataType(buffer.dtype) == constraint.dtype]
            if len(possible_buffers) > 0:
                satisfied_1.append(lambda ctx, size: self.generate_sized_bufferload(
                    ctx.with_additional_information(possible_buffers),
                    size,
                ))

            # Shuffle
            if size - 1 - constraint.dtype.lanes >= constraint.dtype.lanes:
                satisfied_1.append(self.generate_sized_shuffle)

            # Call
            def id_ctx(ctx): return ctx
            def to_tir_t(dtype): return DataType(dtype) if isinstance(dtype, str) else dtype
            def cast_to_scalar(v, dst_scalar_type):
                    u = v
                    
                    type_str_arr = str(v.dtype).split('x')
                    cur_scalar_type = DataType(type_str_arr[0])
                    dst_scalar_type = DataType(str(dst_scalar_type))

                    if len(type_str_arr) > 1: # vector found. extract it to a scalar.
                        u = tir.Shuffle([v], [random.randint(0, DataType(str(v.dtype)).lanes - 1)])              

                    if cur_scalar_type != dst_scalar_type:
                        u = tir.Cast(dst_scalar_type, u)

                    return u

            def unary_op_callback(func_str, ctype = 'float', rtype = 'float', stype = 'float'):
                rtype = to_tir_t(rtype) # Must be a scalar type.
                stype = to_tir_t(stype)
                ctype = to_tir_t(ctype) # Maybe a vector type.

                return (lambda x, *args: call_intrin(rtype,
                        func_str, 
                        cast_to_scalar(x, stype),
                        *args).astype(ctype), [id_ctx])

            def binary_op_callback(func_str, ctype = 'float', rtype = 'float', stype = 'float'):
                rtype = to_tir_t(rtype) # Must be a scalar type.
                stype = to_tir_t(stype)
                ctype = to_tir_t(ctype) # Maybe a vector type.
                # FIXME(Jiawei): Casting will fail with lanes.

                return (lambda x, y, *args: call_intrin(
                    rtype,
                    func_str, 
                    cast_to_scalar(x, stype),
                    cast_to_scalar(y, stype),
                    *args).astype(ctype), 
                    [id_ctx, id_ctx])


            dtype = constraint.dtype
            # can be refined according to each intrinsic
            numeric = [
                unary_op_callback('tir.exp', dtype),
                unary_op_callback('tir.exp2', dtype),
                unary_op_callback('tir.exp10', dtype),
                unary_op_callback('tir.erf', dtype),
                unary_op_callback('tir.tanh', dtype),
                unary_op_callback('tir.sigmoid', dtype),
                unary_op_callback('tir.log', dtype),
                unary_op_callback('tir.log2', dtype),
                unary_op_callback('tir.log10', dtype),
                unary_op_callback('tir.log1p', dtype),
                unary_op_callback('tir.tan', dtype),
                unary_op_callback('tir.cos', dtype),
                unary_op_callback('tir.cosh', dtype),
                unary_op_callback('tir.acos', dtype),
                unary_op_callback('tir.acosh', dtype),
                unary_op_callback('tir.sin', dtype),
                unary_op_callback('tir.sinh', dtype),
                unary_op_callback('tir.asin', dtype),
                unary_op_callback('tir.asinh', dtype),
                unary_op_callback('tir.atan', dtype),
                unary_op_callback('tir.atanh', dtype),
                unary_op_callback('tir.sqrt', dtype),
                unary_op_callback('tir.rsqrt', dtype),
                binary_op_callback('tir.ldexp', dtype),
                binary_op_callback('tir.pow', dtype),
                binary_op_callback('tir.atan2', dtype),
            ]

            numeric += [
                unary_op_callback('tir.floor', dtype),
                unary_op_callback('tir.ceil', dtype),
                unary_op_callback('tir.trunc', dtype),
                unary_op_callback('tir.fabs', dtype),
                unary_op_callback('tir.round', dtype),
                unary_op_callback('tir.nearbyint', dtype),
                unary_op_callback('tir.isnan', ctype=dtype, rtype='bool'),
                unary_op_callback('tir.isfinite', ctype=dtype, rtype='bool'),
                unary_op_callback('tir.isinf', ctype=dtype, rtype='bool'),
                unary_op_callback('tir.clz', ctype=dtype, rtype='int', stype='int'),
                (lambda *args: call_intrin(dtype, 'tir.ret', *args).astype(dtype), [id_ctx]),
                unary_op_callback('tir.popcount', ctype=dtype, rtype='int', stype='int'),
                binary_op_callback('tir.nextafter', dtype),
                binary_op_callback('tir.hypot', dtype),
                binary_op_callback('tir.copysign', dtype),
                # (lambda *args: call_intrin(dtype, 'tir.fmod', *args), [id_ctx, id_ctx]), # tvm-llvm do not support tir.fmod.
                (lambda x, y, z, *args: call_intrin(
                        to_tir_t('int'),
                        'tir.q_multiply_shift', 
                        cast_to_scalar(x, 'int'),
                        random.randint(0, 31), # must be constant int.
                        cast_to_scalar(y, 'int'),
                        cast_to_scalar(z, 'int'),
                        *args).astype(dtype), [id_ctx, id_ctx, id_ctx])
            ]
            satisfied_call = []
            if constraint.is_dtype_numeric:
                satisfied_call.extend(numeric)
            if len(satisfied_call) > 0:
                satisfied_1.append(lambda ctx, size: self.generate_sized_call(
                    ctx.with_additional_information(satisfied_call),
                    size,
                ))

            return satisfied_1

    def generate_sized_prim_expr(self, context: Context, size: int) -> TIRNode:
        constraint = context.constraint
        assert isinstance(constraint, PrimExprConstraint)
        satsified = self.get_satisfied_options_for_prim_expr(context, size)
        return random.choice(satsified)(context, size)

    def generate_var(self, context: Context) -> TIRNode:
        constraint = context.constraint
        if isinstance(constraint, PrimExprConstraint):
            if constraint.var_should_be_bound:
                return random.choice(context.additional_information)  # type: ignore # noqa
            else:
                return tir.Var(util.fresh_name(), constraint.dtype)
        else:
            assert isinstance(constraint, VarConstraint)
            if constraint.var_reference is not None:
                return constraint.var_reference
            return tir.Var(util.fresh_name(), constraint.dtype)

    def generate_sized_prim_func(self, size: int) -> TIRNode:
        params_size = int(0.2 * size)
        body_size = int(0.8 * size)
        buffer_map = random.choice(domain.primitive.BUFFER_MAPS)

        params = self.generate_sized_list_of_stuff(
            lambda: Context(VarConstraint(self.random_dtype())),
            params_size,
        )

        return tir.PrimFunc(
            params + [v for v in buffer_map],
            self.generate_sized_stmt(Context(
                StmtConstraint(),
                params + [buffer.data for buffer in buffer_map.values()],
                [buffer for buffer in buffer_map.values()],
            ),
                body_size
            ),
            buffer_map=buffer_map,
        )

    def generate_sized_load(self, context: Context, size: int) -> TIRNode:
        constraint = context.constraint
        assert isinstance(constraint, PrimExprConstraint)
        index_size, predicate_size = util.uniformly_split_num(size - 1, 2)
        # ptr_type = tvm.ir.PointerType(tvm.ir.PrimType(self.random_dtype()))
        buffer_var: tir.Var = random.choice(
            context.additional_information)  # type: ignore
        return tir.Load(
            constraint.dtype, buffer_var,
            self.generate_sized(
                context.with_constraint_being(PrimExprConstraint('uint')),
                index_size),
            None if predicate_size == 0 else self.generate_sized(
                context.with_constraint_being(PrimExprConstraint('bool')),
                predicate_size)
        )

    def generate_sized_bufferload(self, context: Context, size: int) -> TIRNode:
        buffer: tir.Buffer = random.choice(
            context.additional_information)  # type: ignore
        sizes = util.uniformly_split_num(size - 1, len(buffer.shape))
        return tir.BufferLoad(
            buffer,
            [self.generate_sized(context.with_constraint_being(
                PrimExprConstraint(self.random_dtype())),
                size) for size in sizes]
        )

    # def generate_sized_producerload(self, context: Context, size: int) -> TIRNode:
    #     tir.ProducerLoad()
    #     from tvm import te
    #     from tvm.te import Tensor

    def generate_sized_let(self, context: Context, size: int) -> TIRNode:
        value_size, body_size = util.uniformly_split_num(size - 1, 2)
        new_var = self.generate_var(context.with_constraint_being(
            VarConstraint(self.random_dtype())))
        return tir.Let(
            new_var,
            self.generate_sized(context.with_constraint_being(
                PrimExprConstraint(new_var.dtype)),
                value_size),
            self.generate_sized(context.with_bound_vars_being(
                [*context.bound_variables, new_var]),
                body_size),
        )

    def generate_sized_call(self, context: Context, size: int) -> TIRNode:
        call_op, arg_context_getters = random.choice(
            context.additional_information)  # type: ignore
        sizes = util.uniformly_split_num(size - 1, len(arg_context_getters))
        return call_op(*[self.generate_sized(ctx_getter(context), size)
                         for ctx_getter, size in zip(arg_context_getters, sizes)])

    def generate_sized_binop(self, context: Context, size: int, cls: type) -> TIRNode:
        a_size, b_size = util.uniformly_split_num(size - 1, 2)
        a = self.generate_sized(context, a_size)
        b = self.generate_sized(context, b_size)
        if a.dtype != b.dtype:
            print(
                f'{a.dtype, DataType(a.dtype).lanes}: {a}, {b.dtype, DataType(b.dtype).lanes}: {b}')
            breakpoint()
            raise AssertionError
        return cls(a, b)

    def generate_sized_add(self, context: Context, size: int) -> TIRNode:
        return self.generate_sized_binop(context, size, tir.Add)

    def generate_sized_sub(self, context: Context, size: int) -> TIRNode:
        return self.generate_sized_binop(context, size, tir.Sub)

    def generate_sized_mul(self, context: Context, size: int) -> TIRNode:
        return self.generate_sized_binop(context, size, tir.Mul)

    def generate_sized_div(self, context: Context, size: int) -> TIRNode:
        return self.generate_sized_binop(context, size, tir.Div)

    def generate_sized_mod(self, context: Context, size: int) -> TIRNode:
        return self.generate_sized_binop(context, size, tir.Mod)

    def generate_sized_floordiv(self, context: Context, size: int) -> TIRNode:
        return self.generate_sized_binop(context, size, tir.FloorDiv)

    def generate_sized_floormod(self, context: Context, size: int) -> TIRNode:
        return self.generate_sized_binop(context, size, tir.FloorMod)

    def generate_sized_min(self, context: Context, size: int) -> TIRNode:
        return self.generate_sized_binop(context, size, tir.Min)

    def generate_sized_max(self, context: Context, size: int) -> TIRNode:
        return self.generate_sized_binop(context, size, tir.Max)

    def generate_sized_cmp(self, context: Context, size: int, cls) -> TIRNode:
        constraint = context.constraint
        assert isinstance(constraint, PrimExprConstraint)
        assert constraint.is_dtype_bool
        dtype = self.random_dtype_with_lanes(constraint.dtype.lanes)
        new_context = context.with_constraint_being(PrimExprConstraint(dtype))
        return self.generate_sized_binop(new_context, size, cls)

    def generate_sized_eq(self, context: Context, size: int) -> TIRNode:
        return self.generate_sized_cmp(context, size, tir.EQ)

    def generate_sized_ne(self, context: Context, size: int) -> TIRNode:
        return self.generate_sized_cmp(context, size, tir.NE)

    def generate_sized_lt(self, context: Context, size: int) -> TIRNode:
        return self.generate_sized_cmp(context, size, tir.LT)

    def generate_sized_le(self, context: Context, size: int) -> TIRNode:
        return self.generate_sized_cmp(context, size, tir.LE)

    def generate_sized_gt(self, context: Context, size: int) -> TIRNode:
        return self.generate_sized_cmp(context, size, tir.GT)

    def generate_sized_ge(self, context: Context, size: int) -> TIRNode:
        return self.generate_sized_cmp(context, size, tir.GE)

    def generate_sized_and(self, context: Context, size: int) -> TIRNode:
        return self.generate_sized_binop(context, size, tir.And)

    def generate_sized_or(self, context: Context, size: int) -> TIRNode:
        return self.generate_sized_binop(context, size, tir.Or)

    def generate_sized_reduce(self, context: Context, size: int) -> TIRNode:
        """TODO: IDK how to handle this"""
        # CommReducer
        # tir.comm_reducer()
        raise NotImplementedError

    def generate_sized_cast(self, context: Context, size: int) -> TIRNode:
        constraint = context.constraint
        assert isinstance(constraint, PrimExprConstraint)
        dtype_to_cast = self.random_dtype_with_lanes(constraint.dtype.lanes)
        return tir.Cast(
            constraint.dtype, self.generate_sized(
                context.with_constraint_being(
                    PrimExprConstraint(dtype_to_cast)),
                size - 1,
            )
        )

    def generate_sized_not(self, context: Context, size: int) -> TIRNode:
        return tir.Not(self.generate_sized(context, size - 1))

    def generate_sized_select(self, context: Context, size: int) -> TIRNode:
        sizes = util.uniformly_split_num(size - 1, 3)
        condition_size, true_size, false_size = sizes
        return tir.Select(
            self.generate_sized(
                context.with_constraint_being(PrimExprConstraint('bool')),
                condition_size,
            ),
            self.generate_sized(context, true_size),
            self.generate_sized(context, false_size),
        )

    def generate_sized_ramp(self, context: Context, size: int) -> TIRNode:
        constraint = context.constraint
        assert isinstance(constraint, PrimExprConstraint)
        assert constraint.dtype.lanes > 1
        base_size, stride_size = util.uniformly_split_num(size - 1, 2)
        new_context = context.with_constraint_being(PrimExprConstraint(
            DataType.CODE2STR[constraint.dtype.type_code]
        ))
        return tir.Ramp(
            self.generate_sized(
                new_context,
                base_size,
            ),
            self.generate_sized(
                new_context,
                stride_size
            ),
            constraint.dtype.lanes,
        )

    def generate_sized_broadcast(self, context: Context, size: int) -> TIRNode:
        constraint = context.constraint
        assert isinstance(constraint, PrimExprConstraint)
        return tir.Broadcast(
            self.generate_sized(
                Context(
                    PrimExprConstraint(str(constraint.dtype).split('x')[0])),
                size - 1,
            ),
            constraint.dtype.lanes,
        )

    def generate_sized_shuffle(self, context: Context, size: int) -> TIRNode:
        constraint = context.constraint
        assert isinstance(constraint, PrimExprConstraint)
        array_size = size - 1 - constraint.dtype.lanes
        indices_size = constraint.dtype.lanes
        assert array_size + indices_size == size - 1
        sizes = util.uniformly_split_num(size, array_size + indices_size)

        array_sizes = sizes[:array_size]
        indices_sizes = sizes[array_size:]
        assert len(array_sizes) == array_size and len(
            indices_sizes) == indices_size

        # array_constraint = PrimExprConstraint(constraint.dtype)
        array = [
            self.generate_sized(context.with_constraint_being(PrimExprConstraint(
                self.random_multi_dim_dtype(base_type=str(constraint.dtype)))),
                size)
            for size in array_sizes
        ]
        indices = [
            self.generate_sized(context.with_constraint_being(PrimExprConstraint(
                self.random_multi_dim_dtype())),
                size)
            for size in indices_sizes
        ]
        assert len(indices) == indices_size
        assert len(array) == array_size
        return tir.Shuffle(
            array,
            indices,
        )

    def generate_intimm(self, context: Context) -> TIRNode:
        constraint = context.constraint
        assert isinstance(constraint, PrimExprConstraint)
        return tir.IntImm(
            constraint.dtype,
            (util.random_uint() if constraint.is_dtype_uint
             else random.choice([True, False]) if constraint.is_dtype_bool
             else util.random_integer())
        )

    def generate_floatimm(self, context: Context) -> TIRNode:
        constraint = context.constraint
        assert isinstance(constraint, PrimExprConstraint)
        return tir.FloatImm(constraint.dtype, util.random_float())

    def generate_stringimm(self) -> TIRNode:
        return tir.StringImm(util.random_string())

    def generate_sized_stmt(self, context: Context, size: int) -> TIRNode:
        constraint = context.constraint
        assert isinstance(constraint, StmtConstraint)
        if size == 0:
            options_0 = [lambda context: tir.Evaluate(tir.const(0))]
            if len(context.buffers) > 0:
                options_0.append(lambda context: self.generate_prefetch(
                    context.with_additional_information(context.buffers),
                ))
            return random.choice(options_0)(context)
        options = [
            self.generate_sized_attrstmt,
            self.generate_sized_ifthenelse,
            self.generate_sized_letstmt,
            self.generate_sized_for,
            self.generate_sized_while,
            self.generate_sized_allocate,
            # self.generate_sized_bufferrealize,
            self.generate_sized_assertstmt,
            # self.generate_sized_producerstore,
            # self.generate_sized_producerrealize,
            self.generate_sized_seqstmt,
            self.generate_sized_evaluate,
            # self.generate_sized_block,
            # self.generate_sized_blockrealize,
        ]
        possible_vars = [
            v for v in context.bound_variables
            if v.dtype == 'handle'
            and 'type_annotation' in dir(v)
            and isinstance(v.type_annotation, tvm.ir.PointerType)
        ]
        if len(possible_vars) > 0:
            options.append(lambda ctx, size: self.generate_sized_store(
                ctx.with_additional_information(possible_vars),
                size,
            ))

        possible_buffers = context.buffers
        if len(possible_buffers) > 0:
            options.append(lambda ctx, size: self.generate_sized_bufferstore(
                ctx.with_additional_information(possible_buffers),
                size,
            ))
        return random.choice(options)(context, size)

    # src_doms = [ir.Range, tir.Var, IterType, ThreadTag]
    # target_dom = tir.IterVar

    # def __call__(self, dom: ir.Range, var: tir.Var, iter_type: IterType, thread_tag: ThreadTag) -> tir.PrimExpr:  # type: ignore
    #     return tir.IterVar(dom, var, iter_type, thread_tag)
    def generate_itervar(self, context: Context) -> tir.IterVar:
        dom = random.choice(domain.primitive.RANGES)
        var = self.generate_var(context.with_constraint_being(
            VarConstraint(self.random_dtype())))
        iter_type = random.choice(domain.primitive.ITERTYPES)
        thread_tag = random.choice(domain.primitive.THREADTAGS)
        return tir.IterVar(dom, var, iter_type, thread_tag)

    def generate_sized_attrstmt(self, context: Context, size: int) -> TIRNode:
        node = self.generate_itervar(context)
        attr_key = random.choice(domain.primitive.ATTRKEYS)
        value_size, body_size = util.uniformly_split_num(size - 1, 2)
        return tir.AttrStmt(node, attr_key, self.generate_sized(
            context.with_constraint_being(PrimExprConstraint(
                self.random_dtype(),
                var_should_be_bound=False)),
            value_size,
        ), self.generate_sized(
            context.with_constraint_being(
                StmtConstraint(var_should_be_bound=False)),
            body_size,
        ))

    def generate_sized_ifthenelse(self, context: Context, size: int) -> TIRNode:
        sizes = util.uniformly_split_num(size - 1, 3)
        condition_size, then_size, else_size = sizes
        return tir.IfThenElse(
            self.generate_sized(
                context.with_constraint_being(PrimExprConstraint('bool')),
                condition_size,
            ), self.generate_sized(
                context.with_constraint_being(StmtConstraint()),
                then_size,
            ), self.generate_sized(
                context.with_constraint_being(StmtConstraint()),
                else_size,
            )
        )

    def generate_sized_letstmt(self, context: Context, size: int) -> TIRNode:
        value_size, body_size = util.uniformly_split_num(size - 1, 2)
        var = self.generate_var(
            context.with_constraint_being(
                VarConstraint(self.random_dtype())))
        return tir.LetStmt(
            var,
            self.generate_sized(
                context.with_constraint_being(PrimExprConstraint(var.dtype)),
                value_size,
            ), self.generate_sized(
                context
                .with_constraint_being(StmtConstraint())
                .with_bound_vars_being([*context.bound_variables, var]),
                body_size,
            )
        )

    def generate_sized_for(self, context: Context, size: int) -> TIRNode:
        loop_var = self.generate_var(context.with_constraint_being(
            VarConstraint(random.choice(['int', 'uint']))))
        min_value_size, extent_size, body_size = util.uniformly_split_num(
            size - 1,
            3,
        )
        expr_context = context.with_constraint_being(
            PrimExprConstraint(loop_var.dtype))
        return tir.For(
            loop_var,
            self.generate_sized(expr_context, min_value_size),
            self.generate_sized(expr_context, extent_size),
            random.choice(domain.primitive.FOR_KINDS),
            self.generate_sized(
                context
                .with_constraint_being(StmtConstraint())
                .with_bound_vars_being([*context.bound_variables, loop_var]),
                body_size
            ),
        )

    def generate_sized_while(self, context: Context, size: int) -> TIRNode:
        condition_size, body_size = util.uniformly_split_num(size - 1, 2)
        return tir.stmt.While(
            tir.Cast('bool', self.generate_sized(
                context.with_constraint_being(PrimExprConstraint('bool')),
                condition_size,
            )),
            self.generate_sized(context, body_size)
        )

    def generate_sized_allocate(self, context: Context, size: int) -> TIRNode:
        constraint = context.constraint
        assert isinstance(constraint, StmtConstraint)
        extents_size, condition_size, body_size = util.uniformly_split_num(
            size, 3)
        dtype = self.random_dtype()
        return tir.Allocate(
            self.generate_var(context.with_constraint_being(VarConstraint(
                tvm.ir.PointerType(tvm.ir.PrimType(dtype))
            ))),
            dtype,
            self.generate_sized_list_of_stuff(lambda: context.with_constraint_being(
                PrimExprConstraint(dtype),
            ), extents_size),
            self.generate_sized(
                context.with_constraint_being(PrimExprConstraint('bool')),
                condition_size,
            ),
            self.generate_sized(
                context.with_constraint_being(StmtConstraint()),
                body_size,
            )
        )

    def generate_sized_store(self, context: Context, size: int) -> TIRNode:
        constraint = context.constraint
        assert isinstance(constraint, StmtConstraint)
        value_size, index_size, predicate_size = util.uniformly_split_num(
            size - 1, 3)
        buffer_var: tir.Var = random.choice(
            context.additional_information)  # type: ignore
        return tir.Store(
            random.choice(context.additional_information),  # type: ignore
            self.generate_sized(
                context.with_constraint_being(PrimExprConstraint(
                    buffer_var.type_annotation.element_type.dtype)),
                value_size),
            self.generate_sized(
                context.with_constraint_being(PrimExprConstraint('uint')),
                index_size),
            None
        )

    def generate_sized_bufferstore(self, context: Context, size: int) -> TIRNode:
        constraint = context.constraint
        assert isinstance(constraint, StmtConstraint)

        buffer: tir.Buffer = random.choice(
            context.additional_information)  # type: ignore
        sizes = util.uniformly_split_num(size - 1, 1 + len(buffer.shape))
        value_size = sizes[0]
        indices_sizes = sizes[1:]

        return tir.BufferStore(
            buffer,
            self.generate_sized(
                context.with_constraint_being(PrimExprConstraint(
                    buffer.data.type_annotation.element_type.dtype)),
                value_size),
            [self.generate_sized(context.with_constraint_being(
                PrimExprConstraint(self.random_dtype())),
                size) for size in indices_sizes]
        )

    def generate_sized_bufferrealize(self, context: Context, size: int) -> TIRNode:
        """TODO"""
        raise NotImplementedError

    def generate_sized_assertstmt(self, context: Context, size: int) -> TIRNode:
        condition_size, body_size = util.uniformly_split_num(size - 1, 2)
        return tir.AssertStmt(
            self.generate_sized(context.with_constraint_being(
                PrimExprConstraint('bool')), condition_size),
            random.choice([
                self.generate_stringimm(),
                self.generate_intimm(context.with_constraint_being(
                    PrimExprConstraint('int32')
                )),
            ]),
            self.generate_sized(context, body_size)
        )

    # def generate_sized_producerstore(self, context: Context, size: int) -> TIRNode:
    #     raise NotImplementedError

    # def generate_sized_producerrealize(self, context: Context, size: int) -> TIRNode:
        # raise NotImplementedError

    def generate_prefetch(self, context: Context) -> TIRNode:
        return tir.Prefetch(
            random.choice(context.additional_information),  # type: ignore
            domain.primitive.RANGES,
        )

    def generate_sized_seqstmt(self, context: Context, size: int) -> TIRNode:
        return tir.SeqStmt(self.generate_sized_list_of_stuff(lambda: context, size - 1))

    def generate_sized_evaluate(self, context: Context, size: int) -> TIRNode:
        return tir.Evaluate(
            self.generate_sized(
                context.with_constraint_being(
                    PrimExprConstraint(self.random_dtype())),
                size - 1,
            )
        )

    # def generate_sized_block(self, context: Context, size: int) -> TIRNode:
    #     raise NotImplementedError

    # def generate_sized_blockrealize(self, context: Context, size: int) -> TIRNode:
    #     raise NotImplementedError

    def generate_sized_list_of_stuff(
        self,
        context_getter: Callable[[], Context],
        size: int
    ) -> List[TIRNode]:
        if size == 0:
            return []
        base_size, inductive_size = util.uniformly_split_num(size - 1, 2)
        return [
            self.generate_sized(context_getter(), base_size),
            *self.generate_sized_list_of_stuff(context_getter, inductive_size),
        ]

    def random_dtype(self) -> str:
        return random.choice([
            # 'int8',
            # 'int16',
            'int32',
            # 'int64',
            # 'uint8',
            # 'uint16',
            'uint32',
            # 'uint64',
            'uint1',
            'float32',
            # 'float64'
        ])

    def random_dtype_with_lanes(self, lanes: int) -> str:
        return f'{self.random_dtype()}x{lanes}'

    def random_lanes(self) -> int:
        return random.randint(1, 4)

    def random_multi_dim_dtype(self, base_type: Optional[str] = None) -> str:
        if base_type is None:
            base_type = self.random_dtype()
        lanes = self.random_lanes()
        return f'{base_type}x{lanes}'
    # def generate_sized_any(self, context: Context, size: int) -> TIRNode:
    #     NotImplementedError
