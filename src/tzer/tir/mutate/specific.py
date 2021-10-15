import random
from tvm import tir
from tzer.tir.semantic.constraint import PrimExprConstraint, StmtConstraint, VarConstraint
from .generate import SizedGenerator
from .mutator import Mutator
from tzer.tir.domain import primitive
from tzer.tir.util import TIRNode
from tzer.tir.semantic import Context


class SpecificMutator(Mutator):
    def __init__(self) -> None:
        super().__init__()
        self.expr_pools = []
        self.stmt_pools = []
        self.generator = SizedGenerator(lambda: random.randint(1, 50))

    # PrimExpr Mutator
    def select(self, op: TIRNode, context: Context) -> TIRNode:
        condition = self.generator.generate(Context(PrimExprConstraint('bool')))
        false_value = self.generator.generate(Context(PrimExprConstraint(op.dtype)))
        new = tir.Select(condition=condition, true_value=op, false_value=false_value)
        return new

    def buffer_expr_mutate(self, op: TIRNode, context: Context) -> TIRNode:
        possible_buffers = [buffer for buffer in context.buffers if buffer.dtype == op.dtype]
        if len(possible_buffers) > 0:
            buffer = random.choice(possible_buffers)
            return random.choices([
                lambda: tir.BufferLoad(buffer, [op]),
                lambda: tir.Load(op.dtype, buffer.data, [op], self.generator.generate(Context(PrimExprConstraint('bool'))))
            ])()
        else:
            return op

    # Stmt Mutator
    def allocate_mutate(self, op: TIRNode, context: Context) -> TIRNode:
        possible_buffers = [buffer for buffer in context.buffers]
        if len(possible_buffers) > 0:
            buffer = random.choice(possible_buffers)
            var = buffer.data
            condition = self.generator.generate_sized(Context(PrimExprConstraint('bool')), size=random.randint(1, 10))
            dtype = random.choice([
                "int8", "int16", "int32", "int64",
                "uint8", "uint16", "uint32", "uint64",
                "float32", "float64",
                "bool", "handle"
            ])
            extents = self.generator.generate_sized(Context(PrimExprConstraint(dtype)), size=random.randint(1, 10))
            new = tir.Allocate(buffer_var=var, dtype=var.dtype, extents=extents, condition=condition, body=op)
            return new
        else:
            return op

    def assert_mutate(self, op: TIRNode, context: Context) -> TIRNode:
        condition = self.generator.generate_sized(Context(PrimExprConstraint('bool')), size=random.randint(1, 10))
        message = random.choice(primitive.STRINGIMMS)
        new = tir.stmt.AssertStmt(condition=condition, message=message, body=op)
        return new

    def attr_mutate(self, op: TIRNode, context: Context) -> TIRNode:
        node = self.generator.generate_itervar(Context(PrimExprConstraint('int')))
        key = random.choice(primitive.ATTRKEYS)
        dtype = random.choice([
            "int8", "int16", "int32", "int64",
            "uint8", "uint16", "uint32", "uint64",
            "float32", "float64",
            "bool", "handle"
        ])
        value = self.generator.generate_sized(Context(PrimExprConstraint(dtype)), size=random.randint(1, 10))
        new = tir.AttrStmt(node=node, attr_key=key, value=value, body=op)
        return new

    def if_then_mutate(self, op: TIRNode, context: Context) -> TIRNode:
        condition = self.generator.generate_sized(Context(PrimExprConstraint('bool')), size=random.randint(1, 10))
        else_case = None
        if random.choice([True, False]):
            else_case = self.generator.generate_sized(Context(StmtConstraint()), size=random.randint(1, 10))
        new = tir.stmt.IfThenElse(condition=condition, then_case=op, else_case=else_case)
        return new

    def loop_mutate(self, op: TIRNode, context: Context) -> TIRNode:
        def attr_body(body):
            node = self.generator.generate_itervar(Context(PrimExprConstraint('int')))
            key = random.choice(['pragma_unroll_explicit', 'pragma_auto_unroll_max_step'])
            value = random.randint(-128, 127)
            value = tir.IntImm('int32', value)
            return tir.AttrStmt(node=node, attr_key=key, value=value, body=body)
        
        def ret_body(body):
            return body

        possible_buffers = [buffer for buffer in context.buffers]
        is_constant = random.choice([True, False])
        is_seq = random.choice([len(possible_buffers) > 0, False])
        max_loop_depth = 3
        for_kinds = list(primitive.FOR_KINDS)
        last_loop_var = None

        if is_seq:
            buffer = random.choice(possible_buffers)
            index = self.generator.generate_sized(
                context.with_constraint_being(PrimExprConstraint('uint')),
                size=random.randint(1, 10))
            expr = self.generator.generate_sized(
                Context(PrimExprConstraint(buffer.data.type_annotation.element_type.dtype)),
                size=random.randint(1, 10))
            predicate =  self.generator.generate(Context(PrimExprConstraint('bool')))
            new = random.choice([
                lambda: tir.Store(buffer_var=buffer.data, value=expr, index=index, predicate=predicate),
                lambda: tir.BufferStore(buffer_var=buffer, value=expr, index=[index]),
            ])()
        else:
            new = op


        for _ in range(random.randint(1, max_loop_depth)):
            loop_var = self.generator.generate_var(context.with_constraint_being(
                VarConstraint(random.choice(['int', 'uint']))))
            
            if is_constant or last_loop_var is None:
                min_val = random.randint(-127, 128)
                extent = random.randint(1, 128)
            else:
                expr_context = context.with_constraint_being(
                    PrimExprConstraint(loop_var.dtype))
                min_val = random.choice([lambda: last_loop_var, lambda: random.randint(-127, 128), self.generator.generate_sized(expr_context, random.randint(1, 10))])()
                extent = random.choice([lambda: last_loop_var, lambda: random.randint(1, 1000), self.generator.generate_sized(expr_context, random.randint(1, 10))])()

            thread_binding = None

            for_kind = random.choice(for_kinds)
            if for_kind == tir.ForKind.PARALLEL:
                for_kinds.remove(for_kind)
            elif for_kind == tir.ForKind.THREAD_BINDING:
                thread_binding = self.generator.generate_itervar(Context(PrimExprConstraint('int')))
            
            if is_seq:
                if isinstance(new, tir.Store):
                    new = tir.Store(buffer_var=new.buffer_var, value=new.value, index=index+loop_var, predicate=new.predicate)
                else:
                    new = tir.BufferStore(buffer=new.buffer, value=new.value, indices=[new.indices[0]+loop_var])
            else:
                new = random.choice([attr_body, ret_body])(new)

            new = tir.For(
                loop_var,
                min_val,
                extent,
                for_kind,
                new,
                thread_binding
            )

            last_loop_var = loop_var

        if is_seq:
            return tir.SeqStmt([new, op])
        else:
            return new

    def mutate(self, op: TIRNode, context: Context) -> TIRNode:
        if isinstance(op, tir.Stmt):
            mutator = random.choice([self.allocate_mutate, self.assert_mutate, self.attr_mutate, self.if_then_mutate, self.loop_mutate])
            return mutator(op, context)
        elif isinstance(op, tir.PrimExpr):
            mutator = random.choice([self.select, self.buffer_expr_mutate])
            return mutator(op, context)
        else:
            return op

    def will_modify(self, op: TIRNode, context: Context) -> bool:
        return isinstance(op, (
            tir.Stmt,
            tir.PrimExpr
        ))