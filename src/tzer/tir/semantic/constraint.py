from dataclasses import dataclass
from typing import List, Optional, Type, Union
from tvm import tir
import tvm
from tzer.tir.util import TIRNode
from tvm.runtime import DataType, DataTypeCode


class PrimExprConstraint:
    def __init__(self, dtype: Union[str, DataType], var_should_be_bound: bool = True) -> None:
        if isinstance(dtype, str) and dtype.startswith('bool'):
            dtype = 'uint1' + dtype[4:]
        self.dtype = DataType(dtype) if isinstance(dtype, str) else dtype
        self.var_should_be_bound = var_should_be_bound

    @property
    def is_dtype_uint(self) -> bool:
        return self.dtype.type_code == DataTypeCode.UINT

    @property
    def is_dtype_int(self) -> bool:
        return self.dtype.type_code == DataTypeCode.INT

    @property
    def is_dtype_numeric(self) -> bool:
        return self.is_dtype_int or self.is_dtype_float or self.is_dtype_bool or self.is_dtype_uint

    @property
    def is_dtype_float(self) -> bool:
        return self.dtype.type_code == DataTypeCode.FLOAT

    @property
    def is_dtype_bool(self) -> bool:
        return self.dtype.type_code == DataTypeCode.UINT and self.dtype.bits == 1


class StmtConstraint:
    def __init__(self, var_should_be_bound: bool = True) -> None:
        self.var_should_be_bound = var_should_be_bound


@dataclass
class PrimFuncConstraint:
    ...


class VarConstraint:
    def __init__(
        self,
        dtype: Union[str, tvm.ir.PointerType],
        var_reference: Optional[tir.Var] = None
    ) -> None:
        self.dtype = dtype
        self.var_reference = var_reference


@dataclass
class BlockConstraint:
    ...


# Specifies the constraints of a hole in an expression (e.g. [ ] * 1.1).
Constraint = Union[
    PrimExprConstraint,
    StmtConstraint,
    VarConstraint,
    BlockConstraint,
    PrimFuncConstraint,
]


def satisfies(node: TIRNode, constraint: Constraint) -> bool:
    """Returns whether `node` satisfies the constraint specified"""
    if isinstance(constraint, PrimExprConstraint):
        return isinstance(node, tir.PrimExpr) and DataType(node.dtype) == constraint.dtype
    elif isinstance(constraint, StmtConstraint):
        return isinstance(node, tir.Stmt)
    elif isinstance(constraint, VarConstraint):
        return isinstance(node, tir.Var)
    elif isinstance(constraint, BlockConstraint):
        return isinstance(node, tir.Block)
    elif isinstance(constraint, PrimFuncConstraint):
        return isinstance(node, tir.PrimFunc)
    else:
        raise NotImplementedError(constraint)


def hole_type(self, constraint: Constraint) -> type:
    """Returns the TIR node type of the hole in this constraint"""
    if isinstance(constraint, PrimExprConstraint):
        return tir.PrimExpr
    elif isinstance(constraint, StmtConstraint):
        return tir.Stmt
    elif isinstance(constraint, VarConstraint):
        return tir.Var
    elif isinstance(constraint, BlockConstraint):
        return tir.Block
    elif isinstance(constraint, PrimFuncConstraint):
        return tir.PrimFunc
    else:
        raise NotImplementedError
