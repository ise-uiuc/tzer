from random import randint
import tvm
from tzer import tir
from . import construct as cons
from typing import Any, Optional, List, Union
from tzer.tir import util, visit
from dataclasses import dataclass
from ..visit import get_free_vars, rebind_buffer_var

Component = Union[tvm.ir.Node, Any]


@dataclass
class LazyDomNode:
    dom: cons.Dom
    # None for leaf node
    cons: Optional[cons.Cons]
    # None for unevaluated non-leaf node
    value: Optional[cons.DomValue]
    children: List['LazyDomNode']

    @staticmethod
    def make_leaf(dom: cons.Dom, value: cons.DomValue) -> 'LazyDomNode':
        return LazyDomNode(dom, None, value, [])

    @staticmethod
    def make_nonleaf(dom: cons.Dom, cons: cons.Cons, children: List['LazyDomNode']) -> 'LazyDomNode':
        return LazyDomNode(dom, cons, None, children)

    @property
    def all_nodes(self) -> List['LazyDomNode']:
        return [self] if self.is_leaf else util.list_union([
            *[child.all_nodes for child in self.children], [self]
        ])

    @property
    def all_child_nodes(self) -> List['LazyDomNode']:
        return self.all_nodes[:-1]

    @property
    def is_leaf(self) -> bool:
        return self.cons is None

    def get(self) -> cons.DomValue:
        if self.value is not None:
            return self.value
        if self.value is None and self.cons is None:
            # If both are None, we consider None to be the value of the node
            return None
        assert not self.is_leaf
        params = [child.get() for child in self.children]
        assert self.cons is not None
        self.value = self.cons(*params)
        return self.value

    def _semval_param_append(self):
        fvars = get_free_vars(self.value.body)
        if fvars:
            if self.value.buffer_map:
                for bf_var, bf in dict(self.value.buffer_map).items():
                    if bf.data in fvars:
                        fvars.remove(bf.data)
                        fvars.add(bf_var)
            params = list(fvars | {k for k in self.value.buffer_map})
            self.value = tvm.tir.PrimFunc(
                params,
                self.value.body,
                self.value.ret_type,
                self.value.buffer_map,
                self.value.attrs,
                self.value.span
            )

    def _semval_rebind_buffer(self):
        assert isinstance(self.value, tvm.tir.PrimFunc)
        try:
            self.value = rebind_buffer_var(self.value)
        except tvm.TVMError:
            pass

    def _semval_constize(self):
        fvars = get_free_vars(self.value.body)

        # Find out lvalue of free variables
        lvars = []
        def find_lvalue(stmt):
            nonlocal lvars
            if isinstance(stmt, tvm.tir.Store):
                lvars.append(stmt.buffer_var)
            elif isinstance(stmt, tvm.tir.LetStmt):
                lvars.append(stmt.var)
            elif isinstance(stmt, tvm.tir.For):
                lvars.append(stmt.loop_var)

        tvm.tir.stmt_functor.post_order_visit(self.value.body, find_lvalue)

        # FIXME(Jiawei): Constize variable often leads to grammar error. 
        for v in fvars:
            if v in lvars:
                continue
            cval = None
            if v.dtype.startswith('int'):
                cval = randint(1, 500)
            elif v.dtype.startswith('float'):
                cval = float(randint(1, 500))
            else:
                continue
                # raise NotImplementedError(f"Tzer does not model var type in `{v.dtype}`")
            self.value = visit.swap_tir(self.value, v, tvm.tir.const(value=cval, dtype=v.dtype))


    def semantic_validation(self) -> tvm.tir.PrimFunc:
        """
        Make sure the semantic of TIR correct to reduce unnecessary compilation error at
        parsing level. Related work like Squirrel also show this necessity since we want
        to test the internel of TVM not just parsing & semantic analysis.
        """
        assert self.value is not None, 'Come on... Please do `.get()` first...'
        try:
            self._semval_constize()
        except tvm.TVMError:
            pass
        self._semval_param_append()
        self._semval_rebind_buffer()
        return self.value
        

    def replace(self, old: 'LazyDomNode', new: 'LazyDomNode') -> 'LazyDomNode':
        if self is old:
            return new
        return LazyDomNode(
            self.dom,
            self.cons,
            self.value,
            [child.replace(old, new) for child in self.children]
        )

    def check(self):
        if self.is_leaf:
            assert len(self.children) == 0
            assert self.cons is None
            # assert self.value is not None
        else:
            assert self.cons is not None
            if not self.dom == self.cons.target_dom:
                print(self.dom, self.cons)
                assert False
            assert len(self.cons.src_doms) == len(self.children)
        for child in self.children:
            if not isinstance(child, LazyDomNode):
                print(self.dom, self.cons)
                assert False
            child.check()

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, o: object) -> bool:
        return self is o
