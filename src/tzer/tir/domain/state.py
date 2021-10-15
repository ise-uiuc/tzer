from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from tvm import tir
from .kernel import KERNEL
from .laze import laze
from .node import LazyDomNode
from . import construct as cons


class GenerationError(Exception):
    pass


@dataclass
class DomState:
    values: List[cons.DomValue]
    constructors: Set[cons.Cons]


class State:
    def __init__(
        self,
        seeds: List[tir.PrimFunc] = [],
        max_node_size: Optional[int] = None
    ) -> None:
        self.data: Dict[cons.Dom, DomState] = dict()
        self.no_empty_dom: Set[cons.Dom] = set()
        self.valid_constructors: Set[cons.Cons] = set()

        for dom, data in KERNEL.data.items():
            self.data[dom] = DomState(
                values=[prim for prim in data.primitives],
                constructors={cons for cons in data.constructors}
            )
            if len(data.primitives) > 0:
                self.no_empty_dom.add(dom)

        # Additional conss and values by seeds
        for seed in seeds:
            seed_node = laze(seed)
            for node in seed_node.all_nodes:
                if not node.dom in self.data:
                    self.data[node.dom] = DomState([], set())
                if max_node_size is None or len(node.all_nodes) < max_node_size:
                    self.data[node.dom].values.append(node.get())
                self.no_empty_dom.add(node.dom)
                if not node.is_leaf:
                    assert node.cons is not None
                    for dom in node.cons.src_doms:
                        if dom not in self.data:
                            self.data[dom] = DomState([], set())
                    if node.cons not in self.data[node.dom].constructors:
                        self.data[node.dom].constructors.add(node.cons)

        self.update_valid_constructors()

    def some_value(self, dom: cons.Dom, expansion_num: int) -> List[LazyDomNode]:
        new_nodes: List[LazyDomNode] = []

        root = LazyDomNode.make_leaf(dom, None)
        to_be_expanded: List[LazyDomNode] = [root]
        cannot_be_expanded: List[LazyDomNode] = []

        n = 0
        while n < expansion_num and len(to_be_expanded) > 0:
            node = to_be_expanded.pop(
                random.randint(0, len(to_be_expanded) - 1))
            assert node.cons is None
            conss = self[node.dom].constructors
            if len(conss) == 0:
                cannot_be_expanded.append(node)
            else:
                cons = random.choice(list(conss))
                # assert cons.target_dom == node.dom
                node.cons = cons
                node.children = [LazyDomNode.make_leaf(
                    dom, None) for dom in cons.src_doms]
                new_nodes.append(node)
                to_be_expanded.extend(node.children)
                n += 1

        expanded = to_be_expanded + cannot_be_expanded
        while len(expanded) > 0:
            node = expanded.pop()
            if node.dom not in self.no_empty_dom:
                conss = self[node.dom].constructors
                no_empty_constructors = conss & self.valid_constructors
                if len(no_empty_constructors) > 0:
                    cons = random.choice(list(no_empty_constructors))
                else:
                    cons = random.choice(list(conss))
                node.cons = cons
                node.children = [LazyDomNode.make_leaf(
                    dom, None) for dom in cons.src_doms]
                expanded.extend(node.children)
            else:
                node.value = random.choice(self[node.dom].values)

        return new_nodes

    def update_valid_constructors(self):
        all_consturcotrs = set()
        for v in self.data.values():
            all_consturcotrs.update(v.constructors)

        for cons in all_consturcotrs - self.valid_constructors:
            if len(set(cons.src_doms) - self.no_empty_dom) == 0:
                self.valid_constructors.add(cons)

    def __getitem__(self, key: cons.Dom) -> DomState:
        return self.data[key]

    def __str__(self) -> str:
        return str(self.data)
