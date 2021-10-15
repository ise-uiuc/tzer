from typing import Dict, List
from . import construct as cons, primitive as prim
from dataclasses import dataclass


@dataclass
class KernelDomData:
    primitives: List[cons.DomValue]
    constructors: List[cons.Cons]


@dataclass
class Kernel:
    """Set of primitives and constructors that can be used to form
    any desired TIR (and any value of its related domains)"""
    data: Dict[cons.Dom, KernelDomData]

    def all_doms(self) -> List[cons.Dom]:
        return cons.ALL_DOMS

    def __getitem__(self, key: cons.Dom) -> KernelDomData:
        return self.data[key]


def _make_kernel() -> Kernel:
    res: Dict[cons.Dom, KernelDomData] = {
        dom: KernelDomData([], []) for dom in cons.ALL_DOMS}
    for c in cons.ALL_CONSS:
        res[c.target_dom].constructors.append(c)
    for dom, values in prim.ALL_PRIMITIVES.items():
        res[dom].primitives.extend(values)
    return Kernel(res)


KERNEL = _make_kernel()
