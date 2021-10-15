from typing import List

import tvm
# from tvm.script import ty
import tvm.relay as relay
from tvm.relay.backend import graph_executor_codegen
from tvm import tir

from tzer import relay_seeds


def relay_to_tir(relay_seeds) -> List[tir.PrimFunc]:
    tirs: List[tir.PrimFunc] = []
    for module, params in relay_seeds:
        # compile the model
        target = tvm.target.Target('llvm')
        dev = tvm.cpu()
        with tvm.transform.PassContext(opt_level=0):
            opt_mod, _ = relay.optimize(module, target, params)  # compile
            graph_json, lowered_func, params = graph_executor_codegen.GraphExecutorCodegen(
                None, target=target).codegen(opt_mod['main'])  # relay -> tir
            mod = lowered_func[target]
            tirs.extend([mod[v] for v in mod.get_global_vars()])
    return tirs


def get_all_seeds() -> List[tir.PrimFunc]:
    return relay_to_tir(relay_seeds.MODEL_SEEDS)

