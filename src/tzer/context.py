from typing import Dict, List
from dataclasses import dataclass, field
import tvm
from tvm import relay
import pickle
import random
import numpy as np
import random

from copy import deepcopy

from .tvmpass import PassDependenceGraph, PassNode

# TODO: Add parameters.
# TODO: Add more passes.
_RELAY_FUNCTION_HARD_PASSES_ = [ # Note these are types.
    relay.transform.RemoveUnusedFunctions,
    relay.transform.Inline,
    relay.transform.PartitionGraph,
    relay.transform.ToGraphNormalForm,
    relay.transform.SimplifyInference,
    relay.transform.FoldConstant,
    relay.transform.AnnotateSpans,
    relay.transform.DefuseOps,
    relay.transform.FuseOps,
    relay.transform.SimplifyExpr,
    # relay.transform.ToBasicBlockNormalForm,
    relay.transform.BatchingOps,
    relay.transform.AlterOpLayout,
    relay.transform.FoldScaleAxis,
    relay.transform.CanonicalizeOps,
    relay.transform.CanonicalizeCast,
    relay.transform.DeadCodeElimination,
    relay.transform.EliminateCommonSubexpr,
    relay.transform.CombineParallelConv2D,
    relay.transform.CombineParallelDense,
    relay.transform.CombineParallelBatchMatmul,
    relay.transform.FastMath,
    relay.transform.DynamicToStatic,
    relay.transform.FoldExplicitPadding,
]


_RANDOM_WALK_MAP_ = np.ones((len(_RELAY_FUNCTION_HARD_PASSES_), len(_RELAY_FUNCTION_HARD_PASSES_)))
_RANDOM_WALK_MAP_[_RELAY_FUNCTION_HARD_PASSES_.index(relay.transform.AnnotateSpans)][_RELAY_FUNCTION_HARD_PASSES_.index(relay.transform.FuseOps)] = 0

graph = PassDependenceGraph(tvm.target.Target('llvm'))
_ALL_DIR_PASS_NODES_ = list(graph.tir_pass_nodes.values())

@dataclass
class CompileConfig:
    target       :tvm.target.Target    = None
    relay_pass_types :List[relay.transform.FunctionPass] = None # actually, there're some module passes...
    tir_pass_nodes :List[PassNode] = None
    def mutate(self):
        # TODO: Think about better mutation strategies.
        # Target
        self.target = random.choice(self._target_space())
        # Passes
        n_pass = random.randint(1, len(_RELAY_FUNCTION_HARD_PASSES_) - 1)
        self.relay_pass_types = []
        pidx = random.randint(1, len(_RELAY_FUNCTION_HARD_PASSES_) - 1)
        for _ in range(n_pass):
            self.relay_pass_types.append(_RELAY_FUNCTION_HARD_PASSES_[pidx])
            candidates_idx = _RANDOM_WALK_MAP_[pidx].nonzero()[0]
            if len(candidates_idx) == 0:
                break
            pidx = candidates_idx[random.randint(1, len(candidates_idx) - 1)]

        self.tir_pass_nodes = graph.random_tir_passes(n_pass)



    def hard_relay_passes() -> List[relay.transform.FunctionPass]:
        """passes that do not leverage (great) approximation.
        """
        return _RELAY_FUNCTION_HARD_PASSES_

    def get_device(self):
        if self.target.export()['kind'] == 'cuda':
            return tvm.cuda()
        if self.target.export()['kind'] == 'rocm':
            return tvm.rocm()
        return tvm.cpu()
    
    def check(self):
        assert self.target != None
        assert self.relay_pass_types != None

    @staticmethod
    def _target_space():
        # To get "-mcpu=?", do "cat /proc/cpuinfo". Then search the `model name` on ark.intel.com
        # There can more targets... Let's forget it for a while.
        # tvm.target.Target('c') is too weak...
        _targets = [tvm.target.Target('llvm')]
        # TODO: Allow devices.
        # if tvm.cuda().exist:
        #     _targets.append(tvm.target.cuda())
        #     if cudnn.exists():
        #         _targets.append(tvm.target.Target('cuda -libs=cudnn'))
        # if tvm.rocm().exist:
        #     _targets.append(tvm.target.rocm())
        return _targets


# When using CHI distribution on [0, +inf)
_SAMPLE_CHI_DIST_DF_ = 3

_MAX_SAMPLE_SIZE_ = 64
_MAX_TEST_BATCH_ = _MAX_SAMPLE_SIZE_

_MIN_TEST_HW_ = 128
_MAX_TEST_HW_ = 1024
_HW_NORMAL_DIST_MU_ = (_MIN_TEST_HW_ + _MAX_TEST_HW_ * 3 // 5) // 2
# 3 sigma is hard... we make it 4...
_HW_NORMAL_DIST_SIGMA_ = _HW_NORMAL_DIST_MU_ // 4

@dataclass
class ExecutionConfig:
    module        :tvm.IRModule
    params        :Dict
    n_inp_node    :int
    exe_mode      :str                      = None
    inputs        :List[List[tvm.nd.array]] = field(default_factory=list)
    oracle        :List[List[tvm.nd.array]] = None # None if not required.
    oracle_name   :str                      = "NOT_SET"

    def from_keras(self, model, shape=None, layout="NCHW"):
        self.module, self.params = relay.frontend.from_keras(model, shape, layout)

    @staticmethod
    def exe_mode_space(dynamic_shape=False):
        if dynamic_shape:
            return ['vm', 'debug']
        else:
            return ['vm', 'graph', 'debug']

    def check(self):
        assert isinstance(self.module, tvm.IRModule)
        assert self.params is not None
        assert self.n_inp_node > 0
        assert self.exe_mode != None
        assert self.inputs

    def mutate(self):
        # TODO: Think about better mutation strategies.
        # Create some inputs...
        input_shapes = self.module['main'].checked_type.arg_types[:self.n_inp_node]

        dynamic_batch_input_id = []
        dynamic_input_ids = []
        for i, s in enumerate(input_shapes):
            if relay.ty.is_dynamic(s):
                dynamic_input_ids.append(i)
                if isinstance(s.shape[0], tvm.tir.Any):
                    dynamic_batch_input_id.append(i)
        
        dy_batch_size_list = [] # if we support dynamic batch.
        n_sample = 1 # if len(dynamic_input_ids) == 0 
        # else: np.random.chisquare
        # We use chisquare dist which give more probability on small samples (faster).
        # See: https://en.wikipedia.org/wiki/Chi-square_distribution
        # Normal dist: \mu and \sigma
        # Chi dist: \mu, \sigma, v
        if len(dynamic_input_ids) != 0:
            n_sample = max(1, int(np.random.chisquare(3)))
            n_sample = min(n_sample, _MAX_SAMPLE_SIZE_)
            if len(dynamic_batch_input_id) != 0:
                start = 0
                for _ in range(n_sample):
                    start += int(np.random.chisquare(_SAMPLE_CHI_DIST_DF_))
                    if start <= _MAX_TEST_BATCH_:
                        dy_batch_size_list.append(start)
                    else:
                        dynamic_input_ids.append(1)
                # From small to big. Crash in small batch is fast path.
                dynamic_input_ids.sort()
            

        # We assume there's a batch dim
        # TODO: Make it more genral...
        def _concretize_non_batch_dim(shape :relay.TensorType):
            concrete_shape = []
            for idx, x in enumerate(shape.shape):
                if isinstance(x, tvm.tir.Any):
                    if idx == 0:
                        concrete_shape.append(tvm.tir.Any())
                    else:
                        dim = int(np.random.uniform(_HW_NORMAL_DIST_MU_, _HW_NORMAL_DIST_SIGMA_))
                        dim = min(dim, _MAX_TEST_HW_)
                        dim = max(dim, _MIN_TEST_HW_)
                        concrete_shape.append(dim)
                else:
                    concrete_shape.append(int(x))
            return relay.TensorType(shape=concrete_shape, dtype=shape.dtype)

        # clear inputs
        self.inputs = []
        for i in range(n_sample):
            this_input = []
            for shape in input_shapes:
                shape_type = _concretize_non_batch_dim(shape)
                shape_ = list(shape_type.shape)
                dtype_ = shape_type.dtype
                if relay.ty.is_dynamic(shape_type):
                    # Still dynamic means batch dim is dynamic
                    shape_[0] = dy_batch_size_list[i]
                # nd.array empty is dangerous! (causing inf)
                shape_ = [int(x) for x in shape_]
                data = np.zeros(shape=shape_, dtype=dtype_)
                this_input.append(tvm.nd.array(data))
            self.inputs.append(this_input)
        
        self.exe_mode = 'graph' # TODO: Test more runtimes.
        # random.choice(self.exe_mode_space(len(dynamic_input_ids) != 0))

    def __deepcopy__(self, meno):
        module = tvm.parser.parse(self.module.astext())
        params = {k:tvm.nd.array(v.numpy()) for k,v in self.params.items()}
        n_inp_node = self.n_inp_node
        exe_mode = deepcopy(self.exe_mode, meno)
        inputs = [[tvm.nd.array(i.numpy()) for i in inp]for inp in self.inputs]
        oracle = None if self.oracle is None else [[tvm.nd.array(i.numpy()) for i in inp]for inp in self.oracle]
        oracle_name = deepcopy(self.oracle_name, meno)
        
        return ExecutionConfig(
            module, params, n_inp_node, exe_mode, inputs, oracle, oracle_name
        )


@dataclass
class Context:
    """Top-level configuration of fuzzer.
    """
    runtime  :ExecutionConfig
    compile  :CompileConfig

    def dump(self, path): # Fix this ...
        to_store_params = {}
        for k, v in self.runtime.params.items():
            to_store_params[k] = v.numpy()
        with open(path, 'wb') as f:
            runtime_conf = {
                'module': self.runtime.module.astext(),
                'params': to_store_params,
                'n_inp_node': self.runtime.n_inp_node,
                'exe_mode': self.runtime.exe_mode,
                'inputs': [[x.numpy() for x in inp] for inp in self.runtime.inputs],
                'oracle': self.runtime.oracle,
                'oracle_name': self.runtime.oracle_name
            }

            compile_conf = {
                'target': self.compile.target,
                'relay_pass_types': self.compile.relay_pass_types,
                'tir_pass_nodes': graph.export_name(self.compile.tir_pass_nodes)
            }

            pickle.dump({
                'runtime': runtime_conf,
                'compile': compile_conf
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.compile.target = data['compile']['target']
            self.compile.relay_pass_types = data['compile']['relay_pass_types']
            self.compile.tir_pass_nodes = graph.recover(data['compile']['tir_pass_nodes'])
    
            for k, v in data['runtime'].items():
                if k == 'module':
                    self.runtime.module = tvm.parser.fromtext(v)
                elif k == 'params':
                    self.runtime.params = {}
                    for k_, v_ in v.items():
                        self.runtime.params[k_] = tvm.nd.array(v_)
                elif k == 'inputs':
                    self.runtime.inputs = [[tvm.nd.array(x) for x in inp] for inp in v],
                else:
                    setattr(self.runtime, k, v)

    def mutate(self):
        self.runtime.mutate()
        self.compile.mutate()

    def check(self):
        self.runtime.check()
        self.compile.check()
