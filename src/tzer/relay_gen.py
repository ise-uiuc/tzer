import random
import tvm.relay as relay
from tvm.relay import testing

class FuseOpGen:
    def __init__(self) -> None:
        self._candidate_map = {}

        # self._candidate_map['S'] = ['IS', 'CS', 'e']
        # self._candidate_map['I'] = ['iI', 'ir', 'e']
        # self._candidate_map['C'] = ['cI']
        # IS -> iIS | irS | S
        # CS -> cIS
        # let U -> IS, V -> CS
        # So we got:
        self._candidate_map['S'] = ['U', 'V', 'e']
        self._candidate_map['U'] = ['iU', 'ir', 'S'] # NOTE: ir -> irS cause invalidity
        self._candidate_map['V'] = ['cU']
    
    def gen(self, n_round = 10):
        ret = 'S'
        i = 0
        while ret[-1].isupper():
            candidates = self._candidate_map[ret[-1]].copy()
            if i < n_round and candidates.count('e') != 0:
                # Do not terminate until i >= n_round
                candidates.pop(candidates.index('e'))
            elif i >= n_round and candidates.count('e') != 0:
                candidates = ['e']
            out = random.choice(candidates)
            ret = ret[:-1]
            ret += out
            print(ret)
            i += 1
        if ret[-1] == 'e':
            ret = ret[:-1]
        return ret
    
    def gen_ir(self, n_round = 10):
        expr = self.gen(n_round)
        net = relay.var("data", relay.TensorType((1, 3, 128, 128), "float32"))
        for v in expr:
            if v == 'i':
                net = random.choice(self.injective_candidates())(net)
            elif v == 'r':
                net = random.choice(self.reduce_candidates())(net)
            elif v == 'c':
                net = random.choice(self.out_ewise_fusable_candidates())(net)
        net = relay.Function(relay.analysis.free_vars(net), net)
        print(net.astext())
        return testing.create_workload(net)


    def injective_candidates(self):
        return [
            lambda x : relay.nn.upsampling(x, 1.25, 1.25),
            lambda x : relay.nn.pad(x, ((0, 0), (0, 0), (1, 1), (2, 2))),
            lambda x : relay.sqrt(x),
            lambda x : relay.nn.relu(x),
            # lambda x : relay.nn.prelu(x, 0.1),
            lambda x : relay.abs(x),
            lambda x : relay.ceil(x),
            lambda x : relay.floor(x),
            lambda x : relay.exp(x),
        ]

    def reduce_candidates(self):
        return [
            lambda x : relay.sum(x),
            lambda x : relay.max(x),
            lambda x : relay.min(x),
            lambda x : relay.argmax(x)
        ]

    def out_ewise_fusable_candidates(self):
        return [
            lambda x : relay.nn.conv2d(data=x, weight=relay.var("weight"), kernel_size=(5, 5), channels=8, padding=(1, 1)),
            # lambda x : relay.nn.bitserial_conv2d(data=x, weight=relay.var("weight"), kernel_size=(5, 5), channels=8, padding=(1, 1)), # FIXME: Error???
            lambda x : relay.nn.conv2d_transpose(data=x, weight=relay.var("weight"), kernel_size=(5, 5), channels=8, padding=(1, 1)),
            lambda x : relay.nn.max_pool2d(x, pool_size=(3, 3)),
            lambda x : relay.nn.avg_pool2d(x, pool_size=(3, 3)),
            # lambda x : relay.nn.global_max_pool2d(x)
        ]