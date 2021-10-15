from tzer import context, relay_gen

import tvm.relay as relay
from tvm.relay import testing
import onnx

def example(batch_dim=1):
    out_channels = 32

    data = relay.var("data", relay.TensorType((batch_dim, 3, 224, 224), "float32"))
    weight = relay.var("weight")
    bn_gamma = relay.var("bn_gamma")
    bn_beta = relay.var("bn_beta")
    bn_mmean = relay.var("bn_mean")
    bn_mvar = relay.var("bn_var")

    simple_net = relay.nn.conv2d(
        data=data, weight=weight, kernel_size=(5, 5), channels=out_channels, padding=(1, 1)
    )
    simple_net = relay.nn.batch_norm(simple_net, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
    simple_net = relay.nn.relu(simple_net)
    simple_net = relay.Function(relay.analysis.free_vars(simple_net), simple_net)

    return testing.create_workload(simple_net)

# _RESNET18_ONNX_ = onnx.load('resnet18-v2-7.onnx')
# _RESNET50_ONNX_ = onnx.load('resnet50-v2-7.onnx')

# def resnet18():
#     input_name = "data"
#     shape_dict = {input_name: [1, 3, 224, 224]}
#     mod, params = relay.frontend.from_onnx(_RESNET18_ONNX_, shape_dict)
#     mod = relay.transform.InferType()(mod)
#     return mod, params

# def resnet50():
#     input_name = "data"
#     shape_dict = {input_name: [128, 3, 224, 224]}
#     mod, params = relay.frontend.from_onnx(_RESNET50_ONNX_, shape_dict)
#     mod = relay.transform.InferType()(mod)
#     return mod, params

gen = relay_gen.FuseOpGen()

def make_context(model=None):
    batch_dim = 1 # TODO: Support VM.
    # mod, params = resnet50()
    if model == None:
        mod, params = example(batch_dim)
    else:
        mod, params = model
    # mod, params = gen.gen_ir()
    
    rt = context.ExecutionConfig(module=mod, params=params, n_inp_node=1)
    compile = context.CompileConfig()

    ctx = context.Context(runtime=rt, compile=compile)
    ctx.mutate()
    ctx.check()

    # print(ctx.runtime.exe_mode)
    # print(ctx.compile.target, ctx.compile.get_device())
    # for inp in ctx.runtime.inputs:
    #     for t in inp:
    #         print(t.shape)
    # print(ctx.compile.get_device())
    # for rpass in ctx.compile.relay_passes:
    #     print(rpass)
    return ctx

