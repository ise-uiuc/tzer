from sys import meta_path
import tvm
from tvm import runtime
from tvm.ir import transform
from tvm.ir.module import IRModule
import tvm.relay as relay
from tvm import tir
from tvm.relay.backend import graph_executor_codegen

import time
import numpy as np
from tvm.contrib import graph_executor
from tvm.relay.build_module import bind_params_by_name

from .verify import *
from .context import Context

_FUZZ_TVM_NO_OPT_TAG_ = 'TVM_NO_OPT'
_FUZZ_TVM_OPT_TAG_ = 'TVM_OPT'


def execute_both_mode(ctx :Context) -> Context:
    params = ctx.runtime.params
    module = ctx.runtime.module
    target = ctx.compile.target
    dev = tvm.cpu(0)

    if params is not None:
        module = IRModule.from_expr(bind_params_by_name(module["main"], params))

    with tvm.transform.PassContext(opt_level=0):
        libo0 = relay.build(module, target=target, params=params)
        graph_exe_o0 = graph_executor.GraphModule(libo0["default"](ctx.compile.get_device()))

    with tvm.transform.PassContext(opt_level=0):
        module, params = relay.optimize(module, target=target, params=params)

    with tvm.transform.PassContext(opt_level=4):
        with ctx.compile.target: # There can be target-aware passes...
            # compile the model
            module = tvm.transform.Sequential(passes=[f() for f in ctx.compile.relay_pass_types], opt_level=4)(module)
    
    # convert relay ir to tir
    with tvm.transform.PassContext(opt_level=0):
        module, params = relay.optimize(module, target=target, params=params)

        graph, lowered_func, params = graph_executor_codegen.GraphExecutorCodegen(None, target=target).codegen(module['main'])
        ir_m = lowered_func[target]

    with tvm.transform.PassContext(opt_level=4):
        opt = tvm.transform.Sequential(
                passes=[n.mutate() for n in ctx.compile.tir_pass_nodes],
                opt_level=4
        )

        opt_ir_m = opt(ir_m)

        opt_execute = tvm.build(opt_ir_m, target=target)

        graph_exe_opt = tvm.contrib.graph_executor.create(graph, opt_execute, dev)
        graph_exe_opt.load_params(runtime.save_param_dict(params))


    opt_perf_times = []
    nopt_perf_times = []
    for inp in ctx.runtime.inputs:
        graph_exe_o0.set_input('data', inp[0]) # TODO: support multi-tensor.

        # Write result.
        nopt_begin = time.perf_counter() # TODO support multiple inputs
        graph_exe_o0.run()
        nopt_result = graph_exe_o0.get_output(0)
        nopt_perf_times.append(time.perf_counter() - nopt_begin)

        # !Check non-optimzed output
        if ctx.runtime.oracle:
            assert_allclose(
                obtained=nopt_result, 
                desired=ctx.runtime.oracle,
                obtained_name=_FUZZ_TVM_NO_OPT_TAG_,
                oracle_name=ctx.runtime.oracle_name)

        # Write result.
        graph_exe_opt.set_input('data', inp[0])
        opt_begin = time.perf_counter() # TODO support multiple inputs
        graph_exe_opt.run()
        opt_result = graph_exe_opt.get_output(0)
        opt_perf_times.append(time.perf_counter() - opt_begin)

        # !Check optimzed output
        assert_allclose(
            obtained=opt_result, 
            desired=nopt_result,
            obtained_name=_FUZZ_TVM_OPT_TAG_,
            oracle_name=_FUZZ_TVM_NO_OPT_TAG_)

    # Think about graph executor only.
    # TODO: Test performance later.
    # for nopt_time, opt_time in zip(nopt_perf_times, opt_perf_times):
    #     # !Check optimization.
    #     assert_no_perf_degrad(optimzed_time=opt_time, non_optimized_time=nopt_time)

    return ctx

def execute_tir_mode(ctx :Context) -> Context:
    out_shape = ctx.runtime.module['main'].ret_type.shape
    params = ctx.runtime.params
    module = ctx.runtime.module
    target = ctx.compile.target
    dev = tvm.cpu(0)

    if params is not None:
        module = IRModule.from_expr(bind_params_by_name(module["main"], params))

    with tvm.transform.PassContext(opt_level=0):
        libo0 = relay.build(module, target=target, params=params)
        graph_exe_o0 = graph_executor.GraphModule(libo0["default"](ctx.compile.get_device()))


    # convert relay ir to tir
    with tvm.transform.PassContext(opt_level=0):
        module, params = relay.optimize(module, target=target, params=params)

        graph, lowered_func, params = graph_executor_codegen.GraphExecutorCodegen(None, target=target).codegen(module['main'])
        ir_m = lowered_func[target]

    with tvm.transform.PassContext(opt_level=4):
        opt = tvm.transform.Sequential(
                passes=[n.mutate() for n in ctx.compile.tir_pass_nodes],
                opt_level=4
        )

        opt_ir_m = opt(ir_m)

        opt_execute = tvm.build(opt_ir_m, target=target)

        graph_exe_opt = tvm.contrib.graph_executor.create(graph, opt_execute, dev)
        graph_exe_opt.load_params(runtime.save_param_dict(params))


    opt_perf_times = []
    nopt_perf_times = []
    for inp in ctx.runtime.inputs:
        graph_exe_o0.set_input('data', inp[0]) # TODO: support multi-tensor.

        # Write result.
        nopt_begin = time.perf_counter() # TODO support multiple inputs
        graph_exe_o0.run()
        nopt_result = graph_exe_o0.get_output(0, tvm.nd.empty(out_shape))
        nopt_perf_times.append(time.perf_counter() - nopt_begin)

        # !Check non-optimzed output
        if ctx.runtime.oracle:
            assert_allclose(
                obtained=nopt_result, 
                desired=ctx.runtime.oracle,
                obtained_name=_FUZZ_TVM_NO_OPT_TAG_,
                oracle_name=ctx.runtime.oracle_name)

        # Write result.
        graph_exe_opt.set_input('data', inp[0])
        opt_begin = time.perf_counter() # TODO support multiple inputs
        graph_exe_opt.run()
        opt_result = graph_exe_opt.get_output(0, tvm.nd.empty(out_shape))
        opt_perf_times.append(time.perf_counter() - opt_begin)

        # !Check optimzed output
        assert_allclose(
            obtained=opt_result, 
            desired=nopt_result,
            obtained_name=_FUZZ_TVM_OPT_TAG_,
            oracle_name=_FUZZ_TVM_NO_OPT_TAG_)

    # Think about graph executor only.
    # TODO: Test performance later.
    # for nopt_time, opt_time in zip(nopt_perf_times, opt_perf_times):
    #     # !Check optimization.
    #     assert_no_perf_degrad(optimzed_time=opt_time, non_optimized_time=nopt_time)

    return ctx



def execute(ctx :Context) -> Context:
    out_shape = ctx.runtime.module['main'].ret_type.shape

    # runtime non-optimized model
    with tvm.transform.PassContext(opt_level=0):
        libo0 = relay.build(ctx.runtime.module, target=ctx.compile.target, params=ctx.runtime.params)
        graph_exe_o0 = graph_executor.GraphModule(libo0["default"](ctx.compile.get_device()))

    with tvm.transform.PassContext(opt_level=0):
        module, params = relay.optimize(ctx.runtime.module, target=ctx.compile.target, params=ctx.runtime.params)

    with tvm.transform.PassContext(opt_level=4):
        with ctx.compile.target: # There can be target-aware passes...
            # compile the model
            # module = IRModule.from_expr(bind_params_by_name(module["main"], params))
            # module = pass_optimizer.optimize_module(module, ctx.compile.relay_pass_types, ctx.compile.target)
            # print(pass_optimizer.hit_cnt)
            # module = IRModule.from_expr(bind_params_by_name(module["main"], params))
            module = tvm.transform.Sequential(
                passes=[f() for f in ctx.compile.relay_pass_types], opt_level=4)(module)
            lib_opt = relay.build(module, target=ctx.compile.target, params=params)
            graph_exe_opt = graph_executor.GraphModule(lib_opt["default"](ctx.compile.get_device()))

    opt_perf_times = []
    nopt_perf_times = []
    for inp in ctx.runtime.inputs:
        graph_exe_o0.set_input('data', inp[0]) # TODO: support multi-tensor.

        # Write result.
        nopt_begin = time.perf_counter() # TODO support multiple inputs
        graph_exe_o0.run()
        nopt_result = graph_exe_o0.get_output(0, tvm.nd.empty(out_shape))
        nopt_perf_times.append(time.perf_counter() - nopt_begin)

        # !Check non-optimzed output
        if ctx.runtime.oracle:
            assert_allclose(
                obtained=nopt_result, 
                desired=ctx.runtime.oracle,
                obtained_name=_FUZZ_TVM_NO_OPT_TAG_,
                oracle_name=ctx.runtime.oracle_name)

        # Write result.
        graph_exe_opt.set_input('data', inp[0])
        opt_begin = time.perf_counter() # TODO support multiple inputs
        graph_exe_opt.run()
        opt_result = graph_exe_opt.get_output(0, tvm.nd.empty(out_shape))
        opt_perf_times.append(time.perf_counter() - opt_begin)

        # !Check optimzed output
        assert_allclose(
            obtained=opt_result, 
            desired=nopt_result,
            obtained_name=_FUZZ_TVM_OPT_TAG_,
            oracle_name=_FUZZ_TVM_NO_OPT_TAG_)

    # Think about graph executor only.
    # TODO: Test performance later.
    # for nopt_time, opt_time in zip(nopt_perf_times, opt_perf_times):
    #     # !Check optimization.
    #     assert_no_perf_degrad(optimzed_time=opt_time, non_optimized_time=nopt_time)

    return ctx
