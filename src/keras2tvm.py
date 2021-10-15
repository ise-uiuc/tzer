from tensorflow import keras
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tvm.relay.testing.tf as tf_testing

import tvm
from tvm import relay
import argparse
from tvm.ir.module import IRModule
import tvm.relay as relay
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.backend import graph_executor_codegen

tf.enable_eager_execution()

def keras2tf(model):
    full_model = tf.function(lambda x: model(x))
    freeze_shape = model.inputs[0].shape

    shape_list = []
    for v in freeze_shape:
        try:
            shape_list.append(int(v))
        except TypeError as e:
            shape_list.append(1)

    full_model = full_model.get_concrete_function(
        tf.TensorSpec(tf.TensorShape(shape_list), model.inputs[0].dtype))
    frozen_func = convert_variables_to_constants_v2(full_model)
    
    # print(frozen_func.graph.as_graph_def())
    return frozen_func.graph.as_graph_def()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to the model file.')
    args = parser.parse_args()

    model = keras.models.load_model(args.model)
    print(model.inputs[0].name)
    print(model.inputs[0].shape)
    shape_list = []
    for v in model.inputs[0].shape:
        try:
            shape_list.append(int(v))
        except TypeError as e:
            shape_list.append(1)
    
    # TVM:
    graph_def = keras2tf(model)
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    module, params = relay.frontend.from_tensorflow(graph_def)

    # compile the model
    target = tvm.target.Target("llvm")

    if params is not None:
        module = IRModule.from_expr(bind_params_by_name(module["main"], params))

    # convert relay ir to tir
    with tvm.transform.PassContext(opt_level=0):
        module, params = relay.optimize(module, target=target, params=params)

        module = relay.transform.DynamicToStatic()(module)
        graph, lowered_func, params = graph_executor_codegen.GraphExecutorCodegen(None, target=target).codegen(module['main'])
        ir_m = lowered_func[target]
    
    with open(f"{args.model}.tir", 'w') as f:
        f.write(tvm.ir.save_json(ir_m))
