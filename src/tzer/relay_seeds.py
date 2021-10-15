import tvm.relay as relay
from tvm.relay import testing
import time

MODEL_SEEDS = [
    # get_lstm(),
    testing.resnet.get_workload(batch_size=1, num_layers=18, image_shape=(128, 128, 3), layout="NHWC"),
    testing.squeezenet.get_workload(batch_size=1, num_classes=100, image_shape=(3, 128, 128), dtype='float32'),
    testing.mobilenet.get_workload(image_shape=(3, 128, 128)),
    testing.mlp.get_workload(batch_size=1, num_classes=10, image_shape=(1, 64, 64)),
    testing.dcgan.get_workload(batch_size=1),
    testing.inception_v3.get_workload(),
    testing.vgg.get_workload(batch_size=1),
    testing.densenet.get_workload(),
]

if __name__ == '__main__':
    from tvm.contrib import coverage
    for mod, params in MODEL_SEEDS:
        before_cov = coverage.get_now()
        before_time = time.time()
        relay.build(ir_mod=mod, params=params, target='llvm')
        print(f'Increased cov: {coverage.get_now() - before_cov} in {time.time() - before_time} s')

    print(coverage.get_now())