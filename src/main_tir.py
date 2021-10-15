import tzer.tir
import random
import numpy as np

if __name__ == '__main__':
    random.seed(2333)
    np.random.seed(2333)

    parser = tzer.tir.config.make_arg_parser()
    args = parser.parse_args()

    config = tzer.tir.config.config_from_args(args)
    fuzzer = tzer.tir.fuzz.Fuzzer(config)
    fuzzer.start()
