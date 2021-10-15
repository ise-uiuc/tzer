import argparse
import os
import pickle
import tvm

from tzer.tir.seed import get_all_seeds, get_simplenet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seeds', type=str, help='allseeds / simplenet')
    parser.add_argument('-f', '--folders', type=str, help='seeds function folder')
    args = parser.parse_args()

    if args.seeds == 'allseeds':
        tir_functions = get_all_seeds()
    elif args.seeds == 'simplenet':
        tir_functions = get_simplenet()
    else:
        print('no seeeds ...')
        exit()

    if not os.path.exists(args.folders):
        os.makedirs(args.folders)

    for i in range(len(tir_functions)):
        tir_func = tir_functions[i]
        tir_func_mod = tvm.lower(tir_func)

        ctx_path = os.path.join(args.folders, f'{i}.ctx')
        with open(ctx_path, 'wb') as f:
            pickle.dump(tir_func_mod, f)

        json_path = os.path.join(args.folders, f'{i}.json')
        with open(json_path, 'w') as f:
            f.write(tvm.ir.save_json(tir_func_mod))
