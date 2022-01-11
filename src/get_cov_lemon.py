import argparse
import pickle
import matplotlib.pyplot as plt
import os
from tvm.contrib import coverage
import tvm
import multiprocessing as mp

import datetime

from tzer.tir.error import MaybeDeadLoop, RuntimeFailure


def load_corpus(path, timeout):

    def run(func, args, kwargs, return_dict):
        return_dict['err'] = ''
        return_dict['result'] = None

        try:
            result = func(*args, **kwargs)
            return_dict['result'] = result
        except Exception as e:
            return_dict['err'] = e

        return_dict['cov'] = coverage.get_now()
        return_dict['hitmap'] = coverage.get_hitmap()


    def load_and_build(path):
        with open(path, 'r') as f:
            data = f.read()
        m = tvm.ir.load_json(data)
        tvm.build(m)


    with mp.Manager() as manager:
        return_dict = manager.dict()

        p = mp.Process(target=run, args=(load_and_build, (path, ), {}, return_dict))
        p.start()
        p.join(timeout=timeout)

        coverage.set_now(return_dict['cov'])
        coverage.set_hitmap(return_dict['hitmap'])

        if not p.exitcode == 0:
            msg = f'{path} terminated abnormally with exit code: {p.exitcode}'
            raise RuntimeFailure(msg)
    
        if return_dict['err'] != '':
            raise return_dict['err']


class CovGetter:
    def __init__(self, timeout) -> None:
        self.timeout = timeout  # type: ignore

    def save(self, folder, build_folder, timeout, overwrite):
        if not os.path.exists(folder):
            os.mkdir(folder)

        cov_by_time_fname = os.path.join(folder, 'cov_by_time.txt')
        if os.path.exists(cov_by_time_fname) and not overwrite:
            response = input(
                f'{cov_by_time_fname} exists. Overwrite it? [y/n]')
        else:
            response = 'y'

        if response == 'y':
            coverage.reset()

            tir_models =  build_folder
            timestamp_log = os.path.join(tir_models, 'lemon_results_timestamp.log')

            with open(timestamp_log, 'rb') as f:
                file_time_mapping = pickle.load(f)

            file_time_mapping = sorted(file_time_mapping, key=lambda x:x[0])
            start_time = file_time_mapping[0][0]
            
            all_seeds_cnt = len(file_time_mapping)
            valid_seeds_cnt = 0
            new_cov_valid_seeds_cnt = 0
            
            last_cov = coverage.get_now()
            for t, c in file_time_mapping:
                time = t - start_time
                if time > timeout:
                    break

                valid = False
                try:
                    c = os.path.join(tir_models, f'{c}.tir')
                    load_corpus(c, 120)
                    valid = True
                except Exception as e:
                    pass

                cov_now = coverage.get_now()

                if valid:
                    valid_seeds_cnt += 1
                    if cov_now > last_cov:
                        print(c)
                        new_cov_valid_seeds_cnt += 1
                        last_cov = cov_now

                with open(cov_by_time_fname, 'a') as f:
                    f.write(f'{time:.2f},{cov_now},\n')
            
            print(f'compile rate:{valid_seeds_cnt*100/all_seeds_cnt:.2f}')
            print(f'new coverage seed: {new_cov_valid_seeds_cnt}')


if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--report-folder', type=str, help='coverage report folder')
    parser.add_argument('-r', '--result-folder', type=str, help='tvm-libfuzz build folder')
    parser.add_argument('-t', '--timeout', type=float, default=180, help='timeout (seconds)')          
    parser.add_argument('-b', '--build-timeout', type=float, default=120, nargs='?', help='building timeout (seconds)')
    parser.add_argument('-w', '--overwrite', help="Overwrite cov_by_time.txt even if it exists",
                        action="store_true")
    args = parser.parse_args()

    getter = CovGetter(args.build_timeout)
    getter.save(args.report_folder, args.result_folder, args.timeout, args.overwrite)
