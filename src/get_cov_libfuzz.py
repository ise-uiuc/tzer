import argparse
import matplotlib.pyplot as plt
import os
from tvm.contrib import coverage
import tvm
import multiprocessing as mp

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
        print(f'{path} load success')
        tvm.build(m)
        print(f'{path} build success')


    with mp.Manager() as manager:
        return_dict = manager.dict()

        p = mp.Process(target=run, args=(load_and_build, (path, ), {}, return_dict))
        p.start()
        p.join(timeout=timeout)

        coverage.set_now(return_dict['cov'])
        coverage.set_hitmap(return_dict['hitmap'])

        if not p.exitcode == 0:
            print(f'{path} crashes')
            msg = f'{path} terminated abnormally with exit code: {p.exitcode}'
            raise RuntimeFailure(msg)
    
        if return_dict['err'] != '':
            raise return_dict['err']


class CovGetter:
    def __init__(self, timeout) -> None:
        self.timeout = timeout  # type: ignore

    def save(self, folder, build_folder, overwrite):
        if not os.path.exists(folder):
            os.mkdir(folder)

        cov_by_time_fname = os.path.join(folder, 'cov_by_time.txt')
        if os.path.exists(cov_by_time_fname) and not overwrite:
            response = input(
                f'{cov_by_time_fname} exists. Overwrite it? [y/n]')
        else:
            response = 'y'
        if response == 'y':
            with open(cov_by_time_fname, 'w') as f:
                pass

            start_time_fname = os.path.join(build_folder, 'start_time.txt')
            with open(start_time_fname, 'r') as f:
                data = f.read()
            start_time = float(data)
            coverage.reset()

            crashes = os.listdir(build_folder)
            crashes = [i for i in crashes if i.startswith('crash')]
            crashes = [os.path.join(build_folder, i) for i in crashes]

            corpus_dir = os.path.join(build_folder, 'corpus')
            corpus = os.listdir(corpus_dir)
            corpus = [os.path.join(corpus_dir, i) for i in corpus]
            
            file_time_mapping = []
            for i in corpus + crashes:
                t = os.path.getctime(i)
                file_time_mapping.append((t, i))

            file_time_mapping = sorted(file_time_mapping, key=lambda x:x[0])
            
            all_seeds_cnt = len(file_time_mapping)
            valid_seeds_cnt = 0
            new_cov_valid_seeds_cnt = 0
            
            last_cov = coverage.get_now()
            for t, c in file_time_mapping:
                valid = False
                try:
                    load_corpus(c, 120)
                    valid = True
                except Exception as e:
                    pass

                time = t - start_time
                cov_now = coverage.get_now()

                if valid:
                    valid_seeds_cnt += 1
                    if cov_now > last_cov:
                        print(c)
                        new_cov_valid_seeds_cnt += 1
                        last_cov = cov_now

                with open(cov_by_time_fname, 'a') as f:
                    f.write(f'{time:.2f},{cov_now},\n')

            valid_seed_fname = os.path.join(folder, 'valid_seed_new_cov_count.txt')
            with open(valid_seed_fname, 'w') as f:
                f.write(str(new_cov_valid_seeds_cnt))
            
            print(f'compile rate:{valid_seeds_cnt*100/all_seeds_cnt:.2f}')
            print(f'new coverage seed: {new_cov_valid_seeds_cnt}')


if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--report-folder', type=str, help='coverage report folder')
    parser.add_argument('-b', '--build-folder', type=str, help='tvm-libfuzz build folder')                 
    parser.add_argument('-t', '--timeout', type=float, default=120,
                        nargs='?', help='building timeout (seconds)')
    parser.add_argument('-w', '--overwrite', help="Overwrite cov_by_time.txt even if it exists",
                        action="store_true")
    args = parser.parse_args()

    getter = CovGetter(args.timeout)
    getter.save(args.report_folder, args.build_folder, args.overwrite)
