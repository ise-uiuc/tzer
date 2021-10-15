import argparse
from typing import List, Tuple
import matplotlib.pyplot as plt
import dill as pickle
from dill import UnpicklingError
import os
from tqdm import tqdm
from tvm._ffi.base import TVMError
from tvm.contrib import coverage
from tzer.tir import error, oracle
from tzer.tir import report


class CovGetter:
    def __init__(self, timeout) -> None:
        self.timeout = timeout  # type: ignore

    def save(self, folder, overwrite):
        tir_by_time_file = open(os.path.join(
            folder, 'tir_by_time.pickle'), 'rb')

        cov_by_time_fname = os.path.join(folder, 'cov_by_time.txt')
        if os.path.exists(cov_by_time_fname) and not overwrite:
            response = input(
                f'{cov_by_time_fname} exists. Overwrite it? [y/n]')
        else:
            response = 'y'
        if response == 'y':
            self.reporter = report.Reporter(
                folder,
                use_coverage=True,
                record_tir=False,
                use_existing_dir=True
            )

            coverage.reset()
            cov_now = coverage.get_now()
            cov_hitmap = coverage.get_hitmap()

            print(f'Processing {self.reporter.report_folder}..')
            total_length = tir_by_time_file.seek(0, 2)
            tir_by_time_file.seek(0, 0)
            last_pos = 0
            it = 0
            valid_seed_new_cov_count = 0
            with tqdm(total=total_length) as pbar:
                while True:
                    try:
                        time, func, passes = pickle.load(tir_by_time_file)
                    except EOFError:
                        break
                    except (TVMError, UnpicklingError):
                        tir_by_time_file.seek(1, 1)
                        continue
                    try:
                        coverage.set_now(cov_now)
                        coverage.set_hitmap(cov_hitmap)
                        oracle.build_and_test(
                            func,
                            passes,
                            self.timeout,
                            0,
                            True,
                        )
                        valid = True
                    except (error.RuntimeFailure, error.MaybeDeadLoop, TVMError):
                        valid = False
                    if valid and coverage.get_now() > cov_now:
                        valid_seed_new_cov_count += 1
                    cov_now = coverage.get_now()
                    cov_hitmap = coverage.get_hitmap()
                    self.reporter.record_coverage(time)
                    self.reporter.record_valid_seed_achieving_new_cov_count(
                        valid_seed_new_cov_count
                    )
                    it += 1
                    pbar.update(tir_by_time_file.tell() - last_pos)
                    last_pos = tir_by_time_file.tell()

            print(f'Finished processing {self.reporter.report_folder}!')
        tir_by_time_file.close()


if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folders', type=str,
                        nargs='+', help='bug report folder')
    parser.add_argument('-t', '--timeout', type=float, default=2,
                        nargs='?', help='building timeout (seconds)')
    parser.add_argument('-w', '--overwrite', help="Overwrite cov_by_time.txt even if it exists",
                        action="store_true")
    args = parser.parse_args()

    getter = CovGetter(args.timeout)

    for f in args.folders:
        getter.save(f, args.overwrite)
