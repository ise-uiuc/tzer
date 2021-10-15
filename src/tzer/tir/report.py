import dill as pickle
from typing import List, Optional
from tvm import tir
import tvm
import time
import os
import uuid
import datetime
import git

__TVM_INSTRUMENTED__ = False
try:
    from tvm.contrib import coverage
    __TVM_INSTRUMENTED__ = True
except Exception as e:
    print(f'No coverage in linked TVM. {e}')

if __TVM_INSTRUMENTED__:
    assert os.getenv(
        'NO_COV') is None, "Since you want coverage disabled, why linking an instrumented TVM?"

_METADATA_NAME_ = 'meta.txt'
_COV_BY_TIME_NAME_ = 'cov_by_time.txt'
_COMPILATION_RATE_ = 'compile_rate.txt'
_TIR_BY_TIME_NAME_ = 'tir_by_time.pickle'
_ITERATION_ = 'iterations.txt'
_VALID_SEED_NEW_COV_COUNT_ = 'valid_seed_new_cov_count.txt'

class TVMFuzzerUsageError(Exception):
    def __init__(self, msg):
        self.message = msg

    def __str__(self):
        return f'Please TVMFuzzer in the right way... {self.message}'


class Reporter:
    def __init__(self, report_folder=None, use_coverage=True, record_tir=False, use_existing_dir=False) -> None:
        # Checks
        tvm_home = os.getenv('TVM_HOME')
        if not tvm_home or not os.path.exists(tvm_home):
            raise TVMFuzzerUsageError('got incorrect env var `TVM_HOME`: "{tvm_home}"')

        self.start_time = time.perf_counter()
        self.report_folder = report_folder

        if report_folder is None:
            self.report_folder = f'fuzzing-report-{uuid.uuid4()}'

        if use_existing_dir:
            assert os.path.exists(self.report_folder)
        else:
            # TODO: Allow continous fuzzing...
            if os.path.exists(self.report_folder):
                raise TVMFuzzerUsageError(
                    f'{self.report_folder} already exist... We want an empty folder to report...')
            os.mkdir(self.report_folder)
            print(f'Create report folder: {self.report_folder}')

        print(f'Using `{self.report_folder}` as the fuzzing report folder')
        with open(os.path.join(self.report_folder, _METADATA_NAME_), 'w') as f:
            fuzz_repo = git.Repo(search_parent_directories=True)
            tvm_repo = git.Repo(search_parent_directories=True)

            def _log_repo(f, tag, repo: git.Repo):
                f.write(f'{tag} GIT HASH: {repo.head.object.hexsha}\n')
                f.write(f'{tag} GIT STATUS: ')
                f.write(
                    '\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')
                f.write(repo.git.status())
                f.write(
                    '\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n')

            f.write(f'START TIME: {datetime.datetime.now()}')
            _log_repo(f, 'Fuzzer', fuzz_repo)
            _log_repo(f, 'TVM', tvm_repo)

        self.cov_by_time_file = None
        if use_coverage:
            self.cov_by_time_file = open(os.path.join(
                self.report_folder, _COV_BY_TIME_NAME_), 'w')

        self.tir_by_time_file = None
        if record_tir:
            self.tir_by_time_file = open(os.path.join(
                self.report_folder, _TIR_BY_TIME_NAME_), 'wb')

        self.n_bug = 0

    def record_tir_and_passes(self, tir, passes):
        assert self.tir_by_time_file
        pickle.dump((time.perf_counter() - self.start_time, tir, passes),
                    self.tir_by_time_file)

    def record_coverage(self, t=None):
        if t is None:
            t = time.perf_counter() - self.start_time
        assert self.cov_by_time_file
        self.cov_by_time_file.write(
            f'{t:.2f},{coverage.get_now()},\n')

    def record_compile_rate(self, rate):
        with open(os.path.join(self.report_folder, _COMPILATION_RATE_), 'w') as f:
            f.write(rate)

    def record_valid_seed_achieving_new_cov_count(self, count: int):
        with open(os.path.join(self.report_folder, _VALID_SEED_NEW_COV_COUNT_), 'w') as f:
            f.write(str(count))

    def record_iteration(self, iteration: int):
        with open(os.path.join(self.report_folder, _ITERATION_), 'w') as f:
            f.write(str(iteration))

    def report_tir_bug(
        self,
        err: Exception,
        func: tir.PrimFunc,
        passes: Optional[List[tvm.transform.Pass]],
        parameters: Optional[list],
        msg: str
    ):
        bug_prefix = f'{type(err).__name__}__{uuid.uuid4()}'

        with open(os.path.join(self.report_folder, f'{bug_prefix}.ctx'), 'wb') as f:
            pickle.dump({
                'func': func,
                'passes': passes,
                'args': parameters
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.report_folder, f'{bug_prefix}.error_message.txt'), 'w') as f1:
            f1.write(msg)  # type: ignore

        self.n_bug += 1
