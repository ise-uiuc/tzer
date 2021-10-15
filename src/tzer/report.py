from .context import Context

try:
    from tvm.contrib import coverage
except Exception as e:
    print(f'No coverage in linked TVM. {e}')

import time
import os
import uuid
import datetime
import git

_METADATA_NAME_ = 'meta.txt'
_COV_BY_TIME_NAME_ = 'cov_by_time.txt'

class TVMFuzzerUsageError(Exception):
    def __str__(self):
        return f'Please TVMFuzzer in the right way... {self.message}'

class Reporter:
    def __init__(self, report_folder=None) -> None:
        # Checks
        tvm_home = os.getenv('TVM_HOME')
        if not tvm_home or not os.path.exists(tvm_home):
            raise TVMFuzzerUsageError('got incorrect env var `TVM_HOME`: "{tvm_home}"')

        self.start_time = time.perf_counter()
        self.report_folder = report_folder
        
        if report_folder is None:
            self.report_folder = f'fuzzing-report-{uuid.uuid4()}'
        
        if os.path.exists(self.report_folder):
            # TODO: Allow continous fuzzing...
            raise TVMFuzzerUsageError(
                f'{self.report_folder} already exist... We want an empty folder to report...')
        
        os.mkdir(self.report_folder)
        print(f'Create report folder: {self.report_folder}')
        
        print(f'Using `{self.report_folder}` as the fuzzing report folder')
        with open(os.path.join(self.report_folder, _METADATA_NAME_), 'w') as f:
            fuzz_repo = git.Repo(search_parent_directories=True)
            tvm_repo = git.Repo(search_parent_directories=True)

            def _log_repo(f, tag, repo :git.Repo):
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

        self.n_bug = 0

    def report_bug(self, err_type: Exception, ctx :Context, message: str):
        bug_prefix = f'{type(err_type).__name__}__{uuid.uuid4()}'
        ctx.dump(os.path.join(self.report_folder, f'{bug_prefix}.ctx'))
        with open(os.path.join(self.report_folder, f'{bug_prefix}.error_message.txt'), 'w') as f:
            f.write(message)
        self.n_bug += 1

    def record_coverage(self):
        with open(os.path.join(self.report_folder, _COV_BY_TIME_NAME_), 'a') as f:
            f.write(
                f'{time.perf_counter() - self.start_time :.2f},{coverage.get_now()}\n')
