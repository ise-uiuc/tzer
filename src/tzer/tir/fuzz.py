from typing import Tuple
import time
import traceback
import random

from termcolor import colored
from tqdm import tqdm
import numpy as np
import tvm
import tvm.testing
from tvm import tir

from . import report, error, seed, oracle
from .config import Config
from .pass_fuzz.pass_mutator import SimplePassMutator
from .visit import get_node_size
from .joint_seed_pool import JointSeedPool, __USE_RANDOM_PASS_GEN__, __USE_FULL_PASS__, __PASS_BASELINE_TESTING__
from .pass_fuzz.pass_mutator import random_tir_passes, tir_pass_graph

try:
    from tvm.contrib import coverage
    __USE_COV__ = True
except Exception as e:
    print(f'No coverage in linked TVM. {e}')
    __USE_COV__ = False

__MAX_TIR_FAIL__: int
__MAX_PASS_FAIL__ = 1
__MIN_SEED_POOL__ = 10

def assert_no_cov(func, *args, **kwargs):
    if __USE_COV__:
        before_cov = coverage.get_now()
    func(*args, **kwargs)
    if __USE_COV__:
        after_cov = coverage.get_now()
        assert before_cov == after_cov, f'Statement {func} has coverage! {before_cov} -> {after_cov}'

class Fuzzer:
    def __init__(self, config: Config) -> None:
        self.config = config
        global __MAX_TIR_FAIL__
        __MAX_TIR_FAIL__ = config.tolerance

        if self.config.use_seeds:
            seeds = seed.get_all_seeds()
            print(colored('Running experiments with all seeds', 'green'))
        else:
            seeds = []
            print(colored('Running experiments without seed.', 'yellow'))

        if self.config.use_coverage:
            # Reset coverage in the beginning to avoid influence on seed generation.
            coverage.reset()

        # self.state = domain.State(seeds, self.config.max_node_size)

        self.pass_mutator = SimplePassMutator()
        self.reporter = report.Reporter(
            config.report_folder, config.use_coverage, config.record_tir)

        self.iter = 0
        self.n_filtered_ast = 0

        self.joint_seed_pool = JointSeedPool(
            tir_func_list=seeds,
            general_cfg_mut=self.config.mutate_control_flow_with_general_purpose_mutators,
            use_none=self.config.use_none,
            max_gen_size=self.config.max_generation_size)

    def run_and_get_cov_increase(self, func: tir.PrimFunc, passes=None) -> Tuple[int, float]:
        assert isinstance(func, tir.PrimFunc) or func is None
        if passes is None:
            passes = []

        if self.config.use_coverage:
            old_now = coverage.get_now()

        t0 = time.time()

        useful_pass_mask = np.ones((len(passes)))

        try:
            oracle.build_and_test(
                func,
                passes,
                self.config.building_timeout_in_seconds,
                self.config.diff_test_rounds,
                self.config.use_coverage,
                useful_pass_mask
            )
            self.n_pass_compilation += 1
        except (error.RuntimeFailure, error.MaybeDeadLoop) as e:
            self.reporter.report_tir_bug(e, func, passes, None, str(e))
            self.n_pass_compilation += 1
        except (error.IncorrectResult, error.PerfDegradation) as e:
            params = e.args[0]
            self.reporter.report_tir_bug(e, func, passes, params, str(e))
            self.n_pass_compilation += 1
        except tvm.TVMError as e:
            self.n_failed += 1
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print(f'TZER Implementation error here..')
            assert_no_cov(traceback.print_exc)

        cov_increase: int = coverage.get_now(
        ) - old_now if self.config.use_coverage else 1

        if self.reporter.cov_by_time_file:
            self.reporter.record_coverage()

        if self.reporter.tir_by_time_file:
            self.reporter.record_tir_and_passes(func, passes)

        return cov_increase, time.time() - t0, useful_pass_mask

    def ready(self):
        # Fuzzing progress
        self.start_point = 0 if self.config.iterations is not None else time.time()
        self.end_point = self.config.iterations if self.config.iterations is not None \
            else self.start_point + self.config.fuzzing_time_in_minutes * 60
        self.current_point = self.start_point

        # Time information
        self.start_time = time.time()
        self.last_time = self.start_time

        # Compilation rate
        self.n_pass_compilation = 0
        self.n_failed = 0

    def start(self):
        self.ready()
        with tqdm(total=int(self.end_point - self.start_point)) as pbar:
            while self.current_point < self.end_point:
                self.fuzz_new(pbar)

    def fuzz_new(self, pbar):
        if self.joint_seed_pool.size() == 0:
            self.joint_seed_pool.put()  # empty tir by default.

        try:
            seed_idx, seed = self.joint_seed_pool.random_pick()

            fallback = lambda : None
            # IR mutant
            if not self.config.use_pass:
                func_mutant = self.joint_seed_pool.mutate_ir(seed.tir_func)
            elif seed.n_ir_cont_fail < __MAX_TIR_FAIL__:
                func_mutant = self.joint_seed_pool.mutate_ir(seed.tir_func)
                def ir_fail():
                    self.joint_seed_pool.seeds[seed_idx].n_ir_cont_fail += 1
                fallback = ir_fail
            else:
                func_mutant = seed.tir_func
            # Pass mutant
            if not self.config.use_pass:
                pass_mutant = []
            elif __USE_FULL_PASS__:
                pass_mutant = list(tir_pass_graph.tir_pass_nodes.values())
                random.shuffle(pass_mutant)
            elif __USE_RANDOM_PASS_GEN__:
                pass_mutant = random_tir_passes()
            elif seed.n_ir_cont_fail >= __MAX_TIR_FAIL__ and seed.n_pass_cont_fail < __MAX_PASS_FAIL__:
                pass_mutant = random_tir_passes()
                def pass_fail():
                    self.joint_seed_pool.seeds[seed_idx].n_pass_cont_fail += 1
                fallback = pass_fail
            else:
                pass_mutant = seed.pass_seq

        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print("Generation failure")
            # assert_no_cov(traceback.print_exc)
            self.n_failed += 1
            self.update_loop_info(pbar, 0, 0, 'gen-failure')
            return

        passes = [p.mutate()
                  for p in pass_mutant] if pass_mutant is not None else None

        n_pass_compilation_prev = self.n_pass_compilation
        cov_increase, build_time, useful_pass_mask = self.run_and_get_cov_increase(
            func_mutant, passes)

        if (cov_increase > 0 or not self.config.use_coverage_feedback) \
                and self.n_pass_compilation != n_pass_compilation_prev:
            assert self.n_pass_compilation == n_pass_compilation_prev + 1
            # only append valid seed
            # Shrink the pass mutant.
            pass_seq = []
            for idx, mask in enumerate(useful_pass_mask):
                if mask: # this pass triggered coverage!
                    pass_seq.append(pass_mutant[idx])
            if not self.config.use_pass:
                assert pass_seq == []
            if not self.config.use_pass:
                assert self.joint_seed_pool.seeds[seed_idx].n_ir_cont_fail < __MAX_TIR_FAIL__
            if self.joint_seed_pool.seeds[seed_idx].n_ir_cont_fail < __MAX_TIR_FAIL__:
                self.joint_seed_pool.put(func_mutant, pass_seq)
            self.joint_seed_pool.seeds[seed_idx].pass_seq = pass_seq
            self.joint_seed_pool.seeds[seed_idx].n_ir_cont_fail = 0
            self.joint_seed_pool.seeds[seed_idx].n_pass_cont_fail = 0
        else:
            if self.config.use_pass:
                fallback()
                if __PASS_BASELINE_TESTING__:  # IR only
                    if self.joint_seed_pool.seeds[seed_idx].n_ir_cont_fail >= __MAX_TIR_FAIL__:
                        self.joint_seed_pool.seeds[seed_idx].n_ir_cont_fail = 0
                        self.joint_seed_pool.seeds[seed_idx].n_pass_cont_fail = 0
                # IR + Pass
                elif self.joint_seed_pool.seeds[seed_idx].n_ir_cont_fail >= __MAX_TIR_FAIL__ and \
                    self.joint_seed_pool.seeds[seed_idx].n_pass_cont_fail >= __MAX_PASS_FAIL__:
                    self.joint_seed_pool.seeds[seed_idx].n_ir_cont_fail = 0
                    self.joint_seed_pool.seeds[seed_idx].n_pass_cont_fail = 0

            # if self.joint_seed_pool.size() < __MIN_SEED_POOL__:
            #     self.joint_seed_pool.put() # empty tir by default.

        node_count = '?' if self.config.use_none else get_node_size(
            func_mutant)
        self.update_loop_info(pbar, build_time, node_count, 'mut-new')

    def update_loop_info(self, pbar, build_time, node_count, phase):
        # Loop information updating.
        self.iter += 1
        t = time.time()
        compile_rate = self.n_pass_compilation / (
            self.n_pass_compilation + self.n_failed)
        pbar.set_description(
            f'#it: {self.iter}, '
            f'#bugs: {self.reporter.n_bug}, '
            f'#pool: {self.joint_seed_pool.size()}, '
            f'%compile: {compile_rate:.2f}, '
            f'T(build+run): {build_time:.2f}, '
            f'time: {time.time() - self.last_time:.2f}, '
            f'#cov: {f"{coverage.get_now()} / {coverage.get_total()}" if self.config.use_coverage else "?"}, '
            f'#node: {node_count}, '
        )
        update = t - self.last_time if self.config.iterations is None else 1
        pbar.update(update)
        self.current_point += update

        self.last_time = t
        self.reporter.record_compile_rate(f'{compile_rate:.2f}')
        self.reporter.record_iteration(self.iter)
        if self.config.use_coverage:
            # The count can be directly calculated if Tzer uses coverage
            self.reporter.record_valid_seed_achieving_new_cov_count(
                len(self.joint_seed_pool.seeds) - self.joint_seed_pool.initial_pool_size
            )
