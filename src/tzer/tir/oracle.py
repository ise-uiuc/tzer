import os
from typing import Any, List, Optional, Union
import multiprocessing as mp
import time
import numpy as np
import psutil
from dataclasses import dataclass
import tvm
import tvm.testing
from tvm import tir
from . import util, error
from enum import Enum


__USE_PASS__ = os.getenv('PASS') is not None

try:
    from tvm.contrib import coverage
except Exception as e:
    print(f'No coverage in linked TVM. {e}')

class TIRCreationError(Exception):
    pass


def tir_primfunc_to_mod(func: tir.PrimFunc, target: tvm.target.Target = tvm.target.Target('llvm') ) -> tvm.ir.IRModule:
    func = func.with_attr('target', target)
    func = func.with_attr('global_symbol', 'main')
    func = func.with_attr('tir.noalias', True)
    func = func.with_attr('from_legacy_te_schedule', True)
    
    return tvm.ir.IRModule({'main': func})


class BuildStage(Enum):
    COMPILE_NOPT = 'compile nopt'
    COMPILE_OPT = 'compile opt'
    DIFF_TEST = 'perform diff. test.'
    FINISHED = 'finished'


class RunOutcomeStatus(Enum):
    TIMEOUT = 'timeout'
    SUCCESS = 'success'
    EXCEPTION = 'EXCEPTION'
    CRASH = 'CRASH'


@dataclass
class RunOutcome:
    params: Optional[List[Union[Any, np.ndarray]]]
    ret_value: Optional[Any]
    status: RunOutcomeStatus


def outcome_equal(lhs: RunOutcome, rhs: RunOutcome) -> bool:
    if lhs.status != rhs.status:
        return False
    if lhs.status == RunOutcomeStatus.SUCCESS:
        assert lhs.params is not None and rhs.params is not None
        for lhs_param, rhs_param in zip(lhs.params, rhs.params):
            if not util.all_close(lhs_param, rhs_param):
                return False
        return util.all_close(lhs.ret_value, rhs.ret_value)
    elif lhs.status == RunOutcomeStatus.EXCEPTION:
        assert isinstance(lhs.ret_value, Exception) and isinstance(
            rhs.ret_value, Exception)
        return type(lhs.ret_value) == type(rhs.ret_value)
    else:
        return True


def run_module(
    mod: tvm.driver.build_module.OperatorModule,
    np_params: List[Union[Any, np.ndarray]],
) -> Any:
    d = dict()
    try:
        tvm_params = util.np_params_to_tvm_params(np_params)
        d['ret'] = mod(*tvm_params)
        d['params'] = util.tvm_params_to_np_params(tvm_params)
    except Exception as e:
        d['exc'] = e
        raise e

    if 'exc' in d:
        # assert 'ret' not in d and 'params' not in d)
        return RunOutcome(None, d['exc'], RunOutcomeStatus.EXCEPTION)
    else:
        # assert 'ret' in d and 'params' in d and 'exc' not in d
        return RunOutcome(d['params'], d['ret'], RunOutcomeStatus.SUCCESS)

def build_and_test(
    func: tir.PrimFunc,
    passes: List[tvm.ir.transform.Pass],
    build_timeout: float,
    diff_test_round: int,
    use_cov: bool,
    useful_pass_mask = None
):
    def wrapper(
        func: tir.PrimFunc,
        passes: List[tvm.ir.transform.Pass],
        use_cov: bool,
        d: dict,
    ):
        d['stage'] = BuildStage.COMPILE_NOPT
        try:
            useless_pass_idx = []
            mod = tir_primfunc_to_mod(func)
            if diff_test_round > 0 or not __USE_PASS__: # diff. test.
                with tvm.transform.PassContext(opt_level=0):
                    nopt_mod = tvm.build(mod)

            if __USE_PASS__:
                if use_cov:
                    last_cov = coverage.get_now()
                d['stage'] = BuildStage.COMPILE_OPT
                with tvm.transform.PassContext(opt_level=4):
                    for idx, single_pass in enumerate(passes):
                        mod = tvm.transform.Sequential(
                            [single_pass],
                            opt_level=4
                        )(mod)
                        if use_cov:
                            cur_cov = coverage.get_now()
                            if last_cov == cur_cov:
                                useless_pass_idx.append(idx)
                            last_cov = cur_cov
                    opt_mod = tvm.build(mod)
        except Exception as e:
            raise e
        finally:
            if use_cov:
                d['cov'] = coverage.get_now(), coverage.get_hitmap()
                d['useless_pass_idx'] = useless_pass_idx

        if diff_test_round > 0:
            assert __USE_PASS__
            d['stage'] = BuildStage.DIFF_TEST
            for _ in range(diff_test_round):
                # no crash
                params = util.gen_np_params_for_tir(func)
                d['params'] = params

                t0 = time.time()
                opt_result = run_module(opt_mod, params)
                opt_time = time.time() - t0

                t0 = time.time()
                nopt_result = run_module(nopt_mod, params)
                nopt_time = time.time() - t0

                if not util.no_perf_degrad(opt_time, nopt_time):
                    raise error.PerfDegradation

                if not outcome_equal(opt_result, nopt_result):
                    raise error.IncorrectResult

        d['stage'] = BuildStage.FINISHED

    def no_exception_wrapper(
        func: tir.PrimFunc,
        passes: List[tvm.ir.transform.Pass],
        use_cov: bool,
        d: dict,
    ):
        try:
            wrapper(func, passes, use_cov, d)
        except AssertionError as e:
            raise e
        except Exception as e:
            d['exc'] = e

    with mp.Manager() as manager:
        d: dict = manager.dict()  # type: ignore
        p = mp.Process(target=no_exception_wrapper, args=(
            func, 
            passes, 
            use_cov,
            d
        ))

        p_duration = None
        try:
            p.start()
            p_strt_time = time.time()
            p.join(timeout=build_timeout)
            p_duration = time.time() - p_strt_time
        finally:
            if p.is_alive():
                for child in psutil.Process(p.pid).children(recursive=False):
                    child: psutil.Process  # type: ignore
                    try:
                        child.terminate()
                        child.wait()
                        # print(f'{child} is terminated')
                    except psutil.NoSuchProcess:
                        pass
                p.terminate()
                p.join()

                if p_duration is not None and p_duration >= build_timeout and (
                    d['stage'] == BuildStage.COMPILE_NOPT or d['stage'] == BuildStage.DIFF_TEST):
                    # assert not 'cov_now' in d
                    raise error.MaybeDeadLoop

            if use_cov and 'cov' in d:
                now, hitmap = d['cov']
                if coverage.get_now() < now:
                    coverage.set_now(now)
                    coverage.set_hitmap(hitmap)

                    if 'useless_pass_idx' in d and useful_pass_mask is not None:
                        useful_pass_mask[d['useless_pass_idx']] = 0
        
        assert not p.is_alive(), 'The build process is expected to be dead.'

        if p.exitcode != 0:
            if d['stage'] == BuildStage.COMPILE_NOPT:
                # assert not 'cov_now' in d
                raise error.RuntimeFailure('Failed in non-opt compilation')
            elif d['stage'] == BuildStage.COMPILE_OPT:
                # assert 'cov_now' in d
                raise error.RuntimeFailure(
                    'Failed in optimized compilation but succeeded in non-opt comp.')
            else:
                raise error.RuntimeFailure(f"Failed in stage: {d['stage']}")
        else:
            # assert 'cov_now' in d
            if 'exc' in d:
                e: Exception = d['exc']
                if type(e) == error.PerfDegradation:
                    raise error.PerfDegradation(d['params'])
                elif type(e) == error.IncorrectResult:
                    raise error.IncorrectResult(d['params'])
                else:
                    # print('=====================================================================')
                    # print(f'Error during compilation ({type(e)})!!! Check the semantic validity of TIR generated by Tzer')
                    # print(func)
                    # print(e)
                    raise e
            else:
                assert d['stage'] == BuildStage.FINISHED