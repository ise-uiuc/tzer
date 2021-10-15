from __future__ import print_function
from numpy import testing

from .error import *

def assert_allclose(obtained, desired, obtained_name, oracle_name):
    try:
        index = 1 # TODO support multiple inputs
        # for l, r in zip(obtained, desired):
        # testing.assert_allclose(l, r, rtol=1e-07, atol=1e-06)
        testing.assert_allclose(obtained.numpy(), desired.numpy(), rtol=1e-02, atol=1e-05)
        index += 1
    except AssertionError as err:
        print(err)
        raise IncorrectResult(
            f'{obtained_name} v.s. {oracle_name} mismatch in #{index} tensor:')

def assert_no_perf_degrad(optimzed_time :float, non_optimized_time :float, rtol: float = 5e-2):
    """Assert no performance degradation.

    Args:
        optimzed_time (float): [description]
        non_optimized_time (float): [description]
        atol (float, optional): Max tolerant runtime degradation. Defaults to 5e-3.
    """
    frac_down = max(non_optimized_time, optimzed_time)
    frac_up = min(non_optimized_time, optimzed_time)
    relative_err = 1 - frac_up / frac_down 
    if optimzed_time > non_optimized_time and relative_err > rtol:
        raise PerfDegradation(
            f'{optimzed_time} > {non_optimized_time}: relative err: {relative_err}')
