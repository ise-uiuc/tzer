"""This is a generic timeout mechanism in python!!!
NOTE: cannot be used with functions having complex return values
such as IRModule of tvm, which is related to C++ cannot be easily
transmitted between processes"""

import struct
from typing import Callable, Dict, Any, Optional, Tuple
import uuid
from tvm._ffi._ctypes.object import ObjectBase
import numpy as np
from tvm import tir
import tvm
import tvm.testing
from tvm._ffi.runtime_ctypes import Device
import pickle
from typing import List, Set, TypeVar, Union
import hashlib
import random
import string
import math
import time


def cov_id(coverage: bytearray) -> bytes:
    pickled = pickle.dumps(coverage)
    return hashlib.md5(pickled).digest()


T = TypeVar('T')


def flatten(sets: List[Set[T]]) -> Set[T]:
    res: Set[T] = set()
    for s in sets:
        res |= s
    return res


TIRNode = Union[tir.PrimFunc, tir.PrimExpr, tir.Stmt]


def dict_union(*dicts: Dict[Any, list]) -> Dict[Any, list]:
    res: Dict[Any, list] = {}
    for d in dicts:
        for k, v in d.items():
            if not k in res:
                res[k] = []
            res[k] += v
    return res


def list_union(lists: List[List[T]]) -> List[T]:
    res: list = []
    for l in lists:
        res += l
    return res


_DUMMY_BUFFER = ''.join(random.choice(string.digits) for _ in range(1024))


def gen_np_params_for_tir(root: tir.PrimFunc) -> list:
    result: list = []
    for param in root.params:
        if param.dtype.startswith('int'):
            result.append(random.randint(1, 5))
        elif param.dtype.startswith('uint'):
            result.append(random.randint(1, 5))
        elif param.dtype.startswith('bool'):
            result.append(random.choice([True, False]))
        elif param.dtype.startswith('float') or param.dtype.startswith('bfloat'):
            result.append(float(random.random() * 2))
        elif param.dtype.startswith('str'):
            result.append(_DUMMY_BUFFER)
        elif param.dtype == 'handle':
            if param in root.buffer_map:
                buffer = root.buffer_map[param]
                shape = [x.value if 'value' in dir(x) and isinstance(
                    x.value, int) else 1 for x in buffer.shape]
                if len(shape) > 0:
                    result.append(
                        (2 * np.random.rand(*shape)).astype(buffer.dtype))
                else:
                    result.append(2 * np.random.rand())
            else:
                result.append((2 * np.random.rand(1)).astype('float32'))
        else:
            raise NotImplementedError
    return result


def np_params_to_tvm_params(params: list, dev: Device = tvm.cpu()) -> list:
    return [tvm.nd.array(x, dev) if isinstance(x, np.ndarray)
            else x for x in params]


def tvm_params_to_np_params(params: list) -> list:
    result = []
    for x in params:
        try:
            result.append(x.numpy() if isinstance(
                x, tvm.runtime.NDArray) else x)
        except (ValueError, tvm.TVMError):
            # Errors occurring here may account for some internal bugs
            result.append(None)
    return result


def gen_params_for_tir(root: tir.PrimFunc, dev: Device = tvm.cpu()) -> list:
    params = gen_np_params_for_tir(root)
    return np_params_to_tvm_params(params)


def copy_np_params(params: List[Union[Any, np.ndarray]]) -> List[Union[Any, tvm.nd.array]]:
    return [x.copy() if isinstance(x, np.ndarray) else x for x in params]


def fresh_name() -> str:
    return str(uuid.uuid4())[:3]

def random_uint() -> int:
    return random.randint(0, np.iinfo(np.int32).max)

def random_integer() -> int:
    return random.randint(np.iinfo(np.int32).min, np.iinfo(np.int32).max)

def str_to_bool(s: str) -> bool:
    if not s in ['True', 'False']:
        raise RuntimeError
    return True if s == 'True' else False


def all_close(
    lhs: Union[np.ndarray, int, float, bool, None],
    rhs: Union[np.ndarray, int, float, bool, None],
) -> bool:
    if lhs is None:
        return rhs is None
    if rhs is None:
        return lhs is None
    try:
        tvm.testing.assert_allclose(lhs, rhs, 1e-3, 1e-3)
        return True
    except AssertionError:
        return False


def no_perf_degrad(optimzed_time: float, non_optimized_time: float, rtol: float = 5e-2) -> bool:
    """Assert no performance degradation.

    Args:
        optimzed_time (float): [description]
        non_optimized_time (float): [description]
        rtol (float, optional): Max tolerant runtime degradation. Defaults to 5e-3.
    """
    frac_down = max(non_optimized_time, optimzed_time)
    frac_up = min(non_optimized_time, optimzed_time)
    relative_err = 1 - frac_up / frac_down
    return optimzed_time > non_optimized_time and relative_err > rtol


def get_id(obj) -> Optional[int]:
    if isinstance(obj, ObjectBase):
        if obj.handle is None:
            return None
        else:
            return obj.handle.value
    else:
        return id(obj)


def weighted_select(weighted_data: List[Tuple[int, T]]) -> T:
    weights, options = list(zip(*weighted_data))
    weights_np = np.array(weights)
    distribution = weights_np / np.sum(weights_np)
    return np.random.choice(options, p=distribution)


def uniformly_split_num(
    sum: int,
    n: int,
) -> List[int]:
    """Generate `n` non-negative numbers that sum up to `sum`"""
    assert n > 0, 'n should be > 0'
    assert sum >= 0, 'sum should be >= 0'
    random_numbers = [random.randint(0, sum) for _ in range(n - 1)]
    values = [0] + sorted(random_numbers) + [sum]
    values.sort()
    intervals = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    return intervals


def efficiently_fixed_sum(
    sum: int,
    n: int,
) -> List[int]:
    raise NotImplementedError

def random_float() -> float:
    return random.random()


def random_string() -> str:
    return str(time.time())

def permutation(l: list) -> list:
    return list(np.random.permutation(l))