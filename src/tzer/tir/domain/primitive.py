from typing import Dict, List
from typing import get_args, get_origin, get_type_hints
from tvm import ir, tir
from tvm.ir.expr import Range
from . import construct as cons
from .basic_type import *
import sys

_INT = 'int32'
_UINT = 'uint32'
_BOOL = 'bool'
_FLOAT = 'float32'
_HANDLE = 'handle'

_SIZEVAR_N = tir.SizeVar('N', _UINT)

DTYPES: List[DType] = [
    DType(dtype) for dtype in [
        _INT,
        _UINT,
        _BOOL,
        _FLOAT,
        _HANDLE,
    ]
]

ATTRKEYS: List[AttrKey] = [
    AttrKey(key) for key in [
        ""
        "tzer",
        "pragma_auto_unroll_max_step",
        "pragma_unroll_explicit",
        "thread_extent",
        "virtual_thread",
        "coproc_scope",
        "coproc_uop_scope",
        "volatile_scope",
        "extern_scope",
        "compute_scope",
        "storage_alignment",
        "realize_scope",
        "device_id",
        "device_type",
        "loop_scope",
        "reduce_scope",
        "pragma_",
        "pragma_import_c",
        "pragma_import_llvm",
        "pragma_debug_skip_region",
        "buffer_bind_scope",
        "channel_read_scope",
        "channel_read_advance",
        "channel_write_scope",
        "channel_write_advance",
        "pipeline_stage_scope",
        "pipeline_exec_scope",
        "device_scope",
        "fragment_shape",
        "fragment_layout",
        "buffer_dim_align",
        "buffer_bound",
        "hand_threaded"
    ]
]

THREADTAGS: List[ThreadTag] = [
    ThreadTag(thread_tag) for thread_tag in [
        "",
        "tzer",
        "vthread",
        "vthread.x",
        "vthread.y",
        "vthread.z",
        "threadIdx.x",
        "threadIdx.y",
        "threadIdx.z",
    ]
]

ITERTYPES: List[IterType] = [IterType(i) for i in range(1, 6)]

LANES: List[Lanes] = [Lanes(i) for i in range(0, 128)]

RANGES: List[ir.Range] = [
    ir.Range(0, 128),
    ir.Range(0, 16),
    ir.Range(0, 8),
    ir.Range(-8, 8),
    ir.Range(-8, 0),
    ir.Range(-16, 0),
    ir.Range(-128, 0),
    ir.Range(-128, 128),
]

_BI1 = tir.decl_buffer((_SIZEVAR_N,), _INT, 'bi1')
_BU1 = tir.decl_buffer((_SIZEVAR_N,), _UINT, 'bu1')
_BB1 = tir.decl_buffer((_SIZEVAR_N,), _BOOL, 'bb1')
_BF1 = tir.decl_buffer((_SIZEVAR_N,), _FLOAT, 'bf1')

_BI2 = tir.decl_buffer((_SIZEVAR_N, _SIZEVAR_N, _SIZEVAR_N,), _INT, 'bi2')
_BU2 = tir.decl_buffer((_SIZEVAR_N, _SIZEVAR_N), _UINT, 'bu2')
_BB2 = tir.decl_buffer((_SIZEVAR_N,), _BOOL, 'bb2')
_BF2 = tir.decl_buffer((_SIZEVAR_N), _FLOAT, 'bf2')

_BI3 = tir.decl_buffer((_SIZEVAR_N, _SIZEVAR_N), _INT, 'bi3')
_BU3 = tir.decl_buffer((_SIZEVAR_N, _SIZEVAR_N), _UINT, 'bu3')
_BB3 = tir.decl_buffer((_SIZEVAR_N, _SIZEVAR_N), _BOOL, 'bb3')
_BF3 = tir.decl_buffer((_SIZEVAR_N, _SIZEVAR_N), _FLOAT, 'bf3')

_H1 = tir.Var('h1', _HANDLE)
_H2 = tir.Var('h2', _HANDLE)
_H3 = tir.Var('h3', _HANDLE)
_H4 = tir.Var('h4', _HANDLE)

_BUFFER_MAP = {_H1: _BI1, _H2: _BU1, _H3: _BB1, _H4: _BF1}
_BUFFER_MAP_1 = {_H1: _BI2, _H2: _BU2, _H3: _BB2, _H4: _BF2}
_BUFFER_MAP_2 = {_H1: _BI3, _H2: _BU3, _H3: _BB3, _H4: _BF3}

BUFFER_MAPS: List[Dict[tir.Var, tir.Buffer]] = [
    {},
    _BUFFER_MAP,
    _BUFFER_MAP_1,
    _BUFFER_MAP_2,
]

BUFFERS: List[tir.Buffer] = [
    _BI1,
    _BU1,
    _BB1,
    _BF1,
    _BI2,
    _BU2,
    _BB2,
    _BF2,
    _BI3,
    _BU3,
    _BB3,
    _BF3,
]

INTIMMS: List[tir.IntImm] = [
    tir.IntImm(_INT, 0),
    tir.IntImm(_INT, 1),
    tir.IntImm(_INT, -1234567),
    tir.IntImm(_INT, 1234567),
    tir.IntImm(_INT, 1000),
    tir.IntImm(_INT, -1001),

    tir.IntImm(_BOOL, True),
    tir.IntImm(_BOOL, False),

    tir.IntImm(_UINT, 0),
    tir.IntImm(_UINT, 1),
    tir.IntImm(_UINT, 9876543)
]

FLOATIMMS: List[tir.FloatImm] = [
    tir.FloatImm(_FLOAT, 0.0),
    tir.FloatImm(_FLOAT, 1.0),
    tir.FloatImm(_FLOAT, 99999999.999999),
    tir.FloatImm(_FLOAT, -99999999.999999),
    tir.FloatImm(_FLOAT, 88888.88888),
    tir.FloatImm(_FLOAT, -77777.77777),
]

STRINGIMMS: List[tir.StringImm] = [
    tir.StringImm('s1'),
    tir.StringImm('s2'),
    tir.StringImm('s3'),
    tir.StringImm('s4'),
    tir.StringImm('s5'),
]

VARS: List[tir.Var] = [
    tir.Var('f1', _FLOAT),
    tir.Var('f2', _FLOAT),
    tir.Var('f3', _FLOAT),
    tir.Var('f4', _FLOAT),
    tir.Var('f5', _FLOAT),

    tir.Var('i1', _INT),
    tir.Var('i2', _INT),
    tir.Var('i3', _INT),
    tir.Var('i4', _INT),
    tir.Var('i5', _INT),

    tir.Var('u1', _UINT),
    tir.Var('u2', _UINT),
    tir.Var('u3', _UINT),
    tir.Var('u4', _UINT),
    tir.Var('u5', _UINT),

    tir.Var('b1', _BOOL),
    tir.Var('b2', _BOOL),
    tir.Var('b3', _BOOL),
    tir.Var('b4', _BOOL),
    tir.Var('b5', _BOOL),

    _SIZEVAR_N,

    _H1,
    _H2,
    _H3,
    _H4,

    _BI1.data,
    _BU1.data,
    _BB1.data,
    _BF1.data,
]

PRIMEXPRS: List[tir.PrimExpr] = INTIMMS + FLOATIMMS + STRINGIMMS + VARS

FOR_KINDS: List[tir.ForKind] = [
    tir.ForKind.SERIAL,
    tir.ForKind.PARALLEL,
    tir.ForKind.VECTORIZED,
    tir.ForKind.UNROLLED,
    tir.ForKind.THREAD_BINDING,
]

LST_PRIMEXPR: List[List[tir.PrimExpr]] = [[], [tir.const(0)], ]
LST_STMT: List[List[tir.Stmt]] = [[], ]
LST_VAR: List[List[tir.Var]] = [[], ]


_ALL_TYPE_HINTS = get_type_hints(sys.modules[__name__])


def _get_all_primitives() -> Dict[cons.Dom, List[cons.DomValue]]:
    res: Dict[cons.Dom, List[cons.DomValue]] = {}
    for vname in dir(sys.modules[__name__]):
        try:
            hint = _ALL_TYPE_HINTS[vname]
        except KeyError:
            continue
        if get_origin(hint) == list:
            dom = get_args(hint)[0]
            if dom in cons.ALL_DOMS:
                res[dom] = globals()[vname]
    return res


ALL_PRIMITIVES: Dict[cons.Dom, List[cons.DomValue]] = _get_all_primitives()
