import pickle
import hashlib

def cov_id(coverage: bytearray) -> bytes:
    pickled = pickle.dumps(coverage)
    return hashlib.md5(pickled).digest()

if __name__ == '__main__':
    from tvm.contrib import coverage as memcov
    print(cov_id(memcov.get_now()))
    from tvm import te
    te.var('a') + te.var('b')
    print(cov_id(memcov.get_now()))
