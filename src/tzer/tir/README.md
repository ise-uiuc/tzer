# tzer.tir

## Quick start

Note these commands might be out-of-date all the time so please look at `src/tzer/tir/config.py` for detailed specification. 

```shell
python src/main_tir.py --fuzz-time 240

# file mutation only
python src/main_tir.py --fuzz-time 240
# with pass
PASS=1 python src/main_tir.py --fuzz-time 240
# with all seeds
ALL_SEEDS=1 python src/main_tir.py --fuzz-time 240
# baseline: random generation w/o looking at coverage
NO_COV=1 python src/main_tir.py --fuzz-time 240

## EXPERIMENTAL
# Provide incorrect values on purpose during fuzzing
NONE=1 python src/main_tir.py --fuzz-time 240
```

## Other options

- Please refer to `src/tzer/tir/config.py`.

About reproducibility:

We fixed the random seed for reproducibility and fair comparison.

## Seeds

Look into `src/tzer/tir/seed.py`. Note that `ALL_SEEDS` is deprecated.

## How to retrieve fuzzed tir.PrimFunc

`population: List[tir.PrimFunc] = fuzzer.state[tir.PrimFunc].values`
