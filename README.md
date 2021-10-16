<p align="center">
    <img src="./docs/imgs/Tzer-Logo.svg", width="550">
</p>

---

<p align="center">
    <a href="#Detected-Bugs">Reproduce Bugs</a>  •
    <a href="#Quick-Start">Quick Start</a> •
    <a href="#Installation">Installation</a> •
    <a href="#Usage">Usage</a> •
    <a href="#Want-to-Extend-Tzer?">Extend Tzer</a>
</p>

# Coverage-Guided Tensor Compiler Fuzzing with Joint IR-Pass Mutation

This is the artifact of Tzer for anonymous review in OOPSLA'22. 

## Reproduce Bugs

Till submission, Tzer has been detected **40** bugs for TVM with **30 confirmed** and **24 fixed** (merged in the latest branch). Due to the anonymous review policy of OOPSLA, the links of actual bug reports will be provided after the review process.

We provide strong reproducibility of our work. **To reproduce all bugs, all you need to do is a single click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tzer-AnonBot/tzer/blob/main/bug-report.ipynb) on your browser**. Since some bugs need to be triggered by some complex GPU settings, to maximumly ease the hardware and software effort, the bugs are summarized in a Google Colab environment (No GPU required, but just a browser!).

## Quick Start

Quick start with Docker!

```shell
docker run --rm -it tzerbot/oopsla
cd tzer
python3 src/main_tir.py --fuzz-time 10     --report-folder ten-minute-fuzz
#                       run for 10 min.    bugs in folder `ten-minute-fuzz`
```

## Installation

### Expected Environments 

- **Hardware**: 8GB RAM; 100GB Storage; X86 CPU; Good Network to GitHub;
- **Software**: Linux (tested under Manjaro and Ubuntu20.04)

### Docker Hub (Pre-built)

```shell
docker run --rm -it tzerbot/oopsla
```

### Docker (Build from Scratch)

1. Make sure you have [docker](https://docs.docker.com/get-docker/) installed.
2. `git clone https://github.com/Tzer-AnonBot/tzer.git && cd tzer`
3. `docker build --tag tzer-oopsla:eval .`
4. `docker run --rm -it tzer-oopsla:eval`

### Manual (Linux)

```shell
# Arch Linux / Manjaro
sudo pacman -Syy
sudo pacman -S compiler-rt llvm llvm-libs compiler-rt clang cmake git python3
# Ubuntu
sudo apt update
sudo apt install -y libfuzzer-12-dev # If you fail, try "libfuzzer-11-dev", "-10-dev", ...
sudo apt install -y clang cmake git python3
```

#### Installation

```shell
git clone https://github.com/Tzer-AnonBot/tzer.git
cd tzer/tvm_cov_patch

# Build TVM with intruments
bash ./build_tvm.sh # If you fail, check the script for step-by-step instruction;
cd ../../../
# If success, tvm is installed under `tvm_cov_patch/tvm`

# Set up TVM_HOME and PYTHONPATH env var. (so that you can import tvm)
export TVM_HOME=$(realpath tvm_cov_patch/tvm)
export PYTHONPATH=$TVM_HOME/python

# Install Python dependency
python3 -m pip install -r requirements.txt
python3 src/main_tir.py --fuzz-time 10 --report-folder ten-minute-fuzz
# There you go!
```

## Usage

```shell
cd tzer # Under the tzer folder.
python3 src/main_tir.py --fuzz-time 10     --report-folder ten-minute-fuzz
#                       run for 10 min.    bugs in folder `ten-minute-fuzz`
```

Successful installation looks like:

![](./docs/imgs/tzer-terminal-output.png)

Coverage by time: `ten-minute-fuzz/cov_by_time.txt` where 1st column means time (second) and 2nd one means basic-block coverage.

## Want to Extend Tzer?

We implemented many re-usable functionalities for future and open research! To easily implement other coverage-guided fuzzing algorithm for TVM, after your installing TVM with [memcov](https://github.com/Tzer-AnonBot/memcov) by applying `tvm_cov_patch/memcov4tvm.patch` to TVM (See [](tvm_cov_patch/build_tvm.sh)), you can get current coverage of TVM by:

```python
from tvm.contrib import coverage

print(coverage.get_now()) # Current visited # basic blocks
print(coverage.get_total()) # Total number of # basic blocks

coverage.push() # store current coverage snapshot to a stack and reset it to empty (useful for multi-process scenario)
coverage.pop()  # merge the top snapshot from the stack. 
```

**Usage `push-pop` combo**: Some times the target program might crash, but we don't want the fuzzer to be affected by the failure. Therefore, you can set a "safe guard" by:

1. push: save current snapshot and reset the coverage hitmap;
2. raise a sub-process to compile target IR & passes with TVM;
3. pop: merge the snapshot of the sub-process and last stored snapshot (top of the stack) to get a complete coverage.

Latency of such combo is usually around 1ms since we applied bit-level optimization.
