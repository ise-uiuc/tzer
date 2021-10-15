<p align="center">
    <img src="./docs/imgs/Tzer-Logo.svg", width="550">
</p>

---

<p align="center">
    <a href="#Installation">Installation</a> •
    <a href="#Quick-Start">Quick Start</a> •
    <a href="#Detected-Bugs">Detected Bugs</a> 
</p>

# Coverage-Guided Tensor Compiler Fuzzing with Joint IR-Pass Mutation

This is the artifact of Tzer for anonymous review in OOPSLA'22. 

## Installation

### Expected Environments 

- **Hardware**: 8GB RAM; 100GB Storage; X86 CPU; Good Network to GitHub;
- **Software**: Linux (tested under Manjaro and Ubuntu20.04)

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

# Add TVM into python path
export PYTHONPATH=$(realpath tvm_cov_patch/tvm/python)

# Install Python dependency
python3 -m pip install -r requirements.txt
python3 src/main_tir.py --fuzz-time 10 --report-folder ten-minute-fuzz
# There you go!
```

## Quick Start

```shell
# Under the tzer folder.
python3 src/main_tir.py --fuzz-time 10     --report-folder ten-minute-fuzz
#                       run for 10 min.    bugs in folder `ten-minute-fuzz`
```

Successful installation looks like:

![](./docs/imgs/tzer-terminal-output.png)

Coverage by time: `ten-minute-fuzz/cov_by_time.txt` where 1st column means time (second) and 2nd one means basic-block coverage.

## Detected Bugs

Till submission, Tzer has been detected **40** bugs for TVM with **30 confirmed** and 24 fixed (merged in the latest branch).

To maximumly ease your effort, we provide an online executable environments for you. To reproduce all the bugs, all you need to do is a single click on your browser.
