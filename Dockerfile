FROM ubuntu:20.04

RUN apt update --allow-unauthenticated

RUN DEBIAN_FRONTEND="noninteractive" apt install libfuzzer-12-dev clang cmake git python3

RUN git clone https://github.com/Tzer-AnonBot/tzer.git

RUN cd /tzer/tvm_cov_patch && ./build_tvm.sh

RUN python3 -m pip install -r /tzer/requirements.txt

ENV PYTHONPATH=/tzer/tvm_cov_patch/tvm/python
