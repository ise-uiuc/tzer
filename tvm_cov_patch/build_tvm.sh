set -e

echo "Installing latest TVM... If it fails you may want to use earlier versions."
git clone https://github.com/apache/tvm.git
cd tvm
git submodule init && git submodule update --recursive
patch -p1 < ../memcov4tvm.patch
mkdir -p build && cd build
echo "We use LLVM Sanitizers. Please ensure you have clang & compiler-rt installed."
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CXX_COMPILER=$(which clang++) \
         -DCMAKE_C_COMPILER=$(which clang) \
         -DFETCHCONTENT_QUIET=0
make -j$(nproc)
echo "Successfully installed TVM-cov! Please set PYTHONPATH=/path/to/tvm/python before use"
