# sparse_vector_bmp

This demo is based on sparse vector index implemented in infinity, using BMP algorithm.

# build and run

```
# 1. build
mkdir third_party
git clone https://github.com/google/googletest.git
mkdir build
cd build
cmake ..
cmake --build .
# 2. run
./tests/SparseVectorBMP_test
```
BTW, change the CMAKE_CXX_FLAGS_RELEASE flag in tests/CMakeLists.txt to use other levels(e.g. -Ofast) in compilation.
