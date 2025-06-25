cmake_minimum_required(VERSION 3.14)
project(SparseVectorBMP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(src)

add_subdirectory(src)
add_subdirectory(third_party/googletest)
add_subdirectory(tests)
