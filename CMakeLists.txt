cmake_minimum_required(VERSION 3.14)
project(SparseVectorBMP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(src)

add_subdirectory(src)
add_subdirectory(third_party/googletest)
add_subdirectory(tests)


# find_package(Threads REQUIRED)
# target_link_libraries(${CMAKE_PROJECT_NAME} Threads::Threads)

# option(BUILD_TESTING "Build tests" ON)
# if(BUILD_TESTING)
#     enable_testing()
#     add_subdirectory(tests/unit)
# endif()

# include(FetchContent)
# FetchContent_Declare(
#     googletest
#     GIT_REPOSITORY https://github.com/google/googletest.git
#     GIT_TAG release-1.12.1
# )
# FetchContent_MakeAvailable(googletest)