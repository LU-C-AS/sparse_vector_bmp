# add_executable(bmp_tests
#   bmp_index_test.cpp
# )

# target_link_libraries(math_tests
#     PRIVATE 
#         MyMathLib          # 被测试库
#         GTest::GTest      # gtest主库
#         GTest::Main       
# )


# gtest_discover_tests(math_tests
#     EXTRA_ARGS
#         --gtest_output=xml:${CMAKE_BINARY_DIR}/test_results/
#     WORKING_DIRECTORY
#         ${CMAKE_BINARY_DIR}
# )

set(BINARY ${CMAKE_PROJECT_NAME}_test)
file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false *.h *.cpp)
set(SOURCES ${TEST_SOURCES})
add_executable(${BINARY} ${TEST_SOURCES})
add_test(NAME ${BINARY} COMMAND ${BINARY})
# 链接src生成的lib库和gtest库
target_link_libraries(${BINARY} PUBLIC ${CMAKE_PROJECT_NAME}_lib gtest)
