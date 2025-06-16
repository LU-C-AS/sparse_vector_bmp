set(BINARY ${CMAKE_PROJECT_NAME})
 
file(GLOB_RECURSE SOURCES LIST_DIRECTORIES true *.hpp *.cpp)


set(SOURCES ${SOURCES})
add_executable(${BINARY}_run ${SOURCES})
find_package(Threads REQUIRED)
target_link_libraries(${BINARY}_run Threads::Threads)

set_target_properties(${BINARY}_run PROPERTIES
	    LINKER_LANGUAGE CXX  # 强制指定C++链接<sup>3</sup>
)
# 为了让单元测试的时候src下的代码能被作为静态链接库使用
add_library(${BINARY}_lib STATIC ${SOURCES})
target_link_libraries(${BINARY}_lib Threads::Threads)

set_target_properties(${BINARY}_lib PROPERTIES
	    LINKER_LANGUAGE CXX  # 强制指定C++链接<sup>3</sup>
)
