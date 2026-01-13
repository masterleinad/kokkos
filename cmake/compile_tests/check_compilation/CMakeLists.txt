cmake_minimum_required(VERSION 3.22)
project(CheckCompilation ${TEST_LANGUAGE})
add_executable(CheckCompilationExec dummy.cpp)

set_source_files_properties(dummy.cpp PROPERTIES LANGUAGE ${TEST_LANGUAGE})
target_compile_options(CheckCompilationExec PRIVATE ${TEST_COMPILE_FLAGS})
target_link_options(CheckCompilationExec PRIVATE ${TEST_LINK_OPTIONS})
target_link_libraries(CheckCompilationExec ${TEST_LINK_LIBRARIES})
