cmake_minimum_required (VERSION 2.6)

project (Tutorial)

add_executable(test_kokkos test_kokkos.cpp)

FIND_PACKAGE(Kokkos REQUIRED PATHS /tmp/kokkos/build_cmake/build/install/lib/CMake/Kokkos)

target_link_libraries(test_kokkos PRIVATE ${Kokkos_LIBRARIES})
target_include_directories(test_kokkos SYSTEM PRIVATE ${Kokkos_INCLUDE_DIRS})
