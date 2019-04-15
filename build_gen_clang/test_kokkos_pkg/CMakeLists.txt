cmake_minimum_required (VERSION 2.6)

project (Tutorial)

add_executable(test_kokkos test_kokkos.cpp)

find_package(PkgConfig REQUIRED)
pkg_check_modules(KOKKOS IMPORTED_TARGET kokkos)
target_link_libraries(test_kokkos PRIVATE PkgConfig::KOKKOS)

target_link_libraries(test_kokkos PRIVATE /tmp/kokkos/build_gen_clang/build/install/lib/libkokkos.a)

find_package(OpenMP REQUIRED)
target_link_libraries(test_kokkos PRIVATE OpenMP::OpenMP_CXX)

target_link_libraries(test_kokkos PRIVATE dl)
target_link_libraries(test_kokkos PRIVATE rt)
target_link_libraries(test_kokkos PRIVATE /usr/local/cuda-9.2/targets/x86_64-linux/lib/libcudart_static.a)
