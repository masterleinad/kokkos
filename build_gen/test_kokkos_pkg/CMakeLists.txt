cmake_minimum_required (VERSION 2.6)

project (Tutorial)

add_executable(test_kokkos test_kokkos.cpp)

find_package(PkgConfig REQUIRED)
pkg_check_modules(KOKKOS IMPORTED_TARGET kokkos)
target_link_libraries(test_kokkos PRIVATE PkgConfig::KOKKOS)

target_link_libraries(test_kokkos PRIVATE /tmp/kokkos/build_gen/build/install/lib/libkokkos.a)
