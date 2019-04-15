cmake_minimum_required (VERSION 2.6)

project (Tutorial)

add_executable(test_kokkos test_kokkos.cpp)

find_package(PkgConfig REQUIRED)
pkg_check_modules(KOKKOS IMPORTED_TARGET kokkos)

target_link_libraries(test_kokkos ${KOKKOS_LDFLAGS})
target_include_directories(test_kokkos PUBLIC ${KOKKOS_INCLUDE_DIRS})
target_compile_options(test_kokkos PUBLIC ${KOKKOS_CFLAGS_OTHER})
