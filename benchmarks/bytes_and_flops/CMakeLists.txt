cmake_minimum_required(VERSION 3.16)

project(BytesAndFlops CXX)

find_package(Kokkos REQUIRED)

add_executable(bytes_and_flops.exe
  main.cpp
  bench_double.cpp
  bench_float.cpp
  bench_int32_t.cpp
  bench_int64_t.cpp
  )

target_include_directories(bytes_and_flops.exe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bytes_and_flops.exe Kokkos::kokkos)
