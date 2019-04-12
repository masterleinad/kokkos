cmake_minimum_required (VERSION 2.6)

project (Tutorial)

#SET(CMAKE_CXX_FLAGS "--std=c++11 -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored -arch=sm_70")

add_executable(test_kokkos test_kokkos.cpp)

find_package(Kokkos REQUIRED
  HINTS ${KOKKOS_DIR} $ENV{KOKKOS_DIR}
  )

message(STATUS "${KOKKOS_INCLUDE_DIRS}")

target_link_libraries(test_kokkos PRIVATE Kokkos::kokkos)
#target_include_directories(test_kokkos SYSTEM PUBLIC ${KOKKOS_INCLUDE_DIRS})


