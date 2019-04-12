cmake_minimum_required (VERSION 2.6)

project (Tutorial)

SET(CMAKE_CXX_FLAGS "--std=c++11 -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored -arch=sm_70")

add_executable(test_kokkos test_kokkos.cpp)

MESSAGE(STATUS "${KOKKOS_DIR}")
FIND_PATH(KOKKOS_INCLUDE_DIR KokkosCore_config.h
          HINTS ${KOKKOS_DIR} $ENV{KOKKOS_DIR}
          PATH_SUFFIXES include)

IF(NOT KOKKOS_INCLUDE_DIR)
  MESSAGE(STATUS "${KOKKOS_INCLUDE_DIR}")
  MESSAGE(FATAL_ERROR
          "Error! Could not locate Kokkos.")
ENDIF()

FIND_LIBRARY(KOKKOS_LIBRARY NAMES kokkos
             HINTS ${KOKKOS_DIR} $ENV{KOKKOS_DIR}
             PATH_SUFFIXES lib)

IF(NOT KOKKOS_LIBRARY)
  MESSAGE(STATUS "${KOKKOS_LIBRARY}")
  MESSAGE(FATAL_ERROR
          "Error! Could not locate Kokkos.")
ENDIF()

target_link_libraries(test_kokkos PRIVATE KOKKOS_LIBRARY)
target_include_directories(test_kokkos SYSTEM PRIVATE ${KOKKOS_INCLUDE_DIR})


